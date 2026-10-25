"""
Metaculus forecasting bot powered by Claude Code.

Research: one Claude run with WebSearch/WebFetch writes a dated research brief.
Forecast: several independent Claude runs (FORECAST_MODELS) forecast from that brief,
and their answers are combined into the single prediction that gets published.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import statistics
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from forecasting_tools import (
    BinaryQuestion,
    ConditionalPrediction,
    ConditionalQuestion,
    DateQuestion,
    ForecastBot,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    PredictionAffirmed,
    ReasonedPrediction,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

import claude_cli
import prompts

logger = logging.getLogger(__name__)

T = TypeVar("T")
Member = tuple[str, Any, str]  # (model, parsed value, rationale)


def _env_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _percent(probability: float) -> str:
    return f"{probability * 100:.1f}%"


class ClaudeForecaster(ForecastBot):
    research_model = os.getenv("RESEARCH_MODEL", "sonnet")
    research_effort = os.getenv("RESEARCH_EFFORT", "high") or None
    fallback_model = os.getenv("FALLBACK_MODEL", "opus")
    forecast_models = _env_list("FORECAST_MODELS", "opus,opus,sonnet")
    forecast_effort = os.getenv("FORECAST_EFFORT", "high") or None
    research_timeout = float(os.getenv("RESEARCH_TIMEOUT_SECONDS", "900"))
    forecast_timeout = float(os.getenv("FORECAST_TIMEOUT_SECONDS", "600"))

    min_binary_probability = 0.015
    max_binary_probability = 0.985
    min_option_probability = 0.005

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("research_reports_per_question", 1)
        # One "prediction" per question: the ensemble runs inside each forecast method.
        kwargs.setdefault("predictions_per_research_report", 1)
        super().__init__(**kwargs)

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, Any]:
        return {
            "default": f"claude-code ensemble ({', '.join(cls.forecast_models)})",
            "researcher": f"claude-code {cls.research_model} with WebSearch/WebFetch",
            "summarizer": None,
            "parser": None,
        }

    ##################################### CLAUDE CALLS #####################################

    async def _call_with_fallback(self, prompt: str, *, model: str, **kwargs: Any) -> claude_cli.ClaudeResult:
        try:
            return await claude_cli.run_claude(prompt, model=model, **kwargs)
        except claude_cli.ClaudeUsageLimitError:
            raise
        except claude_cli.ClaudeCliError as error:
            alternate = self.fallback_model if model != self.fallback_model else self.research_model
            if alternate == model:
                raise
            logger.warning(f"{model} failed ({error}); retrying once with {alternate}")
            return await claude_cli.run_claude(prompt, model=alternate, **kwargs)

    async def _ensemble(self, prompt: str, schema: dict, parse: Callable[[dict], T]) -> list[Member]:
        async def forecast_once(model: str) -> Member:
            result = await self._call_with_fallback(
                prompt,
                model=model,
                system_prompt=prompts.FORECAST_SYSTEM_PROMPT,
                json_schema=schema,
                effort=self.forecast_effort,
                timeout_seconds=self.forecast_timeout,
            )
            assert result.structured is not None
            value = parse(result.structured)
            rationale = str(result.structured.get("rationale", "")).strip()
            return result.model, value, rationale

        outcomes = await asyncio.gather(
            *(forecast_once(model) for model in self.forecast_models), return_exceptions=True
        )
        members = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        for failure in failures:
            logger.warning(f"Forecast run failed: {type(failure).__name__}: {failure}")

        needed = math.ceil(len(self.forecast_models) / 2)
        if len(members) < needed:
            limit_errors = [f for f in failures if isinstance(f, claude_cli.ClaudeUsageLimitError)]
            if limit_errors:
                raise limit_errors[0]
            raise claude_cli.ClaudeCliError(
                f"Only {len(members)} of {len(self.forecast_models)} forecast runs succeeded: {failures}"
            )
        return members

    @staticmethod
    def _combined_rationale(members: list[Member], readable: Callable[[Any], str], final: str) -> str:
        parts = [f"**Combined forecast:** {final}", ""]
        for index, (model, value, rationale) in enumerate(members, start=1):
            parts.append(f"### Forecaster {index} ({model}): {readable(value)}\n{rationale}\n")
        return "\n".join(parts)

    @staticmethod
    def _split_condition(research: str) -> tuple[str, str]:
        """Conditional branches carry their assumption in front of the research; split it off."""
        marker = "# Conditional question"
        if research.startswith(marker):
            context, _, rest = research.partition("\n\n#### RESEARCH ####\n")
            return context + "\n", rest
        return "", research

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        if isinstance(question, ConditionalQuestion):
            prompt = (
                prompts.research_prompt(question.child)
                + "\n\nThis question is conditional on the following related question; research both.\n\n"
                + prompts.question_block(question.parent)
            )
        else:
            prompt = prompts.research_prompt(question)

        result = await self._call_with_fallback(
            prompt,
            model=self.research_model,
            system_prompt=prompts.RESEARCH_SYSTEM_PROMPT,
            tools=("WebSearch", "WebFetch"),
            effort=self.research_effort,
            timeout_seconds=self.research_timeout,
        )
        research = result.text.strip()
        if len(research) < 200:
            raise claude_cli.ClaudeCliError(f"Research came back too short ({len(research)} chars)")
        logger.info(
            f"Research for {question.page_url}: {len(research)} chars, {result.web_searches} web searches"
        )
        return research

    async def summarize_research(self, question: MetaculusQuestion, research: str) -> str:
        # No extra model call: the comment's summary is the brief's "Current status" section.
        match = re.search(r"## Current status\s*(.+?)(?=\n## |\Z)", research, re.S)
        excerpt = (match.group(1) if match else research).strip()
        return excerpt if len(excerpt) <= 1500 else excerpt[:1500] + "..."

    ##################################### BINARY #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        context, research = self._split_condition(research)
        prompt = prompts.forecast_prompt(question, research, prompts.BINARY_TASK, context)
        members = await self._ensemble(prompt, prompts.binary_schema(), lambda data: float(data["probability"]))
        probability = self.combine_binary([value for _, value, _ in members])
        return ReasonedPrediction(
            prediction_value=probability,
            reasoning=self._combined_rationale(members, _percent, _percent(probability)),
        )

    @classmethod
    def combine_binary(cls, probabilities: list[float]) -> float:
        """Average in log-odds space (geometric mean of odds), then clip."""
        clipped = [min(max(p, 0.005), 0.995) for p in probabilities]
        mean_log_odds = statistics.fmean(math.log(p / (1 - p)) for p in clipped)
        combined = 1 / (1 + math.exp(-mean_log_odds))
        return min(max(combined, cls.min_binary_probability), cls.max_binary_probability)

    ##################################### MULTIPLE CHOICE #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        context, research = self._split_condition(research)
        options = list(question.options)
        prompt = prompts.forecast_prompt(question, research, prompts.MULTIPLE_CHOICE_TASK, context)

        def parse(data: dict) -> dict[str, float]:
            raw = {option: 0.0 for option in options}
            for item in data["probabilities"]:
                if item["option"] in raw:
                    raw[item["option"]] += float(item["probability"])
            total = sum(raw.values())
            if total <= 0:
                raise ValueError("All option probabilities were zero")
            return {option: probability / total for option, probability in raw.items()}

        members = await self._ensemble(prompt, prompts.multiple_choice_schema(options), parse)
        final = self.combine_options([value for _, value, _ in members], options)
        prediction = PredictedOptionList(
            predicted_options=[PredictedOption(option_name=o, probability=p) for o, p in final.items()]
        )

        def readable(value: dict[str, float]) -> str:
            return ", ".join(f"{option}: {_percent(p)}" for option, p in value.items())

        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=self._combined_rationale(members, readable, readable(final)),
        )

    @classmethod
    def combine_options(cls, forecasts: list[dict[str, float]], options: list[str]) -> dict[str, float]:
        averaged = {option: statistics.fmean(f[option] for f in forecasts) for option in options}
        floored = {option: max(p, cls.min_option_probability) for option, p in averaged.items()}
        total = sum(floored.values())
        return {option: p / total for option, p in floored.items()}

    ##################################### NUMERIC AND DATE #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        context, research = self._split_condition(research)
        prompt = prompts.forecast_prompt(question, research, prompts.NUMERIC_TASK, context)

        def parse(data: dict) -> list[float]:
            return [float(data["percentiles"][f"p{level}"]) for level in prompts.PERCENTILE_LEVELS]

        members = await self._ensemble(prompt, prompts.percentile_schema("number"), parse)
        values = self.average_percentiles([value for _, value, _ in members])
        distribution = self.build_distribution(values, question)

        def readable(value: list[float]) -> str:
            return self._readable_percentiles(value, lambda v: f"{v:,.6g}")

        return ReasonedPrediction(
            prediction_value=distribution,
            reasoning=self._combined_rationale(members, readable, readable(values)),
        )

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        context, research = self._split_condition(research)
        prompt = prompts.forecast_prompt(question, research, prompts.DATE_TASK, context)

        def parse(data: dict) -> list[float]:
            return [_date_to_timestamp(data["percentiles"][f"p{level}"]) for level in prompts.PERCENTILE_LEVELS]

        members = await self._ensemble(prompt, prompts.percentile_schema("date"), parse)
        values = self.average_percentiles([value for _, value, _ in members])
        distribution = self.build_distribution(values, question)

        def readable(value: list[float]) -> str:
            return self._readable_percentiles(
                value, lambda v: datetime.fromtimestamp(v, tz=timezone.utc).date().isoformat()
            )

        return ReasonedPrediction(
            prediction_value=distribution,
            reasoning=self._combined_rationale(members, readable, readable(values)),
        )

    @staticmethod
    def average_percentiles(forecasts: list[list[float]]) -> list[float]:
        """Quantile averaging: mean value at each percentile level, kept non-decreasing."""
        averaged = [statistics.fmean(level_values) for level_values in zip(*(sorted(f) for f in forecasts))]
        for index in range(1, len(averaged)):
            averaged[index] = max(averaged[index], averaged[index - 1])
        return averaged

    @staticmethod
    def _readable_percentiles(values: list[float], fmt: Callable[[float], str]) -> str:
        shown = {10, 50, 90}
        return ", ".join(
            f"p{level}={fmt(value)}" for level, value in zip(prompts.PERCENTILE_LEVELS, values) if level in shown
        )

    @staticmethod
    def build_distribution(values: list[float], question: NumericQuestion | DateQuestion) -> NumericDistribution:
        if isinstance(question, DateQuestion):
            lower, upper = question.lower_bound.timestamp(), question.upper_bound.timestamp()
        else:
            lower, upper = question.lower_bound, question.upper_bound
        span = upper - lower
        floor = lower if not question.open_lower_bound else lower - 0.5 * span
        ceiling = upper if not question.open_upper_bound else upper + 0.5 * span
        if question.zero_point is not None:
            floor = max(floor, question.zero_point + 1e-9 * max(abs(span), 1.0))

        clamped = [min(max(value, floor), ceiling) for value in values]
        for index in range(1, len(clamped)):
            clamped[index] = max(clamped[index], clamped[index - 1])
        percentiles = [
            Percentile(percentile=level / 100, value=value)
            for level, value in zip(prompts.PERCENTILE_LEVELS, clamped)
        ]
        distribution = NumericDistribution.from_question(percentiles, question)
        distribution.get_cdf()  # validate now so a bad distribution fails before publishing
        return distribution

    ##################################### CONDITIONAL #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        def branch_research(branch: str) -> str:
            context = prompts.CONDITIONAL_CONTEXT.format(branch=branch, parent_text=question.parent.question_text)
            return f"{context}\n\n#### RESEARCH ####\n{research}"

        yes_info, no_info = await asyncio.gather(
            self._make_prediction(question.question_yes, branch_research("YES")),
            self._make_prediction(question.question_no, branch_research("NO")),
        )
        reasoning = (
            f"## If the related question resolves YES\n{yes_info.reasoning}\n\n"
            f"## If the related question resolves NO\n{no_info.reasoning}"
        )
        return ReasonedPrediction(
            reasoning=reasoning,
            prediction_value=ConditionalPrediction(
                parent=PredictionAffirmed(),
                child=PredictionAffirmed(),
                prediction_yes=yes_info.prediction_value,  # type: ignore[arg-type]
                prediction_no=no_info.prediction_value,  # type: ignore[arg-type]
            ),
        )


def _date_to_timestamp(text: str) -> float:
    text = str(text).strip()
    if len(text) == 10:
        parsed = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()
