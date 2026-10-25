"""Prompts and JSON schemas for the research and forecasting calls."""

from __future__ import annotations

from datetime import datetime, timezone

from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

PERCENTILE_LEVELS = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)

RESEARCH_SYSTEM_PROMPT = """\
You are the research assistant for a professional forecasting team. You gather and organize evidence; you never give the final forecast yourself.

Use the WebSearch and WebFetch tools. Be efficient: roughly 5-12 searches, and fetch only pages that look decisive (official sources, data releases, reputable news). Prefer the newest information and always note the publication date of each fact. Never invent facts, numbers or sources; if you cannot find something, say so.

Your brief must cover:
1. Resolution mechanics: what exactly must happen under the resolution criteria and fine print, and which source decides it. Flag subtle wording.
2. Current status: the latest relevant facts and data, each with its date.
3. Base rates: reference classes, historical frequencies, typical timelines.
4. What is scheduled between now and the resolution date (votes, releases, deadlines, meetings).
5. Markets and expert views: prices or forecasts on the same or closely related questions (Polymarket, Kalshi, Manifold, Metaculus, official forecasts, analyst consensus), with dates.
6. For numeric or date questions: the latest values, recent trend, volatility, and any consensus estimates.
7. Whether the outcome already looks determined (the event already happened, or has become impossible).

Write the brief in Markdown, under 900 words, with exactly these headings:
## Resolution mechanics
## Current status
## Base rates
## Upcoming events
## Markets and expert views
## Key uncertainties
## Sources
"""

FORECAST_SYSTEM_PROMPT = """\
You are an expert forecaster with an excellent calibration record, competing in a forecasting tournament scored with a proper (logarithmic) scoring rule, so honest, well-calibrated probabilities maximize your score.

Principles:
- Read the resolution criteria and fine print literally; outcomes often hinge on exact wording, data sources and deadlines.
- Start from base rates (the outside view), then adjust for the specific evidence (the inside view).
- Respect the status quo: most things change more slowly than news coverage suggests. For "will X happen by <date>" questions, weigh the chance per week against the time remaining.
- Give substantial weight to liquid prediction markets and to official or consensus forecasts when they address the question directly.
- If the evidence shows the outcome is already effectively determined, forecast accordingly, after checking that it truly satisfies the resolution criteria.
- Avoid both overconfidence and timid hedging. Probabilities below 3% or above 97% are for outcomes that are nearly settled.
- The research brief was gathered today and may contain mistakes; weigh its claims by their sources.

Respond only through the provided JSON schema. The "rationale" field is a concise written justification (100-250 words) a reader can follow: base rate, key evidence, status quo, and the main reasons for your final answer.
"""


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _fmt_time(value: datetime | None) -> str:
    if value is None:
        return "not stated"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _number(value: float | None) -> str:
    if value is None:
        return "not stated"
    return f"{value:,.6g}"


def describe_question_type(question: MetaculusQuestion) -> str:
    if isinstance(question, BinaryQuestion):
        return "Binary (resolves Yes or No)."
    if isinstance(question, MultipleChoiceQuestion):
        options = "\n".join(f"- {option}" for option in question.options)
        return f"Multiple choice. Exactly one of these options will be correct:\n{options}"
    if isinstance(question, DateQuestion):
        return (
            "Date (the answer is a date).\n"
            f"- Range on the question: {question.lower_bound.date().isoformat()} "
            f"to {question.upper_bound.date().isoformat()}\n"
            f"- {_bound_sentence(question.open_lower_bound, 'lower', 'before ' + question.lower_bound.date().isoformat())}\n"
            f"- {_bound_sentence(question.open_upper_bound, 'upper', 'after ' + question.upper_bound.date().isoformat())}"
        )
    if isinstance(question, NumericQuestion):
        lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
        upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        unit = question.unit_of_measure or "not stated (infer from the question)"
        lines = [
            "Numeric (the answer is a number).",
            f"- Units: {unit}",
            f"- Range on the question: {_number(lower)} to {_number(upper)}",
            f"- {_bound_sentence(question.open_lower_bound, 'lower', 'below ' + _number(lower))}",
            f"- {_bound_sentence(question.open_upper_bound, 'upper', 'above ' + _number(upper))}",
        ]
        if question.question_type == "discrete" or (question.cdf_size and question.cdf_size < 201):
            lines.append("- The outcome takes discrete values (e.g. whole numbers).")
        if question.zero_point is not None:
            lines.append("- The question uses a logarithmic scale.")
        return "\n".join(lines)
    return f"{type(question).__name__}"


def _bound_sentence(is_open: bool, side: str, beyond: str) -> str:
    if is_open:
        return f"The {side} bound is open: outcomes {beyond} are possible but the question creator thinks they are unlikely."
    return f"The {side} bound is closed: outcomes {beyond} are impossible (or resolve at the bound)."


def question_block(question: MetaculusQuestion) -> str:
    group = (
        f'\nThis question is the sub-question for "{question.group_question_option}" within a group of related questions.\n'
        if question.group_question_option
        else ""
    )
    return f"""\
# Question
{question.question_text}
{group}
# Question type
{describe_question_type(question)}

# Background
{question.background_info or "(none given)"}

# Resolution criteria
{question.resolution_criteria or "(none given)"}

# Fine print
{question.fine_print or "(none given)"}

# Timing
Forecasting closes: {_fmt_time(question.close_time)}
Scheduled resolution: {_fmt_time(question.scheduled_resolution_time)}
"""


def research_prompt(question: MetaculusQuestion) -> str:
    return f"""\
Today's date (UTC): {_today()}

Research this forecasting question and write the brief described in your instructions.

{question_block(question)}"""


def forecast_prompt(question: MetaculusQuestion, research: str, task: str, extra_context: str = "") -> str:
    return f"""\
Today's date (UTC): {_today()}

{question_block(question)}
{extra_context}
# Research brief (gathered today by your research assistant)
{research or "(no research available)"}

# Your task
{task}
"""


BINARY_TASK = "Give the probability (between 0 and 1) that this question resolves Yes."
MULTIPLE_CHOICE_TASK = (
    "Give a probability for every option; the probabilities must sum to 1. "
    "Leave some probability on less likely options to account for surprises."
)
NUMERIC_TASK = (
    "Give your forecast as percentiles of the outcome, in the question's units, strictly increasing "
    "from p1 to p99. Never use scientific notation. Set wide tails (p1/p99, p5/p95) to cover unknown unknowns."
)
DATE_TASK = (
    "Give your forecast as percentiles of the date the outcome occurs, as YYYY-MM-DD dates, "
    "strictly increasing from p1 to p99. Set wide tails (p1/p99, p5/p95) to cover unknown unknowns."
)


def binary_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "probability": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["rationale", "probability"],
        "additionalProperties": False,
    }


def multiple_choice_schema(options: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "probabilities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "option": {"type": "string", "enum": list(options)},
                        "probability": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["option", "probability"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["rationale", "probabilities"],
        "additionalProperties": False,
    }


def percentile_schema(value_type: str) -> dict:
    value_schema = {"type": "number"} if value_type == "number" else {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}"}
    keys = [f"p{level}" for level in PERCENTILE_LEVELS]
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "percentiles": {
                "type": "object",
                "properties": {key: value_schema for key in keys},
                "required": keys,
                "additionalProperties": False,
            },
        },
        "required": ["rationale", "percentiles"],
        "additionalProperties": False,
    }


CONDITIONAL_CONTEXT = """\
# Conditional question
This is one branch of a conditional question. Assume the following related question resolves {branch}:
"{parent_text}"
Forecast only the question above under that assumption; do not forecast the related question itself.
"""
