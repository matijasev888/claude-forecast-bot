import asyncio
import json
import math
from datetime import datetime, timezone

import pytest
from forecasting_tools import DateQuestion, NumericQuestion

import claude_cli
import prompts
from bot import ClaudeForecaster, _date_to_timestamp


def make_numeric(open_lower=False, open_upper=True, lower=0.0, upper=100.0, cdf_size=201):
    return NumericQuestion(
        question_text="How many widgets?",
        id_of_question=1,
        id_of_post=1,
        upper_bound=upper,
        lower_bound=lower,
        open_upper_bound=open_upper,
        open_lower_bound=open_lower,
        unit_of_measure="widgets",
        cdf_size=cdf_size,
    )


def make_date():
    return DateQuestion(
        question_text="When will it happen?",
        id_of_question=2,
        id_of_post=2,
        upper_bound=datetime(2027, 12, 31, tzinfo=timezone.utc),
        lower_bound=datetime(2026, 11, 1, tzinfo=timezone.utc),
        open_upper_bound=True,
        open_lower_bound=False,
    )


def test_combine_binary_is_log_odds_mean_and_clipped():
    assert ClaudeForecaster.combine_binary([0.5, 0.5, 0.5]) == pytest.approx(0.5)
    combined = ClaudeForecaster.combine_binary([0.2, 0.3, 0.4])
    logits = [math.log(p / (1 - p)) for p in (0.2, 0.3, 0.4)]
    assert combined == pytest.approx(1 / (1 + math.exp(-sum(logits) / 3)))
    assert ClaudeForecaster.combine_binary([0.0, 0.001, 0.0]) == ClaudeForecaster.min_binary_probability
    assert ClaudeForecaster.combine_binary([1.0, 0.999]) == ClaudeForecaster.max_binary_probability


def test_combine_options_floors_and_normalizes():
    options = ["A", "B", "C"]
    final = ClaudeForecaster.combine_options([{"A": 1.0, "B": 0.0, "C": 0.0}, {"A": 0.8, "B": 0.2, "C": 0.0}], options)
    assert sum(final.values()) == pytest.approx(1.0)
    assert final["C"] >= ClaudeForecaster.min_option_probability * 0.99
    assert final["A"] > final["B"] > final["C"]


def test_average_percentiles_monotone():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    b = [3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20]
    averaged = ClaudeForecaster.average_percentiles([a, b])
    assert averaged[0] == pytest.approx(2)
    assert averaged[-1] == pytest.approx(16.5)
    assert all(x <= y for x, y in zip(averaged, averaged[1:]))


@pytest.mark.parametrize("open_lower,open_upper", [(False, False), (False, True), (True, True)])
def test_build_distribution_valid_cdf(open_lower, open_upper):
    question = make_numeric(open_lower=open_lower, open_upper=open_upper)
    values = [2, 8, 15, 25, 33, 40, 47, 54, 61, 70, 82, 90, 98]
    distribution = ClaudeForecaster.build_distribution(values, question)
    cdf = [p.percentile for p in distribution.get_cdf()]
    assert len(cdf) == 201
    assert all(x <= y for x, y in zip(cdf, cdf[1:]))
    if not open_lower:
        assert cdf[0] == pytest.approx(0.0, abs=1e-9)
    if not open_upper:
        assert cdf[-1] == pytest.approx(1.0, abs=1e-9)


def test_build_distribution_handles_values_outside_closed_bounds():
    question = make_numeric(open_lower=False, open_upper=False)
    values = [-50, -20, -5, 1, 2, 3, 4, 5, 6, 7, 150, 200, 400]
    distribution = ClaudeForecaster.build_distribution(values, question)
    cdf = [p.percentile for p in distribution.get_cdf()]
    assert cdf[-1] == pytest.approx(1.0, abs=1e-9)


def test_build_distribution_discrete_question():
    question = make_numeric(open_lower=False, open_upper=True, lower=-0.5, upper=20.5, cdf_size=22)
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 17]
    distribution = ClaudeForecaster.build_distribution(values, question)
    assert len(distribution.get_cdf()) == 22


def test_build_distribution_date_question():
    question = make_date()
    dates = ["2026-11-15", "2026-12-01", "2026-12-20", "2027-01-15", "2027-02-10", "2027-03-01",
             "2027-03-20", "2027-04-15", "2027-05-10", "2027-06-15", "2027-08-01", "2027-10-01", "2028-03-01"]
    values = [_date_to_timestamp(d) for d in dates]
    distribution = ClaudeForecaster.build_distribution(values, question)
    cdf = [p.percentile for p in distribution.get_cdf()]
    assert cdf[0] == pytest.approx(0.0, abs=1e-9)
    assert all(x <= y for x, y in zip(cdf, cdf[1:]))


def test_date_parsing():
    assert _date_to_timestamp("2027-01-01") == datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp()
    assert _date_to_timestamp("2027-01-01T12:00:00Z") == datetime(2027, 1, 1, 12, tzinfo=timezone.utc).timestamp()


def test_split_condition_roundtrip():
    research = prompts.CONDITIONAL_CONTEXT.format(branch="YES", parent_text="Will X?") + "\n\n#### RESEARCH ####\nBrief"
    context, rest = ClaudeForecaster._split_condition(research)
    assert "resolves YES" in context and rest == "Brief"
    assert ClaudeForecaster._split_condition("Just research") == ("", "Just research")


def test_schemas_are_json_serializable():
    for schema in (
        prompts.binary_schema(),
        prompts.multiple_choice_schema(["A", "B"]),
        prompts.percentile_schema("number"),
        prompts.percentile_schema("date"),
    ):
        json.dumps(schema)


def test_parse_cli_json_tolerates_leading_noise():
    payload = {"result": "ok", "is_error": False}
    assert claude_cli._parse_cli_json("warning line\n" + json.dumps(payload)) == payload


def test_search_keywords_drop_stopwords_dates_and_small_numbers():
    import signals

    keywords = signals.search_keywords("Will the Federal Reserve cut rates at its December 2026 meeting before January 15?")
    assert keywords == "Federal Reserve cut rates meeting"


def test_market_context_survives_api_failures(monkeypatch):
    import signals

    def boom(query):
        raise OSError("network down")

    monkeypatch.setattr(signals, "manifold_lines", boom)
    monkeypatch.setattr(signals, "polymarket_lines", lambda query: ["- [Polymarket, real money] \"Q\": Yes: 40%"])
    context = asyncio.run(signals.market_context("Will something notable happen?"))
    assert "Polymarket" in context and "Manifold" not in context.split("\n", 1)[1]


def test_usage_limit_short_circuits(monkeypatch):
    monkeypatch.setattr(claude_cli, "usage_limit_hit", True)
    with pytest.raises(claude_cli.ClaudeUsageLimitError):
        asyncio.run(claude_cli.run_claude("hi", model="sonnet", system_prompt="x"))
