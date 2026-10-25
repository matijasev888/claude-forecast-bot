"""
Forecast new questions in the Metaculus FutureEval bot tournament (and MiniBench).

Usage:
    uv run python main.py --mode tournament          # live tournament + MiniBench (default)
    uv run python main.py --mode test --dry-run      # bot-testing-area, nothing published
    uv run python main.py --mode test                # bot-testing-area, published (safe to resubmit)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import warnings
from datetime import datetime, timezone

from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=r".*does not support cost tracking.*")
load_dotenv()

import claude_cli  # noqa: E402
from bot import ClaudeForecaster  # noqa: E402
from forecasting_tools import ForecastReport, MetaculusClient, MetaculusQuestion  # noqa: E402

logger = logging.getLogger("main")

TEST_TOURNAMENT = "bot-testing-area"


def collect_questions(
    client: MetaculusClient, sources: list[int | str], include_forecasted: bool
) -> list[MetaculusQuestion]:
    """Open questions from each source in priority order, soonest-closing first within a source."""
    far_future = datetime.max.replace(tzinfo=timezone.utc)
    selected: list[MetaculusQuestion] = []
    seen: set[int | None] = set()
    for source in sources:
        try:
            questions = client.get_all_open_questions_from_tournament(source)
        except Exception as error:
            logger.error(f"Could not load questions from {source}: {error}")
            continue
        fresh = [q for q in questions if include_forecasted or not q.already_forecasted]
        fresh.sort(key=lambda q: q.close_time or far_future)
        logger.info(f"{source}: {len(questions)} open, {len(fresh)} to forecast")
        for question in fresh:
            if question.id_of_question in seen:
                continue
            seen.add(question.id_of_question)
            selected.append(question)
    return selected


async def forecast_all(
    bot: ClaudeForecaster, questions: list[MetaculusQuestion], concurrency: int
) -> list[ForecastReport | BaseException]:
    semaphore = asyncio.Semaphore(concurrency)

    async def forecast_one(question: MetaculusQuestion) -> ForecastReport | BaseException:
        async with semaphore:
            if claude_cli.usage_limit_hit:
                return claude_cli.ClaudeUsageLimitError(f"Skipped {question.page_url}: usage limit reached")
            logger.info(f"Forecasting {question.page_url}: {question.question_text}")
            return await bot.forecast_question(question, return_exceptions=True)

    return await asyncio.gather(*(forecast_one(q) for q in questions))


def write_step_summary(reports: list[ForecastReport | BaseException], published: bool, hide_values: bool) -> None:
    lines = [f"## Forecast run {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", ""]
    if not reports:
        lines.append("No new questions to forecast.")
    for report in reports:
        if isinstance(report, ForecastReport):
            verb = "published" if published else "dry run"
            # Public repos expose run logs, so live tournament values stay out of them.
            readable = (
                "(value hidden)"
                if hide_values
                else type(report).make_readable_prediction(report.prediction).replace("\n", " ")
            )
            lines.append(f"- ✅ ({verb}) [{report.question.question_text}]({report.question.page_url}): {readable}")
        else:
            lines.append(f"- ❌ {type(report).__name__}: {str(report)[:300]}")
    text = "\n".join(lines) + "\n"
    print(text)
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["tournament", "test", "cup"], default="tournament")
    parser.add_argument("--dry-run", action="store_true", help="Forecast but do not publish anything")
    parser.add_argument("--max-questions", type=int, default=int(os.getenv("MAX_QUESTIONS_PER_RUN", "6")))
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("QUESTION_CONCURRENCY", "2")))
    parser.add_argument("--no-minibench", action="store_true", help="Skip MiniBench in tournament mode")
    parser.add_argument("--reports-dir", default=os.getenv("REPORTS_DIR", "reports"))
    parser.add_argument(
        "--check-only", action="store_true", help="Only count questions to forecast (writes count to GITHUB_OUTPUT)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    if not (os.getenv("METACULUS_TOKEN") or "").strip():
        print("METACULUS_TOKEN is not set. Create a bot at https://www.metaculus.com/accounts/settings/bots/")
        return 1

    client = MetaculusClient()
    if args.mode == "tournament":
        sources: list[int | str] = [client.CURRENT_AI_COMPETITION_ID]
        if not args.no_minibench and os.getenv("SKIP_MINIBENCH", "").lower() not in {"1", "true", "yes"}:
            sources.append(client.CURRENT_MINIBENCH_ID)
    elif args.mode == "test":
        sources = [TEST_TOURNAMENT]
    else:
        sources = [client.CURRENT_METACULUS_CUP_ID]

    # The testing area and the Cup allow resubmission; the scored bot tournaments do not.
    include_forecasted = args.mode != "tournament"
    questions = collect_questions(client, sources, include_forecasted)
    if args.max_questions >= 0:
        questions = questions[: args.max_questions]

    if args.check_only:
        print(f"Questions to forecast: {len(questions)}")
        output_path = os.getenv("GITHUB_OUTPUT")
        if output_path:
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(f"count={len(questions)}\n")
        return 0

    published = not args.dry_run
    hide_values = args.mode == "tournament" and os.getenv("HIDE_FORECAST_VALUES", "").lower() in {"1", "true", "yes"}
    reports_dir = None if hide_values else args.reports_dir
    if reports_dir:
        os.makedirs(reports_dir, exist_ok=True)
    bot = ClaudeForecaster(
        publish_reports_to_metaculus=published,
        skip_previously_forecasted_questions=False,  # filtered above
        folder_to_save_reports_to=reports_dir,
        extra_metadata_in_explanation=True,
        metaculus_client=client,
    )
    print(f"Mode={args.mode} publish={published} questions={len(questions)} concurrency={args.concurrency}")
    reports = asyncio.run(forecast_all(bot, questions, args.concurrency))
    write_step_summary(reports, published, hide_values)
    if claude_cli.usage_limit_hit:
        print("Stopped early: the Claude usage limit was reached. Remaining questions will be picked up next run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
