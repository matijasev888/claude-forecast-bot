# Claude forecast bot

A forecasting bot for the [Metaculus FutureEval bot tournament](https://www.metaculus.com/futureeval/) and MiniBench, powered by Claude through the [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI.

## How it works

For every new tournament question:

1. **Research**: one Claude run (default `sonnet`) with web search and page fetching writes a dated research brief: resolution mechanics, current status, base rates, upcoming events, prediction-market and expert views, key uncertainties, sources.
2. **Forecast**: several independent Claude runs (default `opus, opus, sonnet`) read the brief and return a structured forecast with a written rationale.
3. **Combine**: binary forecasts are averaged in log-odds space and clipped to 1.5%–98.5%; multiple-choice probabilities are averaged with a small floor; numeric and date forecasts average 13 percentiles (p1–p99) and are turned into a full CDF with [`forecasting-tools`](https://github.com/Metaculus/forecasting-tools).
4. **Publish**: the forecast and a private comment with the research and every rationale are posted through the Metaculus API.

The main tournament is prioritised over MiniBench, and soonest-closing questions go first. If the Claude subscription's usage limit is reached, the run stops cleanly and the remaining questions are picked up by the next run.

## Running it

GitHub Actions runs `.github/workflows/forecast.yaml` every 20 minutes. It needs two repository secrets:

| Secret | Where it comes from |
|---|---|
| `METACULUS_TOKEN` | metaculus.com → Settings → My Forecasting Bots → your bot's API token |
| `CLAUDE_CODE_OAUTH_TOKEN` | `claude setup-token` (requires a Claude subscription) |

`.github/workflows/test_bot.yaml` (manual) runs the unit tests and forecasts the [bot-testing-area](https://www.metaculus.com/tournament/bot-testing-area/), by default without publishing.

Locally:

```bash
uv sync
uv run pytest -q
uv run python main.py --mode test --dry-run   # needs METACULUS_TOKEN in .env and a logged-in `claude` CLI
```

Optional settings (environment variables or repository variables): `RESEARCH_MODEL`, `FORECAST_MODELS` (comma-separated), `FORECAST_EFFORT`, `MAX_QUESTIONS_PER_RUN`, `QUESTION_CONCURRENCY`, `CLAUDE_MAX_PARALLEL`, `SKIP_MINIBENCH`.

## Tournament rules

The bot runs with no human in the loop: it forecasts each tournament question once, directly through the API, and posts private comments with its reasoning. Development testing happens only on the bot-testing-area.

## Credits

Built on Metaculus's MIT-licensed [`forecasting-tools`](https://github.com/Metaculus/forecasting-tools); the overall setup follows Metaculus's [bot template](https://github.com/Metaculus/metac-bot-template). Developed with Claude Code.
