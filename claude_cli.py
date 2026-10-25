"""
Async wrapper around the Claude Code CLI in headless mode (`claude -p`).

The bot runs on a Claude subscription rather than API keys: in GitHub Actions
the CLAUDE_CODE_OAUTH_TOKEN secret (created locally with `claude setup-token`)
authenticates the CLI. Locally, whatever account the CLI is logged in with is used.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Substrings (lowercased) that identify a subscription usage/rate limit in the CLI's result text.
_LIMIT_MARKERS = (
    "usage limit",
    "limit reached",
    "hit your limit",
    "rate limit",
    "rate_limit",
    "quota",
)
# Substrings that identify a safety-classifier refusal; another model may answer.
_REFUSAL_MARKERS = (
    "safeguards flagged",
    "usage policy",
)


class ClaudeCliError(RuntimeError):
    """The CLI failed or returned an error result."""


class ClaudeUsageLimitError(ClaudeCliError):
    """The subscription's usage limit was hit; no new work should start this run."""


class ClaudeRefusalError(ClaudeCliError):
    """A safety classifier declined the request; retrying with another model may help."""


@dataclass
class ClaudeResult:
    text: str
    structured: dict[str, Any] | None
    cost_usd: float
    num_turns: int
    web_searches: int
    model: str
    raw: dict[str, Any] = field(repr=False)


_semaphore: asyncio.Semaphore | None = None
usage_limit_hit = False


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(int(os.getenv("CLAUDE_MAX_PARALLEL", "3")))
    return _semaphore


def _claude_binary() -> str:
    return os.getenv("CLAUDE_BIN") or shutil.which("claude") or "claude"


def _parse_cli_json(stdout: str) -> dict[str, Any]:
    stdout = stdout.strip()
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        # Tolerate stray log lines before the final JSON object.
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        raise


async def run_claude(
    prompt: str,
    *,
    model: str,
    system_prompt: str,
    tools: Sequence[str] = (),
    json_schema: dict[str, Any] | None = None,
    effort: str | None = None,
    timeout_seconds: float = 600,
) -> ClaudeResult:
    """
    Run one headless Claude Code turn and return its result.

    `tools` lists the built-in tools Claude may use (e.g. WebSearch, WebFetch);
    an empty list disables tools. With `json_schema`, the reply is validated
    against the schema and returned in `structured`.
    """
    global usage_limit_hit
    if usage_limit_hit:
        raise ClaudeUsageLimitError("Usage limit was already reached earlier in this run")

    args = [
        _claude_binary(),
        "-p",
        "--output-format",
        "json",
        "--model",
        model,
        "--no-session-persistence",
        "--system-prompt",
        system_prompt,
        "--tools",
        ",".join(tools),
    ]
    if tools:
        args += ["--allowedTools", ",".join(tools)]
    if json_schema is not None:
        args += ["--json-schema", json.dumps(json_schema)]
    if effort:
        args += ["--effort", effort]

    async with _get_semaphore():
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise ClaudeCliError(f"claude ({model}) timed out after {timeout_seconds:.0f}s")

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    try:
        data = _parse_cli_json(stdout)
    except json.JSONDecodeError:
        raise ClaudeCliError(
            f"claude ({model}) returned non-JSON output (exit {process.returncode}): "
            f"{stdout[-400:]!r} {stderr[-400:]!r}"
        )

    result_text = str(data.get("result") or "")
    if data.get("is_error") or process.returncode != 0:
        lowered = f"{result_text} {stderr}".lower()
        if any(marker in lowered for marker in _LIMIT_MARKERS):
            usage_limit_hit = True
            raise ClaudeUsageLimitError(result_text or stderr)
        if any(marker in lowered for marker in _REFUSAL_MARKERS):
            raise ClaudeRefusalError(f"claude ({model}) refused: {result_text[:300]}")
        raise ClaudeCliError(f"claude ({model}) error: {(result_text or stderr)[:600]}")

    structured = data.get("structured_output")
    if json_schema is not None and not isinstance(structured, dict):
        try:
            structured = json.loads(result_text)
        except json.JSONDecodeError:
            # Model output stays out of exception text: run logs of a public repo are public.
            raise ClaudeCliError(f"claude ({model}) gave no structured output ({len(result_text)} chars of text)")

    model_usage = data.get("modelUsage") or {}
    web_searches = sum(int(entry.get("webSearchRequests") or 0) for entry in model_usage.values())
    result = ClaudeResult(
        text=result_text,
        structured=structured,
        cost_usd=float(data.get("total_cost_usd") or 0.0),
        num_turns=int(data.get("num_turns") or 0),
        web_searches=int(web_searches),
        model=model,
        raw=data,
    )
    logger.info(
        f"claude {model}: {result.num_turns} turns, {result.web_searches} searches, "
        f"~${result.cost_usd:.3f} equivalent"
    )
    return result
