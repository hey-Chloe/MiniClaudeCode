"""asyncio concurrent live benchmark runner.

The agent loop itself stays synchronous (deterministic and testable); this
runner is the asyncio orchestration layer: a bounded ``asyncio.Semaphore``
keeps ``--concurrency`` live tasks in flight, each task's sync agent loop runs
inside ``asyncio.to_thread``, and the LLM I/O goes through
``AsyncOpenAIProvider`` bridged by ``RunInLoopProvider`` (one provider and
event loop per worker thread). The report is aggregated with the same
``CodingReport`` used by the synchronous runner, so artifacts are comparable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.coding.models import CodingReport
from evaluation.coding.runner import (
    _persist_report,
    _run_live_case,
    _select_tasks,
)
from miniclaude.config import AppConfig
from miniclaude.llm import AsyncOpenAIProvider, OpenAIProviderConfig, RunInLoopProvider
from miniclaude.metrics import CostCalculator, Pricing


def _async_provider_factory(config: AppConfig):
    """One bridge provider per worker thread, each with its own event loop."""

    def factory() -> RunInLoopProvider:
        async_provider = AsyncOpenAIProvider(
            OpenAIProviderConfig(
                model=config.model or "",
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
        )
        return RunInLoopProvider(async_provider)

    return factory


async def run_tasks_concurrently(
    tasks,
    *,
    concurrency: int,
    case_kwargs: dict[str, Any],
) -> tuple:
    """Run live cases with bounded asyncio concurrency."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def run_one(task):
        nonlocal active, peak
        async with semaphore:
            async with lock:
                active += 1
                peak = max(peak, active)
            try:
                return await asyncio.to_thread(
                    _run_live_case, task, **case_kwargs
                )
            finally:
                async with lock:
                    active -= 1

    results = await asyncio.gather(*(run_one(task) for task in tasks))
    return tuple(results), peak


def main() -> int:
    # Windows proactor teardown emits WinError 6 "Cancelling an overlapped
    # future failed" noise when httpx connections are closed with the loop;
    # the selector policy is the standard fix and keeps the same semantics.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )
    parser = argparse.ArgumentParser(
        description=(
            "Run the coding benchmark concurrently with asyncio; LLM I/O uses "
            "AsyncOpenAIProvider."
        )
    )
    parser.add_argument("--tasks", help="comma-separated task ids")
    parser.add_argument(
        "--category",
        choices={
            "failing_test_fix",
            "small_feature",
            "code_search",
            "safe_refactor",
            "config_repair",
            "dependency_issue",
            "permission_security",
        },
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--runtime", choices=["local", "docker"], default="local")
    parser.add_argument("--docker-image", default="python:3.12-slim")
    parser.add_argument(
        "--permission-mode",
        choices=["default", "plan", "accept-edits", "bypass"],
        default="default",
    )
    parser.add_argument("--approvals", choices=["auto", "none"], default="auto")
    parser.add_argument("--skills-dir", default=None)
    parser.add_argument("--no-skills", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    tasks = _select_tasks(args)
    if not tasks:
        print(json.dumps({"error": "no tasks selected"}, ensure_ascii=False))
        return 2

    config = AppConfig.from_env(model=args.model, workspace=Path.cwd())
    if not config.model:
        print(
            "live benchmark requires --model or MINICLAUDE_MODEL",
            file=sys.stderr,
        )
        return 2
    skills_dir: Path | None = None
    if not args.no_skills:
        if args.skills_dir:
            skills_dir = Path(args.skills_dir)
        else:
            candidate = Path(__file__).resolve().parents[2] / "skills"
            skills_dir = candidate if candidate.is_dir() else None
    cost_calculator = None
    if (
        config.input_price_per_million is not None
        and config.output_price_per_million is not None
    ):
        cost_calculator = CostCalculator(
            {
                config.model: Pricing(
                    input_per_million=config.input_price_per_million,
                    output_per_million=config.output_price_per_million,
                )
            }
        )

    case_kwargs: dict[str, Any] = dict(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        max_turns=args.max_turns,
        timeout=args.timeout,
        max_retries=config.max_retries,
        permission_mode=args.permission_mode,
        runtime_name=args.runtime,
        docker_image=args.docker_image,
        skills_dir=skills_dir,
        cost_calculator=cost_calculator,
        auto_approve=args.approvals == "auto",
        provider_factory=_async_provider_factory(config),
    )
    cases, peak = asyncio.run(
        run_tasks_concurrently(
            tasks,
            concurrency=args.concurrency,
            case_kwargs=case_kwargs,
        )
    )
    report = CodingReport.aggregate(
        cases,
        run_type="live_async",
        model=config.model,
    )
    payload = report.to_dict()
    payload["concurrency"] = {
        "requested": args.concurrency,
        "peak_active": peak,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    _persist_report(rendered, payload, args.output)
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
