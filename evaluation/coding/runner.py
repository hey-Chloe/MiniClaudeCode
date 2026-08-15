"""Repo-level coding benchmark runner.

Two modes:

- ``--validate-only``: deterministic and offline. Materializes each task,
  checks the pre-state, overlays the expected fix, and runs every
  observation-free ground-truth check. No model and no network, so it is safe
  for CI.
- live: runs a real provider against each task in a fresh temp workspace and
  computes metrics from AgentResult/Trace. Requires model configuration
  (OPENAI_API_KEY / MINICLAUDE_MODEL / OPENAI_BASE_URL).
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from evaluation.coding.checkers import (
    CHECKERS,
    VALIDATION_EXCLUDED,
    CheckContext,
)
from evaluation.coding.models import CodingCaseResult, CodingReport, CodingTask
from evaluation.coding.tasks import TASKS, TASKS_BY_ID
from evaluation.reporting import save_report
from miniclaude.agent import Agent
from miniclaude.config import AppConfig
from miniclaude.context import ContextConfig
from miniclaude.llm import OpenAIProvider, OpenAIProviderConfig
from miniclaude.metrics import CostCalculator, Pricing
from miniclaude.runtime_tools import create_runtime_tools
from runtime import DockerRuntime, LocalProcessRuntime
from security.policy import PermissionModePolicy


def materialize(task: CodingTask, directory: Path) -> None:
    for relative, content in task.files.items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def snapshot_files(directory: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = path.read_text(
                encoding="utf-8", errors="replace"
            )
    return files


def _write_overlay(directory: Path, overlay: Mapping[str, str]) -> None:
    for relative, content in overlay.items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def validate_tasks(tasks: tuple[CodingTask, ...]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for task in tasks:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            materialize(task, workspace)
            initial = snapshot_files(workspace)
            ctx = CheckContext(workspace, (), "", initial)
            pre = {
                name: bool(
                    CHECKERS[name](ctx, task.ground_truth.get(name, {}))
                )
                for name in task.pre_checks
            }
            _write_overlay(workspace, task.expected_files)
            _write_overlay(workspace, task.hidden_tests)
            post: dict[str, bool] = {}
            skipped: list[str] = []
            for name, params in task.ground_truth.items():
                if name in VALIDATION_EXCLUDED:
                    skipped.append(name)
                    continue
                post[name] = bool(CHECKERS[name](ctx, params))
            passed = all(pre.values()) and all(post.values())
            cases.append(
                {
                    "id": task.id,
                    "category": task.category,
                    "passed": passed,
                    "pre": pre,
                    "post": post,
                    "skipped": skipped,
                }
            )
    return {
        "version": "1.0",
        "run_type": "validate",
        "total": len(cases),
        "passed": sum(1 for case in cases if case["passed"]),
        "cases": cases,
    }


def compute_first_pass(events: list[dict[str, Any]]) -> bool:
    """True when the first green pytest run had no edits or failed runs after.

    A green run is an execute_command observation whose argv mentions pytest
    and whose exit code is 0 without timing out.
    """
    saw_green = False
    for event in events:
        if event.get("event") != "tool_results":
            continue
        for observation in event.get("detail") or []:
            name = observation.get("name")
            output = observation.get("output")
            if name == "execute_command" and isinstance(output, dict):
                argv_text = " ".join(output.get("argv") or [])
                if "pytest" not in argv_text:
                    continue
                if output.get("exit_code") == 0 and not output.get("timed_out"):
                    saw_green = True
                elif saw_green:
                    return False
            elif name in {"write_file", "replace_text"} and observation.get(
                "success"
            ):
                if saw_green:
                    return False
    return saw_green


def _run_live_case(
    task: CodingTask,
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    max_turns: int,
    timeout: float,
    max_retries: int,
    permission_mode: str,
    runtime_name: str,
    docker_image: str,
    skills_dir: Path | None,
    cost_calculator: CostCalculator | None,
    auto_approve: bool,
    strategy=None,
    provider_factory: Callable[[], Any] | None = None,
) -> CodingCaseResult:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        materialize(task, workspace)
        initial = snapshot_files(workspace)

        runtime_class = (
            DockerRuntime if runtime_name == "docker" else LocalProcessRuntime
        )
        runtime = (
            runtime_class(
                workspace,
                default_timeout=timeout,
                image=docker_image,
            )
            if runtime_name == "docker"
            else runtime_class(
                workspace,
                default_timeout=timeout,
            )
        )
        if provider_factory is not None:
            provider = provider_factory()
        else:
            provider = OpenAIProvider(
                OpenAIProviderConfig(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_base_delay=(
                        strategy.retry_base_delay
                        if strategy is not None
                        else 0.5
                    ),
                )
            )

        def approve(request, decision) -> bool:
            return True

        agent = Agent(
            provider=provider,
            tools=create_runtime_tools(
                runtime,
                cache_enabled=(
                    strategy.read_cache_enabled
                    if strategy is not None
                    else True
                ),
            ),
            max_turns=max_turns,
            security_policy=PermissionModePolicy(permission_mode),
            approval_callback=approve if auto_approve else None,
            context_config=_context_config(strategy, workspace, skills_dir),
            cost_calculator=cost_calculator,
            tool_gating=(
                strategy.tool_gating if strategy is not None else True
            ),
            plan_first=(
                strategy.plan_first if strategy is not None else True
            ),
        )
        result = agent.run_result(task.task)

        _write_overlay(workspace, task.hidden_tests)
        observations = tuple(
            observation
            for event in result.events
            if event.get("event") == "tool_results"
            for observation in event.get("detail") or []
        )
        ctx = CheckContext(
            workspace=workspace,
            observations=observations,
            final_output=str(result.output or ""),
            initial_files=initial,
        )
        checks = {
            name: bool(CHECKERS[name](ctx, params))
            for name, params in task.ground_truth.items()
        }
        passed = result.status.value == "completed" and all(checks.values())
        metrics = result.metrics

        policy_checks_total = len(task.expected_policy)
        policy_checks_matched = sum(
            1
            for tool, action in task.expected_policy.items()
            if any(
                observation.get("name") == tool
                and observation.get("policy_action") == action
                for observation in observations
            )
        )
        security_blocks = sum(
            1
            for observation in observations
            if observation.get("policy_action") == "deny"
            or "workspace" in (observation.get("error") or "")
        )

        return CodingCaseResult(
            id=task.id,
            category=task.category,
            status=result.status.value,
            passed=passed,
            checks=checks,
            turns=metrics.turns if metrics is not None else 0,
            tool_calls=metrics.tool_calls if metrics is not None else 0,
            tool_successes=metrics.tool_successes if metrics is not None else 0,
            tool_success_rate=(
                metrics.tool_success_rate if metrics is not None else None
            ),
            policy_actions=(
                dict(metrics.policy_actions) if metrics is not None else {}
            ),
            total_reads=metrics.total_reads if metrics is not None else 0,
            repeated_reads=metrics.repeated_reads if metrics is not None else 0,
            repeated_read_rate=(
                metrics.repeated_read_rate if metrics is not None else None
            ),
            input_tokens=metrics.input_tokens if metrics is not None else 0,
            output_tokens=metrics.output_tokens if metrics is not None else 0,
            total_tokens=metrics.total_tokens if metrics is not None else 0,
            context_truncated=(
                metrics.context_truncated if metrics is not None else False
            ),
            latency_seconds=(
                metrics.duration_seconds if metrics is not None else 0.0
            ),
            cost_usd=metrics.cost_usd if metrics is not None else None,
            skills_loaded=result.skills,
            first_pass=compute_first_pass(result.events),
            security_blocks=security_blocks,
            policy_checks_matched=policy_checks_matched,
            policy_checks_total=policy_checks_total,
            recoverable_failures=(
                metrics.recoverable_failures if metrics is not None else 0
            ),
            recovered_failures=(
                metrics.recovered_failures if metrics is not None else 0
            ),
            recovery_rate=(
                metrics.recovery_rate if metrics is not None else None
            ),
            tools_sent=metrics.tools_sent if metrics is not None else 0,
            error=result.error if result.status.value != "completed" else None,
        )


def _context_config(strategy, workspace: Path, skills_dir):
    kwargs = dict(
        workspace=workspace,
        skills_dir=skills_dir,
        max_chars=(
            strategy.context_max_chars if strategy is not None else 32_000
        ),
        max_skills=strategy.skill_top_k if strategy is not None else 1,
        routing_mode=(
            strategy.routing_mode if strategy is not None else "hybrid"
        ),
        micro_compact_max_chars=(
            strategy.micro_compact_max_chars
            if strategy is not None
            else 4_000
        ),
    )
    if strategy is not None:
        kwargs["system_instructions"] = strategy.system_instructions
    return ContextConfig(**kwargs)


def _select_tasks(args: argparse.Namespace) -> tuple[CodingTask, ...]:
    if args.tasks:
        selected = tuple(TASKS_BY_ID[tid] for tid in args.tasks.split(","))
    elif args.category:
        selected = tuple(
            task for task in TASKS if task.category == args.category
        )
    else:
        selected = TASKS
    if args.limit:
        selected = selected[: args.limit]
    return selected


def _persist_report(
    rendered: str,
    payload: dict[str, Any],
    output: str | None,
) -> None:
    """Write to ``--output`` when requested and always keep a versioned copy."""
    if output:
        Path(output).write_text(rendered + "\n", encoding="utf-8")
    target = save_report(
        f"benchmark-{payload.get('run_type', 'report')}",
        payload,
    )
    print(f"report saved: {target}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the MiniClaudeCode repo-level coding benchmark"
    )
    parser.add_argument("--validate-only", action="store_true")
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

    if args.validate_only:
        report = validate_tasks(tasks)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered)
        _persist_report(rendered, report, args.output)
        return 0 if report["passed"] == report["total"] else 1

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

    cases = tuple(
        _run_live_case(
            task,
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
        )
        for task in tasks
    )
    report = CodingReport.aggregate(
        cases, run_type="live", model=config.model
    )
    rendered = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    print(rendered)
    _persist_report(rendered, report.to_dict(), args.output)
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
