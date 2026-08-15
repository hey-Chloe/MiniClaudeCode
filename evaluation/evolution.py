"""Benchmark-driven strategy evolution for MiniClaudeCode.

The agent itself does not evolve. The loop operates on an explicit strategy
space (skill routing depth, context budget, micro-compaction, retry policy,
tool gating, plan-first) and:

1. generates deterministic candidate variants of the base strategy;
2. scores them on a fixed training subset;
3. promotes a candidate only when it improves a fixed holdout subset without
   regressing task success (otherwise the run keeps/rolls back to the base).

Every step is persisted as a versioned report under ``reports/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from evaluation.coding.models import CodingReport
from evaluation.coding.runner import _run_live_case
from evaluation.coding.tasks import TASKS, TASKS_BY_ID
from evaluation.reporting import save_report
from miniclaude.config import AppConfig
from miniclaude.context import DEFAULT_SYSTEM_INSTRUCTIONS
from miniclaude.llm import OpenAIProvider, OpenAIProviderConfig


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """One versioned point in the evolvable strategy space.

    Every field here is actually wired into the agent at run time
    (``evaluation/coding/runner._run_live_case``), so promotion is real.
    """

    version: str
    skill_top_k: int = 1
    routing_mode: str = "hybrid"
    context_max_chars: int = 32_000
    micro_compact_max_chars: int = 4_000
    retry_max_retries: int = 2
    retry_base_delay: float = 0.5
    tool_gating: bool = True
    read_cache_enabled: bool = True
    plan_first: bool = True
    system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("strategy version must not be empty")
        if self.skill_top_k < 1:
            raise ValueError("skill_top_k must be at least 1")
        if self.routing_mode not in {"keyword", "hybrid", "semantic"}:
            raise ValueError(
                "routing_mode must be one of keyword/hybrid/semantic"
            )
        if self.context_max_chars < 1:
            raise ValueError("context_max_chars must be positive")
        if self.micro_compact_max_chars < 1:
            raise ValueError("micro_compact_max_chars must be positive")
        if self.retry_max_retries < 0:
            raise ValueError("retry_max_retries must not be negative")
        if self.retry_base_delay < 0:
            raise ValueError("retry_base_delay must not be negative")
        if not self.system_instructions.strip():
            raise ValueError("system_instructions must not be empty")


_MUTATIONS: Mapping[str, tuple[Any, ...]] = {
    "skill_top_k": (2,),
    "routing_mode": ("keyword", "semantic"),
    "context_max_chars": (24_000, 40_000),
    "micro_compact_max_chars": (2_000, 8_000),
    "retry_max_retries": (1, 3),
    "retry_base_delay": (0.2, 1.0),
    "tool_gating": (False,),
    "read_cache_enabled": (False,),
    "plan_first": (False,),
}


def generate_candidates(
    base: StrategyConfig,
    max_candidates: int = 10,
) -> tuple[StrategyConfig, ...]:
    """Deterministic template mutations around the base strategy."""
    candidates: list[StrategyConfig] = [base]
    for field_name, values in _MUTATIONS.items():
        for value in values:
            if getattr(base, field_name) == value:
                continue
            candidates.append(
                replace(
                    base,
                    version=f"cand-{field_name}-{value}",
                    **{field_name: value},
                )
            )
    return tuple(candidates[: max(1, max_candidates)])


def default_splits() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Category-stratified, deterministic training/holdout split of the 26 tasks."""
    by_category: dict[str, list[str]] = {}
    for task in TASKS:
        by_category.setdefault(task.category, []).append(task.id)
    train: list[str] = []
    holdout: list[str] = []
    for category_ids in by_category.values():
        for index, task_id in enumerate(category_ids):
            (train if index % 2 == 0 else holdout).append(task_id)
    return tuple(train), tuple(holdout)


def aggregate_metrics(report: CodingReport) -> dict[str, float]:
    """Normalize a benchmark report into the scalar score vector used here."""
    return {
        "success_rate": report.task_success_rate,
        "average_tokens": report.average_tokens,
        "average_latency_seconds": report.average_latency_seconds,
        "recovery_rate": report.recovery_rate or 0.0,
        "average_tools_sent_per_turn": report.average_tools_sent_per_turn,
    }


@dataclass(frozen=True, slots=True)
class CandidateResult:
    version: str
    train_score: Mapping[str, float]
    holdout_score: Mapping[str, float]
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "train": dict(self.train_score),
            "holdout": dict(self.holdout_score),
            "selected": self.selected,
        }


@dataclass(frozen=True, slots=True)
class GenerationResult:
    generation: int
    base_version: str
    decision: str
    base_holdout: Mapping[str, float]
    candidates: tuple[CandidateResult, ...] = ()
    promoted_version: str | None = None
    promoted_strategy: StrategyConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "base_version": self.base_version,
            "decision": self.decision,
            "promoted_version": self.promoted_version,
            "base_holdout": dict(self.base_holdout),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class EvolutionRun:
    final_version: str
    generations: tuple[GenerationResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "run_type": "evolution",
            "final_version": self.final_version,
            "generations": [
                generation.to_dict() for generation in self.generations
            ],
        }


EvaluateFn = Callable[
    [StrategyConfig, tuple[str, ...]], Mapping[str, float]
]


def _rank(score: Mapping[str, float]) -> tuple[float, float, float, float]:
    return (
        score["success_rate"],
        -score["average_tokens"],
        -score["average_latency_seconds"],
        score["recovery_rate"],
    )


def _improves_holdout(
    candidate: Mapping[str, float],
    base: Mapping[str, float],
) -> bool:
    """Promote only when success does not regress and something else improves."""
    success_ok = candidate["success_rate"] >= base["success_rate"] - 1e-9
    other_improvement = (
        candidate["average_tokens"] < base["average_tokens"]
        or candidate["average_latency_seconds"]
        < base["average_latency_seconds"]
        or candidate["recovery_rate"] > base["recovery_rate"]
    )
    return success_ok and other_improvement


def _evolve_generation(
    base: StrategyConfig,
    evaluate_fn: EvaluateFn,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    generation: int,
    max_candidates: int,
    extra_candidates: tuple[StrategyConfig, ...] = (),
) -> GenerationResult:
    base_holdout = dict(evaluate_fn(base, holdout_ids))
    candidates = generate_candidates(base, max_candidates=max_candidates)
    known = {candidate.version for candidate in candidates}
    candidates = candidates + tuple(
        candidate
        for candidate in extra_candidates
        if candidate.version not in known
    )
    results: list[CandidateResult] = []
    for candidate in candidates:
        results.append(
            CandidateResult(
                version=candidate.version,
                train_score=dict(evaluate_fn(candidate, train_ids)),
                holdout_score=dict(evaluate_fn(candidate, holdout_ids)),
            )
        )
    best = max(results, key=lambda result: _rank(result.train_score))
    best_strategy = next(
        candidate
        for candidate in candidates
        if candidate.version == best.version
    )
    if best.version == base.version:
        decision = "kept_base"
    elif _improves_holdout(best.holdout_score, base_holdout):
        decision = "promoted"
    else:
        decision = "kept_base"  # rejected candidate; run rolls back to the base

    return GenerationResult(
        generation=generation,
        base_version=base.version,
        decision=decision,
        base_holdout=base_holdout,
        candidates=tuple(
            replace(result, selected=result.version == best.version)
            for result in results
        ),
        promoted_version=best.version if decision == "promoted" else None,
        promoted_strategy=(
            best_strategy if decision == "promoted" else None
        ),
    )


def evolve(
    base: StrategyConfig,
    evaluate_fn: EvaluateFn,
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    *,
    generations: int = 1,
    max_candidates: int = 10,
    extra_candidates: tuple[StrategyConfig, ...] = (),
) -> EvolutionRun:
    """Run the evolution loop; each generation may promote a new base."""
    if generations < 1:
        raise ValueError("generations must be at least 1")
    current = base
    results: list[GenerationResult] = []
    for generation in range(generations):
        result = _evolve_generation(
            current,
            evaluate_fn,
            train_ids,
            holdout_ids,
            generation=generation,
            max_candidates=max_candidates,
            extra_candidates=extra_candidates,
        )
        results.append(result)
        if result.decision == "promoted" and result.promoted_strategy:
            current = result.promoted_strategy
        else:
            break
    return EvolutionRun(
        final_version=current.version,
        generations=tuple(results),
    )


def evaluate_live(
    strategy: StrategyConfig,
    task_ids: tuple[str, ...],
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    max_turns: int = 20,
    timeout: float = 120.0,
    max_retries: int = 2,
    permission_mode: str = "default",
    runtime_name: str = "local",
    docker_image: str = "python:3.12-slim",
    skills_dir: str | Path | None = None,
    auto_approve: bool = True,
) -> dict[str, float]:
    """Score a strategy on a task subset with a real model."""
    tasks = tuple(
        TASKS_BY_ID[task_id]
        for task_id in task_ids
        if task_id in TASKS_BY_ID
    )
    if not tasks:
        raise ValueError("no valid task ids")
    cases = tuple(
        _run_live_case(
            task,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_turns=max_turns,
            timeout=timeout,
            max_retries=max_retries,
            permission_mode=permission_mode,
            runtime_name=runtime_name,
            docker_image=docker_image,
            skills_dir=Path(skills_dir) if skills_dir else None,
            cost_calculator=None,
            auto_approve=auto_approve,
            strategy=strategy,
        )
        for task in tasks
    )
    report = CodingReport.aggregate(cases, run_type="live", model=model)
    return aggregate_metrics(report)


def estimate_context_chars(
    strategy: StrategyConfig,
    task: str,
    tool_names: tuple[str, ...],
) -> dict[str, Any]:
    """Offline estimate of per-turn context cost under a strategy.

    This is an estimate, not a measurement: tool schema size is derived from
    the built-in runtime tools and history cost is approximated by the
    configured budget. It exists so candidate A/B can be triaged without
    spending tokens.
    """
    import tempfile

    from miniclaude.runtime_tools import create_runtime_tools
    from runtime import LocalProcessRuntime

    with tempfile.TemporaryDirectory() as directory:
        schemas = [
            tool.schema()
            for tool in create_runtime_tools(
                LocalProcessRuntime(directory)
            )
        ]
    selected = [
        schema
        for schema in schemas
        if not strategy.tool_gating or schema["name"] in tool_names
    ]
    return {
        "task": task,
        "tools_sent": len(selected),
        "tools_total": len(schemas),
        "schema_chars": sum(len(json.dumps(schema)) for schema in selected),
        "history_budget": strategy.context_max_chars,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark-driven strategy evolution"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="score candidates with a real model (requires MINICLAUDE_MODEL)",
    )
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--base-version", default="v1")
    parser.add_argument("--tasks", default=None)
    parser.add_argument(
        "--attribution-trace",
        default=None,
        help=(
            "path to a previous run's events JSON (AgentResult.events); "
            "failures are attributed and seed extra strategy candidates"
        ),
    )
    args = parser.parse_args()

    base = StrategyConfig(version=args.base_version)
    train_ids, holdout_ids = default_splits()
    if args.tasks:
        task_ids = tuple(part.strip() for part in args.tasks.split(","))
        train_ids = task_ids
        holdout_ids = task_ids

    if not args.live:
        candidates = generate_candidates(base, args.max_candidates)
        payload = {
            "version": "1.0",
            "run_type": "evolution_dry_run",
            "base_version": base.version,
            "candidates": [
                {
                    "version": candidate.version,
                    "estimate": estimate_context_chars(
                        candidate, "sample task", ("read_file", "grep_files")
                    ),
                }
                for candidate in candidates
            ],
        }
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        print(rendered)
        save_report(f"evolution-{args.base_version}-dry-run", payload)
        return 0

    config = AppConfig.from_env(model=None, workspace=Path.cwd())
    if not config.model:
        print("--live requires MINICLAUDE_MODEL", file=sys.stderr)
        return 2

    def evaluate(
        strategy: StrategyConfig, task_ids: tuple[str, ...]
    ) -> Mapping[str, float]:
        return evaluate_live(
            strategy,
            task_ids,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
            permission_mode="default",
        )

    extra_candidates: tuple[StrategyConfig, ...] = ()
    attribution = None
    if args.attribution_trace:
        from evaluation.attribution import (
            attribute_run,
            generate_attribution_candidates,
        )

        payload = json.loads(
            Path(args.attribution_trace).read_text(encoding="utf-8")
        )
        events = payload if isinstance(payload, list) else payload.get(
            "events", []
        )
        phases = payload.get("phases", ()) if isinstance(payload, dict) else ()
        attribution = attribute_run(events, phases)
        extra_candidates = generate_attribution_candidates(
            base, attribution
        )
        print(
            f"attribution: {json.dumps(attribution.to_dict(), ensure_ascii=False)}",
            file=sys.stderr,
        )

    run = evolve(
        base,
        evaluate,
        train_ids,
        holdout_ids,
        generations=args.generations,
        max_candidates=args.max_candidates,
        extra_candidates=extra_candidates,
    )
    payload = run.to_dict()
    if attribution is not None:
        payload["attribution"] = attribution.to_dict()
        payload["attribution_candidates"] = [
            candidate.version for candidate in extra_candidates
        ]
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    save_report(f"evolution-{args.base_version}", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
