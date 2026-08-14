"""Evaluate skill routing (keyword vs hybrid) against the coding benchmark.

Each benchmark task belongs to a category that maps to an expected skill.
The evaluator measures how often ``SkillRegistry.select`` picks that skill in
``keyword`` and ``hybrid`` modes and persists a versioned report under
``reports/``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.coding.tasks import TASKS
from evaluation.reporting import save_report
from miniclaude.skills import SkillRegistry


CATEGORY_EXPECTED_SKILL: dict[str, str] = {
    "failing_test_fix": "bug-fix",
    "small_feature": "bug-fix",
    "code_search": "repo-analysis",
    "safe_refactor": "bug-fix",
    "config_repair": "bug-fix",
    "dependency_issue": "bug-fix",
    "permission_security": "code-review",
}


def evaluate_routing(tasks, registry: SkillRegistry) -> dict[str, Any]:
    """Return per-task selections and aggregate hit rates for both modes."""
    cases: list[dict[str, Any]] = []
    keyword_hits = 0
    hybrid_hits = 0
    expected_total = 0
    for task in tasks:
        expected = CATEGORY_EXPECTED_SKILL.get(task.category)
        if expected is None:
            continue
        expected_total += 1
        keyword = registry.select(task.task, top_k=1, mode="keyword")
        hybrid = registry.select(task.task, top_k=1, mode="hybrid")
        keyword_name = keyword[0].name if keyword else None
        hybrid_name = hybrid[0].name if hybrid else None
        keyword_hits += keyword_name == expected
        hybrid_hits += hybrid_name == expected
        cases.append(
            {
                "id": task.id,
                "category": task.category,
                "expected_skill": expected,
                "keyword_skill": keyword_name,
                "hybrid_skill": hybrid_name,
            }
        )
    return {
        "version": "1.0",
        "run_type": "skill_routing",
        "skills": registry.names(),
        "total": expected_total,
        "keyword_hit_rate": (
            keyword_hits / expected_total if expected_total else 0.0
        ),
        "hybrid_hit_rate": (
            hybrid_hits / expected_total if expected_total else 0.0
        ),
        "cases": cases,
    }


def _skills_dir() -> Path:
    candidate = Path(__file__).resolve().parents[1] / "skills"
    return candidate if candidate.is_dir() else Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate skill routing hit rates on the coding benchmark"
    )
    parser.add_argument("--skills-dir", default=None)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    registry = SkillRegistry(
        Path(args.skills_dir) if args.skills_dir else _skills_dir()
    )
    report = evaluate_routing(TASKS, registry)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if not args.no_report:
        target = save_report("skill-routing", report)
        print(f"report saved: {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
