"""Offline A/B measurement of tool-gating context savings (no API calls).

The driver sends only task-activated tool schemas when gating is on. This
module replays that exact activation logic against the 26-task coding
benchmark and measures, per task and in aggregate, how many schemas and
schema characters are sent with gating on versus the full toolset. No model
and no network are involved: the numbers come from the real
``ToolRegistry.activate_for_task`` / skill-tool activation path.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.coding.tasks import TASKS
from evaluation.reporting import save_report
from miniclaude.context import ContextConfig, ContextManager
from miniclaude.runtime_tools import create_runtime_tools
from miniclaude.tools import ToolRegistry
from runtime.local import LocalProcessRuntime
from security.approval import ApprovalManager


def _schema_chars(registry: ToolRegistry) -> int:
    return sum(
        len(json.dumps(schema, ensure_ascii=False))
        for schema in registry.schemas()
    )


def measure_task(
    task_text: str,
    registry: ToolRegistry,
    context: ContextManager,
) -> dict[str, Any]:
    """Measure gated vs full schema payloads for one task."""
    context.start(task_text)
    skill_tools = context.selected_skill_tools()

    registry.activate_all()
    full_count = len(registry.schemas())
    full_chars = _schema_chars(registry)

    registry.activate_for_task(task_text)
    activated = list(registry.active_tools())
    if skill_tools:
        registry.activate(skill_tools)
    gated = list(registry.active_tools())
    gated_count = len(gated)
    gated_chars = _schema_chars(registry)

    fallback_full = set(gated) == set(registry.list_tools())
    return {
        "tools_total": full_count,
        "tools_sent_gated": gated_count,
        "schema_chars_full": full_chars,
        "schema_chars_gated": gated_chars,
        "saved_chars": full_chars - gated_chars,
        "saved_pct": (
            round((full_chars - gated_chars) / full_chars * 100, 2)
            if full_chars
            else 0.0
        ),
        "gated": gated_count < full_count,
        "fallback_full": fallback_full,
        "skill_tools": list(skill_tools),
    }


def run_ab(reports_dir: Path | None = None) -> dict[str, Any]:
    """Measure gating on every benchmark task and return the report payload."""
    skills_dir = Path(__file__).resolve().parents[1] / "skills"
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        for index in range(4):
            (workspace / "src").mkdir(parents=True, exist_ok=True)
            (workspace / "src" / f"mod_{index}.py").write_text(
                f"# module {index}\ndef value_{index}():\n    return {index}\n",
                encoding="utf-8",
            )
        runtime = LocalProcessRuntime(workspace)
        tools = create_runtime_tools(runtime)
        registry = ToolRegistry(approvals=ApprovalManager(lambda *_: True))
        for tool in tools:
            registry.register(tool)
        context = ContextManager(
            ContextConfig(
                skills_dir=skills_dir if skills_dir.is_dir() else None,
            )
        )
        for task in TASKS:
            measurement = measure_task(task.task, registry, context)
            cases.append(
                {
                    "id": task.id,
                    "category": task.category,
                    **measurement,
                }
            )

    full_chars = sum(case["schema_chars_full"] for case in cases)
    gated_chars = sum(case["schema_chars_gated"] for case in cases)
    per_category: dict[str, dict[str, float]] = {}
    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_category.setdefault(case["category"], []).append(case)
    for category, items in sorted(by_category.items()):
        category_full = sum(item["schema_chars_full"] for item in items)
        category_gated = sum(item["schema_chars_gated"] for item in items)
        per_category[category] = {
            "tasks": len(items),
            "schema_chars_full": category_full,
            "schema_chars_gated": category_gated,
            "saved_pct": (
                round((category_full - category_gated) / category_full * 100, 2)
                if category_full
                else 0.0
            ),
        }

    return {
        "version": "1.0",
        "run_type": "ab_tool_gating",
        "note": (
            "Offline replay of the real activation path over the 26-task "
            "benchmark; no model calls."
        ),
        "totals": {
            "tasks": len(cases),
            "tools_total": cases[0]["tools_total"] if cases else 0,
            "schema_chars_full": full_chars,
            "schema_chars_gated": gated_chars,
            "saved_chars": full_chars - gated_chars,
            "saved_pct": (
                round((full_chars - gated_chars) / full_chars * 100, 2)
                if full_chars
                else 0.0
            ),
            "gated_tasks": sum(1 for case in cases if case["gated"]),
            "fallback_tasks": sum(1 for case in cases if case["fallback_full"]),
        },
        "per_category": per_category,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure tool-gating schema savings on the 26-task benchmark "
            "and persist a versioned A/B report."
        )
    )
    parser.parse_args()
    payload = run_ab()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    target = save_report("ab-tool-gating", payload)
    print(f"report saved: {target}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
