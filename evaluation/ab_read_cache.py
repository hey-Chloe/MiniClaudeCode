"""Offline A/B measurement of the freshness file-read cache (no API calls).

The agent's ``read_file`` tool serves repeated reads of an unchanged file from
a cache keyed by ``(path, mtime, size)``. This module replays a deterministic
repeated-read session through the real tool registry with the cache enabled
and disabled, and reports cache hit rate, repeated-read rate, and observed
wall-clock medians. Every number is computed from the replay; nothing is
hard-coded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from evaluation.reporting import save_report
from miniclaude.runtime_tools import create_runtime_tools
from miniclaude.tools import ToolRegistry
from runtime.local import LocalProcessRuntime
from security.approval import ApprovalManager


def _build_session(workspace: Path) -> list[tuple[str, str, str]]:
    """Deterministic read/write session: repeated reads plus mutations."""
    calls: list[tuple[str, str, str]] = []
    for round_index in range(8):
        for file_index in range(6):
            path = f"src/mod_{file_index}.py"
            calls.append(
                (
                    f"read_r{round_index}_f{file_index}",
                    "read_file",
                    json.dumps({"path": path}),
                )
            )
        if round_index in {2, 5}:
            calls.append(
                (
                    f"write_r{round_index}",
                    "write_file",
                    json.dumps(
                        {
                            "path": "src/mod_0.py",
                            "content": f"# rewritten round {round_index}\n",
                        }
                    ),
                )
            )
    return calls


def _run_session(
    registry: ToolRegistry,
    calls: list[tuple[str, str, str]],
) -> tuple[dict[str, int], float]:
    started = time.perf_counter()
    seen: set[str] = set()
    repeated = 0
    cache_hits = 0
    total_reads = 0
    for call_id, name, arguments in calls:
        observation = registry.dispatch(call_id, name, arguments)
        if name == "read_file" and observation.success:
            total_reads += 1
            output = observation.output or {}
            if output.get("cache_hit") is True:
                cache_hits += 1
            path = (observation.arguments or {}).get("path")
            if path in seen:
                repeated += 1
            seen.add(path)
    elapsed = time.perf_counter() - started
    stats = {
        "total_reads": total_reads,
        "repeated_reads": repeated,
        "cache_hits": cache_hits,
        "repeated_read_rate": (
            round(repeated / total_reads, 4) if total_reads else None
        ),
        "cache_hit_rate": (
            round(cache_hits / total_reads, 4) if total_reads else None
        ),
    }
    return stats, elapsed


def run_ab(*, repeats: int = 5) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        (workspace / "src").mkdir(parents=True, exist_ok=True)
        for index in range(6):
            (workspace / "src" / f"mod_{index}.py").write_text(
                "\n".join(
                    f"def f_{line}():\n    return {line}"
                    for line in range(300)
                ),
                encoding="utf-8",
            )
        calls = _build_session(workspace)

        def build(cache_enabled: bool) -> tuple[ToolRegistry, list[float]]:
            registry = ToolRegistry(
                approvals=ApprovalManager(lambda *_: True)
            )
            for tool in create_runtime_tools(
                LocalProcessRuntime(workspace),
                cache_enabled=cache_enabled,
            ):
                registry.register(tool)
            samples: list[float] = []
            final_stats: dict[str, int] | None = None
            for _ in range(repeats):
                stats, elapsed = _run_session(registry, calls)
                final_stats = stats
                samples.append(elapsed)
            assert final_stats is not None
            return registry, samples

        _, samples_off = build(cache_enabled=False)
        registry_on, samples_on = build(cache_enabled=True)

        # Re-run once on a fresh registry so stats reflect the final session.
        registry_off = ToolRegistry(approvals=ApprovalManager(lambda *_: True))
        for tool in create_runtime_tools(
            LocalProcessRuntime(workspace),
            cache_enabled=False,
        ):
            registry_off.register(tool)
        stats_off, _ = _run_session(registry_off, calls)
        stats_on, _ = _run_session(registry_on, calls)

    median_off = statistics.median(samples_off)
    median_on = statistics.median(samples_on)
    return {
        "version": "1.0",
        "run_type": "ab_read_cache",
        "note": (
            "Offline replay of a deterministic repeated-read session through "
            "the real read_file tool; wall-clock medians on this machine."
        ),
        "session": {
            "calls": len(calls),
            "repeats": repeats,
        },
        "results": {
            "cache_off": {
                **stats_off,
                "median_seconds": round(median_off, 6),
                "samples_seconds": [round(sample, 6) for sample in samples_off],
            },
            "cache_on": {
                **stats_on,
                "median_seconds": round(median_on, 6),
                "samples_seconds": [round(sample, 6) for sample in samples_on],
            },
            "cache_hit_rate_on": stats_on["cache_hit_rate"],
            "repeated_read_rate_off": stats_off["repeated_read_rate"],
            "repeated_read_rate_on": stats_on["repeated_read_rate"],
            "time_saved_pct": (
                round((median_off - median_on) / median_off * 100, 2)
                if median_off
                else None
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A/B the freshness read cache on a synthetic repeated-read "
            "session and persist a versioned report."
        )
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    payload = run_ab(repeats=args.repeats)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    target = save_report("ab-read-cache", payload)
    print(f"report saved: {target}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
