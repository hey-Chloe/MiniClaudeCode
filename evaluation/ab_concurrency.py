"""Offline micro-benchmark of read-only batch dispatch (sequential vs pooled).

The driver runs independent read-only tool calls in a bounded thread pool
(``ThreadPoolExecutor``, max 4 workers) and runs mutating batches
sequentially. This module measures the wall-clock difference on a synthetic
read-only batch through the real ``ToolRegistry.dispatch`` path, reporting
per-mode medians over repeated runs. It is an honest, machine-local
measurement: the report states that numbers are observed wall-clock medians,
not a portable guarantee.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from evaluation.reporting import save_report
from miniclaude.runtime_tools import create_runtime_tools
from miniclaude.tools import ToolRegistry
from runtime.local import LocalProcessRuntime
from security.approval import ApprovalManager


def _build_batch(workspace: Path) -> list[tuple[str, str, str]]:
    """A deterministic read-only batch: reads plus greps across fixture files."""
    calls: list[tuple[str, str, str]] = []
    for index in range(12):
        path = f"src/mod_{index % 4}.py"
        calls.append(
            ("call_read_%d" % index, "read_file", json.dumps({"path": path}))
        )
    for index in range(4):
        calls.append(
            (
                f"call_grep_{index}",
                "grep_files",
                json.dumps(
                    {"pattern": r"def helper", "path": "src", "limit": 200}
                ),
            )
        )
    return calls


def _dispatch_batch(registry: ToolRegistry, calls, parallel: bool) -> list[Any]:
    if not parallel or len(calls) <= 1:
        return [
            registry.dispatch(call_id, name, arguments)
            for call_id, name, arguments in calls
        ]
    workers = min(4, len(calls))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                lambda call: registry.dispatch(*call),
                calls,
            )
        )


def _median_time(registry: ToolRegistry, calls, *, parallel: bool, repeats: int) -> dict[str, Any]:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        observations = _dispatch_batch(registry, calls, parallel=parallel)
        elapsed = time.perf_counter() - started
        samples.append(elapsed)
        if not all(observation.success for observation in observations):
            raise RuntimeError("dispatch benchmark produced a failed observation")
    return {
        "samples_seconds": [round(sample, 6) for sample in samples],
        "median_seconds": round(statistics.median(samples), 6),
    }


def run_benchmark(
    *,
    repeats: int = 5,
    batch_workers: int = 4,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        (workspace / "src").mkdir(parents=True, exist_ok=True)
        for index in range(4):
            (workspace / "src" / f"mod_{index}.py").write_text(
                "\n".join(
                    f"def helper_{line}():\n    return '{'x' * 80}'"
                    for line in range(8_000)
                )
                + f"\ndef value_{index}():\n    return {index}\n",
                encoding="utf-8",
            )
        runtime = LocalProcessRuntime(workspace)
        registry = ToolRegistry(approvals=ApprovalManager(lambda *_: True))
        for tool in create_runtime_tools(runtime):
            registry.register(tool)
        calls = _build_batch(workspace)
        sequential = _median_time(
            registry, calls, parallel=False, repeats=repeats
        )
        pooled = _median_time(
            registry, calls, parallel=True, repeats=repeats
        )

    sequential_median = sequential["median_seconds"]
    pooled_median = pooled["median_seconds"]
    return {
        "version": "1.0",
        "run_type": "ab_concurrency",
        "note": (
            "Observed wall-clock medians on this machine (LocalProcessRuntime, "
            "read-only batch); not a portable performance guarantee."
        ),
        "batch": {
            "calls": len(calls),
            "workers": batch_workers,
            "modes": {"sequential": 1, "pooled": batch_workers},
        },
        "results": {
            "sequential": sequential,
            "pooled": pooled,
            "speedup": (
                round(sequential_median / pooled_median, 3)
                if pooled_median
                else None
            ),
            "pooled_pct_change": (
                round(
                    (pooled_median - sequential_median)
                    / sequential_median
                    * 100,
                    2,
                )
                if sequential_median
                else None
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Micro-benchmark sequential vs pooled read-only dispatch and "
            "persist a versioned report."
        )
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    payload = run_benchmark(repeats=args.repeats)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    target = save_report("ab-concurrency", payload)
    print(f"report saved: {target}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
