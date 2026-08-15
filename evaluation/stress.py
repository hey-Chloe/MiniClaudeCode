"""Synthetic long-session replay for context-compression measurement.

The agent's history can grow unboundedly on long tasks; the harness applies
compression layers (``stale_snip``, ``micro_compact``, optional
``auto_compact``) before sending a snapshot to the model. This module builds a
deterministic, long synthetic session that contains exactly the message
patterns those layers act on (superseded snapshot outputs, oversized tool
outputs, repeated reads) and replays it through the real ``ContextManager``
with compression on and off. Every number in the resulting report is computed
from the replay, so a compression claim can point at a concrete artifact.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any

from miniclaude.context import ContextConfig, ContextManager
from evaluation.reporting import save_report


@dataclass(frozen=True, slots=True)
class ReplayStats:
    """One compression configuration's outcome for the same session."""

    label: str
    messages_in: int
    messages_out: int
    chars_in: int
    chars_out: int
    compression: dict[str, int]
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "messages_in": self.messages_in,
            "messages_out": self.messages_out,
            "chars_in": self.chars_in,
            "chars_out": self.chars_out,
            "chars_removed": self.chars_in - self.chars_out,
            "chars_removed_pct": (
                round((self.chars_in - self.chars_out) / self.chars_in * 100, 2)
                if self.chars_in
                else 0.0
            ),
            "messages_removed_pct": (
                round(
                    (self.messages_in - self.messages_out)
                    / self.messages_in
                    * 100,
                    2,
                )
                if self.messages_in
                else 0.0
            ),
            "compression": dict(self.compression),
            "truncated": self.truncated,
        }


def _tool_message(name: str, output: Any, size: int = 0) -> str:
    """One tool observation serialized exactly like ``ToolObservation``."""
    payload = {
        "call_id": f"call_{name}",
        "name": name,
        "success": True,
        "output": output,
        "policy_action": "allow",
        "policy_reason": "read-only",
        "duration_seconds": 0.01,
    }
    rendered = json.dumps(payload, ensure_ascii=False, default=str)
    if size and len(rendered) < size:
        rendered = rendered[:-1] + (" " * (size - len(rendered))) + "}"
    return rendered


def synthesize_session(
    turns: int = 40,
    *,
    seed: int = 7,
    oversized_output_chars: int = 9_000,
    snapshot_chars: int = 5_000,
) -> list[dict[str, str]]:
    """Deterministically generate a long conversation for one replay.

    The session interleaves assistant turns with tool observations:
    repeated reads of the same modules, periodically superseded
    ``workspace_diff`` snapshots (stale_snip material), and oversized
    ``grep_files`` outputs (micro_compact material).
    """
    rng = random.Random(seed)
    messages: list[dict[str, str]] = []
    modules = [f"src/mod_{index}.py" for index in range(6)]
    snapshot_counter = 0
    grep_counter = 0
    for turn in range(1, turns + 1):
        module = modules[turn % len(modules)]
        messages.append(
            {
                "role": "assistant",
                "content": f"Turn {turn}: inspecting {module}.",
            }
        )
        content = (
            "line = "
            + str(rng.getrandbits(32))
            + "\ndef handler():\n    return 0\n"
        )
        messages.append(
            {
                "role": "tool",
                "content": _tool_message(
                    "read_file",
                    {"path": module, "content": content, "cache_hit": False},
                ),
            }
        )
        if turn % 8 == 0:
            snapshot_counter += 1
            diff = "\n".join(
                f"@@ -{line} +{line} @@\n- old_{line}\n+ new_{line}"
                for line in range(1, 60)
            )
            messages.append(
                {
                    "role": "tool",
                    "content": _tool_message(
                        "workspace_diff",
                        {"changed_files": [f"src/mod_{turn}.py"], "diff": diff},
                        size=snapshot_chars,
                    ),
                }
            )
        if turn % 12 == 0:
            grep_counter += 1
            matches = [
                {
                    "path": module,
                    "line": line,
                    "text": f"match {line} " + ("x" * 40),
                }
                for line in range(1, 80)
            ]
            messages.append(
                {
                    "role": "tool",
                    "content": _tool_message(
                        "grep_files",
                        {"pattern": "handler", "matches": matches},
                        size=oversized_output_chars,
                    ),
                }
            )
    return messages


def replay(
    messages: list[dict[str, str]],
    *,
    compression_layers: tuple[str, ...],
    max_chars: int,
    micro_compact_max_chars: int = 4_000,
) -> ReplayStats:
    """Feed the session through a real ContextManager and measure the result."""
    config = ContextConfig(
        max_chars=max_chars,
        compression_layers=compression_layers,
        micro_compact_max_chars=micro_compact_max_chars,
    )
    manager = ContextManager(config)
    manager.start("synthetic stress session")
    for message in messages:
        if message["role"] == "tool":
            manager.add_tool(message["content"])
        else:
            manager.add_assistant(message["content"])
    snapshot = manager.snapshot()
    chars_in = sum(
        len(message["role"]) + len(message["content"]) for message in messages
    )
    chars_out = sum(
        len(message.role) + len(message.content)
        for message in snapshot.messages
    )
    return ReplayStats(
        label="+".join(compression_layers) or "none",
        messages_in=len(messages),
        messages_out=len(snapshot.messages),
        chars_in=chars_in,
        chars_out=chars_out,
        compression=dict(snapshot.compression),
        truncated=snapshot.truncated,
    )


def run_stress(
    *,
    turns: int = 40,
    seed: int = 7,
    max_chars: int = 200_000,
    micro_compact_max_chars: int = 4_000,
) -> dict[str, Any]:
    """Run baseline vs compressed replay and return the report payload."""
    messages = synthesize_session(turns, seed=seed)
    baseline = replay(
        messages,
        compression_layers=(),
        max_chars=max_chars,
        micro_compact_max_chars=micro_compact_max_chars,
    )
    compressed = replay(
        messages,
        compression_layers=("stale_snip", "micro_compact"),
        max_chars=max_chars,
        micro_compact_max_chars=micro_compact_max_chars,
    )
    budget_only = replay(
        messages,
        compression_layers=(),
        max_chars=32_000,
        micro_compact_max_chars=micro_compact_max_chars,
    )
    return {
        "version": "1.0",
        "run_type": "stress_compression",
        "session": {
            "turns": turns,
            "seed": seed,
            "messages": len(messages),
            "patterns": {
                "repeated_reads": True,
                "superseded_snapshots": True,
                "oversized_outputs": True,
            },
        },
        "configs": {
            "baseline": baseline.to_dict(),
            "compressed": compressed.to_dict(),
            "budget_only": budget_only.to_dict(),
        },
        "compression_gain": {
            "chars_removed": (
                baseline.chars_out - compressed.chars_out
            ),
            "chars_removed_pct": (
                round(
                    (baseline.chars_out - compressed.chars_out)
                    / baseline.chars_out
                    * 100,
                    2,
                )
                if baseline.chars_out
                else 0.0
            ),
            "messages_removed": (
                baseline.messages_out - compressed.messages_out
            ),
            "messages_removed_pct": (
                round(
                    (baseline.messages_out - compressed.messages_out)
                    / baseline.messages_out
                    * 100,
                    2,
                )
                if baseline.messages_out
                else 0.0
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a synthetic long session through context compression "
            "and persist a versioned report."
        )
    )
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--micro-compact-max-chars", type=int, default=4_000)
    args = parser.parse_args()

    payload = run_stress(
        turns=args.turns,
        seed=args.seed,
        max_chars=args.max_chars,
        micro_compact_max_chars=args.micro_compact_max_chars,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    target = save_report("stress-compression", payload)
    print(f"report saved: {target}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
