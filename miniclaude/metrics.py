"""Per-run metrics assembled from agent state and trace events.

Metrics are intentionally computed from data the harness already produces
(AgentState usage counters, trace ``tool_results`` observations, timestamps)
so they never require instrumenting the provider a second time.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Mapping


_READ_TOOL_NAME = "read_file"


@dataclass(frozen=True, slots=True)
class Pricing:
    """USD cost per one million tokens for a model."""

    input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        if self.input_per_million < 0 or self.output_per_million < 0:
            raise ValueError("prices must not be negative")


class CostCalculator:
    """Estimate USD cost from a configurable per-model pricing table.

    Returns None when the model is unknown or no pricing is configured, so a
    missing price is never silently reported as zero.
    """

    def __init__(self, pricing: Mapping[str, Pricing] | None = None):
        self.pricing = dict(pricing or {})

    def estimate(
        self,
        model: str | None,
        input_tokens: int,
        output_tokens: int,
    ) -> float | None:
        if not model:
            return None
        price = self.pricing.get(model)
        if price is None:
            return None
        return (
            (input_tokens / 1_000_000) * price.input_per_million
            + (output_tokens / 1_000_000) * price.output_per_million
        )


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Aggregated, audit-friendly numbers for one agent run."""

    turns: int = 0
    tool_calls: int = 0
    tool_successes: int = 0
    policy_actions: Mapping[str, int] = field(default_factory=dict)
    total_reads: int = 0
    repeated_reads: int = 0
    recoverable_failures: int = 0
    recovered_failures: int = 0
    tools_sent: int = 0
    cache_hits: int = 0
    context_compression: Mapping[str, int] = field(default_factory=dict)
    parallel_batches: int = 0
    max_parallelism: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    context_truncated: bool = False
    duration_seconds: float = 0.0
    model_name: str | None = None
    cost_usd: float | None = None
    skills_loaded: tuple[str, ...] = ()

    @property
    def tool_success_rate(self) -> float | None:
        if self.tool_calls == 0:
            return None
        return self.tool_successes / self.tool_calls

    @property
    def repeated_read_rate(self) -> float | None:
        if self.total_reads == 0:
            return None
        return self.repeated_reads / self.total_reads

    @property
    def recovery_rate(self) -> float | None:
        """Share of failed tool calls followed by a later success of the same tool."""
        if self.recoverable_failures == 0:
            return None
        return self.recovered_failures / self.recoverable_failures

    @property
    def safety_block_rate(self) -> float | None:
        """Share of tool calls blocked by the security policy (DENY)."""
        if self.tool_calls == 0:
            return None
        denied = self.policy_actions.get("deny", 0)
        return denied / self.tool_calls

    @property
    def average_tools_per_turn(self) -> float | None:
        if self.turns == 0:
            return None
        return self.tools_sent / self.turns

    @property
    def cache_hit_rate(self) -> float | None:
        if self.total_reads == 0:
            return None
        return self.cache_hits / self.total_reads

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "tool_successes": self.tool_successes,
            "tool_success_rate": self.tool_success_rate,
            "policy_actions": dict(self.policy_actions),
            "total_reads": self.total_reads,
            "repeated_reads": self.repeated_reads,
            "repeated_read_rate": self.repeated_read_rate,
            "recoverable_failures": self.recoverable_failures,
            "recovered_failures": self.recovered_failures,
            "recovery_rate": self.recovery_rate,
            "safety_block_rate": self.safety_block_rate,
            "tools_sent": self.tools_sent,
            "average_tools_per_turn": self.average_tools_per_turn,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hit_rate,
            "context_compression": dict(self.context_compression),
            "parallel_batches": self.parallel_batches,
            "max_parallelism": self.max_parallelism,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "context_truncated": self.context_truncated,
            "duration_seconds": round(self.duration_seconds, 3),
            "model_name": self.model_name,
            "cost_usd": self.cost_usd,
            "skills_loaded": list(self.skills_loaded),
        }

    @classmethod
    def from_run(cls, state, trace, started: float) -> "RunMetrics":
        """Build metrics from mutable state and a finished trace.

        ``state`` is an AgentState; the type is not imported here to keep the
        metrics module free of model dependencies.
        """
        policy_counts: dict[str, int] = {}
        tool_calls = 0
        tool_successes = 0
        seen_paths: set[str] = set()
        total_reads = 0
        repeated_reads = 0
        cache_hits = 0
        observations: list[dict[str, Any]] = []

        for event in trace.events:
            if event.get("event") != "tool_results":
                continue
            for observation in event.get("detail") or []:
                observations.append(observation)
                tool_calls += 1
                if observation.get("success"):
                    tool_successes += 1
                action = observation.get("policy_action")
                if action:
                    policy_counts[action] = policy_counts.get(action, 0) + 1
                if observation.get("name") == _READ_TOOL_NAME:
                    total_reads += 1
                    output = observation.get("output")
                    if (
                        isinstance(output, dict)
                        and output.get("cache_hit") is True
                    ):
                        cache_hits += 1
                    arguments = observation.get("arguments")
                    path = (
                        arguments.get("path")
                        if isinstance(arguments, dict)
                        else None
                    )
                    if isinstance(path, str):
                        if path in seen_paths:
                            repeated_reads += 1
                        seen_paths.add(path)

        recoverable_failures = 0
        recovered_failures = 0
        for index, observation in enumerate(observations):
            if observation.get("success"):
                continue
            later = observations[index + 1 :]
            if not later:
                continue
            recoverable_failures += 1
            name = observation.get("name")
            if any(
                later_observation.get("name") == name
                and later_observation.get("success")
                for later_observation in later
            ):
                recovered_failures += 1

        return cls(
            turns=state.turn_count,
            tool_calls=tool_calls,
            tool_successes=tool_successes,
            policy_actions=policy_counts,
            total_reads=total_reads,
            repeated_reads=repeated_reads,
            recoverable_failures=recoverable_failures,
            recovered_failures=recovered_failures,
            tools_sent=getattr(state, "tools_sent", 0),
            cache_hits=cache_hits,
            context_compression=dict(
                getattr(state, "context_compression", {})
            ),
            parallel_batches=getattr(state, "parallel_batches", 0),
            max_parallelism=getattr(state, "max_parallelism", 1),
            input_tokens=state.usage_input_tokens,
            output_tokens=state.usage_output_tokens,
            total_tokens=state.usage_input_tokens + state.usage_output_tokens,
            context_truncated=bool(state.context_truncated),
            duration_seconds=max(0.0, time.monotonic() - started),
            model_name=state.model_name,
            skills_loaded=tuple(state.skills_loaded),
        )
