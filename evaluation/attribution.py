"""Failure attribution from a run trace, plus attribution-driven candidates.

The agent already records every tool observation and phase. This module turns
that audit trail into a structured attribution (which tools failed, with what
kind of error, in which phase, and whether a later call recovered) and then
generates bounded, deterministic strategy candidates that target those
failures. It is the honest, implementable version of "failure attribution
generates candidate strategies": candidates are parameter variants seeded by
attribution, not new skills or routing rules written by an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Iterable, Mapping

from evaluation.evolution import StrategyConfig


@dataclass(frozen=True, slots=True)
class FailureAttribution:
    """Structured summary of what failed in one run and whether it recovered."""

    failed_tools: tuple[str, ...] = ()
    error_kinds: tuple[str, ...] = ()
    failed_phase: str | None = None
    recoverable_failures: int = 0
    recovered_failures: int = 0
    policy_denials: int = 0

    @property
    def recovery_rate(self) -> float | None:
        if self.recoverable_failures == 0:
            return None
        return self.recovered_failures / self.recoverable_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "failed_tools": list(self.failed_tools),
            "error_kinds": list(self.error_kinds),
            "failed_phase": self.failed_phase,
            "recoverable_failures": self.recoverable_failures,
            "recovered_failures": self.recovered_failures,
            "recovery_rate": self.recovery_rate,
            "policy_denials": self.policy_denials,
        }


def _error_kind(observation: Mapping[str, Any]) -> str:
    error = str(observation.get("error") or "")
    if "blocked" in error:
        return "policy_denial"
    if "timeout" in error or "timed out" in error:
        return "timeout"
    if "unknown tool" in error:
        return "unknown_tool"
    if "validation" in error.lower() or "argument" in error.lower():
        return "validation"
    return "execution"


def attribute_run(
    events: Iterable[Mapping[str, Any]],
    phases: Iterable[str] = (),
) -> FailureAttribution:
    """Attribute failures from a trace event list (AgentResult.events)."""
    observations = [
        observation
        for event in events
        if event.get("event") == "tool_results"
        for observation in event.get("detail") or []
    ]
    failed_tools: list[str] = []
    error_kinds: list[str] = []
    policy_denials = 0
    recoverable = 0
    recovered = 0
    for index, observation in enumerate(observations):
        if observation.get("success"):
            continue
        name = str(observation.get("name") or "unknown")
        kind = _error_kind(observation)
        if name not in failed_tools:
            failed_tools.append(name)
        if kind not in error_kinds:
            error_kinds.append(kind)
        if kind == "policy_denial":
            policy_denials += 1
        later = observations[index + 1 :]
        if not later:
            continue
        recoverable += 1
        if any(
            later_observation.get("name") == name
            and later_observation.get("success")
            for later_observation in later
        ):
            recovered += 1
    phase_list = tuple(phases)
    failed_phase = "reflect" if "reflect" in phase_list else (
        phase_list[-1] if phase_list else None
    )
    return FailureAttribution(
        failed_tools=tuple(failed_tools),
        error_kinds=tuple(error_kinds),
        failed_phase=failed_phase,
        recoverable_failures=recoverable,
        recovered_failures=recovered,
        policy_denials=policy_denials,
    )


def generate_attribution_candidates(
    base: StrategyConfig,
    attribution: FailureAttribution,
    *,
    max_candidates: int = 3,
) -> tuple[StrategyConfig, ...]:
    """Deterministic candidate variants targeting the attributed failures."""
    candidates: list[StrategyConfig] = []

    if attribution.failed_tools:
        tool_list = ", ".join(attribution.failed_tools)
        hint = (
            "\n\nRecovery guidance: when a tool call fails, read the error, "
            "retry once with corrected arguments, and prefer read-only tools "
            "for diagnosis before mutating anything. "
            f"Recently failing tools: {tool_list}."
        )
        candidates.append(
            replace(
                base,
                version="attr-recovery-hint",
                system_instructions=base.system_instructions + hint,
            )
        )

    if (
        attribution.recovery_rate is not None
        and attribution.recovery_rate < 0.5
        and base.skill_top_k < 2
    ):
        candidates.append(
            replace(
                base,
                version="attr-skill_top_k-2",
                skill_top_k=2,
            )
        )

    if (
        "timeout" in attribution.error_kinds or "execution" in attribution.error_kinds
    ) and base.retry_max_retries < 3:
        candidates.append(
            replace(
                base,
                version="attr-retry_max_retries-3",
                retry_max_retries=3,
            )
        )

    return tuple(candidates[:max_candidates])
