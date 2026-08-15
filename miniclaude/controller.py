"""Bounded state machine that owns the MiniClaudeCode agent loop."""

import time
from typing import Protocol

from miniclaude.metrics import RunMetrics
from miniclaude.models import (
    AgentPhase,
    AgentResult,
    AgentState,
    LoopDecision,
    RunStatus,
)
from miniclaude.trace import Trace


class LoopDriver(Protocol):
    """Decision boundary consumed by the loop.

    This is intentionally not an LLM provider. Phase 2 will introduce the
    provider abstraction and an adapter that satisfies this protocol.
    """

    def next(self, state: AgentState) -> LoopDecision:
        """Return the next decision for the current run state."""


class CompatibilityLoopDriver:
    """Deterministic driver preserving the v4.1 command-line behavior."""

    _STEPS = (
        LoopDecision("planning", "create plan", phase=AgentPhase.PLAN),
        LoopDecision("tool_selection", "pytest", phase=AgentPhase.ACT),
        LoopDecision(
            "verification",
            "passed",
            terminal=True,
            phase=AgentPhase.VERIFY,
        ),
    )

    def next(self, state: AgentState) -> LoopDecision:
        try:
            return self._STEPS[state.turn_count]
        except IndexError as exc:
            raise RuntimeError("compatibility driver exhausted") from exc


class AgentController:
    """Runs decisions until completion, failure, or the turn budget is spent."""

    def __init__(self, driver: LoopDriver, max_turns: int = 20):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.driver = driver
        self.max_turns = max_turns

    def run(
        self,
        task: str,
        trace: Trace | None = None,
        *,
        initial_state: AgentState | None = None,
    ) -> AgentResult:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        current_trace = trace if trace is not None else Trace()
        if initial_state is not None:
            if initial_state.task != task:
                raise ValueError("initial state does not match the task")
            state = initial_state
            current_trace.add("resumed", {"turn_count": state.turn_count})
        else:
            current_trace.clear()
            current_trace.add("task", task)
            state = AgentState(
                task=task,
                max_turns=self.max_turns,
                status=RunStatus.RUNNING,
            )
        started = time.monotonic()

        while state.turn_count < state.max_turns:
            try:
                decision = self.driver.next(state)
            except Exception as exc:
                state.status = RunStatus.FAILED
                state.error = str(exc)
                current_trace.add("error", state.error)
                return self._result(state, current_trace, started)

            self._validate_decision(decision)
            state.turn_count += 1
            for extra_phase in decision.extra_phases:
                state.phases.append(extra_phase.value)
            state.phases.append(decision.phase.value)
            for extra_event, extra_detail in decision.extra_events:
                current_trace.add(extra_event, extra_detail)
            current_trace.add(decision.event, decision.detail)
            if decision.event == "tool_results":
                self._record_file_modifications(current_trace, decision.detail)
            if state.turn_count == 1 and state.skills_loaded:
                current_trace.add("skill_loaded", list(state.skills_loaded))

            if decision.terminal:
                state.status = RunStatus.COMPLETED
                state.output = decision.detail
                return self._result(state, current_trace, started)

        state.status = RunStatus.MAX_TURNS
        state.error = f"maximum turn limit reached ({state.max_turns})"
        current_trace.add("termination", state.error)
        return self._result(state, current_trace, started)

    @staticmethod
    def _validate_decision(decision: LoopDecision) -> None:
        if not isinstance(decision, LoopDecision):
            raise TypeError("loop driver must return LoopDecision")
        if not decision.event.strip():
            raise ValueError("loop decision event must not be empty")

    @staticmethod
    def _record_file_modifications(trace: Trace, detail) -> None:
        """Derive explicit file-modification events from tool observations."""
        for observation in detail or []:
            name = observation.get("name")
            if name not in {"write_file", "replace_text"}:
                continue
            if not observation.get("success"):
                continue
            arguments = observation.get("arguments") or {}
            path = arguments.get("path") if isinstance(arguments, dict) else None
            if isinstance(path, str):
                trace.add(
                    "file_modified",
                    {"tool": name, "path": path},
                )

    @staticmethod
    def _result(
        state: AgentState, trace: Trace, started: float
    ) -> AgentResult:
        return AgentResult(
            status=state.status,
            task=state.task,
            turns=state.turn_count,
            output=state.output,
            error=state.error,
            events=trace.export(),
            metrics=RunMetrics.from_run(state, trace, started),
            skills=tuple(state.skills_loaded),
            phases=tuple(state.phases),
            provider_response_id=state.provider_response_id,
        )

