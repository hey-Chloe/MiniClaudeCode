"""Bounded state machine that owns the MiniClaudeCode agent loop."""

from typing import Protocol

from miniclaude.models import AgentResult, AgentState, LoopDecision, RunStatus
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
        LoopDecision("planning", "create plan"),
        LoopDecision("tool_selection", "pytest"),
        LoopDecision("verification", "passed", terminal=True),
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

    def run(self, task: str, trace: Trace | None = None) -> AgentResult:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        current_trace = trace if trace is not None else Trace()
        current_trace.clear()
        current_trace.add("task", task)
        state = AgentState(task=task, max_turns=self.max_turns, status=RunStatus.RUNNING)

        while state.turn_count < state.max_turns:
            try:
                decision = self.driver.next(state)
            except Exception as exc:
                state.status = RunStatus.FAILED
                state.error = str(exc)
                current_trace.add("error", state.error)
                return self._result(state, current_trace)

            self._validate_decision(decision)
            state.turn_count += 1
            current_trace.add(decision.event, decision.detail)

            if decision.terminal:
                state.status = RunStatus.COMPLETED
                state.output = decision.detail
                return self._result(state, current_trace)

        state.status = RunStatus.MAX_TURNS
        state.error = f"maximum turn limit reached ({state.max_turns})"
        current_trace.add("termination", state.error)
        return self._result(state, current_trace)

    @staticmethod
    def _validate_decision(decision: LoopDecision) -> None:
        if not isinstance(decision, LoopDecision):
            raise TypeError("loop driver must return LoopDecision")
        if not decision.event.strip():
            raise ValueError("loop decision event must not be empty")

    @staticmethod
    def _result(state: AgentState, trace: Trace) -> AgentResult:
        return AgentResult(
            status=state.status,
            task=state.task,
            turns=state.turn_count,
            output=state.output,
            error=state.error,
            events=trace.export(),
        )

