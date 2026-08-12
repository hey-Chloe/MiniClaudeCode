"""Core state models for the MiniClaudeCode agent loop."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    """Lifecycle state of one agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class LoopDecision:
    """One decision emitted by a loop driver.

    Phase 1 deliberately keeps this independent from any LLM or tool schema.
    Later phases can adapt provider responses into these decisions.
    """

    event: str
    detail: Any
    terminal: bool = False


@dataclass(slots=True)
class AgentState:
    """Mutable state owned by a single agent run."""

    task: str
    max_turns: int
    status: RunStatus = RunStatus.PENDING
    turn_count: int = 0
    output: Any = None
    error: str | None = None
    provider_response_id: str | None = None
    tool_outputs: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Structured result produced when an agent run terminates."""

    status: RunStatus
    task: str
    turns: int
    output: Any = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
