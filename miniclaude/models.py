"""Core state models for the MiniClaudeCode agent loop."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from miniclaude.metrics import RunMetrics


class RunStatus(str, Enum):
    """Lifecycle state of one agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    FAILED = "failed"


class AgentPhase(str, Enum):
    """Stages of the bounded agent loop, recorded per decision."""

    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    REFLECT = "reflect"
    VERIFY = "verify"
    FINALIZE = "finalize"


@dataclass(frozen=True, slots=True)
class LoopDecision:
    """One decision emitted by a loop driver.

    Phase 1 deliberately keeps this independent from any LLM or tool schema.
    Later phases can adapt provider responses into these decisions.
    """

    event: str
    detail: Any
    terminal: bool = False
    phase: AgentPhase = AgentPhase.ACT
    # Extra phases/events recorded alongside the main decision without
    # consuming an extra turn. A tool round is one decision that advances
    # through act -> observe (plus reflect on failure), so all three stages
    # are auditable in ``AgentResult.phases`` and the trace.
    extra_phases: tuple[AgentPhase, ...] = ()
    extra_events: tuple[tuple[str, Any], ...] = ()


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
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    model_name: str | None = None
    context_truncated: bool = False
    skills_loaded: tuple[str, ...] = ()
    tools_sent: int = 0
    context_compression: dict[str, int] = field(default_factory=dict)
    parallel_batches: int = 0
    max_parallelism: int = 1
    phases: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AgentResult:
    """Structured result produced when an agent run terminates."""

    status: RunStatus
    task: str
    turns: int
    output: Any = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    metrics: RunMetrics | None = None
    skills: tuple[str, ...] = ()
    phases: tuple[str, ...] = ()
    provider_response_id: str | None = None
