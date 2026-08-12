"""Provider-neutral language-model contracts."""

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot produce a valid model response."""


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Provider-neutral model input with local context metadata."""

    task: str
    turn: int = 0
    tools: tuple[dict[str, Any], ...] = ()
    tool_outputs: tuple[dict[str, str], ...] = ()
    previous_response_id: str | None = None
    instructions: str | None = None
    messages: tuple[dict[str, str], ...] = ()
    context_truncated: bool = False


@dataclass(frozen=True, slots=True)
class LLMToolCall:
    """Provider-neutral tool request emitted by a model."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Normalized token accounting reported by a provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized result of one provider request."""

    text: str = ""
    tool_calls: tuple[LLMToolCall, ...] = ()
    response_id: str | None = None
    model: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.text and not self.tool_calls:
            raise ValueError("LLM response must contain text or tool calls")


class LLMProvider(Protocol):
    """Synchronous provider boundary consumed by the agent loop."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return one normalized response for a model request."""
