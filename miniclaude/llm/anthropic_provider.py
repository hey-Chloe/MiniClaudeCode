"""Anthropic Messages API adapter for the provider-neutral boundary."""

import json
import random
import time
from dataclasses import dataclass
from typing import Any

from miniclaude.llm.base import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    LLMUsage,
)


@dataclass(frozen=True, slots=True)
class AnthropicProviderConfig:
    """Configuration for the Anthropic Messages API adapter."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 120.0
    max_tokens: int = 4096
    instructions: str | None = None
    max_retries: int = 2
    retry_base_delay: float = 0.5
    retry_max_delay: float = 8.0
    retry_jitter: bool = True

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.retry_base_delay < 0 or self.retry_max_delay < 0:
            raise ValueError("retry delays must not be negative")


class AnthropicProvider:
    """Calls the Messages API and normalizes responses into ``LLMResponse``."""

    def __init__(self, config: AnthropicProviderConfig, client: Any | None = None):
        self.config = config
        self.client = client if client is not None else self._create_client(config)
        self._messages: list[dict[str, Any]] = []
        self._pending_tool_uses: list[dict[str, Any]] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")
        if request.turn == 0:
            self._messages = []
            self._pending_tool_uses = []
        if request.tool_outputs:
            self._append_tool_results(request.tool_outputs)

        parameters: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": list(self._messages),
        }
        system = "\n\n".join(
            value
            for value in (self.config.instructions, request.instructions)
            if value
        )
        if system:
            parameters["system"] = system
        if request.tools:
            parameters["tools"] = [
                self._anthropic_tool(tool) for tool in request.tools
            ]

        try:
            response = self._with_retry(
                lambda: self.client.messages.create(**parameters)
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Anthropic request failed: {exc}") from exc

        blocks = [
            block
            for block in (getattr(response, "content", None) or ())
        ]
        text = "".join(
            self._block_value(block, "text") or ""
            for block in blocks
            if self._block_value(block, "type") == "text"
        )
        tool_calls: list[LLMToolCall] = []
        tool_blocks: list[dict[str, Any]] = []
        for block in blocks:
            if self._block_value(block, "type") != "tool_use":
                continue
            block_id = str(self._block_value(block, "id") or "")
            name = str(self._block_value(block, "name") or "")
            block_input = self._block_value(block, "input") or {}
            tool_blocks.append(
                {
                    "type": "tool_use",
                    "id": block_id,
                    "name": name,
                    "input": block_input,
                }
            )
            tool_calls.append(
                LLMToolCall(
                    call_id=block_id,
                    name=name,
                    arguments=json.dumps(
                        block_input, ensure_ascii=False, default=str
                    ),
                )
            )
        if tool_blocks:
            self._pending_tool_uses = tool_blocks
            self._messages.append(
                {"role": "assistant", "content": tool_blocks}
            )
        else:
            self._messages.append({"role": "assistant", "content": text})

        usage = getattr(response, "usage", None) or {}
        input_tokens = int(self._block_value(usage, "input_tokens") or 0)
        output_tokens = int(self._block_value(usage, "output_tokens") or 0)
        return LLMResponse(
            text=text,
            tool_calls=tuple(tool_calls),
            response_id=getattr(response, "id", None),
            model=getattr(response, "model", None),
            usage=LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            raw=response,
        )

    def complete_stream(self, request: LLMRequest):
        """Yield incremental text deltas from the Messages API stream.

        Streaming is text-only at the provider level (the agent loop uses the
        synchronous ``complete`` contract); tool-use blocks are still handled
        through ``complete``.
        """
        if request.turn == 0:
            self._messages = []
            self._pending_tool_uses = []
        if request.tool_outputs:
            self._append_tool_results(request.tool_outputs)

        parameters: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": list(self._messages),
            "stream": True,
        }
        system = "\n\n".join(
            value
            for value in (self.config.instructions, request.instructions)
            if value
        )
        if system:
            parameters["system"] = system
        if request.tools:
            parameters["tools"] = [
                self._anthropic_tool(tool) for tool in request.tools
            ]

        try:
            stream = self._with_retry(
                lambda: self.client.messages.create(**parameters)
            )
            for event in stream:
                if self._block_value(event, "type") != "content_block_delta":
                    continue
                delta = self._block_value(event, "delta") or {}
                if self._block_value(delta, "type") != "text_delta":
                    continue
                text = str(self._block_value(delta, "text") or "")
                if text:
                    yield text
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Anthropic stream failed: {exc}") from exc

    def _append_tool_results(self, tool_outputs) -> None:
        if self._pending_tool_uses:
            self._messages.append(
                {"role": "assistant", "content": self._pending_tool_uses}
            )
            self._pending_tool_uses = []
        results = [
            {
                "type": "tool_result",
                "tool_use_id": output["call_id"],
                "content": output["output"],
            }
            for output in tool_outputs
        ]
        self._messages.append({"role": "user", "content": results})

    def export_state(self) -> dict[str, Any]:
        return {
            "messages": [dict(message) for message in self._messages],
            "pending_tool_uses": [
                dict(block) for block in self._pending_tool_uses
            ],
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self._messages = [
            dict(message) for message in state.get("messages", [])
        ]
        self._pending_tool_uses = [
            dict(block) for block in state.get("pending_tool_uses", [])
        ]

    def restore(self, messages) -> None:
        self._messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message.get("role") in {"user", "assistant"}
        ]
        self._pending_tool_uses = []

    def _with_retry(self, operation):
        attempt = 0
        while True:
            try:
                return operation()
            except Exception as exc:
                attempt += 1
                if (
                    attempt > self.config.max_retries
                    or not self._is_retryable(exc)
                ):
                    raise
                cap = min(
                    self.config.retry_max_delay,
                    self.config.retry_base_delay * (2 ** (attempt - 1)),
                )
                delay = (
                    random.uniform(0, cap)
                    if self.config.retry_jitter
                    else cap
                )
                time.sleep(delay)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return True
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status in {408, 429} or 500 <= status < 600
        return False

    @staticmethod
    def _block_value(block: Any, key: str) -> Any:
        if isinstance(block, dict):
            return block.get(key)
        return getattr(block, key, None)

    @staticmethod
    def _anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }

    @staticmethod
    def _create_client(config: AnthropicProviderConfig) -> Any:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMProviderError(
                "Anthropic SDK is not installed; install the project dependencies"
            ) from exc
        options: dict[str, Any] = {"timeout": config.timeout}
        if config.api_key is not None:
            options["api_key"] = config.api_key
        if config.base_url is not None:
            options["base_url"] = config.base_url
        try:
            return anthropic.Anthropic(**options)
        except Exception as exc:
            raise LLMProviderError(
                f"Anthropic client configuration failed: {exc}"
            ) from exc
