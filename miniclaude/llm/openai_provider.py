"""OpenAI Responses API provider implementation."""

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
class OpenAIProviderConfig:
    """Configuration for an OpenAI-compatible Responses API endpoint."""

    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 120.0
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
        if self.max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if self.retry_base_delay < 0 or self.retry_max_delay < 0:
            raise ValueError("retry delays must not be negative")


class OpenAIProvider:
    """Calls the Responses API and normalizes SDK response objects."""

    def __init__(self, config: OpenAIProviderConfig, client: Any | None = None):
        self.config = config
        self.client = client if client is not None else self._create_client(config)
        self._chat_messages: list[dict[str, Any]] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")

        if self._uses_chat_completions():
            return self._complete_chat(request)

        model_input: Any = request.task
        if request.tool_outputs:
            model_input = [
                {
                    "type": "function_call_output",
                    "call_id": output["call_id"],
                    "output": output["output"],
                }
                for output in request.tool_outputs
            ]

        parameters: dict[str, Any] = {"model": self.config.model, "input": model_input}
        instructions = "\n\n".join(
            value
            for value in (self.config.instructions, request.instructions)
            if value
        )
        if instructions:
            parameters["instructions"] = instructions
        if request.tools:
            parameters["tools"] = list(request.tools)
        if request.previous_response_id:
            parameters["previous_response_id"] = request.previous_response_id

        try:
            response = self._with_retry(
                lambda: self.client.responses.create(**parameters)
            )
            return self._normalize_response(response)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"OpenAI request failed: {exc}") from exc

    def _uses_chat_completions(self) -> bool:
        return bool(
            self.config.base_url
            and "api.deepseek.com" in self.config.base_url.lower()
        )

    def _complete_chat(self, request: LLMRequest) -> LLMResponse:
        messages = self._prepare_chat_input(request)

        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
        }
        if request.tools:
            parameters["tools"] = [self._chat_tool(tool) for tool in request.tools]

        try:
            response = self._with_retry(
                lambda: self.client.chat.completions.create(**parameters)
            )
            message = response.choices[0].message
            raw_tool_calls = getattr(message, "tool_calls", None) or []
            tool_calls = tuple(
                LLMToolCall(
                    call_id=str(call.id),
                    name=str(call.function.name),
                    arguments=str(call.function.arguments),
                )
                for call in raw_tool_calls
            )
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": getattr(message, "content", None),
            }
            if raw_tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in raw_tool_calls
                ]
            self._chat_messages.append(assistant_message)
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            return LLMResponse(
                text=getattr(message, "content", None) or "",
                tool_calls=tool_calls,
                response_id=getattr(response, "id", None),
                model=getattr(response, "model", None),
                usage=LLMUsage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=int(
                        getattr(usage, "total_tokens", 0)
                        or prompt_tokens + completion_tokens
                    ),
                ),
                raw=response,
            )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"DeepSeek chat request failed: {exc}") from exc

    def complete_stream(self, request: LLMRequest):
        """Stream text deltas for one request (Responses or chat path)."""
        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")
        if self._uses_chat_completions():
            yield from self._complete_chat_stream(request)
        else:
            yield from self._complete_responses_stream(request)

    def _prepare_chat_input(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Build the chat message list, mutating provider state on turn zero."""
        if request.turn == 0:
            self._chat_messages = []
            instructions = "\n\n".join(
                value
                for value in (self.config.instructions, request.instructions)
                if value
            )
            if instructions:
                self._chat_messages.append(
                    {"role": "system", "content": instructions}
                )
            self._chat_messages.append(
                {"role": "user", "content": request.task}
            )
        for output in request.tool_outputs:
            self._chat_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": output["call_id"],
                    "content": output["output"],
                }
            )
        return list(self._chat_messages)

    def _complete_chat_stream(self, request: LLMRequest):
        messages = self._prepare_chat_input(request)
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
        }
        if request.tools:
            parameters["tools"] = [
                self._chat_tool(tool) for tool in request.tools
            ]
        try:
            response = self._with_retry(
                lambda: self.client.chat.completions.create(**parameters)
            )
            parts: list[str] = []
            for chunk in response:
                choices = getattr(chunk, "choices", None) or ()
                if not choices:
                    continue
                delta = getattr(choices[0].message, "content", None)
                if delta:
                    parts.append(delta)
                    yield delta
            content = "".join(parts)
            if content:
                self._chat_messages.append(
                    {"role": "assistant", "content": content}
                )
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                f"DeepSeek chat stream failed: {exc}"
            ) from exc

    def _complete_responses_stream(self, request: LLMRequest):
        model_input: Any = request.task
        if request.tool_outputs:
            model_input = [
                {
                    "type": "function_call_output",
                    "call_id": output["call_id"],
                    "output": output["output"],
                }
                for output in request.tool_outputs
            ]
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "input": model_input,
            "stream": True,
        }
        instructions = "\n\n".join(
            value
            for value in (self.config.instructions, request.instructions)
            if value
        )
        if instructions:
            parameters["instructions"] = instructions
        if request.tools:
            parameters["tools"] = list(request.tools)
        if request.previous_response_id:
            parameters["previous_response_id"] = request.previous_response_id
        try:
            response = self._with_retry(
                lambda: self.client.responses.create(**parameters)
            )
            for event in response:
                if getattr(event, "type", None) != "response.output_text.delta":
                    continue
                delta = getattr(event, "delta", None)
                if delta:
                    yield delta
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"OpenAI stream failed: {exc}") from exc

    def restore(self, messages: tuple[dict[str, str], ...]) -> None:
        """Seed chat-completions history when resuming a session.

        The Responses API path resumes through ``previous_response_id``; this
        method only matters for the DeepSeek chat adapter, which keeps its own
        in-memory message list.
        """
        self._chat_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant", "tool"} and content:
                self._chat_messages.append(
                    {"role": role, "content": content}
                )

    def export_state(self) -> dict[str, Any]:
        """Export provider-local state (chat history) for checkpointing.

        The chat-completions path needs the exact message list, including
        assistant ``tool_calls`` and tool ``tool_call_id`` links, to resume
        without losing the tool-call correlation.
        """
        return {
            "chat_messages": [
                dict(message) for message in self._chat_messages
            ]
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore provider-local state from a checkpoint."""
        self._chat_messages = [
            dict(message)
            for message in state.get("chat_messages", [])
        ]

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
    def _chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }

    @staticmethod
    def _create_client(config: OpenAIProviderConfig) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMProviderError(
                "OpenAI SDK is not installed; install the project dependencies"
            ) from exc

        options: dict[str, Any] = {"timeout": config.timeout}
        if config.api_key is not None:
            options["api_key"] = config.api_key
        if config.base_url is not None:
            options["base_url"] = config.base_url

        try:
            return OpenAI(**options)
        except Exception as exc:
            raise LLMProviderError(f"OpenAI client configuration failed: {exc}") from exc

    @classmethod
    def _normalize_response(cls, response: Any) -> LLMResponse:
        text = getattr(response, "output_text", "") or ""
        tool_calls: list[LLMToolCall] = []

        for item in getattr(response, "output", ()) or ():
            if cls._value(item, "type") != "function_call":
                continue
            tool_calls.append(
                LLMToolCall(
                    call_id=str(cls._value(item, "call_id") or ""),
                    name=str(cls._value(item, "name") or ""),
                    arguments=str(cls._value(item, "arguments") or ""),
                )
            )

        usage_object = getattr(response, "usage", None)
        input_tokens = int(cls._value(usage_object, "input_tokens") or 0)
        output_tokens = int(cls._value(usage_object, "output_tokens") or 0)
        total_tokens = int(
            cls._value(usage_object, "total_tokens") or input_tokens + output_tokens
        )

        try:
            return LLMResponse(
                text=text,
                tool_calls=tuple(tool_calls),
                response_id=getattr(response, "id", None),
                model=getattr(response, "model", None),
                usage=LLMUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                ),
                raw=response,
            )
        except ValueError as exc:
            raise LLMProviderError("OpenAI returned an empty response") from exc

    @staticmethod
    def _value(value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)
