"""AsyncOpenAIProvider: asyncio I/O for OpenAI-compatible endpoints.

The agent loop keeps a synchronous ``complete`` contract for deterministic,
testable execution. This provider is the async I/O boundary: it uses
``openai.AsyncOpenAI`` so callers (and the asyncio benchmark runner) can do
real concurrent network I/O without blocking. Normalization is shared with
the synchronous provider.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from miniclaude.llm.base import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    LLMUsage,
)
from miniclaude.llm.openai_provider import OpenAIProvider, OpenAIProviderConfig


class AsyncOpenAIProvider:
    """Async twin of ``OpenAIProvider`` for Responses / chat endpoints."""

    def __init__(self, config: OpenAIProviderConfig, client: Any | None = None):
        self.config = config
        self.client = (
            client if client is not None else self._create_client(config)
        )
        self._chat_messages: list[dict[str, Any]] = []

    def _uses_chat_completions(self) -> bool:
        return bool(
            self.config.base_url
            and "api.deepseek.com" in self.config.base_url.lower()
        )

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest")
        if self._uses_chat_completions():
            return await self._acomplete_chat(request)
        return await self._acomplete_responses(request)

    async def _acomplete_responses(self, request: LLMRequest) -> LLMResponse:
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
            response = await self._with_retry_async(
                lambda: self.client.responses.create(**parameters)
            )
            return OpenAIProvider._normalize_response(response)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                f"OpenAI async request failed: {exc}"
            ) from exc

    async def _acomplete_chat(self, request: LLMRequest) -> LLMResponse:
        messages = self._prepare_chat_input(request)
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if request.tools:
            parameters["tools"] = [
                OpenAIProvider._chat_tool(tool) for tool in request.tools
            ]
        try:
            response = await self._with_retry_async(
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
            completion_tokens = int(
                getattr(usage, "completion_tokens", 0) or 0
            )
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
            raise LLMProviderError(
                f"DeepSeek async chat request failed: {exc}"
            ) from exc

    async def acomplete_stream(self, request: LLMRequest):
        """Yield incremental text deltas (async iterator)."""
        if self._uses_chat_completions():
            async for delta in self._astream_chat(request):
                yield delta
        else:
            async for delta in self._astream_responses(request):
                yield delta

    async def _astream_chat(self, request: LLMRequest):
        messages = self._prepare_chat_input(request)
        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
        }
        if request.tools:
            parameters["tools"] = [
                OpenAIProvider._chat_tool(tool) for tool in request.tools
            ]
        try:
            stream = await self._with_retry_async(
                lambda: self.client.chat.completions.create(**parameters)
            )
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or ()
                if not choices:
                    continue
                delta = getattr(choices[0].message, "content", None)
                if delta:
                    yield delta
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                f"DeepSeek async chat stream failed: {exc}"
            ) from exc

    async def _astream_responses(self, request: LLMRequest):
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
        try:
            stream = await self._with_retry_async(
                lambda: self.client.responses.create(**parameters)
            )
            async for event in stream:
                if getattr(event, "type", None) != "response.output_text.delta":
                    continue
                delta = getattr(event, "delta", None)
                if delta:
                    yield delta
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                f"OpenAI async stream failed: {exc}"
            ) from exc

    def _prepare_chat_input(self, request: LLMRequest) -> list[dict[str, Any]]:
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

    async def _with_retry_async(self, operation):
        attempt = 0
        while True:
            try:
                return await operation()
            except Exception as exc:
                attempt += 1
                if (
                    attempt > self.config.max_retries
                    or not OpenAIProvider._is_retryable(exc)
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
                await asyncio.sleep(delay)

    def export_state(self) -> dict[str, Any]:
        return {
            "chat_messages": [
                dict(message) for message in self._chat_messages
            ]
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self._chat_messages = [
            dict(message)
            for message in state.get("chat_messages", [])
        ]

    def restore(self, messages: tuple[dict[str, str], ...]) -> None:
        self._chat_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant", "tool"} and content:
                self._chat_messages.append(
                    {"role": role, "content": content}
                )

    @staticmethod
    def _create_client(config: OpenAIProviderConfig) -> Any:
        try:
            from openai import AsyncOpenAI
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
            return AsyncOpenAI(**options)
        except Exception as exc:
            raise LLMProviderError(
                f"Async OpenAI client configuration failed: {exc}"
            ) from exc
