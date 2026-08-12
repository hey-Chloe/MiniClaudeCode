"""OpenAI Responses API provider implementation."""

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

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")


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
            response = self.client.responses.create(**parameters)
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
        if request.turn == 0:
            self._chat_messages = []
            instructions = "\n\n".join(
                value
                for value in (self.config.instructions, request.instructions)
                if value
            )
            if instructions:
                self._chat_messages.append({"role": "system", "content": instructions})
            self._chat_messages.append({"role": "user", "content": request.task})

        for output in request.tool_outputs:
            self._chat_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": output["call_id"],
                    "content": output["output"],
                }
            )

        parameters: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(self._chat_messages),
            "stream": False,
        }
        if request.tools:
            parameters["tools"] = [self._chat_tool(tool) for tool in request.tools]

        try:
            response = self.client.chat.completions.create(**parameters)
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
