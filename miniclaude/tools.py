"""Typed tool definitions, validation, registration, and dispatch."""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from security.approval import ApprovalManager
from security.policy import (
    DefaultSecurityPolicy,
    PolicyAction,
    PolicyRequest,
    SecurityPolicy,
    ToolRisk,
)


class ToolError(RuntimeError):
    """Base error raised by the tool system."""


class ToolRegistrationError(ToolError):
    """Raised when a tool cannot be registered safely."""


class ToolValidationError(ToolError):
    """Raised when model-provided arguments do not match a tool schema."""


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A callable tool and its model-facing JSON Schema."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[..., Any] = field(repr=False, compare=False)
    risk: ToolRisk = ToolRisk.READ_ONLY

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name must be a non-empty string")
        if not self.description.strip():
            raise ValueError("tool description must be a non-empty string")
        if self.parameters.get("type") != "object":
            raise ValueError("tool parameters must use an object JSON Schema")
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")
        if not isinstance(self.risk, ToolRisk):
            raise TypeError("tool risk must be a ToolRisk")

    def schema(self) -> dict[str, Any]:
        """Return the provider-neutral function tool schema."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
            "strict": True,
        }


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """Uniform result of one attempted tool call."""

    call_id: str
    name: str
    success: bool
    output: Any = None
    error: str | None = None
    policy_action: str | None = None
    policy_reason: str | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "policy_action": self.policy_action,
            "policy_reason": self.policy_reason,
            "duration_seconds": self.duration_seconds,
        }

    def model_output(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class ToolRegistry:
    """Owns explicitly registered tools and dispatches calls by exact name."""

    def __init__(
        self,
        policy: SecurityPolicy | None = None,
        approvals: ApprovalManager | None = None,
    ):
        self.tools: dict[str, ToolDefinition] = {}
        self.policy = policy if policy is not None else DefaultSecurityPolicy()
        self.approvals = approvals if approvals is not None else ApprovalManager()

    def register(self, tool: ToolDefinition) -> None:
        if not isinstance(tool, ToolDefinition):
            raise TypeError("tool must be a ToolDefinition")
        if tool.name in self.tools:
            raise ToolRegistrationError(f"tool already registered: {tool.name}")
        self.tools[tool.name] = tool

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def dispatch(self, call_id: str, name: str, arguments: str) -> ToolObservation:
        started = time.monotonic()
        tool = self.tools.get(name)
        if tool is None:
            return ToolObservation(
                call_id=call_id,
                name=name,
                success=False,
                error=f"unknown tool: {name}",
                duration_seconds=time.monotonic() - started,
            )

        try:
            values = json.loads(arguments or "{}")
            if not isinstance(values, dict):
                raise ToolValidationError("tool arguments must decode to an object")
            self._validate_object(values, tool.parameters)
            request = PolicyRequest(tool.name, tool.risk, values)
            decision = self.policy.evaluate(request)
            allowed, reason = self.approvals.authorize(request, decision)
            if not allowed:
                return ToolObservation(
                    call_id,
                    name,
                    False,
                    error=f"tool blocked: {reason}",
                    policy_action=decision.action.value,
                    policy_reason=reason,
                    duration_seconds=time.monotonic() - started,
                )
            output = tool.handler(**values)
            return ToolObservation(
                call_id,
                name,
                True,
                output=output,
                policy_action=decision.action.value,
                policy_reason=reason,
                duration_seconds=time.monotonic() - started,
            )
        except (json.JSONDecodeError, ToolValidationError, TypeError) as exc:
            return ToolObservation(
                call_id, name, False, error=str(exc),
                duration_seconds=time.monotonic() - started,
            )
        except Exception as exc:
            return ToolObservation(
                call_id,
                name,
                False,
                error=f"tool execution failed: {exc}",
                duration_seconds=time.monotonic() - started,
            )

    @classmethod
    def _validate_object(
        cls, values: Mapping[str, Any], schema: Mapping[str, Any]
    ) -> None:
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        missing = [key for key in required if key not in values]
        if missing:
            raise ToolValidationError(
                f"missing required arguments: {', '.join(sorted(missing))}"
            )

        if schema.get("additionalProperties") is False:
            extra = set(values) - set(properties)
            if extra:
                raise ToolValidationError(
                    f"unexpected arguments: {', '.join(sorted(extra))}"
                )

        for key, value in values.items():
            property_schema = properties.get(key)
            if property_schema:
                cls._validate_value(key, value, property_schema)

    @classmethod
    def _validate_value(
        cls, name: str, value: Any, schema: Mapping[str, Any]
    ) -> None:
        expected = schema.get("type")
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
            "null": type(None),
        }
        python_type = type_map.get(expected)
        if python_type is not None:
            valid = isinstance(value, python_type)
            if expected in {"integer", "number"} and isinstance(value, bool):
                valid = False
            if not valid:
                raise ToolValidationError(f"argument '{name}' must be {expected}")

        if "enum" in schema and value not in schema["enum"]:
            raise ToolValidationError(f"argument '{name}' is not an allowed value")
        if expected == "array" and "items" in schema:
            for index, item in enumerate(value):
                cls._validate_value(f"{name}[{index}]", item, schema["items"])
