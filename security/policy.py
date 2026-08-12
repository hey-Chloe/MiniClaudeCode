"""Tool-level security policy primitives."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol


class PolicyAction(str, Enum):
    """Explicit result of a security policy evaluation."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class ToolRisk(str, Enum):
    """Declared side-effect class of a tool."""

    READ_ONLY = "read_only"
    MUTATING = "mutating"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    tool_name: str
    risk: ToolRisk
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: PolicyAction
    reason: str


class SecurityPolicy(Protocol):
    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        """Return an explicit authorization decision for a tool call."""


class DefaultSecurityPolicy:
    """Conservative default policy based on declared tool risk."""

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        if request.tool_name == "execute_command":
            from security.command_analysis import assess_argv

            assessment = assess_argv(request.arguments.get("argv"))
            return PolicyDecision(assessment.action, assessment.reason)
        if request.risk is ToolRisk.READ_ONLY:
            return PolicyDecision(PolicyAction.ALLOW, "read-only tool")
        if request.risk is ToolRisk.MUTATING:
            return PolicyDecision(PolicyAction.ASK, "mutating tool requires approval")
        return PolicyDecision(PolicyAction.DENY, "destructive tool denied by default")


class PermissionModePolicy(DefaultSecurityPolicy):
    """Maps CLI permission modes onto explicit policy decisions."""

    def __init__(self, mode: str = "default"):
        if mode not in {"default", "plan", "accept-edits", "bypass"}:
            raise ValueError(f"unsupported permission mode: {mode}")
        self.mode = mode

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        if self.mode == "plan" and request.risk is not ToolRisk.READ_ONLY:
            return PolicyDecision(PolicyAction.DENY, "plan mode forbids side effects")
        if self.mode == "bypass":
            return PolicyDecision(PolicyAction.ALLOW, "bypass mode")
        if self.mode == "accept-edits" and request.risk is ToolRisk.MUTATING:
            if request.tool_name == "write_file":
                return PolicyDecision(PolicyAction.ALLOW, "accept-edits mode")
        return super().evaluate(request)
