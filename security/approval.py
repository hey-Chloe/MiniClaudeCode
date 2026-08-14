"""Session-scoped human approval handling."""

import json
import threading
from collections.abc import Callable

from security.policy import PolicyAction, PolicyDecision, PolicyRequest


ApprovalCallback = Callable[[PolicyRequest, PolicyDecision], bool]


class ApprovalManager:
    """Resolves ASK decisions and remembers exact approvals for one session."""

    def __init__(self, callback: ApprovalCallback | None = None):
        self.callback = callback
        self._approved: set[str] = set()
        self._lock = threading.Lock()

    def authorize(
        self, request: PolicyRequest, decision: PolicyDecision
    ) -> tuple[bool, str]:
        if decision.action is PolicyAction.ALLOW:
            return True, decision.reason
        if decision.action is PolicyAction.DENY:
            return False, decision.reason

        key = self._key(request)
        with self._lock:
            if key in self._approved:
                return True, "approved earlier in this session"
        if self.callback is None:
            return False, "approval required but no approval callback is configured"
        if not self.callback(request, decision):
            return False, "user denied approval"

        with self._lock:
            self._approved.add(key)
        return True, "user approved for this session"

    def clear(self) -> None:
        with self._lock:
            self._approved.clear()

    @staticmethod
    def _key(request: PolicyRequest) -> str:
        arguments = json.dumps(
            request.arguments, ensure_ascii=False, sort_keys=True, default=str
        )
        return f"{request.tool_name}:{arguments}"
