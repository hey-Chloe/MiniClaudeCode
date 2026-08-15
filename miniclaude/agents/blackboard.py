"""Shared evidence blackboard for multi-agent collaboration."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class Evidence:
    """One piece of shared evidence with provenance and verification state."""

    id: str
    agent: str
    kind: str
    content: str
    source: str = ""
    created_at: float = field(default_factory=time.time)
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "kind": self.kind,
            "content": self.content,
            "source": self.source,
            "created_at": self.created_at,
            "verified": self.verified,
        }


class CollaborationBlackboard:
    """Thread-safe shared store: publish, dedup, query, and verify evidence."""

    def __init__(self):
        self._items: dict[str, Evidence] = {}
        self._lock = threading.Lock()

    def publish(
        self,
        agent: str,
        kind: str,
        content: str,
        *,
        source: str = "",
    ) -> Evidence:
        """Publish evidence; identical (kind, content, source) dedups."""
        if not agent.strip() or not kind.strip() or not content.strip():
            raise ValueError("evidence agent, kind, and content must not be empty")
        digest = hashlib.sha1(
            f"{kind}\x00{content}\x00{source}".encode("utf-8")
        ).hexdigest()[:16]
        with self._lock:
            existing = next(
                (item for item in self._items.values() if item.id == digest),
                None,
            )
            if existing is not None:
                return existing
            evidence = Evidence(
                id=digest,
                agent=agent,
                kind=kind,
                content=content,
                source=source,
            )
            self._items[digest] = evidence
            return evidence

    def get(self, evidence_id: str) -> Evidence | None:
        with self._lock:
            return self._items.get(evidence_id)

    def query(
        self,
        *,
        kind: str | None = None,
        keyword: str | None = None,
        verified: bool | None = None,
    ) -> list[Evidence]:
        with self._lock:
            items = list(self._items.values())
        if kind is not None:
            items = [item for item in items if item.kind == kind]
        if keyword is not None:
            lowered = keyword.lower()
            items = [
                item
                for item in items
                if lowered in f"{item.content} {item.source}".lower()
            ]
        if verified is not None:
            items = [item for item in items if item.verified is verified]
        return sorted(items, key=lambda item: item.created_at)

    def verify(self, evidence_id: str, ok: bool) -> bool:
        """Mark evidence verified (or not); returns False if unknown id."""
        with self._lock:
            item = self._items.get(evidence_id)
            if item is None:
                return False
            self._items[evidence_id] = Evidence(
                id=item.id,
                agent=item.agent,
                kind=item.kind,
                content=item.content,
                source=item.source,
                created_at=item.created_at,
                verified=ok,
            )
            return True

    def items(self) -> list[Evidence]:
        with self._lock:
            return list(self._items.values())

    def stats(self) -> dict[str, int]:
        with self._lock:
            verified = sum(1 for item in self._items.values() if item.verified)
            return {
                "evidence": len(self._items),
                "verified": verified,
                "unverified": len(self._items) - verified,
            }

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def evidence_to_dicts(items: Iterable[Evidence]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in items]
