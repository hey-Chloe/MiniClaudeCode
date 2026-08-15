"""Freshness-checked file read cache.

This is the honest, bounded version of "memory": repeated reads of an
unchanged file are served from memory, keyed by path + mtime + size, and
invalidated when the harness itself writes the file. It reduces repeated disk
reads without pretending to be a semantic memory system.
"""

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FileCache:
    """Keyed by ``(path, mtime, size)`` so stale content is never served."""

    _entries: dict[str, tuple[float, int, str]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    invalidations: int = 0

    def get(self, path: str, mtime: float, size: int) -> str | None:
        entry = self._entries.get(path)
        if entry is not None and entry[0] == mtime and entry[1] == size:
            self.hits += 1
            return entry[2]
        self.misses += 1
        return None

    def put(self, path: str, mtime: float, size: int, content: str) -> None:
        self._entries[path] = (mtime, size, content)

    def invalidate(self, path: str) -> None:
        if path in self._entries:
            self.invalidations += 1
            del self._entries[path]

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "invalidations": self.invalidations,
        }


@dataclass(slots=True)
class MemoryEntry:
    """One stored memory with a TTL and optional metadata."""

    key: str
    content: str
    created_at: float
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def expired(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.expires_at is not None and now > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "content": self.content,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryEntry":
        return cls(
            key=str(payload.get("key", "")),
            content=str(payload.get("content", "")),
            created_at=float(payload.get("created_at", 0.0)),
            expires_at=(
                float(payload["expires_at"])
                if payload.get("expires_at") is not None
                else None
            ),
            metadata=dict(payload.get("metadata") or {}),
        )


_TOKEN_RE = re.compile(r"[a-z0-9_\-]+")


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) >= 3
    }


def _rank(query: str, entries: list[MemoryEntry]) -> list[MemoryEntry]:
    """Deterministic keyword-overlap ranking (freshness breaks ties)."""
    query_tokens = _tokens(query)
    scored: list[tuple[int, float, MemoryEntry]] = []
    for entry in entries:
        haystack = f"{entry.key} {entry.content}"
        overlap = len(query_tokens & _tokens(haystack))
        scored.append((overlap, -entry.created_at, entry))
    scored.sort(key=lambda item: (-item[0], item[1], item[2].key))
    return [entry for _, _, entry in scored]


class WorkingMemory:
    """Bounded, in-run key-value memory with TTL and keyword retrieval.

    This is the honest Working Memory: it remembers within one run, expires
    entries by TTL, and retrieves by deterministic keyword overlap. It does
    not pretend to be a semantic or episodic memory system.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 3_600.0,
        max_entries: int = 64,
    ):
        if ttl_seconds <= 0 or max_entries < 1:
            raise ValueError("ttl_seconds and max_entries must be positive")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._entries: dict[str, MemoryEntry] = {}
        self._lock = threading.Lock()

    def put(
        self,
        key: str,
        content: str,
        *,
        ttl: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        if not key.strip() or not content.strip():
            raise ValueError("memory key and content must not be empty")
        now = time.time()
        entry = MemoryEntry(
            key=key,
            content=content,
            created_at=now,
            expires_at=now + (ttl if ttl is not None else self._ttl_seconds),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._entries[key] = entry
            if len(self._entries) > self._max_entries:
                oldest = min(
                    self._entries.values(), key=lambda item: item.created_at
                )
                del self._entries[oldest.key]
        return entry

    def get(self, key: str) -> MemoryEntry | None:
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expired(now):
                del self._entries[key]
                return None
            return entry

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        max_chars: int = 800,
    ) -> list[MemoryEntry]:
        now = time.time()
        with self._lock:
            live = [
                entry
                for entry in self._entries.values()
                if not entry.expired(now)
            ]
        ranked = _rank(query, live)[: max(1, top_k)]
        total = 0
        selected: list[MemoryEntry] = []
        for entry in ranked:
            total += len(entry.content)
            if total > max_chars:
                break
            selected.append(entry)
        return selected

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "expired": sum(
                    1 for entry in self._entries.values() if entry.expired()
                ),
            }


class PersistentMemory:
    """JSONL-backed cross-session memory with TTL and freshness checks.

    Entries survive process restarts, expire by TTL, and are retrieved by the
    same deterministic keyword ranking as ``WorkingMemory``. Writes append to
    a JSONL file; ``save`` rewrites it atomically.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        ttl_seconds: float = 86_400.0,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._path = Path(path)
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, MemoryEntry] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        now = time.time()
        loaded: dict[str, MemoryEntry] = {}
        with self._lock:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = MemoryEntry.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    continue
                if not entry.expired(now):
                    loaded[entry.key] = entry
            self._entries = loaded
        if self._path.exists():
            self.save()

    def put(
        self,
        key: str,
        content: str,
        *,
        ttl: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        if not key.strip() or not content.strip():
            raise ValueError("memory key and content must not be empty")
        now = time.time()
        entry = MemoryEntry(
            key=key,
            content=content,
            created_at=now,
            expires_at=now + (ttl if ttl is not None else self._ttl_seconds),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._entries[key] = entry
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
                )
        return entry

    def get(self, key: str) -> MemoryEntry | None:
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expired(now):
                del self._entries[key]
                self.save()
                return None
            return entry

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        max_chars: int = 800,
    ) -> list[MemoryEntry]:
        now = time.time()
        with self._lock:
            live = [
                entry
                for entry in self._entries.values()
                if not entry.expired(now)
            ]
        return _rank(query, live)[: max(1, top_k)]

    def invalidate(self, key: str) -> None:
        with self._lock:
            if self._entries.pop(key, None) is not None:
                self.save()

    def save(self) -> None:
        rendered = "\n".join(
            json.dumps(entry.to_dict(), ensure_ascii=False)
            for entry in self._entries.values()
        )
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(rendered + ("\n" if rendered else ""), encoding="utf-8")
        temporary.replace(self._path)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._entries)}
