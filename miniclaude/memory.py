"""Freshness-checked file read cache.

This is the honest, bounded version of "memory": repeated reads of an
unchanged file are served from memory, keyed by path + mtime + size, and
invalidated when the harness itself writes the file. It reduces repeated disk
reads without pretending to be a semantic memory system.
"""

from dataclasses import dataclass, field


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
