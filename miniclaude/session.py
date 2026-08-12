"""Atomic persistence for completed agent session records."""

import json
import os
import tempfile
from pathlib import Path

from miniclaude.models import AgentResult


class SessionStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, result: AgentResult, detailed_events=()) -> Path:
        target = self._path(session_id)
        payload = {
            "version": 1, "session_id": session_id, "status": result.status.value,
            "task": result.task, "turns": result.turns, "output": result.output,
            "error": result.error, "events": result.events,
            "detailed_events": list(detailed_events),
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.directory, delete=False) as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target

    def load(self, session_id: str) -> dict:
        return json.loads(self._path(session_id).read_text(encoding="utf-8"))

    def _path(self, session_id: str) -> Path:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not session_id or any(character not in allowed for character in session_id):
            raise ValueError("invalid session id")
        return self.directory / f"{session_id}.json"
