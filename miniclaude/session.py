"""Atomic persistence for completed agent session records."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miniclaude.models import AgentResult


@dataclass(frozen=True, slots=True)
class SessionCheckpoint:
    """Resumable state captured from a run that did not finish."""

    task: str
    max_turns: int
    turn_count: int
    status: str
    provider_response_id: str | None = None
    output: Any = None
    error: str | None = None
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    model_name: str | None = None
    context_truncated: bool = False
    skills_loaded: tuple[str, ...] = ()
    messages: tuple[dict[str, str], ...] = ()
    provider_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "max_turns": self.max_turns,
            "turn_count": self.turn_count,
            "status": self.status,
            "provider_response_id": self.provider_response_id,
            "output": self.output,
            "error": self.error,
            "usage_input_tokens": self.usage_input_tokens,
            "usage_output_tokens": self.usage_output_tokens,
            "model_name": self.model_name,
            "context_truncated": self.context_truncated,
            "skills_loaded": list(self.skills_loaded),
            "messages": list(self.messages),
            "provider_state": self.provider_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionCheckpoint":
        return cls(
            task=str(data["task"]),
            max_turns=int(data["max_turns"]),
            turn_count=int(data["turn_count"]),
            status=str(data["status"]),
            provider_response_id=data.get("provider_response_id"),
            output=data.get("output"),
            error=data.get("error"),
            usage_input_tokens=int(data.get("usage_input_tokens", 0)),
            usage_output_tokens=int(data.get("usage_output_tokens", 0)),
            model_name=data.get("model_name"),
            context_truncated=bool(data.get("context_truncated", False)),
            skills_loaded=tuple(data.get("skills_loaded", ())),
            messages=tuple(
                {"role": message["role"], "content": message["content"]}
                for message in data.get("messages", ())
            ),
            provider_state=data.get("provider_state"),
        )


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
            "metrics": result.metrics.to_dict() if result.metrics is not None else None,
            "skills": list(result.skills),
        }
        return self._write_atomic(target, payload)

    def save_checkpoint(
        self, session_id: str, checkpoint: SessionCheckpoint
    ) -> Path:
        target = self._checkpoint_path(session_id)
        return self._write_atomic(
            target,
            {"version": 1, "checkpoint": checkpoint.to_dict()},
        )

    def load_checkpoint(self, session_id: str) -> SessionCheckpoint:
        data = json.loads(
            self._checkpoint_path(session_id).read_text(encoding="utf-8")
        )
        return SessionCheckpoint.from_dict(data["checkpoint"])

    def _write_atomic(self, target: Path, payload: dict[str, Any]) -> Path:
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

    def _validate_id(self, session_id: str) -> None:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not session_id or any(character not in allowed for character in session_id):
            raise ValueError("invalid session id")

    def _path(self, session_id: str) -> Path:
        self._validate_id(session_id)
        return self.directory / f"{session_id}.json"

    def _checkpoint_path(self, session_id: str) -> Path:
        self._validate_id(session_id)
        return self.directory / f"{session_id}.checkpoint.json"
