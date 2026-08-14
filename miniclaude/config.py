"""Environment and CLI configuration for MiniClaudeCode."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    model: str | None
    api_key: str | None
    base_url: str | None
    workspace: Path
    max_turns: int = 20
    timeout: float = 120.0
    max_output_chars: int = 100_000
    permission_mode: str = "default"
    runtime: str = "local"
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    max_retries: int = 2

    @classmethod
    def from_env(cls, **overrides):
        _load_dotenv()
        values = {
            "model": os.getenv("MINICLAUDE_MODEL"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "workspace": Path(os.getenv("MINICLAUDE_WORKSPACE", ".")).resolve(),
            "max_turns": _positive_int("MINICLAUDE_MAX_TURNS", 20),
            "timeout": _positive_float("MINICLAUDE_TIMEOUT", 120.0),
            "max_output_chars": _positive_int("MINICLAUDE_MAX_OUTPUT_CHARS", 100_000),
            "permission_mode": os.getenv("MINICLAUDE_PERMISSION_MODE", "default"),
            "runtime": os.getenv("MINICLAUDE_RUNTIME", "local"),
            "input_price_per_million": _optional_non_negative_float(
                "MINICLAUDE_INPUT_PRICE_PER_1M"
            ),
            "output_price_per_million": _optional_non_negative_float(
                "MINICLAUDE_OUTPUT_PRICE_PER_1M"
            ),
            "max_retries": _non_negative_int("MINICLAUDE_MAX_RETRIES", 2),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.workspace.is_dir():
            raise ValueError(f"workspace does not exist: {self.workspace}")
        if self.permission_mode not in {"default", "plan", "accept-edits", "bypass"}:
            raise ValueError(f"unsupported permission mode: {self.permission_mode}")
        if self.runtime not in {"local", "docker"}:
            raise ValueError(f"unsupported runtime: {self.runtime}")
        if (self.input_price_per_million is None) != (
            self.output_price_per_million is None
        ):
            raise ValueError(
                "MINICLAUDE_INPUT_PRICE_PER_1M and "
                "MINICLAUDE_OUTPUT_PRICE_PER_1M must be set together"
            )


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _non_negative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _optional_non_negative_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    value = float(raw)
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value

