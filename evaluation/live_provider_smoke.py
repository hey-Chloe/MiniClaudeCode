"""Explicit live smoke test for the configured OpenAI-compatible provider.

This module is intentionally not collected by pytest because it consumes network
and provider quota. Run it manually with ``python -m evaluation.live_provider_smoke``.
"""

import json
import sys

from miniclaude.config import AppConfig
from miniclaude.llm import LLMProviderError, LLMRequest, OpenAIProvider, OpenAIProviderConfig


def _masked(value: str | None) -> str:
    if not value:
        return "absent"
    if len(value) < 6:
        return "present:***"
    return f"present:{value[:3]}***{value[-2:]}"


def main() -> int:
    config = AppConfig.from_env()
    print(
        json.dumps(
            {
                "model": config.model,
                "base_url": config.base_url,
                "api_key": _masked(config.api_key),
            },
            ensure_ascii=False,
        )
    )
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", config.api_key),
            ("OPENAI_BASE_URL", config.base_url),
            ("MINICLAUDE_MODEL", config.model),
        )
        if not value
    ]
    if missing:
        print(f"configuration error: missing {', '.join(missing)}", file=sys.stderr)
        return 2

    provider = OpenAIProvider(
        OpenAIProviderConfig(
            model=config.model or "",
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
    )
    try:
        response = provider.complete(
            LLMRequest(task="Reply with exactly: MiniClaudeCode provider smoke OK")
        )
    except LLMProviderError as exc:
        message = str(exc)
        category = "authentication_rejected" if "401" in message else "provider_error"
        print(f"{category}: {message}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "model": response.model,
                "text": response.text,
                "usage": response.usage.total_tokens,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
