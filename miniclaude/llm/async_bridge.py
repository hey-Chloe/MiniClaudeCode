"""RunInLoopProvider: synchronous facade over an async provider.

The agent loop keeps a synchronous ``complete`` contract. When callers want
real asyncio I/O (``AsyncOpenAIProvider``) from the synchronous loop, this
bridge owns a dedicated event loop per instance and runs each request to
completion. Each instance must be used from a single thread, which the
asyncio benchmark runner guarantees by creating one provider per worker
thread.
"""

from __future__ import annotations

import asyncio
from typing import Any

from miniclaude.llm.base import LLMRequest, LLMResponse


class RunInLoopProvider:
    """Run an async provider's ``acomplete`` inside a private event loop."""

    def __init__(self, async_provider: Any):
        self._async_provider = async_provider
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop

    def complete(self, request: LLMRequest) -> LLMResponse:
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_provider.acomplete(request)
        )

    def complete_stream(self, request: LLMRequest):
        loop = self._get_loop()

        async def collect() -> list[str]:
            return [
                delta
                async for delta in self._async_provider.acomplete_stream(
                    request
                )
            ]

        yield from loop.run_until_complete(collect())

    def export_state(self) -> dict[str, Any]:
        exporter = getattr(self._async_provider, "export_state", None)
        return exporter() if exporter is not None else {}

    def restore_state(self, state: dict[str, Any]) -> None:
        restorer = getattr(self._async_provider, "restore_state", None)
        if restorer is not None:
            restorer(state)

    def restore(self, messages) -> None:
        restorer = getattr(self._async_provider, "restore", None)
        if restorer is not None:
            restorer(messages)
