"""Bounded Reviewer second pass: an LLM-backed verifier before finalize.

This is the honest, single-agent-plus-reviewer version of "Multi-Agent":
after the main agent edits files and declares itself done, a separate
reviewer LLM call inspects the diff and either approves or requests changes.
A rejection is fed back into the main loop (like a failed verification), so
the agent gets a bounded chance to address the comments before finalizing.
There is no shared blackboard or coordinator: the reviewer is a second pass
over the same run, which is what the codebase can genuinely support.
"""

from __future__ import annotations

from typing import Any, Callable

from miniclaude.llm.base import LLMProvider, LLMRequest
from runtime.base import Runtime


REVIEW_INSTRUCTIONS = """You are a reviewer for a coding agent. Inspect the
task and the diff below. Reply with exactly one line:

APPROVED
or
CHANGES_REQUESTED <one short reason>

Then, on a new line, add at most three concrete comments if changes are
requested. Do not modify any files."""


def _collect_diff(runtime: Runtime, max_chars: int = 8_000) -> str:
    try:
        result = runtime.execute(["git", "diff", "--no-ext-diff"], timeout=30)
    except Exception:
        return "(diff unavailable)"
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.succeeded and output.strip():
        return output[:max_chars]
    return "(no diff)"


def build_review_verifier(
    provider: LLMProvider,
    runtime: Runtime,
    *,
    max_diff_chars: int = 8_000,
    max_review_chars: int = 4_000,
) -> Callable[[], dict[str, Any]]:
    """Build a verifier that runs a reviewer LLM pass over the workspace diff.

    The returned callable matches the ``verifier`` contract of
    ``Agent``/``LLMLoopDriver``: ``{"passed": bool, "output": str}``. A
    rejection is fed back to the main agent as verification output.
    """

    def verifier() -> dict[str, Any]:
        diff = _collect_diff(runtime, max_diff_chars)
        response = provider.complete(
            LLMRequest(
                task="Review the changes made by the agent.",
                instructions=REVIEW_INSTRUCTIONS,
                messages=(
                    {
                        "role": "user",
                        "content": f"Diff:\n{diff}",
                    },
                ),
            )
        )
        text = (response.text or "").strip()
        approved = text.upper().startswith("APPROVED")
        return {
            "passed": approved,
            "output": text[:max_review_chars],
        }

    return verifier
