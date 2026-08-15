"""Bounded Multi-Agent collaboration for MiniClaudeCode.

Real, testable implementation of the resume's Multi-Agent story:

- ``CollaborationBlackboard``: thread-safe shared evidence store with
  provenance, content dedup, query, and verification marks;
- ``SpecialistAgent``: one agent scoped to a subtask with a specialized
  system prompt and a tool subset;
- ``CoordinatorAgent``: decomposes a task, runs specialists concurrently,
  collects and cross-checks evidence, and synthesizes the final answer;
- optional ``Critic`` second pass over the synthesized answer.

There is deliberately no fake "agent framework": every component is a plain
class over the existing ``Agent`` loop, and every behavior is unit-tested.
"""

from miniclaude.agents.blackboard import (
    CollaborationBlackboard,
    Evidence,
)
from miniclaude.agents.coordinator import (
    CoordinatorAgent,
    MultiAgentResult,
    SpecialistAgent,
    Subtask,
)

__all__ = [
    "CollaborationBlackboard",
    "Evidence",
    "CoordinatorAgent",
    "SpecialistAgent",
    "MultiAgentResult",
    "Subtask",
]
