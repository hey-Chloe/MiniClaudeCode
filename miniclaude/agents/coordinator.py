"""Coordinator/Specialist orchestration over the existing Agent loop."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from miniclaude.agent import Agent
from miniclaude.agents.blackboard import CollaborationBlackboard
from miniclaude.context import ContextConfig
from miniclaude.llm.base import LLMProvider, LLMRequest
from miniclaude.tools import ToolDefinition


DEFAULT_SPECIALIST_PROMPTS: Mapping[str, str] = {
    "analyzer": (
        "You are the repository-analysis specialist. Explore the workspace "
        "structure, locate relevant files and markers, and report concrete "
        "findings with file paths. Prefer read-only tools."
    ),
    "implementer": (
        "You are the implementation specialist. Locate the root cause, apply "
        "a minimal fix, and report exactly what changed and why. Prefer "
        "read-before-write and verify after editing."
    ),
    "verifier": (
        "You are the verification specialist. Run the relevant checks, "
        "inspect the diff, and report whether the change is sound."
    ),
}


@dataclass(frozen=True, slots=True)
class Subtask:
    """One decomposed unit of work assigned to a specialist."""

    id: str
    description: str
    specialist: str
    tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SpecialistResult:
    """Outcome of one specialist run, including published evidence ids."""

    subtask_id: str
    specialist: str
    status: str
    output: str
    turns: int
    tokens: int
    evidence_ids: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "specialist": self.specialist,
            "status": self.status,
            "output": self.output,
            "turns": self.turns,
            "tokens": self.tokens,
            "evidence_ids": list(self.evidence_ids),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class MultiAgentResult:
    """Aggregated outcome of one coordinated multi-agent run."""

    status: str
    output: str
    subtasks: tuple[Subtask, ...]
    specialist_results: tuple[SpecialistResult, ...]
    blackboard_stats: dict[str, int]
    evidence: tuple[dict[str, Any], ...]
    critic: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output": self.output,
            "subtasks": [
                {
                    "id": subtask.id,
                    "description": subtask.description,
                    "specialist": subtask.specialist,
                }
                for subtask in self.subtasks
            ],
            "specialists": [
                result.to_dict() for result in self.specialist_results
            ],
            "blackboard": self.blackboard_stats,
            "evidence": list(self.evidence),
            "critic": self.critic,
        }


def default_decompose(task: str) -> tuple[Subtask, ...]:
    """Deterministic keyword-based decomposition into specialist subtasks."""
    lowered = task.lower()
    subtasks: list[Subtask] = []
    if any(
        keyword in lowered
        for keyword in (
            "search",
            "find",
            "todo",
            "structure",
            "review",
            "analyze",
            "inspect",
        )
    ):
        subtasks.append(
            Subtask(
                id="analyze",
                description=(
                    "Analyze the workspace: structure, markers, and relevant "
                    "files for the task."
                ),
                specialist="analyzer",
                tools=(
                    "file_tree",
                    "todo_scan",
                    "grep_files",
                    "read_file",
                    "file_stat",
                    "glob_files",
                    "list_directory",
                ),
            )
        )
    if any(
        keyword in lowered
        for keyword in (
            "fix",
            "bug",
            "test",
            "implement",
            "refactor",
            "change",
            "write",
        )
    ):
        subtasks.append(
            Subtask(
                id="implement",
                description="Locate the root cause and implement the change.",
                specialist="implementer",
                tools=(
                    "read_file",
                    "grep_files",
                    "write_file",
                    "replace_text",
                    "execute_command",
                    "git_diff",
                    "workspace_diff",
                    "file_stat",
                ),
            )
        )
    if not subtasks:
        subtasks.append(
            Subtask(
                id="analyze",
                description=f"Work on the task: {task}",
                specialist="analyzer",
            )
        )
    if len(subtasks) == 2:
        subtasks.append(
            Subtask(
                id="verify",
                description="Run the relevant checks and verify the result.",
                specialist="verifier",
                tools=(
                    "execute_command",
                    "read_file",
                    "git_diff",
                    "workspace_diff",
                ),
            )
        )
    return tuple(subtasks)


class SpecialistAgent:
    """One agent scoped to a subtask with its own prompt and tool subset."""

    def __init__(
        self,
        *,
        name: str,
        provider: LLMProvider,
        workspace: Path,
        system_instructions: str,
        tools: Sequence[ToolDefinition],
        blackboard: CollaborationBlackboard | None = None,
        max_turns: int = 10,
    ):
        self.name = name
        self._provider = provider
        self._workspace = workspace
        self._system_instructions = system_instructions
        self._tools = tools
        self._blackboard = blackboard
        self._max_turns = max_turns

    def run(self, subtask: Subtask) -> SpecialistResult:
        selected = [
            tool
            for tool in self._tools
            if not subtask.tools or tool.name in subtask.tools
        ]
        agent = Agent(
            provider=self._provider,
            tools=selected,
            max_turns=self._max_turns,
            approval_callback=lambda *_: True,
            context_config=ContextConfig(
                workspace=self._workspace,
                system_instructions=self._system_instructions,
            ),
            plan_first=False,
        )
        try:
            result = agent.run_result(subtask.description)
        except Exception as exc:
            return SpecialistResult(
                subtask_id=subtask.id,
                specialist=self.name,
                status="failed",
                output="",
                turns=0,
                tokens=0,
                error=str(exc),
            )
        metrics = result.metrics
        evidence_ids: list[str] = []
        if self._blackboard is not None:
            if result.output:
                evidence_ids.append(
                    self._blackboard.publish(
                        self.name,
                        "answer",
                        str(result.output)[:1000],
                    ).id
                )
            for event in result.events:
                if event.get("event") != "tool_results":
                    continue
                for observation in event.get("detail") or []:
                    arguments = observation.get("arguments") or {}
                    source = (
                        str(arguments.get("path"))
                        if isinstance(arguments, dict)
                        else ""
                    )
                    content = (
                        f"{observation.get('name')}: "
                        + (
                            "ok"
                            if observation.get("success")
                            else (
                                "failed: "
                                + str(observation.get("error"))
                            )
                        )
                    )
                    evidence_ids.append(
                        self._blackboard.publish(
                            self.name,
                            "tool_observation",
                            content[:500],
                            source=source,
                        ).id
                    )
        return SpecialistResult(
            subtask_id=subtask.id,
            specialist=self.name,
            status=result.status.value,
            output=str(result.output or ""),
            turns=result.turns,
            tokens=(
                metrics.total_tokens
                if metrics is not None
                else 0
            ),
            evidence_ids=tuple(evidence_ids),
            error=(
                result.error
                if result.status.value != "completed"
                else None
            ),
        )


class CoordinatorAgent:
    """Decomposes a task, runs specialists concurrently, and synthesizes."""

    def __init__(
        self,
        *,
        provider_factory: Callable[[str], LLMProvider],
        workspace: Path,
        tools: Sequence[ToolDefinition],
        blackboard: CollaborationBlackboard | None = None,
        decomposer: Callable[[str], tuple[Subtask, ...]] | None = None,
        specialist_prompts: Mapping[str, str] | None = None,
        max_specialists: int = 3,
        concurrency: int = 2,
        max_turns: int = 10,
        critic_provider: LLMProvider | None = None,
    ):
        self._provider_factory = provider_factory
        self._workspace = workspace
        self._tools = tools
        self._blackboard = blackboard if blackboard is not None else CollaborationBlackboard()
        self._decomposer = decomposer or default_decompose
        self._prompts = dict(DEFAULT_SPECIALIST_PROMPTS)
        self._prompts.update(specialist_prompts or {})
        self._max_specialists = max_specialists
        self._concurrency = concurrency
        self._max_turns = max_turns
        self._critic_provider = critic_provider

    def decompose(self, task: str) -> tuple[Subtask, ...]:
        subtasks = self._decomposer(task)
        return tuple(subtasks[: self._max_specialists])

    def run(self, task: str) -> MultiAgentResult:
        subtasks = self.decompose(task)
        if not subtasks:
            raise ValueError("decomposer returned no subtasks")
        providers: dict[str, LLMProvider] = {}
        lock = threading.Lock()

        def make_specialist(subtask: Subtask) -> SpecialistAgent:
            with lock:
                if subtask.specialist not in providers:
                    providers[subtask.specialist] = self._provider_factory(
                        subtask.specialist
                    )
                provider = providers[subtask.specialist]
            return SpecialistAgent(
                name=subtask.specialist,
                provider=provider,
                workspace=self._workspace,
                system_instructions=self._prompts.get(
                    subtask.specialist,
                    "You are a coding specialist.",
                ),
                tools=list(self._tools),
                blackboard=self._blackboard,
                max_turns=self._max_turns,
            )

        with ThreadPoolExecutor(
            max_workers=min(self._concurrency, len(subtasks))
        ) as pool:
            futures = [
                pool.submit(make_specialist(subtask).run, subtask)
                for subtask in subtasks
            ]
            results = tuple(future.result() for future in futures)

        for evidence in self._blackboard.items():
            if evidence.source:
                self._blackboard.verify(
                    evidence.id,
                    (self._workspace / evidence.source).exists(),
                )

        status = (
            "completed"
            if all(result.status == "completed" for result in results)
            else "partial"
        )
        sections = [
            f"[{result.specialist}] {result.output}"
            for result in results
            if result.output
        ]
        output = "\n\n".join(sections) if sections else "(no specialist output)"
        critic = None
        if self._critic_provider is not None:
            critic = self._run_critic(output)
        return MultiAgentResult(
            status=status,
            output=output,
            subtasks=subtasks,
            specialist_results=results,
            blackboard_stats=self._blackboard.stats(),
            evidence=tuple(item.to_dict() for item in self._blackboard.items()),
            critic=critic,
        )

    def _run_critic(self, output: str) -> dict[str, Any]:
        response = self._critic_provider.complete(
            LLMRequest(
                task="Critically review the synthesized answer.",
                instructions=(
                    "Reply with exactly one line: APPROVED or "
                    "CHANGES_REQUESTED <reason>, then optional comments."
                ),
                messages=({"role": "user", "content": output},),
            )
        )
        text = (response.text or "").strip()
        return {
            "approved": text.upper().startswith("APPROVED"),
            "comments": text[:2000],
        }
