"""Adapter from provider responses to loop decisions with explicit phases."""

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from typing import Any

from miniclaude.context import ContextManager
from miniclaude.llm.base import LLMProvider, LLMRequest
from miniclaude.models import AgentPhase, AgentState, LoopDecision
from miniclaude.tools import ToolRegistry
from security.policy import ToolRisk


_WRITE_TOOLS = {"write_file", "replace_text"}


class LLMLoopDriver:
    """Converts normalized provider output into agent-loop decisions."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry | None = None,
        context: ContextManager | None = None,
        verifier: Callable[[], dict[str, Any]] | None = None,
        plan_first: bool = True,
        tool_gating: bool = True,
    ):
        self.provider = provider
        self.tools = tools if tools is not None else ToolRegistry()
        self.context = context if context is not None else ContextManager()
        self.verifier = verifier
        self.plan_first = plan_first
        self.tool_gating = tool_gating
        self._dirty = False
        self._final_text: str | None = None

    def next(self, state: AgentState) -> LoopDecision:
        if self._final_text is not None:
            text = self._final_text
            self._final_text = None
            self.context.add_assistant(text)
            return LoopDecision(
                "answer",
                text,
                terminal=True,
                phase=AgentPhase.FINALIZE,
            )

        if state.turn_count == 0:
            snapshot = self.context.start(state.task)
            state.skills_loaded = snapshot.skills
        else:
            snapshot = self.context.snapshot()
        if snapshot.compression:
            state.context_compression = {
                key: state.context_compression.get(key, 0) + value
                for key, value in snapshot.compression.items()
            }
        if self.tool_gating and state.turn_count == 0:
            self.tools.activate_for_task(state.task)
            for tool_name in self.context.selected_skill_tools():
                self.tools.activate([tool_name])
        schemas = tuple(self.tools.schemas())
        state.tools_sent += len(schemas)
        response = self.provider.complete(
            LLMRequest(
                task=state.task,
                turn=state.turn_count,
                tools=schemas,
                tool_outputs=tuple(state.tool_outputs),
                previous_response_id=state.provider_response_id,
                instructions=snapshot.instructions,
                messages=tuple(
                    {"role": message.role, "content": message.content}
                    for message in snapshot.messages
                ),
                context_truncated=snapshot.truncated,
            )
        )
        state.provider_response_id = response.response_id
        state.tool_outputs.clear()
        state.usage_input_tokens += response.usage.input_tokens
        state.usage_output_tokens += response.usage.output_tokens
        if response.model:
            state.model_name = response.model
        if snapshot.truncated:
            state.context_truncated = True

        if response.tool_calls:
            self.context.add_assistant(
                "Tool calls: "
                + ", ".join(call.name for call in response.tool_calls)
            )
            observations, parallel_count = self._dispatch_batch(
                response.tool_calls
            )
            if parallel_count > 1:
                state.parallel_batches += 1
                state.max_parallelism = max(
                    state.max_parallelism, parallel_count
                )
            if any(
                observation.success
                and observation.name in _WRITE_TOOLS
                for observation in observations
            ):
                self._dirty = True
            if self.tool_gating:
                self.tools.activate(
                    observation.name
                    for observation in observations
                    if observation.success
                )
            state.tool_outputs.extend(
                {
                    "call_id": observation.call_id,
                    "output": observation.model_output(),
                }
                for observation in observations
            )
            for observation in observations:
                self.context.add_tool(observation.model_output())
            return LoopDecision(
                event="tool_results",
                detail=[observation.to_dict() for observation in observations],
                phase=(
                    AgentPhase.REFLECT
                    if any(
                        not observation.success
                        for observation in observations
                    )
                    else AgentPhase.ACT
                ),
            )

        if self.verifier is not None and self._dirty:
            result = self.verifier()
            passed = bool(result.get("passed"))
            if passed:
                self._dirty = False
                self._final_text = response.text
            self.context.add_tool(
                "Verification: "
                + ("passed" if passed else "failed")
                + "\n"
                + str(result.get("output", ""))
            )
            return LoopDecision(
                event="verification",
                detail=result,
                phase=AgentPhase.VERIFY,
            )

        if state.turn_count == 0 and self.plan_first:
            self.context.add_assistant(response.text)
            return LoopDecision(
                event="plan",
                detail=response.text,
                phase=AgentPhase.PLAN,
            )

        self.context.add_assistant(response.text)
        return LoopDecision(
            event="answer",
            detail=response.text,
            terminal=True,
            phase=AgentPhase.FINALIZE,
        )

    def _dispatch_batch(self, calls):
        """Dispatch a tool batch, running read-only calls concurrently.

        Batches containing mutating calls run sequentially in order, so
        approvals and write ordering stay deterministic. There is no DAG
        dependency analysis; concurrency is limited to independent calls.
        Returns ``(observations, parallelism_used)``.
        """
        if len(calls) <= 1:
            return [self._dispatch_call(call) for call in calls], 1
        mutating = any(
            self._is_mutating(call.name) for call in calls
        )
        if mutating:
            return (
                [self._dispatch_call(call) for call in calls],
                1,
            )
        workers = min(4, len(calls))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            observations = list(executor.map(self._dispatch_call, calls))
        return observations, workers

    def _dispatch_call(self, call):
        return self.tools.dispatch(
            call_id=call.call_id,
            name=call.name,
            arguments=call.arguments,
        )

    def _is_mutating(self, name: str) -> bool:
        tool = self.tools.tools.get(name)
        return tool is not None and tool.risk is not ToolRisk.READ_ONLY
