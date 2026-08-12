"""Adapter from provider responses to Phase 1 loop decisions."""

from miniclaude.context import ContextManager
from miniclaude.llm.base import LLMProvider, LLMRequest
from miniclaude.models import AgentState, LoopDecision
from miniclaude.tools import ToolRegistry


class LLMLoopDriver:
    """Converts normalized provider output into agent-loop decisions."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry | None = None,
        context: ContextManager | None = None,
    ):
        self.provider = provider
        self.tools = tools if tools is not None else ToolRegistry()
        self.context = context if context is not None else ContextManager()

    def next(self, state: AgentState) -> LoopDecision:
        if state.turn_count == 0:
            self.context.start(state.task)
        snapshot = self.context.snapshot()
        response = self.provider.complete(
            LLMRequest(
                task=state.task,
                turn=state.turn_count,
                tools=tuple(self.tools.schemas()),
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

        if response.tool_calls:
            self.context.add_assistant(
                "Tool calls: "
                + ", ".join(call.name for call in response.tool_calls)
            )
            observations = [
                self.tools.dispatch(
                    call_id=call.call_id,
                    name=call.name,
                    arguments=call.arguments,
                )
                for call in response.tool_calls
            ]
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
            )

        self.context.add_assistant(response.text)
        return LoopDecision(event="answer", detail=response.text, terminal=True)
