from dataclasses import replace
from typing import Any, Callable

from miniclaude.controller import AgentController, CompatibilityLoopDriver, LoopDriver
from miniclaude.context import ContextConfig, ContextManager
from miniclaude.llm.base import LLMProvider
from miniclaude.llm.driver import LLMLoopDriver
from miniclaude.metrics import CostCalculator
from miniclaude.models import AgentResult, AgentState, RunStatus
from miniclaude.session import SessionCheckpoint
from miniclaude.trace import Trace
from miniclaude.tools import ToolDefinition, ToolRegistry
from security.approval import ApprovalCallback, ApprovalManager
from security.policy import SecurityPolicy

class Agent:
    def __init__(
        self,
        driver: LoopDriver | None = None,
        max_turns: int = 20,
        provider: LLMProvider | None = None,
        tools: list[ToolDefinition] | None = None,
        security_policy: SecurityPolicy | None = None,
        approval_callback: ApprovalCallback | None = None,
        context_config: ContextConfig | None = None,
        cost_calculator: CostCalculator | None = None,
        verifier: Callable[[], dict[str, Any]] | None = None,
        plan_first: bool = True,
        tool_gating: bool = True,
    ):
        if driver is not None and provider is not None:
            raise ValueError("provide either driver or provider, not both")
        self.trace=Trace()
        self.tools=ToolRegistry(
            policy=security_policy,
            approvals=ApprovalManager(approval_callback),
        )
        for tool in tools or []:
            self.tools.register(tool)
        self.context = ContextManager(context_config)
        selected_driver = (
            driver
            if driver is not None
            else LLMLoopDriver(
                provider,
                self.tools,
                self.context,
                verifier=verifier,
                plan_first=plan_first,
                tool_gating=tool_gating,
            )
            if provider is not None
            else CompatibilityLoopDriver()
        )
        self.controller = AgentController(
            driver=selected_driver,
            max_turns=max_turns,
        )
        self.cost_calculator = cost_calculator

    def run(self, task):
        """Run a task and return the v4.1-compatible event list."""
        return self.run_result(task).events

    def run_result(self, task) -> AgentResult:
        """Run a task and return its structured v5 result."""
        result = self.controller.run(task, trace=self.trace)
        if (
            self.cost_calculator is not None
            and result.metrics is not None
            and result.metrics.model_name
        ):
            cost = self.cost_calculator.estimate(
                result.metrics.model_name,
                result.metrics.input_tokens,
                result.metrics.output_tokens,
            )
            if cost is not None:
                result = replace(
                    result,
                    metrics=replace(result.metrics, cost_usd=cost),
                )
        return result

    def build_checkpoint(self, result: AgentResult) -> SessionCheckpoint:
        """Capture resumable state from a run that did not finish."""
        metrics = result.metrics
        provider = getattr(self.controller.driver, "provider", None)
        export_state = getattr(provider, "export_state", None)
        provider_state = (
            export_state() if export_state is not None else None
        )
        self.trace.add(
            "checkpoint_built",
            {
                "turn_count": result.turns,
                "status": result.status.value,
                "provider_response_id": result.provider_response_id,
            },
        )
        return SessionCheckpoint(
            task=result.task,
            max_turns=self.controller.max_turns,
            turn_count=result.turns,
            status=result.status.value,
            output=result.output,
            error=result.error,
            provider_response_id=result.provider_response_id,
            usage_input_tokens=metrics.input_tokens if metrics is not None else 0,
            usage_output_tokens=metrics.output_tokens if metrics is not None else 0,
            model_name=metrics.model_name if metrics is not None else None,
            context_truncated=(
                metrics.context_truncated if metrics is not None else False
            ),
            skills_loaded=result.skills,
            messages=self.context.export_messages(),
            provider_state=provider_state,
        )

    def resume(
        self,
        task: str,
        checkpoint: SessionCheckpoint,
    ) -> AgentResult:
        """Continue a checkpointed run from where it stopped."""
        if checkpoint.task != task:
            raise ValueError("checkpoint task does not match the resumed task")
        self.context.restore(task, checkpoint.messages)
        provider = getattr(self.controller.driver, "provider", None)
        restore = getattr(provider, "restore", None)
        restore_state = getattr(provider, "restore_state", None)
        if checkpoint.provider_state is not None and restore_state is not None:
            restore_state(checkpoint.provider_state)
        elif restore is not None:
            restore(checkpoint.messages)
        state = AgentState(
            task=task,
            max_turns=checkpoint.max_turns,
            status=RunStatus.RUNNING,
            turn_count=checkpoint.turn_count,
            output=checkpoint.output,
            error=checkpoint.error,
            provider_response_id=checkpoint.provider_response_id,
            usage_input_tokens=checkpoint.usage_input_tokens,
            usage_output_tokens=checkpoint.usage_output_tokens,
            model_name=checkpoint.model_name,
            context_truncated=checkpoint.context_truncated,
            skills_loaded=checkpoint.skills_loaded,
        )
        return self.controller.run(
            task,
            trace=self.trace,
            initial_state=state,
        )
