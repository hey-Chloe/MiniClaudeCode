from miniclaude.controller import AgentController, CompatibilityLoopDriver, LoopDriver
from miniclaude.context import ContextConfig, ContextManager
from miniclaude.llm.base import LLMProvider
from miniclaude.llm.driver import LLMLoopDriver
from miniclaude.models import AgentResult
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
        selected_driver = (
            driver
            if driver is not None
            else LLMLoopDriver(
                provider,
                self.tools,
                ContextManager(context_config),
            )
            if provider is not None
            else CompatibilityLoopDriver()
        )
        self.controller = AgentController(
            driver=selected_driver,
            max_turns=max_turns,
        )

    def run(self, task):
        """Run a task and return the v4.1-compatible event list."""
        return self.run_result(task).events

    def run_result(self, task) -> AgentResult:
        """Run a task and return its structured v5 result."""
        return self.controller.run(task, trace=self.trace)
