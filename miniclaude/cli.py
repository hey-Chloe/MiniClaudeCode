"""Command-line interface for legacy compatibility and real v5 execution."""

import argparse
import json
import os
import shlex
import sys
import uuid
from getpass import getpass
from pathlib import Path

from miniclaude.agent import Agent
from miniclaude.config import AppConfig
from miniclaude.context import ContextConfig
from miniclaude.llm import (
    AnthropicProvider,
    AnthropicProviderConfig,
    OpenAIProvider,
    OpenAIProviderConfig,
)
from miniclaude.metrics import CostCalculator, Pricing
from miniclaude.mcp import MCPClient, MCPServerConfig
from miniclaude.runtime_tools import create_runtime_tools
from miniclaude.session import SessionStore
from miniclaude.tools import ToolDefinition
from runtime import DockerRuntime, LocalProcessRuntime
from security.policy import ToolRisk
from security.policy import PermissionModePolicy


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        config = AppConfig.from_env(
            model=args.model,
            workspace=args.workspace,
            max_turns=args.max_turns,
            timeout=args.timeout,
            permission_mode=args.permission_mode,
            runtime=args.runtime,
        )
        use_legacy = args.legacy or (args.mode == "auto" and not config.model)
        if use_legacy:
            return _run_legacy(args.task)
        if not config.model:
            parser.error("v5 mode requires --model or MINICLAUDE_MODEL")
        return _run_v5(
            args.task,
            config,
            args.json,
            args.session_id,
            args.sessions_dir,
            verify=args.verify,
            review=args.review,
            tool_gating=not args.no_tool_gating,
            resume=args.resume,
            provider=args.provider,
            mcp_demo=args.mcp_demo,
            mcp_servers=args.mcp,
            multi_agent=args.multi_agent,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="miniclaude")
    parser.add_argument("task")
    parser.add_argument("--mode", choices=["auto", "v5"], default="auto")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--model")
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "anthropic"],
        default="auto",
    )
    parser.add_argument("--workspace", type=lambda value: __import__("pathlib").Path(value).resolve())
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--runtime", choices=["local", "docker"])
    parser.add_argument(
        "--permission-mode",
        choices=["default", "plan", "accept-edits", "bypass"],
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--session-id")
    parser.add_argument("--sessions-dir")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted run from its saved checkpoint",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="run pytest before finalizing when files were edited",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help=(
            "run a reviewer LLM pass over the diff before finalizing "
            "(second-pass verification; implies the file-editing gate)"
        ),
    )
    parser.add_argument(
        "--no-tool-gating",
        action="store_true",
        help="send every tool schema on every turn instead of activating a subset",
    )
    parser.add_argument(
        "--mcp-demo",
        action="store_true",
        help="attach the bundled demo MCP server (stdio)",
    )
    parser.add_argument(
        "--mcp",
        action="append",
        default=None,
        metavar="NAME=COMMAND[ ARGS...]",
        help="attach an external MCP stdio server; repeatable",
    )
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help=(
            "run the task through the coordinator/specialist multi-agent "
            "pipeline (shared blackboard, concurrent specialists, optional critic)"
        ),
    )
    return parser


def _run_legacy(task: str) -> int:
    for item in Agent().run(task):
        print(item)
    return 0


def _run_v5(
    task: str,
    config: AppConfig,
    json_output: bool,
    session_id=None,
    sessions_dir=None,
    *,
    verify: bool = False,
    review: bool = False,
    tool_gating: bool = True,
    resume: bool = False,
    provider: str = "auto",
    mcp_demo: bool = False,
    mcp_servers: list[str] | None = None,
    multi_agent: bool = False,
) -> int:
    def provider_factory():
        if provider == "anthropic":
            return AnthropicProvider(
                AnthropicProviderConfig(
                    model=config.model or "",
                    api_key=os.getenv("ANTHROPIC_API_KEY") or config.api_key,
                    base_url=config.base_url,
                    timeout=config.timeout,
                    max_retries=config.max_retries,
                )
            )
        return OpenAIProvider(
            OpenAIProviderConfig(
                model=config.model or "",
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
        )

    llm_provider = provider_factory()
    runtime_class = DockerRuntime if config.runtime == "docker" else LocalProcessRuntime
    runtime = runtime_class(
        config.workspace,
        default_timeout=config.timeout,
        max_output_chars=config.max_output_chars,
    )
    cost_calculator = None
    if (
        config.input_price_per_million is not None
        and config.output_price_per_million is not None
    ):
        cost_calculator = CostCalculator(
            {
                (config.model or ""): Pricing(
                    input_per_million=config.input_price_per_million,
                    output_per_million=config.output_price_per_million,
                )
            }
        )
    skills_dir = Path(__file__).resolve().parents[1] / "skills"
    verifier = None
    if verify:
        def verifier():
            result = runtime.execute(
                ["pytest", "-q"],
                cwd=".",
                timeout=min(config.timeout, 120.0),
            )
            return {
                "passed": bool(result.succeeded),
                "output": (
                    (result.stdout or "")[-4000:]
                    + "\n"
                    + (result.stderr or "")[-2000:]
                ),
            }
    if review:
        from miniclaude.reviewer import build_review_verifier

        reviewer_provider = OpenAIProvider(
            OpenAIProviderConfig(
                model=config.model or "",
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
        )
        verifier = build_review_verifier(reviewer_provider, runtime)
    mcp_clients: list[MCPClient] = []
    try:
        mcp_clients, mcp_tools = _attach_mcp(
            mcp_demo=mcp_demo,
            mcp_servers=mcp_servers,
        )
        if multi_agent:
            from miniclaude.agents import (
                CollaborationBlackboard,
                CoordinatorAgent,
            )

            coordinator = CoordinatorAgent(
                provider_factory=lambda name: provider_factory(),
                workspace=config.workspace,
                tools=create_runtime_tools(runtime) + mcp_tools,
                blackboard=CollaborationBlackboard(),
                concurrency=2,
            )
            multi_result = coordinator.run(task)
            print(
                json.dumps(
                    multi_result.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            return 0 if multi_result.status == "completed" else 1
        agent = Agent(
            provider=llm_provider,
            tools=create_runtime_tools(runtime) + mcp_tools,
            max_turns=config.max_turns,
            security_policy=PermissionModePolicy(config.permission_mode),
            approval_callback=_approval_callback if config.permission_mode == "default" else None,
            context_config=ContextConfig(
                workspace=config.workspace,
                skills_dir=skills_dir if skills_dir.is_dir() else None,
            ),
            cost_calculator=cost_calculator,
            verifier=verifier,
            tool_gating=tool_gating,
        )
        store = None
        if session_id or sessions_dir or resume:
            store = SessionStore(sessions_dir or config.workspace / "sessions")
        if resume:
            if not session_id:
                raise ValueError("--resume requires --session-id")
            checkpoint = store.load_checkpoint(session_id)
            result = agent.resume(task, checkpoint)
        else:
            result = agent.run_result(task)
        if store is not None:
            resolved_id = session_id or str(uuid.uuid4())
            store.save(resolved_id, result, agent.trace.export_detailed())
            if result.status.value != "completed":
                checkpoint = agent.build_checkpoint(result)
                store.save_checkpoint(resolved_id, checkpoint)
    finally:
        for client in mcp_clients:
            client.stop()
    if json_output:
        print(json.dumps({
            "status": result.status.value,
            "turns": result.turns,
            "output": result.output,
            "error": result.error,
            "events": result.events,
            "metrics": result.metrics.to_dict() if result.metrics is not None else None,
            "skills": list(result.skills),
            "phases": list(result.phases),
        }, ensure_ascii=False, indent=2, default=str))
    else:
        for event in result.events:
            print(event)
        if result.output is not None:
            print(result.output)
        if result.metrics is not None:
            _print_metrics_table(result.metrics)
    return 0 if result.status.value == "completed" else 1


def _attach_mcp(
    *,
    mcp_demo: bool,
    mcp_servers: list[str] | None,
) -> tuple[list[MCPClient], list[ToolDefinition]]:
    """Start configured MCP servers and collect their tools.

    Returns ``(clients, tools)``; callers must stop the clients when done.
    """
    configs: list[MCPServerConfig] = []
    if mcp_demo:
        configs.append(
            MCPServerConfig(
                name="demo",
                command=sys.executable,
                args=("-m", "miniclaude.mcp.demo_server"),
                risk=ToolRisk.MUTATING,
                activation_keywords=("demo", "note", "echo"),
            )
        )
    for spec in mcp_servers or []:
        name, separator, command = spec.partition("=")
        if not separator or not name.strip() or not command.strip():
            raise ValueError(
                f"--mcp expects NAME=COMMAND [ARGS...], got: {spec!r}"
            )
        parts = shlex.split(command, posix=False)
        if not parts:
            raise ValueError(f"--mcp command is empty for server {name!r}")
        configs.append(
            MCPServerConfig(
                name=name.strip(),
                command=parts[0],
                args=tuple(parts[1:]),
                risk=ToolRisk.MUTATING,
            )
        )
    clients = [MCPClient(config) for config in configs]
    tools: list[ToolDefinition] = []
    try:
        for client in clients:
            client.start()
            tools.extend(client.list_tools())
    except Exception:
        for client in clients:
            client.stop()
        raise
    return clients, tools


def _print_metrics_table(metrics) -> None:
    """Render run metrics with Rich when available; fall back to JSON."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print(
            "metrics: "
            + json.dumps(
                metrics.to_dict(), ensure_ascii=False, default=str
            )
        )
        return
    data = metrics.to_dict()
    table = Table(title="Run metrics")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="green")
    for name in (
        "turns",
        "tool_calls",
        "tool_success_rate",
        "recovery_rate",
        "repeated_read_rate",
        "cache_hit_rate",
        "average_tools_per_turn",
        "parallel_batches",
        "max_parallelism",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "context_truncated",
        "duration_seconds",
        "model_name",
        "cost_usd",
    ):
        table.add_row(name, str(data.get(name)))
    Console().print(table)


def _approval_callback(request, decision) -> bool:
    if not sys.stdin.isatty():
        return False
    rendered = json.dumps(dict(request.arguments), ensure_ascii=False, indent=2, default=str)
    print(f"\nApproval required: {request.tool_name} ({request.risk.value})")
    print(f"Reason: {decision.reason}\nArguments:\n{rendered}")
    answer = getpass("Allow this exact call for this session? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
