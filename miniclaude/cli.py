"""Command-line interface for legacy compatibility and real v5 execution."""

import argparse
import json
import sys
import uuid
from getpass import getpass

from miniclaude.agent import Agent
from miniclaude.config import AppConfig
from miniclaude.context import ContextConfig
from miniclaude.llm import OpenAIProvider, OpenAIProviderConfig
from miniclaude.runtime_tools import create_runtime_tools
from miniclaude.session import SessionStore
from runtime import DockerRuntime, LocalProcessRuntime
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
        return _run_v5(args.task, config, args.json, args.session_id, args.sessions_dir)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="miniclaude")
    parser.add_argument("task")
    parser.add_argument("--mode", choices=["auto", "v5"], default="auto")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--model")
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
    return parser


def _run_legacy(task: str) -> int:
    for item in Agent().run(task):
        print(item)
    return 0


def _run_v5(task: str, config: AppConfig, json_output: bool, session_id=None, sessions_dir=None) -> int:
    provider = OpenAIProvider(
        OpenAIProviderConfig(
            model=config.model or "",
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
    )
    runtime_class = DockerRuntime if config.runtime == "docker" else LocalProcessRuntime
    runtime = runtime_class(
        config.workspace,
        default_timeout=config.timeout,
        max_output_chars=config.max_output_chars,
    )
    agent = Agent(
        provider=provider,
        tools=create_runtime_tools(runtime),
        max_turns=config.max_turns,
        security_policy=PermissionModePolicy(config.permission_mode),
        approval_callback=_approval_callback if config.permission_mode == "default" else None,
        context_config=ContextConfig(workspace=config.workspace),
    )
    result = agent.run_result(task)
    if session_id or sessions_dir:
        store = SessionStore(sessions_dir or config.workspace / "sessions")
        store.save(session_id or str(uuid.uuid4()), result, agent.trace.export_detailed())
    if json_output:
        print(json.dumps({
            "status": result.status.value,
            "turns": result.turns,
            "output": result.output,
            "error": result.error,
            "events": result.events,
        }, ensure_ascii=False, indent=2, default=str))
    else:
        for event in result.events:
            print(event)
        if result.output is not None:
            print(result.output)
    return 0 if result.status.value == "completed" else 1


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
