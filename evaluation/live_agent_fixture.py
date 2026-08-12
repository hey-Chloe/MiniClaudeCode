"""Opt-in live acceptance harness for a disposable calculator fixture."""

import json
import sys
from pathlib import Path

from miniclaude.agent import Agent
from miniclaude.config import AppConfig
from miniclaude.context import ContextConfig
from miniclaude.llm import OpenAIProvider, OpenAIProviderConfig
from miniclaude.runtime_tools import create_runtime_tools
from runtime import LocalProcessRuntime
from security.policy import PermissionModePolicy


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    workspace = project / "tmp_demo"
    python = project / ".venv" / "Scripts" / "python.exe"
    config = AppConfig.from_env(workspace=workspace, permission_mode="default")
    approvals: list[dict[str, object]] = []

    def approve(request, decision) -> bool:
        entry = {
            "tool": request.tool_name,
            "action": decision.action.value,
            "reason": decision.reason,
            "arguments": dict(request.arguments),
            "approved": True,
        }
        approvals.append(entry)
        print("APPROVAL " + json.dumps(entry, ensure_ascii=False), flush=True)
        return True

    provider = OpenAIProvider(
        OpenAIProviderConfig(
            model=config.model or "",
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
    )
    runtime = LocalProcessRuntime(workspace, default_timeout=config.timeout)
    agent = Agent(
        provider=provider,
        tools=create_runtime_tools(runtime),
        max_turns=config.max_turns,
        security_policy=PermissionModePolicy("default"),
        approval_callback=approve,
        context_config=ContextConfig(workspace=workspace),
    )
    task = (
        "修复 calculator.py，使测试通过。必须按顺序完成：先读取 calculator.py 和 "
        "test_calculator.py；再修复 calculator.py；修改后调用 git_diff；最后使用 "
        f"execute_command 运行 {python} -m pytest -v -p no:cacheprovider。"
        "必须依据真实 pytest 输出给出最终修改摘要。"
    )
    result = agent.run_result(task)
    events = list(result.events)
    print(json.dumps({
        "status": result.status.value,
        "turns": result.turns,
        "output": result.output,
        "error": result.error,
        "approvals": approvals,
        "events": events,
    }, ensure_ascii=False, indent=2, default=str))

    observations = [
        observation
        for event in events
        if event.get("event") == "tool_results"
        for observation in event.get("detail", [])
    ]
    names = [item.get("name") for item in observations if item.get("success")]
    checks = {
        "read_file": names.count("read_file") >= 2,
        "write": "write_file" in names or "replace_text" in names,
        "git_diff": any(
            item.get("name") == "git_diff"
            and item.get("success")
            and item.get("output", {}).get("exit_code") == 0
            and "-    return left - right" in item.get("output", {}).get("stdout", "")
            and "+    return left + right" in item.get("output", {}).get("stdout", "")
            for item in observations
        ),
        "pytest": any(
            item.get("name") == "execute_command"
            and "pytest" in json.dumps(item.get("output"), ensure_ascii=False)
            and item.get("output", {}).get("exit_code") == 0
            for item in observations
        ),
        "approval": any(item.get("action") == "ask" for item in approvals),
    }
    print("ACCEPTANCE " + json.dumps(checks, ensure_ascii=False))
    return 0 if result.status.value == "completed" and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
