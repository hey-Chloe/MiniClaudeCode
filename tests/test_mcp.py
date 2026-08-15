import json
import subprocess
import sys
import tempfile
import unittest

from miniclaude.agent import Agent
from miniclaude.llm import LLMResponse, LLMToolCall
from miniclaude.mcp import MCPClient, MCPError, MCPServerConfig
from miniclaude.models import RunStatus
from miniclaude.tools import ToolRegistry
from security.approval import ApprovalManager
from security.policy import ToolRisk


FAKE_SERVER = r"""
import json, sys
for line in sys.stdin:
    message = json.loads(line)
    method = message["method"]
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "fake", "version": "1.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "fake_echo",
                    "description": "Echo supplied text.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                }
            ]
        }
    elif method == "tools/call":
        arguments = message["params"]["arguments"]
        result = {
            "content": [
                {"type": "text", "text": "echo:" + arguments.get("text", "")}
            ]
        }
    else:
        result = {}
    sys.stdout.write(
        json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result})
        + "\n"
    )
    sys.stdout.flush()
"""


class MCPClientTests(unittest.TestCase):
    def _client(self, risk=ToolRisk.MUTATING):
        return MCPClient(
            MCPServerConfig(
                name="fake",
                command=sys.executable,
                args=("-c", FAKE_SERVER),
                risk=risk,
            )
        )

    def test_lists_and_calls_mcp_tool(self):
        with self._client() as client:
            tools = client.list_tools()

            self.assertEqual([tool.name for tool in tools], ["fake_echo"])
            registry = ToolRegistry(
                approvals=ApprovalManager(lambda *_: True)
            )
            for tool in tools:
                registry.register(tool)

            result = registry.dispatch(
                "call_1", "fake_echo", '{"text": "hi"}'
            )

        self.assertTrue(result.success)
        self.assertEqual(result.output, "echo:hi")

    def test_mcp_tools_default_to_mutating_risk(self):
        self.assertEqual(
            self._client().config.risk, ToolRisk.MUTATING
        )
        tools = self._client().list_tools()
        self.assertEqual(tools[0].risk, ToolRisk.MUTATING)

    def test_denied_approval_blocks_mcp_call(self):
        with self._client() as client:
            tools = client.list_tools()
            registry = ToolRegistry(
                approvals=ApprovalManager(lambda *_: False)
            )
            for tool in tools:
                registry.register(tool)

            result = registry.dispatch(
                "call_1", "fake_echo", '{"text": "hi"}'
            )

        self.assertFalse(result.success)
        self.assertEqual(result.policy_action, "ask")

    def test_server_error_is_raised(self):
        server = (
            "import json,sys\n"
            "for line in sys.stdin:\n"
            "    m=json.loads(line)\n"
            "    sys.stdout.write(json.dumps({'jsonrpc':'2.0','id':m['id'],"
            "'error':{'code':-32601,'message':'method not found'}})+'\\n')\n"
            "    sys.stdout.flush()\n"
        )
        client = MCPClient(
            MCPServerConfig(
                name="error",
                command=sys.executable,
                args=("-c", server),
            )
        )
        with self.assertRaisesRegex(MCPError, "method not found"):
            client.list_tools()
        client.stop()

    def test_missing_command_is_reported(self):
        client = MCPClient(
            MCPServerConfig(
                name="missing",
                command="definitely-not-a-real-command-xyz",
            )
        )
        with self.assertRaisesRegex(MCPError, "could not start"):
            client.start()


class BundledDemoServerTests(unittest.TestCase):
    """Integration tests for the bundled ``--mcp-demo`` server."""

    def _client(self, root):
        return MCPClient(
            MCPServerConfig(
                name="demo",
                command=sys.executable,
                args=("-m", "miniclaude.mcp.demo_server"),
                risk=ToolRisk.MUTATING,
            )
        )

    def test_read_only_hint_is_mapped_to_read_only_risk(self):
        with tempfile.TemporaryDirectory() as directory:
            import os

            os.environ["MINICLAUDE_DEMO_ROOT"] = directory
            try:
                with self._client(directory) as client:
                    tools = {tool.name: tool for tool in client.list_tools()}
            finally:
                os.environ.pop("MINICLAUDE_DEMO_ROOT", None)

        self.assertEqual(
            {name: tool.risk.value for name, tool in tools.items()},
            {
                "demo_echo": "read_only",
                "demo_read_file": "read_only",
                "demo_append_note": "mutating",
            },
        )

    def test_read_only_demo_tool_runs_through_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            import os

            os.environ["MINICLAUDE_DEMO_ROOT"] = directory
            try:
                with self._client(directory) as client:
                    tools = client.list_tools()
                    registry = ToolRegistry(
                        approvals=ApprovalManager(lambda *_: True)
                    )
                    for tool in tools:
                        registry.register(tool)
                    result = registry.dispatch(
                        "call_1", "demo_echo", '{"text": "hi"}'
                    )
            finally:
                os.environ.pop("MINICLAUDE_DEMO_ROOT", None)

        self.assertTrue(result.success)
        self.assertEqual(result.output, "echo:hi")

    def test_mutating_demo_tool_is_blocked_without_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            import os

            os.environ["MINICLAUDE_DEMO_ROOT"] = directory
            try:
                with self._client(directory) as client:
                    tools = client.list_tools()
                    registry = ToolRegistry(
                        approvals=ApprovalManager(lambda *_: False)
                    )
                    for tool in tools:
                        registry.register(tool)
                    result = registry.dispatch(
                        "call_1", "demo_append_note", '{"text": "note"}'
                    )
            finally:
                os.environ.pop("MINICLAUDE_DEMO_ROOT", None)

        self.assertFalse(result.success)
        self.assertEqual(result.policy_action, "ask")

    def test_agent_loop_calls_demo_tool(self):
        class ScriptedProvider:
            def __init__(self, responses):
                self.responses = list(responses)
                self.requests = []

            def complete(self, request):
                self.requests.append(request)
                return self.responses.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            import os

            os.environ["MINICLAUDE_DEMO_ROOT"] = directory
            try:
                with self._client(directory) as client:
                    tools = client.list_tools()
                    provider = ScriptedProvider(
                        [
                            LLMResponse(
                                response_id="resp_1",
                                tool_calls=(
                                    LLMToolCall(
                                        "call_1",
                                        "demo_echo",
                                        '{"text": "hello"}',
                                    ),
                                ),
                            ),
                            LLMResponse(response_id="resp_2", text="done"),
                        ]
                    )
                    result = Agent(
                        provider=provider,
                        tools=tools,
                        tool_gating=False,
                    ).run_result("echo hello")
            finally:
                os.environ.pop("MINICLAUDE_DEMO_ROOT", None)

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.output, "done")
        observations = [
            observation
            for event in result.events
            if event.get("event") == "tool_results"
            for observation in event.get("detail", [])
        ]
        self.assertEqual(observations[0]["name"], "demo_echo")
        self.assertTrue(observations[0]["success"])
        self.assertEqual(observations[0]["output"], "echo:hello")


if __name__ == "__main__":
    unittest.main()
