import json
import subprocess
import sys
import unittest

from miniclaude.mcp import MCPClient, MCPError, MCPServerConfig
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


if __name__ == "__main__":
    unittest.main()
