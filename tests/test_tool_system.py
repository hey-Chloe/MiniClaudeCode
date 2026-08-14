import json
import unittest

from miniclaude.agent import Agent
from miniclaude.llm import LLMResponse, LLMToolCall
from miniclaude.models import RunStatus
from miniclaude.tools import (
    ToolDefinition,
    ToolRegistrationError,
    ToolRegistry,
)


def echo_tool():
    return ToolDefinition(
        name="echo",
        description="Return supplied text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        handler=lambda text: {"text": text},
    )


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class ToolRegistryTests(unittest.TestCase):
    def test_register_list_and_export_schema(self):
        registry = ToolRegistry()
        registry.register(echo_tool())

        self.assertEqual(registry.list_tools(), ["echo"])
        schema = registry.schemas()[0]
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["name"], "echo")
        self.assertTrue(schema["strict"])

    def test_duplicate_registration_is_rejected(self):
        registry = ToolRegistry()
        registry.register(echo_tool())

        with self.assertRaisesRegex(ToolRegistrationError, "already registered"):
            registry.register(echo_tool())

    def test_valid_call_returns_observation(self):
        registry = ToolRegistry()
        registry.register(echo_tool())

        result = registry.dispatch("call_1", "echo", '{"text":"hello"}')

        self.assertTrue(result.success)
        self.assertEqual(result.output, {"text": "hello"})
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.duration_seconds, 0)

    def test_unknown_tool_returns_failed_observation(self):
        result = ToolRegistry().dispatch("call_1", "missing", "{}")

        self.assertFalse(result.success)
        self.assertIn("unknown tool", result.error)

    def test_invalid_json_and_schema_do_not_execute_handler(self):
        calls = []
        tool = ToolDefinition(
            name="record",
            description="Record a value.",
            parameters={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
                "additionalProperties": False,
            },
            handler=lambda count: calls.append(count),
        )
        registry = ToolRegistry()
        registry.register(tool)

        malformed = registry.dispatch("call_1", "record", "not-json")
        wrong_type = registry.dispatch("call_2", "record", '{"count":"one"}')
        extra = registry.dispatch("call_3", "record", '{"count":1,"extra":2}')

        self.assertFalse(malformed.success)
        self.assertFalse(wrong_type.success)
        self.assertFalse(extra.success)
        self.assertEqual(calls, [])

    def test_handler_failure_is_captured(self):
        def fail():
            raise RuntimeError("boom")

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="fail",
                description="Always fail.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=fail,
            )
        )

        result = registry.dispatch("call_1", "fail", "{}")

        self.assertFalse(result.success)
        self.assertIn("boom", result.error)


class ToolActivationTests(unittest.TestCase):
    def _registry(self):
        def handler():
            return {}

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="read_file",
                description="Read a file.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=handler,
                activation_keywords=("read", "inspect"),
            )
        )
        registry.register(
            ToolDefinition(
                name="execute_command",
                description="Run a command.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=handler,
                activation_keywords=("run", "pytest"),
            )
        )
        registry.register(
            ToolDefinition(
                name="plain",
                description="No keywords.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=handler,
            )
        )
        return registry

    def test_activate_for_task_selects_matching_tools(self):
        registry = self._registry()

        activated = registry.activate_for_task("read the file and run pytest")

        self.assertEqual(activated, ["execute_command", "read_file"])
        self.assertEqual(
            [schema["name"] for schema in registry.schemas()],
            ["read_file", "execute_command"],
        )

    def test_activate_for_task_falls_back_to_all(self):
        registry = self._registry()

        activated = registry.activate_for_task("unrelated task")

        self.assertEqual(activated, [])
        self.assertEqual(len(registry.schemas()), 3)

    def test_activate_unions_and_expands_active_set(self):
        registry = self._registry()
        registry.activate_for_task("read the file")

        registry.activate(["plain"])

        self.assertEqual(
            [schema["name"] for schema in registry.schemas()],
            ["read_file", "plain"],
        )

    def test_activate_all_restores_full_set(self):
        registry = self._registry()
        registry.activate_for_task("read the file")

        registry.activate_all()

        self.assertEqual(len(registry.schemas()), 3)
        self.assertEqual(registry.active_tools(), ["read_file", "execute_command", "plain"])

    def test_unknown_activation_names_are_ignored(self):
        registry = self._registry()
        registry.activate(["missing"])
        self.assertEqual(registry.active_tools(), [])


class ToolLoopIntegrationTests(unittest.TestCase):
    def test_tool_result_is_returned_to_provider(self):
        provider = ScriptedProvider(
            [
                LLMResponse(
                    response_id="resp_1",
                    tool_calls=(
                        LLMToolCall("call_1", "echo", '{"text":"hello"}'),
                    ),
                ),
                LLMResponse(response_id="resp_2", text="done"),
            ]
        )

        result = Agent(provider=provider, tools=[echo_tool()]).run_result("echo hello")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.turns, 2)
        self.assertEqual(result.events[1]["event"], "tool_results")
        self.assertTrue(result.events[1]["detail"][0]["success"])
        second_request = provider.requests[1]
        self.assertEqual(second_request.previous_response_id, "resp_1")
        model_result = json.loads(second_request.tool_outputs[0]["output"])
        self.assertEqual(model_result["output"], {"text": "hello"})

    def test_tool_schema_is_sent_to_provider(self):
        provider = ScriptedProvider([LLMResponse(text="done")])

        Agent(provider=provider, tools=[echo_tool()]).run_result("inspect tools")

        self.assertEqual(provider.requests[0].tools[0]["name"], "echo")

    def test_tool_gating_limits_schemas_to_matching_tools(self):
        registry_tools = [
            ToolDefinition(
                name="read_file",
                description="Read a file.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=lambda: {},
                activation_keywords=("read",),
            ),
            ToolDefinition(
                name="execute_command",
                description="Run a command.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=lambda: {},
                activation_keywords=("run",),
            ),
        ]
        provider = ScriptedProvider(
            [
                LLMResponse(text="I will read the file."),
                LLMResponse(text="done"),
            ]
        )

        Agent(provider=provider, tools=registry_tools).run_result(
            "read the file"
        )

        self.assertEqual(
            [tool["name"] for tool in provider.requests[0].tools],
            ["read_file"],
        )

    def test_tool_gating_can_be_disabled(self):
        provider = ScriptedProvider(
            [
                LLMResponse(text="I will read the file."),
                LLMResponse(text="done"),
            ]
        )
        tools = [
            ToolDefinition(
                name="read_file",
                description="Read a file.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=lambda: {},
                activation_keywords=("read",),
            ),
            ToolDefinition(
                name="execute_command",
                description="Run a command.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=lambda: {},
                activation_keywords=("run",),
            ),
        ]

        Agent(
            provider=provider,
            tools=tools,
            tool_gating=False,
        ).run_result("read the file")

        self.assertEqual(
            [tool["name"] for tool in provider.requests[0].tools],
            ["read_file", "execute_command"],
        )


if __name__ == "__main__":
    unittest.main()
