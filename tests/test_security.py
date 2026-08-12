import tempfile
import unittest
from pathlib import Path

from miniclaude.tools import ToolDefinition, ToolRegistry
from security.approval import ApprovalManager
from security.command_analysis import assess_argv, assess_command
from security.paths import PathSecurityError, WorkspacePathPolicy
from security.policy import (
    DefaultSecurityPolicy,
    PolicyAction,
    ToolRisk,
)


def make_tool(name, risk, handler):
    return ToolDefinition(
        name=name,
        description=f"{name} test tool.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        handler=lambda value: handler(value),
        risk=risk,
    )


class ToolSecurityTests(unittest.TestCase):
    def test_read_only_tool_is_allowed_without_approval(self):
        calls = []
        registry = ToolRegistry()
        registry.register(make_tool("read", ToolRisk.READ_ONLY, calls.append))

        result = registry.dispatch("call_1", "read", '{"value":"x"}')

        self.assertTrue(result.success)
        self.assertEqual(calls, ["x"])
        self.assertEqual(result.policy_action, PolicyAction.ALLOW.value)

    def test_mutating_tool_is_blocked_without_approval_callback(self):
        calls = []
        registry = ToolRegistry()
        registry.register(make_tool("write", ToolRisk.MUTATING, calls.append))

        result = registry.dispatch("call_1", "write", '{"value":"x"}')

        self.assertFalse(result.success)
        self.assertEqual(calls, [])
        self.assertEqual(result.policy_action, PolicyAction.ASK.value)
        self.assertIn("approval required", result.error)

    def test_approved_mutation_is_cached_for_exact_arguments(self):
        approvals = []

        def approve(request, decision):
            approvals.append((request, decision))
            return True

        calls = []
        registry = ToolRegistry(approvals=ApprovalManager(approve))
        registry.register(make_tool("write", ToolRisk.MUTATING, calls.append))

        first = registry.dispatch("call_1", "write", '{"value":"x"}')
        second = registry.dispatch("call_2", "write", '{"value":"x"}')
        different = registry.dispatch("call_3", "write", '{"value":"y"}')

        self.assertTrue(first.success)
        self.assertTrue(second.success)
        self.assertTrue(different.success)
        self.assertEqual(len(approvals), 2)
        self.assertIn("earlier", second.policy_reason)

    def test_denied_approval_never_executes_handler(self):
        calls = []
        registry = ToolRegistry(approvals=ApprovalManager(lambda *_: False))
        registry.register(make_tool("write", ToolRisk.MUTATING, calls.append))

        result = registry.dispatch("call_1", "write", '{"value":"x"}')

        self.assertFalse(result.success)
        self.assertEqual(calls, [])
        self.assertIn("user denied", result.error)

    def test_destructive_tool_is_denied_without_prompting(self):
        approval_calls = []
        registry = ToolRegistry(
            policy=DefaultSecurityPolicy(),
            approvals=ApprovalManager(lambda *_: approval_calls.append(True) or True),
        )
        registry.register(make_tool("delete", ToolRisk.DESTRUCTIVE, lambda _: None))

        result = registry.dispatch("call_1", "delete", '{"value":"x"}')

        self.assertFalse(result.success)
        self.assertEqual(result.policy_action, PolicyAction.DENY.value)
        self.assertEqual(approval_calls, [])


class WorkspacePathPolicyTests(unittest.TestCase):
    def test_resolves_paths_inside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = WorkspacePathPolicy(directory)
            resolved = policy.resolve("src/example.py")

            self.assertEqual(resolved, Path(directory).resolve() / "src/example.py")

    def test_rejects_parent_and_absolute_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = WorkspacePathPolicy(directory)

            with self.assertRaises(PathSecurityError):
                policy.resolve("../outside.txt")
            with self.assertRaises(PathSecurityError):
                policy.resolve(Path(directory).resolve().parent / "outside.txt")


class CommandAnalysisTests(unittest.TestCase):
    def test_read_only_commands_are_allowed(self):
        self.assertEqual(assess_command("git status").action, PolicyAction.ALLOW)
        self.assertEqual(assess_command("rg TODO").action, PolicyAction.ALLOW)

    def test_mutating_and_unknown_commands_require_approval(self):
        self.assertEqual(assess_command("git commit -m test").action, PolicyAction.ASK)
        self.assertEqual(assess_command("python script.py").action, PolicyAction.ASK)

    def test_destructive_commands_are_denied(self):
        self.assertEqual(assess_command("rm file.txt").action, PolicyAction.DENY)
        self.assertEqual(assess_command("format C:").action, PolicyAction.DENY)

    def test_shell_operators_require_approval(self):
        self.assertEqual(assess_command("git status | more").action, PolicyAction.ASK)
        self.assertEqual(assess_command("dir && echo done").action, PolicyAction.ASK)

    def test_argument_vector_analysis_denies_destructive_commands(self):
        self.assertEqual(assess_argv(["rm", "file"]).action, PolicyAction.DENY)
        self.assertEqual(assess_argv(["git", "status"]).action, PolicyAction.ALLOW)


if __name__ == "__main__":
    unittest.main()
