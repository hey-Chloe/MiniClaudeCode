import os
import sys
import tempfile
import unittest
from pathlib import Path

from miniclaude.runtime_tools import create_runtime_tools
from miniclaude.tools import ToolRegistry
from runtime import LocalProcessRuntime, RuntimeErrorBase, SandboxRuntime
from security.approval import ApprovalManager
from security.paths import PathSecurityError


class LocalProcessRuntimeTests(unittest.TestCase):
    def test_runtime_reports_that_local_process_is_not_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalProcessRuntime(directory)

            self.assertEqual(runtime.info.name, "local-process")
            self.assertFalse(runtime.info.isolated)

    def test_executes_argument_vector_without_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalProcessRuntime(directory)
            result = runtime.execute(
                [sys.executable, "-c", "print('hello')"], timeout=10
            )

            self.assertTrue(result.succeeded)
            self.assertEqual(result.stdout.strip(), "hello")
            self.assertEqual(result.stderr, "")
            self.assertFalse(result.isolated)

    def test_shell_operators_are_plain_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalProcessRuntime(directory)
            result = runtime.execute(
                [sys.executable, "-c", "import sys; print(sys.argv[1])", "&& touch bad"],
                timeout=10,
            )

            self.assertEqual(result.stdout.strip(), "&& touch bad")
            self.assertFalse((Path(directory) / "bad").exists())

    def test_timeout_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalProcessRuntime(directory)
            result = runtime.execute(
                [sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.05
            )

            self.assertTrue(result.timed_out)
            self.assertIsNone(result.exit_code)
            self.assertFalse(result.succeeded)

    def test_output_is_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalProcessRuntime(directory, max_output_chars=5)
            result = runtime.execute(
                [sys.executable, "-c", "print('abcdefghij', end='')"], timeout=10
            )

            self.assertEqual(result.stdout, "abcde")
            self.assertTrue(result.output_truncated)

    def test_cwd_must_stay_inside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalProcessRuntime(directory)

            with self.assertRaises(PathSecurityError):
                runtime.execute([sys.executable, "--version"], cwd="..")

    def test_environment_is_filtered_and_additions_are_allowlisted(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalProcessRuntime(directory, allowed_env_keys={"SAFE_VALUE", "PATH"})
            result = runtime.execute(
                [sys.executable, "-c", "import os; print(os.getenv('SAFE_VALUE', ''))"],
                env={"SAFE_VALUE": "visible"},
                timeout=10,
            )

            self.assertEqual(result.stdout.strip(), "visible")
            with self.assertRaisesRegex(RuntimeErrorBase, "not allowed"):
                runtime.execute(
                    [sys.executable, "--version"], env={"SECRET_VALUE": "hidden"}
                )

    def test_pythonpath_can_be_forwarded_for_approved_test_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalProcessRuntime(directory)
            result = runtime.execute(
                [sys.executable, "-c", "import os; print(os.environ['PYTHONPATH'])"],
                env={"PYTHONPATH": "approved-test-path"},
                timeout=10,
            )

            self.assertEqual(result.stdout.strip(), "approved-test-path")

    def test_reads_and_atomically_writes_workspace_files(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalProcessRuntime(directory)

            byte_count = runtime.write_text("example.txt", "hello 世界")

            self.assertEqual(runtime.read_text("example.txt"), "hello 世界")
            self.assertEqual(byte_count, len("hello 世界".encode("utf-8")))
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_file_operations_reject_workspace_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalProcessRuntime(directory)

            with self.assertRaises(PathSecurityError):
                runtime.read_text("../outside.txt")
            with self.assertRaises(PathSecurityError):
                runtime.write_text("../outside.txt", "blocked")

    def test_compatibility_sandbox_is_honest_about_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SandboxRuntime(directory)

            self.assertFalse(runtime.info.isolated)


class RuntimeToolTests(unittest.TestCase):
    def test_read_tool_is_allowed_and_write_requires_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalProcessRuntime(directory)
            runtime.write_text("input.txt", "hello")
            registry = ToolRegistry()
            for tool in create_runtime_tools(runtime):
                registry.register(tool)

            read = registry.dispatch("call_1", "read_file", '{"path":"input.txt"}')
            write = registry.dispatch(
                "call_2", "write_file", '{"path":"output.txt","content":"new"}'
            )

            self.assertTrue(read.success)
            self.assertEqual(read.output, "hello")
            self.assertFalse(write.success)
            self.assertFalse((Path(directory) / "output.txt").exists())

    def test_approved_runtime_tools_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalProcessRuntime(directory)
            registry = ToolRegistry(approvals=ApprovalManager(lambda *_: True))
            for tool in create_runtime_tools(runtime):
                registry.register(tool)

            write = registry.dispatch(
                "call_1", "write_file", '{"path":"output.txt","content":"new"}'
            )
            command = registry.dispatch(
                "call_2",
                "execute_command",
                '{"argv":["' + sys.executable.replace("\\", "\\\\") + '","-c","print(7)"]}',
            )

            self.assertTrue(write.success)
            self.assertEqual(runtime.read_text("output.txt"), "new")
            self.assertTrue(command.success)
            self.assertEqual(command.output["stdout"].strip(), "7")
            self.assertFalse(command.output["isolated"])

    def test_command_argument_items_are_type_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ToolRegistry(approvals=ApprovalManager(lambda *_: True))
            for tool in create_runtime_tools(LocalProcessRuntime(directory)):
                registry.register(tool)

            result = registry.dispatch(
                "call_1", "execute_command", '{"argv":["python", 123]}'
            )

            self.assertFalse(result.success)
            self.assertIn("argv[1]", result.error)


if __name__ == "__main__":
    unittest.main()
