import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from git.diff import generate_diff
from miniclaude.runtime_tools import create_runtime_tools
from miniclaude.tools import ToolRegistry
from runtime import LocalProcessRuntime
from security.approval import ApprovalManager


class CodingToolTests(unittest.TestCase):
    def registry(self, directory, approve=True):
        registry = ToolRegistry(
            approvals=ApprovalManager(lambda *_: approve)
        )
        for tool in create_runtime_tools(LocalProcessRuntime(directory)):
            registry.register(tool)
        return registry

    def test_list_glob_and_grep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("# TODO\nprint('hi')\n", encoding="utf-8")
            registry = self.registry(directory)

            listing = registry.dispatch("1", "list_directory", '{"path":"src"}')
            globbed = registry.dispatch("2", "glob_files", '{"pattern":"**/*.py"}')
            grepped = registry.dispatch("3", "grep_files", '{"pattern":"TODO"}')

            self.assertEqual(listing.output[0]["name"], "app.py")
            self.assertEqual(globbed.output, ["src/app.py"])
            self.assertEqual(grepped.output[0]["line"], 1)

    def test_replace_requires_unique_match_and_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "app.py"
            path.write_text("old\n", encoding="utf-8")
            blocked = self.registry(directory, approve=False).dispatch(
                "1", "replace_text", json.dumps({"path": "app.py", "old": "old", "new": "new"})
            )
            self.assertFalse(blocked.success)
            self.assertEqual(path.read_text(encoding="utf-8"), "old\n")

            changed = self.registry(directory).dispatch(
                "2", "replace_text", json.dumps({"path": "app.py", "old": "old", "new": "new"})
            )
            self.assertTrue(changed.success)
            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")

    def test_destructive_execute_is_denied_before_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.registry(directory).dispatch(
                "1", "execute_command", '{"argv":["rm","file"]}'
            )
            self.assertFalse(result.success)
            self.assertEqual(result.policy_action, "deny")

    def test_unified_diff(self):
        diff = generate_diff("old\n", "new\n", "app.py")
        self.assertIn("--- a/app.py", diff)
        self.assertIn("+++ b/app.py", diff)
        self.assertIn("-old", diff)
        self.assertIn("+new", diff)

    def test_git_diff_nonzero_exit_is_a_failed_observation(self):
        class FailedGitRuntime:
            info = SimpleNamespace(name="test", workspace=Path.cwd())

            def read_text(self, path):
                return ""

            def write_text(self, path, content):
                return len(content)

            def execute(self, argv, **kwargs):
                return SimpleNamespace(
                    succeeded=False,
                    exit_code=129,
                    stderr="not a git repository",
                    stdout="",
                )

        registry = ToolRegistry()
        for tool in create_runtime_tools(FailedGitRuntime()):
            registry.register(tool)

        result = registry.dispatch("1", "git_diff", "{}")

        self.assertFalse(result.success)
        self.assertIn("exit code 129", result.error)
        self.assertIn("not a git repository", result.error)


if __name__ == "__main__":
    unittest.main()
