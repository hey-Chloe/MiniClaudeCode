import tempfile
import unittest
from pathlib import Path

from miniclaude.runtime_tools import create_runtime_tools
from miniclaude.tools import ToolRegistry
from runtime import LocalProcessRuntime


class WorkspaceDiffTests(unittest.TestCase):
    def _registry(self, workspace):
        registry = ToolRegistry()
        for tool in create_runtime_tools(LocalProcessRuntime(workspace)):
            registry.register(tool)
        return registry

    def test_reports_changed_files_with_unified_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "a.py").write_text("old\n", encoding="utf-8")
            (workspace / "b.py").write_text("same\n", encoding="utf-8")
            registry = self._registry(workspace)

            (workspace / "a.py").write_text("new\n", encoding="utf-8")
            result = registry.dispatch("call_1", "workspace_diff", "{}")

            self.assertTrue(result.success)
            self.assertEqual(result.output["changed_files"], ["a.py"])
            self.assertIn("-old", result.output["diff"])
            self.assertIn("+new", result.output["diff"])
            self.assertNotIn("b.py", result.output["diff"])

    def test_no_changes_returns_empty_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "a.py").write_text("same\n", encoding="utf-8")
            result = self._registry(workspace).dispatch(
                "call_1", "workspace_diff", "{}"
            )

            self.assertTrue(result.success)
            self.assertEqual(result.output["changed_files"], [])
            self.assertEqual(result.output["diff"], "")

    def test_cache_artifacts_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "a.py").write_text("x\n", encoding="utf-8")
            registry = self._registry(workspace)

            cache = workspace / "__pycache__"
            cache.mkdir()
            (cache / "a.cpython-314.pyc").write_bytes(b"junk")
            (workspace / "a.py").write_text("y\n", encoding="utf-8")
            result = registry.dispatch("call_1", "workspace_diff", "{}")

            self.assertEqual(result.output["changed_files"], ["a.py"])
            self.assertNotIn("__pycache__", result.output["diff"])


if __name__ == "__main__":
    unittest.main()
