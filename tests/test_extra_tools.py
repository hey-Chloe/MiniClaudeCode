import tempfile
import unittest
from pathlib import Path

from miniclaude.runtime_tools import create_runtime_tools
from miniclaude.tools import ToolRegistry
from runtime import LocalProcessRuntime
from security.approval import ApprovalManager


def build_registry(directory):
    registry = ToolRegistry(
        approvals=ApprovalManager(lambda *_: True)
    )
    for tool in create_runtime_tools(LocalProcessRuntime(directory)):
        registry.register(tool)
    return registry


class ExtendedToolTests(unittest.TestCase):
    def _workspace(self):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        (root / "src" / "nested").mkdir(parents=True)
        (root / "src" / "app.py").write_text(
            "# TODO: handle empty input\n"
            "def run():\n"
            "    return 1\n",
            encoding="utf-8",
        )
        (root / "src" / "nested" / "util.py").write_text(
            "FIXME: rename me\n",
            encoding="utf-8",
        )
        (root / "README.md").write_text("docs\n", encoding="utf-8")
        return directory, root

    def test_builtin_toolset_has_thirteen_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = build_registry(directory)
            names = registry.list_tools()

            self.assertEqual(len(names), 13)
            self.assertEqual(len(set(names)), 13)
            self.assertIn("file_tree", names)
            self.assertIn("todo_scan", names)
            self.assertIn("file_stat", names)

    def test_file_tree_lists_nested_paths_and_respects_depth(self):
        directory, _ = self._workspace()
        try:
            registry = build_registry(directory.name)

            result = registry.dispatch(
                "1", "file_tree", '{"path": ".", "depth": 1}'
            )

            self.assertTrue(result.success)
            paths = {entry["path"] for entry in result.output}
            self.assertIn("README.md", paths)
            self.assertIn("src", paths)
            self.assertNotIn("src/nested", paths)
            self.assertIn("src", [entry["path"] for entry in result.output])
        finally:
            directory.cleanup()

    def test_todo_scan_finds_markers(self):
        directory, _ = self._workspace()
        try:
            registry = build_registry(directory.name)

            result = registry.dispatch(
                "2",
                "todo_scan",
                '{"pattern": "TODO|FIXME", "path": "."}',
            )

            self.assertTrue(result.success)
            paths = {entry["path"] for entry in result.output}
            self.assertIn("src/app.py", paths)
            self.assertIn("src/nested/util.py", paths)
            todo = next(
                entry
                for entry in result.output
                if entry["path"] == "src/app.py"
            )
            self.assertEqual(todo["line"], 1)
            self.assertIn("TODO", todo["text"])
        finally:
            directory.cleanup()

    def test_file_stat_reports_metadata(self):
        directory, _ = self._workspace()
        try:
            registry = build_registry(directory.name)

            file_result = registry.dispatch(
                "3", "file_stat", '{"path": "src/app.py"}'
            )
            dir_result = registry.dispatch(
                "4", "file_stat", '{"path": "src"}'
            )

            self.assertTrue(file_result.success)
            self.assertEqual(file_result.output["type"], "file")
            self.assertEqual(file_result.output["line_count"], 3)
            self.assertGreater(file_result.output["size_bytes"], 0)
            self.assertEqual(dir_result.output["type"], "directory")
        finally:
            directory.cleanup()

    def test_missing_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = build_registry(directory)

            result = registry.dispatch(
                "5", "file_stat", '{"path": "missing.txt"}'
            )

            self.assertFalse(result.success)
            self.assertIn("does not exist", result.error)


if __name__ == "__main__":
    unittest.main()
