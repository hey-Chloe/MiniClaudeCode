import tempfile
import unittest
from pathlib import Path

from miniclaude.memory import FileCache
from miniclaude.runtime_tools import create_runtime_tools
from miniclaude.tools import ToolRegistry
from runtime import LocalProcessRuntime
from security.approval import ApprovalManager


class FileCacheTests(unittest.TestCase):
    def test_hit_requires_same_mtime_and_size(self):
        cache = FileCache()
        cache.put("a.py", mtime=10.0, size=3, content="abc")

        self.assertEqual(cache.get("a.py", 10.0, 3), "abc")
        self.assertIsNone(cache.get("a.py", 11.0, 3))
        self.assertIsNone(cache.get("a.py", 10.0, 4))
        self.assertEqual(cache.stats()["hits"], 1)
        self.assertEqual(cache.stats()["misses"], 2)

    def test_invalidate_drops_entry(self):
        cache = FileCache()
        cache.put("a.py", 1.0, 1, "x")
        cache.invalidate("a.py")

        self.assertIsNone(cache.get("a.py", 1.0, 1))
        self.assertEqual(cache.stats()["invalidations"], 1)


class CachedReadToolTests(unittest.TestCase):
    def registry(self, directory):
        registry = ToolRegistry(
            approvals=ApprovalManager(lambda *_: True)
        )
        for tool in create_runtime_tools(LocalProcessRuntime(directory)):
            registry.register(tool)
        return registry

    def test_repeated_read_uses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("print('hi')\n", encoding="utf-8")
            registry = self.registry(directory)

            first = registry.dispatch("1", "read_file", '{"path":"app.py"}')
            second = registry.dispatch("2", "read_file", '{"path":"app.py"}')

            self.assertEqual(first.output["cache_hit"], False)
            self.assertEqual(second.output["cache_hit"], True)
            self.assertEqual(first.output["content"], second.output["content"])

    def test_write_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("one\n", encoding="utf-8")
            registry = self.registry(directory)

            first = registry.dispatch("1", "read_file", '{"path":"app.py"}')
            registry.dispatch(
                "2",
                "write_file",
                '{"path":"app.py","content":"two\\n"}',
            )
            third = registry.dispatch("3", "read_file", '{"path":"app.py"}')

            self.assertEqual(first.output["cache_hit"], False)
            self.assertEqual(third.output["cache_hit"], False)
            self.assertEqual(third.output["content"], "two\n")

    def test_external_change_is_not_served_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "app.py"
            target.write_text("one\n", encoding="utf-8")
            registry = self.registry(directory)

            first = registry.dispatch("1", "read_file", '{"path":"app.py"}')
            target.write_text("changed\n", encoding="utf-8")
            second = registry.dispatch("2", "read_file", '{"path":"app.py"}')

            self.assertEqual(first.output["cache_hit"], False)
            self.assertEqual(second.output["cache_hit"], False)
            self.assertEqual(second.output["content"], "changed\n")


if __name__ == "__main__":
    unittest.main()
