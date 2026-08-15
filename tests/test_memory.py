import tempfile
import time
import unittest
from pathlib import Path

from miniclaude.agent import Agent
from miniclaude.llm import LLMResponse
from miniclaude.memory import FileCache, PersistentMemory, WorkingMemory
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


class WorkingMemoryTests(unittest.TestCase):
    def test_put_get_and_ttl_expiry(self):
        memory = WorkingMemory(ttl_seconds=0.05)
        memory.put("tool_1", "read_file ok")

        self.assertEqual(memory.get("tool_1").content, "read_file ok")
        time.sleep(0.06)
        self.assertIsNone(memory.get("tool_1"))

    def test_retrieve_ranks_by_keyword_overlap(self):
        memory = WorkingMemory()
        memory.put("a", "pytest failing test suite")
        memory.put("b", "unrelated shopping list")

        top = memory.retrieve("failing pytest", top_k=1)
        self.assertEqual(top[0].key, "a")

    def test_invalidate_and_eviction_bound(self):
        memory = WorkingMemory(max_entries=2)
        memory.put("a", "one")
        memory.put("b", "two")
        memory.put("c", "three")

        self.assertEqual(memory.stats()["entries"], 2)
        memory.invalidate("b")
        self.assertIsNone(memory.get("b"))


class PersistentMemoryTests(unittest.TestCase):
    def test_round_trip_across_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            memory = PersistentMemory(path)
            memory.put("task-1", "status=completed; turns=3")

            reloaded = PersistentMemory(path)
            self.assertEqual(
                reloaded.get("task-1").content,
                "status=completed; turns=3",
            )

    def test_expired_entries_are_pruned_on_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            memory = PersistentMemory(path)
            memory.put(
                "stale",
                "old result",
                ttl=0.01,
            )
            time.sleep(0.02)
            reloaded = PersistentMemory(path)

            self.assertIsNone(reloaded.get("stale"))

    def test_retrieve_and_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            memory = PersistentMemory(path)
            memory.put("task-x", "pytest failing on module core")

            top = memory.retrieve("failing pytest", top_k=1)
            self.assertEqual(top[0].key, "task-x")
            memory.invalidate("task-x")
            self.assertIsNone(memory.get("task-x"))


class AgentMemoryIntegrationTests(unittest.TestCase):
    class ScriptedProvider:
        def __init__(self, responses):
            self.responses = list(responses)
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return self.responses.pop(0)

    def test_cross_session_memory_is_injected_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentMemory(Path(directory) / "memory.jsonl")
            first = Agent(
                provider=self.ScriptedProvider(
                    [
                        LLMResponse(text="I will handle it."),
                        LLMResponse(text="done"),
                    ]
                ),
                memory=memory,
            )
            result = first.run_result("echo hi")
            self.assertEqual(result.status.value, "completed")

            self.assertIsNotNone(memory.get("echo hi"))

            second_provider = self.ScriptedProvider(
                [
                    LLMResponse(text="I will handle it again."),
                    LLMResponse(text="done again"),
                ]
            )
            second = Agent(
                provider=second_provider,
                memory=PersistentMemory(Path(directory) / "memory.jsonl"),
            )
            second.run_result("echo hi")

            messages = second_provider.requests[0].messages
            roles = [message["role"] for message in messages]
            self.assertIn("assistant", roles)
            memory_message = next(
                message["content"]
                for message in messages
                if message["role"] == "assistant"
            )
            self.assertIn("Run memory from previous sessions", memory_message)
            self.assertIn("status=completed", memory_message)

    def test_cache_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("print('hi')\n", encoding="utf-8")
            registry = ToolRegistry(
                approvals=ApprovalManager(lambda *_: True)
            )
            for tool in create_runtime_tools(
                LocalProcessRuntime(directory),
                cache_enabled=False,
            ):
                registry.register(tool)

            first = registry.dispatch("1", "read_file", '{"path":"app.py"}')
            second = registry.dispatch("2", "read_file", '{"path":"app.py"}')

            self.assertEqual(first.output["cache_hit"], False)
            self.assertEqual(second.output["cache_hit"], False)


if __name__ == "__main__":
    unittest.main()
