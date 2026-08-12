import json
import tempfile
import unittest
from pathlib import Path

from miniclaude.agent import Agent
from miniclaude.session import SessionStore
from miniclaude.trace import Trace


class TraceTests(unittest.TestCase):
    def test_legacy_and_detailed_exports(self):
        trace = Trace()
        trace.add("task", "hello")
        self.assertEqual(trace.export(), [{"event": "task", "detail": "hello"}])
        detailed = trace.export_detailed()[0]
        self.assertEqual(detailed["event"], "task")
        self.assertIn("timestamp", detailed)
        self.assertIn("run_id", detailed)

    def test_jsonl_export(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Trace()
            trace.add("task", "hello")
            path = Path(directory) / "trace.jsonl"
            trace.write_jsonl(path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["event"], "task")


class SessionTests(unittest.TestCase):
    def test_save_and_load_session(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent()
            result = agent.run_result("task")
            store = SessionStore(directory)
            path = store.save("session-1", result, agent.trace.export_detailed())
            loaded = store.load("session-1")
            self.assertTrue(path.is_file())
            self.assertEqual(loaded["task"], "task")
            self.assertEqual(loaded["status"], "completed")

    def test_invalid_session_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                SessionStore(directory).load("../escape")


if __name__ == "__main__":
    unittest.main()
