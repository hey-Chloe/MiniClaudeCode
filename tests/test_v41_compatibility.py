import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

from miniclaude.agent import Agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AgentCompatibilityTests(unittest.TestCase):
    def test_agent_returns_v41_trace_shape(self):
        result = Agent().run("inspect architecture")

        self.assertEqual(
            result,
            [
                {"event": "task", "detail": "inspect architecture"},
                {"event": "planning", "detail": "create plan"},
                {"event": "tool_selection", "detail": "pytest"},
                {"event": "verification", "detail": "passed"},
            ],
        )


class MainEntryPointCompatibilityTests(unittest.TestCase):
    def test_main_py_accepts_positional_task_and_prints_trace(self):
        environment = os.environ.copy()
        environment["MINICLAUDE_MODEL"] = ""
        environment["OPENAI_API_KEY"] = ""
        completed = subprocess.run(
            [sys.executable, "main.py", "inspect architecture"],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        events = [ast.literal_eval(line) for line in lines]
        self.assertEqual(events[0], {"event": "task", "detail": "inspect architecture"})
        self.assertEqual(events[-1], {"event": "verification", "detail": "passed"})
        self.assertEqual(len(events), 4)


if __name__ == "__main__":
    unittest.main()
