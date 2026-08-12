import unittest

from miniclaude.agent import Agent
from miniclaude.controller import AgentController
from miniclaude.models import AgentState, LoopDecision, RunStatus


class ScriptedDriver:
    def __init__(self, decisions):
        self.decisions = decisions

    def next(self, state: AgentState) -> LoopDecision:
        return self.decisions[state.turn_count]


class EndlessDriver:
    def next(self, state: AgentState) -> LoopDecision:
        return LoopDecision("thinking", f"turn-{state.turn_count + 1}")


class FailingDriver:
    def next(self, state: AgentState) -> LoopDecision:
        raise RuntimeError("decision failed")


class InvalidDriver:
    def next(self, state: AgentState):
        return {"event": "invalid"}


class AgentLoopTests(unittest.TestCase):
    def test_completes_after_multiple_turns(self):
        driver = ScriptedDriver(
            [
                LoopDecision("thinking", "inspect"),
                LoopDecision("answer", "done", terminal=True),
            ]
        )

        result = Agent(driver=driver, max_turns=5).run_result("fix the bug")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.turns, 2)
        self.assertEqual(result.output, "done")
        self.assertIsNone(result.error)
        self.assertEqual(result.events[-1], {"event": "answer", "detail": "done"})

    def test_stops_at_maximum_turn_limit(self):
        result = Agent(driver=EndlessDriver(), max_turns=2).run_result("keep thinking")

        self.assertEqual(result.status, RunStatus.MAX_TURNS)
        self.assertEqual(result.turns, 2)
        self.assertEqual(result.events[-1]["event"], "termination")
        self.assertIn("2", result.error)

    def test_driver_failure_becomes_failed_result(self):
        result = Agent(driver=FailingDriver()).run_result("fail safely")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.turns, 0)
        self.assertEqual(result.error, "decision failed")
        self.assertEqual(result.events[-1]["event"], "error")

    def test_invalid_driver_result_is_rejected(self):
        controller = AgentController(InvalidDriver())

        with self.assertRaisesRegex(TypeError, "LoopDecision"):
            controller.run("validate driver")

    def test_reusing_agent_starts_with_fresh_trace(self):
        agent = Agent()

        first = agent.run("first task")
        second = agent.run("second task")

        self.assertEqual(len(first), 4)
        self.assertEqual(len(second), 4)
        self.assertEqual(second[0], {"event": "task", "detail": "second task"})

    def test_empty_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            Agent().run_result("   ")


if __name__ == "__main__":
    unittest.main()
