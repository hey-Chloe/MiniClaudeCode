import tempfile
import threading
import time
import unittest
from pathlib import Path

from miniclaude.agents import (
    CollaborationBlackboard,
    CoordinatorAgent,
    SpecialistAgent,
    Subtask,
)
from miniclaude.agents.coordinator import default_decompose
from miniclaude.llm import LLMResponse
from miniclaude.tools import ToolDefinition


def echo_tool(name="echo"):
    return ToolDefinition(
        name=name,
        description=f"Tool {name}.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        handler=lambda text: {"text": text},
    )


class ScriptedProvider:
    def __init__(self, responses, failure=None):
        self.responses = list(responses)
        self.failure = failure

    def complete(self, request):
        if self.failure is not None:
            raise self.failure
        return self.responses.pop(0)


class BlackboardTests(unittest.TestCase):
    def test_publish_dedup_query_and_verify(self):
        board = CollaborationBlackboard()
        first = board.publish("a", "finding", "module core is broken")
        duplicate = board.publish("b", "finding", "module core is broken")
        board.publish("a", "finding", "another issue", source="src/x.py")

        self.assertEqual(first.id, duplicate.id)
        self.assertEqual(board.stats()["evidence"], 2)
        self.assertEqual(len(board.query(kind="finding")), 2)
        self.assertEqual(len(board.query(keyword="core")), 1)
        board.verify(first.id, True)
        self.assertTrue(board.get(first.id).verified)
        self.assertEqual(board.stats()["verified"], 1)

    def test_verify_unknown_id_returns_false(self):
        board = CollaborationBlackboard()
        self.assertFalse(board.verify("missing", True))


class CoordinatorTests(unittest.TestCase):
    def _coordinator(self, providers, workspace, tools, **overrides):
        def factory(name):
            return providers[name]

        return CoordinatorAgent(
            provider_factory=factory,
            workspace=workspace,
            tools=tools,
            concurrency=2,
            **overrides,
        )

    def test_default_decomposition_covers_roles(self):
        subtasks = default_decompose("fix the failing test and review the repo")

        self.assertEqual(
            [subtask.specialist for subtask in subtasks],
            ["analyzer", "implementer", "verifier"],
        )

    def test_run_specialists_concurrently_and_synthesizes(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "src").mkdir()
            (workspace / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
            providers = {
                "analyzer": ScriptedProvider(
                    [LLMResponse(text="found: src/app.py")]
                ),
                "implementer": ScriptedProvider(
                    [LLMResponse(text="wrote the fix")]
                ),
                "verifier": ScriptedProvider(
                    [LLMResponse(text="tests pass")]
                ),
            }
            coordinator = self._coordinator(
                providers,
                workspace,
                [echo_tool()],
            )

            result = coordinator.run("review the repo and fix the failing test")

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.specialist_results), 3)
        self.assertIn("found: src/app.py", result.output)
        self.assertIn("wrote the fix", result.output)
        self.assertIn("tests pass", result.output)
        self.assertEqual(result.blackboard_stats["evidence"], 3)

    def test_concurrency_is_bounded(self):
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()

        class SlowProvider:
            def complete(self, request):
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.05)
                with lock:
                    state["active"] -= 1
                return LLMResponse(text="ok")

        with tempfile.TemporaryDirectory() as directory:
            coordinator = CoordinatorAgent(
                provider_factory=lambda name: SlowProvider(),
                workspace=Path(directory),
                tools=[echo_tool()],
                concurrency=2,
            )
            result = coordinator.run("review the repo and fix the failing test")

        self.assertEqual(len(result.specialist_results), 3)
        self.assertEqual(state["peak"], 2)

    def test_partial_failure_keeps_remaining_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            providers = {
                "analyzer": ScriptedProvider(
                    [LLMResponse(text="analysis done")]
                ),
                "implementer": ScriptedProvider(
                    [LLMResponse(text="fix done")],
                    failure=RuntimeError("provider down"),
                ),
                "verifier": ScriptedProvider(
                    [LLMResponse(text="verified")]
                ),
            }
            coordinator = self._coordinator(
                providers,
                workspace,
                [echo_tool()],
            )

            result = coordinator.run("review the repo and fix the failing test")

        self.assertEqual(result.status, "partial")
        failed = next(
            item
            for item in result.specialist_results
            if item.specialist == "implementer"
        )
        self.assertEqual(failed.status, "failed")
        self.assertIn("provider down", failed.error)
        self.assertEqual(result.blackboard_stats["evidence"], 2)

    def test_critic_verdict_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            providers = {
                "analyzer": ScriptedProvider(
                    [LLMResponse(text="answer one")]
                )
            }
            critic = ScriptedProvider(
                [LLMResponse(text="CHANGES_REQUESTED add citations")]
            )
            coordinator = self._coordinator(
                providers,
                Path(directory),
                [echo_tool()],
                critic_provider=critic,
            )

            result = coordinator.run("analyze the repo")

        self.assertIsNotNone(result.critic)
        self.assertFalse(result.critic["approved"])
        self.assertIn("add citations", result.critic["comments"])

    def test_evidence_source_is_verified_against_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "src").mkdir()
            (workspace / "src" / "real.py").write_text("x\n", encoding="utf-8")
            board = CollaborationBlackboard()
            board.publish(
                "analyzer",
                "tool_observation",
                "read_file: ok",
                source="src/real.py",
            )
            board.publish(
                "analyzer",
                "tool_observation",
                "read_file: ok",
                source="src/missing.py",
            )
            providers = {
                "analyzer": ScriptedProvider(
                    [LLMResponse(text="analysis")]
                )
            }
            coordinator = self._coordinator(
                providers,
                workspace,
                [echo_tool()],
                blackboard=board,
            )

            coordinator.run("analyze the repo")

            by_source = {
                item.source: item.verified for item in board.items()
            }
            self.assertTrue(by_source["src/real.py"])
            self.assertFalse(by_source["src/missing.py"])


class SpecialistTests(unittest.TestCase):
    def test_specialist_publishes_answer_and_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            board = CollaborationBlackboard()
            specialist = SpecialistAgent(
                name="implementer",
                provider=ScriptedProvider(
                    [LLMResponse(text="fixed it")]
                ),
                workspace=Path(directory),
                system_instructions="implement",
                tools=[echo_tool()],
                blackboard=board,
            )

            result = specialist.run(
                Subtask(
                    id="implement",
                    description="fix",
                    specialist="implementer",
                )
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.output, "fixed it")
        self.assertEqual(board.stats()["evidence"], 1)
        self.assertEqual(board.query(kind="answer")[0].agent, "implementer")


if __name__ == "__main__":
    unittest.main()
