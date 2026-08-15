import tempfile
import unittest
from pathlib import Path

from miniclaude.agent import Agent
from miniclaude.llm import LLMResponse, LLMToolCall
from miniclaude.models import RunStatus
from miniclaude.tools import ToolDefinition


def write_tool(directory: Path) -> ToolDefinition:
    def write_file(path: str, content: str):
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": path}

    return ToolDefinition(
        name="write_file",
        description="Write a file inside the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        handler=write_file,
    )


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return self.responses.pop(0)


class LoopPhaseTests(unittest.TestCase):
    def _result(self, provider, tools, verifier=None, plan_first=True):
        return (
            Agent(
                provider=provider,
                tools=tools,
                verifier=verifier,
                plan_first=plan_first,
            ).run_result("write a file")
        )

    def test_plan_act_verify_finalize_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = ScriptedProvider(
                [
                    LLMResponse(text="I will write the file."),
                    LLMResponse(
                        response_id="resp_1",
                        tool_calls=(
                            LLMToolCall(
                                "call_1",
                                "write_file",
                                '{"path": "hello.txt", "content": "hi"}',
                            ),
                        ),
                    ),
                    LLMResponse(text="done"),
                ]
            )
            verifier_calls = []

            def verifier():
                verifier_calls.append(1)
                return {"passed": True, "output": "1 passed"}

            result = self._result(provider, [write_tool(Path(directory))], verifier)

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(
            result.phases,
            ("plan", "act", "observe", "verify", "finalize"),
        )
        self.assertEqual(
            [event["event"] for event in result.events],
            [
                "task",
                "plan",
                "tool_calls",
                "tool_results",
                "file_modified",
                "verification",
                "answer",
            ],
        )
        self.assertEqual(len(verifier_calls), 1)
        self.assertEqual(provider.calls, 3)  # verification never re-invokes the model

    def test_failed_verification_returns_to_model(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = ScriptedProvider(
                [
                    LLMResponse(text="I will write the file."),
                    LLMResponse(
                        response_id="resp_1",
                        tool_calls=(
                            LLMToolCall(
                                "call_1",
                                "write_file",
                                '{"path": "hello.txt", "content": "hi"}',
                            ),
                        ),
                    ),
                    LLMResponse(text="done"),
                    LLMResponse(text="fixed"),
                ]
            )
            verifier_runs = []

            def verifier():
                verifier_runs.append(1)
                return {
                    "passed": len(verifier_runs) > 1,
                    "output": "0 passed" if len(verifier_runs) == 1 else "1 passed",
                }

            result = self._result(provider, [write_tool(Path(directory))], verifier)

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(
            result.phases,
            ("plan", "act", "observe", "verify", "verify", "finalize"),
        )
        self.assertEqual(len(verifier_runs), 2)
        self.assertEqual(provider.calls, 4)

    def test_no_edits_skips_verification(self):
        provider = ScriptedProvider(
            [
                LLMResponse(text="no changes needed."),
                LLMResponse(text="answer"),
            ]
        )
        verifier_runs = []

        def verifier():
            verifier_runs.append(1)
            return {"passed": True, "output": "unused"}

        result = self._result(provider, [], verifier)

        self.assertEqual(result.phases, ("plan", "finalize"))
        self.assertEqual(verifier_runs, [])
        self.assertEqual(provider.calls, 2)

    def test_plan_first_can_be_disabled(self):
        provider = ScriptedProvider([LLMResponse(text="answer")])
        result = self._result(provider, [], plan_first=False)

        self.assertEqual(result.phases, ("finalize",))
        self.assertEqual(result.events[-1]["event"], "answer")
        self.assertEqual(provider.calls, 1)

    def test_reflect_phase_and_file_modified_events(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = ScriptedProvider(
                [
                    LLMResponse(text="I will fix it."),
                    LLMResponse(
                        response_id="resp_1",
                        tool_calls=(
                            LLMToolCall(
                                "call_1",
                                "no_such_tool",
                                "{}",
                            ),
                            LLMToolCall(
                                "call_2",
                                "write_file",
                                '{"path": "note.txt", "content": "x"}',
                            ),
                        ),
                    ),
                    LLMResponse(text="done"),
                ]
            )
            result = self._result(
                provider,
                [write_tool(Path(directory))],
                verifier=None,
            )
            self.assertTrue((Path(directory) / "note.txt").is_file())

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(
            result.phases,
            ("plan", "act", "observe", "reflect", "finalize"),
        )
        self.assertIn(
            {"event": "file_modified", "detail": {"tool": "write_file", "path": "note.txt"}},
            result.events,
        )

    def test_legacy_driver_reports_phases(self):
        result = Agent().run_result("inspect architecture")
        self.assertEqual(result.phases, ("plan", "act", "verify"))


if __name__ == "__main__":
    unittest.main()
