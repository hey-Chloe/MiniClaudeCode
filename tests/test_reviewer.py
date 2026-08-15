import tempfile
import unittest
from pathlib import Path

from miniclaude.agent import Agent
from miniclaude.llm import LLMResponse, LLMToolCall
from miniclaude.models import RunStatus
from miniclaude.reviewer import REVIEW_INSTRUCTIONS, build_review_verifier
from miniclaude.tools import ToolDefinition
from runtime import LocalProcessRuntime


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
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class ReviewerVerifierTests(unittest.TestCase):
    def _verifier(self, review_text):
        provider = ScriptedProvider([LLMResponse(text=review_text)])
        with tempfile.TemporaryDirectory() as directory:
            verifier = build_review_verifier(
                provider,
                LocalProcessRuntime(directory),
            )
            return verifier(), provider.requests

    def test_approved_review_passes(self):
        verdict, requests = self._verifier("APPROVED\nLGTM, tests pass.")

        self.assertTrue(verdict["passed"])
        self.assertIn("LGTM", verdict["output"])
        self.assertIn(REVIEW_INSTRUCTIONS, requests[0].instructions)

    def test_changes_requested_fails_and_returns_comments(self):
        verdict, _ = self._verifier(
            "CHANGES_REQUESTED missing edge case\n"
            "1. handle empty input\n2. add a test"
        )

        self.assertFalse(verdict["passed"])
        self.assertIn("empty input", verdict["output"])


class ReviewerLoopIntegrationTests(unittest.TestCase):
    def test_reviewer_rejection_feeds_back_and_approval_finalizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main_provider = ScriptedProvider(
                [
                    LLMResponse(text="I will add the file."),
                    LLMResponse(
                        response_id="resp_1",
                        tool_calls=(
                            LLMToolCall(
                                "call_1",
                                "write_file",
                                '{"path": "note.txt", "content": "x"}',
                            ),
                        ),
                    ),
                    LLMResponse(text="done"),
                    LLMResponse(text="done after review"),
                ]
            )
            reviewer_provider = ScriptedProvider(
                [
                    LLMResponse(text="CHANGES_REQUESTED missing test\nadd a test"),
                    LLMResponse(text="APPROVED\nlooks good now"),
                ]
            )
            verifier = build_review_verifier(
                reviewer_provider,
                LocalProcessRuntime(root),
            )
            result = Agent(
                provider=main_provider,
                tools=[write_tool(root)],
                verifier=verifier,
            ).run_result("add a file")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.output, "done after review")
        self.assertEqual(result.phases.count("verify"), 2)
        verification_events = [
            event for event in result.events if event.get("event") == "verification"
        ]
        self.assertEqual(len(verification_events), 2)
        self.assertFalse(verification_events[0]["detail"]["passed"])
        self.assertTrue(verification_events[1]["detail"]["passed"])
        self.assertIn(
            "missing test",
            verification_events[0]["detail"]["output"],
        )
        self.assertEqual(main_provider.requests[-1].tool_outputs, ())


if __name__ == "__main__":
    unittest.main()
