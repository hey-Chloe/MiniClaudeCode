import tempfile
import unittest
from pathlib import Path

from miniclaude.agent import Agent
from miniclaude.context import ContextConfig, ContextManager
from miniclaude.llm import LLMResponse
from miniclaude.skills import SkillRegistry


BUG_FIX_SKILL = """\
---
name: bug-fix
description: Locate, patch, and verify failing code.
when_to_use: failing test, bug, broken, traceback, pytest, fix
version: 1.0.0
tools: read_file, grep_files, execute_command
---

# Bug Fix Workflow
1. Read the failing tests first.
2. Reproduce the failure.
3. Make the smallest change.
4. Re-run the tests until green.
"""

CODE_REVIEW_SKILL = """\
---
name: code-review
description: Review a diff for correctness and safety.
when_to_use: review, pull request, diff, audit
version: 1.0.0
---

# Code Review Workflow
1. Scope the diff.
2. Check correctness, security, clarity.
3. Rank findings by severity.
"""


def _write_skills(directory: Path) -> Path:
    root = directory / "skills"
    (root / "bug-fix").mkdir(parents=True)
    (root / "code-review").mkdir()
    (root / "bug-fix" / "SKILL.md").write_text(BUG_FIX_SKILL, encoding="utf-8")
    (root / "code-review" / "SKILL.md").write_text(
        CODE_REVIEW_SKILL, encoding="utf-8"
    )
    return root


class SkillRegistryTests(unittest.TestCase):
    def test_discover_and_select(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_skills(Path(directory))
            registry = SkillRegistry(root)

            self.assertEqual(registry.names(), ["bug-fix", "code-review"])
            selected = registry.select("fix the failing pytest test", top_k=1)
            self.assertEqual([spec.name for spec in selected], ["bug-fix"])
            self.assertEqual(
                [spec.name for spec in registry.select("audit this diff", top_k=1)],
                ["code-review"],
            )

    def test_unrelated_task_selects_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(_write_skills(Path(directory)))
            self.assertEqual(registry.select("write a poem", top_k=1), [])

    def test_hybrid_matches_keyword_recall(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(_write_skills(Path(directory)))
            self.assertEqual(
                [spec.name for spec in registry.select("fix the failing pytest test", top_k=1)],
                ["bug-fix"],
            )
            self.assertEqual(
                [spec.name for spec in registry.select("audit this diff", top_k=1)],
                ["code-review"],
            )

    def test_semantic_mode_scores_by_similarity(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(_write_skills(Path(directory)))
            selected = registry.select(
                "the failing test suite keeps erroring",
                top_k=1,
                mode="semantic",
            )
            self.assertEqual([spec.name for spec in selected], ["bug-fix"])

    def test_tools_front_matter_is_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(_write_skills(Path(directory)))
            self.assertEqual(
                registry.get("bug-fix").tools,
                ("read_file", "grep_files", "execute_command"),
            )
            self.assertEqual(registry.get("code-review").tools, ())

    def test_unsupported_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = SkillRegistry(_write_skills(Path(directory)))
            with self.assertRaisesRegex(ValueError, "unsupported selection mode"):
                registry.select("fix tests", mode="vector")

    def test_missing_front_matter_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad").mkdir()
            (root / "bad" / "SKILL.md").write_text("no front matter", encoding="utf-8")
            with self.assertRaises(ValueError):
                SkillRegistry(root)


class ContextSkillsTests(unittest.TestCase):
    def test_matching_skill_is_injected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_skills(Path(directory))
            manager = ContextManager(
                ContextConfig(
                    system_instructions="system",
                    skills_dir=root,
                )
            )
            snapshot = manager.start("fix the failing pytest test")

            self.assertEqual(snapshot.skills, ("bug-fix",))
            self.assertEqual(
                manager.selected_skill_tools(),
                ("read_file", "grep_files", "execute_command"),
            )
            self.assertIn("## Skill: bug-fix", snapshot.instructions)
            self.assertIn("Read the failing tests first", snapshot.instructions)

    def test_unrelated_task_injects_no_skills(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_skills(Path(directory))
            manager = ContextManager(
                ContextConfig(
                    system_instructions="system",
                    skills_dir=root,
                )
            )
            snapshot = manager.start("write a poem")

            self.assertEqual(snapshot.skills, ())
            self.assertNotIn("## Skill:", snapshot.instructions)

    def test_skill_budget_truncation_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_skills(Path(directory))
            manager = ContextManager(
                ContextConfig(
                    system_instructions="system",
                    skills_dir=root,
                    skill_budget_chars=30,
                )
            )
            snapshot = manager.start("fix the failing pytest test")

            self.assertTrue(snapshot.truncated)
            self.assertEqual(snapshot.skills, ("bug-fix",))


class AgentSkillsIntegrationTests(unittest.TestCase):
    class ScriptedProvider:
        def __init__(self, response):
            self.response = response
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return self.response

    def test_result_and_trace_expose_loaded_skill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_skills(Path(directory))
            provider = self.ScriptedProvider(LLMResponse(text="done"))
            agent = Agent(
                provider=provider,
                context_config=ContextConfig(
                    system_instructions="system",
                    skills_dir=root,
                ),
            )
            result = agent.run_result("fix the failing pytest test")

            self.assertEqual(result.skills, ("bug-fix",))
            self.assertTrue(
                any(event.get("event") == "skill_loaded" for event in result.events)
            )
            self.assertIn(
                "## Skill: bug-fix", provider.requests[0].instructions
            )


if __name__ == "__main__":
    unittest.main()
