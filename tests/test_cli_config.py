import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from miniclaude.config import AppConfig
from security.policy import PermissionModePolicy, PolicyAction, PolicyRequest, ToolRisk


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_loads_environment_and_overrides(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "MINICLAUDE_MODEL": "env-model",
                "MINICLAUDE_WORKSPACE": directory,
                "MINICLAUDE_MAX_TURNS": "7",
            },
            clear=True,
        ):
            config = AppConfig.from_env(model="cli-model")

        self.assertEqual(config.model, "cli-model")
        self.assertEqual(config.max_turns, 7)
        self.assertEqual(config.workspace, Path(directory).resolve())

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported permission"):
            AppConfig.from_env(permission_mode="unsafe")


class PermissionModeTests(unittest.TestCase):
    def request(self, name="write_file", risk=ToolRisk.MUTATING):
        return PolicyRequest(name, risk, {})

    def test_plan_denies_mutation(self):
        decision = PermissionModePolicy("plan").evaluate(self.request())
        self.assertEqual(decision.action, PolicyAction.DENY)

    def test_accept_edits_allows_write_but_not_commands(self):
        policy = PermissionModePolicy("accept-edits")
        self.assertEqual(policy.evaluate(self.request()).action, PolicyAction.ALLOW)
        self.assertEqual(
            policy.evaluate(
                PolicyRequest(
                    "execute_command",
                    ToolRisk.MUTATING,
                    {"argv": ["python", "script.py"]},
                )
            ).action,
            PolicyAction.ASK,
        )

    def test_bypass_allows_explicitly(self):
        decision = PermissionModePolicy("bypass").evaluate(
            self.request("delete", ToolRisk.DESTRUCTIVE)
        )
        self.assertEqual(decision.action, PolicyAction.ALLOW)


class CliTests(unittest.TestCase):
    def test_auto_mode_without_model_preserves_legacy_entry(self):
        environment = os.environ.copy()
        # Empty values intentionally shadow a developer's real .env file.
        environment["MINICLAUDE_MODEL"] = ""
        environment["OPENAI_API_KEY"] = ""
        completed = subprocess.run(
            [sys.executable, "main.py", "compatibility"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("'event': 'task'", completed.stdout)

    def test_explicit_v5_requires_model(self):
        environment = os.environ.copy()
        environment["MINICLAUDE_MODEL"] = ""
        completed = subprocess.run(
            [sys.executable, "main.py", "task", "--mode", "v5"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("requires --model", completed.stderr)


if __name__ == "__main__":
    unittest.main()
