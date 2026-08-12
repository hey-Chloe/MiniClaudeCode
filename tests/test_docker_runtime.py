import tempfile
import unittest
from unittest.mock import patch

from runtime.base import RuntimeErrorBase
from runtime.docker import DockerRuntime


class DockerRuntimeTests(unittest.TestCase):
    def test_missing_docker_is_reported(self):
        with tempfile.TemporaryDirectory() as directory, patch("runtime.docker.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeErrorBase, "not found"):
                DockerRuntime(directory)

    def test_runtime_reports_isolation(self):
        with tempfile.TemporaryDirectory() as directory, patch("runtime.docker.shutil.which", return_value="docker"):
            runtime = DockerRuntime(directory)
            self.assertTrue(runtime.info.isolated)
            self.assertEqual(runtime.info.name, "docker")


if __name__ == "__main__":
    unittest.main()
