"""Tests for Modal cloud runner scaffolding (no actual Modal calls)."""

import unittest
from unittest.mock import patch

from volvence_labs.framework.parallel import ModalRunner
from volvence_labs.framework.parallel.base import CloudRunnerNotConfigured


class TestModalRunnerSetup(unittest.TestCase):
    def test_setup_raises_when_modal_missing(self):
        """Without modal installed, setup() must raise with install hint."""
        with patch.dict("sys.modules", {"modal": None}):
            runner = ModalRunner()
            with self.assertRaises(CloudRunnerNotConfigured) as ctx:
                runner.setup()
            self.assertIn("pip install modal", str(ctx.exception))

    def test_submit_unit_raises_without_setup(self):
        """submit_unit() with no setup() should also raise (auto-setup tries first)."""
        with patch.dict("sys.modules", {"modal": None}):
            runner = ModalRunner()
            with self.assertRaises(CloudRunnerNotConfigured):
                runner.submit_unit("refusal-direction-v1", "baseline", 0, "shadow")


class TestModalAppDoesNotCrashOnImport(unittest.TestCase):
    def test_modal_app_module_imports(self):
        """The modal_app module must import without raising even when modal is absent."""
        from volvence_labs.framework.parallel import modal_app
        self.assertTrue(hasattr(modal_app, "app"))
        self.assertTrue(hasattr(modal_app, "run_unit"))


if __name__ == "__main__":
    unittest.main()
