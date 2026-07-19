"""Tests for process-level ModelRuntime cache."""

import unittest

from volvence_labs.framework.runtime import (
    ModelRuntime,
    cache_size,
    clear_model_cache,
    get_model_runtime,
)


class TestModelCache(unittest.TestCase):
    def setUp(self):
        clear_model_cache()

    def tearDown(self):
        clear_model_cache()

    def test_same_args_returns_same_instance(self):
        rt1 = get_model_runtime("sshleifer/tiny-gpt2", dtype="fp32")
        rt2 = get_model_runtime("sshleifer/tiny-gpt2", dtype="fp32")
        self.assertIs(rt1, rt2)
        self.assertEqual(cache_size(), 1)

    def test_different_args_returns_different_instances(self):
        rt1 = get_model_runtime("sshleifer/tiny-gpt2", dtype="fp32")
        rt2 = get_model_runtime("sshleifer/tiny-gpt2", dtype="fp16")
        self.assertIsNot(rt1, rt2)
        self.assertEqual(cache_size(), 2)

    def test_clear_cache(self):
        get_model_runtime("sshleifer/tiny-gpt2", dtype="fp32")
        self.assertEqual(cache_size(), 1)
        clear_model_cache()
        self.assertEqual(cache_size(), 0)

    def test_returns_model_runtime_instance(self):
        rt = get_model_runtime("sshleifer/tiny-gpt2", dtype="fp32")
        self.assertIsInstance(rt, ModelRuntime)

    def test_load_only_happens_once(self):
        """If two callers get the same runtime, load_model() is idempotent."""
        rt1 = get_model_runtime("sshleifer/tiny-gpt2", dtype="fp32")
        rt1.load_model()
        first_model = rt1._model

        rt2 = get_model_runtime("sshleifer/tiny-gpt2", dtype="fp32")
        rt2.load_model()  # already loaded — should be no-op
        self.assertIs(rt2._model, first_model)


if __name__ == "__main__":
    unittest.main()
