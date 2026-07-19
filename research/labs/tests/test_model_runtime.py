"""Test ModelRuntime: TinyLlama load + forward pass + feature cache.

NOTE: This test downloads TinyLlama (~2.2GB) on first run. Subsequent runs
use the HF cache. Set SKIP_MODEL_TESTS=1 to skip in CI.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

# Skip if explicitly requested (CI without GPU/model access)
SKIP = os.environ.get("SKIP_MODEL_TESTS", "0") == "1"


@unittest.skipIf(SKIP, "SKIP_MODEL_TESTS=1")
class TestModelRuntime(unittest.TestCase):
    """Integration tests for ModelRuntime with a tiny test model.

    Uses sshleifer/tiny-gpt2 (~2MB) for fast CI. Production probes use TinyLlama.
    """

    TEST_MODEL = "sshleifer/tiny-gpt2"

    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        import torch
        from volvence_labs.framework.runtime import ModelRuntime
        cls.rt = ModelRuntime(cls.TEST_MODEL, dtype="fp32")
        cls.rt.load_model()

    @classmethod
    def tearDownClass(cls):
        cls.rt.unload()

    def test_model_loads_and_has_sha(self):
        self.assertIsNotNone(self.rt.model)
        self.assertIsNotNone(self.rt.tokenizer)
        self.assertTrue(len(self.rt.model_sha) == 64)

    def test_forward_lm_produces_logits(self):
        import torch
        text = "The capital of France is"
        encoded = self.rt.tokenizer(text, return_tensors="pt")
        result = self.rt.forward_lm(encoded["input_ids"], return_hidden_states=True)

        self.assertIn("logits", result)
        self.assertIn("hidden_states", result)
        # logits shape: (1, seq_len, vocab_size)
        self.assertEqual(result["logits"].dim(), 3)
        self.assertEqual(result["logits"].shape[0], 1)

    def test_get_logits_for_text(self):
        result = self.rt.get_logits_for_text("Hello world, this is a test.")
        self.assertIn("logits", result)
        self.assertIn("hidden_states", result)
        self.assertIn("tokens", result)
        # logits: (seq_len, vocab)
        self.assertEqual(result["logits"].dim(), 2)

    def test_encode_text_batch(self):
        texts = ["Hello world", "This is a test", "Another sentence here"]
        result = self.rt.encode_text(texts)
        self.assertIn("embeddings", result)
        # embeddings: (3, hidden_dim)
        self.assertEqual(result["embeddings"].shape[0], 3)

    def test_deterministic_forward(self):
        """Same input produces same output (fp32 determinism)."""
        import torch
        text = "Determinism test: 1 + 1 ="
        r1 = self.rt.get_logits_for_text(text)
        r2 = self.rt.get_logits_for_text(text)
        # On CPU with eval mode, should be identical
        self.assertTrue(
            torch.allclose(r1["logits"], r2["logits"], atol=1e-5),
            "Forward pass not deterministic within epsilon",
        )


@unittest.skipIf(SKIP, "SKIP_MODEL_TESTS=1")
class TestFeatureCache(unittest.TestCase):
    """Test feature cache disk persistence."""

    def test_put_get_roundtrip(self):
        from volvence_labs.framework.runtime.cache import FeatureCache
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["VOLVENCE_LABS_ROOT"] = tmp
            cache = FeatureCache("model_sha_test", "dataset_sha_test", root=tmp)

            features = np.random.randn(10, 384).astype(np.float32)
            cache.put("val", 0, features)

            loaded = cache.get("val", 0)
            self.assertIsNotNone(loaded)
            np.testing.assert_array_equal(features, loaded)

            # Miss
            self.assertIsNone(cache.get("val", 999))

            # Count
            self.assertEqual(cache.count("val"), 1)

            # Clear
            cleared = cache.clear("val")
            self.assertEqual(cleared, 1)
            self.assertEqual(cache.count("val"), 0)

            os.environ.pop("VOLVENCE_LABS_ROOT", None)

    def test_batch_operations(self):
        from volvence_labs.framework.runtime.cache import FeatureCache
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["VOLVENCE_LABS_ROOT"] = tmp
            cache = FeatureCache("m", "d", root=tmp)

            batch = {i: np.random.randn(384).astype(np.float32) for i in range(5)}
            cache.put_batch("train", batch)

            loaded = cache.get_batch("train", [0, 1, 2, 3, 4, 99])
            for i in range(5):
                np.testing.assert_array_equal(batch[i], loaded[i])
            self.assertIsNone(loaded[99])

            os.environ.pop("VOLVENCE_LABS_ROOT", None)


if __name__ == "__main__":
    unittest.main()
