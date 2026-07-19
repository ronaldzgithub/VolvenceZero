"""Tests for epistemic/aleatoric uncertainty separator."""

import unittest

import numpy as np

from volvence_labs.framework.runtime.uncertainty import (
    cross_entropy_per_token,
    epistemic_aleatoric_split,
    softmax,
)


class TestEpistemicAleatoricSplit(unittest.TestCase):
    def test_zero_disagreement_means_zero_epistemic(self):
        """If all N samples agree, epistemic must be 0."""
        rng = np.random.default_rng(0)
        T, V = 5, 10
        single_logits = rng.standard_normal((T, V)).astype(np.float32)
        # Stack identical samples
        samples = np.stack([single_logits] * 4, axis=0)
        epistemic, aleatoric = epistemic_aleatoric_split(samples)
        np.testing.assert_array_almost_equal(epistemic, np.zeros(T), decimal=5)
        # Aleatoric = entropy of single distribution > 0 for non-degenerate logits
        self.assertTrue(np.all(aleatoric > 0))

    def test_high_disagreement_means_high_epistemic(self):
        """If samples disagree a lot, epistemic should be > 0."""
        rng = np.random.default_rng(0)
        N, T, V = 5, 3, 10
        # Each sample peaks at a different vocab index
        samples = np.zeros((N, T, V), dtype=np.float32)
        for n in range(N):
            for t in range(T):
                # Sharp peak at different position per sample
                samples[n, t, n] = 10.0
        epistemic, aleatoric = epistemic_aleatoric_split(samples)
        self.assertTrue(np.all(epistemic > 0.5))
        # Aleatoric is small (each sample is sharp, low entropy)
        self.assertTrue(np.all(aleatoric < 0.1))

    def test_uniform_logits_high_aleatoric_zero_epistemic(self):
        """Uniform predictions = max aleatoric, zero epistemic."""
        N, T, V = 5, 3, 10
        samples = np.zeros((N, T, V), dtype=np.float32)
        epistemic, aleatoric = epistemic_aleatoric_split(samples)
        np.testing.assert_array_almost_equal(epistemic, np.zeros(T), decimal=5)
        max_h = np.log(V)
        np.testing.assert_array_almost_equal(aleatoric, np.full(T, max_h), decimal=5)

    def test_decomposition_is_nonnegative(self):
        """Both components should be >= 0 for arbitrary inputs."""
        rng = np.random.default_rng(42)
        N, T, V = 8, 10, 50
        samples = rng.standard_normal((N, T, V)).astype(np.float32) * 3
        epistemic, aleatoric = epistemic_aleatoric_split(samples)
        self.assertTrue(np.all(epistemic >= 0))
        self.assertTrue(np.all(aleatoric >= 0))

    def test_total_uncertainty_equals_sum(self):
        """H_total = aleatoric + epistemic (within numerical tolerance)."""
        rng = np.random.default_rng(7)
        N, T, V = 4, 5, 20
        samples = rng.standard_normal((N, T, V)).astype(np.float32)
        probs = softmax(samples, axis=-1)
        mean_p = probs.mean(axis=0)
        h_total = -(mean_p * np.log(np.clip(mean_p, 1e-8, None))).sum(axis=-1)
        epistemic, aleatoric = epistemic_aleatoric_split(samples)
        np.testing.assert_array_almost_equal(epistemic + aleatoric, h_total, decimal=5)


class TestCrossEntropy(unittest.TestCase):
    def test_perfect_prediction_zero_ce(self):
        T, V = 4, 10
        logits = np.full((T, V), -100.0, dtype=np.float32)
        targets = np.array([2, 5, 1, 7])
        for t, tgt in enumerate(targets):
            logits[t, tgt] = 100.0
        ce = cross_entropy_per_token(logits, targets)
        np.testing.assert_array_almost_equal(ce, np.zeros(T), decimal=4)

    def test_uniform_logits_log_v(self):
        T, V = 3, 10
        logits = np.zeros((T, V), dtype=np.float32)
        targets = np.array([0, 1, 2])
        ce = cross_entropy_per_token(logits, targets)
        np.testing.assert_array_almost_equal(ce, np.full(T, np.log(V)), decimal=5)


if __name__ == "__main__":
    unittest.main()
