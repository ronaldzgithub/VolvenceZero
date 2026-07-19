"""Tests for the gate module: Two-Gate + SGM e-value + aggregator.

4 synthetic scenarios:
1. Clear pass: probe_on consistently beats baseline
2. Hold: marginal improvement, not enough evidence
3. Reject (capacity): too many parameters
4. Reject (Goodhart): counterfactual outperforms probe_on
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import Any, Optional

from volvence_labs.framework.gate import (
    TwoGate,
    SGMGate,
    GateAggregator,
    GateDecision,
    EValue,
)
from volvence_labs.framework.gate.sgm import HoeffdingBound


class TestTwoGate(unittest.TestCase):
    def test_capacity_readonly_probe(self):
        """Read-only probe (0 params) always passes capacity."""
        gate = TwoGate(vc_threshold=100.0)
        result = gate.check_capacity(n_parameters=0, n_samples=100, n_outputs=50)
        self.assertTrue(result.passed)
        self.assertEqual(result.vc_dim_estimate, 0.0)

    def test_capacity_large_model_fails(self):
        """Large model exceeds capacity threshold."""
        gate = TwoGate(vc_threshold=50.0)
        result = gate.check_capacity(n_parameters=1000, n_samples=100, n_outputs=20)
        self.assertFalse(result.passed)
        self.assertGreater(result.vc_dim_estimate, 50.0)

    def test_margin_clear_win(self):
        """Probe clearly beats baseline."""
        gate = TwoGate(margin_delta=0.01, conservative_margin=2.0)
        result = gate.check_margin(
            probe_on_values=[0.9, 0.92, 0.88, 0.91],
            baseline_values=[0.7, 0.72, 0.68, 0.71],
            metric_name="accuracy",
        )
        self.assertTrue(result.passed)
        self.assertGreater(result.margin, 0.15)

    def test_margin_insufficient(self):
        """Probe barely beats baseline — below delta."""
        gate = TwoGate(margin_delta=0.1, conservative_margin=2.0)
        result = gate.check_margin(
            probe_on_values=[0.71, 0.72, 0.70],
            baseline_values=[0.70, 0.69, 0.71],
            metric_name="accuracy",
        )
        self.assertFalse(result.passed)

    def test_margin_negative(self):
        """Probe is worse than baseline."""
        gate = TwoGate(margin_delta=0.01)
        result = gate.check_margin(
            probe_on_values=[0.5, 0.52, 0.48],
            baseline_values=[0.7, 0.72, 0.68],
            metric_name="accuracy",
        )
        self.assertFalse(result.passed)
        self.assertLess(result.margin, 0)

    def test_margin_empty_data(self):
        """Empty data fails gracefully."""
        gate = TwoGate()
        result = gate.check_margin([], [0.7], metric_name="x")
        self.assertFalse(result.passed)


class TestSGMGate(unittest.TestCase):
    def test_strong_signal_approves(self):
        """Consistent strong signal should approve with enough observations."""
        # delta=0.1, sigma=0.2 → lambda=2.5, strong signal of 0.2 per step
        sgm = SGMGate(delta=0.1, sigma=0.2, alpha=0.05)
        # 8 observations where probe is consistently 0.2 better
        for _ in range(8):
            sgm.update(0.9, 0.7)
        self.assertEqual(sgm.decision(), "approve")
        self.assertGreater(sgm.e_value.value, 20.0)  # > 1/0.05

    def test_no_signal_holds(self):
        """No difference should not approve."""
        sgm = SGMGate(delta=0.05, sigma=0.5, alpha=0.05)
        for _ in range(10):
            sgm.update(0.7, 0.7)
        self.assertEqual(sgm.decision(), "hold")
        # e-value should be around 1 or below
        self.assertLess(sgm.e_value.value, 20.0)

    def test_negative_signal_shrinks(self):
        """Probe worse than baseline should shrink e-value."""
        sgm = SGMGate(delta=0.05, sigma=0.5, alpha=0.05)
        for _ in range(10):
            sgm.update(0.5, 0.7)  # probe is worse
        self.assertLess(sgm.e_value.value, 1.0)

    def test_reset(self):
        """Reset brings e-value back to 1."""
        sgm = SGMGate(delta=0.05, sigma=0.5)
        sgm.update(0.9, 0.7)
        sgm.reset()
        self.assertEqual(sgm.e_value.value, 1.0)
        self.assertEqual(sgm.e_value.n_updates, 0)

    def test_batch_update(self):
        """Batch update equivalent to sequential."""
        sgm1 = SGMGate(delta=0.05, sigma=0.5)
        sgm2 = SGMGate(delta=0.05, sigma=0.5)

        probes = [0.9, 0.85, 0.92, 0.88]
        bases = [0.7, 0.72, 0.68, 0.71]

        for p, b in zip(probes, bases):
            sgm1.update(p, b)
        sgm2.update_batch(probes, bases)

        self.assertAlmostEqual(sgm1.e_value.log_value, sgm2.e_value.log_value, places=10)


class TestHoeffdingBound(unittest.TestCase):
    def test_confidence_shrinks_with_n(self):
        """More observations → tighter confidence."""
        hb = HoeffdingBound(range_bound=1.0, alpha=0.05)
        widths = []
        for i in range(20):
            hb.update(0.8, 0.6)
            widths.append(hb.confidence_width())
        # Width should be monotonically decreasing
        for i in range(1, len(widths)):
            self.assertLess(widths[i], widths[i - 1])

    def test_significance_with_strong_signal(self):
        """Strong consistent signal becomes significant."""
        hb = HoeffdingBound(range_bound=1.0, alpha=0.05)
        for _ in range(50):
            hb.update(0.9, 0.5)
        self.assertTrue(hb.is_significant(0.0))
        self.assertGreater(hb.lower_bound(), 0.0)


class TestGateAggregator(unittest.TestCase):
    """Integration tests for the full aggregator."""

    @dataclass
    class FakeUnit:
        ok: bool
        run_id: str
        cell: str
        metrics: dict
        seed: int = 0

    def test_clear_pass(self):
        """Probe clearly better → approve."""
        agg = GateAggregator(
            vc_threshold=100.0,
            margin_delta=0.01,
            sgm_delta=0.1,
            sgm_sigma=0.2,
            sgm_alpha=0.05,
            conservative_margin=1.0,
        )
        units = []
        for i in range(8):
            units.append(self.FakeUnit(ok=True, run_id=f"run_p_{i}", cell="probe_on", metrics={"accuracy": 0.9 + i * 0.01}))
            units.append(self.FakeUnit(ok=True, run_id=f"run_b_{i}", cell="baseline", metrics={"accuracy": 0.6 + i * 0.01}))

        decision = agg.evaluate(
            "test-probe",
            units,
            metric_name="accuracy",
            n_parameters=0,
        )
        self.assertEqual(decision.decision, GateDecision.APPROVE)
        self.assertIn("all gates pass", decision.reason)

    def test_hold_insufficient_evidence(self):
        """Marginal improvement with few seeds → hold."""
        agg = GateAggregator(
            margin_delta=0.1,
            sgm_delta=0.1,
            sgm_sigma=1.0,
            conservative_margin=2.0,
        )
        units = [
            self.FakeUnit(ok=True, run_id="p0", cell="probe_on", metrics={"accuracy": 0.72}),
            self.FakeUnit(ok=True, run_id="b0", cell="baseline", metrics={"accuracy": 0.70}),
        ]
        decision = agg.evaluate("test-probe", units, metric_name="accuracy", n_parameters=0)
        # Margin is 0.02 < 0.1*2.0 = 0.2, so hold
        self.assertIn(decision.decision, (GateDecision.HOLD, GateDecision.REJECT))

    def test_reject_capacity(self):
        """Too many parameters → reject."""
        agg = GateAggregator(vc_threshold=10.0)
        # Need enough samples/outputs so VC estimate exceeds threshold
        units = [
            self.FakeUnit(ok=True, run_id=f"p{i}", cell="probe_on", metrics={"accuracy": 0.95})
            for i in range(50)
        ] + [
            self.FakeUnit(ok=True, run_id=f"b{i}", cell="baseline", metrics={"accuracy": 0.5})
            for i in range(50)
        ]
        decision = agg.evaluate(
            "test-probe", units, metric_name="accuracy", n_parameters=10000
        )
        self.assertEqual(decision.decision, GateDecision.REJECT)
        self.assertIn("capacity", decision.reason)

    def test_reject_goodhart(self):
        """Counterfactual outperforms probe_on → Goodhart reject."""
        agg = GateAggregator(
            vc_threshold=100.0,
            margin_delta=0.01,
            conservative_margin=1.0,
        )
        units = [
            self.FakeUnit(ok=True, run_id="p0", cell="probe_on", metrics={"accuracy": 0.7}),
            self.FakeUnit(ok=True, run_id="b0", cell="baseline", metrics={"accuracy": 0.5}),
            # Counterfactual is suspiciously better than probe_on
            self.FakeUnit(ok=True, run_id="cf0", cell="counterfactual", metrics={"accuracy": 0.95}),
        ]
        decision = agg.evaluate("test-probe", units, metric_name="accuracy", n_parameters=0)
        self.assertEqual(decision.decision, GateDecision.REJECT)
        self.assertIn("Goodhart", decision.reason)

    def test_reject_negative_margin(self):
        """Probe worse than baseline → reject."""
        agg = GateAggregator(conservative_margin=1.0)
        units = [
            self.FakeUnit(ok=True, run_id="p0", cell="probe_on", metrics={"accuracy": 0.4}),
            self.FakeUnit(ok=True, run_id="b0", cell="baseline", metrics={"accuracy": 0.7}),
        ]
        decision = agg.evaluate("test-probe", units, metric_name="accuracy", n_parameters=0)
        self.assertEqual(decision.decision, GateDecision.REJECT)
        self.assertIn("negative margin", decision.reason)


if __name__ == "__main__":
    unittest.main()
