"""Tests for Prometheus metrics exporter wiring in _build_report."""

import unittest

from volvence_labs.framework.readout.metrics_exporter import (
    GATE_DECISION,
    METRIC_VALUE,
    RUNS_TOTAL,
)
from volvence_labs.framework.scheduler.runner import run_experiment
from volvence_labs.framework.wiring import get_profile
from volvence_labs.probes import (  # noqa: F401  registers probes
    primitive_7_readonly_monitoring,
)


def _counter_total() -> float:
    """Sum all RUNS_TOTAL samples."""
    total = 0.0
    for sample in RUNS_TOTAL.collect()[0].samples:
        if sample.name.endswith("_total"):
            total += sample.value
    return total


def _gate_total() -> float:
    total = 0.0
    for sample in GATE_DECISION.collect()[0].samples:
        total += sample.value
    return total


class TestRunnerWiresPrometheus(unittest.TestCase):
    def test_run_experiment_increments_runs_counter(self):
        """After running 2 units, RUNS_TOTAL should grow by exactly 2."""
        before = _counter_total()
        report = run_experiment("refusal-direction-v1", get_profile("dev"))
        after = _counter_total()
        self.assertEqual(len(report.units), 2)
        self.assertAlmostEqual(after - before, 2.0)

    def test_run_experiment_records_gate_decision(self):
        """A completed run should set the gate decision gauge."""
        run_experiment("refusal-direction-v1", get_profile("dev"))
        labels_set = {
            sample.labels.get("probe_id"): sample.value
            for sample in GATE_DECISION.collect()[0].samples
        }
        self.assertIn("refusal-direction-v1", labels_set)

    def test_metric_value_populated(self):
        """At least one METRIC_VALUE gauge should be set after a run."""
        run_experiment("refusal-direction-v1", get_profile("dev"))
        nonzero = [
            sample
            for sample in METRIC_VALUE.collect()[0].samples
            if sample.value != 0
        ]
        self.assertGreater(len(nonzero), 0)


if __name__ == "__main__":
    unittest.main()
