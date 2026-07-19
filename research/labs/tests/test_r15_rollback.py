"""R15 rollback: 端到端跑 pe-baseline-v0 + r15-rollback-v0，验证 gate 通过。"""

from __future__ import annotations

import os
import tempfile
import unittest

# Importing probes triggers registration.
import volvence_labs.probes  # noqa: F401
from volvence_labs.framework.scheduler import run_experiment
from volvence_labs.framework.wiring import get_profile


class TestR15Rollback(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ["VOLVENCE_LABS_ROOT"] = self.tmp.name
        self.addCleanup(lambda: os.environ.pop("VOLVENCE_LABS_ROOT", None))

    def test_smoke_probe_runs(self) -> None:
        report = run_experiment(
            "pe-baseline-v0",
            get_profile("dev"),
            parallel=False,
            root=self.tmp.name,
        )
        for unit in report.units:
            self.assertTrue(unit.ok, f"unit failed: {unit.error}")
            self.assertIn("mean_pe", unit.metrics)
        self.assertIsNotNone(report.gate)
        self.assertTrue(report.gate.passed, f"gate failed: {report.gate.reason}")

    def test_r15_rollback_probe_passes(self) -> None:
        report = run_experiment(
            "r15-rollback-v0",
            get_profile("dev"),
            parallel=False,
            root=self.tmp.name,
        )
        for unit in report.units:
            self.assertTrue(unit.ok, f"unit failed: {unit.error}")
            self.assertEqual(
                unit.metrics.get("passed"),
                1.0,
                f"rollback failed: {unit.readouts.get('artifacts', {})}",
            )
        self.assertIsNotNone(report.gate)
        self.assertTrue(report.gate.passed, f"r15 gate failed: {report.gate.reason}")

    def test_parallel_runner_matches_sequential(self) -> None:
        """Parallel 跑同一 probe 不能破坏 bit-exact 性（同一输入 → 同一 output sha）。"""
        seq = run_experiment(
            "pe-baseline-v0",
            get_profile("dev"),
            parallel=False,
            root=self.tmp.name,
        )
        par = run_experiment(
            "pe-baseline-v0",
            get_profile("dev"),
            parallel=True,
            root=self.tmp.name,
        )
        # 同 cell 同 seed 的 output_sha / input_sha 应一致。
        def key(u):
            return (u.cell, u.seed)

        seq_map = {key(u): u for u in seq.units if u.ok}
        par_map = {key(u): u for u in par.units if u.ok}
        self.assertEqual(set(seq_map), set(par_map))
        for k, seq_u in seq_map.items():
            par_u = par_map[k]
            self.assertEqual(seq_u.input_sha, par_u.input_sha, f"input_sha diverged for {k}")
            self.assertEqual(seq_u.output_sha, par_u.output_sha, f"output_sha diverged for {k}")
            self.assertEqual(
                seq_u.readouts_sha,
                par_u.readouts_sha,
                f"readouts_sha diverged for {k}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
