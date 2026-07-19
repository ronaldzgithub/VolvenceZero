"""Stress test: CursorRunner (subprocess fallback) under concurrent load.

Verifies that CAS/SQLite don't corrupt under 8 concurrent subprocess units,
and that output_sha is consistent across runs.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import volvence_labs.probes  # noqa: F401
from volvence_labs.framework.parallel import CursorRunner
from volvence_labs.framework.parallel.cursor_runner import CursorRunnerConfig
from volvence_labs.framework.snapshot import CASStore, RunLog, default_paths
from volvence_labs.framework.wiring import get_profile


class TestCursorRunnerStress(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ["VOLVENCE_LABS_ROOT"] = self.tmp.name
        self.addCleanup(lambda: os.environ.pop("VOLVENCE_LABS_ROOT", None))

    def test_concurrent_8_units_no_corruption(self) -> None:
        """8 concurrent units (4 seeds x 2 cells) must all succeed."""
        config = CursorRunnerConfig(max_concurrent=8)
        runner = CursorRunner(root=self.tmp.name, config=config)
        profile = get_profile("canary")  # 4 seeds x 2 cells = 8 units
        report = runner.run("pe-baseline-v0", profile)

        failed = [u for u in report.units if not u.ok]
        self.assertEqual(
            len(failed), 0,
            f"units failed: {[(u.cell, u.seed, u.error) for u in failed]}",
        )
        self.assertEqual(len(report.units), 8)

    def test_output_sha_consistent_across_runs(self) -> None:
        """Same (probe, cell, seed) must produce same output_sha."""
        config = CursorRunnerConfig(max_concurrent=4)
        runner = CursorRunner(root=self.tmp.name, config=config)
        profile = get_profile("dev")  # 1 seed x 2 cells

        report1 = runner.run("pe-baseline-v0", profile)
        report2 = runner.run("pe-baseline-v0", profile)

        def key(u):
            return (u.cell, u.seed)

        map1 = {key(u): u for u in report1.units if u.ok}
        map2 = {key(u): u for u in report2.units if u.ok}

        self.assertEqual(set(map1.keys()), set(map2.keys()))
        for k in map1:
            self.assertEqual(
                map1[k].input_sha, map2[k].input_sha,
                f"input_sha diverged for {k}",
            )
            self.assertEqual(
                map1[k].output_sha, map2[k].output_sha,
                f"output_sha diverged for {k}",
            )
            self.assertEqual(
                map1[k].readouts_sha, map2[k].readouts_sha,
                f"readouts_sha diverged for {k}",
            )

    def test_sqlite_integrity_after_stress(self) -> None:
        """After 8 concurrent writes, SQLite DB passes integrity check."""
        config = CursorRunnerConfig(max_concurrent=8)
        runner = CursorRunner(root=self.tmp.name, config=config)
        profile = get_profile("canary")
        runner.run("pe-baseline-v0", profile)

        paths = default_paths(self.tmp.name)
        store = CASStore(paths)
        log = RunLog(paths, store)

        records = log.list(limit=100)
        self.assertEqual(len(records), 8)

        # Verify all records have valid shas in CAS
        for r in records:
            self.assertTrue(store.exists(r.manifest_sha), f"missing manifest for {r.run_id}")
            self.assertTrue(store.exists(r.readouts_sha), f"missing readouts for {r.run_id}")
            self.assertTrue(store.exists(r.output_sha), f"missing output for {r.run_id}")

        log.close()
        store.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
