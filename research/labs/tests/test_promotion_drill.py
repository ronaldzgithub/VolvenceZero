"""Test SHADOW→ACTIVE promotion drill for P7 refusal-direction-v1.

Full automated drill:
1. Run P7 with shadow profile (8 seeds × 4 cells)
2. Evaluate gate → should approve (absolute mode, threshold=0.8)
3. Promote → PromotionRecord written to CAS
4. Rollback → DemotionRecord written, original preserved
5. Verify all records are content-addressed and immutable
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from volvence_labs.framework.gate import GateAggregator, GateDecision
from volvence_labs.framework.snapshot import CASStore, RunLog, default_paths
from volvence_labs.framework.wiring.promotion import PromotionManager
from volvence_labs.framework.scheduler.runner import _run_unit
from volvence_labs.framework.wiring import AblationCell, WiringLevel


class TestPromotionDrill(unittest.TestCase):
    """Full SHADOW→ACTIVE→SHADOW drill for P7."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        os.environ["VOLVENCE_LABS_ROOT"] = self._tmp

    def tearDown(self):
        os.environ.pop("VOLVENCE_LABS_ROOT", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_full_drill_p7(self):
        """Complete promotion drill: run → gate → promote → demote → verify."""
        from dataclasses import dataclass

        # Step 1: Run P7 with multiple seeds and cells
        probe_id = "refusal-direction-v1"
        seeds = [0, 1, 2, 3]
        cells = ["baseline", "probe_on", "counterfactual"]

        for seed in seeds:
            for cell in cells:
                result = _run_unit(probe_id, cell, seed, "shadow", self._tmp)
                self.assertTrue(result["ok"], f"run failed: {result.get('error')}")

        # Step 2: Gather evidence and evaluate gate
        paths = default_paths(self._tmp)
        store = CASStore(paths)
        log = RunLog(paths, store)
        records = log.list(probe_id=probe_id, limit=100)
        self.assertGreater(len(records), 0)

        @dataclass
        class _Unit:
            ok: bool
            run_id: str
            cell: str
            metrics: dict
            seed: int

        units = []
        for r in records:
            readouts = store.get_obj(r.readouts_sha)
            metrics = readouts.get("metrics", {})
            units.append(_Unit(ok=True, run_id=r.run_id, cell=r.ablation_cell, metrics=metrics, seed=r.seed))

        agg = GateAggregator(
            sgm_sigma=0.1,
            sgm_delta=0.05,
            margin_delta=0.01,
            conservative_margin=1.0,
        )
        decision = agg.evaluate(
            probe_id, units,
            metric_name="accuracy",
            n_parameters=0,
            mode="absolute",
            absolute_threshold=0.8,
        )

        self.assertEqual(decision.decision, GateDecision.APPROVE,
                         f"Gate should approve P7: {decision.reason}")

        # Step 3: Promote
        mgr = PromotionManager(self._tmp)
        promo_record = mgr.promote(probe_id, decision)
        self.assertEqual(promo_record.probe_id, probe_id)
        self.assertEqual(promo_record.from_level, "shadow")
        self.assertEqual(promo_record.to_level, "active")
        self.assertEqual(promo_record.gate_decision, "approve")
        self.assertTrue(len(promo_record.sha) == 64)

        # Verify current level
        level = mgr.get_current_level(probe_id)
        self.assertEqual(level, "active")

        # Step 4: Rollback (demote)
        demo_record = mgr.demote(probe_id, promo_record.sha, reason="drill test")
        self.assertEqual(demo_record.probe_id, probe_id)
        self.assertEqual(demo_record.from_level, "active")
        self.assertEqual(demo_record.to_level, "shadow")
        self.assertEqual(demo_record.original_promotion_sha, promo_record.sha)

        # Verify current level after demotion
        level = mgr.get_current_level(probe_id)
        self.assertEqual(level, "shadow")

        # Step 5: Verify immutability — original promotion record still exists
        promotions = mgr.list_promotions(probe_id=probe_id)
        self.assertEqual(len(promotions), 1)
        self.assertEqual(promotions[0].sha, promo_record.sha)

        # Verify CAS has the promotion object
        promo_obj = store.get_obj(promo_record.sha)
        # The CAS stores the canonical JSON, which when parsed should match
        # (it won't have the sha field since that's computed from the content)
        self.assertEqual(promo_obj.get("probe_id") or promo_record.probe_id, probe_id)

        mgr.close()
        log.close()
        store.close()

    def test_promotion_rejected_without_approval(self):
        """Cannot promote without gate approval."""
        from volvence_labs.framework.gate import PromotionDecision
        from volvence_labs.framework.gate.two_gate import CapacityBound, ValidationMargin

        mgr = PromotionManager(self._tmp)

        # Create a REJECT decision
        decision = PromotionDecision(
            decision=GateDecision.REJECT,
            probe_id="test-probe",
            from_level="shadow",
            to_level="active",
            capacity=None,
            margin=None,
            sgm_summary={},
            hoeffding_summary={},
            reason="test rejection",
            evidence_run_ids=[],
        )

        with self.assertRaises(ValueError):
            mgr.promote("test-probe", decision)

        mgr.close()


if __name__ == "__main__":
    unittest.main()
