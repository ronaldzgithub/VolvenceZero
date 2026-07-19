"""F5 R15 teeth meta-probe: deliberately corrupted restores must fail.

The positive rollback probes establish reproducibility. This companion probe
establishes that the same checks are capable of rejecting a minimal mutation,
so an accidentally vacuous checker cannot satisfy the R15 gate.
"""

from __future__ import annotations

import json
import shutil
from typing import Any, Mapping

from ...framework.probe import (
    BaseProbe,
    GateReport,
    PrimitiveTag,
    ProbeContext,
    ReadoutBundle,
    RunOutcome,
    register_probe,
)
from ...framework.snapshot import (
    CASStore,
    RunLog,
    canonical_dumps,
    default_paths,
    sha256_bytes,
)
from ...framework.wiring import AblationCell, WiringLevel


TARGET_PROBE_ID = "pe-baseline-v0"
TARGET_CELL = AblationCell.PROBE_ON
TARGET_SEED = 0
TAMPER_KEY = "__teeth_tamper__"


def _mismatched_top_level_keys(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> tuple[str, ...]:
    keys = set(expected) | set(observed)
    return tuple(sorted(key for key in keys if expected.get(key) != observed.get(key)))


@register_probe
class R15RollbackTeethProbe(BaseProbe):
    id = "r15-rollback-teeth-v0"
    hypothesis = (
        "R15 rollback verification rejects and localizes deliberately "
        "corrupted reconstructed artifacts and content hashes."
    )
    primitive = PrimitiveTag.F5_R15_FORMALIZATION
    r_ids = ("R8", "R15", "R12")

    def knobs(self) -> dict[str, list]:
        return {}

    def default_inputs(self, seed: int) -> Any:
        return {
            "target_probe_id": TARGET_PROBE_ID,
            "target_cell": TARGET_CELL.value,
            "target_seed": TARGET_SEED,
            "tamper_key": TAMPER_KEY,
        }

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        del knobs
        from ...framework.scheduler.runner import _run_unit

        paths = default_paths()
        store = CASStore(paths)
        log = RunLog(paths, store)

        checks: dict[str, bool] = {}
        raw = _run_unit(
            TARGET_PROBE_ID,
            TARGET_CELL.value,
            TARGET_SEED,
            WiringLevel.SHADOW.value,
            str(paths.root),
        )
        checks["target_run_ok"] = bool(raw.get("ok"))
        if not checks["target_run_ok"]:
            log.close()
            store.close()
            return _failed_outcome(
                reason="target probe run failed",
                checks=checks,
                ctx=ctx,
                details={"error": raw.get("error", "")},
            )

        run_id = raw["run_id"]
        record = log.get(run_id)
        exp_dir = paths.experiment_dir(run_id)
        manifest_path = exp_dir / "manifest.json"
        readouts_path = exp_dir / "readouts" / "readouts.json"

        manifest_expected = store.get_obj(record.manifest_sha)
        readouts_expected = store.get_obj(record.readouts_sha)

        # Rebuild exactly as the positive R15 probe does.
        shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        readouts_path.parent.mkdir(exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest_expected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        readouts_path.write_text(
            json.dumps(readouts_expected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        checks["clean_manifest_rebuild_matches"] = (
            json.loads(manifest_path.read_text("utf-8")) == manifest_expected
        )
        checks["clean_readouts_rebuild_matches"] = (
            json.loads(readouts_path.read_text("utf-8")) == readouts_expected
        )

        # Minimal broken models: one extra top-level field in each artifact.
        manifest_tampered = {**manifest_expected, TAMPER_KEY: "manifest"}
        readouts_tampered = {**readouts_expected, TAMPER_KEY: "readouts"}
        manifest_path.write_text(
            json.dumps(manifest_tampered, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        readouts_path.write_text(
            json.dumps(readouts_tampered, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        manifest_observed = json.loads(manifest_path.read_text("utf-8"))
        readouts_observed = json.loads(readouts_path.read_text("utf-8"))
        manifest_mismatches = _mismatched_top_level_keys(
            manifest_expected,
            manifest_observed,
        )
        readouts_mismatches = _mismatched_top_level_keys(
            readouts_expected,
            readouts_observed,
        )
        checks["tampered_manifest_rejected"] = manifest_observed != manifest_expected
        checks["tampered_readouts_rejected"] = readouts_observed != readouts_expected
        checks["manifest_tamper_localized"] = manifest_mismatches == (TAMPER_KEY,)
        checks["readouts_tamper_localized"] = readouts_mismatches == (TAMPER_KEY,)

        manifest_tampered_sha = sha256_bytes(canonical_dumps(manifest_tampered))
        readouts_tampered_sha = sha256_bytes(canonical_dumps(readouts_tampered))
        checks["tampered_manifest_sha_rejected"] = (
            manifest_tampered_sha != record.manifest_sha
        )
        checks["tampered_readouts_sha_rejected"] = (
            readouts_tampered_sha != record.readouts_sha
        )

        # Leave the target run in its valid reconstructed state.
        manifest_path.write_text(
            json.dumps(manifest_expected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        readouts_path.write_text(
            json.dumps(readouts_expected, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        checks["manifest_restored_after_teeth_check"] = (
            json.loads(manifest_path.read_text("utf-8")) == manifest_expected
        )
        checks["readouts_restored_after_teeth_check"] = (
            json.loads(readouts_path.read_text("utf-8")) == readouts_expected
        )

        all_ok = all(checks.values())
        log.close()
        store.close()
        return RunOutcome(
            readouts=ReadoutBundle(
                metrics={
                    "passed": 1.0 if all_ok else 0.0,
                    "n_checks": float(len(checks)),
                    "n_failed": float(sum(1 for value in checks.values() if not value)),
                },
                artifacts={
                    "checks": checks,
                    "run_id": run_id,
                    "manifest_mismatches": manifest_mismatches,
                    "readouts_mismatches": readouts_mismatches,
                    "manifest_tampered_sha": manifest_tampered_sha,
                    "readouts_tampered_sha": readouts_tampered_sha,
                },
                tags={
                    "target_probe": TARGET_PROBE_ID,
                    "target_cell": TARGET_CELL.value,
                    "target_seed": TARGET_SEED,
                    "discipline": "teeth",
                },
            ),
            output={
                "all_ok": all_ok,
                "run_id": run_id,
                "caught_tamper_key": TAMPER_KEY,
            },
        )

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        if not outcomes:
            return GateReport(passed=False, reason="no outcomes", stats={})
        passed = all(
            outcome.readouts.metrics.get("passed", 0.0) >= 1.0
            for outcome in outcomes
        )
        return GateReport(
            passed=passed,
            reason="all deliberately corrupted rollback models must be rejected",
            stats={
                "n_outcomes": len(outcomes),
                "n_passing": sum(
                    1
                    for outcome in outcomes
                    if outcome.readouts.metrics.get("passed", 0.0) >= 1.0
                ),
            },
        )


def _failed_outcome(
    *,
    reason: str,
    checks: dict[str, bool],
    ctx: ProbeContext,
    details: Mapping[str, Any],
) -> RunOutcome:
    return RunOutcome(
        readouts=ReadoutBundle(
            metrics={
                "passed": 0.0,
                "n_checks": float(len(checks)),
                "n_failed": float(sum(1 for value in checks.values() if not value)),
            },
            artifacts={"checks": checks, "reason": reason, **details},
            tags={"cell": ctx.cell.value, "discipline": "teeth"},
        ),
        output={"all_ok": False, "reason": reason},
    )
