"""F5 R15 meta-probe v1: ε-tolerance rollback for real hooks.

Extends r15_rollback_v0 to handle real model weights + HF datasets where:
- model_sha / dataset_sha must be bit-exact across reruns (same identity)
- Numeric readouts (float metrics) use ε-tolerance: abs(a - b) < 1e-5
- Token-level outputs (integer ids) remain bit-exact
- Logical state is bit-exact; only floating-point non-determinism is tolerated

This is the necessary generalization of R15 for stage 1+ where fp16/bf16
non-determinism in CUDA kernels makes strict bit-exact impossible for floats.
"""

from __future__ import annotations

import json
import math
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
from ...framework.snapshot import CASStore, RunLog, default_paths
from ...framework.wiring import AblationCell, WiringLevel


TARGET_PROBE_ID = "pe-curiosity-critic-v1"
TARGET_CELL = AblationCell.PROBE_ON
TARGET_SEED = 42
EPSILON = 1e-5


def _floats_close(a: Any, b: Any, eps: float = EPSILON) -> bool:
    """Recursively compare two JSON-like structures with ε-tolerance for floats."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) < eps
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, bool) and isinstance(b, bool):
        return a == b
    if a is None and b is None:
        return True
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_floats_close(x, y, eps) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_floats_close(a[k], b[k], eps) for k in a)
    return a == b


def _count_float_diffs(a: Any, b: Any, eps: float = EPSILON) -> int:
    """Count number of float values that differ beyond epsilon."""
    if isinstance(a, float) and isinstance(b, float):
        return 0 if abs(a - b) < eps else 1
    if isinstance(a, list) and isinstance(b, list):
        # Lenient by design: this is a diagnostic counter, so a length
        # mismatch just means the overlapping prefix is compared.
        return sum(_count_float_diffs(x, y, eps) for x, y in zip(a, b, strict=False))
    if isinstance(a, dict) and isinstance(b, dict):
        return sum(_count_float_diffs(a[k], b[k], eps) for k in a if k in b)
    return 0


@register_probe
class R15RollbackV1Probe(BaseProbe):
    id = "r15-rollback-v1"
    hypothesis = (
        "Real-hooks experiments can be rolled back from CAS+RunLog with ε-tolerance "
        "for float metrics and bit-exact for logical state (model_sha, token ids)."
    )
    primitive = PrimitiveTag.F5_R15_FORMALIZATION
    r_ids = ("R8", "R15", "R12")

    def knobs(self) -> dict[str, list]:
        return {"epsilon": [1e-5, 1e-4]}

    def default_inputs(self, seed: int) -> Any:
        return {
            "target_probe_id": TARGET_PROBE_ID,
            "target_cell": TARGET_CELL.value,
            "target_seed": TARGET_SEED,
            "epsilon": EPSILON,
        }

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        from ...framework.scheduler.runner import _run_unit

        eps = knobs.get("epsilon", EPSILON)
        paths = default_paths()
        store = CASStore(paths)
        log = RunLog(paths, store)

        checks: dict[str, bool] = {}
        notes: list[str] = []

        # Step 1: run target probe (run A)
        raw_a = _run_unit(
            TARGET_PROBE_ID,
            TARGET_CELL.value,
            TARGET_SEED,
            WiringLevel.SHADOW.value,
            str(paths.root),
        )
        checks["target_run_ok"] = bool(raw_a.get("ok"))
        if not checks["target_run_ok"]:
            return _fail_v1("target probe run failed", raw_a.get("error", ""), ctx, checks)

        run_id_a = raw_a["run_id"]
        record_a = log.get(run_id_a)
        exp_dir_a = paths.experiment_dir(run_id_a)

        # Step 2: delete and rebuild from CAS
        manifest_before = json.loads((exp_dir_a / "manifest.json").read_text("utf-8"))
        readouts_before = json.loads((exp_dir_a / "readouts" / "readouts.json").read_text("utf-8"))

        shutil.rmtree(exp_dir_a)
        checks["dir_deleted"] = not exp_dir_a.exists()

        manifest_obj = store.get_obj(record_a.manifest_sha)
        readouts_obj = store.get_obj(record_a.readouts_sha)

        exp_dir_a.mkdir(parents=True, exist_ok=True)
        (exp_dir_a / "readouts").mkdir(exist_ok=True)
        (exp_dir_a / "manifest.json").write_text(
            json.dumps(manifest_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (exp_dir_a / "readouts" / "readouts.json").write_text(
            json.dumps(readouts_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        checks["manifest_bit_exact"] = manifest_obj == manifest_before
        checks["readouts_bit_exact"] = readouts_obj == readouts_before

        # Step 3: re-run (run B)
        raw_b = _run_unit(
            TARGET_PROBE_ID,
            TARGET_CELL.value,
            TARGET_SEED,
            WiringLevel.SHADOW.value,
            str(paths.root),
        )
        checks["rerun_ok"] = bool(raw_b.get("ok"))
        if not checks["rerun_ok"]:
            return _fail_v1("target probe re-run failed", raw_b.get("error", ""), ctx, checks)

        # Step 4: ε-tolerance comparison
        # Logical state shas (knobs, input) must be bit-exact
        for k in ("knobs_sha", "input_sha"):
            checks[f"same_{k}"] = raw_a[k] == raw_b[k]
            if raw_a[k] != raw_b[k]:
                notes.append(f"{k} diverged: A={raw_a[k][:12]}... B={raw_b[k][:12]}...")

        # Output and readouts: ε-tolerance for floats
        readouts_a = raw_a.get("readouts", {})
        readouts_b = raw_b.get("readouts", {})
        checks["readouts_epsilon_close"] = _floats_close(readouts_a, readouts_b, eps)
        n_diffs = _count_float_diffs(readouts_a, readouts_b, eps)
        if n_diffs > 0:
            notes.append(f"readouts have {n_diffs} float values beyond ε={eps}")

        # Output shas: with numpy/torch, output may differ slightly
        # We check ε-tolerance on the actual output objects
        if raw_a["output_sha"] == raw_b["output_sha"]:
            checks["output_bit_exact"] = True
        else:
            # Load and compare with tolerance
            try:
                out_a = store.get_obj(raw_a["output_sha"])
                out_b = store.get_obj(raw_b["output_sha"])
                checks["output_epsilon_close"] = _floats_close(out_a, out_b, eps)
            except Exception:
                checks["output_epsilon_close"] = False

        # model_sha / dataset_sha in tags must be bit-exact (if present)
        tags_a = readouts_a.get("tags", {})
        tags_b = readouts_b.get("tags", {})
        for tag_key in ("model_sha", "dataset_sha"):
            if tag_key in tags_a and tag_key in tags_b:
                checks[f"same_{tag_key}"] = tags_a[tag_key] == tags_b[tag_key]

        all_ok = all(checks.values())

        log.close()
        store.close()

        readouts = ReadoutBundle(
            metrics={
                "passed": 1.0 if all_ok else 0.0,
                "n_checks": float(len(checks)),
                "n_failed": float(sum(1 for v in checks.values() if not v)),
                "epsilon": eps,
                "n_float_diffs": float(n_diffs),
            },
            artifacts={
                "checks": checks,
                "notes": notes,
                "run_id_a": run_id_a,
                "run_id_b": raw_b.get("run_id", ""),
            },
            tags={
                "target_probe": TARGET_PROBE_ID,
                "target_cell": TARGET_CELL.value,
                "target_seed": TARGET_SEED,
                "version": "v1",
            },
        )

        return RunOutcome(
            readouts=readouts,
            output={"all_ok": all_ok, "run_id_a": run_id_a, "run_id_b": raw_b.get("run_id", "")},
        )

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        if not outcomes:
            return GateReport(passed=False, reason="no outcomes", stats={})
        passed = all(o.readouts.metrics.get("passed", 0.0) >= 1.0 for o in outcomes)
        return GateReport(
            passed=passed,
            reason="all ε-tolerance rollback checks must pass",
            stats={
                "n_outcomes": len(outcomes),
                "n_passing": sum(1 for o in outcomes if o.readouts.metrics.get("passed", 0.0) >= 1.0),
            },
        )


def _fail_v1(reason: str, error: str, ctx: ProbeContext, checks: dict) -> RunOutcome:
    return RunOutcome(
        readouts=ReadoutBundle(
            metrics={
                "passed": 0.0,
                "n_checks": float(len(checks)),
                "n_failed": 1.0,
                "epsilon": EPSILON,
                "n_float_diffs": 0.0,
            },
            artifacts={"checks": checks, "reason": reason, "error": error},
            tags={"cell": ctx.cell.value, "version": "v1"},
        ),
        output={"all_ok": False, "reason": reason},
    )
