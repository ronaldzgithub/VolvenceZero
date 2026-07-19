"""F5 R15 meta-probe: 证明 rollback 正确。

Hypothesis（DESIGN.md §7）：
    任何运行完的实验，都能从 snapshot 完整还原其 input / output / readouts，
    且"重跑"相同 (probe_id, seed, cell) 得到 bit-exact 相同的输出。

本 meta-probe 的运行方式：
    1. 拿一个 "target probe"（pe-baseline-v0 的一个 seed）跑一次，得到 run_id A；
    2. 删除 experiments/<A>/；
    3. 从 CAS + RunLog 重建 manifest / readouts 文件；校验重建后内容等于原内容；
    4. 重新跑一次同一 (probe, seed, cell)，得到 run_id B；
    5. 校验 B 的 input_sha / output_sha / readouts_sha / knobs_sha 与 A 完全一致
       （bit-exact via content addressing）。
    任何一步失败 -> 整个框架不能升 ACTIVE。
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
from ...framework.snapshot import CASStore, RunLog, default_paths
from ...framework.wiring import AblationCell, WiringLevel


TARGET_PROBE_ID = "pe-baseline-v0"
TARGET_CELL = AblationCell.PROBE_ON
TARGET_SEED = 0


@register_probe
class R15RollbackProbe(BaseProbe):
    id = "r15-rollback-v0"
    hypothesis = (
        "Any completed experiment can be rolled back from CAS+RunLog bit-exact, "
        "and a re-run with the same (probe, seed, cell) yields identical shas."
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
        }

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        # Lazy import to avoid circular import at module load time.
        from ...framework.scheduler.runner import _run_unit

        paths = default_paths()
        store = CASStore(paths)
        log = RunLog(paths, store)

        # -------------------------------------------------------------
        # Step 1: run target probe once (run A).
        # -------------------------------------------------------------
        raw_a = _run_unit(
            TARGET_PROBE_ID,
            TARGET_CELL.value,
            TARGET_SEED,
            WiringLevel.SHADOW.value,
            str(paths.root),
        )
        checks: dict[str, bool] = {}
        notes: list[str] = []

        checks["target_run_ok"] = bool(raw_a.get("ok"))
        if not checks["target_run_ok"]:
            return _fail(
                "target probe run failed",
                extra_tags={"error": raw_a.get("error", "")},
                ctx=ctx,
                checks=checks,
            )

        run_id_a = raw_a["run_id"]
        record_a = log.get(run_id_a)
        exp_dir_a = paths.experiment_dir(run_id_a)
        manifest_path_a = exp_dir_a / "manifest.json"
        readouts_path_a = exp_dir_a / "readouts" / "readouts.json"

        checks["manifest_exists"] = manifest_path_a.exists()
        checks["readouts_exists"] = readouts_path_a.exists()

        # -------------------------------------------------------------
        # Step 2: delete experiments/<A>/, then rebuild from CAS.
        # -------------------------------------------------------------
        manifest_before = json.loads(manifest_path_a.read_text("utf-8"))
        readouts_before = json.loads(readouts_path_a.read_text("utf-8"))

        shutil.rmtree(exp_dir_a)
        checks["dir_deleted"] = not exp_dir_a.exists()

        # Rebuild
        manifest_obj = store.get_obj(record_a.manifest_sha)
        readouts_obj = store.get_obj(record_a.readouts_sha)

        exp_dir_a.mkdir(parents=True, exist_ok=True)
        (exp_dir_a / "readouts").mkdir(exist_ok=True)
        manifest_path_a.write_text(
            json.dumps(manifest_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        readouts_path_a.write_text(
            json.dumps(readouts_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        checks["manifest_bit_exact"] = manifest_obj == manifest_before
        checks["readouts_bit_exact"] = readouts_obj == readouts_before

        # -------------------------------------------------------------
        # Step 3: re-run same (probe, cell, seed) as run B.
        # -------------------------------------------------------------
        raw_b = _run_unit(
            TARGET_PROBE_ID,
            TARGET_CELL.value,
            TARGET_SEED,
            WiringLevel.SHADOW.value,
            str(paths.root),
        )
        checks["rerun_ok"] = bool(raw_b.get("ok"))
        if not checks["rerun_ok"]:
            return _fail(
                "target probe re-run failed",
                extra_tags={"error": raw_b.get("error", "")},
                ctx=ctx,
                checks=checks,
            )

        # -------------------------------------------------------------
        # Step 4: content-hash equality of all artifact shas.
        # -------------------------------------------------------------
        sha_fields = ("knobs_sha", "input_sha", "output_sha", "readouts_sha")
        for k in sha_fields:
            a = raw_a[k]
            b = raw_b[k]
            checks[f"same_{k}"] = (a == b)
            if a != b:
                notes.append(f"{k} diverged: A={a[:12]}... B={b[:12]}...")

        # manifest_sha 会因为 created_at 不同而不同 — 这是预期的（meta-time ≠ logical state）。
        # 我们只要求 *逻辑状态* bit-exact；时间戳允许变化。
        checks["manifest_sha_distinct_ok"] = raw_a["manifest_sha"] != raw_b["manifest_sha"]

        all_ok = all(checks.values())

        log.close()
        store.close()

        readouts = ReadoutBundle(
            metrics={
                "passed": 1.0 if all_ok else 0.0,
                "n_checks": float(len(checks)),
                "n_failed": float(sum(1 for v in checks.values() if not v)),
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
            },
        )

        return RunOutcome(
            readouts=readouts,
            output={
                "all_ok": all_ok,
                "run_id_a": run_id_a,
                "run_id_b": raw_b.get("run_id", ""),
            },
        )

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        if not outcomes:
            return GateReport(passed=False, reason="no outcomes", stats={})
        passed = all(o.readouts.metrics.get("passed", 0.0) >= 1.0 for o in outcomes)
        return GateReport(
            passed=passed,
            reason="all rollback checks must pass bit-exactly",
            stats={
                "n_outcomes": len(outcomes),
                "n_passing": sum(1 for o in outcomes if o.readouts.metrics.get("passed", 0.0) >= 1.0),
            },
        )


def _fail(
    reason: str,
    *,
    extra_tags: dict,
    ctx: ProbeContext,
    checks: dict,
) -> RunOutcome:
    return RunOutcome(
        readouts=ReadoutBundle(
            metrics={
                "passed": 0.0,
                "n_checks": float(len(checks)),
                "n_failed": float(sum(1 for v in checks.values() if not v)),
            },
            artifacts={"checks": checks, "reason": reason},
            tags={"cell": ctx.cell.value, **extra_tags},
        ),
        output={"all_ok": False, "reason": reason},
    )
