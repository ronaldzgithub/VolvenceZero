"""Runner implementations."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..probe import (
    BaseProbe,
    GateReport,
    ProbeContext,
    RunOutcome,
    get_registry,
)
from ..probe.types import ReadoutBundle
from ..snapshot import (
    CASStore,
    LabsPaths,
    RunLog,
    RunRecord,
    canonical_dumps,
    default_paths,
)
from ..wiring import AblationCell, WiringLevel, WiringProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_run_id(probe_id: str, cell: AblationCell, seed: int) -> str:
    # Stable-ish prefix so you can grep; random suffix to avoid collision
    ts = time.strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{ts}_{probe_id}_{cell.value}_s{seed}_{suffix}"


def _pick_knobs(probe: BaseProbe) -> dict[str, Any]:
    """阶段 0 只取每个 knob 的第一个值（稳定的默认组合）。

    阶段 1 再做 grid 展开 / Bayesian 搜索。
    """
    raw = probe.knobs() or {}
    picked: dict[str, Any] = {}
    for name, values in raw.items():
        if not values:
            continue
        picked[name] = values[0]
    return picked


# ---------------------------------------------------------------------------
# Unit execution
# ---------------------------------------------------------------------------

@dataclass
class UnitReport:
    run_id: str
    probe_id: str
    cell: str
    seed: int
    wiring: str
    ok: bool
    error: Optional[str]
    readouts: dict[str, Any]
    metrics: dict[str, float]
    manifest_sha: str
    output_sha: str
    readouts_sha: str
    input_sha: str
    knobs_sha: str
    duration_s: float

    def to_jsonable(self) -> dict:
        return {
            "run_id": self.run_id,
            "probe_id": self.probe_id,
            "cell": self.cell,
            "seed": self.seed,
            "wiring": self.wiring,
            "ok": self.ok,
            "error": self.error,
            "metrics": self.metrics,
            "readouts": self.readouts,
            "input_sha": self.input_sha,
            "output_sha": self.output_sha,
            "readouts_sha": self.readouts_sha,
            "knobs_sha": self.knobs_sha,
            "manifest_sha": self.manifest_sha,
            "duration_s": self.duration_s,
        }


def _ensure_probes_imported() -> None:
    """Import the builtin probes package so registrations are present.

    Safe to call multiple times. Needed in 'spawn' child processes.
    """
    # Relative import from 'framework.scheduler' to 'volvence_labs.probes'.
    import importlib
    importlib.import_module("volvence_labs.probes")


def _run_unit(
    probe_id: str,
    cell_value: str,
    seed: int,
    level_value: str,
    root_str: str,
    knob_overrides: Optional[dict] = None,
) -> dict:
    """单 unit 执行体（必须在子进程可 pickle，故参数都是 str/int）。

    返回 UnitReport 的 jsonable dict。
    """
    _ensure_probes_imported()
    start = time.time()
    cell = AblationCell(cell_value)
    level = WiringLevel(level_value)
    paths = default_paths(root_str)
    store = CASStore(paths)
    log = RunLog(paths, store)

    probe_cls = get_registry().get(probe_id)
    probe: BaseProbe = probe_cls()

    try:
        knobs = _pick_knobs(probe)
        # Merge overrides (CLI --knob flags or profile-level overrides)
        if knob_overrides:
            knobs.update(knob_overrides)
        knobs_sha = store.put_obj(knobs, kind="knobs", meta={"probe_id": probe_id})

        # Choose input source: real model or synthetic
        use_real = knobs.get("use_real_model", False)
        if use_real and hasattr(probe, "real_inputs"):
            inputs = probe.real_inputs(seed, knobs)
        else:
            inputs = probe.default_inputs(seed)
        input_sha = store.put_obj(
            inputs,
            kind="probe_input",
            meta={"probe_id": probe_id, "seed": seed, "cell": cell_value},
        )

        ctx = ProbeContext(
            level=level,
            cell=cell,
            seed=seed,
            inputs=inputs,
            inputs_sha=input_sha,
        )
        outcome: RunOutcome = probe.run_cell(ctx, knobs)
        readouts = outcome.readouts
        if not isinstance(readouts, ReadoutBundle):
            raise TypeError(
                f"probe {probe_id!r} run_cell must return RunOutcome with ReadoutBundle readouts"
            )

        readouts_sha = store.put_obj(
            readouts.to_jsonable(),
            kind="readouts",
            meta={"probe_id": probe_id, "cell": cell_value, "seed": seed},
        )
        output_sha = store.put_obj(
            outcome.output if outcome.output is not None else {},
            kind="probe_output",
            meta={"probe_id": probe_id, "cell": cell_value, "seed": seed},
        )

        run_id = _new_run_id(probe_id, cell, seed)

        manifest = {
            "run_id": run_id,
            "probe_id": probe_id,
            "primitive": probe.primitive.value,
            "r_ids": list(probe.r_ids),
            "hypothesis": probe.hypothesis,
            "cell": cell.value,
            "seed": seed,
            "wiring": level.value,
            "knobs_sha": knobs_sha,
            "input_sha": input_sha,
            "output_sha": output_sha,
            "readouts_sha": readouts_sha,
            "created_at": time.time(),
        }
        manifest_sha = store.put_obj(manifest, kind="manifest", meta={"run_id": run_id})

        record = RunRecord(
            run_id=run_id,
            probe_id=probe_id,
            wiring=level.value,
            ablation_cell=cell.value,
            seed=seed,
            knobs_sha=knobs_sha,
            input_sha=input_sha,
            output_sha=output_sha,
            readouts_sha=readouts_sha,
            manifest_sha=manifest_sha,
            created_at=manifest["created_at"],
        )
        log.record(record)

        # Human-readable mirror
        exp_dir = paths.experiment_dir(run_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ro_dir = exp_dir / "readouts"
        ro_dir.mkdir(exist_ok=True)
        (ro_dir / "readouts.json").write_text(
            json.dumps(readouts.to_jsonable(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        duration = time.time() - start
        return UnitReport(
            run_id=run_id,
            probe_id=probe_id,
            cell=cell.value,
            seed=seed,
            wiring=level.value,
            ok=True,
            error=None,
            readouts=readouts.to_jsonable(),
            metrics=dict(readouts.metrics),
            manifest_sha=manifest_sha,
            output_sha=output_sha,
            readouts_sha=readouts_sha,
            input_sha=input_sha,
            knobs_sha=knobs_sha,
            duration_s=duration,
        ).to_jsonable()
    except Exception:
        err = traceback.format_exc()
        duration = time.time() - start
        return UnitReport(
            run_id="",
            probe_id=probe_id,
            cell=cell_value,
            seed=seed,
            wiring=level_value,
            ok=False,
            error=err,
            readouts={},
            metrics={},
            manifest_sha="",
            output_sha="",
            readouts_sha="",
            input_sha="",
            knobs_sha="",
            duration_s=duration,
        ).to_jsonable()
    finally:
        log.close()
        store.close()


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

@dataclass
class ExperimentReport:
    probe_id: str
    profile_name: str
    units: list[UnitReport] = field(default_factory=list)
    gate: Optional[GateReport] = None

    def to_jsonable(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "profile": self.profile_name,
            "units": [u.to_jsonable() for u in self.units],
            "gate": None
            if self.gate is None
            else {
                "passed": self.gate.passed,
                "reason": self.gate.reason,
                "stats": dict(self.gate.stats),
            },
        }


class SequentialRunner:
    def __init__(self, root: Optional[str] = None, knob_overrides: Optional[dict] = None):
        self.paths: LabsPaths = default_paths(root)
        self.knob_overrides = knob_overrides

    def run(self, probe_id: str, profile: WiringProfile) -> ExperimentReport:
        level = profile.level_for(probe_id)
        units: list[UnitReport] = []
        for cell in profile.cells:
            for seed in profile.seeds:
                raw = _run_unit(probe_id, cell.value, seed, level.value, str(self.paths.root), self.knob_overrides)
                units.append(_unit_from_jsonable(raw))
        return _build_report(probe_id, profile, units)


class ParallelRunner:
    def __init__(self, root: Optional[str] = None, max_workers: Optional[int] = None, knob_overrides: Optional[dict] = None):
        self.paths: LabsPaths = default_paths(root)
        self.max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
        self.knob_overrides = knob_overrides

    def run(self, probe_id: str, profile: WiringProfile) -> ExperimentReport:
        level = profile.level_for(probe_id)
        args = [
            (probe_id, cell.value, seed, level.value, str(self.paths.root), self.knob_overrides)
            for cell in profile.cells
            for seed in profile.seeds
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=self.max_workers) as pool:
            raw_reports = pool.starmap(_run_unit, args)
        units = [_unit_from_jsonable(r) for r in raw_reports]
        return _build_report(probe_id, profile, units)


def _unit_from_jsonable(d: dict) -> UnitReport:
    return UnitReport(
        run_id=d["run_id"],
        probe_id=d["probe_id"],
        cell=d["cell"],
        seed=d["seed"],
        wiring=d["wiring"],
        ok=d["ok"],
        error=d.get("error"),
        readouts=d.get("readouts", {}),
        metrics=d.get("metrics", {}),
        manifest_sha=d.get("manifest_sha", ""),
        output_sha=d.get("output_sha", ""),
        readouts_sha=d.get("readouts_sha", ""),
        input_sha=d.get("input_sha", ""),
        knobs_sha=d.get("knobs_sha", ""),
        duration_s=d.get("duration_s", 0.0),
    )


def _build_report(
    probe_id: str,
    profile: WiringProfile,
    units: list[UnitReport],
) -> ExperimentReport:
    _ensure_probes_imported()
    probe_cls = get_registry().get(probe_id)
    probe: BaseProbe = probe_cls()
    outcomes: list[RunOutcome] = []
    for u in units:
        if not u.ok:
            continue
        outcomes.append(
            RunOutcome(
                readouts=ReadoutBundle(
                    metrics=u.metrics,
                    artifacts=u.readouts.get("artifacts", {}),
                    tags=u.readouts.get("tags", {}),
                ),
                output=None,
            )
        )
    gate = probe.gate(outcomes) if outcomes else GateReport(
        passed=False, reason="no successful units", stats={}
    )

    try:
        from ..readout.metrics_exporter import record_run, record_gate_decision

        for u in units:
            record_run(
                probe_id=probe_id,
                cell=u.cell,
                ok=u.ok,
                elapsed_s=u.duration_s,
                metrics=u.metrics,
                seed=u.seed,
            )
        record_gate_decision(probe_id, "approve" if gate.passed else "reject")
    except Exception:
        pass

    return ExperimentReport(probe_id=probe_id, profile_name=profile.name, units=units, gate=gate)


def run_experiment(
    probe_id: str,
    profile: WiringProfile,
    *,
    parallel: bool = False,
    root: Optional[str] = None,
    knob_overrides: Optional[dict] = None,
) -> ExperimentReport:
    runner: Any = ParallelRunner(root=root, knob_overrides=knob_overrides) if parallel else SequentialRunner(root=root, knob_overrides=knob_overrides)
    return runner.run(probe_id, profile)
