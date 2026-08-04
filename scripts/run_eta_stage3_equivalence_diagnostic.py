"""Run the preregistered, attribution-only ETA Stage-3 P1 diagnostic."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import torch
import transformers

from companion_test_plan_common import (
    MPSAvailability,
    exclusive_mps_lock,
    mps_payload,
    require_mps,
)
from msc_prediction_checkpoint import PredictionRunCheckpointStore
from preregister_eta_stage3_equivalence_diagnostic import (
    FROZEN_SOURCE_FILES,
    PREREGISTRATION_SCHEMA_VERSION,
    SOURCE_STAGE3_REPORT,
)
from volvence_zero.agent.eta_proof_benchmark import (
    ETAOpenWeightRuntimeConfig,
    generate_eta_proof_corpus,
)
from volvence_zero.agent.eta_stage3_equivalence_diagnostic import (
    Stage3EquivalenceDiagnosticReport,
    Stage3EquivalenceThresholds,
    Stage3SteeringControlPoint,
    run_eta_stage3_equivalence_diagnostic,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_ID = "eta-stage3-equivalence-mps.v1"
CHECKPOINT_SCHEMA_NAMESPACE = "eta-stage3-equivalence"


class PreregistrationMismatch(RuntimeError):
    """Frozen P1 protocol disagrees with the requested execution."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    return result.stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class EquivalencePointCache:
    def __init__(self, store: PredictionRunCheckpointStore) -> None:
        self._store = store

    @staticmethod
    def _unit(*, training_mode: str, seed: int) -> str:
        return f"points/{training_mode}/seed-{seed}"

    @classmethod
    def _relative_path(cls, *, training_mode: str, seed: int) -> str:
        return f"checkpoints/{cls._unit(training_mode=training_mode, seed=seed)}.json"

    def load_point(
        self, *, training_mode: str, seed: int
    ) -> Stage3SteeringControlPoint | None:
        payload = self._store.load_json(
            unit=self._unit(training_mode=training_mode, seed=seed),
            relative_path=self._relative_path(
                training_mode=training_mode,
                seed=seed,
            ),
        )
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("P1 checkpoint payload must be an object.")
        return Stage3SteeringControlPoint(**payload)

    def store_point(self, point: Stage3SteeringControlPoint) -> None:
        self._store.save_json(
            unit=self._unit(
                training_mode=point.training_mode,
                seed=point.seed,
            ),
            relative_path=self._relative_path(
                training_mode=point.training_mode,
                seed=point.seed,
            ),
            payload=asdict(point),
        )


def _requested_configuration(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_id": args.model_id,
        "model_source": args.model_source,
        "device": args.device,
        "layer_indices": [20, 21, 22],
        "activation_width": 8,
        "seed_schedule": list(range(args.seeds)),
        "n_z": args.n_z,
        "alpha": args.alpha,
        "updates_per_run": args.updates,
        "learning_rate": args.learning_rate,
        "switch_threshold": args.switch_threshold,
        "reference_gate2_accuracy": args.reference_gate2_accuracy,
        "corpus": {
            "corpus_seed": args.corpus_seed,
            "objective_count": args.objective_count,
            "corridor_count": args.corridor_count,
            "extra_edge_probability": args.extra_edge_probability,
            "train_routes": args.train_routes,
            "heldout_routes": args.heldout_routes,
            "train_lengths": list(args.train_lengths),
            "heldout_lengths": list(args.heldout_lengths),
        },
        "observation_protocol": "partially-observable-staged-plan.v4",
        "posterior_parameterization": "smooth",
        "rate_gating": "switch-gated",
        "gate_mode": "hard-st",
        "training_modes": ["full", "bias-only"],
        "control_ablations": ["zero-z", "cyclic-permuted-z"],
        "oracle_boundary": (
            "active_subgoal[t] != active_subgoal[t-1]; readout-only"
        ),
    }


def _validate_preregistration(
    payload: object,
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise PreregistrationMismatch("preregistration root must be an object")
    if payload.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION:
        raise PreregistrationMismatch("preregistration schema_version mismatch")
    if payload.get("claim_scope") != "stage3-attribution-only-no-readjudication":
        raise PreregistrationMismatch("preregistration claim_scope mismatch")
    if payload.get("configuration") != _requested_configuration(args):
        raise PreregistrationMismatch(
            "preregistered configuration does not match requested execution"
        )
    expected_thresholds = asdict(Stage3EquivalenceThresholds())
    if payload.get("thresholds") != expected_thresholds:
        raise PreregistrationMismatch("preregistered thresholds drifted")

    source = payload.get("source_stage3_artifact")
    if not isinstance(source, dict):
        raise PreregistrationMismatch("source_stage3_artifact is missing")
    report_path = _REPO_ROOT / SOURCE_STAGE3_REPORT
    if source.get("path") != SOURCE_STAGE3_REPORT:
        raise PreregistrationMismatch("source Stage-3 report path drifted")
    if source.get("sha256") != _sha256(report_path):
        raise PreregistrationMismatch("source Stage-3 report hash drifted")
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        source.get("verdict") != "kill-eta"
        or source_report.get("verdict") != "kill-eta"
    ):
        raise PreregistrationMismatch("source Stage-3 verdict is not kill-eta")

    frozen = payload.get("frozen_source_files")
    if not isinstance(frozen, dict):
        raise PreregistrationMismatch("frozen_source_files is missing")
    for name in FROZEN_SOURCE_FILES:
        actual = _sha256(_REPO_ROOT / name)
        if frozen.get(name) != actual:
            raise PreregistrationMismatch(
                f"frozen source hash drift for {name}: "
                f"preregistered={frozen.get(name)!r}, actual={actual!r}"
            )
    return payload


def _report_markdown(report: Stage3EquivalenceDiagnosticReport) -> str:
    probe = report.exact_entry_probe
    attribution = report.attribution
    lines = [
        "# ETA Stage-3 P1 等价性诊断",
        "",
        "> 范围：只归因，不重判已封存的 Stage-3 `kill-eta`，不改 production WiringLevel。",
        "",
        "## 单页结论",
        "",
        f"- 主归因：`{attribution.dominant_attribution}`",
        (
            "- exact-entry probe："
            f"{probe.accuracy:.3f}（chance {probe.chance_accuracy:.3f}，"
            f"Gate-2 参考 {probe.reference_gate2_accuracy:.3f}）"
        ),
        (
            "- bias-only 改善回收："
            f"{attribution.mean_bias_only_recovery:.3f}；zero-z 改善回收："
            f"{attribution.mean_zero_z_recovery:.3f}"
        ),
        (
            "- cyclic-permuted-z distortion penalty："
            f"{attribution.mean_permuted_z_penalty:.3f}"
        ),
        (
            "- oracle subgoal F1 − action-change F1："
            f"{attribution.mean_oracle_minus_action_boundary_f1:.3f}"
        ),
        "",
        "## 匹配控制",
        "",
        "| seed | mode | heldout D | zero-z D | permuted-z D | action F1 | oracle F1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for point in report.control_points:
        zero = (
            f"{point.heldout_zero_z_distortion:.3f}"
            if point.heldout_zero_z_distortion is not None
            else "—"
        )
        permuted = (
            f"{point.heldout_permuted_z_distortion:.3f}"
            if point.heldout_permuted_z_distortion is not None
            else "—"
        )
        lines.append(
            f"| {point.seed} | {point.training_mode} | "
            f"{point.heldout_distortion:.3f} | {zero} | {permuted} | "
            f"{point.action_boundary_f1:.3f} | "
            f"{point.oracle_subgoal_boundary_f1:.3f} |"
        )
    lines.extend(
        (
            "",
            "## 决策边界",
            "",
            (
                "P1 只能说明信息是否死在 exact entry、free bias 是否绕开 z、"
                "以及 z 是否具有因果作用。任何读数都不撤销 Stage-3 verdict；"
                "忠实 ETA rewrite 必须另立 claim / prereg。"
            ),
            "",
        )
    )
    return "\n".join(lines)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run preregistered ETA Stage-3 P1 diagnostics."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
    )
    parser.add_argument(
        "--model-id", default=ETAOpenWeightRuntimeConfig().model_id
    )
    parser.add_argument(
        "--model-source",
        default="artifacts/eta_stage2_merged_v2_20260803",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-z", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--switch-threshold", type=float, default=0.55)
    parser.add_argument("--reference-gate2-accuracy", type=float, default=0.944)
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--corridor-count", type=int, default=2)
    parser.add_argument("--extra-edge-probability", type=float, default=0.35)
    parser.add_argument("--train-routes", type=int, default=64)
    parser.add_argument("--heldout-routes", type=int, default=24)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--heldout-lengths", type=int, nargs="+", default=(3, 4))
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.updates < 1:
        parser.error("--seeds and --updates must be positive")

    preregistration = _validate_preregistration(
        json.loads(args.preregistration.read_text(encoding="utf-8")),
        args=args,
    )
    preregistration_sha256 = _sha256(args.preregistration)
    source_sha256 = {
        name: _sha256(_REPO_ROOT / name) for name in FROZEN_SOURCE_FILES
    }
    checkpoint_store = PredictionRunCheckpointStore(
        output_dir=args.output_dir,
        configuration={
            **_requested_configuration(args),
            "preregistration_sha256": preregistration_sha256,
            "source_sha256": source_sha256,
        },
        resume=args.resume,
        schema_namespace=CHECKPOINT_SCHEMA_NAMESPACE,
    )
    corpus = generate_eta_proof_corpus(
        seed=args.corpus_seed,
        objective_count=args.objective_count,
        corridor_count=args.corridor_count,
        extra_edge_probability=args.extra_edge_probability,
        train_route_count=args.train_routes,
        heldout_route_count=args.heldout_routes,
        train_lengths=tuple(args.train_lengths),
        heldout_lengths=tuple(args.heldout_lengths),
    )
    config = ETAOpenWeightRuntimeConfig(
        model_id=args.model_id,
        model_source=args.model_source,
        device=args.device,
        layer_indices=(20, 21, 22),
        activation_width=8,
        model_dtype="float32",
    )
    uses_mps = args.device.startswith("mps")
    started = time.perf_counter()
    with ExitStack() as stack:
        mps: MPSAvailability | None = None
        if uses_mps:
            stack.enter_context(
                exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID)
            )
            mps = require_mps()
        report = run_eta_stage3_equivalence_diagnostic(
            corpus=corpus,
            open_weight_config=config,
            source_stage3_verdict="kill-eta",
            seed_schedule=tuple(range(args.seeds)),
            n_z=args.n_z,
            alpha=args.alpha,
            updates_per_run=args.updates,
            learning_rate=args.learning_rate,
            switch_threshold=args.switch_threshold,
            reference_gate2_accuracy=args.reference_gate2_accuracy,
            thresholds=Stage3EquivalenceThresholds(),
            point_cache=EquivalencePointCache(checkpoint_store),
            progress=lambda message: print(message, flush=True),
        )
        elapsed = time.perf_counter() - started
    checkpoint_store.mark_complete()

    _write_json(args.output_dir / "report.json", asdict(report))
    points_path = args.output_dir / "points.jsonl"
    with points_path.open("w", encoding="utf-8") as handle:
        for point in report.control_points:
            handle.write(json.dumps(asdict(point), ensure_ascii=False) + "\n")
    (args.output_dir / "report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    result_files = ("report.json", "points.jsonl", "report.md")
    manifest = {
        "schema_version": "eta-stage3-equivalence-manifest.v1",
        "experiment_id": "eta-stage3-equivalence-diagnostic-p1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "claim_scope": report.claim_scope,
        "source_stage3_verdict": report.source_stage3_verdict,
        "source_stage3_report": preregistration["source_stage3_artifact"],
        "preregistered": True,
        "preregistration_path": str(args.preregistration),
        "preregistration_sha256": preregistration_sha256,
        "attribution_authoritative": True,
        "formal_claim_allowed": False,
        "production_wiring_changed": False,
        "resumed": bool(args.resume),
        "checkpoint_units": checkpoint_store.immutable_file_manifest(),
        "elapsed_seconds": round(elapsed, 3),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_exclusive_lock": str(args.mps_lock) if uses_mps else "not-required",
        "mps_attestation": mps_payload(mps) if mps is not None else "not-required",
        "source_files": source_sha256,
        "result_files": {
            name: _sha256(args.output_dir / name) for name in result_files
        },
    }
    _write_json(args.output_dir / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {
                "source_stage3_verdict": report.source_stage3_verdict,
                "dominant_attribution": report.attribution.dominant_attribution,
                "exact_entry_accuracy": round(
                    report.exact_entry_probe.accuracy,
                    4,
                ),
                "bias_only_recovery": round(
                    report.attribution.mean_bias_only_recovery,
                    4,
                ),
                "zero_z_recovery": round(
                    report.attribution.mean_zero_z_recovery,
                    4,
                ),
                "permuted_z_penalty": round(
                    report.attribution.mean_permuted_z_penalty,
                    4,
                ),
                "elapsed_seconds": round(elapsed, 1),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
