"""Freeze the S2 no-bias causal-steering protocol before execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from volvence_zero.agent.eta_s2_causal_steering import (
    S2CausalSteeringThresholds,
)
from volvence_zero.substrate import FrozenResidualReadoutArtifact


PREREGISTRATION_SCHEMA_VERSION = "eta-s2-causal-steering-prereg.v1"
_REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_S1_REPORT = "artifacts/eta_s1_residual_readout_v2_20260804/report.json"
SOURCE_S1_MANIFEST = (
    "artifacts/eta_s1_residual_readout_v2_20260804/artifact_manifest.json"
)
SOURCE_S1_ARTIFACT = (
    "artifacts/eta_s1_residual_readout_v2_20260804/readout_artifact.json"
)
FROZEN_SOURCE_FILES = (
    "packages/vz-substrate/src/volvence_zero/substrate/residual_contracts.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "packages/vz-substrate/src/volvence_zero/substrate/steered_action_scoring.py",
    "packages/vz-substrate/src/volvence_zero/substrate/frozen_residual_readout.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_rate_distortion_evidence.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_s2_causal_steering.py",
    "scripts/run_eta_s2_causal_steering.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_preregistration(
    *,
    device: str,
    max_length: int,
    scale_fractions: tuple[float, ...],
    primary_scale_fraction: float,
    shuffled_class_shifts: tuple[int, ...],
    control_norm_ratio: float,
    probe_train_rows: int,
    batch_size: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    corpus_seed: int,
    objective_count: int,
    corridor_count: int,
    extra_edge_probability: float,
    train_routes: int,
    heldout_routes: int,
    train_lengths: tuple[int, ...],
    heldout_lengths: tuple[int, ...],
) -> dict[str, object]:
    report_path = _REPO_ROOT / SOURCE_S1_REPORT
    manifest_path = _REPO_ROOT / SOURCE_S1_MANIFEST
    artifact_path = _REPO_ROOT / SOURCE_S1_ARTIFACT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = FrozenResidualReadoutArtifact.from_json(
        artifact_path.read_text(encoding="utf-8")
    )
    admission = report.get("admission")
    if not isinstance(admission, dict) or admission.get("admitted") is not True:
        raise RuntimeError("S2 preregistration requires S1 admission PASS.")
    if manifest.get("s1_admitted_for_s2") is not True:
        raise RuntimeError("S1 manifest does not authorize S2 evidence.")
    if report.get("artifact_id") != artifact.artifact_id:
        raise RuntimeError("S1 report/artifact lineage mismatch.")
    thresholds = S2CausalSteeringThresholds()
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "experiment_id": "eta-s2-no-bias-causal-residual-steering",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "s2-heldout-axis-causal-steering",
        "source_s1": {
            "report_path": SOURCE_S1_REPORT,
            "report_sha256": _sha256(report_path),
            "manifest_path": SOURCE_S1_MANIFEST,
            "manifest_sha256": _sha256(manifest_path),
            "artifact_path": SOURCE_S1_ARTIFACT,
            "artifact_file_sha256": _sha256(artifact_path),
            "artifact_id": artifact.artifact_id,
            "admitted": True,
        },
        "configuration": {
            "model_id": artifact.model_fingerprint.model_id,
            "model_source": "artifacts/eta_stage2_merged_v2_20260803",
            "model_version": artifact.model_fingerprint.version,
            "model_weights_sha256": artifact.model_fingerprint.weights_sha256,
            "device": device,
            "model_dtype": "float32",
            "max_length": max_length,
            "token_budget_policy": "fail-loud-preflight.v1",
            "injection_layer_index": 20,
            "hidden_size": 896,
            "control_norm_ratio": control_norm_ratio,
            "scale_fractions": list(scale_fractions),
            "primary_scale_fraction": primary_scale_fraction,
            "shuffled_class_shifts": list(shuffled_class_shifts),
            "probe_train_rows": probe_train_rows,
            "batch_size": batch_size,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_confidence": bootstrap_confidence,
            "corpus": {
                "corpus_seed": corpus_seed,
                "objective_count": objective_count,
                "corridor_count": corridor_count,
                "extra_edge_probability": extra_edge_probability,
                "train_routes": train_routes,
                "heldout_routes": heldout_routes,
                "train_lengths": list(train_lengths),
                "heldout_lengths": list(heldout_lengths),
            },
            "observation_protocol": "partially-observable-staged-plan.v4",
            "evaluation_split": "heldout-only",
            "controls": [
                "target-plus-axis",
                "target-minus-axis",
                "noop",
                "mean-of-three-cyclic-shuffled-class-axes",
            ],
            "aggregation": "mean-within-route-then-bootstrap-routes",
        },
        "thresholds": asdict(thresholds),
        "decision_rules": [
            "Only scale_fraction=0.50 adjudicates the primary S2 claim.",
            "All three route-level effects and bootstrap lower bounds must be positive.",
            "Target-plus must beat noop, sign reversal, and shuffled class axes.",
            "Every route-win-rate must reach the preregistered 0.65 threshold.",
            "Scale 0.25 and 1.00 are dose diagnostics and cannot rescue primary failure.",
        ],
        "prohibited_after_execution": [
            "selecting the best scale after reading outcomes",
            "changing thresholds, shifts, batches, or bootstrap after outcomes",
            "adding a trainable or constant steering bias",
            "fitting any parameter on heldout action NLL",
            "treating repeated steps as independent bootstrap units",
            "installing the artifact or changing production WiringLevel",
            "feeding action NLL or S2 admission into learning",
        ],
        "frozen_source_files": {
            name: _sha256(_REPO_ROOT / name) for name in FROZEN_SOURCE_FILES
        },
    }


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze ETA S2 no-bias causal residual steering."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument(
        "--scale-fractions", type=float, nargs="+", default=(0.25, 0.50, 1.00)
    )
    parser.add_argument("--primary-scale-fraction", type=float, default=0.50)
    parser.add_argument(
        "--shuffled-class-shifts", type=int, nargs="+", default=(1, 3, 5)
    )
    parser.add_argument("--control-norm-ratio", type=float, default=0.25)
    parser.add_argument("--probe-train-rows", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--corridor-count", type=int, default=2)
    parser.add_argument("--extra-edge-probability", type=float, default=0.35)
    parser.add_argument("--train-routes", type=int, default=64)
    parser.add_argument("--heldout-routes", type=int, default=24)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--heldout-lengths", type=int, nargs="+", default=(3, 4))
    args = parser.parse_args(argv)
    payload = build_preregistration(
        device=args.device,
        max_length=args.max_length,
        scale_fractions=tuple(args.scale_fractions),
        primary_scale_fraction=args.primary_scale_fraction,
        shuffled_class_shifts=tuple(args.shuffled_class_shifts),
        control_norm_ratio=args.control_norm_ratio,
        probe_train_rows=args.probe_train_rows,
        batch_size=args.batch_size,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        corpus_seed=args.corpus_seed,
        objective_count=args.objective_count,
        corridor_count=args.corridor_count,
        extra_edge_probability=args.extra_edge_probability,
        train_routes=args.train_routes,
        heldout_routes=args.heldout_routes,
        train_lengths=tuple(args.train_lengths),
        heldout_lengths=tuple(args.heldout_lengths),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"sha256 {_sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
