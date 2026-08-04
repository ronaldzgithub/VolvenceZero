"""Freeze the S1 full-width residual-readout admission before execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from volvence_zero.agent.eta_proof_benchmark import ETAOpenWeightRuntimeConfig
from volvence_zero.agent.eta_s1_residual_readout import (
    S1ResidualReadoutThresholds,
)
from volvence_zero.substrate import fingerprint_model_weight_files


PREREGISTRATION_SCHEMA_VERSION = "eta-s1-residual-readout-prereg.v1"
_REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_P1_REPORT = (
    "artifacts/eta_stage3_equivalence_diagnostic_20260804/report.json"
)
FROZEN_SOURCE_FILES = (
    "packages/vz-substrate/src/volvence_zero/substrate/residual_contracts.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "packages/vz-substrate/src/volvence_zero/substrate/forward_representation.py",
    "packages/vz-substrate/src/volvence_zero/substrate/frozen_residual_readout.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_rate_distortion_evidence.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_s1_residual_readout.py",
    "scripts/run_eta_s1_residual_readout.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_preregistration(
    *,
    model_id: str,
    model_source: str,
    model_version: str,
    device: str,
    max_length: int,
    ridge_alpha: float,
    corpus_seed: int,
    objective_count: int,
    corridor_count: int,
    extra_edge_probability: float,
    train_routes: int,
    heldout_routes: int,
    train_lengths: tuple[int, ...],
    heldout_lengths: tuple[int, ...],
) -> dict[str, object]:
    source_report_path = _REPO_ROOT / SOURCE_P1_REPORT
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    attribution = source_report.get("attribution")
    if not isinstance(attribution, dict) or (
        attribution.get("dominant_attribution")
        != "incentive-bypass-via-free-bias"
    ):
        raise RuntimeError(
            "S1 preregistration requires the completed P1 free-bias attribution."
        )
    model_root = (_REPO_ROOT / model_source).resolve()
    model_weights_sha256 = fingerprint_model_weight_files(model_root)
    thresholds = S1ResidualReadoutThresholds()
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "experiment_id": "eta-s1-full-width-frozen-residual-readout",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "s1-readout-admission-no-causal-claim",
        "source_p1_artifact": {
            "path": SOURCE_P1_REPORT,
            "sha256": _sha256(source_report_path),
            "dominant_attribution": "incentive-bypass-via-free-bias",
        },
        "configuration": {
            "model_id": model_id,
            "model_source": model_source,
            "model_version": model_version,
            "model_weights_sha256": model_weights_sha256,
            "device": device,
            "model_dtype": "float32",
            "max_length": max_length,
            "fail_on_truncation": True,
            "layer_indices": [20],
            "activation_widths": [896],
            "ridge_alpha": ridge_alpha,
            "training_mode": "closed-form-standardized-ridge-one-hot.v1",
            "class_axis": "class-weight-minus-other-class-mean-l2.v1",
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
            "probe_surface": "cumulative-trajectory-prefix",
        },
        "thresholds": asdict(thresholds),
        "decision_rules": [
            "Admit S1 only if every preregistered threshold passes on heldout routes.",
            "The layer is fixed at the later S2 injection layer (20) before readout.",
            "Class axes exclude classifier bias and are not optimized against S2 outcomes.",
            "S1 establishes decodability only; it cannot establish causal steering.",
            "Any source above max_length fails loudly; silent truncation is forbidden.",
        ],
        "prohibited_after_execution": [
            "selecting another layer or ridge alpha after reading heldout scores",
            "changing any threshold after reading S1 results",
            "feeding heldout labels, scores, margins, or admission into learning",
            "installing the artifact into production or changing a WiringLevel",
            "claiming S2 steering success from S1 decodability",
        ],
        "frozen_source_files": {
            name: _sha256(_REPO_ROOT / name) for name in FROZEN_SOURCE_FILES
        },
    }


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze ETA S1 full-width residual-readout admission."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-id", default=ETAOpenWeightRuntimeConfig().model_id
    )
    parser.add_argument(
        "--model-source",
        default="artifacts/eta_stage2_merged_v2_20260803",
    )
    parser.add_argument("--model-version", default="eta-stage2-merged-v2")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--corridor-count", type=int, default=2)
    parser.add_argument("--extra-edge-probability", type=float, default=0.35)
    parser.add_argument("--train-routes", type=int, default=64)
    parser.add_argument("--heldout-routes", type=int, default=24)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--heldout-lengths", type=int, nargs="+", default=(3, 4))
    args = parser.parse_args(argv)
    if args.max_length < 64 or args.ridge_alpha <= 0.0:
        parser.error("--max-length must be >= 64 and --ridge-alpha must be positive")
    payload = build_preregistration(
        model_id=args.model_id,
        model_source=args.model_source,
        model_version=args.model_version,
        device=args.device,
        max_length=args.max_length,
        ridge_alpha=args.ridge_alpha,
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
