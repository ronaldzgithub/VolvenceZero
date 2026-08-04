"""Freeze the attribution-only ETA Stage-3 P1 diagnostic before execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from volvence_zero.agent.eta_proof_benchmark import ETAOpenWeightRuntimeConfig
from volvence_zero.agent.eta_stage3_equivalence_diagnostic import (
    Stage3EquivalenceThresholds,
)

PREREGISTRATION_SCHEMA_VERSION = "eta-stage3-equivalence-prereg.v1"

_REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_STAGE3_REPORT = (
    "artifacts/eta_stage3_rate_distortion_20260803/report.json"
)
FROZEN_SOURCE_FILES = (
    "packages/vz-substrate/src/volvence_zero/substrate/steered_action_scoring.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "packages/vz-temporal/src/volvence_zero/temporal/torch_store_ssl.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_rate_distortion_evidence.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_stage3_equivalence_diagnostic.py",
    "scripts/run_eta_stage3_equivalence_diagnostic.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_preregistration(
    *,
    model_id: str,
    model_source: str,
    device: str,
    seeds: int,
    n_z: int,
    alpha: float,
    updates: int,
    learning_rate: float,
    switch_threshold: float,
    reference_gate2_accuracy: float,
    corpus_seed: int,
    objective_count: int,
    corridor_count: int,
    extra_edge_probability: float,
    train_routes: int,
    heldout_routes: int,
    train_lengths: tuple[int, ...],
    heldout_lengths: tuple[int, ...],
) -> dict[str, object]:
    source_report_path = _REPO_ROOT / SOURCE_STAGE3_REPORT
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("verdict") != "kill-eta":
        raise RuntimeError(
            "P1 preregistration requires the sealed Stage-3 kill-eta report."
        )
    thresholds = Stage3EquivalenceThresholds()
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "experiment_id": "eta-stage3-equivalence-diagnostic-p1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "stage3-attribution-only-no-readjudication",
        "source_stage3_artifact": {
            "path": SOURCE_STAGE3_REPORT,
            "sha256": _sha256(source_report_path),
            "verdict": "kill-eta",
        },
        "configuration": {
            "model_id": model_id,
            "model_source": model_source,
            "device": device,
            "layer_indices": [20, 21, 22],
            "activation_width": 8,
            "seed_schedule": list(range(seeds)),
            "n_z": n_z,
            "alpha": alpha,
            "updates_per_run": updates,
            "learning_rate": learning_rate,
            "switch_threshold": switch_threshold,
            "reference_gate2_accuracy": reference_gate2_accuracy,
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
            "posterior_parameterization": "smooth",
            "rate_gating": "switch-gated",
            "gate_mode": "hard-st",
            "training_modes": ["full", "bias-only"],
            "control_ablations": ["zero-z", "cyclic-permuted-z"],
            "oracle_boundary": (
                "active_subgoal[t] != active_subgoal[t-1]; readout-only"
            ),
        },
        "thresholds": asdict(thresholds),
        "decision_rules": [
            (
                "The exact Stage-3 entry is readable iff heldout active-subgoal "
                "accuracy is at least entry_chance_multiple times uniform chance."
            ),
            (
                "The free-bias bypass is open iff matched bias-only recovery or "
                "zero-z recovery reaches its preregistered minimum."
            ),
            (
                "Learned z is causal iff zero-z recovery stays below its bypass "
                "threshold and cyclic permutation worsens distortion by at least "
                "permuted_z_penalty_min."
            ),
            (
                "Oracle active-subgoal boundaries are labels for readout only and "
                "must never enter the loss or threshold calibration."
            ),
        ],
        "prohibited_after_execution": [
            "changing any attribution threshold after reading a diagnostic point",
            "using P1 to change or relabel the sealed Stage-3 kill-eta verdict",
            "feeding probe, boundary, or evaluation labels into training",
            "changing any production WiringLevel",
            "promoting a faithful ETA rewrite without a new claim and preregistration",
        ],
        "frozen_source_files": {
            name: _sha256(_REPO_ROOT / name) for name in FROZEN_SOURCE_FILES
        },
    }


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze ETA Stage-3 P1 equivalence diagnostics."
    )
    parser.add_argument("--output", type=Path, required=True)
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
    payload = build_preregistration(
        model_id=args.model_id,
        model_source=args.model_source,
        device=args.device,
        seeds=args.seeds,
        n_z=args.n_z,
        alpha=args.alpha,
        updates=args.updates,
        learning_rate=args.learning_rate,
        switch_threshold=args.switch_threshold,
        reference_gate2_accuracy=args.reference_gate2_accuracy,
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
