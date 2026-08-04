"""Freeze the Branch-B faithful ETA directional screen before execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from volvence_zero.agent.eta_faithful_rewrite_screen import (
    FaithfulETAScreenThresholds,
)
from volvence_zero.substrate import fingerprint_model_weight_files


PREREGISTRATION_SCHEMA_VERSION = "eta-faithful-rewrite-screen-prereg.v1"
_REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCE_STAGE3_REPORT = (
    "artifacts/eta_stage3_rate_distortion_20260803/report.json"
)
SOURCE_P1_REPORT = (
    "artifacts/eta_stage3_equivalence_diagnostic_20260804/report.json"
)
SOURCE_S2_REPORT = "artifacts/eta_s2_causal_steering_20260804/report.json"
SOURCE_S2_MANIFEST = (
    "artifacts/eta_s2_causal_steering_20260804/artifact_manifest.json"
)
MODEL_SOURCE = "artifacts/eta_stage2_merged_v2_20260803"

FROZEN_SOURCE_FILES = (
    "packages/vz-substrate/src/volvence_zero/substrate/residual_contracts.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "packages/vz-substrate/src/volvence_zero/substrate/steered_action_scoring.py",
    "packages/vz-temporal/src/volvence_zero/temporal/metacontroller_components.py",
    "packages/vz-temporal/src/volvence_zero/temporal/torch_store_ssl.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_rate_distortion_evidence.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_faithful_rewrite_screen.py",
    "scripts/run_eta_faithful_rewrite_screen.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_lineage() -> dict[str, object]:
    stage3_path = _REPO_ROOT / SOURCE_STAGE3_REPORT
    p1_path = _REPO_ROOT / SOURCE_P1_REPORT
    s2_path = _REPO_ROOT / SOURCE_S2_REPORT
    s2_manifest_path = _REPO_ROOT / SOURCE_S2_MANIFEST
    stage3 = json.loads(stage3_path.read_text(encoding="utf-8"))
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    s2 = json.loads(s2_path.read_text(encoding="utf-8"))
    s2_manifest = json.loads(s2_manifest_path.read_text(encoding="utf-8"))
    if stage3.get("verdict") != "kill-eta":
        raise RuntimeError("Faithful rewrite requires sealed Stage-3 kill-eta.")
    if p1.get("attribution", {}).get("dominant_attribution") != (
        "incentive-bypass-via-free-bias"
    ):
        raise RuntimeError("Faithful rewrite requires the P1 free-bias finding.")
    if s2.get("primary_admission", {}).get("admitted") is not False:
        raise RuntimeError("Faithful rewrite requires the preregistered S2 FAIL.")
    if s2_manifest.get("s2_causal_supported") is not False:
        raise RuntimeError("S2 manifest does not close the residual-axis route.")
    return {
        "stage3_report_path": SOURCE_STAGE3_REPORT,
        "stage3_report_sha256": _sha256(stage3_path),
        "stage3_verdict": "kill-eta",
        "p1_report_path": SOURCE_P1_REPORT,
        "p1_report_sha256": _sha256(p1_path),
        "p1_dominant_attribution": "incentive-bypass-via-free-bias",
        "s2_report_path": SOURCE_S2_REPORT,
        "s2_report_sha256": _sha256(s2_path),
        "s2_manifest_path": SOURCE_S2_MANIFEST,
        "s2_manifest_sha256": _sha256(s2_manifest_path),
        "s2_causal_supported": False,
    }


def build_preregistration(
    *,
    device: str,
    max_length: int,
    alpha_grid: tuple[float, ...],
    primary_alpha: float,
    seed_schedule: tuple[int, ...],
    updates_per_run: int,
    learning_rate: float,
    switch_threshold: float,
    n_z: int,
    injection_layer_index: int,
    residual_width: int,
    steering_rank: int,
    control_norm_ratio: float,
    screen_train_routes: int,
    screen_heldout_routes: int,
    corpus_seed: int,
    objective_count: int,
    corridor_count: int,
    extra_edge_probability: float,
    train_routes: int,
    heldout_routes: int,
    train_lengths: tuple[int, ...],
    heldout_lengths: tuple[int, ...],
) -> dict[str, object]:
    model_root = _REPO_ROOT / MODEL_SOURCE
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "experiment_id": "eta-faithful-rewrite-directional-screen",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "faithful-eta-rewrite-directional-screen",
        "source_lineage": _source_lineage(),
        "configuration": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_source": MODEL_SOURCE,
            "model_weights_sha256": fingerprint_model_weight_files(model_root),
            "device": device,
            "model_dtype": "float32",
            "max_length": max_length,
            "token_budget_policy": "fail-loud-preflight.v1",
            "observation_protocol": "partially-observable-staged-plan.v4",
            "observation_surface": "stage2-v4-cumulative-causal-prefix",
            "action_prompt_suffix": "\nNext move:",
            "scorer_internal_prompt_suffix": "",
            "capture_injection_alignment": (
                "controller e_t and steering injection use the identical scored "
                "prefix ending at the action-prompt token"
            ),
            "screen_selection": "first-generated-routes-before-screen-outcomes",
            "screen_train_routes": screen_train_routes,
            "screen_heldout_routes": screen_heldout_routes,
            "alpha_grid": list(alpha_grid),
            "primary_alpha": primary_alpha,
            "seed_schedule": list(seed_schedule),
            "updates_per_run": updates_per_run,
            "learning_rate": learning_rate,
            "switch_threshold": switch_threshold,
            "n_z": n_z,
            "injection_layer_index": injection_layer_index,
            "residual_width": residual_width,
            "steering_rank": steering_rank,
            "steering_parameterization": "low-rank-multiplicative",
            "low_rank_equation": "A @ diag(tanh(C @ z_t)) @ B.T @ e_t",
            "free_steering_bias": False,
            "zero_code_strict_noop": True,
            "current_observation_mode": "learned-projection",
            "posterior_parameterization": "smooth",
            "rate_gating": "switch-gated",
            "gate_mode": "hard-st",
            "control_norm_ratio": control_norm_ratio,
            "substrate_training": False,
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
        },
        "thresholds": asdict(FaithfulETAScreenThresholds()),
        "decision_rules": [
            "Only alpha=0.30 adjudicates z causality and boundary alignment.",
            "All cells must update the learned input projection and low-rank factors.",
            "Every cell must attest no free bias and exact zero-code no-op.",
            "Zero-z and cyclic-permuted-z penalties must clear 0.02 at alpha=0.30.",
            "Both primary-alpha seeds must have positive zero/permuted penalties.",
            "Oracle active-subgoal boundaries are readout-only and never train beta.",
            "Passing admits only a separately preregistered authoritative sweep.",
        ],
        "prohibited_after_execution": [
            "changing alpha, primary alpha, seeds, updates, learning rate, or thresholds",
            "selecting the best alpha after reading outcomes",
            "adding an additive or constant steering bias",
            "folding or truncating the 896-wide controller input to n_z",
            "capturing e_t from a different token position than steering injection",
            "using active_subgoal identity or boundary labels in the training loss",
            "training the substrate or feeding evaluation admission into learning",
            "relabeling the sealed Stage-3 verdict",
            "installing the screen controller or changing production WiringLevel",
        ],
        "frozen_source_files": {
            name: _sha256(_REPO_ROOT / name) for name in FROZEN_SOURCE_FILES
        },
    }


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the faithful ETA rewrite directional screen."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument(
        "--alpha-grid", type=float, nargs="+", default=(0.03, 0.30, 3.00)
    )
    parser.add_argument("--primary-alpha", type=float, default=0.30)
    parser.add_argument("--seed-schedule", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--updates-per-run", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--switch-threshold", type=float, default=0.55)
    parser.add_argument("--n-z", type=int, default=16)
    parser.add_argument("--injection-layer-index", type=int, default=20)
    parser.add_argument("--residual-width", type=int, default=896)
    parser.add_argument("--steering-rank", type=int, default=8)
    parser.add_argument("--control-norm-ratio", type=float, default=0.25)
    parser.add_argument("--screen-train-routes", type=int, default=16)
    parser.add_argument("--screen-heldout-routes", type=int, default=8)
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
        alpha_grid=tuple(args.alpha_grid),
        primary_alpha=args.primary_alpha,
        seed_schedule=tuple(args.seed_schedule),
        updates_per_run=args.updates_per_run,
        learning_rate=args.learning_rate,
        switch_threshold=args.switch_threshold,
        n_z=args.n_z,
        injection_layer_index=args.injection_layer_index,
        residual_width=args.residual_width,
        steering_rank=args.steering_rank,
        control_norm_ratio=args.control_norm_ratio,
        screen_train_routes=args.screen_train_routes,
        screen_heldout_routes=args.screen_heldout_routes,
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
