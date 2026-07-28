from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import subprocess
import sys
import time

import yaml
import torch
import transformers

from volvence_zero.agent.eta_proof_benchmark import ETAOpenWeightRuntimeConfig
from volvence_zero.agent.eta_segment_credit_evidence import (
    export_eta_segment_credit_evidence,
    run_eta_segment_credit_evidence,
)


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ETA segment-level versus turn-level delayed-credit evidence."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--backend",
        choices=("transformers-open-weight", "trace"),
        default="transformers-open-weight",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--model-id",
        default=ETAOpenWeightRuntimeConfig().model_id,
        help="HF model id for the transformers-open-weight backend.",
    )
    parser.add_argument(
        "--model-source",
        default=None,
        help="Optional local model directory or alternate pretrained source.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads if the selected model is not cached.",
    )
    parser.add_argument("--max-observation-lag", type=int, default=3)
    parser.add_argument(
        "--training-mode",
        choices=("ssl-rl-alternating", "rl-only"),
        default="ssl-rl-alternating",
    )
    parser.add_argument("--training-cycles", type=int, default=3)
    parser.add_argument("--ssl-updates-per-cycle", type=int, default=1)
    parser.add_argument("--controller-dim", type=int, default=16)
    parser.add_argument("--ssl-alpha", type=float, default=0.1)
    parser.add_argument("--switch-prior", type=float, default=0.10)
    parser.add_argument("--switch-rate-weight", type=float, default=0.05)
    parser.add_argument("--switch-binary-weight", type=float, default=0.01)
    parser.add_argument("--switch-group-weight", type=float, default=0.01)
    parser.add_argument(
        "--proposal-prediction-weight",
        type=float,
        default=0.50,
    )
    parser.add_argument("--gate-choice-weight", type=float, default=1.0)
    parser.add_argument("--gate-choice-temperature", type=float, default=0.02)
    parser.add_argument("--prediction-horizon", type=int, default=3)
    parser.add_argument(
        "--distortion-target",
        choices=("absolute", "innovation"),
        default="innovation",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if args.max_observation_lag < 1:
        parser.error("--max-observation-lag must be at least 1")
    if args.training_cycles < 1:
        parser.error("--training-cycles must be at least 1")
    if args.ssl_updates_per_cycle < 1:
        parser.error("--ssl-updates-per-cycle must be at least 1")
    if args.controller_dim != 3 and args.controller_dim < 4:
        parser.error("--controller-dim must be 3 or at least 4")
    if args.training_mode == "ssl-rl-alternating" and args.controller_dim <= 3:
        parser.error("--training-mode ssl-rl-alternating requires --controller-dim >= 4")
    if args.ssl_alpha < 0.0:
        parser.error("--ssl-alpha must be non-negative")
    if not 0.0 < args.switch_prior < 1.0:
        parser.error("--switch-prior must be strictly between 0 and 1")
    if any(
        value < 0.0
        for value in (
            args.switch_rate_weight,
            args.switch_binary_weight,
            args.switch_group_weight,
            args.proposal_prediction_weight,
            args.gate_choice_weight,
        )
    ):
        parser.error("--switch loss weights must be non-negative")
    if args.prediction_horizon < 1:
        parser.error("--prediction-horizon must be at least 1")
    if args.gate_choice_temperature <= 0.0:
        parser.error("--gate-choice-temperature must be positive")

    started = time.perf_counter()
    config = ETAOpenWeightRuntimeConfig(
        model_id=args.model_id,
        model_source=args.model_source,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    report = run_eta_segment_credit_evidence(
        seed_schedule=tuple(range(args.seeds)),
        backend_label=args.backend,
        open_weight_config=config,
        max_observation_lag=args.max_observation_lag,
        training_mode=args.training_mode,
        training_cycles=args.training_cycles,
        ssl_updates_per_cycle=args.ssl_updates_per_cycle,
        controller_dim=args.controller_dim,
        ssl_alpha=args.ssl_alpha,
        switch_prior=args.switch_prior,
        switch_rate_weight=args.switch_rate_weight,
        switch_binary_weight=args.switch_binary_weight,
        switch_group_weight=args.switch_group_weight,
        proposal_prediction_weight=args.proposal_prediction_weight,
        gate_choice_weight=args.gate_choice_weight,
        gate_choice_temperature=args.gate_choice_temperature,
        prediction_horizon=args.prediction_horizon,
        distortion_target=args.distortion_target,
    )
    elapsed = time.perf_counter() - started
    written = export_eta_segment_credit_evidence(
        report,
        output_dir=args.output_dir,
    )
    manifest = {
        "schema_version": "eta-segment-credit-manifest.v12",
        "experiment_id": "eta-segment-credit-vs-turn",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "backend": args.backend,
        "model_id": config.model_id if args.backend == "transformers-open-weight" else "trace",
        "model_source": config.model_source,
        "local_files_only": config.local_files_only,
        "device": args.device if args.backend == "transformers-open-weight" else "cpu",
        "runtime_origin": report.runtime_origin,
        "fallback_active": report.fallback_active,
        "seed_schedule": list(report.seed_schedule),
        "controller_initialization_seed": (
            report.controller_initialization_seed
        ),
        "experience_seed_semantics": report.experience_seed_semantics,
        "max_observation_lag": args.max_observation_lag,
        "training_mode": report.training_mode,
        "training_cycles": report.training_cycles,
        "ssl_updates_per_cycle": report.ssl_updates_per_cycle,
        "controller_dim": report.controller_dim,
        "ssl_alpha": report.ssl_alpha,
        "switch_prior": report.switch_prior,
        "switch_rate_weight": report.switch_rate_weight,
        "switch_binary_weight": report.switch_binary_weight,
        "switch_group_weight": report.switch_group_weight,
        "proposal_prediction_weight": report.proposal_prediction_weight,
        "gate_choice_weight": report.gate_choice_weight,
        "gate_choice_temperature": report.gate_choice_temperature,
        "prediction_horizon": report.prediction_horizon,
        "distortion_target": report.distortion_target,
        "ssl_supervision_target": report.ssl_supervision_target,
        "expert_action_supervision": report.expert_action_supervision,
        "outcome_target": report.outcome_target,
        "rollout_replacement_mode": report.rollout_replacement_mode,
        "temporal_fast_prior_enabled": report.temporal_fast_prior_enabled,
        "episode_recurrent_state_isolated": (
            report.episode_recurrent_state_isolated
        ),
        "elapsed_seconds": elapsed,
        "claim_status": report.claim_status,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "mps_available": bool(torch.backends.mps.is_available()),
        "artifact_files": [path.name for path in written],
    }
    manifest_path = args.output_dir / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    interval_map = {
        interval.metric_name: asdict(interval)
        for interval in report.metric_intervals
    }
    summary = {
        "elapsed_seconds": round(elapsed, 3),
        "claim_status": report.claim_status,
        "credit_f1_delta": interval_map["credit_f1_delta"],
        "false_credit_reduction": interval_map["false_credit_reduction"],
        "family_assignment_delta": interval_map["family_assignment_delta"],
        "pe_reduction_rate_delta": interval_map["pe_reduction_rate_delta"],
        "segment_boundary_f1": interval_map["segment_boundary_f1"],
        "mean_active_family_count": sum(
            row.active_family_count for row in report.run_metrics
        ) / len(report.run_metrics),
        "beta_boundary_count": sum(
            row.beta_boundary_count for row in report.run_metrics
        ),
        "true_boundary_count": sum(
            row.true_boundary_count for row in report.run_metrics
        ),
        "mean_ssl_trained_steps": sum(
            row.ssl_trained_step_count for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_ssl_prediction_loss": sum(
            row.ssl_prediction_loss_mean for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_ssl_kl_loss": sum(
            row.ssl_kl_loss_mean for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_ssl_switch_frequency": sum(
            row.ssl_switch_frequency_mean for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_final_ssl_switch_frequency": sum(
            row.ssl_switch_frequency_final for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_ssl_switch_probability": sum(
            row.ssl_switch_probability_mean for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_final_ssl_switch_probability": sum(
            row.ssl_switch_probability_final for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_ssl_switch_rate_loss": sum(
            row.ssl_switch_rate_loss_mean for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_ssl_gate_choice_loss": sum(
            row.ssl_gate_choice_loss_mean for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_ssl_target_variance": sum(
            row.ssl_target_variance_mean for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_final_ssl_action_boundary_f1": sum(
            row.ssl_action_boundary_f1_final for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_final_ssl_boundary_switch_probability": sum(
            row.ssl_boundary_switch_probability_final
            for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_final_ssl_continuation_switch_probability": sum(
            row.ssl_continuation_switch_probability_final
            for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_final_ssl_switch_threshold": sum(
            row.ssl_switch_threshold_final for row in report.run_metrics
        ) / len(report.run_metrics),
        "mean_final_runtime_switch_threshold": sum(
            row.runtime_switch_threshold_final for row in report.run_metrics
        ) / len(report.run_metrics),
        "ssl_optimizer_final_step_min": min(
            row.ssl_optimizer_final_step for row in report.run_metrics
        ),
        "ssl_optimizer_reuse_count_min": min(
            row.ssl_optimizer_reuse_count for row in report.run_metrics
        ),
        "ssl_writeback_count": sum(
            row.ssl_writeback_count for row in report.run_metrics
        ),
        "retain_gates": report.retain_gates,
        "output_dir": str(args.output_dir.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
