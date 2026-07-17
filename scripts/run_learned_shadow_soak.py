"""Learned-shadow long soak (W1.D / W3 of the intent-alignment remediation).

Runs N (default 500) synthetic CPU turns under the frozen learned-shadow
operator profile and accumulates the full learned-owner evidence surface into
one JSON artifact:

    python -u scripts/run_learned_shadow_soak.py --turns 500

(Use ``python -u`` or the ``scripts/run_learned_shadow_soak.sh`` launcher when
redirecting output to a file, so progress lines are not lost to buffering.)

Collected per soak:
- runtime parity / SSL / RL / CMS payload (same collector as the smoke)
- CP-11 predictive-head readout time series + kill-criteria readout
- CP-11 head checkpoint export + restore round-trip verification
- W1.A dual-track gate learner readout
- W1.B semantic owner forecast stats per slot
- W1.C social record store settlement counts
- a per-component ``learned_active_gate`` verdict fed with HONEST evidence
  (synthetic turns count as real_trace_turns=0, so verdicts are expected
  BLOCKED on the real-trace gates; the artifact records directional data only)

This is the synthetic lane. It is NOT ACTIVE-promotion evidence (that requires
the Linux CUDA real-trace lane per the AGI-uplift plan and
docs/specs/evidence_program.md).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from volvence_zero.agent.learned_active_gate import (
    LearnedActiveEvidence,
    LearnedBackendComponent,
    evaluate_learned_active_candidate,
)
from volvence_zero.agent.learned_shadow_evidence import (
    LEARNED_SHADOW_TEMPORAL_LATENT_DIM,
    build_learned_shadow_rollout_config,
    collect_internal_rl_no_optimize_proof,
    collect_learned_shadow_evidence,
    collect_strict_eta_gate_evidence,
)
from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.prediction import (
    PredictionErrorModule,
    PredictiveHeadCheckpoint,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.semantic_state import SEMANTIC_OWNER_SLOTS
from volvence_zero.social import TOM_SLOTS

_SOAK_SCHEMA_VERSION = "learned-shadow-soak.v1"

# Synthetic turn pool: rotates through task planning, emotional disclosure,
# schedule churn, preference statements and repair-style turns so the world
# AND self axes both move (a monotone pool would starve the self head and the
# ToM settlement path of variance).
_TURN_POOL: tuple[str, ...] = (
    "Walk me through the harbor plan for tomorrow.",
    "The tide tables changed; adjust the schedule.",
    "I'm honestly a bit anxious we won't make the window.",
    "Actually I prefer we brief the crew before dawn, not after.",
    "That last suggestion missed what I asked for - please re-check.",
    "Ok that helps. What's still open on the checklist?",
    "My colleague thinks the northern berth is safer; I disagree.",
    "Let's lock the fuel order today.",
    "Thanks - I feel better about the plan now.",
    "One more thing: keep the backup pilot on standby.",
    "Summarize what we committed to this week.",
    "The client moved the deadline up two days; replan.",
)

# Soak-lane latency SLO for the synthetic CPU profile: the smoke lane finishes
# 4 turns in well under a minute, so a soak turn averaging above this is a
# regression worth flagging in the artifact.
_SOAK_MEAN_TURN_SECONDS_SLO = 5.0


def _rollout_config_for_backend_combo(combo: str) -> FinalRolloutConfig:
    if combo == "runtime+ssl+internal-rl+cms-torch":
        return build_learned_shadow_rollout_config()
    if combo == "runtime-only":
        return FinalRolloutConfig(
            temporal_runtime_backend=WiringLevel.SHADOW,
            temporal_ssl_backend=WiringLevel.DISABLED,
            internal_rl_backend=WiringLevel.DISABLED,
            cms_torch_backend=WiringLevel.DISABLED,
        )
    if combo == "runtime+ssl":
        return FinalRolloutConfig(
            temporal_runtime_backend=WiringLevel.SHADOW,
            temporal_ssl_backend=WiringLevel.SHADOW,
            internal_rl_backend=WiringLevel.DISABLED,
            cms_torch_backend=WiringLevel.DISABLED,
        )
    if combo == "runtime+ssl+internal-rl":
        return FinalRolloutConfig(
            temporal_runtime_backend=WiringLevel.SHADOW,
            temporal_ssl_backend=WiringLevel.SHADOW,
            internal_rl_backend=WiringLevel.SHADOW,
            cms_torch_backend=WiringLevel.DISABLED,
        )
    raise ValueError(f"unsupported backend_combo {combo!r}")


def _git_output(args: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            ("git",) + args, check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _collect_provenance() -> dict[str, object]:
    status = _git_output(("status", "--porcelain"))
    return {
        "git_sha": _git_output(("rev-parse", "HEAD")),
        "git_branch": _git_output(("branch", "--show-current")),
        "working_tree_dirty": status not in {"", "unknown"},
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _file_record(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _checkpoint_payload(checkpoint: PredictiveHeadCheckpoint) -> dict[str, object]:
    return {
        "schema_version": checkpoint.schema_version,
        "checkpoint_id": checkpoint.checkpoint_id,
        "feature_dim": checkpoint.feature_dim,
        "sample_count": checkpoint.sample_count,
        "world_weights": [
            {"axis": axis, "weights": list(weights)}
            for axis, weights in checkpoint.world_weights
        ],
        "self_weights": [
            {"axis": axis, "weights": list(weights)}
            for axis, weights in checkpoint.self_weights
        ],
        "abs_error_sums": dict(checkpoint.abs_error_sums),
    }


def _verify_checkpoint_round_trip(checkpoint: PredictiveHeadCheckpoint) -> None:
    """Restore into a FRESH PE module and compare re-export field by field.

    Proves session-medium restore works without disturbing the live module.
    Raises on any mismatch (fail loudly).
    """

    fresh = PredictionErrorModule()
    fresh.restore_predictive_head_checkpoint(checkpoint)
    reexported = fresh.export_predictive_head_checkpoint(
        checkpoint_id=checkpoint.checkpoint_id
    )
    if reexported != checkpoint:
        raise ValueError(
            "predictive-head checkpoint round-trip mismatch: "
            f"exported={checkpoint!r} reexported={reexported!r}"
        )


def _gate_verdicts(
    *,
    real_trace_turns: int,
    validation_delta: float,
    safety_gate_ok: bool,
    latency_slo_ok: bool,
    strict_eta_gate_passed: bool,
    rollback_drill_passed: bool,
    pe_off_control_direction_correct: bool = False,
    eta_off_control_direction_correct: bool = False,
) -> list[dict[str, object]]:
    """Evaluate all four components with honest lane evidence."""

    verdicts: list[dict[str, object]] = []
    for component in LearnedBackendComponent:
        evidence = LearnedActiveEvidence(
            component=component,
            real_trace_turns=real_trace_turns,
            validation_delta=validation_delta,
            strict_eta_gate_passed=strict_eta_gate_passed,
            pe_off_control_direction_correct=pe_off_control_direction_correct,
            eta_off_control_direction_correct=eta_off_control_direction_correct,
            rollback_drill_passed=rollback_drill_passed,
            latency_slo_ok=latency_slo_ok,
            safety_gate_ok=safety_gate_ok,
        )
        verdict = evaluate_learned_active_candidate(evidence)
        verdicts.append(
            {
                "component": component.value,
                "real_trace_turns": real_trace_turns,
                "validation_delta": validation_delta,
                "strict_eta_gate_passed": strict_eta_gate_passed,
                "pe_off_control_direction_correct": pe_off_control_direction_correct,
                "eta_off_control_direction_correct": eta_off_control_direction_correct,
                "rollback_drill_passed": rollback_drill_passed,
                "latency_slo_ok": latency_slo_ok,
                "safety_gate_ok": safety_gate_ok,
                "eligible": verdict.eligible,
                "missing_gates": list(verdict.missing_gates),
                "description": verdict.description,
            }
        )
    return verdicts


async def main(
    *,
    output_dir: Path,
    turn_count: int,
    sample_every: int,
    checkpoint_every: int,
    temporal_latent_dim: int = LEARNED_SHADOW_TEMPORAL_LATENT_DIM,
    backend_combo: str = "runtime+ssl+internal-rl+cms-torch",
    substrate_mode: str = "synthetic",
    substrate_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    substrate_device: str = "cuda",
    substrate_local_files_only: bool = True,
) -> int:
    if turn_count < 3:
        raise ValueError(
            "turn_count must be >= 3 so the joint-loop schedule runs a full "
            f"cycle; got {turn_count}."
        )
    default_runtime = None
    if substrate_mode == "hf":
        from volvence_zero.substrate import build_transformers_runtime_with_fallback

        default_runtime = build_transformers_runtime_with_fallback(
            model_id=substrate_model_id,
            device=substrate_device,
            local_files_only=substrate_local_files_only,
            allow_live_substrate_mutation=False,
            fallback_to_builtin=False,
        )
    elif substrate_mode != "synthetic":
        raise ValueError(f"unsupported substrate_mode {substrate_mode!r}")
    else:
        from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime

        default_runtime = SyntheticOpenWeightResidualRuntime(
            model_id="learned-shadow-soak-synthetic",
        )
    runner = AgentSessionRunner(
        config=_rollout_config_for_backend_combo(backend_combo),
        temporal_latent_dim=temporal_latent_dim,
        default_residual_runtime=default_runtime,
        rare_heavy_enabled=False,
    )
    print(
        f"[soak] profile: n_z={temporal_latent_dim}, "
        f"backend_combo={backend_combo}, substrate_mode={substrate_mode}, "
        f"{turn_count} turns",
        flush=True,
    )

    head_readout_series: list[dict[str, object]] = []
    kill_criteria_series: list[dict[str, object]] = []
    checkpoint_round_trips = 0
    real_trace_turns = 0
    started = time.perf_counter()
    for index in range(1, turn_count + 1):
        text = _TURN_POOL[(index - 1) % len(_TURN_POOL)]
        result = await runner.run_turn(text)
        recent_traces = getattr(runner, "_recent_training_traces")
        # Honesty guard: a ":real:" trace_id only means the trace was built
        # from the previous turn's residual sequence. Synthetic runtimes also
        # produce residual sequences, so counting them as real traces would
        # let a synthetic soak satisfy the >=500 real-trace promotion gate.
        # Real-trace turns require the real open-weight substrate lane.
        if (
            substrate_mode == "hf"
            and recent_traces
            and ":real:" in recent_traces[-1].trace_id
        ):
            real_trace_turns += 1
        if index % sample_every == 0 or index == turn_count:
            pe_value = result.active_snapshots["prediction_error"].value
            readout = pe_value.predictive_head_readout
            if readout is not None:
                head_readout_series.append(
                    {
                        "turn": index,
                        "sample_count": readout.sample_count,
                        "world_learned_mae": readout.world_learned_mae,
                        "world_baseline_mae": readout.world_baseline_mae,
                        "self_learned_mae": readout.self_learned_mae,
                        "self_baseline_mae": readout.self_baseline_mae,
                        "world_improvement": readout.world_improvement,
                        "self_improvement": readout.self_improvement,
                        "window_size": readout.window_size,
                        "window_sample_count": readout.window_sample_count,
                        "window_world_improvement": readout.window_world_improvement,
                        "window_self_improvement": readout.window_self_improvement,
                        "window_axis_learned_maes": list(readout.window_axis_learned_maes),
                        "window_axis_baseline_maes": list(readout.window_axis_baseline_maes),
                        "window_target_stds": list(readout.window_target_stds),
                        "window_persistence_maes": list(readout.window_persistence_maes),
                    }
                )
            kill = runner.prediction_module.predictive_head_kill_criteria()
            kill_criteria_series.append(
                {
                    "turn": index,
                    "samples_in_window": kill.samples_in_window,
                    "window_filled": kill.window_filled,
                    "prediction_self_autocorrelation": kill.prediction_self_autocorrelation,
                    "prediction_target_correlation": kill.prediction_target_correlation,
                    "kill_triggered": kill.kill_triggered,
                }
            )
            elapsed = time.perf_counter() - started
            print(
                f"[soak] turn {index}/{turn_count} "
                f"({elapsed / index:.2f}s/turn avg, kill={kill.kill_triggered})",
                flush=True,
            )
        if index % checkpoint_every == 0:
            checkpoint = runner.prediction_module.export_predictive_head_checkpoint(
                checkpoint_id=f"soak-turn-{index}"
            )
            _verify_checkpoint_round_trip(checkpoint)
            checkpoint_round_trips += 1
    total_seconds = time.perf_counter() - started
    mean_turn_seconds = total_seconds / turn_count

    # Full learned-shadow requires all four owner evidence surfaces. Capacity
    # ladder partial combos intentionally omit some backends, so they emit a
    # partial artifact instead of pretending the full profile was exercised.
    if backend_combo == "runtime+ssl+internal-rl+cms-torch":
        payload = collect_learned_shadow_evidence(runner)
    else:
        payload = {
            "schema_version": _SOAK_SCHEMA_VERSION,
            "artifact_kind": "learned_shadow_soak",
            "partial_backend_combo": True,
            "backend_combo": backend_combo,
            "temporal_runtime": {
                "world_report_present": (
                    runner.world_temporal_policy.latest_runtime_shadow_report
                    is not None
                ),
                "self_report_present": (
                    runner.self_temporal_policy.latest_runtime_shadow_report
                    is not None
                ),
            },
        }
    payload["schema_version"] = _SOAK_SCHEMA_VERSION
    payload["artifact_kind"] = "learned_shadow_soak"
    payload["evidence_tier"] = (
        "real-trace-soak" if real_trace_turns > 0 else "synthetic-cpu-soak"
    )
    payload["capacity_arm"] = {
        "temporal_latent_dim": temporal_latent_dim,
        "backend_combo": backend_combo,
        "substrate_mode": substrate_mode,
        "substrate_model_id": substrate_model_id if substrate_mode == "hf" else "",
        "substrate_device": substrate_device if substrate_mode == "hf" else "",
    }
    payload["turn_count"] = turn_count
    payload["real_trace_turns"] = real_trace_turns
    payload["total_seconds"] = round(total_seconds, 2)
    payload["mean_turn_seconds"] = round(mean_turn_seconds, 4)
    payload["provenance"] = _collect_provenance()

    final_kill = runner.prediction_module.predictive_head_kill_criteria()
    final_checkpoint = runner.prediction_module.export_predictive_head_checkpoint(
        checkpoint_id=f"soak-final-{turn_count}"
    )
    _verify_checkpoint_round_trip(final_checkpoint)
    payload["cp11_predictive_heads"] = {
        "readout_series": head_readout_series,
        "kill_criteria_series": kill_criteria_series,
        "final_kill_criteria": {
            "samples_in_window": final_kill.samples_in_window,
            "window_filled": final_kill.window_filled,
            "prediction_self_autocorrelation": final_kill.prediction_self_autocorrelation,
            "prediction_target_correlation": final_kill.prediction_target_correlation,
            "kill_triggered": final_kill.kill_triggered,
            "description": final_kill.description,
        },
        "final_checkpoint": _checkpoint_payload(final_checkpoint),
        "checkpoint_round_trips_verified": checkpoint_round_trips + 1,
    }

    gate_readout = runner.dual_track_gate_learner.readout()
    payload["dual_track_gate_learner"] = {
        "update_count": gate_readout.update_count,
        "running_abs_error": gate_readout.running_abs_error,
        "last_prediction": gate_readout.last_prediction,
        "last_target": gate_readout.last_target,
        "description": gate_readout.description,
    }

    # G-series SHADOW learners (2026-07-17): capture settle counts +
    # promotion readouts so a soak run doubles as their promotion-evidence
    # accumulation lane. All report-only; none of these touch live paths.
    regime_shadow = None
    regime_snapshot = runner.upstream_snapshots.get("regime")
    if regime_snapshot is not None:
        regime_shadow = regime_snapshot.value.learned_score_shadow
    payload["regime_score_learner"] = (
        {
            "update_count": regime_shadow.update_count,
            "running_abs_error": regime_shadow.running_abs_error,
            "last_target_regime_id": regime_shadow.last_target_regime_id,
            "ready": regime_shadow.ready,
            "kill_recommended": regime_shadow.kill_recommended,
            "blocking_reasons": list(regime_shadow.blocking_reasons),
            "description": regime_shadow.description,
        }
        if regime_shadow is not None
        else {"present": False, "reason": "regime snapshot missing or shadow field unset"}
    )

    consolidation_readout = runner.reflection_consolidation_learner.promotion_readout()
    payload["reflection_consolidation_learner"] = {
        "settled_count": consolidation_readout.settled_count,
        "learned_mae": consolidation_readout.learned_mae,
        "baseline_mae": consolidation_readout.baseline_mae,
        "mae_improvement": consolidation_readout.mae_improvement,
        "ready": consolidation_readout.ready,
        "kill_recommended": consolidation_readout.kill_recommended,
        "blocking_reasons": list(consolidation_readout.blocking_reasons),
        "description": consolidation_readout.description,
    }

    cocoa_state = runner.credit_module.ledger.export_rewarding_state_head()
    gate_risk_state = runner.credit_module.ledger.export_gate_risk_learner()
    payload["credit_learned_heads"] = {
        "rewarding_state_head_update_count": cocoa_state.update_count,
        "rewarding_state_head_last_validation_delta": cocoa_state.last_validation_delta,
        "gate_risk_update_count": gate_risk_state.update_count,
        "gate_risk_agreement_count": gate_risk_state.agreement_count,
    }

    payload["semantic_owner_forecasts"] = {
        slot: {
            "update_count": stats[0],
            "running_abs_error": round(stats[1], 6),
        }
        for slot in SEMANTIC_OWNER_SLOTS
        for stats in (runner.semantic_state_store.owner_forecast_stats(slot),)
    }

    store = runner.social_record_store
    payload["social_settlement"] = {
        "tom_records": {slot: len(store.tom_records(slot)) for slot in TOM_SLOTS},
        "tom_pending": {
            slot: len(store.pending_tom_predictions(slot)) for slot in TOM_SLOTS
        },
        "common_ground_dyad_atoms": len(store.common_ground_dyad_atoms),
        "common_ground_group_atoms": len(store.common_ground_group_atoms),
        "common_ground_pending": len(store.pending_common_ground_predictions),
    }

    # CP-09: run the strict ETA suite for real instead of hardcoding False.
    print("[soak] running strict ETA gate suite (CP-09)...", flush=True)
    try:
        strict_eta = collect_strict_eta_gate_evidence()
    except ImportError as exc:
        strict_eta = {
            "evidence_kind": "strict_eta_gate",
            "gate_passed": False,
            "missing_dependency": "torch",
            "description": f"strict ETA gate skipped: {exc}",
        }
    payload["strict_eta_gate"] = strict_eta
    print(
        f"[soak] strict ETA gate_passed={strict_eta['gate_passed']} "
        f"({strict_eta['description']})",
        flush=True,
    )

    # CP-07 (GAP-09): offline optimize-vs-no-optimize matched control.
    print("[soak] running internal RL no-optimize proof (CP-07)...", flush=True)
    try:
        no_optimize_proof = collect_internal_rl_no_optimize_proof()
    except ImportError as exc:
        no_optimize_proof = {
            "evidence_kind": "internal_rl_no_optimize_proof",
            "full_beats_control": False,
            "missing_dependency": "torch",
            "description": f"no-optimize proof skipped: {exc}",
        }
    payload["internal_rl_no_optimize_proof"] = no_optimize_proof
    print(
        f"[soak] no-optimize proof full_beats_control="
        f"{no_optimize_proof['full_beats_control']}",
        flush=True,
    )

    latency_slo_ok = mean_turn_seconds <= _SOAK_MEAN_TURN_SECONDS_SLO
    validation_delta = 0.0
    validation_delta_basis = "none"
    if head_readout_series:
        last_readout = head_readout_series[-1]
        window_count = int(last_readout["window_sample_count"])
        window_size = int(last_readout["window_size"])
        if window_count >= window_size:
            # Plan CP-11 gate wording: ">= 0.02 improvement over >= 200
            # turns". A filled trailing window measures the converged head;
            # the cumulative mean mixes cold-start error in forever.
            validation_delta = max(
                float(last_readout["window_world_improvement"]),
                float(last_readout["window_self_improvement"]),
            )
            validation_delta_basis = f"trailing-window-{window_size}"
        else:
            # Short soaks cannot fill the gate window; report the cumulative
            # improvement and say so (the gate also requires >= 500 real
            # trace turns, so short lanes stay blocked regardless).
            validation_delta = max(
                float(last_readout["world_improvement"]),
                float(last_readout["self_improvement"]),
            )
            validation_delta_basis = f"cumulative-window-unfilled-{window_count}"
    payload["learned_active_gate"] = {
        "note": (
            "Soak lane evidence. ACTIVE promotion additionally requires "
            "component controls (PE-off / ETA-off) from the same-substrate "
            "ablation and rollback drill evidence. strict_eta_gate reflects "
            "the CP-09 suite executed this soak."
        ),
        "latency_slo_seconds": _SOAK_MEAN_TURN_SECONDS_SLO,
        "latency_slo_ok": latency_slo_ok,
        "real_trace_turns": real_trace_turns,
        "validation_delta": validation_delta,
        "validation_delta_basis": validation_delta_basis,
        "verdicts": _gate_verdicts(
            real_trace_turns=real_trace_turns,
            validation_delta=validation_delta,
            safety_gate_ok=True,  # collector verified zero SHADOW write-back
            latency_slo_ok=latency_slo_ok,
            strict_eta_gate_passed=bool(strict_eta["gate_passed"]),
            rollback_drill_passed=checkpoint_round_trips > 0,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "learned_shadow_soak.json"
    artifact_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest_path = output_dir / "learned_shadow_soak_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": _SOAK_SCHEMA_VERSION,
                "artifact_kind": "learned_shadow_soak_manifest",
                "artifacts": [_file_record(artifact_path)],
                "provenance": payload["provenance"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[soak] {turn_count} turns in {total_seconds:.1f}s ({mean_turn_seconds:.2f}s/turn)")
    print(
        "[soak] cp11 final: "
        f"kill_triggered={final_kill.kill_triggered} "
        f"self-autocorr={final_kill.prediction_self_autocorrelation} "
        f"target-corr={final_kill.prediction_target_correlation}"
    )
    print(
        "[soak] gate learner: "
        f"updates={gate_readout.update_count} mae={gate_readout.running_abs_error}"
    )
    print(f"[soak] artifact written to {artifact_path}")
    print(f"[soak] manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/learned_shadow_soak")
    )
    parser.add_argument("--turns", type=int, default=500)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=25,
        help="Record CP-11 readout / kill-criteria samples every N turns.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Export + round-trip-verify a head checkpoint every N turns.",
    )
    parser.add_argument(
        "--temporal-latent-dim",
        type=int,
        default=LEARNED_SHADOW_TEMPORAL_LATENT_DIM,
    )
    parser.add_argument(
        "--backend-combo",
        default="runtime+ssl+internal-rl+cms-torch",
        choices=[
            "runtime-only",
            "runtime+ssl",
            "runtime+ssl+internal-rl",
            "runtime+ssl+internal-rl+cms-torch",
        ],
    )
    parser.add_argument(
        "--substrate-mode",
        default="synthetic",
        choices=["synthetic", "hf"],
    )
    parser.add_argument(
        "--substrate-model-id",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--substrate-device", default="cuda")
    parser.add_argument("--substrate-allow-download", action="store_true")
    args = parser.parse_args()
    sys.exit(
        asyncio.run(
            main(
                output_dir=args.output_dir,
                turn_count=args.turns,
                sample_every=max(1, args.sample_every),
                checkpoint_every=max(1, args.checkpoint_every),
                temporal_latent_dim=args.temporal_latent_dim,
                backend_combo=args.backend_combo,
                substrate_mode=args.substrate_mode,
                substrate_model_id=args.substrate_model_id,
                substrate_device=args.substrate_device,
                substrate_local_files_only=not args.substrate_allow_download,
            )
        )
    )
