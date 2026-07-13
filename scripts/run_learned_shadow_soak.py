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
    collect_learned_shadow_evidence,
)
from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.prediction import (
    PredictionErrorModule,
    PredictiveHeadCheckpoint,
)
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
    *, safety_gate_ok: bool, latency_slo_ok: bool
) -> list[dict[str, object]]:
    """Evaluate all four components with honest synthetic-lane evidence.

    real_trace_turns=0 because synthetic soak turns are NOT real traces;
    control/rollback/eta gates are False because this lane does not run them.
    Verdicts are therefore expected BLOCKED - that is the honest record.
    """

    verdicts: list[dict[str, object]] = []
    for component in LearnedBackendComponent:
        evidence = LearnedActiveEvidence(
            component=component,
            real_trace_turns=0,
            validation_delta=0.0,
            strict_eta_gate_passed=False,
            pe_off_control_direction_correct=False,
            eta_off_control_direction_correct=False,
            rollback_drill_passed=False,
            latency_slo_ok=latency_slo_ok,
            safety_gate_ok=safety_gate_ok,
        )
        verdict = evaluate_learned_active_candidate(evidence)
        verdicts.append(
            {
                "component": component.value,
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
) -> int:
    if turn_count < 3:
        raise ValueError(
            "turn_count must be >= 3 so the joint-loop schedule runs a full "
            f"cycle; got {turn_count}."
        )
    runner = AgentSessionRunner(
        config=build_learned_shadow_rollout_config(),
        temporal_latent_dim=LEARNED_SHADOW_TEMPORAL_LATENT_DIM,
        rare_heavy_enabled=False,
    )
    print(
        f"[soak] profile: n_z={LEARNED_SHADOW_TEMPORAL_LATENT_DIM}, "
        f"four backends SHADOW, {turn_count} synthetic turns",
        flush=True,
    )

    head_readout_series: list[dict[str, object]] = []
    kill_criteria_series: list[dict[str, object]] = []
    checkpoint_round_trips = 0
    started = time.perf_counter()
    for index in range(1, turn_count + 1):
        text = _TURN_POOL[(index - 1) % len(_TURN_POOL)]
        result = await runner.run_turn(text)
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

    # Fails loudly if any owner is missing evidence or wrote back under SHADOW.
    payload = collect_learned_shadow_evidence(runner)
    payload["schema_version"] = _SOAK_SCHEMA_VERSION
    payload["artifact_kind"] = "learned_shadow_soak"
    payload["evidence_tier"] = "synthetic-cpu-soak"
    payload["turn_count"] = turn_count
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

    latency_slo_ok = mean_turn_seconds <= _SOAK_MEAN_TURN_SECONDS_SLO
    payload["learned_active_gate"] = {
        "note": (
            "Synthetic soak lane: real_trace_turns=0 by definition, so BLOCKED "
            "verdicts are the expected honest outcome. Directional readouts "
            "only; ACTIVE promotion requires the Linux CUDA real-trace lane."
        ),
        "latency_slo_seconds": _SOAK_MEAN_TURN_SECONDS_SLO,
        "latency_slo_ok": latency_slo_ok,
        "verdicts": _gate_verdicts(
            safety_gate_ok=True,  # collector verified zero SHADOW write-back
            latency_slo_ok=latency_slo_ok,
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
    args = parser.parse_args()
    sys.exit(
        asyncio.run(
            main(
                output_dir=args.output_dir,
                turn_count=args.turns,
                sample_every=max(1, args.sample_every),
                checkpoint_every=max(1, args.checkpoint_every),
            )
        )
    )
