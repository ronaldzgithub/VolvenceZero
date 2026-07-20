"""Evaluation mid-layer SHADOW matched-control evidence.

Runs the same scripted turn sequence twice on a deterministic synthetic
substrate — once with ``evaluation_mid=SHADOW`` (current default) and once
with ``evaluation_mid=DISABLED`` (rollback baseline) — and verifies:

1. **Zero behavior impact**: per-turn semantic digests of the active chain
   (regime / prediction_error / dual_track / evaluation) are identical
   between the two runs. SHADOW must not leak into live behavior (R15).
2. **Signal presence**: the SHADOW run publishes non-empty
   ``evaluation_mid`` snapshots (credit / PE / regime readout re-emission),
   i.e. promoting the layer would add observable readout surface.

This is the SHADOW evidence packet required by
``docs/specs/evaluation-cascade.md`` §迁移协议 Step 2 before any
SHADOW → ACTIVE discussion. Readout-only; no acceptance gate consumes the
artifact (R12).

Example:

    python scripts/run_evaluation_mid_shadow_evidence.py --turns 6
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.integration.final_wiring import FinalRolloutConfig
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime

_ARTIFACT_SCHEMA_VERSION = "evaluation-mid-shadow-evidence.v1"

# Fixed scripted inputs covering task / support / repair / casual pressure
# so the regime + evaluation surfaces move across turns.
_SCRIPTED_INPUTS: tuple[str, ...] = (
    "I need help planning the migration of our billing service this week.",
    "Honestly I'm exhausted, this project has been draining me for a month.",
    "You misunderstood what I asked for yesterday and it cost me a meeting.",
    "Thanks for hearing me out. What small step should I take first?",
    "Let's get back to the migration plan — what are the risk points?",
    "That helps. I feel a bit better about the whole thing now.",
    "One more thing: can you double-check the rollback path in the plan?",
    "Good. Same time tomorrow to review progress?",
)


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


def behavior_digest(active_snapshots: dict[str, Any]) -> dict[str, object]:
    """Stable per-turn semantic digest of the active chain.

    Only reads owner-published snapshot fields (R8); rounds floats so the
    digest is insensitive to sub-1e-9 nondeterminism but catches any real
    behavioral divergence.
    """
    digest: dict[str, object] = {}
    regime = active_snapshots.get("regime")
    if regime is not None:
        value = regime.value
        digest["regime_active"] = value.active_regime.regime_id
        digest["regime_candidates"] = [
            (regime_id, round(score, 6)) for regime_id, score in value.candidate_regimes
        ]
    pe = active_snapshots.get("prediction_error")
    if pe is not None:
        error = pe.value.error
        digest["pe"] = [
            round(error.task_error, 6),
            round(error.relationship_error, 6),
            round(error.regime_error, 6),
            round(error.action_error, 6),
            round(error.magnitude, 6),
        ]
    dual_track = active_snapshots.get("dual_track")
    if dual_track is not None:
        value = dual_track.value
        digest["dual_track"] = [
            round(value.world_track.tension_level, 6),
            round(value.self_track.tension_level, 6),
            round(value.cross_track_tension, 6),
        ]
    evaluation = active_snapshots.get("evaluation")
    if evaluation is not None:
        digest["evaluation_turn_scores"] = [
            (score.metric_name, round(score.value, 6))
            for score in evaluation.value.turn_scores
        ]
    return digest


def _mid_readout(shadow_snapshots: dict[str, Any]) -> dict[str, object] | None:
    snapshot = shadow_snapshots.get("evaluation_mid")
    if snapshot is None:
        return None
    value = snapshot.value
    return {
        "aggregated_score_count": len(value.aggregated_scores),
        "aggregated_metric_names": sorted(
            {score.metric_name for score in value.aggregated_scores}
        ),
        "counterfactual_readout_count": len(value.counterfactual_readouts),
        "description": value.description,
    }


async def run_arm(
    *, arm_label: str, evaluation_mid: WiringLevel, turns: int
) -> tuple[list[dict[str, object]], list[dict[str, object] | None]]:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="evaluation-mid-shadow-evidence")
    runner = AgentSessionRunner(
        session_id=f"evaluation-mid-evidence-{arm_label}",
        config=FinalRolloutConfig(evaluation_mid=evaluation_mid),
        default_residual_runtime=runtime,
    )
    digests: list[dict[str, object]] = []
    mid_readouts: list[dict[str, object] | None] = []
    for turn_index in range(turns):
        user_input = _SCRIPTED_INPUTS[turn_index % len(_SCRIPTED_INPUTS)]
        result = await runner.run_turn(user_input)
        digests.append(behavior_digest(result.active_snapshots))
        mid_readouts.append(_mid_readout(result.shadow_snapshots))
    return digests, mid_readouts


async def main(*, output_dir: Path, turns: int) -> int:
    print(f"[eval-mid] running SHADOW arm ({turns} turns)...", flush=True)
    shadow_digests, shadow_mid = await run_arm(
        arm_label="shadow", evaluation_mid=WiringLevel.SHADOW, turns=turns
    )
    print(f"[eval-mid] running DISABLED arm ({turns} turns)...", flush=True)
    disabled_digests, disabled_mid = await run_arm(
        arm_label="disabled", evaluation_mid=WiringLevel.DISABLED, turns=turns
    )

    mismatched_turns = [
        index
        for index, (a, b) in enumerate(zip(shadow_digests, disabled_digests, strict=True))
        if a != b
    ]
    behavior_identical = not mismatched_turns
    signal_turns = [
        index
        for index, readout in enumerate(shadow_mid)
        if readout is not None and readout["aggregated_score_count"]
    ]
    disabled_published = [r for r in disabled_mid if r is not None]
    mid_signal_present = bool(signal_turns)
    disabled_stays_silent = not disabled_published
    overall_pass = behavior_identical and mid_signal_present and disabled_stays_silent

    payload: dict[str, object] = {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "evaluation_mid_shadow_evidence",
        "provenance": _collect_provenance(),
        "turns": turns,
        "arms": {
            "shadow": {"evaluation_mid": "shadow"},
            "disabled": {"evaluation_mid": "disabled"},
        },
        "behavior_identical": behavior_identical,
        "mismatched_turns": mismatched_turns,
        "mid_signal_present": mid_signal_present,
        "mid_signal_turns": signal_turns,
        "disabled_stays_silent": disabled_stays_silent,
        "shadow_mid_readouts": shadow_mid,
        "per_turn_digests_shadow": shadow_digests,
        "per_turn_digests_disabled": disabled_digests,
        "overall_pass": overall_pass,
        "note": (
            "SHADOW evidence packet for evaluation-cascade 迁移协议 Step 2. "
            "Readout-only; promotion to ACTIVE additionally requires consumer "
            "opt-in review and rollback drill."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "evaluation_mid_shadow_evidence.json"
    artifact_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest_path = output_dir / "evaluation_mid_shadow_evidence_manifest.json"
    data = artifact_path.read_bytes()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "evaluation-mid-shadow-evidence-manifest.v1",
                "artifacts": [
                    {
                        "path": str(artifact_path),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "size_bytes": len(data),
                    }
                ],
                "provenance": payload["provenance"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[eval-mid] behavior_identical={behavior_identical} mismatched={mismatched_turns}")
    print(f"[eval-mid] mid_signal_present={mid_signal_present} signal_turns={signal_turns}")
    print(f"[eval-mid] disabled_stays_silent={disabled_stays_silent}")
    print(f"[eval-mid] overall_pass={overall_pass}")
    print(f"[eval-mid] artifact written to {artifact_path}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/evaluation_mid_shadow_evidence")
    )
    parser.add_argument("--turns", type=int, default=6)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(output_dir=args.output_dir, turns=args.turns)))
