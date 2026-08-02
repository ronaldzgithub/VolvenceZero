"""Freeze the ETA rate-distortion sweep before any of it is executed.

Everything that could otherwise be tuned after seeing a curve lives here:
the alpha grid, the seed schedule, the optimisation budget, the gap-detection
thresholds, the arm-separation rule, and the closed verdict set. The runner
refuses to attach a preregistration whose values disagree with the code it is
about to execute, and refuses to claim an authoritative verdict without one.

The frozen source SHAs exist because a previous campaign was invalidated when
runtime source changed mid-matrix; a mismatch is a hard stop, not a warning.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

from volvence_zero.agent.eta_proof_benchmark import ETAOpenWeightRuntimeConfig
from volvence_zero.agent.eta_rate_distortion_evidence import (
    OBSERVATION_PROTOCOL_V1,
    OBSERVATION_PROTOCOL_V2,
    OBSERVATION_PROTOCOL_V3,
    assess_gap,
)
from volvence_zero.temporal.metacontroller_components import (
    POSTERIOR_PARAMETERIZATION_LEGACY,
    POSTERIOR_PARAMETERIZATION_SMOOTH,
    RATE_GATING_PER_STEP,
    RATE_GATING_SWITCH,
)

PREREGISTRATION_SCHEMA_VERSION = "eta-rate-distortion-prereg.v1"

_REPO_ROOT = Path(__file__).resolve().parent.parent

FROZEN_SOURCE_FILES = (
    "packages/vz-substrate/src/volvence_zero/substrate/steered_action_scoring.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "packages/vz-temporal/src/volvence_zero/temporal/torch_store_ssl.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_rate_distortion_evidence.py",
    "scripts/run_eta_rate_distortion.py",
)

VERDICT_SET = (
    "retain-eta",
    "retain-weak",
    "kill-eta",
    "instrument-invalid",
    "inconclusive-joint-arm-gap",
    "incomplete-sweep",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_preregistration(
    *,
    alphas: tuple[float, ...],
    seeds: int,
    n_z: int,
    updates: int,
    learning_rate: float,
    substrate_learning_rate: float,
    switch_threshold: float,
    model_id: str,
    device: str,
    arms: tuple[str, ...],
    model_source: str | None = None,
    corpus: dict[str, object] | None = None,
    rate_axis_gate: dict[str, object] | None = None,
    observation_protocol: str = OBSERVATION_PROTOCOL_V1,
    posterior_parameterization: str = POSTERIOR_PARAMETERIZATION_LEGACY,
    rate_gating: str = RATE_GATING_PER_STEP,
) -> dict[str, object]:
    gap_defaults = inspect.signature(assess_gap).parameters
    return {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "experiment_id": "eta-rate-distortion-criterion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "eta-temporal-abstraction-criterion-only",
        "sweep": {
            "alpha_grid": list(alphas),
            "seed_schedule": list(range(seeds)),
            "n_z": n_z,
            "updates_per_run": updates,
            "learning_rate": learning_rate,
            "substrate_learning_rate": substrate_learning_rate,
            "switch_threshold": switch_threshold,
            "arms": list(arms),
            "model_id": model_id,
            "model_source": model_source or model_id,
            "device": device,
            "corpus": corpus or {"corpus_origin": "default-hardcoded-7-route"},
            "observation_protocol": observation_protocol,
            "posterior_parameterization": posterior_parameterization,
            "rate_gating": rate_gating,
        },
        "rate_axis_gate": rate_axis_gate
        or {
            "gate_id": "gate-1-rate-axis-response",
            "arm": "frozen",
            "spearman_alpha_rate_max": -0.8,
            "rate_span_min": 0.30,
            "baseline_rate_span_7_route": 0.20,
            "rule": (
                "Gate 1 passes iff the frozen arm's spearman(alpha, rate) <= "
                "-0.8 AND rate_span >= rate_span_min AND rate_span is "
                "materially larger than the 7-route baseline. Failing means "
                "the rate axis still does not trade information for accuracy: "
                "fix posterior variance parameterization, do NOT proceed to "
                "Stage 2 pretraining."
            ),
        },
        "gap_thresholds": {
            "drop_share_threshold": gap_defaults[
                "drop_share_threshold"
            ].default,
            "rate_share_threshold": gap_defaults[
                "rate_share_threshold"
            ].default,
            "noise_multiple": gap_defaults["noise_multiple"].default,
        },
        "arm_separation_rule": "max(2.0 * pooled_distortion_std, 0.02)",
        "verdict_set": list(VERDICT_SET),
        "decision_rules": [
            (
                "The joint arm is a mandatory validity control. If the frozen and joint "
                "curves are indistinguishable the round yields instrument-invalid and "
                "no thesis conclusion may be drawn."
            ),
            (
                "retain-eta requires the frozen arm to show the gap, the joint arm not "
                "to, and beta boundary F1 to be higher inside the gap region than outside."
            ),
            "kill-eta requires a distinguishable arm pair and no gap on the frozen arm.",
            (
                "Distortion is the expert-action NLL through the steered frozen model; "
                "rate is the mean per-dimension posterior KL of z_t. No other term may "
                "enter the loss."
            ),
        ],
        "prohibited_after_execution": [
            "changing the alpha grid, seed schedule, or optimisation budget",
            "changing any gap threshold or the arm-separation rule",
            "reporting a verdict outside the frozen verdict set",
            "using a forward-head capacity ladder to support an ETA capacity claim",
        ],
        "frozen_source_files": {
            name: _sha256(_REPO_ROOT / name) for name in FROZEN_SOURCE_FILES
        },
    }


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the ETA rate-distortion sweep preregistration."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-z", type=int, default=16)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument(
        "--substrate-learning-rate", type=float, default=1e-4
    )
    parser.add_argument("--switch-threshold", type=float, default=0.55)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--model-id", default=ETAOpenWeightRuntimeConfig().model_id
    )
    parser.add_argument(
        "--arms", nargs="+", choices=("frozen", "joint"),
        default=("frozen", "joint"),
    )
    parser.add_argument("--model-source", default=None)
    parser.add_argument("--corpus-seed", type=int, default=None)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--corridor-count", type=int, default=2)
    parser.add_argument("--extra-edge-probability", type=float, default=0.35)
    parser.add_argument("--train-routes", type=int, default=200)
    parser.add_argument("--heldout-routes", type=int, default=60)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--heldout-lengths", type=int, nargs="+", default=(3, 4))
    parser.add_argument(
        "--observation-protocol",
        choices=(
            OBSERVATION_PROTOCOL_V1,
            OBSERVATION_PROTOCOL_V2,
            OBSERVATION_PROTOCOL_V3,
        ),
        default=OBSERVATION_PROTOCOL_V1,
    )
    parser.add_argument(
        "--posterior-parameterization",
        choices=(
            POSTERIOR_PARAMETERIZATION_LEGACY,
            POSTERIOR_PARAMETERIZATION_SMOOTH,
        ),
        default=POSTERIOR_PARAMETERIZATION_LEGACY,
    )
    parser.add_argument(
        "--rate-gating",
        choices=(RATE_GATING_PER_STEP, RATE_GATING_SWITCH),
        default=RATE_GATING_PER_STEP,
    )
    args = parser.parse_args(argv)

    output: Path = args.output
    if output.exists():
        raise SystemExit(
            f"refusing to overwrite an existing preregistration: {output}"
        )
    if args.corpus_seed is not None:
        corpus = {
            "corpus_origin": "generated-seeded",
            "corpus_seed": args.corpus_seed,
            "objective_count": args.objective_count,
            "corridor_count": args.corridor_count,
            "extra_edge_probability": args.extra_edge_probability,
            "train_routes": args.train_routes,
            "heldout_routes": args.heldout_routes,
            "train_lengths": list(args.train_lengths),
            "heldout_lengths": list(args.heldout_lengths),
        }
    else:
        corpus = {"corpus_origin": "default-hardcoded-7-route"}
    payload = build_preregistration(
        alphas=tuple(args.alphas),
        seeds=args.seeds,
        n_z=args.n_z,
        updates=args.updates,
        learning_rate=args.learning_rate,
        substrate_learning_rate=args.substrate_learning_rate,
        switch_threshold=args.switch_threshold,
        model_id=args.model_id,
        device=args.device,
        arms=tuple(args.arms),
        model_source=args.model_source,
        corpus=corpus,
        observation_protocol=args.observation_protocol,
        posterior_parameterization=args.posterior_parameterization,
        rate_gating=args.rate_gating,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "preregistration": str(output),
                "sha256": _sha256(output),
                "cells": (
                    len(payload["sweep"]["alpha_grid"])  # type: ignore[index]
                    * len(payload["sweep"]["seed_schedule"])  # type: ignore[index]
                    * len(payload["sweep"]["arms"])  # type: ignore[index]
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
