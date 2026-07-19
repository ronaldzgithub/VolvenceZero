"""Workstream E runner -> research/ant/results/matched_control.json.

    python scripts/run_ant_matched_control.py [--ticks 60] [--with-latent]

Emits:
- the authoritative latent-space proofs (optimize-vs-no-optimize Internal RL on
  z_t + strict-ETA bottleneck ladder) when --with-latent and torch is present;
- the directional behavioural matched-control arms (learned / pe_off /
  no_optimize / eta_off / fixed_rule / random).

``no_optimize`` and ``eta_off`` need a ``JointLoopSchedule`` object, which the
embodiment package must not import (import-boundary rule); this orchestration
script constructs it and passes it via ``extra_kernel_arms``.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path

from volvence_ant.evidence import collect_ant_provenance, write_ant_artifact_bundle
from volvence_ant.proofs import run_multiseed_matched_control
from volvence_ant.runtime.ant_session import AntSessionConfig

_RESULTS_DIR = Path("research/ant/results")
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _schedule_gated_arms(*, seed: int, n_z: int) -> dict[str, AntSessionConfig]:
    """Build the schedule-gated kernel arms (script-side JointLoopSchedule)."""

    from volvence_zero.joint_loop import JointLoopSchedule
    from volvence_zero.temporal import (
        LearnedLiteTemporalPolicy,
        MetacontrollerParameterStore,
    )

    frozen = JointLoopSchedule(ssl_interval=0, rl_interval=0)
    active = JointLoopSchedule(ssl_interval=1, rl_interval=3)
    return {
        # no_optimize observes the same PE/SSL schedule but forbids owner writeback.
        "no_optimize": AntSessionConfig(
            temporal_latent_dim=n_z,
            seed=seed,
            external_prediction_error_drive=True,
            joint_schedule=active,
            joint_apply_writeback=False,
        ),
        # ETA-off retains the same substrate/world while disabling learned
        # latent replacement/switching and SSL/RL.
        "eta_off": AntSessionConfig(
            temporal_latent_dim=n_z,
            seed=seed,
            external_prediction_error_drive=True,
            joint_schedule=frozen,
            joint_apply_writeback=False,
            temporal_policy=LearnedLiteTemporalPolicy(
                parameter_store=MetacontrollerParameterStore(n_z=n_z)
            ),
        ),
    }


def _learned_config(seed: int, n_z: int) -> AntSessionConfig:
    from volvence_zero.joint_loop import JointLoopSchedule

    return AntSessionConfig(
        temporal_latent_dim=n_z,
        seed=seed,
        external_prediction_error_drive=True,
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=3),
        joint_apply_writeback=True,
    )


def _pe_off_config(seed: int, n_z: int) -> AntSessionConfig:
    from volvence_zero.joint_loop import JointLoopSchedule

    return AntSessionConfig(
        temporal_latent_dim=n_z,
        seed=seed,
        external_prediction_error_drive=False,
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=3),
        joint_apply_writeback=True,
    )


async def main(
    *,
    ticks: int,
    seeds: tuple[int, ...],
    n_z: int,
    with_latent: bool,
    include_e2e_rl: bool,
) -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "ticks": ticks,
        "seeds": seeds,
        "n_z": n_z,
        "include_e2e_rl": include_e2e_rl,
    }
    payload: dict = {
        "artifact_kind": "digital-ant-fair-learning-matrix",
        "experiment": "workstream_e_matched_control",
        **config,
    }

    if with_latent:
        try:
            from volvence_ant.proofs import run_ant_latent_proofs

            latent = run_ant_latent_proofs()
            payload["latent_proofs"] = {
                "learning_is_real": latent.learning_is_real,
                "eta_bottleneck_holds": latent.eta_bottleneck_holds,
                "description": latent.description,
                "internal_rl_no_optimize_proof": latent.internal_rl_no_optimize_proof,
                "strict_eta_gate": latent.strict_eta_gate,
            }
            print(f"[matched-control] {latent.description}")
        except ImportError as exc:
            payload["latent_proofs"] = {"skipped": f"torch unavailable: {exc}"}
            print(f"[matched-control] latent proofs skipped: {exc}")

    report = await run_multiseed_matched_control(
        seeds=seeds,
        ticks=ticks,
        temporal_latent_dim=n_z,
        kernel_arm_factory=lambda seed, dim: _schedule_gated_arms(
            seed=seed, n_z=dim
        ),
        learned_config_factory=_learned_config,
        pe_off_config_factory=_pe_off_config,
        include_e2e_rl=include_e2e_rl,
    )
    payload["behavioral"] = {
        "per_seed": [asdict(seed_report) for seed_report in report.reports],
        "aggregates": [asdict(aggregate) for aggregate in report.aggregates],
        "learned_minus_no_optimize": report.learned_minus_no_optimize,
        "verdict": (
            "PASS"
            if report.learned_minus_no_optimize is not None
            and report.learned_minus_no_optimize > 0.0
            else "BLOCK"
        ),
    }
    provenance = collect_ant_provenance(
        repo_root=_REPO_ROOT,
        seeds=seeds,
        config=config,
    )
    manifest = write_ant_artifact_bundle(
        artifact_path=_RESULTS_DIR / "matched_control.json",
        payload=payload,
        provenance=provenance,
        repo_root=_REPO_ROOT,
    )
    print(
        "[matched-control] "
        f"learned-minus-no-optimize={report.learned_minus_no_optimize}"
    )
    print(f"[matched-control] manifest={manifest}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=60)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n-z", type=int, default=16)
    parser.add_argument("--with-latent", action="store_true")
    parser.add_argument("--no-e2e-rl", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                ticks=args.ticks,
                seeds=tuple(int(value) for value in args.seeds.split(",")),
                n_z=args.n_z,
                with_latent=args.with_latent,
                include_e2e_rl=not args.no_e2e_rl,
            )
        )
    )
