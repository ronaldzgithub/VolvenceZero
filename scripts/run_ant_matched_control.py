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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path

from volvence_ant.evidence import (
    ANT_RUNTIME_EXPLORATION_STRENGTH,
    ANT_RUNTIME_MODULATION_STRENGTH,
    SeedPartialStore,
    ant_runtime_replay_rollout_config,
    ant_stage_fingerprint,
    collect_ant_provenance,
    stable_json_digest,
    verify_ant_artifact_manifest,
    write_ant_artifact_bundle,
)
from volvence_ant.proofs import (
    ArmMetrics,
    MatchedControlReport,
    aggregate_matched_control_reports,
    run_single_seed_matched_control,
)
from volvence_ant.runtime.ant_session import AntSessionConfig

_RESULTS_DIR = Path("research/ant/results")
_REPO_ROOT = Path(__file__).resolve().parents[1]
# Backward-compatible script constants; values are owned by runtime_profile.
_ANT_RL_RUNTIME_MODULATION_STRENGTH = ANT_RUNTIME_MODULATION_STRENGTH
_ANT_RL_RUNTIME_EXPLORATION_STRENGTH = ANT_RUNTIME_EXPLORATION_STRENGTH


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
        # no_optimize observes the same PE/SSL schedule, rollout and optimizer
        # report, but restores the post-SSL/pre-RL checkpoint so reward-driven
        # policy/critic updates cannot accumulate. Reflection writeback remains
        # matched to learned; this arm isolates Internal-RL optimization only.
        "no_optimize": AntSessionConfig(
            temporal_latent_dim=n_z,
            seed=seed,
            external_prediction_error_drive=True,
            joint_schedule=active,
            joint_apply_writeback=True,
            joint_apply_policy_optimization=False,
            rollout_config=ant_runtime_replay_rollout_config(),
        ),
        # ETA-off retains the same substrate/world while disabling learned
        # latent replacement/switching and SSL/RL.
        "eta_off": AntSessionConfig(
            temporal_latent_dim=n_z,
            seed=seed,
            external_prediction_error_drive=True,
            joint_schedule=frozen,
            joint_apply_writeback=False,
            joint_apply_policy_optimization=False,
            rollout_config=ant_runtime_replay_rollout_config(),
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
        joint_apply_policy_optimization=True,
        rollout_config=ant_runtime_replay_rollout_config(),
    )


def _pe_off_config(seed: int, n_z: int) -> AntSessionConfig:
    from volvence_zero.joint_loop import JointLoopSchedule

    return AntSessionConfig(
        temporal_latent_dim=n_z,
        seed=seed,
        external_prediction_error_drive=False,
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=3),
        joint_apply_writeback=True,
        joint_apply_policy_optimization=True,
        rollout_config=ant_runtime_replay_rollout_config(),
    )


def _report_from_dict(payload: dict) -> MatchedControlReport:
    arms = payload.get("arms")
    if not isinstance(arms, list):
        raise ValueError("matched-control partial arms must be a list")
    return MatchedControlReport(
        ticks=int(payload["ticks"]),
        seed=int(payload["seed"]),
        arms=tuple(ArmMetrics(**arm) for arm in arms),
        learned_beats_random_food=bool(payload["learned_beats_random_food"]),
        description=str(payload["description"]),
    )


def _configure_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        return


def _run_seed_worker(
    seed: int,
    ticks: int,
    train_ticks: int,
    n_z: int,
    include_e2e_rl: bool,
) -> MatchedControlReport:
    _configure_worker_threads()
    return asyncio.run(
        run_single_seed_matched_control(
            seed=seed,
            ticks=ticks,
            training_ticks=train_ticks,
            temporal_latent_dim=n_z,
            kernel_arms=_schedule_gated_arms(seed=seed, n_z=n_z),
            learned_config=_learned_config(seed, n_z),
            pe_off_config=_pe_off_config(seed, n_z),
            include_e2e_rl=include_e2e_rl,
        )
    )


def _final_artifact_matches(*, config: dict) -> bool:
    artifact_path = _RESULTS_DIR / "matched_control.json"
    manifest_path = artifact_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return False
    verify_ant_artifact_manifest(
        manifest_path=manifest_path,
        repo_root=_REPO_ROOT,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    actual = payload.get("provenance", {}).get("config_digest")
    expected = stable_json_digest(config)
    if actual != expected:
        print(
            "[matched-control] ignoring stale final artifact; "
            f"config actual={actual!r}, expected={expected!r}"
        )
        return False
    return True


async def main(
    *,
    ticks: int,
    train_ticks: int,
    seeds: tuple[int, ...],
    n_z: int,
    with_latent: bool,
    include_e2e_rl: bool,
    workers: int,
    resume: bool,
) -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "ticks": ticks,
        "train_ticks": train_ticks,
        "seeds": seeds,
        "n_z": n_z,
        "with_latent": with_latent,
        "include_e2e_rl": include_e2e_rl,
        "internal_rl_runtime_modulation_strength": (
            ANT_RUNTIME_MODULATION_STRENGTH
        ),
        "internal_rl_runtime_exploration_strength": (
            ANT_RUNTIME_EXPLORATION_STRENGTH
        ),
        "internal_rl_runtime_replay": "active",
    }
    if resume and _final_artifact_matches(config=config):
        print("[matched-control] resumed complete final artifact")
        return 0

    fingerprint = ant_stage_fingerprint(stage="matched_control", config=config)
    partials = SeedPartialStore(
        results_root=_RESULTS_DIR,
        stage="matched_control",
        fingerprint=fingerprint,
        requested_seeds=seeds,
    )
    completed_payloads = partials.load() if resume else {}
    reports_by_seed = {
        seed: _report_from_dict(dict(report))
        for seed, report in completed_payloads.items()
    }
    remaining = tuple(seed for seed in seeds if seed not in reports_by_seed)
    if reports_by_seed:
        print(
            "[matched-control] resumed seeds="
            + ",".join(str(seed) for seed in sorted(reports_by_seed))
        )

    payload: dict = {
        "artifact_kind": "digital-ant-fair-learning-matrix",
        "experiment": "workstream_e_matched_control",
        "diagnostic_order": (
            "exploration-food-contact",
            "outcome-pe-credit",
            "runtime-replay-settlement",
            "policy-vs-no-optimize-divergence",
            "z-and-turn-action-effect",
            "held-out-generalization",
        ),
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

    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers == 1:
        for seed in remaining:
            report = await run_single_seed_matched_control(
                seed=seed,
                ticks=ticks,
                training_ticks=train_ticks,
                temporal_latent_dim=n_z,
                kernel_arms=_schedule_gated_arms(seed=seed, n_z=n_z),
                learned_config=_learned_config(seed, n_z),
                pe_off_config=_pe_off_config(seed, n_z),
                include_e2e_rl=include_e2e_rl,
            )
            partials.commit(seed=seed, report=asdict(report))
            reports_by_seed[seed] = report
            print(f"[matched-control] completed seed={seed}")
    elif remaining:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(remaining)),
            mp_context=context,
        ) as executor:
            future_seeds = {
                executor.submit(
                    _run_seed_worker,
                    seed,
                    ticks,
                    train_ticks,
                    n_z,
                    include_e2e_rl,
                ): seed
                for seed in remaining
            }
            for future in as_completed(future_seeds):
                seed = future_seeds[future]
                report = future.result()
                if report.seed != seed:
                    raise ValueError(
                        f"worker seed mismatch: expected={seed}, actual={report.seed}"
                    )
                partials.commit(seed=seed, report=asdict(report))
                reports_by_seed[seed] = report
                print(f"[matched-control] completed seed={seed}")

    report = aggregate_matched_control_reports(
        tuple(reports_by_seed.values()),
        seed_order=seeds,
    )
    validation_delta = (
        report.learned_minus_no_optimize / max(ticks, 1)
        if report.learned_minus_no_optimize is not None
        else None
    )
    payload["behavioral"] = {
        "per_seed": [asdict(seed_report) for seed_report in report.reports],
        "aggregates": [asdict(aggregate) for aggregate in report.aggregates],
        "learned_minus_no_optimize": report.learned_minus_no_optimize,
        "validation_delta": validation_delta,
        "verdict": (
            "PASS"
            if validation_delta is not None and validation_delta >= 0.02
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
        f"learned-minus-no-optimize={report.learned_minus_no_optimize} "
        f"validation-delta={validation_delta}"
    )
    print(f"[matched-control] manifest={manifest}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=60)
    parser.add_argument("--train-ticks", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n-z", type=int, default=16)
    parser.add_argument("--with-latent", action="store_true")
    parser.add_argument("--no-e2e-rl", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                ticks=args.ticks,
                train_ticks=args.train_ticks,
                seeds=tuple(int(value) for value in args.seeds.split(",")),
                n_z=args.n_z,
                with_latent=args.with_latent,
                include_e2e_rl=not args.no_e2e_rl,
                workers=args.workers,
                resume=args.resume,
            )
        )
    )
