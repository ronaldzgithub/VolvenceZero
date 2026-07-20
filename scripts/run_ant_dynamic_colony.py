"""Run the Dynamic Stigmergy Regime-Shift Benchmark v1.

Example smoke run:

    python scripts/run_ant_dynamic_colony.py \
      --seeds 0 --n-ants 1 --training-rounds 0 \
      --pre-shift-rounds 1 --post-shift-rounds 1 --recovery-window 1

The frozen formal claim requires ten paired seeds. Smaller runs intentionally
produce ``BLOCK`` while still exercising the full seven-arm evidence path.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
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
from volvence_ant.experiments import (
    DynamicColonyAggregateReport,
    DynamicColonyArm,
    DynamicColonyConfig,
    DynamicColonySeedReport,
    DynamicPerturbationKind,
    aggregate_dynamic_colony_reports,
    run_dynamic_colony_seed,
)
from volvence_ant.runtime import AntSessionConfig

_RESULTS_DIR = Path("research/ant/results")
_ARTIFACT_PATH = _RESULTS_DIR / "dynamic_colony.v1.json"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_VERSION = "dynamic-stigmergy-regime-shift.v1"
_REPORT_SCHEMA_VERSION = "dynamic-colony-report.v3"


def _implementation_digest() -> str:
    source_roots = (
        _REPO_ROOT / "packages/vz-contracts/src",
        _REPO_ROOT / "packages/vz-substrate/src",
        _REPO_ROOT / "packages/vz-cognition/src",
        _REPO_ROOT / "packages/vz-temporal/src",
        _REPO_ROOT / "packages/vz-runtime/src",
        _REPO_ROOT / "packages/vz-embodiment-ant/src",
    )
    paths = [Path(__file__).resolve()]
    for root in source_roots:
        paths.extend(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(str(path.relative_to(_REPO_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _configure_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        return


def _kernel_configs(
    *,
    seed: int,
    n_z: int,
) -> tuple[AntSessionConfig, AntSessionConfig, AntSessionConfig]:
    from volvence_zero.joint_loop import JointLoopSchedule

    schedule = JointLoopSchedule(ssl_interval=1, rl_interval=3)
    rollout = ant_runtime_replay_rollout_config(enable_sparse_exploration=True)
    learned = AntSessionConfig(
        temporal_latent_dim=n_z,
        seed=seed,
        external_prediction_error_drive=True,
        joint_schedule=schedule,
        joint_apply_writeback=True,
        joint_apply_policy_optimization=True,
        rollout_config=rollout,
    )
    no_optimize = AntSessionConfig(
        temporal_latent_dim=n_z,
        seed=seed,
        external_prediction_error_drive=True,
        joint_schedule=schedule,
        joint_apply_writeback=True,
        joint_apply_policy_optimization=False,
        rollout_config=rollout,
    )
    pe_off = AntSessionConfig(
        temporal_latent_dim=n_z,
        seed=seed,
        external_prediction_error_drive=False,
        joint_schedule=schedule,
        joint_apply_writeback=True,
        joint_apply_policy_optimization=True,
        rollout_config=rollout,
    )
    return learned, no_optimize, pe_off


def _run_seed_worker(
    seed: int,
    perturbation_value: str,
    config_payload: dict,
) -> DynamicColonySeedReport:
    _configure_worker_threads()
    config = DynamicColonyConfig(**config_payload)
    learned, no_optimize, pe_off = _kernel_configs(
        seed=seed,
        n_z=config.temporal_latent_dim,
    )
    return asyncio.run(
        run_dynamic_colony_seed(
            seed=seed,
            perturbation=DynamicPerturbationKind(perturbation_value),
            config=config,
            learned_config=learned,
            no_optimize_config=no_optimize,
            pe_off_config=pe_off,
        )
    )


def _report_from_dict(payload: dict) -> DynamicColonySeedReport:
    config = payload.get("config")
    arms = payload.get("arms")
    if not isinstance(config, dict):
        raise ValueError("dynamic-colony partial config must be an object")
    if not isinstance(arms, list):
        raise ValueError("dynamic-colony partial arms must be a list")
    return DynamicColonySeedReport(
        seed=int(payload["seed"]),
        training_world_seed=int(payload["training_world_seed"]),
        evaluation_world_seed=int(payload["evaluation_world_seed"]),
        perturbation=str(payload["perturbation"]),
        config=DynamicColonyConfig(**config),
        arms=tuple(DynamicColonyArm(**arm) for arm in arms),
    )


def _final_artifact_matches(*, config: dict) -> bool:
    manifest_path = _ARTIFACT_PATH.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return False
    verify_ant_artifact_manifest(
        manifest_path=manifest_path,
        repo_root=_REPO_ROOT,
    )
    payload = json.loads(_ARTIFACT_PATH.read_text(encoding="utf-8"))
    actual = payload.get("provenance", {}).get("config_digest")
    return actual == stable_json_digest(config)


async def _run_scenario(
    *,
    perturbation: DynamicPerturbationKind,
    config: DynamicColonyConfig,
    seeds: tuple[int, ...],
    workers: int,
    resume: bool,
) -> DynamicColonyAggregateReport:
    semantic_config = {
        "benchmark_version": _BENCHMARK_VERSION,
        "report_schema_version": _REPORT_SCHEMA_VERSION,
        "implementation_digest": _implementation_digest(),
        "perturbation": perturbation.value,
        **asdict(config),
        "seeds": seeds,
        "runtime_replay": "active",
        "runtime_modulation_strength": ANT_RUNTIME_MODULATION_STRENGTH,
        "runtime_exploration_strength": ANT_RUNTIME_EXPLORATION_STRENGTH,
    }
    stage = f"dynamic_colony_{perturbation.value}"
    fingerprint = ant_stage_fingerprint(stage=stage, config=semantic_config)
    partials = SeedPartialStore(
        results_root=_RESULTS_DIR,
        stage=stage,
        fingerprint=fingerprint,
        requested_seeds=seeds,
    )
    completed_payloads = partials.load() if resume else {}
    reports_by_seed = {
        seed: _report_from_dict(dict(report))
        for seed, report in completed_payloads.items()
    }
    remaining = tuple(seed for seed in seeds if seed not in reports_by_seed)

    config_payload = asdict(config)
    if workers == 1:
        for seed in remaining:
            learned, no_optimize, pe_off = _kernel_configs(
                seed=seed,
                n_z=config.temporal_latent_dim,
            )
            report = await run_dynamic_colony_seed(
                seed=seed,
                perturbation=perturbation,
                config=config,
                learned_config=learned,
                no_optimize_config=no_optimize,
                pe_off_config=pe_off,
            )
            partials.commit(seed=seed, report=asdict(report))
            reports_by_seed[seed] = report
            print(
                f"[dynamic-colony:{perturbation.value}] completed seed={seed}"
            )
    elif remaining:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(remaining)),
            mp_context=context,
        ) as executor:
            future_seeds = {
                executor.submit(
                    _run_seed_worker,
                    seed,
                    perturbation.value,
                    config_payload,
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
                print(
                    f"[dynamic-colony:{perturbation.value}] completed seed={seed}"
                )
    return aggregate_dynamic_colony_reports(
        tuple(reports_by_seed.values()),
        seed_order=seeds,
    )


async def main(
    *,
    perturbations: tuple[DynamicPerturbationKind, ...],
    config: DynamicColonyConfig,
    seeds: tuple[int, ...],
    workers: int,
    resume: bool,
) -> int:
    if not perturbations:
        raise ValueError("perturbations must be non-empty")
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if len(set(perturbations)) != len(perturbations):
        raise ValueError("perturbations must be distinct")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    semantic_config = {
        "benchmark_version": _BENCHMARK_VERSION,
        "report_schema_version": _REPORT_SCHEMA_VERSION,
        "implementation_digest": _implementation_digest(),
        "perturbations": tuple(item.value for item in perturbations),
        **asdict(config),
        "seeds": seeds,
        "runtime_replay": "active",
        "runtime_modulation_strength": ANT_RUNTIME_MODULATION_STRENGTH,
        "runtime_exploration_strength": ANT_RUNTIME_EXPLORATION_STRENGTH,
    }
    if resume and _final_artifact_matches(config=semantic_config):
        print("[dynamic-colony] resumed complete final artifact")
        return 0

    scenario_reports = []
    for perturbation in perturbations:
        scenario_reports.append(
            await _run_scenario(
                perturbation=perturbation,
                config=config,
                seeds=seeds,
                workers=workers,
                resume=resume,
            )
        )
    required_perturbations = set(DynamicPerturbationKind)
    suite_complete = (
        len(perturbations) == len(required_perturbations)
        and set(perturbations) == required_perturbations
    )
    overall_verdict = (
        "PASS"
        if suite_complete
        and all(report.verdict == "PASS" for report in scenario_reports)
        else "BLOCK"
    )
    payload = {
        "artifact_kind": "digital-ant-dynamic-colony",
        "experiment": "dynamic_stigmergy_regime_shift_v1",
        **semantic_config,
        "scenario_reports": [asdict(report) for report in scenario_reports],
        "suite_complete": suite_complete,
        "verdict": overall_verdict,
    }
    manifest = write_ant_artifact_bundle(
        artifact_path=_ARTIFACT_PATH,
        payload=payload,
        provenance=collect_ant_provenance(
            repo_root=_REPO_ROOT,
            seeds=seeds,
            config=semantic_config,
        ),
        repo_root=_REPO_ROOT,
    )
    print(
        "[dynamic-colony] "
        f"verdict={overall_verdict}; manifest={manifest}"
    )
    for report in scenario_reports:
        print(f"[dynamic-colony] {report.description}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--perturbations",
        default="obstacle_block,food_relocation,motor_bias",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--n-ants", type=int, default=8)
    parser.add_argument("--training-rounds", type=int, default=200)
    parser.add_argument("--pre-shift-rounds", type=int, default=50)
    parser.add_argument("--post-shift-rounds", type=int, default=100)
    parser.add_argument("--recovery-window", type=int, default=20)
    parser.add_argument("--n-z", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                perturbations=tuple(
                    DynamicPerturbationKind(value)
                    for value in args.perturbations.split(",")
                ),
                config=DynamicColonyConfig(
                    n_ants=args.n_ants,
                    training_rounds=args.training_rounds,
                    pre_shift_rounds=args.pre_shift_rounds,
                    post_shift_rounds=args.post_shift_rounds,
                    recovery_window=args.recovery_window,
                    temporal_latent_dim=args.n_z,
                ),
                seeds=tuple(
                    int(value) for value in args.seeds.split(",")
                ),
                workers=args.workers,
                resume=args.resume,
            )
        )
    )
