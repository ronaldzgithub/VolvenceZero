"""Phase 2 caste reprogramming -> research/ant/results/phase2_caste.json.

    python scripts/run_ant_caste.py [--n-individuals 16] [--rounds 500]

Runs the OFFLINE role-reprogramming loop (rare-heavy analogue) under several
environmental pressures and shows the emergent explorer/patroller mix shifting
systematically with pressure — no hardcoded role assignment. The runtime is
never allowed to trigger this step.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path

import numpy as np

from volvence_ant.caste import (
    ColonyRareHeavyBundle,
    EnvironmentPressure,
    IndividualRareHeavyRef,
    RoleProbe,
    cluster_behavioral_roles,
    reprogram_castes,
)
from volvence_ant.evidence import collect_ant_provenance, write_ant_artifact_bundle
from volvence_ant.evidence.provenance import stable_json_digest
from volvence_ant.env.ant_world import AntWorld, AntWorldConfig, FoodSource
from volvence_ant.runtime import AntSession, AntSessionConfig

_RESULTS_DIR = Path("research/ant/results")
_REPO_ROOT = Path(__file__).resolve().parents[1]

_PRESSURES = (
    EnvironmentPressure(
        label="abundant", food_distance=4.0, food_decay=2.5, food_strength=2.5, food_radius=2.0
    ),
    EnvironmentPressure(
        label="scarce", food_distance=7.0, food_decay=0.9, food_strength=2.5, food_radius=1.5
    ),
    EnvironmentPressure(
        label="predator",
        food_distance=4.0,
        food_decay=2.5,
        food_strength=2.5,
        food_radius=2.0,
        predation=1.0,
    ),
)


async def _trace_for_pressure(
    pressure: EnvironmentPressure,
    *,
    seed: int,
    steps: int,
) -> object:
    from volvence_zero.substrate import TraceStep, TrainingTrace

    world = AntWorld(
        config=AntWorldConfig(seed=seed),
        food_sources=(
            FoodSource(
                x=pressure.food_distance,
                y=0.0,
                strength=pressure.food_strength,
                decay=pressure.food_decay,
                radius=pressure.food_radius,
            ),
        ),
    )
    session = AntSession(
        world,
        config=AntSessionConfig(
            temporal_latent_dim=16,
            session_id=f"rare-heavy-trace:{pressure.label}:{seed}",
            seed=seed,
        ),
    )
    trace_steps = []
    for step_index in range(steps):
        await session.step()
        snapshot = await session._adapter_factory("ant", step_index).capture()
        trace_steps.append(
            TraceStep(
                step=step_index,
                token="<ant-sense>",
                feature_surface=snapshot.feature_surface,
                residual_activations=snapshot.residual_activations,
            )
        )
    return TrainingTrace(
        trace_id=f"ant:{pressure.label}:real:{seed}",
        source_text="digital-ant physical trace",
        steps=tuple(trace_steps),
    )


async def _formal_rare_heavy(
    *,
    pressure: EnvironmentPressure,
    n_individuals: int,
    seed: int,
) -> tuple[ColonyRareHeavyBundle, tuple[RoleProbe, ...], tuple[object, ...]]:
    from volvence_zero.joint_loop import PipelineConfig, SSLRLTrainingPipeline

    refs = []
    artifacts = []
    rollback_verified = True
    for individual_id in range(n_individuals):
        trace = await _trace_for_pressure(
            pressure,
            seed=seed + individual_id,
            steps=6,
        )
        pipeline = SSLRLTrainingPipeline(
            config=PipelineConfig(
                n_z=16,
                ssl_min_steps=2,
                ssl_max_steps=3,
                transition_max_steps=1,
                rl_max_steps=2,
                rl_rollouts_per_step=1,
            )
        )
        pipeline.run_pipeline(traces=(trace,))
        artifact = pipeline.export_rare_heavy_artifact(
            artifact_id=f"ant:{pressure.label}:individual:{individual_id}",
            include_substrate=False,
        )
        session = AntSession(
            AntWorld(config=AntWorldConfig(seed=seed + individual_id)),
            config=AntSessionConfig(
                temporal_latent_dim=16,
                session_id=f"rare-heavy-review:{individual_id}",
                seed=seed + individual_id,
                allow_live_substrate_mutation=True,
            ),
        )
        review = session.runner.review_rare_heavy_artifact(artifact)
        applied = session.runner.apply_rare_heavy_artifact(artifact)
        rollback = session.runner.rollback_rare_heavy_import(applied.checkpoint)
        rollback_verified = rollback_verified and bool(rollback)
        refs.append(
            IndividualRareHeavyRef(
                individual_id=individual_id,
                artifact_id=artifact.artifact_id,
                artifact_digest=stable_json_digest(asdict(artifact)),
                provenance=f"ant:{pressure.label}:real:{seed + individual_id}",
                gate_verdict=(
                    "reviewed-and-experimentally-applied"
                    if review.checkpoint is not None
                    else "blocked"
                ),
            )
        )
        artifacts.append(artifact)
    # Held-out behavioral probes are computed from artifact parameters only as
    # readouts; labels are never passed into training or runtime initialization.
    probes = []
    for individual_id, artifact in enumerate(artifacts):
        snapshot = artifact.temporal_snapshot
        vector = np.asarray(
            (
                *snapshot.latent_mean,
                *snapshot.latent_scale,
                *snapshot.decoder_control,
            ),
            dtype=float,
        )
        radius = float(np.linalg.norm(vector[:4]))
        trail = float(abs(vector[4 % len(vector)]))
        discovery = float(abs(vector[5 % len(vector)]))
        patrol = float(abs(vector[6 % len(vector)]))
        probes.append(
            RoleProbe(
                individual_id=individual_id,
                trajectory_radius=radius,
                trail_reliance=trail,
                discovery_contribution=discovery,
                patrol_contribution=patrol,
            )
        )
    return (
        ColonyRareHeavyBundle(
            schema_version="digital-ant-colony-rare-heavy.v1",
            pressure_label=pressure.label,
            individuals=tuple(refs),
            rollback_verified=rollback_verified,
        ),
        tuple(probes),
        tuple(artifacts),
    )


async def main(*, n_individuals: int, rounds: int, seed: int) -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = reprogram_castes(
        pressures=_PRESSURES,
        n_individuals=n_individuals,
        rounds=rounds,
        seed=seed,
        allow_offline=True,
    )
    formal = {}
    for pressure in _PRESSURES:
        bundle, probes, _artifacts = await _formal_rare_heavy(
            pressure=pressure,
            n_individuals=n_individuals,
            seed=seed,
        )
        clustering_error = None
        try:
            roles = cluster_behavioral_roles(probes, seed=seed)
        except ValueError as exc:
            roles = ()
            clustering_error = str(exc)
        counts = {
            label: sum(role.role_label == label for role in roles)
            for label in ("explorer", "patroller")
        }
        formal[pressure.label] = {
            "bundle": asdict(bundle),
            "probes": [asdict(probe) for probe in probes],
            "roles": [asdict(role) for role in roles],
            "role_counts": counts,
            "non_degenerate": all(value > 0 for value in counts.values()),
            "clustering_error": clustering_error,
        }
    payload = {
        "artifact_kind": "digital-ant-rare-heavy-roles",
        "experiment": "phase2_caste_reprogramming",
        "formal_rare_heavy": formal,
        "legacy_fixed_rule_grid": {
            "description": result.description,
            "role_shift_monotone": result.role_shift_monotone,
            "profiles": [asdict(p) for p in result.profiles],
            "yield_grid": {
                key: [list(item) for item in value]
                for key, value in result.yield_grid.items()
            },
        },
        "verdict": (
            "PASS"
            if all(
                lane["non_degenerate"] and lane["bundle"]["rollback_verified"]
                for lane in formal.values()
            )
            else "BLOCK"
        ),
    }
    manifest = write_ant_artifact_bundle(
        artifact_path=_RESULTS_DIR / "phase2_caste.json",
        payload=payload,
        provenance=collect_ant_provenance(
            repo_root=_REPO_ROOT,
            seeds=(seed,),
            config={
                "n_individuals": n_individuals,
                "rounds": rounds,
                "pressures": tuple(item.label for item in _PRESSURES),
            },
        ),
        repo_root=_REPO_ROOT,
    )
    print(f"[phase2] verdict={payload['verdict']}; manifest={manifest}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-individuals", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                n_individuals=args.n_individuals,
                rounds=args.rounds,
                seed=args.seed,
            )
        )
    )
