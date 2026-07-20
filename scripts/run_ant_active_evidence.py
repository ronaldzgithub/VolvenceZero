"""Build the versioned multi-seed digital-ant ACTIVE evidence bundle."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, replace
from pathlib import Path

from volvence_zero.agent.learned_active_gate import LearnedBackendComponent
from volvence_zero.runtime import WiringLevel

from volvence_ant.evidence import (
    ant_runtime_replay_rollout_config,
    collect_ant_active_evidence,
    collect_ant_provenance,
    write_ant_artifact_bundle,
)
from volvence_ant.runtime import AntSessionConfig

_RESULTS_DIR = Path("research/ant/results")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_NAME = "digital-ant-evidence-bundle.v2.json"


def _arm_configs(
    seed: int,
    n_z: int,
    component: LearnedBackendComponent,
) -> dict[str, AntSessionConfig]:
    from volvence_zero.joint_loop import JointLoopSchedule
    from volvence_zero.temporal import (
        LearnedLiteTemporalPolicy,
        MetacontrollerParameterStore,
    )

    active = JointLoopSchedule(ssl_interval=1, rl_interval=3)
    frozen = JointLoopSchedule(ssl_interval=0, rl_interval=0)
    order = (
        LearnedBackendComponent.TEMPORAL_RUNTIME,
        LearnedBackendComponent.TEMPORAL_SSL,
        LearnedBackendComponent.INTERNAL_RL,
        LearnedBackendComponent.CMS_TORCH,
    )
    candidate_index = order.index(component)
    rollout = ant_runtime_replay_rollout_config()
    backend_fields = tuple(item.value for item in order)
    rollout = replace(
        rollout,
        **{
            field_name: (
                WiringLevel.ACTIVE
                if index <= candidate_index
                else WiringLevel.DISABLED
            )
            for index, field_name in enumerate(backend_fields)
        },
    )
    return {
        "learned": AntSessionConfig(
            temporal_latent_dim=n_z,
            seed=seed,
            rollout_config=rollout,
            joint_schedule=active,
            joint_apply_writeback=True,
            joint_apply_policy_optimization=True,
        ),
        "no_optimize": AntSessionConfig(
            temporal_latent_dim=n_z,
            seed=seed,
            rollout_config=rollout,
            joint_schedule=active,
            joint_apply_writeback=True,
            joint_apply_policy_optimization=False,
        ),
        "pe_off": AntSessionConfig(
            temporal_latent_dim=n_z,
            seed=seed,
            rollout_config=rollout,
            joint_schedule=active,
            joint_apply_writeback=True,
            joint_apply_policy_optimization=True,
            external_prediction_error_drive=False,
        ),
        "eta_off": AntSessionConfig(
            temporal_latent_dim=n_z,
            seed=seed,
            rollout_config=rollout,
            joint_schedule=frozen,
            joint_apply_writeback=False,
            joint_apply_policy_optimization=False,
            temporal_policy=LearnedLiteTemporalPolicy(
                parameter_store=MetacontrollerParameterStore(n_z=n_z)
            ),
        ),
    }


def _rollback_verified() -> bool:
    path = _RESULTS_DIR / "phase2_caste.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    lanes = payload.get("formal_rare_heavy", {})
    return bool(lanes) and all(
        lane["bundle"]["rollback_verified"] for lane in lanes.values()
    )


async def main(
    *,
    trace_turns: int,
    train_ticks: int,
    ticks: int,
    seeds: tuple[int, ...],
    n_z: int,
    with_latent: bool,
) -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    components = (
        LearnedBackendComponent.TEMPORAL_RUNTIME,
        LearnedBackendComponent.TEMPORAL_SSL,
        LearnedBackendComponent.INTERNAL_RL,
        LearnedBackendComponent.CMS_TORCH,
    )
    rollback_ok = _rollback_verified()
    component_results: list[dict] = []
    prior_runtime_active = False
    prior_ssl_active = False
    prior_internal_rl_active = False
    for component in components:
        blocked_by_prior = (
            component is not LearnedBackendComponent.TEMPORAL_RUNTIME
            and not prior_runtime_active
        ) or (
            component is LearnedBackendComponent.INTERNAL_RL
            and not prior_ssl_active
        ) or (
            component is LearnedBackendComponent.CMS_TORCH
            and not prior_internal_rl_active
        )
        if blocked_by_prior:
            component_results.append(
                {
                    "component": component.value,
                    "status": "not_evaluated_due_to_prior_gate",
                }
            )
            continue
        seed_results = []
        for seed in seeds:
            configs = _arm_configs(seed, n_z, component)
            bundle = await collect_ant_active_evidence(
                trace_turns=trace_turns,
                training_ticks=train_ticks,
                behavioral_ticks=ticks,
                seed=seed,
                n_z=n_z,
                with_latent=with_latent,
                component=component,
                learned_config=configs["learned"],
                no_optimize_config=configs["no_optimize"],
                pe_off_config=configs["pe_off"],
                eta_off_config=configs["eta_off"],
                rollback_drill_passed=rollback_ok,
                prior_runtime_active=prior_runtime_active,
                prior_ssl_active=prior_ssl_active,
            )
            seed_results.append(
                {
                    "seed": seed,
                    "evidence": {
                        **asdict(bundle.evidence),
                        "component": bundle.evidence.component.value,
                    },
                    "verdict": {
                        **asdict(bundle.verdict),
                        "component": bundle.verdict.component.value,
                    },
                    "metrics": bundle.metrics,
                }
            )
        eligible = all(item["verdict"]["eligible"] for item in seed_results)
        component_results.append(
            {
                "component": component.value,
                "status": "PASS" if eligible else "BLOCK",
                "per_seed": seed_results,
            }
        )
        if component is LearnedBackendComponent.TEMPORAL_RUNTIME:
            prior_runtime_active = eligible
        elif component is LearnedBackendComponent.TEMPORAL_SSL:
            prior_ssl_active = eligible
        elif component is LearnedBackendComponent.INTERNAL_RL:
            prior_internal_rl_active = eligible

    artifact = {
        "artifact_kind": "digital-ant-evidence-bundle.v2",
        "substrate": "digital-ant-v0",
        "trace_provenance": ":ant:real:",
        "trace_turns_per_seed": trace_turns,
        "seeds": seeds,
        "component_results": component_results,
        "production_defaults_changed": False,
        "rollback_verified": rollback_ok,
        "overall_verdict": (
            "PASS"
            if all(item["status"] == "PASS" for item in component_results)
            else "BLOCK"
        ),
    }
    candidate_inputs = tuple(
        path
        for path in (
            _RESULTS_DIR / "phase0_homing.json",
            _RESULTS_DIR / "phase0_route_learning.json",
            _RESULTS_DIR / "matched_control.json",
            _RESULTS_DIR / "motor_calibration.v1.json",
            _RESULTS_DIR / "phase1_colony.json",
            _RESULTS_DIR / "phase2_caste.json",
            _RESULTS_DIR / "dual_substrate.json",
            _RESULTS_DIR / "g2_perturbation.json",
            _RESULTS_DIR / "g3_bio_overlay.json",
            _RESULTS_DIR / "g4_safety_reflex.json",
        )
        if path.is_file()
    )
    manifest = write_ant_artifact_bundle(
        artifact_path=_RESULTS_DIR / _ARTIFACT_NAME,
        payload=artifact,
        provenance=collect_ant_provenance(
            repo_root=_REPO_ROOT,
            seeds=seeds,
            config={
                "trace_turns": trace_turns,
                "train_ticks": train_ticks,
                "ticks": ticks,
                "n_z": n_z,
                "with_latent": with_latent,
            },
        ),
        input_paths=candidate_inputs,
        repo_root=_REPO_ROOT,
    )
    print(
        f"[ant-active-evidence] overall={artifact['overall_verdict']} "
        f"manifest={manifest}"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-turns", type=int, default=500)
    parser.add_argument("--train-ticks", type=int, default=200)
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n-z", type=int, default=16)
    parser.add_argument("--no-latent", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                trace_turns=args.trace_turns,
                train_ticks=args.train_ticks,
                ticks=args.ticks,
                seeds=tuple(int(value) for value in args.seeds.split(",")),
                n_z=args.n_z,
                with_latent=not args.no_latent,
            )
        )
    )
