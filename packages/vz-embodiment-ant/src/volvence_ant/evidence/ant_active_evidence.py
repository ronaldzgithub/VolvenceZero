"""Collect ACTIVE-promotion evidence on the digital-ant substrate.

The gate (``evaluate_learned_active_candidate``) is imported from the
vz-runtime facade and is substrate-agnostic. This module only supplies the
evidence fields, sourced from ant lanes:

- ``real_trace_turns``   : genuine kernel turns driven by the ant substrate.
- ``validation_delta``   : learned arm minus no-optimize control on a held-out
                           foraging metric.
- ``strict_eta_gate``    : from the reused strict-ETA bottleneck proof.
- ``pe_off`` / ``eta_off`` controls : from the behavioural matched-control arms.
- ``rollback_drill``     : two same-seed sessions reproduce identically (a
                           deterministic pure-Python baseline to roll back to).
- ``latency_slo``        : measured per-turn wall time under the SLO.
- ``safety_gate``        : the hardwired escape reflex fires on alarm.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from volvence_zero.agent.learned_active_gate import (
    LearnedActiveEvidence,
    LearnedActiveGateVerdict,
    LearnedBackendComponent,
    evaluate_learned_active_candidate,
)
from volvence_ant.env.ant_world import AntWorld, AntWorldConfig, FoodSource
from volvence_ant.evidence.runtime_profile import (
    ant_runtime_replay_rollout_config,
)
from volvence_ant.proofs.matched_control import run_behavioral_matched_control
from volvence_ant.runtime.ant_session import AntSession, AntSessionConfig

_LATENCY_SLO_S = 5.0


@dataclass(frozen=True)
class AntActiveEvidenceBundle:
    evidence: LearnedActiveEvidence
    verdict: LearnedActiveGateVerdict
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return self.verdict.eligible


def _world(seed: int) -> AntWorld:
    return AntWorld(
        config=AntWorldConfig(seed=seed),
        food_sources=(FoodSource(x=6.0, y=0.0, strength=1.0, decay=5.0),),
    )


async def _real_trace_lane(
    *,
    seed: int,
    trace_turns: int,
    config: AntSessionConfig,
) -> tuple[int, float]:
    """Run genuine ant-substrate kernel turns; return (turn_count, mean_latency)."""

    world = _world(seed)
    session = AntSession(world, config=config)
    latencies: list[float] = []
    for _ in range(trace_turns):
        start = time.perf_counter()
        await session.step()
        latencies.append(time.perf_counter() - start)
    mean_latency = float(sum(latencies) / len(latencies)) if latencies else 0.0
    return len(session.trajectory), mean_latency


async def _rollback_drill(*, seed: int, ticks: int, n_z: int) -> bool:
    """Mutate owner state, restore it, and verify the exported fingerprint."""

    session = AntSession(
        _world(seed),
        config=AntSessionConfig(
            temporal_latent_dim=n_z,
            seed=seed,
            joint_apply_writeback=True,
            rollout_config=ant_runtime_replay_rollout_config(
                enable_sparse_exploration=True
            ),
        ),
    )
    initial = session.export_learning_checkpoint(checkpoint_id="rollback-drill")
    await session.run(ticks)
    session.restore_learning_checkpoint(initial)
    restored = session.export_learning_checkpoint(checkpoint_id="rollback-drill")
    return restored.fingerprint == initial.fingerprint


async def _safety_drill(*, seed: int, config: AntSessionConfig) -> bool:
    world = _world(seed)
    session = AntSession(world, config=config)
    world.trigger_alarm(magnitude=1.0)
    session.holder.update(
        observation=world.observe(),
        navigator_state=session.navigator.state,
        step=world.tick,
    )
    record = await session.step()
    return bool(
        session.actuator.is_reflex_triggered(1.0)
        and record.command.turn_command == 0.0
        and record.command.step_command == world.config.step_size
    )


async def collect_ant_active_evidence(
    *,
    trace_turns: int = 30,
    training_ticks: int = 0,
    behavioral_ticks: int = 40,
    seed: int = 0,
    n_z: int = 4,
    with_latent: bool = True,
    component: LearnedBackendComponent = LearnedBackendComponent.TEMPORAL_RUNTIME,
    learned_config: AntSessionConfig | None = None,
    no_optimize_config: AntSessionConfig | None = None,
    pe_off_config: AntSessionConfig | None = None,
    eta_off_config: AntSessionConfig | None = None,
    rollback_drill_passed: bool = False,
    prior_runtime_active: bool = False,
    prior_ssl_active: bool = False,
) -> AntActiveEvidenceBundle:
    learned_cfg = learned_config or AntSessionConfig(
        temporal_latent_dim=n_z,
        seed=seed,
        external_prediction_error_drive=True,
        rollout_config=ant_runtime_replay_rollout_config(
            enable_sparse_exploration=True
        ),
    )
    no_optimize_cfg = no_optimize_config or AntSessionConfig(
        temporal_latent_dim=n_z,
        seed=seed,
        external_prediction_error_drive=True,
        joint_apply_policy_optimization=False,
        rollout_config=ant_runtime_replay_rollout_config(
            enable_sparse_exploration=True
        ),
    )
    pe_off_cfg = pe_off_config or AntSessionConfig(
        temporal_latent_dim=n_z,
        seed=seed,
        external_prediction_error_drive=False,
        rollout_config=ant_runtime_replay_rollout_config(
            enable_sparse_exploration=True
        ),
    )
    # --- real-trace lane (also yields latency) ---
    real_trace_turns, mean_latency = await _real_trace_lane(
        seed=seed, trace_turns=trace_turns, config=learned_cfg
    )
    latency_slo_ok = mean_latency <= _LATENCY_SLO_S

    # --- fair train -> owner checkpoint -> held-out behavioural arms ---
    extra_arms = {"no_optimize": no_optimize_cfg}
    if eta_off_config is not None:
        extra_arms["eta_off"] = eta_off_config
    report = await run_behavioral_matched_control(
        ticks=behavioral_ticks,
        training_ticks=training_ticks,
        seed=seed,
        temporal_latent_dim=n_z,
        learned_config=learned_cfg,
        pe_off_config=pe_off_cfg,
        extra_kernel_arms=extra_arms,
    )
    by_arm = {arm.arm: arm for arm in report.arms}
    learned = by_arm["learned"]
    no_optimize = by_arm["no_optimize"]
    pe_off = by_arm["pe_off"]
    eta_off = by_arm.get("eta_off")
    validation_delta = (
        learned.food_delivered - no_optimize.food_delivered
    ) / max(behavioral_ticks, 1)
    pe_off_direction_correct = (
        learned.food_delivered >= pe_off.food_delivered
        and learned.food_pickups >= pe_off.food_pickups
        and (
            learned.food_delivered > pe_off.food_delivered
            or learned.food_pickups > pe_off.food_pickups
        )
    )
    eta_off_direction_correct = bool(
        eta_off is not None
        and learned.food_delivered >= eta_off.food_delivered
        and learned.food_pickups >= eta_off.food_pickups
        and (
            learned.food_delivered > eta_off.food_delivered
            or learned.food_pickups > eta_off.food_pickups
        )
    )

    # --- strict-ETA proof (reused, torch) ---
    strict_eta_passed = bool(
        eta_off is not None
        and learned.switch_count > eta_off.switch_count
        and eta_off_direction_correct
        and learned.temporal_parameters_changed
        # no-optimize intentionally shares SSL with learned, so its temporal
        # structure may change.  The causal isolation is that Internal-RL
        # policy/critic parameters do not persist.
        and no_optimize.temporal_parameters_changed
        and not no_optimize.policy_parameters_changed
    )
    latent_desc = "skipped"
    if with_latent:
        try:
            from volvence_ant.proofs import run_ant_latent_proofs

            latent = run_ant_latent_proofs()
            latent_desc = latent.description
        except ImportError as exc:
            latent_desc = f"torch unavailable: {exc}"

    # --- drills ---
    reproducible_baseline = await _rollback_drill(
        seed=seed,
        ticks=min(behavioral_ticks, 12),
        n_z=n_z,
    )
    safety_ok = await _safety_drill(seed=seed, config=learned_cfg)

    evidence = LearnedActiveEvidence(
        component=component,
        real_trace_turns=real_trace_turns,
        validation_delta=validation_delta,
        strict_eta_gate_passed=strict_eta_passed,
        pe_off_control_direction_correct=pe_off_direction_correct,
        eta_off_control_direction_correct=eta_off_direction_correct,
        rollback_drill_passed=reproducible_baseline,
        latency_slo_ok=latency_slo_ok,
        safety_gate_ok=safety_ok,
        prior_runtime_active=prior_runtime_active,
        prior_ssl_active=prior_ssl_active,
    )
    verdict = evaluate_learned_active_candidate(evidence)
    metrics = {
        "mean_latency_s": mean_latency,
        "learned": asdict(learned),
        "no_optimize": asdict(no_optimize),
        "pe_off": asdict(pe_off),
        "eta_off": asdict(eta_off) if eta_off is not None else None,
        "validation_delta": validation_delta,
        "training_ticks": training_ticks,
        "held_out_ticks": behavioral_ticks,
        "train_eval_isolated": True,
        "backend_wiring": {
            "temporal_runtime_backend": learned_cfg.rollout_config.temporal_runtime_backend.value,
            "temporal_ssl_backend": learned_cfg.rollout_config.temporal_ssl_backend.value,
            "internal_rl_backend": learned_cfg.rollout_config.internal_rl_backend.value,
            "internal_rl_runtime_replay": (
                learned_cfg.rollout_config.internal_rl_runtime_replay.value
            ),
            "internal_rl_runtime_exploration_strength": (
                learned_cfg.rollout_config.internal_rl_runtime_exploration_strength
            ),
            "cms_torch_backend": learned_cfg.rollout_config.cms_torch_backend.value,
        }
        if learned_cfg.rollout_config is not None
        else {
            "temporal_runtime_backend": "disabled",
            "temporal_ssl_backend": "disabled",
            "internal_rl_backend": "disabled",
            "internal_rl_runtime_replay": "disabled",
            "internal_rl_runtime_exploration_strength": 0.0,
            "cms_torch_backend": "disabled",
        },
        "latent_desc": latent_desc,
        "trace_turns_requested": trace_turns,
        "substrate": "digital-ant-v0",
        "trace_tag": ":ant:real:",
        "strict_eta_source": "ant-task-arms",
        "generic_latent_proof_reference_only": latent_desc,
        "reproducible_baseline": reproducible_baseline,
        "legacy_external_rollback_claim": rollback_drill_passed,
    }
    return AntActiveEvidenceBundle(evidence=evidence, verdict=verdict, metrics=metrics)
