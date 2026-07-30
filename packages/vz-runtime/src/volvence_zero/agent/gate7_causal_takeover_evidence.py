"""Preregistered five-arm Gate 7 SSL→Internal-RL causal evidence harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

from volvence_zero.agent.eta_proof_benchmark import (
    build_default_eta_proof_environment,
)
from volvence_zero.agent.gate78_shared_trace import (
    GATE78_SOURCE_DESCRIPTOR,
    GATE78_TRACE_SCHEMA_VERSION,
    GATE78_TRACE_SEEDS,
    Gate78EpisodePlan,
    load_gate78_partition,
    verify_gate78_shared_trace_bundle,
)
from volvence_zero.internal_rl import HierarchicalRouteSpec, InternalRLSandbox
from volvence_zero.memory import Track
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    ExpertActionTarget,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
    build_training_trace,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterSnapshot,
    MetacontrollerParameterStore,
    MetacontrollerSSLTrainer,
    build_training_trace_from_substrate_snapshots,
)


GATE7_SCHEMA_VERSION = "gate7-causal-takeover.v1"
GATE7_ARMS = (
    "full",
    "no-ssl",
    "no-rl",
    "reverse-order",
    "joint-unfrozen",
)
GATE7_REQUIRED_FILES = (
    "manifest.yaml",
    "predictions.jsonl",
    "outcomes.jsonl",
    "prediction_errors.jsonl",
    "segments.jsonl",
    "credit.jsonl",
    "state_diff.jsonl",
    "action_selection.jsonl",
    "ablation_results.json",
    "promotion_verdict.json",
    "rollback_evidence.json",
    "report.md",
)
_ACTION_FAMILIES = ("alpha", "beta", "gamma", "delta", "epsilon")
_TAKEOVER_THRESHOLDS = {
    "posterior_agreement": 0.35,
    "switch_sparsity_retention": 0.05,
    "family_reuse_retention": 0.20,
    "decoder_effect_retention": 0.01,
    "heldout_prefix_stability": 0.20,
}


@dataclass(frozen=True)
class Gate7TakeoverMetrics:
    passed: bool
    posterior_agreement: float
    switch_sparsity_retention: float
    family_reuse_retention: float
    decoder_effect_retention: float
    heldout_prefix_stability: float
    transition_count: int
    failed_reasons: tuple[str, ...]


@dataclass(frozen=True)
class Gate7ArmResult:
    seed: int
    partition: str
    arm: str
    train_episode_count: int
    evaluation_episode_count: int
    ssl_update_count: int
    ssl_trained_step_count: int
    ssl_prediction_loss_mean: float
    ssl_kl_loss_mean: float
    rl_update_count: int
    rl_parameter_change_count: int
    takeover: Gate7TakeoverMetrics
    terminal_return: float
    terminal_success_rate: float
    composition_score: float
    delayed_credit_assignment_count: int
    active_family_count: int
    future_residual_leakage_count: int
    token_space_rl_mutation_count: int
    structure_fingerprint_before_rl: str
    structure_fingerprint_after_rl: str
    structure_fingerprint_change_during_rl: int
    policy_fingerprint_before: str
    policy_fingerprint_after: str
    rollback_exact: bool
    rollback_fingerprint_before: str
    rollback_fingerprint_after: str


@dataclass(frozen=True)
class Gate7EvidenceReport:
    schema_version: str
    source_schema_version: str
    source_fingerprint: str
    partition: str
    seed_schedule: tuple[int, ...]
    arm_schedule: tuple[str, ...]
    controller_initialization_seed: int
    controller_dim: int
    ssl_updates: int
    rl_cycles: int
    formal_locked_run: bool
    source_consumer_admission: bool
    arm_results: tuple[Gate7ArmResult, ...]
    aggregate_metrics: tuple[tuple[str, float], ...]
    mechanism_gates: tuple[tuple[str, bool, float], ...]
    causal_gates: tuple[tuple[str, bool, float], ...]
    verdict: str
    description: str


@dataclass(frozen=True)
class _EpisodeMaterial:
    plan: Gate78EpisodePlan
    snapshots: tuple[SubstrateSnapshot, ...]
    training_trace: object
    proof_episode: object


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mean(values: tuple[float, ...]) -> float:
    return statistics.fmean(values) if values else 0.0


def _structure_fingerprint(
    snapshot: MetacontrollerParameterSnapshot,
) -> str:
    family_topology = tuple(
        {
            "family_id": family.family_id,
            "latent_centroid": family.latent_centroid,
            "decoder_centroid": family.decoder_centroid,
            "support": family.support,
            "stability": family.stability,
            "switch_bias": family.switch_bias,
        }
        for family in snapshot.action_families
    )
    structure = {
        "encoder_weights": snapshot.encoder_weights,
        "encoder_recurrence": snapshot.encoder_recurrence,
        "switch_weights": snapshot.switch_weights,
        "beta_threshold": snapshot.beta_threshold,
        "decoder_matrix": snapshot.decoder_matrix,
        "decoder_hidden": snapshot.decoder_hidden,
        "ndim_encoder_parameters": snapshot.ndim_encoder_parameters,
        "ndim_switch_parameters": snapshot.ndim_switch_parameters,
        "ndim_decoder_parameters": snapshot.ndim_decoder_parameters,
        # Runtime reuse/competition counters and summaries are observation
        # telemetry, not discovered structure.  The frozen fingerprint keeps
        # family identity, centroids and discovery confidence while excluding
        # those counters.
        "action_family_topology": family_topology,
    }
    return _sha256(structure)


def _checkpoint_fingerprint(checkpoint: object) -> str:
    payload = asdict(checkpoint)
    payload.pop("checkpoint_id", None)
    return _sha256(payload)


def _episode_material(
    plan: Gate78EpisodePlan,
    *,
    controller_dim: int,
) -> _EpisodeMaterial:
    tokens: list[str] = []
    target_ids: list[str] = []
    for family_id, segment_length in zip(
        plan.action_family_ids,
        plan.segment_lengths,
        strict=True,
    ):
        for segment_index in range(segment_length):
            tokens.append(
                f"{plan.context_id}-{family_id}-segment-{segment_index}"
            )
            target_ids.append(family_id)
    trace = build_training_trace(
        trace_id=f"{plan.episode_id}:source",
        source_text=" ".join(tokens),
    )
    snapshots: list[SubstrateSnapshot] = []
    expert_targets: list[ExpertActionTarget] = []
    for index, step in enumerate(trace.steps):
        prefix = tuple(
            ResidualSequenceStep(
                step=prefix_step.step,
                token=prefix_step.token,
                feature_surface=prefix_step.feature_surface,
                residual_activations=prefix_step.residual_activations,
                description=(
                    f"{plan.episode_id} causal prefix step "
                    f"{prefix_step.step}"
                ),
            )
            for prefix_step in trace.steps[: index + 1]
        )
        snapshots.append(
            SubstrateSnapshot(
                model_id=f"gate78-frozen:{plan.seed}",
                is_frozen=True,
                surface_kind=SurfaceKind.RESIDUAL_STREAM,
                token_logits=tuple(
                    sum(signal.values) / len(signal.values)
                    for signal in step.feature_surface
                ),
                feature_surface=step.feature_surface,
                residual_activations=step.residual_activations,
                residual_sequence=prefix,
                unavailable_fields=(),
                description=(
                    f"{plan.episode_id} prefix {index}; "
                    f"boundary={plan.next_session_boundary}"
                ),
            )
        )
        family_id = target_ids[index]
        family_index = _ACTION_FAMILIES.index(family_id)
        expert_targets.append(
            ExpertActionTarget(
                action_id=family_id,
                values=tuple(
                    1.0 if dimension == family_index else 0.0
                    for dimension in range(controller_dim)
                ),
                source=f"gate78-environment:{plan.episode_id}",
                description="Structured current-action target; no future outcome fields.",
            )
        )
    training_trace = build_training_trace_from_substrate_snapshots(
        trace_id=f"{plan.episode_id}:ssl",
        source_text=" ".join(tokens),
        snapshots=tuple(snapshots),
        expert_action_targets=tuple(expert_targets),
    )
    route = HierarchicalRouteSpec(
        case_id=plan.episode_id,
        split=plan.partition,
        source_text=" ".join(plan.session_two_turns),
        waypoints=plan.route,
        split_detail=f"{plan.partition}:next-session",
        description=(
            f"Gate 7 composition route for {plan.context_id} at "
            f"difficulty {plan.difficulty:.3f}."
        ),
    )
    proof_episode = build_default_eta_proof_environment().build_proof_episode(
        route
    )
    return _EpisodeMaterial(
        plan=plan,
        snapshots=tuple(snapshots),
        training_trace=training_trace,
        proof_episode=proof_episode,
    )


def _future_leakage_count(
    materials: tuple[_EpisodeMaterial, ...],
) -> int:
    leakage = 0
    for material in materials:
        for index, snapshot in enumerate(material.snapshots):
            if len(snapshot.residual_sequence) != index + 1:
                leakage += 1
                continue
            if snapshot.residual_sequence[-1].step != index:
                leakage += 1
            if any(step.step > index for step in snapshot.residual_sequence):
                leakage += 1
    return leakage


def _rollouts(
    sandbox: InternalRLSandbox,
    materials: tuple[_EpisodeMaterial, ...],
    *,
    label: str,
) -> tuple[object, ...]:
    sandbox.policy.parameter_store.set_learning_phase(
        "rl", structure_frozen=True
    )
    return tuple(
        sandbox.rollout(
            rollout_id=f"gate7:{label}:{material.plan.episode_id}",
            substrate_steps=material.snapshots,
            track=Track.SHARED,
            replacement_mode="causal",
            proof_episode=material.proof_episode,
        )
        for material in materials
    )


def _takeover_metrics(
    rollouts: tuple[object, ...],
) -> Gate7TakeoverMetrics:
    transitions = tuple(
        transition
        for rollout in rollouts
        for transition in rollout.transitions
    )
    if not transitions:
        return Gate7TakeoverMetrics(
            passed=False,
            posterior_agreement=0.0,
            switch_sparsity_retention=0.0,
            family_reuse_retention=0.0,
            decoder_effect_retention=0.0,
            heldout_prefix_stability=0.0,
            transition_count=0,
            failed_reasons=("takeover-rollout-readiness-below-threshold",),
        )
    posterior_agreement = _mean(
        tuple(
            1.0
            - (
                sum(
                    abs(action - latent)
                    for action, latent in zip(
                        transition.policy_action,
                        transition.latent_code,
                        strict=True,
                    )
                )
                / len(transition.policy_action)
            )
            for transition in transitions
        )
    )
    switch_sparsity_retention = _mean(
        tuple(
            1.0 - transition.controller_state.switch_gate
            for transition in transitions
        )
    )
    family_reuse_retention = _mean(
        tuple(
            float(
                transition.active_family_id
                not in {None, "unassigned"}
            )
            for transition in transitions
        )
    )
    decoder_effect_retention = _mean(
        tuple(
            min(
                sum(abs(value) for value in transition.downstream_effect)
                / max(len(transition.downstream_effect), 1),
                1.0,
            )
            for transition in transitions
        )
    )
    heldout_prefix_stability = _mean(
        tuple(
            max(0.0, 1.0 - transition.controller_state.switch_gate)
            * max(
                0.0,
                min(transition.policy_replacement_quality, 1.0),
            )
            for transition in transitions
        )
    )
    values = {
        "posterior_agreement": posterior_agreement,
        "switch_sparsity_retention": switch_sparsity_retention,
        "family_reuse_retention": family_reuse_retention,
        "decoder_effect_retention": decoder_effect_retention,
        "heldout_prefix_stability": heldout_prefix_stability,
    }
    failed_reasons = tuple(
        f"{name.replace('_', '-')}-below-threshold"
        for name, threshold in _TAKEOVER_THRESHOLDS.items()
        if values[name] < threshold
    )
    return Gate7TakeoverMetrics(
        passed=not failed_reasons,
        posterior_agreement=posterior_agreement,
        switch_sparsity_retention=switch_sparsity_retention,
        family_reuse_retention=family_reuse_retention,
        decoder_effect_retention=decoder_effect_retention,
        heldout_prefix_stability=heldout_prefix_stability,
        transition_count=len(transitions),
        failed_reasons=failed_reasons,
    )


def _run_ssl_updates(
    *,
    trainer: MetacontrollerSSLTrainer,
    policy: FullLearnedTemporalPolicy,
    materials: tuple[_EpisodeMaterial, ...],
    seed: int,
    arm: str,
    start_index: int,
    count: int,
) -> tuple[object, ...]:
    reports: list[object] = []
    policy.parameter_store.set_learning_phase(
        "ssl", structure_frozen=False
    )
    for update_index in range(count):
        reports.append(
            trainer.optimize_batch(
                policy=policy,
                traces=tuple(
                    material.training_trace for material in materials
                ),
                batch_id=(
                    f"gate7:{seed}:{arm}:ssl:"
                    f"{start_index + update_index}"
                ),
            )
        )
    return tuple(reports)


def _run_rl_updates(
    *,
    sandbox: InternalRLSandbox,
    materials: tuple[_EpisodeMaterial, ...],
    seed: int,
    arm: str,
    start_index: int,
    count: int,
) -> tuple[object, ...]:
    reports: list[object] = []
    for cycle_index in range(count):
        rollouts = _rollouts(
            sandbox,
            materials,
            label=f"{seed}:{arm}:train:{start_index + cycle_index}",
        )
        reports.append(sandbox.optimize(rollouts))
    return tuple(reports)


def _run_arm(
    *,
    seed: int,
    partition: str,
    arm: str,
    train_materials: tuple[_EpisodeMaterial, ...],
    evaluation_materials: tuple[_EpisodeMaterial, ...],
    controller_dim: int,
    ssl_updates: int,
    rl_cycles: int,
) -> Gate7ArmResult:
    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(
            n_z=controller_dim,
            initialization_seed=42,
        )
    )
    sandbox = InternalRLSandbox(policy=policy)
    trainer = MetacontrollerSSLTrainer(
        n_z=controller_dim,
        ssl_backend=WiringLevel.ACTIVE,
    )
    initial_checkpoint = sandbox.create_checkpoint(
        checkpoint_id=f"gate7:{seed}:{arm}:initial"
    )
    rollback_before = _checkpoint_fingerprint(initial_checkpoint)
    substrate_before = _sha256(
        tuple(
            material.snapshots
            for material in (*train_materials, *evaluation_materials)
        )
    )
    policy_before = initial_checkpoint.policy_optimization_fingerprint
    ssl_reports: list[object] = []
    rl_reports: list[object] = []
    structure_before_rl = _structure_fingerprint(
        policy.parameter_store.export_parameter_snapshot()
    )
    takeover_rollouts: tuple[object, ...] = ()

    if arm == "full":
        ssl_reports.extend(
            _run_ssl_updates(
                trainer=trainer,
                policy=policy,
                materials=train_materials,
                seed=seed,
                arm=arm,
                start_index=0,
                count=ssl_updates,
            )
        )
        structure_before_rl = _structure_fingerprint(
            policy.parameter_store.export_parameter_snapshot()
        )
        takeover_rollouts = _rollouts(
            sandbox,
            evaluation_materials,
            label=f"{seed}:{arm}:takeover",
        )
        if _takeover_metrics(takeover_rollouts).passed:
            rl_reports.extend(
                _run_rl_updates(
                    sandbox=sandbox,
                    materials=train_materials,
                    seed=seed,
                    arm=arm,
                    start_index=0,
                    count=rl_cycles,
                )
            )
    elif arm == "no-ssl":
        structure_before_rl = _structure_fingerprint(
            policy.parameter_store.export_parameter_snapshot()
        )
        takeover_rollouts = _rollouts(
            sandbox,
            evaluation_materials,
            label=f"{seed}:{arm}:takeover",
        )
        rl_reports.extend(
            _run_rl_updates(
                sandbox=sandbox,
                materials=train_materials,
                seed=seed,
                arm=arm,
                start_index=0,
                count=rl_cycles,
            )
        )
    elif arm == "no-rl":
        ssl_reports.extend(
            _run_ssl_updates(
                trainer=trainer,
                policy=policy,
                materials=train_materials,
                seed=seed,
                arm=arm,
                start_index=0,
                count=ssl_updates,
            )
        )
        structure_before_rl = _structure_fingerprint(
            policy.parameter_store.export_parameter_snapshot()
        )
        takeover_rollouts = _rollouts(
            sandbox,
            evaluation_materials,
            label=f"{seed}:{arm}:takeover",
        )
    elif arm == "reverse-order":
        structure_before_rl = _structure_fingerprint(
            policy.parameter_store.export_parameter_snapshot()
        )
        rl_reports.extend(
            _run_rl_updates(
                sandbox=sandbox,
                materials=train_materials,
                seed=seed,
                arm=arm,
                start_index=0,
                count=rl_cycles,
            )
        )
        ssl_reports.extend(
            _run_ssl_updates(
                trainer=trainer,
                policy=policy,
                materials=train_materials,
                seed=seed,
                arm=arm,
                start_index=0,
                count=ssl_updates,
            )
        )
        takeover_rollouts = _rollouts(
            sandbox,
            evaluation_materials,
            label=f"{seed}:{arm}:takeover",
        )
    elif arm == "joint-unfrozen":
        for cycle_index in range(rl_cycles):
            ssl_reports.extend(
                _run_ssl_updates(
                    trainer=trainer,
                    policy=policy,
                    materials=train_materials,
                    seed=seed,
                    arm=arm,
                    start_index=cycle_index,
                    count=1,
                )
            )
            if cycle_index == 0:
                structure_before_rl = _structure_fingerprint(
                    policy.parameter_store.export_parameter_snapshot()
                )
            rl_reports.extend(
                _run_rl_updates(
                    sandbox=sandbox,
                    materials=train_materials,
                    seed=seed,
                    arm=arm,
                    start_index=cycle_index,
                    count=1,
                )
            )
            ssl_reports.extend(
                _run_ssl_updates(
                    trainer=trainer,
                    policy=policy,
                    materials=train_materials,
                    seed=seed,
                    arm=arm,
                    start_index=ssl_updates + cycle_index,
                    count=1,
                )
            )
        takeover_rollouts = _rollouts(
            sandbox,
            evaluation_materials,
            label=f"{seed}:{arm}:takeover",
        )
    else:
        raise ValueError(
            f"Unsupported Gate 7 arm {arm!r}; expected one of {GATE7_ARMS}"
        )

    takeover = _takeover_metrics(takeover_rollouts)
    evaluation_rollouts = _rollouts(
        sandbox,
        evaluation_materials,
        label=f"{seed}:{arm}:evaluation",
    )
    structure_after_rl = _structure_fingerprint(
        policy.parameter_store.export_parameter_snapshot()
    )
    active_family_count = len(policy.parameter_store.action_families)
    final_checkpoint = sandbox.create_checkpoint(
        checkpoint_id=f"gate7:{seed}:{arm}:final"
    )
    policy_after = final_checkpoint.policy_optimization_fingerprint
    substrate_after = _sha256(
        tuple(
            material.snapshots
            for material in (*train_materials, *evaluation_materials)
        )
    )
    composition_scores = tuple(
        len(rollout.completed_subgoals)
        / max(len(material.proof_episode.subgoals), 1)
        for rollout, material in zip(
            evaluation_rollouts,
            evaluation_materials,
            strict=True,
        )
    )
    delayed_credit_count = sum(
        len(rollout.delayed_credit_assignments)
        for rollout in evaluation_rollouts
    )
    sandbox.restore_checkpoint(initial_checkpoint)
    restored_checkpoint = sandbox.create_checkpoint(
        checkpoint_id=f"gate7:{seed}:{arm}:restored"
    )
    rollback_after = _checkpoint_fingerprint(restored_checkpoint)
    ssl_parameter_changes = sum(
        report.torch_parameters_changed for report in ssl_reports
    )
    rl_parameter_changes = sum(
        int(report.parameters_changed) for report in rl_reports
    )
    return Gate7ArmResult(
        seed=seed,
        partition=partition,
        arm=arm,
        train_episode_count=len(train_materials),
        evaluation_episode_count=len(evaluation_materials),
        ssl_update_count=len(ssl_reports),
        ssl_trained_step_count=sum(
            report.trained_step_count for report in ssl_reports
        ),
        ssl_prediction_loss_mean=_mean(
            tuple(report.torch_prediction_loss for report in ssl_reports)
        ),
        ssl_kl_loss_mean=_mean(
            tuple(report.torch_kl_loss for report in ssl_reports)
        ),
        rl_update_count=len(rl_reports),
        rl_parameter_change_count=(
            rl_parameter_changes + ssl_parameter_changes * 0
        ),
        takeover=takeover,
        terminal_return=_mean(
            tuple(rollout.total_reward for rollout in evaluation_rollouts)
        ),
        terminal_success_rate=_mean(
            tuple(float(rollout.terminal_success) for rollout in evaluation_rollouts)
        ),
        composition_score=_mean(composition_scores),
        delayed_credit_assignment_count=delayed_credit_count,
        active_family_count=active_family_count,
        future_residual_leakage_count=_future_leakage_count(
            evaluation_materials
        ),
        token_space_rl_mutation_count=int(
            substrate_before != substrate_after
        ),
        structure_fingerprint_before_rl=structure_before_rl,
        structure_fingerprint_after_rl=structure_after_rl,
        structure_fingerprint_change_during_rl=int(
            structure_before_rl != structure_after_rl
        ),
        policy_fingerprint_before=policy_before,
        policy_fingerprint_after=policy_after,
        rollback_exact=rollback_before == rollback_after,
        rollback_fingerprint_before=rollback_before,
        rollback_fingerprint_after=rollback_after,
    )


def _aggregate(
    results: tuple[Gate7ArmResult, ...],
    *,
    source_admission: bool,
) -> tuple[
    tuple[tuple[str, float], ...],
    tuple[tuple[str, bool, float], ...],
    tuple[tuple[str, bool, float], ...],
    str,
]:
    arm_means = {
        arm: {
            "terminal_return": _mean(
                tuple(
                    row.terminal_return
                    for row in results
                    if row.arm == arm
                )
            ),
            "composition_score": _mean(
                tuple(
                    row.composition_score
                    for row in results
                    if row.arm == arm
                )
            ),
        }
        for arm in GATE7_ARMS
    }
    terminal_gain_no_ssl = (
        arm_means["full"]["terminal_return"]
        - arm_means["no-ssl"]["terminal_return"]
    )
    terminal_gain_no_rl = (
        arm_means["full"]["terminal_return"]
        - arm_means["no-rl"]["terminal_return"]
    )
    composition_gain_no_ssl = (
        arm_means["full"]["composition_score"]
        - arm_means["no-ssl"]["composition_score"]
    )
    composition_gain_no_rl = (
        arm_means["full"]["composition_score"]
        - arm_means["no-rl"]["composition_score"]
    )
    future_leakage = sum(
        row.future_residual_leakage_count for row in results
    )
    token_mutation = sum(
        row.token_space_rl_mutation_count for row in results
    )
    rollback_mismatch = sum(not row.rollback_exact for row in results)
    full_structure_change = sum(
        row.structure_fingerprint_change_during_rl
        for row in results
        if row.arm == "full"
    )
    full_takeover_rate = _mean(
        tuple(
            float(row.takeover.passed)
            for row in results
            if row.arm == "full"
        )
    )
    metrics = (
        ("full_terminal_return", arm_means["full"]["terminal_return"]),
        ("full_composition_score", arm_means["full"]["composition_score"]),
        ("terminal_return_gain_vs_no_ssl", terminal_gain_no_ssl),
        ("terminal_return_gain_vs_no_rl", terminal_gain_no_rl),
        ("composition_gain_vs_no_ssl", composition_gain_no_ssl),
        ("composition_gain_vs_no_rl", composition_gain_no_rl),
        ("full_takeover_rate", full_takeover_rate),
        ("future_residual_leakage_count", float(future_leakage)),
        ("token_space_rl_mutation_count", float(token_mutation)),
        ("full_structure_change_count", float(full_structure_change)),
        ("rollback_mismatch_count", float(rollback_mismatch)),
    )
    mechanism_gates = (
        ("source-consumer-admission", source_admission, float(source_admission)),
        ("future-residual-leakage-zero", future_leakage == 0, float(future_leakage)),
        ("token-space-rl-mutation-zero", token_mutation == 0, float(token_mutation)),
        ("full-structure-frozen", full_structure_change == 0, float(full_structure_change)),
        ("whole-cycle-rollback-exact", rollback_mismatch == 0, float(rollback_mismatch)),
    )
    causal_gates = (
        ("full-takeover-passes", full_takeover_rate == 1.0, full_takeover_rate),
        (
            "terminal-return-gain-vs-no-ssl",
            terminal_gain_no_ssl >= 0.02,
            terminal_gain_no_ssl,
        ),
        (
            "terminal-return-gain-vs-no-rl",
            terminal_gain_no_rl >= 0.02,
            terminal_gain_no_rl,
        ),
        (
            "composition-gain-vs-no-ssl",
            composition_gain_no_ssl >= 0.02,
            composition_gain_no_ssl,
        ),
        (
            "composition-gain-vs-no-rl",
            composition_gain_no_rl >= 0.02,
            composition_gain_no_rl,
        ),
    )
    if not all(passed for _name, passed, _value in mechanism_gates):
        verdict = "invalid"
    elif all(passed for _name, passed, _value in causal_gates):
        verdict = "causal-supported"
    else:
        verdict = "not-supported"
    return metrics, mechanism_gates, causal_gates, verdict


def run_gate7_evidence(
    *,
    trace_root: str | Path,
    seed_schedule: tuple[int, ...] = GATE78_TRACE_SEEDS,
    partition: str = "trace-development-heldout",
    controller_dim: int = 8,
    ssl_updates: int = 2,
    rl_cycles: int = 2,
    train_limit: int | None = None,
    evaluation_limit: int | None = None,
    formal_locked_run: bool = False,
) -> Gate7EvidenceReport:
    if not seed_schedule:
        raise ValueError("Gate 7 seed_schedule must not be empty")
    if any(seed not in GATE78_TRACE_SEEDS for seed in seed_schedule):
        raise ValueError("Gate 7 seed_schedule contains an unregistered seed")
    if formal_locked_run and partition != "trace-locked-confirmation":
        raise ValueError(
            "Formal Gate 7 run must use trace-locked-confirmation"
        )
    if not formal_locked_run and partition == "trace-locked-confirmation":
        raise ValueError(
            "Development Gate 7 run must not consume locked confirmation"
        )
    if controller_dim < len(_ACTION_FAMILIES):
        raise ValueError(
            "Gate 7 controller_dim must cover every action family"
        )
    if ssl_updates < 1 or rl_cycles < 1:
        raise ValueError("Gate 7 update counts must be positive")
    source_verification = verify_gate78_shared_trace_bundle(trace_root)
    if not source_verification["consumer_admission"]:
        raise RuntimeError("Gate 7 source corpus failed consumer admission")
    all_results: list[Gate7ArmResult] = []
    for seed in seed_schedule:
        train_plans = load_gate78_partition(
            trace_root,
            seed=seed,
            partition="trace-train",
        )
        evaluation_plans = load_gate78_partition(
            trace_root,
            seed=seed,
            partition=partition,
        )
        if train_limit is not None:
            train_plans = train_plans[:train_limit]
        if evaluation_limit is not None:
            evaluation_plans = evaluation_plans[:evaluation_limit]
        train_materials = tuple(
            _episode_material(plan, controller_dim=controller_dim)
            for plan in train_plans
        )
        evaluation_materials = tuple(
            _episode_material(plan, controller_dim=controller_dim)
            for plan in evaluation_plans
        )
        for arm in GATE7_ARMS:
            all_results.append(
                _run_arm(
                    seed=seed,
                    partition=partition,
                    arm=arm,
                    train_materials=train_materials,
                    evaluation_materials=evaluation_materials,
                    controller_dim=controller_dim,
                    ssl_updates=ssl_updates,
                    rl_cycles=rl_cycles,
                )
            )
    results = tuple(all_results)
    metrics, mechanism_gates, causal_gates, verdict = _aggregate(
        results,
        source_admission=bool(source_verification["consumer_admission"]),
    )
    return Gate7EvidenceReport(
        schema_version=GATE7_SCHEMA_VERSION,
        source_schema_version=GATE78_TRACE_SCHEMA_VERSION,
        source_fingerprint=_sha256(GATE78_SOURCE_DESCRIPTOR),
        partition=partition,
        seed_schedule=seed_schedule,
        arm_schedule=GATE7_ARMS,
        controller_initialization_seed=42,
        controller_dim=controller_dim,
        ssl_updates=ssl_updates,
        rl_cycles=rl_cycles,
        formal_locked_run=formal_locked_run,
        source_consumer_admission=True,
        arm_results=results,
        aggregate_metrics=metrics,
        mechanism_gates=mechanism_gates,
        causal_gates=causal_gates,
        verdict=verdict,
        description=(
            f"Gate 7 five-arm evidence on {partition}: verdict={verdict}; "
            f"seeds={seed_schedule}, arms={GATE7_ARMS}."
        ),
    )


def _write_jsonl(
    path: Path,
    rows: tuple[Mapping[str, object], ...],
) -> None:
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def export_gate7_evidence_bundle(
    report: Gate7EvidenceReport,
    *,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    arm_rows = tuple(_jsonable(row) for row in report.arm_results)
    rows_by_file: dict[str, tuple[Mapping[str, object], ...]] = {
        "predictions.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "partition": row.partition,
                "takeover": _jsonable(row.takeover),
            }
            for row in report.arm_results
        ),
        "outcomes.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "terminal_return": row.terminal_return,
                "terminal_success_rate": row.terminal_success_rate,
                "composition_score": row.composition_score,
            }
            for row in report.arm_results
        ),
        "prediction_errors.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "composition_error": 1.0 - row.composition_score,
                "future_residual_leakage_count": (
                    row.future_residual_leakage_count
                ),
            }
            for row in report.arm_results
        ),
        "segments.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "switch_sparsity_retention": (
                    row.takeover.switch_sparsity_retention
                ),
                "active_family_count": row.active_family_count,
            }
            for row in report.arm_results
        ),
        "credit.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "delayed_credit_assignment_count": (
                    row.delayed_credit_assignment_count
                ),
            }
            for row in report.arm_results
        ),
        "state_diff.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "structure_before_rl": (
                    row.structure_fingerprint_before_rl
                ),
                "structure_after_rl": (
                    row.structure_fingerprint_after_rl
                ),
                "structure_changed": (
                    row.structure_fingerprint_change_during_rl
                ),
                "policy_before": row.policy_fingerprint_before,
                "policy_after": row.policy_fingerprint_after,
                "token_space_rl_mutation_count": (
                    row.token_space_rl_mutation_count
                ),
            }
            for row in report.arm_results
        ),
        "action_selection.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "family_reuse_retention": (
                    row.takeover.family_reuse_retention
                ),
                "active_family_count": row.active_family_count,
                "rl_parameter_change_count": (
                    row.rl_parameter_change_count
                ),
            }
            for row in report.arm_results
        ),
    }
    written: list[Path] = []
    for filename, rows in rows_by_file.items():
        path = target / filename
        _write_jsonl(path, rows)
        written.append(path)
    ablation_path = target / "ablation_results.json"
    ablation_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "arm_schedule": report.arm_schedule,
                "arm_results": arm_rows,
                "aggregate_metrics": report.aggregate_metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(ablation_path)
    verdict_path = target / "promotion_verdict.json"
    verdict_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "verdict": report.verdict,
                "mechanism_gates": report.mechanism_gates,
                "causal_gates": report.causal_gates,
                "locked_consumed": report.formal_locked_run,
                "retuning_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(verdict_path)
    rollback_path = target / "rollback_evidence.json"
    rollback_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "rows": [
                    {
                        "seed": row.seed,
                        "arm": row.arm,
                        "exact": row.rollback_exact,
                        "before": row.rollback_fingerprint_before,
                        "after": row.rollback_fingerprint_after,
                    }
                    for row in report.arm_results
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(rollback_path)
    manifest = {
        "schema_version": report.schema_version,
        "suite_id": "gate7-causal-takeover",
        "source_schema_version": report.source_schema_version,
        "source_fingerprint": report.source_fingerprint,
        "partition": report.partition,
        "seed_schedule": report.seed_schedule,
        "arm_schedule": report.arm_schedule,
        "controller_initialization_seed": (
            report.controller_initialization_seed
        ),
        "controller_dim": report.controller_dim,
        "ssl_updates": report.ssl_updates,
        "rl_cycles": report.rl_cycles,
        "formal_locked_run": report.formal_locked_run,
        "required_files": GATE7_REQUIRED_FILES,
    }
    manifest_path = target / "manifest.yaml"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written.append(manifest_path)
    report_path = target / "report.md"
    report_path.write_text(
        (
            "# Gate 7 causal takeover evidence\n\n"
            f"- partition: `{report.partition}`\n"
            f"- formal locked run: `{report.formal_locked_run}`\n"
            f"- verdict: `{report.verdict}`\n"
            f"- source fingerprint: `{report.source_fingerprint}`\n\n"
            "## Mechanism gates\n\n"
            + "".join(
                f"- {name}: `{passed}` ({value:.6f})\n"
                for name, passed, value in report.mechanism_gates
            )
            + "\n## Causal gates\n\n"
            + "".join(
                f"- {name}: `{passed}` ({value:.6f})\n"
                for name, passed, value in report.causal_gates
            )
        ),
        encoding="utf-8",
    )
    written.append(report_path)
    return tuple(written)


def verify_gate7_evidence_bundle(
    output_dir: str | Path,
) -> dict[str, object]:
    target = Path(output_dir)
    missing = tuple(
        filename
        for filename in GATE7_REQUIRED_FILES
        if not (target / filename).is_file()
    )
    if missing:
        return {
            "passed": False,
            "missing_files": missing,
            "verdict": "invalid",
        }
    manifest = json.loads(
        (target / "manifest.yaml").read_text(encoding="utf-8")
    )
    verdict = json.loads(
        (target / "promotion_verdict.json").read_text(encoding="utf-8")
    )
    passed = (
        manifest["schema_version"] == GATE7_SCHEMA_VERSION
        and tuple(manifest["arm_schedule"]) == GATE7_ARMS
        and tuple(manifest["required_files"]) == GATE7_REQUIRED_FILES
        and verdict["verdict"]
        in {"invalid", "not-supported", "causal-supported"}
    )
    return {
        "passed": passed,
        "missing_files": (),
        "verdict": verdict["verdict"],
        "formal_locked_run": manifest["formal_locked_run"],
        "partition": manifest["partition"],
    }
