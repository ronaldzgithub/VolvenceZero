"""P0 mechanism audit for the digital-ant ecology learning path.

The audit is intentionally diagnostic: it can only report PASS/BLOCK and never
produces a promotable checkpoint.  It binds deterministic paired action probes,
segment-closure telemetry, and owner-scoped frozen-evaluation fingerprints to
the exact checkpoints observed during a short matched training schedule.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from volvence_ant.experiments.ecology_curriculum import (
    EcologyCurriculumConfig,
    EcologyDataSplit,
    EcologyEvaluationScenario,
    EcologyStage,
    EcologyTrainingEpisodePlan,
    EcologyTrainingEpisodeReport,
    EcologyTrainingTier,
    _run_training_episode,
    _session_config,
    _world,
)
from volvence_ant.experiments.ecology_probe import (
    EcologyCheckpointActionProbe,
    EcologyProbeKind,
    run_ecology_checkpoint_action_probes,
)
from volvence_ant.runtime import AntLearningCheckpoint, KernelColonyRunner


ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION = (
    "digital-ant-ecology-mechanism-audit.v2"
)


@dataclass(frozen=True)
class EcologyMechanismAuditConfig:
    n_ants: int = 4
    temporal_latent_dim: int = 16
    episode_rounds: int = 12
    episodes_per_stage: int = 3
    evaluation_rounds: int = 24
    seed: int = 0
    code_delta_threshold: float = 1e-8
    turn_delta_threshold: float = 1e-4
    retention_ratio_threshold: float = 0.25
    body_pass_ratio: float = 0.8

    def __post_init__(self) -> None:
        if self.n_ants < 1:
            raise ValueError("n_ants must be >= 1")
        if self.temporal_latent_dim < 3:
            raise ValueError("temporal_latent_dim must be >= 3")
        if self.episode_rounds < 1 or self.episodes_per_stage < 1:
            raise ValueError("training audit budgets must be >= 1")
        if self.evaluation_rounds < 3:
            raise ValueError(
                "evaluation_rounds must be >= 3 so replay can settle"
            )
        if self.code_delta_threshold <= 0.0:
            raise ValueError("code_delta_threshold must be positive")
        if self.turn_delta_threshold <= 0.0:
            raise ValueError("turn_delta_threshold must be positive")
        if not 0.0 < self.retention_ratio_threshold <= 1.0:
            raise ValueError(
                "retention_ratio_threshold must be within (0, 1]"
            )
        if not 0.0 < self.body_pass_ratio <= 1.0:
            raise ValueError("body_pass_ratio must be within (0, 1]")


@dataclass(frozen=True)
class EcologyActionChainSnapshot:
    arm: str
    label: str
    stage: str
    tier: str
    episode_index: int
    body_reports: tuple[EcologyCheckpointActionProbe, ...]
    required_body_passes: int
    passing_bodies: int
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class EcologySegmentTelemetry:
    arm: str
    stage: str
    tier: str
    episode_index: int
    switch_count: int
    closed_segment_count: int
    longest_segment_length: int
    close_reason_counts: tuple[tuple[str, int], ...]
    switch_gate_min: float
    switch_gate_mean: float
    switch_gate_max: float
    track_switch_gate_ranges: tuple[tuple[str, float, float], ...]
    action_chain_guard_passed: bool
    action_chain_rollback_applied: bool
    action_chain_failures: tuple[str, ...]


@dataclass(frozen=True)
class EcologyOwnerDifference:
    body_id: int
    tick: int
    owner_name: str
    before_fingerprint: str
    after_fingerprint: str


@dataclass(frozen=True)
class EcologyFrozenEvaluationAudit:
    scenario: str
    seed: int
    rounds: int
    policy_stable: bool
    temporal_learning_stable: bool
    first_differences: tuple[EcologyOwnerDifference, ...]
    replay_settlement_coverage: float
    replay_lineage_coverage: float
    replay_drop_count: int
    passed: bool


@dataclass(frozen=True)
class EcologyMechanismGate:
    name: str
    passed: bool
    observed: str
    threshold: str


@dataclass(frozen=True)
class EcologyMechanismAuditReport:
    schema_version: str
    config: EcologyMechanismAuditConfig
    action_chain_snapshots: tuple[EcologyActionChainSnapshot, ...]
    segment_telemetry: tuple[EcologySegmentTelemetry, ...]
    frozen_evaluations: tuple[EcologyFrozenEvaluationAudit, ...]
    gates: tuple[EcologyMechanismGate, ...]
    verdict: str
    diagnostic_breakpoints: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _curriculum_config(
    config: EcologyMechanismAuditConfig,
) -> EcologyCurriculumConfig:
    return EcologyCurriculumConfig(
        n_ants=config.n_ants,
        temporal_latent_dim=config.temporal_latent_dim,
        stage_rounds=config.episode_rounds,
        stage_episodes=config.episodes_per_stage,
        mastery_min_episodes=min(3, config.episodes_per_stage),
        validation_rounds=config.evaluation_rounds,
        validation_seeds=(config.seed + 43,),
        heldout_rounds=config.evaluation_rounds,
        heldout_seeds=(config.seed + 101,),
        seed=config.seed,
        action_probe_code_delta_threshold=config.code_delta_threshold,
        action_probe_turn_delta_threshold=config.turn_delta_threshold,
        action_probe_retention_ratio=config.retention_ratio_threshold,
        action_probe_body_pass_ratio=config.body_pass_ratio,
    )


def _probe_by_kind(
    body_report: EcologyCheckpointActionProbe,
    kind: EcologyProbeKind,
):
    return next(probe for probe in body_report.probes if probe.kind is kind)


def _evaluate_action_snapshot(
    *,
    config: EcologyMechanismAuditConfig,
    arm: str,
    label: str,
    stage: str,
    tier: str,
    episode_index: int,
    body_reports: tuple[EcologyCheckpointActionProbe, ...],
    initial_reports: tuple[EcologyCheckpointActionProbe, ...],
) -> EcologyActionChainSnapshot:
    failures: list[str] = []
    passing_bodies = 0
    initial_by_body = {item.body_id: item for item in initial_reports}
    for body_report in body_reports:
        body_failures: list[str] = []
        baseline = initial_by_body[body_report.body_id]
        for kind in (EcologyProbeKind.FOOD, EcologyProbeKind.HEAT):
            probe = _probe_by_kind(body_report, kind)
            initial_probe = _probe_by_kind(baseline, kind)
            retention_floor = (
                initial_probe.turn_delta
                * config.retention_ratio_threshold
            )
            if not probe.input_reachable:
                body_failures.append(f"{kind.value}:input-unreachable")
            if probe.turn_delta < config.turn_delta_threshold:
                body_failures.append(
                    f"{kind.value}:turn-delta={probe.turn_delta:.9g}"
                )
            if probe.turn_delta < retention_floor:
                body_failures.append(
                    f"{kind.value}:retention={probe.turn_delta:.9g}/"
                    f"{initial_probe.turn_delta:.9g}"
                )
        obstacle = _probe_by_kind(
            body_report,
            EcologyProbeKind.OBSTACLE,
        )
        if not obstacle.input_reachable:
            body_failures.append("obstacle:input-unreachable")
        if body_failures:
            failures.extend(
                f"body:{body_report.body_id}:{item}"
                for item in body_failures
            )
        else:
            passing_bodies += 1
    required = max(1, math.ceil(len(body_reports) * config.body_pass_ratio))
    return EcologyActionChainSnapshot(
        arm=arm,
        label=label,
        stage=stage,
        tier=tier,
        episode_index=episode_index,
        body_reports=body_reports,
        required_body_passes=required,
        passing_bodies=passing_bodies,
        passed=passing_bodies >= required,
        failures=tuple(failures),
    )


def _segment_telemetry(
    *,
    arm: str,
    plan: EcologyTrainingEpisodePlan,
    runner: KernelColonyRunner,
    training_report: EcologyTrainingEpisodeReport,
) -> EcologySegmentTelemetry:
    records = tuple(
        step for round_record in runner.rounds for step in round_record.ant_steps
    )
    latest = tuple(
        session.trajectory[-1]
        for session in runner.sessions
        if session.trajectory
    )
    reason_counts: dict[str, int] = {}
    for record in latest:
        for reason, count in record.runtime_segment_close_reason_counts:
            reason_counts[reason] = reason_counts.get(reason, 0) + count
    switch_gates = tuple(record.switch_gate for record in records)
    track_values: dict[str, list[float]] = {}
    for record in records:
        for track_name, value in record.track_switch_gates:
            track_values.setdefault(track_name, []).append(value)
    return EcologySegmentTelemetry(
        arm=arm,
        stage=plan.stage.value,
        tier=plan.tier.value,
        episode_index=plan.episode_index,
        switch_count=sum(int(record.is_switching) for record in records),
        closed_segment_count=sum(
            record.runtime_closed_segments for record in latest
        ),
        longest_segment_length=max(
            (record.runtime_longest_segment_length for record in latest),
            default=0,
        ),
        close_reason_counts=tuple(sorted(reason_counts.items())),
        switch_gate_min=min(switch_gates, default=0.0),
        switch_gate_mean=(
            sum(switch_gates) / len(switch_gates)
            if switch_gates
            else 0.0
        ),
        switch_gate_max=max(switch_gates, default=0.0),
        track_switch_gate_ranges=tuple(
            (
                track_name,
                min(values),
                max(values),
            )
            for track_name, values in sorted(track_values.items())
        ),
        action_chain_guard_passed=(
            training_report.action_chain_guard_passed
        ),
        action_chain_rollback_applied=(
            training_report.action_chain_rollback_applied
        ),
        action_chain_failures=training_report.action_chain_failures,
    )


async def _probe_checkpoints(
    *,
    config: EcologyMechanismAuditConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    seed_offset: int,
) -> tuple[EcologyCheckpointActionProbe, ...]:
    return await run_ecology_checkpoint_action_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=config.seed + seed_offset,
        checkpoints=checkpoints,
        code_delta_threshold=config.code_delta_threshold,
        turn_delta_threshold=config.turn_delta_threshold,
    )


async def _train_audit_arm(
    *,
    audit_config: EcologyMechanismAuditConfig,
    curriculum_config: EcologyCurriculumConfig,
    initial: tuple[AntLearningCheckpoint, ...],
    arm: str,
    optimize: bool,
    initial_probes: tuple[EcologyCheckpointActionProbe, ...],
) -> tuple[
    tuple[AntLearningCheckpoint, ...],
    tuple[EcologyActionChainSnapshot, ...],
    tuple[EcologySegmentTelemetry, ...],
]:
    checkpoints = initial
    snapshots: list[EcologyActionChainSnapshot] = []
    segment_reports: list[EcologySegmentTelemetry] = []
    stages = (
        EcologyStage.BUTTER,
        EcologyStage.BURNING_MATCH,
        EcologyStage.COMPOSITE,
    )
    tiers = (
        EcologyTrainingTier.NEAR,
        EcologyTrainingTier.MEDIUM,
        EcologyTrainingTier.FAR,
    )
    for stage_index, stage in enumerate(stages):
        for episode_index in range(audit_config.episodes_per_stage):
            tier = tiers[min(episode_index, len(tiers) - 1)]
            plan = EcologyTrainingEpisodePlan(
                stage=stage,
                tier=tier,
                seed=(
                    audit_config.seed
                    + stage_index * 10_000
                    + episode_index * 101
                ),
                episode_index=episode_index,
                interleaved=False,
                forced_escape=(
                    stage is EcologyStage.BURNING_MATCH
                    and episode_index == 0
                ),
            )
            runner, checkpoints, training_report = (
                await _run_training_episode(
                    config=curriculum_config,
                    checkpoints=checkpoints,
                    arm=arm,
                    optimize=optimize,
                    local_valence_enabled=True,
                    segment_credit_enabled=True,
                    plan=plan,
                    action_probe_baseline=initial,
                    action_probe_baseline_reports=initial_probes,
                )
            )
            body_reports = await _probe_checkpoints(
                config=audit_config,
                checkpoints=checkpoints,
                # Retention is a within-probe comparison: every checkpoint
                # must see the exact same paired worlds and session seeds.
                seed_offset=700_003,
            )
            snapshots.append(
                _evaluate_action_snapshot(
                    config=audit_config,
                    arm=arm,
                    label=(
                        f"{stage.value}:{tier.value}:"
                        f"episode:{episode_index}"
                    ),
                    stage=stage.value,
                    tier=tier.value,
                    episode_index=episode_index,
                    body_reports=body_reports,
                    initial_reports=initial_probes,
                )
            )
            segment_reports.append(
                _segment_telemetry(
                    arm=arm,
                    plan=plan,
                    runner=runner,
                    training_report=training_report,
                )
            )
    return checkpoints, tuple(snapshots), tuple(segment_reports)


def _owner_map(
    checkpoint: AntLearningCheckpoint,
) -> dict[str, str]:
    return dict(checkpoint.learning_owner_fingerprints)


async def _frozen_evaluation_audit(
    *,
    audit_config: EcologyMechanismAuditConfig,
    curriculum_config: EcologyCurriculumConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    scenario: EcologyEvaluationScenario,
    seed: int,
) -> EcologyFrozenEvaluationAudit:
    stage = {
        EcologyEvaluationScenario.BUTTER_ONLY: EcologyStage.BUTTER,
        EcologyEvaluationScenario.HEAT_FORCED_ESCAPE: (
            EcologyStage.BURNING_MATCH
        ),
    }[scenario]
    runner = KernelColonyRunner(
        _world(
            config=curriculum_config,
            stage=stage,
            seed=seed,
            data_split=EcologyDataSplit.HELDOUT,
            tier=EcologyTrainingTier.FAR,
            forced_escape=(
                scenario is EcologyEvaluationScenario.HEAT_FORCED_ESCAPE
            ),
        ),
        base_config=_session_config(
            config=curriculum_config,
            seed=seed,
            session_id=f"ecology:p0:frozen:{scenario.value}:{seed}",
            optimize=False,
            learning_enabled=False,
            sparse_exploration_enabled=False,
        ),
    )
    runner.restore_learning_checkpoints(checkpoints)
    previous = runner.export_learning_checkpoints(
        checkpoint_prefix="ecology:p0:frozen:before",
        include_runtime_replay=False,
    )
    first_differences: dict[tuple[int, str], EcologyOwnerDifference] = {}
    for tick in range(audit_config.evaluation_rounds):
        await runner.step_round()
        current = runner.export_learning_checkpoints(
            checkpoint_prefix=f"ecology:p0:frozen:tick:{tick}",
            include_runtime_replay=False,
        )
        for body_id, (before, after) in enumerate(
            zip(previous, current, strict=True)
        ):
            before_map = _owner_map(before)
            after_map = _owner_map(after)
            for owner_name in sorted(before_map.keys() | after_map.keys()):
                if before_map.get(owner_name) != after_map.get(owner_name):
                    first_differences.setdefault(
                        (body_id, owner_name),
                        EcologyOwnerDifference(
                            body_id=body_id,
                            tick=tick,
                            owner_name=owner_name,
                            before_fingerprint=before_map.get(
                                owner_name,
                                "",
                            ),
                            after_fingerprint=after_map.get(owner_name, ""),
                        ),
                    )
        previous = current
    final = previous
    policy_stable = all(
        before.policy_fingerprint == after.policy_fingerprint
        for before, after in zip(checkpoints, final, strict=True)
    )
    temporal_stable = all(
        before.temporal_learning_fingerprint
        == after.temporal_learning_fingerprint
        for before, after in zip(checkpoints, final, strict=True)
    )
    latest = tuple(
        session.trajectory[-1]
        for session in runner.sessions
        if session.trajectory
    )
    captured = sum(item.runtime_replay_captured for item in latest)
    settled = sum(item.runtime_replay_settled for item in latest)
    lineage = sum(item.runtime_replay_lineage_matches for item in latest)
    pending = sum(item.runtime_replay_pending_captures for item in latest)
    eligible = captured - pending
    drops = sum(len(item.runtime_replay_drop_reasons) for item in latest)
    settlement_coverage = settled / eligible if eligible else 0.0
    lineage_coverage = lineage / settled if settled else 0.0
    passed = (
        policy_stable
        and temporal_stable
        and settlement_coverage >= 0.99
        and lineage_coverage >= 0.99
        and drops == 0
    )
    return EcologyFrozenEvaluationAudit(
        scenario=scenario.value,
        seed=seed,
        rounds=audit_config.evaluation_rounds,
        policy_stable=policy_stable,
        temporal_learning_stable=temporal_stable,
        first_differences=tuple(first_differences.values()),
        replay_settlement_coverage=settlement_coverage,
        replay_lineage_coverage=lineage_coverage,
        replay_drop_count=drops,
        passed=passed,
    )


async def run_ecology_mechanism_audit(
    config: EcologyMechanismAuditConfig,
) -> EcologyMechanismAuditReport:
    """Run the complete P0 audit without emitting a promotion checkpoint."""

    curriculum_config = _curriculum_config(config)
    bootstrap = KernelColonyRunner(
        _world(
            config=curriculum_config,
            stage=EcologyStage.COMPOSITE,
            seed=config.seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
        ),
        base_config=_session_config(
            config=curriculum_config,
            seed=config.seed,
            session_id="ecology:p0:shared-initial",
            optimize=True,
        ),
    )
    initial = bootstrap.export_learning_checkpoints(
        checkpoint_prefix="ecology:p0:shared-initial",
        include_runtime_replay=False,
    )
    initial_probes = await _probe_checkpoints(
        config=config,
        checkpoints=initial,
        seed_offset=700_003,
    )
    initial_snapshot = _evaluate_action_snapshot(
        config=config,
        arm="shared-initial",
        label="shared-initial",
        stage="initial",
        tier="initial",
        episode_index=-1,
        body_reports=initial_probes,
        initial_reports=initial_probes,
    )
    learned, learned_snapshots, learned_segments = await _train_audit_arm(
        audit_config=config,
        curriculum_config=curriculum_config,
        initial=initial,
        arm="learned",
        optimize=True,
        initial_probes=initial_probes,
    )
    no_optimize, no_opt_snapshots, no_opt_segments = (
        await _train_audit_arm(
            audit_config=config,
            curriculum_config=curriculum_config,
            initial=initial,
            arm="no_optimize",
            optimize=False,
            initial_probes=initial_probes,
        )
    )
    frozen = (
        await _frozen_evaluation_audit(
            audit_config=config,
            curriculum_config=curriculum_config,
            checkpoints=learned,
            scenario=EcologyEvaluationScenario.BUTTER_ONLY,
            seed=config.seed + 101,
        ),
        await _frozen_evaluation_audit(
            audit_config=config,
            curriculum_config=curriculum_config,
            checkpoints=learned,
            scenario=EcologyEvaluationScenario.HEAT_FORCED_ESCAPE,
            seed=config.seed + 211,
        ),
    )
    action_snapshots = (
        (initial_snapshot,) + learned_snapshots + no_opt_snapshots
    )
    segment_telemetry = learned_segments + no_opt_segments
    action_chain_ok = all(item.passed for item in action_snapshots)
    no_optimize_stable = all(
        before.policy_fingerprint == after.policy_fingerprint
        for before, after in zip(initial, no_optimize, strict=True)
    )
    learned_segment_reason_counts: dict[str, int] = {}
    for item in learned_segments:
        for reason, count in item.close_reason_counts:
            learned_segment_reason_counts[reason] = (
                learned_segment_reason_counts.get(reason, 0) + count
            )
    temporal_switch_ok = (
        any(item.switch_count > 0 for item in learned_segments)
        and learned_segment_reason_counts.get("beta-switch", 0) > 0
        and (
            learned_segment_reason_counts.get(
                "environment-milestone",
                0,
            )
            > 0
        )
    )
    frozen_ok = all(item.passed for item in frozen)
    gates = (
        EcologyMechanismGate(
            name="action_chain",
            passed=action_chain_ok,
            observed=(
                f"passing_snapshots={sum(item.passed for item in action_snapshots)}"
                f"/{len(action_snapshots)}"
            ),
            threshold=(
                "food/heat code reachable and turn-sensitive with retained "
                "sensitivity for the required body ratio"
            ),
        ),
        EcologyMechanismGate(
            name="no_optimize_policy_stable",
            passed=no_optimize_stable,
            observed=str(no_optimize_stable),
            threshold="no-optimize policy fingerprints equal shared initial",
        ),
        EcologyMechanismGate(
            name="temporal_switch_and_closure",
            passed=temporal_switch_ok,
            observed=str(
                {
                    "switches": sum(
                        item.switch_count for item in learned_segments
                    ),
                    "close_reasons": tuple(
                        sorted(learned_segment_reason_counts.items())
                    ),
                }
            ),
            threshold=(
                "at least one learned beta switch, beta-switch closure, and "
                "environment-milestone closure"
            ),
        ),
        EcologyMechanismGate(
            name="frozen_evaluation",
            passed=frozen_ok,
            observed=str(
                tuple(
                    (
                        item.scenario,
                        item.policy_stable,
                        item.temporal_learning_stable,
                        item.replay_settlement_coverage,
                        item.replay_lineage_coverage,
                        item.replay_drop_count,
                    )
                    for item in frozen
                )
            ),
            threshold=(
                "policy/temporal-learning stable; replay settlement and "
                "lineage >=0.99 with no drops"
            ),
        ),
    )
    breakpoints = tuple(gate.name for gate in gates if not gate.passed)
    verdict = "PASS" if not breakpoints else "BLOCK"
    return EcologyMechanismAuditReport(
        schema_version=ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION,
        config=config,
        action_chain_snapshots=action_snapshots,
        segment_telemetry=segment_telemetry,
        frozen_evaluations=frozen,
        gates=gates,
        verdict=verdict,
        diagnostic_breakpoints=breakpoints,
        description=(
            f"{verdict}: "
            + (", ".join(breakpoints) if breakpoints else "all P0 gates passed")
        ),
    )


__all__ = [
    "ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION",
    "EcologyActionChainSnapshot",
    "EcologyFrozenEvaluationAudit",
    "EcologyMechanismAuditConfig",
    "EcologyMechanismAuditReport",
    "EcologyMechanismGate",
    "EcologyOwnerDifference",
    "EcologySegmentTelemetry",
    "run_ecology_mechanism_audit",
]
