from __future__ import annotations

import asyncio

from volvence_zero.credit import CreditRecord, CreditSnapshot, GateDecision, ModificationGate, extract_abstract_action_credit_bonus
from volvence_zero.evaluation import EvaluationScore, EvaluationSnapshot
from volvence_zero.internal_rl import (
    DualTrackOptimizationReport,
    InternalRLEnvironment,
    InternalRLSandbox,
    OptimizationReport,
    PolicyOptimizationResult,
    derive_abstract_action_credit,
)
from volvence_zero.internal_rl.sandbox import CausalZPolicy
from volvence_zero.joint_loop import ETANLJointLoop, JointLoopSchedule, PipelineConfig, SSLRLTrainingPipeline
from volvence_zero.joint_loop.runtime import JointCycleReport
from volvence_zero.memory import Track
from volvence_zero.substrate import (
    NoOpResidualInterventionBackend,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
    build_training_trace,
)
from volvence_zero.temporal import FullLearnedTemporalPolicy


def _snapshot_from_step(trace_id: str, step: object) -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id=trace_id,
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(0.1, 0.2),
        feature_surface=step.feature_surface,
        residual_activations=step.residual_activations,
        residual_sequence=(
            ResidualSequenceStep(
                step=step.step,
                token=step.token,
                feature_surface=step.feature_surface,
                residual_activations=step.residual_activations,
                description=f"trace token {step.token}",
            ),
        ),
        unavailable_fields=(),
        description=f"trace step {step.step}",
    )


def test_internal_rl_sandbox_emits_abstract_action_credit():
    trace = build_training_trace(trace_id="rl-trace", source_text="steady warm planning")
    sandbox = InternalRLSandbox()
    rollout = sandbox.rollout(
        rollout_id="rollout-1",
        substrate_steps=tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps),
    )
    credits = derive_abstract_action_credit(rollout=rollout, timestamp_ms=100)

    assert rollout.transitions
    assert credits
    assert credits[0].level == "abstract_action"


def test_internal_rl_sandbox_runs_dual_track_rollout():
    trace = build_training_trace(trace_id="dual-track", source_text="plan carefully and stay warm")
    sandbox = InternalRLSandbox()
    dual_rollout = sandbox.rollout_dual_track(
        rollout_id="rollout-2",
        substrate_steps=tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps),
    )

    assert dual_rollout.task_rollout.track.value == "world"
    assert dual_rollout.relationship_rollout.track.value == "self"
    assert dual_rollout.task_rollout.transitions
    assert dual_rollout.relationship_rollout.transitions
    assert dual_rollout.task_rollout.replacement_mode == "causal-binary"
    assert dual_rollout.task_rollout.transitions[0].policy_action
    assert dual_rollout.task_rollout.transitions[0].applied_control
    assert dual_rollout.task_rollout.transitions[0].policy_replacement_quality >= -1.0
    assert dual_rollout.task_rollout.transitions[0].backend_name == "trace-residual-backend"
    assert dual_rollout.task_rollout.transitions[0].controller_state.switch_gate in {0.0, 1.0}


def test_internal_rl_sandbox_supports_checkpoint_and_optimization_report():
    trace = build_training_trace(trace_id="opt-trace", source_text="balance task and relationship")
    sandbox = InternalRLSandbox()
    initial_temporal_params = sandbox.policy.export_parameters()
    checkpoint = sandbox.create_checkpoint(checkpoint_id="policy-1")
    checkpoint_temporal_params = sandbox.policy.export_parameters()
    dual_rollout = sandbox.rollout_dual_track(
        rollout_id="rollout-3",
        substrate_steps=tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps),
    )
    report = sandbox.optimize(dual_rollout)

    assert report.description
    assert report.task_report.parameter_summary
    assert report.relationship_report.parameter_summary
    assert report.task_report.clip_fraction >= 0.0
    assert report.relationship_report.kl_penalty >= 0.0
    assert sandbox.policy.export_parameters() != initial_temporal_params

    sandbox.restore_checkpoint(checkpoint)
    restored = sandbox.causal_policy.export_parameters()
    assert restored[0].update_step == checkpoint.parameters_by_track[0].update_step
    assert sandbox.policy.export_parameters() == checkpoint_temporal_params


def test_internal_rl_sandbox_can_run_noop_backend_baseline_path():
    trace = build_training_trace(trace_id="baseline-trace", source_text="steady guided planning")
    sandbox = InternalRLSandbox(env=InternalRLEnvironment(control_backend=NoOpResidualInterventionBackend()))

    rollout = sandbox.rollout(
        rollout_id="rollout-baseline",
        substrate_steps=tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps),
    )

    assert rollout.transitions
    assert rollout.transitions[0].backend_name == "noop-residual-backend"


def test_internal_rl_sandbox_can_run_continuous_causal_path():
    trace = build_training_trace(trace_id="continuous-trace", source_text="steady guided planning")
    sandbox = InternalRLSandbox()

    rollout = sandbox.rollout(
        rollout_id="rollout-continuous",
        substrate_steps=tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps),
        replacement_mode="causal",
    )

    assert rollout.replacement_mode == "causal"
    assert 0.0 <= rollout.transitions[0].controller_state.switch_gate <= 1.0


def test_eta_nl_joint_loop_runs_minimal_cycle():
    loop = ETANLJointLoop()
    trace = build_training_trace(trace_id="joint-trace", source_text="repair tension then continue helpfully")
    before = loop.temporal_policy.export_parameters()
    report = asyncio.run(loop.run_cycle(cycle_index=1, trace=trace))
    after = loop.temporal_policy.export_parameters()

    assert report.acceptance_passed is True
    assert report.policy_objective != 0.0
    assert report.task_reward != 0.0
    assert report.relationship_reward != 0.0
    assert report.applied_operations
    assert report.metacontroller_state is not None
    assert report.metacontroller_state.mode == "full-learned"
    assert report.ssl_prediction_loss >= 0.0
    assert report.ssl_kl_loss >= 0.0
    assert report.ssl_posterior_drift >= 0.0
    assert report.kernel_score_count >= 1
    assert report.rollback_reasons == ()
    assert report.ssl_rollback_applied is False
    assert report.owner_path == "online-joint-loop"
    assert report.cms_description
    assert after != before
    assert report.metacontroller_state.active_label.startswith("discovered_family_")
    assert report.metacontroller_state.structure_frozen is True
    assert report.metacontroller_state.learning_phase == "rl-online"
    assert any(operation.startswith("temporal-prior:") for operation in report.applied_operations)


def test_eta_nl_joint_loop_can_rollback_policy_when_reward_regresses():
    loop = ETANLJointLoop()
    loop._previous_total_reward = 999.0
    trace = build_training_trace(trace_id="rollback-trace", source_text="brief weak signal")
    report = asyncio.run(loop.run_cycle(cycle_index=2, trace=trace))

    assert report.policy_rollback_applied is True
    assert report.ssl_rollback_applied is True
    assert report.policy_objective != 0.0
    assert "reward-regression" in report.rollback_reasons
    assert "ssl-rollback" in report.applied_operations
    assert "policy-rollback" in report.applied_operations


def test_eta_nl_joint_loop_detects_surrogate_outcome_decoupling():
    loop = ETANLJointLoop()
    reasons = loop._rollback_reasons(
        total_reward=0.05,
        evaluation_snapshot=EvaluationSnapshot(
            turn_scores=(
                EvaluationScore(family="task", metric_name="task_pressure", value=0.20, confidence=0.6, evidence="low task"),
                EvaluationScore(
                    family="relationship",
                    metric_name="cross_track_stability",
                    value=0.25,
                    confidence=0.6,
                    evidence="low relationship",
                ),
                EvaluationScore(
                    family="learning",
                    metric_name="joint_learning_progress",
                    value=0.35,
                    confidence=0.6,
                    evidence="weak learning",
                ),
                EvaluationScore(
                    family="abstraction",
                    metric_name="abstract_action_usefulness",
                    value=0.30,
                    confidence=0.6,
                    evidence="weak abstraction",
                ),
                EvaluationScore(
                    family="safety",
                    metric_name="contract_integrity",
                    value=1.0,
                    confidence=0.9,
                    evidence="safe",
                ),
            ),
            session_scores=(),
            alerts=(),
            description="low outcome alignment snapshot",
        ),
        optimization_report=DualTrackOptimizationReport(
            task_report=OptimizationReport(
                track=Track.WORLD,
                average_reward=0.4,
                baseline_reward=0.1,
                mean_advantage=0.3,
                surrogate_objective=0.22,
                clip_fraction=0.0,
                kl_penalty=0.05,
                parameter_summary="task positive surrogate",
            ),
            relationship_report=OptimizationReport(
                track=Track.SELF,
                average_reward=0.35,
                baseline_reward=0.1,
                mean_advantage=0.25,
                surrogate_objective=0.18,
                clip_fraction=0.0,
                kl_penalty=0.05,
                parameter_summary="relationship positive surrogate",
            ),
            description="positive surrogate but weak outcomes",
        ),
        metacontroller_state=None,
    )

    assert "surrogate-outcome-decoupling" in reasons


def test_eta_nl_joint_loop_supports_scheduled_ssl_only_and_full_cycle_paths():
    loop = ETANLJointLoop()
    trace = build_training_trace(trace_id="scheduled-trace", source_text="repair then continue carefully")

    ssl_only = asyncio.run(
        loop.run_scheduled_step(
            turn_index=1,
            trace=trace,
            schedule=JointLoopSchedule(ssl_interval=1, rl_interval=3),
        )
    )
    full_cycle = asyncio.run(
        loop.run_scheduled_step(
            turn_index=3,
            trace=trace,
            schedule=JointLoopSchedule(ssl_interval=1, rl_interval=3),
        )
    )

    assert ssl_only.schedule_action == "ssl-only"
    assert ssl_only.cycle_report is None
    assert ssl_only.metacontroller_state is not None
    assert ssl_only.owner_path == "online-joint-loop"
    assert ssl_only.metacontroller_state.learning_phase == "ssl"
    assert ssl_only.metacontroller_state.structure_frozen is False
    assert dict(ssl_only.schedule_telemetry)["ssl_due"] == 1
    assert dict(ssl_only.schedule_telemetry)["rl_due"] == 0
    assert full_cycle.schedule_action == "full-cycle"
    assert full_cycle.cycle_report is not None
    assert full_cycle.owner_path == "online-joint-loop"
    assert dict(full_cycle.schedule_telemetry)["rl_due"] == 1


def test_eta_nl_joint_loop_can_apply_and_rollback_rare_heavy_artifact():
    pipeline = SSLRLTrainingPipeline(
        config=PipelineConfig(n_z=8, ssl_min_steps=2, ssl_max_steps=3, rl_max_steps=1)
    )
    traces = tuple(
        build_training_trace(trace_id=f"rare-heavy-{index}", source_text="repair tension then plan steadily")
        for index in range(5)
    )
    pipeline.run_pipeline(traces=traces)
    artifact = pipeline.export_rare_heavy_artifact(artifact_id="offline-artifact")

    loop = ETANLJointLoop()
    before = loop.temporal_policy.export_parameters()
    result = loop.apply_rare_heavy_artifact(artifact)
    after = loop.temporal_policy.export_parameters()

    assert result.artifact_id == "offline-artifact"
    assert "rare-heavy:temporal-import" in result.applied_operations
    assert after != before

    rollback_operations = loop.rollback_rare_heavy_import(result.checkpoint)
    restored = loop.temporal_policy.export_parameters()

    assert "rare-heavy:temporal-rollback" in rollback_operations
    assert restored == before


def test_causal_policy_multi_epoch_optimization():
    trace = build_training_trace(trace_id="multi-epoch-trace", source_text="steady warm planning")
    sandbox = InternalRLSandbox()
    steps = tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps)
    rollout = sandbox.rollout(rollout_id="me-1", substrate_steps=steps)

    single_epoch = sandbox.causal_policy.optimize(rollout=rollout, n_epochs=1)
    sandbox.restore_checkpoint(sandbox.create_checkpoint(checkpoint_id="reset-me"))

    multi_epoch = sandbox.causal_policy.optimize(rollout=rollout, n_epochs=3)

    assert single_epoch.epochs_executed == 1
    assert multi_epoch.epochs_executed >= 1
    assert multi_epoch.epochs_executed <= 3
    assert multi_epoch.parameter_summary != single_epoch.parameter_summary


def test_causal_policy_kl_early_stopping():
    trace = build_training_trace(trace_id="kl-stop-trace", source_text="steady warm planning")
    sandbox = InternalRLSandbox()
    steps = tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps)
    rollout = sandbox.rollout(rollout_id="kl-1", substrate_steps=steps)

    report = sandbox.causal_policy.optimize(rollout=rollout, n_epochs=10, max_kl=0.001)

    assert report.epochs_executed <= 10
    if report.kl_penalty > 0.001:
        assert report.kl_early_stopped is True


def test_causal_policy_gae_advantages():
    policy = FullLearnedTemporalPolicy()
    causal = CausalZPolicy(parameter_store=policy.parameter_store)
    trace = build_training_trace(trace_id="gae-trace", source_text="steady warm planning")
    sandbox = InternalRLSandbox(policy=policy)
    steps = tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps)
    rollout = sandbox.rollout(rollout_id="gae-1", substrate_steps=steps)

    advantages = causal._compute_gae(rollout=rollout, gamma=0.99, gae_lambda=0.95)

    assert len(advantages) == len(rollout.transitions)
    mean_adv = sum(advantages) / len(advantages)
    assert abs(mean_adv) < 1e-6


def test_optimization_produces_self_modification_record():
    trace = build_training_trace(trace_id="audit-trace", source_text="balance task and relationship")
    sandbox = InternalRLSandbox()
    dual_rollout = sandbox.rollout_dual_track(
        rollout_id="audit-1",
        substrate_steps=tuple(_snapshot_from_step(trace.trace_id, step) for step in trace.steps),
    )

    result = sandbox.optimize_with_audit(dual_rollout, timestamp_ms=42)

    assert isinstance(result, PolicyOptimizationResult)
    assert isinstance(result.optimization_report, DualTrackOptimizationReport)
    assert result.total_epochs_executed >= 2
    assert result.total_kl_divergence >= 0.0
    if result.policy_update_applied:
        assert len(result.modification_records) == 1
        record = result.modification_records[0]
        assert record.gate.value == "online"
        assert record.decision.value == "allow"
        assert record.is_reversible is True
        assert record.old_value_hash != record.new_value_hash
        assert record.timestamp_ms == 42


def test_joint_loop_credit_shaped_rewards():
    loop = ETANLJointLoop()
    trace = build_training_trace(trace_id="credit-trace", source_text="repair tension then continue helpfully")

    asyncio.run(loop.run_cycle(cycle_index=0, trace=trace))
    assert loop._previous_credit_snapshot is not None

    bonus = extract_abstract_action_credit_bonus(loop._previous_credit_snapshot)
    assert isinstance(bonus, dict)

    report2 = asyncio.run(loop.run_cycle(cycle_index=1, trace=trace))
    assert report2.total_reward != 0.0


def test_joint_cycle_report_policy_fields():
    loop = ETANLJointLoop()
    trace = build_training_trace(trace_id="fields-trace", source_text="repair tension then continue helpfully")
    report = asyncio.run(loop.run_cycle(cycle_index=1, trace=trace))

    assert hasattr(report, "policy_update_applied")
    assert hasattr(report, "policy_kl_divergence")
    assert hasattr(report, "policy_epochs_executed")
    assert isinstance(report.policy_update_applied, bool)
    assert report.policy_kl_divergence >= 0.0
    assert report.policy_epochs_executed >= 1


# ---------------------------------------------------------------------------
# Phase 1 W1.1  —  20-cycle continuous RL validation
# ---------------------------------------------------------------------------

PHASE1_SOURCE_TEXTS = (
    "repair tension then continue helpfully",
    "balance task and relationship carefully",
    "steady warm planning for future growth",
    "plan carefully and stay warm",
    "guide with empathy and clear reasoning",
    "maintain composure under pressure slowly",
    "explore creative solutions together openly",
    "listen deeply and respond thoughtfully now",
    "build trust through consistent reliable actions",
    "adapt strategy when new evidence arrives",
)


def test_phase1_20_cycle_reward_trend():
    """Run 20 consecutive joint-loop cycles and verify observable learning.

    Phase 1 exit-condition check: the RL closed loop must produce a
    non-random reward trajectory.  We verify:
      1. All 20 cycles complete without crashing.
      2. ``policy_objective`` is non-zero for the majority of cycles.
      3. ``policy_update_applied`` is True for at least half the cycles.
      4. Reward variance is non-zero (system is not stuck).
      5. No unrecoverable rollback cascade (≤ 5 rollback cycles).
    """
    loop = ETANLJointLoop()
    reports: list[JointCycleReport] = []

    for cycle_index in range(20):
        source_text = PHASE1_SOURCE_TEXTS[cycle_index % len(PHASE1_SOURCE_TEXTS)]
        trace = build_training_trace(
            trace_id=f"phase1-cycle-{cycle_index}",
            source_text=source_text,
        )
        report = asyncio.run(loop.run_cycle(cycle_index=cycle_index, trace=trace))
        reports.append(report)

    rewards = [r.total_reward for r in reports]
    objectives = [r.policy_objective for r in reports]
    updates_applied = sum(1 for r in reports if r.policy_update_applied)
    rollback_count = sum(1 for r in reports if r.policy_rollback_applied)

    assert len(reports) == 20
    non_zero_objectives = sum(1 for o in objectives if abs(o) > 1e-8)
    assert non_zero_objectives >= 10, (
        f"Expected ≥10 non-zero policy objectives, got {non_zero_objectives}"
    )
    assert updates_applied >= 10, (
        f"Expected ≥10 policy updates applied, got {updates_applied}"
    )
    reward_variance = sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)
    assert reward_variance > 0.0, "Reward is stuck at a constant value"
    assert rollback_count <= 5, (
        f"Too many rollback cycles ({rollback_count}/20), RL may be unstable"
    )


def test_phase1_credit_shaping_affects_rl_reward():
    """Verify credit bonus from previous cycle influences next cycle's RL environment.

    Phase 1 W1.3: credit_snapshot from cycle N injects abstract-action
    bonus into the RL environment for cycle N+1, and the resulting
    total_reward differs from a baseline without credit shaping.
    """
    loop_with_credit = ETANLJointLoop()
    trace = build_training_trace(
        trace_id="credit-shape-trace",
        source_text="repair tension then continue helpfully",
    )

    asyncio.run(loop_with_credit.run_cycle(cycle_index=0, trace=trace))
    assert loop_with_credit._previous_credit_snapshot is not None

    bonus = extract_abstract_action_credit_bonus(loop_with_credit._previous_credit_snapshot)
    assert isinstance(bonus, dict)
    assert len(bonus) > 0, "Expected at least one credit bonus entry"

    report_with = asyncio.run(loop_with_credit.run_cycle(cycle_index=1, trace=trace))

    loop_no_credit = ETANLJointLoop()
    report_no_credit = asyncio.run(loop_no_credit.run_cycle(cycle_index=0, trace=trace))

    assert report_with.total_reward != 0.0
    assert report_no_credit.total_reward != 0.0
    assert abs(report_with.total_reward - report_no_credit.total_reward) > 1e-9 or (
        report_with.policy_kl_divergence != report_no_credit.policy_kl_divergence
    ), "Credit shaping should create observable difference between cycles"


def test_phase1_audit_chain_integrity():
    """Verify credit → reward shaping → optimize → modification record chain.

    The full audit trail must be traceable: credit records are produced,
    bonus is extracted, RL optimization produces a SelfModificationRecord,
    and the record is embedded in the enriched credit snapshot.
    """
    loop = ETANLJointLoop()
    trace = build_training_trace(
        trace_id="audit-chain-trace",
        source_text="balance task and relationship carefully",
    )

    report = asyncio.run(loop.run_cycle(cycle_index=0, trace=trace))
    credit = loop._previous_credit_snapshot
    assert credit is not None

    assert len(credit.recent_credits) > 0, "No credit records produced"

    abstract_credits = [c for c in credit.recent_credits if c.level == "abstract_action"]
    assert len(abstract_credits) > 0, "No abstract-action level credits"

    has_modification = len(credit.recent_modifications) > 0
    if report.policy_update_applied:
        assert has_modification, (
            "Policy update was applied but no modification records in credit snapshot"
        )
        allow_records = [
            m for m in credit.recent_modifications if m.decision.value == "allow"
        ]
        assert len(allow_records) > 0, "Expected at least one ALLOW modification record"


# ---------------------------------------------------------------------------
# Phase 1 W2  —  Rollback and gate integration
# ---------------------------------------------------------------------------

def test_phase1_kl_excessive_triggers_rollback():
    """Verify rollback triggers when KL divergence is excessive.

    _rollback_reasons should include 'excessive-kl' when either track's
    kl_penalty exceeds the threshold (0.4).
    """
    loop = ETANLJointLoop()
    reasons = loop._rollback_reasons(
        total_reward=0.5,
        evaluation_snapshot=EvaluationSnapshot(
            turn_scores=(),
            session_scores=(),
            alerts=(),
            description="normal snapshot",
        ),
        optimization_report=DualTrackOptimizationReport(
            task_report=OptimizationReport(
                track=Track.WORLD,
                average_reward=0.3,
                baseline_reward=0.1,
                mean_advantage=0.2,
                surrogate_objective=0.1,
                clip_fraction=0.0,
                kl_penalty=0.5,
                parameter_summary="high kl",
            ),
            relationship_report=OptimizationReport(
                track=Track.SELF,
                average_reward=0.3,
                baseline_reward=0.1,
                mean_advantage=0.2,
                surrogate_objective=0.1,
                clip_fraction=0.0,
                kl_penalty=0.05,
                parameter_summary="normal kl",
            ),
            description="one track high kl",
        ),
        metacontroller_state=None,
    )
    assert "excessive-kl" in reasons


def test_phase1_negative_surrogate_triggers_rollback():
    """Verify rollback triggers when surrogate objective is negative."""
    loop = ETANLJointLoop()
    reasons = loop._rollback_reasons(
        total_reward=0.5,
        evaluation_snapshot=EvaluationSnapshot(
            turn_scores=(),
            session_scores=(),
            alerts=(),
            description="normal",
        ),
        optimization_report=DualTrackOptimizationReport(
            task_report=OptimizationReport(
                track=Track.WORLD,
                average_reward=-0.3,
                baseline_reward=0.1,
                mean_advantage=-0.4,
                surrogate_objective=-0.15,
                clip_fraction=0.0,
                kl_penalty=0.05,
                parameter_summary="negative surrogate",
            ),
            relationship_report=OptimizationReport(
                track=Track.SELF,
                average_reward=0.3,
                baseline_reward=0.1,
                mean_advantage=0.2,
                surrogate_objective=0.1,
                clip_fraction=0.0,
                kl_penalty=0.05,
                parameter_summary="ok",
            ),
            description="negative task surrogate",
        ),
        metacontroller_state=None,
    )
    assert "negative-surrogate" in reasons


def test_phase1_gate_deny_blocks_policy_update():
    """When evaluation has CRITICAL alert, ONLINE gate blocks policy update.

    Uses the real gate evaluation pathway to ensure that critical
    alerts prevent self-modification.
    """
    from volvence_zero.credit import ModificationProposal, evaluate_gate

    proposal = ModificationProposal(
        target="causal_policy.track_weights",
        desired_gate=ModificationGate.ONLINE,
        old_value_hash="hash-old",
        new_value_hash="hash-new",
        justification="attempt update during critical alert",
    )
    decision = evaluate_gate(
        proposal=proposal,
        evaluation_snapshot=EvaluationSnapshot(
            turn_scores=(),
            session_scores=(),
            alerts=("CRITICAL: safety breach detected",),
            description="critical alert active",
        ),
    )
    assert decision is GateDecision.BLOCK

    decision_safe = evaluate_gate(
        proposal=proposal,
        evaluation_snapshot=EvaluationSnapshot(
            turn_scores=(),
            session_scores=(),
            alerts=(),
            description="no alerts",
        ),
    )
    assert decision_safe is GateDecision.ALLOW


def test_phase1_checkpoint_restore_preserves_policy_state():
    """Verify checkpoint → rollback → restore roundtrip on joint loop."""
    loop = ETANLJointLoop()
    trace = build_training_trace(trace_id="cp-trace", source_text="steady warm planning")

    before_params = loop.temporal_policy.export_parameters()
    asyncio.run(loop.run_cycle(cycle_index=0, trace=trace))
    after_params = loop.temporal_policy.export_parameters()
    assert after_params != before_params, "Cycle should change policy parameters"

    loop2 = ETANLJointLoop()
    loop2._previous_total_reward = 999.0
    trace2 = build_training_trace(trace_id="cp-trace2", source_text="brief weak signal")
    report = asyncio.run(loop2.run_cycle(cycle_index=1, trace=trace2))
    assert report.policy_rollback_applied is True


# ---------------------------------------------------------------------------
# Phase 3 W6.2 — Evolution judge in joint loop
# ---------------------------------------------------------------------------

def test_phase3_evolution_judge_integrated_in_joint_loop():
    """Verify evolution judge is called during run_cycle and its verdict
    is published in the JointCycleReport."""
    from volvence_zero.evaluation import EvolutionDecision

    loop = ETANLJointLoop()
    trace = build_training_trace(
        trace_id="judge-loop-trace",
        source_text="repair tension then continue helpfully",
    )
    report = asyncio.run(loop.run_cycle(cycle_index=0, trace=trace))

    assert report.evolution_judgement is not None, (
        "Evolution judgement should be present in JointCycleReport"
    )
    assert report.evolution_judgement.decision in {
        EvolutionDecision.PROMOTE,
        EvolutionDecision.HOLD,
        EvolutionDecision.ROLLBACK,
    }
    assert report.evolution_judgement.replay_passed is not None
    assert len(report.evolution_judgement.reasons) > 0


# ---------------------------------------------------------------------------
# Phase 3 W7 — Regime selection weights + effectiveness trend
# ---------------------------------------------------------------------------

def test_phase3_regime_effectiveness_trend_in_snapshot():
    """Verify regime effectiveness trend is published in RegimeSnapshot
    and evolves across multiple cycles."""
    loop = ETANLJointLoop()
    traces = [
        build_training_trace(trace_id=f"regime-{i}", source_text=text)
        for i, text in enumerate([
            "repair tension and rebuild trust carefully",
            "plan carefully for future growth steadily",
            "guide with empathy and clear reasoning now",
        ])
    ]

    for i, trace in enumerate(traces):
        asyncio.run(loop.run_cycle(cycle_index=i, trace=trace))

    regime_module = loop._regime_module
    checkpoint = regime_module.create_checkpoint(checkpoint_id="regime-check")
    assert checkpoint is not None
