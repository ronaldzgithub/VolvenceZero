from __future__ import annotations

import asyncio
from dataclasses import replace
import math

import pytest

from volvence_zero.agent.response import LLMResponseSynthesizer, ResponseContext
from volvence_zero.application.runtime import (
    ResponseAssemblySnapshot,
    ResponseMode,
    RiskBand,
)
from volvence_zero.credit import CreditModule
from volvence_zero.integration import (
    FinalRolloutConfig,
    resolve_final_rollout_config,
)
from volvence_zero.integration.final_wiring import build_final_runtime_modules
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionError,
    PredictionErrorSnapshot,
    ForwardRepresentationBatchSnapshot,
    ForwardRepresentationSettlement,
    bind_steering_terminal_prediction_error_decisions,
    settle_steering_terminal_prediction_error,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.steering_contracts import (
    STEERING_ARTIFACT_BUNDLE_SCHEMA_VERSION,
    STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION,
    STEERING_GATE_ARTIFACT_SCHEMA_VERSION,
    STEERING_READER_ARTIFACT_SCHEMA_VERSION,
    SteeringArtifactBundle,
    SteeringExecutorArtifact,
    SteeringGateAction,
    SteeringGateArtifact,
    SteeringGateDecision,
    SteeringReaderArtifact,
)
from volvence_zero.steering_executor import SteeringExecutorModule
from volvence_zero.steering_gate import SteeringGateModule
from volvence_zero.steering_sensor import SteeringSensorModule
from volvence_zero.substrate import (
    HashingWhitespaceTokenizer,
    GenerationResult,
    PlaceholderSubstrateAdapter,
    ResidualActivation,
    ResidualControlApplication,
    SubstrateSnapshot,
    SubstrateFingerprint,
    SubstrateForwardRepresentationLineage,
    SurfaceKind,
    TransformersOpenWeightResidualRuntime,
)


_DIGEST = "a" * 64
_PREREG = "b" * 64
_HEAD_FINGERPRINT = "c" * 64
_TARGET_FINGERPRINT = "d" * 64


def _bundle() -> SteeringArtifactBundle:
    reader = SteeringReaderArtifact(
        schema_version=STEERING_READER_ARTIFACT_SCHEMA_VERSION,
        artifact_id="reader-v1",
        model_id="tiny-steering-model",
        model_weights_sha256=_DIGEST,
        source_preregistration_sha256=_PREREG,
        layer_index=0,
        residual_width=2,
        class_labels=("relationship", "task"),
        weights=((1.0, -1.0), (0.0, 0.0)),
        feature_mean=(0.0, 0.0),
        feature_scale=(1.0, 1.0),
        ridge_lambda=0.1,
        description="Frozen test reader.",
    )
    executor = SteeringExecutorArtifact(
        schema_version=STEERING_EXECUTOR_ARTIFACT_SCHEMA_VERSION,
        artifact_id="executor-v1",
        model_id=reader.model_id,
        model_weights_sha256=_DIGEST,
        source_preregistration_sha256=_PREREG,
        reader_artifact_id=reader.artifact_id,
        layer_index=0,
        residual_width=2,
        rank=2,
        class_labels=reader.class_labels,
        u_factors=((1.0, 0.0), (0.0, 1.0)),
        v_factors=((1.0, 0.0), (0.0, 1.0)),
        condition_codes=((1.0, 0.0), (0.0, 1.0)),
        control_norm_cap_ratio=0.25,
        free_bias_present=False,
        zero_code_strict_noop=True,
        description="Frozen test operator.",
    )
    gate = SteeringGateArtifact(
        schema_version=STEERING_GATE_ARTIFACT_SCHEMA_VERSION,
        artifact_id="gate-v1",
        source_preregistration_sha256=_PREREG,
        feature_names=(
            "belief_disagrees_fresh",
            "prediction_error_magnitude",
        ),
        weights=((0.0, 4.0), (0.0, 1.0)),
        bias=(0.5, -0.5),
        policy_version=1,
        description="Frozen test gate.",
    )
    return SteeringArtifactBundle(
        schema_version=STEERING_ARTIFACT_BUNDLE_SCHEMA_VERSION,
        bundle_id="steering-bundle-v1",
        reader=reader,
        executor=executor,
        gate=gate,
        description="Model-bound steering test bundle.",
    )


def _bundle_with_sensor_off() -> SteeringArtifactBundle:
    bundle = _bundle()
    unconditional = replace(
        bundle.executor,
        artifact_id="executor-unconditional-v1",
        condition_codes=((0.5, -0.5), (0.5, -0.5)),
        description="Matched-budget unconditional test operator.",
    )
    return replace(bundle, sensor_off_executor=unconditional)


def _substrate(first: float = 2.0, second: float = 1.0) -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="tiny-steering-model",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(0.7, 0.3),
        feature_surface=(),
        residual_activations=(
            ResidualActivation(
                layer_index=0,
                activation=(first, second),
                step=0,
            ),
        ),
        residual_sequence=(),
        unavailable_fields=(),
        description="Tiny frozen residual snapshot.",
    )


def _prediction_error(magnitude: float = 0.4) -> PredictionErrorSnapshot:
    predicted = PredictedOutcome(
        source_turn_index=1,
        target_turn_index=2,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.0,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.0,
        confidence=0.8,
        description="Frozen prediction.",
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=predicted,
        actual_outcome=ActualOutcome(
            observed_turn_index=2,
            task_progress=0.6,
            relationship_delta=0.1,
            regime_stability=0.5,
            action_payoff=0.2,
            description="Observed outcome.",
        ),
        next_prediction=replace(
            predicted,
            source_turn_index=2,
            target_turn_index=3,
        ),
        error=PredictionError(
            task_error=-0.1,
            relationship_error=-0.1,
            regime_error=0.0,
            action_error=-0.2,
            magnitude=magnitude,
            signed_reward=0.2,
            description="PE-owned mismatch.",
        ),
        turn_index=2,
        bootstrap=False,
        description="Prediction-error snapshot.",
    )


def _forward_snapshot(
    *,
    batch_id: str,
    mse: float,
    cosine: float,
    predicted: tuple[float, float],
) -> ForwardRepresentationBatchSnapshot:
    lineage = SubstrateForwardRepresentationLineage(
        schema_version="substrate-forward-representation.v1",
        snapshot_fingerprint=_TARGET_FINGERPRINT,
        model_fingerprint=SubstrateFingerprint(
            model_id="tiny-steering-model",
            version="test-v1",
            weights_sha256=_DIGEST,
        ),
        runtime_origin="hf-local",
        readout_kind="latest-token-selected-layer-residual-l2.v1",
        layer_indices=(0,),
        activation_widths=(2,),
        representation_dim=2,
    )
    return ForwardRepresentationBatchSnapshot(
        batch_id=batch_id,
        n_z=2,
        sample_count=1,
        update_applied=False,
        mean_squared_error=mse,
        mean_cosine_similarity=cosine,
        persistence_mean_squared_error=0.5,
        persistence_mean_cosine_similarity=0.0,
        mse_improvement_over_persistence=0.5 - mse,
        elapsed_ms=1.0,
        parameter_fingerprint=_HEAD_FINGERPRINT,
        target_lineage=lineage,
        settlements=(
            ForwardRepresentationSettlement(
                sample_id="episode-1:terminal",
                history_turns=4,
                predicted_representation=predicted,
                actual_representation=(1.0, 0.0),
                signed_error=(1.0 - predicted[0], -predicted[1]),
                mean_squared_error=mse,
                cosine_similarity=cosine,
                persistence_mean_squared_error=0.5,
                persistence_cosine_similarity=0.0,
            ),
        ),
        description="Frozen heldout terminal forward.",
    )


def _gate_decision(action: SteeringGateAction) -> SteeringGateDecision:
    return SteeringGateDecision(
        decision_id=f"test-{action.value}",
        action=action,
        steer_probability=0.8 if action is SteeringGateAction.STEER else 0.2,
        observations=(("belief_margin", 0.5),),
        policy_artifact_id="gate-v1",
        policy_version=1,
        terminal_credit_pending=True,
        decision_mode="test",
        description="Test gate decision.",
    )


class _TraceDirectRuntime:
    model_id = "tiny-steering-model"

    def __init__(self) -> None:
        self.calls: list[tuple[float, ...]] = []

    def apply_direct_residual_delta(self, **kwargs: object) -> ResidualControlApplication:
        delta = kwargs["residual_delta"]
        assert isinstance(delta, tuple)
        self.calls.append(delta)
        substrate = kwargs["substrate_snapshot"]
        assert isinstance(substrate, SubstrateSnapshot)
        return ResidualControlApplication(
            applied_snapshot=substrate,
            downstream_effect=(0.1, 0.0, -0.1),
            control_energy=0.1,
            backend_name="trace-direct-runtime",
            description="Trace-only direct steering preview.",
        )


class _ActiveGenerateRuntime:
    model_id = "tiny-steering-model"

    def __init__(self) -> None:
        self.interventions: list[object] = []

    def generate(self, **kwargs: object) -> GenerationResult:
        intervention = kwargs.get("steering_intervention")
        self.interventions.append(intervention)
        assert intervention is not None
        return GenerationResult(
            text="steered answer",
            token_count=2,
            capture=None,
            description="Active steering test generation.",
            steering_intervention_applied=True,
            steering_action=intervention.action.value,
            steering_executor_artifact_id=(
                intervention.executor_artifact_id
            ),
            steering_gate_policy_version=intervention.gate_policy_version,
        )


def _response_context(**kwargs: object) -> ResponseContext:
    return ResponseContext(
        regime_id="steady",
        regime_name="Steady",
        regime_switched=False,
        abstract_action=None,
        alert_count=0,
        temporal_switch_gate=0.0,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="none",
        user_input="hello",
        **kwargs,
    )


def _response_assembly() -> ResponseAssemblySnapshot:
    return ResponseAssemblySnapshot(
        regime_id="steady",
        regime_name="Steady",
        abstract_action=None,
        response_mode=ResponseMode.SUPPORT,
        answer_depth_limit="high-level-only",
        citation_mode="none",
        clarification_required=False,
        refer_out_required=False,
        ordering_plan=(),
        knowledge_briefs=(),
        case_briefs=(),
        playbook_ordering=(),
        required_disclaimers=(),
        required_disclaimer_phrases=(),
        control_code=(),
        control_scale=0.0,
        max_questions=0,
        prompt_residue_summary="",
        prompt_residue_ratio=0.0,
        knowledge_hit_count=0,
        case_hit_count=0,
        playbook_rule_count=0,
        risk_band=RiskBand.LOW,
        description="Steering response test assembly.",
    )


def test_sensor_gate_executor_chain_preserves_lag_and_norm_cap() -> None:
    bundle = _bundle()
    sensor = SteeringSensorModule(artifact=bundle.reader)
    first = asyncio.run(sensor.process_standalone(substrate=_substrate())).value
    second = asyncio.run(
        sensor.process_standalone(substrate=_substrate(first=-2.0))
    ).value

    assert first.belief_label == first.fresh_belief_label == "relationship"
    assert second.belief_label == "relationship"
    assert second.fresh_belief_label == "task"
    assert second.belief_disagrees_fresh is True

    gate = SteeringGateModule(artifact=bundle.gate)
    decision = asyncio.run(
        gate.process_standalone(
            belief=second,
            prediction_error=_prediction_error(),
        )
    ).value
    assert decision.action is SteeringGateAction.STEER

    executor = SteeringExecutorModule(
        artifact=bundle.executor,
        apply_shadow_hook=False,
    )
    intervention = asyncio.run(
        executor.process_standalone(
            substrate=_substrate(first=3.0, second=4.0),
            belief=first,
            gate=decision,
        )
    ).value
    assert intervention.action is SteeringGateAction.STEER
    assert intervention.control_norm <= intervention.control_norm_cap
    assert intervention.control_norm_cap == pytest.approx(1.25)
    assert math.isclose(
        math.sqrt(sum(value * value for value in intervention.residual_delta)),
        intervention.control_norm,
    )
    assert intervention.shadow_hook_executed is False


def test_model_bound_artifact_bundle_round_trips_canonical_json() -> None:
    bundle = _bundle()
    payload = bundle.to_json()

    assert SteeringArtifactBundle.from_json(payload) == bundle
    assert payload == bundle.to_json()


def test_sensor_off_artifact_bundle_round_trips_and_runs_matched_preview() -> None:
    bundle = _bundle_with_sensor_off()
    assert SteeringArtifactBundle.from_json(bundle.to_json()) == bundle
    runtime = _TraceDirectRuntime()
    belief = asyncio.run(
        SteeringSensorModule(artifact=bundle.reader).process_standalone(
            substrate=_substrate()
        )
    ).value

    intervention = asyncio.run(
        SteeringExecutorModule(
            artifact=bundle.executor,
            sensor_off_artifact=bundle.sensor_off_executor,
            runtime=runtime,
            source_text="hello",
            apply_shadow_hook=True,
        ).process_standalone(
            substrate=_substrate(),
            belief=belief,
            gate=_gate_decision(SteeringGateAction.STEER),
        )
    ).value

    assert len(runtime.calls) == 2
    assert intervention.noop_context is not None
    assert intervention.action_context is not None
    assert intervention.sensor_off_action_context is not None
    assert intervention.noop_context.conditioned is False
    assert intervention.action_context.conditioned is True
    assert intervention.sensor_off_action_context.conditioned is True
    assert (
        intervention.sensor_off_executor_artifact_id
        == bundle.sensor_off_executor.artifact_id
    )
    assert intervention.sensor_off_control_norm <= intervention.control_norm_cap


def test_shadow_hook_off_registers_no_direct_preview_and_noop_is_exact() -> None:
    bundle = _bundle()
    runtime = _TraceDirectRuntime()
    sensor = SteeringSensorModule(artifact=bundle.reader)
    belief = asyncio.run(sensor.process_standalone(substrate=_substrate())).value
    executor = SteeringExecutorModule(
        artifact=bundle.executor,
        runtime=runtime,
        source_text="hello",
        apply_shadow_hook=False,
    )

    intervention = asyncio.run(
        executor.process_standalone(
            substrate=_substrate(),
            belief=belief,
            gate=_gate_decision(SteeringGateAction.NOOP),
        )
    ).value

    assert runtime.calls == []
    assert intervention.residual_delta == (0.0, 0.0)
    assert intervention.zero_code_noop is True
    assert intervention.application_mode == "shadow-noop"


def test_shadow_hook_on_uses_transformers_only_preview_seam() -> None:
    bundle = _bundle()
    runtime = _TraceDirectRuntime()
    belief = asyncio.run(
        SteeringSensorModule(artifact=bundle.reader).process_standalone(
            substrate=_substrate()
        )
    ).value
    executor = SteeringExecutorModule(
        artifact=bundle.executor,
        runtime=runtime,
        source_text="hello",
        apply_shadow_hook=True,
    )

    intervention = asyncio.run(
        executor.process_standalone(
            substrate=_substrate(),
            belief=belief,
            gate=_gate_decision(SteeringGateAction.STEER),
        )
    ).value

    assert len(runtime.calls) == 1
    assert intervention.shadow_hook_executed is True
    assert intervention.application_mode == "shadow-preview"


def test_shadow_intervention_cannot_enter_response_context() -> None:
    bundle = _bundle()
    belief = asyncio.run(
        SteeringSensorModule(artifact=bundle.reader).process_standalone(
            substrate=_substrate()
        )
    ).value
    shadow = asyncio.run(
        SteeringExecutorModule(
            artifact=bundle.executor,
            apply_shadow_hook=False,
        ).process_standalone(
            substrate=_substrate(),
            belief=belief,
            gate=_gate_decision(SteeringGateAction.STEER),
        )
    ).value

    with pytest.raises(ValueError, match="SHADOW previews"):
        _response_context(steering_intervention=shadow)


def test_active_intervention_is_forwarded_and_attested_in_response() -> None:
    bundle = _bundle()
    belief = asyncio.run(
        SteeringSensorModule(artifact=bundle.reader).process_standalone(
            substrate=_substrate()
        )
    ).value
    active = asyncio.run(
        SteeringExecutorModule(
            artifact=bundle.executor,
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone(
            substrate=_substrate(),
            belief=belief,
            gate=_gate_decision(SteeringGateAction.STEER),
        )
    ).value
    runtime = _ActiveGenerateRuntime()

    response = LLMResponseSynthesizer(runtime=runtime).synthesize(
        context=_response_context(steering_intervention=active),
        assembly=_response_assembly(),
    )

    assert runtime.interventions == [active]
    assert any(
        tag.startswith("steering=steer:executor-v1:policy-1:applied-true")
        for tag in response.rationale_tags
    )


def test_c1_routes_matched_n_plus_one_pe_through_credit_to_gate() -> None:
    bundle = _bundle()
    sensor = SteeringSensorModule(artifact=bundle.reader)
    belief = asyncio.run(
        sensor.process_standalone(substrate=_substrate())
    ).value
    gate = SteeringGateModule(artifact=bundle.gate, learning_rate=0.1)
    first = asyncio.run(
        gate.process_standalone(
            belief=belief,
            prediction_error=_prediction_error(),
        )
    ).value
    second = asyncio.run(
        gate.process_standalone(
            belief=belief,
            prediction_error=_prediction_error(),
        )
    ).value
    terminal_pe = settle_steering_terminal_prediction_error(
        episode_id="episode-1",
        decision_ids=(first.decision_id, second.decision_id),
        action_snapshot=_forward_snapshot(
            batch_id="episode-1:steered",
            mse=0.1,
            cosine=0.9,
            predicted=(0.9, 0.1),
        ),
        noop_snapshot=_forward_snapshot(
            batch_id="episode-1:noop",
            mse=0.4,
            cosine=0.2,
            predicted=(0.2, 0.8),
        ),
    )
    assert terminal_pe.relative_mse_improvement == pytest.approx(0.75)
    assert terminal_pe.cosine_error_improvement == pytest.approx(0.7)

    credit_owner = CreditModule()
    credit_snapshot = credit_owner.settle_steering_terminal_prediction_errors(
        (terminal_pe,),
        timestamp_ms=10,
    )
    terminal_records = tuple(
        record
        for record in credit_snapshot.recent_credits
        if record.level == "steering_terminal_prediction_error"
    )
    assert len(terminal_records) == 2
    assert all(record.credit_value == pytest.approx(0.75) for record in terminal_records)
    assert all(record.source_event.startswith("steering-terminal:") for record in terminal_records)

    report = gate.settle_terminal_credit(credit_snapshot)
    assert report.update_applied is True
    assert report.old_policy_version == 1
    assert report.new_policy_version == 2
    assert gate.artifact.policy_version == 2
    assert gate.settle_terminal_credit(credit_snapshot).update_applied is False
    with pytest.raises(ValueError, match="already credited"):
        credit_owner.settle_steering_terminal_prediction_errors(
            (terminal_pe,),
            timestamp_ms=11,
        )


def test_noop_decision_receives_opposite_directional_counterfactual_credit() -> None:
    artifact = replace(
        _bundle().gate,
        feature_names=("belief_margin",),
        weights=((2.0, -2.0),),
        bias=(1.0, -1.0),
    )
    gate = SteeringGateModule(artifact=artifact, learning_rate=0.1)
    decision = gate.replay_observations((("belief_margin", 1.0),)).value
    assert decision.action is SteeringGateAction.NOOP
    terminal_pe = settle_steering_terminal_prediction_error(
        episode_id="counterfactual-source",
        decision_ids=("source-decision",),
        action_snapshot=_forward_snapshot(
            batch_id="counterfactual-steer",
            mse=0.1,
            cosine=0.9,
            predicted=(0.9, 0.1),
        ),
        noop_snapshot=_forward_snapshot(
            batch_id="counterfactual-noop",
            mse=0.4,
            cosine=0.2,
            predicted=(0.2, 0.8),
        ),
    )
    rebound = bind_steering_terminal_prediction_error_decisions(
        terminal_pe,
        episode_id="counterfactual-replay",
        decision_ids=(decision.decision_id,),
    )
    credit = CreditModule().settle_steering_terminal_prediction_errors(
        (rebound,), timestamp_ms=1
    )

    report = gate.settle_terminal_credit(credit)

    assert report.mean_terminal_credit == pytest.approx(0.75)
    assert report.mean_directional_terminal_credit == pytest.approx(-0.75)
    assert report.update_applied is True
    assert gate.artifact.weights != artifact.weights


def test_stochastic_gate_checkpoint_restores_exact_continuation() -> None:
    artifact = replace(
        _bundle().gate,
        feature_names=("belief_margin",),
        weights=((0.0, 0.0),),
        bias=(0.0, 0.0),
    )
    original = SteeringGateModule(
        artifact=artifact,
        learning_enabled=False,
        decision_mode="evidence-stochastic",
        exploration_seed=71,
    )
    for value in (0.1, 0.2, 0.3):
        original.replay_observations((("belief_margin", value),))
    checkpoint = original.export_checkpoint(checkpoint_id="exact-continuation")
    decoded = type(checkpoint).from_json(checkpoint.to_json())
    restored = SteeringGateModule(
        artifact=artifact,
        learning_enabled=False,
        decision_mode="evidence-stochastic",
        exploration_seed=71,
    )
    restored.restore_checkpoint(decoded)

    expected = tuple(
        original.replay_observations((("belief_margin", value),)).value
        for value in (0.4, 0.5, 0.6, 0.7)
    )
    actual = tuple(
        restored.replay_observations((("belief_margin", value),)).value
        for value in (0.4, 0.5, 0.6, 0.7)
    )

    assert actual == expected


def test_c1_rejects_unmatched_noop_target_or_head() -> None:
    action = _forward_snapshot(
        batch_id="action",
        mse=0.1,
        cosine=0.9,
        predicted=(0.9, 0.1),
    )
    noop = replace(
        _forward_snapshot(
            batch_id="noop",
            mse=0.4,
            cosine=0.2,
            predicted=(0.2, 0.8),
        ),
        parameter_fingerprint="e" * 64,
    )

    with pytest.raises(ValueError, match="parameter_fingerprint drift"):
        settle_steering_terminal_prediction_error(
            episode_id="episode-1",
            decision_ids=("decision-1",),
            action_snapshot=action,
            noop_snapshot=noop,
        )


def test_ordered_active_promotion_guards_fail_loudly() -> None:
    with pytest.raises(ValueError, match="requires ACTIVE steering_sensor"):
        FinalRolloutConfig(steering_executor=WiringLevel.ACTIVE)
    with pytest.raises(ValueError, match="explicit gate-off"):
        FinalRolloutConfig(
            steering_sensor=WiringLevel.ACTIVE,
            steering_executor=WiringLevel.ACTIVE,
        )
    with pytest.raises(ValueError, match="requires ACTIVE steering_executor"):
        FinalRolloutConfig(steering_gate=WiringLevel.ACTIVE)

    config = FinalRolloutConfig(
        steering_sensor=WiringLevel.ACTIVE,
        steering_executor=WiringLevel.ACTIVE,
        steering_ungated_action="noop",
    )
    assert config.steering_gate is WiringLevel.SHADOW


def test_final_wiring_builds_one_ordered_shadow_owner_chain() -> None:
    bundle = _bundle()
    modules = build_final_runtime_modules(
        config=FinalRolloutConfig(steering_shadow_hook=False),
        substrate_adapter=PlaceholderSubstrateAdapter(
            model_id=bundle.reader.model_id
        ),
        user_input="hello",
        steering_bundle=bundle,
    )
    slots = tuple(module.slot_name for module in modules)

    assert slots.count("steering_condition_belief") == 1
    assert slots.count("steering_gate_decision") == 1
    assert slots.count("steering_intervention") == 1
    assert slots.index("substrate") < slots.index("steering_condition_belief")
    assert slots.index("prediction_error") < slots.index(
        "steering_gate_decision"
    )
    assert slots.index("steering_gate_decision") < slots.index(
        "steering_intervention"
    )
    steering_modules = tuple(
        module
        for module in modules
        if module.slot_name.startswith("steering_")
    )
    assert all(
        module.wiring_level is WiringLevel.SHADOW
        for module in steering_modules
    )


def test_final_wiring_drops_sensor_off_control_for_active_executor() -> None:
    bundle = _bundle_with_sensor_off()

    modules = build_final_runtime_modules(
        config=FinalRolloutConfig(
            steering_sensor=WiringLevel.ACTIVE,
            steering_executor=WiringLevel.ACTIVE,
            steering_gate=WiringLevel.SHADOW,
            steering_shadow_hook=False,
            steering_ungated_action="always_on",
        ),
        substrate_adapter=PlaceholderSubstrateAdapter(
            model_id=bundle.reader.model_id
        ),
        user_input="hello",
        steering_bundle=bundle,
    )

    sensor = next(
        module
        for module in modules
        if module.slot_name == "steering_condition_belief"
    )
    gate = next(
        module
        for module in modules
        if module.slot_name == "steering_gate_decision"
    )
    executor = next(
        module
        for module in modules
        if module.slot_name == "steering_intervention"
    )
    assert sensor.wiring_level is WiringLevel.ACTIVE
    assert gate.wiring_level is WiringLevel.SHADOW
    assert executor.wiring_level is WiringLevel.ACTIVE


def test_steering_env_overrides_are_owner_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VZ_STEERING_SENSOR", "active")
    monkeypatch.setenv("VZ_STEERING_EXECUTOR", "active")
    monkeypatch.setenv("VZ_STEERING_UNGATED_ACTION", "noop")
    monkeypatch.setenv("VZ_STEERING_SHADOW_HOOK", "off")

    config = resolve_final_rollout_config(FinalRolloutConfig())

    assert config.steering_sensor is WiringLevel.ACTIVE
    assert config.steering_executor is WiringLevel.ACTIVE
    assert config.steering_gate is WiringLevel.SHADOW
    assert config.steering_ungated_action == "noop"
    assert config.steering_shadow_hook is False


def test_legacy_torch_env_does_not_promote_steering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VZ_TORCH_BACKENDS", "active")

    config = resolve_final_rollout_config(FinalRolloutConfig())

    assert config.temporal_ssl_backend is WiringLevel.ACTIVE
    assert config.steering_sensor is WiringLevel.SHADOW
    assert config.steering_executor is WiringLevel.SHADOW
    assert config.steering_gate is WiringLevel.SHADOW


def test_transformers_generation_applies_active_intervention_without_capture() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_ctx=32,
            n_embd=2,
            n_layer=1,
            n_head=1,
        )
    )
    runtime = TransformersOpenWeightResidualRuntime(
        model_id="tiny-steering-model",
        model=model,
        tokenizer=HashingWhitespaceTokenizer(vocab_size=64),
        device="cpu",
        layer_indices=(0,),
        activation_width=2,
        runtime_origin="hf-local",
        loaded_base_model_weights_sha256=_DIGEST,
    )
    bundle = _bundle()
    belief = asyncio.run(
        SteeringSensorModule(artifact=bundle.reader).process_standalone(
            substrate=_substrate()
        )
    ).value
    active_executor = SteeringExecutorModule(
        artifact=bundle.executor,
        wiring_level=WiringLevel.ACTIVE,
    )
    intervention = asyncio.run(
        active_executor.process_standalone(
            substrate=_substrate(),
            belief=belief,
            gate=_gate_decision(SteeringGateAction.STEER),
        )
    ).value

    result = runtime.generate(
        prompt="hello",
        max_new_tokens=1,
        temperature=0.0,
        capture_residuals=False,
        steering_intervention=intervention,
    )

    assert result.capture is None
    assert result.steering_intervention_applied is True
    assert result.steering_action == SteeringGateAction.STEER.value
    assert result.steering_executor_artifact_id == bundle.executor.artifact_id
    assert result.steering_gate_policy_version == bundle.gate.policy_version
    assert torch.isfinite(model.transformer.h[0].mlp.c_fc.weight).all()
