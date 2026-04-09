from __future__ import annotations

import asyncio

from volvence_zero.agent.response import LLMResponseSynthesizer, ResponseContext
from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.credit import (
    CreditModule,
    derive_credit_records_from_prediction_error_first,
    derive_prediction_error_credit_records,
)
from volvence_zero.dual_track import DualTrackSnapshot, TrackState
from volvence_zero.evaluation import EvaluationScore, EvaluationSnapshot
from volvence_zero.internal_rl import InternalRLEnvironment
from volvence_zero.memory import Track
from volvence_zero.prediction import PredictionErrorModule, PredictionErrorSnapshot, derive_actual_outcome_from_substrate
from volvence_zero.regime import RegimeIdentity, RegimeSnapshot
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    FeatureSignal,
    GenerationResult,
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
    SyntheticOpenWeightResidualRuntime,
)
from volvence_zero.temporal import ControllerState, TemporalAbstractionSnapshot


class CaptureRuntime(SyntheticOpenWeightResidualRuntime):
    def __init__(self) -> None:
        super().__init__(model_id="capture-runtime")
        self.last_control_scale = 0.0
        self.last_control_parameters: tuple[float, ...] = ()

    def generate(
        self,
        *,
        prompt: str,
        system_context: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        control_parameters: tuple[float, ...] = (),
        control_scale: float = 0.0,
    ) -> GenerationResult:
        del max_new_tokens, temperature
        self.last_control_scale = control_scale
        self.last_control_parameters = control_parameters
        capture = self.capture(source_text=f"{system_context} {prompt}".strip())
        return GenerationResult(
            text="generated",
            token_count=1,
            capture=capture,
            description="fake generation",
        )


def _evaluation_snapshot(task: float = 0.7, relationship: float = 0.6, abstraction: float = 0.55) -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(
            EvaluationScore("task", "task_pressure", task, 0.8, "task"),
            EvaluationScore("relationship", "relationship_continuity", relationship, 0.8, "rel"),
            EvaluationScore("learning", "joint_learning_progress", 0.6, 0.8, "learn"),
            EvaluationScore("abstraction", "abstract_action_usefulness", abstraction, 0.8, "abs"),
            EvaluationScore("safety", "contract_integrity", 0.9, 0.9, "safe"),
        ),
        session_scores=(),
        alerts=(),
        description="eval",
    )


def _dual_track_snapshot(tension: float = 0.2) -> DualTrackSnapshot:
    return DualTrackSnapshot(
        world_track=TrackState(track=Track.WORLD, active_goals=("task",), recent_credits=(), controller_code=(0.2, 0.3), tension_level=0.4),
        self_track=TrackState(track=Track.SELF, active_goals=("support",), recent_credits=(), controller_code=(0.3, 0.2), tension_level=0.5),
        cross_track_tension=tension,
        description="dual",
    )


def _regime_snapshot(effectiveness: float = 0.7) -> RegimeSnapshot:
    regime = RegimeIdentity(
        regime_id="problem_solving",
        name="problem solving",
        embedding=(0.1, 0.2, 0.3),
        entry_conditions="task",
        exit_conditions="done",
        historical_effectiveness=effectiveness,
    )
    return RegimeSnapshot(
        active_regime=regime,
        previous_regime=None,
        switch_reason="test",
        candidate_regimes=(("problem_solving", effectiveness),),
        turns_in_current_regime=1,
        description="regime",
        effectiveness_trend=(("problem_solving", effectiveness),),
    )


def _temporal_snapshot(switch_gate: float = 0.8) -> TemporalAbstractionSnapshot:
    return TemporalAbstractionSnapshot(
        controller_state=ControllerState(
            code=(0.4, 0.5, 0.6),
            code_dim=3,
            switch_gate=switch_gate,
            is_switching=switch_gate > 0.5,
            steps_since_switch=0,
        ),
        active_abstract_action="repair_controller",
        controller_params_hash="hash",
        description="temporal",
        action_family_version=1,
    )


def _substrate_snapshot(
    *,
    task_pull: float,
    support_pull: float,
    repair_pull: float,
    exploration_pull: float,
    directive_pull: float,
    token: str = "hello",
) -> SubstrateSnapshot:
    residual = (
        ResidualActivation(layer_index=0, activation=(task_pull, support_pull, directive_pull), step=0),
    )
    sequence = (
        ResidualSequenceStep(
            step=0,
            token=token,
            feature_surface=(
                FeatureSignal(name="semantic_task_pull", values=(task_pull,), source="test"),
                FeatureSignal(name="semantic_support_pull", values=(support_pull,), source="test"),
                FeatureSignal(name="semantic_repair_pull", values=(repair_pull,), source="test"),
                FeatureSignal(name="semantic_exploration_pull", values=(exploration_pull,), source="test"),
                FeatureSignal(name="semantic_directive_pull", values=(directive_pull,), source="test"),
            ),
            residual_activations=residual,
            description="step",
        ),
    )
    return SubstrateSnapshot(
        model_id="test-substrate",
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(0.5, 0.4),
        feature_surface=sequence[0].feature_surface,
        residual_activations=residual,
        residual_sequence=sequence,
        unavailable_fields=(),
        description="substrate",
    )


def test_prediction_error_module_bootstrap_then_error():
    module = PredictionErrorModule(wiring_level=WiringLevel.ACTIVE)
    first = asyncio.run(
        module.process_standalone(
            turn_index=1,
            evaluation_snapshot=_evaluation_snapshot(),
            dual_track_snapshot=_dual_track_snapshot(),
            regime_snapshot=_regime_snapshot(),
            temporal_snapshot=_temporal_snapshot(),
        )
    )
    assert isinstance(first.value, PredictionErrorSnapshot)
    assert first.value.turn_index == 1
    assert first.value.bootstrap is True
    assert first.value.evaluated_prediction is None
    prev_prediction = first.value.next_prediction
    second = asyncio.run(
        module.process_standalone(
            previous_prediction=prev_prediction,
            turn_index=2,
            evaluation_snapshot=_evaluation_snapshot(task=0.4, relationship=0.8, abstraction=0.7),
            dual_track_snapshot=_dual_track_snapshot(tension=0.1),
            regime_snapshot=_regime_snapshot(effectiveness=0.9),
            temporal_snapshot=_temporal_snapshot(switch_gate=0.9),
        )
    )
    assert second.value.turn_index == 2
    assert second.value.bootstrap is False
    assert second.value.evaluated_prediction == prev_prediction
    assert second.value.error.magnitude >= 0.0
    assert second.value.error.signed_reward != 0.0 or second.value.error.magnitude > 0.0


def test_prediction_error_credit_records():
    module = PredictionErrorModule(wiring_level=WiringLevel.ACTIVE)
    snap = asyncio.run(
        module.process_standalone(
            turn_index=1,
            evaluation_snapshot=_evaluation_snapshot(task=0.2),
            dual_track_snapshot=_dual_track_snapshot(),
            regime_snapshot=_regime_snapshot(),
            temporal_snapshot=_temporal_snapshot(),
        )
    )
    records = derive_prediction_error_credit_records(
        prediction_error=snap.value.error,
        timestamp_ms=1,
    )
    assert len(records) == 4
    assert {r.source_event for r in records} == {"pe:task", "pe:relationship", "pe:regime", "pe:action"}


def test_llm_response_synthesizer_passes_control_signal():
    runtime = CaptureRuntime()
    synth = LLMResponseSynthesizer(runtime=runtime)
    response = synth.synthesize(
        context=ResponseContext(
            regime_id="problem_solving",
            regime_name="problem solving",
            regime_switched=False,
            abstract_action="repair_controller",
            alert_count=0,
            retrieved_memory_count=1,
            temporal_switch_gate=0.8,
            temporal_is_switching=True,
            reflection_lesson_count=0,
            reflection_tension_count=0,
            reflection_writeback_applied=False,
            primary_reflection_lesson=None,
            primary_reflection_tension=None,
            joint_schedule_action="full-cycle",
            user_input="Help me think through this",
            retrieved_memories=("You prefer structured plans.",),
            controller_description="controller active",
            control_code=(0.4, 0.5, 0.6),
        )
    )
    assert response.text == "generated"
    assert runtime.last_control_parameters == (0.4, 0.5, 0.6)
    assert runtime.last_control_scale > 0.0


def test_agent_session_runner_exposes_prediction_error_from_second_turn():
    runner = AgentSessionRunner(session_id="pe-session")
    first = asyncio.run(runner.run_turn("First turn for bootstrap."))
    assert first.prediction_error is not None
    assert first.evaluated_prediction is None
    assert first.next_prediction is not None
    second = asyncio.run(runner.run_turn("Second turn should evaluate previous prediction."))
    assert second.prediction_error is not None
    assert second.evaluated_prediction is not None
    assert second.actual_outcome is not None
    assert second.next_prediction is not None
    assert second.evaluated_prediction.target_turn_index == second.actual_outcome.observed_turn_index
    assert second.active_snapshots.get("prediction_error") is not None


def test_generate_returns_capture_when_control_is_applied():
    runtime = CaptureRuntime()
    result = runtime.generate(
        prompt="Plan the next step.",
        system_context="You are calm and structured.",
        control_parameters=(0.4, 0.3, 0.2),
        control_scale=0.8,
    )
    assert result.token_count >= 0
    assert result.capture is not None
    assert result.capture.residual_sequence
    assert runtime.last_control_scale > 0.0


def test_actual_outcome_uses_substrate_level_deltas():
    previous = _substrate_snapshot(
        task_pull=0.2,
        support_pull=0.2,
        repair_pull=0.2,
        exploration_pull=0.3,
        directive_pull=0.3,
        token="prev",
    )
    current = _substrate_snapshot(
        task_pull=0.8,
        support_pull=0.7,
        repair_pull=0.6,
        exploration_pull=0.5,
        directive_pull=0.8,
        token="curr",
    )
    outcome = derive_actual_outcome_from_substrate(
        observed_turn_index=2,
        substrate_snapshot=current,
        previous_substrate_snapshot=previous,
    )
    assert outcome.task_progress > 0.5
    assert outcome.relationship_delta > 0.5
    assert "Substrate-derived outcome" in outcome.description


def test_prediction_error_module_process_is_stateful_owner():
    module = PredictionErrorModule(wiring_level=WiringLevel.ACTIVE)
    upstream_turn1 = {
        "substrate": type("S", (), {"value": _substrate_snapshot(task_pull=0.3, support_pull=0.3, repair_pull=0.2, exploration_pull=0.4, directive_pull=0.3), "timestamp_ms": 1})(),
        "evaluation": type("S", (), {"value": _evaluation_snapshot(), "timestamp_ms": 1})(),
        "dual_track": type("S", (), {"value": _dual_track_snapshot(), "timestamp_ms": 1})(),
        "regime": type("S", (), {"value": _regime_snapshot(), "timestamp_ms": 1})(),
    }
    first = asyncio.run(module.process(upstream_turn1))
    assert first.value.bootstrap is True
    upstream_turn2 = {
        "substrate": type("S", (), {"value": _substrate_snapshot(task_pull=0.8, support_pull=0.7, repair_pull=0.6, exploration_pull=0.4, directive_pull=0.8), "timestamp_ms": 2})(),
        "evaluation": type("S", (), {"value": _evaluation_snapshot(task=0.4, relationship=0.8, abstraction=0.7), "timestamp_ms": 2})(),
        "dual_track": type("S", (), {"value": _dual_track_snapshot(tension=0.1), "timestamp_ms": 2})(),
        "regime": type("S", (), {"value": _regime_snapshot(effectiveness=0.9), "timestamp_ms": 2})(),
    }
    second = asyncio.run(module.process(upstream_turn2))
    assert second.value.bootstrap is False
    assert second.value.evaluated_prediction is not None
    assert second.value.actual_outcome.observed_turn_index == 2


def test_pe_first_reward_becomes_primary_term():
    env = InternalRLEnvironment(evaluation_family_signals={"prediction_error_reward": 0.8, "task": 0.7, "relationship": 0.2, "learning": 0.6, "abstraction": 0.55, "safety": 0.9})
    step = _temporal_snapshot(0.8)
    components = env._reward_components(
        track=Track.WORLD,
        temporal_step=type("T", (), {"controller_state": step.controller_state})(),
        downstream_effect=(0.2, 0.1, 0.3),
        control_energy=0.2,
        policy_replacement_quality=0.6,
    )
    comp_map = dict(components)
    assert "primary_prediction_error" in comp_map
    assert abs(comp_map["primary_prediction_error"]) > abs(comp_map["task_outcome_delta"])


def test_credit_derivation_prefers_prediction_error_when_available():
    module = PredictionErrorModule(wiring_level=WiringLevel.ACTIVE)
    substrate_prev = _substrate_snapshot(
        task_pull=0.2, support_pull=0.2, repair_pull=0.2, exploration_pull=0.3, directive_pull=0.3, token="prev"
    )
    substrate_curr = _substrate_snapshot(
        task_pull=0.8, support_pull=0.7, repair_pull=0.6, exploration_pull=0.5, directive_pull=0.8, token="curr"
    )
    first = asyncio.run(
        module.process_standalone(
            turn_index=1,
            substrate_snapshot=substrate_prev,
            previous_substrate_snapshot=None,
            evaluation_snapshot=_evaluation_snapshot(),
            dual_track_snapshot=_dual_track_snapshot(),
            regime_snapshot=_regime_snapshot(),
        )
    )
    second = asyncio.run(
        module.process_standalone(
            turn_index=2,
            previous_prediction=first.value.next_prediction,
            substrate_snapshot=substrate_curr,
            previous_substrate_snapshot=substrate_prev,
            evaluation_snapshot=_evaluation_snapshot(task=0.4, relationship=0.8, abstraction=0.7),
            dual_track_snapshot=_dual_track_snapshot(tension=0.1),
            regime_snapshot=_regime_snapshot(effectiveness=0.9),
        )
    )
    credits = derive_credit_records_from_prediction_error_first(
        dual_track_snapshot=_dual_track_snapshot(tension=0.1),
        evaluation_snapshot=_evaluation_snapshot(task=0.4, relationship=0.8, abstraction=0.7),
        prediction_error_snapshot=second.value,
        timestamp_ms=2,
    )
    pe_records = [r for r in credits if r.level == "prediction_error"]
    readout_records = [r for r in credits if r.level == "evaluation_readout"]
    assert pe_records, "PE-first path must produce prediction_error-level records"
    assert readout_records, "Readout records should still exist as lightweight evidence"
    assert max(abs(r.credit_value) for r in pe_records) >= max(abs(r.credit_value) for r in readout_records)


def test_credit_module_process_uses_prediction_error_dependency():
    module = CreditModule(wiring_level=WiringLevel.ACTIVE)
    pe_module = PredictionErrorModule(wiring_level=WiringLevel.ACTIVE)
    substrate_prev = _substrate_snapshot(
        task_pull=0.2, support_pull=0.2, repair_pull=0.2, exploration_pull=0.3, directive_pull=0.3, token="prev"
    )
    substrate_curr = _substrate_snapshot(
        task_pull=0.8, support_pull=0.7, repair_pull=0.6, exploration_pull=0.5, directive_pull=0.8, token="curr"
    )
    first_pe = asyncio.run(
        pe_module.process_standalone(
            turn_index=1,
            substrate_snapshot=substrate_prev,
            previous_substrate_snapshot=None,
            evaluation_snapshot=_evaluation_snapshot(),
            dual_track_snapshot=_dual_track_snapshot(),
            regime_snapshot=_regime_snapshot(),
        )
    )
    second_pe = asyncio.run(
        pe_module.process_standalone(
            turn_index=2,
            previous_prediction=first_pe.value.next_prediction,
            substrate_snapshot=substrate_curr,
            previous_substrate_snapshot=substrate_prev,
            evaluation_snapshot=_evaluation_snapshot(task=0.4, relationship=0.8, abstraction=0.7),
            dual_track_snapshot=_dual_track_snapshot(tension=0.1),
            regime_snapshot=_regime_snapshot(effectiveness=0.9),
        )
    )
    credit_snapshot = asyncio.run(
        module.process_standalone(
            dual_track_snapshot=_dual_track_snapshot(tension=0.1),
            evaluation_snapshot=_evaluation_snapshot(task=0.4, relationship=0.8, abstraction=0.7),
            prediction_error_snapshot=second_pe.value,
            timestamp_ms=2,
        )
    ).value
    assert any(record.level == "prediction_error" for record in credit_snapshot.recent_credits)
