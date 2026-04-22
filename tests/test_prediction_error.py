from __future__ import annotations

import asyncio
from unittest.mock import patch

from volvence_zero.agent.response import LLMResponseSynthesizer, ResponseContext
from volvence_zero.agent.prompts import build_system_prompt
from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.credit import (
    CreditModule,
    derive_credit_records_from_prediction_error_first,
    derive_prediction_error_credit_records,
)
from volvence_zero.dual_track import DualTrackSnapshot, TrackState
from volvence_zero.evaluation import EvaluationBackbone, EvaluationScore, EvaluationSnapshot
from volvence_zero.internal_rl import InternalRLEnvironment
from volvence_zero.memory import Track
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionErrorModule,
    PredictionErrorSnapshot,
    derive_actual_outcome_from_substrate,
)
from volvence_zero.regime import RegimeIdentity, RegimeSnapshot
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    build_builtin_transformers_runtime,
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
        self.last_chat_messages: tuple[tuple[str, str], ...] = ()

    def generate(
        self,
        *,
        prompt: str,
        system_context: str = "",
        chat_messages: tuple[tuple[str, str], ...] = (),
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        control_parameters: tuple[float, ...] = (),
        control_scale: float = 0.0,
    ) -> GenerationResult:
        del max_new_tokens, temperature
        self.last_control_scale = control_scale
        self.last_control_parameters = control_parameters
        self.last_chat_messages = chat_messages
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
    assert runtime.last_chat_messages[0][0] == "system"
    assert runtime.last_chat_messages[-1] == ("user", "Help me think through this")


def test_system_prompt_explicitly_forbids_dialogue_continuation():
    prompt = build_system_prompt(
        context=ResponseContext(
            regime_id="casual_social",
            regime_name="casual social",
            regime_switched=False,
            abstract_action=None,
            alert_count=0,
            retrieved_memory_count=0,
            temporal_switch_gate=0.0,
            temporal_is_switching=False,
            reflection_lesson_count=0,
            reflection_tension_count=0,
            reflection_writeback_applied=False,
            primary_reflection_lesson=None,
            primary_reflection_tension=None,
            joint_schedule_action="ssl-only",
            user_input="hi",
        )
    )

    assert "Reply as the assistant to the latest user message only." in prompt
    assert "Do not continue the conversation on behalf of the user." in prompt


def test_system_prompt_includes_knowledge_and_boundary_context():
    prompt = build_system_prompt(
        context=ResponseContext(
            regime_id="problem_solving",
            regime_name="problem solving",
            regime_switched=False,
            abstract_action="structured_planning",
            alert_count=0,
            retrieved_memory_count=0,
            temporal_switch_gate=0.4,
            temporal_is_switching=False,
            reflection_lesson_count=0,
            reflection_tension_count=0,
            reflection_writeback_applied=False,
            primary_reflection_lesson=None,
            primary_reflection_tension=None,
            joint_schedule_action="ssl-only",
            user_input="What should I do next?",
            knowledge_hit_count=2,
            knowledge_summaries=("Keep guidance high-level and jurisdiction-aware.",),
            citation_required=True,
            boundary_risk_band="medium",
            boundary_answer_depth_limit="high-level-only",
            boundary_clarification_required=True,
            boundary_refer_out_required=False,
            boundary_required_disclaimers=("jurisdiction-variance",),
        )
    )

    assert "Relevant domain guidance" in prompt
    assert "sourceable information" in prompt
    assert "missing local or factual detail" in prompt
    assert "jurisdiction-variance" in prompt


def test_system_prompt_includes_case_patterns_when_available():
    prompt = build_system_prompt(
        context=ResponseContext(
            regime_id="guided_exploration",
            regime_name="guided exploration",
            regime_switched=False,
            abstract_action="stabilize_then_structure",
            alert_count=0,
            retrieved_memory_count=0,
            temporal_switch_gate=0.3,
            temporal_is_switching=False,
            reflection_lesson_count=0,
            reflection_tension_count=0,
            reflection_writeback_applied=False,
            primary_reflection_lesson=None,
            primary_reflection_tension=None,
            joint_schedule_action="ssl-only",
            user_input="What should I do first?",
            case_hit_count=2,
            case_patterns=("family-transition-high-emotion", "needs-structure"),
        )
    )

    assert "Relevant prior case patterns" in prompt
    assert "family-transition-high-emotion" in prompt


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


def test_transformers_runtime_generate_passes_attention_mask():
    runtime = build_builtin_transformers_runtime()
    captured_kwargs: dict[str, object] = {}
    torch = runtime._torch

    def fake_generate(**kwargs):
        captured_kwargs.update(kwargs)
        input_ids = kwargs["input_ids"]
        appended = torch.full((1, 1), 1, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat((input_ids, appended), dim=1)

    with patch.object(runtime._model, "generate", side_effect=fake_generate):
        runtime.generate(
            prompt="hello",
            system_context="be helpful",
            chat_messages=(("system", "be helpful"), ("user", "hello")),
            max_new_tokens=8,
        )

    assert "attention_mask" in captured_kwargs


def test_transformers_runtime_uses_chat_template_when_available():
    runtime = build_builtin_transformers_runtime()
    torch = runtime._torch

    def fake_apply_chat_template(messages, *, tokenize, add_generation_prompt, return_tensors=None, return_dict=None):
        assert add_generation_prompt is True
        assert messages == [
            {"role": "system", "content": "System guidance"},
            {"role": "user", "content": "Hello there"},
        ]
        if tokenize is True:
            assert return_tensors == "pt"
            assert return_dict is True
            return {
                "input_ids": torch.tensor([[7, 8, 9]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
        return "<chat-template-rendered>"

    runtime._tokenizer.apply_chat_template = fake_apply_chat_template

    source_text, model_inputs = runtime._build_generation_inputs(
        prompt="Hello there",
        system_context="System guidance",
        chat_messages=(("system", "System guidance"), ("user", "Hello there")),
    )

    assert "system: System guidance" in source_text
    assert "user: Hello there" in source_text
    assert "attention_mask" in model_inputs


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


def test_prediction_error_weighting_respects_prediction_confidence():
    module = PredictionErrorModule(wiring_level=WiringLevel.ACTIVE)
    high_conf_prediction = PredictedOutcome(
        source_turn_index=1,
        target_turn_index=2,
        predicted_task_progress=0.95,
        predicted_relationship_delta=0.92,
        predicted_regime_stability=0.90,
        predicted_action_payoff=0.94,
        confidence=0.95,
        description="high confidence",
    )
    low_conf_prediction = PredictedOutcome(
        source_turn_index=1,
        target_turn_index=2,
        predicted_task_progress=0.55,
        predicted_relationship_delta=0.54,
        predicted_regime_stability=0.53,
        predicted_action_payoff=0.56,
        confidence=0.10,
        description="low confidence",
    )
    actual = ActualOutcome(
        observed_turn_index=2,
        task_progress=0.35,
        relationship_delta=0.34,
        regime_stability=0.33,
        action_payoff=0.36,
        description="actual",
    )
    high_conf_error = module.compute_prediction_error(
        predicted=high_conf_prediction,
        actual_outcome=actual,
    )
    low_conf_error = module.compute_prediction_error(
        predicted=low_conf_prediction,
        actual_outcome=actual,
    )
    assert high_conf_error.magnitude > low_conf_error.magnitude
    assert "weighted_axes[" in high_conf_error.description


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


def test_prediction_error_evaluation_scores_remain_readouts():
    module = PredictionErrorModule(wiring_level=WiringLevel.ACTIVE)
    backbone = EvaluationBackbone()
    first = asyncio.run(
        module.process_standalone(
            turn_index=1,
            substrate_snapshot=_substrate_snapshot(
                task_pull=0.2,
                support_pull=0.2,
                repair_pull=0.2,
                exploration_pull=0.3,
                directive_pull=0.3,
                token="prev",
            ),
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
            substrate_snapshot=_substrate_snapshot(
                task_pull=0.8,
                support_pull=0.7,
                repair_pull=0.6,
                exploration_pull=0.5,
                directive_pull=0.8,
                token="curr",
            ),
            previous_substrate_snapshot=_substrate_snapshot(
                task_pull=0.2,
                support_pull=0.2,
                repair_pull=0.2,
                exploration_pull=0.3,
                directive_pull=0.3,
                token="prev",
            ),
            evaluation_snapshot=_evaluation_snapshot(task=0.4, relationship=0.8, abstraction=0.7),
            dual_track_snapshot=_dual_track_snapshot(tension=0.1),
            regime_snapshot=_regime_snapshot(effectiveness=0.9),
        )
    )
    base = _evaluation_snapshot()
    enriched = backbone.record_prediction_error_evidence(
        session_id="s",
        wave_id="w",
        timestamp_ms=2,
        base_snapshot=base,
        prediction_error_snapshot=second.value,
    )
    score_map = {score.metric_name: score for score in enriched.turn_scores}
    assert "prediction_error_magnitude" in score_map
    assert "prediction_error_reward" in score_map
    assert "predictive_accuracy" in score_map
    assert "PE-owner" in score_map["prediction_error_magnitude"].evidence
    assert "prediction_confidence" in score_map["predictive_accuracy"].evidence


def test_pe_readout_only_reward_keeps_readout_without_primary_dominance():
    env = InternalRLEnvironment(
        evaluation_family_signals={
            "prediction_error_reward_readout": -0.4,
            "task": 0.7,
            "relationship": 0.2,
            "learning": 0.6,
            "abstraction": 0.55,
            "safety": 0.9,
        },
        primary_prediction_error_enabled=False,
    )
    step = _temporal_snapshot(0.8)
    components = env._reward_components(
        track=Track.WORLD,
        temporal_step=type("T", (), {"controller_state": step.controller_state})(),
        downstream_effect=(0.2, 0.1, 0.3),
        control_energy=0.2,
        policy_replacement_quality=0.6,
    )
    comp_map = dict(components)
    assert "primary_prediction_error" not in comp_map
    assert "prediction_error_readout" in comp_map
    assert comp_map["prediction_error_readout"] < 0.0


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
