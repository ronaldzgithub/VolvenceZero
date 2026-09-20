from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

from volvence_zero.agent.response import (
    ExpressionExactBindingSourceError,
    LLMResponseSynthesizer,
    ResponseContext,
    StructuredExpressionOutputError,
)
from volvence_zero.application.runtime import (
    ResponseActionRealization,
    ResponseAssemblySnapshot,
    ResponseMode,
    RiskBand,
)
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
)
from volvence_zero.expression_output import (
    ExpressionBindingSource,
    ExpressionExactBinding,
    ExpressionOutputContract,
)
from volvence_zero.substrate import (
    GenerationResult,
    SubstrateFingerprint,
    SyntheticOpenWeightResidualRuntime,
)


class _RecordingRuntime:
    model_id = "test-runtime"

    def __init__(self, *, applies_personal_conditioning: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._applies_personal_conditioning = applies_personal_conditioning

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        # Mirror the GenerationResult contract: the runtime reports whether
        # it actually injected the conditioning it received.
        applied = (
            self._applies_personal_conditioning
            and kwargs.get("personal_conditioning") is not None
        )
        return SimpleNamespace(
            text="hello",
            token_count=1,
            personal_conditioning_applied=applied,
            character_prefix_applied=False,
            character_prefix_id=None,
            character_prefix_wiring_level="disabled",
            conditioning_bank_carriers_applied=(),
        )


def _context() -> ResponseContext:
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
        user_input="hi",
    )


def _assembly() -> ResponseAssemblySnapshot:
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
        description="test assembly",
    )


def test_llm_synthesizer_disables_residual_capture_for_expression_generate() -> None:
    runtime = _RecordingRuntime()
    synthesizer = LLMResponseSynthesizer(runtime=runtime)

    response = synthesizer.synthesize(context=_context(), assembly=_assembly())

    assert response.text == "hello"
    assert runtime.calls
    assert runtime.calls[0]["capture_residuals"] is False
    assert runtime.calls[0]["stop_after_complete_json_object"] is False


def _intent_output_contract(*, exact_action: bool = False) -> ExpressionOutputContract:
    return ExpressionOutputContract(
        schema_name="lifeform_intent_v1",
        schema_json=json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["utterance", "intended_action"],
                "properties": {
                    "utterance": {"type": "string", "minLength": 1},
                    "intended_action": {"type": "string", "minLength": 1},
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        exact_bindings=(
            (
                ExpressionExactBinding(
                    json_pointer="/intended_action",
                    source=(
                        ExpressionBindingSource.RESPONSE_ACTION_REALIZATION_ACTION_STATEMENT
                    ),
                ),
            )
            if exact_action
            else ()
        ),
    )


def _action_assembly() -> ResponseAssemblySnapshot:
    return replace(
        _assembly(),
        action_realization=ResponseActionRealization(
            abstract_action="discovered_family_0",
            source_case_id="case:gate-water-mark",
            action_labels=("block-door", "signal-retreat"),
            action_statement="我会先挡在门前，再示意阿兰后退。",
            grounding_confidence=0.91,
            description="owner-published action",
        ),
    )


class _SequenceRuntime(_RecordingRuntime):
    def __init__(self, outputs: list[str]) -> None:
        super().__init__()
        self.outputs = list(outputs)

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            text=self.outputs.pop(0),
            token_count=4,
            personal_conditioning_applied=False,
            character_prefix_applied=False,
            character_prefix_id=None,
            character_prefix_wiring_level="disabled",
            conditioning_bank_carriers_applied=(),
        )


def test_strict_expression_retries_format_only_against_same_context() -> None:
    runtime = _SequenceRuntime(
        [
            '{"utterance":"我知道了"}',
            '{"utterance":"我知道了","intended_action":"去检查水痕"}',
        ]
    )
    context = replace(
        _context(), expression_output_contract=_intent_output_contract()
    )

    response = LLMResponseSynthesizer(runtime=runtime).synthesize(
        context=context,
        assembly=_assembly(),
    )

    assert response.text == (
        '{"utterance":"我知道了","intended_action":"去检查水痕"}'
    )
    assert len(runtime.calls) == 2
    assert [
        call["stop_after_complete_json_object"] for call in runtime.calls
    ] == [True, True]
    assert runtime.calls[0]["temperature"] == 0.0
    assert "Expression delivery contract" in runtime.calls[0]["chat_messages"][0][1]
    assert runtime.calls[1]["prompt"] == runtime.calls[0]["prompt"]
    assert runtime.calls[1]["control_parameters"] == runtime.calls[0][
        "control_parameters"
    ]
    assert ("assistant", '{"utterance":"我知道了"}') in runtime.calls[1][
        "chat_messages"
    ]
    assert "expression_format_retries=1" in response.rationale_tags


def test_strict_expression_fails_loudly_after_one_retry() -> None:
    runtime = _SequenceRuntime(["not json", '{"utterance":"still partial"}'])
    context = replace(
        _context(), expression_output_contract=_intent_output_contract()
    )

    with pytest.raises(StructuredExpressionOutputError):
        LLMResponseSynthesizer(runtime=runtime).synthesize(
            context=context,
            assembly=_assembly(),
        )

    assert len(runtime.calls) == 2


def test_exact_action_binding_retries_then_preserves_owner_unicode() -> None:
    owner_action = "我会先挡在门前，再示意阿兰后退。"
    runtime = _SequenceRuntime(
        [
            '{"utterance":"我看见了","intended_action":"我去追问来人。"}',
            json.dumps(
                {
                    "utterance": "我看见了",
                    "intended_action": owner_action,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ]
    )
    context = replace(
        _context(),
        expression_output_contract=_intent_output_contract(exact_action=True),
    )

    response = LLMResponseSynthesizer(runtime=runtime).synthesize(
        context=context,
        assembly=_action_assembly(),
    )

    assert json.loads(response.text)["intended_action"] == owner_action
    assert len(runtime.calls) == 2
    first_system = runtime.calls[0]["chat_messages"][0][1]
    assert "Exact owner-published output bindings" in first_system
    assert owner_action in first_system
    retry_messages = runtime.calls[1]["chat_messages"]
    assert owner_action in retry_messages[-1][1]


def test_exact_action_binding_fails_before_generation_without_owner_readout() -> None:
    runtime = _SequenceRuntime([])
    context = replace(
        _context(),
        expression_output_contract=_intent_output_contract(exact_action=True),
    )

    with pytest.raises(
        ExpressionExactBindingSourceError,
        match="ResponseAssembly.action_realization",
    ):
        LLMResponseSynthesizer(runtime=runtime).synthesize(
            context=context,
            assembly=_assembly(),
        )

    assert runtime.calls == []


def test_exact_action_binding_cannot_invent_after_retry() -> None:
    runtime = _SequenceRuntime(
        [
            '{"utterance":"一","intended_action":"改写一"}',
            '{"utterance":"二","intended_action":"改写二"}',
        ]
    )
    context = replace(
        _context(),
        expression_output_contract=_intent_output_contract(exact_action=True),
    )

    with pytest.raises(
        StructuredExpressionOutputError,
        match="exact binding mismatch",
    ):
        LLMResponseSynthesizer(runtime=runtime).synthesize(
            context=context,
            assembly=_action_assembly(),
        )

    assert len(runtime.calls) == 2


def test_exact_binding_requires_required_string_schema_target() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"intended_action": {"type": "string"}},
    }

    with pytest.raises(ValueError, match="must be required"):
        ExpressionOutputContract(
            schema_name="invalid_owner_binding",
            schema_json=json.dumps(schema, separators=(",", ":"), sort_keys=True),
            exact_bindings=(
                ExpressionExactBinding(
                    json_pointer="/intended_action",
                    source=(
                        ExpressionBindingSource.RESPONSE_ACTION_REALIZATION_ACTION_STATEMENT
                    ),
                ),
            ),
        )


def test_evidence_profile_captures_normalized_conditioned_runtime_context() -> None:
    class EvidenceRuntime(SyntheticOpenWeightResidualRuntime):
        def __init__(self) -> None:
            super().__init__(model_id="evidence-model")
            self.capture_requested = False

        def generate(self, **kwargs: Any) -> GenerationResult:
            self.capture_requested = kwargs["capture_residuals"] is True
            source = str(kwargs["prompt"])
            return GenerationResult(
                text="captured response",
                token_count=2,
                capture=self.capture(source_text=source),
                description="evidence fixture",
                input_token_count=5,
                source_sha256=hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest(),
            )

    runtime = EvidenceRuntime()
    response = LLMResponseSynthesizer(
        runtime=runtime,
        capture_runtime_context=True,
        runtime_model_fingerprint=SubstrateFingerprint(
            model_id="evidence-model",
            version="fixture-v1",
            weights_sha256="a" * 64,
        ),
    ).synthesize(context=_context(), assembly=_assembly())

    evidence = response.runtime_context_evidence
    assert runtime.capture_requested
    assert evidence is not None
    assert evidence.input_token_count == 5
    assert evidence.output_token_count == 2
    assert evidence.generation_latency_ms >= 0.0
    values = evidence.representation.representations[0].values
    assert sum(value * value for value in values) == pytest.approx(1.0)


def test_llm_synthesizer_forwards_immutable_character_id() -> None:
    runtime = _RecordingRuntime()
    synthesizer = LLMResponseSynthesizer(
        runtime=runtime,
        character_id="zhang-wuji",
    )

    response = synthesizer.synthesize(context=_context(), assembly=_assembly())

    assert runtime.calls[0]["character_id"] == "zhang-wuji"
    assert "character_id=zhang-wuji" in response.rationale_tags


def _conditioning() -> PersonalConditioningSnapshot:
    return PersonalConditioningSnapshot(
        schema_version=PERSONAL_CONDITIONING_SCHEMA_VERSION,
        state_vector=tuple(0.5 for _ in PERSONAL_CONDITIONING_VECTOR_LABELS),
        vector_labels=PERSONAL_CONDITIONING_VECTOR_LABELS,
        source_versions=(("user_model", 2),),
        source_fingerprint="conditioning-test",
        confidence=0.7,
        is_cold_start=False,
        description="test",
    )


def test_llm_synthesizer_forwards_active_personal_conditioning() -> None:
    runtime = _RecordingRuntime(applies_personal_conditioning=True)
    synthesizer = LLMResponseSynthesizer(runtime=runtime)
    conditioning = _conditioning()

    response = synthesizer.synthesize(
        context=replace(_context(), personal_conditioning=conditioning),
        assembly=_assembly(),
    )

    assert response.text == "hello"
    assert runtime.calls[0]["personal_conditioning"] is conditioning
    assert (
        "personal_conditioning=personal-conditioning.v1:0.70:conditioning"
        in response.rationale_tags
    )
    assert not any(
        tag.startswith("personal_conditioning_not_applied=")
        for tag in response.rationale_tags
    )


def test_llm_synthesizer_audit_tag_reflects_non_injection() -> None:
    # The runtime received the snapshot but did not inject it (e.g.
    # trace-only synthetic backend). The audit tag must not claim
    # injection; it must record the explicit not-applied lineage instead.
    runtime = _RecordingRuntime(applies_personal_conditioning=False)
    synthesizer = LLMResponseSynthesizer(runtime=runtime)
    conditioning = _conditioning()

    response = synthesizer.synthesize(
        context=replace(_context(), personal_conditioning=conditioning),
        assembly=_assembly(),
    )

    assert runtime.calls[0]["personal_conditioning"] is conditioning
    assert (
        "personal_conditioning_not_applied="
        "personal-conditioning.v1:0.70:conditioning"
        in response.rationale_tags
    )
    assert not any(
        tag.startswith("personal_conditioning=")
        for tag in response.rationale_tags
    )


def test_llm_synthesizer_text_conditioning_uses_prompt_only() -> None:
    runtime = _RecordingRuntime(applies_personal_conditioning=True)
    synthesizer = LLMResponseSynthesizer(runtime=runtime)
    statement = (
        "Current relational state estimate (typed readout only, confidence 0.70):\n"
        "- User: overall stability moderate (0.50)."
    )

    response = synthesizer.synthesize(
        context=replace(
            _context(),
            personal_conditioning=None,
            personal_conditioning_statement=statement,
            personal_conditioning_statement_ref=(
                "personal-conditioning.v1:0.70:conditioning"
            ),
        ),
        assembly=_assembly(),
    )

    call = runtime.calls[0]
    assert call["personal_conditioning"] is None
    assert statement in call["system_context"]
    assert "Private background state for calibration only." in call["system_context"]
    assert (
        "personal_conditioning_text="
        "personal-conditioning.v1:0.70:conditioning"
        in response.rationale_tags
    )
    assert not any(
        tag.startswith(
            ("personal_conditioning=", "personal_conditioning_not_applied=")
        )
        for tag in response.rationale_tags
    )


def test_response_context_rejects_dual_conditioning_delivery() -> None:
    with pytest.raises(ValueError, match="exactly one delivery path"):
        replace(
            _context(),
            personal_conditioning=_conditioning(),
            personal_conditioning_statement="rendered state",
            personal_conditioning_statement_ref=(
                "personal-conditioning.v1:0.70:conditioning"
            ),
        )


def test_response_context_requires_text_conditioning_lineage() -> None:
    with pytest.raises(ValueError, match="audit lineage"):
        replace(
            _context(),
            personal_conditioning_statement="rendered state",
        )
