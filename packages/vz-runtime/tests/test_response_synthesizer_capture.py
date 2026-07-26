from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from volvence_zero.agent.response import LLMResponseSynthesizer, ResponseContext
from volvence_zero.application.runtime import (
    ResponseAssemblySnapshot,
    ResponseMode,
    RiskBand,
)
from volvence_zero.personal_conditioning_contracts import (
    PERSONAL_CONDITIONING_SCHEMA_VERSION,
    PERSONAL_CONDITIONING_VECTOR_LABELS,
    PersonalConditioningSnapshot,
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
