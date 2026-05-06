"""Wave 1 part A: rationale_tags is the structured audit surface.

These tests lock the typed tag schema so consumers (gates, evaluators,
reflection) can read ``AgentResponse.rationale_tags`` rather than
substring-matching the human-readable ``rationale``. Each test names
the tag it expects to see; if the synthesizer phrasing changes, the
tag still has to be present \u2014 substring matches in the rendered
text are no longer load-bearing.

See ``docs/specs/expression-layer.md``.
"""

from __future__ import annotations

from lifeform_expression.prompt_planner import PromptPlanner, TurnIntent
from lifeform_expression.response_synthesizer import GroundedResponseSynthesizer

from volvence_zero.agent.response import (
    AgentResponse,
    RepairExpressionAdvisory,
    ResponseContext,
    ResponseSynthesizer,
)
from volvence_zero.application.runtime import (
    ResponseAssemblySnapshot,
    ResponseMode,
    RiskBand,
)


def _context(
    *,
    regime_id: str = "emotional_support",
    repair_advisory: RepairExpressionAdvisory | None = None,
) -> ResponseContext:
    return ResponseContext(
        regime_id=regime_id,
        regime_name=regime_id,
        regime_switched=False,
        abstract_action=None,
        alert_count=0,
        temporal_switch_gate=0.1,
        temporal_is_switching=False,
        reflection_lesson_count=0,
        reflection_tension_count=0,
        reflection_writeback_applied=False,
        primary_reflection_lesson=None,
        primary_reflection_tension=None,
        joint_schedule_action="idle",
        user_input="",
        repair_advisory=repair_advisory,
    )


def _assembly(
    *,
    regime_id: str = "emotional_support",
    expression_intent: str = "support-first",
    response_mode: ResponseMode = ResponseMode.SUPPORT,
) -> ResponseAssemblySnapshot:
    return ResponseAssemblySnapshot(
        regime_id=regime_id,
        regime_name=regime_id,
        abstract_action=None,
        response_mode=response_mode,
        answer_depth_limit="standard",
        citation_mode="optional",
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
        max_questions=1,
        prompt_residue_summary="",
        prompt_residue_ratio=0.0,
        knowledge_hit_count=0,
        case_hit_count=0,
        playbook_rule_count=0,
        risk_band=RiskBand.LOW,
        description="test assembly",
        continuum_target_position=0.5,
        ordering_driver="playbook-only",
        expression_intent=expression_intent,
    )


def test_kernel_response_synthesizer_emits_typed_rationale_tags() -> None:
    """Even the kernel-default synthesizer (no PromptPlanner) must
    populate ``rationale_tags``. Downstream evaluators must never have
    to fall back to substring scanning the rationale text.
    """

    response = ResponseSynthesizer().synthesize(
        context=_context(),
        assembly=_assembly(),
    )

    assert isinstance(response, AgentResponse)
    assert response.rationale_tags, "kernel synthesize must emit typed tags"
    assert any(
        tag.startswith("regime=") for tag in response.rationale_tags
    ), f"missing regime= tag in {response.rationale_tags!r}"
    assert any(
        tag.startswith("switch_gate=") for tag in response.rationale_tags
    ), f"missing switch_gate= tag in {response.rationale_tags!r}"
    assert any(
        tag.startswith("risk=") for tag in response.rationale_tags
    ), f"missing risk= tag in {response.rationale_tags!r}"


def test_grounded_synthesizer_emits_intent_and_section_tags() -> None:
    synthesizer = GroundedResponseSynthesizer(planner=PromptPlanner())
    response = synthesizer.synthesize(
        context=_context(),
        assembly=_assembly(),
    )

    assert response.rationale_tags, "grounded synthesize must emit tags"
    assert any(
        tag == "intent=support-first" for tag in response.rationale_tags
    ), f"missing intent=support-first in {response.rationale_tags!r}"
    assert any(
        tag.startswith("plan=intent:") for tag in response.rationale_tags
    ), f"missing plan summary tag in {response.rationale_tags!r}"


def test_repair_alpha_emits_typed_section_variant_tags() -> None:
    """Wave 1 keystone: the relationship_repair_alpha gate reads typed
    ``acknowledge_section=repair_alpha`` (and friends) instead of
    matching substrings of the rendered prose. This test pins the
    contract.
    """

    advisory = RepairExpressionAdvisory(
        rupture_kind="over_directive",
        confidence=0.85,
        signal_strength=0.9,
        description="Rupture kind=over_directive strength=0.90 confidence=0.85.",
    )
    synthesizer = GroundedResponseSynthesizer(
        planner=PromptPlanner(repair_alpha_enabled=True)
    )
    response = synthesizer.synthesize(
        context=_context(repair_advisory=advisory),
        assembly=_assembly(expression_intent="repair-first"),
    )

    tags = response.rationale_tags
    assert "intent=repair-first" in tags, f"missing repair-first intent in {tags!r}"
    assert (
        "repair_alpha=over_directive" in tags
    ), f"missing repair_alpha tag in {tags!r}"
    assert (
        "acknowledge_section=repair_alpha" in tags
    ), f"missing acknowledge_section variant in {tags!r}"


def test_no_substring_search_required_for_repair_alpha_phrase() -> None:
    """Regression for the Wave 1 substring-gate violation: the
    repair-alpha rendering's signature is the typed
    ``acknowledge_section=repair_alpha`` tag; the rendered prose is
    NOT a contract and may evolve. This test rejects the previous
    pattern of asserting on ``"pause the optimizing frame" in text``.
    """

    advisory = RepairExpressionAdvisory(
        rupture_kind="over_directive",
        confidence=0.85,
        signal_strength=0.9,
        description="Rupture kind=over_directive strength=0.90 confidence=0.85.",
    )
    synthesizer = GroundedResponseSynthesizer(
        planner=PromptPlanner(repair_alpha_enabled=True)
    )
    response = synthesizer.synthesize(
        context=_context(repair_advisory=advisory),
        assembly=_assembly(expression_intent="repair-first"),
    )

    typed_signal = "acknowledge_section=repair_alpha" in response.rationale_tags
    assert typed_signal, (
        "Repair-alpha activation must be visible via typed rationale_tags. "
        f"Got tags: {response.rationale_tags!r}"
    )


def test_attach_plan_rationale_preserves_kernel_tags() -> None:
    """``_attach_plan_rationale`` is used on the DELEGATE_TO_BASE path.
    Kernel-emitted tags must survive the attach step; otherwise vitals
    and reflection signals would vanish on refer-out / direct-answer
    turns.
    """

    synthesizer = GroundedResponseSynthesizer(planner=PromptPlanner())
    response = synthesizer.synthesize(
        context=_context(regime_id="casual_social"),
        assembly=_assembly(
            regime_id="casual_social",
            expression_intent="direct-answer",
        ),
    )

    assert response.rationale_tags, "delegate path must still emit tags"
    assert any(
        tag.startswith("regime=") for tag in response.rationale_tags
    ), f"kernel-side regime tag was dropped: {response.rationale_tags!r}"
    assert any(
        tag.startswith("plan=intent:") for tag in response.rationale_tags
    ), f"plan summary tag missing on delegate path: {response.rationale_tags!r}"
