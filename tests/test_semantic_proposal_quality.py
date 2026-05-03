from __future__ import annotations

from volvence_zero.semantic_state import SemanticProposalOperation
from volvence_zero.semantic_state.llm_runtime import LLMSemanticProposalRuntime
from volvence_zero.semantic_state.quality import (
    SemanticProposalQualityCase,
    evaluate_semantic_proposal_quality,
)


class _ScriptedProvider:
    def __init__(self, responses: tuple[str, ...]) -> None:
        self._responses = list(responses)

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
    ) -> str:
        del prompt, max_new_tokens, temperature
        if not self._responses:
            raise RuntimeError("scripted provider exhausted")
        return self._responses.pop(0)


def test_boundary_consent_quality_harness_reports_precision_and_recall() -> None:
    runtime = LLMSemanticProposalRuntime(
        provider=_ScriptedProvider(
            (
                (
                    '{"runtime_id": "test", "schema_version": 1, "description": "deny", '
                    '"proposals": [{"proposal_id": "ignored", "target_slot": "boundary_consent", '
                    '"operation": "block", "summary": "external action denied", '
                    '"detail": "User denied external action.", "confidence": 0.82, '
                    '"evidence": "Do not act externally", "control_signal": 0.7}]}'
                ),
                (
                    '{"runtime_id": "test", "schema_version": 1, "description": "grant", '
                    '"proposals": [{"proposal_id": "ignored", "target_slot": "boundary_consent", '
                    '"operation": "observe", "summary": "memory scope granted", '
                    '"detail": "User granted memory scope.", "confidence": 0.72, '
                    '"evidence": "You can remember my planning preference"}]}'
                ),
                (
                    '{"runtime_id": "test", "schema_version": 1, "description": "ambiguous", '
                    '"proposals": []}'
                ),
            )
        )
    )
    cases = (
        SemanticProposalQualityCase(
            case_id="deny-external-action",
            target_slot="boundary_consent",
            user_input="Do not act externally without asking me.",
            expected_operations=(SemanticProposalOperation.BLOCK,),
            min_confidence=0.70,
        ),
        SemanticProposalQualityCase(
            case_id="grant-memory-scope",
            target_slot="boundary_consent",
            user_input="You can remember my planning preference.",
            expected_operations=(),
        ),
        SemanticProposalQualityCase(
            case_id="ambiguous-consent",
            target_slot="boundary_consent",
            user_input="Maybe keep that in mind, I am not sure yet.",
            expected_operations=(),
        ),
    )

    report = evaluate_semantic_proposal_quality(runtime=runtime, cases=cases)

    assert report.target_slot == "boundary_consent"
    assert report.passed_case_count == 3
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.false_positive_count == 0
    assert report.missing_count == 0
    assert report.would_block_count == 0
    assert report.would_allow_count == 3
    assert report.shadow_gate_reasons == ()
    assert "precision=1.000" in report.description


def test_boundary_consent_quality_harness_catches_false_positive() -> None:
    runtime = LLMSemanticProposalRuntime(
        provider=_ScriptedProvider(
            (
                (
                    '{"runtime_id": "test", "schema_version": 1, "description": "false-positive", '
                    '"proposals": [{"proposal_id": "ignored", "target_slot": "boundary_consent", '
                    '"operation": "block", "summary": "overclassified denial", '
                    '"detail": "Runtime overclassified ambiguous consent.", "confidence": 0.80, '
                    '"evidence": "Maybe keep that in mind"}]}'
                ),
            )
        )
    )
    cases = (
        SemanticProposalQualityCase(
            case_id="ambiguous-consent",
            target_slot="boundary_consent",
            user_input="Maybe keep that in mind, I am not sure yet.",
            expected_operations=(),
        ),
    )

    report = evaluate_semantic_proposal_quality(runtime=runtime, cases=cases)

    assert report.passed_case_count == 0
    assert report.precision == 0.0
    assert report.recall == 1.0
    assert report.false_positive_count == 1
    assert report.would_block_count == 1
    assert report.shadow_gate_reasons == (
        ("ambiguous-consent", ("false-positive",)),
    )
    assert report.case_results[0].would_block is True
    assert report.case_results[0].passed is False


def test_goal_value_quality_harness_reports_tradeoff_recall() -> None:
    runtime = LLMSemanticProposalRuntime(
        provider=_ScriptedProvider(
            (
                (
                    '{"runtime_id": "test", "schema_version": 1, "description": "goal-tradeoff", '
                    '"proposals": [{"proposal_id": "ignored", "target_slot": "goal_value", '
                    '"operation": "defer", "summary": "value tradeoff", '
                    '"detail": "User needs value clarification before choosing.", '
                    '"confidence": 0.76, "evidence": "I need to think about the tradeoff", '
                    '"control_signal": 0.55}]}'
                ),
                (
                    '{"runtime_id": "test", "schema_version": 1, "description": "goal-create", '
                    '"proposals": [{"proposal_id": "ignored", "target_slot": "goal_value", '
                    '"operation": "create", "summary": "safer launch goal", '
                    '"detail": "User wants a safer launch path.", "confidence": 0.81, '
                    '"evidence": "I want a safer launch"}]}'
                ),
            )
        )
    )
    cases = (
        SemanticProposalQualityCase(
            case_id="goal-tradeoff",
            target_slot="goal_value",
            user_input="I need to think about the tradeoff before choosing.",
            expected_operations=(SemanticProposalOperation.DEFER,),
            min_confidence=0.70,
        ),
        SemanticProposalQualityCase(
            case_id="goal-create",
            target_slot="goal_value",
            user_input="I want a safer launch path.",
            expected_operations=(SemanticProposalOperation.CREATE,),
            min_confidence=0.70,
        ),
    )

    report = evaluate_semantic_proposal_quality(runtime=runtime, cases=cases)

    assert report.target_slot == "goal_value"
    assert report.passed_case_count == 2
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.fallback_count == 0
    assert report.would_block_count == 0


def test_quality_harness_reports_fallback_count_for_unparseable_payload() -> None:
    runtime = LLMSemanticProposalRuntime(
        provider=_ScriptedProvider(("not json",))
    )
    cases = (
        SemanticProposalQualityCase(
            case_id="bad-boundary-payload",
            target_slot="boundary_consent",
            user_input="Do not act externally.",
            expected_operations=(SemanticProposalOperation.BLOCK,),
        ),
    )

    report = evaluate_semantic_proposal_quality(runtime=runtime, cases=cases)

    assert report.passed_case_count == 0
    assert report.fallback_count == 1
    assert report.missing_count == 1
    assert report.would_block_count == 1
    assert report.shadow_gate_reasons == (
        ("bad-boundary-payload", ("missing-expected-operation", "runtime-fallback")),
    )
    assert report.case_results[0].fell_back is True
    assert report.case_results[0].would_block is True


def test_quality_harness_shadow_gate_blocks_low_confidence() -> None:
    runtime = LLMSemanticProposalRuntime(
        provider=_ScriptedProvider(
            (
                (
                    '{"runtime_id": "test", "schema_version": 1, "description": "low-confidence", '
                    '"proposals": [{"proposal_id": "ignored", "target_slot": "goal_value", '
                    '"operation": "create", "summary": "weak goal", '
                    '"detail": "Low confidence goal proposal.", "confidence": 0.41, '
                    '"evidence": "I might want this"}]}'
                ),
            )
        )
    )
    cases = (
        SemanticProposalQualityCase(
            case_id="low-confidence-goal",
            target_slot="goal_value",
            user_input="I might want this.",
            expected_operations=(SemanticProposalOperation.CREATE,),
            min_confidence=0.70,
        ),
    )

    report = evaluate_semantic_proposal_quality(runtime=runtime, cases=cases)

    assert report.passed_case_count == 0
    assert report.would_block_count == 1
    assert report.shadow_gate_reasons == (
        ("low-confidence-goal", ("confidence-below-floor",)),
    )


def test_quality_harness_rejects_mixed_target_slots() -> None:
    runtime = LLMSemanticProposalRuntime(provider=_ScriptedProvider(()))
    cases = (
        SemanticProposalQualityCase(
            case_id="boundary",
            target_slot="boundary_consent",
            user_input="Do not act externally.",
            expected_operations=(),
        ),
        SemanticProposalQualityCase(
            case_id="goal",
            target_slot="goal_value",
            user_input="I want a safer launch.",
            expected_operations=(),
        ),
    )

    try:
        evaluate_semantic_proposal_quality(runtime=runtime, cases=cases)
    except ValueError as exc:
        assert "same slot" in str(exc)
    else:
        raise AssertionError("mixed target slots must fail loudly")
