from volvence_zero.state_kv_credit_longitudinal import (
    CreditLongitudinalSample,
    CreditOutcomeJudgeResult,
    build_credit_longitudinal_verdict,
)


def _judge(*, model: str, family: str, improvements: int, regressions: int = 0):
    return CreditOutcomeJudgeResult(
        judge_model_id=model,
        judge_family=family,
        shadow_matched_count=3,
        active_matched_count=3 + improvements - regressions,
        improvement_count=improvements,
        regression_count=regressions,
        sample_count=8,
    )


def _samples(*, late_increment: float):
    return tuple(
        CreditLongitudinalSample(
            turn_index=index,
            shadow_confidence=0.5,
            active_confidence=0.5 + (0.0 if index < 4 else late_increment),
            shadow_credit_delta=0.02,
            active_credit_delta=0.02,
            responses_differ=index >= 4,
        )
        for index in range(8)
    )


def test_mechanism_does_not_overclaim_outcome_quality() -> None:
    verdict = build_credit_longitudinal_verdict(
        samples=_samples(late_increment=0.02),
        artifact_id="credit-longitudinal:test",
    )

    assert verdict.mechanism_state == "pass"
    assert verdict.outcome_claim_state == "insufficient_data"
    assert verdict.gate_state == "mechanism_supported"


def test_flat_increment_fails_growth_gate() -> None:
    verdict = build_credit_longitudinal_verdict(
        samples=_samples(late_increment=0.0),
        artifact_id="credit-longitudinal:test",
    )

    assert verdict.mechanism_state == "fail"
    assert verdict.gate_state == "fail"


def test_dual_frozen_outcome_judges_produce_terminal_pass() -> None:
    verdict = build_credit_longitudinal_verdict(
        samples=_samples(late_increment=0.02),
        artifact_id="credit-longitudinal:test",
        outcome_judges=(
            _judge(model="bge", family="xlm-roberta", improvements=1),
            _judge(model="m3e", family="bert", improvements=2),
        ),
    )

    assert verdict.outcome_claim_state == "pass"
    assert verdict.gate_state == "pass"


def test_dual_outcome_panel_fails_when_one_judge_has_no_gain() -> None:
    verdict = build_credit_longitudinal_verdict(
        samples=_samples(late_increment=0.02),
        artifact_id="credit-longitudinal:test",
        outcome_judges=(
            _judge(model="bge", family="xlm-roberta", improvements=1),
            _judge(model="m3e", family="bert", improvements=0),
        ),
    )

    assert verdict.outcome_claim_state == "fail"
    assert verdict.gate_state == "mechanism_supported"
