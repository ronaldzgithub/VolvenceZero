from volvence_zero.state_kv_credit_longitudinal import (
    CreditLongitudinalSample,
    build_credit_longitudinal_verdict,
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
