from __future__ import annotations

import pytest

from volvence_zero.state_kv_bank_gain_gate import (
    BANK_GAIN_SCHEMA_VERSION,
    IrrelevantBankControlSample,
    PairedBankGainSample,
    build_bank_gain_verdict,
)


def _gain_samples(bank_type: str) -> tuple[PairedBankGainSample, ...]:
    return tuple(
        PairedBankGainSample(
            probe_id=f"{bank_type}-{index}",
            bank_type=bank_type,
            dual_output=f"dual-{bank_type}-{index}",
            ablated_output=f"ablated-{bank_type}-{index}",
            dual_match_correct=True,
            ablated_match_correct=False,
        )
        for index in range(8)
    )


def _irrelevant_controls(
    *,
    router_score: float = 0.05,
    with_bank_match_correct: bool = False,
) -> tuple[IrrelevantBankControlSample, ...]:
    return tuple(
        IrrelevantBankControlSample(
            probe_id=f"irrelevant-{index}",
            bank_type="relationship",
            router_score=router_score,
            without_bank_match_correct=False,
            with_bank_match_correct=with_bank_match_correct,
        )
        for index in range(8)
    )


def _verdict(
    *,
    paired_samples: tuple[PairedBankGainSample, ...],
    irrelevant_controls: tuple[IrrelevantBankControlSample, ...],
):
    return build_bank_gain_verdict(
        paired_samples=paired_samples,
        irrelevant_controls=irrelevant_controls,
        artifact_id="bank-gain-test",
        substrate_fingerprint="substrate-fp",
        router_version="topk-semantic.v1",
    )


def test_all_bank_gain_and_negative_control_claims_pass() -> None:
    verdict = _verdict(
        paired_samples=(
            *_gain_samples("personal"),
            *_gain_samples("relationship"),
        ),
        irrelevant_controls=_irrelevant_controls(),
    )

    assert verdict.gate_state == "pass"
    assert verdict.bank_count_frozen is False
    assert all(claim.state == "pass" for claim in verdict.claims)
    assert verdict.as_json_dict()["schema_version"] == BANK_GAIN_SCHEMA_VERSION


def test_missing_bank_evidence_is_insufficient_without_freezing_count() -> None:
    verdict = _verdict(
        paired_samples=_gain_samples("personal"),
        irrelevant_controls=_irrelevant_controls(),
    )

    assert verdict.gate_state == "insufficient_data"
    assert verdict.bank_count_frozen is False
    assert verdict.freeze_reason == ""
    relationship = next(
        claim
        for claim in verdict.claims
        if claim.claim == "claim_relationship_independent_gain"
    )
    assert relationship.state == "insufficient_data"


@pytest.mark.parametrize(
    "controls",
    (
        _irrelevant_controls(router_score=0.8),
        _irrelevant_controls(with_bank_match_correct=True),
    ),
)
def test_irrelevant_bank_signal_fails_negative_control(
    controls: tuple[IrrelevantBankControlSample, ...],
) -> None:
    verdict = _verdict(
        paired_samples=(
            *_gain_samples("personal"),
            *_gain_samples("relationship"),
        ),
        irrelevant_controls=controls,
    )

    assert verdict.gate_state == "fail"
    assert verdict.bank_count_frozen is True
    assert verdict.claims[-1].state == "fail"


def test_paired_judge_outcomes_must_be_complete() -> None:
    with pytest.raises(ValueError, match="both blind-judge"):
        PairedBankGainSample(
            probe_id="broken",
            bank_type="personal",
            dual_output="dual",
            ablated_output="ablated",
            dual_match_correct=True,
            ablated_match_correct=None,
        )
