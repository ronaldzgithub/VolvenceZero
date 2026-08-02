from __future__ import annotations

import pytest

from volvence_zero.state_kv_bank_gain_gate import (
    BANK_GAIN_SCHEMA_VERSION,
    BankPersonaContrast,
    IrrelevantBankControlSample,
    NonBankPersonaControlSample,
    PairedBankGainSample,
    build_bank_gain_panel_verdict,
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


def _non_bank_controls(
    *,
    match_correct: bool | None = None,
) -> tuple[NonBankPersonaControlSample, ...]:
    return tuple(
        NonBankPersonaControlSample(
            probe_id=f"{bank_type}:n{index}",
            bank_type=bank_type,
            match_correct=(
                match_correct
                if match_correct is not None
                else index % 2 == 0
            ),
        )
        for bank_type in ("personal", "relationship")
        for index in range(8)
    )


def _verdict(
    *,
    paired_samples: tuple[PairedBankGainSample, ...],
    irrelevant_controls: tuple[IrrelevantBankControlSample, ...],
    non_bank_persona_controls: (
        tuple[NonBankPersonaControlSample, ...] | None
    ) = None,
    persona_contrasts: tuple[BankPersonaContrast, ...] | None = None,
):
    return build_bank_gain_verdict(
        paired_samples=paired_samples,
        irrelevant_controls=irrelevant_controls,
        non_bank_persona_controls=(
            non_bank_persona_controls
            if non_bank_persona_controls is not None
            else _non_bank_controls()
        ),
        persona_contrasts=(
            persona_contrasts
            if persona_contrasts is not None
            else tuple(
                BankPersonaContrast(
                    bank_type=bank_type,
                    probe_count=4,
                    material_contrast_count=4,
                    fingerprint_contrast_count=4,
                )
                for bank_type in ("personal", "relationship")
            )
        ),
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


def test_collapsed_persona_state_is_insufficient_not_causal_failure() -> None:
    verdict = _verdict(
        paired_samples=(
            *_gain_samples("personal"),
            *_gain_samples("relationship"),
        ),
        irrelevant_controls=_irrelevant_controls(),
        persona_contrasts=(
            BankPersonaContrast(
                bank_type="personal",
                probe_count=4,
                material_contrast_count=4,
                fingerprint_contrast_count=4,
            ),
            BankPersonaContrast(
                bank_type="relationship",
                probe_count=4,
                material_contrast_count=0,
                fingerprint_contrast_count=0,
            ),
        ),
    )

    assert verdict.gate_state == "insufficient_data"
    assert verdict.bank_count_frozen is False
    claims = {claim.claim: claim.state for claim in verdict.claims}
    assert claims["claim_relationship_state_contrast"] == "insufficient_data"
    assert (
        claims["claim_relationship_independent_gain"]
        == "insufficient_data"
    )


def test_non_bank_persona_leakage_is_insufficient_not_bank_failure() -> None:
    verdict = _verdict(
        paired_samples=(
            *_gain_samples("personal"),
            *_gain_samples("relationship"),
        ),
        irrelevant_controls=_irrelevant_controls(),
        non_bank_persona_controls=_non_bank_controls(match_correct=True),
    )

    assert verdict.gate_state == "insufficient_data"
    assert verdict.bank_count_frozen is False
    claims = {claim.claim: claim.state for claim in verdict.claims}
    assert claims["claim_personal_non_bank_isolation"] == "insufficient_data"
    assert claims["claim_relationship_non_bank_isolation"] == "insufficient_data"
    assert claims["claim_personal_independent_gain"] == "insufficient_data"
    assert claims["claim_relationship_independent_gain"] == "insufficient_data"


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


def test_v4_panel_requires_two_distinct_judges_on_same_observations() -> None:
    first = build_bank_gain_verdict(
        paired_samples=(*_gain_samples("personal"), *_gain_samples("relationship")),
        irrelevant_controls=_irrelevant_controls(),
        non_bank_persona_controls=_non_bank_controls(),
        persona_contrasts=tuple(
            BankPersonaContrast(
                bank_type=bank_type,
                probe_count=4,
                material_contrast_count=4,
                fingerprint_contrast_count=4,
            )
            for bank_type in ("personal", "relationship")
        ),
        artifact_id="bank-gain-test",
        substrate_fingerprint="substrate-fp",
        router_version="topk-semantic.v1",
        judge_model_id="judge-a",
        observation_artifact_sha256="a" * 64,
    )
    second = build_bank_gain_verdict(
        paired_samples=(*_gain_samples("personal"), *_gain_samples("relationship")),
        irrelevant_controls=_irrelevant_controls(),
        non_bank_persona_controls=_non_bank_controls(),
        persona_contrasts=tuple(
            BankPersonaContrast(
                bank_type=bank_type,
                probe_count=4,
                material_contrast_count=4,
                fingerprint_contrast_count=4,
            )
            for bank_type in ("personal", "relationship")
        ),
        artifact_id="bank-gain-test",
        substrate_fingerprint="substrate-fp",
        router_version="topk-semantic.v1",
        judge_model_id="judge-b",
        observation_artifact_sha256="a" * 64,
    )

    panel = build_bank_gain_panel_verdict(
        judge_verdicts=(first, second),
        preregistration_sha256="b" * 64,
    )

    assert panel.gate_state == "pass"
    assert panel.as_json_dict()["schema_version"] == "state-kv-bank-gain.v4"
    assert len(panel.as_json_dict()["judge_panel"]) == 2


def test_v4_panel_rejects_duplicate_judge() -> None:
    verdict = build_bank_gain_verdict(
        paired_samples=(*_gain_samples("personal"), *_gain_samples("relationship")),
        irrelevant_controls=_irrelevant_controls(),
        non_bank_persona_controls=_non_bank_controls(),
        persona_contrasts=tuple(
            BankPersonaContrast(
                bank_type=bank_type,
                probe_count=4,
                material_contrast_count=4,
                fingerprint_contrast_count=4,
            )
            for bank_type in ("personal", "relationship")
        ),
        artifact_id="bank-gain-test",
        substrate_fingerprint="substrate-fp",
        router_version="topk-semantic.v1",
        judge_model_id="judge-a",
        observation_artifact_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="distinct"):
        build_bank_gain_panel_verdict(
            judge_verdicts=(verdict, verdict),
            preregistration_sha256="b" * 64,
        )
