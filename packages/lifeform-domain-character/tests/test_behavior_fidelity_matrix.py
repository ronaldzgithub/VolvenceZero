from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lifeform_domain_character import (
    BehaviorFidelityCaseKind,
    PromotionExpectation,
    build_zhang_wuji_profile,
    load_behavior_fidelity_matrix,
    load_zhang_wuji_action_applicability_matrix,
    read_ledger_json,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEDGER = (
    _REPO_ROOT
    / "artifacts"
    / "character-live-through"
    / "zhang_wuji.reviewed_ledger.json"
)
_EXPECTED_MATRIX_DIGEST = (
    "5cf094b9446cad43bdf0544cdcf9c8d37fcc5cc8fbeb75731886bf71cae9e1b7"
)


def test_zhang_wuji_matrix_freezes_balanced_holdout_and_thresholds() -> None:
    matrix = load_zhang_wuji_action_applicability_matrix()

    assert matrix.suite_id == "zhang-wuji-action-applicability-v1"
    assert matrix.character_id == "zhang-wuji"
    assert matrix.target_schema_id == (
        "intervene-to-stop-imminent-third-party-harm"
    )
    assert matrix.source_chapter_ids == ("ch-11", "ch-17")
    assert len(matrix.cases) == 16
    assert dict(matrix.thresholds.required_case_counts) == {
        BehaviorFidelityCaseKind.POSITIVE: 4,
        BehaviorFidelityCaseKind.NEAR_NEGATIVE: 4,
        BehaviorFidelityCaseKind.INSUFFICIENT_EVIDENCE: 4,
        BehaviorFidelityCaseKind.COMPETING_BEHAVIOR: 4,
    }
    assert matrix.thresholds.minimum_positive_promotion_hits == 3
    assert matrix.thresholds.maximum_non_positive_promotion_hits == 0
    assert matrix.thresholds.minimum_case_fidelity_score == 0.75
    assert (
        matrix.thresholds.minimum_positive_mean_baked_cold_delta
        == 0.2
    )
    assert matrix.thresholds.require_source_digest_verified is True
    assert matrix.thresholds.require_no_feedback is True
    assert matrix.thresholds.require_competing_family_match is True
    assert matrix.digest == _EXPECTED_MATRIX_DIGEST


def test_zhang_wuji_matrix_is_oracle_separated_and_source_held_out() -> None:
    matrix = load_zhang_wuji_action_applicability_matrix()
    ledger = read_ledger_json(_LEDGER)
    profile = build_zhang_wuji_profile()
    serialized_sources = repr((ledger, profile))

    for case in matrix.cases:
        assert case.stimulus.character_id == matrix.character_id
        assert case.stimulus.setting not in serialized_sources
        assert case.stimulus.decision_point not in serialized_sources
        assert (
            case.reference.canonical_action
            not in case.stimulus.setting_prompt
        )
        assert (
            case.reference.canonical_action
            not in case.stimulus.decision_prompt
        )
        assert (
            case.reference.canonical_outcome
            not in case.stimulus.setting_prompt
        )
        assert (
            case.reference.canonical_outcome
            not in case.stimulus.decision_prompt
        )


def test_zhang_wuji_matrix_freezes_promotion_and_competing_behavior() -> None:
    matrix = load_zhang_wuji_action_applicability_matrix()

    positives = tuple(
        case
        for case in matrix.cases
        if case.kind is BehaviorFidelityCaseKind.POSITIVE
    )
    non_positives = tuple(
        case
        for case in matrix.cases
        if case.kind is not BehaviorFidelityCaseKind.POSITIVE
    )
    competing = tuple(
        case
        for case in matrix.cases
        if case.kind is BehaviorFidelityCaseKind.COMPETING_BEHAVIOR
    )

    assert all(
        case.promotion_expectation is PromotionExpectation.REQUIRED
        for case in positives
    )
    assert all(
        case.promotion_expectation is PromotionExpectation.FORBIDDEN
        for case in non_positives
    )
    assert {
        case.expected_behavior_family for case in competing
    } == {
        "gentle-aid",
        "fact-finding",
        "relationship-repair",
        "boundary-preservation",
    }


def test_matrix_contract_fails_closed_on_semantic_role_mismatch() -> None:
    matrix = load_zhang_wuji_action_applicability_matrix()
    positive = next(
        case
        for case in matrix.cases
        if case.kind is BehaviorFidelityCaseKind.POSITIVE
    )

    with pytest.raises(
        ValueError,
        match="positive case requires promotion expectation 'required'",
    ):
        replace(
            positive,
            promotion_expectation=PromotionExpectation.FORBIDDEN,
        )

    with pytest.raises(
        ValueError,
        match="matrix case counts do not match frozen thresholds",
    ):
        replace(matrix, cases=matrix.cases[:-1])


def test_matrix_loader_rejects_unregistered_fields(tmp_path: Path) -> None:
    matrix_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "lifeform_domain_character"
        / "evaluation"
        / "zhang_wuji_action_applicability_v1.json"
    )
    raw = json.loads(matrix_path.read_text(encoding="utf-8"))
    raw["runtime_reward"] = 1.0
    invalid_path = tmp_path / "invalid-matrix.json"
    invalid_path.write_text(
        json.dumps(raw, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="matrix keys mismatch"):
        load_behavior_fidelity_matrix(invalid_path)


def test_matrix_loader_rejects_wrong_types_and_extra_counts(
    tmp_path: Path,
) -> None:
    matrix_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "lifeform_domain_character"
        / "evaluation"
        / "zhang_wuji_action_applicability_v1.json"
    )
    raw = json.loads(matrix_path.read_text(encoding="utf-8"))
    raw["thresholds"]["require_no_feedback"] = "true"
    invalid_type_path = tmp_path / "invalid-type-matrix.json"
    invalid_type_path.write_text(
        json.dumps(raw, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="require_no_feedback"):
        load_behavior_fidelity_matrix(invalid_type_path)

    raw = json.loads(matrix_path.read_text(encoding="utf-8"))
    raw["thresholds"]["required_case_counts"]["unknown"] = 1
    invalid_count_path = tmp_path / "invalid-count-matrix.json"
    invalid_count_path.write_text(
        json.dumps(raw, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="required_case_counts keys mismatch"):
        load_behavior_fidelity_matrix(invalid_count_path)
