from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from lifeform_domain_emogpt.lab.relationship_residual_fit_corpus import (
    RelationshipResidualFitInput,
    build_relationship_residual_fit_corpus,
    load_relationship_residual_fit_protocol,
)
from lifeform_domain_emogpt.relationship_action_contracts import RelationshipAction
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGateDecision,
    RelationshipActionGateMode,
    RelationshipGateAction,
)
from volvence_zero.agent.named_action_steering_artifact_training import (
    NamedActionSteeringCorpus,
)
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    RelationshipConditionReadout,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
)


_DIRECTIONAL = (
    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE.value,
    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION.value,
)
_OUTCOMES = ("helped", "felt_heard", "missed", "over_directive")
_ACTION_TEXT = (
    "公开的用户上下文只说明此刻的关系边界。\n"
    "Choose one directional action: stay_present_without_probe | "
    "respect_space_with_return_option\n"
    "Action:"
)


def _candidate(action_id: str, offset: int) -> SocialActionCandidatePrediction:
    values = (
        (0.4, 0.3, 0.2, 0.1),
        (0.1, 0.2, 0.3, 0.4),
        (0.25, 0.25, 0.25, 0.25),
    )[offset]
    return SocialActionCandidatePrediction(
        action_id=action_id,
        outcomes=tuple(
            SocialActionOutcomeProbability(outcome_id, probability)
            for outcome_id, probability in zip(_OUTCOMES, values, strict=True)
        ),
    )


def _input(
    *,
    row_id: str,
    subject_scope: str,
    recommended_action_id: str,
) -> RelationshipResidualFitInput:
    digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
    readout = RelationshipConditionReadout(
        condition_label=(
            "belonging_uncertainty"
            if recommended_action_id == _DIRECTIONAL[0]
            else "agency_pressure"
        ),
        confidence=0.8,
        normalized_margin=0.3,
        candidate_scores=(
            (
                "belonging_uncertainty"
                if recommended_action_id == _DIRECTIONAL[0]
                else "agency_pressure",
                0.8,
            ),
            (
                "agency_pressure"
                if recommended_action_id == _DIRECTIONAL[0]
                else "belonging_uncertainty",
                0.2,
            ),
        ),
        reader_artifact_id="a" * 64,
        source_observation_sha256=digest,
    )
    forecast = PreferenceActionForecast(
        forecast_id=f"forecast-{row_id}",
        decision_id=f"decision-{row_id}",
        interlocutor_id=f"person-{subject_scope}",
        candidate_predictions=tuple(
            _candidate(action.value, index)
            for index, action in enumerate(RelationshipAction)
        ),
        recommended_action_id=recommended_action_id,
        confidence=0.8,
        source_record_ids=(f"record-{row_id}",),
        issued_turn=7,
        evidence=("owner:preference-about-other", "reader:named-condition"),
        session_scope=f"session-{subject_scope}",
        condition_readout=readout,
    )
    gate = RelationshipActionGateDecision(
        decision_id=forecast.decision_id,
        forecast_id=forecast.forecast_id,
        gate_action=RelationshipGateAction.STEER,
        selected_action_id=recommended_action_id,
        recommended_action_id=recommended_action_id,
        steer_probability=0.75,
        features=(0.2, 0.1, 0.8, 0.3, 0.4),
        mode=RelationshipActionGateMode.LEARNED,
        artifact_id="relationship-action-gate-test",
        artifact_version=1,
        update_count=3,
        evidence_refs=(f"record-{row_id}", "pe-credit-checkpoint-3"),
        rationale_codes=(
            "policy:bounded-logistic-gate",
            "learning:pe-credit-only",
        ),
    )
    return RelationshipResidualFitInput(
        row_id=row_id,
        subject_scope=subject_scope,
        public_action_text=_ACTION_TEXT,
        forecast=forecast,
        gate_decision=gate,
    )


def _split(prefix: str) -> tuple[RelationshipResidualFitInput, ...]:
    return tuple(
        _input(
            row_id=f"{prefix}-{action_index}-{repeat}",
            subject_scope=f"{prefix}-subject-{action_index}-{repeat}",
            recommended_action_id=action_id,
        )
        for action_index, action_id in enumerate(_DIRECTIONAL)
        for repeat in range(2)
    )


def test_protocol_freezes_relationship_actions_and_honest_claim_boundary() -> None:
    protocol = load_relationship_residual_fit_protocol()

    assert protocol.action_ids == _DIRECTIONAL
    assert protocol.strict_noop_action_id == RelationshipAction.NEUTRAL_NOOP.value
    assert protocol.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert "would not by itself prove raw strict-JSON generation" in (
        protocol.claim_boundary
    )


def test_corpus_uses_only_pre_action_owner_and_gate_lineage() -> None:
    corpus = build_relationship_residual_fit_corpus(
        train_inputs=_split("train"),
        heldout_inputs=_split("heldout"),
    )

    assert corpus.action_ids == _DIRECTIONAL
    assert {row.subject_scope for row in corpus.train_rows}.isdisjoint(
        row.subject_scope for row in corpus.heldout_rows
    )
    for row in (*corpus.train_rows, *corpus.heldout_rows):
        assert row.action_text.count(_DIRECTIONAL[0]) == 1
        assert row.action_text.count(_DIRECTIONAL[1]) == 1
        assert RelationshipAction.NEUTRAL_NOOP.value not in row.action_text
        assert row.target_action_id in row.condition_text
        assert '"evaluation_present":false' in row.condition_text
        assert '"observed_or_future_outcome_present":false' in row.condition_text
    assert NamedActionSteeringCorpus.from_payload(corpus.to_payload()) == corpus
    assert NamedActionSteeringCorpus.from_payload(corpus.to_payload()).corpus_id == (
        corpus.corpus_id
    )


def test_oracle_or_untrained_gate_cannot_supply_fit_rows() -> None:
    item = _split("train")[0]
    oracle = replace(
        item.gate_decision,
        mode=RelationshipActionGateMode.ORACLE,
        evaluator_only=True,
    )
    with pytest.raises(ValueError, match="oracle/evaluator"):
        build_relationship_residual_fit_corpus(
            train_inputs=(replace(item, gate_decision=oracle), *_split("train")[1:]),
            heldout_inputs=_split("heldout"),
        )

    untrained = replace(item.gate_decision, update_count=0)
    with pytest.raises(ValueError, match="PE-trained"):
        build_relationship_residual_fit_corpus(
            train_inputs=(
                replace(item, gate_decision=untrained),
                *_split("train")[1:],
            ),
            heldout_inputs=_split("heldout"),
        )


def test_public_action_text_rejects_evaluator_field_leakage() -> None:
    item = _split("train")[0]
    leaked = replace(
        item,
        public_action_text=f"{item.public_action_text}\npreferred_action_id=hidden",
    )
    with pytest.raises(ValueError, match="forbidden fields"):
        build_relationship_residual_fit_corpus(
            train_inputs=(leaked, *_split("train")[1:]),
            heldout_inputs=_split("heldout"),
        )


def test_train_and_heldout_subject_scopes_must_be_disjoint() -> None:
    train = _split("train")
    heldout = _split("heldout")
    overlapping = replace(heldout[0], subject_scope=train[0].subject_scope)

    with pytest.raises(ValueError, match="subject scopes must be disjoint"):
        build_relationship_residual_fit_corpus(
            train_inputs=train,
            heldout_inputs=(overlapping, *heldout[1:]),
        )
