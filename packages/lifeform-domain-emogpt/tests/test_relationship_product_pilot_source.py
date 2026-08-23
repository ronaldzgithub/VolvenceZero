from __future__ import annotations

import json

import pytest

from lifeform_domain_emogpt.lab.environment import ReactiveRelationshipEnvironment
from lifeform_domain_emogpt.lab.relationship_product_pilot_source import (
    build_relationship_product_pilot_environment,
    build_relationship_product_pilot_evaluator_bundle,
    build_relationship_product_pilot_public_view,
    load_relationship_product_pilot_source_protocol,
    relationship_product_pilot_source_protocol_path,
)
from lifeform_domain_emogpt.relationship_action_contracts import RelationshipAction
from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind


_FORBIDDEN_PUBLIC_KEYS = {
    "active_policy_mode",
    "condition_id",
    "dynamic_id",
    "environment_seed",
    "phase_id",
    "policy_id",
    "preferred_action_id",
    "scene_id",
    "stage_id",
    "subject_seed",
}


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def test_protocol_freezes_eight_development_worlds_and_complementary_reversal() -> None:
    protocol = load_relationship_product_pilot_source_protocol()

    assert len(protocol.subject_seeds) == 8
    assert len(set(protocol.subject_seeds)) == 8
    assert protocol.onboarding_sessions_per_subject == 4
    assert protocol.decision_sessions_per_subject == 24
    assert protocol.per_arm_exogenous_world_clone is True
    assert protocol.arm_identity_affects_source_or_environment_seed is False
    assert protocol.p1m_output_dependency is False
    assert protocol.difficulty_tuned_from_p1m is False
    assert protocol.model_output_count == 0
    assert protocol.formal_evidence_authorized is False
    assert protocol.runtime_owner_added is False
    assert protocol.runtime_slot_added is False

    alpha = protocol.policy("alpha")
    beta = protocol.policy("beta")
    for condition_id in protocol.condition_ids:
        assert alpha.action_for(condition_id) is not beta.action_for(condition_id)
        assert {
            alpha.action_for(condition_id),
            beta.action_for(condition_id),
        } == {
            RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
            RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
        }
    assert tuple(protocol.base_policy_id(index) for index in range(8)) == (
        "alpha",
        "beta",
        "alpha",
        "beta",
        "alpha",
        "beta",
        "alpha",
        "beta",
    )
    assert all(item.active_policy_mode == "base" for item in protocol.phase_specs[:12])
    assert all(item.active_policy_mode == "complement" for item in protocol.phase_specs[12:])
    assert protocol.phase_specs[12].virtual_day - protocol.phase_specs[11].virtual_day >= 14
    assert {item.stage_id for item in protocol.phase_specs} >= {
        "domain_switch",
        "post_gap_reversal",
        "reversal",
        "correction",
        "return_after_gap",
    }
    assert {item.domain_id for item in protocol.phase_specs} == set(protocol.domain_ids)
    assert len([item for item in protocol.phase_specs if item.public_correction_target_index is not None]) == 2


def test_public_and_sealed_views_are_type_and_payload_separated_with_context_pressure() -> None:
    protocol = load_relationship_product_pilot_source_protocol()
    public = build_relationship_product_pilot_public_view(protocol)
    evaluator = build_relationship_product_pilot_evaluator_bundle(protocol)

    assert len(public.subjects) == 8
    assert all(len(subject.onboarding_sessions) == 4 for subject in public.subjects)
    assert all(len(subject.decision_sessions) == 24 for subject in public.subjects)
    assert len(evaluator.onboarding_sessions) == 32
    assert len(evaluator.decision_sessions) == 192
    assert evaluator.evaluation_or_judge_feedback_to_learning is False
    assert len(evaluator.sealed_bundle_sha256) == 64

    public_payload = public.to_sut_payload()
    assert not (_all_keys(public_payload) & _FORBIDDEN_PUBLIC_KEYS)
    encoded = json.dumps(public_payload, ensure_ascii=False, sort_keys=True)
    for condition_id in protocol.condition_ids:
        assert condition_id not in encoded
    assert '"alpha"' not in encoded
    assert '"beta"' not in encoded
    assert "relationship-product-pilot-evaluator-bundle" not in encoded

    public_session_ids = {session.session_id for subject in public.subjects for session in subject.decision_sessions}
    sealed_session_ids = {session.session_id for session in evaluator.decision_sessions}
    assert public_session_ids == sealed_session_ids
    for subject in public.subjects:
        assert subject.public_source_character_count >= protocol.minimum_public_source_characters_per_subject
        assert subject.public_source_utf8_byte_count >= protocol.minimum_public_source_utf8_bytes_per_subject
        assert len({item.public_context_chunk for item in subject.decision_sessions}) == 24
        receipt = subject.to_sut_payload()["context_pressure_receipt"]
        assert isinstance(receipt, dict)
        assert receipt["token_measurement_status"] == "not_measured_not_claimed"
        assert receipt["token_count"] is None

    correction_sessions = [
        session
        for subject in public.subjects
        for session in subject.decision_sessions
        if session.public_correction_target_session_id is not None
    ]
    assert len(correction_sessions) == 16
    assert all("纠正" in session.current_input for session in correction_sessions)


def test_each_arm_can_clone_the_same_existing_reactive_environment() -> None:
    protocol = load_relationship_product_pilot_source_protocol()
    public = build_relationship_product_pilot_public_view(protocol)
    evaluator = build_relationship_product_pilot_evaluator_bundle(protocol)
    sealed = evaluator.decision_sessions[0]

    first = build_relationship_product_pilot_environment(evaluator, subject_id=sealed.subject_id)
    second = build_relationship_product_pilot_environment(evaluator, subject_id=sealed.subject_id)
    assert isinstance(first, ReactiveRelationshipEnvironment)
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert public.subjects[0].world_clone_id == sealed.world_clone_id

    preferred = RelationshipAction(sealed.preferred_action_id)
    alternative = (
        RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION
        if preferred is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
        else RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
    )
    preferred_distribution = first.distribution_for(scene_id=sealed.scene_id, action=preferred)
    alternative_distribution = first.distribution_for(scene_id=sealed.scene_id, action=alternative)
    positive = {DialogueExternalOutcomeKind.HELPED, DialogueExternalOutcomeKind.FELT_HEARD}
    preferred_positive = sum(preferred_distribution.probability_of(kind) for kind in positive)
    alternative_positive = sum(alternative_distribution.probability_of(kind) for kind in positive)
    assert preferred_positive == pytest.approx(0.90)
    assert alternative_positive == pytest.approx(0.15)

    first_outcome = first.settle(
        scene_id=sealed.scene_id,
        decision_id=sealed.decision_id,
        action=preferred,
        seed=sealed.environment_seed,
    )
    second_outcome = second.settle(
        scene_id=sealed.scene_id,
        decision_id=sealed.decision_id,
        action=preferred,
        seed=sealed.environment_seed,
    )
    assert first_outcome == second_outcome
    assert first_outcome.environment_evidence_ref == second_outcome.environment_evidence_ref

    matching_condition = sealed.condition_id
    after_reversal = next(
        item
        for item in evaluator.decision_sessions
        if item.subject_id == sealed.subject_id
        and item.decision_index >= 12
        and item.condition_id == matching_condition
    )
    assert after_reversal.policy_id != sealed.policy_id
    assert after_reversal.preferred_action_id != sealed.preferred_action_id
    with pytest.raises(ValueError, match="twenty-four"):
        build_relationship_product_pilot_environment(evaluator, subject_id="missing-subject")


def test_protocol_tampering_cannot_enable_p1m_dependency(tmp_path) -> None:
    raw = json.loads(relationship_product_pilot_source_protocol_path().read_text(encoding="utf-8"))
    raw["owner"]["p1m_output_dependency"] = True
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="P1m"):
        load_relationship_product_pilot_source_protocol(tampered)
