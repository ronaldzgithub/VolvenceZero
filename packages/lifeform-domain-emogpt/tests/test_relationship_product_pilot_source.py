from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

import lifeform_domain_emogpt.lab.relationship_product_pilot_source as legacy_source_owner
from lifeform_domain_emogpt.lab.environment import ReactiveRelationshipEnvironment
from lifeform_domain_emogpt.lab.relationship_product_pilot_source import (
    RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION as RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION_V1,
    RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION as RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V1,
    build_relationship_product_pilot_environment,
    build_relationship_product_pilot_evaluator_bundle,
    build_relationship_product_pilot_public_view,
    load_relationship_product_pilot_source_protocol,
    relationship_product_pilot_source_protocol_path,
)
from lifeform_domain_emogpt.lab.relationship_product_pilot_source_v2 import (
    RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION as RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION_V2,
    RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2,
    RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3,
    RELATIONSHIP_PRODUCT_PILOT_SOURCE_V2_REGISTRY,
    archived_relationship_product_pilot_source_v2_protocol_path,
    build_relationship_product_pilot_evaluator_bundle as build_independent_product_pilot_evaluator_bundle,
    build_relationship_product_pilot_public_view as build_independent_product_pilot_public_view,
    load_archived_relationship_product_pilot_source_v2_protocol,
    load_relationship_product_pilot_source_protocol as load_independent_product_pilot_source_protocol,
    relationship_product_pilot_source_protocol_path as independent_product_pilot_source_protocol_path,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
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


def _semantic_surfaces_by_condition(protocol: object, public: object, subject_index: int) -> dict[str, list[str]]:
    condition_ids = protocol.condition_ids
    subject = public.subjects[subject_index]
    surfaces = {condition_id: [] for condition_id in condition_ids}
    onboarding_conditions = (condition_ids[0], condition_ids[0], condition_ids[1], condition_ids[1])
    for session, condition_id in zip(subject.onboarding_sessions, onboarding_conditions, strict=True):
        surfaces[condition_id].append(session.user_utterance)
    for session, phase in zip(subject.decision_sessions, protocol.phase_specs, strict=True):
        surfaces[phase.condition_id].append(session.current_input)
    return surfaces


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


def test_v1_golden_source_and_materialized_outputs_remain_byte_exact() -> None:
    path = relationship_product_pilot_source_protocol_path()
    protocol = load_relationship_product_pilot_source_protocol()
    public = build_relationship_product_pilot_public_view(protocol)
    evaluator = build_relationship_product_pilot_evaluator_bundle(protocol)

    assert path.name == "relationship_product_pilot_source_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "c623c33f7f8cdbb31a0e5055cb4b802af479ae2a7aa6cb73c9724c614e8539b0"
    )
    assert protocol.schema_version == RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V1
    raw_protocol = json.loads(path.read_text(encoding="utf-8"))
    assert raw_protocol["context_pressure"]["rendering_version"] == (
        RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION_V1
    )
    assert protocol.protocol_sha256 == "048b73d4a412b4444fb469be0d9daa6d2a26e9920c743804da8f36dc331691ae"
    assert public.public_plan_sha256 == "93474269cb5b9d066e68253d6f2e51fbc0d3bf3b6a7fe2a748b140d136bb812b"
    assert evaluator.sealed_bundle_sha256 == "d502b78364dcb7024b229f4bb10c0cddb002488c3a360edd7aa0932c345d8b5a"
    legacy_owner_path = pathlib.Path(legacy_source_owner.__file__).resolve()
    assert hashlib.sha256(legacy_owner_path.read_bytes()).hexdigest() == (
        "15162708b3e23071830d0413f1ff7a2a75512f507e37f6e30a47570a593972ae"
    )


def test_independent_source_registry_materialization_is_explicit_and_deterministic() -> None:
    assert tuple(RELATIONSHIP_PRODUCT_PILOT_SOURCE_V2_REGISTRY) == (
        RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3,
    )
    path = independent_product_pilot_source_protocol_path(
        RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3
    )
    first = load_independent_product_pilot_source_protocol(path)
    second = load_independent_product_pilot_source_protocol()
    first_public = build_independent_product_pilot_public_view(first)
    second_public = build_independent_product_pilot_public_view(second)
    first_evaluator = build_independent_product_pilot_evaluator_bundle(first)
    second_evaluator = build_independent_product_pilot_evaluator_bundle(second)

    assert path.name == "relationship_product_pilot_source_v3.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "09b0fe4adad95a23dda06570e6720381ebac46f1ebacfb13b57924671f45b22f"
    )
    assert first == second
    assert first.schema_version == RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3
    assert first.rendering_version == RELATIONSHIP_PRODUCT_PILOT_PUBLIC_RENDERING_VERSION_V2
    assert first.independent_semantic_surfaces_per_condition == 14
    assert first.model_output_count == 0
    assert first.formal_evidence_authorized is False
    assert first.protocol_sha256 == "d17a49edc5dc549a325648f9c430340d0d3e7cabe634ba339506bd7e56b24be8"
    assert first_public.to_sut_payload() == second_public.to_sut_payload()
    assert first_public.public_plan_sha256 == "4f5e8c1508b533d9d64434c5e3ba6a8e9b95814b070fec21ff2d03841746bb05"
    assert first_evaluator.sealed_bundle_sha256 == second_evaluator.sealed_bundle_sha256
    assert first_evaluator.sealed_bundle_sha256 == (
        "7ddec3e160967381a51e6ffdf2362924619249e8b43ffc0a531aaf2015cdc18d"
    )
    assert len(first_public.subjects) == 8
    assert all(len(subject.onboarding_sessions) == 4 for subject in first_public.subjects)
    assert all(len(subject.decision_sessions) == 24 for subject in first_public.subjects)
    assert len(first_evaluator.onboarding_sessions) == 32
    assert len(first_evaluator.decision_sessions) == 192


def test_independent_cohort_schedule_seeds_domains_and_identities_are_disjoint_from_v1() -> None:
    v1 = load_relationship_product_pilot_source_protocol()
    v2 = load_independent_product_pilot_source_protocol()
    v1_public = build_relationship_product_pilot_public_view(v1)
    v2_public = build_independent_product_pilot_public_view(v2)
    v1_evaluator = build_relationship_product_pilot_evaluator_bundle(v1)
    v2_evaluator = build_independent_product_pilot_evaluator_bundle(v2)

    assert v1.cohort_id != v2.cohort_id
    assert v2.identity_namespace == "relationship-product-pilot-independent-v2"
    assert set(v1.subject_seeds).isdisjoint(v2.subject_seeds)
    assert v1.environment_seed_namespace != v2.environment_seed_namespace
    assert set(v1.domain_ids).isdisjoint(v2.domain_ids)
    assert {phase.phase_id for phase in v1.phase_specs}.isdisjoint(phase.phase_id for phase in v2.phase_specs)
    assert all(phase.phase_id.startswith("independent_v2_") for phase in v2.phase_specs)
    condition_counts = {
        condition: sum(phase.condition_id == condition for phase in v2.phase_specs) for condition in v2.condition_ids
    }
    assert condition_counts == {condition: 12 for condition in v2.condition_ids}
    assert tuple(v2.base_policy_id(index) for index in range(8)).count("alpha") == 4
    assert tuple(v2.base_policy_id(index) for index in range(8)).count("beta") == 4
    assert all(phase.active_policy_mode == "base" for phase in v2.phase_specs[:12])
    assert all(phase.active_policy_mode == "complement" for phase in v2.phase_specs[12:])
    corrections = [phase for phase in v2.phase_specs if phase.stage_id == "correction"]
    assert {phase.condition_id for phase in corrections} == set(v2.condition_ids)
    assert v2.phase_specs[12].virtual_day - v2.phase_specs[11].virtual_day >= 14
    assert v2.phase_specs[20].virtual_day - v2.phase_specs[19].virtual_day >= 14

    def public_ids(public: object) -> set[str]:
        return {
            value
            for subject in public.subjects
            for value in (
                *(session.session_id for session in subject.onboarding_sessions),
                *(session.event_id for session in subject.onboarding_sessions),
                *(session.session_id for session in subject.decision_sessions),
                *(session.decision_id for session in subject.decision_sessions),
            )
        }

    assert public_ids(v1_public).isdisjoint(public_ids(v2_public))
    assert {item.subject_id for item in v1_evaluator.decision_sessions}.isdisjoint(
        item.subject_id for item in v2_evaluator.decision_sessions
    )
    assert {item.scene_id for item in v1_evaluator.decision_sessions}.isdisjoint(
        item.scene_id for item in v2_evaluator.decision_sessions
    )
    assert {item.environment_seed for item in v1_evaluator.decision_sessions}.isdisjoint(
        item.environment_seed for item in v2_evaluator.decision_sessions
    )


def test_independent_semantic_surfaces_are_unique_balanced_and_exactly_disjoint_from_v1() -> None:
    v1 = load_relationship_product_pilot_source_protocol()
    v2 = load_independent_product_pilot_source_protocol()
    v1_public = build_relationship_product_pilot_public_view(v1)
    v2_public = build_independent_product_pilot_public_view(v2)
    all_v1_surfaces: set[str] = set()
    all_v2_surfaces: set[str] = set()
    all_v1_public_fragments: set[str] = set()
    all_v2_public_fragments: set[str] = set()

    for subject_index in range(8):
        v1_surfaces = _semantic_surfaces_by_condition(v1, v1_public, subject_index)
        v2_surfaces = _semantic_surfaces_by_condition(v2, v2_public, subject_index)
        all_v1_surfaces.update(surface for values in v1_surfaces.values() for surface in values)
        subject_v2_surfaces = [surface for values in v2_surfaces.values() for surface in values]
        all_v2_surfaces.update(subject_v2_surfaces)
        for session in (
            *v1_public.subjects[subject_index].onboarding_sessions,
            *v1_public.subjects[subject_index].decision_sessions,
        ):
            all_v1_public_fragments.update(session.public_text_fragments())
        for session in (
            *v2_public.subjects[subject_index].onboarding_sessions,
            *v2_public.subjects[subject_index].decision_sessions,
        ):
            all_v2_public_fragments.update(session.public_text_fragments())
        assert set(v2_surfaces) == set(v2.condition_ids)
        assert all(len(values) == 14 for values in v2_surfaces.values())
        assert all(len(set(values)) == 14 for values in v2_surfaces.values())
        assert len(subject_v2_surfaces) == 28
        assert len(set(subject_v2_surfaces)) == 28

    assert len(all_v2_surfaces) == 8 * 28
    assert all_v1_surfaces.isdisjoint(all_v2_surfaces)
    assert all_v1_public_fragments.isdisjoint(all_v2_public_fragments)


def test_independent_semantic_surfaces_preserve_truth_firewall_without_action_directives() -> None:
    protocol = load_independent_product_pilot_source_protocol()
    public = build_independent_product_pilot_public_view(protocol)
    payload = public.to_sut_payload()
    assert not (_all_keys(payload) & _FORBIDDEN_PUBLIC_KEYS)

    sealed_literals = {
        *protocol.condition_ids,
        "alpha",
        "beta",
        *(action.value for action in RELATIONSHIP_ACTIONS),
        *(outcome.value for outcome in RELATIONSHIP_OUTCOMES),
    }
    direct_action_hints = {
        "请陪着我",
        "请给我空间",
        "请退开",
        "请主动靠近",
        "请替我决定",
        "选择陪伴",
        "选择退开",
    }
    for subject_index in range(8):
        surfaces = _semantic_surfaces_by_condition(protocol, public, subject_index)
        for surface in (surface for values in surfaces.values() for surface in values):
            assert all(literal not in surface for literal in sealed_literals)
            assert all(hint not in surface for hint in direct_action_hints)


def test_independent_owner_routing_and_revision_binding_fail_closed(tmp_path) -> None:
    v3_path = independent_product_pilot_source_protocol_path()
    archived_v2_path = archived_relationship_product_pilot_source_v2_protocol_path()
    with pytest.raises(ValueError, match="not owned"):
        independent_product_pilot_source_protocol_path(
            RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2
        )
    with pytest.raises(ValueError, match="only routes source-v3"):
        load_independent_product_pilot_source_protocol(archived_v2_path)

    raw = json.loads(v3_path.read_text(encoding="utf-8"))
    raw["owner"]["module"] = "lifeform_domain_emogpt.lab.relationship_product_pilot_source"
    wrong_owner = tmp_path / "wrong-owner.json"
    wrong_owner.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="owner binding drifted"):
        load_independent_product_pilot_source_protocol(wrong_owner)

    raw = json.loads(v3_path.read_text(encoding="utf-8"))
    raw["base_source"]["raw_sha256"] = "0" * 64
    wrong_base = tmp_path / "wrong-base.json"
    wrong_base.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="base-source binding drifted"):
        load_independent_product_pilot_source_protocol(wrong_base)


def test_archived_source_v2_raw_and_materialization_remain_replayable() -> None:
    path = archived_relationship_product_pilot_source_v2_protocol_path()
    protocol = load_archived_relationship_product_pilot_source_v2_protocol()
    public = build_independent_product_pilot_public_view(protocol)
    evaluator = build_independent_product_pilot_evaluator_bundle(protocol)

    assert path.name == "relationship_product_pilot_source_v2.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "ef35ba2637e53c96c2ed86b16a8bb69281cc10e1a2f30c81112a66841f4b23f7"
    )
    assert protocol.schema_version == RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2
    assert protocol.protocol_sha256 == "9f4ad004f9332a705d3231cb9a3394b4922417878f18487806b0f030bb863161"
    assert public.public_plan_sha256 == "7a99d247ecc7c7d0b25cdaba104d4177eb55b83c3860de44c2e8eef5af9157bf"
    assert evaluator.sealed_bundle_sha256 == "da6cac8fa99476f3ce0cd6b57a0dc3f27b1de417eab8e9a8aa65e777d75a34c6"
