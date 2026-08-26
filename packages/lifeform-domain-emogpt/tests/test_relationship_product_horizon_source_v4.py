from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace

import pytest

from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 import (
    RELATIONSHIP_PRODUCT_HORIZON_SOURCE_SCHEMA_VERSION,
    build_relationship_product_horizon_environment,
    build_relationship_product_horizon_evaluator_bundle,
    build_relationship_product_horizon_phase_specs,
    build_relationship_product_horizon_public_view,
    load_relationship_product_horizon_source_protocol,
    relationship_product_horizon_source_protocol_path,
)
from lifeform_domain_emogpt.lab.relationship_product_pilot_source_v2 import (
    build_relationship_product_pilot_evaluator_bundle,
    build_relationship_product_pilot_public_view,
    load_relationship_product_pilot_source_protocol,
    relationship_product_pilot_source_protocol_path,
)
from lifeform_domain_emogpt.relationship_action_contracts import RELATIONSHIP_ACTIONS


def test_source_v4_freezes_112_distinct_long_horizon_roots() -> None:
    protocol = load_relationship_product_horizon_source_protocol()
    public = build_relationship_product_horizon_public_view(protocol)
    evaluator = build_relationship_product_horizon_evaluator_bundle(protocol)

    assert protocol.schema_version == RELATIONSHIP_PRODUCT_HORIZON_SOURCE_SCHEMA_VERSION
    assert protocol.protocol_id == "dbf0526299558842b52f293875520e4524afff0e5a1636ab1fd10da9f74d1d91"
    assert public.public_plan_sha256 == "f46336a95aeac2c7be60616388a31333fb45e46cd18320dae2f9bd25179a86d6"
    assert evaluator.sealed_bundle_sha256 == "51900ec798a8afdbcdab7547b1ad2c7c22fc098122900ec96050a5931c308936"
    assert protocol.root_count == 112
    assert protocol.decision_sessions_per_root == 48
    assert len(public.roots) == 112
    assert len(evaluator.root_manifests) == 112
    assert len(evaluator.onboarding_sessions) == 448
    assert len(evaluator.decision_sessions) == 5_376
    assert len({item.root_seed for item in evaluator.root_manifests}) == 112
    assert len({item.tape_seed for item in evaluator.root_manifests}) == 112
    assert len({item.world_clone_id for item in evaluator.root_manifests}) == 112
    assert len({item.public_trajectory_sha256 for item in evaluator.root_manifests}) == 112
    assert len({item.causal_tape_signature for item in evaluator.root_manifests}) == 112
    assert len({item.environment_seed for item in evaluator.decision_sessions}) == 5_376
    assert len({item.session_id for item in evaluator.decision_sessions}) == 5_376
    assert len({item.decision_id for item in evaluator.decision_sessions}) == 5_376
    assert len({item.scene_id for item in evaluator.decision_sessions}) == 5_376
    assert min(item.public_source_characters for item in public.roots) >= 12_000
    assert min(item.public_source_utf8_bytes for item in public.roots) >= 30_000


def test_source_v4_schedule_has_balanced_segments_and_disjoint_collection() -> None:
    protocol = load_relationship_product_horizon_source_protocol()
    public = build_relationship_product_horizon_public_view(protocol)
    evaluator = build_relationship_product_horizon_evaluator_bundle(protocol)
    manifests_by_subject = {item.subject_id: item for item in evaluator.root_manifests}
    public_by_subject = {item.subject_id: item for item in public.roots}

    assert len(manifests_by_subject) == 112
    collection_policy_counts: Counter[str] = Counter()
    for root_index, manifest in enumerate(evaluator.root_manifests):
        sessions = evaluator.sessions_for(manifest.subject_id)
        assert [item.decision_index for item in sessions] == list(range(48))
        assert all(item.policy_mode == "complement" for item in sessions[:8])
        assert all(item.policy_mode == "complement" for item in sessions[8:])
        onboarding = tuple(
            item for item in evaluator.onboarding_sessions if item.subject_id == manifest.subject_id
        )
        assert [item.virtual_day for item in onboarding] == [0, 1, 2, 3]
        assert sessions[0].virtual_day == 4
        assert sessions[8].virtual_day - sessions[7].virtual_day - 1 >= 14
        assert sessions[32].virtual_day - sessions[31].virtual_day - 1 >= 14
        assert [item.virtual_day for item in public_by_subject[manifest.subject_id].decision_sessions] == [
            item.virtual_day for item in sessions
        ]
        assert {item.policy_id for item in onboarding} != {sessions[0].policy_id}
        collection_policy_counts[sessions[0].policy_id] += 1
        phases = build_relationship_product_horizon_phase_specs(protocol, root_index=root_index)
        assert phases[8].virtual_day == sessions[8].virtual_day
        assert phases[32].virtual_day == sessions[32].virtual_day
        for segment_id in (
            "matched_collection",
            "post_reversal",
            "correction",
            "post_correction",
            "return_after_gap",
            "mixed_stress",
        ):
            segment = tuple(item for item in sessions if item.segment_id == segment_id)
            assert len(segment) == 8
            assert Counter(item.condition_id for item in segment) == {
                "connection_under_exclusion": 4,
                "agency_under_override": 4,
            }
        for item in sessions[16:24]:
            assert item.correction_target_index is not None
            assert item.correction_target_index < 8
            target = sessions[item.correction_target_index]
            assert target.decision_index < item.decision_index
            assert target.condition_id == item.condition_id
        for condition_id in ("connection_under_exclusion", "agency_under_override"):
            collection_targets = {
                item.decision_index for item in sessions[:8] if item.condition_id == condition_id
            }
            correction_targets = {
                item.correction_target_index
                for item in sessions[16:24]
                if item.condition_id == condition_id
            }
            assert correction_targets == collection_targets
    assert collection_policy_counts == {"alpha": 56, "beta": 56}


def test_source_v4_tape_and_rendering_are_bound_to_protocol_identity() -> None:
    protocol = load_relationship_product_horizon_source_protocol()
    revised = replace(protocol, protocol_id="0" * 64)
    original_phases = build_relationship_product_horizon_phase_specs(protocol, root_index=0)
    revised_phases = build_relationship_product_horizon_phase_specs(revised, root_index=0)
    original_public = build_relationship_product_horizon_public_view(protocol).roots[0]
    revised_public = build_relationship_product_horizon_public_view(revised).roots[0]
    original_evaluator = build_relationship_product_horizon_evaluator_bundle(protocol).root_manifests[0]
    revised_evaluator = build_relationship_product_horizon_evaluator_bundle(revised).root_manifests[0]

    assert original_phases != revised_phases
    assert original_public.public_trajectory_sha256 != revised_public.public_trajectory_sha256
    assert original_evaluator.tape_seed != revised_evaluator.tape_seed
    assert original_evaluator.causal_tape_signature != revised_evaluator.causal_tape_signature


def test_source_v4_replays_all_16128_action_conditioned_branches() -> None:
    evaluator = build_relationship_product_horizon_evaluator_bundle()
    evidence_refs: set[str] = set()
    branch_count = 0
    for manifest in evaluator.root_manifests:
        environment = build_relationship_product_horizon_environment(
            evaluator,
            subject_id=manifest.subject_id,
        )
        for session in evaluator.sessions_for(manifest.subject_id):
            for action in RELATIONSHIP_ACTIONS:
                distribution = environment.distribution_for(scene_id=session.scene_id, action=action)
                assert distribution.action_id is action
                assert sum(item.probability for item in distribution.outcomes) == pytest.approx(1.0)
                outcome = environment.settle(
                    scene_id=session.scene_id,
                    decision_id=session.decision_id,
                    action=action,
                    seed=session.environment_seed,
                )
                assert outcome.selected_action is action
                assert outcome.environment_evidence_ref not in evidence_refs
                if session.decision_index == 0:
                    assert outcome == environment.settle(
                        scene_id=session.scene_id,
                        decision_id=session.decision_id,
                        action=action,
                        seed=session.environment_seed,
                    )
                evidence_refs.add(outcome.environment_evidence_ref)
                branch_count += 1
    assert branch_count == 16_128
    assert len(evidence_refs) == 16_128


def test_source_v4_public_payload_is_truth_free_and_text_disjoint_from_v3() -> None:
    public_v4 = build_relationship_product_horizon_public_view()
    payload = public_v4.to_sut_payload()
    forbidden = {
        "causal_tape_signature",
        "condition_id",
        "environment_seed",
        "policy_id",
        "policy_mode",
        "preferred_action_id",
        "root_seed",
        "scene_id",
        "segment_id",
        "surface_recipe_id",
        "tape_seed",
        "world_clone_id",
    }

    public_strings: set[str] = set()

    def assert_truth_free(value: object) -> None:
        if isinstance(value, dict):
            assert not (set(value) & forbidden)
            for child in value.values():
                assert_truth_free(child)
        elif isinstance(value, list):
            for child in value:
                assert_truth_free(child)
        elif isinstance(value, str):
            public_strings.add(value)

    assert_truth_free(payload)
    for sealed_literal in (
        "connection_under_exclusion",
        "agency_under_override",
        "matched_collection",
        "post_reversal",
        "correction",
        "post_correction",
        "return_after_gap",
        "mixed_stress",
        "alpha",
        "beta",
    ):
        assert not any(sealed_literal in value for value in public_strings)
    for meta_prompt_literal in ("模型", "评估器", "隐藏状态", "系统只能"):
        assert not any(meta_prompt_literal in value for value in public_strings)

    source_v3 = load_relationship_product_pilot_source_protocol()
    public_v3 = build_relationship_product_pilot_public_view(source_v3)
    v3_texts = {
        fragment
        for subject in public_v3.subjects
        for session in (*subject.onboarding_sessions, *subject.decision_sessions)
        for fragment in session.public_text_fragments()
    }
    v4_texts = {
        text
        for root in public_v4.roots
        for text in (
            *(
                fragment
                for session in root.onboarding_sessions
                for fragment in (
                    session.public_context_chunk,
                    session.user_utterance,
                    session.rendered_user_reaction,
                )
            ),
            *(
                fragment
                for session in root.decision_sessions
                for fragment in (session.public_context_chunk, session.current_input)
            ),
        )
    }
    assert not (v3_texts & v4_texts)


def test_source_v4_loader_fails_loudly_on_schema_or_firewall_drift(tmp_path) -> None:
    source = load_relationship_product_horizon_source_protocol()
    protocol_path = tmp_path / "source-v4.json"
    canonical_path = relationship_product_horizon_source_protocol_path()
    payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "relationship-product-horizon-source.v3"
    protocol_path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="refuses another schema"):
        load_relationship_product_horizon_source_protocol(protocol_path)

    payload["schema_version"] = source.schema_version
    payload["firewall"]["formal_evidence_authorized"] = True
    protocol_path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="formal_evidence_authorized"):
        load_relationship_product_horizon_source_protocol(protocol_path)

    canonical_bytes = canonical_path.read_bytes()
    protocol_path.write_bytes(canonical_bytes.replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="LF-only"):
        load_relationship_product_horizon_source_protocol(protocol_path)

    protocol_path.write_bytes(canonical_bytes + b"\n")
    with pytest.raises(ValueError, match="ending in one LF"):
        load_relationship_product_horizon_source_protocol(protocol_path)

    duplicate_key = canonical_bytes.replace(
        b'{\n  "schema_version":',
        b'{\n  "schema_version": "duplicate",\n  "schema_version":',
        1,
    )
    protocol_path.write_bytes(duplicate_key)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_relationship_product_horizon_source_protocol(protocol_path)

    collapsed = json.loads(canonical_bytes)
    collapsed["policies"]["profiles"]["alpha"]["agency_under_override"] = (
        "stay_present_without_probe"
    )
    collapsed["policies"]["profiles"]["beta"]["agency_under_override"] = (
        "respect_space_with_return_option"
    )
    protocol_path.write_bytes((json.dumps(collapsed, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="each source-v4 policy must use both"):
        load_relationship_product_horizon_source_protocol(protocol_path)

    inverted_environment = json.loads(canonical_bytes)
    inverted_environment["reactive_environment"]["preferred_action_probabilities"] = [
        0.01,
        0.01,
        0.49,
        0.49,
    ]
    protocol_path.write_bytes(
        (json.dumps(inverted_environment, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    with pytest.raises(ValueError, match="preferred action must dominate"):
        load_relationship_product_horizon_source_protocol(protocol_path)


def test_source_v4_envelopes_reject_mutable_or_cross_root_inventory() -> None:
    public = build_relationship_product_horizon_public_view()
    evaluator = build_relationship_product_horizon_evaluator_bundle()
    with pytest.raises(ValueError, match="immutable tuple"):
        replace(public, roots=list(public.roots))

    foreign = replace(evaluator.decision_sessions[0], subject_id="foreign-subject")
    with pytest.raises(ValueError, match="subject joins drifted"):
        replace(evaluator, decision_sessions=(foreign, *evaluator.decision_sessions[1:]))

    duplicated_tape = replace(
        evaluator.root_manifests[1],
        causal_tape_signature=evaluator.root_manifests[0].causal_tape_signature,
    )
    with pytest.raises(ValueError, match="causal_tape_signature values must be unique"):
        replace(
            evaluator,
            root_manifests=(evaluator.root_manifests[0], duplicated_tape, *evaluator.root_manifests[2:]),
        )

    with pytest.raises(ValueError, match="canonical root/decision order"):
        replace(
            evaluator,
            decision_sessions=(
                evaluator.decision_sessions[1],
                evaluator.decision_sessions[0],
                *evaluator.decision_sessions[2:],
            ),
        )

    first_root = public.roots[0]
    with pytest.raises(ValueError, match="public onboarding rows must be in canonical order"):
        replace(
            first_root,
            onboarding_sessions=(
                first_root.onboarding_sessions[1],
                first_root.onboarding_sessions[0],
                *first_root.onboarding_sessions[2:],
            ),
        )

    with pytest.raises(ValueError, match="segment schedule drifted"):
        replace(evaluator.decision_sessions[0], segment_id="correction")


def test_source_v4_preserves_live_source_v3_lineage() -> None:
    source_v4_path = relationship_product_horizon_source_protocol_path()
    path = relationship_product_pilot_source_protocol_path()
    protocol = load_relationship_product_pilot_source_protocol()
    public = build_relationship_product_pilot_public_view(protocol)
    evaluator = build_relationship_product_pilot_evaluator_bundle(protocol)

    assert hashlib.sha256(source_v4_path.read_bytes()).hexdigest() == (
        "29162022c011b311369816f74087b16c9b262dc50a7a7434227d150b3b2e8bd3"
    )
    assert len(source_v4_path.read_bytes()) == 4_977
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "09b0fe4adad95a23dda06570e6720381ebac46f1ebacfb13b57924671f45b22f"
    )
    assert len(path.read_bytes()) == 1_410
    assert protocol.protocol_sha256 == "d17a49edc5dc549a325648f9c430340d0d3e7cabe634ba339506bd7e56b24be8"
    assert public.public_plan_sha256 == "4f5e8c1508b533d9d64434c5e3ba6a8e9b95814b070fec21ff2d03841746bb05"
    assert evaluator.sealed_bundle_sha256 == "7ddec3e160967381a51e6ffdf2362924619249e8b43ffc0a531aaf2015cdc18d"
