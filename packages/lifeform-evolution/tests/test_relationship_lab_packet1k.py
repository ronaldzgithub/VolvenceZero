from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from lifeform_domain_emogpt.lab import (
    RelationshipAction,
    load_relationship_consumer_training_view,
)
from lifeform_evolution.relationship_lab_baseline import StatelessActionCompletion
from lifeform_evolution.relationship_lab_contexts import (
    RELATIONSHIP_P1_ARMS,
    PersistedRelationshipP1StateDigest,
    RelationshipP1Context,
    RelationshipP1ContextBundle,
)
from lifeform_evolution.relationship_lab_packet1i import (
    load_relationship_p1i_frozen_consumer_protocol,
)
from lifeform_evolution.relationship_lab_packet1k import (
    RELATIONSHIP_P1K_CONTEXT_ARM,
    RELATIONSHIP_P1K_PREPARED_NEXT_ACTION,
    RELATIONSHIP_P1K_TIERS,
    RelationshipP1kOracleTier,
    RelationshipP1kVerdict,
    assess_relationship_p1k_diagnostic,
    build_relationship_p1k_checkpoint,
    build_relationship_p1k_disclosure,
    execute_relationship_p1k_diagnostic,
    freeze_relationship_p1k_protocol,
    load_relationship_p1k_progress,
    load_relationship_p1k_report,
    persist_relationship_p1k_decision,
    persist_relationship_p1k_readout,
    render_relationship_p1k_request,
    validate_relationship_p1k_progress,
    validate_relationship_p1k_protocol_lineage,
    write_relationship_p1k_checkpoint,
    write_relationship_p1k_report,
)


_FROZEN_AT = "2026-08-21T08:00:00+00:00"
_CREATED_AT = "2026-08-21T09:00:00+00:00"


def _consumer():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    return load_relationship_p1i_frozen_consumer_protocol(
        repo_root
        / "artifacts"
        / "relationship_lab"
        / "qwen25_3b_packet1i_v3_training_replay_20260820"
        / "frozen_consumer_protocol.json"
    )


def _fake_contexts(consumer, dataset) -> RelationshipP1ContextBundle:
    contexts = tuple(
        RelationshipP1Context(
            arm=arm,
            scene_id=observation.scene_id,
            background_depth=depth,
            context_text=(
                ""
                if arm.value == "stateless"
                else "\n".join(
                    (
                        "[public relationship outcome evidence]",
                        f"scene_id: {observation.scene_id}",
                        f"arm: {arm.value}",
                        f"depth: {depth}",
                    )
                )
            ),
            source_evidence_refs=(
                ()
                if arm.value == "stateless"
                else tuple(item.event_id for item in observation.histories)
            ),
        )
        for observation in dataset.observations
        for depth in consumer.background_depths
        for arm in RELATIONSHIP_P1_ARMS
    )
    digest = hashlib.sha256(b"p1k-fake-state").hexdigest()
    return RelationshipP1ContextBundle(
        dataset_fingerprint=dataset.dataset_fingerprint,
        background_depths=consumer.background_depths,
        background_templates_sha256=consumer.background_templates_sha256,
        rag_config_sha256=consumer.rag_config_sha256,
        contexts=contexts,
        persisted_state=PersistedRelationshipP1StateDigest(
            structured_scope_digests=(("fake-v3", digest, 4),),
            rag_scope_digests=(("fake-v3", digest, 4),),
        ),
    )


class _FakeOraclePolicy:
    def __init__(self, consumer, dataset, *, mode: str) -> None:
        self.model_id = consumer.model_id
        self.weights_sha256 = consumer.expected_weights_sha256
        self.generation_config_sha256 = consumer.expected_generation_config_sha256
        self.prompt_sha256 = hashlib.sha256(b"unused").hexdigest()
        self._expected = {
            observation.scene_id: dynamic.preferred_action
            for observation, dynamic in (
                (observation, dataset.dynamic_for_scene(observation.scene_id))
                for observation in dataset.observations
            )
        }
        self._mode = mode

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        raise AssertionError("P1k must not call the stateless policy surface")

    def choose_from_messages(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        seed: int,
    ) -> StatelessActionCompletion:
        request = messages[1]["content"]
        scene_id = next(scene for scene in self._expected if scene in request)
        disclose_policy = "该用户在每种抽象关系条件下需要的关系动作" in request
        disclose_binding = "每条公开历史所处的抽象关系条件" in request
        if self._mode == "always_stay":
            action = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
        elif self._mode == "policy_only" and disclose_policy:
            action = self._expected[scene_id]
        elif self._mode == "through_binding" and (disclose_policy or disclose_binding):
            action = self._expected[scene_id]
        elif self._mode == "all_correct":
            action = self._expected[scene_id]
        else:
            action = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
        scores = (
            {
                "stay_present_without_probe_score": 1,
                "respect_space_with_return_option_score": -1,
            }
            if action is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
            else {
                "stay_present_without_probe_score": -1,
                "respect_space_with_return_option_score": 1,
            }
        )
        return StatelessActionCompletion(
            raw_output=json.dumps(scores),
            chosen_action_id=None,
            prompt_tokens=80,
            completion_tokens=12,
        )

    def count_tokens(self, text: str) -> int:
        return len(text)


def _frozen_fixture():
    consumer = _consumer()
    dataset = load_relationship_consumer_training_view().training_dataset
    contexts = _fake_contexts(consumer, dataset)
    protocol = freeze_relationship_p1k_protocol(
        consumer=consumer,
        dataset=dataset,
        contexts=contexts,
        seed_schedule=consumer.seed_schedule,
        frozen_at_iso=_FROZEN_AT,
    )
    return consumer, dataset, contexts, protocol


def test_p1k_disclosure_ladder_does_not_emit_sealed_ids() -> None:
    dataset = load_relationship_consumer_training_view().training_dataset
    scene_id = dataset.observations[0].scene_id
    dynamic = dataset.dynamic_for_scene(scene_id)
    for tier in RELATIONSHIP_P1K_TIERS:
        text = build_relationship_p1k_disclosure(
            dataset=dataset,
            scene_id=scene_id,
            tier=tier,
        )
        assert dynamic.probe_condition_id not in text
        assert dynamic.policy_id not in text
        assert "latent_condition_" not in text
        assert "latent_policy_" not in text
        assert "preferred_action" not in text
        assert "条件甲" in text and "条件乙" in text
        request = render_relationship_p1k_request(
            context_text="public history",
            disclosure_text=text,
            current_input="current",
        )
        assert "<sealed_concept_disclosure>" in request
    policy_text = build_relationship_p1k_disclosure(
        dataset=dataset,
        scene_id=scene_id,
        tier=RelationshipP1kOracleTier.CONDITION_AND_POLICY,
    )
    assert dynamic.preferred_action.value in policy_text
    binding_text = build_relationship_p1k_disclosure(
        dataset=dataset,
        scene_id=scene_id,
        tier=RelationshipP1kOracleTier.CONDITION_AND_BINDING,
    )
    assert dynamic.preferred_action.value not in binding_text
    assert dataset.observations[0].histories[0].event_id in binding_text


def test_p1k_freezes_non_competitive_protocol_before_qwen_output() -> None:
    consumer, dataset, contexts, protocol = _frozen_fixture()

    assert protocol.next_action == RELATIONSHIP_P1K_PREPARED_NEXT_ACTION
    assert protocol.context_arm == RELATIONSHIP_P1K_CONTEXT_ARM.value
    assert protocol.observation_count == 12
    assert protocol.planned_output_count == 36
    assert protocol.tiers == tuple(tier.value for tier in RELATIONSHIP_P1K_TIERS)
    assert not protocol.to_payload()["experiment_guards"]["competitive"]
    assert not protocol.to_payload()["experiment_guards"]["p2_enabled"]
    validate_relationship_p1k_protocol_lineage(
        protocol,
        consumer=consumer,
        dataset=dataset,
        contexts=contexts,
    )


def _run_complete(tmp_path, *, mode: str):
    consumer, dataset, contexts, protocol = _frozen_fixture()
    checkpoint = build_relationship_p1k_checkpoint(protocol=protocol, dataset=dataset)
    write_relationship_p1k_checkpoint(checkpoint=checkpoint, output_dir=tmp_path)
    policy = _FakeOraclePolicy(consumer, dataset, mode=mode)
    first = execute_relationship_p1k_diagnostic(
        policy,
        protocol=protocol,
        dataset=dataset,
        contexts=contexts,
        existing_progress=load_relationship_p1k_progress(tmp_path),
        max_new_readouts=4,
        readout_observer=lambda index, item: persist_relationship_p1k_readout(
            checkpoint=checkpoint,
            output_dir=tmp_path,
            index=index,
            readout=item,
        ),
        decision_observer=lambda index, item: persist_relationship_p1k_decision(
            checkpoint=checkpoint,
            output_dir=tmp_path,
            index=index,
            decision=item,
        ),
    )
    assert first.new_outputs == 4
    first_readout = tmp_path / "records" / "0000.readout.json"
    first_hash = hashlib.sha256(first_readout.read_bytes()).hexdigest()
    execute_relationship_p1k_diagnostic(
        policy,
        protocol=protocol,
        dataset=dataset,
        contexts=contexts,
        existing_progress=load_relationship_p1k_progress(tmp_path),
        readout_observer=lambda index, item: persist_relationship_p1k_readout(
            checkpoint=checkpoint,
            output_dir=tmp_path,
            index=index,
            readout=item,
        ),
        decision_observer=lambda index, item: persist_relationship_p1k_decision(
            checkpoint=checkpoint,
            output_dir=tmp_path,
            index=index,
            decision=item,
        ),
    )
    complete = load_relationship_p1k_progress(tmp_path)
    validate_relationship_p1k_progress(
        complete,
        protocol=protocol,
        dataset=dataset,
        contexts=contexts,
    )
    assert complete.is_complete
    assert hashlib.sha256(first_readout.read_bytes()).hexdigest() == first_hash
    report = assess_relationship_p1k_diagnostic(
        protocol=protocol,
        progress=complete,
        created_at_iso=_CREATED_AT,
    )
    return protocol, complete, report


def test_p1k_checkpoint_resume_and_policy_floor(tmp_path) -> None:
    _protocol, _complete, report = _run_complete(tmp_path, mode="always_stay")
    assert report.verdict is RelationshipP1kVerdict.SUBSTRATE_APPLICATION_FLOOR
    assert report.next_action == "stop_scenario_lane_change_substrate_or_readout_floor"
    assert not report.to_payload()["experiment_guards"]["p2_enabled"]
    report_path, _markdown = write_relationship_p1k_report(
        report=report,
        output_dir=tmp_path,
    )
    assert load_relationship_p1k_report(report_path) == report


def test_p1k_localizes_policy_induction_when_only_full_disclosure_works(
    tmp_path,
) -> None:
    _protocol, _complete, report = _run_complete(tmp_path, mode="policy_only")
    assert report.verdict is RelationshipP1kVerdict.POLICY_INDUCTION_BOTTLENECK
    by_tier = {item.tier: item for item in report.tier_metrics}
    assert by_tier[RelationshipP1kOracleTier.CONDITION_AND_POLICY.value].functional
    assert not by_tier[RelationshipP1kOracleTier.CONDITION_AND_BINDING.value].functional


def test_p1k_localizes_unaided_gap_when_every_oracle_rung_works(tmp_path) -> None:
    _protocol, _complete, report = _run_complete(tmp_path, mode="all_correct")
    assert report.verdict is RelationshipP1kVerdict.UNAIDED_INDUCTION_BOTTLENECK
    assert all(item.functional for item in report.tier_metrics)
    assert report.output_count == 36


def test_p1k_rejects_policy_drift(tmp_path) -> None:
    consumer, dataset, contexts, protocol = _frozen_fixture()
    checkpoint = build_relationship_p1k_checkpoint(protocol=protocol, dataset=dataset)
    write_relationship_p1k_checkpoint(checkpoint=checkpoint, output_dir=tmp_path)
    policy = _FakeOraclePolicy(consumer, dataset, mode="all_correct")
    policy.weights_sha256 = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(ValueError, match="substrate"):
        execute_relationship_p1k_diagnostic(
            policy,
            protocol=protocol,
            dataset=dataset,
            contexts=contexts,
            existing_progress=load_relationship_p1k_progress(tmp_path),
        )
