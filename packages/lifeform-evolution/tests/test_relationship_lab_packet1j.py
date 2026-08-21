from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import replace

import pytest

from lifeform_domain_emogpt.lab import (
    RelationshipAction,
    load_relationship_consumer_split_bundle,
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
from lifeform_evolution.relationship_lab_packet1j import (
    RELATIONSHIP_P1J_PREPARED_NEXT_ACTION,
    RelationshipP1jQualificationProtocol,
    RelationshipP1jVerdict,
    assess_relationship_p1j_qualification,
    build_relationship_p1j_checkpoint,
    execute_relationship_p1j_qualification,
    freeze_relationship_p1j_protocol,
    load_relationship_p1j_progress,
    load_relationship_p1j_report,
    persist_relationship_p1j_decision,
    persist_relationship_p1j_readout,
    validate_relationship_p1j_progress,
    validate_relationship_p1j_protocol_lineage,
    validate_relationship_p1j_report_lineage,
    validate_relationship_p1j_terminal_files,
    write_relationship_p1j_checkpoint,
    write_relationship_p1j_report,
)


_FROZEN_AT = "2026-08-21T01:00:00+00:00"
_CREATED_AT = "2026-08-21T02:00:00+00:00"


def _consumer():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    return load_relationship_p1i_frozen_consumer_protocol(
        repo_root
        / "artifacts"
        / "relationship_lab"
        / "qwen25_3b_packet1i_v3_training_replay_20260820"
        / "frozen_consumer_protocol.json"
    )


def _fake_contexts(consumer, split_bundle) -> RelationshipP1ContextBundle:
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
        for observation in split_bundle.qualification_dataset.observations
        for depth in consumer.background_depths
        for arm in RELATIONSHIP_P1_ARMS
    )
    digest = hashlib.sha256(b"p1j-fake-state").hexdigest()
    return RelationshipP1ContextBundle(
        dataset_fingerprint=consumer.qualification_dataset_fingerprint,
        background_depths=consumer.background_depths,
        background_templates_sha256=consumer.background_templates_sha256,
        rag_config_sha256=consumer.rag_config_sha256,
        contexts=contexts,
        persisted_state=PersistedRelationshipP1StateDigest(
            structured_scope_digests=(("fake-v4", digest, 4),),
            rag_scope_digests=(("fake-v4", digest, 4),),
        ),
    )


class _FakeQualificationPolicy:
    def __init__(self, consumer, split_bundle) -> None:
        self.model_id = consumer.model_id
        self.weights_sha256 = consumer.expected_weights_sha256
        self.generation_config_sha256 = consumer.expected_generation_config_sha256
        self.prompt_sha256 = hashlib.sha256(b"unused").hexdigest()
        exact_scenes: set[str] = set()
        expected = {}
        for pair_index, (_pair_id, members) in enumerate(
            split_bundle.qualification_dataset.mirrored_pairs()
        ):
            for observation, dynamic in members:
                expected[observation.scene_id] = dynamic.preferred_action
                if pair_index < 6:
                    exact_scenes.add(observation.scene_id)
        self._expected = expected
        self._exact_scenes = exact_scenes

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        raise AssertionError("P1j must not call the stateless policy surface")

    def choose_from_messages(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        seed: int,
    ) -> StatelessActionCompletion:
        request = messages[1]["content"]
        scene_id = next(scene for scene in self._expected if scene in request)
        action = (
            self._expected[scene_id]
            if scene_id in self._exact_scenes
            else RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
        )
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
            prompt_tokens=100,
            completion_tokens=12,
        )

    def count_tokens(self, text: str) -> int:
        return len(text)


def _frozen_fixture():
    consumer = _consumer()
    split_bundle = load_relationship_consumer_split_bundle()
    contexts = _fake_contexts(consumer, split_bundle)
    protocol = freeze_relationship_p1j_protocol(
        consumer=consumer,
        split_bundle=split_bundle,
        contexts=contexts,
        context_manifest_artifact_id=contexts.artifact_id,
        frozen_at_iso=_FROZEN_AT,
    )
    return consumer, split_bundle, contexts, protocol


def test_p1j_freezes_exact_one_shot_protocol_before_qwen_output() -> None:
    consumer, split_bundle, contexts, protocol = _frozen_fixture()

    assert RelationshipP1jQualificationProtocol.from_json(protocol.to_json()) == protocol
    assert protocol.consumer_protocol_id == consumer.protocol_id
    assert protocol.qualification_observation_count == 24
    assert protocol.planned_qwen_output_count == 72
    assert protocol.qualification_qwen_outputs_observed_before_freeze == 0
    assert protocol.one_shot
    assert protocol.next_action == RELATIONSHIP_P1J_PREPARED_NEXT_ACTION
    assert not protocol.qualification_feedback_to_consumer
    assert not protocol.p2_enabled
    validate_relationship_p1j_protocol_lineage(
        protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        contexts=contexts,
        context_manifest_artifact_id=contexts.artifact_id,
    )

    with pytest.raises(ValueError, match="freeze order"):
        replace(protocol, qualification_qwen_outputs_observed_before_freeze=1)
    with pytest.raises(ValueError, match="lineage mismatch"):
        validate_relationship_p1j_protocol_lineage(
            replace(
                protocol,
                qualification_context_surface_sha256=hashlib.sha256(
                    b"drift"
                ).hexdigest(),
            ),
            consumer=consumer,
            split_bundle=split_bundle,
            contexts=contexts,
            context_manifest_artifact_id=contexts.artifact_id,
        )


def test_p1j_checkpoint_resume_preserves_records_and_publishes_terminal_report(
    tmp_path,
) -> None:
    consumer, split_bundle, contexts, protocol = _frozen_fixture()
    checkpoint = build_relationship_p1j_checkpoint(
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
    )
    write_relationship_p1j_checkpoint(checkpoint=checkpoint, output_dir=tmp_path)
    policy = _FakeQualificationPolicy(consumer, split_bundle)

    progress = load_relationship_p1j_progress(tmp_path)
    first = execute_relationship_p1j_qualification(
        policy,
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        contexts=contexts,
        existing_progress=progress,
        max_new_readouts=5,
        readout_observer=lambda index, item: persist_relationship_p1j_readout(
            checkpoint=checkpoint,
            output_dir=tmp_path,
            index=index,
            readout=item,
        ),
        decision_observer=lambda index, item: persist_relationship_p1j_decision(
            checkpoint=checkpoint,
            output_dir=tmp_path,
            index=index,
            decision=item,
        ),
    )
    assert first.new_qwen_outputs == 5
    assert not first.complete
    first_readout_path = tmp_path / "records" / "0000.readout.json"
    first_decision_path = tmp_path / "records" / "0000.decision.json"
    first_hashes = (
        hashlib.sha256(first_readout_path.read_bytes()).hexdigest(),
        hashlib.sha256(first_decision_path.read_bytes()).hexdigest(),
    )

    partial = load_relationship_p1j_progress(tmp_path)
    validate_relationship_p1j_progress(
        partial,
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        contexts=contexts,
    )
    complete_execution = execute_relationship_p1j_qualification(
        policy,
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        contexts=contexts,
        existing_progress=partial,
        readout_observer=lambda index, item: persist_relationship_p1j_readout(
            checkpoint=checkpoint,
            output_dir=tmp_path,
            index=index,
            readout=item,
        ),
        decision_observer=lambda index, item: persist_relationship_p1j_decision(
            checkpoint=checkpoint,
            output_dir=tmp_path,
            index=index,
            decision=item,
        ),
    )
    assert complete_execution.complete
    assert complete_execution.new_qwen_outputs == 67
    assert first_hashes == (
        hashlib.sha256(first_readout_path.read_bytes()).hexdigest(),
        hashlib.sha256(first_decision_path.read_bytes()).hexdigest(),
    )

    complete = load_relationship_p1j_progress(tmp_path)
    report = assess_relationship_p1j_qualification(
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
        progress=complete,
        created_at_iso=_CREATED_AT,
    )
    assert report.verdict is RelationshipP1jVerdict.QUALIFIED
    assert report.qualification_qwen_output_count == 72
    assert tuple(
        (item.accuracy, item.pair_flip_rate) for item in report.arm_metrics
    ) == ((0.75, 0.5), (0.75, 0.5), (0.75, 0.5))
    assert not report.qualification_feedback_to_consumer
    assert not report.consumer_revision_after_qualification
    assert not report.p2_enabled
    validate_relationship_p1j_report_lineage(
        report,
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
    )
    report_path, _markdown_path = write_relationship_p1j_report(
        report=report,
        progress=complete,
        output_dir=tmp_path,
    )
    assert load_relationship_p1j_report(report_path) == report
    validate_relationship_p1j_terminal_files(
        report=report,
        progress=complete,
        output_dir=tmp_path,
    )


def test_p1j_rejects_policy_and_evaluator_drift(tmp_path) -> None:
    consumer, split_bundle, contexts, protocol = _frozen_fixture()
    checkpoint = build_relationship_p1j_checkpoint(
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
    )
    write_relationship_p1j_checkpoint(checkpoint=checkpoint, output_dir=tmp_path)
    progress = load_relationship_p1j_progress(tmp_path)
    policy = _FakeQualificationPolicy(consumer, split_bundle)
    policy.weights_sha256 = hashlib.sha256(b"wrong").hexdigest()
    with pytest.raises(ValueError, match="substrate"):
        execute_relationship_p1j_qualification(
            policy,
            protocol=protocol,
            consumer=consumer,
            split_bundle=split_bundle,
            contexts=contexts,
            existing_progress=progress,
        )


@pytest.mark.parametrize(
    ("entry_name", "is_directory"),
    (("0072.readout.json", False), ("0000.readout.json", True)),
)
def test_p1j_rejects_record_entries_outside_frozen_plan(
    tmp_path,
    entry_name: str,
    is_directory: bool,
) -> None:
    consumer, split_bundle, _contexts, protocol = _frozen_fixture()
    checkpoint = build_relationship_p1j_checkpoint(
        protocol=protocol,
        consumer=consumer,
        split_bundle=split_bundle,
    )
    write_relationship_p1j_checkpoint(checkpoint=checkpoint, output_dir=tmp_path)
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    entry = records_dir / entry_name
    if is_directory:
        entry.mkdir()
    else:
        entry.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the frozen plan"):
        load_relationship_p1j_progress(tmp_path)
