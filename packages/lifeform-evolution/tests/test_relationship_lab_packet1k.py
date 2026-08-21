from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import replace

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
from lifeform_evolution.relationship_lab_packet1j import (
    RelationshipP1jVerdict,
    load_relationship_p1j_protocol,
    load_relationship_p1j_report,
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
    load_relationship_p1k_protocol,
    load_relationship_p1k_report,
    persist_relationship_p1k_decision,
    persist_relationship_p1k_readout,
    relationship_p1k_execution_gate,
    render_relationship_p1k_request,
    validate_relationship_p1k_progress,
    validate_relationship_p1k_protocol_lineage,
    validate_relationship_p1k_report_lineage,
    validate_relationship_p1k_terminal_files,
    write_relationship_p1k_checkpoint,
    write_relationship_p1k_protocol,
    write_relationship_p1k_report,
)


_FROZEN_AT = "2026-08-21T13:00:00+00:00"
_CREATED_AT = "2026-08-21T14:00:00+00:00"


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _consumer():
    return load_relationship_p1i_frozen_consumer_protocol(
        _repo_root()
        / "artifacts"
        / "relationship_lab"
        / "qwen25_3b_packet1i_v3_training_replay_20260820"
        / "frozen_consumer_protocol.json"
    )


def _source_p1j():
    source = _repo_root() / "artifacts" / "relationship_lab" / "qwen25_3b_packet1j_v4_one_shot_20260821"
    return (
        load_relationship_p1j_protocol(source / "packet1j_protocol.json"),
        load_relationship_p1j_report(source / "packet1j_report.json"),
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
                () if arm.value == "stateless" else tuple(item.event_id for item in observation.histories)
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
            observation.scene_id: dataset.dynamic_for_scene(observation.scene_id).preferred_action
            for observation in dataset.observations
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
        has_policy = "该用户在每种抽象关系条件下需要的关系动作" in request
        has_binding = "每条公开历史所处的抽象关系条件" in request
        has_current = "当前消息所处的抽象关系条件" in request
        is_application = has_policy and has_binding and has_current
        is_induction = not has_policy and has_binding and has_current
        is_recognition = has_policy and has_binding and not has_current
        is_binding = not has_policy and not has_binding and has_current
        cell_count = sum((is_application, is_induction, is_recognition, is_binding))
        if cell_count != 1:
            raise AssertionError(f"invalid diagnostic disclosure count: {cell_count}")
        correct = {
            "application_fail": False,
            "policy_induction_fail": is_application or is_recognition,
            "probe_recognition_fail": not is_recognition,
            "history_binding_fail": not is_binding,
            "multiple_middle_fail": is_application,
            "all_correct": True,
        }.get(self._mode)
        if self._mode == "machinery_invalid":
            return StatelessActionCompletion(
                raw_output="{}",
                chosen_action_id=None,
                prompt_tokens=80,
                completion_tokens=2,
            )
        if correct is None:
            raise AssertionError(f"unknown fake mode: {self._mode}")
        action = self._expected[scene_id] if correct else RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
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
    source_p1j_protocol, source_p1j_report = _source_p1j()
    dataset = load_relationship_consumer_training_view().training_dataset
    contexts = _fake_contexts(consumer, dataset)
    protocol = freeze_relationship_p1k_protocol(
        consumer=consumer,
        source_p1j_protocol=source_p1j_protocol,
        source_p1j_report=source_p1j_report,
        dataset=dataset,
        contexts=contexts,
        seed_schedule=consumer.seed_schedule,
        frozen_at_iso=_FROZEN_AT,
    )
    return (
        consumer,
        source_p1j_protocol,
        source_p1j_report,
        dataset,
        contexts,
        protocol,
    )


def _persist_callbacks(checkpoint, output_dir):
    def persist_readout(index, item) -> None:
        persist_relationship_p1k_readout(
            checkpoint=checkpoint,
            output_dir=output_dir,
            index=index,
            readout=item,
        )

    def persist_decision(index, item) -> None:
        persist_relationship_p1k_decision(
            checkpoint=checkpoint,
            output_dir=output_dir,
            index=index,
            decision=item,
        )

    return persist_readout, persist_decision


def _run_terminal(tmp_path, *, mode: str):
    consumer, _p1j_protocol, _p1j_report, dataset, contexts, protocol = _frozen_fixture()
    checkpoint = build_relationship_p1k_checkpoint(protocol=protocol, dataset=dataset)
    write_relationship_p1k_checkpoint(checkpoint=checkpoint, output_dir=tmp_path)
    persist_readout, persist_decision = _persist_callbacks(checkpoint, tmp_path)
    policy = _FakeOraclePolicy(consumer, dataset, mode=mode)
    first = execute_relationship_p1k_diagnostic(
        policy,
        protocol=protocol,
        dataset=dataset,
        contexts=contexts,
        existing_progress=load_relationship_p1k_progress(tmp_path),
        max_new_readouts=4,
        readout_observer=persist_readout,
        decision_observer=persist_decision,
    )
    assert first.new_outputs == 4
    first_readout = tmp_path / "records" / "0000.readout.json"
    first_hash = hashlib.sha256(first_readout.read_bytes()).hexdigest()
    for _ in RELATIONSHIP_P1K_TIERS:
        progress = load_relationship_p1k_progress(tmp_path)
        gate = relationship_p1k_execution_gate(protocol=protocol, progress=progress)
        if gate.terminal:
            break
        execute_relationship_p1k_diagnostic(
            policy,
            protocol=protocol,
            dataset=dataset,
            contexts=contexts,
            existing_progress=progress,
            readout_observer=persist_readout,
            decision_observer=persist_decision,
        )
    terminal = load_relationship_p1k_progress(tmp_path)
    validate_relationship_p1k_progress(
        terminal,
        protocol=protocol,
        dataset=dataset,
        contexts=contexts,
    )
    assert relationship_p1k_execution_gate(
        protocol=protocol,
        progress=terminal,
    ).terminal
    assert hashlib.sha256(first_readout.read_bytes()).hexdigest() == first_hash
    report = assess_relationship_p1k_diagnostic(
        protocol=protocol,
        progress=terminal,
        created_at_iso=_CREATED_AT,
    )
    return protocol, terminal, report


def test_p1k_matrix_disclosures_are_orthogonal_and_hide_sealed_ids() -> None:
    dataset = load_relationship_consumer_training_view().training_dataset
    scene_id = dataset.observations[0].scene_id
    dynamic = dataset.dynamic_for_scene(scene_id)
    disclosures = {
        tier: build_relationship_p1k_disclosure(
            dataset=dataset,
            scene_id=scene_id,
            tier=tier,
        )
        for tier in RELATIONSHIP_P1K_TIERS
    }
    for text in disclosures.values():
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
    application = disclosures[RelationshipP1kOracleTier.POLICY_APPLICATION]
    induction = disclosures[RelationshipP1kOracleTier.POLICY_INDUCTION]
    recognition = disclosures[RelationshipP1kOracleTier.PROBE_RECOGNITION]
    binding = disclosures[RelationshipP1kOracleTier.HISTORY_BINDING]
    assert dynamic.preferred_action.value in application
    assert "每条公开历史所处的抽象关系条件" in induction
    assert dynamic.preferred_action.value not in induction
    assert "当前消息所处的抽象关系条件" not in recognition
    assert dynamic.preferred_action.value in recognition
    assert "当前消息所处的抽象关系条件" in binding
    assert "每条公开历史所处的抽象关系条件" not in binding
    assert dynamic.preferred_action.value not in binding


def test_p1k_freezes_p1j_bound_noncompetitive_matrix_at_zero_outputs(tmp_path) -> None:
    consumer, p1j_protocol, p1j_report, dataset, contexts, protocol = _frozen_fixture()
    assert protocol.next_action == RELATIONSHIP_P1K_PREPARED_NEXT_ACTION
    assert protocol.context_arm == RELATIONSHIP_P1K_CONTEXT_ARM.value
    assert protocol.source_p1j_protocol_id == p1j_protocol.protocol_id
    assert protocol.source_p1j_report_artifact_id == p1j_report.artifact_id
    assert protocol.source_p1j_verdict == RelationshipP1jVerdict.UNDERQUALIFIED.value
    assert protocol.observation_count == 12
    assert protocol.planned_output_count == 48
    assert protocol.tiers == tuple(tier.value for tier in RELATIONSHIP_P1K_TIERS)
    assert not protocol.to_payload()["experiment_guards"]["competitive"]
    validate_relationship_p1k_protocol_lineage(
        protocol,
        consumer=consumer,
        source_p1j_protocol=p1j_protocol,
        source_p1j_report=p1j_report,
        dataset=dataset,
        contexts=contexts,
    )
    protocol_path = write_relationship_p1k_protocol(protocol, tmp_path / "protocol.json")
    assert load_relationship_p1k_protocol(protocol_path) == protocol


def test_p1k_rejects_nonfailed_p1j_prerequisite() -> None:
    consumer = _consumer()
    p1j_protocol, p1j_report = _source_p1j()
    dataset = load_relationship_consumer_training_view().training_dataset
    contexts = _fake_contexts(consumer, dataset)
    with pytest.raises(ValueError, match="underqualification"):
        freeze_relationship_p1k_protocol(
            consumer=consumer,
            source_p1j_protocol=p1j_protocol,
            source_p1j_report=replace(
                p1j_report,
                verdict=RelationshipP1jVerdict.QUALIFIED,
            ),
            dataset=dataset,
            contexts=contexts,
            seed_schedule=consumer.seed_schedule,
            frozen_at_iso=_FROZEN_AT,
        )


@pytest.mark.parametrize(
    ("mode", "verdict", "output_count", "skipped"),
    (
        (
            "application_fail",
            RelationshipP1kVerdict.SUBSTRATE_APPLICATION_FLOOR,
            12,
            3,
        ),
        (
            "machinery_invalid",
            RelationshipP1kVerdict.MACHINERY_REGRESSION,
            12,
            3,
        ),
        (
            "policy_induction_fail",
            RelationshipP1kVerdict.POLICY_INDUCTION_BOTTLENECK,
            36,
            1,
        ),
        (
            "multiple_middle_fail",
            RelationshipP1kVerdict.MULTIPLE_DIAGNOSTIC_BOTTLENECKS,
            36,
            1,
        ),
        (
            "probe_recognition_fail",
            RelationshipP1kVerdict.CONDITION_RECOGNITION_BOTTLENECK,
            48,
            0,
        ),
        (
            "history_binding_fail",
            RelationshipP1kVerdict.HISTORY_BINDING_BOTTLENECK,
            48,
            0,
        ),
        (
            "all_correct",
            RelationshipP1kVerdict.UNAIDED_ABSTRACTION_OR_TRANSFER,
            48,
            0,
        ),
    ),
)
def test_p1k_staged_gate_routes_terminal_owner(
    tmp_path,
    mode,
    verdict,
    output_count,
    skipped,
) -> None:
    protocol, terminal, report = _run_terminal(tmp_path, mode=mode)
    assert report.verdict is verdict
    assert report.output_count == output_count
    assert report.planned_output_count == 48
    assert len(report.skipped_tiers) == skipped
    assert len(terminal.decisions) == output_count
    validate_relationship_p1k_report_lineage(report, protocol=protocol)


def test_p1k_terminal_artifact_round_trip_and_tamper_detection(tmp_path) -> None:
    protocol, terminal, report = _run_terminal(tmp_path, mode="policy_induction_fail")
    report_path, _markdown = write_relationship_p1k_report(
        report=report,
        progress=terminal,
        output_dir=tmp_path,
    )
    loaded = load_relationship_p1k_report(report_path)
    assert loaded == report
    validate_relationship_p1k_report_lineage(loaded, protocol=protocol)
    validate_relationship_p1k_terminal_files(
        report=loaded,
        progress=terminal,
        output_dir=tmp_path,
    )
    (tmp_path / "readouts.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="readout ledger"):
        validate_relationship_p1k_terminal_files(
            report=loaded,
            progress=terminal,
            output_dir=tmp_path,
        )


def test_p1k_rejects_unplanned_record_file(tmp_path) -> None:
    _consumer_value, _p1j_protocol, _p1j_report, dataset, _contexts, protocol = _frozen_fixture()
    checkpoint = build_relationship_p1k_checkpoint(protocol=protocol, dataset=dataset)
    write_relationship_p1k_checkpoint(checkpoint=checkpoint, output_dir=tmp_path)
    records = tmp_path / "records"
    records.mkdir()
    (records / "surprise.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the frozen plan"):
        load_relationship_p1k_progress(tmp_path)


def test_p1k_rejects_policy_drift(tmp_path) -> None:
    consumer, _p1j_protocol, _p1j_report, dataset, contexts, protocol = _frozen_fixture()
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
