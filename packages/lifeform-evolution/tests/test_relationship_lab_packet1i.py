from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import replace

import pytest

import lifeform_evolution.relationship_lab_packet1i as packet1i
from lifeform_domain_emogpt.lab import (
    RelationshipAction,
    load_relationship_consumer_training_view,
)
from lifeform_evolution.relationship_lab_baseline import StatelessActionCompletion
from lifeform_evolution.relationship_lab_contexts import (
    PersistedRelationshipP1StateDigest,
    RelationshipP1Arm,
    RelationshipP1Context,
    RelationshipP1ContextBundle,
    load_relationship_p1_context_replay_manifest,
)
from lifeform_evolution.relationship_lab_packet1i import (
    RELATIONSHIP_P1I_NEXT_ACTION,
    RelationshipP1iCalibrationProtocol,
    RelationshipP1iFrozenConsumerProtocol,
    assess_relationship_p1i_calibration,
    build_relationship_p1i_candidate_checkpoint,
    finalize_relationship_p1i_candidate_checkpoint,
    freeze_relationship_p1i_consumer_protocol,
    load_relationship_p1i_calibration_protocol,
    load_relationship_p1i_candidate_artifact,
    load_relationship_p1i_candidate_progress,
    persist_relationship_p1i_decision,
    persist_relationship_p1i_readout,
    relationship_p1i_run_from_progress,
    relationship_p1i_training_context_surface_sha256,
    run_relationship_p1i_candidate,
    summarize_relationship_p1i_candidate,
    validate_relationship_p1i_candidate_files,
    validate_relationship_p1i_candidate_progress,
    validate_relationship_p1i_frozen_consumer_lineage,
    validate_relationship_p1i_local_lineage,
    write_relationship_p1i_candidate_artifact,
    write_relationship_p1i_candidate_checkpoint,
    write_relationship_p1i_report_and_protocol,
)


_CREATED_AT = "2026-08-20T03:00:00+00:00"


def _fake_context_bundle(
    protocol: RelationshipP1iCalibrationProtocol,
) -> RelationshipP1ContextBundle:
    view = load_relationship_consumer_training_view()
    contexts = []
    for observation in view.training_dataset.observations:
        for depth in protocol.background_depths:
            for arm in (
                RelationshipP1Arm.PROMPT_STEELMAN,
                RelationshipP1Arm.RAG_STEELMAN,
                RelationshipP1Arm.STRUCTURED_STATE,
            ):
                contexts.append(
                    RelationshipP1Context(
                        arm=arm,
                        scene_id=observation.scene_id,
                        background_depth=depth,
                        context_text=(
                            f"[public relationship outcome evidence]\n"
                            f"scene_id: {observation.scene_id}\n"
                            f"arm: {arm.value}\ndepth: {depth}"
                        ),
                        source_evidence_refs=(f"fake:{observation.scene_id}",),
                    )
                )
    digest = hashlib.sha256(b"fake-state").hexdigest()
    return RelationshipP1ContextBundle(
        dataset_fingerprint=protocol.training_dataset_fingerprint,
        background_depths=protocol.background_depths,
        background_templates_sha256=protocol.background_templates_sha256,
        rag_config_sha256=protocol.rag_config_sha256,
        contexts=tuple(contexts),
        persisted_state=PersistedRelationshipP1StateDigest(
            structured_scope_digests=(("fake-scope", digest, 4),),
            rag_scope_digests=(("fake-scope", digest, 4),),
        ),
    )


class _FakeCalibrationPolicy:
    def __init__(self, protocol: RelationshipP1iCalibrationProtocol) -> None:
        self.model_id = protocol.model_id
        self.weights_sha256 = protocol.expected_weights_sha256
        self.generation_config_sha256 = protocol.expected_generation_config_sha256
        self.prompt_sha256 = hashlib.sha256(b"unused-stateless").hexdigest()
        view = load_relationship_consumer_training_view()
        self._expected = {
            observation.scene_id: view.training_dataset.dynamic_for_scene(
                observation.scene_id
            ).preferred_action
            for observation in view.training_dataset.observations
        }
        self._candidate_by_hash = {
            hashlib.sha256(
                (
                    packet1i._asset_dir() / "prompts" / item.prompt_asset
                ).read_text(encoding="utf-8").strip().encode("utf-8")
            ).hexdigest(): item.candidate_id
            for item in protocol.candidates
        }

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        raise AssertionError("P1i must not run a stateless calibration arm")

    def choose_from_messages(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        seed: int,
    ) -> StatelessActionCompletion:
        prompt_hash = hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest()
        candidate_id = self._candidate_by_hash[prompt_hash]
        user_message = messages[1]["content"]
        scene_id = next(scene for scene in self._expected if scene in user_message)
        if candidate_id == "latent_partition_v1":
            action = self._expected[scene_id]
        elif candidate_id == "conditioned_match_v1":
            action = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
        else:
            action = RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION
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


def test_p1i_protocol_is_training_only_and_content_addressed() -> None:
    protocol = load_relationship_p1i_calibration_protocol()
    view = validate_relationship_p1i_local_lineage(protocol)

    assert RelationshipP1iCalibrationProtocol.from_json(protocol.to_json()) == protocol
    assert protocol.protocol_id == (
        "080c908d7824b25081c501abbb6e76a2405a16508dbc0fb9d119b831447eef40"
    )
    assert protocol.training_package_name == "relationship_transfer_v3"
    assert protocol.qualification_package_name == "relationship_transfer_v4"
    assert len(protocol.candidates) == protocol.maximum_revision_rounds == 3
    assert protocol.selection_method == "leave_one_surface_family_out_training_only"
    assert protocol.qualification_inputs_observed_before_freeze == 0
    assert protocol.qualification_qwen_outputs_observed_before_freeze == 0
    assert "qualification_dataset" not in view.__dataclass_fields__

    with pytest.raises(ValueError, match="before observing v4"):
        replace(protocol, qualification_inputs_observed_before_freeze=1)
    with pytest.raises(ValueError, match="three rounds"):
        replace(protocol, maximum_revision_rounds=4)


def test_p1i_loads_frozen_p1g_context_replay_manifest() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    training_view = load_relationship_consumer_training_view()
    manifest = load_relationship_p1_context_replay_manifest(
        repo_root
        / "artifacts"
        / "relationship_lab"
        / "qwen25_3b_packet1g_v3_conditioned_top4_20260820"
        / "p1b_candidate"
        / "contexts.json",
        dataset=training_view.training_dataset,
    )

    assert manifest.artifact_id == (
        "aea311e2604bffcbf087cba1f8e10314e808ffc6e337533c797d9e8316259791"
    )
    assert len(manifest.context_hashes) == 144
    assert len(manifest.rag_orders) == 36
    replay = next(
        item
        for item in manifest.rag_orders
        if item.scene_id == "rtv3_scene_002a" and item.background_depth == 0
    )
    assert replay.turn_ids == (
        "rtv3_evt_011",
        "rtv3_evt_012",
        "rtv3_evt_009",
        "rtv3_evt_010",
    )


def test_p1i_fake_calibration_preserves_all_candidates_and_freezes_one(
    tmp_path,
) -> None:
    base_protocol = load_relationship_p1i_calibration_protocol()
    training_view = load_relationship_consumer_training_view()
    contexts = _fake_context_bundle(base_protocol)
    protocol = replace(
        base_protocol,
        training_context_surface_sha256=(
            relationship_p1i_training_context_surface_sha256(bundle=contexts)
        ),
    )
    policy = _FakeCalibrationPolicy(protocol)
    artifacts = []
    for candidate in protocol.candidates:
        run = run_relationship_p1i_candidate(
            policy,
            protocol=protocol,
            candidate=candidate,
            training_view=training_view,
            contexts=contexts,
        )
        artifact = summarize_relationship_p1i_candidate(run)
        write_relationship_p1i_candidate_artifact(
            run=run,
            artifact=artifact,
            output_dir=tmp_path / f"candidate_{candidate.round_index:02d}",
        )
        artifacts.append(artifact)

    report = assess_relationship_p1i_calibration(
        protocol=protocol,
        training_view=training_view,
        candidate_artifacts=tuple(artifacts),
        created_at_iso=_CREATED_AT,
    )
    assert report.selected_candidate_id == "latent_partition_v1"
    assert report.ranking[0] == report.selected_candidate_id
    assert len(report.candidate_artifacts) == 3
    assert report.qualification_inputs_observed == 0
    assert report.qualification_qwen_outputs_observed == 0
    assert not report.p2_enabled
    assert report.next_action == RELATIONSHIP_P1I_NEXT_ACTION
    selected_metrics = dict(report.selected_candidate.selection_metrics)
    assert selected_metrics["minimum_primary_macro_accuracy"] == 1.0
    assert selected_metrics["minimum_primary_macro_pair_flip_rate"] == 1.0

    frozen = freeze_relationship_p1i_consumer_protocol(
        calibration_protocol=protocol,
        report=report,
        training_view=training_view,
        frozen_at_iso=_CREATED_AT,
    )
    assert frozen.selected_candidate.candidate_id == "latent_partition_v1"
    assert frozen.qualification_inputs_observed_before_freeze == 0
    assert frozen.qualification_qwen_outputs_observed_before_freeze == 0
    assert RelationshipP1iFrozenConsumerProtocol.from_json(frozen.to_json()) == frozen
    validate_relationship_p1i_frozen_consumer_lineage(
        frozen,
        calibration_protocol=protocol,
        report=report,
        training_view=training_view,
    )
    with pytest.raises(ValueError, match="lineage mismatch"):
        validate_relationship_p1i_frozen_consumer_lineage(
            replace(frozen, rag_top_k=3),
            calibration_protocol=protocol,
            report=report,
            training_view=training_view,
        )

    report_path, _markdown_path, consumer_path = (
        write_relationship_p1i_report_and_protocol(
            report=report,
            consumer_protocol=frozen,
            output_dir=tmp_path,
        )
    )
    assert report_path.is_file()
    assert consumer_path.is_file()
    validate_relationship_p1i_candidate_files(report=report, output_dir=tmp_path)


def test_p1i_rejects_context_and_policy_lineage_drift() -> None:
    base_protocol = load_relationship_p1i_calibration_protocol()
    training_view = load_relationship_consumer_training_view()
    contexts = _fake_context_bundle(base_protocol)
    protocol = replace(
        base_protocol,
        training_context_surface_sha256=(
            relationship_p1i_training_context_surface_sha256(bundle=contexts)
        ),
    )
    policy = _FakeCalibrationPolicy(protocol)
    candidate = protocol.candidates[0]

    with pytest.raises(ValueError, match="training context surface"):
        run_relationship_p1i_candidate(
            policy,
            protocol=replace(
                protocol,
                training_context_surface_sha256=hashlib.sha256(b"drift").hexdigest(),
            ),
            candidate=candidate,
            training_view=training_view,
            contexts=contexts,
        )
    policy.weights_sha256 = hashlib.sha256(b"wrong-weights").hexdigest()
    with pytest.raises(ValueError, match="substrate lineage"):
        run_relationship_p1i_candidate(
            policy,
            protocol=protocol,
            candidate=candidate,
            training_view=training_view,
            contexts=contexts,
        )


def test_p1i_candidate_checkpoint_recovers_without_changing_records(tmp_path) -> None:
    base_protocol = load_relationship_p1i_calibration_protocol()
    training_view = load_relationship_consumer_training_view()
    contexts = _fake_context_bundle(base_protocol)
    protocol = replace(
        base_protocol,
        training_context_surface_sha256=(
            relationship_p1i_training_context_surface_sha256(bundle=contexts)
        ),
    )
    policy = _FakeCalibrationPolicy(protocol)
    candidate = protocol.candidates[0]
    candidate_dir = tmp_path / "candidate_01"
    checkpoint = build_relationship_p1i_candidate_checkpoint(
        policy,
        protocol=protocol,
        candidate=candidate,
        training_view=training_view,
        contexts=contexts,
    )
    write_relationship_p1i_candidate_checkpoint(
        checkpoint=checkpoint,
        candidate_dir=candidate_dir,
    )

    class _StopAfterFive(RuntimeError):
        pass

    readout_index = 0
    decision_index = 0

    def checkpoint_readout(readout) -> None:
        nonlocal readout_index
        persist_relationship_p1i_readout(
            checkpoint=checkpoint,
            candidate_dir=candidate_dir,
            index=readout_index,
            readout=readout,
        )
        readout_index += 1

    def checkpoint_decision(decision) -> None:
        nonlocal decision_index
        persist_relationship_p1i_decision(
            checkpoint=checkpoint,
            candidate_dir=candidate_dir,
            index=decision_index,
            decision=decision,
        )
        decision_index += 1
        if decision_index == 5:
            raise _StopAfterFive

    with pytest.raises(_StopAfterFive):
        run_relationship_p1i_candidate(
            policy,
            protocol=protocol,
            candidate=candidate,
            training_view=training_view,
            contexts=contexts,
            readout_observer=checkpoint_readout,
            decision_observer=checkpoint_decision,
        )
    partial = load_relationship_p1i_candidate_progress(candidate_dir)
    validate_relationship_p1i_candidate_progress(
        partial,
        protocol=protocol,
        candidate=candidate,
        training_view=training_view,
        contexts=contexts,
    )
    assert len(partial.readouts) == len(partial.decisions) == 5
    assert not partial.is_complete

    readout_index = 0
    decision_index = 0

    def checkpoint_decision_resume(decision) -> None:
        nonlocal decision_index
        persist_relationship_p1i_decision(
            checkpoint=checkpoint,
            candidate_dir=candidate_dir,
            index=decision_index,
            decision=decision,
        )
        decision_index += 1

    run = run_relationship_p1i_candidate(
        policy,
        protocol=protocol,
        candidate=candidate,
        training_view=training_view,
        contexts=contexts,
        readout_observer=checkpoint_readout,
        decision_observer=checkpoint_decision_resume,
    )
    complete = load_relationship_p1i_candidate_progress(candidate_dir)
    validate_relationship_p1i_candidate_progress(
        complete,
        protocol=protocol,
        candidate=candidate,
        training_view=training_view,
        contexts=contexts,
    )
    assert complete.is_complete
    assert relationship_p1i_run_from_progress(
        complete,
        training_view=training_view,
    ) == run
    artifact = summarize_relationship_p1i_candidate(run)
    summary_path = finalize_relationship_p1i_candidate_checkpoint(
        run=run,
        artifact=artifact,
        candidate_dir=candidate_dir,
    )
    assert summary_path.is_file()
    assert load_relationship_p1i_candidate_artifact(candidate_dir) == artifact
