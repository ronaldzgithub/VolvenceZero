from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from dataclasses import replace

import pytest

from companion_ref_harness import HashingEmbedder
from lifeform_domain_emogpt.lab import (
    RelationshipAction,
    RelationshipDatasetSplit,
    load_relationship_transfer_dataset,
    sha256_json,
)
from lifeform_evolution.relationship_lab_baseline import StatelessActionCompletion
from lifeform_evolution.relationship_lab_contexts import (
    RELATIONSHIP_P1_ARMS,
    RelationshipP1Arm,
    build_relationship_p1_context_bundle,
    probe_relationship_p1_persisted_state,
    relationship_p1_evaluated_context_surface_sha256,
    relationship_p1_structural_metrics,
    run_relationship_p1_console_control_probe,
)
from lifeform_evolution.relationship_lab_gate0 import FrozenBaselineAttestation
from lifeform_evolution.relationship_lab_packet1 import (
    RelationshipP1RecoveryEvidence,
    assess_relationship_packet1,
    relationship_p1_prompt_path,
    reassess_relationship_packet1_report_v1,
    run_relationship_packet1_arms,
    write_relationship_packet1_artifacts,
)


_CREATED_AT = "2026-08-19T12:00:00+00:00"


class _MappedContextPolicy:
    model_id = "frozen-p1-test-model"
    weights_sha256 = sha256_json("p1-test-weights")
    generation_config_sha256 = sha256_json("p1-test-generation")
    prompt_sha256 = hashlib.sha256(relationship_p1_prompt_path(RelationshipP1Arm.STATELESS).read_bytes()).hexdigest()

    def __init__(self, choices: dict[str, RelationshipAction]) -> None:
        self._choices = choices

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        assert current_input
        assert seed >= 0
        return self._completion(RelationshipAction.NEUTRAL_NOOP, current_input)

    def choose_from_messages(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        seed: int,
    ) -> StatelessActionCompletion:
        assert seed >= 0
        user_content = messages[-1]["content"]
        return self._completion(self._choices[user_content], user_content)

    def count_tokens(self, text: str) -> int:
        return max(0, len(text) // 4)

    @staticmethod
    def _completion(
        action: RelationshipAction,
        source: str,
    ) -> StatelessActionCompletion:
        raw = json.dumps({"action_id": action.value}, separators=(",", ":"))
        return StatelessActionCompletion(
            raw_output=raw,
            chosen_action_id=action,
            prompt_tokens=max(1, len(source) // 4) + 80,
            completion_tokens=10,
        )


def _choices_for_qualified_steelmen(bundle) -> dict[str, RelationshipAction]:
    dataset = load_relationship_transfer_dataset()
    choices: dict[str, RelationshipAction] = {}
    first_scene_by_arm: dict[RelationshipP1Arm, str] = {}
    for arm in (
        RelationshipP1Arm.PROMPT_STEELMAN,
        RelationshipP1Arm.RAG_STEELMAN,
        RelationshipP1Arm.STRUCTURED_STATE,
    ):
        eligible = [
            observation
            for observation in dataset.observations
            if dataset.dynamic_for_scene(observation.scene_id).split
            in {RelationshipDatasetSplit.TRAIN, RelationshipDatasetSplit.VALIDATION}
        ]
        first_scene_by_arm[arm] = eligible[0].scene_id
        for observation in eligible:
            dynamic = dataset.dynamic_for_scene(observation.scene_id)
            context = bundle.context(scene_id=observation.scene_id, arm=arm)
            action = dynamic.preferred_action
            if arm is not RelationshipP1Arm.STRUCTURED_STATE and observation.scene_id == first_scene_by_arm[arm]:
                action = (
                    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION
                    if action is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
                    else RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
                )
            choices[context.render_user_message(observation.current_input)] = action
    return choices


def _gate0_baseline(policy: _MappedContextPolicy) -> FrozenBaselineAttestation:
    dataset = load_relationship_transfer_dataset()
    return FrozenBaselineAttestation(
        arm_id="stateless",
        dataset_fingerprint=dataset.dataset_fingerprint,
        model_id=policy.model_id,
        weights_sha256=policy.weights_sha256,
        prompt_sha256=policy.prompt_sha256,
        generation_config_sha256=policy.generation_config_sha256,
        seed_schedule_sha256=sha256_json((101, 211, 307)),
        decision_ledger_sha256=sha256_json("p1-gate0-ledger"),
        evaluated_split="calibration",
        valid_decisions=24,
        correct_decisions=4,
        evaluated_decisions=24,
        context_tokens_total=1000,
        hidden_test_opened=False,
        frozen_at_iso="2026-08-19T10:00:00+00:00",
    )


def test_p1_contexts_use_both_persistent_owners_without_truth_leakage(
    tmp_path,
) -> None:
    dataset = load_relationship_transfer_dataset()
    bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "state",
        rag_embedder=HashingEmbedder(),
    )
    assert bundle.background_depths == (0, 8, 32)
    assert len(bundle.contexts) == len(dataset.observations) * 3 * 4
    assert {item.arm for item in bundle.contexts} == set(RELATIONSHIP_P1_ARMS)
    assert relationship_p1_structural_metrics(bundle=bundle)["passed"]
    recovered = probe_relationship_p1_persisted_state(state_root=tmp_path / "state")
    assert recovered.artifact_id == bundle.persisted_state.artifact_id
    assert {row[2] for row in recovered.structured_scope_digests} == {34}
    assert {row[2] for row in recovered.rag_scope_digests} == {34}
    sealed_tokens = {dynamic.dynamic_id for dynamic in dataset.dynamics} | {
        dynamic.mirror_pair_id for dynamic in dataset.dynamics
    }
    for context in bundle.contexts:
        assert not any(token in context.context_text for token in sealed_tokens)
        assert "preferred_action" not in context.context_text
    deepest_rag_contexts = tuple(
        context
        for context in bundle.contexts
        if context.arm is RelationshipP1Arm.RAG_STEELMAN and context.background_depth == 32
    )
    assert {len(context.source_evidence_refs) for context in deepest_rag_contexts} == {4}


def test_p1_context_bundle_versions_rag_top_k_in_its_config(tmp_path) -> None:
    default_bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "default",
        rag_embedder=HashingEmbedder(),
    )
    top_two_bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "top-two",
        rag_embedder=HashingEmbedder(),
        rag_top_k=2,
    )
    assert default_bundle.rag_config_sha256 != top_two_bundle.rag_config_sha256
    top_two_rag_contexts = tuple(
        context
        for context in top_two_bundle.contexts
        if context.arm is RelationshipP1Arm.RAG_STEELMAN and context.background_depth == 32
    )
    assert {len(context.source_evidence_refs) for context in top_two_rag_contexts} == {2}
    with pytest.raises(ValueError):
        build_relationship_p1_context_bundle(
            state_root=tmp_path / "invalid",
            rag_embedder=HashingEmbedder(),
            rag_top_k=0,
        )


def test_p1_evaluated_context_lineage_excludes_volatile_owner_ids_and_repo_heldout_rag(
    tmp_path,
) -> None:
    first = build_relationship_p1_context_bundle(
        state_root=tmp_path / "first",
        rag_embedder=HashingEmbedder(),
        rag_top_k=2,
    )
    second = build_relationship_p1_context_bundle(
        state_root=tmp_path / "second",
        rag_embedder=HashingEmbedder(),
        rag_top_k=2,
    )
    assert first.artifact_id != second.artifact_id
    assert relationship_p1_evaluated_context_surface_sha256(
        bundle=first
    ) == relationship_p1_evaluated_context_surface_sha256(bundle=second)

    dataset = load_relationship_transfer_dataset()
    heldout_pair = next(
        members
        for _pair_id, members in dataset.mirrored_pairs()
        if members[0][1].split is RelationshipDatasetSplit.HELDOUT
    )
    left_scene = heldout_pair[0][0].scene_id
    right_scene = heldout_pair[1][0].scene_id
    left_rag = first.context(
        scene_id=left_scene,
        arm=RelationshipP1Arm.RAG_STEELMAN,
    )
    contexts = tuple(
        replace(
            item,
            context_text=left_rag.context_text,
            source_evidence_refs=left_rag.source_evidence_refs,
        )
        if item.scene_id == right_scene
        and item.arm is RelationshipP1Arm.RAG_STEELMAN
        and item.background_depth == first.max_background_depth
        else item
        for item in first.contexts
    )
    with_heldout_rag_collision = replace(first, contexts=contexts)
    metrics = relationship_p1_structural_metrics(
        bundle=with_heldout_rag_collision,
        dataset=dataset,
    )
    assert metrics["evaluated_context_pairs"] == 4
    assert metrics["contextual_histories_distinct"]
    assert metrics["scope_isolation"]
    assert metrics["passed"]


def test_p1_state_recovery_is_reproducible_in_a_fresh_process(tmp_path) -> None:
    bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "state",
        rag_embedder=HashingEmbedder(),
    )
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_relationship_lab_packet1.py"),
            "probe-state",
            "--state-root",
            str(tmp_path / "state"),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["artifact_id"] == bundle.persisted_state.artifact_id


def test_p1_console_rewrite_delete_and_sibling_isolation_persist(tmp_path) -> None:
    evidence = run_relationship_p1_console_control_probe(root=tmp_path / "console")
    assert evidence.passed
    assert evidence.rewrite_persisted
    assert evidence.delete_persisted
    assert evidence.sibling_scope_unchanged
    assert evidence.original_entry_sha256 != evidence.rewritten_entry_sha256


def test_p1_runner_qualifies_non_saturated_steelmen_and_writes_artifacts(
    tmp_path,
) -> None:
    bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "state",
        rag_embedder=HashingEmbedder(),
    )
    policy = _MappedContextPolicy(_choices_for_qualified_steelmen(bundle))
    observed = []
    run = run_relationship_packet1_arms(
        policy,
        contexts=bundle,
        decision_observer=observed.append,
    )
    assert tuple(observed) == run.decisions
    assert len(run.decisions) == 32
    assert run.arm_metrics(RelationshipP1Arm.STATELESS)["accuracy"] == 0.0
    assert run.arm_metrics(RelationshipP1Arm.PROMPT_STEELMAN)["accuracy"] == 0.875
    assert run.arm_metrics(RelationshipP1Arm.RAG_STEELMAN)["accuracy"] == 0.875
    assert run.arm_metrics(RelationshipP1Arm.STRUCTURED_STATE)["accuracy"] == 1.0

    recovery = RelationshipP1RecoveryEvidence(
        expected_state_artifact_id=bundle.persisted_state.artifact_id,
        recovered_state_artifact_id=bundle.persisted_state.artifact_id,
        fresh_process=True,
    )
    console = run_relationship_p1_console_control_probe(root=tmp_path / "console")
    report = assess_relationship_packet1(
        run=run,
        contexts=bundle,
        recovery=recovery,
        console=console,
        gate0_baseline=_gate0_baseline(policy),
        created_at_iso=_CREATED_AT,
    )
    assert report.machinery_ready
    assert report.gate1_passed
    paths = write_relationship_packet1_artifacts(
        run=run,
        report=report,
        recovery=recovery,
        console=console,
        persisted_state=bundle.persisted_state,
        contexts=bundle,
        output_dir=tmp_path / "evidence",
    )
    assert len(paths) == 8
    assert json.loads(paths[6].read_text(encoding="utf-8"))["artifact_id"] == (report.artifact_id)
    with pytest.raises(FileExistsError):
        write_relationship_packet1_artifacts(
            run=run,
            report=report,
            recovery=recovery,
            console=console,
            persisted_state=bundle.persisted_state,
            contexts=bundle,
            output_dir=tmp_path / "evidence",
        )


def test_p1_gate_rejects_a_saturated_prompt_steelman(tmp_path) -> None:
    bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "state",
        rag_embedder=HashingEmbedder(),
    )
    choices = _choices_for_qualified_steelmen(bundle)
    dataset = load_relationship_transfer_dataset()
    for observation in dataset.observations:
        dynamic = dataset.dynamic_for_scene(observation.scene_id)
        if dynamic.split not in {
            RelationshipDatasetSplit.TRAIN,
            RelationshipDatasetSplit.VALIDATION,
        }:
            continue
        context = bundle.context(
            scene_id=observation.scene_id,
            arm=RelationshipP1Arm.PROMPT_STEELMAN,
        )
        choices[context.render_user_message(observation.current_input)] = dynamic.preferred_action
    policy = _MappedContextPolicy(choices)
    run = run_relationship_packet1_arms(policy, contexts=bundle)
    recovery = RelationshipP1RecoveryEvidence(
        expected_state_artifact_id=bundle.persisted_state.artifact_id,
        recovered_state_artifact_id=bundle.persisted_state.artifact_id,
        fresh_process=True,
    )
    report = assess_relationship_packet1(
        run=run,
        contexts=bundle,
        recovery=recovery,
        console=run_relationship_p1_console_control_probe(root=tmp_path / "console"),
        gate0_baseline=_gate0_baseline(policy),
        created_at_iso=_CREATED_AT,
    )
    assert report.machinery_ready
    assert not report.gate1_passed
    steelman = next(item for item in report.checks if item.check_id == "steelman_qualification")
    assert dict(steelman.metrics)["prompt_accuracy"] == 1.0


def test_p1_gate_requires_structured_state_to_change_mirrored_choice(
    tmp_path,
) -> None:
    bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "state",
        rag_embedder=HashingEmbedder(),
    )
    choices = _choices_for_qualified_steelmen(bundle)
    dataset = load_relationship_transfer_dataset()
    for observation in dataset.observations:
        dynamic = dataset.dynamic_for_scene(observation.scene_id)
        if dynamic.split not in {
            RelationshipDatasetSplit.TRAIN,
            RelationshipDatasetSplit.VALIDATION,
        }:
            continue
        context = bundle.context(
            scene_id=observation.scene_id,
            arm=RelationshipP1Arm.STRUCTURED_STATE,
        )
        choices[context.render_user_message(observation.current_input)] = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
    policy = _MappedContextPolicy(choices)
    run = run_relationship_packet1_arms(policy, contexts=bundle)
    recovery = RelationshipP1RecoveryEvidence(
        expected_state_artifact_id=bundle.persisted_state.artifact_id,
        recovered_state_artifact_id=bundle.persisted_state.artifact_id,
        fresh_process=True,
    )
    report = assess_relationship_packet1(
        run=run,
        contexts=bundle,
        recovery=recovery,
        console=run_relationship_p1_console_control_probe(root=tmp_path / "console"),
        gate0_baseline=_gate0_baseline(policy),
        created_at_iso=_CREATED_AT,
    )
    assert report.machinery_ready
    assert not report.gate1_passed
    swap_effect = next(item for item in report.checks if item.check_id == "structured_state_user_swap_effect")
    assert dict(swap_effect.metrics)["structured_state_pair_flip_rate"] == 0.0


def test_p1_v1_report_reassessment_is_content_addressed(tmp_path) -> None:
    bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "state",
        rag_embedder=HashingEmbedder(),
    )
    policy = _MappedContextPolicy(_choices_for_qualified_steelmen(bundle))
    run = run_relationship_packet1_arms(policy, contexts=bundle)
    state_id = bundle.persisted_state.artifact_id
    report = assess_relationship_packet1(
        run=run,
        contexts=bundle,
        recovery=RelationshipP1RecoveryEvidence(
            expected_state_artifact_id=state_id,
            recovered_state_artifact_id=state_id,
            fresh_process=True,
        ),
        console=run_relationship_p1_console_control_probe(root=tmp_path / "console"),
        gate0_baseline=_gate0_baseline(policy),
        created_at_iso=_CREATED_AT,
    )
    v1_payload = json.loads(report.to_json())
    del v1_payload["artifact_id"]
    del v1_payload["source_report_artifact_id"]
    v1_payload["schema_version"] = "relationship-p1-report.v1"
    del v1_payload["config"]["minimum_structured_state_pair_flip_rate"]
    v1_payload["checks"] = [
        item for item in v1_payload["checks"] if item["check_id"] != "structured_state_user_swap_effect"
    ]
    v1_payload["artifact_id"] = sha256_json(v1_payload)
    source = tmp_path / "report.v1.json"
    source.write_text(json.dumps(v1_payload), encoding="utf-8")

    reassessed = reassess_relationship_packet1_report_v1(source_report_path=source)
    assert reassessed.source_report_artifact_id == v1_payload["artifact_id"]
    assert reassessed.gate1_passed
    assert reassessed.schema_version == "relationship-p1-report.v3"

    v1_payload["arm_metrics"]["stateless"]["accuracy"] = 1.0
    source.write_text(json.dumps(v1_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_id mismatch"):
        reassess_relationship_packet1_report_v1(source_report_path=source)
