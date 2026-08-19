from __future__ import annotations

import hashlib
import json
import pathlib
import runpy

import pytest

from companion_ref_harness import HashingEmbedder
from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME,
    RelationshipAction,
    RelationshipDatasetSplit,
    load_relationship_transfer_dataset,
    relationship_transfer_package_dir,
    sha256_json,
)
from lifeform_evolution.relationship_lab_baseline import StatelessActionCompletion
from lifeform_evolution.relationship_lab_contexts import (
    RelationshipP1Arm,
    RelationshipP1RagCandidateSurface,
    build_relationship_p1_context_bundle,
)
from lifeform_evolution.relationship_lab_gate0 import (
    FrozenBaselineAttestation,
    Gate0CalibrationConfig,
    run_relationship_gate0_calibration,
)
from lifeform_evolution.relationship_lab_packet1 import relationship_p1_prompt_path
from lifeform_evolution.relationship_lab_packet1b import (
    RelationshipP1bReadoutProfile,
    RelationshipP1bReport,
    RelationshipP1bVerdict,
    relationship_p1b_readout_prompt_path,
    render_relationship_p1b_readout_request,
    run_relationship_packet1b_arms,
)
from lifeform_evolution.relationship_lab_packet1e import (
    RelationshipP1eVerdict,
    assess_relationship_packet1e,
    load_relationship_p1e_consumer_protocol,
    load_relationship_packet1e_report,
    validate_relationship_p1e_local_lineage,
    write_relationship_packet1e_report,
)


_CREATED_AT = "2026-08-20T08:00:00+00:00"


class _MappedConditionedPolicy:
    model_id = "frozen-p1e-test-model"
    weights_sha256 = sha256_json("p1e-test-weights")
    generation_config_sha256 = sha256_json("p1e-test-generation")
    prompt_sha256 = hashlib.sha256(
        relationship_p1_prompt_path(RelationshipP1Arm.STATELESS).read_bytes()
    ).hexdigest()

    def __init__(self, raw_by_user_message: dict[str, str]) -> None:
        self._raw_by_user_message = raw_by_user_message

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        assert current_input and seed >= 0
        return StatelessActionCompletion(
            raw_output=json.dumps(
                {"action_id": RelationshipAction.NEUTRAL_NOOP.value}
            ),
            chosen_action_id=RelationshipAction.NEUTRAL_NOOP,
            prompt_tokens=80,
            completion_tokens=8,
        )

    def choose_from_messages(
        self,
        *,
        messages: tuple[dict[str, str], ...],
        seed: int,
    ) -> StatelessActionCompletion:
        assert seed >= 0
        raw = self._raw_by_user_message[messages[-1]["content"]]
        return StatelessActionCompletion(
            raw_output=raw,
            chosen_action_id=None,
            prompt_tokens=max(1, len(messages[-1]["content"]) // 4),
            completion_tokens=18,
        )

    def count_tokens(self, text: str) -> int:
        return max(0, len(text) // 4)


def _score_json(action: RelationshipAction) -> str:
    stay, space = {
        RelationshipAction.STAY_PRESENT_WITHOUT_PROBE: (1, -1),
        RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION: (-1, 1),
    }[action]
    return json.dumps(
        {
            "stay_present_without_probe_score": stay,
            "respect_space_with_return_option_score": space,
        },
        separators=(",", ":"),
    )


def _baseline(*, valid_decisions: int = 24) -> FrozenBaselineAttestation:
    protocol = load_relationship_p1e_consumer_protocol()
    return FrozenBaselineAttestation(
        arm_id="stateless",
        dataset_fingerprint=protocol.dataset_fingerprint,
        model_id=protocol.model_id,
        weights_sha256=sha256_json("p1e-candidate-weights"),
        prompt_sha256=protocol.stateless_prompt_sha256,
        generation_config_sha256=protocol.expected_generation_config_sha256,
        seed_schedule_sha256=sha256_json(protocol.baseline_seed_schedule),
        decision_ledger_sha256=sha256_json("p1e-gate0-ledger"),
        evaluated_split="calibration",
        valid_decisions=valid_decisions,
        correct_decisions=10,
        evaluated_decisions=24,
        context_tokens_total=2048,
        hidden_test_opened=False,
        frozen_at_iso=_CREATED_AT,
    )


def _arm_metrics(
    *,
    accuracy: float,
    pair_flip_rate: float,
) -> tuple[tuple[str, object], ...]:
    decisions = 8
    return tuple(
        sorted(
            {
                "accuracy": accuracy,
                "completion_tokens_total": 80,
                "correct_decisions": int(accuracy * decisions),
                "decisions": decisions,
                "pair_flip_rate": pair_flip_rate,
                "pair_groups": 4,
                "prompt_tokens_total": 800,
                "readouts": decisions,
                "valid_decisions": decisions,
                "valid_pair_groups": 4,
                "valid_rate": 1.0,
                "valid_readouts": decisions,
            }.items()
        )
    )


def _p1b_report(
    baseline: FrozenBaselineAttestation,
    *,
    verdict: RelationshipP1bVerdict,
    prompt_accuracy: float,
    prompt_pair_flip: float,
    rag_accuracy: float,
    rag_pair_flip: float,
    structured_pair_flip: float,
    machinery_ready: bool = True,
) -> RelationshipP1bReport:
    protocol = load_relationship_p1e_consumer_protocol()
    saturated_arms = tuple(
        arm
        for arm, accuracy in (
            ("prompt-steelman", prompt_accuracy),
            ("rag-steelman", rag_accuracy),
        )
        if accuracy > 0.875
    )
    return RelationshipP1bReport(
        created_at_iso=_CREATED_AT,
        dataset_fingerprint=protocol.dataset_fingerprint,
        context_bundle_artifact_id=sha256_json("p1e-context-bundle"),
        evaluated_context_surface_sha256=(
            protocol.evaluated_context_surface_sha256
        ),
        background_templates_sha256=protocol.background_templates_sha256,
        rag_config_sha256=protocol.rag_config_sha256,
        seed_schedule_sha256=sha256_json(protocol.p1b_seed_schedule),
        p1_gate_config_sha256=protocol.p1_gate_config_sha256,
        model_id=protocol.model_id,
        weights_sha256=baseline.weights_sha256,
        generation_config_sha256=baseline.generation_config_sha256,
        gate0_baseline_attestation_id=baseline.artifact_id,
        readout_prompt_sha256=protocol.readout_prompt_sha256,
        readout_request_template_sha256=(
            protocol.readout_request_template_sha256
        ),
        readout_schema_sha256=protocol.readout_schema_sha256,
        compiler_version=protocol.compiler_version,
        run_artifact_id=sha256_json("p1e-p1b-run"),
        p1_report_artifact_id=sha256_json("p1e-p1-report"),
        readout_ledger_sha256=sha256_json("p1e-readout-ledger"),
        verdict=verdict,
        p1_machinery_ready=machinery_ready,
        all_readouts_valid=True,
        saturated_arms=saturated_arms,
        arm_metrics=(
            (
                "prompt-steelman",
                _arm_metrics(
                    accuracy=prompt_accuracy,
                    pair_flip_rate=prompt_pair_flip,
                ),
            ),
            (
                "rag-steelman",
                _arm_metrics(
                    accuracy=rag_accuracy,
                    pair_flip_rate=rag_pair_flip,
                ),
            ),
            (
                "structured-state",
                _arm_metrics(
                    accuracy=0.75,
                    pair_flip_rate=structured_pair_flip,
                ),
            ),
        ),
    )


def test_p1e_protocol_is_frozen_before_model_output_and_rejects_weakening() -> None:
    protocol = load_relationship_p1e_consumer_protocol()
    validate_relationship_p1e_local_lineage(protocol)
    assert protocol.package_name == RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME
    assert protocol.histories_per_user == 4
    assert protocol.current_message_participates
    assert protocol.all_four_histories_available
    assert protocol.rag_top_k == 4
    assert protocol.rag_candidate_surface == "relationship_outcomes_only"
    assert protocol.readout_profile == "v2_condition_aware"
    assert not protocol.formal_hidden_test_opened

    raw = json.loads(protocol.to_json())
    raw["run_config"]["rag_top_k"] = 2
    raw["protocol_id"] = sha256_json("tampered-p1e")
    with pytest.raises(ValueError, match="top-k"):
        type(protocol).from_json(json.dumps(raw))


def test_p1e_contexts_publish_all_four_histories_without_truth(tmp_path) -> None:
    dataset = load_relationship_transfer_dataset(
        package_name=RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME
    )
    bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "state",
        rag_embedder=HashingEmbedder(),
        dataset=dataset,
        background_depths=(0, 8),
        rag_top_k=4,
        rag_candidate_surface=(
            RelationshipP1RagCandidateSurface.RELATIONSHIP_OUTCOMES_ONLY
        ),
    )
    sealed_tokens = {
        item.condition_id for item in dataset.abstract_conditions
    } | {item.policy_id for item in dataset.policy_profiles}
    persisted_counts = {
        records for _scope, _digest, records in bundle.persisted_state.structured_scope_digests
    }
    assert persisted_counts == {12}
    for observation in dataset.observations:
        signal_ids = {item.event_id for item in observation.histories}
        prompt = bundle.context(
            scene_id=observation.scene_id,
            arm=RelationshipP1Arm.PROMPT_STEELMAN,
        )
        rag = bundle.context(
            scene_id=observation.scene_id,
            arm=RelationshipP1Arm.RAG_STEELMAN,
        )
        structured = bundle.context(
            scene_id=observation.scene_id,
            arm=RelationshipP1Arm.STRUCTURED_STATE,
        )
        assert prompt.context_text.count("[public relationship outcome evidence]") == 4
        assert set(rag.source_evidence_refs) == signal_ids
        assert rag.context_text.count("[public relationship outcome evidence]") == 4
        assert len(structured.source_evidence_refs) == 4
        assert structured.context_text.count("[public relationship outcome evidence]") == 4
        for context in (prompt, rag, structured):
            assert not any(token in context.context_text for token in sealed_tokens)


def test_p1e_conditioned_profile_drives_all_three_contextual_arms(tmp_path) -> None:
    protocol = load_relationship_p1e_consumer_protocol()
    dataset = load_relationship_transfer_dataset(
        package_name=RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME
    )
    bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "state",
        rag_embedder=HashingEmbedder(),
        dataset=dataset,
        background_depths=(0, 8),
        rag_top_k=4,
        rag_candidate_surface=(
            RelationshipP1RagCandidateSurface.RELATIONSHIP_OUTCOMES_ONLY
        ),
    )
    profile = RelationshipP1bReadoutProfile.V2_CONDITION_AWARE
    mapping: dict[str, str] = {}
    for observation in dataset.observations:
        dynamic = dataset.dynamic_for_scene(observation.scene_id)
        if dynamic.split not in {
            RelationshipDatasetSplit.TRAIN,
            RelationshipDatasetSplit.VALIDATION,
        }:
            continue
        for arm in (
            RelationshipP1Arm.PROMPT_STEELMAN,
            RelationshipP1Arm.RAG_STEELMAN,
            RelationshipP1Arm.STRUCTURED_STATE,
        ):
            context = bundle.context(scene_id=observation.scene_id, arm=arm)
            mapping[
                render_relationship_p1b_readout_request(
                    context_text=context.context_text,
                    current_input=observation.current_input,
                    profile=profile,
                )
            ] = _score_json(dynamic.preferred_action)
    run = run_relationship_packet1b_arms(
        _MappedConditionedPolicy(mapping),
        contexts=bundle,
        dataset=dataset,
        readout_profile=profile,
    )
    assert len(run.readouts) == 24
    assert all(item.valid for item in run.readouts)
    assert run.readout_prompt_sha256 == protocol.readout_prompt_sha256
    assert (
        run.readout_request_template_sha256
        == protocol.readout_request_template_sha256
    )
    for arm in (
        RelationshipP1Arm.PROMPT_STEELMAN,
        RelationshipP1Arm.RAG_STEELMAN,
        RelationshipP1Arm.STRUCTURED_STATE,
    ):
        assert run.action_run.arm_metrics(arm)["accuracy"] == 1.0
        assert run.action_run.arm_metrics(arm)["pair_flip_rate"] == 1.0


@pytest.mark.parametrize(
    ("p1b_kwargs", "expected"),
    (
        (
            {
                "verdict": RelationshipP1bVerdict.QUALIFIED,
                "prompt_accuracy": 0.75,
                "prompt_pair_flip": 0.5,
                "rag_accuracy": 0.75,
                "rag_pair_flip": 0.5,
                "structured_pair_flip": 0.75,
            },
            RelationshipP1eVerdict.FORMAL_PREREG_FREEZE_CANDIDATE,
        ),
        (
            {
                "verdict": RelationshipP1bVerdict.DATASET_SATURATED,
                "prompt_accuracy": 1.0,
                "prompt_pair_flip": 1.0,
                "rag_accuracy": 0.75,
                "rag_pair_flip": 0.5,
                "structured_pair_flip": 0.75,
            },
            RelationshipP1eVerdict.SCENARIO_STILL_SATURATED,
        ),
        (
            {
                "verdict": RelationshipP1bVerdict.BASELINE_UNDERQUALIFIED,
                "prompt_accuracy": 0.5,
                "prompt_pair_flip": 0.25,
                "rag_accuracy": 0.5,
                "rag_pair_flip": 0.25,
                "structured_pair_flip": 0.25,
            },
            RelationshipP1eVerdict.REWRITE_PUBLIC_EVIDENCE_CONTRACT,
        ),
    ),
)
def test_p1e_routes_frozen_qualification_results(
    tmp_path,
    p1b_kwargs,
    expected,
) -> None:
    protocol = load_relationship_p1e_consumer_protocol()
    baseline = _baseline()
    gate0 = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(),
        baseline=baseline,
        package_root=relationship_transfer_package_dir(protocol.package_name),
        created_at_iso=_CREATED_AT,
    )
    assert gate0.gate0_passed
    report = assess_relationship_packet1e(
        protocol=protocol,
        baseline=baseline,
        gate0_report=gate0,
        p1b_report=_p1b_report(baseline, **p1b_kwargs),
        created_at_iso=_CREATED_AT,
    )
    assert report.verdict is expected
    paths = write_relationship_packet1e_report(report, output_dir=tmp_path)
    assert load_relationship_packet1e_report(paths[0]) == report


def test_p1e_runner_checkpoint_is_resumable_and_scope_bound(tmp_path) -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    runner = runpy.run_path(
        str(repo_root / "scripts" / "run_relationship_lab_packet1e.py"),
        run_name="relationship_p1e_runner_test",
    )
    protocol_id = sha256_json("p1e-checkpoint-test")
    stage, artifacts = runner["_load_or_initialize_checkpoint"](
        tmp_path / "run",
        protocol_id=protocol_id,
    )
    assert stage == "initialized"
    assert artifacts == {}
    runner["_write_checkpoint"](
        tmp_path / "run",
        protocol_id=protocol_id,
        stage="gate0_running",
        artifacts={"active_gate0_dir": "gate0_candidate"},
    )
    resumed_stage, resumed = runner["_load_or_initialize_checkpoint"](
        tmp_path / "run",
        protocol_id=protocol_id,
    )
    assert resumed_stage == "gate0_running"
    assert resumed == {"active_gate0_dir": "gate0_candidate"}
    with pytest.raises(ValueError, match="escapes output root"):
        runner["_artifact_path"](
            tmp_path / "run",
            {"p1e_report": "../outside.json"},
            "p1e_report",
        )


def test_p1e_prompt_uses_current_semantics_without_sealed_truth() -> None:
    dataset = load_relationship_transfer_dataset(
        package_name=RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME
    )
    prompt = relationship_p1b_readout_prompt_path(
        RelationshipP1bReadoutProfile.V2_CONDITION_AWARE
    ).read_text(encoding="utf-8")
    assert "当前消息必须参与归纳" in prompt
    assert "不得忽略当前消息后对全部历史做动作多数票" in prompt
    assert "不要依赖单个词" in prompt
    assert not any(
        item.condition_id in prompt for item in dataset.abstract_conditions
    )
    assert not any(item.policy_id in prompt for item in dataset.policy_profiles)
