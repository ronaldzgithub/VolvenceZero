from __future__ import annotations

import hashlib
import json
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
    RelationshipP1Arm,
    build_relationship_p1_context_bundle,
    run_relationship_p1_console_control_probe,
)
from lifeform_evolution.relationship_lab_gate0 import FrozenBaselineAttestation
from lifeform_evolution.relationship_lab_packet1 import (
    RelationshipP1RecoveryEvidence,
    assess_relationship_packet1,
    relationship_p1_prompt_path,
)
from lifeform_evolution.relationship_lab_packet1b import (
    RELATIONSHIP_P1B_RAG_TOP_K,
    RelationshipP1bVerdict,
    assess_relationship_packet1b,
    compile_relationship_evidence_scores,
    load_relationship_packet1b_report,
    parse_relationship_evidence_scores,
    render_relationship_p1b_readout_request,
    run_relationship_packet1b_arms,
    write_relationship_packet1b_artifacts,
)


_CREATED_AT = "2026-08-19T14:00:00+00:00"


class _MappedReadoutPolicy:
    model_id = "frozen-p1b-test-model"
    weights_sha256 = sha256_json("p1b-test-weights")
    generation_config_sha256 = sha256_json("p1b-test-generation")
    prompt_sha256 = hashlib.sha256(relationship_p1_prompt_path(RelationshipP1Arm.STATELESS).read_bytes()).hexdigest()

    def __init__(self, raw_by_user_message: dict[str, str]) -> None:
        self._raw_by_user_message = raw_by_user_message

    def choose(self, *, current_input: str, seed: int) -> StatelessActionCompletion:
        assert current_input
        assert seed >= 0
        raw = json.dumps({"action_id": RelationshipAction.NEUTRAL_NOOP.value})
        return StatelessActionCompletion(
            raw_output=raw,
            chosen_action_id=RelationshipAction.NEUTRAL_NOOP,
            prompt_tokens=100,
            completion_tokens=9,
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
    scores = {
        RelationshipAction.STAY_PRESENT_WITHOUT_PROBE: (1, -1),
        RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION: (-1, 1),
    }
    stay, space = scores[action]
    return json.dumps(
        {
            "stay_present_without_probe_score": stay,
            "respect_space_with_return_option_score": space,
        },
        separators=(",", ":"),
    )


def _readout_mapping(bundle, *, one_error_per_steelman: bool) -> dict[str, str]:
    dataset = load_relationship_transfer_dataset()
    mapping: dict[str, str] = {}
    first_scene_by_arm: dict[RelationshipP1Arm, str] = {}
    for arm in (
        RelationshipP1Arm.PROMPT_STEELMAN,
        RelationshipP1Arm.RAG_STEELMAN,
        RelationshipP1Arm.STRUCTURED_STATE,
    ):
        eligible = tuple(
            observation
            for observation in dataset.observations
            if dataset.dynamic_for_scene(observation.scene_id).split
            in {RelationshipDatasetSplit.TRAIN, RelationshipDatasetSplit.VALIDATION}
        )
        first_scene_by_arm[arm] = eligible[0].scene_id
        for observation in eligible:
            dynamic = dataset.dynamic_for_scene(observation.scene_id)
            action = dynamic.preferred_action
            if (
                one_error_per_steelman
                and arm
                in {
                    RelationshipP1Arm.PROMPT_STEELMAN,
                    RelationshipP1Arm.RAG_STEELMAN,
                }
                and observation.scene_id == first_scene_by_arm[arm]
            ):
                action = (
                    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION
                    if action is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
                    else RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
                )
            context = bundle.context(scene_id=observation.scene_id, arm=arm)
            mapping[
                render_relationship_p1b_readout_request(
                    context_text=context.context_text,
                    current_input=observation.current_input,
                )
            ] = _score_json(action)
    return mapping


def _gate0_baseline(policy: _MappedReadoutPolicy) -> FrozenBaselineAttestation:
    dataset = load_relationship_transfer_dataset()
    return FrozenBaselineAttestation(
        arm_id="stateless",
        dataset_fingerprint=dataset.dataset_fingerprint,
        model_id=policy.model_id,
        weights_sha256=policy.weights_sha256,
        prompt_sha256=policy.prompt_sha256,
        generation_config_sha256=policy.generation_config_sha256,
        seed_schedule_sha256=sha256_json((101,)),
        decision_ledger_sha256=sha256_json("p1b-gate0-ledger"),
        evaluated_split="calibration",
        valid_decisions=24,
        correct_decisions=4,
        evaluated_decisions=24,
        context_tokens_total=1000,
        hidden_test_opened=False,
        frozen_at_iso="2026-08-19T10:00:00+00:00",
    )


def _assess(tmp_path, bundle, policy, run):
    state_id = bundle.persisted_state.artifact_id
    p1_report = assess_relationship_packet1(
        run=run.action_run,
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
    return p1_report, assess_relationship_packet1b(
        run=run,
        p1_report=p1_report,
        contexts=bundle,
        created_at_iso=_CREATED_AT,
    )


def test_p1b_score_parser_and_compiler_are_strict() -> None:
    raw = _score_json(RelationshipAction.STAY_PRESENT_WITHOUT_PROBE)
    assert parse_relationship_evidence_scores(raw) == (1, -1)
    assert (
        compile_relationship_evidence_scores(stay_score=1, space_score=-1)
        is RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
    )
    assert compile_relationship_evidence_scores(stay_score=0, space_score=0) is RelationshipAction.NEUTRAL_NOOP
    assert parse_relationship_evidence_scores('{"stay_present_without_probe_score":true}') == (
        None,
        None,
    )
    assert parse_relationship_evidence_scores(
        '{"stay_present_without_probe_score":1,"respect_space_with_return_option_score":-1,"extra":0}'
    ) == (None, None)
    with pytest.raises(ValueError):
        compile_relationship_evidence_scores(stay_score=2, space_score=0)
    rendered = render_relationship_p1b_readout_request(
        context_text="typed history",
        current_input="same current message",
    )
    assert "typed history" in rendered
    assert "same current message" in rendered
    assert rendered.endswith("</output_contract>")
    assert "expected_action" not in rendered


def test_p1b_detects_when_a_strong_readout_saturates_the_dataset(tmp_path) -> None:
    bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "state",
        rag_embedder=HashingEmbedder(),
        rag_top_k=RELATIONSHIP_P1B_RAG_TOP_K,
    )
    policy = _MappedReadoutPolicy(_readout_mapping(bundle, one_error_per_steelman=False))
    events: list[tuple[str, str, str]] = []
    run = run_relationship_packet1b_arms(
        policy,
        contexts=bundle,
        readout_observer=lambda item: events.append(("readout", item.arm.value, item.scene_id)),
        decision_observer=lambda item: events.append(("decision", item.arm.value, item.scene_id)),
    )
    assert len(run.readouts) == 24
    assert len(run.action_run.decisions) == 32
    first_readout_payload = run.readouts[0].to_payload()
    assert first_readout_payload["schema_version"] == "relationship-p1b-readout.v2"
    assert first_readout_payload["request_template_sha256"] == (run.readout_request_template_sha256)
    assert "expected_action_id" not in first_readout_payload
    for index, event in enumerate(events):
        if event[0] != "readout":
            continue
        assert events[index + 1] == ("decision", event[1], event[2])
    p1_report, report = _assess(tmp_path, bundle, policy, run)
    assert p1_report.machinery_ready
    assert not p1_report.gate1_passed
    assert report.verdict is RelationshipP1bVerdict.DATASET_SATURATED
    assert report.saturated_arms == ("prompt-steelman", "rag-steelman")
    machinery_failed = assess_relationship_packet1b(
        run=run,
        p1_report=replace(
            p1_report,
            machinery_ready=False,
            gate1_passed=False,
        ),
        contexts=bundle,
        created_at_iso=_CREATED_AT,
    )
    assert machinery_failed.verdict is RelationshipP1bVerdict.BASELINE_UNDERQUALIFIED
    assert machinery_failed.saturated_arms == ("prompt-steelman", "rag-steelman")


def test_p1b_qualifies_non_saturated_steelmen_and_writes_artifacts(
    tmp_path,
) -> None:
    bundle = build_relationship_p1_context_bundle(
        state_root=tmp_path / "state",
        rag_embedder=HashingEmbedder(),
        rag_top_k=RELATIONSHIP_P1B_RAG_TOP_K,
    )
    policy = _MappedReadoutPolicy(_readout_mapping(bundle, one_error_per_steelman=True))
    run = run_relationship_packet1b_arms(policy, contexts=bundle)
    with pytest.raises(ValueError, match="readout lineage"):
        replace(
            run,
            readouts=(
                replace(
                    run.readouts[0],
                    request_template_sha256=sha256_json("tampered-template"),
                ),
                *run.readouts[1:],
            ),
        )
    p1_report, report = _assess(tmp_path, bundle, policy, run)
    assert p1_report.gate1_passed
    assert report.verdict is RelationshipP1bVerdict.QUALIFIED
    assert report.gate1_passed
    paths = write_relationship_packet1b_artifacts(
        run=run,
        report=report,
        output_dir=tmp_path / "evidence",
    )
    assert len(paths) == 4
    assert json.loads(paths[2].read_text(encoding="utf-8"))["artifact_id"] == (report.artifact_id)
    assert load_relationship_packet1b_report(paths[2]) == report
    tampered = json.loads(paths[2].read_text(encoding="utf-8"))
    tampered["model_id"] = "tampered-model"
    paths[2].write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_id mismatch"):
        load_relationship_packet1b_report(paths[2])
    markdown = paths[3].read_text(encoding="utf-8")
    assert "readout_request_template_sha256" in markdown
    assert "| prompt-steelman |" in markdown
    with pytest.raises(FileExistsError):
        write_relationship_packet1b_artifacts(
            run=run,
            report=report,
            output_dir=tmp_path / "evidence",
        )
