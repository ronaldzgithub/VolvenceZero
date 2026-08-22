from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from lifeform_domain_emogpt.lab import (
    RelationshipAction,
    load_relationship_transfer_dataset,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    RelationshipConditionPrototype,
    RelationshipConditionReaderArtifact,
)
from lifeform_evolution.relationship_lab_packet1m_qualification import (
    RelationshipP1mQualificationArm,
    RelationshipP1mQualificationDecision,
    RelationshipP1mQualificationProtocol,
    RelationshipP1mQualificationReport,
    RelationshipP1mQualificationVerdict,
    RelationshipP1mQwenReadout,
    RelationshipP1mArmMetrics,
    frozen_snapshot_manifest_sha256,
    load_relationship_p1m_qualification_report,
    load_relationship_p1m_qualification_protocol,
    relationship_p1m_arm_metrics,
    render_relationship_p1m_forced_choice_request,
    wilson_one_sided_lower,
    write_relationship_p1m_qualification_protocol,
    write_relationship_p1m_qualification_report,
)


_SHA = "a" * 64
_PRESENCE = RelationshipAction.STAY_PRESENT_WITHOUT_PROBE
_SPACE = RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION


def test_frozen_snapshot_manifest_digest_is_platform_neutral(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "config.json").write_bytes(b"{}\n")
    (nested / "weights.bin").write_bytes(b"weights")
    expected_manifest = (
        (
            "config.json",
            3,
            "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
        ),
        (
            "nested/weights.bin",
            7,
            "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
        ),
    )
    encoded = json.dumps(
        expected_manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert frozen_snapshot_manifest_sha256(tmp_path) == hashlib.sha256(encoded).hexdigest()


def _reader_artifact() -> RelationshipConditionReaderArtifact:
    return RelationshipConditionReaderArtifact(
        embedding_model_id="fixture-bge",
        embedding_weights_sha256=_SHA,
        prototypes=(
            RelationshipConditionPrototype(
                label="agency_displacement",
                summary="当事人的声音、选择或决定被别人越过和取代。",
            ),
            RelationshipConditionPrototype(
                label="belonging_erasure",
                summary="当事人在共同经历和关系网络中的位置被遗漏。",
            ),
        ),
    )


def _protocol() -> RelationshipP1mQualificationProtocol:
    return RelationshipP1mQualificationProtocol(
        frozen_at_iso="2026-08-22T15:00:00+08:00",
        source_p1k_report_artifact_id=_SHA,
        source_generation_attestation_id="b" * 64,
        source_generation_protocol_id="c" * 64,
        source_transport_id="d" * 64,
        source_seed_inventory_sha256="e" * 64,
        package_name="relationship_transfer_p1m_v1",
        dataset_fingerprint="f" * 64,
        pair_count=24,
        scene_count=48,
        qwen_model_source="Qwen/Qwen2.5-3B-Instruct",
        qwen_model_revision="main",
        qwen_model_id="fixture-qwen",
        qwen_weights_sha256="1" * 64,
        qwen_snapshot_sha256="2" * 64,
        qwen_device="cpu",
        qwen_torch_dtype="bfloat16",
        prompt_sha256="3" * 64,
        request_template_sha256="4" * 64,
        token_a_id=32,
        token_b_id=33,
        scoring_method="exact_first_assistant_token_logits_A_vs_B",
        qwen_config_sha256="5" * 64,
        bge_model_source="BAAI/bge-m3",
        bge_model_revision="main",
        bge_weights_sha256=_SHA,
        reader_artifact=_reader_artifact(),
        rag_top_k=4,
        plan_artifact_id="6" * 64,
        planned_qwen_readouts=96,
        planned_structured_readouts=48,
        qualification_inputs_observed_before_freeze=0,
        qwen_outputs_observed_before_freeze=0,
        structured_outputs_observed_before_freeze=0,
        first_qualification_attempt_only=True,
        evaluation_feedback_allowed=False,
    )


def test_p1m_protocol_roundtrip_freezes_zero_output_first_attempt(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    path = write_relationship_p1m_qualification_protocol(
        protocol,
        output_dir=tmp_path,
    )
    loaded = load_relationship_p1m_qualification_protocol(path)
    assert loaded == protocol
    assert loaded.protocol_id == protocol.protocol_id
    assert loaded.qwen_outputs_observed_before_freeze == 0
    assert loaded.structured_outputs_observed_before_freeze == 0
    assert loaded.reader_artifact.artifact_id == protocol.reader_artifact.artifact_id


def test_p1m_forced_choice_request_rotates_mapping_without_truth() -> None:
    observation = load_relationship_transfer_dataset(
        package_name="relationship_transfer_v3"
    ).observations[0]
    order = tuple(item.event_id for item in observation.histories)
    forward = render_relationship_p1m_forced_choice_request(
        observation=observation,
        ordered_event_ids=order,
        candidate_a=_PRESENCE,
        candidate_b=_SPACE,
    )
    reverse = render_relationship_p1m_forced_choice_request(
        observation=observation,
        ordered_event_ids=order,
        candidate_a=_SPACE,
        candidate_b=_PRESENCE,
    )
    assert "A = stay_present_without_probe" in forward
    assert "B = respect_space_with_return_option" in forward
    assert "A = respect_space_with_return_option" in reverse
    assert "B = stay_present_without_probe" in reverse
    for sealed in ("preferred_action", "policy_id", "probe_condition_id"):
        assert sealed not in forward
        assert sealed not in reverse


def test_p1m_exact_logit_readout_rejects_letter_bias_as_hidden_mapping() -> None:
    readout = RelationshipP1mQwenReadout(
        protocol_id=_SHA,
        record_index=0,
        arm=RelationshipP1mQualificationArm.PROMPT_STEELMAN,
        scene_id="scene-a",
        model_input_sha256="b" * 64,
        logit_a=2.0,
        logit_b=1.0,
        chosen_label="A",
        chosen_action_id=_SPACE,
        prompt_tokens=100,
    )
    assert readout.valid
    assert readout.chosen_label == "A"
    assert readout.chosen_action_id is _SPACE


def _metric_decisions() -> tuple[RelationshipP1mQualificationDecision, ...]:
    decisions = []
    for pair_index in range(24):
        pair_id = f"pair-{pair_index:02d}"
        expected = (_PRESENCE, _SPACE)
        chosen = expected if pair_index < 18 else (_PRESENCE, _PRESENCE)
        for member_index in range(2):
            decisions.append(
                RelationshipP1mQualificationDecision(
                    protocol_id=_SHA,
                    arm=RelationshipP1mQualificationArm.PROMPT_STEELMAN,
                    record_index=len(decisions),
                    scene_id=f"scene-{pair_index:02d}-{member_index}",
                    mirror_pair_id=pair_id,
                    readout_artifact_id=f"{len(decisions) + 1:064x}",
                    chosen_action_id=chosen[member_index],
                    expected_action_id=expected[member_index],
                )
            )
    return tuple(decisions)


def test_p1m_metrics_use_one_sided_wilson_lower_for_accuracy_and_flip() -> None:
    metrics = relationship_p1m_arm_metrics(
        _metric_decisions(),
        arm=RelationshipP1mQualificationArm.PROMPT_STEELMAN,
    )
    assert metrics.correct_decisions == 42
    assert metrics.accuracy == 0.875
    assert metrics.pair_flips == 18
    assert metrics.pair_flip_rate == 0.75
    assert math.isclose(
        metrics.accuracy_wilson_lower,
        wilson_one_sided_lower(42, 48),
    )
    assert math.isclose(
        metrics.pair_flip_wilson_lower,
        wilson_one_sided_lower(18, 24),
    )
    assert metrics.accuracy_wilson_lower >= 0.5
    assert metrics.pair_flip_wilson_lower > 0.35


def _arm_metrics(
    arm: RelationshipP1mQualificationArm,
    *,
    correct: int,
    flips: int,
) -> RelationshipP1mArmMetrics:
    return RelationshipP1mArmMetrics(
        arm=arm,
        decisions=48,
        valid_decisions=48,
        correct_decisions=correct,
        accuracy=correct / 48,
        accuracy_wilson_lower=wilson_one_sided_lower(correct, 48),
        mirrored_pairs=24,
        pair_flips=flips,
        pair_flip_rate=flips / 24,
        pair_flip_wilson_lower=wilson_one_sided_lower(flips, 24),
    )


def _failed_report() -> RelationshipP1mQualificationReport:
    return RelationshipP1mQualificationReport(
        created_at_iso="2026-08-22T18:00:00+08:00",
        protocol_id="1" * 64,
        plan_artifact_id="2" * 64,
        dataset_fingerprint="3" * 64,
        qwen_readout_ledger_sha256="4" * 64,
        structured_readout_ledger_sha256="5" * 64,
        qwen_decision_ledger_sha256="6" * 64,
        structured_decision_ledger_sha256="7" * 64,
        arm_metrics=(
            _arm_metrics(
                RelationshipP1mQualificationArm.PROMPT_STEELMAN,
                correct=24,
                flips=0,
            ),
            _arm_metrics(
                RelationshipP1mQualificationArm.RAG_STEELMAN,
                correct=24,
                flips=0,
            ),
            _arm_metrics(
                RelationshipP1mQualificationArm.STRUCTURED_STATE,
                correct=46,
                flips=24,
            ),
        ),
        verdict=RelationshipP1mQualificationVerdict.BASELINE_TOO_WEAK,
        qualification_passed=False,
        scenario_versioning_closed=True,
        evaluation_feedback_to_system=False,
    )


def test_p1m_report_roundtrip_rejects_artifact_or_derived_metric_drift(
    tmp_path: Path,
) -> None:
    report = _failed_report()
    path = write_relationship_p1m_qualification_report(
        report,
        output_dir=tmp_path,
    )
    assert load_relationship_p1m_qualification_report(path) == report

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["arm_metrics"][0]["accuracy"] = 0.75
    path.write_text(json.dumps(raw), encoding="utf-8")
    try:
        load_relationship_p1m_qualification_report(path)
    except ValueError as exc:
        assert "metric accuracy drift" in str(exc)
    else:
        raise AssertionError("P1m report metric tampering was accepted")
