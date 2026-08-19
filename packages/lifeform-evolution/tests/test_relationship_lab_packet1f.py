from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RelationshipTransferDataset,
    load_relationship_transfer_dataset,
)
from lifeform_evolution.relationship_lab_packet1e import (
    RelationshipP1eReport,
    load_relationship_packet1e_report,
)
from lifeform_evolution.relationship_lab_packet1f import (
    RelationshipP1fVerdict,
    assess_relationship_packet1f,
    canonical_relationship_p1f_embedder_name,
    load_relationship_packet1f_report,
    write_relationship_packet1f_report,
)


_CREATED_AT = "2026-08-20T12:00:00+00:00"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_P1E_REPORT = (
    _REPO_ROOT
    / "artifacts"
    / "relationship_lab"
    / "qwen25_3b_packet1e_v2_conditioned_top4_20260820"
    / "packet1e_report.json"
)


class _ExactSemanticEmbedder:
    def __init__(self, *, name: str, vectors: dict[str, tuple[float, ...]]) -> None:
        self._name = name
        self._vectors = vectors

    @property
    def dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return self._name

    def embed(self, text: str) -> tuple[float, ...]:
        try:
            return self._vectors[text]
        except KeyError as exc:
            raise AssertionError(f"unexpected P1f embedding text: {text}") from exc


def _source_report() -> RelationshipP1eReport:
    return load_relationship_packet1e_report(_SOURCE_P1E_REPORT)


def _dataset_and_vectors() -> tuple[
    RelationshipTransferDataset,
    dict[str, tuple[float, ...]],
    dict[str, tuple[float, ...]],
]:
    dataset = load_relationship_transfer_dataset(package_name=RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME)
    condition_ids = tuple(sorted(condition.condition_id for condition in dataset.abstract_conditions))
    condition_vectors = {
        condition_ids[0]: (1.0, 0.0),
        condition_ids[1]: (0.0, 1.0),
    }
    vectors: dict[str, tuple[float, ...]] = {}

    def bind(text: str, condition_id: str) -> None:
        vector = condition_vectors[condition_id]
        previous = vectors.setdefault(text, vector)
        assert previous == vector

    for condition in dataset.abstract_conditions:
        bind(condition.hidden_summary, condition.condition_id)
    history_bindings = dict(dataset.history_condition_bindings)
    for observation in dataset.observations:
        for history in observation.histories:
            bind(
                f"{history.user_utterance}\n{history.user_reaction}",
                history_bindings[history.event_id],
            )
        dynamic = dataset.dynamic_for_scene(observation.scene_id)
        assert dynamic.probe_condition_id is not None
        bind(observation.current_input, dynamic.probe_condition_id)
    return dataset, vectors, condition_vectors


def _embedder(
    dataset: RelationshipTransferDataset,
    vectors: dict[str, tuple[float, ...]],
) -> _ExactSemanticEmbedder:
    contract = dataset.public_evidence_contract
    assert contract is not None
    return _ExactSemanticEmbedder(
        name=canonical_relationship_p1f_embedder_name(contract),
        vectors=vectors,
    )


def test_p1f_audits_all_public_units_and_round_trips(tmp_path: Path) -> None:
    dataset, vectors, _condition_vectors = _dataset_and_vectors()
    contract = dataset.public_evidence_contract
    assert contract is not None
    report = assess_relationship_packet1f(
        dataset=dataset,
        source_p1e_report=_source_report(),
        embedder=_embedder(dataset, vectors),
        weights_sha256=contract.semantic_audit_weights_sha256,
        created_at_iso=_CREATED_AT,
    )
    assert len(report.evidence_units) == 60
    assert sum(unit.evidence_kind == "history" for unit in report.evidence_units) == 48
    assert sum(unit.evidence_kind == "probe" for unit in report.evidence_units) == 12
    assert report.correct_count == 60
    assert report.top1_accuracy == 1.0
    assert report.minimum_correct_anchor_margin == 1.0
    assert report.mean_correct_anchor_margin == 1.0
    assert report.semantic_auditor_version == ("relationship-public-evidence-auditor.v1")
    assert report.semantic_similarity == "cosine"
    assert report.human_anchor_status == "pending_before_formal"
    assert report.verdict is (RelationshipP1fVerdict.CONSUMER_PROTOCOL_FREEZE_CANDIDATE)

    json_path, markdown_path = write_relationship_packet1f_report(
        report,
        output_dir=tmp_path,
    )
    loaded = load_relationship_packet1f_report(json_path)
    assert loaded == report
    assert loaded.artifact_id == report.artifact_id
    assert markdown_path.is_file()
    encoded = json_path.read_text(encoding="utf-8")
    for condition in dataset.abstract_conditions:
        assert condition.condition_id not in encoded
        assert condition.hidden_summary not in encoded
    for observation in dataset.observations:
        assert observation.current_input not in encoded


def test_p1f_failed_semantic_unit_blocks_consumer_freeze() -> None:
    dataset, vectors, condition_vectors = _dataset_and_vectors()
    contract = dataset.public_evidence_contract
    assert contract is not None
    first_observation = sorted(dataset.observations, key=lambda item: item.scene_id)[0]
    first_history = first_observation.histories[0]
    first_text = f"{first_history.user_utterance}\n{first_history.user_reaction}"
    expected_condition = dict(dataset.history_condition_bindings)[first_history.event_id]
    competing_vector = next(
        vector for condition_id, vector in condition_vectors.items() if condition_id != expected_condition
    )
    vectors[first_text] = competing_vector
    report = assess_relationship_packet1f(
        dataset=dataset,
        source_p1e_report=_source_report(),
        embedder=_embedder(dataset, vectors),
        weights_sha256=contract.semantic_audit_weights_sha256,
        created_at_iso=_CREATED_AT,
    )
    assert report.correct_count == 59
    assert report.minimum_correct_anchor_margin == -1.0
    assert report.verdict is (RelationshipP1fVerdict.REWRITE_PUBLIC_EVIDENCE_CONTRACT_AGAIN)


def test_p1f_rejects_source_or_embedder_lineage_drift() -> None:
    dataset, vectors, _condition_vectors = _dataset_and_vectors()
    contract = dataset.public_evidence_contract
    assert contract is not None
    changed_source = dataclasses.replace(
        _source_report(),
        created_at_iso="2026-08-20T12:01:00+00:00",
    )
    with pytest.raises(ValueError, match="source P1e"):
        assess_relationship_packet1f(
            dataset=dataset,
            source_p1e_report=changed_source,
            embedder=_embedder(dataset, vectors),
            weights_sha256=contract.semantic_audit_weights_sha256,
            created_at_iso=_CREATED_AT,
        )
    wrong_name = _ExactSemanticEmbedder(name="wrong/embedder", vectors=vectors)
    with pytest.raises(ValueError, match="embedder identity"):
        assess_relationship_packet1f(
            dataset=dataset,
            source_p1e_report=_source_report(),
            embedder=wrong_name,
            weights_sha256=contract.semantic_audit_weights_sha256,
            created_at_iso=_CREATED_AT,
        )
    with pytest.raises(ValueError, match="weights"):
        assess_relationship_packet1f(
            dataset=dataset,
            source_p1e_report=_source_report(),
            embedder=_embedder(dataset, vectors),
            weights_sha256="0" * 64,
            created_at_iso=_CREATED_AT,
        )


def test_p1f_report_rejects_metric_or_artifact_tampering() -> None:
    dataset, vectors, _condition_vectors = _dataset_and_vectors()
    contract = dataset.public_evidence_contract
    assert contract is not None
    report = assess_relationship_packet1f(
        dataset=dataset,
        source_p1e_report=_source_report(),
        embedder=_embedder(dataset, vectors),
        weights_sha256=contract.semantic_audit_weights_sha256,
        created_at_iso=_CREATED_AT,
    )
    raw = json.loads(report.to_json())
    raw["metrics"]["correct_count"] = 59
    with pytest.raises(ValueError, match="aggregate metrics"):
        type(report).from_json(json.dumps(raw))
    raw = json.loads(report.to_json())
    raw["artifact_id"] = "0" * 64
    with pytest.raises(ValueError, match="artifact_id"):
        type(report).from_json(json.dumps(raw))
