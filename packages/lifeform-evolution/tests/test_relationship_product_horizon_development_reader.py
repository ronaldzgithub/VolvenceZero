from __future__ import annotations

import hashlib
import pathlib
import shutil

import pytest

import lifeform_evolution.relationship_product_horizon_development_reader as subject
from lifeform_evolution.relationship_lab_product_model_adapters import (
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    BGE_M3_WEIGHT_BYTES_SHA256,
    bge_m3_weight_pinned_embedder_identity,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PREFLIGHT = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_condition_reader_qualification_preflight_v6_"
    "20260826_p723796027a64_c7381743b0"
)
_SOURCE_V4 = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_source_v4_admission_20260826_b3988b21"
)


class _FakePinnedBge:
    model_source = BGE_M3_MODEL_ID
    model_revision = BGE_M3_MODEL_REVISION
    weights_sha256 = BGE_M3_WEIGHT_BYTES_SHA256
    sentence_transformers_version = BGE_M3_SENTENCE_TRANSFORMERS_VERSION
    name = bge_m3_weight_pinned_embedder_identity(
        model_revision=BGE_M3_MODEL_REVISION,
        weights_sha256=BGE_M3_WEIGHT_BYTES_SHA256,
        sentence_transformers_version=BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
        identity_kind="model-adapter-v2",
    )

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls.append(text)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [0.0] * 1024
        values[int.from_bytes(digest[:2], "big") % 1024] = 1.0
        values[int.from_bytes(digest[2:4], "big") % 1024] += 0.5
        return tuple(values)


def _minimal_sources(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    preflight = tmp_path / "preflight"
    campaign = tmp_path / "source-v4"
    for relative_path in (
        "public/public_corpus.json",
        "sealed/condition_training_labels.json",
    ):
        target = preflight / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_PREFLIGHT / relative_path, target)
    for relative_path in ("manifest.json", "public/source_plan.json"):
        target = campaign / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_SOURCE_V4 / relative_path, target)
    return preflight, campaign


def test_public_inventory_needs_no_challenge_labels_or_sealed_campaign(
    tmp_path: pathlib.Path,
) -> None:
    preflight, campaign = _minimal_sources(tmp_path)
    protocol = subject.load_relationship_product_horizon_development_reader_protocol()

    training = subject._build_training_projection(protocol, preflight)
    inventory = subject._build_public_text_inventory(
        protocol=protocol,
        preflight_root=preflight,
        source_v4_admission_root=campaign,
    )

    assert len(training["rows"]) == 4
    assert [item["source_position"] for item in training["rows"]] == [0, 1, 2, 3]
    assert {item["condition_label"] for item in training["rows"]} == {
        "agency_displacement",
        "belonging_erasure",
    }
    assert len(inventory) == 2109
    assert not (preflight / "sealed/challenge_labels.json").exists()
    assert not (campaign / "sealed").exists()


def test_materialization_is_create_only_and_model_free_replayable(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight, campaign = _minimal_sources(tmp_path)
    output = tmp_path / "reader-input"
    fake = _FakePinnedBge()
    monkeypatch.setattr(
        subject,
        "bge_m3_public_semantic_embedder",
        lambda **_kwargs: fake,
    )

    manifest = subject.materialize_relationship_product_horizon_development_reader(
        preflight_root=preflight,
        source_v4_admission_root=campaign,
        output_dir=output,
        implementation_git_commit="a" * 40,
    )

    assert manifest["status"] == "development_unqualified_reader_materialized"
    assert manifest["embedding_call_count"] == 2109
    assert manifest["embedding_table_record_count"] == 2109
    assert manifest["training_input_count"] == 4
    assert manifest["preflight_label_free_challenge_text_count"] == 224
    assert "source_v3_public_calibration_text_count" not in manifest
    assert manifest["challenge_label_file_read_count"] == 0
    assert manifest["source_v4_sealed_file_read_count"] == 0
    assert manifest["claims"]["condition_reader_qualified"] is False
    assert manifest["claims"]["campaign_execution_authorized"] is False
    assert len(fake.calls) == len(set(fake.calls)) == 2109

    replayed = subject.validate_relationship_product_horizon_development_reader(
        preflight_root=preflight,
        source_v4_admission_root=campaign,
        output_dir=output,
        expected_protocol_id=manifest["protocol_id"],
        expected_artifact_id=manifest["artifact_id"],
    )
    assert replayed == manifest
    with pytest.raises(ValueError, match="external expected.*protocol"):
        subject.validate_relationship_product_horizon_development_reader(
            preflight_root=preflight,
            source_v4_admission_root=campaign,
            output_dir=output,
            expected_protocol_id="0" * 64,
            expected_artifact_id=manifest["artifact_id"],
        )
    with pytest.raises(FileExistsError, match="already exists"):
        subject.materialize_relationship_product_horizon_development_reader(
            preflight_root=preflight,
            source_v4_admission_root=campaign,
            output_dir=output,
            implementation_git_commit="a" * 40,
        )
    (output / "unexpected.txt").write_text("not evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or extra"):
        subject.validate_relationship_product_horizon_development_reader(
            preflight_root=preflight,
            source_v4_admission_root=campaign,
            output_dir=output,
            expected_protocol_id=manifest["protocol_id"],
            expected_artifact_id=manifest["artifact_id"],
        )
