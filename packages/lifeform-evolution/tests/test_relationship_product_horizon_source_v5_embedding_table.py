from __future__ import annotations

import copy
import json
import pathlib
import shutil

import pytest

import lifeform_evolution.relationship_product_horizon_source_v5_embedding_table as subject
from lifeform_evolution.relationship_lab_product_model_adapters import (
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    BGE_M3_WEIGHT_BYTES_SHA256,
    bge_m3_weight_pinned_embedder_identity,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SOURCE_V5 = _REPO_ROOT / (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_source_v5_admission_20260827_"
    "pd07bdb21_i8fb2e751"
)


class _FakePinnedBatchBge:
    model_source = BGE_M3_MODEL_ID
    model_revision = BGE_M3_MODEL_REVISION
    weights_sha256 = BGE_M3_WEIGHT_BYTES_SHA256
    sentence_transformers_version = BGE_M3_SENTENCE_TRANSFORMERS_VERSION
    name = bge_m3_weight_pinned_embedder_identity(
        model_revision=model_revision,
        weights_sha256=weights_sha256,
        sentence_transformers_version=sentence_transformers_version,
        identity_kind="model-adapter-v2",
    )

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def embed_many(
        self,
        texts: tuple[str, ...],
        *,
        batch_size: int,
    ) -> tuple[tuple[float, ...], ...]:
        self.calls.append((texts, batch_size))
        return tuple((float(index + 1), -1.0) for index, _text in enumerate(texts))


def _admission_receipt(
    protocol: subject.RelationshipProductHorizonSourceV5EmbeddingTableProtocol,
) -> subject._SourceV5AdmissionReceipt:
    admission = protocol.source_admission
    public_input = protocol.public_embedding_input
    return subject._SourceV5AdmissionReceipt(
        protocol_id=admission["protocol_id"],
        artifact_id=admission["artifact_id"],
        status=admission["required_status"],
        source_protocol_id=public_input["source_protocol_id_lineage_only"],
        public_plan_content_id=public_input["public_plan_content_id"],
        root_count=public_input["root_count"],
    )


def _validation_inputs(tmp_path: pathlib.Path) -> subject.SourceV5AdmissionValidationInputs:
    inputs = tmp_path / "inputs"
    return subject.SourceV5AdmissionValidationInputs(
        source_v3_admission_root=inputs / "source-v3",
        source_v4_admission_root=inputs / "source-v4",
        development_reader_root=inputs / "development-reader",
        attempt03_embedding_table_path=inputs / "attempt03/table.json",
        attempt03_reobservation_path=inputs / "attempt03/reobservation.json",
        qualification_v5_embedding_table_path=inputs / "qualification/table.json",
    )


def test_public_inventory_rebuild_needs_only_the_typed_public_plan(
    tmp_path: pathlib.Path,
) -> None:
    protocol = subject.load_relationship_product_horizon_source_v5_embedding_table_protocol()
    public_only_root = tmp_path / "public-only"
    target = public_only_root / "public/source_plan.json"
    target.parent.mkdir(parents=True)
    shutil.copyfile(_SOURCE_V5 / "public/source_plan.json", target)

    inventory = subject._load_public_inventory(
        protocol=protocol,
        source_v5_admission_root=public_only_root,
        admission_receipt=_admission_receipt(protocol),
    )

    assert len(inventory) == 3946
    assert subject.sha256_json(inventory) == (
        "13794eb5b73c9d9f6d69553278c03b0f3121b5eb450efdff41f85a1987dbb082"
    )
    assert not (public_only_root / "sealed").exists()
    assert not (public_only_root / "lineage").exists()


def test_table_builder_uses_one_fixed_batch_call_and_no_reader() -> None:
    frozen = subject.load_relationship_product_horizon_source_v5_embedding_table_protocol()
    payload = copy.deepcopy(dict(frozen.payload))
    payload["public_embedding_input"]["reader_text_unique_count"] = 3
    payload["semantic_model"]["embedding_width"] = 2
    payload["execution_contract"]["batch_size"] = 2
    protocol = subject.RelationshipProductHorizonSourceV5EmbeddingTableProtocol(
        payload=payload,
        protocol_id="a" * 64,
        raw_sha256="b" * 64,
        raw_bytes=1,
    )
    inventory = tuple(
        sorted(
            (
                subject.hashlib.sha256(text.encode("utf-8")).hexdigest(),
                text,
            )
            for text in ("public alpha", "public beta", "public gamma")
        )
    )
    fake = _FakePinnedBatchBge()

    table = subject._build_embedding_table(
        protocol=protocol,
        inventory=inventory,
        embedder=fake,
    )

    expected_texts = tuple(text for _digest, text in inventory)
    assert fake.calls == [(expected_texts, 2)]
    assert tuple(record.text for record in table.records) == expected_texts
    assert table.embedding_width == 2


def test_manifest_names_batch_vectors_and_reader_non_intervention(
    tmp_path: pathlib.Path,
) -> None:
    frozen = subject.load_relationship_product_horizon_source_v5_embedding_table_protocol()
    payload = copy.deepcopy(dict(frozen.payload))
    payload["public_embedding_input"]["reader_text_unique_count"] = 1
    payload["semantic_model"]["embedding_width"] = 2
    payload["execution_contract"]["embedding_vector_count"] = 1
    protocol = subject.RelationshipProductHorizonSourceV5EmbeddingTableProtocol(
        payload=payload,
        protocol_id="a" * 64,
        raw_sha256="b" * 64,
        raw_bytes=1,
    )
    text = "public only"
    inventory = ((subject.hashlib.sha256(text.encode("utf-8")).hexdigest(), text),)
    table = subject._build_embedding_table(
        protocol=protocol,
        inventory=inventory,
        embedder=_FakePinnedBatchBge(),
    )
    (tmp_path / "protocol.json").write_bytes(b"{}\n")
    (tmp_path / "embedding_table.json").write_bytes(table.to_json().encode("utf-8"))

    manifest = subject._manifest(
        root=tmp_path,
        protocol=protocol,
        implementation_git_commit="c" * 40,
        admission_receipt=_admission_receipt(frozen),
        table=table,
    )

    assert manifest["embedding_api_call_count"] == 1
    assert manifest["embedding_vector_count"] == 1
    assert manifest["embedding_table_record_count"] == 1
    assert manifest["reader_fit_count"] == 0
    assert manifest["reader_artifact_input_count"] == 0
    assert manifest["reader_artifact_output_count"] == 0
    assert manifest["reader_inference_count"] == 0
    assert manifest["embedding_stage_sealed_payload_count"] == 0


def test_validate_existing_replays_without_constructing_a_live_embedder(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = subject.load_relationship_product_horizon_source_v5_embedding_table_protocol()
    payload = copy.deepcopy(dict(frozen.payload))
    payload["public_embedding_input"]["reader_text_unique_count"] = 1
    payload["semantic_model"]["embedding_width"] = 2
    payload["execution_contract"]["embedding_vector_count"] = 1
    protocol = subject.RelationshipProductHorizonSourceV5EmbeddingTableProtocol(
        payload=payload,
        protocol_id="a" * 64,
        raw_sha256="b" * 64,
        raw_bytes=1,
    )
    text = "public only"
    inventory = ((subject.hashlib.sha256(text.encode("utf-8")).hexdigest(), text),)
    table = subject._build_embedding_table(
        protocol=protocol,
        inventory=inventory,
        embedder=_FakePinnedBatchBge(),
    )
    protocol_raw = (
        subject.relationship_product_horizon_source_v5_embedding_table_protocol_path().read_bytes()
    )
    (tmp_path / "protocol.json").write_bytes(protocol_raw)
    (tmp_path / "embedding_table.json").write_bytes(table.to_json().encode("utf-8"))
    receipt = _admission_receipt(frozen)
    manifest = subject._manifest(
        root=tmp_path,
        protocol=protocol,
        implementation_git_commit="c" * 40,
        admission_receipt=receipt,
        table=table,
    )
    (tmp_path / "manifest.json").write_bytes(subject._artifact_bytes(manifest))
    monkeypatch.setattr(subject, "_load_public_inventory", lambda **_kwargs: inventory)
    monkeypatch.setattr(
        subject,
        "bge_m3_public_semantic_embedder",
        lambda **_kwargs: pytest.fail("validate-existing loaded BGE"),
    )

    replayed = subject._validate_persisted_bundle(
        protocol=protocol,
        source_v5_admission_root=tmp_path / "unused-admission",
        admission_receipt=receipt,
        output_dir=tmp_path,
        expected_protocol_id=protocol.protocol_id,
        expected_artifact_id=manifest["artifact_id"],
    )

    assert replayed == manifest
    assert replayed["validate_existing_embedding_api_call_count"] == 0


def test_admission_failure_occurs_before_first_embedding_call(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = subject.load_relationship_product_horizon_source_v5_embedding_table_protocol()
    fake = _FakePinnedBatchBge()

    def fail_admission(*_args: object, **_kwargs: object) -> object:
        raise ValueError("frozen admission failed")

    monkeypatch.setattr(
        subject,
        "validate_relationship_product_horizon_source_v5_admission",
        fail_admission,
    )
    with pytest.raises(ValueError, match="frozen admission failed"):
        subject._materialize_with_embedder(
            protocol=protocol,
            source_v5_admission_root=_SOURCE_V5,
            admission_validation_inputs=_validation_inputs(tmp_path),
            output_dir=tmp_path / "must-not-exist",
            implementation_git_commit="a" * 40,
            embedder=fake,
            model_snapshot_root=tmp_path / "model-snapshot",
        )
    assert fake.calls == []
    assert not (tmp_path / "must-not-exist").exists()


def test_output_must_be_disjoint_from_every_read_only_root_before_embedding(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = subject.load_relationship_product_horizon_source_v5_embedding_table_protocol()
    inputs = _validation_inputs(tmp_path)
    source_v5 = tmp_path / "inputs/source-v5"
    snapshot = tmp_path / "inputs/model-snapshot"
    fake = _FakePinnedBatchBge()
    monkeypatch.setattr(
        subject,
        "validate_relationship_product_horizon_source_v5_admission",
        lambda *_args, **_kwargs: pytest.fail("disjointness checked too late"),
    )
    upstream_roots = (source_v5, *inputs.read_only_roots(), snapshot)

    for index, upstream in enumerate(upstream_roots):
        output = upstream / f"nested-output-{index}"
        with pytest.raises(ValueError, match="disjoint"):
            subject._materialize_with_embedder(
                protocol=protocol,
                source_v5_admission_root=source_v5,
                admission_validation_inputs=inputs,
                output_dir=output,
                implementation_git_commit="a" * 40,
                embedder=fake,
                model_snapshot_root=snapshot,
            )
        assert not output.exists()
    assert fake.calls == []


def test_admission_revalidation_uses_frozen_external_double_identity(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = subject.load_relationship_product_horizon_source_v5_embedding_table_protocol()
    manifest = json.loads((_SOURCE_V5 / "manifest.json").read_text(encoding="utf-8"))
    calls: list[tuple[pathlib.Path, dict[str, object]]] = []

    def validate(root: pathlib.Path, **kwargs: object) -> object:
        calls.append((root, kwargs))
        return manifest

    monkeypatch.setattr(
        subject,
        "validate_relationship_product_horizon_source_v5_admission",
        validate,
    )
    receipt = subject._revalidate_source_admission(
        protocol=protocol,
        source_v5_admission_root=_SOURCE_V5,
        admission_validation_inputs=_validation_inputs(tmp_path),
    )

    assert receipt == _admission_receipt(protocol)
    assert calls[0][1]["expected_protocol_id"] == protocol.source_admission[
        "protocol_id"
    ]
    assert calls[0][1]["expected_artifact_id"] == protocol.source_admission[
        "artifact_id"
    ]


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ((0.0, 0.0), "norm"),
        ((1.0, float("nan")), "finite"),
        ((1.0,), "width"),
    ],
)
def test_embedding_rows_fail_loudly(vector: tuple[float, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        subject._embedding(vector, expected_width=2)
