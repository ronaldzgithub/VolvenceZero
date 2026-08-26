from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import pathlib
from typing import Callable

import pytest

from lifeform_domain_emogpt.relationship_condition_reader import (
    FrozenLinearRelationshipConditionReaderArtifact,
)

import lifeform_evolution.relationship_condition_reader_qualification_predictor as predictor
from lifeform_evolution.relationship_lab_product_model_adapters import (
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    BGE_M3_WEIGHT_BYTES_SHA256,
    PrecomputedPublicEmbeddingTable,
    bge_m3_weight_pinned_embedder_identity,
)


_LABELS = ("agency_displacement", "belonging_erasure")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _artifact(core: dict[str, object]) -> dict[str, object]:
    payload = dict(core)
    artifact_id = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return {**payload, "artifact_id": artifact_id}


def _raw(payload: dict[str, object]) -> bytes:
    return (_canonical_json(payload) + "\n").encode("utf-8")


def _sha(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class _InputBundle:
    child_request_path: pathlib.Path
    training_corpus_path: pathlib.Path
    predictor_request_path: pathlib.Path
    child_request_artifact_id: str
    vectors: dict[str, tuple[float, float]]
    child_request: dict[str, object]
    training_corpus: dict[str, object]
    predictor_request: dict[str, object]


def _build_inputs(
    root: pathlib.Path,
    *,
    mutate_training: Callable[[dict[str, object]], None] | None = None,
    mutate_predictor: Callable[[dict[str, object]], None] | None = None,
    mutate_child: Callable[[dict[str, object]], None] | None = None,
) -> _InputBundle:
    root.mkdir()
    protocol_id = _sha("qualification-protocol")
    execution_protocol_id = _sha("execution-protocol")
    public_corpus_artifact_id = _sha("public-corpus")
    group_split_artifact_id = _sha("group-split")
    vectors: dict[str, tuple[float, float]] = {}

    training_rows: list[dict[str, object]] = []
    for index, label in enumerate((*_LABELS, *_LABELS)):
        text = f"development text {index}"
        item_id = _sha(f"training-item-{index}")
        training_rows.append(
            {
                "item_id": item_id,
                "text": text,
                "text_sha256": _sha(text),
                "condition_label": label,
            }
        )
        vectors[text] = (1.0, -0.0) if label == _LABELS[0] else (-0.0, 1.0)
    training_rows.sort(key=lambda row: str(row["item_id"]))
    training_core: dict[str, object] = {
        "schema_version": (predictor.RELATIONSHIP_READER_PREDICTION_TRAINING_CORPUS_SCHEMA_VERSION),
        "protocol_id": protocol_id,
        "public_corpus_artifact_id": public_corpus_artifact_id,
        "labels": list(_LABELS),
        "rows": training_rows,
        "row_count": len(training_rows),
        "condition_only": True,
    }
    if mutate_training is not None:
        mutate_training(training_core)
    training = _artifact(training_core)
    training_raw = _raw(training)

    challenge_rows: list[dict[str, object]] = []
    for index in range(224):
        text = f"opaque challenge text {index:03d}"
        item_id = _sha(f"challenge-item-{index}")
        challenge_rows.append(
            {
                "item_id": item_id,
                "text": text,
                "text_sha256": _sha(text),
            }
        )
        vectors[text] = (1.0, -0.0) if index % 2 == 0 else (-0.0, 1.0)
    challenge_rows.sort(key=lambda row: str(row["item_id"]))
    predictor_core: dict[str, object] = {
        "schema_version": predictor.RELATIONSHIP_READER_PREDICTOR_REQUEST_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "public_corpus_artifact_id": public_corpus_artifact_id,
        "challenge_inputs": challenge_rows,
        "challenge_input_count": len(challenge_rows),
    }
    if mutate_predictor is not None:
        mutate_predictor(predictor_core)
    predictor_request = _artifact(predictor_core)
    predictor_raw = _raw(predictor_request)

    child_core: dict[str, object] = {
        "schema_version": (predictor.RELATIONSHIP_READER_PREDICTION_CHILD_REQUEST_SCHEMA_VERSION),
        "protocol_id": protocol_id,
        "execution_protocol_id": execution_protocol_id,
        "public_corpus_artifact_id": public_corpus_artifact_id,
        "training_corpus_artifact_id": training["artifact_id"],
        "training_corpus_raw_sha256": _sha(training_raw),
        "training_corpus_raw_bytes": len(training_raw),
        "predictor_request_artifact_id": predictor_request["artifact_id"],
        "predictor_request_raw_sha256": _sha(predictor_raw),
        "predictor_request_raw_bytes": len(predictor_raw),
        "group_split_artifact_id": group_split_artifact_id,
        "semantic_model": {
            "model_id": BGE_M3_MODEL_ID,
            "model_revision": BGE_M3_MODEL_REVISION,
            "weights_sha256": BGE_M3_WEIGHT_BYTES_SHA256,
            "sentence_transformers_version": BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
            "embedding_width": 2,
            "device": "cuda",
            "network_allowed": False,
            "stub_allowed": False,
        },
        "reader": {
            "schema_version": "relationship-condition-reader-artifact.v2",
            "solver": "unit_normalized_class_centroid_linear",
            "solver_version": "relationship-condition-centroid-solver.v1",
            "labels": list(_LABELS),
        },
        "required_live_embedding_count": 228,
    }
    if mutate_child is not None:
        mutate_child(child_core)
    child_request = _artifact(child_core)

    child_path = root / "child_request.json"
    training_path = root / "training_corpus.json"
    predictor_path = root / "predictor_request.json"
    child_path.write_bytes(_raw(child_request))
    training_path.write_bytes(training_raw)
    predictor_path.write_bytes(predictor_raw)
    return _InputBundle(
        child_request_path=child_path,
        training_corpus_path=training_path,
        predictor_request_path=predictor_path,
        child_request_artifact_id=str(child_request["artifact_id"]),
        vectors=vectors,
        child_request=child_request,
        training_corpus=training,
        predictor_request=predictor_request,
    )


class _FakePinnedEmbedder:
    def __init__(
        self,
        vectors: dict[str, tuple[float, float]],
        *,
        model_revision: str = BGE_M3_MODEL_REVISION,
    ) -> None:
        self._vectors = vectors
        self.calls: list[str] = []
        self.model_source = BGE_M3_MODEL_ID
        self.model_revision = model_revision
        self.weights_sha256 = BGE_M3_WEIGHT_BYTES_SHA256
        self.sentence_transformers_version = BGE_M3_SENTENCE_TRANSFORMERS_VERSION
        self.name = bge_m3_weight_pinned_embedder_identity(
            model_revision=self.model_revision,
            weights_sha256=self.weights_sha256,
            sentence_transformers_version=self.sentence_transformers_version,
            identity_kind="model-adapter-v2",
        )

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls.append(text)
        return self._vectors[text]


def _run_fake(
    bundle: _InputBundle,
    *,
    output_root: pathlib.Path,
    run_ordinal: int,
    run_nonce: str,
    model_revision: str = BGE_M3_MODEL_REVISION,
) -> tuple[dict[str, object], _FakePinnedEmbedder]:
    fake = _FakePinnedEmbedder(bundle.vectors, model_revision=model_revision)
    result = predictor._run_relationship_condition_reader_prediction_child_core(
        child_request_path=bundle.child_request_path,
        expected_child_request_artifact_id=bundle.child_request_artifact_id,
        training_corpus_path=bundle.training_corpus_path,
        predictor_request_path=bundle.predictor_request_path,
        output_root=output_root,
        run_ordinal=run_ordinal,
        run_nonce=run_nonce,
        embedder_factory=lambda _spec: fake,
    )
    return dict(result), fake


def test_prediction_child_two_fresh_runs_are_byte_exact_and_embed_228_once(
    tmp_path: pathlib.Path,
) -> None:
    bundle = _build_inputs(tmp_path / "inputs")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first, first_embedder = _run_fake(
        bundle,
        output_root=first_root,
        run_ordinal=1,
        run_nonce=_sha("first-nonce"),
    )
    second, second_embedder = _run_fake(
        bundle,
        output_root=second_root,
        run_ordinal=2,
        run_nonce=_sha("second-nonce"),
    )

    assert first["live_embedding_call_count"] == 228
    assert second["live_embedding_call_count"] == 228
    assert len(first_embedder.calls) == len(set(first_embedder.calls)) == 228
    assert len(second_embedder.calls) == len(set(second_embedder.calls)) == 228
    assert first["embedding_table_artifact_id"] == second["embedding_table_artifact_id"]
    assert first["reader_artifact_id"] == second["reader_artifact_id"]
    assert first["prediction_ledger_artifact_id"] == second["prediction_ledger_artifact_id"]
    for filename in (
        "embedding_table.json",
        "reader_artifact.json",
        "prediction_ledger.json",
    ):
        assert (first_root / filename).read_bytes() == (second_root / filename).read_bytes()
    assert (first_root / "process_attestation.json").read_bytes() != (
        second_root / "process_attestation.json"
    ).read_bytes()

    table = PrecomputedPublicEmbeddingTable.from_json((first_root / "embedding_table.json").read_text(encoding="utf-8"))
    assert len(table.records) == 228
    assert not any(value.startswith("-0x0.") for record in table.records for value in record.embedding_hex)
    reader = FrozenLinearRelationshipConditionReaderArtifact.from_json(
        (first_root / "reader_artifact.json").read_bytes()
    )
    assert reader.embedding_width == 2
    ledger = json.loads((first_root / "prediction_ledger.json").read_text("utf-8"))
    assert ledger["row_count"] == 224
    assert ledger["execution_protocol_id"] == _sha("execution-protocol")
    assert ledger["challenge_labels_present"] is False
    assert ledger["qualification_scored"] is False
    assert all(
        isinstance(row["confidence_hex"], str)
        and isinstance(row["normalized_margin_hex"], str)
        and all(isinstance(score["score_hex"], str) for score in row["candidate_scores"])
        for row in ledger["rows"]
    )
    assert {path.name for path in first_root.iterdir()} == {
        "embedding_table.json",
        "reader_artifact.json",
        "prediction_ledger.json",
        "process_attestation.json",
        "manifest.json",
    }
    attestation = json.loads((first_root / "process_attestation.json").read_text(encoding="utf-8"))
    assert attestation["schema_version"] == ("relationship-condition-reader-prediction-process-attestation.v5")
    assert attestation["os_security_boundary"] is False
    assert attestation["execution_protocol_id"] == _sha("execution-protocol")
    assert attestation["python_executable"]
    assert isinstance(attestation["argv"], list)
    assert attestation["working_directory"]
    assert isinstance(attestation["sys_path"], list)
    assert attestation["environment_contract"]["projected_keys"] == [
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "KMP_DUPLICATE_LIB_OK",
        "KMP_INIT_AT_FORK",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "PYTHONPYCACHEPREFIX",
        "PYTHONSAFEPATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUTF8",
        "TOKENIZERS_PARALLELISM",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRANSFORMERS_OFFLINE",
    ]
    assert attestation["environment_contract"]["schema_version"] == (
        "relationship-condition-reader-prediction-environment.v4"
    )
    assert attestation["environment_contract"]["all_environment_values_hashed"] is True
    assert attestation["environment_contract"]["unlisted_environment_variables_recorded"] is True
    assert attestation["environment_contract"]["key_name_canonicalization"] == "windows_uppercase"
    assert attestation["environment_contract"]["complete_environment_observation_scope"] == (
        "cpython_visible_mapping"
    )
    assert attestation["environment_contract"]["raw_win32_environment_block_attested"] is False
    assert attestation["environment_key_names"] == sorted(attestation["environment_key_names"])
    assert set(attestation["environment_value_sha256s"]) == set(attestation["environment_key_names"])
    assert all(
        isinstance(value, str) and len(value) == 64 for value in attestation["environment_value_sha256s"].values()
    )
    origins = attestation["loaded_file_backed_module_origins"]
    assert isinstance(origins, list)
    assert any(
        item["module_name"] == "lifeform_evolution.relationship_condition_reader_qualification_predictor"
        for item in origins
    )
    assert [item["module_name"] for item in origins] == sorted(item["module_name"] for item in origins)
    assert len({item["module_name"] for item in origins}) == len(origins)
    assert all(set(item) == {"module_name", "origin"} for item in origins)
    assert all(pathlib.Path(item["origin"]).is_absolute() for item in origins)
    assert isinstance(attestation["volvence_zero_namespace_search_locations"], list)
    assert attestation["volvence_zero_namespace_search_locations"]
    assert isinstance(attestation["interpreter_flags"], dict)
    assert "safe_path" in attestation["interpreter_flags"]
    assert "pycache_prefix" in attestation
    assert isinstance(attestation["bootstrap_import_roots"], list)
    assert {item["module_name"] for item in attestation["forbidden_module_observations"]} == {
        "lifeform_domain_emogpt.lab.relationship_product_pilot_source",
        "lifeform_evolution.relationship_condition_reader_qualification",
        "lifeform_evolution.relationship_lab_product_horizon",
    }
    assert not any(item["imported_by_worker"] for item in attestation["forbidden_module_observations"])
    manifest = json.loads((first_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_protocol_id"] == _sha("execution-protocol")
    assert first["execution_protocol_id"] == _sha("execution-protocol")


def test_public_worker_exposes_no_embedder_factory() -> None:
    parameters = inspect.signature(predictor.run_relationship_condition_reader_prediction_child).parameters
    assert "embedder_factory" not in parameters
    assert set(parameters) == {
        "child_request_path",
        "expected_child_request_artifact_id",
        "training_corpus_path",
        "predictor_request_path",
        "output_root",
        "run_ordinal",
        "run_nonce",
        "bge_snapshot_path",
    }


def test_prediction_child_rejects_preexisting_output_before_factory(
    tmp_path: pathlib.Path,
) -> None:
    bundle = _build_inputs(tmp_path / "inputs")
    output_root = tmp_path / "existing"
    output_root.mkdir()
    (output_root / "foreign.json").write_text("do not replace", encoding="utf-8")
    factory_called = False

    def factory(_spec: object) -> _FakePinnedEmbedder:
        nonlocal factory_called
        factory_called = True
        return _FakePinnedEmbedder(bundle.vectors)

    with pytest.raises(FileExistsError, match="already exists"):
        predictor._run_relationship_condition_reader_prediction_child_core(
            child_request_path=bundle.child_request_path,
            expected_child_request_artifact_id=bundle.child_request_artifact_id,
            training_corpus_path=bundle.training_corpus_path,
            predictor_request_path=bundle.predictor_request_path,
            output_root=output_root,
            run_ordinal=1,
            run_nonce=_sha("nonce"),
            embedder_factory=factory,
        )
    assert factory_called is False
    assert (output_root / "foreign.json").read_text(encoding="utf-8") == "do not replace"


@pytest.mark.parametrize("variant", ["pretty", "duplicate"])
def test_prediction_child_rejects_noncanonical_or_duplicate_input_json(
    tmp_path: pathlib.Path,
    variant: str,
) -> None:
    bundle = _build_inputs(tmp_path / "inputs")
    if variant == "pretty":
        bundle.predictor_request_path.write_text(
            json.dumps(bundle.predictor_request, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        message = "not canonical"
    else:
        canonical = bundle.predictor_request_path.read_text(encoding="utf-8")
        artifact_id = bundle.predictor_request["artifact_id"]
        bundle.predictor_request_path.write_text(
            canonical.replace(
                f'"artifact_id":"{artifact_id}"',
                f'"artifact_id":"{artifact_id}","artifact_id":"{artifact_id}"',
                1,
            ),
            encoding="utf-8",
            newline="\n",
        )
        message = "duplicate JSON key"
    with pytest.raises(ValueError, match=message):
        _run_fake(
            bundle,
            output_root=tmp_path / "output",
            run_ordinal=1,
            run_nonce=_sha("nonce"),
        )


def test_prediction_child_rejects_unknown_input_key_even_when_readdressed(
    tmp_path: pathlib.Path,
) -> None:
    bundle = _build_inputs(
        tmp_path / "inputs",
        mutate_training=lambda payload: payload.update({"unexpected": True}),
    )
    with pytest.raises(ValueError, match="training corpus keys mismatch"):
        _run_fake(
            bundle,
            output_root=tmp_path / "output",
            run_ordinal=1,
            run_nonce=_sha("nonce"),
        )


def test_prediction_child_rejects_raw_source_pin_tamper(
    tmp_path: pathlib.Path,
) -> None:
    bundle = _build_inputs(
        tmp_path / "inputs",
        mutate_child=lambda payload: payload.update({"predictor_request_raw_sha256": "0" * 64}),
    )
    with pytest.raises(ValueError, match="predictor request raw sha256 pin mismatch"):
        _run_fake(
            bundle,
            output_root=tmp_path / "output",
            run_ordinal=1,
            run_nonce=_sha("nonce"),
        )


@pytest.mark.parametrize("violation", ["count", "label", "text_hash"])
def test_prediction_child_rejects_count_label_and_text_identity_drift(
    tmp_path: pathlib.Path,
    violation: str,
) -> None:
    def mutate(payload: dict[str, object]) -> None:
        rows = payload["rows"]
        assert isinstance(rows, list)
        if violation == "count":
            rows.pop()
            payload["row_count"] = len(rows)
        elif violation == "label":
            row = rows[0]
            assert isinstance(row, dict)
            row["condition_label"] = "unknown_condition"
        else:
            row = rows[0]
            assert isinstance(row, dict)
            row["text_sha256"] = "0" * 64

    bundle = _build_inputs(tmp_path / "inputs", mutate_training=mutate)
    messages = {
        "count": "exactly four training rows",
        "label": "balanced two per class",
        "text_hash": "text_sha256 mismatch",
    }
    with pytest.raises(ValueError, match=messages[violation]):
        _run_fake(
            bundle,
            output_root=tmp_path / "output",
            run_ordinal=1,
            run_nonce=_sha("nonce"),
        )


def test_prediction_child_rejects_embedder_identity_before_live_calls(
    tmp_path: pathlib.Path,
) -> None:
    bundle = _build_inputs(tmp_path / "inputs")
    wrong_revision = "0" * 40
    fake = _FakePinnedEmbedder(bundle.vectors, model_revision=wrong_revision)
    with pytest.raises(ValueError, match="embedder identity mismatch"):
        predictor._run_relationship_condition_reader_prediction_child_core(
            child_request_path=bundle.child_request_path,
            expected_child_request_artifact_id=bundle.child_request_artifact_id,
            training_corpus_path=bundle.training_corpus_path,
            predictor_request_path=bundle.predictor_request_path,
            output_root=tmp_path / "output",
            run_ordinal=1,
            run_nonce=_sha("nonce"),
            embedder_factory=lambda _spec: fake,
        )
    assert fake.calls == []
    assert not (tmp_path / "output").exists()


def test_prediction_output_codec_rejects_tampered_artifact(
    tmp_path: pathlib.Path,
) -> None:
    bundle = _build_inputs(tmp_path / "inputs")
    output_root = tmp_path / "output"
    _run_fake(
        bundle,
        output_root=output_root,
        run_ordinal=1,
        run_nonce=_sha("nonce"),
    )
    ledger_path = output_root / "prediction_ledger.json"
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["rows"][0]["confidence_hex"] = (0.0).hex()
    ledger_path.write_bytes(_raw(payload))
    with pytest.raises(ValueError, match="canonical artifact id mismatch"):
        predictor._load_canonical_artifact(
            ledger_path,
            predictor.RELATIONSHIP_READER_PREDICTION_LEDGER_SCHEMA_VERSION,
        )
