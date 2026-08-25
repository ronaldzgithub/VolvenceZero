from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import platform
import sys
from typing import Mapping

import pytest

from lifeform_domain_emogpt.relationship_condition_reader import (
    FrozenLinearRelationshipConditionReaderRuntime,
    LabeledRelationshipConditionEmbeddingRow,
    build_frozen_linear_relationship_condition_reader_artifact,
)
from volvence_zero.social_cognition import relationship_condition_readout_to_payload

import lifeform_evolution.relationship_condition_reader_qualification_executor as executor
import lifeform_evolution.relationship_condition_reader_qualification_execution_protocol as execution_protocol
import lifeform_evolution.relationship_condition_reader_qualification_scorer as scorer
from lifeform_evolution.relationship_lab_product_model_adapters import (
    PrecomputedPublicEmbeddingRecord,
    PrecomputedPublicEmbeddingTable,
    PrecomputedPublicSemanticEmbedder,
    bge_m3_weight_pinned_embedder_identity,
)


_QUALIFICATION_PROTOCOL_ID = hashlib.sha256(b"qualification-protocol").hexdigest()
_EXECUTION_PROTOCOL_ID = hashlib.sha256(b"execution-protocol").hexdigest()
_RUN_NONCE = hashlib.sha256(b"executor-run").hexdigest()
_SOURCE_TREE_ARTIFACT_ID = hashlib.sha256(b"source-tree").hexdigest()
_BGE_TREE_ARTIFACT_ID = hashlib.sha256(b"bge-tree").hexdigest()
_RUNTIME_IDENTITY_ARTIFACT_ID = hashlib.sha256(b"runtime-identity").hexdigest()
_POST_ANCHOR_INTEGRITY_RECEIPT_ID = hashlib.sha256(b"post-anchor-integrity-receipt").hexdigest()


def _import_binding() -> executor.QualificationChildImportBinding:
    repository_root = pathlib.Path(__file__).resolve().parents[3]
    module_paths = {
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/relationship_condition_reader.py",
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_executor.py",
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_predictor.py",
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_runtime_binding.py",
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_lab_product_model_adapters.py",
        "packages/vz-contracts/src/volvence_zero/social_cognition.py",
        "packages/vz-temporal/src/volvence_zero/temporal/__init__.py",
    }
    frozen_entries: dict[str, Mapping[str, object]] = {}
    for relative in sorted(module_paths):
        raw = (repository_root / pathlib.PurePosixPath(relative)).read_bytes()
        frozen_entries[relative] = {
            "raw_sha256": _sha(raw),
            "raw_bytes": len(raw),
        }
    source_roots = tuple(
        repository_root / pathlib.PurePosixPath(relative)
        for relative in (
            "packages/lifeform-domain-emogpt/src",
            "packages/lifeform-evolution/src",
            "packages/vz-contracts/src",
            "packages/vz-temporal/src",
        )
    )
    return executor.build_qualification_child_import_binding(
        python_executable=pathlib.Path(sys.executable).resolve(),
        repository_root=repository_root,
        repository_source_roots=source_roots,
        frozen_source_entries=frozen_entries,
        frozen_site_packages_root=(pathlib.Path(sys.executable).resolve().parent / "Lib" / "site-packages").resolve(),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _artifact(core: Mapping[str, object]) -> dict[str, object]:
    payload = dict(core)
    return {
        **payload,
        "artifact_id": hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
    }


def _raw(payload: Mapping[str, object]) -> bytes:
    return (_canonical_json(payload) + "\n").encode("utf-8")


def _sha(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write(path: pathlib.Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


@dataclass(frozen=True)
class _PreflightBundle:
    root: pathlib.Path
    execution_root: pathlib.Path
    manifest_artifact_id: str
    publication_request_artifact_id: str
    challenge_labels_path: pathlib.Path
    group_split_path: pathlib.Path
    predictor_rows: tuple[Mapping[str, str], ...]


def _build_preflight(tmp_path: pathlib.Path) -> _PreflightBundle:
    root = tmp_path / "preflight"
    execution_root = tmp_path / "execution"

    training_rows = sorted(
        (
            {
                "item_id": _sha(f"training-item-{index}"),
                "text": f"development text {index}",
                "text_sha256": _sha(f"development text {index}"),
            }
            for index in range(4)
        ),
        key=lambda row: str(row["item_id"]),
    )
    challenge_rows = sorted(
        (
            {
                "item_id": _sha(f"challenge-item-{index:03d}"),
                "text": f"opaque challenge text {index:03d}",
                "text_sha256": _sha(f"opaque challenge text {index:03d}"),
            }
            for index in range(224)
        ),
        key=lambda row: str(row["item_id"]),
    )
    public_corpus = _artifact(
        {
            "schema_version": ("relationship-condition-reader-qualification-public-corpus.v1"),
            "protocol_id": _QUALIFICATION_PROTOCOL_ID,
            "training_inputs": training_rows,
            "challenge_inputs": challenge_rows,
            "training_input_count": 4,
            "challenge_input_count": 224,
            "exact_text_overlap_count": 0,
            "predictor_projection": ("opaque_item_id_exact_text_and_text_sha256_only"),
        }
    )
    predictor_request = _artifact(
        {
            "schema_version": ("relationship-condition-reader-qualification-predictor-request.v1"),
            "protocol_id": _QUALIFICATION_PROTOCOL_ID,
            "public_corpus_artifact_id": public_corpus["artifact_id"],
            "challenge_inputs": challenge_rows,
            "challenge_input_count": 224,
        }
    )

    labels_by_training_id = {
        str(row["item_id"]): ("agency_displacement" if index % 2 == 0 else "belonging_erasure")
        for index, row in enumerate(training_rows)
    }
    training_labels = _artifact(
        {
            "schema_version": ("relationship-condition-reader-qualification-training-labels.v1"),
            "protocol_id": _QUALIFICATION_PROTOCOL_ID,
            "public_corpus_artifact_id": public_corpus["artifact_id"],
            "rows": [
                {
                    "item_id": row["item_id"],
                    "text_sha256": row["text_sha256"],
                    "condition_label": labels_by_training_id[str(row["item_id"])],
                    "source_position": index,
                }
                for index, row in enumerate(training_rows)
            ],
            "row_count": 4,
            "labels": ["agency_displacement", "belonging_erasure"],
            "condition_only": True,
            "action_outcome_pe_credit_evaluation_present": False,
        }
    )

    challenge_label_rows: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []
    for group_index in range(28):
        item_rows = challenge_rows[group_index * 8 : (group_index + 1) * 8]
        group_id = _sha(f"group-{group_index:02d}")
        label = "agency_displacement" if group_index < 14 else "belonging_erasure"
        groups.append(
            {
                "group_id": group_id,
                "item_ids": sorted(str(row["item_id"]) for row in item_rows),
                "row_count": 8,
                "condition_label": label,
            }
        )
        for surface_index, row in enumerate(item_rows):
            challenge_label_rows.append(
                {
                    "item_id": row["item_id"],
                    "text_sha256": row["text_sha256"],
                    "condition_label": label,
                    "group_id": group_id,
                    "subject_index": surface_index,
                    "surface_kind": f"surface-{surface_index}",
                    "source_position": group_index,
                    "source_session_id": f"session-{group_index}",
                }
            )
    challenge_label_rows.sort(key=lambda row: str(row["item_id"]))
    groups.sort(key=lambda group: str(group["group_id"]))
    challenge_labels = _artifact(
        {
            "schema_version": ("relationship-condition-reader-qualification-challenge-labels.v1"),
            "protocol_id": _QUALIFICATION_PROTOCOL_ID,
            "public_corpus_artifact_id": public_corpus["artifact_id"],
            "rows": challenge_label_rows,
            "row_count": 224,
            "label_release_condition": "prediction_ledger_create_only_fsynced",
        }
    )
    group_split = _artifact(
        {
            "schema_version": ("relationship-condition-reader-qualification-group-split.v1"),
            "protocol_id": _QUALIFICATION_PROTOCOL_ID,
            "training_item_ids": sorted(str(row["item_id"]) for row in training_rows),
            "challenge_item_ids": sorted(str(row["item_id"]) for row in challenge_rows),
            "challenge_groups": groups,
            "challenge_group_count": 28,
            "rows_per_challenge_group": 8,
            "training_challenge_text_overlap_count": 0,
            "statistical_independence_claim": False,
            "grouping_owner": "qualification_preflight",
            "grouping_contract": ("surface_kind_and_source_position_across_voice_variants.v1"),
            "group_level_evaluation_unit_count": 28,
        }
    )
    publication = _artifact(
        {
            "schema_version": ("relationship-condition-reader-qualification-publication-request.v1"),
            "protocol_id": _QUALIFICATION_PROTOCOL_ID,
            "protocol_filename": "relationship_condition_reader_qualification_v1.json",
            "protocol_raw_sha256": _sha(b"protocol\n"),
            "protocol_raw_bytes": len(b"protocol\n"),
            "public_corpus_artifact_id": public_corpus["artifact_id"],
            "predictor_request_artifact_id": predictor_request["artifact_id"],
            "training_labels_artifact_id": training_labels["artifact_id"],
            "challenge_labels_artifact_id": challenge_labels["artifact_id"],
            "group_split_artifact_id": group_split["artifact_id"],
            "proposed_execution_root": str(execution_root.resolve()),
            "proposed_execution_root_exists_at_prepare": False,
            "external_observation_required": True,
            "requested_publication_visibility": "public",
            "public_gist_created": False,
            "qualification_execution_authorized": False,
        }
    )

    raw_by_path = {
        "protocol.json": b"protocol\n",
        "public/public_corpus.json": _raw(public_corpus),
        "public/predictor_request.json": _raw(predictor_request),
        "public/publication_request.json": _raw(publication),
        "sealed/condition_training_labels.json": _raw(training_labels),
        "sealed/challenge_labels.json": _raw(challenge_labels),
        "sealed/group_split.json": _raw(group_split),
    }
    artifact_by_path: dict[str, object] = {
        "protocol.json": None,
        "public/public_corpus.json": public_corpus["artifact_id"],
        "public/predictor_request.json": predictor_request["artifact_id"],
        "public/publication_request.json": publication["artifact_id"],
        "sealed/condition_training_labels.json": training_labels["artifact_id"],
        "sealed/challenge_labels.json": challenge_labels["artifact_id"],
        "sealed/group_split.json": group_split["artifact_id"],
    }
    for relative_path, raw in raw_by_path.items():
        _write(root / relative_path, raw)
    manifest = _artifact(
        {
            "schema_version": ("relationship-condition-reader-qualification-preflight-manifest.v1"),
            "protocol_id": _QUALIFICATION_PROTOCOL_ID,
            "files": [
                {
                    "path": relative_path,
                    "raw_sha256": _sha(raw_by_path[relative_path]),
                    "raw_bytes": len(raw_by_path[relative_path]),
                    "artifact_id": artifact_by_path[relative_path],
                }
                for relative_path in (
                    "protocol.json",
                    "public/public_corpus.json",
                    "public/predictor_request.json",
                    "public/publication_request.json",
                    "sealed/condition_training_labels.json",
                    "sealed/challenge_labels.json",
                    "sealed/group_split.json",
                )
            ],
            "file_count": 7,
            "model_output_count": 0,
            "external_public_anchor_created": False,
            "qualification_execution_authorized": False,
        }
    )
    _write(root / "manifest.json", _raw(manifest))
    return _PreflightBundle(
        root=root,
        execution_root=execution_root,
        manifest_artifact_id=str(manifest["artifact_id"]),
        publication_request_artifact_id=str(publication["artifact_id"]),
        challenge_labels_path=(root / "sealed/challenge_labels.json").resolve(),
        group_split_path=(root / "sealed/group_split.json").resolve(),
        predictor_rows=tuple(challenge_rows),
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows path equality folds case")
def test_publication_request_root_join_requires_exact_text_on_windows(
    tmp_path: pathlib.Path,
) -> None:
    bundle = _build_preflight(tmp_path)
    publication_path = bundle.root / "public" / "publication_request.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    case_variant_root = bundle.execution_root.with_name(bundle.execution_root.name.upper())
    publication["proposed_execution_root"] = str(case_variant_root)
    publication["artifact_id"] = _sha("case-variant-publication")

    with pytest.raises(ValueError, match="execution root text mismatch"):
        executor._validate_publication_request(
            publication,
            qualification_protocol_id=_QUALIFICATION_PROTOCOL_ID,
            expected_artifact_id=str(publication["artifact_id"]),
            expected_execution_root=bundle.execution_root.resolve(),
        )


def _file_receipt(
    relative_path: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    raw = _raw(payload)
    return {
        "path": relative_path,
        "raw_sha256": _sha(raw),
        "raw_bytes": len(raw),
        "artifact_id": payload["artifact_id"],
    }


class _FakeLauncher:
    def __init__(self, *, mode: str = "success") -> None:
        self.mode = mode
        self.calls: list[executor._PredictionChildLaunchSpec] = []

    def __call__(
        self,
        spec: executor._PredictionChildLaunchSpec,
    ) -> executor._PredictionChildLaunchResult:
        self.calls.append(spec)
        if self.mode == "nonzero" and spec.run_ordinal == 1:
            return self._result(spec=spec, process_id=1001, exit_code=9)
        if self.mode == "not_exited" and spec.run_ordinal == 1:
            return self._result(
                spec=spec,
                process_id=1001,
                exit_code=None,
                process_exited=False,
                job_object_empty=False,
            )
        if self.mode == "job_busy" and spec.run_ordinal == 1:
            return self._result(
                spec=spec,
                process_id=1001,
                exit_code=0,
                job_object_empty=False,
            )
        self._write_child_outputs(spec, mismatch=self.mode == "mismatch")
        return self._result(
            spec=spec,
            process_id=1000 + spec.run_ordinal,
            exit_code=0,
        )

    def _result(
        self,
        *,
        spec: executor._PredictionChildLaunchSpec,
        process_id: int,
        exit_code: int | None,
        process_exited: bool = True,
        job_object_empty: bool = True,
    ) -> executor._PredictionChildLaunchResult:
        _environment, environment_projection, environment_contract_id = self._environment_for(spec)
        empty_sha = _sha(b"")
        empty_capture = executor._BoundedStreamCapture(
            raw_sha256=empty_sha,
            total_bytes=0,
            retained_prefix=b"",
            retained_prefix_sha256=empty_sha,
            retained_prefix_bytes=0,
            prefix_truncated=False,
        )
        return executor._PredictionChildLaunchResult(
            process_id=process_id,
            process_argv=executor._prediction_child_process_argv(spec),
            exit_code=exit_code,
            process_exited=process_exited,
            job_object_empty=job_object_empty,
            environment_contract_id=environment_contract_id,
            environment_projection=environment_projection,
            creation_flags=executor._FORMAL_CREATION_FLAGS,
            shell=False,
            close_fds=True,
            process_created_suspended=True,
            job_assigned_before_resume=True,
            initial_thread_resume_previous_count=1,
            job_limit_flags=executor._FORMAL_JOB_LIMIT_FLAGS,
            job_active_process_limit=1,
            stdout_capture=empty_capture,
            stderr_capture=empty_capture,
            source_capsule_used=False,
            repository_import_path_used=True,
            bge_snapshot_tree_verified=False,
        )

    @staticmethod
    def _environment_for(
        spec: executor._PredictionChildLaunchSpec,
    ) -> tuple[
        dict[str, str],
        tuple[executor._EnvironmentValueReceipt, ...],
        str,
    ]:
        return executor._build_formal_child_environment(
            {
                "CUDA_PATH": "C:\\poison-toolkit",
                "CUDA_VISIBLE_DEVICES": "7",
                "PATH": "C:\\ambient-shadow;C:\\other-python",
                "PYTHONPATH": "C:\\ambient-shadow",
                "SystemRoot": "C:\\Windows",
                "SECRET_TOKEN": "must-not-cross",
            },
            import_binding=spec.import_binding,
            pycache_prefix=spec.pycache_prefix,
        )

    def _write_child_outputs(
        self,
        spec: executor._PredictionChildLaunchSpec,
        *,
        mismatch: bool,
    ) -> None:
        environment, _projection, _contract_id = self._environment_for(spec)
        child = json.loads(spec.child_request_path.read_text(encoding="utf-8"))
        training = json.loads(spec.training_corpus_path.read_text(encoding="utf-8"))
        predictor = json.loads(spec.predictor_request_path.read_text(encoding="utf-8"))
        width = int(child["semantic_model"]["embedding_width"])
        zero = (0.0).hex()
        agency_vector = ((1.0).hex(), zero, *(zero for _ in range(width - 2)))
        belonging_vector = (zero, (1.0).hex(), *(zero for _ in range(width - 2)))
        records: list[PrecomputedPublicEmbeddingRecord] = []
        for row in training["rows"]:
            vector = agency_vector if row["condition_label"] == "agency_displacement" else belonging_vector
            records.append(
                PrecomputedPublicEmbeddingRecord(
                    text=row["text"],
                    embedding_hex=tuple(vector),
                )
            )
        for index, row in enumerate(predictor["challenge_inputs"]):
            vector = agency_vector if index % 2 == 0 else belonging_vector
            if mismatch and spec.run_ordinal == 2 and index == 0:
                vector = ((0.5).hex(), *vector[1:])
            records.append(
                PrecomputedPublicEmbeddingRecord(
                    text=row["text"],
                    embedding_hex=tuple(vector),
                )
            )
        table_object = PrecomputedPublicEmbeddingTable(
            source_embedder_name=bge_m3_weight_pinned_embedder_identity(
                model_revision=child["semantic_model"]["model_revision"],
                weights_sha256=child["semantic_model"]["weights_sha256"],
                sentence_transformers_version=child["semantic_model"]["sentence_transformers_version"],
                identity_kind="model-adapter-v2",
            ),
            embedding_width=width,
            records=tuple(sorted(records, key=lambda record: (record.text_sha256, record.text))),
        )
        table = table_object.to_payload()
        reader_object = build_frozen_linear_relationship_condition_reader_artifact(
            embedding_model_id=child["semantic_model"]["model_id"],
            embedding_model_revision=child["semantic_model"]["model_revision"],
            embedding_weights_sha256=child["semantic_model"]["weights_sha256"],
            embedding_runtime_version=child["semantic_model"]["sentence_transformers_version"],
            embedding_width=width,
            labels=tuple(child["reader"]["labels"]),
            condition_training_corpus_artifact_id=child["training_corpus_artifact_id"],
            condition_training_corpus_raw_sha256=child["training_corpus_raw_sha256"],
            group_split_artifact_id=child["group_split_artifact_id"],
            rows=tuple(
                LabeledRelationshipConditionEmbeddingRow(
                    example_id=row["item_id"],
                    condition_label=row["condition_label"],
                    embedding_hex=(
                        agency_vector if row["condition_label"] == "agency_displacement" else belonging_vector
                    ),
                )
                for row in training["rows"]
            ),
        )
        reader = reader_object.to_payload()
        replay_runtime = FrozenLinearRelationshipConditionReaderRuntime(
            artifact=reader_object,
            embedder=PrecomputedPublicSemanticEmbedder(table_object),
        )
        ledger_rows: list[dict[str, object]] = []
        for row in predictor["challenge_inputs"]:
            readout = relationship_condition_readout_to_payload(replay_runtime.read_condition(row["text"]))
            ledger_rows.append(
                {
                    "item_id": row["item_id"],
                    "text_sha256": row["text_sha256"],
                    "condition_label": readout["condition_label"],
                    "confidence_hex": float(readout["confidence"]).hex(),
                    "normalized_margin_hex": float(readout["normalized_margin"]).hex(),
                    "candidate_scores": [
                        {
                            "label": score["label"],
                            "score_hex": float(score["score"]).hex(),
                        }
                        for score in readout["candidate_scores"]
                    ],
                    "reader_artifact_id": reader["artifact_id"],
                    "source_observation_sha256": row["text_sha256"],
                }
            )
        ledger = _artifact(
            {
                "schema_version": ("relationship-condition-reader-prediction-ledger.v1"),
                "protocol_id": child["protocol_id"],
                "execution_protocol_id": child["execution_protocol_id"],
                "child_request_artifact_id": child["artifact_id"],
                "predictor_request_artifact_id": child["predictor_request_artifact_id"],
                "embedding_table_artifact_id": table["artifact_id"],
                "reader_artifact_id": reader["artifact_id"],
                "rows": ledger_rows,
                "row_count": 224,
                "challenge_labels_present": False,
                "qualification_scored": False,
            }
        )
        deterministic = [
            _file_receipt("embedding_table.json", table),
            _file_receipt("reader_artifact.json", reader),
            _file_receipt("prediction_ledger.json", ledger),
        ]
        attestation_core: dict[str, object] = {
            "schema_version": ("relationship-condition-reader-prediction-process-attestation.v3"),
            "protocol_id": child["protocol_id"],
            "execution_protocol_id": child["execution_protocol_id"],
            "child_request_artifact_id": child["artifact_id"],
            "run_ordinal": spec.run_ordinal,
            "run_nonce": spec.run_nonce,
            "process_id": 1000 + spec.run_ordinal,
            "parent_process_id": 900,
            "python_executable": str(spec.import_binding.python_executable),
            "python_implementation": "CPython",
            "python_version": platform.python_version(),
            "argv": executor._prediction_child_sys_argv(spec),
            "interpreter_flags": {
                "safe_path": True,
                "no_site": 1,
                "dont_write_bytecode": 1,
                "utf8_mode": 1,
                "isolated": 0,
                "ignore_environment": 0,
                "stdout_write_through": True,
                "stderr_write_through": True,
            },
            "pycache_prefix": str(spec.pycache_prefix),
            "working_directory": str(spec.capsule_root),
            "sys_path": list(
                executor.expected_child_sys_path(
                    spec.import_binding,
                    python_version=platform.python_version(),
                )
            ),
            "bootstrap_import_roots": [str(path) for path in spec.import_binding.import_roots],
            "environment_contract": {
                "schema_version": ("relationship-condition-reader-prediction-environment.v2"),
                "projected_keys": list(executor._PREDICTION_ENVIRONMENT_PROJECTION_KEYS),
                "all_environment_values_hashed": True,
                "unlisted_environment_variables_recorded": True,
            },
            "environment_projection": {
                key: environment.get(key) for key in executor._PREDICTION_ENVIRONMENT_PROJECTION_KEYS
            },
            "environment_key_names": sorted(environment),
            "environment_value_sha256s": {key: _sha(value) for key, value in sorted(environment.items())},
            "loaded_file_backed_module_origins": self._loaded_module_origins(spec),
            "volvence_zero_namespace_search_locations": [
                str(path) for path in spec.import_binding.volvence_zero_namespace_search_locations
            ],
            "embedder_factory_kind": "formal_bge_m3_cuda",
            "model": child["semantic_model"],
            "live_embedding_call_count": 228,
            "training_embedding_count": 4,
            "challenge_embedding_count": 224,
            "prediction_ledger_fsync_completed": True,
            "forbidden_module_observations": [
                {
                    "module_name": module_name,
                    "loaded_at_worker_entry": False,
                    "loaded_at_worker_exit": False,
                    "imported_by_worker": False,
                }
                for module_name in sorted(executor._FORBIDDEN_WORKER_MODULES)
            ],
            "deterministic_outputs": deterministic,
            "os_security_boundary": False,
        }
        if spec.run_ordinal == 1:
            if self.mode == "bad_origin":
                origins = list(attestation_core["loaded_file_backed_module_origins"])
                first_origin = dict(origins[0])
                replacement = (
                    spec.import_binding.repository_root
                    / "packages/lifeform-evolution/src/lifeform_evolution"
                    / "relationship_condition_reader_qualification_executor.py"
                )
                first_origin["origin"] = str(replacement)
                origins[0] = first_origin
                attestation_core["loaded_file_backed_module_origins"] = origins
            elif self.mode == "bad_origin_domain":
                origins = list(attestation_core["loaded_file_backed_module_origins"])
                first_origin = dict(origins[0])
                first_origin["origin"] = str(spec.predictor_request_path.resolve())
                origins[0] = first_origin
                attestation_core["loaded_file_backed_module_origins"] = origins
            elif self.mode == "bad_sys_path":
                attestation_core["sys_path"] = [
                    *list(attestation_core["sys_path"]),
                    str(spec.capsule_root),
                ]
            elif self.mode == "bad_namespace":
                attestation_core["volvence_zero_namespace_search_locations"] = list(
                    reversed(list(attestation_core["volvence_zero_namespace_search_locations"]))
                )
            elif self.mode == "bad_flags":
                flags = dict(attestation_core["interpreter_flags"])
                flags["safe_path"] = False
                attestation_core["interpreter_flags"] = flags
            elif self.mode == "bad_argv":
                attestation_core["argv"] = [*list(attestation_core["argv"]), "shadow"]
        attestation = _artifact(attestation_core)
        manifest = _artifact(
            {
                "schema_version": ("relationship-condition-reader-prediction-manifest.v1"),
                "protocol_id": child["protocol_id"],
                "execution_protocol_id": child["execution_protocol_id"],
                "child_request_artifact_id": child["artifact_id"],
                "files": [
                    *deterministic,
                    _file_receipt("process_attestation.json", attestation),
                ],
                "file_count": 4,
                "deterministic_file_paths": [
                    "embedding_table.json",
                    "reader_artifact.json",
                    "prediction_ledger.json",
                ],
                "prediction_ledger_fsync_completed": True,
            }
        )
        spec.output_root.mkdir(parents=True)
        for filename, payload in (
            ("embedding_table.json", table),
            ("reader_artifact.json", reader),
            ("prediction_ledger.json", ledger),
            ("process_attestation.json", attestation),
            ("manifest.json", manifest),
        ):
            _write(spec.output_root / filename, _raw(payload))

    @staticmethod
    def _loaded_module_origins(
        spec: executor._PredictionChildLaunchSpec,
    ) -> list[Mapping[str, object]]:
        paths_by_module = {
            "lifeform_domain_emogpt.relationship_condition_reader": (
                "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/relationship_condition_reader.py"
            ),
            "lifeform_evolution.relationship_condition_reader_qualification_predictor": (
                "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_predictor.py"
            ),
            "lifeform_evolution.relationship_condition_reader_qualification_runtime_binding": (
                "packages/lifeform-evolution/src/lifeform_evolution/relationship_condition_reader_qualification_runtime_binding.py"
            ),
            "lifeform_evolution.relationship_lab_product_model_adapters": (
                "packages/lifeform-evolution/src/lifeform_evolution/relationship_lab_product_model_adapters.py"
            ),
            "volvence_zero.social_cognition": ("packages/vz-contracts/src/volvence_zero/social_cognition.py"),
        }
        rows: list[Mapping[str, object]] = []
        for module_name, relative in sorted(paths_by_module.items()):
            origin = spec.import_binding.repository_root / pathlib.PurePosixPath(relative)
            rows.append(
                {
                    "module_name": module_name,
                    "origin": str(origin),
                }
            )
        return rows


class _LoaderSpy:
    def __init__(
        self,
        *,
        tamper_commit_receipt: pathlib.Path | None = None,
    ) -> None:
        self.opened: list[pathlib.Path] = []
        self.tamper_commit_receipt = None if tamper_commit_receipt is None else tamper_commit_receipt.resolve()
        self._tampered = False

    def __call__(
        self,
        path: pathlib.Path,
        *,
        expected_schema_version: str,
        max_bytes: int,
    ) -> executor._LoadedArtifact:
        resolved = pathlib.Path(path).resolve()
        self.opened.append(resolved)
        if self.tamper_commit_receipt == resolved and not self._tampered and resolved.exists():
            self._tampered = True
            resolved.write_bytes(resolved.read_bytes() + b" ")
        return executor._load_canonical_artifact(
            resolved,
            expected_schema_version=expected_schema_version,
            max_bytes=max_bytes,
        )


class _FakeIntegrityGuard:
    def __init__(self, *, drift_phase: str | None = None) -> None:
        self.calls: list[str] = []
        self.drift_phase = drift_phase

    def __call__(
        self,
        *,
        phase: str,
        previous_integrity_receipt_artifact_id: str | None,
    ) -> Mapping[str, object]:
        self.calls.append(phase)
        source_id = _SOURCE_TREE_ARTIFACT_ID
        if phase == self.drift_phase:
            source_id = _sha("drifted-source-tree")
        return _artifact(
            {
                "schema_version": ("relationship-condition-reader-qualification-execution-integrity-receipt.v1"),
                "execution_protocol_id": _EXECUTION_PROTOCOL_ID,
                "phase": phase,
                "phase_ordinal": {
                    "pre_prediction_child_1": 1,
                    "post_prediction_child_1": 2,
                    "pre_prediction_child_2": 3,
                    "post_prediction_child_2": 4,
                }[phase],
                "previous_integrity_receipt_artifact_id": (previous_integrity_receipt_artifact_id),
                "source_tree_artifact_id": source_id,
                "source_tree_entry_count": 7,
                "bge_snapshot_tree_artifact_id": _BGE_TREE_ARTIFACT_ID,
                "bge_snapshot_entry_count": 11,
                "runtime_identity_artifact_id": _RUNTIME_IDENTITY_ARTIFACT_ID,
                "source_tree_exact": True,
                "bge_snapshot_tree_exact": True,
                "runtime_identity_exact": True,
                "observer_model_or_cuda_execution_used": False,
                "torch_imported": False,
                "sentence_transformers_imported": False,
                "os_security_boundary": False,
                "windows_directory_entry_durability_attested": False,
            }
        )


def _run(
    bundle: _PreflightBundle,
    *,
    launcher: _FakeLauncher,
    loader: _LoaderSpy,
    integrity_guard: executor._IntegrityGuard | None = None,
    previous_integrity_receipt_artifact_id: str = (_POST_ANCHOR_INTEGRITY_RECEIPT_ID),
) -> Mapping[str, object]:
    guard = integrity_guard or _FakeIntegrityGuard()
    return executor._execute_relationship_condition_reader_qualification_prediction_stage_core(
        preflight_root=bundle.root,
        execution_root=bundle.execution_root,
        expected_qualification_protocol_id=_QUALIFICATION_PROTOCOL_ID,
        expected_preflight_manifest_artifact_id=bundle.manifest_artifact_id,
        expected_publication_request_artifact_id=(bundle.publication_request_artifact_id),
        execution_protocol_id=_EXECUTION_PROTOCOL_ID,
        run_nonce=_RUN_NONCE,
        bge_snapshot_path=None,
        integrity_guard=guard,
        previous_integrity_receipt_artifact_id=(previous_integrity_receipt_artifact_id),
        expected_source_tree_artifact_id=_SOURCE_TREE_ARTIFACT_ID,
        expected_bge_snapshot_tree_artifact_id=_BGE_TREE_ARTIFACT_ID,
        expected_runtime_identity_artifact_id=_RUNTIME_IDENTITY_ARTIFACT_ID,
        import_binding=_import_binding(),
        launcher=launcher,
        artifact_loader=loader,
    )


def _assert_no_evaluator_open(
    bundle: _PreflightBundle,
    loader: _LoaderSpy,
) -> None:
    assert bundle.challenge_labels_path not in loader.opened
    assert bundle.group_split_path not in loader.opened


def test_executor_success_commits_before_handoff_without_opening_evaluator_files(
    tmp_path: pathlib.Path,
) -> None:
    bundle = _build_preflight(tmp_path)
    launcher = _FakeLauncher()
    loader = _LoaderSpy()
    integrity_guard = _FakeIntegrityGuard()

    result = _run(
        bundle,
        launcher=launcher,
        loader=loader,
        integrity_guard=integrity_guard,
    )

    assert len(launcher.calls) == 2
    assert result["fresh_process_count"] == 2
    assert result["deterministic_outputs_byte_exact"] is True
    assert result["parent_opened_challenge_labels"] is False
    assert result["parent_opened_group_split"] is False
    assert result["scorer_launched"] is False
    assert result["os_security_boundary"] is False
    assert result["windows_directory_entry_durability_attested"] is False
    assert integrity_guard.calls == [
        "pre_prediction_child_1",
        "post_prediction_child_1",
        "pre_prediction_child_2",
        "post_prediction_child_2",
    ]
    _assert_no_evaluator_open(bundle, loader)

    scoring_path = bundle.execution_root / "scoring_request.json"
    assert scoring_path.is_file()
    scoring_request = json.loads(scoring_path.read_text(encoding="utf-8"))
    assert pathlib.Path(scoring_request["challenge_labels_path"]).resolve() == (bundle.challenge_labels_path)
    assert pathlib.Path(scoring_request["group_split_path"]).resolve() == (bundle.group_split_path)
    assert scoring_request["minimum_normalized_margin_hex"] == (0.01).hex()

    training = json.loads(
        (bundle.execution_root / "predictor_capsule" / "training_corpus.json").read_text(encoding="utf-8")
    )
    child = json.loads((bundle.execution_root / "predictor_capsule" / "child_request.json").read_text(encoding="utf-8"))
    assert training["schema_version"].endswith("training-corpus.v1")
    assert all("source_position" not in row for row in training["rows"])
    assert child["protocol_id"] == _QUALIFICATION_PROTOCOL_ID
    assert child["execution_protocol_id"] == _EXECUTION_PROTOCOL_ID
    assert "challenge_labels_path" not in child
    assert "group_split_path" not in child

    commit = json.loads((bundle.execution_root / "commit" / "commit_receipt.json").read_text(encoding="utf-8"))
    assert set(commit) == {
        "schema_version",
        "qualification_protocol_id",
        "execution_protocol_id",
        "child_request_artifact_id",
        "predictor_request_artifact_id",
        "prediction_ledger_artifact_id",
        "prediction_ledger_raw_sha256",
        "prediction_ledger_raw_bytes",
        "prediction_run_manifest_artifact_ids",
        "prediction_run_attestation_artifact_ids",
        "fresh_process_count",
        "predictor_processes_exited",
        "predictor_job_objects_empty",
        "embedding_tables_byte_exact",
        "reader_artifacts_byte_exact",
        "prediction_ledgers_byte_exact",
        "ledger_file_fsync_completed",
        "ledger_same_descriptor_readback",
        "ledger_closed_reopen_readback",
        "windows_directory_entry_durability_attested",
        "artifact_id",
    }
    assert commit["windows_directory_entry_durability_attested"] is False
    committed_ledger = json.loads(
        (bundle.execution_root / "commit" / "prediction_ledger.json").read_text(encoding="utf-8")
    )
    scorer._validate_scoring_request(scoring_request)
    scorer._validate_commit_receipt(commit, request=scoring_request)
    scorer._validate_prediction_ledger(
        committed_ledger,
        request=scoring_request,
        commit=commit,
    )


def test_real_frozen_integrity_guard_chains_through_executor_core(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_preflight(tmp_path)
    source_tree = {
        "artifact_id": _SOURCE_TREE_ARTIFACT_ID,
        "entry_count": 7,
    }
    bge_tree = {
        "artifact_id": _BGE_TREE_ARTIFACT_ID,
        "entry_count": 11,
    }
    runtime_identity = {"artifact_id": _RUNTIME_IDENTITY_ARTIFACT_ID}
    frozen_protocol = {
        "execution_source_tree": source_tree,
        "bge_snapshot_tree": bge_tree,
        "runtime_identity": runtime_identity,
    }

    def validate_protocol(
        _payload: Mapping[str, object],
        *,
        expected_protocol_id: str,
    ) -> str:
        assert expected_protocol_id == _EXECUTION_PROTOCOL_ID
        return expected_protocol_id

    monkeypatch.setattr(
        execution_protocol,
        "validate_relationship_condition_reader_qualification_execution_protocol",
        validate_protocol,
    )
    monkeypatch.setattr(
        execution_protocol,
        "build_relationship_condition_reader_execution_source_tree_manifest",
        lambda *, repository_root: dict(source_tree),
    )
    monkeypatch.setattr(
        execution_protocol,
        "build_bge_m3_snapshot_tree_manifest",
        lambda *, snapshot_root: dict(bge_tree),
    )
    monkeypatch.setattr(
        execution_protocol,
        "build_relationship_condition_reader_qualification_runtime_identity",
        lambda: dict(runtime_identity),
    )
    guard = execution_protocol.relationship_condition_reader_qualification_integrity_guard(
        execution_protocol=frozen_protocol,
        expected_execution_protocol_id=_EXECUTION_PROTOCOL_ID,
        repository_root=tmp_path / "repository",
        bge_snapshot_root=tmp_path / "bge-snapshot",
    )
    post_anchor_receipt = guard(
        phase="post_anchor_pre_execution",
        previous_integrity_receipt_artifact_id=None,
    )

    result = _run(
        bundle,
        launcher=_FakeLauncher(),
        loader=_LoaderSpy(),
        integrity_guard=guard,
        previous_integrity_receipt_artifact_id=str(post_anchor_receipt["artifact_id"]),
    )

    expected_phases = [
        "pre_prediction_child_1",
        "post_prediction_child_1",
        "pre_prediction_child_2",
        "post_prediction_child_2",
    ]
    receipts = [
        json.loads((bundle.execution_root / "integrity_receipts" / f"{phase}.json").read_text(encoding="utf-8"))
        for phase in expected_phases
    ]
    assert [receipt["phase"] for receipt in receipts] == expected_phases
    assert [receipt["phase_ordinal"] for receipt in receipts] == [1, 2, 3, 4]
    previous_id = post_anchor_receipt["artifact_id"]
    for receipt in receipts:
        assert receipt["previous_integrity_receipt_artifact_id"] == previous_id
        previous_id = receipt["artifact_id"]
    assert result["last_integrity_receipt_artifact_id"] == previous_id
    assert all(
        receipt["schema_version"] == execution_protocol.RELATIONSHIP_READER_EXECUTION_INTEGRITY_RECEIPT_SCHEMA_VERSION
        for receipt in receipts
    )


@pytest.mark.parametrize("mode", ["nonzero", "not_exited", "job_busy"])
def test_executor_child_failure_never_releases_scoring_request(
    tmp_path: pathlib.Path,
    mode: str,
) -> None:
    bundle = _build_preflight(tmp_path)
    loader = _LoaderSpy()

    with pytest.raises(RuntimeError):
        _run(bundle, launcher=_FakeLauncher(mode=mode), loader=loader)

    _assert_no_evaluator_open(bundle, loader)
    assert not (bundle.execution_root / "scoring_request.json").exists()


def test_executor_rejects_cross_process_byte_mismatch_before_commit(
    tmp_path: pathlib.Path,
) -> None:
    bundle = _build_preflight(tmp_path)
    loader = _LoaderSpy()

    with pytest.raises(RuntimeError, match="byte-exact"):
        _run(bundle, launcher=_FakeLauncher(mode="mismatch"), loader=loader)

    _assert_no_evaluator_open(bundle, loader)
    assert not (bundle.execution_root / "commit").exists()
    assert not (bundle.execution_root / "scoring_request.json").exists()


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("bad_origin", "module name does not match"),
        ("bad_origin_domain", "outside every controlled import domain"),
        ("bad_sys_path", "sys.path differs"),
        ("bad_namespace", "namespace search locations drifted"),
        ("bad_flags", "interpreter flags drifted"),
        ("bad_argv", "sys.argv differs"),
    ],
)
def test_executor_rejects_prediction_runtime_binding_drift_before_handoff(
    tmp_path: pathlib.Path,
    mode: str,
    message: str,
) -> None:
    bundle = _build_preflight(tmp_path)
    loader = _LoaderSpy()

    with pytest.raises(ValueError, match=message):
        _run(bundle, launcher=_FakeLauncher(mode=mode), loader=loader)

    assert not (bundle.execution_root / "scoring_request.json").exists()
    _assert_no_evaluator_open(bundle, loader)


def test_executor_commit_tamper_prevents_scoring_request_and_label_release(
    tmp_path: pathlib.Path,
) -> None:
    bundle = _build_preflight(tmp_path)
    commit_path = bundle.execution_root / "commit" / "commit_receipt.json"
    loader = _LoaderSpy(tamper_commit_receipt=commit_path)

    with pytest.raises(ValueError):
        _run(bundle, launcher=_FakeLauncher(), loader=loader)

    _assert_no_evaluator_open(bundle, loader)
    assert commit_path.exists()
    assert not (bundle.execution_root / "scoring_request.json").exists()


def test_integrity_drift_after_first_child_stops_before_second_and_label_release(
    tmp_path: pathlib.Path,
) -> None:
    bundle = _build_preflight(tmp_path)
    loader = _LoaderSpy()
    launcher = _FakeLauncher()
    guard = _FakeIntegrityGuard(drift_phase="post_prediction_child_1")

    with pytest.raises(ValueError, match="expected source_tree_artifact_id"):
        _run(
            bundle,
            launcher=launcher,
            loader=loader,
            integrity_guard=guard,
        )

    assert len(launcher.calls) == 1
    assert guard.calls == [
        "pre_prediction_child_1",
        "post_prediction_child_1",
    ]
    _assert_no_evaluator_open(bundle, loader)
    assert not (bundle.execution_root / "scoring_request.json").exists()


def test_parent_preflight_open_set_is_exact_and_excludes_full_validator_inputs(
    tmp_path: pathlib.Path,
) -> None:
    bundle = _build_preflight(tmp_path)
    loader = _LoaderSpy()

    _run(bundle, launcher=_FakeLauncher(), loader=loader)

    preflight_opens = {
        path.relative_to(bundle.root.resolve()).as_posix()
        for path in loader.opened
        if path.is_relative_to(bundle.root.resolve())
    }
    assert preflight_opens == {
        "manifest.json",
        "public/publication_request.json",
        "public/public_corpus.json",
        "public/predictor_request.json",
        "sealed/condition_training_labels.json",
    }
    assert "protocol.json" not in preflight_opens
    assert "sealed/challenge_labels.json" not in preflight_opens
    assert "sealed/group_split.json" not in preflight_opens


def test_formal_environment_is_built_from_exact_allowlist_and_hash_binds_values() -> None:
    binding = _import_binding()
    pycache_prefix = pathlib.Path("D:/formal-capsule/pycache-run-1")
    environment, projection, contract_id = executor._build_formal_child_environment(
        {
            "PATH": "C:\\ambient-shadow;C:\\other-python",
            "SystemRoot": "C:\\Windows",
            "CUDA_PATH": "C:\\poison-toolkit",
            "CUDA_VISIBLE_DEVICES": "7",
            "SECRET_TOKEN": "do-not-inherit",
            "AWS_SECRET_ACCESS_KEY": "do-not-inherit-either",
            "PYTHONPATH": "C:\\ambient-shadow",
        },
        import_binding=binding,
        pycache_prefix=pycache_prefix,
    )

    assert "SECRET_TOKEN" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "CUDA_PATH" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["PYTHONPATH"] != "C:\\ambient-shadow"
    assert "ambient-shadow" not in environment["PATH"]
    assert "other-python" not in environment["PATH"]
    assert environment == {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "PATH": os.pathsep.join(
            str(path)
            for path in executor.controlled_child_path(
                binding,
                system_root=pathlib.Path("C:/Windows"),
            )
        ),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(str(path) for path in binding.import_roots),
        "PYTHONPYCACHEPREFIX": str(pycache_prefix),
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "SystemRoot": "C:\\Windows",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert tuple(item.key for item in projection) == tuple(sorted(environment))
    assert all(
        item.value_sha256 == _sha(environment[item.key])
        and item.value_utf8_bytes == len(environment[item.key].encode("utf-8"))
        for item in projection
    )
    assert contract_id == executor._environment_contract_id(projection)


def test_formal_process_contract_is_suspended_single_process_job() -> None:
    assert executor._FORMAL_CREATION_FLAGS & executor._CREATE_SUSPENDED
    assert executor._FORMAL_CREATION_FLAGS & executor._CREATE_NO_WINDOW
    assert executor._FORMAL_CREATION_FLAGS & executor._EXTENDED_STARTUPINFO_PRESENT
    assert executor._FORMAL_JOB_LIMIT_FLAGS & executor._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    assert executor._FORMAL_JOB_LIMIT_FLAGS & executor._JOB_OBJECT_LIMIT_ACTIVE_PROCESS
    binding = _import_binding()
    spec = executor._PredictionChildLaunchSpec(
        child_request_path=pathlib.Path("D:/capsule/child.json"),
        expected_child_request_artifact_id=_sha("child"),
        training_corpus_path=pathlib.Path("D:/capsule/training.json"),
        predictor_request_path=pathlib.Path("D:/capsule/predictor.json"),
        output_root=pathlib.Path("D:/execution/run-1"),
        capsule_root=pathlib.Path("D:/capsule"),
        run_ordinal=1,
        run_nonce=_sha("run-1"),
        bge_snapshot_path=pathlib.Path("D:/bge"),
        import_binding=binding,
        pycache_prefix=pathlib.Path("D:/capsule/pycache-run-1"),
    )
    assert executor._prediction_child_process_argv(spec)[1:10] == (
        "-P",
        "-S",
        "-B",
        "-u",
        "-X",
        "utf8",
        "-X",
        "pycache_prefix=D:\\capsule\\pycache-run-1",
        "-c",
    )


def test_public_formal_entry_requires_explicit_bge_snapshot(
    tmp_path: pathlib.Path,
) -> None:
    binding = _import_binding()
    with pytest.raises(ValueError, match="explicit pinned BGE snapshot"):
        executor.execute_relationship_condition_reader_qualification_prediction_stage(
            preflight_root=tmp_path / "preflight",
            execution_root=tmp_path / "execution",
            expected_qualification_protocol_id=_QUALIFICATION_PROTOCOL_ID,
            expected_preflight_manifest_artifact_id=_sha("manifest"),
            expected_publication_request_artifact_id=_sha("publication"),
            execution_protocol_id=_EXECUTION_PROTOCOL_ID,
            run_nonce=_RUN_NONCE,
            integrity_guard=lambda **_kwargs: {},
            previous_integrity_receipt_artifact_id=(_POST_ANCHOR_INTEGRITY_RECEIPT_ID),
            expected_source_tree_artifact_id=_SOURCE_TREE_ARTIFACT_ID,
            expected_bge_snapshot_tree_artifact_id=_BGE_TREE_ARTIFACT_ID,
            expected_runtime_identity_artifact_id=(_RUNTIME_IDENTITY_ARTIFACT_ID),
            repository_root=binding.repository_root,
            repository_source_roots=binding.repository_source_roots,
            frozen_source_entries={
                entry.path: {
                    "raw_sha256": entry.raw_sha256,
                    "raw_bytes": entry.raw_bytes,
                }
                for entry in binding.frozen_source_entries
            },
            frozen_site_packages_root=binding.frozen_site_packages_root,
            bge_snapshot_path=None,
        )
