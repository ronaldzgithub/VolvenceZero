"""Pinned public-only embedding table for the admitted Product Horizon source-v5.

This development owner does not fit or qualify a reader.  It first delegates
the complete input revalidation to the source-v5 admission owner, then derives
the exact text inventory from the typed public view and passes only those
strings to the pinned BGE adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import pathlib
import re
from typing import Mapping, Protocol, Sequence

from lifeform_domain_emogpt.lab.contracts import canonical_json, sha256_json
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 import (
    RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_VIEW_SCHEMA_VERSION,
    RelationshipProductHorizonPublicView,
)
from lifeform_evolution.relationship_lab_product_batch_model_adapter import (
    bge_m3_batch_public_semantic_embedder,
)
from lifeform_evolution.relationship_lab_product_model_adapters import (
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    BGE_M3_WEIGHT_BYTES_SHA256,
    PRECOMPUTED_PUBLIC_EMBEDDING_RECORD_SCHEMA_VERSION,
    PRECOMPUTED_PUBLIC_EMBEDDING_TABLE_SCHEMA_VERSION,
    PrecomputedPublicEmbeddingRecord,
    PrecomputedPublicEmbeddingTable,
    bge_m3_weight_pinned_embedder_identity,
)
from lifeform_evolution.relationship_product_horizon_source_v5_admission import (
    HORIZON_SOURCE_V5_ADMISSION_MANIFEST_SCHEMA_VERSION,
    validate_relationship_product_horizon_source_v5_admission,
)


SOURCE_V5_EMBEDDING_TABLE_PROTOCOL_SCHEMA_VERSION = (
    "relationship-product-horizon-source-v5-embedding-table-protocol.v2"
)
SOURCE_V5_EMBEDDING_TABLE_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-horizon-source-v5-embedding-table-manifest.v2"
)

_PROTOCOL_FILENAME = "relationship_product_horizon_source_v5_embedding_table_v2.json"
_EXPECTED_OUTPUT_FILES = frozenset(
    {"protocol.json", "embedding_table.json", "manifest.json"}
)
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class _PinnedEmbedder(Protocol):
    name: str
    model_source: str
    model_revision: str | None
    weights_sha256: str | None
    sentence_transformers_version: str | None

    def embed_many(
        self,
        texts: tuple[str, ...],
        *,
        batch_size: int,
    ) -> tuple[tuple[float, ...], ...]: ...


@dataclass(frozen=True)
class RelationshipProductHorizonSourceV5EmbeddingTableProtocol:
    payload: Mapping[str, object]
    protocol_id: str
    raw_sha256: str
    raw_bytes: int

    @property
    def source_admission(self) -> Mapping[str, object]:
        return _mapping(self.payload["source_admission"], "source_admission")

    @property
    def public_embedding_input(self) -> Mapping[str, object]:
        return _mapping(
            self.payload["public_embedding_input"], "public_embedding_input"
        )

    @property
    def semantic_model(self) -> Mapping[str, object]:
        return _mapping(self.payload["semantic_model"], "semantic_model")

    @property
    def execution_contract(self) -> Mapping[str, object]:
        return _mapping(self.payload["execution_contract"], "execution_contract")

    @property
    def output_contract(self) -> Mapping[str, object]:
        return _mapping(self.payload["output_contract"], "output_contract")

    @property
    def claims_ceiling(self) -> Mapping[str, object]:
        return _mapping(self.payload["claims_ceiling"], "claims_ceiling")


@dataclass(frozen=True)
class SourceV5AdmissionValidationInputs:
    """Frozen paths required by the upstream admission owner's full replay."""

    source_v3_admission_root: pathlib.Path
    source_v4_admission_root: pathlib.Path
    development_reader_root: pathlib.Path
    attempt03_embedding_table_path: pathlib.Path
    attempt03_reobservation_path: pathlib.Path
    qualification_v5_embedding_table_path: pathlib.Path

    def as_kwargs(self) -> dict[str, pathlib.Path]:
        return {
            "source_v3_admission_root": pathlib.Path(
                self.source_v3_admission_root
            ),
            "source_v4_admission_root": pathlib.Path(
                self.source_v4_admission_root
            ),
            "development_reader_root": pathlib.Path(self.development_reader_root),
            "attempt03_embedding_table_path": pathlib.Path(
                self.attempt03_embedding_table_path
            ),
            "attempt03_reobservation_path": pathlib.Path(
                self.attempt03_reobservation_path
            ),
            "qualification_v5_embedding_table_path": pathlib.Path(
                self.qualification_v5_embedding_table_path
            ),
        }

    def read_only_roots(self) -> tuple[pathlib.Path, ...]:
        return (
            pathlib.Path(self.source_v3_admission_root),
            pathlib.Path(self.source_v4_admission_root),
            pathlib.Path(self.development_reader_root),
            pathlib.Path(self.attempt03_embedding_table_path).parent,
            pathlib.Path(self.attempt03_reobservation_path).parent,
            pathlib.Path(self.qualification_v5_embedding_table_path).parent,
        )


@dataclass(frozen=True)
class _SourceV5AdmissionReceipt:
    protocol_id: str
    artifact_id: str
    status: str
    source_protocol_id: str
    public_plan_content_id: str
    root_count: int


def relationship_product_horizon_source_v5_embedding_table_protocol_path() -> (
    pathlib.Path
):
    return pathlib.Path(__file__).resolve().parent / "protocols" / _PROTOCOL_FILENAME


def load_relationship_product_horizon_source_v5_embedding_table_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipProductHorizonSourceV5EmbeddingTableProtocol:
    source = pathlib.Path(
        path or relationship_product_horizon_source_v5_embedding_table_protocol_path()
    )
    raw = source.read_bytes()
    payload = _parse_json(raw, source=str(source))
    _exact_keys(
        payload,
        {
            "schema_version",
            "evidence_tier",
            "owner",
            "purpose",
            "predecessor_failure",
            "source_admission",
            "public_embedding_input",
            "semantic_model",
            "execution_contract",
            "output_contract",
            "direct_execution_closure",
            "claims_ceiling",
            "claim_boundary",
        },
        source="source-v5 embedding-table protocol",
    )
    if payload["schema_version"] != SOURCE_V5_EMBEDDING_TABLE_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("source-v5 embedding-table protocol schema drifted")
    if payload["evidence_tier"] != "development":
        raise ValueError("source-v5 embedding-table evidence tier drifted")
    if payload["owner"] != (
        "lifeform_evolution.relationship_product_horizon_source_v5_embedding_table"
    ):
        raise ValueError("source-v5 embedding-table owner drifted")
    if payload["purpose"] != "materialize_source_v5_only_public_embedding_table_without_reader_fit":
        raise ValueError("source-v5 embedding-table purpose drifted")

    predecessor = _mapping(payload["predecessor_failure"], "predecessor_failure")
    expected_predecessor: dict[str, object] = {
        "protocol_id": "e5e3aac68d4248e2e21116b2fd44abb82f4493065eb014992ed9761883a8947f",
        "implementation_git_commit": "db0e949ff0d89ee6c88967fd5f7f8a6ea21cd13e",
        "terminal": "pre_embedding_admission_closure_incompatible",
        "failed_before_first_embedding_api_call": True,
        "embedding_api_call_count": 0,
        "cuda_model_load_count": 0,
        "output_artifact_created": False,
        "retry_under_predecessor_protocol_forbidden": True,
    }
    if predecessor != expected_predecessor:
        raise ValueError("source-v5 embedding predecessor failure lineage drifted")

    admission = _mapping(payload["source_admission"], "source_admission")
    expected_admission: dict[str, object] = {
        "protocol_id": "d07bdb21cadc809b605d36e76ebaba45da8334acbdc8d2b6dc68417bb13efcd4",
        "protocol_raw_sha256": "9112dd1d8dd075443a00fcfb4306fe2cbbcf5d5f361b32665774bc8a13ddeae8",
        "artifact_id": "79a51e5494641140b70b5adb60483e1c7e651ca1ac8c4d939953238d1f8502da",
        "manifest_schema_version": HORIZON_SOURCE_V5_ADMISSION_MANIFEST_SCHEMA_VERSION,
        "manifest_raw_sha256": "444d30e00fbac881e31469fdc24ea0d1202a8c0a4f80af0361d212b8623c05cb",
        "implementation_git_commit": "8fb2e751b367d1daa2ad560a3d6aa75932335a12",
        "required_status": "source_v5_campaign_input_admitted_execution_not_authorized",
        "validate_existing_required": True,
        "external_expected_protocol_and_artifact_ids_required": True,
        "validation_mode": "read_only_full_semantic_rebuild",
        "validation_reads_sealed_upstream": True,
    }
    if admission != expected_admission:
        raise ValueError("source-v5 admission pin drifted")

    public_input = _mapping(
        payload["public_embedding_input"], "public_embedding_input"
    )
    expected_public_input: dict[str, object] = {
        "public_plan_relative_path": "public/source_plan.json",
        "public_plan_schema_version": RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_VIEW_SCHEMA_VERSION,
        "public_plan_raw_sha256": "83f0bbd06911bdc553f3de0d8d3270e8aadae657daf1914e1b8eb1589e9d82cd",
        "public_plan_content_id": "bab2ff2291b95d4eef6107a58ebf4575b08490775bee71b2ad99a5b029e09f6c",
        "source_protocol_id_lineage_only": "71dc200630bf09ee66ce47b9f45460f30ec14cd3ff4e08366c7946497babad9b",
        "source_protocol_raw_sha256_lineage_only": "33623a4409d3e5419207340e08bd90462b6b1675afb089433b89fbdb2d859134",
        "root_count": 112,
        "reader_text_occurrence_count": 5824,
        "reader_text_unique_count": 3946,
        "reader_text_inventory_sha256": "13794eb5b73c9d9f6d69553278c03b0f3121b5eb450efdff41f85a1987dbb082",
        "public_text_field_allowlist": [
            "roots[].onboarding_sessions[].user_utterance",
            "roots[].decision_sessions[].current_input",
        ],
        "deduplication": "sha256_utf8_plus_full_text_collision_check",
        "canonical_order": "text_sha256_then_text",
        "additional_text_source_count": 0,
        "old_embedding_table_record_count": 0,
        "challenge_or_evaluation_label_read_count": 0,
        "outcome_value_forwarded_to_embedder_count": 0,
        "sealed_payload_forwarded_to_embedder_count": 0,
    }
    if public_input != expected_public_input:
        raise ValueError("source-v5 public embedding input contract drifted")

    semantic = _mapping(payload["semantic_model"], "semantic_model")
    expected_semantic = {
        "model_id": BGE_M3_MODEL_ID,
        "model_revision": BGE_M3_MODEL_REVISION,
        "weights_sha256": BGE_M3_WEIGHT_BYTES_SHA256,
        "sentence_transformers_version": BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
        "embedding_width": 1024,
        "device": "cuda",
        "network_allowed": False,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    if semantic != expected_semantic:
        raise ValueError("source-v5 embedding model identity drifted")

    execution = _mapping(payload["execution_contract"], "execution_contract")
    expected_execution: dict[str, object] = {
        "encode_api": "batch_extension.embed_many.v1",
        "batch_size": 32,
        "embedding_api_call_count": 1,
        "embedding_vector_count": 3946,
        "explicit_snapshot_path_required": True,
        "fallback_to_single_text_allowed": False,
        "adaptive_batch_resize_allowed": False,
        "admission_validation_completed_before_first_embedding_call": True,
        "validate_existing_embedding_api_call_count": 0,
        "source_v5_public_embedding_execution_authorized": True,
        "source_v5_cuda_embedding_execution_authorized": True,
        "campaign_model_execution_authorized": False,
        "campaign_cuda_execution_authorized": False,
    }
    if execution != expected_execution:
        raise ValueError("source-v5 embedding execution contract drifted")

    output = _mapping(payload["output_contract"], "output_contract")
    expected_output: dict[str, object] = {
        "embedding_record_schema_version": PRECOMPUTED_PUBLIC_EMBEDDING_RECORD_SCHEMA_VERSION,
        "embedding_table_schema_version": PRECOMPUTED_PUBLIC_EMBEDDING_TABLE_SCHEMA_VERSION,
        "float_serialization": "canonical_float_hex",
        "expected_files": sorted(_EXPECTED_OUTPUT_FILES),
        "maximum_embedding_table_raw_bytes": 100_000_000,
        "reader_fit_count": 0,
        "reader_fit_scope": "this_protocol_and_materialization",
        "reader_artifact_input_count": 0,
        "reader_artifact_output_count": 0,
        "reader_inference_count": 0,
        "manifest_written_last": True,
        "power_loss_durability_claimed": False,
    }
    if output != expected_output:
        raise ValueError("source-v5 embedding output contract drifted")

    expected_closure = (
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/contracts.py",
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/relationship_product_horizon_source_v4.py",
        "packages/lifeform-evolution/src/lifeform_evolution/protocols/relationship_product_horizon_source_v5_embedding_table_v2.json",
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_lab_product_model_adapters.py",
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_lab_product_batch_model_adapter.py",
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_product_horizon_source_v5_admission.py",
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_product_horizon_source_v5_embedding_table.py",
        "scripts/run_relationship_product_horizon_source_v5_embedding_table.py",
    )
    if tuple(
        _text_sequence(payload["direct_execution_closure"], "direct_execution_closure")
    ) != expected_closure:
        raise ValueError("source-v5 embedding direct execution closure drifted")

    expected_claims: dict[str, object] = {
        "upstream_campaign_input_admission_verified": True,
        "source_v5_public_reader_inventory_exact_join_verified": True,
        "source_v5_embedding_table_materialized": True,
        "embedding_model_identity_verified": True,
        "embedding_table_exact_text_coverage_complete": True,
        "materialization_complete": True,
        "reader_fit_count": 0,
        "reader_qualified": False,
        "condition_reader_qualified": False,
        "semantic_novelty_established": False,
        "dgp_independence_established": False,
        "statistical_independence_established": False,
        "process_independence_proven": False,
        "a1_reactive_source_evidence_inherited": False,
        "a2_msc_budget_evidence_inherited": False,
        "theta_handoff_materialized": False,
        "geometric_reachability_established": False,
        "credit_achievability_established": False,
        "treatment_reachability_admitted": False,
        "campaign_protocol_frozen": False,
        "campaign_materialized": False,
        "campaign_execution_authorized": False,
        "campaign_runtime_order_verified": False,
        "forecast_runtime_arm_blinding_verified": False,
        "forecast_runtime_scope_blinding_verified": False,
        "forecast_runtime_order_blinding_verified": False,
        "settlement_count": 0,
        "prediction_error_count": 0,
        "credit_count": 0,
        "gate_update_count": 0,
        "generative_model_output_count": 0,
        "appendable_effect": False,
        "readable_effect": False,
        "learnable_effect": False,
        "steerable_effect": False,
        "formal_evidence_authorized": False,
        "unseen_single_axis_evidence": False,
        "integrated_horizon_authorized": False,
        "four_able_complete": False,
        "human_sample_claimed": False,
        "production_active": False,
    }
    if _mapping(payload["claims_ceiling"], "claims_ceiling") != expected_claims:
        raise ValueError("source-v5 embedding-table claim ceiling drifted")
    _text(payload["claim_boundary"], "claim_boundary")
    return RelationshipProductHorizonSourceV5EmbeddingTableProtocol(
        payload=payload,
        protocol_id=sha256_json(payload),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=len(raw),
    )


def materialize_relationship_product_horizon_source_v5_embedding_table(
    *,
    source_v5_admission_root: pathlib.Path,
    admission_validation_inputs: SourceV5AdmissionValidationInputs,
    output_dir: pathlib.Path,
    implementation_git_commit: str,
    bge_snapshot_path: pathlib.Path,
) -> Mapping[str, object]:
    """Run one fixed BGE batch API call over all unique public source-v5 texts."""

    protocol = load_relationship_product_horizon_source_v5_embedding_table_protocol()
    semantic = protocol.semantic_model
    embedder = bge_m3_batch_public_semantic_embedder(
        device=_text(semantic["device"], "semantic_model.device"),
        model_revision=_text(
            semantic["model_revision"], "semantic_model.model_revision"
        ),
        weights_sha256=_digest(
            semantic["weights_sha256"], "semantic_model.weights_sha256"
        ),
        sentence_transformers_version=_text(
            semantic["sentence_transformers_version"],
            "semantic_model.sentence_transformers_version",
        ),
        snapshot_path=pathlib.Path(bge_snapshot_path).resolve(),
    )
    return _materialize_with_embedder(
        protocol=protocol,
        source_v5_admission_root=pathlib.Path(source_v5_admission_root),
        admission_validation_inputs=admission_validation_inputs,
        output_dir=pathlib.Path(output_dir),
        implementation_git_commit=implementation_git_commit,
        embedder=embedder,
        model_snapshot_root=pathlib.Path(bge_snapshot_path),
    )


def _require_disjoint_output(
    *,
    output_root: pathlib.Path,
    read_only_roots: Sequence[pathlib.Path],
) -> None:
    output = pathlib.Path(output_root).resolve()
    for value in read_only_roots:
        upstream = pathlib.Path(value).resolve()
        if output == upstream or output in upstream.parents or upstream in output.parents:
            raise ValueError(
                "source-v5 embedding-table output must be disjoint from every read-only input root"
            )


def _materialize_with_embedder(
    *,
    protocol: RelationshipProductHorizonSourceV5EmbeddingTableProtocol,
    source_v5_admission_root: pathlib.Path,
    admission_validation_inputs: SourceV5AdmissionValidationInputs,
    output_dir: pathlib.Path,
    implementation_git_commit: str,
    embedder: _PinnedEmbedder,
    model_snapshot_root: pathlib.Path,
) -> Mapping[str, object]:
    commit = _git_commit(implementation_git_commit)
    source_root = pathlib.Path(source_v5_admission_root).resolve()
    root = pathlib.Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"source-v5 embedding-table output already exists: {root}")
    _require_disjoint_output(
        output_root=root,
        read_only_roots=(
            source_root,
            *admission_validation_inputs.read_only_roots(),
            pathlib.Path(model_snapshot_root),
        ),
    )
    _validate_embedder_identity(embedder, protocol)
    admission_receipt = _revalidate_source_admission(
        protocol=protocol,
        source_v5_admission_root=source_root,
        admission_validation_inputs=admission_validation_inputs,
    )
    inventory = _load_public_inventory(
        protocol=protocol,
        source_v5_admission_root=source_root,
        admission_receipt=admission_receipt,
    )
    table = _build_embedding_table(
        protocol=protocol,
        inventory=inventory,
        embedder=embedder,
    )

    table_raw = table.to_json().encode("utf-8")
    maximum_table_bytes = _integer(
        protocol.output_contract["maximum_embedding_table_raw_bytes"],
        "maximum_embedding_table_raw_bytes",
    )
    if len(table_raw) > maximum_table_bytes:
        raise ValueError(
            "source-v5 embedding table exceeds the frozen single-blob byte ceiling"
        )

    root.mkdir(parents=True, exist_ok=False)
    protocol_raw = (
        relationship_product_horizon_source_v5_embedding_table_protocol_path().read_bytes()
    )
    _write_create_only(root / "protocol.json", protocol_raw)
    _write_create_only(root / "embedding_table.json", table_raw)
    manifest = _manifest(
        root=root,
        protocol=protocol,
        implementation_git_commit=commit,
        admission_receipt=admission_receipt,
        table=table,
    )
    _write_create_only(root / "manifest.json", _artifact_bytes(manifest))
    return _validate_persisted_bundle(
        protocol=protocol,
        source_v5_admission_root=source_root,
        admission_receipt=admission_receipt,
        output_dir=root,
        expected_protocol_id=protocol.protocol_id,
        expected_artifact_id=_digest(manifest["artifact_id"], "manifest artifact_id"),
    )


def validate_relationship_product_horizon_source_v5_embedding_table(
    *,
    source_v5_admission_root: pathlib.Path,
    admission_validation_inputs: SourceV5AdmissionValidationInputs,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> Mapping[str, object]:
    """Revalidate admission and then replay the table bundle without a model."""

    protocol = load_relationship_product_horizon_source_v5_embedding_table_protocol()
    source_root = pathlib.Path(source_v5_admission_root).resolve()
    admission_receipt = _revalidate_source_admission(
        protocol=protocol,
        source_v5_admission_root=source_root,
        admission_validation_inputs=admission_validation_inputs,
    )
    return _validate_persisted_bundle(
        protocol=protocol,
        source_v5_admission_root=source_root,
        admission_receipt=admission_receipt,
        output_dir=pathlib.Path(output_dir).resolve(),
        expected_protocol_id=expected_protocol_id,
        expected_artifact_id=expected_artifact_id,
    )


def _revalidate_source_admission(
    *,
    protocol: RelationshipProductHorizonSourceV5EmbeddingTableProtocol,
    source_v5_admission_root: pathlib.Path,
    admission_validation_inputs: SourceV5AdmissionValidationInputs,
) -> _SourceV5AdmissionReceipt:
    pin = protocol.source_admission
    root = pathlib.Path(source_v5_admission_root).resolve()
    protocol_raw = (root / "protocol.json").read_bytes()
    if hashlib.sha256(protocol_raw).hexdigest() != pin["protocol_raw_sha256"]:
        raise ValueError("source-v5 admission protocol raw identity drifted")
    manifest_raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != pin["manifest_raw_sha256"]:
        raise ValueError("source-v5 admission manifest raw identity drifted")
    manifest = validate_relationship_product_horizon_source_v5_admission(
        root,
        expected_protocol_id=_digest(pin["protocol_id"], "admission protocol_id"),
        expected_artifact_id=_digest(pin["artifact_id"], "admission artifact_id"),
        **admission_validation_inputs.as_kwargs(),
    )
    if manifest.get("schema_version") != pin["manifest_schema_version"]:
        raise ValueError("source-v5 admission manifest schema drifted")
    if manifest.get("status") != pin["required_status"]:
        raise ValueError("source-v5 admission status drifted")
    if manifest.get("protocol_id") != pin["protocol_id"]:
        raise ValueError("source-v5 admission protocol identity drifted")
    if manifest.get("artifact_id") != pin["artifact_id"]:
        raise ValueError("source-v5 admission artifact identity drifted")
    claims = _mapping(manifest.get("claims"), "source-v5 admission claims")
    if claims.get("campaign_input_admitted") is not True:
        raise ValueError("source-v5 input is not admitted")
    if claims.get("source_v5_embedding_table_materialized") is not False:
        raise ValueError("source-v5 admission unexpectedly contains an embedding table")
    if claims.get("reader_fit_count") != 0:
        raise ValueError("source-v5 admission reader fit count drifted")
    public_input = protocol.public_embedding_input
    if (
        manifest.get("implementation_git_commit") != pin["implementation_git_commit"]
        or manifest.get("source_protocol_id")
        != public_input["source_protocol_id_lineage_only"]
        or manifest.get("public_plan_sha256")
        != public_input["public_plan_content_id"]
        or manifest.get("root_count") != public_input["root_count"]
    ):
        raise ValueError("source-v5 admission manifest input lineage drifted")
    return _SourceV5AdmissionReceipt(
        protocol_id=_digest(manifest["protocol_id"], "admission protocol_id"),
        artifact_id=_digest(manifest["artifact_id"], "admission artifact_id"),
        status=_text(manifest["status"], "admission status"),
        source_protocol_id=_digest(
            manifest["source_protocol_id"], "source protocol id"
        ),
        public_plan_content_id=_digest(
            manifest["public_plan_sha256"], "public plan content id"
        ),
        root_count=_integer(manifest["root_count"], "root count"),
    )


def _add_public_text(by_digest: dict[str, str], text: str) -> None:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    existing = by_digest.setdefault(digest, text)
    if existing != text:
        raise RuntimeError("SHA-256 collision in source-v5 public reader inventory")


def _load_public_inventory(
    *,
    protocol: RelationshipProductHorizonSourceV5EmbeddingTableProtocol,
    source_v5_admission_root: pathlib.Path,
    admission_receipt: _SourceV5AdmissionReceipt,
) -> tuple[tuple[str, str], ...]:
    pin = protocol.public_embedding_input
    root = pathlib.Path(source_v5_admission_root).resolve()
    if (
        admission_receipt.source_protocol_id
        != pin["source_protocol_id_lineage_only"]
        or admission_receipt.public_plan_content_id != pin["public_plan_content_id"]
        or admission_receipt.root_count != pin["root_count"]
    ):
        raise ValueError("source-v5 admission receipt input lineage drifted")

    public_path = root / _text(
        pin["public_plan_relative_path"], "public plan relative path"
    )
    public_raw = public_path.read_bytes()
    if hashlib.sha256(public_raw).hexdigest() != pin["public_plan_raw_sha256"]:
        raise ValueError("source-v5 public plan raw identity drifted")
    public_payload = _parse_json(public_raw, source=str(public_path))
    public = RelationshipProductHorizonPublicView.from_payload(public_payload)
    if public.schema_version != pin["public_plan_schema_version"]:
        raise ValueError("source-v5 public plan schema drifted")
    if public.protocol_id != pin["source_protocol_id_lineage_only"]:
        raise ValueError("source-v5 public plan protocol identity drifted")
    if public.public_plan_sha256 != pin["public_plan_content_id"]:
        raise ValueError("source-v5 public plan canonical identity drifted")
    if len(public.roots) != pin["root_count"]:
        raise ValueError("source-v5 public root count drifted")

    by_digest: dict[str, str] = {}
    occurrence_count = 0
    for root_item in public.roots:
        for session in root_item.onboarding_sessions:
            occurrence_count += 1
            _add_public_text(by_digest, session.user_utterance)
        for session in root_item.decision_sessions:
            occurrence_count += 1
            _add_public_text(by_digest, session.current_input)
    inventory = tuple(sorted(by_digest.items()))
    if occurrence_count != pin["reader_text_occurrence_count"]:
        raise ValueError("source-v5 reader text occurrence count drifted")
    if len(inventory) != pin["reader_text_unique_count"]:
        raise ValueError("source-v5 reader text unique count drifted")
    if sha256_json(inventory) != pin["reader_text_inventory_sha256"]:
        raise ValueError("source-v5 reader text inventory identity drifted")
    return inventory


def _build_embedding_table(
    *,
    protocol: RelationshipProductHorizonSourceV5EmbeddingTableProtocol,
    inventory: Sequence[tuple[str, str]],
    embedder: _PinnedEmbedder,
) -> PrecomputedPublicEmbeddingTable:
    expected_count = _integer(
        protocol.public_embedding_input["reader_text_unique_count"],
        "reader_text_unique_count",
    )
    if len(inventory) != expected_count:
        raise ValueError("source-v5 embedding inventory count drifted")
    expected_width = _integer(
        protocol.semantic_model["embedding_width"], "embedding_width"
    )
    seen: set[str] = set()
    for expected_digest, text in inventory:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != _digest(expected_digest, "reader text digest"):
            raise ValueError("source-v5 embedding input digest drifted")
        if digest in seen:
            raise ValueError("source-v5 embedding input digest is duplicated")
        seen.add(digest)
    texts = tuple(text for _digest_value, text in inventory)
    vectors = embedder.embed_many(
        texts,
        batch_size=_integer(
            protocol.execution_contract["batch_size"], "batch_size"
        ),
    )
    if len(vectors) != len(texts):
        raise ValueError("source-v5 batch embedding row count drifted")
    records: list[PrecomputedPublicEmbeddingRecord] = []
    for (expected_digest, text), raw_vector in zip(inventory, vectors, strict=True):
        vector = _embedding(raw_vector, expected_width=expected_width)
        record = PrecomputedPublicEmbeddingRecord(
            text=text,
            embedding_hex=tuple(value.hex() for value in vector),
        )
        if record.text_sha256 != expected_digest:
            raise ValueError("source-v5 embedding record digest drifted")
        records.append(
            record
        )
    return PrecomputedPublicEmbeddingTable(
        source_embedder_name=embedder.name,
        embedding_width=expected_width,
        records=tuple(records),
    )


def _validate_persisted_bundle(
    *,
    protocol: RelationshipProductHorizonSourceV5EmbeddingTableProtocol,
    source_v5_admission_root: pathlib.Path,
    admission_receipt: _SourceV5AdmissionReceipt,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> Mapping[str, object]:
    required_protocol = _digest(expected_protocol_id, "expected_protocol_id")
    required_artifact = _digest(expected_artifact_id, "expected_artifact_id")
    if protocol.protocol_id != required_protocol:
        raise ValueError("external expected source-v5 embedding protocol id mismatch")
    root = pathlib.Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source-v5 embedding-table output is missing: {root}")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != _EXPECTED_OUTPUT_FILES:
        raise ValueError("source-v5 embedding-table output contains missing or extra files")
    protocol_raw = (
        relationship_product_horizon_source_v5_embedding_table_protocol_path().read_bytes()
    )
    if (root / "protocol.json").read_bytes() != protocol_raw:
        raise ValueError("source-v5 embedding-table output protocol drifted")

    inventory = _load_public_inventory(
        protocol=protocol,
        source_v5_admission_root=source_v5_admission_root,
        admission_receipt=admission_receipt,
    )
    table_raw = (root / "embedding_table.json").read_bytes()
    if len(table_raw) > protocol.output_contract["maximum_embedding_table_raw_bytes"]:
        raise ValueError("source-v5 embedding table exceeds the frozen byte ceiling")
    table = PrecomputedPublicEmbeddingTable.from_json(table_raw.decode("utf-8"))
    _validate_table(protocol=protocol, inventory=inventory, table=table)
    manifest = _load_artifact(
        root / "manifest.json", SOURCE_V5_EMBEDDING_TABLE_MANIFEST_SCHEMA_VERSION
    )
    commit = _git_commit(manifest["implementation_git_commit"])
    rebuilt = _manifest(
        root=root,
        protocol=protocol,
        implementation_git_commit=commit,
        admission_receipt=admission_receipt,
        table=table,
    )
    if manifest != rebuilt:
        raise ValueError("source-v5 embedding-table manifest drifted")
    if manifest["artifact_id"] != required_artifact:
        raise ValueError("external expected source-v5 embedding artifact id mismatch")
    return manifest


def _validate_table(
    *,
    protocol: RelationshipProductHorizonSourceV5EmbeddingTableProtocol,
    inventory: Sequence[tuple[str, str]],
    table: PrecomputedPublicEmbeddingTable,
) -> None:
    semantic = protocol.semantic_model
    observed_identity = (
        table.source_model_id,
        table.source_model_revision,
        table.source_weights_sha256,
        table.source_sentence_transformers_version,
        table.embedding_width,
    )
    expected_identity = (
        semantic["model_id"],
        semantic["model_revision"],
        semantic["weights_sha256"],
        semantic["sentence_transformers_version"],
        semantic["embedding_width"],
    )
    if observed_identity != expected_identity:
        raise ValueError("source-v5 embedding table model identity drifted")
    expected = tuple(inventory)
    observed = tuple((record.text_sha256, record.text) for record in table.records)
    if observed != expected:
        raise ValueError("source-v5 embedding table public text inventory drifted")


def _validate_embedder_identity(
    embedder: _PinnedEmbedder,
    protocol: RelationshipProductHorizonSourceV5EmbeddingTableProtocol,
) -> None:
    semantic = protocol.semantic_model
    observed = (
        embedder.model_source,
        embedder.model_revision,
        embedder.weights_sha256,
        embedder.sentence_transformers_version,
    )
    expected = (
        semantic["model_id"],
        semantic["model_revision"],
        semantic["weights_sha256"],
        semantic["sentence_transformers_version"],
    )
    if observed != expected:
        raise ValueError("source-v5 embedding-table embedder identity drifted")
    expected_name = bge_m3_weight_pinned_embedder_identity(
        model_revision=_text(semantic["model_revision"], "model_revision"),
        weights_sha256=_digest(semantic["weights_sha256"], "weights_sha256"),
        sentence_transformers_version=_text(
            semantic["sentence_transformers_version"],
            "sentence_transformers_version",
        ),
        identity_kind="model-adapter-v2",
    )
    if embedder.name != expected_name:
        raise ValueError("source-v5 embedding-table embedder name drifted")


def _manifest(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonSourceV5EmbeddingTableProtocol,
    implementation_git_commit: str,
    admission_receipt: _SourceV5AdmissionReceipt,
    table: PrecomputedPublicEmbeddingTable,
) -> dict[str, object]:
    files = []
    for relative in sorted(_EXPECTED_OUTPUT_FILES - {"manifest.json"}):
        raw = (root / relative).read_bytes()
        files.append(
            {
                "path": relative,
                "raw_bytes": len(raw),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    public_input = protocol.public_embedding_input
    semantic = protocol.semantic_model
    execution = protocol.execution_contract
    output = protocol.output_contract
    core: dict[str, object] = {
        "schema_version": SOURCE_V5_EMBEDDING_TABLE_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "protocol_raw_sha256": protocol.raw_sha256,
        "implementation_git_commit": implementation_git_commit,
        "upstream_admission_protocol_id": admission_receipt.protocol_id,
        "upstream_admission_artifact_id": admission_receipt.artifact_id,
        "upstream_admission_status": admission_receipt.status,
        "upstream_admission_validate_existing_passed": True,
        "admission_validation_completed_before_first_embedding_call": execution[
            "admission_validation_completed_before_first_embedding_call"
        ],
        "source_v5_protocol_id": admission_receipt.source_protocol_id,
        "source_v5_public_plan_content_id": admission_receipt.public_plan_content_id,
        "source_v5_public_plan_raw_sha256": public_input[
            "public_plan_raw_sha256"
        ],
        "source_v5_reader_text_inventory_sha256": public_input[
            "reader_text_inventory_sha256"
        ],
        "embedding_input_projection_sha256": public_input[
            "reader_text_inventory_sha256"
        ],
        "source_v5_reader_text_occurrence_count": public_input[
            "reader_text_occurrence_count"
        ],
        "source_v5_reader_text_unique_count": public_input[
            "reader_text_unique_count"
        ],
        "embedding_input_public_text_count": len(table.records),
        "embedding_model_id": semantic["model_id"],
        "embedding_model_revision": semantic["model_revision"],
        "embedding_weights_sha256": semantic["weights_sha256"],
        "embedding_runtime_version": semantic["sentence_transformers_version"],
        "embedding_width": semantic["embedding_width"],
        "encode_api": execution["encode_api"],
        "batch_size": execution["batch_size"],
        "embedding_api_call_count": execution["embedding_api_call_count"],
        "embedding_vector_count": len(table.records),
        "embedding_table_artifact_id": table.artifact_id,
        "embedding_table_record_count": len(table.records),
        "embedding_table_raw_bytes": (root / "embedding_table.json").stat().st_size,
        "requested_device": semantic["device"],
        "network_allowed": semantic["network_allowed"],
        "reader_fit_count": output["reader_fit_count"],
        "reader_fit_scope": output["reader_fit_scope"],
        "reader_artifact_input_count": output["reader_artifact_input_count"],
        "reader_artifact_output_count": output["reader_artifact_output_count"],
        "reader_inference_count": output["reader_inference_count"],
        "embedding_stage_sealed_payload_count": public_input[
            "sealed_payload_forwarded_to_embedder_count"
        ],
        "embedding_stage_label_payload_count": public_input[
            "challenge_or_evaluation_label_read_count"
        ],
        "outcome_value_forwarded_to_embedder_count": public_input[
            "outcome_value_forwarded_to_embedder_count"
        ],
        "validate_existing_embedding_api_call_count": execution[
            "validate_existing_embedding_api_call_count"
        ],
        "manifest_written_last": output["manifest_written_last"],
        "power_loss_durability_claimed": output[
            "power_loss_durability_claimed"
        ],
        "files": files,
        "status": "source_v5_public_embedding_table_materialized_reader_unfitted",
        "claims": protocol.claims_ceiling,
        "claim_boundary": protocol.payload["claim_boundary"],
    }
    return _with_artifact_id(core)


def _embedding(value: object, *, expected_width: int) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != expected_width:
        raise ValueError("source-v5 embedding width drifted")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError("source-v5 embedding values must be numeric")
    vector = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vector):
        raise ValueError("source-v5 embedding values must be finite")
    if math.sqrt(sum(item * item for item in vector)) <= 0.0:
        raise ValueError("source-v5 embedding norm must be positive")
    return vector


def _load_artifact(path: pathlib.Path, schema_version: str) -> dict[str, object]:
    payload = _parse_json(path.read_bytes(), source=str(path))
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"artifact schema drifted: {path}")
    artifact_id = _digest(payload.get("artifact_id"), f"{path} artifact_id")
    core = {key: value for key, value in payload.items() if key != "artifact_id"}
    if sha256_json(core) != artifact_id:
        raise ValueError(f"artifact identity drifted: {path}")
    return payload


def _with_artifact_id(core: Mapping[str, object]) -> dict[str, object]:
    return {"artifact_id": sha256_json(core), **dict(core)}


def _artifact_bytes(payload: Mapping[str, object]) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


def _write_create_only(path: pathlib.Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _object_pairs_no_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, source: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_pairs_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _text_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return tuple(_text(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _digest(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be lowercase SHA-256")
    return text


def _git_commit(value: object) -> str:
    text = _text(value, "implementation_git_commit")
    if _GIT_COMMIT.fullmatch(text) is None:
        raise ValueError("implementation_git_commit must be lowercase 40-hex")
    return text


def _exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    source: str,
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(f"{source} keys drifted: missing={missing}, extra={extra}")


__all__ = [
    "SOURCE_V5_EMBEDDING_TABLE_MANIFEST_SCHEMA_VERSION",
    "SOURCE_V5_EMBEDDING_TABLE_PROTOCOL_SCHEMA_VERSION",
    "RelationshipProductHorizonSourceV5EmbeddingTableProtocol",
    "SourceV5AdmissionValidationInputs",
    "load_relationship_product_horizon_source_v5_embedding_table_protocol",
    "materialize_relationship_product_horizon_source_v5_embedding_table",
    "relationship_product_horizon_source_v5_embedding_table_protocol_path",
    "validate_relationship_product_horizon_source_v5_embedding_table",
]
