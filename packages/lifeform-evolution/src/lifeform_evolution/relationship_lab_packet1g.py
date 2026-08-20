"""Relationship Lab P1g: frozen v3 Qwen consumer qualification.

P1g binds the successful P1f public-evidence audit to one immutable consumer
protocol before any ``relationship_transfer_v3`` Qwen output exists.  It then
consumes only content-addressed Gate 0 and P1b artifacts.  No evaluator label,
PE, credit, learning, controller, steering, or runtime slot enters this lane.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RelationshipTransferDataset,
    load_relationship_transfer_dataset,
    sha256_json,
)
from lifeform_evolution.relationship_lab_baseline import (
    action_choice_schema_path,
    stateless_prompt_path,
)
from lifeform_evolution.relationship_lab_contexts import (
    RelationshipP1ContextBundle,
    RelationshipP1RagCandidateSurface,
    relationship_p1_background_template_path,
    relationship_p1_evaluated_context_surface_sha256,
)
from lifeform_evolution.relationship_lab_gate0 import (
    FrozenBaselineAttestation,
    Gate0CalibrationConfig,
    RelationshipGate0Report,
)
from lifeform_evolution.relationship_lab_packet1 import RelationshipP1GateConfig
from lifeform_evolution.relationship_lab_packet1b import (
    RELATIONSHIP_P1B_COMPILER_VERSION,
    RelationshipP1bReadoutProfile,
    RelationshipP1bReport,
    RelationshipP1bVerdict,
    relationship_p1b_readout_prompt_path,
    relationship_p1b_readout_request_template_path,
    relationship_p1b_readout_schema_path,
)
from lifeform_evolution.relationship_lab_packet1f import (
    RelationshipP1fReport,
    RelationshipP1fVerdict,
)


RELATIONSHIP_P1G_PROTOCOL_SCHEMA_VERSION = "relationship-p1g-consumer-protocol.v1"
RELATIONSHIP_P1G_REPORT_SCHEMA_VERSION = "relationship-p1g-qualification-report.v1"
RELATIONSHIP_P1G_RAG_TOP_K = 4
_HEX_DIGITS = frozenset("0123456789abcdef")
_QUALIFICATION_ARMS = (
    "prompt-steelman",
    "rag-steelman",
    "structured-state",
)
_PROTOCOL_CLAIM_BOUNDARY = (
    "P1g freezes one development-only relationship_transfer_v3 Qwen consumer "
    "before any v3 Qwen output. It binds the successful P1f audit, exact Qwen "
    "and BGE-M3 weight digests, generation/seeds/gates, condition-aware "
    "readout, all four histories, and typed relationship-outcome RAG top-4. "
    "Formal hidden test and P2 remain closed."
)
_REPORT_CLAIM_BOUNDARY = (
    "P1g reports same-substrate strong-baseline qualification or saturation "
    "on the public synthetic v3 development split. It does not prove Volvence "
    "advantage, human readability, Appendable/Readable/Learnable/Steerable, "
    "formal held-out superiority, or product value."
)


def _asset_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def relationship_p1g_consumer_protocol_path() -> pathlib.Path:
    return _asset_dir() / "protocols" / "relationship_p1g_qwen25_3b_v1.json"


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX_DIGITS for char in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")


def _require_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_exact_object(
    value: object,
    *,
    field_name: str,
    expected_fields: set[str],
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{field_name} fields do not match the frozen schema")
    return value


@dataclass(frozen=True)
class RelationshipP1gConsumerProtocol:
    frozen_at_iso: str
    package_name: str
    dataset_schema_version: str
    truth_schema_version: str
    histories_per_user: int
    source_p1f_report_artifact_id: str
    source_p1f_required_verdict: str
    public_evidence_contract_sha256: str
    model_source: str
    model_revision: str
    model_id: str
    expected_weights_sha256: str
    dataset_fingerprint: str
    evaluated_context_surface_sha256: str
    background_templates_sha256: str
    rag_config_sha256: str
    rag_weights_sha256: str
    stateless_prompt_sha256: str
    action_choice_schema_sha256: str
    readout_profile: str
    readout_prompt_sha256: str
    readout_request_template_sha256: str
    readout_schema_sha256: str
    compiler_version: str
    expected_generation_config_sha256: str
    gate0_config_sha256: str
    p1_gate_config_sha256: str
    baseline_seed_schedule: tuple[int, ...]
    p1b_seed_schedule: tuple[int, ...]
    background_depths: tuple[int, ...]
    rag_embedder: str
    rag_model_source: str
    rag_top_k: int
    rag_candidate_surface: str
    device: str
    torch_dtype: str
    temperature: float
    top_p: float
    max_new_tokens: int
    minimum_free_bytes_before_download: int
    maximum_candidate_snapshot_bytes: int
    materialized_weights_required: bool
    current_message_participates: bool
    all_four_histories_available: bool
    v3_qwen_outputs_observed_before_freeze: int
    formal_hidden_test_opened: bool
    p2_enabled: bool
    schema_version: str = RELATIONSHIP_P1G_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1G_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("P1g consumer protocol schema_version mismatch")
        _require_timestamp(self.frozen_at_iso, "frozen_at_iso")
        if self.package_name != RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
            raise ValueError("P1g is bound to relationship_transfer_v3")
        if self.dataset_schema_version != "relationship-transfer-dataset.v3":
            raise ValueError("P1g dataset schema must be v3")
        if self.truth_schema_version != "relationship-transfer-truth.v3":
            raise ValueError("P1g truth schema must be v3")
        if self.histories_per_user != 4:
            raise ValueError("P1g requires exactly four histories per user")
        for field_name, value in (
            ("model_source", self.model_source),
            ("model_revision", self.model_revision),
            ("model_id", self.model_id),
            ("rag_embedder", self.rag_embedder),
            ("rag_model_source", self.rag_model_source),
            ("device", self.device),
            ("torch_dtype", self.torch_dtype),
        ):
            _require_text(value, field_name)
        for field_name, value in (
            ("source_p1f_report_artifact_id", self.source_p1f_report_artifact_id),
            ("public_evidence_contract_sha256", self.public_evidence_contract_sha256),
            ("expected_weights_sha256", self.expected_weights_sha256),
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("evaluated_context_surface_sha256", self.evaluated_context_surface_sha256),
            ("background_templates_sha256", self.background_templates_sha256),
            ("rag_config_sha256", self.rag_config_sha256),
            ("rag_weights_sha256", self.rag_weights_sha256),
            ("stateless_prompt_sha256", self.stateless_prompt_sha256),
            ("action_choice_schema_sha256", self.action_choice_schema_sha256),
            ("readout_prompt_sha256", self.readout_prompt_sha256),
            ("readout_request_template_sha256", self.readout_request_template_sha256),
            ("readout_schema_sha256", self.readout_schema_sha256),
            ("expected_generation_config_sha256", self.expected_generation_config_sha256),
            ("gate0_config_sha256", self.gate0_config_sha256),
            ("p1_gate_config_sha256", self.p1_gate_config_sha256),
        ):
            _require_sha256(value, field_name)
        if self.source_p1f_required_verdict != RelationshipP1fVerdict.CONSUMER_PROTOCOL_FREEZE_CANDIDATE.value:
            raise ValueError("P1g requires the successful P1f freeze verdict")
        if self.readout_profile != RelationshipP1bReadoutProfile.V2_CONDITION_AWARE.value:
            raise ValueError("P1g requires the frozen condition-aware readout")
        if self.compiler_version != RELATIONSHIP_P1B_COMPILER_VERSION:
            raise ValueError("P1g compiler diverges from the typed P1b owner")
        for field_name, values in (
            ("baseline_seed_schedule", self.baseline_seed_schedule),
            ("p1b_seed_schedule", self.p1b_seed_schedule),
        ):
            if (
                not values
                or any(type(item) is not int or item < 0 for item in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(f"{field_name} must contain unique non-negative integers")
        if (
            not self.background_depths
            or self.background_depths != tuple(sorted(set(self.background_depths)))
            or self.background_depths[0] != 0
            or self.background_depths[-1] < 8
        ):
            raise ValueError("P1g background depths must be sorted, unique, start at 0, and reach 8")
        if self.rag_top_k != RELATIONSHIP_P1G_RAG_TOP_K:
            raise ValueError("P1g RAG top-k must equal four")
        if self.rag_candidate_surface != RelationshipP1RagCandidateSurface.RELATIONSHIP_OUTCOMES_ONLY.value:
            raise ValueError("P1g RAG must use typed relationship-outcome candidates")
        if self.device != "cpu" or self.torch_dtype != "bfloat16":
            raise ValueError("P1g freezes the audited CPU bfloat16 runtime")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or self.temperature < 0.0
            or isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not 0.0 < self.top_p <= 1.0
        ):
            raise ValueError("P1g generation sampling config is invalid")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 4:
            raise ValueError("P1g max_new_tokens must be at least four")
        for field_name, value in (
            ("minimum_free_bytes_before_download", self.minimum_free_bytes_before_download),
            ("maximum_candidate_snapshot_bytes", self.maximum_candidate_snapshot_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.materialized_weights_required is not True:
            raise ValueError("P1g requires materialized weights")
        if self.current_message_participates is not True:
            raise ValueError("P1g current message must participate in condition inference")
        if self.all_four_histories_available is not True:
            raise ValueError("P1g cannot hide any of the four public histories")
        if self.v3_qwen_outputs_observed_before_freeze != 0:
            raise ValueError("P1g must freeze before the first v3 Qwen output")
        if self.formal_hidden_test_opened is not False or self.p2_enabled is not False:
            raise ValueError("P1g cannot open formal hidden test or enable P2")

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_at_iso": self.frozen_at_iso,
            "scenario": {
                "package_name": self.package_name,
                "dataset_schema_version": self.dataset_schema_version,
                "truth_schema_version": self.truth_schema_version,
                "histories_per_user": self.histories_per_user,
                "current_message_participates": self.current_message_participates,
                "all_four_histories_available": self.all_four_histories_available,
            },
            "source_lineage": {
                "source_p1f_report_artifact_id": self.source_p1f_report_artifact_id,
                "source_p1f_required_verdict": self.source_p1f_required_verdict,
                "public_evidence_contract_sha256": self.public_evidence_contract_sha256,
            },
            "candidate": {
                "model_source": self.model_source,
                "model_revision": self.model_revision,
                "model_id": self.model_id,
                "expected_weights_sha256": self.expected_weights_sha256,
            },
            "frozen_lineage": {
                "dataset_fingerprint": self.dataset_fingerprint,
                "evaluated_context_surface_sha256": self.evaluated_context_surface_sha256,
                "background_templates_sha256": self.background_templates_sha256,
                "rag_config_sha256": self.rag_config_sha256,
                "rag_weights_sha256": self.rag_weights_sha256,
                "stateless_prompt_sha256": self.stateless_prompt_sha256,
                "action_choice_schema_sha256": self.action_choice_schema_sha256,
                "readout_profile": self.readout_profile,
                "readout_prompt_sha256": self.readout_prompt_sha256,
                "readout_request_template_sha256": self.readout_request_template_sha256,
                "readout_schema_sha256": self.readout_schema_sha256,
                "compiler_version": self.compiler_version,
                "expected_generation_config_sha256": self.expected_generation_config_sha256,
                "gate0_config_sha256": self.gate0_config_sha256,
                "p1_gate_config_sha256": self.p1_gate_config_sha256,
            },
            "run_config": {
                "baseline_seed_schedule": list(self.baseline_seed_schedule),
                "p1b_seed_schedule": list(self.p1b_seed_schedule),
                "background_depths": list(self.background_depths),
                "rag_embedder": self.rag_embedder,
                "rag_model_source": self.rag_model_source,
                "rag_top_k": self.rag_top_k,
                "rag_candidate_surface": self.rag_candidate_surface,
                "device": self.device,
                "torch_dtype": self.torch_dtype,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
            },
            "materialization_guard": {
                "minimum_free_bytes_before_download": self.minimum_free_bytes_before_download,
                "maximum_candidate_snapshot_bytes": self.maximum_candidate_snapshot_bytes,
                "materialized_weights_required": self.materialized_weights_required,
            },
            "experiment_guard": {
                "v3_qwen_outputs_observed_before_freeze": (self.v3_qwen_outputs_observed_before_freeze),
                "formal_hidden_test_opened": self.formal_hidden_test_opened,
                "p2_enabled": self.p2_enabled,
            },
            "claim_boundary": _PROTOCOL_CLAIM_BOUNDARY,
        }

    @property
    def protocol_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["protocol_id"] = self.protocol_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "RelationshipP1gConsumerProtocol":
        raw = _require_exact_object(
            json.loads(encoded),
            field_name="P1g consumer protocol",
            expected_fields={
                "candidate",
                "claim_boundary",
                "experiment_guard",
                "frozen_at_iso",
                "frozen_lineage",
                "materialization_guard",
                "protocol_id",
                "run_config",
                "scenario",
                "schema_version",
                "source_lineage",
            },
        )
        scenario = _require_exact_object(
            raw["scenario"],
            field_name="P1g scenario",
            expected_fields={
                "all_four_histories_available",
                "current_message_participates",
                "dataset_schema_version",
                "histories_per_user",
                "package_name",
                "truth_schema_version",
            },
        )
        source = _require_exact_object(
            raw["source_lineage"],
            field_name="P1g source lineage",
            expected_fields={
                "public_evidence_contract_sha256",
                "source_p1f_report_artifact_id",
                "source_p1f_required_verdict",
            },
        )
        candidate = _require_exact_object(
            raw["candidate"],
            field_name="P1g candidate",
            expected_fields={
                "expected_weights_sha256",
                "model_id",
                "model_revision",
                "model_source",
            },
        )
        lineage = _require_exact_object(
            raw["frozen_lineage"],
            field_name="P1g frozen lineage",
            expected_fields={
                "action_choice_schema_sha256",
                "background_templates_sha256",
                "compiler_version",
                "dataset_fingerprint",
                "evaluated_context_surface_sha256",
                "expected_generation_config_sha256",
                "gate0_config_sha256",
                "p1_gate_config_sha256",
                "rag_config_sha256",
                "rag_weights_sha256",
                "readout_profile",
                "readout_prompt_sha256",
                "readout_request_template_sha256",
                "readout_schema_sha256",
                "stateless_prompt_sha256",
            },
        )
        run = _require_exact_object(
            raw["run_config"],
            field_name="P1g run config",
            expected_fields={
                "background_depths",
                "baseline_seed_schedule",
                "device",
                "max_new_tokens",
                "p1b_seed_schedule",
                "rag_candidate_surface",
                "rag_embedder",
                "rag_model_source",
                "rag_top_k",
                "temperature",
                "top_p",
                "torch_dtype",
            },
        )
        guard = _require_exact_object(
            raw["materialization_guard"],
            field_name="P1g materialization guard",
            expected_fields={
                "materialized_weights_required",
                "maximum_candidate_snapshot_bytes",
                "minimum_free_bytes_before_download",
            },
        )
        experiment = _require_exact_object(
            raw["experiment_guard"],
            field_name="P1g experiment guard",
            expected_fields={
                "formal_hidden_test_opened",
                "p2_enabled",
                "v3_qwen_outputs_observed_before_freeze",
            },
        )
        for field_name in (
            "baseline_seed_schedule",
            "p1b_seed_schedule",
            "background_depths",
        ):
            if not isinstance(run[field_name], list):
                raise ValueError(f"P1g {field_name} must be an array")
        protocol = cls(
            schema_version=raw["schema_version"],
            frozen_at_iso=raw["frozen_at_iso"],
            package_name=scenario["package_name"],
            dataset_schema_version=scenario["dataset_schema_version"],
            truth_schema_version=scenario["truth_schema_version"],
            histories_per_user=scenario["histories_per_user"],
            current_message_participates=scenario["current_message_participates"],
            all_four_histories_available=scenario["all_four_histories_available"],
            source_p1f_report_artifact_id=source["source_p1f_report_artifact_id"],
            source_p1f_required_verdict=source["source_p1f_required_verdict"],
            public_evidence_contract_sha256=source["public_evidence_contract_sha256"],
            model_source=candidate["model_source"],
            model_revision=candidate["model_revision"],
            model_id=candidate["model_id"],
            expected_weights_sha256=candidate["expected_weights_sha256"],
            dataset_fingerprint=lineage["dataset_fingerprint"],
            evaluated_context_surface_sha256=(lineage["evaluated_context_surface_sha256"]),
            background_templates_sha256=lineage["background_templates_sha256"],
            rag_config_sha256=lineage["rag_config_sha256"],
            rag_weights_sha256=lineage["rag_weights_sha256"],
            stateless_prompt_sha256=lineage["stateless_prompt_sha256"],
            action_choice_schema_sha256=lineage["action_choice_schema_sha256"],
            readout_profile=lineage["readout_profile"],
            readout_prompt_sha256=lineage["readout_prompt_sha256"],
            readout_request_template_sha256=(lineage["readout_request_template_sha256"]),
            readout_schema_sha256=lineage["readout_schema_sha256"],
            compiler_version=lineage["compiler_version"],
            expected_generation_config_sha256=(lineage["expected_generation_config_sha256"]),
            gate0_config_sha256=lineage["gate0_config_sha256"],
            p1_gate_config_sha256=lineage["p1_gate_config_sha256"],
            baseline_seed_schedule=tuple(run["baseline_seed_schedule"]),
            p1b_seed_schedule=tuple(run["p1b_seed_schedule"]),
            background_depths=tuple(run["background_depths"]),
            rag_embedder=run["rag_embedder"],
            rag_model_source=run["rag_model_source"],
            rag_top_k=run["rag_top_k"],
            rag_candidate_surface=run["rag_candidate_surface"],
            device=run["device"],
            torch_dtype=run["torch_dtype"],
            temperature=run["temperature"],
            top_p=run["top_p"],
            max_new_tokens=run["max_new_tokens"],
            minimum_free_bytes_before_download=(guard["minimum_free_bytes_before_download"]),
            maximum_candidate_snapshot_bytes=(guard["maximum_candidate_snapshot_bytes"]),
            materialized_weights_required=guard["materialized_weights_required"],
            v3_qwen_outputs_observed_before_freeze=(experiment["v3_qwen_outputs_observed_before_freeze"]),
            formal_hidden_test_opened=experiment["formal_hidden_test_opened"],
            p2_enabled=experiment["p2_enabled"],
        )
        _require_sha256(raw["protocol_id"], "protocol_id")
        if raw["claim_boundary"] != _PROTOCOL_CLAIM_BOUNDARY:
            raise ValueError("P1g protocol claim boundary mismatch")
        if raw["protocol_id"] != protocol.protocol_id:
            raise ValueError("P1g consumer protocol_id mismatch")
        return protocol


def load_relationship_p1g_consumer_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipP1gConsumerProtocol:
    file_path = pathlib.Path(path or relationship_p1g_consumer_protocol_path())
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return RelationshipP1gConsumerProtocol.from_json(file_path.read_text(encoding="utf-8"))


def validate_relationship_p1g_local_lineage(
    protocol: RelationshipP1gConsumerProtocol,
    *,
    source_p1f_report: RelationshipP1fReport,
) -> RelationshipTransferDataset:
    dataset = load_relationship_transfer_dataset(package_name=protocol.package_name)
    contract = dataset.public_evidence_contract
    if contract is None:
        raise ValueError("P1g v3 dataset lost its public evidence contract")
    if (
        protocol.rag_embedder != contract.semantic_audit_embedder
        or protocol.rag_model_source != contract.semantic_audit_model_source
    ):
        raise ValueError("P1g RAG identity diverges from the public evidence contract")
    expected = {
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "public_evidence_contract_sha256": contract.contract_sha256,
        "rag_weights_sha256": contract.semantic_audit_weights_sha256,
        "background_templates_sha256": _sha256_file(relationship_p1_background_template_path(protocol.package_name)),
        "stateless_prompt_sha256": _sha256_file(stateless_prompt_path()),
        "action_choice_schema_sha256": _sha256_file(action_choice_schema_path()),
        "readout_prompt_sha256": _sha256_file(
            relationship_p1b_readout_prompt_path(RelationshipP1bReadoutProfile(protocol.readout_profile))
        ),
        "readout_request_template_sha256": _sha256_file(
            relationship_p1b_readout_request_template_path(RelationshipP1bReadoutProfile(protocol.readout_profile))
        ),
        "readout_schema_sha256": _sha256_file(relationship_p1b_readout_schema_path()),
        "gate0_config_sha256": sha256_json(Gate0CalibrationConfig().to_payload()),
        "p1_gate_config_sha256": sha256_json(RelationshipP1GateConfig().to_payload()),
    }
    expected["expected_generation_config_sha256"] = sha256_json(
        {
            "device": protocol.device,
            "torch_dtype": protocol.torch_dtype,
            "temperature": protocol.temperature,
            "top_p": protocol.top_p,
            "max_new_tokens": protocol.max_new_tokens,
            "schema_sha256": protocol.action_choice_schema_sha256,
            "do_sample": protocol.temperature > 0.0,
        }
    )
    for field_name, value in expected.items():
        if getattr(protocol, field_name) != value:
            raise ValueError(f"P1g local lineage mismatch: {field_name}")
    if (
        dataset.dataset_schema_version != protocol.dataset_schema_version
        or dataset.truth_schema_version != protocol.truth_schema_version
        or {len(item.histories) for item in dataset.observations} != {protocol.histories_per_user}
    ):
        raise ValueError("P1g scenario shape diverges from the frozen protocol")
    if (
        source_p1f_report.artifact_id != protocol.source_p1f_report_artifact_id
        or source_p1f_report.verdict.value != protocol.source_p1f_required_verdict
        or source_p1f_report.package_name != protocol.package_name
        or source_p1f_report.dataset_fingerprint != protocol.dataset_fingerprint
        or source_p1f_report.public_evidence_contract_sha256 != protocol.public_evidence_contract_sha256
        or source_p1f_report.weights_sha256 != protocol.rag_weights_sha256
    ):
        raise ValueError("P1g source P1f report diverges from frozen lineage")
    if _require_timestamp(
        source_p1f_report.created_at_iso,
        "source_p1f_report.created_at_iso",
    ) > _require_timestamp(protocol.frozen_at_iso, "frozen_at_iso"):
        raise ValueError("P1g protocol cannot predate its source P1f report")
    return dataset


def validate_relationship_p1g_context_lineage(
    protocol: RelationshipP1gConsumerProtocol,
    *,
    dataset: RelationshipTransferDataset,
    bundle: RelationshipP1ContextBundle,
) -> None:
    if bundle.dataset_fingerprint != protocol.dataset_fingerprint:
        raise ValueError("P1g context bundle dataset fingerprint mismatch")
    if bundle.background_depths != protocol.background_depths:
        raise ValueError("P1g context bundle background depths mismatch")
    if bundle.background_templates_sha256 != protocol.background_templates_sha256:
        raise ValueError("P1g context background template lineage mismatch")
    if bundle.rag_config_sha256 != protocol.rag_config_sha256:
        raise ValueError("P1g context RAG lineage mismatch")
    surface = relationship_p1_evaluated_context_surface_sha256(
        bundle=bundle,
        dataset=dataset,
    )
    if surface != protocol.evaluated_context_surface_sha256:
        raise ValueError("P1g evaluated context surface mismatch")


class RelationshipP1gVerdict(str, Enum):
    FORMAL_PREREG_FREEZE_CANDIDATE = "formal_prereg_freeze_candidate"
    SCENARIO_SATURATED_AFTER_EVIDENCE_REPAIR = "scenario_saturated_after_evidence_repair"
    CONSUMER_STILL_UNDERQUALIFIED = "consumer_still_underqualified"
    CANDIDATE_GATE0_REJECTED = "candidate_gate0_rejected"
    MACHINERY_REGRESSION = "machinery_regression"


_NEXT_ACTIONS = {
    RelationshipP1gVerdict.FORMAL_PREREG_FREEZE_CANDIDATE: (
        "Freeze the formal preregistration before generating a new secret heldout."
    ),
    RelationshipP1gVerdict.SCENARIO_SATURATED_AFTER_EVIDENCE_REPAIR: (
        "Record that the frozen ordinary context baseline solves v3; do not "
        "weaken it or enter P2. Redesign the causal comparison on a new version."
    ),
    RelationshipP1gVerdict.CONSUMER_STILL_UNDERQUALIFIED: (
        "Record that semantically legible public evidence did not qualify this "
        "frozen Qwen consumer. Do not tune on these outputs; design a new versioned "
        "consumer-training split before another attempt."
    ),
    RelationshipP1gVerdict.CANDIDATE_GATE0_REJECTED: (
        "Repair Gate 0 validity or lineage before interpreting contextual arms."
    ),
    RelationshipP1gVerdict.MACHINERY_REGRESSION: (
        "Repair append/recovery/context/readout machinery and rerun this protocol."
    ),
}


def _expected_p1b_verdict_from_metrics(
    *,
    metrics: tuple[tuple[str, float, float], ...],
    machinery_ready: bool,
    all_readouts_valid: bool,
) -> RelationshipP1bVerdict:
    config = RelationshipP1GateConfig()
    by_arm = {arm: (accuracy, pair_flip) for arm, accuracy, pair_flip in metrics}
    saturated = any(by_arm[arm][0] > config.maximum_steelman_accuracy for arm in ("prompt-steelman", "rag-steelman"))
    steelmen_qualified = all(
        config.minimum_steelman_accuracy <= by_arm[arm][0] <= config.maximum_steelman_accuracy
        and by_arm[arm][1] >= config.minimum_steelman_pair_flip_rate
        for arm in ("prompt-steelman", "rag-steelman")
    )
    structured_qualified = by_arm["structured-state"][1] >= config.minimum_structured_state_pair_flip_rate
    if machinery_ready and all_readouts_valid and saturated:
        return RelationshipP1bVerdict.DATASET_SATURATED
    if machinery_ready and all_readouts_valid and steelmen_qualified and structured_qualified:
        return RelationshipP1bVerdict.QUALIFIED
    return RelationshipP1bVerdict.BASELINE_UNDERQUALIFIED


@dataclass(frozen=True)
class RelationshipP1gReport:
    created_at_iso: str
    consumer_protocol_id: str
    source_p1f_report_artifact_id: str
    public_evidence_contract_sha256: str
    package_name: str
    dataset_fingerprint: str
    model_id: str
    weights_sha256: str
    generation_config_sha256: str
    baseline_attestation_id: str
    gate0_report_artifact_id: str
    gate0_machinery_ready: bool
    gate0_passed: bool
    baseline_accuracy: float
    p1b_report_artifact_id: str | None
    p1b_verdict: str | None
    p1_machinery_ready: bool | None
    all_readouts_valid: bool | None
    qualification_metrics: tuple[tuple[str, float, float], ...]
    verdict: RelationshipP1gVerdict
    schema_version: str = RELATIONSHIP_P1G_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1G_REPORT_SCHEMA_VERSION:
            raise ValueError("P1g report schema_version mismatch")
        _require_timestamp(self.created_at_iso, "created_at_iso")
        if self.package_name != RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
            raise ValueError("P1g report package mismatch")
        _require_text(self.model_id, "model_id")
        for field_name, value in (
            ("consumer_protocol_id", self.consumer_protocol_id),
            ("source_p1f_report_artifact_id", self.source_p1f_report_artifact_id),
            ("public_evidence_contract_sha256", self.public_evidence_contract_sha256),
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("weights_sha256", self.weights_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
            ("baseline_attestation_id", self.baseline_attestation_id),
            ("gate0_report_artifact_id", self.gate0_report_artifact_id),
        ):
            _require_sha256(value, field_name)
        if type(self.gate0_machinery_ready) is not bool or type(self.gate0_passed) is not bool:
            raise ValueError("P1g Gate 0 readiness fields must be boolean")
        if (
            isinstance(self.baseline_accuracy, bool)
            or not isinstance(self.baseline_accuracy, (int, float))
            or not 0.0 <= self.baseline_accuracy <= 1.0
        ):
            raise ValueError("P1g baseline accuracy must be in [0, 1]")
        if not isinstance(self.verdict, RelationshipP1gVerdict):
            raise ValueError("P1g verdict must be typed")
        if self.p1b_report_artifact_id is None:
            if (
                self.p1b_verdict is not None
                or self.p1_machinery_ready is not None
                or self.all_readouts_valid is not None
                or self.qualification_metrics
                or self.gate0_passed
            ):
                raise ValueError("P1g report has inconsistent missing P1b evidence")
            expected = RelationshipP1gVerdict.CANDIDATE_GATE0_REJECTED
        else:
            _require_sha256(self.p1b_report_artifact_id, "p1b_report_artifact_id")
            if not self.gate0_passed:
                raise ValueError("P1g cannot carry P1b evidence after Gate 0 failure")
            if type(self.p1_machinery_ready) is not bool or type(self.all_readouts_valid) is not bool:
                raise ValueError("P1g P1b readiness fields must be boolean")
            if tuple(row[0] for row in self.qualification_metrics) != _QUALIFICATION_ARMS:
                raise ValueError("P1g qualification arms are incomplete or unordered")
            for arm, accuracy, pair_flip_rate in self.qualification_metrics:
                for field_name, value in (
                    (f"{arm}.accuracy", accuracy),
                    (f"{arm}.pair_flip_rate", pair_flip_rate),
                ):
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                        raise ValueError(f"P1g {field_name} must be in [0, 1]")
            expected_p1b = _expected_p1b_verdict_from_metrics(
                metrics=self.qualification_metrics,
                machinery_ready=self.p1_machinery_ready,
                all_readouts_valid=self.all_readouts_valid,
            )
            if self.p1b_verdict != expected_p1b.value:
                raise ValueError("P1g P1b verdict diverges from frozen thresholds")
            if not self.p1_machinery_ready or not self.all_readouts_valid:
                expected = RelationshipP1gVerdict.MACHINERY_REGRESSION
            elif expected_p1b is RelationshipP1bVerdict.DATASET_SATURATED:
                expected = RelationshipP1gVerdict.SCENARIO_SATURATED_AFTER_EVIDENCE_REPAIR
            elif expected_p1b is RelationshipP1bVerdict.QUALIFIED:
                expected = RelationshipP1gVerdict.FORMAL_PREREG_FREEZE_CANDIDATE
            else:
                expected = RelationshipP1gVerdict.CONSUMER_STILL_UNDERQUALIFIED
        if self.verdict is not expected:
            raise ValueError("P1g verdict diverges from frozen qualification evidence")

    @property
    def next_action(self) -> str:
        return _NEXT_ACTIONS[self.verdict]

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "consumer_protocol_id": self.consumer_protocol_id,
            "source_p1f_report_artifact_id": self.source_p1f_report_artifact_id,
            "public_evidence_contract_sha256": self.public_evidence_contract_sha256,
            "package_name": self.package_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "baseline_attestation_id": self.baseline_attestation_id,
            "gate0_report_artifact_id": self.gate0_report_artifact_id,
            "gate0_machinery_ready": self.gate0_machinery_ready,
            "gate0_passed": self.gate0_passed,
            "baseline_accuracy": self.baseline_accuracy,
            "p1b_report_artifact_id": self.p1b_report_artifact_id,
            "p1b_verdict": self.p1b_verdict,
            "p1_machinery_ready": self.p1_machinery_ready,
            "all_readouts_valid": self.all_readouts_valid,
            "qualification_metrics": {
                arm: {"accuracy": accuracy, "pair_flip_rate": pair_flip_rate}
                for arm, accuracy, pair_flip_rate in self.qualification_metrics
            },
            "verdict": self.verdict.value,
            "next_action": self.next_action,
            "claim_boundary": _REPORT_CLAIM_BOUNDARY,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "RelationshipP1gReport":
        payload = _require_exact_object(
            json.loads(encoded),
            field_name="P1g report",
            expected_fields={
                "all_readouts_valid",
                "artifact_id",
                "baseline_accuracy",
                "baseline_attestation_id",
                "claim_boundary",
                "consumer_protocol_id",
                "created_at_iso",
                "dataset_fingerprint",
                "gate0_machinery_ready",
                "gate0_passed",
                "gate0_report_artifact_id",
                "generation_config_sha256",
                "model_id",
                "next_action",
                "p1_machinery_ready",
                "p1b_report_artifact_id",
                "p1b_verdict",
                "package_name",
                "public_evidence_contract_sha256",
                "qualification_metrics",
                "schema_version",
                "source_p1f_report_artifact_id",
                "verdict",
                "weights_sha256",
            },
        )
        raw_metrics = payload["qualification_metrics"]
        if not isinstance(raw_metrics, dict):
            raise ValueError("P1g qualification_metrics must be an object")
        metrics: list[tuple[str, float, float]] = []
        for arm in _QUALIFICATION_ARMS:
            if arm not in raw_metrics:
                continue
            row = _require_exact_object(
                raw_metrics[arm],
                field_name=f"P1g qualification_metrics.{arm}",
                expected_fields={"accuracy", "pair_flip_rate"},
            )
            metrics.append((arm, row["accuracy"], row["pair_flip_rate"]))
        if set(raw_metrics) != {row[0] for row in metrics}:
            raise ValueError("P1g qualification_metrics contains an unknown arm")
        try:
            verdict = RelationshipP1gVerdict(payload["verdict"])
        except (TypeError, ValueError) as exc:
            raise ValueError("P1g verdict is invalid") from exc
        report = cls(
            schema_version=payload["schema_version"],
            created_at_iso=payload["created_at_iso"],
            consumer_protocol_id=payload["consumer_protocol_id"],
            source_p1f_report_artifact_id=(payload["source_p1f_report_artifact_id"]),
            public_evidence_contract_sha256=(payload["public_evidence_contract_sha256"]),
            package_name=payload["package_name"],
            dataset_fingerprint=payload["dataset_fingerprint"],
            model_id=payload["model_id"],
            weights_sha256=payload["weights_sha256"],
            generation_config_sha256=payload["generation_config_sha256"],
            baseline_attestation_id=payload["baseline_attestation_id"],
            gate0_report_artifact_id=payload["gate0_report_artifact_id"],
            gate0_machinery_ready=payload["gate0_machinery_ready"],
            gate0_passed=payload["gate0_passed"],
            baseline_accuracy=payload["baseline_accuracy"],
            p1b_report_artifact_id=payload["p1b_report_artifact_id"],
            p1b_verdict=payload["p1b_verdict"],
            p1_machinery_ready=payload["p1_machinery_ready"],
            all_readouts_valid=payload["all_readouts_valid"],
            qualification_metrics=tuple(metrics),
            verdict=verdict,
        )
        _require_sha256(payload["artifact_id"], "artifact_id")
        if (
            payload["claim_boundary"] != _REPORT_CLAIM_BOUNDARY
            or payload["next_action"] != report.next_action
            or payload["artifact_id"] != report.artifact_id
        ):
            raise ValueError("P1g report derived fields or artifact_id mismatch")
        return report


def _qualification_metrics(
    report: RelationshipP1bReport,
) -> tuple[tuple[str, float, float], ...]:
    by_arm = {arm: dict(metrics) for arm, metrics in report.arm_metrics}
    return tuple(
        (
            arm,
            float(by_arm[arm]["accuracy"]),
            float(by_arm[arm]["pair_flip_rate"]),
        )
        for arm in _QUALIFICATION_ARMS
    )


def assess_relationship_packet1g(
    *,
    protocol: RelationshipP1gConsumerProtocol,
    source_p1f_report: RelationshipP1fReport,
    baseline: FrozenBaselineAttestation,
    gate0_report: RelationshipGate0Report,
    p1b_report: RelationshipP1bReport | None,
    created_at_iso: str | None = None,
) -> RelationshipP1gReport:
    validate_relationship_p1g_local_lineage(
        protocol,
        source_p1f_report=source_p1f_report,
    )
    if (
        baseline.model_id != protocol.model_id
        or baseline.weights_sha256 != protocol.expected_weights_sha256
        or baseline.dataset_fingerprint != protocol.dataset_fingerprint
        or baseline.prompt_sha256 != protocol.stateless_prompt_sha256
        or baseline.generation_config_sha256 != protocol.expected_generation_config_sha256
        or baseline.seed_schedule_sha256 != sha256_json(protocol.baseline_seed_schedule)
        or baseline.hidden_test_opened
    ):
        raise ValueError("P1g Gate 0 baseline diverges from frozen protocol")
    if (
        gate0_report.dataset_fingerprint != protocol.dataset_fingerprint
        or gate0_report.baseline_attestation_id != baseline.artifact_id
        or sha256_json(gate0_report.config.to_payload()) != protocol.gate0_config_sha256
    ):
        raise ValueError("P1g Gate 0 report diverges from frozen protocol")
    timestamp = created_at_iso or datetime.now(timezone.utc).isoformat()
    if _require_timestamp(timestamp, "created_at_iso") < _require_timestamp(
        protocol.frozen_at_iso,
        "protocol.frozen_at_iso",
    ):
        raise ValueError("P1g report cannot predate its frozen protocol")
    common = {
        "created_at_iso": timestamp,
        "consumer_protocol_id": protocol.protocol_id,
        "source_p1f_report_artifact_id": source_p1f_report.artifact_id,
        "public_evidence_contract_sha256": protocol.public_evidence_contract_sha256,
        "package_name": protocol.package_name,
        "dataset_fingerprint": protocol.dataset_fingerprint,
        "model_id": baseline.model_id,
        "weights_sha256": baseline.weights_sha256,
        "generation_config_sha256": baseline.generation_config_sha256,
        "baseline_attestation_id": baseline.artifact_id,
        "gate0_report_artifact_id": gate0_report.artifact_id,
        "gate0_machinery_ready": gate0_report.machinery_ready,
        "baseline_accuracy": baseline.accuracy,
    }
    if not gate0_report.gate0_passed:
        if p1b_report is not None:
            raise ValueError("P1g cannot consume P1b after Gate 0 failure")
        return RelationshipP1gReport(
            **common,
            gate0_passed=False,
            p1b_report_artifact_id=None,
            p1b_verdict=None,
            p1_machinery_ready=None,
            all_readouts_valid=None,
            qualification_metrics=(),
            verdict=RelationshipP1gVerdict.CANDIDATE_GATE0_REJECTED,
        )
    if p1b_report is None:
        raise ValueError("P1g Gate 0 passed but P1b report is missing")
    if (
        p1b_report.dataset_fingerprint != protocol.dataset_fingerprint
        or p1b_report.evaluated_context_surface_sha256 != protocol.evaluated_context_surface_sha256
        or p1b_report.background_templates_sha256 != protocol.background_templates_sha256
        or p1b_report.rag_config_sha256 != protocol.rag_config_sha256
        or p1b_report.seed_schedule_sha256 != sha256_json(protocol.p1b_seed_schedule)
        or p1b_report.p1_gate_config_sha256 != protocol.p1_gate_config_sha256
        or p1b_report.model_id != baseline.model_id
        or p1b_report.weights_sha256 != baseline.weights_sha256
        or p1b_report.generation_config_sha256 != baseline.generation_config_sha256
        or p1b_report.gate0_baseline_attestation_id != baseline.artifact_id
        or p1b_report.readout_prompt_sha256 != protocol.readout_prompt_sha256
        or p1b_report.readout_request_template_sha256 != protocol.readout_request_template_sha256
        or p1b_report.readout_schema_sha256 != protocol.readout_schema_sha256
        or p1b_report.compiler_version != protocol.compiler_version
    ):
        raise ValueError("P1g P1b report diverges from frozen consumer lineage")
    metrics = _qualification_metrics(p1b_report)
    expected_p1b = _expected_p1b_verdict_from_metrics(
        metrics=metrics,
        machinery_ready=p1b_report.p1_machinery_ready,
        all_readouts_valid=p1b_report.all_readouts_valid,
    )
    if p1b_report.verdict is not expected_p1b:
        raise ValueError("P1g P1b verdict diverges from its metrics")
    if not p1b_report.p1_machinery_ready or not p1b_report.all_readouts_valid:
        verdict = RelationshipP1gVerdict.MACHINERY_REGRESSION
    elif expected_p1b is RelationshipP1bVerdict.DATASET_SATURATED:
        verdict = RelationshipP1gVerdict.SCENARIO_SATURATED_AFTER_EVIDENCE_REPAIR
    elif expected_p1b is RelationshipP1bVerdict.QUALIFIED:
        verdict = RelationshipP1gVerdict.FORMAL_PREREG_FREEZE_CANDIDATE
    else:
        verdict = RelationshipP1gVerdict.CONSUMER_STILL_UNDERQUALIFIED
    return RelationshipP1gReport(
        **common,
        gate0_passed=True,
        p1b_report_artifact_id=p1b_report.artifact_id,
        p1b_verdict=p1b_report.verdict.value,
        p1_machinery_ready=p1b_report.p1_machinery_ready,
        all_readouts_valid=p1b_report.all_readouts_valid,
        qualification_metrics=metrics,
        verdict=verdict,
    )


def write_relationship_packet1g_report(
    report: RelationshipP1gReport,
    *,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    target = pathlib.Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "packet1g_report.json"
    markdown_path = target / "packet1g_report.md"
    existing = tuple(path for path in (json_path, markdown_path) if path.exists())
    if existing:
        raise FileExistsError(f"P1g report files already exist: {existing}")
    json_path.write_text(report.to_json(), encoding="utf-8")
    lines = [
        "# Relationship Lab P1g v3 consumer qualification",
        "",
        f"- artifact_id: {report.artifact_id}",
        f"- consumer_protocol_id: {report.consumer_protocol_id}",
        f"- source_p1f_report_artifact_id: {report.source_p1f_report_artifact_id}",
        f"- dataset_fingerprint: {report.dataset_fingerprint}",
        f"- model_id: {report.model_id}",
        f"- gate0_passed: {str(report.gate0_passed).lower()}",
        f"- baseline_accuracy: {report.baseline_accuracy:.3f}",
        f"- verdict: {report.verdict.value}",
        "",
    ]
    if report.qualification_metrics:
        lines.extend(
            [
                "| Arm | accuracy | pair flip |",
                "|---|---:|---:|",
                *(
                    f"| {arm} | {accuracy:.3f} | {pair_flip_rate:.3f} |"
                    for arm, accuracy, pair_flip_rate in report.qualification_metrics
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Required next action",
            "",
            report.next_action,
            "",
            "## Claim boundary",
            "",
            _REPORT_CLAIM_BOUNDARY,
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def load_relationship_packet1g_report(
    path: pathlib.Path,
) -> RelationshipP1gReport:
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return RelationshipP1gReport.from_json(file_path.read_text(encoding="utf-8"))


__all__ = [
    "RELATIONSHIP_P1G_PROTOCOL_SCHEMA_VERSION",
    "RELATIONSHIP_P1G_RAG_TOP_K",
    "RELATIONSHIP_P1G_REPORT_SCHEMA_VERSION",
    "RelationshipP1gConsumerProtocol",
    "RelationshipP1gReport",
    "RelationshipP1gVerdict",
    "assess_relationship_packet1g",
    "load_relationship_p1g_consumer_protocol",
    "load_relationship_packet1g_report",
    "relationship_p1g_consumer_protocol_path",
    "validate_relationship_p1g_context_lineage",
    "validate_relationship_p1g_local_lineage",
    "write_relationship_packet1g_report",
]
