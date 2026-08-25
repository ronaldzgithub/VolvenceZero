"""Relationship Lab P1c: stronger-substrate qualification fork.

P1c does not add another prompt or change a gate. It freezes one development
candidate, requires a fresh Gate 0 attestation for that exact substrate, then
routes the owner-published P1b report to one of three scientific next steps:
freeze a formal preregistration candidate, version a saturated scenario, or
rewrite the public evidence contract. Infrastructure failures remain a
separate non-qualification verdict.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from lifeform_domain_emogpt.lab import (
    load_relationship_transfer_dataset,
    sha256_json,
)
from lifeform_evolution.relationship_lab_baseline import (
    action_choice_schema_path,
    stateless_prompt_path,
)
from lifeform_evolution.relationship_lab_gate0 import (
    FrozenBaselineAttestation,
    Gate0CalibrationConfig,
    RelationshipGate0Report,
)
from lifeform_evolution.relationship_lab_contexts import (
    relationship_p1_background_template_path,
)
from lifeform_evolution.relationship_lab_packet1 import RelationshipP1GateConfig
from lifeform_evolution.relationship_lab_packet1b import (
    RELATIONSHIP_P1B_COMPILER_VERSION,
    RELATIONSHIP_P1B_RAG_TOP_K,
    RelationshipP1bReport,
    RelationshipP1bVerdict,
    relationship_p1b_readout_prompt_path,
    relationship_p1b_readout_request_template_path,
    relationship_p1b_readout_schema_path,
)


RELATIONSHIP_P1C_PROTOCOL_SCHEMA_VERSION = "relationship-p1c-candidate-protocol.v2"
RELATIONSHIP_P1C_REPORT_SCHEMA_VERSION = "relationship-p1c-qualification-report.v1"
_HEX_DIGITS = frozenset("0123456789abcdef")
_QUALIFICATION_ARMS = (
    "prompt-steelman",
    "rag-steelman",
    "structured-state",
)
_PROTOCOL_CLAIM_BOUNDARY = (
    "This protocol freezes a stronger-substrate development qualification run. "
    "The materialized Gate 0 attestation, not the upstream model name or branch, "
    "freezes the actual weights. Formal hidden test remains unopened."
)
_REPORT_CLAIM_BOUNDARY = (
    "P1c is development-only routing evidence. It can qualify a baseline for "
    "formal preregistration, detect scenario saturation, or reject the public "
    "evidence contract. It does not prove Appendable, Readable, Learnable, "
    "Steerable, formal held-out superiority, or product value."
)


def _asset_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def relationship_p1c_candidate_protocol_path() -> pathlib.Path:
    return _asset_dir() / "protocols" / "relationship_p1c_qwen25_3b_v2.json"


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX_DIGITS for char in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")


def _require_iso_timestamp(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")


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
class RelationshipP1cCandidateProtocol:
    candidate_id: str
    model_source: str
    model_revision: str
    model_id: str
    candidate_capacity_tier: str
    reference_model_id: str
    reference_capacity_tier: str
    reference_weights_sha256: str
    reference_p1b_report_artifact_id: str
    dataset_fingerprint: str
    evaluated_context_surface_sha256: str
    background_templates_sha256: str
    rag_config_sha256: str
    stateless_prompt_sha256: str
    action_choice_schema_sha256: str
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
    device: str
    torch_dtype: str
    temperature: float
    top_p: float
    max_new_tokens: int
    minimum_free_bytes_before_download: int
    maximum_candidate_snapshot_bytes: int
    materialized_weights_required: bool
    formal_hidden_test_opened: bool
    schema_version: str = RELATIONSHIP_P1C_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1C_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("P1c candidate protocol schema_version mismatch")
        for field_name, value in (
            ("candidate_id", self.candidate_id),
            ("model_source", self.model_source),
            ("model_revision", self.model_revision),
            ("model_id", self.model_id),
            ("candidate_capacity_tier", self.candidate_capacity_tier),
            ("reference_model_id", self.reference_model_id),
            ("reference_capacity_tier", self.reference_capacity_tier),
            ("rag_embedder", self.rag_embedder),
            ("rag_model_source", self.rag_model_source),
            ("device", self.device),
            ("torch_dtype", self.torch_dtype),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.model_id == self.reference_model_id:
            raise ValueError("P1c candidate must differ from the reference model")
        for field_name, value in (
            ("reference_weights_sha256", self.reference_weights_sha256),
            (
                "reference_p1b_report_artifact_id",
                self.reference_p1b_report_artifact_id,
            ),
            ("dataset_fingerprint", self.dataset_fingerprint),
            (
                "evaluated_context_surface_sha256",
                self.evaluated_context_surface_sha256,
            ),
            ("background_templates_sha256", self.background_templates_sha256),
            ("rag_config_sha256", self.rag_config_sha256),
            ("stateless_prompt_sha256", self.stateless_prompt_sha256),
            ("action_choice_schema_sha256", self.action_choice_schema_sha256),
            ("readout_prompt_sha256", self.readout_prompt_sha256),
            (
                "readout_request_template_sha256",
                self.readout_request_template_sha256,
            ),
            ("readout_schema_sha256", self.readout_schema_sha256),
            (
                "expected_generation_config_sha256",
                self.expected_generation_config_sha256,
            ),
            ("gate0_config_sha256", self.gate0_config_sha256),
            ("p1_gate_config_sha256", self.p1_gate_config_sha256),
        ):
            _require_sha256(value, field_name)
        if self.compiler_version != RELATIONSHIP_P1B_COMPILER_VERSION:
            raise ValueError("P1c compiler version diverges from P1b")
        for field_name, values in (
            ("baseline_seed_schedule", self.baseline_seed_schedule),
            ("p1b_seed_schedule", self.p1b_seed_schedule),
        ):
            if (
                not values
                or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"{field_name} must contain unique non-negative integers")
        if (
            not self.background_depths
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in self.background_depths)
            or self.background_depths != tuple(sorted(set(self.background_depths)))
            or self.background_depths[0] != 0
            or self.background_depths[-1] < 8
        ):
            raise ValueError("P1c background_depths must be sorted, unique, start at 0, and reach 8")
        if self.rag_top_k != RELATIONSHIP_P1B_RAG_TOP_K:
            raise ValueError("P1c must preserve the frozen P1b RAG top-k")
        if self.device != "cpu" or self.torch_dtype != "bfloat16":
            raise ValueError("P1c v1 freezes the audited CPU bfloat16 runtime")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or self.temperature < 0.0
            or isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not 0.0 < self.top_p <= 1.0
        ):
            raise ValueError("P1c generation sampling config is invalid")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 4:
            raise ValueError("P1c max_new_tokens must be >= 4")
        for field_name, value in (
            (
                "minimum_free_bytes_before_download",
                self.minimum_free_bytes_before_download,
            ),
            (
                "maximum_candidate_snapshot_bytes",
                self.maximum_candidate_snapshot_bytes,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.materialized_weights_required is not True:
            raise ValueError("P1c requires materialized weights to be attested")
        if not isinstance(self.formal_hidden_test_opened, bool):
            raise ValueError("P1c formal_hidden_test_opened must be boolean")
        if self.formal_hidden_test_opened:
            raise ValueError("P1c development protocol cannot open formal hidden test")

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate": {
                "candidate_id": self.candidate_id,
                "model_source": self.model_source,
                "model_revision": self.model_revision,
                "model_id": self.model_id,
                "capacity_tier": self.candidate_capacity_tier,
            },
            "reference": {
                "model_id": self.reference_model_id,
                "capacity_tier": self.reference_capacity_tier,
                "weights_sha256": self.reference_weights_sha256,
                "p1b_report_artifact_id": self.reference_p1b_report_artifact_id,
            },
            "frozen_lineage": {
                "dataset_fingerprint": self.dataset_fingerprint,
                "evaluated_context_surface_sha256": (self.evaluated_context_surface_sha256),
                "background_templates_sha256": self.background_templates_sha256,
                "rag_config_sha256": self.rag_config_sha256,
                "stateless_prompt_sha256": self.stateless_prompt_sha256,
                "action_choice_schema_sha256": self.action_choice_schema_sha256,
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
            "formal_hidden_test_opened": self.formal_hidden_test_opened,
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
    def from_json(cls, encoded: str) -> "RelationshipP1cCandidateProtocol":
        raw = json.loads(encoded)
        top = _require_exact_object(
            raw,
            field_name="P1c candidate protocol",
            expected_fields={
                "candidate",
                "claim_boundary",
                "formal_hidden_test_opened",
                "frozen_lineage",
                "materialization_guard",
                "protocol_id",
                "reference",
                "run_config",
                "schema_version",
            },
        )
        candidate = _require_exact_object(
            top["candidate"],
            field_name="P1c candidate",
            expected_fields={
                "candidate_id",
                "capacity_tier",
                "model_id",
                "model_revision",
                "model_source",
            },
        )
        reference = _require_exact_object(
            top["reference"],
            field_name="P1c reference",
            expected_fields={
                "capacity_tier",
                "model_id",
                "p1b_report_artifact_id",
                "weights_sha256",
            },
        )
        lineage = _require_exact_object(
            top["frozen_lineage"],
            field_name="P1c frozen_lineage",
            expected_fields={
                "action_choice_schema_sha256",
                "background_templates_sha256",
                "compiler_version",
                "dataset_fingerprint",
                "evaluated_context_surface_sha256",
                "expected_generation_config_sha256",
                "gate0_config_sha256",
                "p1_gate_config_sha256",
                "readout_prompt_sha256",
                "readout_request_template_sha256",
                "readout_schema_sha256",
                "rag_config_sha256",
                "stateless_prompt_sha256",
            },
        )
        run_config = _require_exact_object(
            top["run_config"],
            field_name="P1c run_config",
            expected_fields={
                "background_depths",
                "baseline_seed_schedule",
                "device",
                "max_new_tokens",
                "p1b_seed_schedule",
                "rag_embedder",
                "rag_model_source",
                "rag_top_k",
                "temperature",
                "top_p",
                "torch_dtype",
            },
        )
        guard = _require_exact_object(
            top["materialization_guard"],
            field_name="P1c materialization_guard",
            expected_fields={
                "materialized_weights_required",
                "maximum_candidate_snapshot_bytes",
                "minimum_free_bytes_before_download",
            },
        )
        for field_name in (
            "baseline_seed_schedule",
            "p1b_seed_schedule",
            "background_depths",
        ):
            value = run_config[field_name]
            if not isinstance(value, list):
                raise ValueError(f"P1c {field_name} must be a list")
        protocol = cls(
            schema_version=top["schema_version"],
            candidate_id=candidate["candidate_id"],
            model_source=candidate["model_source"],
            model_revision=candidate["model_revision"],
            model_id=candidate["model_id"],
            candidate_capacity_tier=candidate["capacity_tier"],
            reference_model_id=reference["model_id"],
            reference_capacity_tier=reference["capacity_tier"],
            reference_weights_sha256=reference["weights_sha256"],
            reference_p1b_report_artifact_id=reference["p1b_report_artifact_id"],
            dataset_fingerprint=lineage["dataset_fingerprint"],
            evaluated_context_surface_sha256=lineage["evaluated_context_surface_sha256"],
            background_templates_sha256=lineage["background_templates_sha256"],
            rag_config_sha256=lineage["rag_config_sha256"],
            stateless_prompt_sha256=lineage["stateless_prompt_sha256"],
            action_choice_schema_sha256=lineage["action_choice_schema_sha256"],
            readout_prompt_sha256=lineage["readout_prompt_sha256"],
            readout_request_template_sha256=lineage["readout_request_template_sha256"],
            readout_schema_sha256=lineage["readout_schema_sha256"],
            compiler_version=lineage["compiler_version"],
            expected_generation_config_sha256=lineage["expected_generation_config_sha256"],
            gate0_config_sha256=lineage["gate0_config_sha256"],
            p1_gate_config_sha256=lineage["p1_gate_config_sha256"],
            baseline_seed_schedule=tuple(run_config["baseline_seed_schedule"]),
            p1b_seed_schedule=tuple(run_config["p1b_seed_schedule"]),
            background_depths=tuple(run_config["background_depths"]),
            rag_embedder=run_config["rag_embedder"],
            rag_model_source=run_config["rag_model_source"],
            rag_top_k=run_config["rag_top_k"],
            device=run_config["device"],
            torch_dtype=run_config["torch_dtype"],
            temperature=run_config["temperature"],
            top_p=run_config["top_p"],
            max_new_tokens=run_config["max_new_tokens"],
            minimum_free_bytes_before_download=guard["minimum_free_bytes_before_download"],
            maximum_candidate_snapshot_bytes=guard["maximum_candidate_snapshot_bytes"],
            materialized_weights_required=guard["materialized_weights_required"],
            formal_hidden_test_opened=top["formal_hidden_test_opened"],
        )
        protocol_id = top["protocol_id"]
        _require_sha256(protocol_id, "protocol_id")
        if top["claim_boundary"] != _PROTOCOL_CLAIM_BOUNDARY:
            raise ValueError("P1c protocol claim boundary mismatch")
        if protocol_id != protocol.protocol_id:
            raise ValueError("P1c candidate protocol_id mismatch")
        return protocol


def load_relationship_p1c_candidate_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipP1cCandidateProtocol:
    file_path = pathlib.Path(path or relationship_p1c_candidate_protocol_path())
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return RelationshipP1cCandidateProtocol.from_json(file_path.read_text(encoding="utf-8"))


def validate_relationship_p1c_local_lineage(
    protocol: RelationshipP1cCandidateProtocol,
) -> None:
    dataset = load_relationship_transfer_dataset()
    expected_hashes = {
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "background_templates_sha256": _sha256_file(relationship_p1_background_template_path()),
        "stateless_prompt_sha256": _sha256_file(stateless_prompt_path()),
        "action_choice_schema_sha256": _sha256_file(action_choice_schema_path()),
        "readout_prompt_sha256": _sha256_file(relationship_p1b_readout_prompt_path()),
        "readout_request_template_sha256": _sha256_file(relationship_p1b_readout_request_template_path()),
        "readout_schema_sha256": _sha256_file(relationship_p1b_readout_schema_path()),
        "gate0_config_sha256": sha256_json(Gate0CalibrationConfig().to_payload()),
        "p1_gate_config_sha256": sha256_json(RelationshipP1GateConfig().to_payload()),
    }
    generation_hash = sha256_json(
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
    expected_hashes["expected_generation_config_sha256"] = generation_hash
    for field_name, observed in expected_hashes.items():
        if getattr(protocol, field_name) != observed:
            raise ValueError(f"P1c local lineage mismatch: {field_name}")


class RelationshipP1cVerdict(str, Enum):
    FORMAL_PREREG_FREEZE_CANDIDATE = "formal_prereg_freeze_candidate"
    VERSION_SCENARIO_DATASET_SATURATED = "version_scenario_dataset_saturated"
    REWRITE_PUBLIC_EVIDENCE_CONTRACT = "rewrite_public_evidence_contract"
    CANDIDATE_GATE0_REJECTED = "candidate_gate0_rejected"
    MACHINERY_REGRESSION = "machinery_regression"


_NEXT_ACTIONS = {
    RelationshipP1cVerdict.FORMAL_PREREG_FREEZE_CANDIDATE: (
        "Freeze the materialized weights and complete formal preregistration before generating secret heldout."
    ),
    RelationshipP1cVerdict.VERSION_SCENARIO_DATASET_SATURATED: (
        "Version relationship_transfer_v2; increase hidden-dynamics and "
        "cross-surface transfer difficulty without weakening baselines."
    ),
    RelationshipP1cVerdict.REWRITE_PUBLIC_EVIDENCE_CONTRACT: (
        "Rewrite the public evidence and label contract before adding latent carriers, PE learning, or steering."
    ),
    RelationshipP1cVerdict.CANDIDATE_GATE0_REJECTED: (
        "Reject this candidate run and repair its Gate 0 validity or lineage before any P1b interpretation."
    ),
    RelationshipP1cVerdict.MACHINERY_REGRESSION: (
        "Repair the append/recovery/lineage/readout machinery regression and rerun the frozen protocol."
    ),
}


@dataclass(frozen=True)
class RelationshipP1cReport:
    created_at_iso: str
    candidate_protocol_id: str
    candidate_model_id: str
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
    verdict: RelationshipP1cVerdict
    schema_version: str = RELATIONSHIP_P1C_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1C_REPORT_SCHEMA_VERSION:
            raise ValueError("P1c report schema_version mismatch")
        _require_iso_timestamp(self.created_at_iso, "created_at_iso")
        if not isinstance(self.candidate_model_id, str) or not self.candidate_model_id.strip():
            raise ValueError("P1c candidate_model_id must be non-empty")
        for field_name, value in (
            ("candidate_protocol_id", self.candidate_protocol_id),
            ("weights_sha256", self.weights_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
            ("baseline_attestation_id", self.baseline_attestation_id),
            ("gate0_report_artifact_id", self.gate0_report_artifact_id),
        ):
            _require_sha256(value, field_name)
        for field_name, value in (
            ("gate0_machinery_ready", self.gate0_machinery_ready),
            ("gate0_passed", self.gate0_passed),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{field_name} must be boolean")
        if (
            isinstance(self.baseline_accuracy, bool)
            or not isinstance(self.baseline_accuracy, (int, float))
            or not 0.0 <= self.baseline_accuracy <= 1.0
        ):
            raise ValueError("P1c baseline_accuracy must be in [0, 1]")
        if not isinstance(self.verdict, RelationshipP1cVerdict):
            raise ValueError("P1c verdict must be typed")
        if self.p1b_report_artifact_id is None:
            if (
                any(
                    value is not None
                    for value in (
                        self.p1b_verdict,
                        self.p1_machinery_ready,
                        self.all_readouts_valid,
                    )
                )
                or self.qualification_metrics
            ):
                raise ValueError("P1c report has P1b fields without a P1b artifact")
            if self.gate0_passed:
                raise ValueError("P1c Gate 0 PASS requires a P1b artifact")
        else:
            _require_sha256(
                self.p1b_report_artifact_id,
                "p1b_report_artifact_id",
            )
            if self.p1b_verdict not in {item.value for item in RelationshipP1bVerdict}:
                raise ValueError("P1c p1b_verdict is invalid")
            if not isinstance(self.p1_machinery_ready, bool) or not isinstance(self.all_readouts_valid, bool):
                raise ValueError("P1c P1b readiness fields must be boolean")
            if tuple(item[0] for item in self.qualification_metrics) != _QUALIFICATION_ARMS:
                raise ValueError("P1c qualification metrics must contain the frozen arms")
            for arm, accuracy, pair_flip_rate in self.qualification_metrics:
                for field_name, value in (
                    ("accuracy", accuracy),
                    ("pair_flip_rate", pair_flip_rate),
                ):
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                        raise ValueError(f"P1c {arm}.{field_name} must be in [0, 1]")
        if not self.gate0_passed:
            if self.verdict not in {
                RelationshipP1cVerdict.CANDIDATE_GATE0_REJECTED,
                RelationshipP1cVerdict.VERSION_SCENARIO_DATASET_SATURATED,
            }:
                raise ValueError("P1c Gate 0 failure cannot publish a P1b qualification verdict")
        elif self.verdict is RelationshipP1cVerdict.CANDIDATE_GATE0_REJECTED:
            raise ValueError("P1c Gate 0 rejection verdict requires Gate 0 failure")
        if self.verdict is RelationshipP1cVerdict.FORMAL_PREREG_FREEZE_CANDIDATE and not (
            self.p1b_verdict == RelationshipP1bVerdict.QUALIFIED.value
            and self.p1_machinery_ready
            and self.all_readouts_valid
        ):
            raise ValueError("P1c formal prereg verdict requires a qualified P1b report")
        if (
            self.verdict is RelationshipP1cVerdict.VERSION_SCENARIO_DATASET_SATURATED
            and self.gate0_passed
            and self.p1b_verdict != RelationshipP1bVerdict.DATASET_SATURATED.value
        ):
            raise ValueError("P1c post-Gate-0 saturation verdict requires P1b saturation")
        if self.verdict is RelationshipP1cVerdict.REWRITE_PUBLIC_EVIDENCE_CONTRACT and not (
            self.p1b_verdict == RelationshipP1bVerdict.BASELINE_UNDERQUALIFIED.value
            and self.p1_machinery_ready
            and self.all_readouts_valid
        ):
            raise ValueError("P1c evidence-contract verdict requires a valid underqualified P1b report")
        if self.verdict is RelationshipP1cVerdict.MACHINERY_REGRESSION and (
            self.p1_machinery_ready and self.all_readouts_valid
        ):
            raise ValueError("P1c machinery regression requires a failed machinery precondition")

    @property
    def next_action(self) -> str:
        return _NEXT_ACTIONS[self.verdict]

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "candidate_protocol_id": self.candidate_protocol_id,
            "candidate_model_id": self.candidate_model_id,
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
                arm: {
                    "accuracy": accuracy,
                    "pair_flip_rate": pair_flip_rate,
                }
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
    def from_json(cls, encoded: str) -> "RelationshipP1cReport":
        raw = json.loads(encoded)
        payload = _require_exact_object(
            raw,
            field_name="P1c report",
            expected_fields={
                "all_readouts_valid",
                "artifact_id",
                "baseline_accuracy",
                "baseline_attestation_id",
                "candidate_model_id",
                "candidate_protocol_id",
                "claim_boundary",
                "created_at_iso",
                "gate0_machinery_ready",
                "gate0_passed",
                "gate0_report_artifact_id",
                "generation_config_sha256",
                "next_action",
                "p1_machinery_ready",
                "p1b_report_artifact_id",
                "p1b_verdict",
                "qualification_metrics",
                "schema_version",
                "verdict",
                "weights_sha256",
            },
        )
        metrics_raw = payload["qualification_metrics"]
        if not isinstance(metrics_raw, dict):
            raise ValueError("P1c qualification_metrics must be an object")
        metrics: list[tuple[str, float, float]] = []
        for arm in _QUALIFICATION_ARMS:
            if arm not in metrics_raw:
                continue
            row = _require_exact_object(
                metrics_raw[arm],
                field_name=f"P1c qualification_metrics.{arm}",
                expected_fields={"accuracy", "pair_flip_rate"},
            )
            metrics.append((arm, row["accuracy"], row["pair_flip_rate"]))
        if set(metrics_raw) != {item[0] for item in metrics}:
            raise ValueError("P1c qualification_metrics contains an unknown arm")
        try:
            verdict = RelationshipP1cVerdict(payload["verdict"])
        except (TypeError, ValueError) as exc:
            raise ValueError("P1c verdict is invalid") from exc
        report = cls(
            schema_version=payload["schema_version"],
            created_at_iso=payload["created_at_iso"],
            candidate_protocol_id=payload["candidate_protocol_id"],
            candidate_model_id=payload["candidate_model_id"],
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
        artifact_id = payload["artifact_id"]
        _require_sha256(artifact_id, "artifact_id")
        if (
            payload["claim_boundary"] != _REPORT_CLAIM_BOUNDARY
            or payload["next_action"] != report.next_action
            or artifact_id != report.artifact_id
        ):
            raise ValueError("P1c report derived fields or artifact_id mismatch")
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


def _expected_p1b_verdict(
    report: RelationshipP1bReport,
) -> RelationshipP1bVerdict:
    config = RelationshipP1GateConfig()
    metrics = {arm: dict(values) for arm, values in report.arm_metrics}
    steelman_arms = ("prompt-steelman", "rag-steelman")
    saturated = any(float(metrics[arm]["accuracy"]) > config.maximum_steelman_accuracy for arm in steelman_arms)
    qualified_steelmen = all(
        config.minimum_steelman_accuracy <= float(metrics[arm]["accuracy"]) <= config.maximum_steelman_accuracy
        and float(metrics[arm]["pair_flip_rate"]) >= config.minimum_steelman_pair_flip_rate
        for arm in steelman_arms
    )
    structured_qualified = (
        float(metrics["structured-state"]["pair_flip_rate"]) >= config.minimum_structured_state_pair_flip_rate
    )
    if report.p1_machinery_ready and report.all_readouts_valid and saturated:
        return RelationshipP1bVerdict.DATASET_SATURATED
    if report.p1_machinery_ready and report.all_readouts_valid and qualified_steelmen and structured_qualified:
        return RelationshipP1bVerdict.QUALIFIED
    return RelationshipP1bVerdict.BASELINE_UNDERQUALIFIED


def assess_relationship_packet1c(
    *,
    protocol: RelationshipP1cCandidateProtocol,
    baseline: FrozenBaselineAttestation,
    gate0_report: RelationshipGate0Report,
    p1b_report: RelationshipP1bReport | None,
    created_at_iso: str | None = None,
) -> RelationshipP1cReport:
    validate_relationship_p1c_local_lineage(protocol)
    if (
        baseline.model_id != protocol.model_id
        or baseline.dataset_fingerprint != protocol.dataset_fingerprint
        or baseline.prompt_sha256 != protocol.stateless_prompt_sha256
        or baseline.generation_config_sha256 != protocol.expected_generation_config_sha256
        or baseline.seed_schedule_sha256 != sha256_json(protocol.baseline_seed_schedule)
    ):
        raise ValueError("P1c Gate 0 baseline diverges from the candidate protocol")
    if (
        gate0_report.dataset_fingerprint != protocol.dataset_fingerprint
        or gate0_report.baseline_attestation_id != baseline.artifact_id
        or sha256_json(gate0_report.config.to_payload()) != protocol.gate0_config_sha256
    ):
        raise ValueError("P1c Gate 0 report diverges from the candidate protocol")

    if not gate0_report.gate0_passed:
        if p1b_report is not None:
            raise ValueError("P1c cannot consume P1b when candidate Gate 0 failed")
        baseline_saturated = (
            gate0_report.machinery_ready
            and baseline.valid_decisions == baseline.evaluated_decisions
            and baseline.accuracy > gate0_report.config.maximum_baseline_accuracy
        )
        verdict = (
            RelationshipP1cVerdict.VERSION_SCENARIO_DATASET_SATURATED
            if baseline_saturated
            else RelationshipP1cVerdict.CANDIDATE_GATE0_REJECTED
        )
        return RelationshipP1cReport(
            created_at_iso=created_at_iso or datetime.now(timezone.utc).isoformat(),
            candidate_protocol_id=protocol.protocol_id,
            candidate_model_id=baseline.model_id,
            weights_sha256=baseline.weights_sha256,
            generation_config_sha256=baseline.generation_config_sha256,
            baseline_attestation_id=baseline.artifact_id,
            gate0_report_artifact_id=gate0_report.artifact_id,
            gate0_machinery_ready=gate0_report.machinery_ready,
            gate0_passed=False,
            baseline_accuracy=baseline.accuracy,
            p1b_report_artifact_id=None,
            p1b_verdict=None,
            p1_machinery_ready=None,
            all_readouts_valid=None,
            qualification_metrics=(),
            verdict=verdict,
        )

    if p1b_report is None:
        raise ValueError("P1c candidate Gate 0 passed but P1b report is missing")
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
        raise ValueError("P1c P1b report diverges from the frozen candidate lineage")
    if p1b_report.verdict is not _expected_p1b_verdict(p1b_report):
        raise ValueError("P1c P1b verdict diverges from its frozen qualification metrics")

    if not p1b_report.p1_machinery_ready or not p1b_report.all_readouts_valid:
        verdict = RelationshipP1cVerdict.MACHINERY_REGRESSION
    elif p1b_report.verdict is RelationshipP1bVerdict.DATASET_SATURATED:
        verdict = RelationshipP1cVerdict.VERSION_SCENARIO_DATASET_SATURATED
    elif p1b_report.verdict is RelationshipP1bVerdict.QUALIFIED:
        verdict = RelationshipP1cVerdict.FORMAL_PREREG_FREEZE_CANDIDATE
    else:
        verdict = RelationshipP1cVerdict.REWRITE_PUBLIC_EVIDENCE_CONTRACT

    return RelationshipP1cReport(
        created_at_iso=created_at_iso or datetime.now(timezone.utc).isoformat(),
        candidate_protocol_id=protocol.protocol_id,
        candidate_model_id=baseline.model_id,
        weights_sha256=baseline.weights_sha256,
        generation_config_sha256=baseline.generation_config_sha256,
        baseline_attestation_id=baseline.artifact_id,
        gate0_report_artifact_id=gate0_report.artifact_id,
        gate0_machinery_ready=gate0_report.machinery_ready,
        gate0_passed=gate0_report.gate0_passed,
        baseline_accuracy=baseline.accuracy,
        p1b_report_artifact_id=p1b_report.artifact_id,
        p1b_verdict=p1b_report.verdict.value,
        p1_machinery_ready=p1b_report.p1_machinery_ready,
        all_readouts_valid=p1b_report.all_readouts_valid,
        qualification_metrics=_qualification_metrics(p1b_report),
        verdict=verdict,
    )


def write_relationship_packet1c_report(
    report: RelationshipP1cReport,
    *,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    target = pathlib.Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "packet1c_report.json"
    markdown_path = target / "packet1c_report.md"
    existing = tuple(path for path in (json_path, markdown_path) if path.exists())
    if existing:
        raise FileExistsError(f"P1c report files already exist: {existing}")
    json_path.write_text(report.to_json(), encoding="utf-8")
    lines = [
        "# Relationship Lab P1c qualification fork",
        "",
        f"- artifact_id: `{report.artifact_id}`",
        f"- candidate_protocol_id: `{report.candidate_protocol_id}`",
        f"- candidate_model_id: `{report.candidate_model_id}`",
        f"- weights_sha256: `{report.weights_sha256}`",
        f"- gate0_passed: **{str(report.gate0_passed).lower()}**",
        f"- baseline_accuracy: **{report.baseline_accuracy:.3f}**",
        f"- verdict: **{report.verdict.value}**",
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


def load_relationship_packet1c_report(path: pathlib.Path) -> RelationshipP1cReport:
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return RelationshipP1cReport.from_json(file_path.read_text(encoding="utf-8"))


__all__ = [
    "RELATIONSHIP_P1C_PROTOCOL_SCHEMA_VERSION",
    "RELATIONSHIP_P1C_REPORT_SCHEMA_VERSION",
    "RelationshipP1cCandidateProtocol",
    "RelationshipP1cReport",
    "RelationshipP1cVerdict",
    "assess_relationship_packet1c",
    "load_relationship_p1c_candidate_protocol",
    "load_relationship_packet1c_report",
    "relationship_p1c_candidate_protocol_path",
    "validate_relationship_p1c_local_lineage",
    "write_relationship_packet1c_report",
]
