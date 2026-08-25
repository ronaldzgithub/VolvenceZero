"""Relationship Lab P1i: training-only ordinary-Qwen consumer calibration.

P1i consumes only :class:`RelationshipConsumerTrainingView`.  It evaluates a
pre-registered, bounded set of prompt consumers on the already-observed v3
package, preserves every candidate ledger, selects one consumer with a
leave-one-surface-family-out rule, and freezes that consumer before P1j may
materialize any v4 input or output.  Training labels remain external baseline
calibration data; this module does not write memory, PE, credit, reward,
controller, steering, or runtime state.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, Callable

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME,
    RelationshipAction,
    RelationshipConsumerTrainingView,
    RelationshipDatasetSplit,
    RelationshipTransferDataset,
    canonical_json,
    load_relationship_consumer_training_view,
    sha256_json,
)
from lifeform_evolution.relationship_lab_baseline import StatelessActionCompletion
from lifeform_evolution.relationship_lab_contexts import (
    RelationshipP1Arm,
    RelationshipP1ContextBundle,
)
from lifeform_evolution.relationship_lab_packet1 import (
    RELATIONSHIP_PACKET1_DECISION_SCHEMA_VERSION,
    ContextualRelationshipActionPolicy,
    RelationshipP1Decision,
    relationship_p1_completion_to_decision,
)
from lifeform_evolution.relationship_lab_packet1b import (
    RELATIONSHIP_P1B_COMPILER_VERSION,
    RelationshipEvidenceReadout,
    parse_relationship_evidence_scores,
    relationship_p1b_readout_schema_path,
)
from lifeform_evolution.relationship_lab_packet1g import (
    RelationshipP1gConsumerProtocol,
    load_relationship_p1g_consumer_protocol,
)


RELATIONSHIP_P1I_PROTOCOL_SCHEMA_VERSION = "relationship-p1i-calibration-protocol.v1"
RELATIONSHIP_P1I_CANDIDATE_SCHEMA_VERSION = "relationship-p1i-candidate-artifact.v1"
RELATIONSHIP_P1I_REPORT_SCHEMA_VERSION = "relationship-p1i-calibration-report.v1"
RELATIONSHIP_P1I_CHECKPOINT_SCHEMA_VERSION = "relationship-p1i-checkpoint.v1"
RELATIONSHIP_P1I_CONSUMER_PROTOCOL_SCHEMA_VERSION = (
    "relationship-p1i-frozen-consumer-protocol.v1"
)
RELATIONSHIP_P1I_NEXT_ACTION = "run_one_shot_v4_qualification"

_HEX_DIGITS = frozenset("0123456789abcdef")
_REQUEST_CONTEXT_MARKER = "{{PUBLIC_HISTORY_EVIDENCE}}"
_REQUEST_CURRENT_INPUT_MARKER = "{{CURRENT_USER_MESSAGE}}"
_EVALUATED_ARMS = (
    RelationshipP1Arm.PROMPT_STEELMAN,
    RelationshipP1Arm.RAG_STEELMAN,
    RelationshipP1Arm.STRUCTURED_STATE,
)
_PRIMARY_ARMS = (
    RelationshipP1Arm.PROMPT_STEELMAN,
    RelationshipP1Arm.RAG_STEELMAN,
)
_PROMPT_ASSETS = {
    "conditioned_match_v1": "relationship_lab_conditioned_evidence_readout_v1.txt",
    "latent_partition_v1": "relationship_lab_latent_partition_readout_v1.txt",
    "counterfactual_contrast_v1": (
        "relationship_lab_counterfactual_contrast_readout_v1.txt"
    ),
}
_REQUEST_ASSET = "relationship_lab_conditioned_evidence_readout_request_v1.txt"
_PROTOCOL_CLAIM_BOUNDARY = (
    "P1i pre-registers at most three ordinary-Qwen consumer candidates and "
    "selects one only from the P1h training view. It does not load or run v4 "
    "qualification, modify Volvence memory, PE, credit, controller, or "
    "steering, open formal hidden test or P2, prove Volvence advantage, human "
    "readability, product value, or any complete "
    "Appendable/Readable/Learnable/Steerable claim."
)
_REPORT_CLAIM_BOUNDARY = (
    "P1i reports bounded prompt-consumer calibration on the already-observed "
    "v3 training package and freezes exactly one external baseline consumer. "
    "It is not v4 qualification, Volvence advantage, Readable evidence, "
    "PE/credit learning, steering, formal held-out evidence, product evidence, "
    "or a complete four-capability claim."
)
_CONSUMER_PROTOCOL_CLAIM_BOUNDARY = (
    "This protocol freezes one ordinary-Qwen external baseline selected only "
    "from v3 training data before any P1i process reads or runs v4. P1j may use "
    "it once for development qualification; qualification feedback cannot "
    "revise it. Formal hidden test, P2, Volvence learning and steering remain "
    "closed."
)


def _asset_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent


def relationship_p1i_calibration_protocol_path() -> pathlib.Path:
    return _asset_dir() / "protocols" / "relationship_p1i_training_calibration_v1.json"


def _prompt_path(asset: str) -> pathlib.Path:
    if asset not in set(_PROMPT_ASSETS.values()):
        raise ValueError("P1i prompt asset is not pre-registered")
    return _asset_dir() / "prompts" / asset


def _request_template_path(asset: str) -> pathlib.Path:
    if asset != _REQUEST_ASSET:
        raise ValueError("P1i request template asset is not pre-registered")
    return _asset_dir() / "prompts" / asset


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX_DIGITS for char in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")
    return value


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _require_timestamp(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return text


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    *,
    field_name: str,
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(
            f"{field_name} fields do not match schema; "
            f"missing={missing}, extra={extra}"
        )


def _require_object(
    value: object,
    expected: set[str],
    *,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    _require_exact_keys(value, expected, field_name=field_name)
    return value


def _require_string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field_name} must be a non-empty string array")
    return tuple(value)


def _atomic_write_text(path: pathlib.Path, content: str) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = pathlib.Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class RelationshipP1iCandidateSpec:
    round_index: int
    candidate_id: str
    prompt_asset: str
    prompt_sha256: str
    request_template_asset: str
    request_template_sha256: str
    readout_schema_sha256: str
    compiler_version: str

    def __post_init__(self) -> None:
        if self.round_index < 1:
            raise ValueError("P1i candidate round_index must be positive")
        expected_asset = _PROMPT_ASSETS.get(self.candidate_id)
        if expected_asset is None or self.prompt_asset != expected_asset:
            raise ValueError("P1i candidate id/asset registration mismatch")
        if self.request_template_asset != _REQUEST_ASSET:
            raise ValueError("P1i candidate request template is not frozen")
        for field_name, value in (
            ("prompt_sha256", self.prompt_sha256),
            ("request_template_sha256", self.request_template_sha256),
            ("readout_schema_sha256", self.readout_schema_sha256),
        ):
            _require_sha256(value, f"P1i candidate {field_name}")
        if self.compiler_version != RELATIONSHIP_P1B_COMPILER_VERSION:
            raise ValueError("P1i candidate compiler diverges from P1b")

    @property
    def pipeline_sha256(self) -> str:
        return sha256_json(
            {
                "prompt_sha256": self.prompt_sha256,
                "request_template_sha256": self.request_template_sha256,
                "schema_sha256": self.readout_schema_sha256,
                "compiler_version": self.compiler_version,
            }
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "candidate_id": self.candidate_id,
            "prompt_asset": self.prompt_asset,
            "prompt_sha256": self.prompt_sha256,
            "request_template_asset": self.request_template_asset,
            "request_template_sha256": self.request_template_sha256,
            "readout_schema_sha256": self.readout_schema_sha256,
            "compiler_version": self.compiler_version,
        }


@dataclass(frozen=True)
class RelationshipP1iCalibrationProtocol:
    frozen_at_iso: str
    consumer_split_contract_id: str
    source_p1g_report_artifact_id: str
    source_p1g_consumer_protocol_id: str
    training_package_name: str
    training_dataset_fingerprint: str
    qualification_package_name: str
    qualification_dataset_fingerprint: str
    maximum_revision_rounds: int
    selection_method: str
    preserve_every_candidate: bool
    allowed_feedback: str
    seed_schedule: tuple[int, ...]
    evaluated_arms: tuple[str, ...]
    candidates: tuple[RelationshipP1iCandidateSpec, ...]
    model_source: str
    model_revision: str
    model_id: str
    expected_weights_sha256: str
    expected_generation_config_sha256: str
    device: str
    torch_dtype: str
    temperature: float
    top_p: float
    max_new_tokens: int
    background_depths: tuple[int, ...]
    evaluated_context_surface_sha256: str
    training_context_surface_sha256: str
    background_templates_sha256: str
    rag_embedder: str
    rag_model_source: str
    rag_weights_sha256: str
    rag_top_k: int
    rag_candidate_surface: str
    rag_config_sha256: str
    qualification_inputs_observed_before_freeze: int
    qualification_qwen_outputs_observed_before_freeze: int
    qualification_feedback_to_consumer: bool
    evaluation_feedback_to_pe_credit_reward_or_steering: bool
    formal_hidden_test_opened: bool
    p2_enabled: bool
    claim_boundary: str
    schema_version: str = RELATIONSHIP_P1I_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1I_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("P1i calibration protocol schema_version mismatch")
        _require_timestamp(self.frozen_at_iso, "P1i frozen_at_iso")
        for field_name, value in (
            ("consumer_split_contract_id", self.consumer_split_contract_id),
            ("source_p1g_report_artifact_id", self.source_p1g_report_artifact_id),
            ("source_p1g_consumer_protocol_id", self.source_p1g_consumer_protocol_id),
            ("training_dataset_fingerprint", self.training_dataset_fingerprint),
            (
                "qualification_dataset_fingerprint",
                self.qualification_dataset_fingerprint,
            ),
            ("expected_weights_sha256", self.expected_weights_sha256),
            (
                "expected_generation_config_sha256",
                self.expected_generation_config_sha256,
            ),
            (
                "evaluated_context_surface_sha256",
                self.evaluated_context_surface_sha256,
            ),
            (
                "training_context_surface_sha256",
                self.training_context_surface_sha256,
            ),
            ("background_templates_sha256", self.background_templates_sha256),
            ("rag_weights_sha256", self.rag_weights_sha256),
            ("rag_config_sha256", self.rag_config_sha256),
        ):
            _require_sha256(value, f"P1i {field_name}")
        if self.training_package_name != RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
            raise ValueError("P1i training package must be relationship_transfer_v3")
        if self.qualification_package_name != RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME:
            raise ValueError("P1i qualification package must be relationship_transfer_v4")
        if self.maximum_revision_rounds != 3:
            raise ValueError("P1i calibration budget must remain three rounds")
        if len(self.candidates) != self.maximum_revision_rounds:
            raise ValueError("P1i must pre-register exactly the bounded candidate set")
        if tuple(item.round_index for item in self.candidates) != (1, 2, 3):
            raise ValueError("P1i candidate rounds must be contiguous")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("P1i candidate ids must be unique")
        if self.selection_method != "leave_one_surface_family_out_training_only":
            raise ValueError("P1i selection method diverges from P1h")
        if not self.preserve_every_candidate:
            raise ValueError("P1i must preserve every candidate")
        if self.allowed_feedback != "training_labels_only_external_baseline_calibration":
            raise ValueError("P1i feedback boundary diverges from P1h")
        if not self.seed_schedule or len(set(self.seed_schedule)) != len(
            self.seed_schedule
        ):
            raise ValueError("P1i seed schedule must be non-empty and unique")
        if any(seed < 0 for seed in self.seed_schedule):
            raise ValueError("P1i seeds must be non-negative")
        if self.evaluated_arms != tuple(arm.value for arm in _EVALUATED_ARMS):
            raise ValueError("P1i evaluated arms are not frozen")
        if self.background_depths != (0, 8, 32):
            raise ValueError("P1i background depths diverge from P1g")
        if self.rag_top_k != 4 or self.rag_candidate_surface != (
            "relationship_outcomes_only"
        ):
            raise ValueError("P1i RAG surface diverges from P1g")
        if self.device != "cpu" or self.torch_dtype != "bfloat16":
            raise ValueError("P1i runtime device/dtype diverges from P1g")
        if self.temperature < 0.0 or not 0.0 < self.top_p <= 1.0:
            raise ValueError("P1i generation sampling config is invalid")
        if self.max_new_tokens < 4:
            raise ValueError("P1i max_new_tokens must be at least four")
        if (
            self.qualification_inputs_observed_before_freeze != 0
            or self.qualification_qwen_outputs_observed_before_freeze != 0
        ):
            raise ValueError("P1i must freeze before observing v4 input or output")
        if any(
            (
                self.qualification_feedback_to_consumer,
                self.evaluation_feedback_to_pe_credit_reward_or_steering,
                self.formal_hidden_test_opened,
                self.p2_enabled,
            )
        ):
            raise ValueError("P1i cannot open qualification feedback, formal, or P2")
        if self.claim_boundary != _PROTOCOL_CLAIM_BOUNDARY:
            raise ValueError("P1i calibration protocol claim boundary mismatch")

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_at_iso": self.frozen_at_iso,
            "source_lineage": {
                "consumer_split_contract_id": self.consumer_split_contract_id,
                "source_p1g_report_artifact_id": self.source_p1g_report_artifact_id,
                "source_p1g_consumer_protocol_id": (
                    self.source_p1g_consumer_protocol_id
                ),
                "training_package_name": self.training_package_name,
                "training_dataset_fingerprint": self.training_dataset_fingerprint,
                "qualification_package_name": self.qualification_package_name,
                "qualification_dataset_fingerprint": (
                    self.qualification_dataset_fingerprint
                ),
            },
            "candidate_search": {
                "maximum_revision_rounds": self.maximum_revision_rounds,
                "selection_method": self.selection_method,
                "preserve_every_candidate": self.preserve_every_candidate,
                "allowed_feedback": self.allowed_feedback,
                "seed_schedule": list(self.seed_schedule),
                "evaluated_arms": list(self.evaluated_arms),
                "candidates": [item.to_payload() for item in self.candidates],
            },
            "frozen_runtime": {
                "model_source": self.model_source,
                "model_revision": self.model_revision,
                "model_id": self.model_id,
                "expected_weights_sha256": self.expected_weights_sha256,
                "expected_generation_config_sha256": (
                    self.expected_generation_config_sha256
                ),
                "device": self.device,
                "torch_dtype": self.torch_dtype,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
                "background_depths": list(self.background_depths),
                "evaluated_context_surface_sha256": (
                    self.evaluated_context_surface_sha256
                ),
                "training_context_surface_sha256": (
                    self.training_context_surface_sha256
                ),
                "background_templates_sha256": self.background_templates_sha256,
                "rag_embedder": self.rag_embedder,
                "rag_model_source": self.rag_model_source,
                "rag_weights_sha256": self.rag_weights_sha256,
                "rag_top_k": self.rag_top_k,
                "rag_candidate_surface": self.rag_candidate_surface,
                "rag_config_sha256": self.rag_config_sha256,
            },
            "experiment_guards": {
                "qualification_inputs_observed_before_freeze": (
                    self.qualification_inputs_observed_before_freeze
                ),
                "qualification_qwen_outputs_observed_before_freeze": (
                    self.qualification_qwen_outputs_observed_before_freeze
                ),
                "qualification_feedback_to_consumer": (
                    self.qualification_feedback_to_consumer
                ),
                "evaluation_feedback_to_pe_credit_reward_or_steering": (
                    self.evaluation_feedback_to_pe_credit_reward_or_steering
                ),
                "formal_hidden_test_opened": self.formal_hidden_test_opened,
                "p2_enabled": self.p2_enabled,
            },
            "claim_boundary": self.claim_boundary,
        }

    @property
    def protocol_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["protocol_id"] = self.protocol_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"



    @classmethod
    def from_json(cls, encoded: str) -> "RelationshipP1iCalibrationProtocol":
        raw = json.loads(encoded)
        if not isinstance(raw, dict):
            raise ValueError("P1i calibration protocol must be an object")
        _require_exact_keys(
            raw,
            {
                "schema_version",
                "frozen_at_iso",
                "source_lineage",
                "candidate_search",
                "frozen_runtime",
                "experiment_guards",
                "claim_boundary",
                "protocol_id",
            },
            field_name="P1i calibration protocol",
        )
        source = _require_object(
            raw["source_lineage"],
            {
                "consumer_split_contract_id",
                "source_p1g_report_artifact_id",
                "source_p1g_consumer_protocol_id",
                "training_package_name",
                "training_dataset_fingerprint",
                "qualification_package_name",
                "qualification_dataset_fingerprint",
            },
            field_name="P1i source_lineage",
        )
        search = _require_object(
            raw["candidate_search"],
            {
                "maximum_revision_rounds",
                "selection_method",
                "preserve_every_candidate",
                "allowed_feedback",
                "seed_schedule",
                "evaluated_arms",
                "candidates",
            },
            field_name="P1i candidate_search",
        )
        runtime = _require_object(
            raw["frozen_runtime"],
            {
                "model_source",
                "model_revision",
                "model_id",
                "expected_weights_sha256",
                "expected_generation_config_sha256",
                "device",
                "torch_dtype",
                "temperature",
                "top_p",
                "max_new_tokens",
                "background_depths",
                "evaluated_context_surface_sha256",
                "training_context_surface_sha256",
                "background_templates_sha256",
                "rag_embedder",
                "rag_model_source",
                "rag_weights_sha256",
                "rag_top_k",
                "rag_candidate_surface",
                "rag_config_sha256",
            },
            field_name="P1i frozen_runtime",
        )
        guards = _require_object(
            raw["experiment_guards"],
            {
                "qualification_inputs_observed_before_freeze",
                "qualification_qwen_outputs_observed_before_freeze",
                "qualification_feedback_to_consumer",
                "evaluation_feedback_to_pe_credit_reward_or_steering",
                "formal_hidden_test_opened",
                "p2_enabled",
            },
            field_name="P1i experiment_guards",
        )
        candidates_raw = search["candidates"]
        if not isinstance(candidates_raw, list) or not candidates_raw:
            raise ValueError("P1i candidates must be a non-empty array")
        candidates: list[RelationshipP1iCandidateSpec] = []
        candidate_fields = {
            "round_index",
            "candidate_id",
            "prompt_asset",
            "prompt_sha256",
            "request_template_asset",
            "request_template_sha256",
            "readout_schema_sha256",
            "compiler_version",
        }
        for index, item in enumerate(candidates_raw):
            parsed = _require_object(
                item,
                candidate_fields,
                field_name=f"P1i candidates[{index}]",
            )
            candidates.append(
                RelationshipP1iCandidateSpec(
                    round_index=_require_int(
                        parsed["round_index"],
                        f"P1i candidates[{index}].round_index",
                    ),
                    candidate_id=_require_text(
                        parsed["candidate_id"],
                        f"P1i candidates[{index}].candidate_id",
                    ),
                    prompt_asset=_require_text(
                        parsed["prompt_asset"],
                        f"P1i candidates[{index}].prompt_asset",
                    ),
                    prompt_sha256=_require_sha256(
                        parsed["prompt_sha256"],
                        f"P1i candidates[{index}].prompt_sha256",
                    ),
                    request_template_asset=_require_text(
                        parsed["request_template_asset"],
                        f"P1i candidates[{index}].request_template_asset",
                    ),
                    request_template_sha256=_require_sha256(
                        parsed["request_template_sha256"],
                        f"P1i candidates[{index}].request_template_sha256",
                    ),
                    readout_schema_sha256=_require_sha256(
                        parsed["readout_schema_sha256"],
                        f"P1i candidates[{index}].readout_schema_sha256",
                    ),
                    compiler_version=_require_text(
                        parsed["compiler_version"],
                        f"P1i candidates[{index}].compiler_version",
                    ),
                )
            )
        seed_schedule_raw = search["seed_schedule"]
        background_depths_raw = runtime["background_depths"]
        if not isinstance(seed_schedule_raw, list) or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in seed_schedule_raw
        ):
            raise ValueError("P1i seed_schedule must be an integer array")
        if not isinstance(background_depths_raw, list) or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in background_depths_raw
        ):
            raise ValueError("P1i background_depths must be an integer array")
        boolean_fields = (
            ("preserve_every_candidate", search["preserve_every_candidate"]),
            (
                "qualification_feedback_to_consumer",
                guards["qualification_feedback_to_consumer"],
            ),
            (
                "evaluation_feedback_to_pe_credit_reward_or_steering",
                guards["evaluation_feedback_to_pe_credit_reward_or_steering"],
            ),
            ("formal_hidden_test_opened", guards["formal_hidden_test_opened"]),
            ("p2_enabled", guards["p2_enabled"]),
        )
        for field_name, value in boolean_fields:
            if not isinstance(value, bool):
                raise ValueError(f"P1i {field_name} must be boolean")
        protocol_id = _require_sha256(raw["protocol_id"], "P1i protocol_id")
        protocol = cls(
            schema_version=_require_text(raw["schema_version"], "P1i schema_version"),
            frozen_at_iso=_require_timestamp(
                raw["frozen_at_iso"], "P1i frozen_at_iso"
            ),
            consumer_split_contract_id=_require_sha256(
                source["consumer_split_contract_id"],
                "P1i consumer split contract",
            ),
            source_p1g_report_artifact_id=_require_sha256(
                source["source_p1g_report_artifact_id"],
                "P1i source P1g report",
            ),
            source_p1g_consumer_protocol_id=_require_sha256(
                source["source_p1g_consumer_protocol_id"],
                "P1i source P1g protocol",
            ),
            training_package_name=_require_text(
                source["training_package_name"], "P1i training package"
            ),
            training_dataset_fingerprint=_require_sha256(
                source["training_dataset_fingerprint"],
                "P1i training fingerprint",
            ),
            qualification_package_name=_require_text(
                source["qualification_package_name"],
                "P1i qualification package",
            ),
            qualification_dataset_fingerprint=_require_sha256(
                source["qualification_dataset_fingerprint"],
                "P1i qualification fingerprint",
            ),
            maximum_revision_rounds=_require_int(
                search["maximum_revision_rounds"], "P1i revision budget"
            ),
            selection_method=_require_text(
                search["selection_method"], "P1i selection method"
            ),
            preserve_every_candidate=search["preserve_every_candidate"],
            allowed_feedback=_require_text(
                search["allowed_feedback"], "P1i allowed feedback"
            ),
            seed_schedule=tuple(seed_schedule_raw),
            evaluated_arms=_require_string_list(
                search["evaluated_arms"], "P1i evaluated_arms"
            ),
            candidates=tuple(candidates),
            model_source=_require_text(runtime["model_source"], "P1i model_source"),
            model_revision=_require_text(
                runtime["model_revision"], "P1i model_revision"
            ),
            model_id=_require_text(runtime["model_id"], "P1i model_id"),
            expected_weights_sha256=_require_sha256(
                runtime["expected_weights_sha256"], "P1i model weights"
            ),
            expected_generation_config_sha256=_require_sha256(
                runtime["expected_generation_config_sha256"],
                "P1i generation config",
            ),
            device=_require_text(runtime["device"], "P1i device"),
            torch_dtype=_require_text(runtime["torch_dtype"], "P1i torch_dtype"),
            temperature=_require_number(runtime["temperature"], "P1i temperature"),
            top_p=_require_number(runtime["top_p"], "P1i top_p"),
            max_new_tokens=_require_int(
                runtime["max_new_tokens"], "P1i max_new_tokens"
            ),
            background_depths=tuple(background_depths_raw),
            evaluated_context_surface_sha256=_require_sha256(
                runtime["evaluated_context_surface_sha256"],
                "P1i P1g context surface",
            ),
            training_context_surface_sha256=_require_sha256(
                runtime["training_context_surface_sha256"],
                "P1i training context surface",
            ),
            background_templates_sha256=_require_sha256(
                runtime["background_templates_sha256"],
                "P1i background templates",
            ),
            rag_embedder=_require_text(runtime["rag_embedder"], "P1i rag_embedder"),
            rag_model_source=_require_text(
                runtime["rag_model_source"], "P1i rag_model_source"
            ),
            rag_weights_sha256=_require_sha256(
                runtime["rag_weights_sha256"], "P1i RAG weights"
            ),
            rag_top_k=_require_int(runtime["rag_top_k"], "P1i RAG top_k"),
            rag_candidate_surface=_require_text(
                runtime["rag_candidate_surface"], "P1i RAG candidate surface"
            ),
            rag_config_sha256=_require_sha256(
                runtime["rag_config_sha256"], "P1i RAG config"
            ),
            qualification_inputs_observed_before_freeze=_require_int(
                guards["qualification_inputs_observed_before_freeze"],
                "P1i v4 input count",
            ),
            qualification_qwen_outputs_observed_before_freeze=_require_int(
                guards["qualification_qwen_outputs_observed_before_freeze"],
                "P1i v4 output count",
            ),
            qualification_feedback_to_consumer=guards[
                "qualification_feedback_to_consumer"
            ],
            evaluation_feedback_to_pe_credit_reward_or_steering=guards[
                "evaluation_feedback_to_pe_credit_reward_or_steering"
            ],
            formal_hidden_test_opened=guards["formal_hidden_test_opened"],
            p2_enabled=guards["p2_enabled"],
            claim_boundary=_require_text(
                raw["claim_boundary"], "P1i claim boundary"
            ),
        )
        if protocol.protocol_id != protocol_id:
            raise ValueError("P1i calibration protocol_id mismatch")
        return protocol


def load_relationship_p1i_calibration_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipP1iCalibrationProtocol:
    file_path = pathlib.Path(path or relationship_p1i_calibration_protocol_path())
    if not file_path.is_file():
        raise FileNotFoundError(f"P1i calibration protocol is missing: {file_path}")
    return RelationshipP1iCalibrationProtocol.from_json(
        file_path.read_text(encoding="utf-8")
    )


def relationship_p1i_training_context_surface_sha256(
    *,
    bundle: RelationshipP1ContextBundle,
) -> str:
    rows = tuple(
        sorted(
            (
                context.scene_id,
                context.background_depth,
                context.arm.value,
                context.context_sha256,
            )
            for context in bundle.contexts
        )
    )
    if not rows:
        raise ValueError("P1i training context surface must be non-empty")
    return sha256_json(
        {
            "dataset_fingerprint": bundle.dataset_fingerprint,
            "background_depths": list(bundle.background_depths),
            "background_templates_sha256": bundle.background_templates_sha256,
            "rag_config_sha256": bundle.rag_config_sha256,
            "dataset_role": "consumer_training_only",
            "contexts": rows,
        }
    )


def _validate_training_fold_shape(dataset: RelationshipTransferDataset) -> None:
    families: list[str] = []
    for _pair_id, members in dataset.mirrored_pairs():
        pair_families = {observation.probe_surface_family for observation, _ in members}
        if len(members) != 2 or len(pair_families) != 1:
            raise ValueError("P1i LOSO fold requires one two-user surface-family pair")
        families.append(next(iter(pair_families)))
    if len(families) < 3 or len(set(families)) != len(families):
        raise ValueError("P1i LOSO surface families must be unique across pairs")


def validate_relationship_p1i_local_lineage(
    protocol: RelationshipP1iCalibrationProtocol,
    *,
    training_view: RelationshipConsumerTrainingView | None = None,
    source_p1g_protocol: RelationshipP1gConsumerProtocol | None = None,
) -> RelationshipConsumerTrainingView:
    view = training_view or load_relationship_consumer_training_view()
    p1g = source_p1g_protocol or load_relationship_p1g_consumer_protocol()
    contract = view.contract
    mismatches = {
        "consumer_split_contract_id": (
            protocol.consumer_split_contract_id,
            contract.contract_sha256,
        ),
        "source_p1g_report_artifact_id": (
            protocol.source_p1g_report_artifact_id,
            contract.source_p1g_report_artifact_id,
        ),
        "source_p1g_consumer_protocol_id": (
            protocol.source_p1g_consumer_protocol_id,
            p1g.protocol_id,
        ),
        "training_package_name": (
            protocol.training_package_name,
            contract.training_package_name,
        ),
        "training_dataset_fingerprint": (
            protocol.training_dataset_fingerprint,
            contract.training_dataset_fingerprint,
        ),
        "qualification_package_name": (
            protocol.qualification_package_name,
            contract.qualification_package_name,
        ),
        "qualification_dataset_fingerprint": (
            protocol.qualification_dataset_fingerprint,
            contract.qualification_dataset_fingerprint,
        ),
        "maximum_revision_rounds": (
            protocol.maximum_revision_rounds,
            contract.maximum_consumer_revision_rounds,
        ),
        "selection_method": (protocol.selection_method, contract.selection_method),
        "preserve_every_candidate": (
            protocol.preserve_every_candidate,
            contract.preserve_every_candidate,
        ),
        "allowed_feedback": (protocol.allowed_feedback, contract.allowed_feedback),
        "model_source": (protocol.model_source, p1g.model_source),
        "model_revision": (protocol.model_revision, p1g.model_revision),
        "model_id": (protocol.model_id, p1g.model_id),
        "expected_weights_sha256": (
            protocol.expected_weights_sha256,
            p1g.expected_weights_sha256,
        ),
        "expected_generation_config_sha256": (
            protocol.expected_generation_config_sha256,
            p1g.expected_generation_config_sha256,
        ),
        "device": (protocol.device, p1g.device),
        "torch_dtype": (protocol.torch_dtype, p1g.torch_dtype),
        "temperature": (protocol.temperature, p1g.temperature),
        "top_p": (protocol.top_p, p1g.top_p),
        "max_new_tokens": (protocol.max_new_tokens, p1g.max_new_tokens),
        "background_depths": (protocol.background_depths, p1g.background_depths),
        "evaluated_context_surface_sha256": (
            protocol.evaluated_context_surface_sha256,
            p1g.evaluated_context_surface_sha256,
        ),
        "background_templates_sha256": (
            protocol.background_templates_sha256,
            p1g.background_templates_sha256,
        ),
        "rag_embedder": (protocol.rag_embedder, p1g.rag_embedder),
        "rag_model_source": (protocol.rag_model_source, p1g.rag_model_source),
        "rag_weights_sha256": (
            protocol.rag_weights_sha256,
            p1g.rag_weights_sha256,
        ),
        "rag_top_k": (protocol.rag_top_k, p1g.rag_top_k),
        "rag_candidate_surface": (
            protocol.rag_candidate_surface,
            p1g.rag_candidate_surface,
        ),
        "rag_config_sha256": (protocol.rag_config_sha256, p1g.rag_config_sha256),
    }
    drift = sorted(name for name, values in mismatches.items() if values[0] != values[1])
    if drift:
        raise ValueError(f"P1i local lineage mismatch: {drift}")
    if view.training_dataset.dataset_fingerprint != protocol.training_dataset_fingerprint:
        raise ValueError("P1i training view dataset fingerprint mismatch")
    if "qualification_dataset" in {field.name for field in fields(view)}:
        raise ValueError("P1i training view must not expose qualification data")
    _validate_training_fold_shape(view.training_dataset)
    expected_schema_hash = _sha256_file(relationship_p1b_readout_schema_path())
    for candidate in protocol.candidates:
        if _sha256_file(_prompt_path(candidate.prompt_asset)) != candidate.prompt_sha256:
            raise ValueError(f"P1i prompt asset drifted: {candidate.candidate_id}")
        if (
            _sha256_file(_request_template_path(candidate.request_template_asset))
            != candidate.request_template_sha256
        ):
            raise ValueError("P1i request template asset drifted")
        if candidate.readout_schema_sha256 != expected_schema_hash:
            raise ValueError("P1i readout schema asset drifted")
    return view


def validate_relationship_p1i_context_lineage(
    protocol: RelationshipP1iCalibrationProtocol,
    *,
    bundle: RelationshipP1ContextBundle,
) -> None:
    if bundle.dataset_fingerprint != protocol.training_dataset_fingerprint:
        raise ValueError("P1i context dataset fingerprint mismatch")
    if bundle.background_depths != protocol.background_depths:
        raise ValueError("P1i context background depths mismatch")
    if bundle.background_templates_sha256 != protocol.background_templates_sha256:
        raise ValueError("P1i context background templates mismatch")
    if bundle.rag_config_sha256 != protocol.rag_config_sha256:
        raise ValueError("P1i context RAG config mismatch")
    actual_surface = relationship_p1i_training_context_surface_sha256(bundle=bundle)
    if actual_surface != protocol.training_context_surface_sha256:
        raise ValueError(
            "P1i complete training context surface mismatch: "
            f"expected={protocol.training_context_surface_sha256}, "
            f"actual={actual_surface}"
        )


def _render_request(
    *,
    candidate: RelationshipP1iCandidateSpec,
    context_text: str,
    current_input: str,
) -> str:
    if not context_text.strip() or not current_input.strip():
        raise ValueError("P1i request requires context and current input")
    template = _request_template_path(candidate.request_template_asset).read_text(
        encoding="utf-8"
    )
    if (
        template.count(_REQUEST_CONTEXT_MARKER) != 1
        or template.count(_REQUEST_CURRENT_INPUT_MARKER) != 1
    ):
        raise ValueError("P1i request markers must each occur exactly once")
    return (
        template.replace(_REQUEST_CONTEXT_MARKER, context_text)
        .replace(_REQUEST_CURRENT_INPUT_MARKER, current_input)
        .strip()
    )


def load_relationship_p1i_candidate_prompt(
    candidate: RelationshipP1iCandidateSpec,
) -> str:
    """Load the exact frozen prompt asset owned by a P1i consumer."""

    path = _prompt_path(candidate.prompt_asset)
    if _sha256_file(path) != candidate.prompt_sha256:
        raise ValueError("P1i selected prompt asset drifted")
    return path.read_text(encoding="utf-8").strip()


def render_relationship_p1i_candidate_request(
    *,
    candidate: RelationshipP1iCandidateSpec,
    context_text: str,
    current_input: str,
) -> str:
    """Render the exact frozen P1i request without exposing evaluator truth."""

    path = _request_template_path(candidate.request_template_asset)
    if _sha256_file(path) != candidate.request_template_sha256:
        raise ValueError("P1i selected request template asset drifted")
    return _render_request(
        candidate=candidate,
        context_text=context_text,
        current_input=current_input,
    )


def _readout_completion(readout: RelationshipEvidenceReadout) -> StatelessActionCompletion:
    action = readout.compiled_action
    return StatelessActionCompletion(
        raw_output=(
            readout.raw_output
            if action is None
            else canonical_json({"action_id": action.value})
        ),
        chosen_action_id=action,
        prompt_tokens=readout.prompt_tokens,
        completion_tokens=readout.completion_tokens,
    )


def relationship_p1i_readout_completion(
    readout: RelationshipEvidenceReadout,
) -> StatelessActionCompletion:
    """Reconstruct compiler input from a frozen evidence readout."""

    return _readout_completion(readout)


def _candidate_spec_from_payload(payload: object) -> RelationshipP1iCandidateSpec:
    raw = _require_object(
        payload,
        {
            "round_index",
            "candidate_id",
            "prompt_asset",
            "prompt_sha256",
            "request_template_asset",
            "request_template_sha256",
            "readout_schema_sha256",
            "compiler_version",
        },
        field_name="P1i checkpoint candidate",
    )
    return RelationshipP1iCandidateSpec(
        round_index=_require_int(raw["round_index"], "P1i candidate round"),
        candidate_id=_require_text(raw["candidate_id"], "P1i candidate id"),
        prompt_asset=_require_text(raw["prompt_asset"], "P1i candidate prompt asset"),
        prompt_sha256=_require_sha256(
            raw["prompt_sha256"], "P1i candidate prompt hash"
        ),
        request_template_asset=_require_text(
            raw["request_template_asset"], "P1i candidate request asset"
        ),
        request_template_sha256=_require_sha256(
            raw["request_template_sha256"], "P1i candidate request hash"
        ),
        readout_schema_sha256=_require_sha256(
            raw["readout_schema_sha256"], "P1i candidate readout schema"
        ),
        compiler_version=_require_text(
            raw["compiler_version"], "P1i candidate compiler"
        ),
    )


def _planned_record_keys(
    *,
    protocol: RelationshipP1iCalibrationProtocol,
    training_view: RelationshipConsumerTrainingView,
) -> tuple[tuple[RelationshipP1Arm, str, int], ...]:
    keys: list[tuple[RelationshipP1Arm, str, int]] = []
    for _mirror_pair_id, members in training_view.training_dataset.mirrored_pairs():
        for seed in protocol.seed_schedule:
            for arm in _EVALUATED_ARMS:
                keys.extend((arm, observation.scene_id, seed) for observation, _ in members)
    return tuple(keys)


@dataclass(frozen=True)
class RelationshipP1iCandidateCheckpoint:
    calibration_protocol_id: str
    candidate: RelationshipP1iCandidateSpec
    dataset_fingerprint: str
    training_context_surface_sha256: str
    model_id: str
    weights_sha256: str
    generation_config_sha256: str
    seed_schedule: tuple[int, ...]
    planned_record_keys: tuple[tuple[RelationshipP1Arm, str, int], ...]
    schema_version: str = RELATIONSHIP_P1I_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1I_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("P1i checkpoint schema_version mismatch")
        for field_name, value in (
            ("calibration_protocol_id", self.calibration_protocol_id),
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("training_context_surface_sha256", self.training_context_surface_sha256),
            ("weights_sha256", self.weights_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
        ):
            _require_sha256(value, f"P1i checkpoint {field_name}")
        if not self.model_id.strip():
            raise ValueError("P1i checkpoint model_id must be non-empty")
        if not self.seed_schedule or len(set(self.seed_schedule)) != len(
            self.seed_schedule
        ):
            raise ValueError("P1i checkpoint seed schedule is invalid")
        if not self.planned_record_keys or len(set(self.planned_record_keys)) != len(
            self.planned_record_keys
        ):
            raise ValueError("P1i checkpoint record plan must be unique and non-empty")
        if set(seed for _arm, _scene, seed in self.planned_record_keys) != set(
            self.seed_schedule
        ):
            raise ValueError("P1i checkpoint record seeds diverge from schedule")
        if set(arm for arm, _scene, _seed in self.planned_record_keys) != set(
            _EVALUATED_ARMS
        ):
            raise ValueError("P1i checkpoint record arms diverge from protocol")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "calibration_protocol_id": self.calibration_protocol_id,
            "candidate": self.candidate.to_payload(),
            "dataset_fingerprint": self.dataset_fingerprint,
            "training_context_surface_sha256": self.training_context_surface_sha256,
            "model_lineage": {
                "model_id": self.model_id,
                "weights_sha256": self.weights_sha256,
                "generation_config_sha256": self.generation_config_sha256,
            },
            "seed_schedule": list(self.seed_schedule),
            "planned_records": [
                {
                    "index": index,
                    "arm": arm.value,
                    "scene_id": scene_id,
                    "seed": seed,
                }
                for index, (arm, scene_id, seed) in enumerate(
                    self.planned_record_keys
                )
            ],
            "qualification_inputs_observed": 0,
            "qualification_qwen_outputs_observed": 0,
        }

    @property
    def checkpoint_id(self) -> str:
        return sha256_json(self._identity_payload())

    def to_payload(self) -> dict[str, object]:
        return {**self._identity_payload(), "checkpoint_id": self.checkpoint_id}

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"

    @classmethod
    def from_json(cls, raw_json: str) -> RelationshipP1iCandidateCheckpoint:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError("P1i checkpoint is not valid JSON") from exc
        raw = _require_object(
            payload,
            {
                "schema_version",
                "calibration_protocol_id",
                "candidate",
                "dataset_fingerprint",
                "training_context_surface_sha256",
                "model_lineage",
                "seed_schedule",
                "planned_records",
                "qualification_inputs_observed",
                "qualification_qwen_outputs_observed",
                "checkpoint_id",
            },
            field_name="P1i checkpoint",
        )
        model = _require_object(
            raw["model_lineage"],
            {"model_id", "weights_sha256", "generation_config_sha256"},
            field_name="P1i checkpoint model lineage",
        )
        seed_schedule = raw["seed_schedule"]
        if not isinstance(seed_schedule, list) or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in seed_schedule
        ):
            raise ValueError("P1i checkpoint seed_schedule must be an integer array")
        records = raw["planned_records"]
        if not isinstance(records, list) or not records:
            raise ValueError("P1i checkpoint planned_records must be non-empty")
        keys: list[tuple[RelationshipP1Arm, str, int]] = []
        for expected_index, item in enumerate(records):
            record = _require_object(
                item,
                {"index", "arm", "scene_id", "seed"},
                field_name=f"P1i checkpoint planned_records[{expected_index}]",
            )
            index = _require_int(record["index"], "P1i checkpoint record index")
            if index != expected_index:
                raise ValueError("P1i checkpoint record indices must be contiguous")
            keys.append(
                (
                    RelationshipP1Arm(
                        _require_text(record["arm"], "P1i checkpoint record arm")
                    ),
                    _require_text(record["scene_id"], "P1i checkpoint scene id"),
                    _require_int(record["seed"], "P1i checkpoint record seed"),
                )
            )
        if (
            _require_int(
                raw["qualification_inputs_observed"], "P1i checkpoint v4 inputs"
            )
            != 0
            or _require_int(
                raw["qualification_qwen_outputs_observed"],
                "P1i checkpoint v4 outputs",
            )
            != 0
        ):
            raise ValueError("P1i checkpoint must not observe v4")
        checkpoint_id = _require_sha256(
            raw["checkpoint_id"], "P1i checkpoint id"
        )
        checkpoint = cls(
            schema_version=_require_text(raw["schema_version"], "P1i checkpoint schema"),
            calibration_protocol_id=_require_sha256(
                raw["calibration_protocol_id"], "P1i checkpoint protocol"
            ),
            candidate=_candidate_spec_from_payload(raw["candidate"]),
            dataset_fingerprint=_require_sha256(
                raw["dataset_fingerprint"], "P1i checkpoint dataset"
            ),
            training_context_surface_sha256=_require_sha256(
                raw["training_context_surface_sha256"], "P1i checkpoint context"
            ),
            model_id=_require_text(model["model_id"], "P1i checkpoint model id"),
            weights_sha256=_require_sha256(
                model["weights_sha256"], "P1i checkpoint model weights"
            ),
            generation_config_sha256=_require_sha256(
                model["generation_config_sha256"],
                "P1i checkpoint generation config",
            ),
            seed_schedule=tuple(seed_schedule),
            planned_record_keys=tuple(keys),
        )
        if checkpoint.checkpoint_id != checkpoint_id:
            raise ValueError("P1i checkpoint id mismatch")
        return checkpoint


def build_relationship_p1i_candidate_checkpoint(
    policy: ContextualRelationshipActionPolicy,
    *,
    protocol: RelationshipP1iCalibrationProtocol,
    candidate: RelationshipP1iCandidateSpec,
    training_view: RelationshipConsumerTrainingView,
    contexts: RelationshipP1ContextBundle,
) -> RelationshipP1iCandidateCheckpoint:
    validate_relationship_p1i_local_lineage(protocol, training_view=training_view)
    validate_relationship_p1i_context_lineage(protocol, bundle=contexts)
    if candidate not in protocol.candidates:
        raise ValueError("P1i checkpoint candidate is outside the frozen protocol")
    if (
        policy.model_id != protocol.model_id
        or policy.weights_sha256 != protocol.expected_weights_sha256
        or policy.generation_config_sha256
        != protocol.expected_generation_config_sha256
    ):
        raise ValueError("P1i checkpoint policy diverges from frozen runtime")
    return RelationshipP1iCandidateCheckpoint(
        calibration_protocol_id=protocol.protocol_id,
        candidate=candidate,
        dataset_fingerprint=training_view.training_dataset.dataset_fingerprint,
        training_context_surface_sha256=protocol.training_context_surface_sha256,
        model_id=policy.model_id,
        weights_sha256=policy.weights_sha256,
        generation_config_sha256=policy.generation_config_sha256,
        seed_schedule=protocol.seed_schedule,
        planned_record_keys=_planned_record_keys(
            protocol=protocol,
            training_view=training_view,
        ),
    )


@dataclass(frozen=True)
class RelationshipP1iCandidateRun:
    calibration_protocol_id: str
    candidate: RelationshipP1iCandidateSpec
    dataset_fingerprint: str
    training_context_surface_sha256: str
    model_id: str
    weights_sha256: str
    generation_config_sha256: str
    seed_schedule: tuple[int, ...]
    scene_surface_families: tuple[tuple[str, str], ...]
    readouts: tuple[RelationshipEvidenceReadout, ...]
    decisions: tuple[RelationshipP1Decision, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("calibration_protocol_id", self.calibration_protocol_id),
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("training_context_surface_sha256", self.training_context_surface_sha256),
            ("weights_sha256", self.weights_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
        ):
            _require_sha256(value, f"P1i run {field_name}")
        if not self.model_id.strip():
            raise ValueError("P1i run model_id must be non-empty")
        if not self.seed_schedule or len(set(self.seed_schedule)) != len(
            self.seed_schedule
        ):
            raise ValueError("P1i run seed schedule is invalid")
        surface_map = dict(self.scene_surface_families)
        if len(surface_map) != len(self.scene_surface_families) or not surface_map:
            raise ValueError("P1i run scene/surface map must be unique and non-empty")
        readout_keys = tuple(
            (item.arm, item.scene_id, item.seed) for item in self.readouts
        )
        decision_keys = tuple(
            (item.arm, item.scene_id, item.seed) for item in self.decisions
        )
        if (
            not readout_keys
            or len(set(readout_keys)) != len(readout_keys)
            or set(readout_keys) != set(decision_keys)
            or len(set(decision_keys)) != len(decision_keys)
        ):
            raise ValueError("P1i run readout/decision coverage is invalid")
        expected_keys = {
            (arm, scene_id, seed)
            for arm in _EVALUATED_ARMS
            for scene_id in surface_map
            for seed in self.seed_schedule
        }
        if set(readout_keys) != expected_keys:
            raise ValueError("P1i candidate does not cover every training scene and arm")
        decisions = {
            (item.arm, item.scene_id, item.seed): item for item in self.decisions
        }
        for readout in self.readouts:
            if (
                readout.prompt_sha256 != self.candidate.prompt_sha256
                or readout.request_template_sha256
                != self.candidate.request_template_sha256
                or readout.schema_sha256 != self.candidate.readout_schema_sha256
                or readout.model_id != self.model_id
                or readout.weights_sha256 != self.weights_sha256
                or readout.generation_config_sha256
                != self.generation_config_sha256
            ):
                raise ValueError("P1i candidate readout lineage mismatch")
            decision = decisions[(readout.arm, readout.scene_id, readout.seed)]
            if (
                decision.context_sha256 != readout.context_sha256
                or decision.current_input_sha256 != readout.current_input_sha256
                or decision.arm_prompt_sha256 != self.candidate.pipeline_sha256
                or decision.chosen_action_id is not readout.compiled_action
            ):
                raise ValueError("P1i readout-to-decision projection mismatch")

    def readout_ledger_jsonl(self) -> str:
        return "".join(
            canonical_json(
                {
                    "candidate_id": self.candidate.candidate_id,
                    **item.to_payload(),
                    "artifact_id": item.artifact_id,
                }
            )
            + "\n"
            for item in self.readouts
        )

    def decision_ledger_jsonl(self) -> str:
        return "".join(
            canonical_json(
                {
                    "candidate_id": self.candidate.candidate_id,
                    **item.to_payload(),
                }
            )
            + "\n"
            for item in self.decisions
        )

    @property
    def readout_ledger_sha256(self) -> str:
        return hashlib.sha256(self.readout_ledger_jsonl().encode("utf-8")).hexdigest()

    @property
    def decision_ledger_sha256(self) -> str:
        return hashlib.sha256(self.decision_ledger_jsonl().encode("utf-8")).hexdigest()


def _readout_record_payload(
    *,
    candidate_id: str,
    readout: RelationshipEvidenceReadout,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        **readout.to_payload(),
        "artifact_id": readout.artifact_id,
    }


def _decision_record_payload(
    *,
    candidate_id: str,
    decision: RelationshipP1Decision,
) -> dict[str, object]:
    return {"candidate_id": candidate_id, **decision.to_payload()}


def _require_raw_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _readout_from_record_payload(
    payload: object,
    *,
    expected_candidate_id: str,
) -> RelationshipEvidenceReadout:
    raw = _require_object(
        payload,
        {
            "candidate_id",
            "schema_version",
            "arm",
            "scene_id",
            "seed",
            "current_input_sha256",
            "context_sha256",
            "model_id",
            "weights_sha256",
            "generation_config_sha256",
            "prompt_sha256",
            "request_template_sha256",
            "schema_sha256",
            "raw_output",
            "stay_score",
            "space_score",
            "valid",
            "compiled_action_id",
            "prompt_tokens",
            "completion_tokens",
            "artifact_id",
        },
        field_name="P1i checkpoint readout",
    )
    if _require_text(raw["candidate_id"], "P1i readout candidate") != (
        expected_candidate_id
    ):
        raise ValueError("P1i checkpoint readout candidate mismatch")
    stay_score = raw["stay_score"]
    space_score = raw["space_score"]
    for field_name, score in (
        ("stay_score", stay_score),
        ("space_score", space_score),
    ):
        if score is not None and (isinstance(score, bool) or not isinstance(score, int)):
            raise ValueError(f"P1i readout {field_name} must be integer or null")
    readout = RelationshipEvidenceReadout(
        schema_version=_require_text(raw["schema_version"], "P1i readout schema"),
        arm=RelationshipP1Arm(_require_text(raw["arm"], "P1i readout arm")),
        scene_id=_require_text(raw["scene_id"], "P1i readout scene"),
        seed=_require_int(raw["seed"], "P1i readout seed"),
        current_input_sha256=_require_sha256(
            raw["current_input_sha256"], "P1i readout current input"
        ),
        context_sha256=_require_sha256(
            raw["context_sha256"], "P1i readout context"
        ),
        model_id=_require_text(raw["model_id"], "P1i readout model id"),
        weights_sha256=_require_sha256(
            raw["weights_sha256"], "P1i readout model weights"
        ),
        generation_config_sha256=_require_sha256(
            raw["generation_config_sha256"], "P1i readout generation config"
        ),
        prompt_sha256=_require_sha256(
            raw["prompt_sha256"], "P1i readout prompt"
        ),
        request_template_sha256=_require_sha256(
            raw["request_template_sha256"], "P1i readout request template"
        ),
        schema_sha256=_require_sha256(
            raw["schema_sha256"], "P1i readout schema asset"
        ),
        raw_output=_require_raw_string(raw["raw_output"], "P1i readout output"),
        stay_score=stay_score,
        space_score=space_score,
        prompt_tokens=_require_int(raw["prompt_tokens"], "P1i readout prompt tokens"),
        completion_tokens=_require_int(
            raw["completion_tokens"], "P1i readout completion tokens"
        ),
    )
    compiled_raw = raw["compiled_action_id"]
    compiled = (
        None
        if compiled_raw is None
        else RelationshipAction(
            _require_text(compiled_raw, "P1i readout compiled action")
        )
    )
    if (
        _require_bool(raw["valid"], "P1i readout valid") is not readout.valid
        or compiled is not readout.compiled_action
        or _require_sha256(raw["artifact_id"], "P1i readout artifact id")
        != readout.artifact_id
    ):
        raise ValueError("P1i checkpoint readout derived fields mismatch")
    return readout


def _decision_from_record_payload(
    payload: object,
    *,
    expected_candidate_id: str,
) -> RelationshipP1Decision:
    raw = _require_object(
        payload,
        {
            "candidate_id",
            "schema_version",
            "decision_id",
            "arm",
            "scene_id",
            "mirror_pair_id",
            "split",
            "seed",
            "current_input_sha256",
            "context_sha256",
            "arm_prompt_sha256",
            "raw_output",
            "chosen_action_id",
            "expected_action_id",
            "valid",
            "correct",
            "prompt_tokens",
            "completion_tokens",
        },
        field_name="P1i checkpoint decision",
    )
    if _require_text(raw["candidate_id"], "P1i decision candidate") != (
        expected_candidate_id
    ):
        raise ValueError("P1i checkpoint decision candidate mismatch")
    chosen_raw = raw["chosen_action_id"]
    chosen = (
        None
        if chosen_raw is None
        else RelationshipAction(
            _require_text(chosen_raw, "P1i decision chosen action")
        )
    )
    expected = RelationshipAction(
        _require_text(raw["expected_action_id"], "P1i decision expected action")
    )
    decision = RelationshipP1Decision(
        schema_version=_require_text(raw["schema_version"], "P1i decision schema"),
        decision_id=_require_sha256(raw["decision_id"], "P1i decision id"),
        arm=RelationshipP1Arm(_require_text(raw["arm"], "P1i decision arm")),
        scene_id=_require_text(raw["scene_id"], "P1i decision scene"),
        mirror_pair_id=_require_text(
            raw["mirror_pair_id"], "P1i decision mirror pair"
        ),
        split=RelationshipDatasetSplit(
            _require_text(raw["split"], "P1i decision split")
        ),
        seed=_require_int(raw["seed"], "P1i decision seed"),
        current_input_sha256=_require_sha256(
            raw["current_input_sha256"], "P1i decision current input"
        ),
        context_sha256=_require_sha256(
            raw["context_sha256"], "P1i decision context"
        ),
        arm_prompt_sha256=_require_sha256(
            raw["arm_prompt_sha256"], "P1i decision pipeline"
        ),
        raw_output=_require_raw_string(raw["raw_output"], "P1i decision output"),
        chosen_action_id=chosen,
        expected_action_id=expected,
        valid=_require_bool(raw["valid"], "P1i decision valid"),
        correct=_require_bool(raw["correct"], "P1i decision correct"),
        prompt_tokens=_require_int(
            raw["prompt_tokens"], "P1i decision prompt tokens"
        ),
        completion_tokens=_require_int(
            raw["completion_tokens"], "P1i decision completion tokens"
        ),
    )
    if decision.schema_version != RELATIONSHIP_PACKET1_DECISION_SCHEMA_VERSION:
        raise ValueError("P1i checkpoint decision schema mismatch")
    if decision.valid is not (chosen is not None) or decision.correct is not (
        chosen is expected
    ):
        raise ValueError("P1i checkpoint decision derived fields mismatch")
    if decision.prompt_tokens < 0 or decision.completion_tokens < 0:
        raise ValueError("P1i checkpoint decision token counts must be non-negative")
    return decision


def relationship_p1i_readout_record_payload(
    *,
    candidate_id: str,
    readout: RelationshipEvidenceReadout,
) -> dict[str, object]:
    """Serialize one immutable P1i-compatible evidence readout record."""

    return _readout_record_payload(candidate_id=candidate_id, readout=readout)


def relationship_p1i_decision_record_payload(
    *,
    candidate_id: str,
    decision: RelationshipP1Decision,
) -> dict[str, object]:
    """Serialize one immutable P1i-compatible evaluator decision record."""

    return _decision_record_payload(candidate_id=candidate_id, decision=decision)


def relationship_p1i_readout_from_record_payload(
    payload: object,
    *,
    expected_candidate_id: str,
) -> RelationshipEvidenceReadout:
    """Strictly load one P1i-compatible evidence readout record."""

    return _readout_from_record_payload(
        payload,
        expected_candidate_id=expected_candidate_id,
    )


def relationship_p1i_decision_from_record_payload(
    payload: object,
    *,
    expected_candidate_id: str,
) -> RelationshipP1Decision:
    """Strictly load one P1i-compatible evaluator decision record."""

    return _decision_from_record_payload(
        payload,
        expected_candidate_id=expected_candidate_id,
    )


@dataclass(frozen=True)
class RelationshipP1iCandidateProgress:
    checkpoint: RelationshipP1iCandidateCheckpoint
    readouts: tuple[RelationshipEvidenceReadout, ...]
    decisions: tuple[RelationshipP1Decision, ...]

    def __post_init__(self) -> None:
        if len(self.decisions) > len(self.readouts) or (
            len(self.readouts) - len(self.decisions) > 1
        ):
            raise ValueError("P1i checkpoint readout/decision counts are invalid")
        readout_keys = tuple(
            (item.arm, item.scene_id, item.seed) for item in self.readouts
        )
        decision_keys = tuple(
            (item.arm, item.scene_id, item.seed) for item in self.decisions
        )
        if readout_keys != self.checkpoint.planned_record_keys[: len(readout_keys)]:
            raise ValueError("P1i checkpoint readouts are not the planned prefix")
        if decision_keys != self.checkpoint.planned_record_keys[: len(decision_keys)]:
            raise ValueError("P1i checkpoint decisions are not the planned prefix")
        for index, readout in enumerate(self.readouts):
            if (
                readout.model_id != self.checkpoint.model_id
                or readout.weights_sha256 != self.checkpoint.weights_sha256
                or readout.generation_config_sha256
                != self.checkpoint.generation_config_sha256
                or readout.prompt_sha256
                != self.checkpoint.candidate.prompt_sha256
                or readout.request_template_sha256
                != self.checkpoint.candidate.request_template_sha256
                or readout.schema_sha256
                != self.checkpoint.candidate.readout_schema_sha256
            ):
                raise ValueError("P1i checkpoint readout lineage mismatch")
            if index >= len(self.decisions):
                continue
            decision = self.decisions[index]
            regenerated = relationship_p1_completion_to_decision(
                completion=_readout_completion(readout),
                arm=decision.arm,
                scene_id=decision.scene_id,
                mirror_pair_id=decision.mirror_pair_id,
                split=decision.split,
                seed=decision.seed,
                current_input_sha256=decision.current_input_sha256,
                context_sha256=decision.context_sha256,
                arm_prompt_sha256=decision.arm_prompt_sha256,
                expected_action_id=decision.expected_action_id,
                model_id=readout.model_id,
            )
            if regenerated != decision:
                raise ValueError("P1i checkpoint decision is not a readout projection")

    @property
    def is_complete(self) -> bool:
        return len(self.decisions) == len(self.checkpoint.planned_record_keys)


def _checkpoint_path(candidate_dir: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(candidate_dir) / "checkpoint.json"


def _record_path(
    candidate_dir: pathlib.Path,
    *,
    index: int,
    kind: str,
) -> pathlib.Path:
    if kind not in {"readout", "decision"}:
        raise ValueError("P1i checkpoint record kind is invalid")
    return pathlib.Path(candidate_dir) / "records" / f"{index:04d}.{kind}.json"


def write_relationship_p1i_candidate_checkpoint(
    *,
    checkpoint: RelationshipP1iCandidateCheckpoint,
    candidate_dir: pathlib.Path,
) -> pathlib.Path:
    target = pathlib.Path(candidate_dir)
    target.mkdir(parents=True, exist_ok=False)
    (target / "records").mkdir()
    path = _checkpoint_path(target)
    _atomic_write_text(path, checkpoint.to_json())
    return path


def load_relationship_p1i_candidate_checkpoint(
    candidate_dir: pathlib.Path,
) -> RelationshipP1iCandidateCheckpoint:
    path = _checkpoint_path(candidate_dir)
    if not path.is_file():
        raise FileNotFoundError(f"P1i candidate checkpoint is missing: {path}")
    return RelationshipP1iCandidateCheckpoint.from_json(path.read_text(encoding="utf-8"))


def _load_json_record(path: pathlib.Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"P1i checkpoint record is not valid JSON: {path}") from exc


def persist_relationship_p1i_readout(
    *,
    checkpoint: RelationshipP1iCandidateCheckpoint,
    candidate_dir: pathlib.Path,
    index: int,
    readout: RelationshipEvidenceReadout,
) -> pathlib.Path:
    if index < 0 or index >= len(checkpoint.planned_record_keys):
        raise IndexError("P1i readout checkpoint index is outside the plan")
    if (readout.arm, readout.scene_id, readout.seed) != (
        checkpoint.planned_record_keys[index]
    ):
        raise ValueError("P1i readout does not match its checkpoint slot")
    payload = _readout_record_payload(
        candidate_id=checkpoint.candidate.candidate_id,
        readout=readout,
    )
    path = _record_path(candidate_dir, index=index, kind="readout")
    if path.exists():
        existing = _readout_from_record_payload(
            _load_json_record(path),
            expected_candidate_id=checkpoint.candidate.candidate_id,
        )
        if existing != readout:
            raise ValueError("P1i resumed readout diverges from durable checkpoint")
        return path
    _atomic_write_text(path, canonical_json(payload) + "\n")
    return path


def persist_relationship_p1i_decision(
    *,
    checkpoint: RelationshipP1iCandidateCheckpoint,
    candidate_dir: pathlib.Path,
    index: int,
    decision: RelationshipP1Decision,
) -> pathlib.Path:
    if index < 0 or index >= len(checkpoint.planned_record_keys):
        raise IndexError("P1i decision checkpoint index is outside the plan")
    if (decision.arm, decision.scene_id, decision.seed) != (
        checkpoint.planned_record_keys[index]
    ):
        raise ValueError("P1i decision does not match its checkpoint slot")
    readout_path = _record_path(candidate_dir, index=index, kind="readout")
    if not readout_path.is_file():
        raise FileNotFoundError("P1i decision cannot precede its durable readout")
    payload = _decision_record_payload(
        candidate_id=checkpoint.candidate.candidate_id,
        decision=decision,
    )
    path = _record_path(candidate_dir, index=index, kind="decision")
    if path.exists():
        existing = _decision_from_record_payload(
            _load_json_record(path),
            expected_candidate_id=checkpoint.candidate.candidate_id,
        )
        if existing != decision:
            raise ValueError("P1i resumed decision diverges from durable checkpoint")
        return path
    _atomic_write_text(path, canonical_json(payload) + "\n")
    return path


def load_relationship_p1i_candidate_progress(
    candidate_dir: pathlib.Path,
) -> RelationshipP1iCandidateProgress:
    target = pathlib.Path(candidate_dir)
    checkpoint = load_relationship_p1i_candidate_checkpoint(target)
    records_dir = target / "records"
    if not records_dir.is_dir():
        raise FileNotFoundError(f"P1i checkpoint records directory is missing: {records_dir}")
    allowed_names = {
        f"{index:04d}.{kind}.json"
        for index in range(len(checkpoint.planned_record_keys))
        for kind in ("readout", "decision")
    }
    unexpected = sorted(
        path.name
        for path in records_dir.iterdir()
        if path.name not in allowed_names
        and not (path.name.startswith(".") and path.name.endswith(".tmp"))
    )
    if unexpected:
        raise ValueError(f"P1i checkpoint has unexpected records: {unexpected}")
    readouts: list[RelationshipEvidenceReadout] = []
    decisions: list[RelationshipP1Decision] = []
    gap_seen = False
    for index in range(len(checkpoint.planned_record_keys)):
        readout_path = _record_path(target, index=index, kind="readout")
        decision_path = _record_path(target, index=index, kind="decision")
        if not readout_path.exists():
            if decision_path.exists():
                raise ValueError("P1i checkpoint decision exists without readout")
            gap_seen = True
            continue
        if gap_seen:
            raise ValueError("P1i checkpoint readout records contain a gap")
        readouts.append(
            _readout_from_record_payload(
                _load_json_record(readout_path),
                expected_candidate_id=checkpoint.candidate.candidate_id,
            )
        )
        if not decision_path.exists():
            gap_seen = True
            continue
        decisions.append(
            _decision_from_record_payload(
                _load_json_record(decision_path),
                expected_candidate_id=checkpoint.candidate.candidate_id,
            )
        )
    return RelationshipP1iCandidateProgress(
        checkpoint=checkpoint,
        readouts=tuple(readouts),
        decisions=tuple(decisions),
    )


def validate_relationship_p1i_candidate_progress(
    progress: RelationshipP1iCandidateProgress,
    *,
    protocol: RelationshipP1iCalibrationProtocol,
    candidate: RelationshipP1iCandidateSpec,
    training_view: RelationshipConsumerTrainingView,
    contexts: RelationshipP1ContextBundle,
) -> None:
    expected_checkpoint = RelationshipP1iCandidateCheckpoint(
        calibration_protocol_id=protocol.protocol_id,
        candidate=candidate,
        dataset_fingerprint=training_view.training_dataset.dataset_fingerprint,
        training_context_surface_sha256=protocol.training_context_surface_sha256,
        model_id=protocol.model_id,
        weights_sha256=protocol.expected_weights_sha256,
        generation_config_sha256=protocol.expected_generation_config_sha256,
        seed_schedule=protocol.seed_schedule,
        planned_record_keys=_planned_record_keys(
            protocol=protocol,
            training_view=training_view,
        ),
    )
    if progress.checkpoint != expected_checkpoint:
        raise ValueError("P1i resumed checkpoint diverges from frozen lineage")
    dataset = training_view.training_dataset
    pair_by_scene = {
        observation.scene_id: mirror_pair_id
        for mirror_pair_id, members in dataset.mirrored_pairs()
        for observation, _dynamic in members
    }
    observation_by_scene = {
        observation.scene_id: observation for observation in dataset.observations
    }
    for index, readout in enumerate(progress.readouts):
        observation = observation_by_scene[readout.scene_id]
        expected_input_hash = hashlib.sha256(
            observation.current_input.encode("utf-8")
        ).hexdigest()
        context = contexts.context(scene_id=readout.scene_id, arm=readout.arm)
        if (
            readout.current_input_sha256 != expected_input_hash
            or readout.context_sha256 != context.context_sha256
        ):
            raise ValueError("P1i resumed readout input/context lineage mismatch")
        if index >= len(progress.decisions):
            continue
        dynamic = dataset.dynamic_for_scene(readout.scene_id)
        expected_decision = relationship_p1_completion_to_decision(
            completion=_readout_completion(readout),
            arm=readout.arm,
            scene_id=readout.scene_id,
            mirror_pair_id=pair_by_scene[readout.scene_id],
            split=dynamic.split,
            seed=readout.seed,
            current_input_sha256=expected_input_hash,
            context_sha256=context.context_sha256,
            arm_prompt_sha256=candidate.pipeline_sha256,
            expected_action_id=dynamic.preferred_action,
            model_id=readout.model_id,
        )
        if progress.decisions[index] != expected_decision:
            raise ValueError("P1i resumed decision diverges from training truth")


def relationship_p1i_run_from_progress(
    progress: RelationshipP1iCandidateProgress,
    *,
    training_view: RelationshipConsumerTrainingView,
) -> RelationshipP1iCandidateRun:
    if not progress.is_complete:
        raise ValueError("P1i cannot finalize an incomplete candidate checkpoint")
    dataset = training_view.training_dataset
    return RelationshipP1iCandidateRun(
        calibration_protocol_id=progress.checkpoint.calibration_protocol_id,
        candidate=progress.checkpoint.candidate,
        dataset_fingerprint=progress.checkpoint.dataset_fingerprint,
        training_context_surface_sha256=(
            progress.checkpoint.training_context_surface_sha256
        ),
        model_id=progress.checkpoint.model_id,
        weights_sha256=progress.checkpoint.weights_sha256,
        generation_config_sha256=progress.checkpoint.generation_config_sha256,
        seed_schedule=progress.checkpoint.seed_schedule,
        scene_surface_families=tuple(
            sorted(
                (observation.scene_id, observation.probe_surface_family)
                for observation in dataset.observations
            )
        ),
        readouts=progress.readouts,
        decisions=progress.decisions,
    )


def run_relationship_p1i_candidate(
    policy: ContextualRelationshipActionPolicy,
    *,
    protocol: RelationshipP1iCalibrationProtocol,
    candidate: RelationshipP1iCandidateSpec,
    training_view: RelationshipConsumerTrainingView,
    contexts: RelationshipP1ContextBundle,
    readout_observer: Callable[[RelationshipEvidenceReadout], None] | None = None,
    decision_observer: Callable[[RelationshipP1Decision], None] | None = None,
) -> RelationshipP1iCandidateRun:
    validate_relationship_p1i_local_lineage(protocol, training_view=training_view)
    validate_relationship_p1i_context_lineage(protocol, bundle=contexts)
    if candidate not in protocol.candidates:
        raise ValueError("P1i candidate is not part of the frozen protocol")
    if (
        policy.model_id != protocol.model_id
        or policy.weights_sha256 != protocol.expected_weights_sha256
        or policy.generation_config_sha256
        != protocol.expected_generation_config_sha256
    ):
        raise ValueError("P1i policy diverges from frozen substrate lineage")
    prompt = _prompt_path(candidate.prompt_asset).read_text(encoding="utf-8").strip()
    dataset = training_view.training_dataset
    readouts: list[RelationshipEvidenceReadout] = []
    decisions: list[RelationshipP1Decision] = []
    scene_surface_families = tuple(
        sorted(
            (observation.scene_id, observation.probe_surface_family)
            for observation in dataset.observations
        )
    )
    for mirror_pair_id, members in dataset.mirrored_pairs():
        split = members[0][1].split
        for seed in protocol.seed_schedule:
            for arm in _EVALUATED_ARMS:
                for observation, dynamic in members:
                    context = contexts.context(scene_id=observation.scene_id, arm=arm)
                    current_input_sha256 = hashlib.sha256(
                        observation.current_input.encode("utf-8")
                    ).hexdigest()
                    completion = policy.choose_from_messages(
                        messages=(
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": _render_request(
                                    candidate=candidate,
                                    context_text=context.context_text,
                                    current_input=observation.current_input,
                                ),
                            },
                        ),
                        seed=seed,
                    )
                    stay_score, space_score = parse_relationship_evidence_scores(
                        completion.raw_output
                    )
                    readout = RelationshipEvidenceReadout(
                        arm=arm,
                        scene_id=observation.scene_id,
                        seed=seed,
                        current_input_sha256=current_input_sha256,
                        context_sha256=context.context_sha256,
                        model_id=policy.model_id,
                        weights_sha256=policy.weights_sha256,
                        generation_config_sha256=policy.generation_config_sha256,
                        prompt_sha256=candidate.prompt_sha256,
                        request_template_sha256=candidate.request_template_sha256,
                        schema_sha256=candidate.readout_schema_sha256,
                        raw_output=completion.raw_output,
                        stay_score=stay_score,
                        space_score=space_score,
                        prompt_tokens=completion.prompt_tokens,
                        completion_tokens=completion.completion_tokens,
                    )
                    readouts.append(readout)
                    if readout_observer is not None:
                        readout_observer(readout)
                    decision = relationship_p1_completion_to_decision(
                        completion=_readout_completion(readout),
                        arm=arm,
                        scene_id=observation.scene_id,
                        mirror_pair_id=mirror_pair_id,
                        split=split,
                        seed=seed,
                        current_input_sha256=current_input_sha256,
                        context_sha256=context.context_sha256,
                        arm_prompt_sha256=candidate.pipeline_sha256,
                        expected_action_id=dynamic.preferred_action,
                        model_id=policy.model_id,
                    )
                    decisions.append(decision)
                    if decision_observer is not None:
                        decision_observer(decision)
    return RelationshipP1iCandidateRun(
        calibration_protocol_id=protocol.protocol_id,
        candidate=candidate,
        dataset_fingerprint=dataset.dataset_fingerprint,
        training_context_surface_sha256=protocol.training_context_surface_sha256,
        model_id=policy.model_id,
        weights_sha256=policy.weights_sha256,
        generation_config_sha256=policy.generation_config_sha256,
        seed_schedule=protocol.seed_schedule,
        scene_surface_families=scene_surface_families,
        readouts=tuple(readouts),
        decisions=tuple(decisions),
    )


def _decision_metrics(
    decisions: tuple[RelationshipP1Decision, ...],
    readouts: tuple[RelationshipEvidenceReadout, ...],
) -> tuple[tuple[str, object], ...]:
    if not decisions or len(decisions) != len(readouts):
        raise ValueError("P1i metrics require matched non-empty decisions/readouts")
    valid = sum(int(item.valid) for item in decisions)
    correct = sum(int(item.correct) for item in decisions)
    grouped: dict[tuple[str, int], list[RelationshipP1Decision]] = {}
    for item in decisions:
        grouped.setdefault((item.mirror_pair_id, item.seed), []).append(item)
    valid_groups = 0
    flip_groups = 0
    for group in grouped.values():
        if len(group) != 2:
            raise ValueError("P1i mirrored-pair metric group must contain two decisions")
        if all(item.chosen_action_id is not None for item in group):
            valid_groups += 1
            flip_groups += int(
                {item.chosen_action_id for item in group}
                == {
                    RelationshipAction.STAY_PRESENT_WITHOUT_PROBE,
                    RelationshipAction.RESPECT_SPACE_WITH_RETURN_OPTION,
                }
            )
    valid_readouts = sum(int(item.valid) for item in readouts)
    payload: dict[str, object] = {
        "decisions": len(decisions),
        "valid_decisions": valid,
        "valid_rate": valid / len(decisions),
        "correct_decisions": correct,
        "accuracy": correct / len(decisions),
        "pair_groups": len(grouped),
        "valid_pair_groups": valid_groups,
        "pair_flip_rate": flip_groups / valid_groups if valid_groups else 0.0,
        "readouts": len(readouts),
        "valid_readouts": valid_readouts,
        "prompt_tokens_total": sum(item.prompt_tokens for item in readouts),
        "completion_tokens_total": sum(item.completion_tokens for item in readouts),
    }
    return tuple(sorted(payload.items()))


def _validate_candidate_metrics(
    metrics_items: tuple[tuple[str, object], ...],
    *,
    expected_decisions: int,
    expected_pair_groups: int,
    field_name: str,
) -> None:
    metrics = dict(metrics_items)
    expected_fields = {
        "accuracy",
        "completion_tokens_total",
        "correct_decisions",
        "decisions",
        "pair_flip_rate",
        "pair_groups",
        "prompt_tokens_total",
        "readouts",
        "valid_decisions",
        "valid_pair_groups",
        "valid_rate",
        "valid_readouts",
    }
    if len(metrics) != len(metrics_items) or set(metrics) != expected_fields:
        raise ValueError(f"{field_name} fields do not match the frozen metric schema")
    integer_fields = (
        "completion_tokens_total",
        "correct_decisions",
        "decisions",
        "pair_groups",
        "prompt_tokens_total",
        "readouts",
        "valid_decisions",
        "valid_pair_groups",
        "valid_readouts",
    )
    for name in integer_fields:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name}.{name} must be a non-negative integer")
    for name in ("accuracy", "pair_flip_rate", "valid_rate"):
        value = metrics[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{field_name}.{name} must be in [0, 1]")
    if (
        metrics["decisions"] != expected_decisions
        or metrics["readouts"] != expected_decisions
        or metrics["pair_groups"] != expected_pair_groups
    ):
        raise ValueError(f"{field_name} coverage diverges from the frozen folds")
    if not (
        0
        <= metrics["correct_decisions"]
        <= metrics["valid_decisions"]
        == metrics["valid_readouts"]
        <= expected_decisions
    ):
        raise ValueError(f"{field_name} validity/correctness counts are inconsistent")
    if not 0 <= metrics["valid_pair_groups"] <= expected_pair_groups:
        raise ValueError(f"{field_name} valid pair counts are inconsistent")
    if metrics["accuracy"] != metrics["correct_decisions"] / expected_decisions:
        raise ValueError(f"{field_name} accuracy diverges from counts")
    if metrics["valid_rate"] != metrics["valid_decisions"] / expected_decisions:
        raise ValueError(f"{field_name} valid_rate diverges from counts")
    implied_flips = float(metrics["pair_flip_rate"]) * int(
        metrics["valid_pair_groups"]
    )
    if abs(implied_flips - round(implied_flips)) > 1e-12:
        raise ValueError(f"{field_name} pair_flip_rate cannot come from its counts")


@dataclass(frozen=True)
class RelationshipP1iCandidateArtifact:
    candidate: RelationshipP1iCandidateSpec
    calibration_protocol_id: str
    dataset_fingerprint: str
    training_context_surface_sha256: str
    model_id: str
    weights_sha256: str
    generation_config_sha256: str
    seed_schedule: tuple[int, ...]
    readout_ledger_sha256: str
    decision_ledger_sha256: str
    arm_metrics: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]
    fold_metrics: tuple[
        tuple[str, tuple[tuple[str, tuple[tuple[str, object], ...]], ...]], ...
    ]
    schema_version: str = RELATIONSHIP_P1I_CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1I_CANDIDATE_SCHEMA_VERSION:
            raise ValueError("P1i candidate artifact schema_version mismatch")
        for field_name, value in (
            ("calibration_protocol_id", self.calibration_protocol_id),
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("training_context_surface_sha256", self.training_context_surface_sha256),
            ("weights_sha256", self.weights_sha256),
            ("generation_config_sha256", self.generation_config_sha256),
            ("readout_ledger_sha256", self.readout_ledger_sha256),
            ("decision_ledger_sha256", self.decision_ledger_sha256),
        ):
            _require_sha256(value, f"P1i candidate artifact {field_name}")
        if not self.seed_schedule or len(set(self.seed_schedule)) != len(
            self.seed_schedule
        ):
            raise ValueError("P1i candidate artifact seed schedule is invalid")
        if tuple(arm for arm, _ in self.arm_metrics) != tuple(
            arm.value for arm in _EVALUATED_ARMS
        ):
            raise ValueError("P1i candidate artifact arm metrics are incomplete")
        fold_names = tuple(name for name, _ in self.fold_metrics)
        if len(fold_names) != 6 or fold_names != tuple(sorted(set(fold_names))):
            raise ValueError("P1i requires six sorted unique surface-family folds")
        expected_arm_decisions = len(fold_names) * 2 * len(self.seed_schedule)
        expected_arm_pairs = len(fold_names) * len(self.seed_schedule)
        for arm, metrics in self.arm_metrics:
            _validate_candidate_metrics(
                metrics,
                expected_decisions=expected_arm_decisions,
                expected_pair_groups=expected_arm_pairs,
                field_name=f"P1i arm_metrics.{arm}",
            )
        for fold, arm_metrics in self.fold_metrics:
            if tuple(arm for arm, _ in arm_metrics) != tuple(
                arm.value for arm in _EVALUATED_ARMS
            ):
                raise ValueError("P1i fold arm metrics are incomplete")
            for arm, metrics in arm_metrics:
                _validate_candidate_metrics(
                    metrics,
                    expected_decisions=2 * len(self.seed_schedule),
                    expected_pair_groups=len(self.seed_schedule),
                    field_name=f"P1i fold_metrics.{fold}.{arm}",
                )
        for arm, aggregate_items in self.arm_metrics:
            aggregate = dict(aggregate_items)
            folds = [dict(dict(values)[arm]) for _fold, values in self.fold_metrics]
            for field_name in (
                "completion_tokens_total",
                "correct_decisions",
                "decisions",
                "pair_groups",
                "prompt_tokens_total",
                "readouts",
                "valid_decisions",
                "valid_pair_groups",
                "valid_readouts",
            ):
                if aggregate[field_name] != sum(
                    int(metrics[field_name]) for metrics in folds
                ):
                    raise ValueError(
                        f"P1i {arm}.{field_name} diverges from its LOSO folds"
                    )
            aggregate_flips = float(aggregate["pair_flip_rate"]) * int(
                aggregate["valid_pair_groups"]
            )
            fold_flips = sum(
                float(metrics["pair_flip_rate"])
                * int(metrics["valid_pair_groups"])
                for metrics in folds
            )
            if aggregate_flips != fold_flips:
                raise ValueError(f"P1i {arm} pair flips diverge from its LOSO folds")

    def metrics_for_arm(self, arm: RelationshipP1Arm) -> dict[str, object]:
        return dict(dict(self.arm_metrics)[arm.value])

    @property
    def selection_metrics(self) -> tuple[tuple[str, object], ...]:
        primary_global = [self.metrics_for_arm(arm) for arm in _PRIMARY_ARMS]
        primary_fold_metrics = [
            dict(arm_metrics)[arm.value]
            for _family, arm_metrics in self.fold_metrics
            for arm in _PRIMARY_ARMS
        ]
        selection = {
            "all_readouts_valid": all(
                float(metrics["valid_rate"]) == 1.0
                for _arm, values in self.arm_metrics
                for metrics in (dict(values),)
            ),
            "worst_primary_fold_accuracy": min(
                float(dict(metrics)["accuracy"]) for metrics in primary_fold_metrics
            ),
            "minimum_primary_macro_accuracy": min(
                float(metrics["accuracy"]) for metrics in primary_global
            ),
            "worst_primary_fold_pair_flip_rate": min(
                float(dict(metrics)["pair_flip_rate"])
                for metrics in primary_fold_metrics
            ),
            "minimum_primary_macro_pair_flip_rate": min(
                float(metrics["pair_flip_rate"]) for metrics in primary_global
            ),
            "structured_macro_pair_flip_rate": float(
                self.metrics_for_arm(RelationshipP1Arm.STRUCTURED_STATE)[
                    "pair_flip_rate"
                ]
            ),
            "total_prompt_tokens": sum(
                int(dict(metrics)["prompt_tokens_total"])
                for _arm, metrics in self.arm_metrics
            ),
        }
        return tuple(sorted(selection.items()))

    @property
    def selection_key(self) -> tuple[float, ...]:
        metrics = dict(self.selection_metrics)
        return (
            float(bool(metrics["all_readouts_valid"])),
            float(metrics["worst_primary_fold_accuracy"]),
            float(metrics["minimum_primary_macro_accuracy"]),
            float(metrics["worst_primary_fold_pair_flip_rate"]),
            float(metrics["minimum_primary_macro_pair_flip_rate"]),
            float(metrics["structured_macro_pair_flip_rate"]),
            -float(metrics["total_prompt_tokens"]),
            -float(self.candidate.round_index),
        )

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate": self.candidate.to_payload(),
            "pipeline_sha256": self.candidate.pipeline_sha256,
            "calibration_protocol_id": self.calibration_protocol_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "training_context_surface_sha256": self.training_context_surface_sha256,
            "model_id": self.model_id,
            "weights_sha256": self.weights_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "seed_schedule": list(self.seed_schedule),
            "readout_ledger_sha256": self.readout_ledger_sha256,
            "decision_ledger_sha256": self.decision_ledger_sha256,
            "arm_metrics": {
                arm: dict(metrics) for arm, metrics in self.arm_metrics
            },
            "fold_metrics": {
                family: {
                    arm: dict(metrics) for arm, metrics in arm_metrics
                }
                for family, arm_metrics in self.fold_metrics
            },
            "selection_metrics": dict(self.selection_metrics),
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_payload(self) -> dict[str, object]:
        return {**self._canonical_payload(), "artifact_id": self.artifact_id}


def summarize_relationship_p1i_candidate(
    run: RelationshipP1iCandidateRun,
) -> RelationshipP1iCandidateArtifact:
    surface_by_scene = dict(run.scene_surface_families)
    readouts_by_key = {
        (item.arm, item.scene_id, item.seed): item for item in run.readouts
    }
    arm_metrics: list[tuple[str, tuple[tuple[str, object], ...]]] = []
    for arm in _EVALUATED_ARMS:
        decisions = tuple(item for item in run.decisions if item.arm is arm)
        readouts = tuple(
            readouts_by_key[(item.arm, item.scene_id, item.seed)]
            for item in decisions
        )
        arm_metrics.append((arm.value, _decision_metrics(decisions, readouts)))
    fold_metrics: list[
        tuple[str, tuple[tuple[str, tuple[tuple[str, object], ...]], ...]]
    ] = []
    for family in sorted(set(surface_by_scene.values())):
        fold_arms: list[tuple[str, tuple[tuple[str, object], ...]]] = []
        for arm in _EVALUATED_ARMS:
            decisions = tuple(
                item
                for item in run.decisions
                if item.arm is arm and surface_by_scene[item.scene_id] == family
            )
            readouts = tuple(
                readouts_by_key[(item.arm, item.scene_id, item.seed)]
                for item in decisions
            )
            fold_arms.append((arm.value, _decision_metrics(decisions, readouts)))
        fold_metrics.append((family, tuple(fold_arms)))
    return RelationshipP1iCandidateArtifact(
        candidate=run.candidate,
        calibration_protocol_id=run.calibration_protocol_id,
        dataset_fingerprint=run.dataset_fingerprint,
        training_context_surface_sha256=run.training_context_surface_sha256,
        model_id=run.model_id,
        weights_sha256=run.weights_sha256,
        generation_config_sha256=run.generation_config_sha256,
        seed_schedule=run.seed_schedule,
        readout_ledger_sha256=run.readout_ledger_sha256,
        decision_ledger_sha256=run.decision_ledger_sha256,
        arm_metrics=tuple(arm_metrics),
        fold_metrics=tuple(fold_metrics),
    )


def _parse_metrics(value: object, field_name: str) -> tuple[tuple[str, object], ...]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    expected = {
        "accuracy",
        "completion_tokens_total",
        "correct_decisions",
        "decisions",
        "pair_flip_rate",
        "pair_groups",
        "prompt_tokens_total",
        "readouts",
        "valid_decisions",
        "valid_pair_groups",
        "valid_rate",
        "valid_readouts",
    }
    _require_exact_keys(value, expected, field_name=field_name)
    return tuple(sorted(value.items()))


def _candidate_artifact_from_payload(
    raw: object,
) -> RelationshipP1iCandidateArtifact:
    payload = _require_object(
        raw,
        {
            "schema_version",
            "candidate",
            "pipeline_sha256",
            "calibration_protocol_id",
            "dataset_fingerprint",
            "training_context_surface_sha256",
            "model_id",
            "weights_sha256",
            "generation_config_sha256",
            "seed_schedule",
            "readout_ledger_sha256",
            "decision_ledger_sha256",
            "arm_metrics",
            "fold_metrics",
            "selection_metrics",
            "artifact_id",
        },
        field_name="P1i candidate artifact",
    )
    candidate_raw = _require_object(
        payload["candidate"],
        {
            "round_index",
            "candidate_id",
            "prompt_asset",
            "prompt_sha256",
            "request_template_asset",
            "request_template_sha256",
            "readout_schema_sha256",
            "compiler_version",
        },
        field_name="P1i candidate spec",
    )
    candidate = RelationshipP1iCandidateSpec(
        round_index=_require_int(candidate_raw["round_index"], "P1i round_index"),
        candidate_id=_require_text(candidate_raw["candidate_id"], "P1i candidate_id"),
        prompt_asset=_require_text(candidate_raw["prompt_asset"], "P1i prompt_asset"),
        prompt_sha256=_require_sha256(
            candidate_raw["prompt_sha256"], "P1i prompt_sha256"
        ),
        request_template_asset=_require_text(
            candidate_raw["request_template_asset"], "P1i request_template_asset"
        ),
        request_template_sha256=_require_sha256(
            candidate_raw["request_template_sha256"],
            "P1i request_template_sha256",
        ),
        readout_schema_sha256=_require_sha256(
            candidate_raw["readout_schema_sha256"], "P1i readout_schema_sha256"
        ),
        compiler_version=_require_text(
            candidate_raw["compiler_version"], "P1i compiler_version"
        ),
    )
    if payload["pipeline_sha256"] != candidate.pipeline_sha256:
        raise ValueError("P1i candidate pipeline hash mismatch")
    arms_raw = payload["arm_metrics"]
    folds_raw = payload["fold_metrics"]
    if not isinstance(arms_raw, dict) or not isinstance(folds_raw, dict):
        raise ValueError("P1i candidate metrics must be objects")
    arm_metrics = tuple(
        (
            arm.value,
            _parse_metrics(arms_raw.get(arm.value), f"P1i arm_metrics.{arm.value}"),
        )
        for arm in _EVALUATED_ARMS
    )
    if set(arms_raw) != {arm.value for arm in _EVALUATED_ARMS}:
        raise ValueError("P1i candidate arm metric keys mismatch")
    fold_metrics = tuple(
        (
            family,
            tuple(
                (
                    arm.value,
                    _parse_metrics(
                        metrics.get(arm.value),
                        f"P1i fold_metrics.{family}.{arm.value}",
                    ),
                )
                for arm in _EVALUATED_ARMS
            ),
        )
        for family, metrics in sorted(folds_raw.items())
        if isinstance(family, str) and isinstance(metrics, dict)
    )
    if len(fold_metrics) != len(folds_raw):
        raise ValueError("P1i candidate fold metrics contain invalid entries")
    seed_schedule_raw = payload["seed_schedule"]
    if not isinstance(seed_schedule_raw, list) or any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in seed_schedule_raw
    ):
        raise ValueError("P1i candidate seed schedule is invalid")
    artifact = RelationshipP1iCandidateArtifact(
        schema_version=_require_text(payload["schema_version"], "P1i candidate schema"),
        candidate=candidate,
        calibration_protocol_id=_require_sha256(
            payload["calibration_protocol_id"], "P1i calibration protocol id"
        ),
        dataset_fingerprint=_require_sha256(
            payload["dataset_fingerprint"], "P1i dataset fingerprint"
        ),
        training_context_surface_sha256=_require_sha256(
            payload["training_context_surface_sha256"], "P1i context surface"
        ),
        model_id=_require_text(payload["model_id"], "P1i model_id"),
        weights_sha256=_require_sha256(
            payload["weights_sha256"], "P1i weights_sha256"
        ),
        generation_config_sha256=_require_sha256(
            payload["generation_config_sha256"], "P1i generation config"
        ),
        seed_schedule=tuple(seed_schedule_raw),
        readout_ledger_sha256=_require_sha256(
            payload["readout_ledger_sha256"], "P1i readout ledger"
        ),
        decision_ledger_sha256=_require_sha256(
            payload["decision_ledger_sha256"], "P1i decision ledger"
        ),
        arm_metrics=arm_metrics,
        fold_metrics=fold_metrics,
    )
    if payload["selection_metrics"] != dict(artifact.selection_metrics):
        raise ValueError("P1i candidate selection metrics mismatch")
    if payload["artifact_id"] != artifact.artifact_id:
        raise ValueError("P1i candidate artifact_id mismatch")
    return artifact


@dataclass(frozen=True)
class RelationshipP1iCalibrationReport:
    created_at_iso: str
    calibration_protocol_id: str
    consumer_split_contract_id: str
    source_p1g_report_artifact_id: str
    training_dataset_fingerprint: str
    qualification_dataset_fingerprint: str
    candidate_artifacts: tuple[RelationshipP1iCandidateArtifact, ...]
    ranking: tuple[str, ...]
    selected_candidate_id: str
    training_labels_consumed: bool
    qualification_inputs_observed: int
    qualification_qwen_outputs_observed: int
    qualification_feedback_to_consumer: bool
    evaluation_feedback_to_pe_credit_reward_or_steering: bool
    formal_hidden_test_opened: bool
    p2_enabled: bool
    next_action: str
    claim_boundary: str
    schema_version: str = RELATIONSHIP_P1I_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1I_REPORT_SCHEMA_VERSION:
            raise ValueError("P1i calibration report schema_version mismatch")
        _require_timestamp(self.created_at_iso, "P1i report created_at_iso")
        for field_name, value in (
            ("calibration_protocol_id", self.calibration_protocol_id),
            ("consumer_split_contract_id", self.consumer_split_contract_id),
            ("source_p1g_report_artifact_id", self.source_p1g_report_artifact_id),
            ("training_dataset_fingerprint", self.training_dataset_fingerprint),
            (
                "qualification_dataset_fingerprint",
                self.qualification_dataset_fingerprint,
            ),
        ):
            _require_sha256(value, f"P1i report {field_name}")
        candidate_ids = tuple(
            item.candidate.candidate_id for item in self.candidate_artifacts
        )
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("P1i report candidates must be non-empty and unique")
        if tuple(item.candidate.round_index for item in self.candidate_artifacts) != tuple(
            range(1, len(self.candidate_artifacts) + 1)
        ):
            raise ValueError("P1i report must preserve candidates in round order")
        expected_ranking = tuple(
            item.candidate.candidate_id
            for item in sorted(
                self.candidate_artifacts,
                key=lambda artifact: artifact.selection_key,
                reverse=True,
            )
        )
        if self.ranking != expected_ranking:
            raise ValueError("P1i report ranking diverges from frozen selection rule")
        if self.selected_candidate_id != self.ranking[0]:
            raise ValueError("P1i report must select exactly the top-ranked candidate")
        if not self.training_labels_consumed:
            raise ValueError("P1i must disclose training-label calibration")
        if self.qualification_inputs_observed != 0 or self.qualification_qwen_outputs_observed != 0:
            raise ValueError("P1i report cannot contain v4 observations or outputs")
        if any(
            (
                self.qualification_feedback_to_consumer,
                self.evaluation_feedback_to_pe_credit_reward_or_steering,
                self.formal_hidden_test_opened,
                self.p2_enabled,
            )
        ):
            raise ValueError("P1i report cannot open qualification feedback, formal, or P2")
        if self.next_action != RELATIONSHIP_P1I_NEXT_ACTION:
            raise ValueError("P1i report next action is not frozen")
        if self.claim_boundary != _REPORT_CLAIM_BOUNDARY:
            raise ValueError("P1i report claim boundary mismatch")

    @property
    def selected_candidate(self) -> RelationshipP1iCandidateArtifact:
        return next(
            item
            for item in self.candidate_artifacts
            if item.candidate.candidate_id == self.selected_candidate_id
        )

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "calibration_protocol_id": self.calibration_protocol_id,
            "consumer_split_contract_id": self.consumer_split_contract_id,
            "source_p1g_report_artifact_id": self.source_p1g_report_artifact_id,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "qualification_dataset_fingerprint": (
                self.qualification_dataset_fingerprint
            ),
            "candidate_artifacts": [
                item.to_payload() for item in self.candidate_artifacts
            ],
            "ranking": list(self.ranking),
            "selected_candidate_id": self.selected_candidate_id,
            "training_labels_consumed": self.training_labels_consumed,
            "qualification_inputs_observed": self.qualification_inputs_observed,
            "qualification_qwen_outputs_observed": (
                self.qualification_qwen_outputs_observed
            ),
            "qualification_feedback_to_consumer": (
                self.qualification_feedback_to_consumer
            ),
            "evaluation_feedback_to_pe_credit_reward_or_steering": (
                self.evaluation_feedback_to_pe_credit_reward_or_steering
            ),
            "formal_hidden_test_opened": self.formal_hidden_test_opened,
            "p2_enabled": self.p2_enabled,
            "next_action": self.next_action,
            "claim_boundary": self.claim_boundary,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "RelationshipP1iCalibrationReport":
        raw = json.loads(encoded)
        if not isinstance(raw, dict):
            raise ValueError("P1i calibration report must be an object")
        expected = {
            "schema_version",
            "created_at_iso",
            "calibration_protocol_id",
            "consumer_split_contract_id",
            "source_p1g_report_artifact_id",
            "training_dataset_fingerprint",
            "qualification_dataset_fingerprint",
            "candidate_artifacts",
            "ranking",
            "selected_candidate_id",
            "training_labels_consumed",
            "qualification_inputs_observed",
            "qualification_qwen_outputs_observed",
            "qualification_feedback_to_consumer",
            "evaluation_feedback_to_pe_credit_reward_or_steering",
            "formal_hidden_test_opened",
            "p2_enabled",
            "next_action",
            "claim_boundary",
            "artifact_id",
        }
        _require_exact_keys(raw, expected, field_name="P1i calibration report")
        candidates_raw = raw["candidate_artifacts"]
        if not isinstance(candidates_raw, list):
            raise ValueError("P1i report candidate_artifacts must be an array")
        ranking = _require_string_list(raw["ranking"], "P1i report ranking")
        boolean_names = (
            "training_labels_consumed",
            "qualification_feedback_to_consumer",
            "evaluation_feedback_to_pe_credit_reward_or_steering",
            "formal_hidden_test_opened",
            "p2_enabled",
        )
        for name in boolean_names:
            if not isinstance(raw[name], bool):
                raise ValueError(f"P1i report {name} must be boolean")
        artifact_id = _require_sha256(raw["artifact_id"], "P1i report artifact_id")
        report = cls(
            schema_version=_require_text(raw["schema_version"], "P1i report schema"),
            created_at_iso=_require_timestamp(
                raw["created_at_iso"], "P1i report created_at_iso"
            ),
            calibration_protocol_id=_require_sha256(
                raw["calibration_protocol_id"], "P1i report protocol id"
            ),
            consumer_split_contract_id=_require_sha256(
                raw["consumer_split_contract_id"], "P1i report split contract"
            ),
            source_p1g_report_artifact_id=_require_sha256(
                raw["source_p1g_report_artifact_id"], "P1i report P1g source"
            ),
            training_dataset_fingerprint=_require_sha256(
                raw["training_dataset_fingerprint"], "P1i report training fingerprint"
            ),
            qualification_dataset_fingerprint=_require_sha256(
                raw["qualification_dataset_fingerprint"],
                "P1i report qualification fingerprint",
            ),
            candidate_artifacts=tuple(
                _candidate_artifact_from_payload(item) for item in candidates_raw
            ),
            ranking=ranking,
            selected_candidate_id=_require_text(
                raw["selected_candidate_id"], "P1i report selected candidate"
            ),
            training_labels_consumed=raw["training_labels_consumed"],
            qualification_inputs_observed=_require_int(
                raw["qualification_inputs_observed"], "P1i report v4 inputs"
            ),
            qualification_qwen_outputs_observed=_require_int(
                raw["qualification_qwen_outputs_observed"], "P1i report v4 outputs"
            ),
            qualification_feedback_to_consumer=raw[
                "qualification_feedback_to_consumer"
            ],
            evaluation_feedback_to_pe_credit_reward_or_steering=raw[
                "evaluation_feedback_to_pe_credit_reward_or_steering"
            ],
            formal_hidden_test_opened=raw["formal_hidden_test_opened"],
            p2_enabled=raw["p2_enabled"],
            next_action=_require_text(raw["next_action"], "P1i report next_action"),
            claim_boundary=_require_text(
                raw["claim_boundary"], "P1i report claim_boundary"
            ),
        )
        if report.artifact_id != artifact_id:
            raise ValueError("P1i calibration report artifact_id mismatch")
        return report


def assess_relationship_p1i_calibration(
    *,
    protocol: RelationshipP1iCalibrationProtocol,
    training_view: RelationshipConsumerTrainingView,
    candidate_artifacts: tuple[RelationshipP1iCandidateArtifact, ...],
    created_at_iso: str | None = None,
) -> RelationshipP1iCalibrationReport:
    validate_relationship_p1i_local_lineage(protocol, training_view=training_view)
    if tuple(item.candidate for item in candidate_artifacts) != protocol.candidates:
        raise ValueError("P1i report must preserve every frozen candidate in order")
    for artifact in candidate_artifacts:
        if (
            artifact.calibration_protocol_id != protocol.protocol_id
            or artifact.dataset_fingerprint != protocol.training_dataset_fingerprint
            or artifact.model_id != protocol.model_id
            or artifact.weights_sha256 != protocol.expected_weights_sha256
            or artifact.generation_config_sha256
            != protocol.expected_generation_config_sha256
            or artifact.seed_schedule != protocol.seed_schedule
        ):
            raise ValueError("P1i candidate artifact lineage mismatch")
    ranking = tuple(
        item.candidate.candidate_id
        for item in sorted(
            candidate_artifacts,
            key=lambda artifact: artifact.selection_key,
            reverse=True,
        )
    )
    return RelationshipP1iCalibrationReport(
        created_at_iso=(
            created_at_iso
            or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        ),
        calibration_protocol_id=protocol.protocol_id,
        consumer_split_contract_id=training_view.contract.contract_sha256,
        source_p1g_report_artifact_id=protocol.source_p1g_report_artifact_id,
        training_dataset_fingerprint=protocol.training_dataset_fingerprint,
        qualification_dataset_fingerprint=protocol.qualification_dataset_fingerprint,
        candidate_artifacts=candidate_artifacts,
        ranking=ranking,
        selected_candidate_id=ranking[0],
        training_labels_consumed=True,
        qualification_inputs_observed=0,
        qualification_qwen_outputs_observed=0,
        qualification_feedback_to_consumer=False,
        evaluation_feedback_to_pe_credit_reward_or_steering=False,
        formal_hidden_test_opened=False,
        p2_enabled=False,
        next_action=RELATIONSHIP_P1I_NEXT_ACTION,
        claim_boundary=_REPORT_CLAIM_BOUNDARY,
    )


@dataclass(frozen=True)
class RelationshipP1iFrozenConsumerProtocol:
    frozen_at_iso: str
    calibration_protocol_id: str
    calibration_report_artifact_id: str
    consumer_split_contract_id: str
    source_p1g_report_artifact_id: str
    training_package_name: str
    training_dataset_fingerprint: str
    qualification_package_name: str
    qualification_dataset_fingerprint: str
    selection_method: str
    candidate_ranking: tuple[str, ...]
    selected_candidate: RelationshipP1iCandidateSpec
    selected_candidate_artifact_id: str
    selected_pipeline_sha256: str
    model_source: str
    model_revision: str
    model_id: str
    expected_weights_sha256: str
    expected_generation_config_sha256: str
    device: str
    torch_dtype: str
    temperature: float
    top_p: float
    max_new_tokens: int
    seed_schedule: tuple[int, ...]
    background_depths: tuple[int, ...]
    training_context_surface_sha256: str
    background_templates_sha256: str
    rag_embedder: str
    rag_model_source: str
    rag_weights_sha256: str
    rag_top_k: int
    rag_candidate_surface: str
    rag_config_sha256: str
    required_valid_rate: float
    minimum_accuracy: float
    maximum_accuracy: float
    minimum_pair_flip_rate: float
    primary_qualification_arms: tuple[str, ...]
    structured_state_minimum_pair_flip_rate: float
    qualification_inputs_observed_before_freeze: int
    qualification_qwen_outputs_observed_before_freeze: int
    qualification_feedback_to_consumer: bool
    evaluation_feedback_to_pe_credit_reward_or_steering: bool
    formal_hidden_test_opened: bool
    p2_enabled: bool
    next_action: str
    claim_boundary: str
    schema_version: str = RELATIONSHIP_P1I_CONSUMER_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1I_CONSUMER_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("P1i frozen consumer schema_version mismatch")
        _require_timestamp(self.frozen_at_iso, "P1i consumer frozen_at_iso")
        for field_name, value in (
            ("calibration_protocol_id", self.calibration_protocol_id),
            ("calibration_report_artifact_id", self.calibration_report_artifact_id),
            ("consumer_split_contract_id", self.consumer_split_contract_id),
            ("source_p1g_report_artifact_id", self.source_p1g_report_artifact_id),
            ("training_dataset_fingerprint", self.training_dataset_fingerprint),
            (
                "qualification_dataset_fingerprint",
                self.qualification_dataset_fingerprint,
            ),
            ("selected_candidate_artifact_id", self.selected_candidate_artifact_id),
            ("selected_pipeline_sha256", self.selected_pipeline_sha256),
            ("expected_weights_sha256", self.expected_weights_sha256),
            (
                "expected_generation_config_sha256",
                self.expected_generation_config_sha256,
            ),
            (
                "training_context_surface_sha256",
                self.training_context_surface_sha256,
            ),
            ("background_templates_sha256", self.background_templates_sha256),
            ("rag_weights_sha256", self.rag_weights_sha256),
            ("rag_config_sha256", self.rag_config_sha256),
        ):
            _require_sha256(value, f"P1i consumer {field_name}")
        if self.training_package_name != RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
            raise ValueError("P1i consumer training package mismatch")
        if self.qualification_package_name != RELATIONSHIP_TRANSFER_V4_PACKAGE_NAME:
            raise ValueError("P1i consumer qualification package mismatch")
        if (
            not self.candidate_ranking
            or len(set(self.candidate_ranking)) != len(self.candidate_ranking)
        ):
            raise ValueError("P1i frozen consumer ranking must be non-empty and unique")
        if self.selected_candidate.candidate_id != self.candidate_ranking[0]:
            raise ValueError("P1i frozen consumer is not the top-ranked candidate")
        if self.selected_pipeline_sha256 != self.selected_candidate.pipeline_sha256:
            raise ValueError("P1i frozen consumer pipeline hash mismatch")
        if self.selection_method != "leave_one_surface_family_out_training_only":
            raise ValueError("P1i frozen consumer selection method mismatch")
        if (
            self.required_valid_rate != 1.0
            or self.minimum_accuracy != 0.625
            or self.maximum_accuracy != 0.875
            or self.minimum_pair_flip_rate != 0.5
            or self.primary_qualification_arms
            != ("prompt-steelman", "rag-steelman")
            or self.structured_state_minimum_pair_flip_rate != 0.5
        ):
            raise ValueError("P1i frozen consumer qualification gate drifted")
        if (
            self.qualification_inputs_observed_before_freeze != 0
            or self.qualification_qwen_outputs_observed_before_freeze != 0
        ):
            raise ValueError("P1i consumer must freeze before v4 input or output")
        if any(
            (
                self.qualification_feedback_to_consumer,
                self.evaluation_feedback_to_pe_credit_reward_or_steering,
                self.formal_hidden_test_opened,
                self.p2_enabled,
            )
        ):
            raise ValueError("P1i frozen consumer cannot open feedback, formal, or P2")
        if self.next_action != RELATIONSHIP_P1I_NEXT_ACTION:
            raise ValueError("P1i frozen consumer next action mismatch")
        if self.claim_boundary != _CONSUMER_PROTOCOL_CLAIM_BOUNDARY:
            raise ValueError("P1i frozen consumer claim boundary mismatch")

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_at_iso": self.frozen_at_iso,
            "source_lineage": {
                "calibration_protocol_id": self.calibration_protocol_id,
                "calibration_report_artifact_id": (
                    self.calibration_report_artifact_id
                ),
                "consumer_split_contract_id": self.consumer_split_contract_id,
                "source_p1g_report_artifact_id": self.source_p1g_report_artifact_id,
                "training_package_name": self.training_package_name,
                "training_dataset_fingerprint": self.training_dataset_fingerprint,
                "qualification_package_name": self.qualification_package_name,
                "qualification_dataset_fingerprint": (
                    self.qualification_dataset_fingerprint
                ),
            },
            "selection": {
                "selection_method": self.selection_method,
                "candidate_ranking": list(self.candidate_ranking),
                "selected_candidate": self.selected_candidate.to_payload(),
                "selected_candidate_artifact_id": (
                    self.selected_candidate_artifact_id
                ),
                "selected_pipeline_sha256": self.selected_pipeline_sha256,
            },
            "runtime": {
                "model_source": self.model_source,
                "model_revision": self.model_revision,
                "model_id": self.model_id,
                "expected_weights_sha256": self.expected_weights_sha256,
                "expected_generation_config_sha256": (
                    self.expected_generation_config_sha256
                ),
                "device": self.device,
                "torch_dtype": self.torch_dtype,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
                "seed_schedule": list(self.seed_schedule),
                "background_depths": list(self.background_depths),
                "training_context_surface_sha256": (
                    self.training_context_surface_sha256
                ),
                "background_templates_sha256": self.background_templates_sha256,
                "rag_embedder": self.rag_embedder,
                "rag_model_source": self.rag_model_source,
                "rag_weights_sha256": self.rag_weights_sha256,
                "rag_top_k": self.rag_top_k,
                "rag_candidate_surface": self.rag_candidate_surface,
                "rag_config_sha256": self.rag_config_sha256,
            },
            "qualification_gate": {
                "required_valid_rate": self.required_valid_rate,
                "minimum_accuracy": self.minimum_accuracy,
                "maximum_accuracy": self.maximum_accuracy,
                "minimum_pair_flip_rate": self.minimum_pair_flip_rate,
                "primary_qualification_arms": list(
                    self.primary_qualification_arms
                ),
                "structured_state_minimum_pair_flip_rate": (
                    self.structured_state_minimum_pair_flip_rate
                ),
            },
            "experiment_guards": {
                "qualification_inputs_observed_before_freeze": (
                    self.qualification_inputs_observed_before_freeze
                ),
                "qualification_qwen_outputs_observed_before_freeze": (
                    self.qualification_qwen_outputs_observed_before_freeze
                ),
                "qualification_feedback_to_consumer": (
                    self.qualification_feedback_to_consumer
                ),
                "evaluation_feedback_to_pe_credit_reward_or_steering": (
                    self.evaluation_feedback_to_pe_credit_reward_or_steering
                ),
                "formal_hidden_test_opened": self.formal_hidden_test_opened,
                "p2_enabled": self.p2_enabled,
            },
            "next_action": self.next_action,
            "claim_boundary": self.claim_boundary,
        }

    @property
    def protocol_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["protocol_id"] = self.protocol_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "RelationshipP1iFrozenConsumerProtocol":
        raw = json.loads(encoded)
        if not isinstance(raw, dict):
            raise ValueError("P1i frozen consumer protocol must be an object")
        _require_exact_keys(
            raw,
            {
                "schema_version",
                "frozen_at_iso",
                "source_lineage",
                "selection",
                "runtime",
                "qualification_gate",
                "experiment_guards",
                "next_action",
                "claim_boundary",
                "protocol_id",
            },
            field_name="P1i frozen consumer protocol",
        )
        source = _require_object(
            raw["source_lineage"],
            {
                "calibration_protocol_id",
                "calibration_report_artifact_id",
                "consumer_split_contract_id",
                "source_p1g_report_artifact_id",
                "training_package_name",
                "training_dataset_fingerprint",
                "qualification_package_name",
                "qualification_dataset_fingerprint",
            },
            field_name="P1i consumer source lineage",
        )
        selection = _require_object(
            raw["selection"],
            {
                "selection_method",
                "candidate_ranking",
                "selected_candidate",
                "selected_candidate_artifact_id",
                "selected_pipeline_sha256",
            },
            field_name="P1i consumer selection",
        )
        runtime = _require_object(
            raw["runtime"],
            {
                "model_source",
                "model_revision",
                "model_id",
                "expected_weights_sha256",
                "expected_generation_config_sha256",
                "device",
                "torch_dtype",
                "temperature",
                "top_p",
                "max_new_tokens",
                "seed_schedule",
                "background_depths",
                "training_context_surface_sha256",
                "background_templates_sha256",
                "rag_embedder",
                "rag_model_source",
                "rag_weights_sha256",
                "rag_top_k",
                "rag_candidate_surface",
                "rag_config_sha256",
            },
            field_name="P1i consumer runtime",
        )
        gate = _require_object(
            raw["qualification_gate"],
            {
                "required_valid_rate",
                "minimum_accuracy",
                "maximum_accuracy",
                "minimum_pair_flip_rate",
                "primary_qualification_arms",
                "structured_state_minimum_pair_flip_rate",
            },
            field_name="P1i consumer qualification gate",
        )
        guards = _require_object(
            raw["experiment_guards"],
            {
                "qualification_inputs_observed_before_freeze",
                "qualification_qwen_outputs_observed_before_freeze",
                "qualification_feedback_to_consumer",
                "evaluation_feedback_to_pe_credit_reward_or_steering",
                "formal_hidden_test_opened",
                "p2_enabled",
            },
            field_name="P1i consumer experiment guards",
        )
        candidate = _require_object(
            selection["selected_candidate"],
            {
                "round_index",
                "candidate_id",
                "prompt_asset",
                "prompt_sha256",
                "request_template_asset",
                "request_template_sha256",
                "readout_schema_sha256",
                "compiler_version",
            },
            field_name="P1i consumer selected candidate",
        )
        seed_schedule = runtime["seed_schedule"]
        background_depths = runtime["background_depths"]
        if not isinstance(seed_schedule, list) or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in seed_schedule
        ):
            raise ValueError("P1i consumer seed_schedule must be an integer array")
        if not isinstance(background_depths, list) or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in background_depths
        ):
            raise ValueError("P1i consumer background_depths must be an integer array")
        for field_name in (
            "qualification_feedback_to_consumer",
            "evaluation_feedback_to_pe_credit_reward_or_steering",
            "formal_hidden_test_opened",
            "p2_enabled",
        ):
            if not isinstance(guards[field_name], bool):
                raise ValueError(f"P1i consumer {field_name} must be boolean")
        protocol_id = _require_sha256(raw["protocol_id"], "P1i consumer protocol_id")
        selected_candidate = RelationshipP1iCandidateSpec(
            round_index=_require_int(candidate["round_index"], "P1i selected round"),
            candidate_id=_require_text(
                candidate["candidate_id"], "P1i selected candidate id"
            ),
            prompt_asset=_require_text(
                candidate["prompt_asset"], "P1i selected prompt asset"
            ),
            prompt_sha256=_require_sha256(
                candidate["prompt_sha256"], "P1i selected prompt hash"
            ),
            request_template_asset=_require_text(
                candidate["request_template_asset"], "P1i selected request asset"
            ),
            request_template_sha256=_require_sha256(
                candidate["request_template_sha256"], "P1i selected request hash"
            ),
            readout_schema_sha256=_require_sha256(
                candidate["readout_schema_sha256"], "P1i selected schema hash"
            ),
            compiler_version=_require_text(
                candidate["compiler_version"], "P1i selected compiler"
            ),
        )
        protocol = cls(
            schema_version=_require_text(raw["schema_version"], "P1i consumer schema"),
            frozen_at_iso=_require_timestamp(
                raw["frozen_at_iso"], "P1i consumer frozen_at_iso"
            ),
            calibration_protocol_id=_require_sha256(
                source["calibration_protocol_id"], "P1i calibration protocol"
            ),
            calibration_report_artifact_id=_require_sha256(
                source["calibration_report_artifact_id"], "P1i calibration report"
            ),
            consumer_split_contract_id=_require_sha256(
                source["consumer_split_contract_id"], "P1i split contract"
            ),
            source_p1g_report_artifact_id=_require_sha256(
                source["source_p1g_report_artifact_id"], "P1i P1g source"
            ),
            training_package_name=_require_text(
                source["training_package_name"], "P1i training package"
            ),
            training_dataset_fingerprint=_require_sha256(
                source["training_dataset_fingerprint"], "P1i training fingerprint"
            ),
            qualification_package_name=_require_text(
                source["qualification_package_name"], "P1i qualification package"
            ),
            qualification_dataset_fingerprint=_require_sha256(
                source["qualification_dataset_fingerprint"],
                "P1i qualification fingerprint",
            ),
            selection_method=_require_text(
                selection["selection_method"], "P1i selection method"
            ),
            candidate_ranking=_require_string_list(
                selection["candidate_ranking"], "P1i candidate ranking"
            ),
            selected_candidate=selected_candidate,
            selected_candidate_artifact_id=_require_sha256(
                selection["selected_candidate_artifact_id"],
                "P1i selected candidate artifact",
            ),
            selected_pipeline_sha256=_require_sha256(
                selection["selected_pipeline_sha256"], "P1i selected pipeline"
            ),
            model_source=_require_text(runtime["model_source"], "P1i model source"),
            model_revision=_require_text(
                runtime["model_revision"], "P1i model revision"
            ),
            model_id=_require_text(runtime["model_id"], "P1i model id"),
            expected_weights_sha256=_require_sha256(
                runtime["expected_weights_sha256"], "P1i model weights"
            ),
            expected_generation_config_sha256=_require_sha256(
                runtime["expected_generation_config_sha256"],
                "P1i generation config",
            ),
            device=_require_text(runtime["device"], "P1i device"),
            torch_dtype=_require_text(runtime["torch_dtype"], "P1i torch dtype"),
            temperature=_require_number(runtime["temperature"], "P1i temperature"),
            top_p=_require_number(runtime["top_p"], "P1i top_p"),
            max_new_tokens=_require_int(
                runtime["max_new_tokens"], "P1i max_new_tokens"
            ),
            seed_schedule=tuple(seed_schedule),
            background_depths=tuple(background_depths),
            training_context_surface_sha256=_require_sha256(
                runtime["training_context_surface_sha256"], "P1i context surface"
            ),
            background_templates_sha256=_require_sha256(
                runtime["background_templates_sha256"], "P1i background templates"
            ),
            rag_embedder=_require_text(runtime["rag_embedder"], "P1i RAG embedder"),
            rag_model_source=_require_text(
                runtime["rag_model_source"], "P1i RAG model"
            ),
            rag_weights_sha256=_require_sha256(
                runtime["rag_weights_sha256"], "P1i RAG weights"
            ),
            rag_top_k=_require_int(runtime["rag_top_k"], "P1i RAG top_k"),
            rag_candidate_surface=_require_text(
                runtime["rag_candidate_surface"], "P1i RAG candidate surface"
            ),
            rag_config_sha256=_require_sha256(
                runtime["rag_config_sha256"], "P1i RAG config"
            ),
            required_valid_rate=_require_number(
                gate["required_valid_rate"], "P1i required valid rate"
            ),
            minimum_accuracy=_require_number(
                gate["minimum_accuracy"], "P1i minimum accuracy"
            ),
            maximum_accuracy=_require_number(
                gate["maximum_accuracy"], "P1i maximum accuracy"
            ),
            minimum_pair_flip_rate=_require_number(
                gate["minimum_pair_flip_rate"], "P1i minimum pair flip"
            ),
            primary_qualification_arms=_require_string_list(
                gate["primary_qualification_arms"], "P1i primary arms"
            ),
            structured_state_minimum_pair_flip_rate=_require_number(
                gate["structured_state_minimum_pair_flip_rate"],
                "P1i structured pair flip",
            ),
            qualification_inputs_observed_before_freeze=_require_int(
                guards["qualification_inputs_observed_before_freeze"],
                "P1i v4 inputs",
            ),
            qualification_qwen_outputs_observed_before_freeze=_require_int(
                guards["qualification_qwen_outputs_observed_before_freeze"],
                "P1i v4 outputs",
            ),
            qualification_feedback_to_consumer=guards[
                "qualification_feedback_to_consumer"
            ],
            evaluation_feedback_to_pe_credit_reward_or_steering=guards[
                "evaluation_feedback_to_pe_credit_reward_or_steering"
            ],
            formal_hidden_test_opened=guards["formal_hidden_test_opened"],
            p2_enabled=guards["p2_enabled"],
            next_action=_require_text(raw["next_action"], "P1i next action"),
            claim_boundary=_require_text(
                raw["claim_boundary"], "P1i consumer claim boundary"
            ),
        )
        if protocol.protocol_id != protocol_id:
            raise ValueError("P1i frozen consumer protocol_id mismatch")
        return protocol


def freeze_relationship_p1i_consumer_protocol(
    *,
    calibration_protocol: RelationshipP1iCalibrationProtocol,
    report: RelationshipP1iCalibrationReport,
    training_view: RelationshipConsumerTrainingView,
    frozen_at_iso: str | None = None,
) -> RelationshipP1iFrozenConsumerProtocol:
    validate_relationship_p1i_local_lineage(
        calibration_protocol,
        training_view=training_view,
    )
    if (
        report.calibration_protocol_id != calibration_protocol.protocol_id
        or report.consumer_split_contract_id
        != training_view.contract.contract_sha256
        or report.qualification_inputs_observed != 0
        or report.qualification_qwen_outputs_observed != 0
    ):
        raise ValueError("P1i report cannot freeze this consumer protocol")
    selected = report.selected_candidate
    contract = training_view.contract
    return RelationshipP1iFrozenConsumerProtocol(
        frozen_at_iso=(
            frozen_at_iso
            or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        ),
        calibration_protocol_id=calibration_protocol.protocol_id,
        calibration_report_artifact_id=report.artifact_id,
        consumer_split_contract_id=contract.contract_sha256,
        source_p1g_report_artifact_id=report.source_p1g_report_artifact_id,
        training_package_name=calibration_protocol.training_package_name,
        training_dataset_fingerprint=calibration_protocol.training_dataset_fingerprint,
        qualification_package_name=calibration_protocol.qualification_package_name,
        qualification_dataset_fingerprint=(
            calibration_protocol.qualification_dataset_fingerprint
        ),
        selection_method=calibration_protocol.selection_method,
        candidate_ranking=report.ranking,
        selected_candidate=selected.candidate,
        selected_candidate_artifact_id=selected.artifact_id,
        selected_pipeline_sha256=selected.candidate.pipeline_sha256,
        model_source=calibration_protocol.model_source,
        model_revision=calibration_protocol.model_revision,
        model_id=calibration_protocol.model_id,
        expected_weights_sha256=calibration_protocol.expected_weights_sha256,
        expected_generation_config_sha256=(
            calibration_protocol.expected_generation_config_sha256
        ),
        device=calibration_protocol.device,
        torch_dtype=calibration_protocol.torch_dtype,
        temperature=calibration_protocol.temperature,
        top_p=calibration_protocol.top_p,
        max_new_tokens=calibration_protocol.max_new_tokens,
        seed_schedule=calibration_protocol.seed_schedule,
        background_depths=calibration_protocol.background_depths,
        training_context_surface_sha256=(
            calibration_protocol.training_context_surface_sha256
        ),
        background_templates_sha256=(
            calibration_protocol.background_templates_sha256
        ),
        rag_embedder=calibration_protocol.rag_embedder,
        rag_model_source=calibration_protocol.rag_model_source,
        rag_weights_sha256=calibration_protocol.rag_weights_sha256,
        rag_top_k=calibration_protocol.rag_top_k,
        rag_candidate_surface=calibration_protocol.rag_candidate_surface,
        rag_config_sha256=calibration_protocol.rag_config_sha256,
        required_valid_rate=contract.required_valid_rate,
        minimum_accuracy=contract.minimum_accuracy,
        maximum_accuracy=contract.maximum_accuracy,
        minimum_pair_flip_rate=contract.minimum_pair_flip_rate,
        primary_qualification_arms=contract.primary_qualification_arms,
        structured_state_minimum_pair_flip_rate=(
            contract.structured_state_minimum_pair_flip_rate
        ),
        qualification_inputs_observed_before_freeze=0,
        qualification_qwen_outputs_observed_before_freeze=0,
        qualification_feedback_to_consumer=False,
        evaluation_feedback_to_pe_credit_reward_or_steering=False,
        formal_hidden_test_opened=False,
        p2_enabled=False,
        next_action=RELATIONSHIP_P1I_NEXT_ACTION,
        claim_boundary=_CONSUMER_PROTOCOL_CLAIM_BOUNDARY,
    )


def validate_relationship_p1i_frozen_consumer_lineage(
    consumer: RelationshipP1iFrozenConsumerProtocol,
    *,
    calibration_protocol: RelationshipP1iCalibrationProtocol,
    report: RelationshipP1iCalibrationReport,
    training_view: RelationshipConsumerTrainingView,
) -> None:
    validate_relationship_p1i_local_lineage(
        calibration_protocol,
        training_view=training_view,
    )
    selected = report.selected_candidate
    contract = training_view.contract
    expected = {
        "calibration_protocol_id": calibration_protocol.protocol_id,
        "calibration_report_artifact_id": report.artifact_id,
        "consumer_split_contract_id": contract.contract_sha256,
        "source_p1g_report_artifact_id": report.source_p1g_report_artifact_id,
        "training_package_name": calibration_protocol.training_package_name,
        "training_dataset_fingerprint": (
            calibration_protocol.training_dataset_fingerprint
        ),
        "qualification_package_name": (
            calibration_protocol.qualification_package_name
        ),
        "qualification_dataset_fingerprint": (
            calibration_protocol.qualification_dataset_fingerprint
        ),
        "selection_method": calibration_protocol.selection_method,
        "candidate_ranking": report.ranking,
        "selected_candidate": selected.candidate,
        "selected_candidate_artifact_id": selected.artifact_id,
        "selected_pipeline_sha256": selected.candidate.pipeline_sha256,
        "model_source": calibration_protocol.model_source,
        "model_revision": calibration_protocol.model_revision,
        "model_id": calibration_protocol.model_id,
        "expected_weights_sha256": calibration_protocol.expected_weights_sha256,
        "expected_generation_config_sha256": (
            calibration_protocol.expected_generation_config_sha256
        ),
        "device": calibration_protocol.device,
        "torch_dtype": calibration_protocol.torch_dtype,
        "temperature": calibration_protocol.temperature,
        "top_p": calibration_protocol.top_p,
        "max_new_tokens": calibration_protocol.max_new_tokens,
        "seed_schedule": calibration_protocol.seed_schedule,
        "background_depths": calibration_protocol.background_depths,
        "training_context_surface_sha256": (
            calibration_protocol.training_context_surface_sha256
        ),
        "background_templates_sha256": (
            calibration_protocol.background_templates_sha256
        ),
        "rag_embedder": calibration_protocol.rag_embedder,
        "rag_model_source": calibration_protocol.rag_model_source,
        "rag_weights_sha256": calibration_protocol.rag_weights_sha256,
        "rag_top_k": calibration_protocol.rag_top_k,
        "rag_candidate_surface": calibration_protocol.rag_candidate_surface,
        "rag_config_sha256": calibration_protocol.rag_config_sha256,
        "required_valid_rate": contract.required_valid_rate,
        "minimum_accuracy": contract.minimum_accuracy,
        "maximum_accuracy": contract.maximum_accuracy,
        "minimum_pair_flip_rate": contract.minimum_pair_flip_rate,
        "primary_qualification_arms": contract.primary_qualification_arms,
        "structured_state_minimum_pair_flip_rate": (
            contract.structured_state_minimum_pair_flip_rate
        ),
    }
    consumer_fields = vars(consumer)
    drift = sorted(
        name for name, value in expected.items() if consumer_fields[name] != value
    )
    if drift:
        raise ValueError(f"P1i frozen consumer lineage mismatch: {drift}")
    if report.calibration_protocol_id != calibration_protocol.protocol_id:
        raise ValueError("P1i report/calibration protocol lineage mismatch")
    if report.consumer_split_contract_id != contract.contract_sha256:
        raise ValueError("P1i report/split contract lineage mismatch")
    consumer_time = datetime.fromisoformat(
        _require_timestamp(
            consumer.frozen_at_iso,
            "P1i consumer frozen_at_iso",
        ).replace("Z", "+00:00")
    )
    report_time = datetime.fromisoformat(
        _require_timestamp(
            report.created_at_iso,
            "P1i report created_at_iso",
        ).replace("Z", "+00:00")
    )
    if consumer_time < report_time:
        raise ValueError("P1i consumer cannot predate its calibration report")


def write_relationship_p1i_candidate_artifact(
    *,
    run: RelationshipP1iCandidateRun,
    artifact: RelationshipP1iCandidateArtifact,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    if artifact.candidate != run.candidate:
        raise ValueError("P1i candidate summary does not match its run")
    target = pathlib.Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    readout_path = target / "readouts.jsonl"
    decision_path = target / "decisions.jsonl"
    summary_path = target / "candidate.json"
    readout_path.write_text(run.readout_ledger_jsonl(), encoding="utf-8")
    decision_path.write_text(run.decision_ledger_jsonl(), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            artifact.to_payload(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path


def finalize_relationship_p1i_candidate_checkpoint(
    *,
    run: RelationshipP1iCandidateRun,
    artifact: RelationshipP1iCandidateArtifact,
    candidate_dir: pathlib.Path,
) -> pathlib.Path:
    target = pathlib.Path(candidate_dir)
    progress = load_relationship_p1i_candidate_progress(target)
    if (
        artifact.candidate != run.candidate
        or progress.checkpoint.candidate != run.candidate
        or progress.checkpoint.calibration_protocol_id
        != run.calibration_protocol_id
        or progress.checkpoint.dataset_fingerprint != run.dataset_fingerprint
        or progress.checkpoint.training_context_surface_sha256
        != run.training_context_surface_sha256
        or progress.checkpoint.model_id != run.model_id
        or progress.checkpoint.weights_sha256 != run.weights_sha256
        or progress.checkpoint.generation_config_sha256
        != run.generation_config_sha256
        or progress.checkpoint.seed_schedule != run.seed_schedule
        or progress.readouts != run.readouts
        or progress.decisions != run.decisions
        or not progress.is_complete
    ):
        raise ValueError("P1i final candidate diverges from its durable checkpoint")
    readout_path = target / "readouts.jsonl"
    decision_path = target / "decisions.jsonl"
    summary_path = target / "candidate.json"
    if summary_path.exists():
        loaded = load_relationship_p1i_candidate_artifact(target)
        if loaded != artifact:
            raise ValueError("P1i finalized candidate artifact changed on resume")
        return summary_path
    _atomic_write_text(readout_path, run.readout_ledger_jsonl())
    _atomic_write_text(decision_path, run.decision_ledger_jsonl())
    _atomic_write_text(
        summary_path,
        json.dumps(
            artifact.to_payload(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return summary_path


def load_relationship_p1i_candidate_artifact(
    candidate_dir: pathlib.Path,
) -> RelationshipP1iCandidateArtifact:
    target = pathlib.Path(candidate_dir)
    summary_path = target / "candidate.json"
    readout_path = target / "readouts.jsonl"
    decision_path = target / "decisions.jsonl"
    for path in (summary_path, readout_path, decision_path):
        if not path.is_file():
            raise FileNotFoundError(f"P1i candidate artifact is missing: {path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("P1i candidate summary is not valid JSON") from exc
    artifact = _candidate_artifact_from_payload(summary)
    if _sha256_file(readout_path) != artifact.readout_ledger_sha256:
        raise ValueError("P1i candidate readout ledger hash mismatch")
    if _sha256_file(decision_path) != artifact.decision_ledger_sha256:
        raise ValueError("P1i candidate decision ledger hash mismatch")
    checkpoint_path = _checkpoint_path(target)
    if checkpoint_path.exists():
        checkpoint = load_relationship_p1i_candidate_checkpoint(target)
        if checkpoint.candidate != artifact.candidate:
            raise ValueError("P1i candidate summary/checkpoint lineage mismatch")
    return artifact


def write_relationship_p1i_report_and_protocol(
    *,
    report: RelationshipP1iCalibrationReport,
    consumer_protocol: RelationshipP1iFrozenConsumerProtocol,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    if consumer_protocol.calibration_report_artifact_id != report.artifact_id:
        raise ValueError("P1i report/consumer protocol lineage mismatch")
    target = pathlib.Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / "packet1i_report.json"
    markdown_path = target / "packet1i_report.md"
    consumer_path = target / "frozen_consumer_protocol.json"
    existing = tuple(
        path for path in (report_path, markdown_path, consumer_path) if path.exists()
    )
    lines = [
        "# Relationship Lab P1i training-only consumer calibration",
        "",
        f"- Calibration protocol: `{report.calibration_protocol_id}`",
        f"- Report artifact: `{report.artifact_id}`",
        f"- Frozen consumer protocol: `{consumer_protocol.protocol_id}`",
        f"- Selected candidate: `{report.selected_candidate_id}`",
        f"- Ranking: `{', '.join(report.ranking)}`",
        "- v4 inputs/outputs observed: `0 / 0`",
        "- Formal hidden test / P2: `closed / disabled`",
        "",
        "## Candidate metrics",
        "",
        "| Candidate | Arm | Valid | Accuracy | Pair flip |",
        "|---|---|---:|---:|---:|",
    ]
    for artifact in report.candidate_artifacts:
        for arm, metrics_items in artifact.arm_metrics:
            metrics = dict(metrics_items)
            lines.append(
                "| "
                f"{artifact.candidate.candidate_id} | {arm} | "
                f"{metrics['valid_readouts']}/{metrics['readouts']} | "
                f"{float(metrics['accuracy']):.3f} | "
                f"{float(metrics['pair_flip_rate']):.3f} |"
            )
    lines.extend(("", "## Claim boundary", "", report.claim_boundary, ""))
    expected_contents = {
        report_path: report.to_json(),
        consumer_path: consumer_protocol.to_json(),
        markdown_path: "\n".join(lines),
    }
    for path in existing:
        if path.read_text(encoding="utf-8") != expected_contents[path]:
            raise ValueError(f"P1i existing final artifact diverged: {path}")
    for path, content in expected_contents.items():
        if not path.exists():
            _atomic_write_text(path, content)
    return report_path, markdown_path, consumer_path


def load_relationship_p1i_calibration_report(
    path: pathlib.Path,
) -> RelationshipP1iCalibrationReport:
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"P1i calibration report is missing: {file_path}")
    return RelationshipP1iCalibrationReport.from_json(
        file_path.read_text(encoding="utf-8")
    )


def load_relationship_p1i_frozen_consumer_protocol(
    path: pathlib.Path,
) -> RelationshipP1iFrozenConsumerProtocol:
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"P1i frozen consumer protocol is missing: {file_path}")
    return RelationshipP1iFrozenConsumerProtocol.from_json(
        file_path.read_text(encoding="utf-8")
    )


def validate_relationship_p1i_candidate_files(
    *,
    report: RelationshipP1iCalibrationReport,
    output_dir: pathlib.Path,
) -> None:
    root = pathlib.Path(output_dir)
    for artifact in report.candidate_artifacts:
        candidate_dir = root / f"candidate_{artifact.candidate.round_index:02d}"
        loaded = load_relationship_p1i_candidate_artifact(candidate_dir)
        if loaded != artifact:
            raise ValueError("P1i candidate summary diverges from report")


__all__ = [
    "RELATIONSHIP_P1I_CHECKPOINT_SCHEMA_VERSION",
    "RELATIONSHIP_P1I_CONSUMER_PROTOCOL_SCHEMA_VERSION",
    "RELATIONSHIP_P1I_NEXT_ACTION",
    "RELATIONSHIP_P1I_PROTOCOL_SCHEMA_VERSION",
    "RELATIONSHIP_P1I_REPORT_SCHEMA_VERSION",
    "RelationshipP1iCalibrationProtocol",
    "RelationshipP1iCalibrationReport",
    "RelationshipP1iCandidateCheckpoint",
    "RelationshipP1iCandidateArtifact",
    "RelationshipP1iCandidateProgress",
    "RelationshipP1iCandidateRun",
    "RelationshipP1iCandidateSpec",
    "RelationshipP1iFrozenConsumerProtocol",
    "assess_relationship_p1i_calibration",
    "build_relationship_p1i_candidate_checkpoint",
    "finalize_relationship_p1i_candidate_checkpoint",
    "freeze_relationship_p1i_consumer_protocol",
    "load_relationship_p1i_calibration_protocol",
    "load_relationship_p1i_calibration_report",
    "load_relationship_p1i_candidate_prompt",
    "load_relationship_p1i_candidate_artifact",
    "load_relationship_p1i_candidate_checkpoint",
    "load_relationship_p1i_candidate_progress",
    "load_relationship_p1i_frozen_consumer_protocol",
    "relationship_p1i_calibration_protocol_path",
    "relationship_p1i_decision_from_record_payload",
    "relationship_p1i_decision_record_payload",
    "relationship_p1i_readout_completion",
    "relationship_p1i_readout_from_record_payload",
    "relationship_p1i_readout_record_payload",
    "relationship_p1i_run_from_progress",
    "relationship_p1i_training_context_surface_sha256",
    "render_relationship_p1i_candidate_request",
    "run_relationship_p1i_candidate",
    "summarize_relationship_p1i_candidate",
    "validate_relationship_p1i_candidate_files",
    "validate_relationship_p1i_candidate_progress",
    "validate_relationship_p1i_context_lineage",
    "validate_relationship_p1i_frozen_consumer_lineage",
    "validate_relationship_p1i_local_lineage",
    "write_relationship_p1i_candidate_artifact",
    "write_relationship_p1i_candidate_checkpoint",
    "persist_relationship_p1i_decision",
    "persist_relationship_p1i_readout",
    "write_relationship_p1i_report_and_protocol",
]
