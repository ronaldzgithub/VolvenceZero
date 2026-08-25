"""Relationship-domain corpus adapter for named-action residual fitting.

The adapter consumes only frozen *pre-action* owner values: one
``PreferenceActionForecast`` and its non-oracle relationship gate decision.
It cannot accept an evaluator bundle, an observed outcome, a judge score, or a
credit record.  Its output is the domain-neutral corpus contract owned by
``vz-runtime``; model capture and operator fitting happen there.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import pathlib

from volvence_zero.agent.named_action_steering_artifact_training import (
    NamedActionSteeringCorpus,
    NamedActionSteeringRow,
)
from volvence_zero.social_cognition import PreferenceActionForecast

from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RelationshipAction,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGateDecision,
    RelationshipActionGateMode,
    RelationshipGateAction,
)


RELATIONSHIP_RESIDUAL_FIT_PROTOCOL_SCHEMA_VERSION = (
    "relationship-residual-named-action-fit-protocol.v1"
)
RELATIONSHIP_RESIDUAL_CONDITION_PROJECTION_SCHEMA_VERSION = (
    "relationship-owner-forecast-gate-condition.v1"
)
_DEFAULT_PROTOCOL_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "lab_protocols"
    / "relationship_residual_named_action_fit_v1.json"
)


@dataclass(frozen=True)
class RelationshipResidualFitProtocol:
    protocol_sha256: str
    action_ids: tuple[str, ...]
    strict_noop_action_id: str
    required_action_id_occurrences: int
    forbidden_input_fields: tuple[str, ...]
    forbidden_gate_modes: tuple[RelationshipActionGateMode, ...]
    minimum_train_rows_per_action: int
    minimum_heldout_rows_per_action: int
    model_id: str
    model_revision: str
    model_weights_sha256: str
    execution_assets_sha256: str
    injection_layer_index: int
    residual_width: int
    steering_rank: int
    conditional_executor_updates: int
    sensor_off_executor_updates: int
    executor_learning_rate: float
    reader_ridge_lambda: float
    batch_size: int
    seed: int
    control_norm_cap_ratio: float
    claim_boundary: str

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_sha256, "protocol_sha256")
        directional = tuple(
            action.value
            for action in RELATIONSHIP_ACTIONS
            if action is not RelationshipAction.NEUTRAL_NOOP
        )
        if self.action_ids != directional:
            raise ValueError(
                "relationship residual protocol action surface differs from the "
                "frozen relationship action contract"
            )
        if self.strict_noop_action_id != RelationshipAction.NEUTRAL_NOOP.value:
            raise ValueError("relationship residual strict-noop action drift")
        if self.required_action_id_occurrences < 1:
            raise ValueError("required action-id occurrences must be positive")
        if self.forbidden_gate_modes != (RelationshipActionGateMode.ORACLE,):
            raise ValueError("relationship residual forbidden gate-mode set drift")
        for value in (
            self.minimum_train_rows_per_action,
            self.minimum_heldout_rows_per_action,
        ):
            if value < 1:
                raise ValueError("relationship residual split minima must be positive")
        for field_name, value in (
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("claim_boundary", self.claim_boundary),
        ):
            _require_text(value, field_name)
        _require_sha256(self.model_weights_sha256, "model_weights_sha256")
        _require_sha256(self.execution_assets_sha256, "execution_assets_sha256")
        if self.injection_layer_index < 0 or self.residual_width < 1:
            raise ValueError("relationship residual model geometry is invalid")
        if not 1 <= self.steering_rank <= self.residual_width:
            raise ValueError("relationship residual steering rank is invalid")
        for value in (
            self.conditional_executor_updates,
            self.sensor_off_executor_updates,
            self.batch_size,
        ):
            if value < 1:
                raise ValueError("relationship residual fit counts must be positive")
        if self.conditional_executor_updates != self.sensor_off_executor_updates:
            raise ValueError(
                "relationship residual v1 requires matched conditional/sensor-off updates"
            )
        if self.seed < 0:
            raise ValueError("relationship residual seed must be non-negative")
        if self.executor_learning_rate <= 0.0 or self.reader_ridge_lambda <= 0.0:
            raise ValueError("relationship residual fit rates must be positive")
        if not 0.0 < self.control_norm_cap_ratio <= 2.0:
            raise ValueError("relationship residual norm cap ratio is invalid")


@dataclass(frozen=True)
class RelationshipResidualFitInput:
    """One pre-action public prompt joined to exact owner/gate snapshots."""

    row_id: str
    subject_scope: str
    public_action_text: str
    forecast: PreferenceActionForecast
    gate_decision: RelationshipActionGateDecision

    def __post_init__(self) -> None:
        for field_name, value in (
            ("row_id", self.row_id),
            ("subject_scope", self.subject_scope),
            ("public_action_text", self.public_action_text),
        ):
            _require_text(value, field_name)
        if not isinstance(self.forecast, PreferenceActionForecast):
            raise TypeError("forecast must be a frozen PreferenceActionForecast")
        if not isinstance(self.gate_decision, RelationshipActionGateDecision):
            raise TypeError(
                "gate_decision must be a frozen RelationshipActionGateDecision"
            )


def relationship_residual_fit_protocol_path() -> pathlib.Path:
    return _DEFAULT_PROTOCOL_PATH


def load_relationship_residual_fit_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipResidualFitProtocol:
    protocol_path = pathlib.Path(path or _DEFAULT_PROTOCOL_PATH)
    raw_bytes = protocol_path.read_bytes()
    raw = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_reject_duplicates)
    root = _exact_mapping(
        raw,
        expected={
            "schema_version",
            "protocol_name",
            "source_contract",
            "action_surface",
            "split_contract",
            "fit_recipe",
            "claim_boundary",
        },
        label="relationship residual protocol",
    )
    if _mapping_text(root, "schema_version") != (
        RELATIONSHIP_RESIDUAL_FIT_PROTOCOL_SCHEMA_VERSION
    ):
        raise ValueError("relationship residual protocol schema mismatch")
    if _mapping_text(root, "protocol_name") != (
        "relationship_residual_named_action_fit_v1"
    ):
        raise ValueError("relationship residual protocol name drift")
    source = _exact_mapping(
        root["source_contract"],
        expected={
            "forecast_type",
            "gate_decision_type",
            "required_named_readout",
            "required_gate_action",
            "required_gate_mode",
            "forbidden_gate_modes",
            "evaluator_only_required",
            "target_derivation",
            "condition_projection",
            "learning_source",
            "forbidden_input_fields",
        },
        label="relationship residual source contract",
    )
    _require_literal(
        source,
        {
            "forecast_type": "volvence_zero.social_cognition.PreferenceActionForecast",
            "gate_decision_type": (
                "lifeform_domain_emogpt.relationship_action_gate."
                "RelationshipActionGateDecision"
            ),
            "required_named_readout": "forecast.condition_readout",
            "required_gate_action": "steer",
            "required_gate_mode": "learned",
            "evaluator_only_required": False,
            "target_derivation": "forecast.recommended_action_id",
            "condition_projection": (
                RELATIONSHIP_RESIDUAL_CONDITION_PROJECTION_SCHEMA_VERSION
            ),
            "learning_source": (
                "pre_action_owner_forecast_and_pe_trained_gate_only"
            ),
        },
        label="relationship residual source constants",
    )
    action = _exact_mapping(
        root["action_surface"],
        expected={
            "ordered_action_ids",
            "strict_noop_action_id",
            "required_action_id_occurrences_in_public_action_text",
            "fit_scoring_surface",
            "strict_json_consumer_schema",
            "strict_json_serialization",
            "raw_model_strict_json_generation_proven",
            "typed_action_effect_only_until_physical_consumer",
        },
        label="relationship residual action surface",
    )
    _require_literal(
        action,
        {
            "fit_scoring_surface": "restricted_first_token_enum_nll",
            "strict_json_consumer_schema": "relationship_action_choice.schema.json",
            "strict_json_serialization": "canonical_object_with_only_action_id",
            "raw_model_strict_json_generation_proven": False,
            "typed_action_effect_only_until_physical_consumer": True,
        },
        label="relationship residual action constants",
    )
    split = _exact_mapping(
        root["split_contract"],
        expected={
            "subject_scope_disjoint",
            "minimum_train_rows_per_action",
            "minimum_heldout_rows_per_action",
            "selection_order",
        },
        label="relationship residual split contract",
    )
    _require_literal(
        split,
        {
            "subject_scope_disjoint": True,
            "selection_order": "lexicographic_row_id_no_metric_selection",
        },
        label="relationship residual split constants",
    )
    fit = _exact_mapping(
        root["fit_recipe"],
        expected={
            "model_id",
            "model_revision",
            "model_weights_sha256",
            "execution_assets_sha256",
            "injection_layer_index",
            "residual_width",
            "steering_rank",
            "conditional_executor_updates",
            "sensor_off_executor_updates",
            "executor_learning_rate",
            "reader_ridge_lambda",
            "batch_size",
            "seed",
            "control_norm_cap_ratio",
            "substrate_frozen",
            "free_bias_allowed",
            "zero_code_strict_noop_required",
            "matched_sensor_off_required",
        },
        label="relationship residual fit recipe",
    )
    _require_literal(
        fit,
        {
            "substrate_frozen": True,
            "free_bias_allowed": False,
            "zero_code_strict_noop_required": True,
            "matched_sensor_off_required": True,
        },
        label="relationship residual fit safety constants",
    )
    return RelationshipResidualFitProtocol(
        protocol_sha256=_sha256_bytes(_canonical_bytes(root)),
        action_ids=_text_tuple(action["ordered_action_ids"], "ordered_action_ids"),
        strict_noop_action_id=_mapping_text(action, "strict_noop_action_id"),
        required_action_id_occurrences=_mapping_int(
            action,
            "required_action_id_occurrences_in_public_action_text",
        ),
        forbidden_input_fields=_text_tuple(
            source["forbidden_input_fields"],
            "forbidden_input_fields",
        ),
        forbidden_gate_modes=tuple(
            RelationshipActionGateMode(value)
            for value in _text_tuple(
                source["forbidden_gate_modes"],
                "forbidden_gate_modes",
            )
        ),
        minimum_train_rows_per_action=_mapping_int(
            split,
            "minimum_train_rows_per_action",
        ),
        minimum_heldout_rows_per_action=_mapping_int(
            split,
            "minimum_heldout_rows_per_action",
        ),
        model_id=_mapping_text(fit, "model_id"),
        model_revision=_mapping_text(fit, "model_revision"),
        model_weights_sha256=_mapping_text(fit, "model_weights_sha256"),
        execution_assets_sha256=_mapping_text(fit, "execution_assets_sha256"),
        injection_layer_index=_mapping_int(fit, "injection_layer_index"),
        residual_width=_mapping_int(fit, "residual_width"),
        steering_rank=_mapping_int(fit, "steering_rank"),
        conditional_executor_updates=_mapping_int(
            fit,
            "conditional_executor_updates",
        ),
        sensor_off_executor_updates=_mapping_int(
            fit,
            "sensor_off_executor_updates",
        ),
        executor_learning_rate=_mapping_float(fit, "executor_learning_rate"),
        reader_ridge_lambda=_mapping_float(fit, "reader_ridge_lambda"),
        batch_size=_mapping_int(fit, "batch_size"),
        seed=_mapping_int(fit, "seed"),
        control_norm_cap_ratio=_mapping_float(fit, "control_norm_cap_ratio"),
        claim_boundary=_mapping_text(root, "claim_boundary"),
    )


def build_relationship_residual_fit_corpus(
    *,
    train_inputs: tuple[RelationshipResidualFitInput, ...],
    heldout_inputs: tuple[RelationshipResidualFitInput, ...],
    protocol: RelationshipResidualFitProtocol | None = None,
) -> NamedActionSteeringCorpus:
    """Freeze a corpus without ever accepting post-action/evaluator values."""

    active_protocol = protocol or load_relationship_residual_fit_protocol()
    train_rows = _build_split(
        train_inputs,
        protocol=active_protocol,
        split_name="train",
        minimum_per_action=active_protocol.minimum_train_rows_per_action,
    )
    heldout_rows = _build_split(
        heldout_inputs,
        protocol=active_protocol,
        split_name="heldout",
        minimum_per_action=active_protocol.minimum_heldout_rows_per_action,
    )
    return NamedActionSteeringCorpus(
        source_protocol_sha256=active_protocol.protocol_sha256,
        action_ids=active_protocol.action_ids,
        class_labels=active_protocol.action_ids,
        train_rows=train_rows,
        heldout_rows=heldout_rows,
        description=(
            "Relationship-domain residual fit rows derived only from frozen "
            "pre-action preference-owner forecasts and non-oracle gate decisions."
        ),
    )


def _build_split(
    inputs: tuple[RelationshipResidualFitInput, ...],
    *,
    protocol: RelationshipResidualFitProtocol,
    split_name: str,
    minimum_per_action: int,
) -> tuple[NamedActionSteeringRow, ...]:
    if not isinstance(inputs, tuple) or not inputs:
        raise ValueError(f"relationship residual {split_name} inputs must be non-empty")
    if not all(isinstance(item, RelationshipResidualFitInput) for item in inputs):
        raise TypeError(
            f"relationship residual {split_name} inputs must contain typed values"
        )
    rows = tuple(
        _build_row(item, protocol=protocol)
        for item in sorted(inputs, key=lambda value: value.row_id)
    )
    counts = {
        action_id: sum(row.target_action_id == action_id for row in rows)
        for action_id in protocol.action_ids
    }
    if any(value < minimum_per_action for value in counts.values()):
        raise ValueError(
            f"relationship residual {split_name} split misses its frozen per-action "
            f"minimum: {counts!r}"
        )
    return rows


def _build_row(
    value: RelationshipResidualFitInput,
    *,
    protocol: RelationshipResidualFitProtocol,
) -> NamedActionSteeringRow:
    forecast = value.forecast
    gate = value.gate_decision
    readout = forecast.condition_readout
    if readout is None:
        raise ValueError("relationship residual fit requires a named condition readout")
    if gate.mode in protocol.forbidden_gate_modes or gate.evaluator_only:
        raise ValueError("relationship residual fit rejects oracle/evaluator gate decisions")
    if gate.mode is not RelationshipActionGateMode.LEARNED or gate.update_count < 1:
        raise ValueError(
            "relationship residual fit requires a PE-trained learned gate decision"
        )
    if gate.gate_action is not RelationshipGateAction.STEER:
        raise ValueError(
            "relationship residual fit only trains directional executor rows; "
            "noop remains a strict zero-delta gate arm"
        )
    if (
        gate.forecast_id != forecast.forecast_id
        or gate.decision_id != forecast.decision_id
        or gate.recommended_action_id != forecast.recommended_action_id
        or gate.selected_action_id != forecast.recommended_action_id
    ):
        raise ValueError("relationship residual forecast/gate lineage mismatch")
    if forecast.recommended_action_id not in protocol.action_ids:
        raise ValueError(
            "relationship residual target must be a directional frozen action"
        )
    candidate_ids = tuple(item.action_id for item in forecast.candidate_predictions)
    expected_surface = tuple(action.value for action in RELATIONSHIP_ACTIONS)
    if candidate_ids != expected_surface:
        raise ValueError("relationship residual forecast action surface drift")
    _validate_public_action_text(value.public_action_text, protocol=protocol)

    condition_projection = _condition_projection(forecast=forecast, gate=gate)
    condition_text = _canonical_bytes(condition_projection).decode("utf-8")
    return NamedActionSteeringRow(
        row_id=value.row_id,
        subject_scope=value.subject_scope,
        action_text=value.public_action_text,
        condition_text=condition_text,
        condition_label=forecast.recommended_action_id,
        target_action_id=forecast.recommended_action_id,
        source_condition_lineage_sha256=_sha256_bytes(
            _canonical_bytes(condition_projection)
        ),
    )


def _condition_projection(
    *,
    forecast: PreferenceActionForecast,
    gate: RelationshipActionGateDecision,
) -> dict[str, object]:
    readout = forecast.condition_readout
    if readout is None:
        raise ValueError("relationship residual condition projection requires readout")
    return {
        "schema_version": RELATIONSHIP_RESIDUAL_CONDITION_PROJECTION_SCHEMA_VERSION,
        "forecast": {
            "forecast_id": forecast.forecast_id,
            "decision_id": forecast.decision_id,
            "interlocutor_id_sha256": _sha256_bytes(
                forecast.interlocutor_id.encode("utf-8")
            ),
            "recommended_action_id": forecast.recommended_action_id,
            "confidence": forecast.confidence,
            "issued_turn": forecast.issued_turn,
            "session_scope_sha256": _sha256_bytes(
                forecast.session_scope.encode("utf-8")
            ),
            "source_record_ids_sha256": _sha256_bytes(
                _canonical_bytes(list(forecast.source_record_ids))
            ),
            "evidence_sha256": _sha256_bytes(
                _canonical_bytes(list(forecast.evidence))
            ),
            "candidate_predictions": [
                {
                    "action_id": candidate.action_id,
                    "outcomes": [
                        {
                            "outcome_id": outcome.outcome_id,
                            "probability": outcome.probability,
                        }
                        for outcome in candidate.outcomes
                    ],
                }
                for candidate in forecast.candidate_predictions
            ],
            "condition_readout": {
                "condition_label": readout.condition_label,
                "confidence": readout.confidence,
                "normalized_margin": readout.normalized_margin,
                "candidate_scores": [list(item) for item in readout.candidate_scores],
                "reader_artifact_id": readout.reader_artifact_id,
                "source_observation_sha256": readout.source_observation_sha256,
            },
        },
        "gate": {
            "decision_id": gate.decision_id,
            "forecast_id": gate.forecast_id,
            "gate_action": gate.gate_action.value,
            "selected_action_id": gate.selected_action_id,
            "recommended_action_id": gate.recommended_action_id,
            "steer_probability": gate.steer_probability,
            "features": list(gate.features),
            "mode": gate.mode.value,
            "artifact_id": gate.artifact_id,
            "artifact_version": gate.artifact_version,
            "update_count": gate.update_count,
            "evidence_refs_sha256": _sha256_bytes(
                _canonical_bytes(list(gate.evidence_refs))
            ),
            "rationale_codes": list(gate.rationale_codes),
            "evaluator_only": gate.evaluator_only,
        },
        "input_firewall": {
            "pre_action_only": True,
            "evaluation_present": False,
            "judge_present": False,
            "observed_or_future_outcome_present": False,
            "reward_present": False,
            "credit_record_present": False,
        },
    }


def _validate_public_action_text(
    value: str,
    *,
    protocol: RelationshipResidualFitProtocol,
) -> None:
    folded = value.casefold()
    leaked = tuple(
        field
        for field in protocol.forbidden_input_fields
        if _contains_field_marker(folded, field.casefold())
    )
    if leaked:
        raise ValueError(
            f"relationship residual public action text contains forbidden fields: {leaked!r}"
        )
    for action_id in protocol.action_ids:
        if value.count(action_id) != protocol.required_action_id_occurrences:
            raise ValueError(
                "relationship residual public action text must enumerate every "
                "directional action symmetrically"
            )
    if protocol.strict_noop_action_id in value:
        raise ValueError(
            "relationship residual directional fit prompt must not train the "
            "strict-noop action"
        )


def _contains_field_marker(text: str, field: str) -> bool:
    """Detect serialized private fields without rejecting ordinary prose words."""

    return any(
        marker in text
        for marker in (
            f'"{field}"',
            f"'{field}'",
            f"{field}=",
            f"{field}:",
        )
    )


def _require_literal(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(f"{label}.{key} drifted")


def _exact_mapping(
    value: object,
    *,
    expected: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _mapping_text(value: Mapping[str, object], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _mapping_int(value: Mapping[str, object], key: str) -> int:
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, int):
        raise TypeError(f"{key} must be an integer")
    return item


def _mapping_float(value: Mapping[str, object], key: str) -> float:
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(item)


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{label} must be a list of strings")
    result = tuple(value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{label} must be non-empty and unique")
    return result


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _require_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = (
    "RELATIONSHIP_RESIDUAL_CONDITION_PROJECTION_SCHEMA_VERSION",
    "RELATIONSHIP_RESIDUAL_FIT_PROTOCOL_SCHEMA_VERSION",
    "RelationshipResidualFitInput",
    "RelationshipResidualFitProtocol",
    "build_relationship_residual_fit_corpus",
    "load_relationship_residual_fit_protocol",
    "relationship_residual_fit_protocol_path",
)
