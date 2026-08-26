"""PE-downstream relationship-action credit relative to a frozen noop baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

from companion_standard.canonical import stable_hash
from volvence_zero.credit.gate import (
    CreditRecord,
    derive_preference_action_forecast_credit_records,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.memory import Track
from volvence_zero.social.tom import (
    PREFERENCE_ACTION_RELATIONSHIP_UTILITY_SURFACE_ID,
    preference_action_forecast_expected_utility,
    settle_preference_action_forecast,
    social_prediction_error_from_preference_action_forecast_settlement,
)
from volvence_zero.social_cognition import (
    PreferenceActionForecast,
    PreferenceActionForecastSettlement,
    RelationshipConditionReadout,
    SocialActionCandidatePrediction,
    SocialActionOutcomeProbability,
    SocialPredictionError,
    SocialPredictionKind,
    SocialPredictionOutcome,
    SocialScopeKind,
)


RELATIONSHIP_ACTION_COMMON_BASELINE_CREDIT_SCHEMA_VERSION = (
    "relationship-action-common-baseline-credit.v1"
)
RELATIONSHIP_ACTION_COMMON_BASELINE_ACTION_ID = "neutral_noop"
RELATIONSHIP_ACTION_COMMON_BASELINE_FORMULA_ID = (
    "parent-action-pe-plus-frozen-noop-adjustment.v1"
)
_RECORD_ID_PREFIX = "relationship-action-common-baseline-credit-sha256:"


@dataclass(frozen=True)
class RelationshipActionCommonBaselineCredit:
    """Replay-checked common-noop-baseline credit, not a causal-effect claim."""

    record_id: str
    forecast: PreferenceActionForecast
    external_evidence: DialogueExternalOutcomeEvidence
    settlement: PreferenceActionForecastSettlement
    social_prediction_error: SocialPredictionError
    parent_action_credit: CreditRecord
    forecast_sha256: str
    external_evidence_sha256: str
    settlement_sha256: str
    social_prediction_error_sha256: str
    parent_action_credit_sha256: str
    common_baseline_expected_utility_hex: str
    delivered_expected_utility_hex: str
    observed_utility_hex: str
    evidence_confidence_hex: str
    parent_action_credit_value_hex: str
    common_baseline_adjustment_hex: str
    credit_value_hex: str
    common_baseline_action_id: str = RELATIONSHIP_ACTION_COMMON_BASELINE_ACTION_ID
    utility_surface_id: str = PREFERENCE_ACTION_RELATIONSHIP_UTILITY_SURFACE_ID
    formula_id: str = RELATIONSHIP_ACTION_COMMON_BASELINE_FORMULA_ID
    schema_version: str = RELATIONSHIP_ACTION_COMMON_BASELINE_CREDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_ACTION_COMMON_BASELINE_CREDIT_SCHEMA_VERSION:
            raise ValueError("common-baseline credit schema_version mismatch")
        if self.utility_surface_id != PREFERENCE_ACTION_RELATIONSHIP_UTILITY_SURFACE_ID:
            raise ValueError("common-baseline credit utility_surface_id mismatch")
        if self.formula_id != RELATIONSHIP_ACTION_COMMON_BASELINE_FORMULA_ID:
            raise ValueError("common-baseline credit formula_id mismatch")
        if self.common_baseline_action_id != RELATIONSHIP_ACTION_COMMON_BASELINE_ACTION_ID:
            raise ValueError("common-baseline credit action must be neutral_noop")
        expected = _expected_fields(
            forecast=self.forecast,
            external_evidence=self.external_evidence,
            settlement=self.settlement,
            social_prediction_error=self.social_prediction_error,
            parent_action_credit=self.parent_action_credit,
        )
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"common-baseline credit {field_name} mismatch")
        expected_record_id = _RECORD_ID_PREFIX + stable_hash(self._core_payload())
        if self.record_id != expected_record_id:
            raise ValueError("common-baseline credit record_id mismatch")

    @property
    def common_baseline_expected_utility(self) -> float:
        return _float_from_hex(
            self.common_baseline_expected_utility_hex,
            "common_baseline_expected_utility_hex",
        )

    @property
    def delivered_expected_utility(self) -> float:
        return _float_from_hex(
            self.delivered_expected_utility_hex,
            "delivered_expected_utility_hex",
        )

    @property
    def observed_utility(self) -> float:
        return _float_from_hex(self.observed_utility_hex, "observed_utility_hex")

    @property
    def evidence_confidence(self) -> float:
        return _float_from_hex(
            self.evidence_confidence_hex,
            "evidence_confidence_hex",
        )

    @property
    def parent_action_credit_value(self) -> float:
        return _float_from_hex(
            self.parent_action_credit_value_hex,
            "parent_action_credit_value_hex",
        )

    @property
    def common_baseline_adjustment(self) -> float:
        return _float_from_hex(
            self.common_baseline_adjustment_hex,
            "common_baseline_adjustment_hex",
        )

    @property
    def credit_value(self) -> float:
        return _float_from_hex(self.credit_value_hex, "credit_value_hex")

    @classmethod
    def create(
        cls,
        *,
        forecast: PreferenceActionForecast,
        external_evidence: DialogueExternalOutcomeEvidence,
        settlement: PreferenceActionForecastSettlement,
        social_prediction_error: SocialPredictionError,
        parent_action_credit: CreditRecord,
    ) -> "RelationshipActionCommonBaselineCredit":
        fields = _expected_fields(
            forecast=forecast,
            external_evidence=external_evidence,
            settlement=settlement,
            social_prediction_error=social_prediction_error,
            parent_action_credit=parent_action_credit,
        )
        core_payload = _core_payload(
            fields,
            forecast=forecast,
            external_evidence=external_evidence,
            settlement=settlement,
            social_prediction_error=social_prediction_error,
            parent_action_credit=parent_action_credit,
        )
        return cls(
            record_id=_RECORD_ID_PREFIX + stable_hash(core_payload),
            forecast=forecast,
            external_evidence=external_evidence,
            settlement=settlement,
            social_prediction_error=social_prediction_error,
            parent_action_credit=parent_action_credit,
            **fields,
        )

    def to_payload(self) -> dict[str, object]:
        """Return a compact audit projection; parents remain separate artifacts."""

        return {"record_id": self.record_id, **self._core_payload()}

    def _core_payload(self) -> dict[str, object]:
        fields = {
            field_name: getattr(self, field_name)
            for field_name in _DERIVED_FIELD_NAMES
        }
        return _core_payload(
            fields,
            forecast=self.forecast,
            external_evidence=self.external_evidence,
            settlement=self.settlement,
            social_prediction_error=self.social_prediction_error,
            parent_action_credit=self.parent_action_credit,
        )


_DERIVED_FIELD_NAMES = (
    "forecast_sha256",
    "external_evidence_sha256",
    "settlement_sha256",
    "social_prediction_error_sha256",
    "parent_action_credit_sha256",
    "common_baseline_expected_utility_hex",
    "delivered_expected_utility_hex",
    "observed_utility_hex",
    "evidence_confidence_hex",
    "parent_action_credit_value_hex",
    "common_baseline_adjustment_hex",
    "credit_value_hex",
)


def derive_preference_action_common_baseline_credit_records(
    *,
    forecasts: tuple[PreferenceActionForecast, ...],
    external_evidence: tuple[DialogueExternalOutcomeEvidence, ...],
    settlements: tuple[PreferenceActionForecastSettlement, ...],
    social_errors: tuple[SocialPredictionError, ...],
    settled_at_turn: int,
    timestamp_ms: int,
) -> tuple[RelationshipActionCommonBaselineCredit, ...]:
    """Derive current-turn common-baseline records from the exact PE chain."""

    for field_name, values, expected_type in (
        ("forecasts", forecasts, PreferenceActionForecast),
        ("external_evidence", external_evidence, DialogueExternalOutcomeEvidence),
        ("settlements", settlements, PreferenceActionForecastSettlement),
        ("social_errors", social_errors, SocialPredictionError),
    ):
        _require_exact_tuple(field_name, values, expected_type)
    _require_non_negative_int("settled_at_turn", settled_at_turn)
    _require_non_negative_int("timestamp_ms", timestamp_ms)

    forecasts_by_id = _unique_by_id(forecasts, "forecast_id", "forecasts")
    evidence_by_id = _unique_by_id(
        external_evidence,
        "evidence_id",
        "external_evidence",
    )
    errors_by_id = _unique_by_id(social_errors, "error_id", "social_errors")
    parents = derive_preference_action_forecast_credit_records(
        settlements=settlements,
        social_errors=social_errors,
        settled_at_turn=settled_at_turn,
        timestamp_ms=timestamp_ms,
    )
    parents_by_id = _unique_by_id(parents, "record_id", "parent action credits")

    derived: list[RelationshipActionCommonBaselineCredit] = []
    for settlement in settlements:
        if settlement.observed_turn != settled_at_turn:
            continue
        derived.append(
            RelationshipActionCommonBaselineCredit.create(
                forecast=_required_member(
                    forecasts_by_id,
                    settlement.forecast_id,
                    "frozen forecast",
                ),
                external_evidence=_required_member(
                    evidence_by_id,
                    settlement.source_evidence_id,
                    "external evidence",
                ),
                settlement=settlement,
                social_prediction_error=_required_member(
                    errors_by_id,
                    f"social-pe:{settlement.settlement_id}",
                    "social PE",
                ),
                parent_action_credit=_required_member(
                    parents_by_id,
                    f"relationship-action-pe-credit:{settlement.settlement_id}",
                    "parent action credit",
                ),
            )
        )
    return tuple(derived)


def _expected_fields(
    *,
    forecast: PreferenceActionForecast,
    external_evidence: DialogueExternalOutcomeEvidence,
    settlement: PreferenceActionForecastSettlement,
    social_prediction_error: SocialPredictionError,
    parent_action_credit: CreditRecord,
) -> dict[str, str]:
    _validate_parent_chain(
        forecast=forecast,
        external_evidence=external_evidence,
        settlement=settlement,
        social_prediction_error=social_prediction_error,
        parent_action_credit=parent_action_credit,
    )
    baseline = preference_action_forecast_expected_utility(
        forecast=forecast,
        action_id=RELATIONSHIP_ACTION_COMMON_BASELINE_ACTION_ID,
    )
    delivered = preference_action_forecast_expected_utility(
        forecast=forecast,
        action_id=external_evidence.action_id,
    )
    confidence = external_evidence.confidence
    adjustment = confidence * (delivered - baseline) / 2.0
    direct_credit = confidence * (settlement.observed_utility - baseline) / 2.0
    if not math.isclose(
        math.fsum((parent_action_credit.credit_value, adjustment)),
        direct_credit,
        abs_tol=1e-12,
    ):
        raise RuntimeError("common-baseline credit algebra failed to close")
    return {
        "forecast_sha256": stable_hash(forecast),
        "external_evidence_sha256": stable_hash(external_evidence),
        "settlement_sha256": stable_hash(settlement),
        "social_prediction_error_sha256": stable_hash(social_prediction_error),
        "parent_action_credit_sha256": stable_hash(parent_action_credit),
        "common_baseline_expected_utility_hex": baseline.hex(),
        "delivered_expected_utility_hex": delivered.hex(),
        "observed_utility_hex": settlement.observed_utility.hex(),
        "evidence_confidence_hex": confidence.hex(),
        "parent_action_credit_value_hex": parent_action_credit.credit_value.hex(),
        "common_baseline_adjustment_hex": adjustment.hex(),
        "credit_value_hex": direct_credit.hex(),
    }


def _validate_parent_chain(
    *,
    forecast: PreferenceActionForecast,
    external_evidence: DialogueExternalOutcomeEvidence,
    settlement: PreferenceActionForecastSettlement,
    social_prediction_error: SocialPredictionError,
    parent_action_credit: CreditRecord,
) -> None:
    for field_name, value, expected_type in (
        ("forecast", forecast, PreferenceActionForecast),
        ("external_evidence", external_evidence, DialogueExternalOutcomeEvidence),
        ("settlement", settlement, PreferenceActionForecastSettlement),
        ("social_prediction_error", social_prediction_error, SocialPredictionError),
        ("parent_action_credit", parent_action_credit, CreditRecord),
    ):
        if type(value) is not expected_type:
            raise TypeError(f"{field_name} has an invalid type")
    _require_immutable_parent_shapes(
        forecast,
        social_prediction_error,
        parent_action_credit,
    )
    if type(external_evidence.kind) is not DialogueExternalOutcomeKind:
        raise TypeError("external_evidence.kind must be DialogueExternalOutcomeKind")
    if type(external_evidence.source) is not DialogueExternalOutcomeEvidenceSource:
        raise TypeError(
            "external_evidence.source must be DialogueExternalOutcomeEvidenceSource"
        )
    if type(settlement.outcome) is not SocialPredictionOutcome:
        raise TypeError("settlement.outcome must be SocialPredictionOutcome")
    if type(social_prediction_error.kind) is not SocialPredictionKind:
        raise TypeError("social_prediction_error.kind must be SocialPredictionKind")
    if type(social_prediction_error.outcome) is not SocialPredictionOutcome:
        raise TypeError("social_prediction_error.outcome must be SocialPredictionOutcome")
    if type(social_prediction_error.scope_kind) is not SocialScopeKind:
        raise TypeError("social_prediction_error.scope_kind must be SocialScopeKind")
    if type(parent_action_credit.track) is not Track:
        raise TypeError("parent_action_credit.track must be Track")
    _require_non_negative_int(
        "parent_action_credit.timestamp_ms",
        parent_action_credit.timestamp_ms,
    )
    for field_name, value in (
        ("external_evidence.confidence", external_evidence.confidence),
        ("settlement.predicted_probability", settlement.predicted_probability),
        ("settlement.negative_log_likelihood", settlement.negative_log_likelihood),
        ("settlement.magnitude", settlement.magnitude),
        ("settlement.evidence_confidence", settlement.evidence_confidence),
        ("settlement.expected_utility", settlement.expected_utility),
        ("settlement.observed_utility", settlement.observed_utility),
        (
            "settlement.signed_utility_prediction_error",
            settlement.signed_utility_prediction_error,
        ),
        ("social_prediction_error.magnitude", social_prediction_error.magnitude),
        ("parent_action_credit.credit_value", parent_action_credit.credit_value),
    ):
        _require_exact_float(field_name, value)
    if external_evidence.source is not DialogueExternalOutcomeEvidenceSource.ENVIRONMENT:
        raise ValueError("common-baseline credit requires ENVIRONMENT evidence")
    if external_evidence.confidence != 1.0:
        raise ValueError("common-baseline credit requires environment confidence 1.0")
    if settle_preference_action_forecast(
        forecast=forecast,
        evidence=external_evidence,
    ) != settlement:
        raise ValueError("common-baseline credit settlement replay mismatch")
    if (
        social_prediction_error_from_preference_action_forecast_settlement(
            settlement
        )
        != social_prediction_error
    ):
        raise ValueError("common-baseline credit social PE replay mismatch")
    expected_parent = derive_preference_action_forecast_credit_records(
        settlements=(settlement,),
        social_errors=(social_prediction_error,),
        settled_at_turn=settlement.observed_turn,
        timestamp_ms=parent_action_credit.timestamp_ms,
    )
    if expected_parent != (parent_action_credit,):
        raise ValueError("common-baseline credit parent action-PE replay mismatch")


def _require_immutable_parent_shapes(
    forecast: PreferenceActionForecast,
    social_error: SocialPredictionError,
    parent_credit: CreditRecord,
) -> None:
    condition_readout = forecast.condition_readout
    if condition_readout is not None:
        if type(condition_readout) is not RelationshipConditionReadout:
            raise TypeError(
                "forecast.condition_readout must be an exact "
                "RelationshipConditionReadout"
            )
        for field_name, value in (
            ("condition_label", condition_readout.condition_label),
            ("reader_artifact_id", condition_readout.reader_artifact_id),
            (
                "source_observation_sha256",
                condition_readout.source_observation_sha256,
            ),
        ):
            if type(value) is not str:
                raise TypeError(
                    f"forecast.condition_readout.{field_name} must be an exact str"
                )
        _require_exact_float(
            "forecast.condition_readout.confidence",
            condition_readout.confidence,
        )
        _require_exact_float(
            "forecast.condition_readout.normalized_margin",
            condition_readout.normalized_margin,
        )
        if type(condition_readout.candidate_scores) is not tuple:
            raise TypeError(
                "forecast.condition_readout.candidate_scores must be an exact tuple"
            )
        for index, pair in enumerate(condition_readout.candidate_scores):
            if type(pair) is not tuple:
                raise TypeError(
                    "forecast.condition_readout.candidate_scores"
                    f"[{index}] must be an exact tuple"
                )
            if len(pair) != 2:
                raise ValueError(
                    "forecast.condition_readout candidate score pairs require two "
                    "entries"
                )
            label, score = pair
            if type(label) is not str:
                raise TypeError(
                    "forecast.condition_readout.candidate_scores"
                    f"[{index}].label must be an exact str"
                )
            _require_exact_float(
                "forecast.condition_readout.candidate_scores"
                f"[{index}].score",
                score,
            )
    _require_exact_tuple(
        "forecast.candidate_predictions",
        forecast.candidate_predictions,
        SocialActionCandidatePrediction,
    )
    for index, candidate in enumerate(forecast.candidate_predictions):
        _require_exact_tuple(
            f"forecast.candidate_predictions[{index}].outcomes",
            candidate.outcomes,
            SocialActionOutcomeProbability,
        )
    _require_exact_tuple("forecast.source_record_ids", forecast.source_record_ids, str)
    _require_exact_tuple("forecast.evidence", forecast.evidence, str)
    _require_exact_tuple("social_prediction_error.evidence", social_error.evidence, str)
    _require_exact_tuple(
        "parent_action_credit.conditioning_bank_set",
        parent_credit.conditioning_bank_set,
        str,
    )
    _require_exact_tuple(
        "parent_action_credit.conditioning_bank_fingerprints",
        parent_credit.conditioning_bank_fingerprints,
        tuple,
    )
    for index, pair in enumerate(parent_credit.conditioning_bank_fingerprints):
        _require_exact_tuple(
            f"parent_action_credit.conditioning_bank_fingerprints[{index}]",
            pair,
            str,
        )
        if len(pair) != 2:
            raise ValueError("conditioning bank fingerprint pairs require two entries")


def _core_payload(
    fields: dict[str, str],
    *,
    forecast: PreferenceActionForecast,
    external_evidence: DialogueExternalOutcomeEvidence,
    settlement: PreferenceActionForecastSettlement,
    social_prediction_error: SocialPredictionError,
    parent_action_credit: CreditRecord,
) -> dict[str, object]:
    return {
        "schema_version": RELATIONSHIP_ACTION_COMMON_BASELINE_CREDIT_SCHEMA_VERSION,
        "utility_surface_id": PREFERENCE_ACTION_RELATIONSHIP_UTILITY_SURFACE_ID,
        "formula_id": RELATIONSHIP_ACTION_COMMON_BASELINE_FORMULA_ID,
        "common_baseline_action_id": RELATIONSHIP_ACTION_COMMON_BASELINE_ACTION_ID,
        "forecast_id": forecast.forecast_id,
        "external_evidence_id": external_evidence.evidence_id,
        "settlement_id": settlement.settlement_id,
        "social_prediction_error_id": social_prediction_error.error_id,
        "parent_action_credit_id": parent_action_credit.record_id,
        "delivered_action_id": external_evidence.action_id,
        "observed_outcome_id": settlement.observed_outcome_id,
        "timestamp_ms": parent_action_credit.timestamp_ms,
        **fields,
    }


def _require_exact_tuple(
    field_name: str,
    values: object,
    expected_item_type: type[object],
) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{field_name} must be an exact tuple")
    if any(type(item) is not expected_item_type for item in values):
        raise TypeError(f"{field_name} contains an invalid item type")


def _require_exact_float(field_name: str, value: object) -> None:
    if type(value) is not float:
        raise TypeError(f"{field_name} must be an exact float")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _require_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _float_from_hex(value: str, field_name: str) -> float:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be canonical float hex text")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be canonical float hex text") from exc
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise ValueError(f"{field_name} must use finite canonical float.hex()")
    return parsed


def _unique_by_id(
    values: tuple[object, ...],
    attribute: str,
    field_name: str,
) -> dict[str, object]:
    by_id = {getattr(item, attribute): item for item in values}
    if len(by_id) != len(values):
        raise ValueError(f"{field_name} must have unique {attribute} values")
    return by_id


def _required_member(
    by_id: dict[str, object],
    item_id: str,
    field_name: str,
) -> object:
    try:
        return by_id[item_id]
    except KeyError as exc:
        raise ValueError(f"common-baseline credit requires {field_name} {item_id!r}") from exc


__all__ = [
    "RELATIONSHIP_ACTION_COMMON_BASELINE_ACTION_ID",
    "RELATIONSHIP_ACTION_COMMON_BASELINE_CREDIT_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_COMMON_BASELINE_FORMULA_ID",
    "RelationshipActionCommonBaselineCredit",
    "derive_preference_action_common_baseline_credit_records",
]
