"""Closed-alpha relationship-intelligence product orchestration.

This service layer composes existing owners without becoming a second owner:

* ``preference_about_other`` publishes the pre-action forecast through the
  Brain facade;
* the bounded vertical gate chooses ``noop`` or ``steer``;
* ``self_temporal`` records the typed advisory under SHADOW;
* explicit outcomes remain collection-only until a verified real-user typing
  qualification artifact passes, after which they enter the sole
  ``dialogue_external_outcome`` channel with an exact forecast join.

Operational alpha evidence and opt-in offline training candidates are written
to different roots and different schemas.  Neither store contains raw dialogue
text or plaintext identity/session keys.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
    RelationshipAction,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGate,
    RelationshipActionGateCheckpoint,
    RelationshipActionGateDecision,
    RelationshipActionGateMode,
    RelationshipGateAction,
    temporal_action_advisory_from_gate_decision,
)
from lifeform_domain_emogpt.relationship_forecast import (
    BoundedRelationshipPreferenceForecastRuntime,
)
from lifeform_service.relationship_outcome_typing import (
    RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION,
    RELATIONSHIP_OUTCOME_UNKNOWN,
    RelationshipOutcomeTypingRuntime,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.semantic_state import BoundaryConsentSnapshot
from volvence_zero.social import PreferenceActionForecastRequest
from volvence_zero.social_cognition import PreferenceActionForecast
from volvence_zero.temporal import (
    TemporalAbstractionSnapshot,
    TemporalActionAdvisoryStatus,
)


RELATIONSHIP_ALPHA_AUDIT_SCHEMA_VERSION = "relationship-alpha-action-audit.v1"
RELATIONSHIP_ALPHA_OUTCOME_SCHEMA_VERSION = "relationship-alpha-outcome.v1"
RELATIONSHIP_TRAINING_CANDIDATE_SCHEMA_VERSION = (
    "relationship-alpha-training-candidate.v1"
)
RELATIONSHIP_TYPING_QUALIFICATION_SCHEMA_VERSION = (
    "relationship-outcome-typing-qualification.v1"
)
RELATIONSHIP_TYPING_COLLECTION_ONLY = "collection_only"
RELATIONSHIP_TYPING_PASSED = "passed"
RELATIONSHIP_ACTION_EXPOSURE_BASELINE_NOOP = "baseline_noop_exposed"
RELATIONSHIP_ACTION_EXPOSURE_SHADOW_COUNTERFACTUAL = "shadow_counterfactual"
RELATIONSHIP_RUNTIME_OUTCOME_SUBMITTED = "submitted"
RELATIONSHIP_RUNTIME_OUTCOME_TYPING_BLOCKED = "typing_not_qualified"
RELATIONSHIP_RUNTIME_OUTCOME_REVIEW_BLOCKED = "unknown_or_review_required"
RELATIONSHIP_RUNTIME_OUTCOME_EXPOSURE_BLOCKED = "action_not_exposed"
_TYPED_RELATIONSHIP_OUTCOMES = frozenset(
    outcome.value for outcome in RELATIONSHIP_OUTCOMES
)


@dataclass(frozen=True)
class RelationshipOutcomeTypingQualification:
    qualification_id: str
    typing_method: str
    structured_runtime_id: str
    structured_output_schema_id: str
    rater_count: int
    independent_rater_artifact_ids: tuple[str, ...]
    sample_count: int
    hidden_labels: bool
    majority_agreement: float
    typing_anchor_agreement: float
    preregistered_anchor_threshold: float
    validation_anchor_only: bool
    learning_use_authorized: bool
    keyword_or_regex_routing_used: bool
    unknown_outcome_supported: bool
    reviewed_by: tuple[str, ...]
    qualified_at_iso: str
    schema_version: str = RELATIONSHIP_TYPING_QUALIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name, value in (
            ("qualification_id", self.qualification_id),
            ("typing_method", self.typing_method),
            ("structured_runtime_id", self.structured_runtime_id),
            ("structured_output_schema_id", self.structured_output_schema_id),
            ("qualified_at_iso", self.qualified_at_iso),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be non-empty")
        parsed = datetime.fromisoformat(self.qualified_at_iso)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("qualified_at_iso must include a timezone")
        if self.rater_count < 1 or self.sample_count < 1:
            raise ValueError("rater_count/sample_count must be positive")
        if self.typing_method != "llm_structured_output":
            raise ValueError("typing_method must be llm_structured_output")
        if (
            len(self.independent_rater_artifact_ids) != self.rater_count
            or len(set(self.independent_rater_artifact_ids)) != self.rater_count
            or any(not item for item in self.independent_rater_artifact_ids)
        ):
            raise ValueError(
                "independent_rater_artifact_ids must contain one unique "
                "artifact per rater"
            )
        for field_name, value in (
            ("majority_agreement", self.majority_agreement),
            ("typing_anchor_agreement", self.typing_anchor_agreement),
            (
                "preregistered_anchor_threshold",
                self.preregistered_anchor_threshold,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.preregistered_anchor_threshold <= 0.0:
            raise ValueError(
                "preregistered_anchor_threshold must be greater than zero"
            )
        if not self.reviewed_by or any(not item for item in self.reviewed_by):
            raise ValueError("reviewed_by must be non-empty")
        if len(set(self.reviewed_by)) != len(self.reviewed_by):
            raise ValueError("reviewed_by must be unique")
        if self.schema_version != RELATIONSHIP_TYPING_QUALIFICATION_SCHEMA_VERSION:
            raise ValueError("relationship outcome typing qualification schema mismatch")

    @property
    def passed(self) -> bool:
        return bool(
            self.rater_count >= 3
            and self.structured_output_schema_id
            == RELATIONSHIP_OUTCOME_TYPING_RESULT_SCHEMA_VERSION
            and self.hidden_labels
            and self.majority_agreement >= 0.80
            and self.typing_anchor_agreement
            >= self.preregistered_anchor_threshold
            and self.validation_anchor_only
            and not self.learning_use_authorized
            and not self.keyword_or_regex_routing_used
            and self.unknown_outcome_supported
        )

    def to_json(self) -> dict[str, object]:
        return _with_hash(
            {
                "schema_version": self.schema_version,
                "qualification_id": self.qualification_id,
                "typing_method": self.typing_method,
                "structured_runtime_id": self.structured_runtime_id,
                "structured_output_schema_id": self.structured_output_schema_id,
                "rater_count": self.rater_count,
                "independent_rater_artifact_ids": list(
                    self.independent_rater_artifact_ids
                ),
                "sample_count": self.sample_count,
                "hidden_labels": self.hidden_labels,
                "majority_agreement": self.majority_agreement,
                "typing_anchor_agreement": self.typing_anchor_agreement,
                "preregistered_anchor_threshold": (
                    self.preregistered_anchor_threshold
                ),
                "validation_anchor_only": self.validation_anchor_only,
                "learning_use_authorized": self.learning_use_authorized,
                "keyword_or_regex_routing_used": (
                    self.keyword_or_regex_routing_used
                ),
                "unknown_outcome_supported": self.unknown_outcome_supported,
                "reviewed_by": list(self.reviewed_by),
                "qualified_at_iso": self.qualified_at_iso,
                "passed": self.passed,
            }
        )

    @property
    def content_sha256(self) -> str:
        value = self.to_json()["content_sha256"]
        assert isinstance(value, str)
        return value


def load_relationship_outcome_typing_qualification(
    path: Path,
) -> RelationshipOutcomeTypingQualification:
    raw = _read_verified(path.expanduser().resolve())
    expected_keys = {
        "schema_version",
        "qualification_id",
        "typing_method",
        "structured_runtime_id",
        "structured_output_schema_id",
        "rater_count",
        "independent_rater_artifact_ids",
        "sample_count",
        "hidden_labels",
        "majority_agreement",
        "typing_anchor_agreement",
        "preregistered_anchor_threshold",
        "validation_anchor_only",
        "learning_use_authorized",
        "keyword_or_regex_routing_used",
        "unknown_outcome_supported",
        "reviewed_by",
        "qualified_at_iso",
        "passed",
        "content_sha256",
    }
    if set(raw) != expected_keys:
        raise ValueError("typing qualification fields do not match frozen schema")
    qualification = RelationshipOutcomeTypingQualification(
        qualification_id=_required_text(raw, "qualification_id"),
        typing_method=_required_text(raw, "typing_method"),
        structured_runtime_id=_required_text(raw, "structured_runtime_id"),
        structured_output_schema_id=_required_text(
            raw,
            "structured_output_schema_id",
        ),
        rater_count=_required_int(raw, "rater_count"),
        independent_rater_artifact_ids=_required_text_tuple(
            raw,
            "independent_rater_artifact_ids",
        ),
        sample_count=_required_int(raw, "sample_count"),
        hidden_labels=_required_bool(raw, "hidden_labels"),
        majority_agreement=_required_float(raw, "majority_agreement"),
        typing_anchor_agreement=_required_float(raw, "typing_anchor_agreement"),
        preregistered_anchor_threshold=_required_float(
            raw,
            "preregistered_anchor_threshold",
        ),
        validation_anchor_only=_required_bool(raw, "validation_anchor_only"),
        learning_use_authorized=_required_bool(raw, "learning_use_authorized"),
        keyword_or_regex_routing_used=_required_bool(
            raw,
            "keyword_or_regex_routing_used",
        ),
        unknown_outcome_supported=_required_bool(
            raw,
            "unknown_outcome_supported",
        ),
        reviewed_by=_required_text_tuple(raw, "reviewed_by"),
        qualified_at_iso=_required_text(raw, "qualified_at_iso"),
        schema_version=_required_text(raw, "schema_version"),
    )
    if _required_bool(raw, "passed") is not qualification.passed:
        raise ValueError("typing qualification passed verdict is inconsistent")
    if raw["content_sha256"] != qualification.content_sha256:
        raise ValueError("typing qualification canonical content hash mismatch")
    return qualification


def _sha(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("relationship alpha hash input must be non-empty")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _with_hash(payload: dict[str, object]) -> dict[str, object]:
    body = dict(payload)
    body["content_sha256"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    return body


def _validate_content_hash(payload: Mapping[str, object]) -> None:
    stored = payload.get("content_sha256")
    if not isinstance(stored, str):
        raise ValueError("relationship alpha artifact lacks content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    computed = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    if stored != computed:
        raise ValueError("relationship alpha artifact content hash mismatch")


@dataclass(frozen=True)
class RelationshipAlphaActionAudit:
    audit_id: str
    recorded_at_iso: str
    subject_scope_sha256: str
    session_scope_sha256: str
    observation_sha256: str
    turn_index: int
    forecast_id: str
    decision_id: str
    candidate_predictions: tuple[
        tuple[str, tuple[tuple[str, float], ...]], ...
    ]
    recommended_action_id: str
    selected_action_id: str
    gate_action: str
    gate_mode: str
    steer_probability: float
    gate_features: tuple[float, ...]
    gate_update_count: int
    policy_artifact_id: str
    policy_artifact_version: int
    temporal_advisory_status: str
    applied_to_expression: bool
    boundary_external_action_blocked: bool
    boundary_external_action_consent: str
    boundary_autonomy_risk: float
    boundary_overreach_risk: float
    rationale_codes: tuple[str, ...]
    evidence_ref_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("audit_id", self.audit_id),
            ("recorded_at_iso", self.recorded_at_iso),
            ("subject_scope_sha256", self.subject_scope_sha256),
            ("session_scope_sha256", self.session_scope_sha256),
            ("observation_sha256", self.observation_sha256),
            ("forecast_id", self.forecast_id),
            ("decision_id", self.decision_id),
            ("recommended_action_id", self.recommended_action_id),
            ("selected_action_id", self.selected_action_id),
            ("gate_action", self.gate_action),
            ("gate_mode", self.gate_mode),
            ("policy_artifact_id", self.policy_artifact_id),
            ("temporal_advisory_status", self.temporal_advisory_status),
            ("boundary_external_action_consent", self.boundary_external_action_consent),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be non-empty")
        parsed = datetime.fromisoformat(self.recorded_at_iso)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("recorded_at_iso must include a timezone")
        if self.turn_index < 0 or self.gate_update_count < 0:
            raise ValueError("turn_index/gate_update_count must be non-negative")
        if not 0.0 <= self.steer_probability <= 1.0:
            raise ValueError("steer_probability must be in [0, 1]")
        if self.temporal_advisory_status != TemporalActionAdvisoryStatus.SHADOW_RECORDED.value:
            raise ValueError("closed-alpha relationship advisory must remain SHADOW")
        if self.applied_to_expression:
            raise ValueError("closed-alpha relationship advisory cannot affect expression")
        if not self.candidate_predictions:
            raise ValueError("candidate_predictions must be non-empty")
        if not self.rationale_codes or not self.evidence_ref_sha256:
            raise ValueError("audit rationale/evidence lineage must be non-empty")

    def to_json(self) -> dict[str, object]:
        return _with_hash(
            {
                "schema_version": RELATIONSHIP_ALPHA_AUDIT_SCHEMA_VERSION,
                "audit_id": self.audit_id,
                "recorded_at_iso": self.recorded_at_iso,
                "subject_scope_sha256": self.subject_scope_sha256,
                "session_scope_sha256": self.session_scope_sha256,
                "observation_sha256": self.observation_sha256,
                "turn_index": self.turn_index,
                "forecast_id": self.forecast_id,
                "decision_id": self.decision_id,
                "candidate_predictions": [
                    {
                        "action_id": action_id,
                        "outcomes": [
                            {"outcome_id": outcome_id, "probability": probability}
                            for outcome_id, probability in outcomes
                        ],
                    }
                    for action_id, outcomes in self.candidate_predictions
                ],
                "recommended_action_id": self.recommended_action_id,
                "selected_action_id": self.selected_action_id,
                "gate_action": self.gate_action,
                "gate_mode": self.gate_mode,
                "steer_probability": self.steer_probability,
                "gate_features": list(self.gate_features),
                "gate_update_count": self.gate_update_count,
                "policy_artifact_id": self.policy_artifact_id,
                "policy_artifact_version": self.policy_artifact_version,
                "temporal_advisory_status": self.temporal_advisory_status,
                "applied_to_expression": self.applied_to_expression,
                "boundary": {
                    "external_action_blocked": self.boundary_external_action_blocked,
                    "external_action_consent": self.boundary_external_action_consent,
                    "autonomy_risk": self.boundary_autonomy_risk,
                    "overreach_risk": self.boundary_overreach_risk,
                },
                "rationale_codes": list(self.rationale_codes),
                "evidence_ref_sha256": list(self.evidence_ref_sha256),
            }
        )


@dataclass(frozen=True)
class RelationshipAlphaOutcomeReceipt:
    outcome_id: str
    forecast_id: str
    decision_id: str
    action_id: str
    outcome_kind: str
    confidence: float
    outcome_observation_sha256: str
    typing_gate_status: str
    typing_qualification_id: str | None
    typing_qualification_sha256: str | None
    typing_runtime_id: str | None
    typing_schema_version: str | None
    typing_evidence_basis: str | None
    needs_human_review: bool
    action_exposure_status: str
    runtime_submission_status: str
    submitted_to_runtime: bool
    runtime_evidence_id: str | None
    operational_evidence_ref: str
    training_candidate_ref: str | None


@dataclass(frozen=True)
class RelationshipAlphaTurnResult:
    kernel_result: Any
    audit: RelationshipAlphaActionAudit | None
    status: str
    gate_credit_updates: tuple[Any, ...] = ()


class RelationshipAlphaArtifactStore:
    """Create-only audit/outcome store with an optional separate train root."""

    def __init__(
        self,
        *,
        evidence_root: Path,
        training_candidate_root: Path | None = None,
    ) -> None:
        self._evidence_root = evidence_root.expanduser().resolve()
        self._training_root = (
            training_candidate_root.expanduser().resolve()
            if training_candidate_root is not None
            else None
        )
        if self._training_root is not None and self._training_root == self._evidence_root:
            raise ValueError(
                "relationship alpha evidence and training candidate roots must differ"
            )

    def write_audit(self, audit: RelationshipAlphaActionAudit) -> Path:
        path = (
            self._evidence_root
            / "relationship_action_audits"
            / _sha(audit.forecast_id)[:2]
            / f"{_sha(audit.forecast_id)}.json"
        )
        _write_create_only_verified(path, audit.to_json())
        return path

    def load_audit(self, forecast_id: str) -> Mapping[str, object]:
        digest = _sha(forecast_id)
        path = (
            self._evidence_root
            / "relationship_action_audits"
            / digest[:2]
            / f"{digest}.json"
        )
        return _read_verified(path)

    def find_outcome(
        self,
        forecast_id: str,
    ) -> tuple[Mapping[str, object], str, str | None] | None:
        digest = _sha(forecast_id)
        evidence_path = (
            self._evidence_root
            / "relationship_outcomes"
            / digest[:2]
            / f"{digest}.json"
        )
        if not evidence_path.exists():
            return None
        payload = _read_verified(evidence_path)
        training_ref: str | None = None
        if self._training_root is not None:
            candidate_path = (
                self._training_root
                / "relationship_candidates"
                / digest[:2]
                / f"{digest}.json"
            )
            if candidate_path.exists():
                candidate = _read_verified(candidate_path)
                training_ref = (
                    "relationship-training-candidate:"
                    f"{_required_text(candidate, 'content_sha256')}"
                )
        return (
            payload,
            f"relationship-alpha-evidence:{_required_text(payload, 'content_sha256')}",
            training_ref,
        )

    def write_outcome(
        self,
        *,
        audit_payload: Mapping[str, object],
        outcome_kind: str,
        confidence: float,
        outcome_observation_sha256: str,
        typing_gate_status: str,
        typing_qualification_id: str | None,
        typing_qualification_sha256: str | None,
        typing_runtime_id: str | None,
        typing_schema_version: str | None,
        typing_evidence_basis: str | None,
        needs_human_review: bool,
        action_exposure_status: str,
        runtime_submission_status: str,
        submitted_to_runtime: bool,
        runtime_evidence_id: str | None,
        training_use_authorized: bool,
    ) -> tuple[str, str | None]:
        forecast_id = _required_text(audit_payload, "forecast_id")
        decision_id = _required_text(audit_payload, "decision_id")
        action_id = _required_text(audit_payload, "selected_action_id")
        audit_hash = _required_text(audit_payload, "content_sha256")
        outcome_id = f"relationship-alpha-outcome:{_sha(forecast_id)[:24]}"
        payload = _with_hash(
            {
                "schema_version": RELATIONSHIP_ALPHA_OUTCOME_SCHEMA_VERSION,
                "outcome_id": outcome_id,
                "forecast_id": forecast_id,
                "decision_id": decision_id,
                "action_id": action_id,
                "outcome_kind": outcome_kind,
                "confidence": confidence,
                "outcome_observation_sha256": outcome_observation_sha256,
                "typing_gate_status": typing_gate_status,
                "typing_qualification_id": typing_qualification_id,
                "typing_qualification_sha256": typing_qualification_sha256,
                "typing_runtime_id": typing_runtime_id,
                "typing_schema_version": typing_schema_version,
                "typing_evidence_basis": typing_evidence_basis,
                "needs_human_review": needs_human_review,
                "action_exposure_status": action_exposure_status,
                "runtime_submission_status": runtime_submission_status,
                "training_use_authorized": training_use_authorized,
                "submitted_to_runtime": submitted_to_runtime,
                "runtime_evidence_id": runtime_evidence_id,
                "action_audit_sha256": audit_hash,
                "recorded_at_iso": _required_text(
                    audit_payload,
                    "recorded_at_iso",
                ),
                "privacy_profile": "typed-hashed-metadata-only.v1",
            }
        )
        digest = _sha(forecast_id)
        evidence_path = (
            self._evidence_root
            / "relationship_outcomes"
            / digest[:2]
            / f"{digest}.json"
        )
        _write_create_only_verified(evidence_path, payload)
        training_ref: str | None = None
        if (
            training_use_authorized
            and action_exposure_status
            == RELATIONSHIP_ACTION_EXPOSURE_BASELINE_NOOP
        ):
            if self._training_root is None:
                raise ValueError(
                    "training_use_authorized requires a separate training candidate root"
                )
            candidate = _with_hash(
                {
                    "schema_version": RELATIONSHIP_TRAINING_CANDIDATE_SCHEMA_VERSION,
                    "source_outcome_sha256": payload["content_sha256"],
                    "action_audit_sha256": audit_hash,
                    "forecast_id_sha256": digest,
                    "action_id": action_id,
                    "outcome_kind": outcome_kind,
                    "confidence": confidence,
                    "action_exposure_status": action_exposure_status,
                    "promotion_status": "offline_gate_required",
                    "contains_raw_dialogue": False,
                }
            )
            candidate_path = (
                self._training_root
                / "relationship_candidates"
                / digest[:2]
                / f"{digest}.json"
            )
            _write_create_only_verified(candidate_path, candidate)
            training_ref = (
                "relationship-training-candidate:"
                f"{_required_text(candidate, 'content_sha256')}"
            )
        return (
            f"relationship-alpha-evidence:"
            f"{_required_text(payload, 'content_sha256')}",
            training_ref,
        )


class RelationshipIntelligenceController:
    """P4 closed-alpha coordinator; not a kernel state owner."""

    def __init__(
        self,
        *,
        artifact_store: RelationshipAlphaArtifactStore,
        state_root: Path | None = None,
        typing_qualification: RelationshipOutcomeTypingQualification | None = None,
        outcome_typer: RelationshipOutcomeTypingRuntime | None = None,
    ) -> None:
        self._artifact_store = artifact_store
        self._state_root = state_root.expanduser().resolve() if state_root else None
        self._typing_qualification = typing_qualification
        self._outcome_typer = outcome_typer
        self._typing_gate_status = (
            RELATIONSHIP_TYPING_PASSED
            if typing_qualification is not None and typing_qualification.passed
            else RELATIONSHIP_TYPING_COLLECTION_ONLY
        )
        if self._typing_gate_status == RELATIONSHIP_TYPING_PASSED:
            if outcome_typer is None:
                raise ValueError(
                    "passing relationship typing qualification requires its "
                    "structured LLM runtime"
                )
            if typing_qualification is None:  # pragma: no cover - invariant
                raise RuntimeError("typing qualification status lost its artifact")
            if outcome_typer.runtime_id != typing_qualification.structured_runtime_id:
                raise ValueError("typing runtime does not match qualification artifact")
            if (
                outcome_typer.schema_version
                != typing_qualification.structured_output_schema_id
            ):
                raise ValueError("typing schema does not match qualification artifact")
        elif outcome_typer is not None:
            raise ValueError(
                "relationship outcome typer cannot run without a passing "
                "qualification artifact"
            )
        self._gates: dict[str, RelationshipActionGate] = {}
        self._forecast_runtime = BoundedRelationshipPreferenceForecastRuntime()

    @property
    def typing_gate_status(self) -> str:
        return self._typing_gate_status

    @property
    def typing_qualification(self) -> RelationshipOutcomeTypingQualification | None:
        return self._typing_qualification

    async def run_turn(
        self,
        *,
        session: Any,
        user_input: str,
        subject_scope: str,
        session_scope: str,
        gate_mode: RelationshipActionGateMode = RelationshipActionGateMode.LEARNED,
    ) -> RelationshipAlphaTurnResult:
        if gate_mode is RelationshipActionGateMode.ORACLE:
            raise ValueError("closed-alpha product path cannot run oracle mode")
        upcoming_turn = len(session.turn_summaries) + 1
        decision_id = f"relationship-decision:{_sha(session_scope)[:16]}:{upcoming_turn}"
        request = PreferenceActionForecastRequest(
            decision_id=decision_id,
            interlocutor_id="primary",
            current_observation=user_input,
            observation_ref=(
                f"relationship-alpha-observation:{_sha(session_scope)[:16]}:"
                f"turn-{upcoming_turn}"
            ),
            candidate_action_ids=tuple(action.value for action in RELATIONSHIP_ACTIONS),
            outcome_ids=tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES),
            turn_index=upcoming_turn,
            session_scope=session_scope,
        )
        forecast = await session.preview_preference_action_forecast(
            request=request,
            runtime=self._forecast_runtime,
        )
        gate = self._gate(subject_scope)
        decision: RelationshipActionGateDecision | None = None
        if forecast is not None:
            decision = gate.decide(forecast, mode=gate_mode)
            session.stage_self_temporal_action_advisory(
                temporal_action_advisory_from_gate_decision(decision)
            )
        kernel_result = await session.run_turn(user_input)
        updates: list[Any] = []
        for credit in session.relationship_action_credits(
            settled_at_turn=upcoming_turn,
            timestamp_ms=upcoming_turn,
        ):
            updates.append(gate.observe_credit(credit))
        self._persist_gate(subject_scope, gate)
        if forecast is None or decision is None:
            return RelationshipAlphaTurnResult(
                kernel_result=kernel_result,
                audit=None,
                status="insufficient_typed_owner_evidence",
                gate_credit_updates=tuple(updates),
            )
        audit = _build_action_audit(
            subject_scope=subject_scope,
            session_scope=session_scope,
            user_input=user_input,
            forecast=forecast,
            decision=decision,
            kernel_result=kernel_result,
        )
        self._artifact_store.write_audit(audit)
        return RelationshipAlphaTurnResult(
            kernel_result=kernel_result,
            audit=audit,
            status="shadow_action_audited",
            gate_credit_updates=tuple(updates),
        )

    async def submit_outcome_text(
        self,
        *,
        session: Any,
        subject_scope: str,
        session_scope: str,
        forecast_id: str,
        decision_id: str,
        action_id: str,
        outcome_text: str,
        training_use_authorized: bool,
    ) -> RelationshipAlphaOutcomeReceipt:
        if not isinstance(outcome_text, str) or not outcome_text.strip():
            raise ValueError("outcome_text must be a non-empty string")
        audit = self._artifact_store.load_audit(forecast_id)
        if _required_text(audit, "subject_scope_sha256") != _sha(subject_scope):
            raise PermissionError("relationship action audit belongs to another subject")
        if _required_text(audit, "session_scope_sha256") != _sha(session_scope):
            raise ValueError("relationship action audit session mismatch")
        for key, supplied in (
            ("decision_id", decision_id),
            ("selected_action_id", action_id),
        ):
            if _required_text(audit, key) != supplied:
                raise ValueError(f"relationship outcome {key} lineage mismatch")
        existing = self._artifact_store.find_outcome(forecast_id)
        if existing is not None:
            payload, evidence_ref, training_ref = existing
            if (
                _required_text(payload, "outcome_observation_sha256")
                != _sha(outcome_text)
            ):
                raise ValueError(
                    "relationship outcome retry conflicts with prior observation"
                )
            if (
                _required_bool(payload, "training_use_authorized")
                is not training_use_authorized
            ):
                raise ValueError(
                    "relationship outcome retry conflicts with prior training consent"
                )
            return _outcome_receipt_from_payload(
                payload,
                operational_evidence_ref=evidence_ref,
                training_candidate_ref=training_ref,
            )
        typing_result = None
        if self._typing_gate_status == RELATIONSHIP_TYPING_PASSED:
            if self._outcome_typer is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("qualified relationship outcome typer is unavailable")
            typing_result = await asyncio.to_thread(
                self._outcome_typer.classify,
                outcome_text,
            )
        outcome_kind = (
            typing_result.outcome_kind
            if typing_result is not None
            else RELATIONSHIP_OUTCOME_UNKNOWN
        )
        confidence = typing_result.confidence if typing_result is not None else 0.0
        needs_human_review = (
            typing_result.needs_human_review if typing_result is not None else True
        )
        action_exposure_status = (
            RELATIONSHIP_ACTION_EXPOSURE_BASELINE_NOOP
            if _required_text(audit, "gate_action")
            == RelationshipGateAction.NOOP.value
            and action_id == RelationshipAction.NEUTRAL_NOOP.value
            else RELATIONSHIP_ACTION_EXPOSURE_SHADOW_COUNTERFACTUAL
        )
        runtime_evidence: DialogueExternalOutcomeEvidence | None = None
        submitted = bool(
            self._typing_gate_status == RELATIONSHIP_TYPING_PASSED
            and outcome_kind in _TYPED_RELATIONSHIP_OUTCOMES
            and not needs_human_review
            and action_exposure_status
            == RELATIONSHIP_ACTION_EXPOSURE_BASELINE_NOOP
        )
        if self._typing_gate_status != RELATIONSHIP_TYPING_PASSED:
            runtime_submission_status = RELATIONSHIP_RUNTIME_OUTCOME_TYPING_BLOCKED
        elif (
            outcome_kind not in _TYPED_RELATIONSHIP_OUTCOMES
            or needs_human_review
        ):
            runtime_submission_status = RELATIONSHIP_RUNTIME_OUTCOME_REVIEW_BLOCKED
        elif (
            action_exposure_status
            != RELATIONSHIP_ACTION_EXPOSURE_BASELINE_NOOP
        ):
            runtime_submission_status = RELATIONSHIP_RUNTIME_OUTCOME_EXPOSURE_BLOCKED
        else:
            runtime_submission_status = RELATIONSHIP_RUNTIME_OUTCOME_SUBMITTED
        if submitted:
            if self._typing_qualification is None or typing_result is None:
                raise RuntimeError(
                    "submitted relationship outcome lost qualification lineage"
                )
            action_turn = _required_int(audit, "turn_index")
            consuming_turn = len(session.turn_summaries) + 1
            runtime_evidence = session.submit_dialogue_outcome(
                kind=DialogueExternalOutcomeKind(outcome_kind),
                source=(
                    DialogueExternalOutcomeEvidenceSource.QUALIFIED_USER_REPORT
                ),
                confidence=confidence,
                turn_index=consuming_turn,
                action_turn_index=action_turn,
                evidence_ref=(
                    f"relationship-alpha:{_sha(forecast_id)[:24]}:"
                    f"{outcome_kind}:{_sha(outcome_text)[:16]}"
                ),
                forecast_id=forecast_id,
                decision_id=decision_id,
                action_id=action_id,
                typing_qualification_id=(
                    self._typing_qualification.qualification_id
                ),
                typing_qualification_sha256=(
                    self._typing_qualification.content_sha256
                ),
                typing_runtime_id=typing_result.runtime_id,
                typing_schema_version=typing_result.schema_version,
            )
        evidence_ref, training_ref = self._artifact_store.write_outcome(
            audit_payload=audit,
            outcome_kind=outcome_kind,
            confidence=confidence,
            outcome_observation_sha256=_sha(outcome_text),
            typing_gate_status=self._typing_gate_status,
            typing_qualification_id=(
                self._typing_qualification.qualification_id
                if self._typing_qualification is not None
                else None
            ),
            typing_qualification_sha256=(
                self._typing_qualification.content_sha256
                if self._typing_qualification is not None
                else None
            ),
            typing_runtime_id=(
                typing_result.runtime_id if typing_result is not None else None
            ),
            typing_schema_version=(
                typing_result.schema_version if typing_result is not None else None
            ),
            typing_evidence_basis=(
                typing_result.evidence_basis.value
                if typing_result is not None
                else None
            ),
            needs_human_review=needs_human_review,
            action_exposure_status=action_exposure_status,
            runtime_submission_status=runtime_submission_status,
            submitted_to_runtime=submitted,
            runtime_evidence_id=(
                runtime_evidence.evidence_id if runtime_evidence is not None else None
            ),
            training_use_authorized=training_use_authorized,
        )
        return RelationshipAlphaOutcomeReceipt(
            outcome_id=f"relationship-alpha-outcome:{_sha(forecast_id)[:24]}",
            forecast_id=forecast_id,
            decision_id=decision_id,
            action_id=action_id,
            outcome_kind=outcome_kind,
            confidence=confidence,
            outcome_observation_sha256=_sha(outcome_text),
            typing_gate_status=self._typing_gate_status,
            typing_qualification_id=(
                self._typing_qualification.qualification_id
                if self._typing_qualification is not None
                else None
            ),
            typing_qualification_sha256=(
                self._typing_qualification.content_sha256
                if self._typing_qualification is not None
                else None
            ),
            typing_runtime_id=(
                typing_result.runtime_id if typing_result is not None else None
            ),
            typing_schema_version=(
                typing_result.schema_version if typing_result is not None else None
            ),
            typing_evidence_basis=(
                typing_result.evidence_basis.value
                if typing_result is not None
                else None
            ),
            needs_human_review=needs_human_review,
            action_exposure_status=action_exposure_status,
            runtime_submission_status=runtime_submission_status,
            submitted_to_runtime=submitted,
            runtime_evidence_id=(
                runtime_evidence.evidence_id if runtime_evidence is not None else None
            ),
            operational_evidence_ref=evidence_ref,
            training_candidate_ref=training_ref,
        )

    def _gate(self, subject_scope: str) -> RelationshipActionGate:
        key = _sha(subject_scope)
        existing = self._gates.get(key)
        if existing is not None:
            return existing
        checkpoint = self._load_gate_checkpoint(key)
        gate = RelationshipActionGate(checkpoint=checkpoint)
        self._gates[key] = gate
        return gate

    def _checkpoint_path(self, subject_key: str) -> Path | None:
        if self._state_root is None:
            return None
        return (
            self._state_root
            / "relationship_action_gate"
            / subject_key[:2]
            / f"{subject_key}.json"
        )

    def _persist_gate(self, subject_scope: str, gate: RelationshipActionGate) -> None:
        path = self._checkpoint_path(_sha(subject_scope))
        if path is None:
            return
        payload = _checkpoint_to_json(gate.export_checkpoint())
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _load_gate_checkpoint(
        self,
        subject_key: str,
    ) -> RelationshipActionGateCheckpoint | None:
        path = self._checkpoint_path(subject_key)
        if path is None or not path.exists():
            return None
        raw = _read_verified(path)
        return _checkpoint_from_json(raw)


def _build_action_audit(
    *,
    subject_scope: str,
    session_scope: str,
    user_input: str,
    forecast: PreferenceActionForecast,
    decision: RelationshipActionGateDecision,
    kernel_result: Any,
) -> RelationshipAlphaActionAudit:
    temporal_snapshot = kernel_result.active_snapshots.get(
        "self_temporal"
    ) or kernel_result.shadow_snapshots.get("self_temporal")
    if temporal_snapshot is None or not isinstance(
        temporal_snapshot.value,
        TemporalAbstractionSnapshot,
    ):
        raise RuntimeError("relationship alpha turn lacks self_temporal snapshot")
    temporal = temporal_snapshot.value
    if temporal.action_advisory is None:
        raise RuntimeError("self_temporal did not publish the staged action advisory")
    if temporal.action_advisory.prediction_id != forecast.forecast_id:
        raise RuntimeError("self_temporal action advisory forecast lineage mismatch")
    boundary_snapshot = kernel_result.active_snapshots.get(
        "boundary_consent"
    ) or kernel_result.shadow_snapshots.get("boundary_consent")
    if boundary_snapshot is None or not isinstance(
        boundary_snapshot.value,
        BoundaryConsentSnapshot,
    ):
        raise RuntimeError("relationship alpha turn lacks boundary_consent snapshot")
    boundary = boundary_snapshot.value
    candidates = tuple(
        (
            candidate.action_id,
            tuple(
                (outcome.outcome_id, outcome.probability)
                for outcome in candidate.outcomes
            ),
        )
        for candidate in forecast.candidate_predictions
    )
    return RelationshipAlphaActionAudit(
        audit_id=f"relationship-action-audit:{_sha(forecast.forecast_id)[:24]}",
        recorded_at_iso=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        subject_scope_sha256=_sha(subject_scope),
        session_scope_sha256=_sha(session_scope),
        observation_sha256=_sha(user_input),
        turn_index=forecast.issued_turn,
        forecast_id=forecast.forecast_id,
        decision_id=forecast.decision_id,
        candidate_predictions=candidates,
        recommended_action_id=forecast.recommended_action_id,
        selected_action_id=decision.selected_action_id,
        gate_action=decision.gate_action.value,
        gate_mode=decision.mode.value,
        steer_probability=decision.steer_probability,
        gate_features=decision.features,
        gate_update_count=decision.update_count,
        policy_artifact_id=decision.artifact_id,
        policy_artifact_version=decision.artifact_version,
        temporal_advisory_status=temporal.action_advisory_status.value,
        applied_to_expression=(
            temporal.action_advisory_status is TemporalActionAdvisoryStatus.APPLIED
        ),
        boundary_external_action_blocked=boundary.external_action_blocked,
        boundary_external_action_consent=boundary.external_action_consent,
        boundary_autonomy_risk=boundary.autonomy_risk,
        boundary_overreach_risk=boundary.overreach_risk,
        rationale_codes=decision.rationale_codes,
        evidence_ref_sha256=tuple(_sha(item) for item in decision.evidence_refs),
    )


def _outcome_receipt_from_payload(
    payload: Mapping[str, object],
    *,
    operational_evidence_ref: str,
    training_candidate_ref: str | None,
) -> RelationshipAlphaOutcomeReceipt:
    if _required_text(payload, "schema_version") != (
        RELATIONSHIP_ALPHA_OUTCOME_SCHEMA_VERSION
    ):
        raise ValueError("relationship alpha outcome schema mismatch")
    return RelationshipAlphaOutcomeReceipt(
        outcome_id=_required_text(payload, "outcome_id"),
        forecast_id=_required_text(payload, "forecast_id"),
        decision_id=_required_text(payload, "decision_id"),
        action_id=_required_text(payload, "action_id"),
        outcome_kind=_required_text(payload, "outcome_kind"),
        confidence=_required_float(payload, "confidence"),
        outcome_observation_sha256=_required_text(
            payload,
            "outcome_observation_sha256",
        ),
        typing_gate_status=_required_text(payload, "typing_gate_status"),
        typing_qualification_id=_optional_text(payload, "typing_qualification_id"),
        typing_qualification_sha256=_optional_text(
            payload,
            "typing_qualification_sha256",
        ),
        typing_runtime_id=_optional_text(payload, "typing_runtime_id"),
        typing_schema_version=_optional_text(payload, "typing_schema_version"),
        typing_evidence_basis=_optional_text(payload, "typing_evidence_basis"),
        needs_human_review=_required_bool(payload, "needs_human_review"),
        action_exposure_status=_required_text(payload, "action_exposure_status"),
        runtime_submission_status=_required_text(
            payload,
            "runtime_submission_status",
        ),
        submitted_to_runtime=_required_bool(payload, "submitted_to_runtime"),
        runtime_evidence_id=_optional_text(payload, "runtime_evidence_id"),
        operational_evidence_ref=operational_evidence_ref,
        training_candidate_ref=training_candidate_ref,
    )


def _decision_to_json(decision: RelationshipActionGateDecision) -> dict[str, object]:
    return {
        "decision_id": decision.decision_id,
        "forecast_id": decision.forecast_id,
        "gate_action": decision.gate_action.value,
        "selected_action_id": decision.selected_action_id,
        "recommended_action_id": decision.recommended_action_id,
        "steer_probability": decision.steer_probability,
        "features": list(decision.features),
        "mode": decision.mode.value,
        "artifact_id": decision.artifact_id,
        "artifact_version": decision.artifact_version,
        "update_count": decision.update_count,
        "evidence_refs": list(decision.evidence_refs),
        "rationale_codes": list(decision.rationale_codes),
        "evaluator_only": decision.evaluator_only,
    }


def _decision_from_json(raw: Mapping[str, object]) -> RelationshipActionGateDecision:
    return RelationshipActionGateDecision(
        decision_id=_required_text(raw, "decision_id"),
        forecast_id=_required_text(raw, "forecast_id"),
        gate_action=RelationshipGateAction(_required_text(raw, "gate_action")),
        selected_action_id=_required_text(raw, "selected_action_id"),
        recommended_action_id=_required_text(raw, "recommended_action_id"),
        steer_probability=_required_float(raw, "steer_probability"),
        features=_required_float_tuple(raw, "features"),
        mode=RelationshipActionGateMode(_required_text(raw, "mode")),
        artifact_id=_required_text(raw, "artifact_id"),
        artifact_version=_required_int(raw, "artifact_version"),
        update_count=_required_int(raw, "update_count"),
        evidence_refs=_required_text_tuple(raw, "evidence_refs"),
        rationale_codes=_required_text_tuple(raw, "rationale_codes"),
        evaluator_only=_required_bool(raw, "evaluator_only"),
    )


def _checkpoint_to_json(checkpoint: RelationshipActionGateCheckpoint) -> dict[str, object]:
    return _with_hash(
        {
            "schema_version": checkpoint.schema_version,
            "artifact_id": checkpoint.artifact_id,
            "artifact_version": checkpoint.artifact_version,
            "weights": list(checkpoint.weights),
            "bias": checkpoint.bias,
            "update_count": checkpoint.update_count,
            "processed_credit_ids": list(checkpoint.processed_credit_ids),
            "pending_decisions": [
                _decision_to_json(decision)
                for decision in checkpoint.pending_decisions
            ],
        }
    )


def _checkpoint_from_json(raw: Mapping[str, object]) -> RelationshipActionGateCheckpoint:
    pending = raw.get("pending_decisions")
    if not isinstance(pending, list) or any(not isinstance(item, dict) for item in pending):
        raise ValueError("pending_decisions must be an array of objects")
    return RelationshipActionGateCheckpoint(
        artifact_id=_required_text(raw, "artifact_id"),
        artifact_version=_required_int(raw, "artifact_version"),
        weights=_required_float_tuple(raw, "weights"),
        bias=_required_float(raw, "bias"),
        update_count=_required_int(raw, "update_count"),
        processed_credit_ids=_required_text_tuple(raw, "processed_credit_ids", allow_empty=True),
        pending_decisions=tuple(_decision_from_json(item) for item in pending),
        schema_version=_required_text(raw, "schema_version"),
    )


def _write_create_only_verified(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        existing = _read_verified(path)
        if dict(existing) != dict(payload):
            raise ValueError(
                f"relationship alpha artifact conflicts with {path}"
            ) from exc


def _read_verified(path: Path) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read relationship alpha artifact {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"relationship alpha artifact {path} must be an object")
    _validate_content_hash(raw)
    return raw


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _required_text_tuple(
    payload: Mapping[str, object],
    key: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must be an array of non-empty strings")
    if not allow_empty and not value:
        raise ValueError(f"{key} must be non-empty")
    return tuple(value)


def _required_float_tuple(payload: Mapping[str, object], key: str) -> tuple[float, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in value
    ):
        raise ValueError(f"{key} must be an array of numbers")
    return tuple(float(item) for item in value)


__all__ = [
    "RELATIONSHIP_ALPHA_AUDIT_SCHEMA_VERSION",
    "RELATIONSHIP_ALPHA_OUTCOME_SCHEMA_VERSION",
    "RELATIONSHIP_ACTION_EXPOSURE_BASELINE_NOOP",
    "RELATIONSHIP_ACTION_EXPOSURE_SHADOW_COUNTERFACTUAL",
    "RELATIONSHIP_RUNTIME_OUTCOME_EXPOSURE_BLOCKED",
    "RELATIONSHIP_RUNTIME_OUTCOME_REVIEW_BLOCKED",
    "RELATIONSHIP_RUNTIME_OUTCOME_SUBMITTED",
    "RELATIONSHIP_RUNTIME_OUTCOME_TYPING_BLOCKED",
    "RELATIONSHIP_TRAINING_CANDIDATE_SCHEMA_VERSION",
    "RELATIONSHIP_TYPING_QUALIFICATION_SCHEMA_VERSION",
    "RELATIONSHIP_TYPING_COLLECTION_ONLY",
    "RELATIONSHIP_TYPING_PASSED",
    "RelationshipAlphaActionAudit",
    "RelationshipAlphaArtifactStore",
    "RelationshipAlphaOutcomeReceipt",
    "RelationshipAlphaTurnResult",
    "RelationshipIntelligenceController",
    "RelationshipOutcomeTypingQualification",
    "load_relationship_outcome_typing_qualification",
]
