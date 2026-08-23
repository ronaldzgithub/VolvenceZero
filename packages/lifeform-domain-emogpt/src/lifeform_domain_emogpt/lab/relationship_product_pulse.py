"""Frozen two-phase relationship-action pulse for product-shaped Lab runs.

The pre-action phase can only consume public typed observation data, an
owner-published persistence snapshot, a gate checkpoint, and an injected
forecast collaborator.  It freezes the owner forecast and gate decision before
an outcome exists, then applies the selected *typed* action only to the
offline-environment temporal surface.  It never authorizes expression or a
production runtime and it deliberately accepts only a placeholder substrate.

The settlement phase consumes a separately published typed external outcome,
performs the exact owner join, lifts the owner settlement to social PE, derives
the dedicated PE credit, and optionally applies that credit to the pending
gate decision.  Evaluator truth, judge scores, and preferred-action labels are
not fields of either public input contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from volvence_zero.credit import (
    CreditRecord,
    derive_preference_action_forecast_credit_records,
)
from volvence_zero.dialogue_external_outcome import DialogueExternalOutcomeModule
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.memory import Track
from volvence_zero.owner_hydration import OwnerPersistenceSnapshot
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
)
from volvence_zero.social import (
    PreferenceAboutOtherModule,
    PreferenceActionForecastRequest,
    PreferenceActionForecastRuntime,
    SocialPredictionErrorModule,
    SocialRecordStore,
)
from volvence_zero.social_cognition import (
    PreferenceAboutOtherSnapshot,
    PreferenceActionForecast,
    PreferenceActionForecastSettlement,
    PreferenceActionOutcomeEvidence,
    SocialPredictionErrorSnapshot,
)
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind
from volvence_zero.temporal import PlaceholderTemporalPolicy, TrackTemporalModule
from volvence_zero.temporal_types import (
    TemporalAbstractionSnapshot,
    TemporalActionAdvisoryProposal,
    TemporalActionAdvisoryStatus,
)

from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RELATIONSHIP_OUTCOMES,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RelationshipActionGate,
    RelationshipActionGateCheckpoint,
    RelationshipActionGateDecision,
    RelationshipActionGateMode,
    RelationshipActionGateUpdate,
    temporal_action_advisory_from_gate_decision,
)


RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION = "relationship-product-pulse.v1"
_PREFERENCE_SLOT = "preference_about_other"
_ACTION_IDS = tuple(action.value for action in RELATIONSHIP_ACTIONS)
_OUTCOME_IDS = tuple(outcome.value for outcome in RELATIONSHIP_OUTCOMES)


@dataclass(frozen=True)
class RelationshipProductOnboardingInput:
    """One public, already-observed onboarding experience."""

    session_id: str
    session_index: int
    turn_index: int
    public_observation: str
    action_id: str
    observed_outcome_id: str
    reaction_summary: str
    evidence_ref: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("session_id", self.session_id),
            ("public_observation", self.public_observation),
            ("action_id", self.action_id),
            ("observed_outcome_id", self.observed_outcome_id),
            ("reaction_summary", self.reaction_summary),
            ("evidence_ref", self.evidence_ref),
        ):
            _require_text(value, field_name)
        for field_name, value in (
            ("session_index", self.session_index),
            ("turn_index", self.turn_index),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.action_id not in _ACTION_IDS:
            raise ValueError("onboarding action is outside the relationship surface")
        if self.observed_outcome_id not in _OUTCOME_IDS:
            raise ValueError(
                "onboarding outcome is outside the relationship surface"
            )

    @property
    def evidence_id(self) -> str:
        body = {
            "session_id": self.session_id,
            "session_index": self.session_index,
            "turn_index": self.turn_index,
            "public_observation": self.public_observation,
            "action_id": self.action_id,
            "observed_outcome_id": self.observed_outcome_id,
            "reaction_summary": self.reaction_summary,
            "evidence_ref": self.evidence_ref,
        }
        encoded = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"relationship-product-onboarding:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class RelationshipProductOnboardingSnapshot:
    """Frozen owner publication and resumable state after onboarding."""

    onboarding: RelationshipProductOnboardingInput
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    schema_version: str = RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product onboarding schema mismatch")
        if not isinstance(
            self.preference_snapshot.value,
            PreferenceAboutOtherSnapshot,
        ):
            raise TypeError(
                "onboarding preference snapshot published an unexpected value"
            )
        matching = tuple(
            item
            for item in self.preference_snapshot.value.action_outcome_evidence
            if item.evidence_id == self.onboarding.evidence_id
        )
        if len(matching) != 1:
            raise ValueError(
                "onboarding owner snapshot must contain exactly one current evidence"
            )


@dataclass(frozen=True)
class RelationshipProductPulseAuthorization:
    """Narrow authorization for typed offline-environment action exposure."""

    authorization_id: str
    allowed_policy_artifact_id: str
    allowed_policy_artifact_version: int
    environment_consumer_only: bool = True
    expression_authorized: bool = False
    production_authorized: bool = False
    oracle_action_authorized: bool = False

    def __post_init__(self) -> None:
        _require_text(self.authorization_id, "authorization_id")
        _require_text(
            self.allowed_policy_artifact_id,
            "allowed_policy_artifact_id",
        )
        if (
            isinstance(self.allowed_policy_artifact_version, bool)
            or not isinstance(self.allowed_policy_artifact_version, int)
            or self.allowed_policy_artifact_version < 1
        ):
            raise ValueError("allowed_policy_artifact_version must be >= 1")
        if not self.environment_consumer_only:
            raise ValueError(
                "relationship product pulse authorization must be "
                "environment-consumer-only"
            )
        if (
            self.expression_authorized
            or self.production_authorized
            or self.oracle_action_authorized
        ):
            raise ValueError(
                "relationship product pulse authorization firewall is open"
            )


@dataclass(frozen=True)
class RelationshipProductPreActionRequest:
    """Outcome-free public request for one relationship decision pulse."""

    session_id: str
    forecast_request: PreferenceActionForecastRequest
    outcome_turn_index: int

    def __post_init__(self) -> None:
        _require_text(self.session_id, "session_id")
        if not isinstance(
            self.forecast_request,
            PreferenceActionForecastRequest,
        ):
            raise TypeError(
                "forecast_request must be a PreferenceActionForecastRequest"
            )
        if not self.forecast_request.session_scope:
            raise ValueError("forecast_request.session_scope must be non-empty")
        if self.forecast_request.candidate_action_ids != _ACTION_IDS:
            raise ValueError(
                "forecast_request must expose the canonical relationship "
                "action surface"
            )
        if self.forecast_request.outcome_ids != _OUTCOME_IDS:
            raise ValueError(
                "forecast_request must expose the canonical relationship "
                "outcome surface"
            )
        if (
            isinstance(self.outcome_turn_index, bool)
            or not isinstance(self.outcome_turn_index, int)
            or self.outcome_turn_index
            != self.forecast_request.turn_index + 1
        ):
            raise ValueError(
                "outcome_turn_index must be exactly one turn after preaction"
            )


@dataclass(frozen=True)
class RelationshipProductPreActionSnapshot:
    """Frozen owner/gate/temporal exchange published before any outcome."""

    request: RelationshipProductPreActionRequest
    authorization_id: str
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    forecast: PreferenceActionForecast
    gate_decision: RelationshipActionGateDecision
    temporal_snapshot: Snapshot[TemporalAbstractionSnapshot]
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    gate_checkpoint_before: RelationshipActionGateCheckpoint
    gate_checkpoint_after: RelationshipActionGateCheckpoint
    schema_version: str = RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product preaction schema mismatch")
        _require_text(self.authorization_id, "authorization_id")
        if not isinstance(
            self.preference_snapshot.value,
            PreferenceAboutOtherSnapshot,
        ):
            raise TypeError(
                "preaction preference snapshot published an unexpected value"
            )
        if self.forecast.decision_id != self.request.forecast_request.decision_id:
            raise ValueError("preaction forecast decision lineage mismatch")
        if self.forecast.session_scope != self.request.forecast_request.session_scope:
            raise ValueError("preaction forecast session lineage mismatch")
        if self.gate_decision.forecast_id != self.forecast.forecast_id:
            raise ValueError("preaction gate forecast lineage mismatch")
        temporal = self.temporal_snapshot.value
        if not isinstance(temporal, TemporalAbstractionSnapshot):
            raise TypeError(
                "temporal_snapshot must publish TemporalAbstractionSnapshot"
            )
        if temporal.action_advisory_status is not TemporalActionAdvisoryStatus.APPLIED:
            raise ValueError("preaction typed action was not applied")
        if temporal.active_abstract_action != self.gate_decision.selected_action_id:
            raise ValueError("preaction temporal action lineage mismatch")
        if self.gate_checkpoint_before.artifact_id != self.gate_decision.artifact_id:
            raise ValueError("preaction gate artifact lineage mismatch")
        if self.gate_checkpoint_after.update_count != self.gate_decision.update_count:
            raise ValueError("preaction gate update-count lineage mismatch")
        matching = tuple(
            item
            for item in self.preference_snapshot.value.action_forecasts
            if item.forecast_id == self.forecast.forecast_id
        )
        if matching != (self.forecast,):
            raise ValueError(
                "preaction owner snapshot must contain exactly the frozen forecast"
            )
        pending = tuple(
            item
            for item in self.gate_checkpoint_after.pending_decisions
            if item.forecast_id == self.forecast.forecast_id
        )
        expected_pending = (
            (self.gate_decision,)
            if self.gate_decision.mode is RelationshipActionGateMode.LEARNED
            else ()
        )
        if pending != expected_pending:
            raise ValueError("preaction pending gate decision lineage mismatch")


@dataclass(frozen=True)
class RelationshipProductSettlementInput:
    """Post-action typed evidence; contains no evaluator or judge fields."""

    external_outcome: DialogueExternalOutcomeEvidence
    owner_outcome_evidence: PreferenceActionOutcomeEvidence
    credit_timestamp_ms: int
    apply_credit_to_gate: bool

    def __post_init__(self) -> None:
        if not isinstance(
            self.external_outcome,
            DialogueExternalOutcomeEvidence,
        ):
            raise TypeError(
                "external_outcome must be DialogueExternalOutcomeEvidence"
            )
        if not isinstance(
            self.owner_outcome_evidence,
            PreferenceActionOutcomeEvidence,
        ):
            raise TypeError(
                "owner_outcome_evidence must be "
                "PreferenceActionOutcomeEvidence"
            )
        if (
            isinstance(self.credit_timestamp_ms, bool)
            or not isinstance(self.credit_timestamp_ms, int)
            or self.credit_timestamp_ms < 0
        ):
            raise ValueError("credit_timestamp_ms must be a non-negative integer")
        if not isinstance(self.apply_credit_to_gate, bool):
            raise TypeError("apply_credit_to_gate must be a boolean")


@dataclass(frozen=True)
class RelationshipProductSettlementSnapshot:
    """Frozen owner settlement, PE, credit, and next resumable state."""

    preaction: RelationshipProductPreActionSnapshot
    external_outcome_snapshot: Snapshot[DialogueExternalOutcomeSnapshot]
    preference_snapshot: Snapshot[PreferenceAboutOtherSnapshot]
    social_prediction_error_snapshot: Snapshot[SocialPredictionErrorSnapshot]
    settlement: PreferenceActionForecastSettlement
    credit: CreditRecord
    gate_update: RelationshipActionGateUpdate | None
    owner_persistence_snapshot: OwnerPersistenceSnapshot
    gate_checkpoint: RelationshipActionGateCheckpoint
    credit_applied_to_gate: bool
    schema_version: str = RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION:
            raise ValueError("relationship product settlement schema mismatch")
        if not isinstance(
            self.external_outcome_snapshot.value,
            DialogueExternalOutcomeSnapshot,
        ):
            raise TypeError(
                "external outcome snapshot published an unexpected value"
            )
        if self.settlement.forecast_id != self.preaction.forecast.forecast_id:
            raise ValueError("settlement forecast lineage mismatch")
        if self.credit.prediction_id != self.preaction.forecast.forecast_id:
            raise ValueError("settlement credit forecast lineage mismatch")
        if self.credit_applied_to_gate != (self.gate_update is not None):
            raise ValueError("settlement gate-update audit is inconsistent")
        if (
            self.gate_checkpoint.update_count
            != self.preaction.gate_checkpoint_after.update_count
            + int(self.credit_applied_to_gate)
        ):
            raise ValueError("settlement gate checkpoint count drifted")


async def append_relationship_product_onboarding(
    *,
    owner_persistence_snapshot: OwnerPersistenceSnapshot | None,
    onboarding: RelationshipProductOnboardingInput,
) -> RelationshipProductOnboardingSnapshot:
    """Append one public onboarding experience through its unique owner."""

    store = SocialRecordStore()
    if owner_persistence_snapshot is not None:
        store.hydrate_from_persistence(owner_persistence_snapshot)
    evidence = PreferenceActionOutcomeEvidence(
        evidence_id=onboarding.evidence_id,
        interlocutor_id="primary",
        observation_summary=onboarding.public_observation,
        action_id=onboarding.action_id,
        observed_outcome_id=onboarding.observed_outcome_id,
        reaction_summary=onboarding.reaction_summary,
        source_turn=onboarding.turn_index,
        evidence_refs=(onboarding.evidence_ref,),
    )
    owner_snapshot = await PreferenceAboutOtherModule(
        proposal_runtime=_OutcomeEvidenceProposalRuntime(
            evidence=evidence,
            semantic_evidence_ref=onboarding.evidence_ref,
        ),
        user_input=onboarding.public_observation,
        turn_index=onboarding.turn_index,
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
        action_outcome_evidence=evidence,
    ).process({})
    return RelationshipProductOnboardingSnapshot(
        onboarding=onboarding,
        preference_snapshot=owner_snapshot,
        owner_persistence_snapshot=store.export_persistence_snapshot(),
    )


def authorize_relationship_product_pulse_advisory(
    decision: RelationshipActionGateDecision,
    *,
    authorization: RelationshipProductPulseAuthorization,
) -> TemporalActionAdvisoryProposal:
    """Authorize a non-oracle typed decision for an offline environment."""

    if decision.evaluator_only or decision.mode is RelationshipActionGateMode.ORACLE:
        raise ValueError(
            "relationship product pulse cannot authorize evaluator/oracle decisions"
        )
    if decision.artifact_id != authorization.allowed_policy_artifact_id:
        raise ValueError("gate artifact id is outside pulse authorization")
    if (
        decision.artifact_version
        != authorization.allowed_policy_artifact_version
    ):
        raise ValueError("gate artifact version is outside pulse authorization")
    advisory = temporal_action_advisory_from_gate_decision(decision)
    return replace(
        advisory,
        active_authorized=True,
        evidence_refs=(
            *advisory.evidence_refs,
            f"lab-authorization:{authorization.authorization_id}",
        ),
        rationale_codes=(
            *advisory.rationale_codes,
            "scope:offline-reactive-environment-only",
            "expression:forbidden",
            "production:forbidden",
        ),
    )


async def prepare_relationship_product_preaction(
    *,
    request: RelationshipProductPreActionRequest,
    owner_persistence_snapshot: OwnerPersistenceSnapshot,
    gate_checkpoint: RelationshipActionGateCheckpoint | None,
    forecast_runtime: PreferenceActionForecastRuntime,
    gate_mode: RelationshipActionGateMode,
    authorization: RelationshipProductPulseAuthorization,
    substrate_snapshot: SubstrateSnapshot,
) -> RelationshipProductPreActionSnapshot:
    """Freeze owner forecast and expose one typed action before settlement."""

    if gate_mode is RelationshipActionGateMode.ORACLE:
        raise ValueError("relationship product preaction rejects oracle mode")
    if not isinstance(gate_mode, RelationshipActionGateMode):
        raise TypeError("gate_mode must be RelationshipActionGateMode")
    _validate_placeholder_substrate(substrate_snapshot)

    store = SocialRecordStore()
    store.hydrate_from_persistence(owner_persistence_snapshot)
    gate = RelationshipActionGate(checkpoint=gate_checkpoint)
    checkpoint_before = gate.export_checkpoint()
    owner_snapshot = await PreferenceAboutOtherModule(
        turn_index=request.forecast_request.turn_index,
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
        action_forecast_runtime=forecast_runtime,
        action_forecast_request=request.forecast_request,
    ).process({})
    if not isinstance(owner_snapshot.value, PreferenceAboutOtherSnapshot):
        raise TypeError("preaction preference owner published unexpected snapshot")
    forecasts = tuple(
        forecast
        for forecast in owner_snapshot.value.action_forecasts
        if forecast.decision_id == request.forecast_request.decision_id
    )
    if len(forecasts) != 1:
        raise RuntimeError(
            "preaction preference owner must publish exactly one current forecast"
        )
    forecast = forecasts[0]
    decision = gate.decide(forecast, mode=gate_mode)
    advisory = authorize_relationship_product_pulse_advisory(
        decision,
        authorization=authorization,
    )
    temporal_snapshot = await _apply_typed_environment_advisory(
        advisory,
        substrate_snapshot=substrate_snapshot,
    )
    return RelationshipProductPreActionSnapshot(
        request=request,
        authorization_id=authorization.authorization_id,
        preference_snapshot=owner_snapshot,
        forecast=forecast,
        gate_decision=decision,
        temporal_snapshot=temporal_snapshot,
        owner_persistence_snapshot=store.export_persistence_snapshot(),
        gate_checkpoint_before=checkpoint_before,
        gate_checkpoint_after=gate.export_checkpoint(),
    )


async def settle_relationship_product_pulse(
    *,
    preaction: RelationshipProductPreActionSnapshot,
    settlement_input: RelationshipProductSettlementInput,
) -> RelationshipProductSettlementSnapshot:
    """Settle one frozen preaction through owner PE and dedicated credit."""

    _validate_settlement_lineage(preaction, settlement_input)
    if (
        settlement_input.apply_credit_to_gate
        and preaction.gate_decision.mode is not RelationshipActionGateMode.LEARNED
    ):
        raise ValueError("gate credit can only be applied to a learned decision")

    store = SocialRecordStore()
    store.hydrate_from_persistence(preaction.owner_persistence_snapshot)
    gate = RelationshipActionGate(checkpoint=preaction.gate_checkpoint_after)

    # ``SocialRecordStore`` deliberately excludes one-turn ToM pending state
    # from its cross-session persistence contract.  Preaction and settlement
    # are two phases of the *same* session, so ask the owner to rebuild that
    # fast state at the original action turn before presenting the outcome.
    # This is owner-authored replay, not consumer reconstruction of hidden
    # fields, and keeps the split API behavior equal to the former in-process
    # P4.1 loop.
    replayed_preaction_owner = await PreferenceAboutOtherModule(
        turn_index=preaction.request.forecast_request.turn_index,
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
    ).process({})
    if not isinstance(
        replayed_preaction_owner.value,
        PreferenceAboutOtherSnapshot,
    ):
        raise TypeError("preaction owner fast-state replay published unexpected snapshot")

    external_owner = DialogueExternalOutcomeModule(wiring_level=WiringLevel.ACTIVE)
    external_owner.set_turn_index(preaction.request.outcome_turn_index)
    external_owner.append_evidence(settlement_input.external_outcome)
    external_snapshot = await external_owner.process({})

    evidence = settlement_input.owner_outcome_evidence
    settlement_owner = PreferenceAboutOtherModule(
        proposal_runtime=_OutcomeEvidenceProposalRuntime(
            evidence=evidence,
            semantic_evidence_ref=settlement_input.external_outcome.evidence_ref,
        ),
        user_input=evidence.observation_summary,
        turn_index=evidence.source_turn,
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
        action_outcome_evidence=evidence,
    )
    settled_snapshot = await settlement_owner.process(
        {"dialogue_external_outcome": external_snapshot}
    )
    if not isinstance(settled_snapshot.value, PreferenceAboutOtherSnapshot):
        raise TypeError("settlement preference owner published unexpected snapshot")
    current_settlements = tuple(
        settlement
        for settlement in settled_snapshot.value.forecast_settlements
        if settlement.forecast_id == preaction.forecast.forecast_id
        and settlement.observed_turn == preaction.request.outcome_turn_index
    )
    if len(current_settlements) != 1:
        raise RuntimeError(
            "relationship product pulse requires exactly one current settlement"
        )

    social_pe_snapshot = await SocialPredictionErrorModule(
        wiring_level=WiringLevel.ACTIVE
    ).process({"preference_about_other": settled_snapshot})
    if not isinstance(
        social_pe_snapshot.value,
        SocialPredictionErrorSnapshot,
    ):
        raise TypeError("social PE owner published unexpected snapshot")
    credits = derive_preference_action_forecast_credit_records(
        settlements=settled_snapshot.value.forecast_settlements,
        social_errors=social_pe_snapshot.value.errors,
        settled_at_turn=preaction.request.outcome_turn_index,
        timestamp_ms=settlement_input.credit_timestamp_ms,
    )
    matching_credits = tuple(
        credit
        for credit in credits
        if credit.prediction_id == preaction.forecast.forecast_id
    )
    if len(matching_credits) != 1:
        raise RuntimeError(
            "relationship product pulse must derive exactly one current PE credit"
        )
    credit = matching_credits[0]
    gate_update = (
        gate.observe_credit(credit)
        if settlement_input.apply_credit_to_gate
        else None
    )
    return RelationshipProductSettlementSnapshot(
        preaction=preaction,
        external_outcome_snapshot=external_snapshot,
        preference_snapshot=settled_snapshot,
        social_prediction_error_snapshot=social_pe_snapshot,
        settlement=current_settlements[0],
        credit=credit,
        gate_update=gate_update,
        owner_persistence_snapshot=store.export_persistence_snapshot(),
        gate_checkpoint=gate.export_checkpoint(),
        credit_applied_to_gate=settlement_input.apply_credit_to_gate,
    )


class _OutcomeEvidenceProposalRuntime(SemanticProposalRuntime):
    def __init__(
        self,
        *,
        evidence: PreferenceActionOutcomeEvidence,
        semantic_evidence_ref: str,
    ) -> None:
        self._evidence = evidence
        self._semantic_evidence_ref = semantic_evidence_ref
        self.runtime_id = (
            f"relationship-product-pulse-owner:{evidence.evidence_id}"
        )

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: object | None,
        memory_snapshot: object | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del substrate_snapshot, memory_snapshot, previous_snapshot
        if target_slot != _PREFERENCE_SLOT:
            raise ValueError(
                "relationship product outcome runtime only serves "
                "preference_about_other"
            )
        if turn_index != self._evidence.source_turn:
            raise ValueError("relationship product outcome turn lineage mismatch")
        if user_input != self._evidence.observation_summary:
            raise ValueError("relationship product outcome input drifted")
        return SemanticProposalBatch(
            proposals=(
                SemanticProposal(
                    proposal_id=self._evidence.evidence_id,
                    target_slot=_PREFERENCE_SLOT,
                    operation=SemanticProposalOperation.OBSERVE,
                    summary=self._evidence.observation_summary,
                    detail=self._evidence.reaction_summary,
                    confidence=0.90,
                    evidence=self._semantic_evidence_ref,
                    control_signal=0.0,
                ),
            ),
            runtime_id=self.runtime_id,
            schema_version=1,
            description=(
                "One typed relationship product outcome proposed to its owner."
            ),
        )


async def _apply_typed_environment_advisory(
    advisory: TemporalActionAdvisoryProposal,
    *,
    substrate_snapshot: SubstrateSnapshot,
) -> Snapshot[TemporalAbstractionSnapshot]:
    module = TrackTemporalModule(
        track=Track.SELF,
        policy=PlaceholderTemporalPolicy(),
        wiring_level=WiringLevel.ACTIVE,
        action_advisory=advisory,
        action_advisory_level=WiringLevel.ACTIVE,
    )
    snapshot = await module.process_standalone(
        substrate_snapshot=substrate_snapshot
    )
    if snapshot.value.action_advisory_status is not TemporalActionAdvisoryStatus.APPLIED:
        raise RuntimeError("relationship product typed advisory was not applied")
    return snapshot


def _validate_placeholder_substrate(snapshot: SubstrateSnapshot) -> None:
    if not isinstance(snapshot, SubstrateSnapshot):
        raise TypeError("substrate_snapshot must be a SubstrateSnapshot")
    if not snapshot.is_frozen:
        raise ValueError("relationship product pulse requires a frozen substrate")
    if snapshot.surface_kind is not SurfaceKind.PLACEHOLDER:
        raise ValueError(
            "relationship product pulse is typed simulation only and requires "
            "a placeholder substrate"
        )


def _validate_settlement_lineage(
    preaction: RelationshipProductPreActionSnapshot,
    settlement_input: RelationshipProductSettlementInput,
) -> None:
    request = preaction.request
    external = settlement_input.external_outcome
    owner_evidence = settlement_input.owner_outcome_evidence
    expected = (
        ("external outcome turn", external.turn_index, request.outcome_turn_index),
        (
            "external action turn",
            external.action_turn_index,
            request.forecast_request.turn_index,
        ),
        (
            "external session",
            external.session_scope,
            request.forecast_request.session_scope,
        ),
        ("external forecast", external.forecast_id, preaction.forecast.forecast_id),
        (
            "external decision",
            external.decision_id,
            request.forecast_request.decision_id,
        ),
        (
            "external action",
            external.action_id,
            preaction.gate_decision.selected_action_id,
        ),
        ("owner outcome turn", owner_evidence.source_turn, request.outcome_turn_index),
        (
            "owner interlocutor",
            owner_evidence.interlocutor_id,
            request.forecast_request.interlocutor_id,
        ),
        (
            "owner observation",
            owner_evidence.observation_summary,
            request.forecast_request.current_observation,
        ),
        (
            "owner action",
            owner_evidence.action_id,
            preaction.gate_decision.selected_action_id,
        ),
        (
            "owner typed outcome",
            owner_evidence.observed_outcome_id,
            external.kind.value,
        ),
    )
    for field_name, observed, wanted in expected:
        if observed != wanted:
            raise ValueError(f"{field_name} lineage mismatch")
    if external.evidence_ref not in owner_evidence.evidence_refs:
        raise ValueError("owner outcome does not cite external evidence")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


__all__ = [
    "RELATIONSHIP_PRODUCT_PULSE_SCHEMA_VERSION",
    "RelationshipProductOnboardingInput",
    "RelationshipProductOnboardingSnapshot",
    "RelationshipProductPreActionRequest",
    "RelationshipProductPreActionSnapshot",
    "RelationshipProductPulseAuthorization",
    "RelationshipProductSettlementInput",
    "RelationshipProductSettlementSnapshot",
    "append_relationship_product_onboarding",
    "authorize_relationship_product_pulse_advisory",
    "prepare_relationship_product_preaction",
    "settle_relationship_product_pulse",
]
