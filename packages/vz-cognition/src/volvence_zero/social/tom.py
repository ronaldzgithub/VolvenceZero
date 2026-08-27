"""Theory-of-Mind owners + structured LLM proposal runtime (R17).

This module bundles the four ToM owners and their structured LLM
proposal runtime. The runtime is a collaborator of the owners — not an
independent owner — so it lives in the same file rather than in a
separate ``_runtime.py`` shard.

Owner contract: each ToM module is the single owner of its own slot
(`belief_about_other` / `intent_about_other` / `feeling_about_other` /
`preference_about_other`). The LLM runtime only emits typed
:class:`SemanticProposal` records targeted at those slots; it does not
own state and it does not route renderer behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from json import JSONDecodeError
import math
from typing import Any, Mapping, Protocol

from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeKind,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.llm_proposal_diagnostics import LLMProposalAttemptCounters
from volvence_zero.memory import MemorySnapshot
from volvence_zero.owner_hydration import OwnerPersistenceSnapshot
from volvence_zero.runtime import (
    RuntimeModule,
    RuntimePlaceholderValue,
    Snapshot,
    WiringLevel,
)
from volvence_zero.semantic_state import (
    NoOpSemanticProposalRuntime,
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
)
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    FeelingAboutOtherSnapshot,
    IntentAboutOtherSnapshot,
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
    PreferenceActionOutcomeEvidence,
    PreferenceActionOutcomeMutation,
    PreferenceActionOutcomeMutationOperation,
    PreferenceActionOutcomeMutationReceipt,
    PreferenceActionForecast,
    PreferenceActionForecastSettlement,
    PreferenceAboutOtherSnapshot,
    RelationshipConditionReadout,
    SELF_INTERLOCUTOR_ID,
    SocialActionCandidatePrediction,
    SocialPrediction,
    SocialPredictionError,
    SocialPredictionKind,
    SocialPredictionOutcome,
    SocialScopeKind,
    preference_action_outcome_evidence_sha256,
    preference_action_outcome_mutation_sha256,
)
from volvence_zero.substrate import SubstrateSnapshot

from .record_store import (
    PendingSocialPrediction,
    SocialRecordStore,
    apply_outcome_to_record,
    settle_pending_predictions,
)

from volvence_zero.semantic_state._llm_proposal_counters import (
    LLMProposalAttemptAccumulator,
)

from ._llm_debug import log_proposal_attempt, make_attempt_logger
from ._llm_parsing import strip_code_fence


PREFERENCE_ACTION_RELATIONSHIP_UTILITY_SURFACE_ID = (
    "preference-action-relationship-outcome-utility.v1"
)
_PREFERENCE_ACTION_RELATIONSHIP_OUTCOME_UTILITIES = (
    (DialogueExternalOutcomeKind.HELPED.value, 1.0),
    (DialogueExternalOutcomeKind.FELT_HEARD.value, 1.0),
    (DialogueExternalOutcomeKind.MISSED.value, -1.0),
    (DialogueExternalOutcomeKind.OVER_DIRECTIVE.value, -1.0),
)


class _OtherMindOwnerModule(RuntimeModule[Any]):
    record_kind: OtherMindRecordKind
    snapshot_type: type[Any]
    empty_description: str
    dependencies = ("substrate", "memory", "multi_party_identity")
    default_wiring_level = WiringLevel.SHADOW
    min_proposal_confidence = 0.50
    prediction_kind: SocialPredictionKind

    def __init__(
        self,
        *,
        proposal_runtime: SemanticProposalRuntime | None = None,
        user_input: str | None = None,
        turn_index: int = 0,
        wiring_level: WiringLevel | None = None,
        record_store: SocialRecordStore | None = None,
    ) -> None:
        # W1.C (CP-16): the optional session-held ``record_store`` gives
        # this owner cross-turn records + pending-prediction settlement.
        # When None (unit tests / standalone probes) the owner keeps its
        # original per-turn stateless behavior.
        super().__init__(wiring_level=wiring_level)
        self._proposal_runtime = proposal_runtime
        self._user_input = user_input
        self._turn_index = turn_index
        self._record_store = record_store

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[Any]:
        new_records: tuple[OtherMindRecord, ...] = ()
        control_signal = 0.0
        if self._proposal_runtime is not None:
            substrate_snapshot = upstream.get("substrate")
            memory_snapshot = upstream.get("memory")
            batch = self._proposal_runtime.propose(
                target_slot=self.slot_name,
                user_input=self._user_input,
                substrate_snapshot=(
                    substrate_snapshot.value
                    if substrate_snapshot is not None and isinstance(substrate_snapshot.value, SubstrateSnapshot)
                    else None
                ),
                memory_snapshot=(
                    memory_snapshot.value
                    if memory_snapshot is not None and isinstance(memory_snapshot.value, MemorySnapshot)
                    else None
                ),
                previous_snapshot=None,
                turn_index=self._turn_index,
            )
            proposals = tuple(
                proposal
                for proposal in batch.proposals
                if proposal.target_slot == self.slot_name and proposal.confidence >= self.min_proposal_confidence
            )
            new_records = tuple(
                _record_from_proposal(
                    proposal=proposal,
                    kind=self.record_kind,
                    turn_index=self._turn_index,
                )
                for proposal in proposals
            )
            control_signal = _mean_control_signal(proposals)
        records, settled_errors = self._settle_and_merge(new_records)
        proposal_diagnostics = self._extract_proposal_diagnostics()
        return self.publish(
            self._snapshot(
                records=records,
                control_signal=control_signal,
                proposal_diagnostics=proposal_diagnostics,
                settled_errors=settled_errors,
            )
        )

    def _settle_and_merge(
        self, new_records: tuple[OtherMindRecord, ...]
    ) -> tuple[tuple[OtherMindRecord, ...], tuple[SocialPredictionError, ...]]:
        """Cross-turn settlement + promote/retire (W1.C, CP-16 core).

        Without a store the owner stays stateless: this turn's records
        pass through unchanged and nothing settles.
        """

        store = self._record_store
        if store is None:
            return (new_records, ())
        prior_records = store.tom_records(self.slot_name)
        pending = store.pending_tom_predictions(self.slot_name)
        evidence_by_scope: dict[str, tuple[tuple[str, str], ...]] = {}
        for record in new_records:
            evidence_by_scope[record.interlocutor_id] = (
                *evidence_by_scope.get(record.interlocutor_id, ()),
                (record.record_id, record.summary),
            )
        result = settle_pending_predictions(
            pending=pending,
            new_evidence_by_scope=evidence_by_scope,
            turn_index=self._turn_index,
            owner=self.owner,
            similarity=store.similarity,
        )
        outcome_by_record = {
            record_id: (outcome, error_id) for record_id, outcome, error_id in result.outcomes_by_record
        }
        updated_prior = tuple(
            apply_outcome_to_record(
                record,
                outcome_by_record[record.record_id][0],
                error_id=outcome_by_record[record.record_id][1],
            )
            if record.record_id in outcome_by_record
            else record
            for record in prior_records
        )
        merged_by_id: dict[str, OtherMindRecord] = {}
        for record in (*updated_prior, *new_records):
            merged_by_id[record.record_id] = record
        merged = tuple(merged_by_id.values())
        store.set_tom_records(self.slot_name, merged)
        # Rebuild the pending window: still-ambiguous entries keep their
        # original issue turn; every ACTIVE / CONTESTED record without a
        # pending entry issues a fresh prediction (CONTESTED must remain
        # settleable so a second disconfirmation can retire it).
        pending_by_record = {entry.source_record_id: entry for entry in result.still_pending}
        for record in store.tom_records(self.slot_name):
            if record.status is OtherMindRecordStatus.RETIRED:
                continue
            if record.record_id in pending_by_record:
                continue
            pending_by_record[record.record_id] = PendingSocialPrediction(
                prediction=self._prediction_for_record(record),
                source_record_id=record.record_id,
                issued_turn=self._turn_index,
            )
        store.set_pending_tom_predictions(self.slot_name, tuple(pending_by_record.values()))
        return (store.tom_records(self.slot_name), result.settled_errors)

    def _extract_proposal_diagnostics(self) -> LLMProposalAttemptCounters | None:
        """Return the runtime's typed counters when available.

        Returns ``None`` when:
        * No proposal runtime is wired (NoOp / scaffold paths).
        * The wired runtime is not LLM-backed (no ``attempt_counters``).
          We duck-check on the attribute name rather than ``isinstance``
          to keep this owner agnostic to which LLM-backed runtime
          subclass is wired (e.g. test fakes or future variants that
          implement the same counters protocol).
        """
        runtime = self._proposal_runtime
        if runtime is None:
            return None
        counters = getattr(runtime, "attempt_counters", None)
        if isinstance(counters, LLMProposalAttemptCounters):
            return counters
        return None

    def _snapshot(
        self,
        *,
        records: tuple[OtherMindRecord, ...],
        control_signal: float,
        proposal_diagnostics: LLMProposalAttemptCounters | None,
        settled_errors: tuple[SocialPredictionError, ...] = (),
    ) -> Any:
        return self.snapshot_type(
            records=records,
            active_predictions=self._active_predictions(records),
            control_signal=control_signal,
            description=(
                self.empty_description
                if not records
                else (f"{self.owner} published explicit records={len(records)} settled={len(settled_errors)}.")
            ),
            proposal_diagnostics=proposal_diagnostics,
            settled_errors=settled_errors,
        )

    def _prediction_for_record(self, record: OtherMindRecord) -> SocialPrediction:
        return SocialPrediction(
            prediction_id=f"{self.slot_name}:{record.record_id}:prediction",
            kind=self.prediction_kind,
            scope_kind=SocialScopeKind.INTERLOCUTOR,
            scope_id=record.interlocutor_id,
            subject_ids=(record.interlocutor_id,),
            audience_ids=(SELF_INTERLOCUTOR_ID,),
            predicted_outcome=record.summary,
            confidence=record.confidence,
            evidence=(
                f"tom_record:{record.record_id}",
                f"tom_kind:{record.kind.value}",
                record.evidence,
            ),
        )

    def _active_predictions(self, records: tuple[OtherMindRecord, ...]) -> tuple[SocialPrediction, ...]:
        """Publish owner-authored ToM predictions from typed records.

        W1.C: only ACTIVE records predict publicly. CONTESTED records
        stay settleable in the pending store (so a second
        disconfirmation retires them) but do not assert predictions;
        RETIRED records neither predict nor pend.
        """

        return tuple(
            self._prediction_for_record(record) for record in records if record.status is OtherMindRecordStatus.ACTIVE
        )


class BeliefAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "belief_about_other"
    owner = "BeliefAboutOtherModule"
    value_type = BeliefAboutOtherSnapshot
    record_kind = OtherMindRecordKind.BELIEF
    prediction_kind = SocialPredictionKind.BELIEF_ABOUT_OTHER
    snapshot_type = BeliefAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no belief-about-other records yet."


class IntentAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "intent_about_other"
    owner = "IntentAboutOtherModule"
    value_type = IntentAboutOtherSnapshot
    record_kind = OtherMindRecordKind.INTENT
    prediction_kind = SocialPredictionKind.INTENT_ABOUT_OTHER
    snapshot_type = IntentAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no intent-about-other records yet."


class FeelingAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "feeling_about_other"
    owner = "FeelingAboutOtherModule"
    value_type = FeelingAboutOtherSnapshot
    record_kind = OtherMindRecordKind.FEELING
    prediction_kind = SocialPredictionKind.FEELING_ABOUT_OTHER
    snapshot_type = FeelingAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no feeling-about-other records yet."


class PreferenceAboutOtherModule(_OtherMindOwnerModule):
    slot_name = "preference_about_other"
    owner = "PreferenceAboutOtherModule"
    value_type = PreferenceAboutOtherSnapshot
    record_kind = OtherMindRecordKind.PREFERENCE
    prediction_kind = SocialPredictionKind.PREFERENCE_ABOUT_OTHER
    snapshot_type = PreferenceAboutOtherSnapshot
    empty_description = "R17 SHADOW scaffold: no preference-about-other records yet."
    dependencies = (*_OtherMindOwnerModule.dependencies, "dialogue_external_outcome")

    def __init__(
        self,
        *,
        proposal_runtime: SemanticProposalRuntime | None = None,
        user_input: str | None = None,
        turn_index: int = 0,
        wiring_level: WiringLevel | None = None,
        record_store: SocialRecordStore | None = None,
        action_forecast_runtime: "PreferenceActionForecastRuntime | None" = None,
        action_forecast_request: "PreferenceActionForecastRequest | None" = None,
        action_outcome_evidence: PreferenceActionOutcomeEvidence | None = None,
        action_outcome_mutation: PreferenceActionOutcomeMutation | None = None,
    ) -> None:
        super().__init__(
            proposal_runtime=proposal_runtime,
            user_input=user_input,
            turn_index=turn_index,
            wiring_level=wiring_level,
            record_store=record_store,
        )
        if (action_forecast_runtime is None) != (action_forecast_request is None):
            raise ValueError("preference action forecast runtime and request must be provided together")
        if action_forecast_runtime is not None and self.wiring_level is not WiringLevel.SHADOW:
            raise ValueError("preference action forecasts are P2-development SHADOW-only")
        self._action_forecast_runtime = action_forecast_runtime
        self._action_forecast_request = action_forecast_request
        if action_outcome_evidence is not None and action_outcome_evidence.source_turn != turn_index:
            raise ValueError("preference action outcome evidence source_turn must match turn_index")
        if action_outcome_mutation is not None:
            if record_store is None:
                raise ValueError("preference action outcome mutation requires SocialRecordStore")
            if action_outcome_mutation.requested_turn != turn_index:
                raise ValueError("preference action outcome mutation requested_turn must match turn_index")
            if action_outcome_evidence is not None:
                raise ValueError(
                    "preference action outcome mutation cannot share a turn with new action outcome evidence"
                )
            if proposal_runtime is not None:
                raise ValueError("preference action outcome mutation cannot share a turn with a ToM proposal")
        self._action_outcome_evidence = action_outcome_evidence
        self._action_outcome_mutation = action_outcome_mutation
        self._external_outcome_snapshot: DialogueExternalOutcomeSnapshot | None = None

    async def process(
        self,
        upstream: Mapping[str, Snapshot[Any]],
    ) -> Snapshot[PreferenceAboutOtherSnapshot]:
        self._validate_incoming_action_outcome_against_receipts()
        external = upstream.get("dialogue_external_outcome")
        if external is not None and not isinstance(
            external.value,
            RuntimePlaceholderValue,
        ):
            if not isinstance(external.value, DialogueExternalOutcomeSnapshot):
                raise TypeError("preference_about_other expected DialogueExternalOutcomeSnapshot")
            self._external_outcome_snapshot = external.value
            if self._action_outcome_mutation is not None and external.value.entries:
                raise ValueError(
                    "preference action outcome mutation cannot share a turn with external forecast settlement"
                )
        return await super().process(upstream)

    def _snapshot(
        self,
        *,
        records: tuple[OtherMindRecord, ...],
        control_signal: float,
        proposal_diagnostics: LLMProposalAttemptCounters | None,
        settled_errors: tuple[SocialPredictionError, ...] = (),
    ) -> PreferenceAboutOtherSnapshot:
        records = self._apply_action_outcome_mutation(records)
        action_outcomes = self._merge_action_outcome_evidence(records)
        settlement_errors = self._settle_pending_action_forecasts()
        new_forecasts = self._action_forecasts(records, action_outcomes)
        pending_forecasts = self._merge_pending_action_forecasts(new_forecasts)
        settlements = self._record_store.preference_forecast_settlements if self._record_store is not None else ()
        return PreferenceAboutOtherSnapshot(
            records=records,
            active_predictions=self._active_predictions(records),
            control_signal=control_signal,
            description=(
                self.empty_description
                if not records
                else (f"{self.owner} published explicit records={len(records)} settled={len(settled_errors)}.")
            ),
            proposal_diagnostics=proposal_diagnostics,
            settled_errors=(*settled_errors, *settlement_errors),
            action_forecasts=pending_forecasts,
            action_outcome_evidence=action_outcomes,
            forecast_settlements=settlements,
            action_outcome_mutation_receipts=(
                self._record_store.preference_action_outcome_mutation_receipts if self._record_store is not None else ()
            ),
        )

    def _validate_incoming_action_outcome_against_receipts(self) -> None:
        incoming = self._action_outcome_evidence
        store = self._record_store
        if incoming is None or store is None:
            return
        latest = next(
            (
                receipt
                for receipt in reversed(store.preference_action_outcome_mutation_receipts)
                if receipt.target_evidence_id == incoming.evidence_id
            ),
            None,
        )
        if latest is None:
            return
        if latest.operation is PreferenceActionOutcomeMutationOperation.REDACT:
            raise ValueError("redacted preference action outcome evidence cannot be reintroduced")
        if preference_action_outcome_evidence_sha256(incoming) != latest.after_evidence_sha256:
            raise ValueError("stale preference action outcome evidence cannot overwrite a correction")

    def _apply_action_outcome_mutation(
        self,
        records: tuple[OtherMindRecord, ...],
    ) -> tuple[OtherMindRecord, ...]:
        mutation = self._action_outcome_mutation
        if mutation is None:
            return records
        store = self._record_store
        if store is None:
            raise RuntimeError("preference action outcome mutation lost its store")

        command_sha256 = preference_action_outcome_mutation_sha256(mutation)
        prior_receipt = next(
            (
                receipt
                for receipt in store.preference_action_outcome_mutation_receipts
                if receipt.mutation_id == mutation.mutation_id
            ),
            None,
        )
        if prior_receipt is not None:
            if prior_receipt.command_sha256 != command_sha256:
                raise ValueError("preference action outcome mutation id was reused for a different command")
            return store.tom_records(self.slot_name)

        outcomes_by_id = {item.evidence_id: item for item in store.preference_action_outcomes}
        try:
            target_outcome = outcomes_by_id[mutation.target_evidence_id]
        except KeyError as exc:
            raise ValueError(
                f"preference action outcome mutation targets unknown evidence {mutation.target_evidence_id!r}"
            ) from exc
        before_sha256 = preference_action_outcome_evidence_sha256(target_outcome)
        if before_sha256 != mutation.expected_evidence_sha256:
            raise ValueError("preference action outcome mutation expected hash does not match current evidence")
        if mutation.requested_turn < target_outcome.source_turn:
            raise ValueError("preference action outcome mutation cannot precede source evidence")
        records_by_id = {record.record_id: record for record in records}
        try:
            target_record = records_by_id[mutation.target_evidence_id]
        except KeyError as exc:
            raise ValueError("preference action outcome mutation target has no owner record") from exc

        invalidated_forecasts = tuple(
            forecast
            for forecast in store.preference_action_forecasts
            if mutation.target_evidence_id in forecast.source_record_ids
        )
        remaining_forecasts = tuple(
            forecast
            for forecast in store.preference_action_forecasts
            if mutation.target_evidence_id not in forecast.source_record_ids
        )
        if mutation.operation is PreferenceActionOutcomeMutationOperation.CORRECT:
            replacement = mutation.replacement
            if replacement is None:
                raise RuntimeError("validated correction lost replacement evidence")
            if replacement.interlocutor_id != target_outcome.interlocutor_id:
                raise ValueError("correction cannot change preference evidence interlocutor")
            if replacement.action_id != target_outcome.action_id:
                raise ValueError("correction cannot change the exposed action")
            if replacement.source_turn != target_outcome.source_turn:
                raise ValueError("correction cannot change source turn lineage")
            corrected_record = replace(
                target_record,
                summary=replacement.observation_summary,
                detail=replacement.reaction_summary,
                evidence=replacement.evidence_refs[0],
            )
            next_records = tuple(
                corrected_record if record.record_id == mutation.target_evidence_id else record for record in records
            )
            next_outcomes = tuple(
                replacement if item.evidence_id == mutation.target_evidence_id else item
                for item in store.preference_action_outcomes
            )
            next_pending = tuple(
                PendingSocialPrediction(
                    prediction=self._prediction_for_record(corrected_record),
                    source_record_id=corrected_record.record_id,
                    issued_turn=self._turn_index,
                )
                if item.source_record_id == mutation.target_evidence_id
                else item
                for item in store.pending_tom_predictions(self.slot_name)
            )
            after_sha256: str | None = preference_action_outcome_evidence_sha256(replacement)
        else:
            next_records = tuple(record for record in records if record.record_id != mutation.target_evidence_id)
            next_outcomes = tuple(
                item for item in store.preference_action_outcomes if item.evidence_id != mutation.target_evidence_id
            )
            next_pending = tuple(
                item
                for item in store.pending_tom_predictions(self.slot_name)
                if item.source_record_id != mutation.target_evidence_id
            )
            after_sha256 = None

        receipt = PreferenceActionOutcomeMutationReceipt(
            mutation_id=mutation.mutation_id,
            command_sha256=command_sha256,
            target_evidence_id=mutation.target_evidence_id,
            operation=mutation.operation,
            before_evidence_sha256=before_sha256,
            after_evidence_sha256=after_sha256,
            applied_turn=self._turn_index,
            invalidated_forecast_ids=tuple(forecast.forecast_id for forecast in invalidated_forecasts),
            evidence_refs=mutation.evidence_refs,
        )
        store.replace_preference_action_mutation_state(
            records=next_records,
            pending_predictions=next_pending,
            action_outcomes=next_outcomes,
            action_forecasts=remaining_forecasts,
            receipts=(
                *store.preference_action_outcome_mutation_receipts,
                receipt,
            ),
        )
        return store.tom_records(self.slot_name)

    def _merge_action_outcome_evidence(
        self,
        records: tuple[OtherMindRecord, ...],
    ) -> tuple[PreferenceActionOutcomeEvidence, ...]:
        store = self._record_store
        prior = store.preference_action_outcomes if store is not None else ()
        incoming = self._action_outcome_evidence
        merged_by_id = {item.evidence_id: item for item in prior}
        if incoming is not None:
            merged_by_id[incoming.evidence_id] = incoming
        record_ids = {record.record_id for record in records}
        merged = tuple(item for item in merged_by_id.values() if item.evidence_id in record_ids)
        if incoming is not None and incoming.evidence_id not in record_ids:
            raise ValueError("preference action outcome evidence must reference a record published by this owner turn")
        if store is not None:
            store.set_preference_action_outcomes(merged)
            return store.preference_action_outcomes
        return merged

    def _settle_pending_action_forecasts(
        self,
    ) -> tuple[SocialPredictionError, ...]:
        store = self._record_store
        external = self._external_outcome_snapshot
        if store is None or external is None or not external.entries:
            return ()
        pending_by_id = {forecast.forecast_id: forecast for forecast in store.preference_action_forecasts}
        settlements_by_forecast = {
            settlement.forecast_id: settlement for settlement in store.preference_forecast_settlements
        }
        errors: list[SocialPredictionError] = []
        for entry in external.entries:
            if not entry.has_preference_forecast_join:
                continue
            if entry.forecast_id in settlements_by_forecast:
                prior = settlements_by_forecast[entry.forecast_id]
                if prior.source_evidence_id != entry.evidence_id:
                    raise ValueError("preference forecast already settled by different evidence")
                continue
            try:
                forecast = pending_by_id[entry.forecast_id]
            except KeyError as exc:
                raise ValueError(
                    f"external outcome references an unknown pending preference forecast {entry.forecast_id!r}"
                ) from exc
            settlement = settle_preference_action_forecast(
                forecast=forecast,
                evidence=entry,
            )
            settlements_by_forecast[forecast.forecast_id] = settlement
            del pending_by_id[forecast.forecast_id]
            errors.append(
                social_prediction_error_from_preference_action_forecast_settlement(
                    settlement
                )
            )
        store.set_preference_action_forecasts(tuple(pending_by_id.values()))
        store.set_preference_forecast_settlements(tuple(settlements_by_forecast.values()))
        return tuple(errors)

    def _merge_pending_action_forecasts(
        self,
        new_forecasts: tuple[PreferenceActionForecast, ...],
    ) -> tuple[PreferenceActionForecast, ...]:
        store = self._record_store
        prior = store.preference_action_forecasts if store is not None else ()
        by_id = {forecast.forecast_id: forecast for forecast in prior}
        for forecast in new_forecasts:
            existing = by_id.get(forecast.forecast_id)
            if existing is not None and existing != forecast:
                raise ValueError("preference action forecast id collision")
            by_id[forecast.forecast_id] = forecast
        merged = tuple(by_id.values())
        if store is not None:
            store.set_preference_action_forecasts(merged)
            return store.preference_action_forecasts
        return merged

    def _action_forecasts(
        self,
        records: tuple[OtherMindRecord, ...],
        action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...],
    ) -> tuple[PreferenceActionForecast, ...]:
        runtime = self._action_forecast_runtime
        request = self._action_forecast_request
        if runtime is None or request is None:
            return ()
        eligible_records = tuple(
            record
            for record in records
            if record.interlocutor_id == request.interlocutor_id and record.status is OtherMindRecordStatus.ACTIVE
        )
        eligible_action_outcomes = tuple(
            item for item in action_outcomes if item.interlocutor_id == request.interlocutor_id
        )
        proposal = runtime.propose(
            request=request,
            records=eligible_records,
            action_outcomes=eligible_action_outcomes,
        )
        if proposal is None:
            return ()
        action_ids = tuple(prediction.action_id for prediction in proposal.candidate_predictions)
        if action_ids != request.candidate_action_ids:
            raise ValueError("preference forecast proposal action surface does not match request")
        for prediction in proposal.candidate_predictions:
            outcome_ids = tuple(item.outcome_id for item in prediction.outcomes)
            if outcome_ids != request.outcome_ids:
                raise ValueError("preference forecast proposal outcome surface does not match request")
        eligible_ids = {record.record_id for record in eligible_records}
        unknown_source_ids = set(proposal.source_record_ids).difference(eligible_ids)
        if unknown_source_ids:
            raise ValueError(
                f"preference forecast proposal references ineligible owner records: {sorted(unknown_source_ids)!r}"
            )
        if proposal.condition_readout is not None:
            observation_sha256 = hashlib.sha256(
                request.current_observation.encode("utf-8")
            ).hexdigest()
            if (
                proposal.condition_readout.source_observation_sha256
                != observation_sha256
            ):
                raise ValueError(
                    "preference forecast condition readout is bound to a "
                    "different current observation"
                )
        evidence = tuple(
            dict.fromkeys(
                (
                    f"typed_observation:{request.observation_ref}",
                    *proposal.evidence,
                )
            )
        )
        return (
            PreferenceActionForecast(
                forecast_id=(f"{self.slot_name}:{request.decision_id}:forecast:{request.turn_index}"),
                decision_id=request.decision_id,
                interlocutor_id=request.interlocutor_id,
                candidate_predictions=proposal.candidate_predictions,
                recommended_action_id=proposal.recommended_action_id,
                confidence=proposal.confidence,
                source_record_ids=proposal.source_record_ids,
                issued_turn=request.turn_index,
                evidence=evidence,
                session_scope=request.session_scope,
                condition_readout=proposal.condition_readout,
            ),
        )


@dataclass(frozen=True)
class PreferenceActionForecastRequest:
    """Typed pre-action surface visible to the preference owner collaborator."""

    decision_id: str
    interlocutor_id: str
    current_observation: str
    observation_ref: str
    candidate_action_ids: tuple[str, ...]
    outcome_ids: tuple[str, ...]
    turn_index: int
    session_scope: str = ""

    def __post_init__(self) -> None:
        for field_name, value in (
            ("decision_id", self.decision_id),
            ("interlocutor_id", self.interlocutor_id),
            ("current_observation", self.current_observation),
            ("observation_ref", self.observation_ref),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        _require_unique_texts(
            "candidate_action_ids",
            self.candidate_action_ids,
            minimum=2,
        )
        _require_unique_texts("outcome_ids", self.outcome_ids, minimum=1)
        if isinstance(self.turn_index, bool) or not isinstance(self.turn_index, int):
            raise ValueError("turn_index must be an integer")
        if self.turn_index < 0:
            raise ValueError("turn_index must be >= 0")
        if self.session_scope and not self.session_scope.strip():
            raise ValueError("session_scope cannot be whitespace")


@dataclass(frozen=True)
class PreferenceActionForecastProposal:
    """Non-owning proposal; the preference owner stamps public lineage."""

    candidate_predictions: tuple[SocialActionCandidatePrediction, ...]
    recommended_action_id: str
    confidence: float
    source_record_ids: tuple[str, ...]
    evidence: tuple[str, ...]
    condition_readout: RelationshipConditionReadout | None = None

    def __post_init__(self) -> None:
        action_ids = tuple(item.action_id for item in self.candidate_predictions)
        _require_unique_texts(
            "candidate_predictions.action_id",
            action_ids,
            minimum=2,
        )
        if self.recommended_action_id not in action_ids:
            raise ValueError("recommended_action_id must name one of candidate_predictions")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        _require_unique_texts(
            "source_record_ids",
            self.source_record_ids,
            minimum=0,
        )
        _require_unique_texts("evidence", self.evidence, minimum=1)


class PreferenceActionForecastRuntime(Protocol):
    """Proposal collaborator for the single preference owner."""

    runtime_id: str

    def propose(
        self,
        *,
        request: PreferenceActionForecastRequest,
        records: tuple[OtherMindRecord, ...],
        action_outcomes: tuple[PreferenceActionOutcomeEvidence, ...],
    ) -> PreferenceActionForecastProposal | None: ...


def settle_preference_action_forecast(
    *,
    forecast: PreferenceActionForecast,
    evidence: DialogueExternalOutcomeEvidence,
) -> PreferenceActionForecastSettlement:
    """Derive the preference owner's exact settlement from frozen inputs.

    This pure owner API lets evidence consumers verify a published settlement
    without reimplementing the owner's probability, utility, or PE math.
    """

    if not forecast.session_scope:
        raise ValueError("preference forecast is unscoped and cannot be settled")
    if evidence.session_scope != forecast.session_scope:
        raise ValueError("preference forecast settlement session_scope mismatch")
    if evidence.forecast_id != forecast.forecast_id:
        raise ValueError("preference forecast settlement forecast_id mismatch")
    if evidence.decision_id != forecast.decision_id:
        raise ValueError("preference forecast settlement decision_id mismatch")
    if evidence.action_turn_index != forecast.issued_turn:
        raise ValueError("preference forecast settlement action turn mismatch")
    candidates_by_action = {candidate.action_id: candidate for candidate in forecast.candidate_predictions}
    try:
        candidate = candidates_by_action[evidence.action_id]
    except KeyError as exc:
        raise ValueError("preference forecast settlement action is outside forecast surface") from exc
    probabilities_by_outcome = {item.outcome_id: item.probability for item in candidate.outcomes}
    try:
        probability = probabilities_by_outcome[evidence.kind.value]
    except KeyError as exc:
        raise ValueError("preference forecast settlement outcome is outside forecast surface") from exc
    epsilon = 1e-12
    negative_log_likelihood = -math.log(max(epsilon, probability))
    try:
        observed_utility = preference_action_relationship_outcome_utility(
            evidence.kind.value
        )
        expected_utility = preference_action_forecast_expected_utility(
            forecast=forecast,
            action_id=evidence.action_id,
        )
    except ValueError as exc:
        raise ValueError(
            "preference forecast settlement requires the frozen relationship outcome utility surface"
        ) from exc
    signed_utility_prediction_error = max(
        -1.0,
        min(1.0, (observed_utility - expected_utility) / 2.0),
    )
    uniform_probability = 1.0 / len(candidate.outcomes)
    log_score = math.log(max(epsilon, probability) / uniform_probability)
    normalized_score = max(
        -1.0,
        min(1.0, log_score / math.log(len(candidate.outcomes))),
    )
    if normalized_score > 1e-12:
        outcome = SocialPredictionOutcome.CONFIRMED
    elif normalized_score < -1e-12:
        outcome = SocialPredictionOutcome.DISCONFIRMED
    else:
        outcome = SocialPredictionOutcome.UNKNOWN
    return PreferenceActionForecastSettlement(
        settlement_id=f"{forecast.forecast_id}:settled:{evidence.evidence_id}",
        forecast_id=forecast.forecast_id,
        decision_id=forecast.decision_id,
        session_scope=forecast.session_scope,
        interlocutor_id=forecast.interlocutor_id,
        action_id=evidence.action_id,
        observed_outcome_id=evidence.kind.value,
        predicted_probability=probability,
        negative_log_likelihood=negative_log_likelihood,
        outcome=outcome,
        magnitude=abs(normalized_score),
        source_evidence_id=evidence.evidence_id,
        forecast_issued_turn=forecast.issued_turn,
        observed_turn=evidence.turn_index,
        evidence_confidence=evidence.confidence,
        expected_utility=expected_utility,
        observed_utility=observed_utility,
        signed_utility_prediction_error=signed_utility_prediction_error,
    )


def replay_preference_action_forecast_publication_persistence(
    *,
    before: OwnerPersistenceSnapshot,
    forecast: PreferenceActionForecast,
) -> OwnerPersistenceSnapshot:
    """Replay one exact pending-forecast publication through its owner store.

    This is the owner-side inverse of the product pulse's pre-action receipt:
    every stable field is retained and the exact forecast is appended through
    the owner's bounded merge semantics.  Consumers therefore do not need to
    inspect or reconstruct the opaque persistence payload.
    """

    if not isinstance(before, OwnerPersistenceSnapshot):
        raise TypeError("before must be OwnerPersistenceSnapshot")
    if not isinstance(forecast, PreferenceActionForecast):
        raise TypeError("forecast must be PreferenceActionForecast")
    store = SocialRecordStore()
    store.hydrate_from_persistence(before)
    if any(
        item.forecast_id == forecast.forecast_id
        for item in store.preference_action_forecasts
    ):
        raise ValueError("owner publication pre-state already contains forecast id")
    eligible_record_ids = {
        record.record_id
        for record in store.tom_records(PreferenceAboutOtherModule.slot_name)
        if record.interlocutor_id == forecast.interlocutor_id
        and record.status is OtherMindRecordStatus.ACTIVE
    }
    unknown_source_ids = set(forecast.source_record_ids).difference(
        eligible_record_ids
    )
    if unknown_source_ids:
        raise ValueError(
            "owner publication forecast references ineligible records: "
            f"{sorted(unknown_source_ids)!r}"
        )
    store.set_preference_action_forecasts(
        (*store.preference_action_forecasts, forecast)
    )
    return store.export_persistence_snapshot()


def replay_preference_action_forecast_settlement_persistence(
    *,
    before: OwnerPersistenceSnapshot,
    forecast: PreferenceActionForecast,
    external_evidence: DialogueExternalOutcomeEvidence,
    owner_outcome_evidence: PreferenceActionOutcomeEvidence,
) -> OwnerPersistenceSnapshot:
    """Replay one product-shaped settlement through preference owner rules.

    The returned canonical persistence is the unique stable post-state for the
    given owner pre-state and exact typed evidence. It intentionally reuses the
    owner's record settlement, merge windows, forecast consumption, and
    settlement append logic so evidence consumers never duplicate those rules.
    """

    if not isinstance(before, OwnerPersistenceSnapshot):
        raise TypeError("before must be OwnerPersistenceSnapshot")
    if not isinstance(forecast, PreferenceActionForecast):
        raise TypeError("forecast must be PreferenceActionForecast")
    if not isinstance(external_evidence, DialogueExternalOutcomeEvidence):
        raise TypeError(
            "external_evidence must be DialogueExternalOutcomeEvidence"
        )
    if not isinstance(owner_outcome_evidence, PreferenceActionOutcomeEvidence):
        raise TypeError(
            "owner_outcome_evidence must be PreferenceActionOutcomeEvidence"
        )
    if owner_outcome_evidence.interlocutor_id != forecast.interlocutor_id:
        raise ValueError("owner outcome evidence interlocutor mismatch")
    if owner_outcome_evidence.source_turn != external_evidence.turn_index:
        raise ValueError("owner outcome evidence turn mismatch")
    if owner_outcome_evidence.action_id != external_evidence.action_id:
        raise ValueError("owner outcome evidence action mismatch")
    if owner_outcome_evidence.observed_outcome_id != external_evidence.kind.value:
        raise ValueError("owner outcome evidence outcome mismatch")
    if external_evidence.evidence_ref not in owner_outcome_evidence.evidence_refs:
        raise ValueError("owner outcome evidence does not cite external evidence")

    store = SocialRecordStore()
    store.hydrate_from_persistence(before)
    current_forecasts = tuple(
        item
        for item in store.preference_action_forecasts
        if item.forecast_id == forecast.forecast_id
    )
    if current_forecasts != (forecast,):
        raise ValueError(
            "owner transition pre-state must contain the exact pending forecast"
        )

    replay_owner = PreferenceAboutOtherModule(
        turn_index=forecast.issued_turn,
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
    )
    replay_records, replay_errors = replay_owner._settle_and_merge(())
    replay_owner._snapshot(
        records=replay_records,
        control_signal=0.0,
        proposal_diagnostics=None,
        settled_errors=replay_errors,
    )

    proposal = SemanticProposal(
        proposal_id=owner_outcome_evidence.evidence_id,
        target_slot=PreferenceAboutOtherModule.slot_name,
        operation=SemanticProposalOperation.OBSERVE,
        summary=owner_outcome_evidence.observation_summary,
        detail=owner_outcome_evidence.reaction_summary,
        confidence=0.90,
        evidence=external_evidence.evidence_ref,
        control_signal=0.0,
    )
    new_record = _record_from_proposal(
        proposal=proposal,
        kind=PreferenceAboutOtherModule.record_kind,
        turn_index=owner_outcome_evidence.source_turn,
    )
    settlement_owner = PreferenceAboutOtherModule(
        turn_index=owner_outcome_evidence.source_turn,
        wiring_level=WiringLevel.SHADOW,
        record_store=store,
        action_outcome_evidence=owner_outcome_evidence,
    )
    settlement_owner._validate_incoming_action_outcome_against_receipts()
    settlement_owner._external_outcome_snapshot = DialogueExternalOutcomeSnapshot(
        turn_index=external_evidence.turn_index,
        entries=(external_evidence,),
        description="Owner replay of exact external settlement evidence.",
    )
    settled_records, settled_errors = settlement_owner._settle_and_merge(
        (new_record,)
    )
    settlement_owner._snapshot(
        records=settled_records,
        control_signal=0.0,
        proposal_diagnostics=None,
        settled_errors=settled_errors,
    )
    expected_settlement = settle_preference_action_forecast(
        forecast=forecast,
        evidence=external_evidence,
    )
    current_settlements = tuple(
        item
        for item in store.preference_forecast_settlements
        if item.forecast_id == forecast.forecast_id
    )
    if current_settlements != (expected_settlement,):
        raise RuntimeError(
            "preference owner replay did not publish the exact settlement"
        )
    return store.export_persistence_snapshot()


def preference_action_relationship_outcome_utility(outcome_id: str) -> float:
    """Return the preference owner's frozen utility for one typed outcome."""

    if not isinstance(outcome_id, str) or not outcome_id.strip():
        raise TypeError("outcome_id must be non-empty text")
    utilities = dict(_PREFERENCE_ACTION_RELATIONSHIP_OUTCOME_UTILITIES)
    try:
        return utilities[outcome_id]
    except KeyError as exc:
        raise ValueError(
            "outcome_id is outside the frozen preference-action utility surface"
        ) from exc


def preference_action_forecast_expected_utility(
    *,
    forecast: PreferenceActionForecast,
    action_id: str,
) -> float:
    """Interpret one action distribution through the preference owner's utility surface."""

    if not isinstance(forecast, PreferenceActionForecast):
        raise TypeError("forecast must be a PreferenceActionForecast")
    if not isinstance(action_id, str) or not action_id.strip():
        raise TypeError("action_id must be non-empty text")
    candidates = {
        candidate.action_id: candidate
        for candidate in forecast.candidate_predictions
    }
    try:
        candidate = candidates[action_id]
    except KeyError as exc:
        raise ValueError("action_id is outside the frozen forecast surface") from exc
    return math.fsum(
        preference_action_relationship_outcome_utility(item.outcome_id)
        * item.probability
        for item in candidate.outcomes
    )


def social_prediction_error_from_preference_action_forecast_settlement(
    settlement: PreferenceActionForecastSettlement,
) -> SocialPredictionError:
    """Publish the exact social PE associated with one owner settlement."""

    if not isinstance(settlement, PreferenceActionForecastSettlement):
        raise TypeError("settlement must be a PreferenceActionForecastSettlement")
    return SocialPredictionError(
        error_id=f"social-pe:{settlement.settlement_id}",
        prediction_id=settlement.forecast_id,
        kind=SocialPredictionKind.PREFERENCE_ABOUT_OTHER,
        outcome=settlement.outcome,
        magnitude=settlement.magnitude,
        owner=PreferenceAboutOtherModule.owner,
        scope_kind=SocialScopeKind.INTERLOCUTOR,
        scope_id=settlement.interlocutor_id,
        evidence=(
            f"forecast_settlement:{settlement.settlement_id}",
            f"external_outcome:{settlement.source_evidence_id}",
            f"action:{settlement.action_id}",
            f"observed_outcome:{settlement.observed_outcome_id}",
            f"predicted_probability={settlement.predicted_probability:.12f}",
            f"negative_log_likelihood={settlement.negative_log_likelihood:.12f}",
            f"signed_utility_prediction_error={settlement.signed_utility_prediction_error:.12f}",
        ),
    )


def _require_unique_texts(
    field_name: str,
    values: tuple[str, ...],
    *,
    minimum: int,
) -> None:
    if len(values) < minimum:
        raise ValueError(f"{field_name} must contain at least {minimum} entries")
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} entries must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} entries must be unique")


def _record_from_proposal(
    *,
    proposal: SemanticProposal,
    kind: OtherMindRecordKind,
    turn_index: int,
) -> OtherMindRecord:
    return OtherMindRecord(
        record_id=proposal.proposal_id,
        interlocutor_id="primary",
        kind=kind,
        summary=proposal.summary,
        detail=proposal.detail,
        confidence=proposal.confidence,
        status=OtherMindRecordStatus.ACTIVE,
        source_turn=turn_index,
        prediction_error_refs=(),
        evidence=proposal.evidence,
    )


def _mean_control_signal(proposals: tuple[SemanticProposal, ...]) -> float:
    if not proposals:
        return 0.0
    return sum(proposal.control_signal for proposal in proposals) / len(proposals)


# ---------------------------------------------------------------------------
# Structured LLM proposal runtime (collaborator of the four ToM owners above)
# ---------------------------------------------------------------------------


_TOM_TARGET_SLOTS: frozenset[str] = frozenset(
    {
        "belief_about_other",
        "intent_about_other",
        "feeling_about_other",
        "preference_about_other",
    }
)
_MIN_TOM_CONFIDENCE = 0.50


class _GenerateProtocol(Protocol):
    def generate(self, *, prompt: str, max_new_tokens: int = ..., temperature: float = ...) -> str: ...


@dataclass(frozen=True)
class _ToMDecision:
    target_slot: str
    summary: str
    detail: str
    evidence: str
    confidence: float
    control_signal: float


_TOM_PROMPT = (
    "You extract Theory-of-Mind observations from one dialogue turn.\n"
    "Return a JSON array. Each item must have exactly these fields:\n"
    "[\n"
    "  {{\n"
    '    "target_slot": "belief_about_other|intent_about_other|feeling_about_other|preference_about_other",\n'
    '    "summary": "short stable claim",\n'
    '    "detail": "specific evidence-aware detail",\n'
    '    "evidence": "short quote or observation from the user message",\n'
    '    "confidence": 0.0,\n'
    '    "control_signal": 0.0\n'
    "  }}\n"
    "]\n"
    "\n"
    "Do not infer demographics. Do not output markdown. If there is no "
    "clear Theory-of-Mind observation, return [].\n"
    "\n"
    "User message:\n"
    '"""\n'
    "{user_input}\n"
    '"""'
)


class LLMToMProposalRuntime(SemanticProposalRuntime):
    """Structured proposal source for R17 ToM owners."""

    runtime_id = "social-tom-llm-structured"

    def __init__(
        self,
        *,
        provider: _GenerateProtocol,
        base_runtime: SemanticProposalRuntime | None = None,
        max_new_tokens: int = 384,
    ) -> None:
        self._provider = provider
        self._base = base_runtime or NoOpSemanticProposalRuntime()
        self._max_new_tokens = max_new_tokens
        self._cache_key: tuple[str, int] | None = None
        self._cache_decisions: tuple[_ToMDecision, ...] | None = None
        # Opt-in diagnostic sink. ``None`` (the default) means the hot
        # path stays zero-overhead; setting ``VZ_LLM_PROPOSAL_DEBUG_LOG``
        # before host construction binds a JSONL append callable so a
        # diagnostic run can capture raw provider output + parse outcome
        # without changing constructor surface.
        self._debug_logger = make_attempt_logger()
        # Always-on typed counters (Wave E1). Owners read
        # ``attempt_counters`` and surface it on their snapshot so a
        # 0-records evidence run can be diagnosed without env vars.
        self._counters = LLMProposalAttemptAccumulator()

    @property
    def attempt_counters(self) -> LLMProposalAttemptCounters:
        """Return an immutable snapshot of cumulative LLM call counters.

        Owner modules read this each turn and republish on the typed
        snapshot's ``proposal_diagnostics`` field. The returned value
        is frozen; mutating callers must not assume identity.
        """
        return self._counters.snapshot()

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        previous_snapshot: object | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        if target_slot not in _TOM_TARGET_SLOTS or not user_input:
            return self._base.propose(
                target_slot=target_slot,
                user_input=user_input,
                substrate_snapshot=substrate_snapshot,
                memory_snapshot=memory_snapshot,
                previous_snapshot=previous_snapshot,
                turn_index=turn_index,
            )

        decisions = self._decisions_for_turn(user_input=user_input, turn_index=turn_index)
        proposals = tuple(
            SemanticProposal(
                proposal_id=f"{decision.target_slot}:tom-llm:{turn_index}:{index}",
                target_slot=decision.target_slot,
                operation=SemanticProposalOperation.OBSERVE,
                summary=decision.summary,
                detail=decision.detail,
                confidence=decision.confidence,
                evidence=decision.evidence,
                control_signal=decision.control_signal,
            )
            for index, decision in enumerate(decisions)
            if decision.target_slot == target_slot
        )
        return SemanticProposalBatch(
            proposals=proposals,
            runtime_id=self.runtime_id,
            schema_version=1,
            description=(
                f"Structured ToM runtime emitted {len(proposals)} proposal(s) for {target_slot} at turn {turn_index}."
            ),
        )

    def _decisions_for_turn(
        self,
        *,
        user_input: str,
        turn_index: int,
    ) -> tuple[_ToMDecision, ...]:
        cache_key = (user_input, turn_index)
        if self._cache_key == cache_key and self._cache_decisions is not None:
            return self._cache_decisions
        prompt = _TOM_PROMPT.format(user_input=user_input.strip()[:800])
        raw = self._provider.generate(
            prompt=prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=0.0,
        )
        decisions, parse_status, parse_error = _parse_tom_decisions_with_diag(raw)
        log_proposal_attempt(
            self._debug_logger,
            runtime_id=self.runtime_id,
            target_slot=None,
            turn_index=turn_index,
            prompt=prompt,
            raw_output=raw,
            parsed_count=len(decisions),
            parse_status=parse_status,
            parse_error=parse_error,
        )
        # ``parsed_count`` here is decisions surviving the strict schema
        # parser. Owner-side ``min_proposal_confidence`` may further
        # shrink the set; the runtime tracks the parse outcome and the
        # owner reports its own emission count via a separate path
        # (the snapshot still surfaces parse counters here so a parse
        # failure is not hidden behind owner-side filtering).
        self._counters.record_attempt(
            parse_status=parse_status,
            parse_error=parse_error,
            parsed_count=len(decisions),
            emitted_count=len(decisions),
        )
        self._cache_key = cache_key
        self._cache_decisions = decisions
        return decisions


def _parse_tom_decisions(text: str) -> tuple[_ToMDecision, ...] | None:
    decisions, status, _ = _parse_tom_decisions_with_diag(text)
    if status == "parse_error":
        return None
    return decisions


def _parse_tom_decisions_with_diag(
    text: str,
) -> tuple[tuple[_ToMDecision, ...], str, str | None]:
    """Parse with diagnostic categories; never raises.

    Returns ``(decisions, status, parse_error)`` where ``status`` is one
    of ``"ok"`` / ``"parse_error"`` / ``"empty_or_rejected"`` and
    ``parse_error`` is the JSONDecodeError message when applicable. Used
    by both the production parser (``_parse_tom_decisions``) and the
    diagnostic sink in ``LLMToMProposalRuntime``.
    """
    cleaned = strip_code_fence(text)
    try:
        payload = json.loads(cleaned.strip())
    except JSONDecodeError as exc:
        return ((), "parse_error", str(exc))
    if not isinstance(payload, list):
        return ((), "parse_error", f"top-level not a list: {type(payload).__name__}")
    decisions: list[_ToMDecision] = []
    for item in payload:
        decision = _parse_tom_decision(item)
        if decision is not None:
            decisions.append(decision)
    if not decisions:
        return ((), "empty_or_rejected", None)
    return (tuple(decisions), "ok", None)


def _parse_tom_decision(item: object) -> _ToMDecision | None:
    if not isinstance(item, dict):
        return None
    target_slot = item.get("target_slot")
    summary = item.get("summary")
    detail = item.get("detail")
    evidence = item.get("evidence")
    confidence = item.get("confidence")
    control_signal = item.get("control_signal", 0.0)
    if target_slot not in _TOM_TARGET_SLOTS:
        return None
    if not isinstance(summary, str) or not summary.strip():
        return None
    if not isinstance(detail, str) or not detail.strip():
        return None
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if isinstance(control_signal, bool) or not isinstance(control_signal, (int, float)):
        return None
    confidence_value = float(confidence)
    control_value = float(control_signal)
    if confidence_value < _MIN_TOM_CONFIDENCE or confidence_value > 1.0:
        return None
    if control_value < 0.0 or control_value > 1.0:
        return None
    return _ToMDecision(
        target_slot=target_slot,
        summary=summary.strip()[:160],
        detail=detail.strip()[:500],
        evidence=evidence.strip()[:240],
        confidence=confidence_value,
        control_signal=control_value,
    )


__all__ = [
    "PREFERENCE_ACTION_RELATIONSHIP_UTILITY_SURFACE_ID",
    "BeliefAboutOtherModule",
    "FeelingAboutOtherModule",
    "IntentAboutOtherModule",
    "LLMToMProposalRuntime",
    "PreferenceActionForecastProposal",
    "PreferenceActionForecastRequest",
    "PreferenceActionForecastRuntime",
    "PreferenceAboutOtherModule",
    "preference_action_forecast_expected_utility",
    "preference_action_relationship_outcome_utility",
    "replay_preference_action_forecast_publication_persistence",
    "replay_preference_action_forecast_settlement_persistence",
    "settle_preference_action_forecast",
    "social_prediction_error_from_preference_action_forecast_settlement",
]
