"""Session-local dialogue trace store.

This helper records replay/evidence traces. It deliberately reads the
published prediction-error result instead of owning prediction state or
classifying user text.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from volvence_zero.dialogue_trace import (
    DialogueActionKind,
    DialogueActionTrace,
    DialogueOutcomeEvidence,
    DialogueOutcomeKind,
    DialogueOutcomeResolution,
    DialogueOutcomeTrace,
    DialogueResolutionStatus,
    DialogueTraceSnapshot,
    build_unknown_dialogue_outcome,
)
from volvence_zero.environment import EnvironmentEvent
from volvence_zero.prediction.error import (
    ActualOutcome,
    PredictedOutcome,
    PredictionError,
)


class DialogueTraceStore:
    """Bounded session-local store for dialogue replay traces."""

    def __init__(self, *, max_traces: int = 128) -> None:
        self._max_traces = max_traces
        self._traces: list[DialogueActionTrace] = []
        self._resolved_outcomes: list[DialogueOutcomeTrace] = []
        self._unresolved_trace_ids: list[str] = []

    def record_action(
        self,
        *,
        session_id: str,
        wave_id: str,
        turn_index: int,
        environment_event: EnvironmentEvent,
        active_regime: str | None,
        active_abstract_action: str | None,
        response_text: str,
        response_rationale: str,
        next_prediction: PredictedOutcome | None,
        evaluated_prediction: PredictedOutcome | None,
        actual_outcome: ActualOutcome | None,
        prediction_error: PredictionError | None,
        outcome_evidence: tuple[DialogueOutcomeEvidence, ...] = (),
    ) -> tuple[DialogueActionTrace, DialogueOutcomeResolution | None]:
        trace_id = f"{session_id}:{wave_id}:dialogue-trace"
        unresolved_outcome = build_unknown_dialogue_outcome(
            previous_trace_id=trace_id,
            observed_trace_id=trace_id,
            observed_turn_index=turn_index,
        )
        trace = DialogueActionTrace(
            trace_id=trace_id,
            event_id=environment_event.event_id,
            wave_id=wave_id,
            turn_index=turn_index,
            action_kind=DialogueActionKind.ASSISTANT_RESPONSE,
            environment_frame=environment_event.frame,
            environment_event_kind=environment_event.event_kind.value,
            environment_trigger_kind=environment_event.trigger_kind,
            active_regime=active_regime,
            active_abstract_action=active_abstract_action,
            response_rationale=response_rationale,
            prediction_id=_prediction_ref(next_prediction),
            outcome=unresolved_outcome,
            response_text_hash=_stable_text_hash(response_text),
            description=(
                "Dialogue action trace records public turn evidence; "
                "prediction-error semantics remain owned by prediction_error."
            ),
        )
        self._traces.append(trace)
        self._unresolved_trace_ids.append(trace_id)
        resolution = self._resolve_previous_with_current(
            observed_trace=trace,
            evaluated_prediction=evaluated_prediction,
            actual_outcome=actual_outcome,
            prediction_error=prediction_error,
            outcome_evidence=outcome_evidence,
        )
        self._trim()
        return trace, resolution

    def snapshot(self) -> DialogueTraceSnapshot:
        return DialogueTraceSnapshot(
            traces=tuple(self._traces),
            unresolved_trace_ids=tuple(self._unresolved_trace_ids),
            resolved_outcomes=tuple(self._resolved_outcomes),
            description=(
                f"DialogueTraceStore published {len(self._traces)} trace(s), "
                f"unresolved={len(self._unresolved_trace_ids)}, "
                f"resolved={len(self._resolved_outcomes)}."
            ),
        )

    def export_replay_artifact(self) -> dict[str, object]:
        snapshot = self.snapshot()
        return {
            "trace_count": len(snapshot.traces),
            "unresolved_trace_ids": snapshot.unresolved_trace_ids,
            "resolved_outcome_count": len(snapshot.resolved_outcomes),
            "turns": tuple(
                {
                    "trace_id": trace.trace_id,
                    "event_id": trace.event_id,
                    "wave_id": trace.wave_id,
                    "turn_index": trace.turn_index,
                    "action_kind": trace.action_kind.value,
                    "regime": trace.active_regime or "",
                    "abstract_action": trace.active_abstract_action or "",
                    "prediction_id": trace.prediction_id or "",
                    "outcome_kind": trace.outcome.kind.value,
                    "structured_evidence_ids": tuple(
                        evidence.evidence_id
                        for evidence in trace.outcome.structured_evidence
                    ),
                    "response_text_hash": trace.response_text_hash,
                    "active_speaker_id": trace.environment_frame.active_speaker_id,
                    "subject_ids": trace.environment_frame.subject_ids,
                    "audience_ids": trace.environment_frame.audience_ids,
                }
                for trace in snapshot.traces
            ),
        }

    def _resolve_previous_with_current(
        self,
        *,
        observed_trace: DialogueActionTrace,
        evaluated_prediction: PredictedOutcome | None,
        actual_outcome: ActualOutcome | None,
        prediction_error: PredictionError | None,
        outcome_evidence: tuple[DialogueOutcomeEvidence, ...],
    ) -> DialogueOutcomeResolution | None:
        previous_trace_id = _previous_unresolved_id(
            self._unresolved_trace_ids,
            observed_trace.trace_id,
        )
        if previous_trace_id is None:
            return None
        if evaluated_prediction is None or actual_outcome is None or prediction_error is None:
            return None

        prediction_ref = _prediction_ref(evaluated_prediction)
        pe_ref = (
            f"prediction_error:{evaluated_prediction.source_turn_index}"
            f"->{evaluated_prediction.target_turn_index}"
        )
        outcome = build_unknown_dialogue_outcome(
            previous_trace_id=previous_trace_id,
            observed_trace_id=observed_trace.trace_id,
            observed_turn_index=observed_trace.turn_index,
            evidence_refs=(
                f"actual_outcome:{actual_outcome.observed_turn_index}",
                prediction_ref,
                *_evidence_refs_from_structured(outcome_evidence),
            ),
            prediction_error_refs=(pe_ref,),
            structured_evidence=outcome_evidence,
        )
        if outcome_evidence:
            outcome = replace(
                outcome,
                kind=_dialogue_outcome_kind_from_evidence(outcome_evidence),
                description=(
                    "Outcome kind derived from structured owner/evaluation "
                    "evidence; raw text was not inspected."
                ),
            )
        self._replace_trace_outcome(previous_trace_id=previous_trace_id, outcome=outcome)
        self._resolved_outcomes.append(outcome)
        self._unresolved_trace_ids = [
            trace_id
            for trace_id in self._unresolved_trace_ids
            if trace_id != previous_trace_id
        ]
        return DialogueOutcomeResolution(
            previous_trace_id=previous_trace_id,
            observed_trace_id=observed_trace.trace_id,
            status=DialogueResolutionStatus.RESOLVED,
            outcome=outcome,
            description=(
                "Resolved via published prediction_error snapshot; "
                "outcome kind remains conservative."
            ),
        )

    def _replace_trace_outcome(
        self,
        *,
        previous_trace_id: str,
        outcome: DialogueOutcomeTrace,
    ) -> None:
        self._traces = [
            replace(trace, outcome=outcome)
            if trace.trace_id == previous_trace_id
            else trace
            for trace in self._traces
        ]

    def _trim(self) -> None:
        if len(self._traces) <= self._max_traces:
            return
        kept_ids = {trace.trace_id for trace in self._traces[-self._max_traces :]}
        self._traces = self._traces[-self._max_traces :]
        self._unresolved_trace_ids = [
            trace_id for trace_id in self._unresolved_trace_ids if trace_id in kept_ids
        ]
        self._resolved_outcomes = [
            outcome
            for outcome in self._resolved_outcomes
            if outcome.previous_trace_id in kept_ids
        ]


def _previous_unresolved_id(
    unresolved_trace_ids: list[str],
    current_trace_id: str,
) -> str | None:
    for trace_id in reversed(unresolved_trace_ids):
        if trace_id != current_trace_id:
            return trace_id
    return None


def _prediction_ref(prediction: PredictedOutcome | None) -> str | None:
    if prediction is None:
        return None
    return f"prediction:{prediction.source_turn_index}->{prediction.target_turn_index}"


def _stable_text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _dialogue_outcome_kind_from_evidence(
    outcome_evidence: tuple[DialogueOutcomeEvidence, ...],
) -> DialogueOutcomeKind:
    if not outcome_evidence:
        return DialogueOutcomeKind.UNKNOWN
    strongest = max(outcome_evidence, key=lambda evidence: evidence.confidence)
    return strongest.outcome_kind


def _evidence_refs_from_structured(
    outcome_evidence: tuple[DialogueOutcomeEvidence, ...],
) -> tuple[str, ...]:
    refs: list[str] = []
    for evidence in outcome_evidence:
        refs.append(evidence.evidence_id)
        refs.extend(evidence.evidence_refs)
    # Preserve first occurrence order while keeping tuple uniqueness.
    return tuple(dict.fromkeys(refs))


__all__ = ["DialogueTraceStore"]
