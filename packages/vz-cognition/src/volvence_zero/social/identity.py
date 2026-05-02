"""Multi-party identity owner + social prediction lifters (R16).

R8 SSOT note (Slice 12 SSOT cleanup):

- ``MultiPartyIdentityModule`` owns the active identity scope snapshot.
- ``SocialPredictionAggregateModule`` and ``SocialPredictionErrorModule``
  are lifter / pass-through owners that publish the public
  :class:`SocialPrediction` / :class:`SocialPredictionError` snapshots
  by **reading typed PE signals** (``MemorySnapshot.social_pe_signals``)
  that the producing owner (currently ``MemoryModule``) emits itself.
  They never reconstruct social PE state from raw memory fields and
  they never borrow another module's owner name on their own published
  snapshots.

Earlier slices reconstructed ``MEMORY_VISIBILITY`` predictions and
errors here from ``MemorySnapshot.suppressed_cross_scope_entries``,
which violated the SSOT rule that consumers must not rebuild a
producer's internal state. That reconstruction has moved into the
producing ``MemoryModule`` itself, where the typed signal is published
through ``MemorySnapshot.social_pe_signals``.
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.environment import EnvironmentEvent
from volvence_zero.memory import MemorySnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    InterlocutorIdentity,
    MemorySocialPESignal,
    MultiPartyIdentitySnapshot,
    SocialPredictionError,
    SocialPredictionErrorSnapshot,
    SocialPredictionSnapshot,
    build_primary_multi_party_identity_snapshot,
    social_prediction_error_from_memory_signal,
    social_prediction_from_memory_signal,
)


class MultiPartyIdentityModule(RuntimeModule[MultiPartyIdentitySnapshot]):
    """Publishes the R16 identity scope contract.

    The initial implementation intentionally has no upstream dependencies:
    it is a compatibility scaffold that exposes the new contract in SHADOW
    without changing existing flat ``user_model`` / ``relationship_state``
    consumers.
    """

    slot_name = "multi_party_identity"
    owner = "MultiPartyIdentityModule"
    value_type = MultiPartyIdentitySnapshot
    dependencies: tuple[str, ...] = ()
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        environment_event: EnvironmentEvent | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._environment_event = environment_event

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[MultiPartyIdentitySnapshot]:
        if self._environment_event is not None:
            frame = self._environment_event.frame
            identities = {
                frame.active_speaker_id,
                *frame.addressee_ids,
                *frame.subject_ids,
                *frame.audience_ids,
            }
            return self.publish(
                MultiPartyIdentitySnapshot(
                    active_speaker_id=frame.active_speaker_id,
                    addressee_ids=frame.addressee_ids,
                    subject_ids=frame.subject_ids,
                    audience_ids=frame.audience_ids,
                    interlocutors=tuple(
                        InterlocutorIdentity(
                            interlocutor_id=interlocutor_id,
                            evidence=(self._environment_event.event_id,),
                        )
                        for interlocutor_id in sorted(identities)
                    ),
                    identity_predictions=(),
                    description=(
                        "R16 SHADOW scaffold: identity scope consumed from "
                        f"EnvironmentEvent {self._environment_event.event_id}."
                    ),
                )
            )
        return self.publish(
            build_primary_multi_party_identity_snapshot(
                description=(
                    "R16 SHADOW scaffold: single-interlocutor compatibility "
                    "identity scope under 'primary'."
                )
            )
        )


def _memory_snapshot(snapshot: Snapshot[Any] | None) -> MemorySnapshot | None:
    if snapshot is None or not isinstance(snapshot.value, MemorySnapshot):
        return None
    return snapshot.value


def _memory_social_pe_signals(
    snapshot: Snapshot[Any] | None,
) -> tuple[MemorySocialPESignal, ...]:
    memory = _memory_snapshot(snapshot)
    if memory is None:
        return ()
    return memory.social_pe_signals


def _describe_scope_state(memory: MemorySnapshot | None) -> str:
    """Derive a short scope-state hint for empty-signal descriptions."""

    if memory is None:
        return "default-or-missing"
    scope = memory.active_subject_scope
    if not scope:
        return "default-or-missing"
    if scope == ("primary",):
        return "default-or-missing"
    return f"non-default:{scope[0]}"


class SocialPredictionAggregateModule(RuntimeModule[SocialPredictionSnapshot]):
    """Lift typed PE signals from upstream owners into public predictions.

    R8 SSOT contract: this module never reconstructs social predictions
    from raw producer state. It only forwards typed signals (currently
    ``MemorySnapshot.social_pe_signals``) through the
    :func:`social_prediction_from_memory_signal` lifter. The producing
    owner (``MemoryModule``) is the single source of truth.
    """

    slot_name = "social_prediction"
    owner = "SocialPredictionAggregateModule"
    value_type = SocialPredictionSnapshot
    dependencies = ("multi_party_identity", "memory")
    default_wiring_level = WiringLevel.SHADOW

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[SocialPredictionSnapshot]:
        memory_snapshot = upstream.get("memory")
        memory_value = _memory_snapshot(memory_snapshot)
        signals = _memory_social_pe_signals(memory_snapshot)

        if not signals:
            memory_state = (
                "available" if memory_value is not None else "unavailable"
            )
            scope_state = _describe_scope_state(memory_value)
            return self.publish(
                SocialPredictionSnapshot(
                    predictions=(),
                    description=(
                        "R16 social prediction aggregate: no upstream signals; "
                        f"memory={memory_state}; scope={scope_state}."
                    ),
                )
            )

        retrieved_count = (
            len(memory_value.retrieved_entries) if memory_value is not None else 0
        )
        suppressed_count = (
            len(memory_value.suppressed_cross_scope_entries)
            if memory_value is not None
            else 0
        )
        predictions = tuple(
            social_prediction_from_memory_signal(
                signal,
                extra_evidence=(
                    f"retrieved_count={retrieved_count}",
                    f"suppressed_count={suppressed_count}",
                ),
            )
            for signal in signals
        )
        source_owners = sorted({signal.source_owner for signal in signals})
        return self.publish(
            SocialPredictionSnapshot(
                predictions=predictions,
                description=(
                    "R16 social prediction aggregate: lifted "
                    f"{len(predictions)} typed signal(s) from owner(s) "
                    f"{source_owners}."
                ),
            )
        )


class SocialPredictionErrorModule(RuntimeModule[SocialPredictionErrorSnapshot]):
    """Lift settled PE signals into public :class:`SocialPredictionError` records.

    R8 SSOT contract: this module never reconstructs PE outcomes from
    raw producer state. It forwards already-settled typed signals
    (with ``outcome != None``) through
    :func:`social_prediction_error_from_memory_signal` and concatenates
    any externally injected ``pending_errors`` (probe / test path).
    The owner field on each emitted error record comes from the
    signal's ``source_owner`` so the SSOT contract is preserved.
    """

    slot_name = "social_prediction_error"
    owner = "SocialPredictionErrorModule"
    value_type = SocialPredictionErrorSnapshot
    dependencies = ("social_prediction", "multi_party_identity", "memory")
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        pending_errors: tuple[SocialPredictionError, ...] = (),
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._pending_errors = pending_errors

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[SocialPredictionErrorSnapshot]:
        prediction_snapshot = upstream.get("social_prediction")
        prediction_available = (
            prediction_snapshot is not None
            and isinstance(prediction_snapshot.value, SocialPredictionSnapshot)
        )
        memory_snapshot = upstream.get("memory")
        signals = _memory_social_pe_signals(memory_snapshot)

        derived_errors: list[SocialPredictionError] = []
        for signal in signals:
            error = social_prediction_error_from_memory_signal(signal)
            if error is not None:
                derived_errors.append(error)
        derived_errors_tuple = tuple(derived_errors)
        all_errors = self._pending_errors + derived_errors_tuple

        prediction_state = (
            "available" if prediction_available else "compatibility-fallback"
        )
        derivation_summary = f"memory_visibility_pe={len(derived_errors_tuple)}"
        return self.publish(
            SocialPredictionErrorSnapshot(
                errors=all_errors,
                description=(
                    "R16 social PE: pending_injected="
                    f"{len(self._pending_errors)} {derivation_summary} "
                    f"total={len(all_errors)}; prediction={prediction_state}."
                ),
            )
        )


__all__ = [
    "MultiPartyIdentityModule",
    "SocialPredictionAggregateModule",
    "SocialPredictionErrorModule",
]
