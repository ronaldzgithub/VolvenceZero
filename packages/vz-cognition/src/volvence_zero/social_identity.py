"""Multi-party identity owner scaffold (R16).

Slice 2 introduced the SHADOW compatibility ``primary`` snapshot.  Slice 11
adds the first self-emitting social prediction loop: when the upstream
``multi_party_identity`` snapshot is non-default and the ``MemoryModule``
suppressed cross-scope entries, ``SocialPredictionAggregateModule`` and
``SocialPredictionErrorModule`` derive a ``MEMORY_VISIBILITY`` prediction
and matching PE without any injection.
"""

from __future__ import annotations

from typing import Any, Mapping

from volvence_zero.environment import EnvironmentEvent
from volvence_zero.memory import MemorySnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    InterlocutorIdentity,
    MultiPartyIdentitySnapshot,
    SocialPrediction,
    SocialPredictionError,
    SocialPredictionErrorSnapshot,
    SocialPredictionKind,
    SocialPredictionOutcome,
    SocialPredictionSnapshot,
    SocialScopeKind,
    build_primary_multi_party_identity_snapshot,
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


_MEMORY_VISIBILITY_OWNER = "MemoryModule"


def _is_default_subject_scope(subject_ids: tuple[str, ...]) -> bool:
    return subject_ids == (PRIMARY_INTERLOCUTOR_ID,)


def _identity_value(
    snapshot: Snapshot[Any] | None,
) -> MultiPartyIdentitySnapshot | None:
    if snapshot is None or not isinstance(snapshot.value, MultiPartyIdentitySnapshot):
        return None
    return snapshot.value


def _memory_value(snapshot: Snapshot[Any] | None) -> MemorySnapshot | None:
    if snapshot is None or not isinstance(snapshot.value, MemorySnapshot):
        return None
    return snapshot.value


def _memory_visibility_prediction_id(
    *, scope_id: str, memory_version: int
) -> str:
    return f"memory_visibility:{scope_id}:v{memory_version}"


def _memory_visibility_error_id(
    *, scope_id: str, memory_version: int
) -> str:
    return f"memory_visibility_pe:{scope_id}:v{memory_version}"


class SocialPredictionAggregateModule(RuntimeModule[SocialPredictionSnapshot]):
    """Publishes the pre-action social prediction aggregate.

    R16 Slice 11 introduces the first self-emitted prediction: when the
    active multi-party scope is non-default and memory retrieval has run,
    the module declares "memory retrieval will return entries scoped to the
    active subjects". The owner remains :class:`MemoryModule` -- this slot
    only aggregates predictions for downstream PE.
    """

    slot_name = "social_prediction"
    owner = "SocialPredictionAggregateModule"
    value_type = SocialPredictionSnapshot
    dependencies = ("multi_party_identity", "memory")
    default_wiring_level = WiringLevel.SHADOW

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[SocialPredictionSnapshot]:
        identity = _identity_value(upstream.get("multi_party_identity"))
        memory = _memory_value(upstream.get("memory"))
        memory_available = memory is not None
        identity_available = identity is not None
        if (
            identity is None
            or memory is None
            or _is_default_subject_scope(identity.subject_ids)
        ):
            suffix = (
                f"identity={'available' if identity_available else 'compatibility-fallback'}; "
                f"memory={'available' if memory_available else 'unavailable'}; "
                "scope=default-or-missing"
            )
            return self.publish(
                SocialPredictionSnapshot(
                    predictions=(),
                    description=(
                        "R16 social prediction aggregate: no MEMORY_VISIBILITY "
                        f"prediction emitted; {suffix}."
                    ),
                )
            )
        scope_id = identity.subject_ids[0]
        memory_snapshot = upstream.get("memory")
        memory_version = memory_snapshot.version if memory_snapshot is not None else 0
        prediction = SocialPrediction(
            prediction_id=_memory_visibility_prediction_id(
                scope_id=scope_id,
                memory_version=memory_version,
            ),
            kind=SocialPredictionKind.MEMORY_VISIBILITY,
            scope_kind=SocialScopeKind.INTERLOCUTOR,
            scope_id=scope_id,
            subject_ids=identity.subject_ids,
            audience_ids=identity.audience_ids,
            predicted_outcome="memory_subjects_match_active_subjects",
            confidence=0.6,
            evidence=(
                f"active_subject_ids={','.join(identity.subject_ids)}",
                f"retrieved_count={len(memory.retrieved_entries)}",
            ),
        )
        return self.publish(
            SocialPredictionSnapshot(
                predictions=(prediction,),
                description=(
                    "R16 social prediction aggregate: emitted 1 MEMORY_VISIBILITY "
                    f"prediction for scope={scope_id}; "
                    f"retrieved={len(memory.retrieved_entries)} "
                    f"suppressed={len(memory.suppressed_cross_scope_entries)}."
                ),
            )
        )


class SocialPredictionErrorModule(RuntimeModule[SocialPredictionErrorSnapshot]):
    """Publishes typed social PE records derived from social predictions.

    R16 Slice 11: when the upstream memory snapshot suppressed cross-scope
    entries, this owner converts that into a :class:`SocialPredictionError`
    that disconfirms the matching MEMORY_VISIBILITY prediction.  Manual
    ``pending_errors`` injection (Slice 7 probe path) still flows through
    so external probes can layer additional PE without conflict.
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
        prediction_available = isinstance(
            prediction_snapshot.value, SocialPredictionSnapshot
        ) if prediction_snapshot is not None else False
        identity = _identity_value(upstream.get("multi_party_identity"))
        memory = _memory_value(upstream.get("memory"))
        memory_snapshot = upstream.get("memory")
        memory_version = memory_snapshot.version if memory_snapshot is not None else 0
        derived_error = self._derive_memory_visibility_error(
            identity=identity, memory=memory, memory_version=memory_version
        )
        all_errors = self._pending_errors + (
            (derived_error,) if derived_error is not None else ()
        )
        suffix = (
            "prediction=available"
            if prediction_available
            else "prediction=compatibility-fallback"
        )
        derivation_summary = (
            "memory_visibility_pe=1"
            if derived_error is not None
            else "memory_visibility_pe=0"
        )
        return self.publish(
            SocialPredictionErrorSnapshot(
                errors=all_errors,
                description=(
                    "R16 social PE: pending_injected="
                    f"{len(self._pending_errors)} {derivation_summary} "
                    f"total={len(all_errors)}; {suffix}."
                ),
            )
        )

    def _derive_memory_visibility_error(
        self,
        *,
        identity: MultiPartyIdentitySnapshot | None,
        memory: MemorySnapshot | None,
        memory_version: int,
    ) -> SocialPredictionError | None:
        if (
            identity is None
            or memory is None
            or _is_default_subject_scope(identity.subject_ids)
            or not memory.suppressed_cross_scope_entries
        ):
            return None
        suppressed = memory.suppressed_cross_scope_entries
        evaluated_total = len(memory.retrieved_entries) + len(suppressed)
        magnitude = (
            len(suppressed) / evaluated_total if evaluated_total > 0 else 1.0
        )
        scope_id = identity.subject_ids[0]
        evidence = tuple(
            sorted(
                {
                    f"suppressed:{entry.entry_id}:subject={'+'.join(entry.subject_ids)}"
                    for entry in suppressed
                }
            )
        )
        return SocialPredictionError(
            error_id=_memory_visibility_error_id(
                scope_id=scope_id,
                memory_version=memory_version,
            ),
            prediction_id=_memory_visibility_prediction_id(
                scope_id=scope_id,
                memory_version=memory_version,
            ),
            kind=SocialPredictionKind.MEMORY_VISIBILITY,
            outcome=SocialPredictionOutcome.DISCONFIRMED,
            magnitude=min(1.0, max(0.0, magnitude)),
            owner=_MEMORY_VISIBILITY_OWNER,
            scope_kind=SocialScopeKind.INTERLOCUTOR,
            scope_id=scope_id,
            evidence=evidence,
        )


__all__ = [
    "MultiPartyIdentityModule",
    "SocialPredictionAggregateModule",
    "SocialPredictionErrorModule",
]
