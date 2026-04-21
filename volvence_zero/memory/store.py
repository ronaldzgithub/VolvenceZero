from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Mapping
from uuid import uuid4

from volvence_zero.memory.cms import CMSCheckpointState, CMSMemoryCore, CMSState
from volvence_zero.memory.persistence import (
    PersistenceBackend,
    deserialize_checkpoint,
    serialize_checkpoint,
)
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.substrate import FeatureSignal, SubstrateSnapshot, SurfaceKind

if TYPE_CHECKING:
    from volvence_zero.prediction import PredictionErrorSnapshot


class Track(str, Enum):
    WORLD = "world"
    SELF = "self"
    SHARED = "shared"


class MemoryStratum(str, Enum):
    TRANSIENT = "transient"
    EPISODIC = "episodic"
    DURABLE = "durable"
    DERIVED = "derived"


@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str
    content: str
    track: Track
    stratum: str
    created_at_ms: int
    last_accessed_ms: int
    strength: float
    tags: tuple[str, ...]


@dataclass(frozen=True)
class MemoryWriteRequest:
    content: str
    track: Track
    stratum: MemoryStratum
    tags: tuple[str, ...] = ()
    strength: float = 0.5


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    track: Track | None = None
    strata: tuple[MemoryStratum, ...] = ()
    limit: int = 5
    facets: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalResult:
    query: RetrievalQuery
    entries: tuple[MemoryEntry, ...]


@dataclass(frozen=True)
class MemorySnapshot:
    transient_summary: str
    episodic_summary: str
    durable_summary: str
    retrieved_entries: tuple[MemoryEntry, ...]
    total_entries_by_stratum: tuple[tuple[str, int], ...]
    pending_promotions: int
    pending_decays: int
    cms_state: CMSState | None
    description: str
    lifecycle_metrics: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class MemoryStoreCheckpoint:
    checkpoint_id: str
    entries: tuple[MemoryEntry, ...]
    pending_promotions: tuple[str, ...]
    pending_decays: tuple[str, ...]
    cms_state: CMSCheckpointState | None
    promotion_threshold: float
    semantic_index: tuple[tuple[str, tuple[float, ...]], ...]


def _reconstruct_checkpoint(parsed: dict[str, Any]) -> MemoryStoreCheckpoint | None:
    """Reconstruct a MemoryStoreCheckpoint from a deserialized dict.

    Returns None if the dict is missing required fields or has
    incompatible structure.
    """
    try:
        entries_raw = parsed.get("entries", [])
        entries = tuple(
            MemoryEntry(
                entry_id=str(e["entry_id"]),
                content=str(e["content"]),
                track=Track(e["track"]),
                stratum=str(e["stratum"]),
                created_at_ms=int(e["created_at_ms"]),
                last_accessed_ms=int(e["last_accessed_ms"]),
                strength=float(e["strength"]),
                tags=tuple(str(t) for t in e.get("tags", ())),
            )
            for e in entries_raw
        )
        cms_raw = parsed.get("cms_state")
        cms_state: CMSCheckpointState | None = None
        if cms_raw is not None and isinstance(cms_raw, dict):
            cms_state = CMSCheckpointState(
                online_fast=tuple(float(v) for v in cms_raw["online_fast"]),
                session_medium=tuple(float(v) for v in cms_raw["session_medium"]),
                background_slow=tuple(float(v) for v in cms_raw["background_slow"]),
                last_update_ms=int(cms_raw["last_update_ms"]),
                total_observations=int(cms_raw["total_observations"]),
                total_reflections=int(cms_raw["total_reflections"]),
                session_observations_since_update=int(cms_raw["session_observations_since_update"]),
                background_observations_since_update=int(cms_raw["background_observations_since_update"]),
                session_pending_signal=tuple(float(v) for v in cms_raw["session_pending_signal"]),
                background_pending_signal=tuple(float(v) for v in cms_raw["background_pending_signal"]),
                mode=str(cms_raw.get("mode", "vector")),
                mlp_params=tuple(
                    tuple(tuple(float(x) for x in group) for group in band)
                    for band in cms_raw.get("mlp_params", ())
                ),
            )
        semantic_raw = parsed.get("semantic_index", [])
        semantic_index = tuple(
            (str(pair[0]), tuple(float(v) for v in pair[1]))
            for pair in semantic_raw
        )
        return MemoryStoreCheckpoint(
            checkpoint_id=str(parsed.get("checkpoint_id", "restored")),
            entries=entries,
            pending_promotions=tuple(str(p) for p in parsed.get("pending_promotions", ())),
            pending_decays=tuple(str(d) for d in parsed.get("pending_decays", ())),
            cms_state=cms_state,
            promotion_threshold=float(parsed.get("promotion_threshold", 0.3)),
            semantic_index=semantic_index,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _clamp_strength(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    ascii_buffer: list[str] = []
    compact = "".join(char for char in text.lower() if not char.isspace())
    for char in text.lower():
        if char.isascii() and char.isalnum():
            ascii_buffer.append(char)
            continue
        if ascii_buffer:
            tokens.add("".join(ascii_buffer))
            ascii_buffer.clear()
        if not char.isspace():
            tokens.add(char)
    if ascii_buffer:
        tokens.add("".join(ascii_buffer))
    for index in range(len(compact) - 1):
        tokens.add(compact[index : index + 2])
    return tokens


def _semantic_embedding(*, text: str, tags: tuple[str, ...], dim: int = 6) -> tuple[float, ...]:
    tokens = tuple(sorted(_tokenize(text) | {tag.lower() for tag in tags}))
    if not tokens:
        return tuple(0.0 for _ in range(dim))
    vector = [0.0 for _ in range(dim)]
    for token in tokens:
        token_strength = max(len(token), 1)
        for index, char in enumerate(token):
            vector[(index + len(token)) % dim] += (ord(char) % 31) / 31.0 / token_strength
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-6:
        return tuple(0.0 for _ in range(dim))
    return tuple(value / norm for value in vector)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def summarize_entries(entries: Iterable[MemoryEntry], *, fallback: str) -> str:
    collected = tuple(entries)
    if not collected:
        return fallback
    preview = "; ".join(entry.content for entry in collected[:3])
    if len(collected) > 3:
        preview += "; ..."
    return preview


@dataclass(frozen=True)
class LearnedMemoryRecall:
    query_signal: tuple[float, ...]
    core_signal: tuple[float, ...]
    retrieval_confidence: float
    artifact_weight: float
    learned_weight: float
    description: str


class ArtifactStore:
    """Audit-friendly durable artifact layer.

    This layer stores explicit cards, beliefs, and other rollback-friendly
    artifacts. It is not the primary memory substrate; learned multi-timescale
    state lives in the CMS core.
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._by_stratum: dict[MemoryStratum, list[str]] = {
            MemoryStratum.TRANSIENT: [],
            MemoryStratum.EPISODIC: [],
            MemoryStratum.DURABLE: [],
            MemoryStratum.DERIVED: [],
        }
        self._pending_promotions: list[str] = []
        self._pending_decays: list[str] = []

    @property
    def pending_promotions(self) -> tuple[str, ...]:
        return tuple(self._pending_promotions)

    @property
    def pending_decays(self) -> tuple[str, ...]:
        return tuple(self._pending_decays)

    def write(self, entry: MemoryEntry) -> MemoryEntry:
        self._entries[entry.entry_id] = entry
        stratum = MemoryStratum(entry.stratum)
        if entry.entry_id not in self._by_stratum[stratum]:
            self._by_stratum[stratum].append(entry.entry_id)
        if stratum in {MemoryStratum.TRANSIENT, MemoryStratum.EPISODIC} and entry.entry_id not in self._pending_promotions:
            self._pending_promotions.append(entry.entry_id)
        if stratum is MemoryStratum.TRANSIENT and entry.strength < 0.4 and entry.entry_id not in self._pending_decays:
            self._pending_decays.append(entry.entry_id)
        return entry

    def get(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    def touch(self, entry_id: str, *, timestamp_ms: int) -> MemoryEntry | None:
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        updated = replace(entry, last_accessed_ms=timestamp_ms)
        self._entries[entry_id] = updated
        return updated

    def replace_entry(self, entry: MemoryEntry) -> None:
        self._entries[entry.entry_id] = entry
        stratum = MemoryStratum(entry.stratum)
        for candidate in MemoryStratum:
            bucket = self._by_stratum[candidate]
            if candidate is stratum:
                if entry.entry_id not in bucket:
                    bucket.append(entry.entry_id)
            elif entry.entry_id in bucket:
                bucket.remove(entry.entry_id)

    def entries_for(self, stratum: MemoryStratum) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries[entry_id] for entry_id in self._by_stratum[stratum])

    def entries_in(self, strata: tuple[MemoryStratum, ...]) -> tuple[MemoryEntry, ...]:
        return tuple(
            self._entries[entry_id]
            for stratum in strata
            for entry_id in self._by_stratum[stratum]
        )

    def total_entries_by_stratum(self) -> tuple[tuple[str, int], ...]:
        return tuple((stratum.value, len(self._by_stratum[stratum])) for stratum in MemoryStratum)

    def entry_count(self) -> int:
        return len(self._entries)

    def promote(
        self,
        *,
        entry_id: str,
        promotion_threshold: float,
        promotion_boost: float,
        timestamp_ms: int,
    ) -> MemoryEntry | None:
        entry = self._entries.get(entry_id)
        if entry is None or entry.strength < promotion_threshold:
            return None
        updated = replace(
            entry,
            stratum=MemoryStratum.DURABLE.value,
            strength=_clamp_strength(max(entry.strength, 0.55 + promotion_boost * 0.35)),
            last_accessed_ms=timestamp_ms,
        )
        self.replace_entry(updated)
        return updated

    def decay(
        self,
        *,
        entry_id: str,
        decay_scale: float,
        timestamp_ms: int,
    ) -> MemoryEntry | None:
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        updated = replace(
            entry,
            strength=_clamp_strength(entry.strength * max(0.55, 1.0 - decay_scale * 0.35)),
            last_accessed_ms=timestamp_ms,
        )
        self._entries[entry_id] = updated
        return updated

    def export_entries(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries.values())

    def restore(
        self,
        *,
        entries: tuple[MemoryEntry, ...],
        pending_promotions: tuple[str, ...],
        pending_decays: tuple[str, ...],
    ) -> None:
        self._entries = {entry.entry_id: entry for entry in entries}
        self._by_stratum = {
            stratum: [entry.entry_id for entry in entries if entry.stratum == stratum.value]
            for stratum in MemoryStratum
        }
        self._pending_promotions = list(pending_promotions)
        self._pending_decays = list(pending_decays)


class DerivedRetrievalIndex:
    """Rebuildable artifact retrieval support.

    This index is intentionally derived from explicit artifacts and can be
    reconstructed from checkpoints. It should not be treated as memory truth.
    """

    def __init__(self) -> None:
        self._artifact_embeddings: dict[str, tuple[float, ...]] = {}

    def index_entry(self, entry: MemoryEntry) -> None:
        self._artifact_embeddings[entry.entry_id] = _semantic_embedding(
            text=entry.content,
            tags=entry.tags,
        )

    def affinity(self, *, entry: MemoryEntry, query_embedding: tuple[float, ...]) -> float:
        return _cosine_similarity(query_embedding, self._artifact_embeddings.get(entry.entry_id, (0.0,) * len(query_embedding)))

    def export_state(self) -> tuple[tuple[str, tuple[float, ...]], ...]:
        return tuple(sorted(self._artifact_embeddings.items()))

    def restore(self, embeddings: tuple[tuple[str, tuple[float, ...]], ...]) -> None:
        self._artifact_embeddings = dict(embeddings)


def build_default_memory_store(*, latent_dim: int = 8, nested_profile: bool = True) -> "MemoryStore":
    variant = "nested" if nested_profile else "sequential"
    learned_core = CMSMemoryCore(
        mode="mlp",
        d_in=max(latent_dim, 4),
        d_hidden=max(latent_dim * 2, 8),
        variant=variant,
        session_cadence=2,
        background_cadence=4,
    )
    return MemoryStore(learned_core=learned_core)


class MemoryStore:
    """Owner-controlled memory system with learned core + artifact layers."""

    def __init__(
        self,
        *,
        learned_core: CMSMemoryCore | None = None,
        promotion_threshold: float = 0.7,
        persistence_backend: PersistenceBackend | None = None,
    ) -> None:
        self._learned_core = learned_core
        self._artifact_store = ArtifactStore()
        self._derived_index = DerivedRetrievalIndex()
        self._promotion_threshold = _clamp_strength(promotion_threshold)
        self._persistence_backend = persistence_backend
        self._persistence_version = 0
        self._context_reset_count = 0
        self._last_context_reset_ms = 0
        self._last_context_reset_reason = ""
        self._last_context_reset_applied = False
        self._last_context_reset_online_seed_strength = 0.0
        self._last_context_reset_session_seed_strength = 0.0
        self._last_context_reset_transfer_strength = 0.0
        self._last_context_reset_target_distance_before = 0.0
        self._last_context_reset_target_distance_after = 0.0
        self._last_context_reset_target_alignment_gain = 0.0
        self._artifact_consolidation_count = 0
        self._learned_recall_count = 0
        self._last_recall_confidence = 0.0
        self._last_recall_driver = "artifact-only"

    @property
    def learned_core(self) -> CMSMemoryCore | None:
        return self._learned_core

    @property
    def promotion_threshold(self) -> float:
        return self._promotion_threshold

    @property
    def _entries(self) -> dict[str, MemoryEntry]:
        return self._artifact_store._entries

    @property
    def _by_stratum(self) -> dict[MemoryStratum, list[str]]:
        return self._artifact_store._by_stratum

    @property
    def _pending_promotions(self) -> list[str]:
        return self._artifact_store._pending_promotions

    @property
    def _pending_decays(self) -> list[str]:
        return self._artifact_store._pending_decays

    @property
    def _semantic_index(self) -> dict[str, tuple[float, ...]]:
        return self._derived_index._artifact_embeddings

    def reset_nested_context(self, *, reason: str, timestamp_ms: int) -> tuple[str, ...]:
        self._last_context_reset_ms = timestamp_ms
        self._last_context_reset_reason = reason
        if self._learned_core is None or self._learned_core.mode != "mlp" or self._learned_core.variant != "nested":
            self._last_context_reset_applied = False
            self._last_context_reset_online_seed_strength = 0.0
            self._last_context_reset_session_seed_strength = 0.0
            self._last_context_reset_transfer_strength = 0.0
            self._last_context_reset_target_distance_before = 0.0
            self._last_context_reset_target_distance_after = 0.0
            self._last_context_reset_target_alignment_gain = 0.0
            return ()
        before_state = self._learned_core.snapshot()
        before_online = before_state.online_fast.vector
        nested_reset_targets = self._learned_core.nested_reset_targets()
        if nested_reset_targets is None:
            raise RuntimeError("Nested reset targets must be available for nested MLP CMS.")
        online_target, _session_target = nested_reset_targets
        self._learned_core.reset_context()
        after_state = self._learned_core.snapshot()
        after_online = after_state.online_fast.vector
        after_session = after_state.session_medium.vector
        self._context_reset_count += 1
        self._last_context_reset_applied = True
        self._last_context_reset_online_seed_strength = sum(abs(value) for value in after_online) / max(
            len(after_online), 1
        )
        self._last_context_reset_session_seed_strength = sum(abs(value) for value in after_session) / max(
            len(after_session), 1
        )
        self._last_context_reset_transfer_strength = sum(
            abs(after_value - before_value)
            for after_value, before_value in zip(after_online, before_online, strict=True)
        ) / max(len(after_online), 1)
        self._last_context_reset_target_distance_before = sum(
            abs(before_value - target_value)
            for before_value, target_value in zip(before_online, online_target, strict=True)
        ) / max(len(before_online), 1)
        self._last_context_reset_target_distance_after = sum(
            abs(after_value - target_value)
            for after_value, target_value in zip(after_online, online_target, strict=True)
        ) / max(len(after_online), 1)
        self._last_context_reset_target_alignment_gain = (
            self._last_context_reset_target_distance_before - self._last_context_reset_target_distance_after
        )
        return ("nested-context-reset",)

    def write(self, request: MemoryWriteRequest, *, timestamp_ms: int) -> MemoryEntry:
        entry = MemoryEntry(
            entry_id=str(uuid4()),
            content=request.content,
            track=request.track,
            stratum=request.stratum.value,
            created_at_ms=timestamp_ms,
            last_accessed_ms=timestamp_ms,
            strength=_clamp_strength(request.strength),
            tags=request.tags,
        )
        self._artifact_store.write(entry)
        self._derived_index.index_entry(entry)
        self._observe_artifact_entry(entry=entry, timestamp_ms=timestamp_ms, source="write")
        return entry

    def retrieve(self, query: RetrievalQuery, *, timestamp_ms: int) -> RetrievalResult:
        tokens = _tokenize(query.text)
        query_embedding = _semantic_embedding(
            text=" ".join(part for part in (query.text, *query.facets) if part),
            tags=query.facets,
        )
        learned_recall = self._build_learned_recall(query=query, query_embedding=query_embedding)
        strata = query.strata or tuple(MemoryStratum)
        matches: list[tuple[float, MemoryEntry]] = []
        for entry in self._artifact_store.entries_in(strata):
            if query.track is not None and entry.track is not query.track:
                continue
            score = self._score_entry(
                entry,
                query_tokens=tokens,
                query_embedding=query_embedding,
                learned_recall=learned_recall,
            )
            if score <= 0:
                continue
            updated = self._artifact_store.touch(entry.entry_id, timestamp_ms=timestamp_ms)
            if updated is not None:
                matches.append((score, updated))
        matches.sort(key=lambda item: (-item[0], -item[1].strength, -item[1].created_at_ms))
        self._learned_recall_count += 1
        self._last_recall_confidence = learned_recall.retrieval_confidence
        self._last_recall_driver = (
            "learned-core-guided"
            if learned_recall.learned_weight >= learned_recall.artifact_weight
            else "artifact-guided"
        )
        return RetrievalResult(
            query=query,
            entries=tuple(entry for _, entry in matches[: query.limit]),
        )

    def snapshot(self, *, retrieved_entries: tuple[MemoryEntry, ...]) -> MemorySnapshot:
        transient_entries = self._entries_for(MemoryStratum.TRANSIENT)
        episodic_entries = self._entries_for(MemoryStratum.EPISODIC)
        durable_entries = self._entries_for(MemoryStratum.DURABLE)
        total_entries = self._artifact_store.total_entries_by_stratum()
        description = (
            f"Memory store with learned core primary and {self._artifact_store.entry_count()} artifact entries "
            f"across {len(MemoryStratum)} strata; {len(retrieved_entries)} retrieved this turn."
        )
        if self._learned_core is not None:
            description += f" {self._learned_core.snapshot().description}"
        if self._last_context_reset_reason:
            description += (
                f" last_reset_reason={self._last_context_reset_reason} "
                f"applied={self._last_context_reset_applied} count={self._context_reset_count}."
            )
        return MemorySnapshot(
            transient_summary=summarize_entries(
                transient_entries, fallback="No transient working-state memories."
            ),
            episodic_summary=summarize_entries(
                episodic_entries, fallback="No episodic session-state memories."
            ),
            durable_summary=summarize_entries(
                durable_entries, fallback="No durable artifact memories."
            ),
            retrieved_entries=retrieved_entries,
            total_entries_by_stratum=total_entries,
            pending_promotions=len(self._artifact_store.pending_promotions),
            pending_decays=len(self._artifact_store.pending_decays),
            cms_state=self._learned_core.snapshot() if self._learned_core is not None else None,
            lifecycle_metrics=(
                ("nested_profile_active", float(
                    self._learned_core is not None
                    and self._learned_core.mode == "mlp"
                    and self._learned_core.variant == "nested"
                )),
                ("nested_context_reset_count", float(self._context_reset_count)),
                ("last_nested_reset_applied", float(self._last_context_reset_applied)),
                ("last_nested_reset_online_seed_strength", self._last_context_reset_online_seed_strength),
                ("last_nested_reset_session_seed_strength", self._last_context_reset_session_seed_strength),
                ("slow_to_fast_init_benefit", self._last_context_reset_transfer_strength),
                ("slow_to_fast_target_distance_before", self._last_context_reset_target_distance_before),
                ("slow_to_fast_target_distance_after", self._last_context_reset_target_distance_after),
                ("slow_to_fast_target_alignment_gain", self._last_context_reset_target_alignment_gain),
                ("learned_memory_primary", float(self._learned_core is not None)),
                ("artifact_consolidation_count", float(self._artifact_consolidation_count)),
                ("learned_recall_count", float(self._learned_recall_count)),
                ("last_learned_recall_confidence", self._last_recall_confidence),
                ("last_learned_recall_driver_is_core", float(self._last_recall_driver == "learned-core-guided")),
            ),
            description=description,
        )

    def observe_substrate(self, *, substrate_snapshot: SubstrateSnapshot | None, timestamp_ms: int) -> None:
        if self._learned_core is not None:
            self._learned_core.observe_substrate(
                substrate_snapshot=substrate_snapshot,
                timestamp_ms=timestamp_ms,
            )

    def observe_encoder_feedback(
        self,
        *,
        encoder_signal: tuple[float, ...],
        timestamp_ms: int,
    ) -> None:
        if self._learned_core is not None:
            self._learned_core.observe_encoder_feedback(
                encoder_signal=encoder_signal,
                timestamp_ms=timestamp_ms,
            )

    def observe_temporal_feedback(
        self,
        *,
        encoder_signal: tuple[float, ...],
        timestamp_ms: int,
    ) -> None:
        self.observe_encoder_feedback(encoder_signal=encoder_signal, timestamp_ms=timestamp_ms)

    def apply_prediction_error_signal(
        self,
        *,
        prediction_error_snapshot: "PredictionErrorSnapshot | None",
        timestamp_ms: int,
    ) -> tuple[str, ...]:
        if prediction_error_snapshot is None or prediction_error_snapshot.bootstrap:
            return ()
        pe = prediction_error_snapshot.error
        operations: list[str] = []
        magnitude = pe.magnitude
        primary_dimension = max(
            (
                ("task", abs(pe.task_error)),
                ("relationship", abs(pe.relationship_error)),
                ("regime", abs(pe.regime_error)),
                ("action", abs(pe.action_error)),
            ),
            key=lambda item: item[1],
        )[0]
        target_track = (
            Track.WORLD if primary_dimension == "task"
            else Track.SELF if primary_dimension == "relationship"
            else Track.SHARED
        )
        if magnitude >= 0.15:
            entry = self.write(
                MemoryWriteRequest(
                    content=f"prediction_error:{primary_dimension}:{prediction_error_snapshot.error.description}",
                    track=target_track,
                    stratum=MemoryStratum.EPISODIC if magnitude < 1.0 else MemoryStratum.DURABLE,
                    tags=("prediction_error", primary_dimension),
                    strength=_clamp_strength(min(1.0, 0.45 + magnitude * 0.2)),
                ),
                timestamp_ms=timestamp_ms,
            )
            operations.append(f"prediction-error-write:{entry.entry_id}")
        if pe.relationship_error < -0.15:
            self._promotion_threshold = _clamp_strength(self._promotion_threshold - 0.03)
            operations.append("prediction-error:lower-promotion-threshold")
        if pe.task_error < -0.15:
            self._promotion_threshold = _clamp_strength(self._promotion_threshold - 0.02)
            operations.append("prediction-error:task-threshold-adjust")
        if pe.signed_reward > 0.15:
            self._promotion_threshold = _clamp_strength(self._promotion_threshold + 0.01)
            operations.append("prediction-error:raise-promotion-threshold")
        return tuple(operations)

    def apply_reflection_consolidation(
        self,
        *,
        new_durable_entries: tuple[MemoryEntry, ...],
        promoted_entries: tuple[str, ...],
        decayed_entries: tuple[str, ...],
        beliefs_updated: tuple[str, ...],
        promotion_boost: float,
        decay_scale: float,
        lesson_count: int,
        timestamp_ms: int,
    ) -> tuple[str, ...]:
        applied: list[str] = []
        for entry_id in promoted_entries:
            updated = self._artifact_store.promote(
                entry_id=entry_id,
                promotion_threshold=self._promotion_threshold,
                promotion_boost=promotion_boost,
                timestamp_ms=timestamp_ms,
            )
            if updated is None:
                continue
            self._derived_index.index_entry(updated)
            self._observe_artifact_entry(entry=updated, timestamp_ms=timestamp_ms, source="promotion")
            applied.append(f"promoted:{entry_id}")
        for entry in new_durable_entries:
            if entry.strength < self._promotion_threshold:
                continue
            self._artifact_store.write(entry)
            self._derived_index.index_entry(entry)
            self._observe_artifact_entry(entry=entry, timestamp_ms=timestamp_ms, source="reflection-durable")
            applied.append(f"durable:{entry.entry_id}")
        for entry_id in decayed_entries:
            updated = self._artifact_store.decay(
                entry_id=entry_id,
                decay_scale=decay_scale,
                timestamp_ms=timestamp_ms,
            )
            if updated is None:
                continue
            applied.append(f"decayed:{entry_id}")
        for belief in beliefs_updated:
            belief_entry = MemoryEntry(
                entry_id=str(uuid4()),
                content=belief,
                track=Track.SHARED,
                stratum=MemoryStratum.DURABLE.value,
                created_at_ms=timestamp_ms,
                last_accessed_ms=timestamp_ms,
                strength=_clamp_strength(0.5 + promotion_boost * 0.25),
                tags=("belief_update",),
            )
            self._artifact_store.write(belief_entry)
            self._derived_index.index_entry(belief_entry)
            self._observe_artifact_entry(entry=belief_entry, timestamp_ms=timestamp_ms, source="belief-update")
            applied.append(f"belief:{belief_entry.entry_id}")
        if self._learned_core is not None:
            self._learned_core.reflect_lessons(lesson_count=lesson_count, timestamp_ms=timestamp_ms)
        return tuple(applied)

    def apply_promotion_threshold_update(self, *, delta: float) -> str:
        previous = self._promotion_threshold
        self._promotion_threshold = _clamp_strength(self._promotion_threshold + delta)
        return f"promotion-threshold:{previous:.2f}->{self._promotion_threshold:.2f}"

    def create_checkpoint(self, *, checkpoint_id: str | None = None) -> MemoryStoreCheckpoint:
        return MemoryStoreCheckpoint(
            checkpoint_id=checkpoint_id or str(uuid4()),
            entries=self._artifact_store.export_entries(),
            pending_promotions=self._artifact_store.pending_promotions,
            pending_decays=self._artifact_store.pending_decays,
            cms_state=self._learned_core.export_state() if self._learned_core is not None else None,
            promotion_threshold=self._promotion_threshold,
            semantic_index=self._derived_index.export_state(),
        )

    def export_rare_heavy_state(self, *, checkpoint_id: str | None = None) -> MemoryStoreCheckpoint:
        return self.create_checkpoint(checkpoint_id=checkpoint_id or "rare-heavy-memory")

    def restore_checkpoint(self, checkpoint: MemoryStoreCheckpoint) -> None:
        self._artifact_store.restore(
            entries=checkpoint.entries,
            pending_promotions=checkpoint.pending_promotions,
            pending_decays=checkpoint.pending_decays,
        )
        self._promotion_threshold = checkpoint.promotion_threshold
        self._derived_index.restore(checkpoint.semantic_index)
        if self._learned_core is not None and checkpoint.cms_state is not None:
            self._learned_core.restore_state(checkpoint.cms_state)

    def import_rare_heavy_state(self, checkpoint: MemoryStoreCheckpoint) -> tuple[str, ...]:
        self.restore_checkpoint(checkpoint)
        return ("rare-heavy:memory-import",)

    def save_to_backend(self, *, key: str = "memory/store") -> bool:
        """Persist the current checkpoint to the configured backend. Returns False if no backend."""
        if self._persistence_backend is None:
            return False
        checkpoint = self.create_checkpoint(checkpoint_id=f"persist-{key}")
        data = serialize_checkpoint(checkpoint)
        self._persistence_version += 1
        self._persistence_backend.save_checkpoint(
            key=key, data=data, version=self._persistence_version,
        )
        return True

    def load_from_backend(self, *, key: str = "memory/store") -> bool:
        """Load the latest checkpoint from the configured backend. Returns False if unavailable."""
        if self._persistence_backend is None:
            return False
        result = self._persistence_backend.load_checkpoint(key=key)
        if result is None:
            return False
        data, version = result
        parsed = deserialize_checkpoint(data)
        if not parsed:
            return False
        checkpoint = _reconstruct_checkpoint(parsed)
        if checkpoint is None:
            return False
        self.restore_checkpoint(checkpoint)
        self._persistence_version = version
        return True

    def _entries_for(self, stratum: MemoryStratum) -> tuple[MemoryEntry, ...]:
        return self._artifact_store.entries_for(stratum)

    def _score_entry(
        self,
        entry: MemoryEntry,
        query_tokens: set[str],
        query_embedding: tuple[float, ...],
        learned_recall: LearnedMemoryRecall,
    ) -> float:
        lexical_score = 0.0
        content_tokens = _tokenize(entry.content)
        tag_tokens = {token.lower() for token in entry.tags}
        overlap = len(query_tokens & content_tokens)
        lexical_score += overlap * 3.0
        lexical_score += len(query_tokens & tag_tokens) * 2.0
        if lexical_score == 0.0 and any(token in entry.content.lower() for token in query_tokens):
            lexical_score = 1.0
        if not query_tokens:
            lexical_score = 1.0
        artifact_semantic_score = self._derived_index.affinity(
            entry=entry,
            query_embedding=query_embedding,
        )
        learned_affinity = self._entry_learned_affinity(entry=entry, query_signal=learned_recall.query_signal)
        return (
            learned_affinity * learned_recall.learned_weight
            + artifact_semantic_score * learned_recall.artifact_weight
            + lexical_score * 0.8
        )

    def _observe_artifact_entry(
        self,
        *,
        entry: MemoryEntry,
        timestamp_ms: int,
        source: str,
    ) -> None:
        self._artifact_consolidation_count += 1
        if self._learned_core is None:
            return
        signal = self._entry_signal(entry)
        if source in {"reflection-durable", "belief-update", "promotion"}:
            self._learned_core.observe_encoder_feedback(
                encoder_signal=signal,
                timestamp_ms=timestamp_ms,
            )

    def _entry_signal(self, entry: MemoryEntry) -> tuple[float, ...]:
        dim = self._learned_signal_dim()
        semantic = _semantic_embedding(
            text=f"{entry.track.value}:{entry.stratum}:{entry.content}",
            tags=entry.tags + (entry.track.value, entry.stratum),
            dim=dim,
        )
        strength = _clamp_strength(entry.strength)
        return tuple(_clamp_strength(value * (0.55 + strength * 0.45)) for value in semantic)

    def _entry_learned_affinity(
        self,
        *,
        entry: MemoryEntry,
        query_signal: tuple[float, ...],
    ) -> float:
        return _cosine_similarity(query_signal, self._entry_signal(entry))

    def _learned_signal_dim(self) -> int:
        if self._learned_core is not None:
            return len(self._learned_core.snapshot().online_fast.vector)
        return 6

    def _build_learned_recall(
        self,
        *,
        query: RetrievalQuery,
        query_embedding: tuple[float, ...],
    ) -> LearnedMemoryRecall:
        dim = self._learned_signal_dim()
        projected_query = _semantic_embedding(
            text=" ".join(part for part in (query.text, *query.facets) if part),
            tags=query.facets + ((query.track.value,) if query.track is not None else ()),
            dim=dim,
        )
        if self._learned_core is None:
            return LearnedMemoryRecall(
                query_signal=projected_query,
                core_signal=tuple(0.0 for _ in range(dim)),
                retrieval_confidence=0.0,
                artifact_weight=2.6,
                learned_weight=0.0,
                description="Artifact-only retrieval because no learned core is active.",
            )
        cms_state = self._learned_core.snapshot()
        fast = cms_state.online_fast.vector
        medium = cms_state.session_medium.vector
        slow = cms_state.background_slow.vector
        core_signal = tuple(
            _clamp_strength(fast[index] * 0.5 + medium[index] * 0.3 + slow[index] * 0.2)
            for index in range(len(fast))
        )
        query_signal = tuple(
            _clamp_strength(projected_query[index] * 0.55 + core_signal[index] * 0.45)
            for index in range(len(core_signal))
        )
        confidence = _cosine_similarity(query_signal, core_signal)
        learned_weight = 4.0 + max(confidence, 0.0) * 2.0
        artifact_weight = 1.3 + max(0.0, 1.0 - confidence) * 0.8
        return LearnedMemoryRecall(
            query_signal=query_signal,
            core_signal=core_signal,
            retrieval_confidence=confidence,
            artifact_weight=artifact_weight,
            learned_weight=learned_weight,
            description=(
                f"Learned recall blends query with CMS core confidence={confidence:.2f}; "
                f"learned_weight={learned_weight:.2f} artifact_weight={artifact_weight:.2f}."
            ),
        )


def build_memory_write_requests(
    *,
    substrate_snapshot: SubstrateSnapshot | None,
    user_text: str | None,
    track: Track = Track.SHARED,
) -> tuple[MemoryWriteRequest, ...]:
    requests: list[MemoryWriteRequest] = []
    if user_text:
        requests.append(
            MemoryWriteRequest(
                content=user_text,
                track=track,
                stratum=MemoryStratum.TRANSIENT,
                tags=("user_input",),
                strength=0.55,
            )
        )
    if substrate_snapshot is None:
        return tuple(requests)
    sequence_text = _sequence_text_from_substrate(substrate_snapshot)
    if sequence_text:
        requests.append(
            MemoryWriteRequest(
                content=sequence_text,
                track=track,
                stratum=MemoryStratum.TRANSIENT,
                tags=("substrate_sequence", substrate_snapshot.surface_kind.value),
                strength=0.6,
            )
        )
    if substrate_snapshot.feature_surface:
        for feature in substrate_snapshot.feature_surface:
            requests.append(
                MemoryWriteRequest(
                    content=f"feature:{feature.name}",
                    track=track,
                    stratum=MemoryStratum.DERIVED,
                    tags=(feature.name, feature.source),
                    strength=0.4,
                )
            )
    return tuple(requests)


def build_retrieval_query(
    *,
    substrate_snapshot: SubstrateSnapshot | None,
    user_text: str | None,
    track: Track = Track.SHARED,
    query_facets: tuple[str, ...] = (),
) -> RetrievalQuery:
    query_parts: list[str] = []
    if user_text:
        query_parts.append(user_text)
    if substrate_snapshot is not None:
        sequence_text = _sequence_text_from_substrate(substrate_snapshot)
        if sequence_text:
            query_parts.append(sequence_text)
        query_parts.extend(_query_parts_from_feature_surface(substrate_snapshot.feature_surface))
    query_text = " ".join(part for part in query_parts if part).strip()
    if not query_text:
        query_text = "memory baseline"
    return RetrievalQuery(
        text=query_text,
        track=track,
        strata=(
            MemoryStratum.TRANSIENT,
            MemoryStratum.EPISODIC,
            MemoryStratum.DURABLE,
            MemoryStratum.DERIVED,
        ),
        limit=5,
        facets=query_facets,
    )


def _query_parts_from_feature_surface(feature_surface: tuple[FeatureSignal, ...]) -> tuple[str, ...]:
    return tuple(feature.name for feature in feature_surface[:3])


def _sequence_text_from_substrate(substrate_snapshot: SubstrateSnapshot) -> str:
    tokens = tuple(
        step.token.strip()
        for step in substrate_snapshot.residual_sequence[:24]
        if step.token.strip() and not step.token.startswith("<tok:")
    )
    return " ".join(tokens)


class MemoryModule(RuntimeModule[MemorySnapshot]):
    slot_name = "memory"
    owner = "MemoryModule"
    value_type = MemorySnapshot
    dependencies = ("substrate", "temporal_abstraction", "dual_track", "prediction_error")
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        wiring_level: WiringLevel | None = None,
        memory_feedback_signal: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._store = store or MemoryStore()
        self._memory_feedback_signal = memory_feedback_signal

    @property
    def store(self) -> MemoryStore:
        return self._store

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[MemorySnapshot]:
        from volvence_zero.prediction import PredictionErrorSnapshot

        substrate_snapshot = upstream["substrate"]
        temporal_snapshot = upstream.get("temporal_abstraction")
        dual_track_snapshot = upstream.get("dual_track")
        prediction_error_snapshot = upstream.get("prediction_error")
        substrate_value = substrate_snapshot.value if isinstance(substrate_snapshot.value, SubstrateSnapshot) else None
        prediction_error_value = (
            prediction_error_snapshot.value
            if prediction_error_snapshot is not None and isinstance(prediction_error_snapshot.value, PredictionErrorSnapshot)
            else None
        )
        self._store.observe_substrate(
            substrate_snapshot=substrate_value,
            timestamp_ms=substrate_snapshot.timestamp_ms,
        )
        temporal_feedback_signal = _temporal_feedback_signal(temporal_snapshot.value if temporal_snapshot is not None else None)
        if temporal_feedback_signal:
            self._store.observe_temporal_feedback(
                encoder_signal=temporal_feedback_signal,
                timestamp_ms=substrate_snapshot.timestamp_ms,
            )
        elif self._memory_feedback_signal:
            self._store.observe_temporal_feedback(
                encoder_signal=self._memory_feedback_signal,
                timestamp_ms=substrate_snapshot.timestamp_ms,
            )
        self._store.apply_prediction_error_signal(
            prediction_error_snapshot=prediction_error_value,
            timestamp_ms=substrate_snapshot.timestamp_ms,
        )
        for request in build_memory_write_requests(
            substrate_snapshot=substrate_value,
            user_text=None,
            track=Track.SHARED,
        ):
            self._store.write(request, timestamp_ms=substrate_snapshot.timestamp_ms)

        retrieval = self._store.retrieve(
            build_retrieval_query(
                substrate_snapshot=substrate_value,
                user_text=None,
                track=None,
                query_facets=self._runtime_query_facets(
                    substrate_snapshot=substrate_value,
                    temporal_value=temporal_snapshot.value if temporal_snapshot is not None else None,
                    dual_track_value=dual_track_snapshot.value if dual_track_snapshot is not None else None,
                    prediction_error_value=prediction_error_value,
                ),
            ),
            timestamp_ms=substrate_snapshot.timestamp_ms,
        )
        return self.publish(self._store.snapshot(retrieved_entries=retrieval.entries))

    async def process_standalone(self, **kwargs: object) -> Snapshot[MemorySnapshot]:
        from volvence_zero.prediction import PredictionErrorSnapshot

        timestamp_ms = int(kwargs.get("timestamp_ms", 0)) or 1
        user_text = kwargs.get("user_text")
        if user_text is not None and not isinstance(user_text, str):
            raise TypeError("user_text must be a string when provided.")
        query_facets = kwargs.get("query_facets", ())
        if not isinstance(query_facets, tuple):
            raise TypeError("query_facets must be a tuple when provided.")
        temporal_snapshot = kwargs.get("temporal_snapshot")
        dual_track_snapshot = kwargs.get("dual_track_snapshot")
        prediction_error_snapshot = kwargs.get("prediction_error_snapshot")

        substrate_snapshot = kwargs.get("substrate_snapshot")
        substrate_value = substrate_snapshot if isinstance(substrate_snapshot, SubstrateSnapshot) else None
        self._store.observe_substrate(
            substrate_snapshot=substrate_value,
            timestamp_ms=timestamp_ms,
        )
        temporal_feedback_signal = _temporal_feedback_signal(temporal_snapshot)
        if temporal_feedback_signal:
            self._store.observe_temporal_feedback(
                encoder_signal=temporal_feedback_signal,
                timestamp_ms=timestamp_ms,
            )
        elif self._memory_feedback_signal:
            self._store.observe_temporal_feedback(
                encoder_signal=self._memory_feedback_signal,
                timestamp_ms=timestamp_ms,
            )
        self._store.apply_prediction_error_signal(
            prediction_error_snapshot=prediction_error_snapshot if isinstance(prediction_error_snapshot, PredictionErrorSnapshot) else None,
            timestamp_ms=timestamp_ms,
        )

        for request in build_memory_write_requests(
            substrate_snapshot=substrate_value,
            user_text=user_text,
            track=Track.SHARED,
        ):
            self._store.write(request, timestamp_ms=timestamp_ms)

        retrieval = self._store.retrieve(
            build_retrieval_query(
                substrate_snapshot=substrate_value,
                user_text=user_text,
                track=None,
                query_facets=query_facets
                + self._runtime_query_facets(
                    substrate_snapshot=substrate_value,
                    temporal_value=temporal_snapshot,
                    dual_track_value=dual_track_snapshot,
                    prediction_error_value=prediction_error_snapshot if isinstance(prediction_error_snapshot, PredictionErrorSnapshot) else None,
                ),
            ),
            timestamp_ms=timestamp_ms,
        )
        return self.publish(self._store.snapshot(retrieved_entries=retrieval.entries))

    def _runtime_query_facets(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot | None,
        temporal_value: Any = None,
        dual_track_value: Any = None,
        prediction_error_value: "PredictionErrorSnapshot | None" = None,
    ) -> tuple[str, ...]:
        facets: list[str] = []
        if self._store.learned_core is not None:
            cms_state = self._store.learned_core.snapshot()
            facets.extend(
                (
                    f"cms:{cms_state.online_fast.name}",
                    f"cms:{cms_state.session_medium.name}",
                    f"cms:{cms_state.background_slow.name}",
                )
            )
        if substrate_snapshot is not None:
            facets.extend(_query_parts_from_feature_surface(substrate_snapshot.feature_surface))
        facets.extend(_temporal_query_facets(temporal_value))
        facets.extend(_dual_track_query_facets(dual_track_value))
        facets.extend(_prediction_error_query_facets(prediction_error_value))
        return tuple(facets)


def _temporal_query_facets(temporal_value: Any) -> tuple[str, ...]:
    from volvence_zero.temporal.interface import TemporalAbstractionSnapshot

    if not isinstance(temporal_value, TemporalAbstractionSnapshot):
        return ()
    return (
        f"temporal:{temporal_value.active_abstract_action}",
        f"temporal:steps_since_switch:{temporal_value.controller_state.steps_since_switch}",
    )


def _temporal_feedback_signal(temporal_value: Any) -> tuple[float, ...]:
    from volvence_zero.temporal.interface import TemporalAbstractionSnapshot

    if not isinstance(temporal_value, TemporalAbstractionSnapshot):
        return ()
    return temporal_value.memory_feedback_signal


def _dual_track_query_facets(dual_track_value: Any) -> tuple[str, ...]:
    from volvence_zero.dual_track.core import DualTrackSnapshot

    if not isinstance(dual_track_value, DualTrackSnapshot):
        return ()
    facets: list[str] = []
    for goal in dual_track_value.world_track.active_goals[:2]:
        facets.append(f"world-goal:{goal}")
    for goal in dual_track_value.self_track.active_goals[:2]:
        facets.append(f"self-goal:{goal}")
    facets.append(f"cross-track:{dual_track_value.cross_track_tension:.2f}")
    return tuple(facets)


def _prediction_error_query_facets(prediction_error_value: "PredictionErrorSnapshot | None") -> tuple[str, ...]:
    if prediction_error_value is None or prediction_error_value.bootstrap:
        return ()
    pe = prediction_error_value.error
    dominant_dimension = max(
        (
            ("task", abs(pe.task_error)),
            ("relationship", abs(pe.relationship_error)),
            ("regime", abs(pe.regime_error)),
            ("action", abs(pe.action_error)),
        ),
        key=lambda item: item[1],
    )[0]
    return (
        f"prediction_error:{dominant_dimension}",
        f"prediction_reward:{pe.signed_reward:.2f}",
    )
