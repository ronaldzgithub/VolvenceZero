from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping
from uuid import uuid4

from volvence_zero.memory.cms import CMSCheckpointState, CMSMemoryCore, CMSState
from volvence_zero.memory.persistence import (
    PersistenceBackend,
    deserialize_checkpoint,
    serialize_checkpoint,
)
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.substrate import FeatureSignal, SubstrateSnapshot, SurfaceKind


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


class MemoryStore:
    """Owner-controlled memory store with per-stratum indexes."""

    def __init__(
        self,
        *,
        learned_core: CMSMemoryCore | None = None,
        promotion_threshold: float = 0.7,
        persistence_backend: PersistenceBackend | None = None,
    ) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._by_stratum: dict[MemoryStratum, list[str]] = {
            MemoryStratum.TRANSIENT: [],
            MemoryStratum.EPISODIC: [],
            MemoryStratum.DURABLE: [],
            MemoryStratum.DERIVED: [],
        }
        self._pending_promotions: list[str] = []
        self._pending_decays: list[str] = []
        self._learned_core = learned_core
        self._promotion_threshold = _clamp_strength(promotion_threshold)
        self._semantic_index: dict[str, tuple[float, ...]] = {}
        self._persistence_backend = persistence_backend
        self._persistence_version = 0

    @property
    def learned_core(self) -> CMSMemoryCore | None:
        return self._learned_core

    @property
    def promotion_threshold(self) -> float:
        return self._promotion_threshold

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
        self._entries[entry.entry_id] = entry
        self._by_stratum[request.stratum].append(entry.entry_id)
        self._semantic_index[entry.entry_id] = _semantic_embedding(
            text=entry.content,
            tags=entry.tags,
        )

        if request.stratum in {MemoryStratum.TRANSIENT, MemoryStratum.EPISODIC}:
            self._pending_promotions.append(entry.entry_id)
        if request.stratum is MemoryStratum.TRANSIENT and entry.strength < 0.4:
            self._pending_decays.append(entry.entry_id)
        return entry

    def retrieve(self, query: RetrievalQuery, *, timestamp_ms: int) -> RetrievalResult:
        tokens = _tokenize(query.text)
        query_embedding = _semantic_embedding(
            text=" ".join(part for part in (query.text, *query.facets) if part),
            tags=query.facets,
        )
        strata = query.strata or tuple(MemoryStratum)
        matches: list[tuple[float, MemoryEntry]] = []
        for stratum in strata:
            for entry_id in self._by_stratum[stratum]:
                entry = self._entries[entry_id]
                if query.track is not None and entry.track is not query.track:
                    continue
                score = self._score_entry(entry, tokens, query_embedding)
                if score <= 0:
                    continue
                updated = replace(entry, last_accessed_ms=timestamp_ms)
                self._entries[entry.entry_id] = updated
                matches.append((score, updated))
        matches.sort(key=lambda item: (-item[0], -item[1].strength, -item[1].created_at_ms))
        return RetrievalResult(
            query=query,
            entries=tuple(entry for _, entry in matches[: query.limit]),
        )

    def snapshot(self, *, retrieved_entries: tuple[MemoryEntry, ...]) -> MemorySnapshot:
        transient_entries = self._entries_for(MemoryStratum.TRANSIENT)
        episodic_entries = self._entries_for(MemoryStratum.EPISODIC)
        durable_entries = self._entries_for(MemoryStratum.DURABLE)
        total_entries = tuple(
            (stratum.value, len(self._entries_for(stratum))) for stratum in MemoryStratum
        )
        description = (
            f"Memory store with {len(self._entries)} entries across {len(MemoryStratum)} strata; "
            f"{len(retrieved_entries)} retrieved this turn."
        )
        if self._learned_core is not None:
            description += f" {self._learned_core.snapshot().description}"
        return MemorySnapshot(
            transient_summary=summarize_entries(
                transient_entries, fallback="No transient working-state memories."
            ),
            episodic_summary=summarize_entries(
                episodic_entries, fallback="No episodic session-state memories."
            ),
            durable_summary=summarize_entries(
                durable_entries, fallback="No durable semantic memories."
            ),
            retrieved_entries=retrieved_entries,
            total_entries_by_stratum=total_entries,
            pending_promotions=len(self._pending_promotions),
            pending_decays=len(self._pending_decays),
            cms_state=self._learned_core.snapshot() if self._learned_core is not None else None,
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
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            if entry.strength < self._promotion_threshold:
                continue
            updated = replace(
                entry,
                stratum=MemoryStratum.DURABLE.value,
                strength=_clamp_strength(max(entry.strength, 0.55 + promotion_boost * 0.35)),
                last_accessed_ms=timestamp_ms,
            )
            self._entries[entry_id] = updated
            if entry_id in self._by_stratum[MemoryStratum.TRANSIENT]:
                self._by_stratum[MemoryStratum.TRANSIENT].remove(entry_id)
            if entry_id in self._by_stratum[MemoryStratum.EPISODIC]:
                self._by_stratum[MemoryStratum.EPISODIC].remove(entry_id)
            if entry_id not in self._by_stratum[MemoryStratum.DURABLE]:
                self._by_stratum[MemoryStratum.DURABLE].append(entry_id)
            applied.append(f"promoted:{entry_id}")
        for entry in new_durable_entries:
            if entry.strength < self._promotion_threshold:
                continue
            self._entries[entry.entry_id] = entry
            self._semantic_index[entry.entry_id] = _semantic_embedding(
                text=entry.content,
                tags=entry.tags,
            )
            if entry.entry_id not in self._by_stratum[MemoryStratum.DURABLE]:
                self._by_stratum[MemoryStratum.DURABLE].append(entry.entry_id)
            applied.append(f"durable:{entry.entry_id}")
        for entry_id in decayed_entries:
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            self._entries[entry_id] = replace(
                entry,
                strength=_clamp_strength(entry.strength * max(0.55, 1.0 - decay_scale * 0.35)),
                last_accessed_ms=timestamp_ms,
            )
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
            self._entries[belief_entry.entry_id] = belief_entry
            self._semantic_index[belief_entry.entry_id] = _semantic_embedding(
                text=belief_entry.content,
                tags=belief_entry.tags,
            )
            if belief_entry.entry_id not in self._by_stratum[MemoryStratum.DURABLE]:
                self._by_stratum[MemoryStratum.DURABLE].append(belief_entry.entry_id)
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
            entries=tuple(self._entries.values()),
            pending_promotions=tuple(self._pending_promotions),
            pending_decays=tuple(self._pending_decays),
            cms_state=self._learned_core.export_state() if self._learned_core is not None else None,
            promotion_threshold=self._promotion_threshold,
            semantic_index=tuple(sorted(self._semantic_index.items())),
        )

    def export_rare_heavy_state(self, *, checkpoint_id: str | None = None) -> MemoryStoreCheckpoint:
        return self.create_checkpoint(checkpoint_id=checkpoint_id or "rare-heavy-memory")

    def restore_checkpoint(self, checkpoint: MemoryStoreCheckpoint) -> None:
        self._entries = {entry.entry_id: entry for entry in checkpoint.entries}
        self._by_stratum = {
            stratum: [entry.entry_id for entry in checkpoint.entries if entry.stratum == stratum.value]
            for stratum in MemoryStratum
        }
        self._pending_promotions = list(checkpoint.pending_promotions)
        self._pending_decays = list(checkpoint.pending_decays)
        self._promotion_threshold = checkpoint.promotion_threshold
        self._semantic_index = dict(checkpoint.semantic_index)
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
        return tuple(self._entries[entry_id] for entry_id in self._by_stratum[stratum])

    def _score_entry(
        self,
        entry: MemoryEntry,
        query_tokens: set[str],
        query_embedding: tuple[float, ...],
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
        semantic_score = _cosine_similarity(
            query_embedding,
            self._semantic_index.get(entry.entry_id, (0.0, 0.0, 0.0)),
        )
        return lexical_score * 0.7 + semantic_score * 3.0


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
    dependencies = ("substrate", "temporal_abstraction", "dual_track")
    default_wiring_level = WiringLevel.SHADOW

    def __init__(self, *, store: MemoryStore | None = None, wiring_level: WiringLevel | None = None) -> None:
        super().__init__(wiring_level=wiring_level)
        self._store = store or MemoryStore()

    @property
    def store(self) -> MemoryStore:
        return self._store

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[MemorySnapshot]:
        substrate_snapshot = upstream["substrate"]
        temporal_snapshot = upstream.get("temporal_abstraction")
        dual_track_snapshot = upstream.get("dual_track")
        substrate_value = substrate_snapshot.value if isinstance(substrate_snapshot.value, SubstrateSnapshot) else None
        self._store.observe_substrate(
            substrate_snapshot=substrate_value,
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
                ),
            ),
            timestamp_ms=substrate_snapshot.timestamp_ms,
        )
        return self.publish(self._store.snapshot(retrieved_entries=retrieval.entries))

    async def process_standalone(self, **kwargs: object) -> Snapshot[MemorySnapshot]:
        timestamp_ms = int(kwargs.get("timestamp_ms", 0)) or 1
        user_text = kwargs.get("user_text")
        if user_text is not None and not isinstance(user_text, str):
            raise TypeError("user_text must be a string when provided.")
        query_facets = kwargs.get("query_facets", ())
        if not isinstance(query_facets, tuple):
            raise TypeError("query_facets must be a tuple when provided.")
        temporal_snapshot = kwargs.get("temporal_snapshot")
        dual_track_snapshot = kwargs.get("dual_track_snapshot")

        substrate_snapshot = kwargs.get("substrate_snapshot")
        substrate_value = substrate_snapshot if isinstance(substrate_snapshot, SubstrateSnapshot) else None
        self._store.observe_substrate(
            substrate_snapshot=substrate_value,
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
        return tuple(facets)


def _temporal_query_facets(temporal_value: Any) -> tuple[str, ...]:
    from volvence_zero.temporal.interface import TemporalAbstractionSnapshot

    if not isinstance(temporal_value, TemporalAbstractionSnapshot):
        return ()
    return (
        f"temporal:{temporal_value.active_abstract_action}",
        f"temporal:steps_since_switch:{temporal_value.controller_state.steps_since_switch}",
    )


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
