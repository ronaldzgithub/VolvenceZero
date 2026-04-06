from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping
from uuid import uuid4

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
    description: str


def _clamp_strength(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tokenize(text: str) -> set[str]:
    return {part.strip().lower() for part in text.split() if part.strip()}


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

        if request.stratum in {MemoryStratum.TRANSIENT, MemoryStratum.EPISODIC}:
            self._pending_promotions.append(entry.entry_id)
        if request.stratum is MemoryStratum.TRANSIENT and entry.strength < 0.4:
            self._pending_decays.append(entry.entry_id)
        return entry

    def retrieve(self, query: RetrievalQuery, *, timestamp_ms: int) -> RetrievalResult:
        tokens = _tokenize(query.text)
        strata = query.strata or tuple(MemoryStratum)
        matches: list[tuple[int, MemoryEntry]] = []
        for stratum in strata:
            for entry_id in self._by_stratum[stratum]:
                entry = self._entries[entry_id]
                if query.track is not None and entry.track is not query.track:
                    continue
                score = self._score_entry(entry, tokens)
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
            description=description,
        )

    def _entries_for(self, stratum: MemoryStratum) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries[entry_id] for entry_id in self._by_stratum[stratum])

    def _score_entry(self, entry: MemoryEntry, query_tokens: set[str]) -> int:
        if not query_tokens:
            return 1
        score = 0
        content_tokens = _tokenize(entry.content)
        tag_tokens = {token.lower() for token in entry.tags}
        overlap = len(query_tokens & content_tokens)
        score += overlap * 3
        score += len(query_tokens & tag_tokens) * 2
        if score == 0 and any(token in entry.content.lower() for token in query_tokens):
            score = 1
        return score


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
    if substrate_snapshot.surface_kind is SurfaceKind.FEATURE_SURFACE:
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
) -> RetrievalQuery:
    query_parts: list[str] = []
    if user_text:
        query_parts.append(user_text)
    if substrate_snapshot is not None:
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
    )


def _query_parts_from_feature_surface(feature_surface: tuple[FeatureSignal, ...]) -> tuple[str, ...]:
    return tuple(feature.name for feature in feature_surface[:3])


class MemoryModule(RuntimeModule[MemorySnapshot]):
    slot_name = "memory"
    owner = "MemoryModule"
    value_type = MemorySnapshot
    dependencies = ("substrate",)
    default_wiring_level = WiringLevel.SHADOW

    def __init__(self, *, store: MemoryStore | None = None, wiring_level: WiringLevel | None = None) -> None:
        super().__init__(wiring_level=wiring_level)
        self._store = store or MemoryStore()

    @property
    def store(self) -> MemoryStore:
        return self._store

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[MemorySnapshot]:
        substrate_snapshot = upstream["substrate"]
        substrate_value = substrate_snapshot.value if isinstance(substrate_snapshot.value, SubstrateSnapshot) else None
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
                track=Track.SHARED,
            ),
            timestamp_ms=substrate_snapshot.timestamp_ms,
        )
        return self.publish(self._store.snapshot(retrieved_entries=retrieval.entries))

    async def process_standalone(self, **kwargs: object) -> Snapshot[MemorySnapshot]:
        timestamp_ms = int(kwargs.get("timestamp_ms", 0)) or 1
        user_text = kwargs.get("user_text")
        if user_text is not None and not isinstance(user_text, str):
            raise TypeError("user_text must be a string when provided.")

        substrate_snapshot = kwargs.get("substrate_snapshot")
        substrate_value = substrate_snapshot if isinstance(substrate_snapshot, SubstrateSnapshot) else None

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
                track=Track.SHARED,
            ),
            timestamp_ms=timestamp_ms,
        )
        return self.publish(self._store.snapshot(retrieved_entries=retrieval.entries))
