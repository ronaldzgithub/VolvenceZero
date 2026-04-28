from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Mapping
from uuid import uuid4

from volvence_zero.learned_update import LearnedUpdateDecision, LearnedUpdateRuleState
from volvence_zero.memory.cms import (
    CMSCheckpointState,
    CMSHopeSelfModificationState,
    CMSMemoryCore,
    CMSState,
    CMSTowerConsolidationUpdate,
    CMSTowerProfile,
)
from volvence_zero.memory.persistence import (
    PersistenceBackend,
    deserialize_checkpoint,
    serialize_checkpoint,
)
from volvence_zero.memory.runtime_evidence import (
    build_runtime_backbone_evidence,
    cosine_alignment,
)
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.substrate import FeatureSignal, SubstrateSnapshot, SurfaceKind

if TYPE_CHECKING:
    from volvence_zero.prediction.error import PredictionErrorSnapshot


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
            update_rule_raw = cms_raw.get("update_rule_state")
            update_rule_state = None
            if isinstance(update_rule_raw, dict):
                update_rule_state = LearnedUpdateRuleState(
                    rule_id=str(update_rule_raw["rule_id"]),
                    feature_dim=int(update_rule_raw["feature_dim"]),
                    hidden_dim=int(update_rule_raw["hidden_dim"]),
                    update_count=int(update_rule_raw["update_count"]),
                    last_feature_norm=float(update_rule_raw["last_feature_norm"]),
                    last_improvement=float(update_rule_raw["last_improvement"]),
                    last_guard_reason=str(update_rule_raw.get("last_guard_reason", "")),
                    input_projection=tuple(
                        tuple(float(v) for v in row) for row in update_rule_raw.get("input_projection", ())
                    ),
                    hidden_bias=tuple(float(v) for v in update_rule_raw.get("hidden_bias", ())),
                    output_projection=tuple(
                        tuple(float(v) for v in row) for row in update_rule_raw.get("output_projection", ())
                    ),
                    output_bias=tuple(float(v) for v in update_rule_raw.get("output_bias", ())),
                    last_decisions=tuple(
                        LearnedUpdateDecision(
                            target_id=str(item["target_id"]),
                            write_gate=float(item["write_gate"]),
                            step_scale=float(item["step_scale"]),
                            momentum_gate=float(item["momentum_gate"]),
                            slow_mix=float(item["slow_mix"]),
                            reset_mix=float(item["reset_mix"]),
                            bias_delta=float(item["bias_delta"]),
                            confidence=float(item["confidence"]),
                            guard_applied=bool(item.get("guard_applied", False)),
                            guard_reason=str(item.get("guard_reason", "")),
                            description=str(item.get("description", "")),
                        )
                        for item in update_rule_raw.get("last_decisions", ())
                    ),
                    base_learning_rate=float(update_rule_raw.get("base_learning_rate", 0.0)),
                    last_effective_learning_rate=float(
                        update_rule_raw.get("last_effective_learning_rate", 0.0)
                    ),
                    last_reward=float(update_rule_raw.get("last_reward", 0.0)),
                    last_stability=float(update_rule_raw.get("last_stability", 0.0)),
                    last_write_gate=float(update_rule_raw.get("last_write_gate", 0.0)),
                    last_step_scale=float(update_rule_raw.get("last_step_scale", 0.0)),
                    last_momentum_gate=float(update_rule_raw.get("last_momentum_gate", 0.0)),
                    last_slow_mix=float(update_rule_raw.get("last_slow_mix", 0.0)),
                    last_reset_mix=float(update_rule_raw.get("last_reset_mix", 0.0)),
                    last_confidence=float(update_rule_raw.get("last_confidence", 0.0)),
                    description=str(update_rule_raw.get("description", "")),
                )
            hope_raw = cms_raw.get("hope_self_modification_state")
            hope_state = None
            if isinstance(hope_raw, dict):
                hope_state = CMSHopeSelfModificationState(
                    enabled=bool(hope_raw.get("enabled", True)),
                    update_count=int(hope_raw.get("update_count", 0)),
                    last_target_id=str(hope_raw.get("last_target_id", "")),
                    generated_learning_rate=float(hope_raw.get("generated_learning_rate", 0.0)),
                    generated_decay_rate=float(hope_raw.get("generated_decay_rate", 0.0)),
                    generated_reset_rate=float(hope_raw.get("generated_reset_rate", 0.0)),
                    last_improvement=float(hope_raw.get("last_improvement", 0.0)),
                    last_stability=float(hope_raw.get("last_stability", 0.0)),
                    last_reward=float(hope_raw.get("last_reward", 0.0)),
                    guarded=bool(hope_raw.get("guarded", False)),
                    guard_reason=str(hope_raw.get("guard_reason", "")),
                    description=str(hope_raw.get("description", "")),
                )
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
                nested_session_init_target=tuple(
                    float(v) for v in cms_raw.get("nested_session_init_target", ())
                ),
                nested_online_init_target=tuple(
                    float(v) for v in cms_raw.get("nested_online_init_target", ())
                ),
                tower_meta_levels=tuple(
                    (str(level[0]), tuple(float(v) for v in level[1]))
                    for level in cms_raw.get("tower_meta_levels", ())
                ),
                update_rule_state=update_rule_state,
                hope_self_modification_state=hope_state,
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


def _mean_abs(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _align_signal(signal: tuple[float, ...], *, dim: int) -> tuple[float, ...]:
    if len(signal) == dim:
        return signal
    if not signal:
        return tuple(0.0 for _ in range(dim))
    return tuple(signal[index % len(signal)] for index in range(dim))


def _blend_signals(
    *,
    dim: int,
    weighted_signals: tuple[tuple[tuple[float, ...], float], ...],
) -> tuple[float, ...]:
    total_weight = sum(weight for _, weight in weighted_signals if weight > 0.0)
    if total_weight <= 1e-6:
        return tuple(0.0 for _ in range(dim))
    blended = [0.0 for _ in range(dim)]
    for signal, weight in weighted_signals:
        if weight <= 0.0:
            continue
        aligned = _align_signal(signal, dim=dim)
        for index in range(dim):
            blended[index] += aligned[index] * weight
    return tuple(_clamp_strength(value / total_weight) for value in blended)


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
    query_base_signal: tuple[float, ...]
    query_signal: tuple[float, ...]
    core_signal: tuple[float, ...]
    tower_profile_id: str
    tower_depth: int
    retrieval_confidence: float
    tower_alignment: float
    query_only_alignment: float
    composite_alignment: float
    transfer_alignment: float
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

    def index_entry(self, entry: MemoryEntry, *, embedding: tuple[float, ...] | None = None) -> None:
        self._artifact_embeddings[entry.entry_id] = embedding or _semantic_embedding(
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
        self._fast_memory_signal_count = 0
        self._last_fast_memory_signal: tuple[float, ...] = ()
        self._last_fast_memory_signal_norm = 0.0
        self._last_fast_memory_runtime_alignment = 0.0
        self._runtime_backbone_observation_count = 0
        self._last_runtime_backbone_signal: tuple[float, ...] = ()
        self._last_runtime_backbone_signal_norm = 0.0
        self._last_runtime_backbone_signal_quality = 0.0
        self._last_runtime_backbone_strength = 0.0
        self._last_runtime_backbone_hook_coverage = 0.0
        self._last_runtime_backbone_fallback_active = 0.0
        self._last_runtime_backbone_residual_stream_active = 0.0
        self._last_runtime_backbone_sequence_density = 0.0
        self._last_runtime_backbone_activation_density = 0.0
        self._tower_consolidation_count = 0
        self._last_tower_depth = 0
        self._last_tower_alignment = 0.0
        self._last_tower_profile_id = "artifact-only"

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
        self._derived_index.index_entry(entry, embedding=self._entry_signal(entry))
        self._observe_artifact_entry(entry=entry, timestamp_ms=timestamp_ms, source="write")
        return entry

    def retrieve(self, query: RetrievalQuery, *, timestamp_ms: int) -> RetrievalResult:
        tokens = _tokenize(query.text)
        query_embedding = self._query_base_signal(query)
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
        self._last_tower_depth = learned_recall.tower_depth
        self._last_tower_alignment = learned_recall.tower_alignment
        self._last_tower_profile_id = learned_recall.tower_profile_id
        return RetrievalResult(
            query=query,
            entries=tuple(entry for _, entry in matches[: query.limit]),
        )

    def snapshot(self, *, retrieved_entries: tuple[MemoryEntry, ...]) -> MemorySnapshot:
        transient_entries = self._entries_for(MemoryStratum.TRANSIENT)
        episodic_entries = self._entries_for(MemoryStratum.EPISODIC)
        durable_entries = self._entries_for(MemoryStratum.DURABLE)
        total_entries = self._artifact_store.total_entries_by_stratum()
        cms_state = self._learned_core.snapshot() if self._learned_core is not None else None
        updater_state = cms_state.update_rule_state if cms_state is not None else None
        hope_state = cms_state.hope_self_modification_state if cms_state is not None else None
        continuum_profile = cms_state.continuum_profile if cms_state is not None else None
        continuum_band_count = len(continuum_profile.bands) if continuum_profile is not None else 0
        continuum_reconstruction_edge_count = (
            len(continuum_profile.reconstruction_edges) if continuum_profile is not None else 0
        )
        continuum_frequency_span = (
            max((band.update_frequency for band in continuum_profile.bands), default=0.0)
            - min((band.update_frequency for band in continuum_profile.bands), default=0.0)
            if continuum_profile is not None and continuum_profile.bands
            else 0.0
        )
        continuum_retrieval_mass = (
            sum(band.retrieval_weight for band in continuum_profile.bands)
            if continuum_profile is not None
            else 0.0
        )
        description = (
            f"Memory store with learned core primary and {self._artifact_store.entry_count()} artifact entries "
            f"across {len(MemoryStratum)} strata; {len(retrieved_entries)} retrieved this turn."
        )
        if cms_state is not None:
            description += f" {cms_state.description}"
        if hope_state is not None and hope_state.update_count > 0:
            description += f" {hope_state.description}"
        if continuum_profile is not None:
            description += (
                f" continuum_profile={continuum_profile.profile_id} "
                f"bands={continuum_band_count} reconstruction_edges={continuum_reconstruction_edge_count}."
            )
        if self._last_context_reset_reason:
            description += (
                f" last_reset_reason={self._last_context_reset_reason} "
                f"applied={self._last_context_reset_applied} count={self._context_reset_count}."
            )
        touched_bands = ()
        touched_param_count = 0.0
        total_band_param_count = 0.0
        if cms_state is not None:
            touched_bands = tuple(
                band.name
                for band in (
                    cms_state.online_fast,
                    cms_state.session_medium,
                    cms_state.background_slow,
                )
                if band.update_gate > 0.05 or band.effective_learning_rate > 0.0
            )
            touched_param_count = float(
                sum(
                    band.mlp_param_count
                    for band in (
                        cms_state.online_fast,
                        cms_state.session_medium,
                        cms_state.background_slow,
                    )
                    if band.update_gate > 0.05 or band.effective_learning_rate > 0.0
                )
            )
            total_band_param_count = float(
                sum(
                    band.mlp_param_count
                    for band in (
                        cms_state.online_fast,
                        cms_state.session_medium,
                        cms_state.background_slow,
                    )
                )
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
            cms_state=cms_state,
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
                ("tower_consolidation_count", float(self._tower_consolidation_count)),
                ("learned_recall_count", float(self._learned_recall_count)),
                ("last_learned_recall_confidence", self._last_recall_confidence),
                ("last_learned_recall_driver_is_core", float(self._last_recall_driver == "learned-core-guided")),
                ("last_memory_tower_depth", float(self._last_tower_depth)),
                ("last_memory_tower_alignment", self._last_tower_alignment),
                ("continuum_band_count", float(continuum_band_count)),
                ("continuum_reconstruction_edge_count", float(continuum_reconstruction_edge_count)),
                ("continuum_frequency_span", continuum_frequency_span),
                ("continuum_retrieval_mass", continuum_retrieval_mass),
                ("runtime_backbone_observation_count", float(self._runtime_backbone_observation_count)),
                ("last_runtime_backbone_signal_norm", self._last_runtime_backbone_signal_norm),
                ("last_runtime_backbone_signal_quality", self._last_runtime_backbone_signal_quality),
                ("last_runtime_backbone_signal_strength", self._last_runtime_backbone_strength),
                ("last_runtime_backbone_hook_coverage", self._last_runtime_backbone_hook_coverage),
                ("last_runtime_backbone_fallback_active", self._last_runtime_backbone_fallback_active),
                (
                    "last_runtime_backbone_residual_stream_active",
                    self._last_runtime_backbone_residual_stream_active,
                ),
                ("last_runtime_backbone_sequence_density", self._last_runtime_backbone_sequence_density),
                ("last_runtime_backbone_activation_density", self._last_runtime_backbone_activation_density),
                ("fast_memory_signal_count", float(self._fast_memory_signal_count)),
                ("last_fast_memory_signal_norm", self._last_fast_memory_signal_norm),
                ("last_fast_memory_runtime_alignment", self._last_fast_memory_runtime_alignment),
                (
                    "memory_updater_effective_lr",
                    updater_state.last_effective_learning_rate if updater_state is not None else 0.0,
                ),
                ("memory_updater_reward", updater_state.last_reward if updater_state is not None else 0.0),
                (
                    "memory_updater_write_gate",
                    updater_state.last_write_gate if updater_state is not None else 0.0,
                ),
                (
                    "memory_updater_slow_mix",
                    updater_state.last_slow_mix if updater_state is not None else 0.0,
                ),
                (
                    "memory_updater_confidence",
                    updater_state.last_confidence if updater_state is not None else 0.0,
                ),
                (
                    "memory_updater_decision_count",
                    float(len(updater_state.last_decisions)) if updater_state is not None else 0.0,
                ),
                ("memory_updater_active_band_count", float(len(touched_bands))),
                (
                    "memory_updater_touched_param_ratio",
                    touched_param_count / total_band_param_count if total_band_param_count > 0.0 else 0.0,
                ),
                ("hope_self_mod_update_count", float(hope_state.update_count) if hope_state is not None else 0.0),
                (
                    "hope_generated_learning_rate",
                    hope_state.generated_learning_rate if hope_state is not None else 0.0,
                ),
                (
                    "hope_generated_decay_rate",
                    hope_state.generated_decay_rate if hope_state is not None else 0.0,
                ),
                (
                    "hope_generated_reset_rate",
                    hope_state.generated_reset_rate if hope_state is not None else 0.0,
                ),
                ("hope_self_mod_guarded", float(hope_state.guarded) if hope_state is not None else 0.0),
            ),
            description=description,
        )

    def observe_substrate(self, *, substrate_snapshot: SubstrateSnapshot | None, timestamp_ms: int) -> None:
        evidence = build_runtime_backbone_evidence(
            substrate_snapshot=substrate_snapshot,
            dim=self._learned_signal_dim(),
        )
        self._runtime_backbone_observation_count += 1
        self._last_runtime_backbone_signal = evidence.signal
        self._last_runtime_backbone_signal_norm = evidence.signal_norm
        self._last_runtime_backbone_signal_quality = evidence.signal_quality
        self._last_runtime_backbone_strength = evidence.runtime_strength
        self._last_runtime_backbone_hook_coverage = evidence.hook_coverage
        self._last_runtime_backbone_fallback_active = evidence.fallback_active
        self._last_runtime_backbone_residual_stream_active = evidence.residual_stream_active
        self._last_runtime_backbone_sequence_density = evidence.sequence_density
        self._last_runtime_backbone_activation_density = evidence.activation_density
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

    def observe_fast_memory_signal(
        self,
        *,
        signal: tuple[float, ...],
        timestamp_ms: int,
    ) -> None:
        self._fast_memory_signal_count += 1
        self._last_fast_memory_signal = _align_signal(signal, dim=self._learned_signal_dim())
        self._last_fast_memory_signal_norm = _mean_abs(signal)
        self._last_fast_memory_runtime_alignment = cosine_alignment(
            self._last_fast_memory_signal,
            self._last_runtime_backbone_signal,
        )
        if self._learned_core is not None:
            self._learned_core.observe_fast_memory_signal(
                signal=signal,
                timestamp_ms=timestamp_ms,
            )

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
            self._derived_index.index_entry(updated, embedding=self._entry_signal(updated))
            self._observe_artifact_entry(entry=updated, timestamp_ms=timestamp_ms, source="promotion")
            applied.append(f"promoted:{entry_id}")
        for entry in new_durable_entries:
            if entry.strength < self._promotion_threshold:
                continue
            self._artifact_store.write(entry)
            self._derived_index.index_entry(entry, embedding=self._entry_signal(entry))
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
            self._derived_index.index_entry(belief_entry, embedding=self._entry_signal(belief_entry))
            self._observe_artifact_entry(entry=belief_entry, timestamp_ms=timestamp_ms, source="belief-update")
            applied.append(f"belief:{belief_entry.entry_id}")
        if self._learned_core is not None:
            tower_update = self._build_tower_consolidation_update(
                promoted_entries=promoted_entries,
                new_durable_entries=new_durable_entries,
                beliefs_updated=beliefs_updated,
                lesson_count=lesson_count,
                promotion_boost=promotion_boost,
                decay_scale=decay_scale,
            )
            tower_operations = self._learned_core.apply_tower_consolidation(
                update=tower_update,
                timestamp_ms=timestamp_ms,
            )
            applied = list(applied) + list(tower_operations)
            if any(operation.startswith("tower-consolidation:") for operation in tower_operations):
                self._tower_consolidation_count += 1
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
        return self._owner_signal(
            text=entry.content,
            tags=entry.tags,
            track=entry.track,
            stratum=entry.stratum,
            strength=entry.strength,
        )

    def _query_base_signal(self, query: RetrievalQuery) -> tuple[float, ...]:
        return self._owner_signal(
            text=query.text,
            tags=query.facets,
            track=query.track,
            stratum=None,
            strength=0.75,
        )

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

    def _owner_signal(
        self,
        *,
        text: str,
        tags: tuple[str, ...],
        track: Track | None,
        stratum: str | None,
        strength: float,
        base_signal: tuple[float, ...] = (),
    ) -> tuple[float, ...]:
        dim = self._learned_signal_dim()
        semantic = _semantic_embedding(
            text=" ".join(
                part
                for part in (
                    text,
                    track.value if track is not None else "",
                    stratum or "",
                )
                if part
            ),
            tags=tags
            + ((track.value,) if track is not None else ())
            + ((stratum,) if stratum is not None else ()),
            dim=dim,
        )
        metadata = _semantic_embedding(
            text=" ".join(
                part
                for part in (
                    track.value if track is not None else "",
                    stratum or "",
                )
                if part
            ),
            tags=tags,
            dim=dim,
        )
        return _blend_signals(
            dim=dim,
            weighted_signals=(
                (semantic, 0.62 + _clamp_strength(strength) * 0.18),
                (metadata, 0.18),
                (base_signal, 0.2 if base_signal else 0.0),
            ),
        )

    def _tower_profile(self) -> CMSTowerProfile | None:
        if self._learned_core is None:
            return None
        return self._learned_core.snapshot().tower_profile

    def _tower_signal(self) -> tuple[float, ...]:
        tower_profile = self._tower_profile()
        if tower_profile is None:
            return tuple(0.0 for _ in range(self._learned_signal_dim()))
        return tower_profile.readout_vector

    def _tower_level_signal(
        self,
        *,
        tower_profile: CMSTowerProfile | None,
        level_ids: tuple[str, ...],
    ) -> tuple[float, ...]:
        dim = self._learned_signal_dim()
        if tower_profile is None:
            return tuple(0.0 for _ in range(dim))
        matched_levels = tuple(
            level.vector
            for level in tower_profile.levels
            if level.level_id in level_ids
        )
        if not matched_levels:
            return tuple(0.0 for _ in range(dim))
        return _blend_signals(
            dim=dim,
            weighted_signals=tuple((vector, 1.0) for vector in matched_levels),
        )

    def _recent_fast_memory_signal(self) -> tuple[float, ...]:
        return _align_signal(self._last_fast_memory_signal, dim=self._learned_signal_dim())

    def _transfer_alignment_signal(self, *, tower_profile: CMSTowerProfile | None) -> tuple[float, ...]:
        dim = self._learned_signal_dim()
        nested_prior_signal = self._tower_level_signal(
            tower_profile=tower_profile,
            level_ids=("nested-online-prior", "nested-session-prior"),
        )
        if not any(nested_prior_signal):
            return tuple(0.0 for _ in range(dim))
        transfer_pressure = _clamp_strength(
            self._last_context_reset_transfer_strength
            + max(self._last_context_reset_target_alignment_gain, 0.0) * 2.0
            + self._last_context_reset_online_seed_strength * 0.15
            + self._last_context_reset_session_seed_strength * 0.10
        )
        if transfer_pressure <= 1e-6:
            return tuple(0.0 for _ in range(dim))
        seeded_signal = _blend_signals(
            dim=dim,
            weighted_signals=(
                (nested_prior_signal, 0.68),
                (self._tower_signal(), 0.32),
            ),
        )
        return tuple(_clamp_strength(value * transfer_pressure) for value in seeded_signal)

    def _tower_context_signal(
        self,
        *,
        tower_profile: CMSTowerProfile | None,
        core_signal: tuple[float, ...],
    ) -> tuple[float, ...]:
        dim = len(core_signal)
        if tower_profile is None:
            return tuple(0.0 for _ in range(dim))
        online_signal = self._tower_level_signal(tower_profile=tower_profile, level_ids=("online-fast",))
        session_signal = self._tower_level_signal(tower_profile=tower_profile, level_ids=("session-medium",))
        background_signal = self._tower_level_signal(tower_profile=tower_profile, level_ids=("background-slow",))
        nested_prior_signal = self._tower_level_signal(
            tower_profile=tower_profile,
            level_ids=("nested-online-prior", "nested-session-prior"),
        )
        return _blend_signals(
            dim=dim,
            weighted_signals=(
                (core_signal, 0.30),
                (background_signal, 0.24 if any(background_signal) else 0.0),
                (session_signal, 0.20 if any(session_signal) else 0.0),
                (online_signal, 0.12 if any(online_signal) else 0.0),
                (nested_prior_signal, 0.14 if any(nested_prior_signal) else 0.0),
            ),
        )

    def _average_signal(self, signals: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
        if not signals:
            return tuple(0.0 for _ in range(self._learned_signal_dim()))
        return _blend_signals(
            dim=self._learned_signal_dim(),
            weighted_signals=tuple((signal, 1.0) for signal in signals),
        )

    def _build_tower_consolidation_update(
        self,
        *,
        promoted_entries: tuple[str, ...],
        new_durable_entries: tuple[MemoryEntry, ...],
        beliefs_updated: tuple[str, ...],
        lesson_count: int,
        promotion_boost: float,
        decay_scale: float,
    ) -> CMSTowerConsolidationUpdate:
        tower_profile = self._tower_profile()
        promoted_signals = tuple(
            self._entry_signal(entry)
            for entry_id in promoted_entries
            if (entry := self._artifact_store.get(entry_id)) is not None
        )
        durable_signals = tuple(self._entry_signal(entry) for entry in new_durable_entries)
        belief_signals = tuple(
            self._owner_signal(
                text=belief,
                tags=("belief_update",),
                track=Track.SHARED,
                stratum=MemoryStratum.DURABLE.value,
                strength=0.55 + promotion_boost * 0.2,
            )
            for belief in beliefs_updated
        )
        lesson_signal = tuple(
            _clamp_strength(lesson_count / (index + 3))
            for index in range(self._learned_signal_dim())
        )
        recent_fast_signal = self._recent_fast_memory_signal()
        transfer_signal = self._transfer_alignment_signal(tower_profile=tower_profile)
        session_signal = _blend_signals(
            dim=self._learned_signal_dim(),
            weighted_signals=(
                (lesson_signal, 0.3 if lesson_count else 0.0),
                (self._average_signal(promoted_signals), 0.25 if promoted_signals else 0.0),
                (self._average_signal(durable_signals), 0.25 if durable_signals else 0.0),
                (self._average_signal(belief_signals), 0.2 if belief_signals else 0.0),
                (recent_fast_signal, 0.16 if any(recent_fast_signal) else 0.0),
                (transfer_signal, 0.12 if any(transfer_signal) else 0.0),
            ),
        )
        background_signal = _blend_signals(
            dim=self._learned_signal_dim(),
            weighted_signals=(
                (lesson_signal, 0.2 if lesson_count else 0.0),
                (self._average_signal(durable_signals), 0.32 if durable_signals else 0.0),
                (self._average_signal(belief_signals), 0.25 if belief_signals else 0.0),
                (self._tower_signal(), 0.15 if self._learned_core is not None else 0.0),
                (session_signal, 0.12 if any(session_signal) else 0.0),
                (transfer_signal, 0.16 if any(transfer_signal) else 0.0),
            ),
        )
        online_signal = _blend_signals(
            dim=self._learned_signal_dim(),
            weighted_signals=(
                (self._average_signal(promoted_signals), 0.28 if promoted_signals else 0.0),
                (self._average_signal(belief_signals), 0.2 if belief_signals else 0.0),
                (lesson_signal, 0.15 if lesson_count else 0.0),
                (session_signal, 0.3 if any(session_signal) else 0.0),
                (recent_fast_signal, 0.18 if any(recent_fast_signal) else 0.0),
                (transfer_signal, 0.12 if any(transfer_signal) else 0.0),
            ),
        )
        return CMSTowerConsolidationUpdate(
            online_signal=online_signal,
            session_signal=session_signal,
            background_signal=background_signal,
            decay_pressure=_clamp_strength(decay_scale),
            reset_fast_context=(
                self._learned_core is not None
                and self._learned_core.variant == "nested"
                and promotion_boost >= 0.45
                and lesson_count > 0
            ),
            description=(
                f"tower update lessons={lesson_count} promoted={len(promoted_entries)} "
                f"durable={len(new_durable_entries)} beliefs={len(beliefs_updated)} "
                f"promotion_boost={promotion_boost:.2f} decay_scale={decay_scale:.2f}"
            ),
        )

    def _build_learned_recall(
        self,
        *,
        query: RetrievalQuery,
        query_embedding: tuple[float, ...],
    ) -> LearnedMemoryRecall:
        dim = self._learned_signal_dim()
        projected_query = self._query_base_signal(query)
        if self._learned_core is None:
            return LearnedMemoryRecall(
                query_base_signal=projected_query,
                query_signal=projected_query,
                core_signal=tuple(0.0 for _ in range(dim)),
                tower_profile_id="artifact-only",
                tower_depth=0,
                retrieval_confidence=0.0,
                tower_alignment=0.0,
                query_only_alignment=0.0,
                composite_alignment=0.0,
                transfer_alignment=0.0,
                artifact_weight=2.6,
                learned_weight=0.0,
                description="Artifact-only retrieval because no learned core is active.",
            )
        cms_state = self._learned_core.snapshot()
        tower_profile = cms_state.tower_profile
        core_signal = tower_profile.readout_vector if tower_profile is not None else self._tower_signal()
        tower_context_signal = self._tower_context_signal(
            tower_profile=tower_profile,
            core_signal=core_signal,
        )
        fast_memory_signal = self._recent_fast_memory_signal()
        transfer_signal = self._transfer_alignment_signal(tower_profile=tower_profile)
        composite_anchor = _blend_signals(
            dim=len(core_signal),
            weighted_signals=(
                (projected_query, 0.18),
                (core_signal, 0.28),
                (tower_context_signal, 0.24 if any(tower_context_signal) else 0.0),
                (fast_memory_signal, 0.16 if any(fast_memory_signal) else 0.0),
                (transfer_signal, 0.14 if any(transfer_signal) else 0.0),
            ),
        )
        query_signal = _blend_signals(
            dim=len(core_signal),
            weighted_signals=(
                (projected_query, 0.42),
                (composite_anchor, 0.58),
            ),
        )
        query_only_alignment = _cosine_similarity(projected_query, core_signal)
        composite_alignment = _blend_signals(
            dim=1,
            weighted_signals=(
                (( _cosine_similarity(query_signal, core_signal),), 0.45),
                (( _cosine_similarity(query_signal, composite_anchor),), 0.35),
                (( _cosine_similarity(query_signal, tower_context_signal),), 0.20 if any(tower_context_signal) else 0.0),
            ),
        )[0]
        transfer_alignment = (
            _blend_signals(
                dim=1,
                weighted_signals=(
                    ((_cosine_similarity(query_signal, transfer_signal),), 0.7),
                    ((max(self._last_context_reset_target_alignment_gain, 0.0),), 0.3),
                ),
            )[0]
            if any(transfer_signal)
            else 0.0
        )
        alignment = _blend_signals(
            dim=1,
            weighted_signals=(
                ((query_only_alignment,), 0.34),
                ((composite_alignment,), 0.46),
                ((transfer_alignment,), 0.20 if transfer_alignment > 0.0 else 0.0),
            ),
        )[0]
        confidence = _blend_signals(
            dim=1,
            weighted_signals=(
                ((_cosine_similarity(query_signal, core_signal),), 0.45),
                ((alignment,), 0.35),
                ((composite_alignment,), 0.20),
            ),
        )[0]
        depth_bonus = max(float(cms_state.tower_depth) - 3.0, 0.0) * 0.2
        consolidation_bonus = min(self._tower_consolidation_count / 4.0, 1.0) * 0.4
        learned_weight = 3.6 + max(confidence, 0.0) * 1.8 + depth_bonus + consolidation_bonus
        artifact_weight = 1.15 + max(0.0, 1.0 - alignment) * 0.75
        return LearnedMemoryRecall(
            query_base_signal=projected_query,
            query_signal=query_signal,
            core_signal=core_signal,
            tower_profile_id=tower_profile.profile_id if tower_profile is not None else "cms-flat",
            tower_depth=cms_state.tower_depth,
            retrieval_confidence=confidence,
            tower_alignment=alignment,
            query_only_alignment=query_only_alignment,
            composite_alignment=composite_alignment,
            transfer_alignment=transfer_alignment,
            artifact_weight=artifact_weight,
            learned_weight=learned_weight,
            description=(
                f"Learned recall blends owner query with memory tower profile="
                f"{tower_profile.profile_id if tower_profile is not None else 'cms-flat'} "
                f"depth={cms_state.tower_depth} confidence={confidence:.2f} "
                f"alignment={alignment:.2f} query_only={query_only_alignment:.2f} "
                f"composite={composite_alignment:.2f} transfer={transfer_alignment:.2f}; "
                f"learned_weight={learned_weight:.2f} "
                f"artifact_weight={artifact_weight:.2f}."
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
        user_text: str | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._store = store or MemoryStore()
        self._memory_feedback_signal = memory_feedback_signal
        self._user_text = user_text

    @property
    def store(self) -> MemoryStore:
        return self._store

    async def process(self, upstream: Mapping[str, Snapshot[object]]) -> Snapshot[MemorySnapshot]:
        from volvence_zero.prediction.error import PredictionErrorSnapshot

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
            user_text=self._user_text,
            track=Track.SHARED,
        ):
            self._store.write(request, timestamp_ms=substrate_snapshot.timestamp_ms)

        retrieval = self._store.retrieve(
            build_retrieval_query(
                substrate_snapshot=substrate_value,
                user_text=self._user_text,
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
        from volvence_zero.prediction.error import PredictionErrorSnapshot

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
            if cms_state.tower_profile is not None:
                facets.extend(
                    (
                        f"cms:tower:{cms_state.tower_profile.profile_id}",
                        f"cms:tower-depth:{cms_state.tower_depth}",
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
