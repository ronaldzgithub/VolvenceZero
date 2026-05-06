"""Memory public contract types and checkpoint reconstruction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from volvence_zero.learned_update import LearnedUpdateDecision, LearnedUpdateRuleState
from volvence_zero.memory.cms import (
    CMSCheckpointState,
    CMSHopeSelfModificationState,
    CMSState,
)
from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
    MemorySocialPESignal,
)


def _require_non_empty_unique_tuple(field_name: str, values: tuple[str, ...]) -> None:
    if not values:
        raise ValueError(f"{field_name} must contain at least one entry")
    for value in values:
        if not value.strip():
            raise ValueError(f"{field_name} entries must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} entries must be unique")


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
    subject_ids: tuple[str, ...] = (PRIMARY_INTERLOCUTOR_ID,)
    audience_ids: tuple[str, ...] = (SELF_INTERLOCUTOR_ID,)

    def __post_init__(self) -> None:
        _require_non_empty_unique_tuple("subject_ids", self.subject_ids)
        _require_non_empty_unique_tuple("audience_ids", self.audience_ids)


@dataclass(frozen=True)
class MemoryWriteRequest:
    content: str
    track: Track
    stratum: MemoryStratum
    tags: tuple[str, ...] = ()
    strength: float = 0.5
    subject_ids: tuple[str, ...] = (PRIMARY_INTERLOCUTOR_ID,)
    audience_ids: tuple[str, ...] = (SELF_INTERLOCUTOR_ID,)

    def __post_init__(self) -> None:
        _require_non_empty_unique_tuple("subject_ids", self.subject_ids)
        _require_non_empty_unique_tuple("audience_ids", self.audience_ids)


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
    suppressed_cross_scope_entries: tuple[MemoryEntry, ...] = ()
    active_subject_scope: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryAttributeReadout:
    """Phase 1.C: PE + substrate-derived attribute pinned to a memory entry.

    Owner-internal readout that captures *why* an entry was written and
    what runtime state the substrate was in at write time. Published
    via ``MemorySnapshot.attribute_summary`` (recent entries only) and
    intentionally **not** merged onto ``MemoryEntry`` itself: keeping
    ``MemoryEntry`` schema stable means checkpoint / persistence / many
    tests stay byte-for-byte compatible while we still have a place to
    publish PE-driven attributes for inspection and downstream learning.

    Replaces what an A-Mem-style external LLM curator would otherwise
    produce; here every field is sourced from already-published owner
    state (PE owner + substrate owner), so no second curator exists.
    """

    entry_id: str
    pe_intensity: float
    pe_primary_axis: str
    regime_id: str
    substrate_feature_digest: tuple[float, ...]
    epistemic_magnitude: float
    aleatoric_magnitude: float
    timestamp_ms: int


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
    cms_band_vectors: tuple[tuple[str, tuple[float, ...]], ...] = ()
    suppressed_cross_scope_entries: tuple[MemoryEntry, ...] = ()
    active_subject_scope: tuple[str, ...] = ()
    social_pe_signals: tuple[MemorySocialPESignal, ...] = ()
    # Phase 1.C: optional attribute readout summary (most recent N
    # entries), populated by MemoryStore. Default empty tuple keeps
    # legacy consumers unaffected.
    attribute_summary: tuple[MemoryAttributeReadout, ...] = ()


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
                subject_ids=tuple(
                    str(subject_id)
                    for subject_id in e.get("subject_ids", (PRIMARY_INTERLOCUTOR_ID,))
                ),
                audience_ids=tuple(
                    str(audience_id)
                    for audience_id in e.get("audience_ids", (SELF_INTERLOCUTOR_ID,))
                ),
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
                    feature_version=int(update_rule_raw.get("feature_version", 1)),
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
                atlas_replay_active=bool(cms_raw.get("atlas_replay_active", False)),
                titans_pe_gate_active=bool(cms_raw.get("titans_pe_gate_active", False)),
                replay_window_sizes=tuple(
                    (str(item[0]), int(item[1]))
                    for item in cms_raw.get("replay_window_sizes", ())
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
