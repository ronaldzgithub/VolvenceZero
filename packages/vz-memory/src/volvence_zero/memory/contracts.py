"""Memory public contract types and checkpoint reconstruction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

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
    # #89 residual: learned PE write-gate threshold. Default keeps
    # pre-gate checkpoints loadable (they restore the initial 0.15).
    pe_write_gate_threshold: float = 0.15
    entry_attributes: tuple[MemoryAttributeReadout, ...] = ()


MEMORY_CHECKPOINT_RECEIPT_SCHEMA = "volvence.memory.checkpoint-persistence-receipt"
MEMORY_CHECKPOINT_RECEIPT_SCHEMA_VERSION = 1
MemoryCheckpointDurability = Literal[
    "restart_durable",
    "process_local",
    "unknown",
]


@dataclass(frozen=True)
class MemoryCheckpointPersistenceReceipt:
    """Owner-authored proof for one checkpoint persistence operation.

    ``payload_sha256`` is always calculated from the bytes read back from the
    configured persistence backend, never from an in-memory approximation of
    the checkpoint.  A load receipt additionally fingerprints the checkpoint
    exported by the freshly restored owner and states whether it is byte-for-
    byte equivalent to the persisted payload.

    ``durability`` describes the backend boundary honestly.  In particular,
    the in-memory backend publishes ``process_local`` and therefore cannot be
    mistaken for restart durability.
    """

    schema_id: str
    schema_version: int
    operation: str
    checkpoint_id: str
    checkpoint_key: str
    checkpoint_version: int
    payload_sha256: str
    payload_bytes: int
    entry_count: int
    durability: MemoryCheckpointDurability
    completed_at_ms: int
    restored_payload_sha256: str | None = None
    restored_matches_persisted: bool | None = None

    def __post_init__(self) -> None:
        if self.schema_id != MEMORY_CHECKPOINT_RECEIPT_SCHEMA:
            raise ValueError("memory checkpoint receipt schema_id is incompatible")
        if self.schema_version != MEMORY_CHECKPOINT_RECEIPT_SCHEMA_VERSION:
            raise ValueError("memory checkpoint receipt schema_version is incompatible")
        if self.operation not in {"save", "load"}:
            raise ValueError("memory checkpoint receipt operation must be 'save' or 'load'")
        if not self.checkpoint_id.strip():
            raise ValueError("memory checkpoint receipt checkpoint_id must be non-empty")
        if not self.checkpoint_key.strip():
            raise ValueError("memory checkpoint receipt checkpoint_key must be non-empty")
        if self.checkpoint_version < 0:
            raise ValueError("memory checkpoint receipt checkpoint_version cannot be negative")
        if len(self.payload_sha256) != 64:
            raise ValueError("memory checkpoint receipt payload_sha256 must be SHA-256 hex")
        try:
            int(self.payload_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "memory checkpoint receipt payload_sha256 must be SHA-256 hex"
            ) from exc
        if self.payload_bytes < 1:
            raise ValueError("memory checkpoint receipt payload_bytes must be positive")
        if self.entry_count < 0:
            raise ValueError("memory checkpoint receipt entry_count cannot be negative")
        if self.durability not in {"restart_durable", "process_local", "unknown"}:
            raise ValueError(
                "memory checkpoint receipt durability must be restart_durable, "
                "process_local, or unknown"
            )
        if self.completed_at_ms < 1:
            raise ValueError("memory checkpoint receipt completed_at_ms must be positive")
        if self.operation == "save":
            if self.restored_payload_sha256 is not None:
                raise ValueError("save receipt cannot carry restored_payload_sha256")
            if self.restored_matches_persisted is not None:
                raise ValueError("save receipt cannot carry restored_matches_persisted")
            return
        if self.restored_payload_sha256 is None:
            raise ValueError("load receipt requires restored_payload_sha256")
        if len(self.restored_payload_sha256) != 64:
            raise ValueError(
                "memory checkpoint receipt restored_payload_sha256 must be SHA-256 hex"
            )
        try:
            int(self.restored_payload_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "memory checkpoint receipt restored_payload_sha256 must be SHA-256 hex"
            ) from exc
        if not isinstance(self.restored_matches_persisted, bool):
            raise ValueError("load receipt requires bool restored_matches_persisted")

    @property
    def is_restart_durable(self) -> bool:
        """Whether this complete owner-validated receipt crosses restart."""

        return self.durability == "restart_durable"

    def to_json(self) -> dict[str, object]:
        """Serialize the receipt at its owner boundary.

        Consumers must not reconstruct or reinterpret Memory internals.  The
        stable public payload is authored here and may be embedded verbatim in
        a DLaaS scene-end response/ledger.
        """

        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "operation": self.operation,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_key": self.checkpoint_key,
            "checkpoint_version": self.checkpoint_version,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
            "entry_count": self.entry_count,
            "durability": self.durability,
            "completed_at_ms": self.completed_at_ms,
            "restored_payload_sha256": self.restored_payload_sha256,
            "restored_matches_persisted": self.restored_matches_persisted,
        }


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
        entry_attributes = tuple(
            MemoryAttributeReadout(
                entry_id=str(item["entry_id"]),
                pe_intensity=float(item["pe_intensity"]),
                pe_primary_axis=str(item["pe_primary_axis"]),
                regime_id=str(item["regime_id"]),
                substrate_feature_digest=tuple(
                    float(value)
                    for value in item.get("substrate_feature_digest", ())
                ),
                epistemic_magnitude=float(item["epistemic_magnitude"]),
                aleatoric_magnitude=float(item["aleatoric_magnitude"]),
                timestamp_ms=int(item["timestamp_ms"]),
            )
            for item in parsed.get("entry_attributes", ())
        )
        return MemoryStoreCheckpoint(
            checkpoint_id=str(parsed.get("checkpoint_id", "restored")),
            entries=entries,
            pending_promotions=tuple(str(p) for p in parsed.get("pending_promotions", ())),
            pending_decays=tuple(str(d) for d in parsed.get("pending_decays", ())),
            cms_state=cms_state,
            promotion_threshold=float(parsed.get("promotion_threshold", 0.3)),
            semantic_index=semantic_index,
            pe_write_gate_threshold=float(parsed.get("pe_write_gate_threshold", 0.15)),
            entry_attributes=entry_attributes,
        )
    except (KeyError, TypeError, ValueError):
        return None


def reconstruct_checkpoint(parsed: dict[str, Any]) -> MemoryStoreCheckpoint | None:
    """Reconstruct a typed checkpoint from its serialized JSON mapping.

    ``serialize_checkpoint`` is intentionally a wire-format API and returns
    plain JSON data.  Consumers that own a template/artifact boundary still
    need to hand that data back to the ``MemoryStore`` restore API without
    reaching into the store's private artifact tables.  Keep the canonical
    parser in one place and expose this narrow typed reconstruction helper so
    those boundaries do not silently drop all entries after a JSON round trip.
    """

    return _reconstruct_checkpoint(parsed)
