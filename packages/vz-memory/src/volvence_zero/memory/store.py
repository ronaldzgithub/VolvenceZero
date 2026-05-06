"""MemoryStore and MemoryModule runtime implementations.

The frozen contracts, artifact store, and derived retrieval helpers live in
sibling modules. This module keeps the owner runtime and re-exports the
historic public API from ``volvence_zero.memory.store``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable, Mapping
from uuid import uuid4

from volvence_zero.learned_update import LearnedUpdateRuleState
from volvence_zero.memory.artifacts import ArtifactStore
from volvence_zero.memory.cms import (
    CMSCheckpointState,
    CMSMemoryCore,
    CMSState,
    CMSTowerConsolidationUpdate,
    CMSTowerProfile,
)
from volvence_zero.memory.contracts import (
    MemoryAttributeReadout,
    MemoryEntry,
    MemorySnapshot,
    MemoryStoreCheckpoint,
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    RetrievalResult,
    Track,
    _reconstruct_checkpoint,
)
from volvence_zero.memory.persistence import (
    PersistenceBackend,
    deserialize_checkpoint,
    serialize_checkpoint,
)
from volvence_zero.memory.retrieval import (
    DerivedRetrievalIndex,
    LearnedMemoryRecall,
    _align_signal,
    _blend_signals,
    _clamp_strength,
    _cosine_similarity,
    _entry_in_subject_scope,
    _mean_abs,
    _semantic_embedding,
    _substrate_embedding,
    _tokenize,
    summarize_entries,
)
from volvence_zero.memory.runtime_evidence import (
    build_runtime_backbone_evidence,
    cosine_alignment,
)
from volvence_zero.runtime import RuntimeModule, RuntimePlaceholderValue, Snapshot, WiringLevel
from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
    MemorySocialPESignal,
    MultiPartyIdentitySnapshot,
    build_memory_visibility_signals,
)
from volvence_zero.substrate import FeatureSignal, SubstrateSnapshot, SurfaceKind

if TYPE_CHECKING:
    from volvence_zero.prediction.error import PredictionErrorSnapshot


def build_default_memory_store(
    *,
    latent_dim: int = 8,
    nested_profile: bool = True,
    cms_pe_features_enabled: bool = True,
    cms_replay_window_size: int | None = 8,
) -> "MemoryStore":
    """Build a default :class:`MemoryStore` with ATLAS / Titans CMS uplift.

    The uplift is ACTIVE by default after the SHADOW validation ladder in
    ``docs/specs/cms-atlas-titans-uplift-shadow-evidence-2026-05-06.md``
    passed steps 1-8. Rollback is still explicit and local: pass
    ``cms_pe_features_enabled=False`` and ``cms_replay_window_size=None``
    to recover the pre-uplift CMS behavior.

    - ``cms_pe_features_enabled`` activates Titans-style PE-driven write
      gating in the CMS ``LearnedUpdateRule``.
    - ``cms_replay_window_size`` activates ATLAS-style joint optimization
      over the recent K observations on the online band, with K/2 and K/4
      on session and background bands respectively (each clamped to >= 1).

    See ``docs/specs/cms-atlas-titans-uplift.md`` §7.
    """
    variant = "nested" if nested_profile else "sequential"
    replay_window_sizes: dict[str, int] | None = None
    if cms_replay_window_size is not None:
        online_k = max(1, int(cms_replay_window_size))
        replay_window_sizes = {
            "online-fast": online_k,
            "session-medium": max(2, online_k // 2),
            "background-slow": max(2, online_k // 4),
        }
    learned_core = CMSMemoryCore(
        mode="mlp",
        d_in=max(latent_dim, 4),
        d_hidden=max(latent_dim * 2, 8),
        variant=variant,
        session_cadence=2,
        background_cadence=4,
        pe_features_enabled=cms_pe_features_enabled,
        replay_window_sizes=replay_window_sizes,
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
        # Phase 1.C: substrate feature_surface cache + per-entry
        # PE/substrate-derived attribute index. The cache is updated in
        # ``observe_substrate`` so writes / queries hit the same surface
        # within a turn. The attribute index is owner-internal and
        # published as a summary block in ``MemorySnapshot``.
        self._current_substrate_feature_surface: tuple[FeatureSignal, ...] = ()
        self._current_pe_intensity: float = 0.0
        self._current_pe_primary_axis: str = ""
        self._current_pe_regime_id: str = ""
        self._current_pe_epistemic: float = 0.0
        self._current_pe_aleatoric: float = 0.0
        self._entry_attributes: dict[str, MemoryAttributeReadout] = {}
        self._attribute_summary_capacity: int = 16

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
            subject_ids=request.subject_ids,
            audience_ids=request.audience_ids,
        )
        self._artifact_store.write(entry)
        self._derived_index.index_entry(entry, embedding=self._entry_signal(entry))
        self._observe_artifact_entry(entry=entry, timestamp_ms=timestamp_ms, source="write")
        # Phase 1.C: pin PE/substrate-derived attribute readout to the
        # entry. Caches are populated upstream by ``observe_substrate``
        # and ``apply_prediction_error_signal``; if nothing populated
        # them yet, the readout records zero PE intensity (still stable).
        substrate_digest_dim = max(min(self._learned_signal_dim(), 8), 1)
        substrate_digest = _substrate_embedding(
            feature_surface=self._current_substrate_feature_surface,
            dim=substrate_digest_dim,
        )
        self._entry_attributes[entry.entry_id] = MemoryAttributeReadout(
            entry_id=entry.entry_id,
            pe_intensity=self._current_pe_intensity,
            pe_primary_axis=self._current_pe_primary_axis,
            regime_id=self._current_pe_regime_id,
            substrate_feature_digest=substrate_digest,
            epistemic_magnitude=self._current_pe_epistemic,
            aleatoric_magnitude=self._current_pe_aleatoric,
            timestamp_ms=timestamp_ms,
        )
        return entry

    def retrieve(
        self,
        query: RetrievalQuery,
        *,
        timestamp_ms: int,
        active_subject_ids: tuple[str, ...] | None = None,
    ) -> RetrievalResult:
        tokens = _tokenize(query.text)
        query_embedding = self._query_base_signal(query)
        learned_recall = self._build_learned_recall(query=query, query_embedding=query_embedding)
        strata = query.strata or tuple(MemoryStratum)
        matches: list[tuple[float, MemoryEntry]] = []
        suppressed: list[tuple[float, MemoryEntry]] = []
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
            if active_subject_ids is not None and not _entry_in_subject_scope(
                entry, active_subject_ids
            ):
                suppressed.append((score, entry))
                continue
            updated = self._artifact_store.touch(entry.entry_id, timestamp_ms=timestamp_ms)
            if updated is not None:
                matches.append((score, updated))
        matches.sort(key=lambda item: (-item[0], -item[1].strength, -item[1].created_at_ms))
        suppressed.sort(key=lambda item: (-item[0], -item[1].strength, -item[1].created_at_ms))
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
            suppressed_cross_scope_entries=tuple(
                entry for _, entry in suppressed[: query.limit]
            ),
            active_subject_scope=tuple(active_subject_ids) if active_subject_ids is not None else (),
        )

    def snapshot(
        self,
        *,
        retrieved_entries: tuple[MemoryEntry, ...],
        suppressed_cross_scope_entries: tuple[MemoryEntry, ...] = (),
        active_subject_scope: tuple[str, ...] = (),
        social_pe_signals: tuple[MemorySocialPESignal, ...] = (),
    ) -> MemorySnapshot:
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
            cms_band_vectors=(
                (
                    ("online_fast", cms_state.online_fast.vector),
                    ("session_medium", cms_state.session_medium.vector),
                    ("background_slow", cms_state.background_slow.vector),
                )
                if cms_state is not None
                else ()
            ),
            description=description,
            suppressed_cross_scope_entries=suppressed_cross_scope_entries,
            active_subject_scope=active_subject_scope,
            social_pe_signals=social_pe_signals,
            attribute_summary=self._attribute_summary(),
        )

    def _attribute_summary(self) -> tuple[MemoryAttributeReadout, ...]:
        # Phase 1.C: published view of the most recent owner-internal
        # attribute readouts (capped to ``_attribute_summary_capacity``).
        # Sorted by timestamp_ms descending so the latest writes land
        # at the top.
        if not self._entry_attributes:
            return ()
        ordered = sorted(
            self._entry_attributes.values(),
            key=lambda readout: (readout.timestamp_ms, readout.entry_id),
            reverse=True,
        )
        return tuple(ordered[: self._attribute_summary_capacity])

    def observe_substrate(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot | None,
        timestamp_ms: int,
        prediction_error: "PredictionErrorSnapshot | None" = None,
    ) -> None:
        # Phase 1.C: cache feature_surface so retrieval embedding / write
        # attribute readout pull from the same per-turn substrate surface.
        if substrate_snapshot is not None:
            self._current_substrate_feature_surface = substrate_snapshot.feature_surface
        else:
            self._current_substrate_feature_surface = ()
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
                prediction_error=prediction_error,
            )

    def observe_encoder_feedback(
        self,
        *,
        encoder_signal: tuple[float, ...],
        timestamp_ms: int,
        prediction_error: "PredictionErrorSnapshot | None" = None,
    ) -> None:
        if self._learned_core is not None:
            self._learned_core.observe_encoder_feedback(
                encoder_signal=encoder_signal,
                timestamp_ms=timestamp_ms,
                prediction_error=prediction_error,
            )

    def observe_temporal_feedback(
        self,
        *,
        encoder_signal: tuple[float, ...],
        timestamp_ms: int,
        prediction_error: "PredictionErrorSnapshot | None" = None,
    ) -> None:
        self.observe_encoder_feedback(
            encoder_signal=encoder_signal,
            timestamp_ms=timestamp_ms,
            prediction_error=prediction_error,
        )

    def observe_fast_memory_signal(
        self,
        *,
        signal: tuple[float, ...],
        timestamp_ms: int,
        prediction_error: "PredictionErrorSnapshot | None" = None,
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
                prediction_error=prediction_error,
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
        # Phase 1.C: cache PE-derived attribute fields so the next
        # ``write`` (PE-driven or domain-driven) can pin them onto the
        # entry attribute index. Decomposition is optional (Phase 1.B).
        self._current_pe_intensity = float(magnitude)
        self._current_pe_primary_axis = primary_dimension
        self._current_pe_regime_id = (
            prediction_error_snapshot.action_context.regime_id
        )
        decomposition = prediction_error_snapshot.pe_decomposition
        if decomposition is not None:
            self._current_pe_epistemic = float(decomposition.epistemic_magnitude)
            self._current_pe_aleatoric = float(decomposition.aleatoric_magnitude)
        else:
            self._current_pe_epistemic = 0.0
            self._current_pe_aleatoric = 0.0
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
        # Phase 1.C: prefer substrate feature_surface as the dense
        # embedding source. When no substrate has been observed yet the
        # vector is zero and we fall back to the legacy hash embedding,
        # so existing tests / bootstrap paths remain identical.
        substrate_signal = _substrate_embedding(
            feature_surface=self._current_substrate_feature_surface,
            dim=dim,
        )
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
        substrate_norm = math.sqrt(sum(value * value for value in substrate_signal))
        substrate_weight = 0.55 if substrate_norm > 1e-9 else 0.0
        # When substrate is available we let it contribute the dominant
        # share of the dense signal; semantic + metadata still play a
        # stabilising role so write/query keys stay deterministic for
        # bootstrap turns and unit tests that lack substrate. When
        # substrate is absent the legacy semantic+metadata blend is
        # preserved exactly (substrate_weight = 0).
        return _blend_signals(
            dim=dim,
            weighted_signals=(
                (substrate_signal, substrate_weight),
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
    subject_ids: tuple[str, ...] = (PRIMARY_INTERLOCUTOR_ID,),
    audience_ids: tuple[str, ...] = (SELF_INTERLOCUTOR_ID,),
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
                subject_ids=subject_ids,
                audience_ids=audience_ids,
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
                subject_ids=subject_ids,
                audience_ids=audience_ids,
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
                    subject_ids=subject_ids,
                    audience_ids=audience_ids,
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


def _memory_scope_from_identity_snapshot(
    snapshot: Snapshot[object] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if snapshot is None or not isinstance(snapshot.value, MultiPartyIdentitySnapshot):
        return (PRIMARY_INTERLOCUTOR_ID,), (SELF_INTERLOCUTOR_ID,)
    return snapshot.value.subject_ids, snapshot.value.audience_ids


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
    dependencies = (
        "substrate",
        "multi_party_identity",
        "temporal_abstraction",
        "dual_track",
        "prediction_error",
    )
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
        multi_party_identity_snapshot = upstream.get("multi_party_identity")
        substrate_value = substrate_snapshot.value if isinstance(substrate_snapshot.value, SubstrateSnapshot) else None
        subject_ids, audience_ids = _memory_scope_from_identity_snapshot(
            multi_party_identity_snapshot
        )
        prediction_error_value = (
            prediction_error_snapshot.value
            if prediction_error_snapshot is not None and isinstance(prediction_error_snapshot.value, PredictionErrorSnapshot)
            else None
        )
        self._store.observe_substrate(
            substrate_snapshot=substrate_value,
            timestamp_ms=substrate_snapshot.timestamp_ms,
            prediction_error=prediction_error_value,
        )
        temporal_feedback_signal = _temporal_feedback_signal(temporal_snapshot.value if temporal_snapshot is not None else None)
        if temporal_feedback_signal:
            self._store.observe_temporal_feedback(
                encoder_signal=temporal_feedback_signal,
                timestamp_ms=substrate_snapshot.timestamp_ms,
                prediction_error=prediction_error_value,
            )
        elif self._memory_feedback_signal:
            self._store.observe_temporal_feedback(
                encoder_signal=self._memory_feedback_signal,
                timestamp_ms=substrate_snapshot.timestamp_ms,
                prediction_error=prediction_error_value,
            )
        self._store.apply_prediction_error_signal(
            prediction_error_snapshot=prediction_error_value,
            timestamp_ms=substrate_snapshot.timestamp_ms,
        )
        for request in build_memory_write_requests(
            substrate_snapshot=substrate_value,
            user_text=self._user_text,
            track=Track.SHARED,
            subject_ids=subject_ids,
            audience_ids=audience_ids,
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
            active_subject_ids=subject_ids,
        )
        social_pe_signals = self._build_social_pe_signals(retrieval=retrieval)
        return self.publish(
            self._store.snapshot(
                retrieved_entries=retrieval.entries,
                suppressed_cross_scope_entries=retrieval.suppressed_cross_scope_entries,
                active_subject_scope=retrieval.active_subject_scope,
                social_pe_signals=social_pe_signals,
            )
        )

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
        subject_ids = kwargs.get("subject_ids", (PRIMARY_INTERLOCUTOR_ID,))
        audience_ids = kwargs.get("audience_ids", (SELF_INTERLOCUTOR_ID,))
        if not isinstance(subject_ids, tuple):
            raise TypeError("subject_ids must be a tuple when provided.")
        if not isinstance(audience_ids, tuple):
            raise TypeError("audience_ids must be a tuple when provided.")

        substrate_snapshot = kwargs.get("substrate_snapshot")
        substrate_value = substrate_snapshot if isinstance(substrate_snapshot, SubstrateSnapshot) else None
        prediction_error_value = (
            prediction_error_snapshot
            if isinstance(prediction_error_snapshot, PredictionErrorSnapshot)
            else None
        )
        self._store.observe_substrate(
            substrate_snapshot=substrate_value,
            timestamp_ms=timestamp_ms,
            prediction_error=prediction_error_value,
        )
        temporal_feedback_signal = _temporal_feedback_signal(temporal_snapshot)
        if temporal_feedback_signal:
            self._store.observe_temporal_feedback(
                encoder_signal=temporal_feedback_signal,
                timestamp_ms=timestamp_ms,
                prediction_error=prediction_error_value,
            )
        elif self._memory_feedback_signal:
            self._store.observe_temporal_feedback(
                encoder_signal=self._memory_feedback_signal,
                timestamp_ms=timestamp_ms,
                prediction_error=prediction_error_value,
            )
        self._store.apply_prediction_error_signal(
            prediction_error_snapshot=prediction_error_value,
            timestamp_ms=timestamp_ms,
        )

        for request in build_memory_write_requests(
            substrate_snapshot=substrate_value,
            user_text=user_text,
            track=Track.SHARED,
            subject_ids=subject_ids,
            audience_ids=audience_ids,
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
            active_subject_ids=subject_ids,
        )
        social_pe_signals = self._build_social_pe_signals(retrieval=retrieval)
        return self.publish(
            self._store.snapshot(
                retrieved_entries=retrieval.entries,
                suppressed_cross_scope_entries=retrieval.suppressed_cross_scope_entries,
                active_subject_scope=retrieval.active_subject_scope,
                social_pe_signals=social_pe_signals,
            )
        )

    def _build_social_pe_signals(
        self, *, retrieval: RetrievalResult
    ) -> tuple[MemorySocialPESignal, ...]:
        active_scope = retrieval.active_subject_scope
        if not active_scope or active_scope == (PRIMARY_INTERLOCUTOR_ID,):
            return ()
        suppressed_evidence = tuple(
            sorted(
                {
                    f"suppressed:{entry.entry_id}:subject={'+'.join(entry.subject_ids)}"
                    for entry in retrieval.suppressed_cross_scope_entries
                }
            )
        )
        seq = self._version + 1
        signals = build_memory_visibility_signals(
            source_owner=self.owner,
            sequence_index=seq,
            active_subject_scope=active_scope,
            retrieved_count=len(retrieval.entries),
            suppressed_evidence=suppressed_evidence,
        )
        return signals

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
        facets.extend(_published_memory_retrieval_facets(temporal_value))
        facets.extend(_published_memory_retrieval_facets(dual_track_value))
        facets.extend(_published_memory_retrieval_facets(prediction_error_value))
        return tuple(facets)


def _published_memory_retrieval_facets(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, RuntimePlaceholderValue):
        return ()
    return tuple(value.memory_retrieval_facets)


def _temporal_feedback_signal(temporal_value: Any) -> tuple[float, ...]:
    if temporal_value is None:
        return ()
    if isinstance(temporal_value, RuntimePlaceholderValue):
        return ()
    return tuple(temporal_value.memory_feedback_signal)


__all__ = [
    "ArtifactStore",
    "DerivedRetrievalIndex",
    "LearnedMemoryRecall",
    "MemoryEntry",
    "MemoryModule",
    "MemorySnapshot",
    "MemoryStore",
    "MemoryStoreCheckpoint",
    "MemoryStratum",
    "MemoryWriteRequest",
    "RetrievalQuery",
    "RetrievalResult",
    "Track",
    "build_default_memory_store",
    "build_memory_write_requests",
    "build_retrieval_query",
    "summarize_entries",
]
