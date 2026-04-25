from __future__ import annotations

import asyncio

from volvence_zero.memory import (
    CMSMemoryCore,
    build_default_memory_store,
    MemoryModule,
    MemoryStore,
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
)
from volvence_zero.prediction import ActualOutcome, PredictionError, PredictionErrorSnapshot, PredictedOutcome
from volvence_zero.runtime import WiringLevel, propagate
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
    SubstrateModule,
)
from volvence_zero.temporal import ControllerState, TemporalAbstractionSnapshot
from volvence_zero.dual_track import DualTrackSnapshot, TrackState


def test_memory_store_write_and_retrieve_by_track_and_query():
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content="user prefers calm reflective conversations",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            tags=("preference", "calm"),
            strength=0.8,
        ),
        timestamp_ms=10,
    )
    result = store.retrieve(
        query=RetrievalQuery(
            text="calm reflective preference",
            track=Track.SELF,
            strata=(MemoryStratum.DURABLE,),
            limit=3,
        ),
        timestamp_ms=20,
    )

    assert len(result.entries) == 1
    assert result.entries[0].track is Track.SELF
    assert result.entries[0].last_accessed_ms == 20


def test_memory_store_hybrid_semantic_retrieval_can_match_without_exact_overlap():
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content="gentle collaborative planning for difficult conversations",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            tags=("warmth", "planning"),
            strength=0.75,
        ),
        timestamp_ms=10,
    )

    result = store.retrieve(
        query=RetrievalQuery(
            text="supportive teamwork dialogue",
            track=Track.SELF,
            strata=(MemoryStratum.DURABLE,),
            limit=3,
            facets=("warmth", "planning"),
        ),
        timestamp_ms=20,
    )

    assert result.entries
    assert result.entries[0].track is Track.SELF


def test_memory_store_checkpoint_restore_preserves_semantic_retrieval_index():
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content="careful paced support during tense moments",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            tags=("repair", "support"),
            strength=0.8,
        ),
        timestamp_ms=10,
    )
    checkpoint = store.create_checkpoint(checkpoint_id="semantic-checkpoint")
    store.restore_checkpoint(checkpoint)

    restored = store.retrieve(
        RetrievalQuery(
            text="steady reassurance",
            track=Track.SELF,
            strata=(MemoryStratum.DURABLE,),
            limit=2,
            facets=("repair", "support"),
        ),
        timestamp_ms=20,
    )

    assert restored.entries
    assert restored.entries[0].content == "careful paced support during tense moments"


def test_memory_module_uses_temporal_and_dual_track_facets_in_runtime_query():
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content="repair_controller maintain a calm supportive tone while planning next steps",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.85,
            tags=("repair", "support"),
        ),
        timestamp_ms=5,
    )
    module = MemoryModule(store=store, wiring_level=WiringLevel.ACTIVE)
    temporal_snapshot = TemporalAbstractionSnapshot(
        controller_state=ControllerState(
            code=(0.2, 0.8, 0.4),
            code_dim=3,
            switch_gate=0.7,
            is_switching=True,
            steps_since_switch=1,
        ),
        active_abstract_action="repair_controller",
        controller_params_hash="hash",
        description="repair context",
    )
    dual_track_snapshot = DualTrackSnapshot(
        world_track=TrackState(
            track=Track.WORLD,
            active_goals=("planning next steps",),
            recent_credits=(),
            controller_code=(0.6, 0.4),
            tension_level=0.4,
        ),
        self_track=TrackState(
            track=Track.SELF,
            active_goals=("maintain a calm supportive tone",),
            recent_credits=(),
            controller_code=(0.3, 0.7),
            tension_level=0.6,
        ),
        cross_track_tension=0.2,
        description="dual track context",
    )

    snapshot = asyncio.run(
        module.process_standalone(
            timestamp_ms=20,
            temporal_snapshot=temporal_snapshot,
            dual_track_snapshot=dual_track_snapshot,
        )
    )

    assert snapshot.value.retrieved_entries
    assert "repair_controller" in snapshot.value.retrieved_entries[0].content


def test_memory_module_process_standalone_publishes_memory_snapshot():
    module = MemoryModule(store=MemoryStore(), wiring_level=WiringLevel.ACTIVE)
    snapshot = asyncio.run(
        module.process_standalone(
            user_text="remember this conversation tone",
            timestamp_ms=100,
        )
    )

    assert snapshot.value.transient_summary
    assert snapshot.value.pending_promotions >= 1
    assert snapshot.value.description.startswith("Memory store with")


def test_memory_module_consumes_substrate_and_publishes_shadow_snapshot():
    substrate = SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="test-model",
            feature_surface=(
                FeatureSignal(name="trust_signal", values=(0.8,), source="adapter"),
            ),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )
    memory = MemoryModule(store=MemoryStore())
    shadow_snapshots: dict[str, object] = {}

    result = asyncio.run(
        propagate(
            [substrate, memory],
            session_id="s1",
            wave_id="w1",
            shadow_snapshots=shadow_snapshots,
        )
    )

    assert "substrate" in result
    assert "memory" not in result
    memory_snapshot = shadow_snapshots["memory"]
    assert memory_snapshot.value.total_entries_by_stratum
    assert memory_snapshot.value.retrieved_entries


def test_memory_module_process_can_ingest_runtime_user_text_into_owner_memory():
    substrate = SubstrateModule(
        adapter=FeatureSurfaceSubstrateAdapter(
            model_id="runtime-user-text-model",
            feature_surface=(
                FeatureSignal(name="trust_signal", values=(0.8,), source="adapter"),
            ),
        ),
        wiring_level=WiringLevel.ACTIVE,
    )
    memory = MemoryModule(
        store=MemoryStore(),
        user_text="remember this exact runtime phrasing",
    )
    shadow_snapshots: dict[str, object] = {}

    asyncio.run(
        propagate(
            [substrate, memory],
            session_id="s-runtime-user-text",
            wave_id="w-runtime-user-text",
            shadow_snapshots=shadow_snapshots,
        )
    )

    memory_snapshot = shadow_snapshots["memory"]
    contents = tuple(entry.content for entry in memory.store._entries.values())
    assert any(content == "remember this exact runtime phrasing" for content in contents)
    assert any(
        entry.content == "remember this exact runtime phrasing"
        for entry in memory_snapshot.value.retrieved_entries
    )


def test_memory_store_supports_checkpoint_restore_and_cms_core():
    store = MemoryStore(learned_core=CMSMemoryCore())
    substrate_snapshot = asyncio.run(
        SubstrateModule(
            adapter=FeatureSurfaceSubstrateAdapter(
                model_id="cms-model",
                feature_surface=(FeatureSignal(name="cms_signal", values=(0.7,), source="adapter"),),
            ),
            wiring_level=WiringLevel.ACTIVE,
        ).process_standalone()
    ).value
    store.observe_substrate(substrate_snapshot=substrate_snapshot, timestamp_ms=50)
    entry = store.write(
        MemoryWriteRequest(
            content="checkpoint this memory",
            track=Track.SHARED,
            stratum=MemoryStratum.TRANSIENT,
            strength=0.6,
        ),
        timestamp_ms=51,
    )
    checkpoint = store.create_checkpoint(checkpoint_id="memory-1")
    store.apply_reflection_consolidation(
        new_durable_entries=(),
        promoted_entries=(entry.entry_id,),
        decayed_entries=(),
        beliefs_updated=("reinforce:shared:checkpoint",),
        promotion_boost=0.6,
        decay_scale=0.2,
        lesson_count=2,
        timestamp_ms=52,
    )
    assert store.learned_core is not None
    assert store.learned_core.snapshot().description
    assert store.snapshot(retrieved_entries=()).cms_state is not None

    store.restore_checkpoint(checkpoint)
    restored = store.retrieve(
        RetrievalQuery(text="checkpoint", track=Track.SHARED, strata=(MemoryStratum.TRANSIENT,), limit=1),
        timestamp_ms=53,
    )
    assert restored.entries


def test_memory_store_observes_fast_memory_signal_and_publishes_lifecycle_metrics():
    store = MemoryStore(learned_core=CMSMemoryCore())

    store.observe_fast_memory_signal(
        signal=(0.4, 0.2, 0.3, 0.1, 0.2, 0.5),
        timestamp_ms=60,
    )
    snapshot = store.snapshot(retrieved_entries=())
    metrics = dict(snapshot.lifecycle_metrics)

    assert metrics["fast_memory_signal_count"] == 1.0
    assert metrics["last_fast_memory_signal_norm"] > 0.0


def test_memory_store_nested_tower_profile_is_machine_readable():
    store = build_default_memory_store(latent_dim=6, nested_profile=True)

    snapshot = store.snapshot(retrieved_entries=())

    assert snapshot.cms_state is not None
    assert snapshot.cms_state.tower_profile is not None
    assert snapshot.cms_state.tower_depth >= 5
    assert snapshot.cms_state.tower_profile.levels[-1].level_id == "tower-readout"
    assert snapshot.cms_state.tower_profile.readout_vector


def test_memory_store_tower_guided_retrieval_publishes_tower_metrics():
    store = build_default_memory_store(latent_dim=6, nested_profile=True)
    store.write(
        MemoryWriteRequest(
            content="steady reflective repair planning with warmth",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            tags=("repair", "planning", "warmth"),
            strength=0.9,
        ),
        timestamp_ms=10,
    )

    result = store.retrieve(
        RetrievalQuery(
            text="warm supportive repair planning",
            track=Track.SELF,
            strata=(MemoryStratum.DURABLE,),
            limit=2,
            facets=("repair", "planning"),
        ),
        timestamp_ms=20,
    )
    snapshot = store.snapshot(retrieved_entries=result.entries)
    metrics = dict(snapshot.lifecycle_metrics)

    assert result.entries
    assert metrics["last_memory_tower_depth"] >= 5.0
    assert metrics["last_memory_tower_alignment"] >= 0.0
    assert snapshot.cms_state is not None
    assert snapshot.cms_state.tower_profile is not None
    assert snapshot.cms_state.tower_profile.profile_id != "artifact-only"


def test_memory_store_composite_tower_alignment_improves_after_fast_signal_and_nested_reset():
    store = build_default_memory_store(latent_dim=6, nested_profile=True)
    store.write(
        MemoryWriteRequest(
            content="steady reflective repair planning with warmth",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            tags=("repair", "planning", "warmth"),
            strength=0.9,
        ),
        timestamp_ms=10,
    )
    query = RetrievalQuery(
        text="warm supportive repair planning",
        track=Track.SELF,
        strata=(MemoryStratum.DURABLE,),
        limit=2,
        facets=("repair", "planning"),
    )

    baseline = store.retrieve(query, timestamp_ms=20)
    baseline_alignment = dict(store.snapshot(retrieved_entries=baseline.entries).lifecycle_metrics)["last_memory_tower_alignment"]

    signal = store._query_base_signal(query)
    for timestamp in range(21, 25):
        store.observe_fast_memory_signal(signal=signal, timestamp_ms=timestamp)
    store.reset_nested_context(reason="alignment-test", timestamp_ms=25)

    improved = store.retrieve(query, timestamp_ms=26)
    improved_snapshot = store.snapshot(retrieved_entries=improved.entries)
    improved_metrics = dict(improved_snapshot.lifecycle_metrics)

    assert improved.entries
    assert improved_metrics["fast_memory_signal_count"] >= 4.0
    assert improved_metrics["last_nested_reset_applied"] == 1.0
    assert improved_metrics["last_memory_tower_alignment"] > baseline_alignment
    assert improved_snapshot.cms_state is not None
    assert improved_snapshot.cms_state.update_rule_state is not None
    reset_decisions = {
        decision.target_id: decision
        for decision in improved_snapshot.cms_state.update_rule_state.last_decisions
        if "reset" in decision.target_id
    }
    assert "nested-online-reset" in reset_decisions
    assert "nested-session-reset" in reset_decisions
    assert reset_decisions["nested-online-reset"].reset_mix > 0.0
    assert reset_decisions["nested-session-reset"].slow_mix > 0.0


def test_memory_store_publishes_tiny_hope_self_modification_state_and_rolls_back():
    store = MemoryStore(
        learned_core=CMSMemoryCore(
            mode="mlp",
            d_in=4,
            d_hidden=8,
            variant="nested",
            session_cadence=1,
            background_cadence=1,
        )
    )
    core = store.learned_core
    assert core is not None

    core.observe_fast_memory_signal(signal=(0.8, 0.2, 0.6, 0.4), timestamp_ms=10)
    before_checkpoint = store.create_checkpoint(checkpoint_id="hope-before")
    before_state = core.snapshot().hope_self_modification_state
    assert before_state is not None
    assert before_state.update_count > 0
    assert before_state.generated_learning_rate > 0.0
    assert before_state.generated_decay_rate > 0.0

    for timestamp in range(11, 15):
        core.observe_fast_memory_signal(signal=(0.2, 0.7, 0.4, 0.9), timestamp_ms=timestamp)
    changed_state = core.snapshot().hope_self_modification_state
    assert changed_state is not None
    assert changed_state.update_count > before_state.update_count
    assert changed_state.description.startswith("Tiny Hope owner-side self-modification state")

    metrics = dict(store.snapshot(retrieved_entries=()).lifecycle_metrics)
    assert metrics["hope_self_mod_update_count"] == float(changed_state.update_count)
    assert metrics["hope_generated_learning_rate"] == changed_state.generated_learning_rate

    store.restore_checkpoint(before_checkpoint)
    restored_state = core.snapshot().hope_self_modification_state
    assert restored_state is not None
    assert restored_state.update_count == before_state.update_count
    assert restored_state.last_target_id == before_state.last_target_id
    assert restored_state.generated_learning_rate == before_state.generated_learning_rate


def test_memory_store_reflection_consolidation_updates_tower_and_restores_checkpoint():
    store = MemoryStore(
        learned_core=CMSMemoryCore(
            mode="mlp",
            d_in=4,
            d_hidden=8,
            variant="nested",
            session_cadence=1,
            background_cadence=1,
        )
    )
    promoted = store.write(
        MemoryWriteRequest(
            content="repair this durable lesson",
            track=Track.SELF,
            stratum=MemoryStratum.EPISODIC,
            tags=("repair",),
            strength=0.9,
        ),
        timestamp_ms=5,
    )
    checkpoint = store.create_checkpoint(checkpoint_id="tower-before")
    before = store.learned_core.snapshot()

    operations = store.apply_reflection_consolidation(
        new_durable_entries=(),
        promoted_entries=(promoted.entry_id,),
        decayed_entries=(),
        beliefs_updated=("reinforce:self:repair",),
        promotion_boost=0.7,
        decay_scale=0.2,
        lesson_count=3,
        timestamp_ms=6,
    )
    after = store.learned_core.snapshot()

    assert any(operation.startswith("tower-consolidation:") for operation in operations)
    assert checkpoint.cms_state is not None
    assert checkpoint.cms_state.tower_meta_levels
    assert before.tower_profile is not None
    assert after.tower_profile is not None
    assert before.tower_profile.readout_vector != after.tower_profile.readout_vector
    metrics = dict(store.snapshot(retrieved_entries=()).lifecycle_metrics)
    assert metrics["tower_consolidation_count"] == 1.0

    store.restore_checkpoint(checkpoint)
    restored = store.learned_core.snapshot()
    assert restored.tower_profile is not None
    assert restored.tower_profile.readout_vector == before.tower_profile.readout_vector


def test_persistence_backend_round_trip():
    import tempfile
    from volvence_zero.memory import (
        FileSystemPersistenceBackend,
        serialize_checkpoint,
        deserialize_checkpoint,
        MemoryStore,
        MemoryStratum,
        MemoryWriteRequest,
        Track,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FileSystemPersistenceBackend(base_dir=tmpdir)
        store = MemoryStore(persistence_backend=backend)
        store.write(
            MemoryWriteRequest(
                content="persistent memory",
                track=Track.WORLD,
                stratum=MemoryStratum.DURABLE,
                strength=0.9,
            ),
            timestamp_ms=100,
        )
        saved = store.save_to_backend(key="memory/test")
        assert saved is True

        result = backend.load_checkpoint(key="memory/test")
        assert result is not None
        data, version = result
        assert version == 1
        parsed = deserialize_checkpoint(data)
        assert parsed
        assert parsed.get("_schema_version") == 1

        keys = backend.list_checkpoints(prefix="memory/")
        assert len(keys) >= 1


def test_persistence_backend_version_cleanup():
    import tempfile
    import os
    from volvence_zero.memory import FileSystemPersistenceBackend

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FileSystemPersistenceBackend(base_dir=tmpdir, max_versions=3)
        for v in range(7):
            backend.save_checkpoint(
                key="test/cleanup",
                data=f'{{"version": {v}}}'.encode("utf-8"),
                version=v,
            )
        files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
        assert len(files) <= 3

        result = backend.load_checkpoint(key="test/cleanup")
        assert result is not None
        _, latest_version = result
        assert latest_version == 6


# ---------------------------------------------------------------------------
# Phase 4 — Persistence full roundtrip (save → restart → load → verify)
# ---------------------------------------------------------------------------

def test_phase4_persistence_full_roundtrip_restores_store_state():
    """Save MemoryStore to filesystem backend, create a fresh store,
    load from backend, and verify entries are restored."""
    import tempfile
    from volvence_zero.memory import (
        FileSystemPersistenceBackend,
        MemoryStore,
        MemoryStratum,
        MemoryWriteRequest,
        Track,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FileSystemPersistenceBackend(base_dir=tmpdir)
        store1 = MemoryStore(persistence_backend=backend)
        store1.write(
            MemoryWriteRequest(
                content="persistent fact alpha",
                track=Track.WORLD,
                stratum=MemoryStratum.DURABLE,
                strength=0.9,
                tags=("fact", "alpha"),
            ),
            timestamp_ms=100,
        )
        store1.write(
            MemoryWriteRequest(
                content="relationship memory beta",
                track=Track.SELF,
                stratum=MemoryStratum.EPISODIC,
                strength=0.7,
                tags=("relationship",),
            ),
            timestamp_ms=200,
        )
        saved = store1.save_to_backend(key="memory/store")
        assert saved is True

        store2 = MemoryStore(persistence_backend=backend)
        loaded = store2.load_from_backend(key="memory/store")
        assert loaded is True, "load_from_backend should succeed"

        assert len(store2._entries) >= 2, (
            f"Expected ≥2 entries after restore, got {len(store2._entries)}"
        )

        restored_content = {e.content for e in store2._entries.values()}
        assert "persistent fact alpha" in restored_content, (
            f"Expected 'persistent fact alpha' in restored store, got: {restored_content}"
        )
        assert "relationship memory beta" in restored_content, (
            f"Expected 'relationship memory beta' in restored store, got: {restored_content}"
        )

        restored_tracks = {e.content: e.track for e in store2._entries.values()}
        assert restored_tracks["persistent fact alpha"] is Track.WORLD
        assert restored_tracks["relationship memory beta"] is Track.SELF


def test_phase4_persistence_with_cms_mlp_roundtrip():
    """Save CMS MLP state via persistence backend and verify restoration."""
    import tempfile
    from volvence_zero.memory import (
        CMSMemoryCore,
        FileSystemPersistenceBackend,
        MemoryStore,
        MemoryWriteRequest,
        MemoryStratum,
        Track,
    )
    from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind, FeatureSignal

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FileSystemPersistenceBackend(base_dir=tmpdir)
        cms = CMSMemoryCore(mode="mlp", d_in=4, d_hidden=8)
        store1 = MemoryStore(learned_core=cms, persistence_backend=backend)
        substrate = SubstrateSnapshot(
            model_id="persist-test",
            is_frozen=True,
            surface_kind=SurfaceKind.FEATURE_SURFACE,
            token_logits=(0.5,),
            feature_surface=(
                FeatureSignal(name="val", values=(0.7, 0.3), source="test", layer_hint=0),
            ),
            residual_activations=(),
            residual_sequence=(),
            unavailable_fields=(),
            description="persist substrate",
        )
        for i in range(5):
            cms.observe_substrate(substrate_snapshot=substrate, timestamp_ms=i)

        snap1 = cms.snapshot()
        saved = store1.save_to_backend(key="memory/cms-test")
        assert saved is True

        cms2 = CMSMemoryCore(mode="mlp", d_in=4, d_hidden=8)
        store2 = MemoryStore(learned_core=cms2, persistence_backend=backend)
        loaded = store2.load_from_backend(key="memory/cms-test")
        assert loaded is True

        snap2 = cms2.snapshot()
        assert snap1.online_fast.vector == snap2.online_fast.vector, (
            "CMS online band should match after persistence roundtrip"
        )
        assert snap2.update_rule_state is not None
        assert snap1.update_rule_state is not None
        assert snap2.update_rule_state.rule_id == snap1.update_rule_state.rule_id
        assert snap2.update_rule_state.update_count == snap1.update_rule_state.update_count
        assert tuple(decision.target_id for decision in snap2.update_rule_state.last_decisions) == tuple(
            decision.target_id for decision in snap1.update_rule_state.last_decisions
        )
        assert snap2.hope_self_modification_state is not None
        assert snap1.hope_self_modification_state is not None
        assert snap2.hope_self_modification_state.update_count == snap1.hope_self_modification_state.update_count
        assert (
            snap2.hope_self_modification_state.generated_learning_rate
            == snap1.hope_self_modification_state.generated_learning_rate
        )


def test_phase4_persistence_version_incompatibility_safe_degradation():
    """When checkpoint version is incompatible, load should fail gracefully."""
    import tempfile
    from volvence_zero.memory import (
        FileSystemPersistenceBackend,
        MemoryStore,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FileSystemPersistenceBackend(base_dir=tmpdir)
        backend.save_checkpoint(
            key="memory/bad",
            data=b'{"_schema_version": 999, "entries": []}',
            version=1,
        )

        store = MemoryStore(persistence_backend=backend)
        loaded = store.load_from_backend(key="memory/bad")
        assert loaded is False, "Incompatible version should cause safe degradation"


def test_phase4_persistence_missing_key_returns_false():
    """Load with non-existent key should return False."""
    import tempfile
    from volvence_zero.memory import (
        FileSystemPersistenceBackend,
        MemoryStore,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FileSystemPersistenceBackend(base_dir=tmpdir)
        store = MemoryStore(persistence_backend=backend)
        loaded = store.load_from_backend(key="does/not/exist")
        assert loaded is False


def test_memory_store_applies_prediction_error_signal():
    store = MemoryStore()
    pe_snapshot = PredictionErrorSnapshot(
        evaluated_prediction=PredictedOutcome(0, 1, 0.6, 0.6, 0.6, 0.6, 0.7, "pred"),
        actual_outcome=ActualOutcome(1, 0.2, 0.1, 0.4, 0.3, "actual"),
        next_prediction=PredictedOutcome(1, 2, 0.5, 0.5, 0.5, 0.5, 0.6, "next"),
        error=PredictionError(
            task_error=-0.4,
            relationship_error=-0.5,
            regime_error=-0.2,
            action_error=-0.3,
            magnitude=1.4,
            signed_reward=-0.35,
            description="prediction error",
        ),
        turn_index=1,
        bootstrap=False,
        description="pe snapshot",
    )
    before_threshold = store.promotion_threshold
    ops = store.apply_prediction_error_signal(
        prediction_error_snapshot=pe_snapshot,
        timestamp_ms=123,
    )
    assert ops
    assert any("prediction-error-write:" in op for op in ops)
    assert store.promotion_threshold != before_threshold
    assert any("prediction_error:" in entry.content for entry in store._entries.values())


def test_memory_module_consumes_prediction_error_snapshot():
    module = MemoryModule(store=MemoryStore(), wiring_level=WiringLevel.ACTIVE)
    pe_snapshot = PredictionErrorSnapshot(
        evaluated_prediction=PredictedOutcome(0, 1, 0.6, 0.6, 0.6, 0.6, 0.7, "pred"),
        actual_outcome=ActualOutcome(1, 0.2, 0.1, 0.4, 0.3, "actual"),
        next_prediction=PredictedOutcome(1, 2, 0.5, 0.5, 0.5, 0.5, 0.6, "next"),
        error=PredictionError(
            task_error=-0.4,
            relationship_error=-0.5,
            regime_error=-0.2,
            action_error=-0.3,
            magnitude=1.4,
            signed_reward=-0.35,
            description="prediction error",
        ),
        turn_index=1,
        bootstrap=False,
        description="pe snapshot",
    )
    snapshot = asyncio.run(
        module.process_standalone(
            timestamp_ms=50,
            user_text="remember this turn",
            prediction_error_snapshot=pe_snapshot,
        )
    )
    contents = tuple(entry.content for entry in snapshot.value.retrieved_entries)
    all_contents = tuple(entry.content for entry in module.store._entries.values())
    assert any("prediction_error:" in content for content in all_contents)

