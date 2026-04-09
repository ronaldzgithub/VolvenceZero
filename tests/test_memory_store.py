from __future__ import annotations

import asyncio

from volvence_zero.memory import (
    CMSMemoryCore,
    MemoryModule,
    MemoryStore,
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
)
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

