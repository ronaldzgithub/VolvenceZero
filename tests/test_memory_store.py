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
        lesson_count=2,
        timestamp_ms=52,
    )
    assert store.learned_core is not None
    assert store.learned_core.snapshot().description

    store.restore_checkpoint(checkpoint)
    restored = store.retrieve(
        RetrievalQuery(text="checkpoint", track=Track.SHARED, strata=(MemoryStratum.TRANSIENT,), limit=1),
        timestamp_ms=53,
    )
    assert restored.entries

