"""Tests for the schema-v11 cultivation store deltas.

Covers the self-learning workshop additions:

* ``source_template_id`` (adopted-seed provenance) round-trips through
  ``create`` / ``get``.
* The append-only ``cultivation_events`` monitoring log persists and
  reads back oldest-first with the typed payload merged in.
* Supervision status transitions (``paused`` / ``failed``) persist.
* Forward migration adds the new column + table to a pre-existing
  schema-v10 database without losing rows.
"""

from __future__ import annotations

import asyncio
import sqlite3

from dlaas_platform_registry.cultivation_store import (
    CultivationStatus,
    CultivationStore,
)
from dlaas_platform_registry.db import (
    Registry,
    SCHEMA_VERSION,
    init_schema,
    open_connection,
)


def _build_registry() -> Registry:
    return Registry(db_path=":memory:")


async def _create_basic(store: CultivationStore, **overrides):
    payload = dict(
        ai_id="cultivation:child-psych",
        slug="child-psych",
        display_name="儿童心理专家",
        domain="儿童心理",
        runtime_template_id="cultivation.expert.v0",
        seed_persona={"display_name": "儿童心理专家"},
        curriculum={"topics": ["依恋"]},
    )
    payload.update(overrides)
    return await store.create(**payload)


def test_source_template_id_round_trips() -> None:
    async def run() -> None:
        registry = _build_registry()
        store = CultivationStore(registry)
        record = await _create_basic(
            store, source_template_id="tpl_source_123"
        )
        assert record.source_template_id == "tpl_source_123"
        fetched = await store.get(record.cultivation_id)
        assert fetched.source_template_id == "tpl_source_123"
        assert fetched.to_json()["source_template_id"] == "tpl_source_123"
        registry.close()

    asyncio.run(run())


def test_source_template_id_defaults_empty_for_empty_seed() -> None:
    async def run() -> None:
        registry = _build_registry()
        store = CultivationStore(registry)
        record = await _create_basic(store)
        assert record.source_template_id == ""
        registry.close()

    asyncio.run(run())


def test_events_append_and_list_oldest_first() -> None:
    async def run() -> None:
        registry = _build_registry()
        store = CultivationStore(registry)
        record = await _create_basic(store)
        cid = record.cultivation_id
        written = await store.append_events(
            cultivation_id=cid,
            kind="cycle",
            events=(
                {
                    "cycle_index": 0,
                    "topic": "依恋",
                    "docs_researched": 2,
                    "active_regime": "pd",
                },
                {
                    "cycle_index": 1,
                    "topic": "认知",
                    "docs_researched": 1,
                    "active_regime": "pd",
                },
            ),
        )
        assert written == 2
        await store.append_events(
            cultivation_id=cid,
            kind="teach",
            events=(
                {
                    "cycle_index": 1,
                    "text": "请更关注依恋安全",
                    "protocol_uptaken": "uptake:attachment",
                },
            ),
        )
        events = await store.list_events(cid)
        assert len(events) == 3
        # Cycle events recorded first (oldest), then the teach correction.
        assert events[0]["kind"] == "cycle"
        assert events[0]["topic"] == "依恋"
        assert events[-1]["kind"] == "teach"
        assert events[-1]["protocol_uptaken"] == "uptake:attachment"
        # Each item carries store metadata merged with the payload.
        assert "event_id" in events[0]
        assert events[1]["seq"] == 1
        registry.close()

    asyncio.run(run())


def test_provenance_round_trips() -> None:
    async def run() -> None:
        registry = _build_registry()
        store = CultivationStore(registry)
        prov = {
            "source_template_id": "tpl_src",
            "source_kind": "character",
            "source_angle": "character",
            "continuation_mode": "protocol_bundle",
        }
        record = await _create_basic(
            store, source_template_id="tpl_src", provenance=prov
        )
        assert record.provenance == prov
        fetched = await store.get(record.cultivation_id)
        assert fetched.provenance["source_kind"] == "character"
        assert (
            fetched.to_json()["provenance"]["continuation_mode"]
            == "protocol_bundle"
        )
        registry.close()

    asyncio.run(run())


def test_provenance_defaults_empty() -> None:
    async def run() -> None:
        registry = _build_registry()
        store = CultivationStore(registry)
        record = await _create_basic(store)
        assert record.provenance == {}
        registry.close()

    asyncio.run(run())


def test_timeline_kind_filter_splits_progress_from_log() -> None:
    async def run() -> None:
        registry = _build_registry()
        store = CultivationStore(registry)
        record = await _create_basic(store)
        cid = record.cultivation_id
        await store.append_events(
            cultivation_id=cid,
            kind="cycle",
            events=({"cycle_index": 0, "topic": "依恋"},),
        )
        await store.append_events(
            cultivation_id=cid,
            kind="progress",
            events=({"cycle_index": 4, "coherence_score": 0.8},),
        )
        await store.append_events(
            cultivation_id=cid,
            kind="pause",
            events=({"cycle_index": 4},),
        )
        timeline = await store.list_events(cid, kinds=("progress",))
        assert len(timeline) == 1
        assert timeline[0]["kind"] == "progress"
        assert timeline[0]["coherence_score"] == 0.8
        # The full log still returns every kind.
        every = await store.list_events(cid)
        kinds = {e["kind"] for e in every}
        assert kinds == {"cycle", "progress", "pause"}
        registry.close()

    asyncio.run(run())


def test_events_empty_batch_is_noop() -> None:
    async def run() -> None:
        registry = _build_registry()
        store = CultivationStore(registry)
        record = await _create_basic(store)
        written = await store.append_events(
            cultivation_id=record.cultivation_id, kind="cycle", events=()
        )
        assert written == 0
        assert await store.list_events(record.cultivation_id) == ()
        registry.close()

    asyncio.run(run())


def test_events_isolated_per_cultivation() -> None:
    async def run() -> None:
        registry = _build_registry()
        store = CultivationStore(registry)
        a = await _create_basic(store, ai_id="cultivation:a", slug="a")
        b = await _create_basic(store, ai_id="cultivation:b", slug="b")
        await store.append_events(
            cultivation_id=a.cultivation_id,
            kind="cycle",
            events=({"cycle_index": 0, "topic": "a"},),
        )
        assert len(await store.list_events(a.cultivation_id)) == 1
        assert await store.list_events(b.cultivation_id) == ()
        registry.close()

    asyncio.run(run())


def test_supervision_status_transitions_persist() -> None:
    async def run() -> None:
        registry = _build_registry()
        store = CultivationStore(registry)
        record = await _create_basic(store)
        cid = record.cultivation_id
        paused = await store.update_status(
            cultivation_id=cid, status=CultivationStatus.PAUSED
        )
        assert paused.status is CultivationStatus.PAUSED
        resumed = await store.update_status(
            cultivation_id=cid, status=CultivationStatus.STUDYING
        )
        assert resumed.status is CultivationStatus.STUDYING
        failed = await store.update_status(
            cultivation_id=cid,
            status=CultivationStatus.FAILED,
            notes="operator_rejected",
        )
        assert failed.status is CultivationStatus.FAILED
        assert failed.notes == "operator_rejected"
        registry.close()

    asyncio.run(run())


def test_schema_migration_adds_v11_column_and_events_table(tmp_path) -> None:
    """Forward migration upgrades a pre-existing v10 cultivations DB."""

    db_path = tmp_path / "old_v10.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
    )
    # A pre-v11 cultivations table: has the v8 package columns but NOT
    # source_template_id, and no cultivation_events table.
    conn.execute(
        """
        CREATE TABLE cultivations (
            cultivation_id TEXT PRIMARY KEY,
            ai_id TEXT NOT NULL DEFAULT '',
            slug TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            domain TEXT NOT NULL DEFAULT '',
            runtime_template_id TEXT NOT NULL DEFAULT '',
            seed_persona_json TEXT NOT NULL DEFAULT '{}',
            curriculum_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'seeding',
            cycles_completed INTEGER NOT NULL DEFAULT 0,
            coherence_score REAL NOT NULL DEFAULT 0.0,
            coherence_detail_json TEXT NOT NULL DEFAULT '{}',
            regime_history_json TEXT NOT NULL DEFAULT '[]',
            dlaas_template_id TEXT NOT NULL DEFAULT '',
            last_exam_run_id TEXT NOT NULL DEFAULT '',
            inducted_template_id TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            tenant_id TEXT NOT NULL DEFAULT '',
            package_id TEXT NOT NULL DEFAULT '',
            track_id TEXT NOT NULL DEFAULT '',
            direction_json TEXT NOT NULL DEFAULT '{}',
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO cultivations (cultivation_id, slug, created_at_ms, "
        "updated_at_ms) VALUES ('cult_old', 'legacy', 1, 1)"
    )
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('schema_version', '10')"
    )
    conn.commit()
    conn.close()

    new_conn = open_connection(db_path)
    init_schema(new_conn)
    columns = {row[1] for row in new_conn.execute("PRAGMA table_info(cultivations)")}
    assert "source_template_id" in columns
    assert "provenance_json" in columns
    tables = {
        row[0]
        for row in new_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert "cultivation_events" in tables
    # Existing row preserved.
    kept = new_conn.execute(
        "SELECT cultivation_id, source_template_id FROM cultivations "
        "WHERE cultivation_id = 'cult_old'"
    ).fetchone()
    assert kept[0] == "cult_old"
    assert kept[1] == ""
    schema_value = new_conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()[0]
    assert schema_value == str(SCHEMA_VERSION)
    new_conn.close()
