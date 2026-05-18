# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the in-memory + SQLite component stores."""

from __future__ import annotations

import pathlib

import pytest

from companion_ref_harness.session_summary import SessionSummary
from companion_ref_harness.store.sqlite_store import (
    InMemoryHarnessStore,
    SqliteHarnessStore,
    StoreMode,
    open_store,
)


def _make_summary(
    *,
    scope_key: str = "user-a",
    session_id: str = "s1",
    topic: str = "moved to a new flat",
    extracted_at: str = "2026-05-17T00:00:00+00:00",
) -> SessionSummary:
    return SessionSummary(
        scope_key=scope_key,
        session_id=session_id,
        topic=topic,
        commitments=("user: will set up the desk on saturday",),
        open_loops=("assistant has not replied about the cat",),
        extracted_at=extracted_at,
        extractor_model="test/extractor",
    )


# ---------------------------------------------------------------------------
# In-memory
# ---------------------------------------------------------------------------


def test_in_memory_store_round_trip() -> None:
    store = InMemoryHarnessStore()
    summary = _make_summary()
    store.session_summary_put(summary)
    fetched = store.session_summary_get(scope_key="user-a", session_id="s1")
    assert fetched == summary


def test_in_memory_store_overwrites_same_key() -> None:
    store = InMemoryHarnessStore()
    first = _make_summary(topic="first version")
    second = _make_summary(topic="second version", extracted_at="2026-05-18T00:00:00+00:00")
    store.session_summary_put(first)
    store.session_summary_put(second)
    fetched = store.session_summary_get(scope_key="user-a", session_id="s1")
    assert fetched is not None
    assert fetched.topic == "second version"


def test_in_memory_store_list_filters_and_orders() -> None:
    store = InMemoryHarnessStore()
    store.session_summary_put(_make_summary(session_id="s1", extracted_at="2026-05-15T00:00:00+00:00"))
    store.session_summary_put(_make_summary(session_id="s2", extracted_at="2026-05-16T00:00:00+00:00"))
    store.session_summary_put(_make_summary(session_id="s3", extracted_at="2026-05-17T00:00:00+00:00"))
    # Different scope shouldn't leak in.
    store.session_summary_put(_make_summary(scope_key="user-b", session_id="b1"))
    result = store.session_summary_list_for_scope(
        scope_key="user-a", exclude_session_id="s2",
    )
    assert tuple(s.session_id for s in result) == ("s1", "s3")


def test_in_memory_store_list_respects_limit() -> None:
    store = InMemoryHarnessStore()
    for i in range(5):
        store.session_summary_put(
            _make_summary(session_id=f"s{i}", extracted_at=f"2026-05-{10+i:02d}T00:00:00+00:00")
        )
    result = store.session_summary_list_for_scope(scope_key="user-a", limit=3)
    # Limit returns the most recent (tail of the chronologically-sorted list).
    assert tuple(s.session_id for s in result) == ("s2", "s3", "s4")


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


def test_sqlite_store_round_trip(tmp_path: pathlib.Path) -> None:
    store = SqliteHarnessStore(tmp_path / "h.sqlite3")
    summary = _make_summary()
    store.session_summary_put(summary)
    fetched = store.session_summary_get(scope_key="user-a", session_id="s1")
    assert fetched == summary
    store.close()


def test_sqlite_store_persists_across_reopens(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "h.sqlite3"
    first = SqliteHarnessStore(path)
    first.session_summary_put(_make_summary())
    first.close()
    second = SqliteHarnessStore(path)
    fetched = second.session_summary_get(scope_key="user-a", session_id="s1")
    assert fetched is not None
    assert fetched.topic == "moved to a new flat"
    second.close()


def test_sqlite_store_upsert(tmp_path: pathlib.Path) -> None:
    store = SqliteHarnessStore(tmp_path / "h.sqlite3")
    store.session_summary_put(_make_summary(topic="v1"))
    store.session_summary_put(_make_summary(topic="v2", extracted_at="2026-06-01T00:00:00+00:00"))
    fetched = store.session_summary_get(scope_key="user-a", session_id="s1")
    assert fetched is not None
    assert fetched.topic == "v2"
    store.close()


def test_open_store_dispatches_correctly(tmp_path: pathlib.Path) -> None:
    mem = open_store(StoreMode.MEMORY)
    assert isinstance(mem, InMemoryHarnessStore)
    mem.close()

    sql = open_store("sqlite", sqlite_path=tmp_path / "h.sqlite3")
    assert isinstance(sql, SqliteHarnessStore)
    sql.close()


def test_open_store_sqlite_without_path_fails_loudly() -> None:
    with pytest.raises(ValueError, match="sqlite_path"):
        open_store(StoreMode.SQLITE)


def test_sqlite_store_h_b_h_c_tables_exist_but_empty(tmp_path: pathlib.Path) -> None:
    """H-A should create H-B / H-C reserved tables but never write to them."""
    import sqlite3

    store = SqliteHarnessStore(tmp_path / "h.sqlite3")
    store.session_summary_put(_make_summary())
    store.close()
    conn = sqlite3.connect(str(tmp_path / "h.sqlite3"))
    try:
        for table in ("embed_index", "user_facts", "episodic_events"):
            count_row = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()
            assert count_row is not None
            assert count_row[0] == 0, (
                f"H-A unexpectedly wrote to {table}. Only H-B / H-C should "
                "populate these tables; H-A must leave them empty."
            )
    finally:
        conn.close()
