"""Tests for the durable cognition snapshot store (Track 4 FULL).

* In-memory backend keeps the historical behaviour.
* SQLite backend persists across a store re-open (simulating a
  platform-api restart) and enforces ``DLAAS_COGNITION_RETENTION_DAYS``.
"""

from __future__ import annotations

from pathlib import Path

from dlaas_platform_api.cognition import CognitionSnapshotStore


def _row(snapshot_id: str, captured_at_ms: int, ai_id: str = "ai-1") -> dict:
    return {
        "snapshot_id": snapshot_id,
        "tenant_id": "t-1",
        "ai_id": ai_id,
        "session_id": "s-1",
        "source": "interaction",
        "captured_at_ms": captured_at_ms,
        "regime_id": "acquaintance",
        "raw_readout": {"social": {"present": True}},
    }


def test_in_memory_append_and_iterate() -> None:
    store = CognitionSnapshotStore()
    assert store.durable is False
    store.append(_row("c1", 1000))
    store.append(_row("c2", 2000))
    rows = list(store)
    assert {r["snapshot_id"] for r in rows} == {"c1", "c2"}
    # Row shape (incl. nested raw_readout) is preserved verbatim.
    assert rows[0]["raw_readout"]["social"]["present"] is True


def test_sqlite_persists_across_reopen(tmp_path: Path) -> None:
    db = str(tmp_path / "cog.db")
    store = CognitionSnapshotStore(db_path=db)
    assert store.durable is True
    store.append(_row("c1", 1000))
    store.append(_row("c2", 2000))
    # Re-open a fresh store on the same file -> rows survive (durable).
    reopened = CognitionSnapshotStore(db_path=db)
    ids = {r["snapshot_id"] for r in reopened}
    assert ids == {"c1", "c2"}
    assert next(iter(reopened))["raw_readout"]["social"]["present"] is True


def test_sqlite_insert_or_replace_is_idempotent(tmp_path: Path) -> None:
    db = str(tmp_path / "cog.db")
    store = CognitionSnapshotStore(db_path=db)
    store.append(_row("c1", 1000))
    store.append(_row("c1", 1500))  # same id -> replace, not duplicate
    rows = list(store)
    assert len(rows) == 1
    assert rows[0]["captured_at_ms"] == 1500


def test_retention_sweep_deletes_old_rows_memory() -> None:
    # No retention at construction -> no auto-sweep on append; we drive
    # the sweep explicitly with an injected horizon + clock.
    store = CognitionSnapshotStore()
    now = 30 * 86_400_000
    store.append(_row("old", now - 10 * 86_400_000))
    store.append(_row("fresh", now - 1 * 86_400_000))
    deleted = store.sweep(retention_days=7, now_ms=now)
    assert deleted == 1
    assert {r["snapshot_id"] for r in store} == {"fresh"}


def test_retention_sweep_deletes_old_rows_sqlite(tmp_path: Path) -> None:
    db = str(tmp_path / "cog.db")
    store = CognitionSnapshotStore(db_path=db)
    now = 30 * 86_400_000
    store.append(_row("old", now - 10 * 86_400_000))
    store.append(_row("fresh", now - 1 * 86_400_000))
    deleted = store.sweep(retention_days=7, now_ms=now)
    assert deleted == 1
    assert {r["snapshot_id"] for r in store} == {"fresh"}


def test_auto_sweep_on_append_evicts_old_rows() -> None:
    # With retention configured, appending fresh-then-stale rows lets the
    # throttled auto-sweep evict anything past the horizon relative to
    # the real clock. Rows far in the past are removed on the next append.
    store = CognitionSnapshotStore(retention_days=7)
    store.append(_row("ancient", 1000))  # ~1970 -> well past any horizon
    assert "ancient" not in {r["snapshot_id"] for r in store}


def test_sweep_noop_without_retention() -> None:
    store = CognitionSnapshotStore()
    store.append(_row("c1", 1000))
    assert store.sweep(now_ms=10**12) == 0
    assert len(list(store)) == 1
