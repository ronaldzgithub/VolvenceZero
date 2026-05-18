# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""SQLite-backed persistence + in-memory fallback for harness components.

Schema (H-A enables only ``session_summary``; H-B / H-C enable the
remaining three):

* ``session_summary(scope_key TEXT, session_id TEXT, summary_json
  TEXT, extracted_at TEXT, extractor_model TEXT,
  PRIMARY KEY (scope_key, session_id))``
* ``embed_index(scope_key TEXT, turn_id TEXT, role TEXT, content
  TEXT, embedding BLOB, ts TEXT, PRIMARY KEY (scope_key, turn_id))``
  -- H-B enables; H-A leaves empty
* ``user_facts(scope_key TEXT, key TEXT, value TEXT, source_turn
  TEXT, confidence REAL, ts TEXT, PRIMARY KEY (scope_key, key))``
  -- H-C enables; H-A leaves empty
* ``episodic_events(scope_key TEXT, event_id TEXT, summary TEXT,
  source_turn TEXT, ts TEXT, PRIMARY KEY (scope_key, event_id))``
  -- H-C enables; H-A leaves empty

The class surface intentionally exposes *one method per (component,
operation)*. There is no generic ``execute(sql)`` escape hatch.
That keeps the SQL string corpus auditable and prevents downstream
modules from sneaking cross-table joins that would violate
snapshot isolation.

Both :class:`SqliteHarnessStore` and :class:`InMemoryHarnessStore`
implement the same Protocol-shaped surface so tests can swap
backends without touching the consumer code.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import pathlib
import sqlite3
import threading
from typing import Any, Protocol, runtime_checkable

from companion_ref_harness.session_summary import SessionSummary


# ---------------------------------------------------------------------------
# Mode enum + factory
# ---------------------------------------------------------------------------


class StoreMode(str, enum.Enum):
    """Backend selector for :func:`open_store`."""

    MEMORY = "memory"
    SQLITE = "sqlite"


def open_store(
    mode: StoreMode | str,
    *,
    sqlite_path: pathlib.Path | str | None = None,
) -> "HarnessStore":
    """Open a store for the given mode.

    Args:
        mode: ``"memory"`` for ephemeral (tests / SHADOW) or
            ``"sqlite"`` for persistent (ACTIVE).
        sqlite_path: When ``mode == sqlite``, the SQLite file path.
            ``None`` raises ``ValueError`` (no implicit default; the
            CLI passes ``--store-path`` explicitly).
    """

    parsed = StoreMode(mode) if isinstance(mode, str) else mode
    if parsed is StoreMode.MEMORY:
        return InMemoryHarnessStore()
    if parsed is StoreMode.SQLITE:
        if sqlite_path is None:
            raise ValueError(
                "open_store(mode='sqlite') requires sqlite_path; got None"
            )
        return SqliteHarnessStore(pathlib.Path(sqlite_path))
    raise ValueError(f"unknown StoreMode: {parsed!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class HarnessStore(Protocol):
    """The surface every component sees.

    Each method is named ``<component>_<verb>`` so a quick grep
    shows exactly which components touch which storage areas.
    """

    # --- session_summary (H-A) ---
    def session_summary_put(self, summary: SessionSummary) -> None: ...

    def session_summary_get(
        self,
        *,
        scope_key: str,
        session_id: str,
    ) -> SessionSummary | None: ...

    def session_summary_list_for_scope(
        self,
        *,
        scope_key: str,
        exclude_session_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[SessionSummary, ...]: ...

    # --- close + integrity ---
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory implementation (tests / SHADOW)
# ---------------------------------------------------------------------------


class InMemoryHarnessStore:
    """Pure-Python dict store with the same surface as SQLite.

    Used by unit tests and by SHADOW boot mode. Thread-safe via a
    single coarse lock — enough for benchmark workloads (single
    aiohttp event loop, no parallel writers).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # (scope_key, session_id) -> SessionSummary
        self._summaries: dict[tuple[str, str], SessionSummary] = {}

    def session_summary_put(self, summary: SessionSummary) -> None:
        with self._lock:
            self._summaries[(summary.scope_key, summary.session_id)] = summary

    def session_summary_get(
        self,
        *,
        scope_key: str,
        session_id: str,
    ) -> SessionSummary | None:
        with self._lock:
            return self._summaries.get((scope_key, session_id))

    def session_summary_list_for_scope(
        self,
        *,
        scope_key: str,
        exclude_session_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[SessionSummary, ...]:
        with self._lock:
            rows = [
                s
                for (sk, _sid), s in self._summaries.items()
                if sk == scope_key
                and (exclude_session_id is None or s.session_id != exclude_session_id)
            ]
        rows.sort(key=lambda s: s.extracted_at)
        if limit is not None:
            rows = rows[-limit:]
        return tuple(rows)

    def close(self) -> None:
        # Nothing to flush; included so the Protocol surface matches
        # SqliteHarnessStore.
        return


# ---------------------------------------------------------------------------
# SQLite implementation (ACTIVE)
# ---------------------------------------------------------------------------


_SCHEMA_SQL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS session_summary (
        scope_key TEXT NOT NULL,
        session_id TEXT NOT NULL,
        summary_json TEXT NOT NULL,
        extracted_at TEXT NOT NULL,
        extractor_model TEXT NOT NULL,
        PRIMARY KEY (scope_key, session_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_session_summary_scope_ts
        ON session_summary (scope_key, extracted_at)
    """,
    # H-B reserved table
    """
    CREATE TABLE IF NOT EXISTS embed_index (
        scope_key TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding BLOB NOT NULL,
        ts TEXT NOT NULL,
        PRIMARY KEY (scope_key, turn_id)
    )
    """,
    # H-C reserved tables
    """
    CREATE TABLE IF NOT EXISTS user_facts (
        scope_key TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        source_turn TEXT NOT NULL,
        confidence REAL NOT NULL,
        ts TEXT NOT NULL,
        PRIMARY KEY (scope_key, key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS episodic_events (
        scope_key TEXT NOT NULL,
        event_id TEXT NOT NULL,
        summary TEXT NOT NULL,
        source_turn TEXT NOT NULL,
        ts TEXT NOT NULL,
        PRIMARY KEY (scope_key, event_id)
    )
    """,
)


class SqliteHarnessStore:
    """File-backed SQLite store; thread-safe for single-process use.

    A single :class:`sqlite3.Connection` is held with
    ``check_same_thread=False`` and protected by a Python lock.
    aiohttp runs one event loop per worker, so this is sufficient
    for the benchmark workload.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        for stmt in _SCHEMA_SQL:
            self._conn.execute(stmt)
        self._conn.commit()

    @property
    def path(self) -> pathlib.Path:
        return self._path

    # ----- session_summary --------------------------------------------------

    def session_summary_put(self, summary: SessionSummary) -> None:
        payload = json.dumps(summary.to_payload(), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO session_summary
                    (scope_key, session_id, summary_json, extracted_at, extractor_model)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (scope_key, session_id) DO UPDATE SET
                    summary_json = excluded.summary_json,
                    extracted_at = excluded.extracted_at,
                    extractor_model = excluded.extractor_model
                """,
                (
                    summary.scope_key,
                    summary.session_id,
                    payload,
                    summary.extracted_at,
                    summary.extractor_model,
                ),
            )
            self._conn.commit()

    def session_summary_get(
        self,
        *,
        scope_key: str,
        session_id: str,
    ) -> SessionSummary | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT scope_key, session_id, summary_json, extracted_at, extractor_model
                FROM session_summary
                WHERE scope_key = ? AND session_id = ?
                """,
                (scope_key, session_id),
            ).fetchone()
        if row is None:
            return None
        return _row_to_summary(row)

    def session_summary_list_for_scope(
        self,
        *,
        scope_key: str,
        exclude_session_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[SessionSummary, ...]:
        if exclude_session_id is None:
            sql = (
                "SELECT scope_key, session_id, summary_json, extracted_at, extractor_model "
                "FROM session_summary WHERE scope_key = ? "
                "ORDER BY extracted_at ASC"
            )
            params: tuple[Any, ...] = (scope_key,)
        else:
            sql = (
                "SELECT scope_key, session_id, summary_json, extracted_at, extractor_model "
                "FROM session_summary WHERE scope_key = ? AND session_id != ? "
                "ORDER BY extracted_at ASC"
            )
            params = (scope_key, exclude_session_id)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        summaries = tuple(_row_to_summary(row) for row in rows)
        if limit is not None and len(summaries) > limit:
            summaries = summaries[-limit:]
        return summaries

    # ----- lifecycle --------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()


def _row_to_summary(
    row: tuple[str, str, str, str, str],
) -> SessionSummary:
    scope_key, session_id, payload_json, extracted_at, extractor_model = row
    payload = json.loads(payload_json)
    return SessionSummary(
        scope_key=scope_key,
        session_id=session_id,
        topic=payload["topic"],
        commitments=tuple(payload.get("commitments", ())),
        open_loops=tuple(payload.get("open_loops", ())),
        extracted_at=extracted_at,
        extractor_model=extractor_model,
    )


# Defensive: dataclasses module imported for downstream typing
_ = dataclasses
