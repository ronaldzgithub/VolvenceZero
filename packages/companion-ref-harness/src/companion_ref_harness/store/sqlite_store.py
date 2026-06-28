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

from companion_ref_harness.embed import EmbedEntry
from companion_ref_harness.episodic import EpisodicEvent
from companion_ref_harness.session_summary import SessionSummary
from companion_ref_harness.user_model import UserFact


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

    # --- embed_index (H-B) ---
    def embed_index_put(self, entry: EmbedEntry) -> None: ...

    def embed_index_list_for_scope(
        self, *, scope_key: str,
    ) -> tuple[EmbedEntry, ...]: ...

    # --- user_facts (H-C) ---
    def user_fact_put(self, fact: UserFact) -> None: ...

    def user_fact_list_for_scope(
        self, *, scope_key: str,
    ) -> tuple[UserFact, ...]: ...

    # --- episodic_events (H-C) ---
    def episodic_put(self, event: EpisodicEvent) -> None: ...

    def episodic_list_for_scope(
        self, *, scope_key: str, limit: int | None = None,
    ) -> tuple[EpisodicEvent, ...]: ...

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
        # (scope_key, turn_id) -> EmbedEntry
        self._embed: dict[tuple[str, str], EmbedEntry] = {}
        # (scope_key, key) -> UserFact
        self._facts: dict[tuple[str, str], UserFact] = {}
        # (scope_key, event_id) -> EpisodicEvent
        self._episodic: dict[tuple[str, str], EpisodicEvent] = {}

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

    # ----- embed_index (H-B) -----------------------------------------------

    def embed_index_put(self, entry: EmbedEntry) -> None:
        with self._lock:
            self._embed[(entry.scope_key, entry.turn_id)] = entry

    def embed_index_list_for_scope(
        self, *, scope_key: str,
    ) -> tuple[EmbedEntry, ...]:
        with self._lock:
            rows = [e for (sk, _t), e in self._embed.items() if sk == scope_key]
        rows.sort(key=lambda e: e.ts)
        return tuple(rows)

    # ----- user_facts (H-C) ------------------------------------------------

    def user_fact_put(self, fact: UserFact) -> None:
        with self._lock:
            self._facts[(fact.scope_key, fact.key)] = fact

    def user_fact_list_for_scope(
        self, *, scope_key: str,
    ) -> tuple[UserFact, ...]:
        with self._lock:
            rows = [f for (sk, _k), f in self._facts.items() if sk == scope_key]
        rows.sort(key=lambda f: f.ts)
        return tuple(rows)

    # ----- episodic_events (H-C) -------------------------------------------

    def episodic_put(self, event: EpisodicEvent) -> None:
        with self._lock:
            self._episodic[(event.scope_key, event.event_id)] = event

    def episodic_list_for_scope(
        self, *, scope_key: str, limit: int | None = None,
    ) -> tuple[EpisodicEvent, ...]:
        with self._lock:
            rows = [e for (sk, _e), e in self._episodic.items() if sk == scope_key]
        rows.sort(key=lambda e: e.ts)
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

    # ----- embed_index (H-B) -----------------------------------------------

    def embed_index_put(self, entry: EmbedEntry) -> None:
        blob = json.dumps(list(entry.embedding)).encode("utf-8")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO embed_index
                    (scope_key, turn_id, role, content, embedding, ts)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (scope_key, turn_id) DO UPDATE SET
                    role = excluded.role,
                    content = excluded.content,
                    embedding = excluded.embedding,
                    ts = excluded.ts
                """,
                (entry.scope_key, entry.turn_id, entry.role, entry.content, blob, entry.ts),
            )
            self._conn.commit()

    def embed_index_list_for_scope(
        self, *, scope_key: str,
    ) -> tuple[EmbedEntry, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT scope_key, turn_id, role, content, embedding, ts "
                "FROM embed_index WHERE scope_key = ? ORDER BY ts ASC",
                (scope_key,),
            ).fetchall()
        return tuple(_row_to_embed(row) for row in rows)

    # ----- user_facts (H-C) ------------------------------------------------

    def user_fact_put(self, fact: UserFact) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO user_facts
                    (scope_key, key, value, source_turn, confidence, ts)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (scope_key, key) DO UPDATE SET
                    value = excluded.value,
                    source_turn = excluded.source_turn,
                    confidence = excluded.confidence,
                    ts = excluded.ts
                """,
                (fact.scope_key, fact.key, fact.value, fact.source_turn, fact.confidence, fact.ts),
            )
            self._conn.commit()

    def user_fact_list_for_scope(
        self, *, scope_key: str,
    ) -> tuple[UserFact, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT scope_key, key, value, source_turn, confidence, ts "
                "FROM user_facts WHERE scope_key = ? ORDER BY ts ASC",
                (scope_key,),
            ).fetchall()
        return tuple(_row_to_fact(row) for row in rows)

    # ----- episodic_events (H-C) -------------------------------------------

    def episodic_put(self, event: EpisodicEvent) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO episodic_events
                    (scope_key, event_id, summary, source_turn, ts)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (scope_key, event_id) DO UPDATE SET
                    summary = excluded.summary,
                    source_turn = excluded.source_turn,
                    ts = excluded.ts
                """,
                (event.scope_key, event.event_id, event.summary, event.source_turn, event.ts),
            )
            self._conn.commit()

    def episodic_list_for_scope(
        self, *, scope_key: str, limit: int | None = None,
    ) -> tuple[EpisodicEvent, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT scope_key, event_id, summary, source_turn, ts "
                "FROM episodic_events WHERE scope_key = ? ORDER BY ts ASC",
                (scope_key,),
            ).fetchall()
        events = tuple(_row_to_episodic(row) for row in rows)
        if limit is not None and len(events) > limit:
            events = events[-limit:]
        return events

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


def _row_to_embed(row: tuple[str, str, str, str, bytes, str]) -> EmbedEntry:
    scope_key, turn_id, role, content, blob, ts = row
    embedding = tuple(float(x) for x in json.loads(bytes(blob).decode("utf-8")))
    return EmbedEntry(
        scope_key=scope_key,
        turn_id=turn_id,
        role=role,
        content=content,
        embedding=embedding,
        ts=ts,
    )


def _row_to_fact(row: tuple[str, str, str, str, float, str]) -> UserFact:
    scope_key, key, value, source_turn, confidence, ts = row
    return UserFact(
        scope_key=scope_key,
        key=key,
        value=value,
        source_turn=source_turn,
        confidence=float(confidence),
        ts=ts,
    )


def _row_to_episodic(row: tuple[str, str, str, str, str]) -> EpisodicEvent:
    scope_key, event_id, summary, source_turn, ts = row
    return EpisodicEvent(
        scope_key=scope_key,
        event_id=event_id,
        summary=summary,
        source_turn=source_turn,
        ts=ts,
    )


# Defensive: dataclasses module imported for downstream typing
_ = dataclasses
