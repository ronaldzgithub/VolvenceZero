# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""SQLite-backed persistence + in-memory fallback for the CAMEL baseline.

The CAMEL baseline keeps a single component of durable state: a compact
per-session memory record (one per ``(scope_key, session_id)``). At the start
of a new session for the same user the prior records are re-seeded into the
agent's context, giving CAMEL cross-session continuity on top of its
within-session context window.

Schema::

    session_memory(
        scope_key TEXT, session_id TEXT, topic TEXT, salient_json TEXT,
        extracted_at TEXT, extractor_model TEXT,
        PRIMARY KEY (scope_key, session_id)
    )

The class surface exposes one method per (component, operation). There is no
generic ``execute(sql)`` escape hatch, so the SQL corpus stays auditable.

Both :class:`SqliteMemoryStore` and :class:`InMemoryStore` implement the same
Protocol-shaped surface, so tests can swap backends without touching consumers.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import pathlib
import sqlite3
import threading
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SessionMemoryRecord:
    """One compacted per-session memory record, immutable after extraction."""

    scope_key: str
    session_id: str
    topic: str
    salient: tuple[str, ...]
    extracted_at: str  # ISO-8601 UTC
    extractor_model: str

    def to_payload(self) -> dict[str, Any]:
        return {"topic": self.topic, "salient": list(self.salient)}

    def to_prompt_block(self) -> str:
        """Markdown fragment re-seeded into the next session's context."""

        lines: list[str] = [f"- session {self.session_id}: {self.topic}"]
        for item in self.salient:
            lines.append(f"    - {item}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode enum + factory
# ---------------------------------------------------------------------------


class StoreMode(str, enum.Enum):
    """Backend selector for :func:`open_store`."""

    MEMORY = "memory"
    SQLITE = "sqlite"


def open_store(
    mode: "StoreMode | str",
    *,
    sqlite_path: pathlib.Path | str | None = None,
) -> "MemoryStore":
    """Open a store for the given mode.

    Args:
        mode: ``"memory"`` for ephemeral (tests / smoke) or ``"sqlite"`` for
            persistent runs.
        sqlite_path: Required when ``mode == sqlite``; ``None`` raises
            ``ValueError`` (no implicit default — the CLI passes
            ``--store-path`` explicitly).
    """

    parsed = StoreMode(mode) if isinstance(mode, str) else mode
    if parsed is StoreMode.MEMORY:
        return InMemoryStore()
    if parsed is StoreMode.SQLITE:
        if sqlite_path is None:
            raise ValueError(
                "open_store(mode='sqlite') requires sqlite_path; got None"
            )
        return SqliteMemoryStore(pathlib.Path(sqlite_path))
    raise ValueError(f"unknown StoreMode: {parsed!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryStore(Protocol):
    """The surface the server / backend sees."""

    def session_memory_put(self, record: SessionMemoryRecord) -> None: ...

    def session_memory_get(
        self, *, scope_key: str, session_id: str,
    ) -> SessionMemoryRecord | None: ...

    def session_memory_list_for_scope(
        self,
        *,
        scope_key: str,
        exclude_session_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[SessionMemoryRecord, ...]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory implementation (tests / smoke)
# ---------------------------------------------------------------------------


class InMemoryStore:
    """Pure-Python dict store with the same surface as SQLite."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str], SessionMemoryRecord] = {}

    def session_memory_put(self, record: SessionMemoryRecord) -> None:
        with self._lock:
            self._records[(record.scope_key, record.session_id)] = record

    def session_memory_get(
        self, *, scope_key: str, session_id: str,
    ) -> SessionMemoryRecord | None:
        with self._lock:
            return self._records.get((scope_key, session_id))

    def session_memory_list_for_scope(
        self,
        *,
        scope_key: str,
        exclude_session_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[SessionMemoryRecord, ...]:
        with self._lock:
            rows = [
                r
                for (sk, _sid), r in self._records.items()
                if sk == scope_key
                and (exclude_session_id is None or r.session_id != exclude_session_id)
            ]
        rows.sort(key=lambda r: r.extracted_at)
        if limit is not None:
            rows = rows[-limit:]
        return tuple(rows)

    def close(self) -> None:
        return


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------


_SCHEMA_SQL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS session_memory (
        scope_key TEXT NOT NULL,
        session_id TEXT NOT NULL,
        topic TEXT NOT NULL,
        salient_json TEXT NOT NULL,
        extracted_at TEXT NOT NULL,
        extractor_model TEXT NOT NULL,
        PRIMARY KEY (scope_key, session_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_session_memory_scope_ts
        ON session_memory (scope_key, extracted_at)
    """,
)


class SqliteMemoryStore:
    """File-backed SQLite store; thread-safe for single-process use."""

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

    def session_memory_put(self, record: SessionMemoryRecord) -> None:
        payload = json.dumps(record.to_payload(), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO session_memory
                    (scope_key, session_id, topic, salient_json, extracted_at, extractor_model)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (scope_key, session_id) DO UPDATE SET
                    topic = excluded.topic,
                    salient_json = excluded.salient_json,
                    extracted_at = excluded.extracted_at,
                    extractor_model = excluded.extractor_model
                """,
                (
                    record.scope_key,
                    record.session_id,
                    record.topic,
                    payload,
                    record.extracted_at,
                    record.extractor_model,
                ),
            )
            self._conn.commit()

    def session_memory_get(
        self, *, scope_key: str, session_id: str,
    ) -> SessionMemoryRecord | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT scope_key, session_id, topic, salient_json, extracted_at, extractor_model
                FROM session_memory
                WHERE scope_key = ? AND session_id = ?
                """,
                (scope_key, session_id),
            ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def session_memory_list_for_scope(
        self,
        *,
        scope_key: str,
        exclude_session_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[SessionMemoryRecord, ...]:
        if exclude_session_id is None:
            sql = (
                "SELECT scope_key, session_id, topic, salient_json, extracted_at, extractor_model "
                "FROM session_memory WHERE scope_key = ? ORDER BY extracted_at ASC"
            )
            params: tuple[Any, ...] = (scope_key,)
        else:
            sql = (
                "SELECT scope_key, session_id, topic, salient_json, extracted_at, extractor_model "
                "FROM session_memory WHERE scope_key = ? AND session_id != ? "
                "ORDER BY extracted_at ASC"
            )
            params = (scope_key, exclude_session_id)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        records = tuple(_row_to_record(row) for row in rows)
        if limit is not None and len(records) > limit:
            records = records[-limit:]
        return records

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()


def _row_to_record(
    row: tuple[str, str, str, str, str, str],
) -> SessionMemoryRecord:
    scope_key, session_id, topic, salient_json, extracted_at, extractor_model = row
    salient = tuple(json.loads(salient_json).get("salient", ()))
    return SessionMemoryRecord(
        scope_key=scope_key,
        session_id=session_id,
        topic=topic,
        salient=salient,
        extracted_at=extracted_at,
        extractor_model=extractor_model,
    )
