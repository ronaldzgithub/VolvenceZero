"""Content-addressed store.

不变量：
- put_bytes(data) 返回 sha256(data)；重复内容在磁盘上只有一份。
- canonical_dumps(obj) 负责 Python dict/list/primitives 的稳定序列化
  （sort_keys=True, separators 紧凑, UTF-8, 不允许 NaN/Inf）。
- 任何 mutation 都必须产生新 sha；旧 sha 永远保留（R15 的物理基础）。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from .paths import LabsPaths


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _reject_nan(o: Any) -> Any:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        raise ValueError("NaN/Inf not allowed in canonical snapshot JSON")
    return o


def canonical_dumps(obj: Any) -> bytes:
    """Canonical JSON encoding used for every snapshot.

    Stable across Python versions so the resulting sha is reproducible.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=_reject_nan,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS snapshots (
    sha         TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    meta_json   TEXT NOT NULL
);
"""

_SCHEMA_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    probe_id       TEXT NOT NULL,
    wiring         TEXT NOT NULL,
    ablation_cell  TEXT NOT NULL,
    seed           INTEGER NOT NULL,
    knobs_sha      TEXT NOT NULL,
    input_sha      TEXT NOT NULL,
    output_sha     TEXT NOT NULL,
    readouts_sha   TEXT NOT NULL,
    manifest_sha   TEXT NOT NULL,
    created_at     REAL NOT NULL
);
"""

_SCHEMA_RUN_INDEX = """
CREATE INDEX IF NOT EXISTS idx_runs_probe ON runs(probe_id);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=60.0)
    # busy_timeout must be set BEFORE any other PRAGMA that might need a lock.
    conn.execute("PRAGMA busy_timeout=30000;")
    # WAL mode requires a brief exclusive lock on first set; retry if contended.
    for _ in range(5):
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            break
        except sqlite3.OperationalError:
            import time
            time.sleep(0.1)
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_SCHEMA_SNAPSHOTS)
    conn.execute(_SCHEMA_RUNS)
    for stmt in _SCHEMA_RUN_INDEX.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)


# ---------------------------------------------------------------------------
# CASStore
# ---------------------------------------------------------------------------

class CASStore:
    """Content-addressed store. All writes are idempotent.

    Not inherently thread-safe across processes for SQLite writes; rely on WAL
    + short transactions. Callers in the parallel scheduler should keep their
    own process-scoped CASStore instance.
    """

    def __init__(self, paths: LabsPaths):
        self.paths = paths
        self.paths.ensure()
        self._conn = _connect(self.paths.index_db)
        _ensure_schema(self._conn)

    # ---- byte-level ------------------------------------------------------

    def put_bytes(self, data: bytes, *, kind: str, meta: Optional[dict] = None) -> str:
        sha = sha256_bytes(data)
        dst = self._path_for(sha)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Use PID-unique temp name to avoid cross-process collision on .tmp
            tmp = dst.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_bytes(data)
            try:
                tmp.replace(dst)
            except OSError:
                # Another process beat us; that's fine (idempotent).
                tmp.unlink(missing_ok=True)
        meta_json = canonical_dumps(meta or {}).decode("utf-8")
        for attempt in range(5):
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO snapshots(sha, kind, size_bytes, created_at, meta_json) "
                    "VALUES (?, ?, ?, ?, ?);",
                    (sha, kind, len(data), time.time(), meta_json),
                )
                break
            except sqlite3.OperationalError:
                if attempt < 4:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise
        return sha

    def get_bytes(self, sha: str) -> bytes:
        dst = self._path_for(sha)
        if not dst.exists():
            raise KeyError(f"CAS miss for sha={sha}")
        return dst.read_bytes()

    def exists(self, sha: str) -> bool:
        return self._path_for(sha).exists()

    # ---- object-level (canonical JSON) ----------------------------------

    def put_obj(self, obj: Any, *, kind: str, meta: Optional[dict] = None) -> str:
        return self.put_bytes(canonical_dumps(obj), kind=kind, meta=meta)

    def get_obj(self, sha: str) -> Any:
        return json.loads(self.get_bytes(sha).decode("utf-8"))

    # ---- admin -----------------------------------------------------------

    def list_snapshots(self, *, kind: Optional[str] = None) -> Iterable[dict]:
        cur = self._conn.execute(
            "SELECT sha, kind, size_bytes, created_at, meta_json FROM snapshots "
            "WHERE (? IS NULL OR kind = ?) ORDER BY created_at DESC;",
            (kind, kind),
        )
        for row in cur:
            yield {
                "sha": row[0],
                "kind": row[1],
                "size_bytes": row[2],
                "created_at": row[3],
                "meta": json.loads(row[4]),
            }

    def _path_for(self, sha: str) -> Path:
        if len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha):
            raise ValueError(f"invalid sha256: {sha!r}")
        return self.paths.cas_dir / sha[:2] / f"{sha}.bin"

    # sqlite connection is intentionally not closed; process exit handles it.
    # For tests we expose a close() for determinism.
    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
