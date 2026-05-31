from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, fields
from typing import Any


SCHEMA_VERSION = 1


class PersistenceBackend(ABC):
    """Abstract interface for cross-session checkpoint persistence."""

    @abstractmethod
    def save_checkpoint(self, *, key: str, data: bytes, version: int) -> None: ...

    @abstractmethod
    def load_checkpoint(self, *, key: str) -> tuple[bytes, int] | None: ...

    @abstractmethod
    def list_checkpoints(self, *, prefix: str) -> tuple[str, ...]: ...

    @abstractmethod
    def delete_checkpoint(self, *, key: str) -> None: ...


class FileSystemPersistenceBackend(PersistenceBackend):
    """JSON file-based persistence with version-aware save/load and automatic cleanup."""

    def __init__(self, *, base_dir: str, max_versions: int = 5) -> None:
        self._base_dir = base_dir
        self._max_versions = max(max_versions, 1)
        os.makedirs(base_dir, exist_ok=True)

    def save_checkpoint(self, *, key: str, data: bytes, version: int) -> None:
        safe_key = key.replace("/", "__").replace("\\", "__")
        filename = f"{safe_key}_v{version}.json"
        filepath = os.path.join(self._base_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data.decode("utf-8"))
        self._cleanup_old_versions(safe_key=safe_key, current_version=version)

    def load_checkpoint(self, *, key: str) -> tuple[bytes, int] | None:
        safe_key = key.replace("/", "__").replace("\\", "__")
        best_version = -1
        best_filepath: str | None = None
        prefix = f"{safe_key}_v"
        try:
            entries = os.listdir(self._base_dir)
        except FileNotFoundError:
            return None
        for entry in entries:
            if entry.startswith(prefix) and entry.endswith(".json"):
                version_str = entry[len(prefix):-len(".json")]
                try:
                    v = int(version_str)
                except ValueError:
                    continue
                if v > best_version:
                    best_version = v
                    best_filepath = os.path.join(self._base_dir, entry)
        if best_filepath is None:
            return None
        with open(best_filepath, "r", encoding="utf-8") as f:
            return (f.read().encode("utf-8"), best_version)

    def list_checkpoints(self, *, prefix: str) -> tuple[str, ...]:
        safe_prefix = prefix.replace("/", "__").replace("\\", "__")
        try:
            entries = os.listdir(self._base_dir)
        except FileNotFoundError:
            return ()
        keys: set[str] = set()
        for entry in entries:
            if entry.startswith(safe_prefix) and entry.endswith(".json"):
                parts = entry.rsplit("_v", 1)
                if parts:
                    keys.add(parts[0].replace("__", "/"))
        return tuple(sorted(keys))

    def delete_checkpoint(self, *, key: str) -> None:
        safe_key = key.replace("/", "__").replace("\\", "__")
        prefix = f"{safe_key}_v"
        try:
            entries = os.listdir(self._base_dir)
        except FileNotFoundError:
            return
        for entry in entries:
            if entry.startswith(prefix) and entry.endswith(".json"):
                os.remove(os.path.join(self._base_dir, entry))

    def _cleanup_old_versions(self, *, safe_key: str, current_version: int) -> None:
        prefix = f"{safe_key}_v"
        try:
            entries = os.listdir(self._base_dir)
        except FileNotFoundError:
            return
        versioned: list[tuple[int, str]] = []
        for entry in entries:
            if entry.startswith(prefix) and entry.endswith(".json"):
                version_str = entry[len(prefix):-len(".json")]
                try:
                    v = int(version_str)
                except ValueError:
                    continue
                versioned.append((v, entry))
        versioned.sort(reverse=True)
        for _, entry in versioned[self._max_versions:]:
            try:
                os.remove(os.path.join(self._base_dir, entry))
            except FileNotFoundError:
                pass


class InMemoryPersistenceBackend(PersistenceBackend):
    """Process-local, dict-backed persistence backend.

    Implements the same :class:`PersistenceBackend` contract as
    :class:`FileSystemPersistenceBackend` but keeps every checkpoint in
    a thread-safe in-process dict instead of on disk. This is the
    reference backend used to prove the *pluggable* scoped-memory
    contract (debt D2): the store does not care which backend it talks
    to as long as the four CRUD methods behave identically. It is also
    handy for tests that must not touch the filesystem.

    Not durable across process restarts — callers that need durability
    pick :class:`FileSystemPersistenceBackend` or a remote backend.
    """

    def __init__(self, *, max_versions: int = 5) -> None:
        self._max_versions = max(max_versions, 1)
        # key -> {version: data}
        self._store: dict[str, dict[int, bytes]] = {}
        self._lock = threading.Lock()

    def save_checkpoint(self, *, key: str, data: bytes, version: int) -> None:
        with self._lock:
            versions = self._store.setdefault(key, {})
            versions[version] = bytes(data)
            # Keep only the newest ``max_versions`` versions.
            if len(versions) > self._max_versions:
                for stale in sorted(versions)[: -self._max_versions]:
                    versions.pop(stale, None)

    def load_checkpoint(self, *, key: str) -> tuple[bytes, int] | None:
        with self._lock:
            versions = self._store.get(key)
            if not versions:
                return None
            best = max(versions)
            return (versions[best], best)

    def list_checkpoints(self, *, prefix: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                sorted(key for key in self._store if key.startswith(prefix))
            )

    def delete_checkpoint(self, *, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)


class PostgresPersistenceBackend(PersistenceBackend):
    """Postgres-backed scoped-memory persistence (debt D2).

    Stores versioned checkpoints in a single ``memory_checkpoints``
    table keyed by ``(namespace, key, version)``. ``namespace`` lets a
    single Postgres instance hold the scoped memory of every tenant /
    end-user without the per-scope directory fan-out the filesystem
    backend needs — the control-plane registry and the scoped-memory
    store can then share one managed Postgres instead of a shared PVC
    full of SQLite/JSON files.

    The :mod:`psycopg` driver is imported lazily so importing this
    module never requires Postgres to be installed (CI / unit tests run
    without it). Instantiation raises a clear, actionable error when the
    driver is absent rather than failing obscurely later.
    """

    _DDL = (
        """
        CREATE TABLE IF NOT EXISTS memory_checkpoints (
            namespace   TEXT    NOT NULL,
            ckpt_key    TEXT    NOT NULL,
            version     INTEGER NOT NULL,
            data        BYTEA   NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (namespace, ckpt_key, version)
        );
        """,
        "CREATE INDEX IF NOT EXISTS memory_checkpoints_ns_key_idx "
        "ON memory_checkpoints (namespace, ckpt_key);",
    )

    def __init__(
        self,
        *,
        dsn: str,
        namespace: str = "default",
        max_versions: int = 5,
        connection: Any = None,
    ) -> None:
        if not dsn and connection is None:
            raise ValueError(
                "PostgresPersistenceBackend requires a DSN (or an injected connection)"
            )
        self._namespace = namespace
        self._max_versions = max(max_versions, 1)
        self._lock = threading.Lock()
        if connection is not None:
            self._conn = connection
        else:
            try:
                import psycopg  # type: ignore
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "PostgresPersistenceBackend needs the 'psycopg' driver. "
                    "Install it (pip install 'psycopg[binary]') or select a "
                    "different backend via VZ_MEMORY_BACKEND."
                ) from exc
            self._conn = psycopg.connect(dsn, autocommit=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock, self._conn.cursor() as cur:
            for stmt in self._DDL:
                cur.execute(stmt)

    def save_checkpoint(self, *, key: str, data: bytes, version: int) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memory_checkpoints (namespace, ckpt_key, version, data) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (namespace, ckpt_key, version) DO UPDATE SET "
                "data = EXCLUDED.data, updated_at = now()",
                (self._namespace, key, version, data),
            )
            # Prune to the newest ``max_versions`` versions for this key.
            cur.execute(
                "DELETE FROM memory_checkpoints "
                "WHERE namespace = %s AND ckpt_key = %s AND version NOT IN ("
                "  SELECT version FROM memory_checkpoints "
                "  WHERE namespace = %s AND ckpt_key = %s "
                "  ORDER BY version DESC LIMIT %s"
                ")",
                (self._namespace, key, self._namespace, key, self._max_versions),
            )

    def load_checkpoint(self, *, key: str) -> tuple[bytes, int] | None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT data, version FROM memory_checkpoints "
                "WHERE namespace = %s AND ckpt_key = %s "
                "ORDER BY version DESC LIMIT 1",
                (self._namespace, key),
            )
            row = cur.fetchone()
        if row is None:
            return None
        data, version = row[0], int(row[1])
        return (bytes(data), version)

    def list_checkpoints(self, *, prefix: str) -> tuple[str, ...]:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT ckpt_key FROM memory_checkpoints "
                "WHERE namespace = %s AND ckpt_key LIKE %s",
                (self._namespace, prefix.replace("%", r"\%") + "%"),
            )
            rows = cur.fetchall()
        return tuple(sorted(str(row[0]) for row in rows))

    def delete_checkpoint(self, *, key: str) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM memory_checkpoints "
                "WHERE namespace = %s AND ckpt_key = %s",
                (self._namespace, key),
            )


def resolve_persistence_backend(
    *,
    base_dir: str | os.PathLike[str] | None = None,
    namespace: str = "default",
    max_versions: int = 5,
    backend: str | None = None,
    dsn: str | None = None,
) -> PersistenceBackend:
    """SSOT for selecting a scoped-memory :class:`PersistenceBackend`.

    Backend selection (debt D2 — pluggable scoped-memory backend):

    * ``backend`` argument, else env ``VZ_MEMORY_BACKEND``, else
      ``"filesystem"`` (the historical default — behaviour unchanged).
    * ``"filesystem"`` → :class:`FileSystemPersistenceBackend` rooted at
      ``base_dir`` (required for this backend).
    * ``"memory"`` → :class:`InMemoryPersistenceBackend` (process-local).
    * ``"postgres"`` → :class:`PostgresPersistenceBackend` using
      ``dsn`` (else env ``VZ_MEMORY_PG_DSN``) and ``namespace``.

    Keeping selection here means hosts only ever construct the abstract
    backend through one funnel; swapping the deploy-side storage layer
    (SQLite-on-PVC → managed Postgres) is an env flag, not a code edit.
    """

    chosen = (backend or os.environ.get("VZ_MEMORY_BACKEND") or "filesystem").strip().lower()
    if chosen in ("", "filesystem", "file", "fs"):
        if base_dir is None:
            raise ValueError(
                "resolve_persistence_backend: base_dir is required for the "
                "filesystem backend"
            )
        return FileSystemPersistenceBackend(base_dir=str(base_dir), max_versions=max_versions)
    if chosen in ("memory", "inmemory", "in_memory"):
        return InMemoryPersistenceBackend(max_versions=max_versions)
    if chosen in ("postgres", "postgresql", "pg"):
        resolved_dsn = dsn or os.environ.get("VZ_MEMORY_PG_DSN") or ""
        return PostgresPersistenceBackend(
            dsn=resolved_dsn,
            namespace=namespace,
            max_versions=max_versions,
        )
    raise ValueError(
        f"resolve_persistence_backend: unknown backend {chosen!r}. "
        "Expected one of: filesystem, memory, postgres."
    )


def serialize_checkpoint(checkpoint: object) -> bytes:
    """Serialize a frozen dataclass checkpoint to JSON bytes."""
    data: dict[str, Any] = {"_schema_version": SCHEMA_VERSION}
    if hasattr(checkpoint, "__dataclass_fields__"):
        for field in fields(checkpoint):  # type: ignore[arg-type]
            value = getattr(checkpoint, field.name)
            data[field.name] = _to_serializable(value)
    else:
        data["_raw"] = str(checkpoint)
    return json.dumps(data, ensure_ascii=False, indent=None).encode("utf-8")


def deserialize_checkpoint(data: bytes) -> dict[str, Any]:
    """Deserialize JSON bytes back to a plain dict.

    The caller is responsible for reconstructing the typed dataclass
    from the dict. Returns an empty dict if the schema version is
    incompatible.
    """
    parsed = json.loads(data.decode("utf-8"))
    stored_version = parsed.get("_schema_version", 0)
    if stored_version != SCHEMA_VERSION:
        return {}
    return parsed


def _to_serializable(value: object) -> object:
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if hasattr(value, "value") and hasattr(value, "name"):
        return value.value  # type: ignore[union-attr]
    if hasattr(value, "__dataclass_fields__"):
        return {
            field.name: _to_serializable(getattr(value, field.name))
            for field in fields(value)  # type: ignore[arg-type]
        }
    return str(value)
