from __future__ import annotations

import json
import os
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
