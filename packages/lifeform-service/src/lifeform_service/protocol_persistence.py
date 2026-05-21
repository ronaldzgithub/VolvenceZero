"""Disk persistence for approved :class:`BehaviorProtocol` instances.

R8 SSOT
-------

This module is the **single owner** of the on-disk JSON
representation of approved protocols. It does not own:

* The in-memory :class:`volvence_zero.protocol_runtime.ProtocolRegistry`
  (that's owned by :class:`lifeform_service.protocol_uptake.ProtocolUptakeService`).
* The serialisation contract itself (that's owned by
  :func:`lifeform_protocol_runtime.protocol_to_payload` /
  :func:`protocol_from_payload`; we just call them).
* The HTTP surface that exposes the library
  (that's owned by ``protocol_routes.py``).

The store is intentionally small and synchronous. ``ProtocolUptakeService``
already serialises mutations through an ``asyncio.Lock`` so we do
not need our own lock — the constraint is "one tenant per directory",
which matches the single-process service model.

Layout on disk
--------------

::

    <approved_dir>/
        <safe_protocol_id_1>.json
        <safe_protocol_id_2>.json

Each ``.json`` is a payload from
:func:`lifeform_protocol_runtime.protocol_to_payload` — full lossless
round-trip with the schema_version field. File names are derived
from the protocol_id via :func:`_filename_for_protocol_id` so a
protocol with id ``growth_advisor:cheng-laoshi`` lands at
``growth_advisor_cheng-laoshi.json`` (path-safe, still recognisable).

Atomicity
---------

Writes go through a ``.tmp`` neighbour + ``os.replace`` so a
crash mid-write cannot leave a half-written JSON in place. Reads
use ``json.loads`` which fails loudly on corrupted content; the
caller decides whether to skip or hard-fail.

Why we don't index ``_active.json``
-----------------------------------

Per the product decision the user made on 2026-05-22 with this
packet: persistence stores the library; in-memory registry stores
the active set; **the active set is not persisted across restarts**.
The user explicitly wants to re-pick from the library after each
restart through the chat-browser UI. If that decision is ever
reversed, add a small companion ``_active.json`` next to the
library; do NOT move active-set tracking into individual protocol
files (that would conflate two concerns).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from lifeform_protocol_runtime import (
    ProtocolPayloadSchemaError,
    protocol_from_payload,
    protocol_to_payload,
)
from volvence_zero.behavior_protocol import BehaviorProtocol


_LOG = logging.getLogger("lifeform_service.protocol_persistence")


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _filename_for_protocol_id(protocol_id: str) -> str:
    """Map ``protocol_id`` → a path-safe filename stem.

    Replaces every character outside ``[A-Za-z0-9._-]`` with ``_``
    so we can carry namespacing colons (``growth_advisor:cheng-laoshi``)
    and still land on a file path that works on Windows + POSIX +
    container volume mounts. The transformation is **not** required
    to round-trip back to the original id — the id lives inside the
    JSON payload, the filename is just a stable, human-readable
    on-disk handle.
    """

    cleaned = _FILENAME_SAFE_RE.sub("_", protocol_id.strip())
    if not cleaned:
        raise ValueError(
            f"_filename_for_protocol_id: protocol_id {protocol_id!r} "
            "yields empty filename after sanitisation"
        )
    return cleaned


class ProtocolPersistenceStore:
    """Single-directory persistence store for approved protocols.

    Pass an ``approved_dir`` at construction. The directory is
    created if missing. All public methods are synchronous; the
    caller (:class:`ProtocolUptakeService`) serialises mutations
    through its own asyncio lock.

    Public surface:

    * :meth:`list_all` — read every protocol on disk; returns a
      tuple sorted by ``protocol_id``.
    * :meth:`read` — read one protocol by id; raises ``KeyError``
      if the file doesn't exist.
    * :meth:`write` — atomically write one protocol to disk.
    * :meth:`delete` — remove the JSON file; returns ``True`` if
      it existed.
    * :meth:`exists` — boolean check by protocol_id.
    """

    def __init__(self, approved_dir: Path | str) -> None:
        self._approved_dir = Path(approved_dir).resolve()
        self._approved_dir.mkdir(parents=True, exist_ok=True)

    @property
    def approved_dir(self) -> Path:
        return self._approved_dir

    def _path_for(self, protocol_id: str) -> Path:
        return self._approved_dir / f"{_filename_for_protocol_id(protocol_id)}.json"

    def exists(self, protocol_id: str) -> bool:
        return self._path_for(protocol_id).is_file()

    def read(self, protocol_id: str) -> BehaviorProtocol:
        path = self._path_for(protocol_id)
        if not path.is_file():
            raise KeyError(
                f"ProtocolPersistenceStore.read: no persisted protocol "
                f"with id {protocol_id!r} (looked at {path})"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"ProtocolPersistenceStore.read: protocol file {path} "
                f"contains invalid JSON: {exc}"
            ) from exc
        protocol = protocol_from_payload(payload)
        if protocol.protocol_id != protocol_id:
            # The id in the file body is canonical; the filename is
            # just a path-safe handle. Mismatch usually means someone
            # renamed the file by hand. Fail loud so they notice.
            raise ValueError(
                f"ProtocolPersistenceStore.read: file {path.name!r} "
                f"contains protocol_id={protocol.protocol_id!r} (expected "
                f"{protocol_id!r}); rename the file or edit the JSON to "
                "match before reloading."
            )
        return protocol

    def write(self, protocol: BehaviorProtocol) -> Path:
        """Atomically write ``protocol`` to ``approved_dir``.

        Returns the path of the written file. Overwrites any
        existing JSON for the same ``protocol_id``.
        """

        path = self._path_for(protocol.protocol_id)
        payload = protocol_to_payload(protocol)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        tmp_path.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
        _LOG.info(
            "persisted protocol %s -> %s (%d bytes)",
            protocol.protocol_id, path, path.stat().st_size,
        )
        return path

    def delete(self, protocol_id: str) -> bool:
        """Delete ``<approved_dir>/<id>.json``; return True if removed."""

        path = self._path_for(protocol_id)
        if not path.is_file():
            return False
        path.unlink()
        _LOG.info("deleted persisted protocol %s (%s)", protocol_id, path)
        return True

    def list_all(self) -> tuple[BehaviorProtocol, ...]:
        """Read every ``.json`` file in ``approved_dir``.

        Returns the tuple sorted by ``protocol_id``. Files that
        fail to deserialise (bad JSON, schema_version mismatch,
        missing required fields) are skipped with a warning rather
        than aborting the whole load — one bad file should not
        block the rest of the library from showing up in the UI.
        """

        loaded: list[BehaviorProtocol] = []
        for path in sorted(self._approved_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                loaded.append(protocol_from_payload(payload))
            except (json.JSONDecodeError, ProtocolPayloadSchemaError, ValueError, KeyError) as exc:
                _LOG.warning(
                    "skipping persisted protocol file %s: %s (%s)",
                    path.name, type(exc).__name__, exc,
                )
                continue
        return tuple(sorted(loaded, key=lambda p: p.protocol_id))


__all__ = [
    "ProtocolPersistenceStore",
]
