"""Content-addressable storage for raw bytes and cleaned text.

Layout under the configured ``root`` directory::

    root/
      raw/
        {sha256}/
          bytes              # exact bytes the parser saw
          sidecar.json       # {source_url, content_type, byte_len, ts_ms}
      cleaned/
        {raw_sha256}/
          v{N}/
            text.txt         # CleanedDocument.text
            cleaning_log.json# CleaningOpRecord tuple + parser_version

Why this shape:

* Raw bytes are content-addressable by sha256 so the same source URL
  fetched twice never writes twice (idempotent ``put_raw``).
* Cleaned text is keyed by ``raw_sha256`` plus ``v{N}`` so a cleaner
  upgrade adds a new directory next to the old one — debt #28 calls
  this out as a hard requirement (re-clean must produce a new version
  without destroying the previous one).
* All sidecars are JSON so they stay human-inspectable; the cleaning
  log is the only persistent surface for "which ops fired with what
  versions on what input".

The store is the only place ``data/raw/`` and ``data/cleaned/``
directory creation lives. Callers pass an explicit ``root`` path; the
default ``packages/lifeform-domain-figure/data/`` is a convention,
not a global, so tests use ``tmp_path`` fixtures.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from lifeform_domain_figure.cleaning.raw_document import (
    CleanedDocument,
    CleaningOp,
    CleaningOpRecord,
)


@dataclass(frozen=True)
class RawSidecar:
    """Sidecar metadata persisted alongside raw bytes."""

    source_url: str
    content_type: str
    byte_len: int
    stored_at_ms: int


class CleaningStore:
    """Content-addressable filesystem store for cleaning artefacts."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise TypeError(
                f"CleaningStore.root must be a Path; got {type(root).__name__}"
            )
        self._root = root
        self._raw_root = root / "raw"
        self._cleaned_root = root / "cleaned"

    @property
    def root(self) -> Path:
        return self._root

    def put_raw(
        self,
        data: bytes,
        *,
        source_url: str,
        content_type: str,
    ) -> str:
        """Idempotently store ``data``. Return the raw sha256."""

        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(
                f"CleaningStore.put_raw: data must be bytes; got {type(data).__name__}"
            )
        if not source_url.strip():
            raise ValueError("CleaningStore.put_raw: source_url must be non-empty")
        if not content_type.strip():
            raise ValueError("CleaningStore.put_raw: content_type must be non-empty")
        raw_sha256 = hashlib.sha256(bytes(data)).hexdigest()
        raw_dir = self._raw_root / raw_sha256
        raw_dir.mkdir(parents=True, exist_ok=True)
        bytes_path = raw_dir / "bytes"
        if not bytes_path.exists():
            bytes_path.write_bytes(bytes(data))
        sidecar_path = raw_dir / "sidecar.json"
        if not sidecar_path.exists():
            sidecar = RawSidecar(
                source_url=source_url,
                content_type=content_type,
                byte_len=len(data),
                stored_at_ms=int(time.time() * 1000.0),
            )
            sidecar_path.write_text(
                json.dumps(
                    {
                        "source_url": sidecar.source_url,
                        "content_type": sidecar.content_type,
                        "byte_len": sidecar.byte_len,
                        "stored_at_ms": sidecar.stored_at_ms,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return raw_sha256

    def get_raw(self, raw_sha256: str) -> tuple[bytes, RawSidecar]:
        """Read raw bytes and sidecar for ``raw_sha256``.

        Raises ``FileNotFoundError`` (with context) when the entry
        does not exist.
        """

        raw_dir = self._raw_root / raw_sha256
        bytes_path = raw_dir / "bytes"
        sidecar_path = raw_dir / "sidecar.json"
        if not bytes_path.exists() or not sidecar_path.exists():
            raise FileNotFoundError(
                f"CleaningStore.get_raw: no raw entry for sha={raw_sha256!r} "
                f"under {self._raw_root}"
            )
        data = bytes_path.read_bytes()
        sidecar_raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar = RawSidecar(
            source_url=str(sidecar_raw["source_url"]),
            content_type=str(sidecar_raw["content_type"]),
            byte_len=int(sidecar_raw["byte_len"]),
            stored_at_ms=int(sidecar_raw["stored_at_ms"]),
        )
        return data, sidecar

    def list_raw(self) -> Iterator[str]:
        """Yield all stored raw sha256 keys in deterministic order."""

        if not self._raw_root.exists():
            return
        for entry in sorted(self._raw_root.iterdir()):
            if entry.is_dir() and (entry / "bytes").exists():
                yield entry.name

    def put_cleaned(self, cleaned: CleanedDocument) -> Path:
        """Persist ``cleaned`` under its ``raw_sha256`` / version dir.

        Returns the directory containing ``text.txt`` and
        ``cleaning_log.json``. Re-writing an existing version overwrites
        in place (idempotent for identical results; intentional for
        cleaner-determinism debugging).
        """

        cleaned_dir = (
            self._cleaned_root
            / cleaned.raw_sha256
            / f"v{cleaned.cleaner_pipeline_version}"
        )
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        (cleaned_dir / "text.txt").write_text(cleaned.text, encoding="utf-8")
        log_payload = {
            "raw_sha256": cleaned.raw_sha256,
            "cleaner_pipeline_version": cleaned.cleaner_pipeline_version,
            "parser_version": cleaned.parser_version,
            "ops": [
                {
                    "op": record.op.value,
                    "op_version": record.op_version,
                    "chars_before": record.chars_before,
                    "chars_after": record.chars_after,
                }
                for record in cleaned.cleaning_log
            ],
        }
        (cleaned_dir / "cleaning_log.json").write_text(
            json.dumps(log_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return cleaned_dir

    def get_cleaned(
        self, raw_sha256: str, pipeline_version: int
    ) -> CleanedDocument | None:
        """Return the persisted cleaned document, or ``None`` if absent."""

        cleaned_dir = self._cleaned_root / raw_sha256 / f"v{pipeline_version}"
        text_path = cleaned_dir / "text.txt"
        log_path = cleaned_dir / "cleaning_log.json"
        if not text_path.exists() or not log_path.exists():
            return None
        text = text_path.read_text(encoding="utf-8")
        log_payload = json.loads(log_path.read_text(encoding="utf-8"))
        records = tuple(
            CleaningOpRecord(
                op=CleaningOp(entry["op"]),
                op_version=str(entry["op_version"]),
                chars_before=int(entry["chars_before"]),
                chars_after=int(entry["chars_after"]),
            )
            for entry in log_payload["ops"]
        )
        return CleanedDocument(
            text=text,
            raw_sha256=str(log_payload["raw_sha256"]),
            cleaner_pipeline_version=int(log_payload["cleaner_pipeline_version"]),
            cleaning_log=records,
            parser_version=str(log_payload["parser_version"]),
        )

    def list_cleaned_versions(self, raw_sha256: str) -> tuple[int, ...]:
        """Return the set of pipeline versions persisted for ``raw_sha256``."""

        sha_dir = self._cleaned_root / raw_sha256
        if not sha_dir.exists():
            return ()
        versions: list[int] = []
        for entry in sha_dir.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            if not name.startswith("v"):
                continue
            try:
                versions.append(int(name[1:]))
            except ValueError:
                continue
        return tuple(sorted(versions))


__all__ = [
    "CleaningStore",
    "RawSidecar",
]
