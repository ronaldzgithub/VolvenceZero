"""Content-addressed trajectory logging (coding-lab Packet 0).

Every episode writes an append-only JSONL event stream. Because the
hand may be a non-deterministic API model, the trajectory log — not a
rerun — is the replay substrate for every downstream packet (junction
corpora, forecast settlement audits). Files are hashed on close and the
hash is the episode's identity in run manifests.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrajectoryRecord:
    """Closed trajectory file reference."""

    path: pathlib.Path
    sha256: str
    event_count: int


class TrajectoryWriter:
    """Append-only JSONL writer for one episode."""

    def __init__(self, path: pathlib.Path) -> None:
        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            raise FileExistsError(f"trajectory file already exists: {self._path!s}")
        self._handle = self._path.open("w", encoding="utf-8")
        self._event_count = 0
        self._closed = False

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("trajectory writer is closed")
        record = {
            "event_index": self._event_count,
            "event_type": event_type,
            "monotonic_seconds": time.monotonic(),
            "payload": payload,
        }
        self._handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._event_count += 1

    def close(self) -> TrajectoryRecord:
        if self._closed:
            raise RuntimeError("trajectory writer is already closed")
        self._handle.flush()
        self._handle.close()
        self._closed = True
        digest = hashlib.sha256(self._path.read_bytes()).hexdigest()
        return TrajectoryRecord(path=self._path, sha256=digest, event_count=self._event_count)


def read_trajectory(path: pathlib.Path) -> tuple[dict[str, Any], ...]:
    """Load a trajectory JSONL file back into event dicts."""

    events: list[dict[str, Any]] = []
    with pathlib.Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                events.append(json.loads(stripped))
    return tuple(events)


__all__ = ["TrajectoryRecord", "TrajectoryWriter", "read_trajectory"]
