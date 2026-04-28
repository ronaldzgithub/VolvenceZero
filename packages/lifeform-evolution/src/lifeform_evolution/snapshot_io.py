"""Persistence for trained metacontroller snapshots.

The multi-round learning loop produces a ``MetacontrollerParameterSnapshot``
per round; ``MultiRoundLearningLoopReport.best_round()`` selects the
"healthy" one. This module turns that selection into a reusable artifact:
write the snapshot to disk, and let downstream runs load it back as a
``temporal_bootstrap`` for ``Brain`` / ``Lifeform``.

Format choice: pickle.

* The snapshot is a tree of frozen dataclasses, tuples, and primitives.
  ``pickle`` round-trips it natively without us having to maintain a
  reconstruction schema for ~40 fields and several optional dataclass types.
* The artifact is produced by *our own* training pipeline and consumed by
  *our own* runtime; we never load externally-supplied pickles. The risks
  pickle is normally faulted for (untrusted code execution on load) do
  not apply on this path.
* The file format is versioned (header magic + ``schema_version`` field) so
  if we ever switch to JSON we can detect old artifacts and reject them.

If external interoperability becomes a requirement (different Python
versions, non-Python consumers), we will switch the encoder to a
``dataclasses.asdict`` + JSON path. The public API in this file is
designed to keep that swap transparent to callers.
"""

from __future__ import annotations

import pathlib
import pickle
from dataclasses import dataclass

from volvence_zero.temporal import MetacontrollerParameterSnapshot


# ``MAGIC`` lets us reject random pickle files. ``SCHEMA_VERSION`` lets us
# evolve the layout (e.g. add metadata) without breaking old loaders.
MAGIC = b"VZ-METASNAP\x00"
SCHEMA_VERSION = "vz-metasnap.v1"


@dataclass(frozen=True)
class SnapshotArtifact:
    """A pickled snapshot envelope, with trace metadata.

    ``metadata`` is product-side context the lifeform layer wants to keep
    next to the snapshot \u2014 things like "this came from round 2 of a
    4-round loop", "trained on these scenario IDs", or "evaluation
    distance from baseline at the time of save = 0.464". It is never
    consumed by the kernel; only by lifeform-evolution dashboards.
    """

    schema_version: str
    snapshot: MetacontrollerParameterSnapshot
    metadata: dict[str, object]


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_snapshot(
    snapshot: MetacontrollerParameterSnapshot,
    path: str | pathlib.Path,
    *,
    metadata: dict[str, object] | None = None,
) -> pathlib.Path:
    """Write a metacontroller snapshot to disk.

    Returns the absolute path that was written. Parent directories are
    created on demand.
    """
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = SnapshotArtifact(
        schema_version=SCHEMA_VERSION,
        snapshot=snapshot,
        metadata=dict(metadata or {}),
    )
    with out.open("wb") as handle:
        handle.write(MAGIC)
        pickle.dump(artifact, handle)
    return out.resolve()


def load_snapshot(
    path: str | pathlib.Path,
) -> SnapshotArtifact:
    """Read a metacontroller snapshot artifact written by ``save_snapshot``."""
    in_path = pathlib.Path(path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Snapshot artifact not found: {in_path}")
    with in_path.open("rb") as handle:
        header = handle.read(len(MAGIC))
        if header != MAGIC:
            raise ValueError(
                f"File at {in_path} is not a Volvence Zero metacontroller snapshot "
                f"(missing magic header)."
            )
        artifact = pickle.load(handle)
    if not isinstance(artifact, SnapshotArtifact):
        raise ValueError(
            f"File at {in_path} did not deserialize to SnapshotArtifact "
            f"(got {type(artifact).__name__})."
        )
    if artifact.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Snapshot at {in_path} has schema_version={artifact.schema_version!r}; "
            f"this build only loads {SCHEMA_VERSION!r}."
        )
    return artifact


def load_snapshot_only(
    path: str | pathlib.Path,
) -> MetacontrollerParameterSnapshot:
    """Convenience: ``load_snapshot(path).snapshot``."""
    return load_snapshot(path).snapshot
