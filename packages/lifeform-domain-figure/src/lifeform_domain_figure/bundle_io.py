"""Persistence layer for :class:`FigureArtifactBundle`.

A :class:`FigureArtifactBundle` is a deeply-nested tree of frozen
dataclasses (:class:`HistoricalFigureProfile`,
:class:`FigureRetrievalIndex`, :class:`FigureCoverageMap`,
:class:`FigureStylePrior`, optional :class:`FigureSteeringSet`,
optional :class:`FigureLoRAArtifact`, plus the compiled
``DomainExperiencePackage`` + ``VitalsBootstrap``). The runtime
contract is byte-for-byte rollback (R15): a saved bundle, when
loaded, must produce the same :attr:`integrity_hash` byte-for-byte.

Layout on disk (one bundle = one directory)::

    <root_dir>/
        <figure_id>/
            <bundle_id>/
                bundle.pickle    # binary frozen-dataclass tree
                manifest.json    # typed audit surface + listing index

Format choice — pickle for the binary, typed JSON for the manifest:

* The bundle is a tree of frozen dataclasses (~20 distinct types,
  several hundred fields total). ``pickle`` round-trips it natively;
  the alternative (``dataclasses.asdict`` + a per-class reconstructor)
  costs ~150 lines of brittle schema code that must be kept in sync
  with the F2-F6 dataclasses on every change. The existing
  ``volvence_zero`` precedent for this pattern is
  :mod:`lifeform_evolution.snapshot_io` (see its module docstring for
  the same trade-off discussion); we deliberately adopt the same
  format so future readers see one approach to nested-dataclass
  persistence in the repo.
* The artifact is produced by *our own* bake pipeline and consumed by
  *our own* runtime / ops; we never load externally-supplied pickles.
* Every load runs :func:`compute_bundle_integrity_hash` and rejects
  any bundle whose recomputed hash differs from the saved
  :attr:`FigureArtifactBundle.integrity_hash`. That gives us
  fail-loud detection of pickle tampering or schema drift.
* The manifest is typed JSON (``schema_version`` / ``bundle_id`` /
  ``figure_id`` / ``profile_version`` / ``version_window`` /
  ``integrity_hash`` / ``created_at_iso`` / ``steering_present`` /
  ``lora_present``) so :func:`list_figure_bundles` and the rollback
  CLI can scan the filesystem without unpickling.

If external interoperability becomes a hard requirement (different
Python versions, non-Python consumers), we will add a
``bundle.json`` companion produced by ``dataclasses.asdict`` next to
``bundle.pickle`` and switch loaders by feature flag — the public
API in this module is designed to keep that swap transparent.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone

from lifeform_domain_figure.figure_artifact import (
    SCHEMA_VERSION as FIGURE_BUNDLE_SCHEMA_VERSION,
    FigureArtifactBundle,
    compute_bundle_integrity_hash,
)


MANIFEST_SCHEMA_VERSION = "vz-figure-bundle-manifest.v1"
PICKLE_MAGIC = b"VZ-FIGURE-BUNDLE\x00"
_BUNDLE_FILENAME = "bundle.pickle"
_MANIFEST_FILENAME = "manifest.json"

# bundle_id is content-addressed and looks like
# ``figure-bundle:einstein:<hash16>`` — the ``:`` separators are not
# valid path components on Windows. We map them to ``__`` for the
# on-disk directory name (deterministic, collision-free for the
# wheel's id format) and keep the original bundle_id in the manifest
# so consumers do not have to know about the path mapping.
_PATH_RESERVED_CHARS = (":",)
_PATH_REPLACEMENT = "__"


@dataclass(frozen=True)
class BundleManifest:
    """Typed mirror of a saved bundle's ``manifest.json`` file.

    The manifest is the human / ops-readable index over saved
    bundles. ``list_figure_bundles`` returns these without
    unpickling the bundle binaries.
    """

    manifest_schema_version: str
    bundle_schema_version: int
    figure_id: str
    bundle_id: str
    profile_version: str
    version_window: tuple[int, int]
    integrity_hash: str
    created_at_iso: str
    steering_present: bool
    lora_present: bool
    bundle_dir: pathlib.Path

    def __post_init__(self) -> None:
        if not self.bundle_id.strip():
            raise ValueError("BundleManifest.bundle_id must be non-empty")
        if not self.figure_id.strip():
            raise ValueError("BundleManifest.figure_id must be non-empty")
        if not self.integrity_hash.strip():
            raise ValueError(
                "BundleManifest.integrity_hash must be non-empty"
            )


def save_figure_bundle(
    bundle: FigureArtifactBundle,
    *,
    root_dir: str | pathlib.Path,
    created_at_iso: str | None = None,
) -> pathlib.Path:
    """Persist ``bundle`` under ``<root_dir>/<figure_id>/<bundle_id>/``.

    Returns the bundle directory path. Both ``bundle.pickle`` and
    ``manifest.json`` are written atomically (write-then-rename) so a
    crashed save never leaves a half-written manifest pointing at a
    missing pickle. Existing bundle directories are preserved; the
    bundle id is content-addressed (R15) so the same bundle reaches
    the same path regardless of when it was baked.
    """

    root = pathlib.Path(root_dir)
    bundle_dir = root / bundle.figure_id / _filename_safe(bundle.bundle_id)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = bundle_dir / _BUNDLE_FILENAME
    manifest_path = bundle_dir / _MANIFEST_FILENAME

    pickle_payload = PICKLE_MAGIC + pickle.dumps(bundle)
    _atomic_write_bytes(pickle_path, pickle_payload)

    manifest_payload = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "bundle_schema_version": FIGURE_BUNDLE_SCHEMA_VERSION,
        "figure_id": bundle.figure_id,
        "bundle_id": bundle.bundle_id,
        "profile_version": bundle.profile_version,
        "version_window": list(bundle.version_window),
        "integrity_hash": bundle.integrity_hash,
        "created_at_iso": created_at_iso or _now_iso(),
        "steering_present": bundle.steering is not None,
        "lora_present": bundle.lora is not None,
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
    )
    return bundle_dir.resolve()


def load_figure_bundle(
    *,
    root_dir: str | pathlib.Path,
    bundle_id: str,
    figure_id: str | None = None,
) -> FigureArtifactBundle:
    """Reload a bundle previously written by :func:`save_figure_bundle`.

    Raises :class:`FileNotFoundError` if ``bundle_id`` is unknown.
    Raises :class:`ValueError` (fail-loud, no-swallow) when the
    pickle magic header is missing OR the recomputed integrity hash
    differs from the saved manifest's hash. Both indicate either
    pickle tampering or a schema drift between save and load and
    must not be silently tolerated (R15 byte-level rollback contract).
    """

    if not bundle_id.strip():
        raise ValueError("load_figure_bundle: bundle_id must be non-empty")
    bundle_dir = _resolve_bundle_dir(
        root_dir=pathlib.Path(root_dir),
        bundle_id=bundle_id,
        figure_id=figure_id,
    )
    pickle_path = bundle_dir / _BUNDLE_FILENAME
    if not pickle_path.is_file():
        raise FileNotFoundError(
            f"load_figure_bundle: pickle missing at {pickle_path}"
        )

    with pickle_path.open("rb") as handle:
        header = handle.read(len(PICKLE_MAGIC))
        if header != PICKLE_MAGIC:
            raise ValueError(
                f"load_figure_bundle: file at {pickle_path} is not a "
                f"Volvence Zero figure bundle (missing magic header)."
            )
        bundle = pickle.load(handle)
    if not isinstance(bundle, FigureArtifactBundle):
        raise ValueError(
            f"load_figure_bundle: file at {pickle_path} did not "
            f"deserialize to FigureArtifactBundle "
            f"(got {type(bundle).__name__})."
        )
    if bundle.schema_version != FIGURE_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"load_figure_bundle: bundle at {pickle_path} has "
            f"schema_version={bundle.schema_version!r}; this build "
            f"only loads {FIGURE_BUNDLE_SCHEMA_VERSION!r}."
        )

    expected_hash = compute_bundle_integrity_hash(
        figure_id=bundle.figure_id,
        profile_version=bundle.profile_version,
        version_window=bundle.version_window,
        retrieval_integrity=bundle.retrieval_index.integrity_hash,
        coverage_integrity=bundle.coverage_map.integrity_hash,
        style_integrity=bundle.style_prior.integrity_hash,
        steering_integrity=_optional_artifact_hash(bundle.steering),
        lora_integrity=_optional_artifact_hash(bundle.lora),
    )
    if expected_hash != bundle.integrity_hash:
        raise ValueError(
            f"load_figure_bundle: integrity hash mismatch for bundle "
            f"{bundle.bundle_id!r} at {pickle_path}; saved="
            f"{bundle.integrity_hash!r} recomputed={expected_hash!r}. "
            f"This indicates pickle tampering or a schema drift "
            f"between save and load and violates R15."
        )
    return bundle


def list_figure_bundles(
    *,
    root_dir: str | pathlib.Path,
    figure_id: str | None = None,
) -> tuple[BundleManifest, ...]:
    """Return manifests for every saved bundle under ``root_dir``.

    Filters by ``figure_id`` when supplied. Manifests are read from
    ``manifest.json`` files only — the bundle pickles are not
    loaded, so this is fast and does not hold any objects in memory.
    """

    root = pathlib.Path(root_dir)
    if not root.is_dir():
        return ()

    if figure_id is not None and not figure_id.strip():
        raise ValueError("list_figure_bundles: figure_id must be non-empty when provided")

    figure_dirs: list[pathlib.Path]
    if figure_id is None:
        figure_dirs = [child for child in root.iterdir() if child.is_dir()]
    else:
        candidate = root / figure_id
        figure_dirs = [candidate] if candidate.is_dir() else []

    manifests: list[BundleManifest] = []
    for fd in figure_dirs:
        for bundle_dir in sorted(fd.iterdir()):
            if not bundle_dir.is_dir():
                continue
            manifest_path = bundle_dir / _MANIFEST_FILENAME
            if not manifest_path.is_file():
                continue
            manifest = _read_manifest(manifest_path, bundle_dir=bundle_dir)
            manifests.append(manifest)
    manifests.sort(key=lambda m: m.created_at_iso)
    return tuple(manifests)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_bundle_dir(
    *,
    root_dir: pathlib.Path,
    bundle_id: str,
    figure_id: str | None,
) -> pathlib.Path:
    """Locate a bundle directory by ``bundle_id`` (and optional ``figure_id``).

    The on-disk layout is ``<root>/<figure_id>/<bundle_id>``. When
    ``figure_id`` is supplied this is a direct path lookup; otherwise
    we scan the figure subdirectories and return the first match.
    """

    safe_id = _filename_safe(bundle_id)
    if figure_id is not None:
        return root_dir / figure_id / safe_id

    if not root_dir.is_dir():
        raise FileNotFoundError(
            f"load_figure_bundle: root_dir {root_dir} does not exist"
        )
    for child in root_dir.iterdir():
        if not child.is_dir():
            continue
        candidate = child / safe_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"load_figure_bundle: bundle_id {bundle_id!r} not found under "
        f"any figure subdirectory of {root_dir}"
    )


def _read_manifest(
    path: pathlib.Path,
    *,
    bundle_dir: pathlib.Path,
) -> BundleManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("manifest_schema_version")
    if schema != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"_read_manifest: manifest at {path} has "
            f"manifest_schema_version={schema!r}; this build only "
            f"loads {MANIFEST_SCHEMA_VERSION!r}."
        )
    version_window_raw = payload["version_window"]
    if (
        not isinstance(version_window_raw, list)
        or len(version_window_raw) != 2
    ):
        raise ValueError(
            f"_read_manifest: manifest at {path} has invalid "
            f"version_window={version_window_raw!r}"
        )
    return BundleManifest(
        manifest_schema_version=schema,
        bundle_schema_version=int(payload["bundle_schema_version"]),
        figure_id=str(payload["figure_id"]),
        bundle_id=str(payload["bundle_id"]),
        profile_version=str(payload["profile_version"]),
        version_window=(
            int(version_window_raw[0]),
            int(version_window_raw[1]),
        ),
        integrity_hash=str(payload["integrity_hash"]),
        created_at_iso=str(payload["created_at_iso"]),
        steering_present=bool(payload["steering_present"]),
        lora_present=bool(payload["lora_present"]),
        bundle_dir=bundle_dir.resolve(),
    )


def _filename_safe(value: str) -> str:
    """Map a content-addressed bundle id to a path-safe directory name.

    Bundle ids have the format ``figure-bundle:<figure>:<hash>`` —
    the ``:`` separators are invalid in Windows directory names.
    We rewrite each reserved character to :data:`_PATH_REPLACEMENT`;
    the substitution is deterministic and collision-free for the
    wheel's id format (no naturally-occurring ``__`` in
    ``figure-bundle`` ids).
    """

    out = value
    for ch in _PATH_RESERVED_CHARS:
        out = out.replace(ch, _PATH_REPLACEMENT)
    return out


def _atomic_write_bytes(target: pathlib.Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(target: pathlib.Path, payload: str) -> None:
    _atomic_write_bytes(target, payload.encode("utf-8"))


def _optional_artifact_hash(artifact: object | None) -> str:
    """Return the hash slot value for an optional steering / lora artifact.

    Mirrors the ``_artifact_integrity`` helper in
    :mod:`lifeform_domain_figure.compiler` — returns ``"absent"`` for
    ``None`` and the artifact's ``integrity_hash`` field otherwise.
    Fail-loud when a non-None artifact lacks ``integrity_hash``: the
    rollback contract requires every load-bearing slot to be hash
    addressed.
    """

    if artifact is None:
        return "absent"
    integrity = getattr(artifact, "integrity_hash", None)
    if not isinstance(integrity, str) or not integrity.strip():
        raise ValueError(
            "_optional_artifact_hash: bundle artifact slot has no "
            "usable integrity_hash; the slot must hold a frozen "
            "artifact whose identity is part of the bundle hash."
        )
    return integrity


def _now_iso() -> str:
    """Return a timezone-aware ISO 8601 timestamp at second resolution."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "PICKLE_MAGIC",
    "BundleManifest",
    "list_figure_bundles",
    "load_figure_bundle",
    "save_figure_bundle",
]
