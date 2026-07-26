"""Versioned provenance and integrity helpers for digital-ant artifacts.

The artifact payload is the machine-readable SSOT.  A sidecar manifest binds
it to the exact bytes, repository state, seed schedule, configuration and
optional model/reference inputs used by the run.  Dirty worktrees remain
exportable for internal review, but are explicitly not externally retainable.

``research/ant/05_ecology_p0_p1_p2_plan.md`` section 2.1 freezes two rules that
this module owns:

* every run records git SHA, dirty flag, config digest, dependency versions,
  device, training seed, layout seed and model fingerprint;
* every artifact uses a new filename and never overwrites an existing one --
  in particular never an existing ``BLOCK`` artifact.

The first rule is unconditional and lives in :func:`collect_ant_provenance`.

The second rule is **opt-in per lane**, not a property of this shared writer.
:func:`write_ant_artifact_bundle` keeps its historical replace-in-place default
because eleven drivers outside the ecology lane still write fixed output paths
and have no ``--force`` flag; flipping the default would break the very next
invocation of each of them, including the documented ``run_ant_pipeline.py``
resume entry point.  The ecology P0/P1/P2 drivers -- the lane the plan governs
-- pass ``overwrite=False`` explicitly and additionally call
:func:`ensure_artifact_writable` before spending any budget.  Migrating the
remaining drivers to run-id filenames plus an explicit force flag is a separate
convergence package (AGENTS.md section 8: high-ripple shared contracts ship
alone and last).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Envelope contract version.  ``v3`` adds the top-level
#: ``evidence_envelope_schema_version`` key and four provenance fields
#: (``requested_device``, ``effective_backend``, ``training_seeds``,
#: ``layout_seeds``).  Any further change to the envelope keys or to the
#: ``AntRunProvenance`` field set must bump this string --
#: ``test_evidence_provenance.py`` pins both so the bump cannot be forgotten.
ANT_ARTIFACT_SCHEMA_VERSION = "digital-ant-evidence.v3"
ANT_MANIFEST_SCHEMA_VERSION = "digital-ant-manifest.v2"

#: Keys the bundle envelope owns.  A payload that already carries one of them
#: would have its own value silently replaced, so the collision is refused.
ANT_ARTIFACT_RESERVED_KEYS = ("provenance", "evidence_envelope_schema_version")


class AntArtifactIntegrityError(RuntimeError):
    """An evidence artifact does not match its integrity manifest."""


class AntArtifactExistsError(RuntimeError):
    """Refusing to destroy an existing evidence artifact in place."""


@dataclass(frozen=True)
class ArtifactFileDigest:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class AntRunProvenance:
    git_sha: str
    git_branch: str
    working_tree_dirty: bool
    python_version: str
    platform: str
    dependency_versions: tuple[tuple[str, str], ...]
    seed_schedule: tuple[int, ...]
    config_digest: str
    model_fingerprint: str | None = None
    #: Device the operator asked for (``--device`` / ``VZ_TENSOR_DEVICE``).
    requested_device: str = "cpu"
    #: Backend the request actually resolves to.  A CPU run and a CUDA/MPS run
    #: are different experiments and must not be indistinguishable in the
    #: record.
    effective_backend: str = "pure:cpu"
    #: Training seeds and layout seeds stay separately recoverable; collapsing
    #: them into one schedule digest loses the namespace separation the plan
    #: requires between training / validation / held-out seeds.
    training_seeds: tuple[int, ...] = ()
    layout_seeds: tuple[int, ...] = ()

    @property
    def externally_retainable(self) -> bool:
        return bool(self.git_sha) and not self.working_tree_dirty


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _dependency_versions(names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    resolved: list[tuple[str, str]] = []
    for name in names:
        try:
            resolved.append((name, version(name)))
        except PackageNotFoundError:
            resolved.append((name, "not-installed"))
    return tuple(resolved)


def requested_tensor_device(explicit: str | None = None) -> str:
    """The device this process was asked to run on.

    ``train_ant_ecology.py`` exports ``VZ_TENSOR_DEVICE`` for an explicit
    ``--device cuda/mps`` run, so the environment is the authoritative record
    when a caller has no device of its own.
    """

    raw = explicit if explicit is not None else os.environ.get("VZ_TENSOR_DEVICE", "cpu")
    normalized = raw.strip().lower()
    if not normalized:
        raise ValueError("requested tensor device must be a non-empty string")
    return normalized


def resolve_effective_backend(requested_device: str) -> str:
    """Resolve the requested device to the backend that will actually run.

    Only an explicit ``cuda``/``mps`` request engages the torch runtime
    (``ant_runtime_replay_rollout_config``); everything else stays on the pure
    float64 CPU path.  An accelerator request that cannot be honoured raises
    here rather than being silently recorded as something it is not.
    """

    normalized = requested_tensor_device(requested_device)
    if not normalized.startswith(("cuda", "mps")):
        return f"pure:{normalized}"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            f"device {normalized!r} needs the torch runtime, which is not installed"
        ) from exc
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {normalized!r} requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"device {normalized!r} requested but MPS is unavailable")
    # Apple MPS has no float64 kernels; TorchBackend downgrades the dtype, and
    # the record must show it.
    dtype = "float32" if device.type == "mps" else "float64"
    return f"torch:{device.type}:{dtype}"


def stable_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: Any, *, overwrite: bool = True) -> None:
    """Atomically replace a JSON file after its complete bytes reach disk.

    ``overwrite=False`` turns an existing destination into a loud failure.  The
    default stays ``True`` because this primitive also serves rotating journals
    and resume markers, which must advance in place; artifact-level
    no-overwrite is enforced by :func:`write_ant_artifact_bundle`.
    """

    if not overwrite and path.exists():
        raise AntArtifactExistsError(
            f"refusing to overwrite an existing file: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def collect_ant_provenance(
    *,
    repo_root: Path,
    seeds: Sequence[int],
    config: Mapping[str, Any],
    model_fingerprint: str | None = None,
    device: str | None = None,
    training_seeds: Sequence[int] = (),
    layout_seeds: Sequence[int] = (),
) -> AntRunProvenance:
    requested_device = requested_tensor_device(device)
    return AntRunProvenance(
        git_sha=_git(repo_root, "rev-parse", "HEAD"),
        git_branch=_git(repo_root, "branch", "--show-current"),
        working_tree_dirty=bool(_git(repo_root, "status", "--porcelain")),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        dependency_versions=_dependency_versions(
            ("vz-contracts", "vz-substrate", "vz-runtime", "numpy", "torch")
        ),
        seed_schedule=tuple(int(seed) for seed in seeds),
        config_digest=stable_json_digest(dict(config)),
        model_fingerprint=model_fingerprint,
        requested_device=requested_device,
        effective_backend=resolve_effective_backend(requested_device),
        training_seeds=tuple(int(seed) for seed in training_seeds),
        layout_seeds=tuple(int(seed) for seed in layout_seeds),
    )


def file_digest(path: Path, *, relative_to: Path | None = None) -> ArtifactFileDigest:
    resolved = (
        path
        if path.is_absolute() or relative_to is None
        else relative_to / path
    )
    raw = resolved.read_bytes()
    display_path = (
        resolved.relative_to(relative_to)
        if relative_to is not None
        else resolved
    )
    return ArtifactFileDigest(
        path=str(display_path),
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )


def artifact_verdict_summary(path: Path) -> str:
    """Best-effort description of the verdict an existing artifact carries.

    Used only to enrich the refusal raised by :func:`write_ant_artifact_bundle`
    so an operator can see what a forced overwrite would destroy.  A file that
    cannot be parsed still produces a refusal -- the summary just says so.

    ``UnicodeDecodeError`` is caught explicitly: it is a ``ValueError``, not an
    ``OSError``, so a truncated or binary file on the artifact path would
    otherwise escape this helper and replace the "artifact is protected"
    refusal with a decoding traceback.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"unreadable: {type(exc).__name__}"
    if not isinstance(payload, dict):
        return "unreadable: JSON is not an object"
    for key in ("verdict", "promotion_verdict", "overall_verdict"):
        if key in payload:
            return f"{key}={payload[key]!r}"
    if "passed" in payload:
        return f"passed={payload['passed']!r}"
    return "verdict absent"


def ensure_artifact_writable(artifact_path: Path, *, overwrite: bool) -> None:
    """Refuse to destroy an existing artifact bundle at ``artifact_path``.

    Drivers call this *before* spending their budget so a name collision costs
    nothing; :func:`write_ant_artifact_bundle` calls it again at write time.
    """

    if overwrite:
        return
    if artifact_path.exists():
        raise AntArtifactExistsError(
            "refusing to overwrite an existing evidence artifact "
            f"{artifact_path} [{artifact_verdict_summary(artifact_path)}]; "
            "write a new run-id filename, or pass --overwrite to destroy it "
            "deliberately"
        )
    manifest_path = artifact_path.with_suffix(".manifest.json")
    if manifest_path.exists():
        raise AntArtifactExistsError(
            "refusing to overwrite an existing manifest "
            f"{manifest_path} whose artifact is missing; the previous run "
            "left an inconsistent bundle, resolve it before rewriting"
        )


def require_ant_artifact_envelope(
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    """Reject an artifact whose envelope predates the current contract.

    An artifact written before ``digital-ant-evidence.v3`` carries no
    ``requested_device`` / ``effective_backend`` / ``training_seeds`` /
    ``layout_seeds`` in its provenance block, so it cannot answer the questions
    plan section 2.1 requires of a formal artifact.  Reading it as if it could
    would silently certify a run whose device and seed namespaces are unknown,
    which is exactly the failure the version exists to prevent.
    """

    if "evidence_envelope_schema_version" not in payload:
        raise AntArtifactIntegrityError(
            f"artifact {path} predates the evidence envelope contract "
            f"(expected {ANT_ARTIFACT_SCHEMA_VERSION!r}); its provenance "
            "cannot report device or seed namespaces, so it must be rerun"
        )
    actual = payload["evidence_envelope_schema_version"]
    if actual != ANT_ARTIFACT_SCHEMA_VERSION:
        raise AntArtifactIntegrityError(
            f"unsupported evidence envelope schema at {path}: "
            f"expected={ANT_ARTIFACT_SCHEMA_VERSION!r}, actual={actual!r}"
        )


def write_ant_artifact_bundle(
    *,
    artifact_path: Path,
    payload: Mapping[str, Any],
    provenance: AntRunProvenance,
    input_paths: Sequence[Path] = (),
    repo_root: Path,
    overwrite: bool = True,
) -> Path:
    """Write ``payload`` plus its provenance sidecar manifest.

    ``overwrite`` defaults to ``True`` -- the historical behaviour every driver
    was written against.  Drivers that own a run-id filename scheme and a force
    flag (the ecology P0/P1/P2 lane) pass ``overwrite=False`` so plan section
    2.1 holds for them: a new filename per run, and never a destroyed ``BLOCK``
    artifact.  Do not flip this default until every driver has migrated; see
    the module docstring.
    """

    reserved = tuple(key for key in ANT_ARTIFACT_RESERVED_KEYS if key in payload)
    if reserved:
        raise AntArtifactIntegrityError(
            "artifact payload collides with envelope-owned keys: "
            f"{list(reserved)}"
        )
    manifest_path = artifact_path.with_suffix(".manifest.json")
    ensure_artifact_writable(artifact_path, overwrite=overwrite)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    # A payload that declares its own domain schema keeps it -- consumers such
    # as load_p1_prerequisite match on that exact string.  The envelope version
    # is then recorded under its own key rather than being silently dropped.
    enriched = {
        "schema_version": ANT_ARTIFACT_SCHEMA_VERSION,
        **dict(payload),
        "evidence_envelope_schema_version": ANT_ARTIFACT_SCHEMA_VERSION,
        "provenance": asdict(provenance),
    }
    atomic_write_json(artifact_path, enriched, overwrite=True)
    artifact = file_digest(artifact_path, relative_to=repo_root)
    inputs = tuple(file_digest(path, relative_to=repo_root) for path in input_paths)
    manifest = {
        "schema_version": ANT_MANIFEST_SCHEMA_VERSION,
        "artifact": asdict(artifact),
        "inputs": [asdict(item) for item in inputs],
        "provenance": asdict(provenance),
        "externally_retainable": provenance.externally_retainable,
    }
    atomic_write_json(manifest_path, manifest, overwrite=True)
    return manifest_path


def verify_ant_artifact_manifest(*, manifest_path: Path, repo_root: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ANT_MANIFEST_SCHEMA_VERSION:
        raise AntArtifactIntegrityError(
            f"unsupported manifest schema: {manifest.get('schema_version')!r}"
        )
    entries = (manifest["artifact"], *manifest.get("inputs", ()))
    for entry in entries:
        path = repo_root / entry["path"]
        if not path.is_file():
            raise AntArtifactIntegrityError(f"manifest file missing: {path}")
        actual = file_digest(path, relative_to=repo_root)
        if actual.sha256 != entry["sha256"] or actual.size_bytes != entry["size_bytes"]:
            raise AntArtifactIntegrityError(f"manifest digest mismatch: {path}")


__all__ = [
    "ANT_ARTIFACT_RESERVED_KEYS",
    "ANT_ARTIFACT_SCHEMA_VERSION",
    "ANT_MANIFEST_SCHEMA_VERSION",
    "AntArtifactExistsError",
    "AntArtifactIntegrityError",
    "AntRunProvenance",
    "ArtifactFileDigest",
    "artifact_verdict_summary",
    "atomic_write_json",
    "collect_ant_provenance",
    "ensure_artifact_writable",
    "file_digest",
    "require_ant_artifact_envelope",
    "requested_tensor_device",
    "resolve_effective_backend",
    "stable_json_digest",
    "verify_ant_artifact_manifest",
    "write_ant_artifact_bundle",
]
