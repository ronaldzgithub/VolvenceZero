"""Fail-loud local readiness and provenance helpers for Windows P1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
from typing import Iterable


WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth"})
REQUIRED_COMMANDS = (
    "lifeform-serve",
    "companion-ref-harness",
    "companion-camel-baseline",
)
P1_PORTS = (8000, 8001, 8002, 8003, 8004, 8005, 8500, 8600)


class P1ReadinessError(RuntimeError):
    """Raised when a paid P1 run would start from an invalid configuration."""


@dataclass(frozen=True)
class WeightFingerprint:
    model_id: str
    weights_root: str
    weights_sha256: str
    weight_file_count: int


@dataclass(frozen=True)
class P1RunManifest:
    schema_version: str
    git_sha: str
    git_worktree_clean: bool
    substrate: WeightFingerprint
    temporal_bootstrap_sha256: str
    regime_bootstrap_sha256: str
    temporal_runtime_backend: str
    temporal_ssl_backend: str
    internal_rl_backend: str
    cms_torch_backend: str
    user_sim_model: str
    perturn_model: str
    arc_model: str


def require_non_qwen_models(models: Iterable[tuple[str, str]]) -> None:
    for label, model in models:
        if not model.strip():
            raise P1ReadinessError(f"{label} model is empty")
        if "qwen" in model.casefold():
            raise P1ReadinessError(
                f"{label} model {model!r} is Qwen; P1 requires a cross-family model"
            )


def resolve_weights_root(model_id: str, explicit_path: str | None = None) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if explicit_path:
        candidates.append(pathlib.Path(explicit_path))
    model_path = pathlib.Path(model_id)
    if model_path.is_dir():
        candidates.append(model_path)

    hf_home = pathlib.Path(
        os.environ.get("HF_HOME", pathlib.Path.home() / ".cache" / "huggingface")
    )
    repo_cache = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    main_ref = repo_cache / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        if revision:
            candidates.append(repo_cache / "snapshots" / revision)
    snapshots = repo_cache / "snapshots"
    if snapshots.is_dir():
        snapshot_dirs = sorted(path for path in snapshots.iterdir() if path.is_dir())
        if len(snapshot_dirs) == 1:
            candidates.append(snapshot_dirs[0])

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir():
            return resolved
    raise P1ReadinessError(
        "cannot resolve local substrate weights; cache the model or set "
        "VZ_SUBSTRATE_WEIGHTS_PATH"
    )


def fingerprint_weights(model_id: str, root: pathlib.Path) -> WeightFingerprint:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in WEIGHT_SUFFIXES
    )
    if not files:
        raise P1ReadinessError(f"no model weight files found under {root}")

    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return WeightFingerprint(
        model_id=model_id,
        weights_root=str(root),
        weights_sha256=digest.hexdigest(),
        weight_file_count=len(files),
    )


def require_commands() -> None:
    missing = tuple(command for command in REQUIRED_COMMANDS if shutil.which(command) is None)
    if missing:
        raise P1ReadinessError(
            "required console commands are not installed: " + ", ".join(missing)
        )


def require_ports_free(host: str = "127.0.0.1") -> None:
    occupied: list[int] = []
    for port in P1_PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) == 0:
                occupied.append(port)
    if occupied:
        raise P1ReadinessError(
            "P1 service ports are already occupied: "
            + ", ".join(str(port) for port in occupied)
        )


def require_accelerator(device: str) -> str:
    """Fail-loud check that the requested substrate device actually exists.

    ``cuda`` is the default P1 profile (Linux/Windows NVIDIA); ``mps`` is the
    explicit Apple-silicon opt-in; ``cpu`` is allowed but recorded as such so
    the throughput expectation is honest. No silent fallback between devices.
    """

    try:
        import torch
    except ImportError as exc:
        raise P1ReadinessError(
            "torch is not installed; install the hf extras (install.sh / install.ps1)"
        ) from exc
    if device == "cuda" or device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise P1ReadinessError("torch.cuda.is_available() is false")
        return str(torch.cuda.get_device_name(0))
    if device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise P1ReadinessError("torch.backends.mps.is_available() is false")
        return "apple-mps"
    if device == "cpu":
        return "cpu"
    raise P1ReadinessError(
        f"unsupported substrate device {device!r} (expected cuda / cuda:N / mps / cpu)"
    )


def require_cuda() -> str:
    return require_accelerator("cuda")


def git_identity(repo_root: pathlib.Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, not bool(status.strip())


def build_run_manifest(
    *,
    repo_root: pathlib.Path,
    substrate: WeightFingerprint,
    user_sim_model: str,
    perturn_model: str,
    arc_model: str,
) -> P1RunManifest:
    from lifeform_domain_emogpt import require_companion_bootstraps

    bootstraps = require_companion_bootstraps()
    git_sha, clean = git_identity(repo_root)
    return P1RunManifest(
        schema_version="companion-p1-run-manifest.v1",
        git_sha=git_sha,
        git_worktree_clean=clean,
        substrate=substrate,
        temporal_bootstrap_sha256=bootstraps.temporal_sha256,
        regime_bootstrap_sha256=bootstraps.regime_sha256,
        temporal_runtime_backend=os.environ.get("VZ_TEMPORAL_RUNTIME_BACKEND", "disabled"),
        temporal_ssl_backend=os.environ.get("VZ_TEMPORAL_SSL_BACKEND", "disabled"),
        internal_rl_backend=os.environ.get("VZ_INTERNAL_RL_BACKEND", "disabled"),
        cms_torch_backend=os.environ.get("VZ_CMS_TORCH_BACKEND", "disabled"),
        user_sim_model=user_sim_model,
        perturn_model=perturn_model,
        arc_model=arc_model,
    )


def write_run_manifest(manifest: P1RunManifest, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_track_fingerprints(
    fingerprint: WeightFingerprint,
    output_dir: pathlib.Path,
    tracks: Iterable[str],
) -> None:
    for track in tracks:
        path = output_dir / track / "substrate_fingerprint.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "track": track,
                    "substrate_model_id": fingerprint.model_id,
                    "weights_sha256": fingerprint.weights_sha256,
                    "weights_root": fingerprint.weights_root,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
