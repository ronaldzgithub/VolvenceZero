from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping


@dataclass(frozen=True)
class PaperMetricSpec:
    metric_name: str
    role: str
    direction: str
    description: str


@dataclass(frozen=True)
class PaperProfileSpec:
    profile_label: str
    role: str
    description: str


@dataclass(frozen=True)
class PaperSuiteManifest:
    suite_id: str
    suite_kind: str
    suite_tier: str
    version: int
    baseline_label: str
    repeat_count: int
    seed_schedule: tuple[int, ...]
    profiles: tuple[PaperProfileSpec, ...]
    primary_metrics: tuple[PaperMetricSpec, ...]
    secondary_metrics: tuple[PaperMetricSpec, ...]
    case_groups: tuple[tuple[str, tuple[str, ...]], ...]
    artifact_expectations: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class PaperSuiteProvenance:
    git_sha: str
    git_branch: str
    working_tree_dirty: bool
    python_version: str
    platform: str
    dependency_versions: tuple[str, ...]
    dependency_digest: str
    manifest_hash: str
    runtime_descriptor: tuple[tuple[str, str], ...]
    description: str


def _json_normalize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_normalize(inner) for key, inner in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_normalize(inner) for key, inner in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_normalize(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    return value


def manifest_hash(manifest: PaperSuiteManifest) -> str:
    payload = json.dumps(_json_normalize(manifest), sort_keys=True, indent=2)
    return sha256(payload.encode("utf-8")).hexdigest()


def _run_git_command(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _dependency_versions() -> tuple[str, ...]:
    distributions = sorted(
        (
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        ),
        key=str.lower,
    )
    return tuple(distributions)


def collect_paper_suite_provenance(
    *,
    manifest: PaperSuiteManifest,
    repo_root: str | Path,
    runtime_descriptor: Mapping[str, str],
) -> PaperSuiteProvenance:
    root = Path(repo_root)
    try:
        git_sha = _run_git_command(root, "rev-parse", "HEAD")
        git_branch = _run_git_command(root, "rev-parse", "--abbrev-ref", "HEAD")
        working_tree_dirty = bool(_run_git_command(root, "status", "--short"))
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_sha = "unavailable"
        git_branch = "unavailable"
        working_tree_dirty = False
    dependencies = _dependency_versions()
    dependency_digest = sha256("\n".join(dependencies).encode("utf-8")).hexdigest()
    descriptor = tuple((key, runtime_descriptor[key]) for key in sorted(runtime_descriptor))
    return PaperSuiteProvenance(
        git_sha=git_sha,
        git_branch=git_branch,
        working_tree_dirty=working_tree_dirty,
        python_version=sys.version.replace("\n", " "),
        platform=platform.platform(),
        dependency_versions=dependencies,
        dependency_digest=dependency_digest,
        manifest_hash=manifest_hash(manifest),
        runtime_descriptor=descriptor,
        description=(
            f"Paper suite provenance for {manifest.suite_id}@v{manifest.version} "
            f"on {platform.system()} with python={platform.python_version()}."
        ),
    )


def export_json_artifact(
    *,
    payload: Any,
    output_path: str | Path,
) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(_json_normalize(payload), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return target_path
