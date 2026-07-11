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


@dataclass(frozen=True)
class RetainProvenanceRequirements:
    require_clean_tree: bool = True
    require_git_identity: bool = True
    require_dependencies: bool = True
    require_manifest_match: bool = True
    min_seed_count: int = 1
    require_substrate_fingerprint: bool = False
    require_artifact_digests: bool = False


@dataclass(frozen=True)
class ArtifactDigest:
    artifact_path: str
    sha256: str
    size_bytes: int


class RetainProvenanceError(ValueError):
    """Raised when an evidence bundle cannot support a retain-level claim."""


@dataclass(frozen=True)
class ClaimVerdict:
    claim_id: str
    status: str
    required_gate_ids: tuple[str, ...]
    supporting_artifacts: tuple[str, ...]
    evidence: tuple[tuple[str, float | str], ...]
    summary: str
    description: str


@dataclass(frozen=True)
class EvidenceBundle:
    bundle_id: str
    suite_kind: str
    manifest: Any
    provenance: PaperSuiteProvenance
    run_summaries: Any
    aggregate_metrics: Any
    pairwise_effects: tuple[Any, ...] = ()
    reference_artifacts: tuple[tuple[str, Any], ...] = ()
    blind_review_packet: Any | None = None
    human_ratings_aggregate: Any | None = None
    claim_verdicts: tuple[ClaimVerdict, ...] = ()
    description: str = ""


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


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_git_sha(value: str) -> bool:
    if len(value) not in (40, 64):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_retain_provenance(
    *,
    provenance: PaperSuiteProvenance,
    manifest: PaperSuiteManifest,
    requirements: RetainProvenanceRequirements = RetainProvenanceRequirements(),
    substrate_fingerprint_verified: bool | None = None,
    artifact_digests: tuple[ArtifactDigest, ...] = (),
) -> None:
    """Fail loudly when provenance is too weak for a retain-level claim.

    Wiring and SHADOW artifacts may still be exported without calling this
    validator. Any caller that labels evidence as externally retainable must
    call it before emitting the verdict or bundle.
    """

    violations: list[str] = []
    if requirements.require_clean_tree and provenance.working_tree_dirty:
        violations.append("working tree is dirty")
    if requirements.require_git_identity:
        if not _is_git_sha(provenance.git_sha):
            violations.append("git_sha is missing or invalid")
        if not provenance.git_branch or provenance.git_branch == "unavailable":
            violations.append("git_branch is missing")
    if requirements.require_dependencies:
        if not provenance.dependency_versions:
            violations.append("dependency_versions is empty")
        if not _is_sha256(provenance.dependency_digest):
            violations.append("dependency_digest is missing or invalid")
    if requirements.require_manifest_match:
        expected_manifest_hash = manifest_hash(manifest)
        if provenance.manifest_hash != expected_manifest_hash:
            violations.append("manifest_hash does not match the supplied manifest")
    if len(manifest.seed_schedule) < requirements.min_seed_count:
        violations.append(
            f"seed_schedule has {len(manifest.seed_schedule)} seeds; "
            f"requires at least {requirements.min_seed_count}"
        )
    if requirements.require_substrate_fingerprint and substrate_fingerprint_verified is not True:
        violations.append("substrate fingerprint is not verified")
    if requirements.require_artifact_digests:
        if not artifact_digests:
            violations.append("artifact digests are missing")
        for artifact in artifact_digests:
            if not artifact.artifact_path:
                violations.append("artifact path is empty")
            if not _is_sha256(artifact.sha256):
                violations.append(f"artifact sha256 is invalid: {artifact.artifact_path}")
            if artifact.size_bytes <= 0:
                violations.append(f"artifact size is not positive: {artifact.artifact_path}")
    if violations:
        raise RetainProvenanceError(
            "Evidence cannot support a retain-level claim: " + "; ".join(violations)
        )


def validate_evidence_bundle_for_external_use(
    *,
    bundle: EvidenceBundle,
    requirements: RetainProvenanceRequirements = RetainProvenanceRequirements(),
    substrate_fingerprint_verified: bool | None = None,
    artifact_digests: tuple[ArtifactDigest, ...] = (),
) -> None:
    retain_claims = tuple(
        verdict.claim_id for verdict in bundle.claim_verdicts if verdict.status == "retain"
    )
    if not retain_claims:
        return
    validate_retain_provenance(
        provenance=bundle.provenance,
        manifest=bundle.manifest,
        requirements=requirements,
        substrate_fingerprint_verified=substrate_fingerprint_verified,
        artifact_digests=artifact_digests,
    )


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
