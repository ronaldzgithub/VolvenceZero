"""Persistence and promotion loading for learned ecology checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from volvence_zero.agent import (
    AgentLearningCheckpointArchive,
    decode_agent_learning_checkpoint_archive,
    encode_agent_learning_checkpoint_archive,
)

from volvence_ant.evidence.provenance import (
    AntArtifactIntegrityError,
    collect_ant_provenance,
    file_digest,
    verify_ant_artifact_manifest,
    write_ant_artifact_bundle,
)
from volvence_ant.experiments.ecology_curriculum import (
    EcologyCheckpointCandidate,
    EcologyCurriculumConfig,
)
from volvence_ant.runtime import AntLearningCheckpoint, AntSenseSchema


ECOLOGY_CHECKPOINT_BUNDLE_KIND = "digital-ant-ecology-checkpoint.v1"


@dataclass(frozen=True)
class LoadedEcologyCheckpoint:
    checkpoints: tuple[AntLearningCheckpoint, ...]
    fingerprint: str
    verdict: str
    config: EcologyCurriculumConfig
    report_path: str


def ecology_checkpoint_compatibility(
    config: EcologyCurriculumConfig,
) -> tuple[tuple[str, str], ...]:
    return (
        ("artifact_kind", ECOLOGY_CHECKPOINT_BUNDLE_KIND),
        ("sense_schema", AntSenseSchema.ECOLOGY_V2.value),
        ("latent_dim", str(config.temporal_latent_dim)),
        ("n_ants", str(config.n_ants)),
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _aggregate_fingerprint(
    checkpoints: tuple[AntLearningCheckpoint, ...],
) -> str:
    return hashlib.sha256(repr(tuple(item.fingerprint for item in checkpoints)).encode("utf-8")).hexdigest()


def write_ecology_checkpoint_bundle(
    *,
    candidate: EcologyCheckpointCandidate,
    archive_path: Path,
    report_path: Path,
    repo_root: Path,
) -> Path:
    config = candidate.report.config
    archive_bytes = encode_agent_learning_checkpoint_archive(
        candidate.checkpoints,
        compatibility=ecology_checkpoint_compatibility(config),
    )
    _atomic_write_bytes(archive_path, archive_bytes)
    archive_digest = file_digest(archive_path, relative_to=repo_root)
    provenance = collect_ant_provenance(
        repo_root=repo_root,
        seeds=config.heldout_seeds,
        config=asdict(config),
        model_fingerprint=_aggregate_fingerprint(candidate.checkpoints),
    )
    payload = {
        "artifact_kind": ECOLOGY_CHECKPOINT_BUNDLE_KIND,
        "promotion_verdict": candidate.report.verdict,
        "checkpoint_archive": asdict(archive_digest),
        "checkpoint_fingerprint": _aggregate_fingerprint(candidate.checkpoints),
        "report": candidate.report.to_dict(),
    }
    return write_ant_artifact_bundle(
        artifact_path=report_path,
        payload=payload,
        provenance=provenance,
        input_paths=(archive_path,),
        repo_root=repo_root,
    )


def _config_from_report(payload: dict[str, object]) -> EcologyCurriculumConfig:
    raw_report = payload.get("report")
    if not isinstance(raw_report, dict):
        raise AntArtifactIntegrityError("ecology checkpoint report is missing report object")
    raw_config = raw_report.get("config")
    if not isinstance(raw_config, dict):
        raise AntArtifactIntegrityError("ecology checkpoint report is missing config object")
    return EcologyCurriculumConfig(
        n_ants=int(raw_config["n_ants"]),
        temporal_latent_dim=int(raw_config["temporal_latent_dim"]),
        stage_rounds=int(raw_config["stage_rounds"]),
        stage_episodes=int(raw_config["stage_episodes"]),
        heldout_rounds=int(raw_config["heldout_rounds"]),
        heldout_seeds=tuple(int(value) for value in raw_config["heldout_seeds"]),
        seed=int(raw_config["seed"]),
    )


def load_promoted_ecology_checkpoint(
    *,
    report_path: Path,
    repo_root: Path,
) -> LoadedEcologyCheckpoint:
    manifest_path = report_path.with_suffix(".manifest.json")
    verify_ant_artifact_manifest(
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AntArtifactIntegrityError("ecology checkpoint report must be an object")
    if payload.get("artifact_kind") != ECOLOGY_CHECKPOINT_BUNDLE_KIND:
        raise AntArtifactIntegrityError(f"unexpected ecology artifact kind: {payload.get('artifact_kind')!r}")
    verdict = str(payload.get("promotion_verdict", "BLOCK")).upper()
    if verdict != "PASS":
        raise AntArtifactIntegrityError(f"ecology checkpoint is not promoted: verdict={verdict}")
    raw_archive = payload.get("checkpoint_archive")
    if not isinstance(raw_archive, dict) or not raw_archive.get("path"):
        raise AntArtifactIntegrityError("ecology checkpoint archive reference missing")
    archive_path = repo_root / str(raw_archive["path"])
    actual_digest = file_digest(archive_path, relative_to=repo_root)
    if actual_digest.sha256 != str(raw_archive.get("sha256", "")) or actual_digest.size_bytes != int(
        raw_archive.get("size_bytes", -1)
    ):
        raise AntArtifactIntegrityError("ecology checkpoint archive digest mismatch")
    config = _config_from_report(payload)
    archive: AgentLearningCheckpointArchive = decode_agent_learning_checkpoint_archive(
        archive_path.read_bytes(),
        trusted_local_artifact=True,
        expected_compatibility=ecology_checkpoint_compatibility(config),
    )
    if len(archive.checkpoints) != config.n_ants:
        raise AntArtifactIntegrityError("ecology checkpoint count does not match configured ant count")
    fingerprint = _aggregate_fingerprint(archive.checkpoints)
    if fingerprint != str(payload.get("checkpoint_fingerprint", "")):
        raise AntArtifactIntegrityError("ecology aggregate checkpoint fingerprint mismatch")
    return LoadedEcologyCheckpoint(
        checkpoints=archive.checkpoints,
        fingerprint=fingerprint,
        verdict=verdict,
        config=config,
        report_path=str(report_path),
    )


__all__ = [
    "ECOLOGY_CHECKPOINT_BUNDLE_KIND",
    "LoadedEcologyCheckpoint",
    "ecology_checkpoint_compatibility",
    "load_promoted_ecology_checkpoint",
    "write_ecology_checkpoint_bundle",
]
