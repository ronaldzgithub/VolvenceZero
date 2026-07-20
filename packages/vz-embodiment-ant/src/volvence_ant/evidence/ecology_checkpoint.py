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
    decode_agent_learning_archive,
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
    ECOLOGY_CURRICULUM_SCHEMA_VERSION,
    ECOLOGY_REQUIRED_GATE_NAMES,
    EcologyCheckpointCandidate,
    EcologyCurriculumConfig,
)
from volvence_ant.runtime import AntSenseSchema


ECOLOGY_CHECKPOINT_BUNDLE_KIND = "digital-ant-ecology-checkpoint.v2"


@dataclass(frozen=True)
class LoadedEcologyCheckpoint:
    checkpoint_archives: tuple[bytes, ...]
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
        ("runtime_replay", "excluded"),
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
    checkpoint_archives: tuple[bytes, ...],
) -> str:
    records = tuple(
        decode_agent_learning_archive(item)
        for item in checkpoint_archives
    )
    checkpoint_ids = tuple(record.info.checkpoint_id for record in records)
    for body_id, checkpoint_id in enumerate(checkpoint_ids):
        if not checkpoint_id.endswith(f":body:{body_id}"):
            raise AntArtifactIntegrityError(
                "ecology checkpoint body mapping mismatch: "
                f"index={body_id}, checkpoint_id={checkpoint_id!r}"
            )
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise AntArtifactIntegrityError(
            "ecology checkpoint ids must be unique"
        )
    fingerprints = tuple(
        record.info.state_fingerprint for record in records
    )
    payload = json.dumps(
        fingerprints,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_report_verdict(report: dict[str, object]) -> str:
    if report.get("schema_version") != ECOLOGY_CURRICULUM_SCHEMA_VERSION:
        raise AntArtifactIntegrityError(f"unexpected ecology report schema: {report.get('schema_version')!r}")
    raw_gates = report.get("gates")
    if not isinstance(raw_gates, (list, tuple)) or not all(isinstance(gate, dict) for gate in raw_gates):
        raise AntArtifactIntegrityError("ecology checkpoint gates must be structured objects")
    gate_names = tuple(str(gate.get("name", "")) for gate in raw_gates)
    if gate_names != ECOLOGY_REQUIRED_GATE_NAMES:
        raise AntArtifactIntegrityError(f"ecology checkpoint gate set mismatch: {gate_names!r}")
    all_passed = all(gate.get("passed") is True for gate in raw_gates)
    expected_verdict = "PASS" if all_passed else "BLOCK"
    report_verdict = str(report.get("verdict", "BLOCK")).upper()
    if report_verdict != expected_verdict:
        raise AntArtifactIntegrityError("ecology checkpoint verdict contradicts its frozen gates")
    return report_verdict


def write_ecology_checkpoint_bundle(
    *,
    candidate: EcologyCheckpointCandidate,
    archive_path: Path,
    report_path: Path,
    repo_root: Path,
) -> Path:
    config = candidate.report.config
    report_payload = candidate.report.to_dict()
    _validated_report_verdict(report_payload)
    if len(candidate.checkpoints) != config.n_ants:
        raise AntArtifactIntegrityError("ecology in-process checkpoint count does not match configured ant count")
    if len(candidate.checkpoint_archives) != config.n_ants:
        raise AntArtifactIntegrityError("ecology checkpoint archive count does not match configured ant count")
    archive_bytes = encode_agent_learning_checkpoint_archive(
        candidate.checkpoint_archives,
        compatibility=ecology_checkpoint_compatibility(config),
    )
    _atomic_write_bytes(archive_path, archive_bytes)
    archive_digest = file_digest(archive_path, relative_to=repo_root)
    provenance = collect_ant_provenance(
        repo_root=repo_root,
        seeds=config.heldout_seeds,
        config=asdict(config),
        model_fingerprint=_aggregate_fingerprint(candidate.checkpoint_archives),
    )
    payload = {
        "artifact_kind": ECOLOGY_CHECKPOINT_BUNDLE_KIND,
        "promotion_verdict": candidate.report.verdict,
        "checkpoint_archive": asdict(archive_digest),
        "checkpoint_fingerprint": _aggregate_fingerprint(candidate.checkpoint_archives),
        "report": report_payload,
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
    raw_report = payload.get("report")
    if not isinstance(raw_report, dict):
        raise AntArtifactIntegrityError("ecology checkpoint report is missing report object")
    verdict = _validated_report_verdict(raw_report)
    if verdict != str(payload.get("promotion_verdict", "BLOCK")).upper():
        raise AntArtifactIntegrityError("ecology checkpoint promotion verdict mismatch")
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
        expected_compatibility=ecology_checkpoint_compatibility(config),
    )
    if len(archive.checkpoint_archives) != config.n_ants:
        raise AntArtifactIntegrityError("ecology checkpoint count does not match configured ant count")
    fingerprint = _aggregate_fingerprint(archive.checkpoint_archives)
    if fingerprint != str(payload.get("checkpoint_fingerprint", "")):
        raise AntArtifactIntegrityError("ecology aggregate checkpoint fingerprint mismatch")
    return LoadedEcologyCheckpoint(
        checkpoint_archives=archive.checkpoint_archives,
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
