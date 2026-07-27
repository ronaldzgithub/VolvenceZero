"""Persistence and promotion loading for learned ecology checkpoints.

Admission is governed by **P2**, not by the curriculum report.
``research/ant/05_ecology_p0_p1_p2_plan.md`` section 5.7 lists the conditions a
checkpoint must satisfy before it may be marked ``PASS`` *and loaded by the demo
loader*, and section 2.3 says in as many words that a P0/P1 checkpoint is never
loadable for demo/promotion.  Historically this module only validated the
curriculum report, whose own config accepts ``n_ants=1`` / ``stage_rounds=1`` /
a single held-out seed -- so a one-ant, one-round curriculum ``PASS`` was fully
loadable while a genuine P2 ``PASS`` produced no loadable bundle at all.

The bundle therefore carries a ``p2_promotion`` block, and
:func:`load_promoted_ecology_checkpoint` refuses a checkpoint without one.  Two
writers produce it:

* :func:`write_ecology_checkpoint_bundle` -- the curriculum lane.  Without a
  confirmatory verdict it still writes its evidence, but the bundle's
  ``promotion_verdict`` is ``BLOCK`` and carries the reason; that is the plan
  section 2.3 rule made mechanical rather than documentary.
* :func:`write_ecology_p2_promotion_bundle` -- the P2 lane.  It ships the
  learned shard's own journalled colony archive together with the confirmatory
  report that certifies it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from volvence_zero.agent import (
    AgentLearningCheckpointArchive,
    decode_agent_learning_archive,
    decode_agent_learning_checkpoint_archive,
    encode_agent_learning_checkpoint_archive,
)

from volvence_ant.evidence.provenance import (
    AntArtifactExistsError,
    AntArtifactIntegrityError,
    artifact_verdict_summary,
    collect_ant_provenance,
    ensure_artifact_writable,
    file_digest,
    require_ant_artifact_envelope,
    verify_ant_artifact_manifest,
    write_ant_artifact_bundle,
)
from volvence_ant.experiments.ecology_curriculum import (
    ECOLOGY_CURRICULUM_SCHEMA_VERSION,
    ECOLOGY_REQUIRED_GATE_NAMES,
    EcologyCheckpointCandidate,
    EcologyCurriculumConfig,
)
from volvence_ant.experiments.ecology_p2 import (
    ECOLOGY_P2_FORMAL_MIN_HELDOUT_LAYOUTS,
    ECOLOGY_P2_GATE_NAMES,
    ECOLOGY_P2_SCHEMA_VERSION,
)
from volvence_ant.runtime import AntSenseSchema
from volvence_ant.substrate.sense_encode import sense_channels


#: ``v5`` binds the P2 confirmatory verdict into promotion admission.  A ``v4``
#: bundle was admitted on its curriculum report alone, which the plan forbids;
#: it is refused rather than reinterpreted, and its checkpoint must be
#: re-promoted through the P2 lane.
ECOLOGY_CHECKPOINT_BUNDLE_KIND = "digital-ant-ecology-checkpoint.v5"

#: The reason recorded on a bundle written without confirmatory evidence.
ECOLOGY_PROMOTION_WITHOUT_P2_REASON = (
    "no P2 confirmatory verdict: plan section 2.3 forbids loading a P0/P1 "
    "checkpoint for demo or promotion, and section 5.7 makes the full P2 gate "
    "set the admission condition"
)

#: BEHAVIOUR CHANGE, stated in the error the operator actually sees.  Every
#: ecology promotion bundle written before ``v5`` is unloadable -- it was
#: admitted on its curriculum report alone, which the plan forbids -- so
#: ``python -m volvence_ant.app.server --ecology-checkpoint-report <old>``
#: aborts at startup instead of quietly serving an unconfirmed checkpoint.
#: That is the intended contract; this names the way out of it.
ECOLOGY_PROMOTION_REMEDY = (
    "This bundle predates the P2 admission contract and cannot be "
    "reinterpreted. Re-promote its checkpoint through the confirmatory lane "
    "(`python scripts/run_ant_ecology_p2.py promote --confirmatory-report "
    "<p2 report> --progress-dir <shard journal> --training-seed <seed>`), "
    "or start the app without --ecology-checkpoint-report to run the "
    "uncheckpointed arm"
)


@dataclass(frozen=True)
class EcologyP2PromotionEvidence:
    """The confirmatory P2 verdict a promotable checkpoint must carry."""

    schema_version: str
    preregistration_digest: str
    verdict: str
    #: The gate names the confirmatory report actually carried.  Kept verbatim
    #: so the loader can re-check completeness against the frozen set instead
    #: of trusting a summarised verdict.
    gate_names: tuple[str, ...]
    failed_gates: tuple[str, ...]
    training_seeds: tuple[int, ...]
    arms: tuple[str, ...]
    n_ants: int
    temporal_latent_dim: int
    #: The confirmatory device (plan section 5.2 freezes one formal device).
    #: Without it a promoted bundle cannot say which backend produced the
    #: numbers that admitted it.
    device: str
    #: The frozen held-out layout namespace the verdict was scored on (plan
    #: section 2.1: "布局 seed" is part of what every run must record, and the
    #: curriculum lane already records it).  Carried on the evidence so the
    #: promotion provenance can publish it; ``layout_seeds=()`` on a promoted
    #: bundle made the held-out namespace unrecoverable from the artifact.
    heldout_layout_seeds: tuple[int, ...]
    source_git_sha: str
    report_path: str
    report_sha256: str

    def validate(self) -> None:
        """Re-check the invariants that make this evidence admissible.

        Applied both when the block is built from a report and when it is read
        back out of a bundle, so a hand-edited ``p2_promotion`` block cannot
        claim ``PASS`` with a thinned or drifted gate set.
        """

        if self.schema_version != ECOLOGY_P2_SCHEMA_VERSION:
            raise AntArtifactIntegrityError(
                "unexpected P2 confirmatory schema: "
                f"expected={ECOLOGY_P2_SCHEMA_VERSION!r}, "
                f"actual={self.schema_version!r}"
            )
        if self.gate_names != ECOLOGY_P2_GATE_NAMES:
            raise AntArtifactIntegrityError(
                f"P2 confirmatory gate set mismatch: {self.gate_names!r}"
            )
        unknown = tuple(
            name for name in self.failed_gates if name not in self.gate_names
        )
        if unknown:
            raise AntArtifactIntegrityError(
                f"P2 confirmatory failed-gate names are not gates: {unknown!r}"
            )
        expected = "PASS" if not self.failed_gates else "BLOCK"
        if self.verdict != expected:
            raise AntArtifactIntegrityError(
                "P2 confirmatory verdict contradicts its frozen gates: "
                f"verdict={self.verdict}, failed={list(self.failed_gates)}"
            )
        if not self.device.strip():
            raise AntArtifactIntegrityError(
                "P2 confirmatory evidence records no device; plan section 5.2 "
                "freezes one formal device per confirmatory matrix"
            )
        if len(set(self.heldout_layout_seeds)) != len(
            self.heldout_layout_seeds
        ):
            raise AntArtifactIntegrityError(
                "P2 confirmatory held-out layout seeds contain duplicates: "
                f"{list(self.heldout_layout_seeds)}"
            )
        if self.verdict == "PASS" and (
            len(self.heldout_layout_seeds) < ECOLOGY_P2_FORMAL_MIN_HELDOUT_LAYOUTS
        ):
            raise AntArtifactIntegrityError(
                "P2 confirmatory evidence claims PASS with "
                f"{len(self.heldout_layout_seeds)} held-out layout seeds; "
                f"plan section 5.2 freezes >={ECOLOGY_P2_FORMAL_MIN_HELDOUT_LAYOUTS}"
            )


@dataclass(frozen=True)
class LoadedEcologyCheckpoint:
    checkpoint_archives: tuple[bytes, ...]
    fingerprint: str
    verdict: str
    config: EcologyCurriculumConfig
    report_path: str
    #: Always populated by :func:`load_promoted_ecology_checkpoint`; the loader
    #: refuses a bundle that carries none.  It stays optional on the dataclass
    #: only so in-process fixtures can build a checkpoint without a full
    #: confirmatory report.
    p2_promotion: EcologyP2PromotionEvidence | None = None


def p2_promotion_evidence(
    payload: Mapping[str, Any],
    *,
    report_path: str,
    report_sha256: str,
) -> EcologyP2PromotionEvidence:
    """Validate a P2 confirmatory report payload into promotion evidence.

    Mirrors :func:`_validated_report_verdict` one level up: schema version,
    the complete frozen gate set, and a verdict that does not contradict its
    own gates.  A drifted or thinned gate list is refused rather than being
    read as "everything present passed".

    A missing key is an integrity failure of the artifact, not a programming
    error in the caller, so it surfaces as :class:`AntArtifactIntegrityError`
    naming the field -- a bare ``KeyError('arms')` escaping into a CLI told an
    operator nothing about which artifact was malformed.
    """

    raw_gates = payload.get("gates")
    if not isinstance(raw_gates, (list, tuple)) or not all(
        isinstance(gate, dict) for gate in raw_gates
    ):
        raise AntArtifactIntegrityError(
            "P2 confirmatory gates must be structured objects"
        )
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise AntArtifactIntegrityError(
            "P2 confirmatory report is missing its config object"
        )
    with _required_fields("P2 confirmatory report", report_path):
        evidence = EcologyP2PromotionEvidence(
            schema_version=str(payload.get("schema_version")),
            preregistration_digest=str(payload["preregistration_digest"]),
            verdict=str(payload.get("verdict", "BLOCK")).upper(),
            gate_names=tuple(str(gate.get("name", "")) for gate in raw_gates),
            failed_gates=tuple(
                str(gate.get("name"))
                for gate in raw_gates
                if gate.get("passed") is not True
            ),
            training_seeds=tuple(
                int(value) for value in payload["training_seeds"]
            ),
            arms=tuple(str(value) for value in payload["arms"]),
            n_ants=int(raw_config["n_ants"]),
            temporal_latent_dim=int(raw_config["temporal_latent_dim"]),
            device=str(payload["device"]),
            heldout_layout_seeds=tuple(
                int(value) for value in payload["heldout_layout_seeds"]
            ),
            source_git_sha=str(payload["source_git_sha"]),
            report_path=report_path,
            report_sha256=report_sha256,
        )
    evidence.validate()
    return evidence


@contextmanager
def _required_fields(kind: str, source: str) -> Iterator[None]:
    """Turn a missing artifact field into a named integrity error."""

    try:
        yield
    except KeyError as exc:
        raise AntArtifactIntegrityError(
            f"{kind} is missing a required field {exc.args[0]!r}: {source}"
        ) from exc


def _p2_promotion_from_bundle(
    raw: Mapping[str, Any],
) -> EcologyP2PromotionEvidence:
    """Read the ``p2_promotion`` block back out of a bundle, re-validated."""

    with _required_fields("ecology checkpoint p2_promotion block", "bundle"):
        evidence = EcologyP2PromotionEvidence(
            schema_version=str(raw["schema_version"]),
            preregistration_digest=str(raw["preregistration_digest"]),
            verdict=str(raw["verdict"]).upper(),
            gate_names=tuple(str(value) for value in raw["gate_names"]),
            failed_gates=tuple(str(value) for value in raw["failed_gates"]),
            training_seeds=tuple(int(value) for value in raw["training_seeds"]),
            arms=tuple(str(value) for value in raw["arms"]),
            n_ants=int(raw["n_ants"]),
            temporal_latent_dim=int(raw["temporal_latent_dim"]),
            device=str(raw["device"]),
            heldout_layout_seeds=tuple(
                int(value) for value in raw["heldout_layout_seeds"]
            ),
            source_git_sha=str(raw["source_git_sha"]),
            report_path=str(raw["report_path"]),
            report_sha256=str(raw["report_sha256"]),
        )
    evidence.validate()
    return evidence


def ecology_checkpoint_compatibility(
    config: EcologyCurriculumConfig,
) -> tuple[tuple[str, str], ...]:
    return (
        ("artifact_kind", ECOLOGY_CHECKPOINT_BUNDLE_KIND),
        ("sense_schema", AntSenseSchema.ECOLOGY_V2.value),
        ("input_dim", str(len(sense_channels(AntSenseSchema.ECOLOGY_V2)))),
        ("latent_dim", str(config.temporal_latent_dim)),
        ("n_ants", str(config.n_ants)),
        ("runtime_replay", "excluded"),
    )


def ensure_ecology_archive_writable(
    archive_path: Path,
    *,
    report_path: Path,
    overwrite: bool,
) -> None:
    """Refuse to destroy the ``.vzac`` half of an existing promotion bundle.

    ``ensure_artifact_writable`` covers the report and its manifest; the
    archive is a separate file on the same bundle and needs the same guard, or
    the report would be protected while the checkpoint bytes it certifies were
    replaced underneath it.  ``report_path`` is only used to name the verdict a
    forced overwrite would destroy, so the caller passes the report it is
    actually about to write rather than one derived from the archive name.
    """

    if overwrite or not archive_path.exists():
        return
    detail = (
        artifact_verdict_summary(report_path)
        if report_path.exists()
        else "report absent"
    )
    raise AntArtifactExistsError(
        "refusing to overwrite an existing ecology checkpoint archive "
        f"{archive_path} [{detail}]; write a new run-id filename, or pass "
        "overwrite=True to destroy it deliberately"
    )


def _atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if not overwrite and path.exists():
        raise AntArtifactExistsError(
            f"refusing to overwrite an existing archive: {path}"
        )
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


def _promotion_verdict(
    *,
    curriculum_verdict: str | None,
    p2_promotion: EcologyP2PromotionEvidence | None,
) -> tuple[str, str]:
    """The bundle's promotion verdict and, when blocked, why.

    Promotion is the conjunction of every stage that owns a verdict.  A
    curriculum ``PASS`` alone is development evidence (plan section 4.1), so
    without confirmatory evidence the bundle is written ``BLOCK`` *with the
    reason recorded* -- it is not silently downgraded, and the reason travels
    with the artifact.
    """

    if p2_promotion is None:
        return "BLOCK", ECOLOGY_PROMOTION_WITHOUT_P2_REASON
    # Re-checked on every writer and on load, so a hand-built evidence block
    # can never reach a bundle claiming PASS with a thinned gate set.
    p2_promotion.validate()
    if p2_promotion.verdict != "PASS":
        return "BLOCK", (
            "P2 confirmatory verdict is "
            f"{p2_promotion.verdict}: {list(p2_promotion.failed_gates)}"
        )
    if curriculum_verdict is not None and curriculum_verdict != "PASS":
        return "BLOCK", f"curriculum verdict is {curriculum_verdict}"
    return "PASS", ""


def write_ecology_checkpoint_bundle(
    *,
    candidate: EcologyCheckpointCandidate,
    archive_path: Path,
    report_path: Path,
    repo_root: Path,
    p2_promotion: EcologyP2PromotionEvidence | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the ``.vzac`` archive, the report and the sidecar manifest.

    This is an evidence-lane writer, not a shared one: the promotion bundle is
    the artifact plan section 2.1 protects most, so ``overwrite`` defaults to
    ``False``.  The archive and the report are both checked *before* the
    archive bytes are written, so a half-replaced bundle cannot exist.

    ``p2_promotion`` is what makes the bundle loadable.  The curriculum lane
    has no confirmatory verdict of its own, so it normally passes ``None`` and
    the resulting bundle is a ``BLOCK`` diagnostic artifact -- exactly what
    plan section 2.3 requires of a P0/P1 checkpoint.
    """

    config = candidate.report.config
    report_payload = candidate.report.to_dict()
    curriculum_verdict = _validated_report_verdict(report_payload)
    if len(candidate.checkpoints) != config.n_ants:
        raise AntArtifactIntegrityError("ecology in-process checkpoint count does not match configured ant count")
    if len(candidate.checkpoint_archives) != config.n_ants:
        raise AntArtifactIntegrityError("ecology checkpoint archive count does not match configured ant count")
    if p2_promotion is not None and p2_promotion.n_ants != config.n_ants:
        raise AntArtifactIntegrityError(
            "P2 confirmatory evidence describes a different colony: "
            f"p2_n_ants={p2_promotion.n_ants}, "
            f"curriculum_n_ants={config.n_ants}"
        )
    verdict, block_reason = _promotion_verdict(
        curriculum_verdict=curriculum_verdict,
        p2_promotion=p2_promotion,
    )
    ensure_artifact_writable(report_path, overwrite=overwrite)
    ensure_ecology_archive_writable(
        archive_path,
        report_path=report_path,
        overwrite=overwrite,
    )
    archive_bytes = encode_agent_learning_checkpoint_archive(
        candidate.checkpoint_archives,
        compatibility=ecology_checkpoint_compatibility(config),
    )
    _atomic_write_bytes(archive_path, archive_bytes, overwrite=overwrite)
    archive_digest = file_digest(archive_path, relative_to=repo_root)
    provenance = collect_ant_provenance(
        repo_root=repo_root,
        seeds=config.heldout_seeds,
        config=asdict(config),
        model_fingerprint=_aggregate_fingerprint(candidate.checkpoint_archives),
        # The curriculum seed drives training; validation and held-out layout
        # seeds are a disjoint namespace (plan section 2.1) and stay separately
        # recoverable from the record instead of collapsing into one schedule.
        training_seeds=(config.seed,),
        layout_seeds=tuple(config.validation_seeds) + tuple(config.heldout_seeds),
    )
    payload = {
        "artifact_kind": ECOLOGY_CHECKPOINT_BUNDLE_KIND,
        "promotion_verdict": verdict,
        "promotion_block_reason": block_reason,
        "checkpoint_archive": asdict(archive_digest),
        "checkpoint_fingerprint": _aggregate_fingerprint(candidate.checkpoint_archives),
        "checkpoint_shape": {
            "n_ants": config.n_ants,
            "temporal_latent_dim": config.temporal_latent_dim,
        },
        "report": report_payload,
        "p2_promotion": (
            None if p2_promotion is None else asdict(p2_promotion)
        ),
    }
    return write_ant_artifact_bundle(
        artifact_path=report_path,
        payload=payload,
        provenance=provenance,
        input_paths=(archive_path,),
        repo_root=repo_root,
        overwrite=overwrite,
    )


def write_ecology_p2_promotion_bundle(
    *,
    checkpoint_archives: tuple[bytes, ...],
    p2_promotion: EcologyP2PromotionEvidence,
    archive_path: Path,
    report_path: Path,
    repo_root: Path,
    overwrite: bool = False,
) -> Path:
    """Turn a P2 confirmatory verdict into the loadable promotion bundle.

    ``checkpoint_archives`` are the learned shard's own journalled colony
    archives (``ecology_p2.load_shard_checkpoint_archives``): promotion ships
    the audited checkpoint, never a freshly retrained one.  There is no
    curriculum report on this bundle -- P2 is the confirmatory owner, and
    fabricating a curriculum report here would invent evidence.
    """

    if len(checkpoint_archives) != p2_promotion.n_ants:
        raise AntArtifactIntegrityError(
            "P2 promotion archive count does not match the confirmatory ant "
            f"count: archives={len(checkpoint_archives)}, "
            f"n_ants={p2_promotion.n_ants}"
        )
    verdict, block_reason = _promotion_verdict(
        curriculum_verdict=None,
        p2_promotion=p2_promotion,
    )
    # Shape binding only: the bundle's authoritative configuration is
    # ``p2_promotion``.  The curriculum config type is reused because it is
    # what ``ecology_checkpoint_compatibility`` and the app runner read, and
    # only ``n_ants`` / ``temporal_latent_dim`` are consulted from it.
    shape = EcologyCurriculumConfig(
        n_ants=p2_promotion.n_ants,
        temporal_latent_dim=p2_promotion.temporal_latent_dim,
    )
    ensure_artifact_writable(report_path, overwrite=overwrite)
    ensure_ecology_archive_writable(
        archive_path,
        report_path=report_path,
        overwrite=overwrite,
    )
    archive_bytes = encode_agent_learning_checkpoint_archive(
        checkpoint_archives,
        compatibility=ecology_checkpoint_compatibility(shape),
    )
    _atomic_write_bytes(archive_path, archive_bytes, overwrite=overwrite)
    archive_digest = file_digest(archive_path, relative_to=repo_root)
    provenance = collect_ant_provenance(
        repo_root=repo_root,
        seeds=p2_promotion.training_seeds,
        config=asdict(p2_promotion),
        model_fingerprint=_aggregate_fingerprint(checkpoint_archives),
        # Plan section 2.1 requires the device, the training seeds AND the
        # layout seeds on every run record; the curriculum lane already writes
        # all three, and the promotion lane used to drop two of them
        # (``layout_seeds=()``, no device), so the 30 frozen held-out seeds and
        # the confirmatory backend were unrecoverable from a promoted bundle.
        device=p2_promotion.device,
        training_seeds=p2_promotion.training_seeds,
        layout_seeds=p2_promotion.heldout_layout_seeds,
    )
    payload = {
        "artifact_kind": ECOLOGY_CHECKPOINT_BUNDLE_KIND,
        "promotion_verdict": verdict,
        "promotion_block_reason": block_reason,
        "checkpoint_archive": asdict(archive_digest),
        "checkpoint_fingerprint": _aggregate_fingerprint(checkpoint_archives),
        "checkpoint_shape": {
            "n_ants": shape.n_ants,
            "temporal_latent_dim": shape.temporal_latent_dim,
        },
        "p2_promotion": asdict(p2_promotion),
    }
    return write_ant_artifact_bundle(
        artifact_path=report_path,
        payload=payload,
        provenance=provenance,
        input_paths=(archive_path,),
        repo_root=repo_root,
        overwrite=overwrite,
    )


def _checkpoint_shape(payload: dict[str, object]) -> EcologyCurriculumConfig:
    """The colony shape the archive must decode under.

    Only ``n_ants`` and ``temporal_latent_dim`` matter here -- they are what
    ``ecology_checkpoint_compatibility`` binds and what the app runner checks.
    The writer declares them explicitly under ``checkpoint_shape``, so this
    loader is not a second parser of the curriculum owner's config dataclass:
    rebuilding a producer's state field-by-field in a consumer makes every
    curriculum field rename a silent promotion-loading break (AGENTS.md
    section 4.2).  The full curriculum report still travels in the bundle as
    evidence, and is still verdict-validated; it is simply not re-derived here.
    """

    raw_shape = payload.get("checkpoint_shape")
    if not isinstance(raw_shape, dict):
        raise AntArtifactIntegrityError(
            "ecology checkpoint bundle carries no checkpoint_shape "
            "declaration; its archive compatibility cannot be reconstructed"
        )
    return EcologyCurriculumConfig(
        n_ants=int(raw_shape["n_ants"]),
        temporal_latent_dim=int(raw_shape["temporal_latent_dim"]),
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
    require_ant_artifact_envelope(payload, path=report_path)
    if payload.get("artifact_kind") != ECOLOGY_CHECKPOINT_BUNDLE_KIND:
        raise AntArtifactIntegrityError(
            f"unexpected ecology artifact kind: {payload.get('artifact_kind')!r} "
            f"(expected {ECOLOGY_CHECKPOINT_BUNDLE_KIND!r}). "
            + ECOLOGY_PROMOTION_REMEDY
        )
    raw_p2 = payload.get("p2_promotion")
    if not isinstance(raw_p2, dict):
        raise AntArtifactIntegrityError(
            "ecology checkpoint carries no P2 promotion verdict; "
            + ECOLOGY_PROMOTION_WITHOUT_P2_REASON
            + ". "
            + ECOLOGY_PROMOTION_REMEDY
        )
    p2_promotion = _p2_promotion_from_bundle(raw_p2)
    # A curriculum report is optional (the P2 lane ships none) but is still
    # fully validated when present: a promoted bundle may not carry a
    # contradicted or schema-drifted development report.
    raw_report = payload.get("report")
    curriculum_verdict: str | None = None
    if raw_report is not None:
        if not isinstance(raw_report, dict):
            raise AntArtifactIntegrityError("ecology checkpoint report is missing report object")
        curriculum_verdict = _validated_report_verdict(raw_report)
    verdict, block_reason = _promotion_verdict(
        curriculum_verdict=curriculum_verdict,
        p2_promotion=p2_promotion,
    )
    if verdict != str(payload.get("promotion_verdict", "BLOCK")).upper():
        raise AntArtifactIntegrityError("ecology checkpoint promotion verdict mismatch")
    if verdict != "PASS":
        raise AntArtifactIntegrityError(
            f"ecology checkpoint is not promoted: verdict={verdict} "
            f"({block_reason})"
        )
    raw_archive = payload.get("checkpoint_archive")
    if not isinstance(raw_archive, dict) or not raw_archive.get("path"):
        raise AntArtifactIntegrityError("ecology checkpoint archive reference missing")
    archive_path = repo_root / str(raw_archive["path"])
    actual_digest = file_digest(archive_path, relative_to=repo_root)
    if actual_digest.sha256 != str(raw_archive.get("sha256", "")) or actual_digest.size_bytes != int(
        raw_archive.get("size_bytes", -1)
    ):
        raise AntArtifactIntegrityError("ecology checkpoint archive digest mismatch")
    config = _checkpoint_shape(payload)
    if raw_report is not None:
        # ``checkpoint_shape`` is a declaration, not a derivation (see
        # :func:`_checkpoint_shape`).  When the bundle ALSO carries the
        # curriculum report that produced the archive, the two declarations
        # must agree: otherwise an internally contradictory bundle -- a report
        # describing a 4-ant colony next to a 1-ant shape declaration --
        # decodes cleanly and the report travels as evidence for a checkpoint
        # it does not describe.
        with _required_fields("ecology checkpoint curriculum report", str(report_path)):
            raw_report_config = raw_report["config"]
            if not isinstance(raw_report_config, dict):
                raise AntArtifactIntegrityError(
                    "ecology checkpoint curriculum report has no config object"
                )
            report_shape = (
                int(raw_report_config["n_ants"]),
                int(raw_report_config["temporal_latent_dim"]),
            )
        if report_shape != (config.n_ants, config.temporal_latent_dim):
            raise AntArtifactIntegrityError(
                "ecology checkpoint_shape contradicts the curriculum report "
                f"it ships: shape={(config.n_ants, config.temporal_latent_dim)}, "
                f"report={report_shape}"
            )
    if config.n_ants != p2_promotion.n_ants or (
        config.temporal_latent_dim != p2_promotion.temporal_latent_dim
    ):
        raise AntArtifactIntegrityError(
            "ecology checkpoint shape disagrees with its P2 confirmatory "
            f"evidence: bundle=({config.n_ants}, "
            f"{config.temporal_latent_dim}), p2=({p2_promotion.n_ants}, "
            f"{p2_promotion.temporal_latent_dim})"
        )
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
        p2_promotion=p2_promotion,
    )


__all__ = [
    "ECOLOGY_CHECKPOINT_BUNDLE_KIND",
    "ECOLOGY_PROMOTION_REMEDY",
    "ECOLOGY_PROMOTION_WITHOUT_P2_REASON",
    "EcologyP2PromotionEvidence",
    "LoadedEcologyCheckpoint",
    "ecology_checkpoint_compatibility",
    "ensure_ecology_archive_writable",
    "load_promoted_ecology_checkpoint",
    "p2_promotion_evidence",
    "write_ecology_checkpoint_bundle",
    "write_ecology_p2_promotion_bundle",
]
