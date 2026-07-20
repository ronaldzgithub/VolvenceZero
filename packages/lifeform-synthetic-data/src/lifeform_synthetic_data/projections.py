"""Reversible task projections from the immutable master trajectory."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

from companion_standard.trajectory import (
    SCHEMA_VERSION as RELATIONSHIP_SCHEMA_VERSION,
)
from companion_standard.trajectory import (
    InteractionTrajectory,
    LabelSource,
    RelationshipPhase,
    RelationshipStateLabel,
    TrajectorySession,
    TrajectorySource,
    TrajectoryTurn,
)
from companion_standard.trajectory import TurnRole as RelationshipTurnRole
from companion_standard.canonical import to_canonical_json as relationship_json

from .canonical import canonical_json, stable_hash
from .contracts import (
    AnnotationRecord,
    AnnotationSource,
    CorpusSplit,
    ExperienceTrajectory,
    GenerationTier,
    TrainingUse,
    TurnRole,
)
from .storage import sha256_file

PROJECTION_SCHEMA_VERSION = "synthetic-projection.v1"

SEMANTIC_OWNER_NAMES = frozenset(
    {
        "plan_intent",
        "commitment",
        "open_loop",
        "user_model",
        "execution_result",
        "belief_assumption",
        "relationship_state",
        "goal_value",
        "boundary_consent",
    }
)

SOCIAL_OWNER_NAMES = frozenset(
    {
        "conversational_role",
        "belief_about_other",
        "intent_about_other",
        "feeling_about_other",
        "preference_about_other",
        "common_ground",
        "groups",
    }
)

_RELATIONSHIP_VAL_FAMILIES = frozenset({"absence_reengagement", "belief_uncertainty_verification"})
_RELATIONSHIP_TEST_FAMILIES = frozenset({"rupture_repair", "multi_party_identity_privacy"})
_PRECOMPUTED_TRAJECTORY_HASHES: dict[str, str] = {}


class ProjectionView(str, Enum):
    RELATIONSHIP_ENCODER = "relationship_encoder"
    EXPRESSION_SFT = "expression_sft"
    SEMANTIC_OWNER = "semantic_owner"
    SOCIAL_COGNITION = "social_cognition"
    MEMORY_RETRIEVAL = "memory_retrieval"
    TEMPORAL_SSL = "temporal_ssl"
    INTERNAL_RL = "internal_rl"
    EVALUATION_ONLY = "evaluation_only"
    HUMAN_REVIEW_QUEUE = "human_review_queue"


@dataclass(frozen=True)
class ProjectionRecord:
    schema_version: str
    view: ProjectionView
    record_id: str
    master_trajectory_id: str
    master_trajectory_hash: str
    split: CorpusSplit
    training_use: TrainingUse
    payload_json: str
    source_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PROJECTION_SCHEMA_VERSION:
            raise ValueError("ProjectionRecord.schema_version mismatch")
        if not self.record_id.strip():
            raise ValueError("ProjectionRecord.record_id must be non-empty")
        if not self.master_trajectory_id.strip():
            raise ValueError("ProjectionRecord.master_trajectory_id must be non-empty")
        if len(self.master_trajectory_hash) != 64:
            raise ValueError("ProjectionRecord.master_trajectory_hash must be SHA-256")
        try:
            json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("ProjectionRecord.payload_json is invalid") from error
        if not self.source_refs or any(not ref.strip() for ref in self.source_refs):
            raise ValueError("ProjectionRecord.source_refs must be non-empty")


@dataclass(frozen=True)
class ProjectionManifest:
    schema_version: str
    view: ProjectionView
    record_count: int
    split_counts: tuple[tuple[str, int], ...]
    master_trajectory_hashes: tuple[str, ...]
    records_uri: str
    records_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != PROJECTION_SCHEMA_VERSION:
            raise ValueError("ProjectionManifest.schema_version mismatch")
        if self.record_count < 0:
            raise ValueError("ProjectionManifest.record_count must be non-negative")
        if sum(count for _, count in self.split_counts) != self.record_count:
            raise ValueError("ProjectionManifest.split_counts sum mismatch")
        if len(self.records_sha256) != 64:
            raise ValueError("ProjectionManifest.records_sha256 must be SHA-256")


def project_relationship_encoder(
    trajectory: ExperienceTrajectory,
) -> InteractionTrajectory:
    """Project transcript + FSM/world labels to the public standard."""

    sessions = tuple(
        TrajectorySession(
            session_index=session.session_index,
            gap_days_before=session.gap_days_before,
            turns=tuple(
                TrajectoryTurn(
                    turn_index=turn.turn_index,
                    role=(RelationshipTurnRole.USER if turn.role is TurnRole.USER else RelationshipTurnRole.ASSISTANT),
                    text=turn.text,
                )
                for turn in session.turns
                if turn.role in {TurnRole.USER, TurnRole.ASSISTANT}
            ),
        )
        for session in trajectory.sessions
    )
    annotation_by_id = {annotation.annotation_id: annotation for annotation in trajectory.annotations}
    labels: list[RelationshipStateLabel] = []
    for frame in trajectory.truth_frames:
        turn = _turn_by_id(trajectory, frame.turn_ref)
        if turn.role is not TurnRole.USER:
            continue
        evidence_refs = tuple(
            annotation_id
            for annotation_id in frame.annotation_refs
            if _is_generator_truth(annotation_by_id[annotation_id])
        )
        if not evidence_refs:
            raise ValueError(f"relationship label frame {frame.frame_id!r} has no generator truth")
        phase = _relationship_phase(trajectory.family, frame.phase_id)
        trust, continuity, repair = _relationship_readouts(phase)
        labels.append(
            RelationshipStateLabel(
                session_index=turn.session_index,
                turn_index=turn.turn_index,
                phase=phase,
                trust_level=trust,
                continuity_level=continuity,
                repair_pressure=repair,
                source=LabelSource.FSM_GROUND_TRUTH,
                evidence="generator_truth:" + ",".join(evidence_refs),
            )
        )
    labels.sort(key=lambda item: (item.session_index, item.turn_index))
    source = (
        TrajectorySource.SYNTHETIC_LLM if trajectory.provenance.model_id is not None else TrajectorySource.SYNTHETIC_FSM
    )
    return InteractionTrajectory(
        trajectory_id=trajectory.trajectory_id,
        schema_version=RELATIONSHIP_SCHEMA_VERSION,
        source=source,
        family=trajectory.family,
        scenario_ref=trajectory.scenario_hash,
        sessions=sessions,
        labels=tuple(labels),
        metadata=(
            ("master_trajectory_hash", _trajectory_hash(trajectory)),
            ("master_split", trajectory.split.value),
            ("projection_split", _relationship_split(trajectory.family).value),
            ("label_provenance", "generator_truth_only"),
        ),
    )


def project_relationship_record(
    trajectory: ExperienceTrajectory,
) -> ProjectionRecord:
    projected = project_relationship_encoder(trajectory)
    return _record(
        trajectory,
        view=ProjectionView.RELATIONSHIP_ENCODER,
        suffix="trajectory",
        training_use=TrainingUse.TARGET,
        payload=projected,
        source_refs=tuple(
            annotation.annotation_id
            for annotation in trajectory.annotations
            if annotation.source is AnnotationSource.GENERATOR_TRUTH
        ),
        split_override=_relationship_split(trajectory.family),
    )


def project_expression_sft(
    trajectory: ExperienceTrajectory,
) -> tuple[ProjectionRecord, ...]:
    if trajectory.generation_tier is not GenerationTier.RENDERED:
        raise ValueError("expression_sft requires a rendered trajectory")
    if any(not item.passed and item.severity.value == "error" for item in trajectory.quality):
        raise ValueError("expression_sft rejects trajectories with failed hard gates")
    truth_by_id = {frame.frame_id: frame for frame in trajectory.truth_frames}
    history: list[dict[str, str]] = []
    records: list[ProjectionRecord] = []
    preceding_truth_ref: str | None = None
    for session in trajectory.sessions:
        for turn in session.turns:
            if turn.role is TurnRole.USER:
                preceding_truth_ref = turn.latent_frame_ref
                history.append({"role": "user", "content": turn.text})
                continue
            if turn.role is not TurnRole.ASSISTANT:
                continue
            if preceding_truth_ref is None:
                raise ValueError("assistant SFT turn lacks a preceding truth frame")
            truth = truth_by_id[preceding_truth_ref]
            records.append(
                _record(
                    trajectory,
                    view=ProjectionView.EXPRESSION_SFT,
                    suffix=turn.turn_id,
                    training_use=TrainingUse.TARGET,
                    payload={
                        "messages": tuple(history),
                        "response": turn.text,
                        "response_contract": truth.response_contract,
                        "language": trajectory.language,
                    },
                    source_refs=(truth.frame_id, turn.turn_id),
                )
            )
            history.append({"role": "assistant", "content": turn.text})
    return tuple(records)


def project_semantic_owner(
    trajectory: ExperienceTrajectory,
) -> tuple[ProjectionRecord, ...]:
    return _project_annotations(
        trajectory,
        view=ProjectionView.SEMANTIC_OWNER,
        owner_names=SEMANTIC_OWNER_NAMES,
    )


def project_social_cognition(
    trajectory: ExperienceTrajectory,
) -> tuple[ProjectionRecord, ...]:
    return _project_annotations(
        trajectory,
        view=ProjectionView.SOCIAL_COGNITION,
        owner_names=SOCIAL_OWNER_NAMES,
    )


def project_memory_retrieval(
    trajectory: ExperienceTrajectory,
) -> tuple[ProjectionRecord, ...]:
    facts = tuple(
        (frame.frame_id, item.value)
        for frame in trajectory.truth_frames
        for item in frame.observable_facts
        if item.key == "fact"
    )
    negative_candidates = tuple(
        (frame.frame_id, item.value)
        for frame in trajectory.truth_frames
        for item in frame.observable_facts
        if item.key in {"fact", "context_constraint"}
    )
    records: list[ProjectionRecord] = []
    for index, (frame_id, positive) in enumerate(facts):
        negatives = tuple(
            dict.fromkeys(
                fact for candidate_id, fact in negative_candidates if candidate_id != frame_id and fact != positive
            )
        )[:3]
        if not negatives:
            continue
        records.append(
            _record(
                trajectory,
                view=ProjectionView.MEMORY_RETRIEVAL,
                suffix=frame_id,
                training_use=TrainingUse.TARGET,
                payload={
                    "query": positive,
                    "positive": positive,
                    "hard_negatives": negatives,
                    "subject_scope": _metadata_value(
                        trajectory,
                        "persona_id",
                    ),
                    "ordinal": index,
                },
                source_refs=(frame_id,),
            )
        )
    return tuple(records)


def project_temporal_ssl(
    trajectory: ExperienceTrajectory,
) -> tuple[ProjectionRecord, ...]:
    _require_live_through(trajectory, view=ProjectionView.TEMPORAL_SSL)
    snapshots = {frame.snapshot_id: frame for frame in trajectory.snapshot_frames}
    records: list[ProjectionRecord] = []
    for session in trajectory.sessions:
        user_turns = tuple(turn for turn in session.turns if turn.role is TurnRole.USER)
        for index in range(len(user_turns) - 1):
            current = user_turns[index]
            following = user_turns[index + 1]
            records.append(
                _record(
                    trajectory,
                    view=ProjectionView.TEMPORAL_SSL,
                    suffix=f"{current.turn_id}:next",
                    training_use=TrainingUse.TARGET,
                    payload={
                        "current_snapshot_frames": tuple(
                            _snapshot_payload(snapshots[ref]) for ref in current.snapshot_refs
                        ),
                        "next_snapshot_frames": tuple(
                            _snapshot_payload(snapshots[ref]) for ref in following.snapshot_refs
                        ),
                        "semantic_labels": (),
                    },
                    source_refs=current.snapshot_refs + following.snapshot_refs,
                )
            )
    return tuple(records)


def project_internal_rl(
    trajectory: ExperienceTrajectory,
) -> tuple[ProjectionRecord, ...]:
    _require_live_through(trajectory, view=ProjectionView.INTERNAL_RL)
    snapshots = {frame.snapshot_id: frame for frame in trajectory.snapshot_frames}
    records: list[ProjectionRecord] = []
    required_slots = {"temporal_abstraction", "prediction_error", "credit"}
    for session in trajectory.sessions:
        for turn in session.turns:
            if turn.role is not TurnRole.USER:
                continue
            turn_frames = tuple(snapshots[ref] for ref in turn.snapshot_refs)
            by_slot = {frame.slot_name: frame for frame in turn_frames}
            if not required_slots.issubset(by_slot):
                continue
            source_refs = tuple(by_slot[slot].snapshot_id for slot in sorted(required_slots))
            records.append(
                _record(
                    trajectory,
                    view=ProjectionView.INTERNAL_RL,
                    suffix=turn.turn_id,
                    training_use=TrainingUse.TARGET,
                    payload={
                        "controller_observation": _snapshot_payload(by_slot["temporal_abstraction"]),
                        "prediction_error": _snapshot_payload(by_slot["prediction_error"]),
                        "credit": _snapshot_payload(by_slot["credit"]),
                        "learning_space": "z_t",
                        "manual_semantic_label": None,
                    },
                    source_refs=source_refs,
                )
            )
    return tuple(records)


def project_evaluation_only(
    trajectory: ExperienceTrajectory,
) -> tuple[ProjectionRecord, ...]:
    payload = {
        "sessions": tuple(
            {
                "session_index": session.session_index,
                "turns": tuple({"role": turn.role.value, "text": turn.text} for turn in session.turns),
            }
            for session in trajectory.sessions
        ),
        "public_snapshot_refs": tuple(frame.snapshot_id for frame in trajectory.snapshot_frames),
        "sealed_truth_refs": tuple(frame.frame_id for frame in trajectory.truth_frames),
        "quality": trajectory.quality,
        "judge_output": None,
    }
    return (
        _record(
            trajectory,
            view=ProjectionView.EVALUATION_ONLY,
            suffix="evaluation-packet",
            training_use=TrainingUse.EVAL_ONLY,
            payload=payload,
            source_refs=(trajectory.trajectory_id,),
        ),
    )


def project_human_review_queue(
    trajectory: ExperienceTrajectory,
    *,
    sample_rate: float = 0.05,
) -> tuple[ProjectionRecord, ...]:
    if sample_rate < 0.0 or sample_rate > 1.0:
        raise ValueError("sample_rate must be in [0, 1]")
    selected: list[AnnotationRecord] = []
    for annotation in trajectory.annotations:
        digest = hashlib.sha256(annotation.annotation_id.encode("utf-8")).digest()
        draw = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        if (
            annotation.confidence < 1.0
            or annotation.adjudication.value in {"pending", "conflicted"}
            or draw < sample_rate
        ):
            selected.append(annotation)
    if not selected and all(item.passed for item in trajectory.quality):
        return ()
    return (
        _record(
            trajectory,
            view=ProjectionView.HUMAN_REVIEW_QUEUE,
            suffix="review",
            training_use=TrainingUse.QUARANTINED,
            payload={
                "blind_transcript": tuple(
                    {
                        "role": turn.role.value,
                        "text": turn.text,
                    }
                    for session in trajectory.sessions
                    for turn in session.turns
                ),
                "candidate_annotation_refs": tuple(item.annotation_id for item in selected),
                "human_label": None,
                "review_status": "pending",
            },
            source_refs=tuple(item.annotation_id for item in selected) or (trajectory.trajectory_id,),
        ),
    )


def write_projection_view(
    records: tuple[ProjectionRecord, ...],
    *,
    view: ProjectionView,
    output_dir: Path,
) -> ProjectionManifest:
    if any(record.view is not view for record in records):
        raise ValueError("projection records contain a different view")
    ordered = tuple(sorted(records, key=lambda item: item.record_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"{view.value}.jsonl.gz"
    with records_path.open("wb") as raw_sink:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_sink,
            mtime=0,
        ) as sink:
            for record in ordered:
                sink.write(f"{canonical_json(record)}\n".encode("utf-8"))
    split_counts = Counter(record.split.value for record in ordered)
    manifest = ProjectionManifest(
        schema_version=PROJECTION_SCHEMA_VERSION,
        view=view,
        record_count=len(ordered),
        split_counts=tuple((split.value, split_counts[split.value]) for split in CorpusSplit),
        master_trajectory_hashes=tuple(sorted({record.master_trajectory_hash for record in ordered})),
        records_uri=records_path.name,
        records_sha256=sha256_file(records_path),
    )
    manifest_path = output_dir / f"{view.value}.manifest.json"
    manifest_path.write_text(f"{canonical_json(manifest)}\n", encoding="utf-8")
    return manifest


def load_master_run(run_root: Path) -> tuple[ExperienceTrajectory, ...]:
    master_root = run_root / "master"
    shard_paths = tuple(sorted(master_root.glob("shard-*.jsonl.gz")))
    if not shard_paths:
        raise FileNotFoundError(f"no master shards under {master_root}")
    trajectories: list[ExperienceTrajectory] = []
    from .canonical import trajectory_from_json

    for shard_path in shard_paths:
        with gzip.open(shard_path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    trajectories.append(trajectory_from_json(line))
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid trajectory in {shard_path.name}:{line_number}") from error
    return tuple(trajectories)


def project_master_run(
    run_root: Path,
    *,
    output_root: Path | None = None,
    human_review_sample_rate: float = 0.05,
) -> tuple[ProjectionManifest, ...]:
    destination = output_root or run_root / "projections"
    return _project_master_run_streaming(
        run_root,
        destination=destination,
        human_review_sample_rate=human_review_sample_rate,
    )


def _project_master_run_streaming(
    run_root: Path,
    *,
    destination: Path,
    human_review_sample_rate: float,
) -> tuple[ProjectionManifest, ...]:
    from .audit import AuditCheck

    destination.mkdir(parents=True, exist_ok=True)
    relationship_root = destination / "relationship_encoder_dataset"
    for stale in relationship_root.rglob("*.trajectory.json"):
        stale.unlink()

    record_counts = {view: 0 for view in ProjectionView}
    split_counts = {view: Counter() for view in ProjectionView}
    master_hashes = {view: set() for view in ProjectionView}
    leakage = {view: [] for view in ProjectionView}
    hash_failures = {view: [] for view in ProjectionView}
    relationship_families = {split: set() for split in CorpusSplit}
    relationship_records: list[dict[str, str]] = []
    record_paths = {view: destination / f"{view.value}.jsonl.gz" for view in ProjectionView}

    with ExitStack() as stack:
        sinks = {}
        for view, path in record_paths.items():
            raw_sink = stack.enter_context(path.open("wb"))
            sinks[view] = stack.enter_context(
                gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw_sink,
                    mtime=0,
                )
            )

        for trajectory, trajectory_hash in _iter_master_run(run_root):
            _PRECOMPUTED_TRAJECTORY_HASHES[trajectory.trajectory_id] = trajectory_hash
            _trajectory_hash.cache_clear()
            try:
                for view in ProjectionView:
                    records = _project_trajectory_for_view(
                        trajectory,
                        view=view,
                        human_review_sample_rate=(human_review_sample_rate),
                    )
                    for record in records:
                        if record.view is not view:
                            raise ValueError("projection record view mismatch")
                        if (
                            view
                            in {
                                ProjectionView.EVALUATION_ONLY,
                                ProjectionView.HUMAN_REVIEW_QUEUE,
                            }
                            and record.training_use is TrainingUse.TARGET
                        ):
                            leakage[view].append(record.record_id)
                        payload = json.loads(record.payload_json)
                        source = payload.get("source") if isinstance(payload, dict) else None
                        if (
                            source
                            in {
                                AnnotationSource.MODEL_PREDICTION.value,
                                AnnotationSource.EVALUATION_READOUT.value,
                            }
                            and record.training_use is TrainingUse.TARGET
                        ):
                            leakage[view].append(record.record_id)
                        if len(record.master_trajectory_hash) != 64:
                            hash_failures[view].append(record.record_id)
                        sinks[view].write(f"{canonical_json(record)}\n".encode("utf-8"))
                        record_counts[view] += 1
                        split_counts[view][record.split.value] += 1
                        master_hashes[view].add(record.master_trajectory_hash)

                projected = project_relationship_encoder(trajectory)
                projected_split = _relationship_split(trajectory.family)
                relationship_families[projected_split].add(trajectory.family)
                filename = hashlib.sha256(projected.trajectory_id.encode("utf-8")).hexdigest() + ".trajectory.json"
                split_root = relationship_root / projected_split.value
                split_root.mkdir(parents=True, exist_ok=True)
                path = split_root / filename
                path.write_text(
                    f"{relationship_json(projected)}\n",
                    encoding="utf-8",
                )
                relationship_records.append(
                    {
                        "master_trajectory_hash": trajectory_hash,
                        "projected_trajectory_id": (projected.trajectory_id),
                        "projected_split": projected_split.value,
                        "uri": path.relative_to(relationship_root).as_posix(),
                    }
                )
            finally:
                _PRECOMPUTED_TRAJECTORY_HASHES.pop(
                    trajectory.trajectory_id,
                    None,
                )
                _trajectory_hash.cache_clear()

    projection_audits = {}
    manifests: list[ProjectionManifest] = []
    for view in ProjectionView:
        checks = (
            AuditCheck(
                check_id="projection_eval_training_isolation",
                passed=not leakage[view],
                hard_gate=True,
                observed=len(leakage[view]),
                expected="0 eval/model records used as targets",
                details_json=canonical_json({"violations": leakage[view]}),
            ),
            AuditCheck(
                check_id="projection_master_hash_lineage",
                passed=not hash_failures[view],
                hard_gate=True,
                observed=len(hash_failures[view]),
                expected="all records cite a master SHA-256",
                details_json=canonical_json({"violations": hash_failures[view]}),
            ),
        )
        failed = tuple(check.check_id for check in checks if check.hard_gate and not check.passed)
        if failed:
            raise ValueError(f"projection hard gates failed for {view.value}: " + ", ".join(failed))
        projection_audits[view.value] = checks
        manifest = ProjectionManifest(
            schema_version=PROJECTION_SCHEMA_VERSION,
            view=view,
            record_count=record_counts[view],
            split_counts=tuple((split.value, split_counts[view][split.value]) for split in CorpusSplit),
            master_trajectory_hashes=tuple(sorted(master_hashes[view])),
            records_uri=record_paths[view].name,
            records_sha256=sha256_file(record_paths[view]),
        )
        (destination / f"{view.value}.manifest.json").write_text(
            f"{canonical_json(manifest)}\n",
            encoding="utf-8",
        )
        manifests.append(manifest)

    for left in CorpusSplit:
        for right in CorpusSplit:
            if left.value >= right.value:
                continue
            overlap = relationship_families[left] & relationship_families[right]
            if overlap:
                raise ValueError(f"relationship encoder family split leakage: {sorted(overlap)}")
    (relationship_root / "split-manifest.json").write_text(
        canonical_json(
            {
                "schema_version": PROJECTION_SCHEMA_VERSION,
                "policy": "whole_scenario_family",
                "split_families": {split.value: sorted(relationship_families[split]) for split in CorpusSplit},
                "records": relationship_records,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "projection-audit.json").write_text(
        canonical_json(
            {
                "schema_version": PROJECTION_SCHEMA_VERSION,
                "passed": True,
                "views": projection_audits,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return tuple(manifests)


def _iter_master_run(
    run_root: Path,
):
    from .canonical import trajectory_from_json

    shard_paths = tuple(sorted((run_root / "master").glob("shard-*.jsonl.gz")))
    if not shard_paths:
        raise FileNotFoundError(f"no master shards under {run_root / 'master'}")
    for shard_path in shard_paths:
        with gzip.open(shard_path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    trajectory = trajectory_from_json(payload)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid trajectory in {shard_path.name}:{line_number}") from error
                yield (
                    trajectory,
                    hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                )


def _project_trajectory_for_view(
    trajectory: ExperienceTrajectory,
    *,
    view: ProjectionView,
    human_review_sample_rate: float,
) -> tuple[ProjectionRecord, ...]:
    if view is ProjectionView.RELATIONSHIP_ENCODER:
        return (project_relationship_record(trajectory),)
    if view is ProjectionView.EXPRESSION_SFT:
        if trajectory.generation_tier is not GenerationTier.RENDERED:
            return ()
        return project_expression_sft(trajectory)
    if view is ProjectionView.SEMANTIC_OWNER:
        return project_semantic_owner(trajectory)
    if view is ProjectionView.SOCIAL_COGNITION:
        return project_social_cognition(trajectory)
    if view is ProjectionView.MEMORY_RETRIEVAL:
        return project_memory_retrieval(trajectory)
    if view is ProjectionView.TEMPORAL_SSL:
        if trajectory.generation_tier is not GenerationTier.LIVE_THROUGH:
            return ()
        return project_temporal_ssl(trajectory)
    if view is ProjectionView.INTERNAL_RL:
        if trajectory.generation_tier is not GenerationTier.LIVE_THROUGH:
            return ()
        return project_internal_rl(trajectory)
    if view is ProjectionView.EVALUATION_ONLY:
        return project_evaluation_only(trajectory)
    if view is ProjectionView.HUMAN_REVIEW_QUEUE:
        return project_human_review_queue(
            trajectory,
            sample_rate=human_review_sample_rate,
        )
    raise AssertionError(f"unhandled projection view {view!r}")


@lru_cache(maxsize=20_000)
def _trajectory_hash(trajectory: ExperienceTrajectory) -> str:
    precomputed = _PRECOMPUTED_TRAJECTORY_HASHES.get(trajectory.trajectory_id)
    return precomputed if precomputed is not None else stable_hash(trajectory)


def write_relationship_encoder_layout(
    trajectories: tuple[ExperienceTrajectory, ...],
    *,
    output_root: Path,
) -> Path:
    """Write the exact train/val/test document layout companion-encoder reads."""

    split_families: dict[CorpusSplit, set[str]] = {split: set() for split in CorpusSplit}
    records: list[dict[str, str]] = []
    for trajectory in sorted(
        trajectories,
        key=lambda item: item.trajectory_id,
    ):
        split = _relationship_split(trajectory.family)
        split_families[split].add(trajectory.family)
        projected = project_relationship_encoder(trajectory)
        filename = hashlib.sha256(projected.trajectory_id.encode("utf-8")).hexdigest() + ".trajectory.json"
        split_root = output_root / split.value
        split_root.mkdir(parents=True, exist_ok=True)
        path = split_root / filename
        path.write_text(f"{relationship_json(projected)}\n", encoding="utf-8")
        records.append(
            {
                "master_trajectory_hash": _trajectory_hash(trajectory),
                "projected_trajectory_id": projected.trajectory_id,
                "projected_split": split.value,
                "uri": path.relative_to(output_root).as_posix(),
            }
        )
    for left in CorpusSplit:
        for right in CorpusSplit:
            if left.value >= right.value:
                continue
            overlap = split_families[left] & split_families[right]
            if overlap:
                raise ValueError(f"relationship encoder family split leakage: {sorted(overlap)}")
    manifest_path = output_root / "split-manifest.json"
    manifest_path.write_text(
        canonical_json(
            {
                "schema_version": PROJECTION_SCHEMA_VERSION,
                "policy": "whole_scenario_family",
                "split_families": {split.value: sorted(split_families[split]) for split in CorpusSplit},
                "records": records,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _project_annotations(
    trajectory: ExperienceTrajectory,
    *,
    view: ProjectionView,
    owner_names: frozenset[str],
) -> tuple[ProjectionRecord, ...]:
    records: list[ProjectionRecord] = []
    for annotation in trajectory.annotations:
        if annotation.target_owner not in owner_names:
            continue
        if annotation.source not in {
            AnnotationSource.GENERATOR_TRUTH,
            AnnotationSource.ENVIRONMENT_FACT,
            AnnotationSource.HUMAN_ANNOTATION,
        }:
            continue
        records.append(
            _record(
                trajectory,
                view=view,
                suffix=annotation.annotation_id,
                training_use=annotation.training_use,
                payload={
                    "target_owner": annotation.target_owner,
                    "target_ref": annotation.target_ref,
                    "ontology": annotation.ontology,
                    "ontology_version": annotation.ontology_version,
                    "label_key": annotation.label_key,
                    "label_value": json.loads(annotation.label_value_json),
                    "track": annotation.track.value,
                    "timescale": annotation.timescale.value,
                    "scope_ids": annotation.scope_ids,
                    "source": annotation.source.value,
                    "confidence": annotation.confidence,
                },
                source_refs=(annotation.annotation_id,) + annotation.evidence_refs,
            )
        )
    return tuple(records)


def _record(
    trajectory: ExperienceTrajectory,
    *,
    view: ProjectionView,
    suffix: str,
    training_use: TrainingUse,
    payload: object,
    source_refs: tuple[str, ...],
    split_override: CorpusSplit | None = None,
) -> ProjectionRecord:
    return ProjectionRecord(
        schema_version=PROJECTION_SCHEMA_VERSION,
        view=view,
        record_id=f"{view.value}:{trajectory.trajectory_id}:{suffix}",
        master_trajectory_id=trajectory.trajectory_id,
        master_trajectory_hash=_trajectory_hash(trajectory),
        split=split_override or trajectory.split,
        training_use=training_use,
        payload_json=canonical_json(payload),
        source_refs=source_refs,
    )


def _turn_by_id(trajectory: ExperienceTrajectory, turn_id: str):
    matches = tuple(turn for session in trajectory.sessions for turn in session.turns if turn.turn_id == turn_id)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one turn for {turn_id!r}")
    return matches[0]


def _is_generator_truth(annotation: AnnotationRecord) -> bool:
    return annotation.source is AnnotationSource.GENERATOR_TRUTH


def _relationship_phase(family: str, phase_id: str) -> RelationshipPhase:
    if family == "rupture_repair":
        return {
            "opening": RelationshipPhase.ESTABLISHING,
            "development": RelationshipPhase.RUPTURED,
            "outcome": RelationshipPhase.REPAIR_WINDOW,
            "reflection": RelationshipPhase.REPAIRED,
        }[phase_id]
    if family == "absence_reengagement":
        return {
            "opening": RelationshipPhase.ESTABLISHED,
            "development": RelationshipPhase.DORMANT,
            "outcome": RelationshipPhase.RE_ENGAGED,
            "reflection": RelationshipPhase.RE_ENGAGED,
        }[phase_id]
    if family in {
        "boundary_consent_autonomy",
        "safety_adversarial_resilience",
    } and phase_id in {"development", "outcome"}:
        return RelationshipPhase.BOUNDARY_TESTED
    if phase_id == "opening":
        return RelationshipPhase.ESTABLISHING
    return RelationshipPhase.ESTABLISHED


def _relationship_split(family: str) -> CorpusSplit:
    if family in _RELATIONSHIP_TEST_FAMILIES:
        return CorpusSplit.TEST
    if family in _RELATIONSHIP_VAL_FAMILIES:
        return CorpusSplit.VAL
    return CorpusSplit.TRAIN


def _relationship_readouts(
    phase: RelationshipPhase,
) -> tuple[float, float, float]:
    return {
        RelationshipPhase.ESTABLISHING: (0.45, 0.35, 0.10),
        RelationshipPhase.ESTABLISHED: (0.72, 0.78, 0.05),
        RelationshipPhase.RUPTURED: (0.22, 0.48, 0.95),
        RelationshipPhase.REPAIR_WINDOW: (0.35, 0.55, 0.72),
        RelationshipPhase.REPAIRED: (0.62, 0.70, 0.25),
        RelationshipPhase.RE_ENGAGED: (0.58, 0.68, 0.15),
        RelationshipPhase.DORMANT: (0.48, 0.42, 0.10),
        RelationshipPhase.BOUNDARY_TESTED: (0.42, 0.60, 0.45),
    }[phase]


def _metadata_value(trajectory: ExperienceTrajectory, key: str) -> str:
    matches = tuple(item.value for item in trajectory.metadata if item.key == key)
    if len(matches) != 1:
        raise ValueError(f"trajectory metadata requires exactly one {key!r}")
    return matches[0]


def _snapshot_payload(frame) -> dict[str, object]:
    return {
        "snapshot_id": frame.snapshot_id,
        "slot_name": frame.slot_name,
        "owner": frame.owner,
        "version": frame.version,
        "timestamp_ms": frame.timestamp_ms,
        "value_hash": frame.value_hash,
        "payload": json.loads(frame.payload_json),
        "wiring_level": frame.wiring_level,
    }


def _require_live_through(
    trajectory: ExperienceTrajectory,
    *,
    view: ProjectionView,
) -> None:
    if trajectory.generation_tier is not GenerationTier.LIVE_THROUGH:
        raise ValueError(f"{view.value} requires live-through runtime observations")
    if not trajectory.snapshot_frames:
        raise ValueError(f"{view.value} requires non-empty snapshot_frames")


__all__ = [
    "PROJECTION_SCHEMA_VERSION",
    "ProjectionManifest",
    "ProjectionRecord",
    "ProjectionView",
    "project_evaluation_only",
    "project_expression_sft",
    "project_human_review_queue",
    "project_internal_rl",
    "project_memory_retrieval",
    "project_master_run",
    "project_relationship_encoder",
    "project_relationship_record",
    "project_semantic_owner",
    "project_social_cognition",
    "project_temporal_ssl",
    "write_relationship_encoder_layout",
    "write_projection_view",
    "load_master_run",
]
