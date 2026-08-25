"""Immutable contracts for the unified synthetic experience corpus.

The corpus is an offline artifact, not a runtime owner or slot.  Its master
record keeps generator truth, rendered text, and runtime observations in
separate fields so a consumer cannot accidentally treat a model readout as
ground truth.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

SCHEMA_VERSION = "synthetic-experience.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


class CorpusSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class GenerationTier(str, Enum):
    STRUCTURAL = "structural"
    RENDERED = "rendered"
    LIVE_THROUGH = "live_through"


class TurnRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AnnotationSource(str, Enum):
    GENERATOR_TRUTH = "generator_truth"
    ENVIRONMENT_FACT = "environment_fact"
    RUNTIME_SNAPSHOT = "runtime_snapshot"
    HUMAN_ANNOTATION = "human_annotation"
    MODEL_PREDICTION = "model_prediction"
    EVALUATION_READOUT = "evaluation_readout"


class TrainingUse(str, Enum):
    TARGET = "target"
    FEATURE_ONLY = "feature_only"
    EVAL_ONLY = "eval_only"
    QUARANTINED = "quarantined"


class Track(str, Enum):
    WORLD = "world"
    SELF = "self"
    DUAL = "dual"
    SHARED = "shared"
    GROUP = "group"


class Timescale(str, Enum):
    ONLINE_FAST = "online-fast"
    SESSION_MEDIUM = "session-medium"
    BACKGROUND_SLOW = "background-slow"
    RARE_HEAVY = "rare-heavy"
    TURN = "turn"
    SESSION = "session"
    CROSS_SESSION = "cross_session"


class AdjudicationStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CONFLICTED = "conflicted"


class QualitySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class KeyValue:
    key: str
    value: str

    def __post_init__(self) -> None:
        _require_non_empty("KeyValue.key", self.key)


@dataclass(frozen=True)
class CountEntry:
    key: str
    count: int

    def __post_init__(self) -> None:
        _require_non_empty("CountEntry.key", self.key)
        _require_non_negative("CountEntry.count", self.count)


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    uri: str
    sha256: str
    mime_type: str
    license_id: str

    def __post_init__(self) -> None:
        _require_id("ArtifactRef.artifact_id", self.artifact_id)
        _require_non_empty("ArtifactRef.kind", self.kind)
        _require_non_empty("ArtifactRef.uri", self.uri)
        _require_sha256("ArtifactRef.sha256", self.sha256)
        _require_non_empty("ArtifactRef.mime_type", self.mime_type)
        _require_non_empty("ArtifactRef.license_id", self.license_id)


@dataclass(frozen=True)
class AnnotationRecord:
    annotation_id: str
    target_ref: str
    ontology: str
    ontology_version: str
    label_key: str
    label_value_json: str
    source: AnnotationSource
    training_use: TrainingUse
    confidence: float
    evidence_refs: tuple[str, ...]
    target_owner: str | None
    track: Track
    timescale: Timescale
    scope_ids: tuple[str, ...] = ()
    adjudication: AdjudicationStatus = AdjudicationStatus.NOT_REQUIRED
    annotator_id: str | None = None

    def __post_init__(self) -> None:
        _require_id("AnnotationRecord.annotation_id", self.annotation_id)
        _require_non_empty("AnnotationRecord.target_ref", self.target_ref)
        _require_non_empty("AnnotationRecord.ontology", self.ontology)
        _require_non_empty("AnnotationRecord.ontology_version", self.ontology_version)
        _require_non_empty("AnnotationRecord.label_key", self.label_key)
        _require_json("AnnotationRecord.label_value_json", self.label_value_json)
        _require_unit("AnnotationRecord.confidence", self.confidence)
        _require_unique_non_empty("AnnotationRecord.evidence_refs", self.evidence_refs)
        _require_unique_non_empty("AnnotationRecord.scope_ids", self.scope_ids)
        if self.target_owner is not None:
            _require_non_empty("AnnotationRecord.target_owner", self.target_owner)
        if self.annotator_id is not None:
            _require_non_empty("AnnotationRecord.annotator_id", self.annotator_id)
        if (
            self.source
            in {
                AnnotationSource.MODEL_PREDICTION,
                AnnotationSource.EVALUATION_READOUT,
            }
            and self.training_use is TrainingUse.TARGET
        ):
            raise ValueError("model_prediction and evaluation_readout annotations cannot be training targets")
        if self.source is AnnotationSource.HUMAN_ANNOTATION:
            if self.annotator_id is None:
                raise ValueError("human_annotation requires annotator_id")
            if not self.evidence_refs:
                raise ValueError("human_annotation requires evidence_refs")
        elif self.annotator_id is not None:
            raise ValueError("annotator_id is only valid for human_annotation")


@dataclass(frozen=True)
class SnapshotFrame:
    snapshot_id: str
    turn_ref: str
    slot_name: str
    owner: str
    version: int
    timestamp_ms: int
    value_type: str
    value_hash: str
    payload_json: str
    wiring_level: str
    description: str

    def __post_init__(self) -> None:
        _require_id("SnapshotFrame.snapshot_id", self.snapshot_id)
        _require_non_empty("SnapshotFrame.turn_ref", self.turn_ref)
        _require_non_empty("SnapshotFrame.slot_name", self.slot_name)
        _require_non_empty("SnapshotFrame.owner", self.owner)
        _require_non_negative("SnapshotFrame.version", self.version)
        _require_non_negative("SnapshotFrame.timestamp_ms", self.timestamp_ms)
        _require_non_empty("SnapshotFrame.value_type", self.value_type)
        _require_sha256("SnapshotFrame.value_hash", self.value_hash)
        _require_json("SnapshotFrame.payload_json", self.payload_json)
        if self.wiring_level not in {"active", "shadow", "disabled"}:
            raise ValueError("SnapshotFrame.wiring_level must be active, shadow, or disabled")


@dataclass(frozen=True)
class LatentTruthFrame:
    frame_id: str
    turn_ref: str
    phase_id: str
    event_kind: str
    observable_facts: tuple[KeyValue, ...]
    private_facts: tuple[KeyValue, ...]
    response_contract: tuple[str, ...]
    annotation_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id("LatentTruthFrame.frame_id", self.frame_id)
        _require_non_empty("LatentTruthFrame.turn_ref", self.turn_ref)
        _require_non_empty("LatentTruthFrame.phase_id", self.phase_id)
        _require_non_empty("LatentTruthFrame.event_kind", self.event_kind)
        _require_unique_keys("LatentTruthFrame.observable_facts", self.observable_facts)
        _require_unique_keys("LatentTruthFrame.private_facts", self.private_facts)
        _require_unique_non_empty("LatentTruthFrame.response_contract", self.response_contract)
        _require_unique_non_empty("LatentTruthFrame.annotation_refs", self.annotation_refs)


@dataclass(frozen=True)
class ExperienceTurn:
    turn_id: str
    session_index: int
    turn_index: int
    role: TurnRole
    text: str
    event_id: str | None
    latent_frame_ref: str | None
    snapshot_refs: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    metadata: tuple[KeyValue, ...] = ()

    def __post_init__(self) -> None:
        _require_id("ExperienceTurn.turn_id", self.turn_id)
        _require_non_negative("ExperienceTurn.session_index", self.session_index)
        _require_non_negative("ExperienceTurn.turn_index", self.turn_index)
        _require_non_empty("ExperienceTurn.text", self.text)
        if self.event_id is not None:
            _require_non_empty("ExperienceTurn.event_id", self.event_id)
        if self.latent_frame_ref is not None:
            _require_non_empty("ExperienceTurn.latent_frame_ref", self.latent_frame_ref)
        _require_unique_non_empty("ExperienceTurn.snapshot_refs", self.snapshot_refs)
        _require_unique_non_empty("ExperienceTurn.artifact_refs", self.artifact_refs)
        _require_unique_keys("ExperienceTurn.metadata", self.metadata)


@dataclass(frozen=True)
class ExperienceSession:
    session_id: str
    session_index: int
    gap_days_before: int
    turns: tuple[ExperienceTurn, ...]

    def __post_init__(self) -> None:
        _require_id("ExperienceSession.session_id", self.session_id)
        _require_non_negative("ExperienceSession.session_index", self.session_index)
        _require_non_negative("ExperienceSession.gap_days_before", self.gap_days_before)
        if not self.turns:
            raise ValueError("ExperienceSession.turns must be non-empty")
        for position, turn in enumerate(self.turns):
            if turn.session_index != self.session_index:
                raise ValueError("turn session_index must match containing session")
            if turn.turn_index != position:
                raise ValueError(f"turn_index {turn.turn_index} must equal position {position}")


@dataclass(frozen=True)
class ProvenanceRecord:
    run_id: str
    source_kind: str
    generator_version: str
    seed: int
    scenario_hash: str
    git_sha: str
    model_id: str | None
    prompt_hash: str | None
    created_at: str
    license_id: str
    consent_basis: str

    def __post_init__(self) -> None:
        _require_id("ProvenanceRecord.run_id", self.run_id)
        _require_non_empty("ProvenanceRecord.source_kind", self.source_kind)
        _require_non_empty("ProvenanceRecord.generator_version", self.generator_version)
        _require_non_negative("ProvenanceRecord.seed", self.seed)
        _require_sha256("ProvenanceRecord.scenario_hash", self.scenario_hash)
        _require_non_empty("ProvenanceRecord.git_sha", self.git_sha)
        if self.model_id is not None:
            _require_non_empty("ProvenanceRecord.model_id", self.model_id)
        if self.prompt_hash is not None:
            _require_sha256("ProvenanceRecord.prompt_hash", self.prompt_hash)
        _require_non_empty("ProvenanceRecord.created_at", self.created_at)
        _require_non_empty("ProvenanceRecord.license_id", self.license_id)
        _require_non_empty("ProvenanceRecord.consent_basis", self.consent_basis)


@dataclass(frozen=True)
class QualityRecord:
    quality_id: str
    check_kind: str
    passed: bool
    severity: QualitySeverity
    score: float | None
    evidence_refs: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        _require_id("QualityRecord.quality_id", self.quality_id)
        _require_non_empty("QualityRecord.check_kind", self.check_kind)
        if self.score is not None:
            _require_unit("QualityRecord.score", self.score)
        _require_unique_non_empty("QualityRecord.evidence_refs", self.evidence_refs)
        _require_non_empty("QualityRecord.description", self.description)
        if self.severity is QualitySeverity.ERROR and not self.passed:
            return


@dataclass(frozen=True)
class ExperienceTrajectory:
    schema_version: str
    trajectory_id: str
    scenario_ref: str
    scenario_hash: str
    split: CorpusSplit
    family: str
    language: str
    generation_tier: GenerationTier
    sessions: tuple[ExperienceSession, ...]
    truth_frames: tuple[LatentTruthFrame, ...]
    snapshot_frames: tuple[SnapshotFrame, ...]
    annotations: tuple[AnnotationRecord, ...]
    artifacts: tuple[ArtifactRef, ...]
    quality: tuple[QualityRecord, ...]
    provenance: ProvenanceRecord
    metadata: tuple[KeyValue, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}")
        _require_id("ExperienceTrajectory.trajectory_id", self.trajectory_id)
        _require_non_empty("ExperienceTrajectory.scenario_ref", self.scenario_ref)
        _require_sha256("ExperienceTrajectory.scenario_hash", self.scenario_hash)
        _require_non_empty("ExperienceTrajectory.family", self.family)
        if self.language not in {"zh", "en", "bilingual"}:
            raise ValueError("language must be zh, en, or bilingual")
        if not self.sessions:
            raise ValueError("ExperienceTrajectory.sessions must be non-empty")
        for position, session in enumerate(self.sessions):
            if session.session_index != position:
                raise ValueError(f"session_index {session.session_index} must equal position {position}")
        if self.sessions[0].gap_days_before != 0:
            raise ValueError("first session gap_days_before must be 0")
        _require_unique_attr("truth_frames", self.truth_frames, "frame_id")
        _require_unique_attr("snapshot_frames", self.snapshot_frames, "snapshot_id")
        _require_unique_attr("annotations", self.annotations, "annotation_id")
        _require_unique_attr("artifacts", self.artifacts, "artifact_id")
        _require_unique_attr("quality", self.quality, "quality_id")
        _require_unique_keys("ExperienceTrajectory.metadata", self.metadata)
        turn_ids = {turn.turn_id for session in self.sessions for turn in session.turns}
        truth_ids = {frame.frame_id for frame in self.truth_frames}
        snapshot_ids = {frame.snapshot_id for frame in self.snapshot_frames}
        annotation_ids = {item.annotation_id for item in self.annotations}
        artifact_ids = {item.artifact_id for item in self.artifacts}
        for frame in self.truth_frames:
            if frame.turn_ref not in turn_ids:
                raise ValueError(f"truth frame references unknown turn {frame.turn_ref!r}")
            if not set(frame.annotation_refs).issubset(annotation_ids):
                raise ValueError("truth frame references unknown annotation")
        for frame in self.snapshot_frames:
            if frame.turn_ref not in turn_ids:
                raise ValueError(f"snapshot frame references unknown turn {frame.turn_ref!r}")
        for session in self.sessions:
            for turn in session.turns:
                if turn.latent_frame_ref is not None and turn.latent_frame_ref not in truth_ids:
                    raise ValueError("turn references unknown latent truth frame")
                if not set(turn.snapshot_refs).issubset(snapshot_ids):
                    raise ValueError("turn references unknown snapshot")
                if not set(turn.artifact_refs).issubset(artifact_ids):
                    raise ValueError("turn references unknown artifact")


@dataclass(frozen=True)
class ScenarioBlueprint:
    scenario_id: str
    family: str
    split: CorpusSplit
    language: str
    domain: str
    difficulty: str
    risk_level: str
    title: str
    sessions: int
    turns_per_session: tuple[int, ...]
    persona_id: str
    latent_arc_id: str
    regime_candidates: tuple[str, ...]
    track: Track
    timescale: Timescale
    path_id: str
    arc_spec_id: str
    semantic_routing_json: str
    observable_facts: tuple[str, ...]
    private_truth: tuple[str, ...]
    response_contract: tuple[str, ...]
    counterfactual_mutations: tuple[str, ...]
    safety_constraints: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id("ScenarioBlueprint.scenario_id", self.scenario_id)
        _require_non_empty("ScenarioBlueprint.family", self.family)
        if self.language not in {"zh", "en", "bilingual"}:
            raise ValueError("ScenarioBlueprint.language must be zh, en, or bilingual")
        _require_non_empty("ScenarioBlueprint.domain", self.domain)
        if self.difficulty not in {
            "basic",
            "intermediate",
            "advanced",
            "adversarial",
            "medium",
            "hard",
            "expert",
        }:
            raise ValueError("unsupported ScenarioBlueprint.difficulty")
        if self.risk_level not in {"low", "medium", "high", "critical"}:
            raise ValueError("unsupported ScenarioBlueprint.risk_level")
        _require_non_empty("ScenarioBlueprint.title", self.title)
        if self.sessions < 1:
            raise ValueError("ScenarioBlueprint.sessions must be >= 1")
        if len(self.turns_per_session) != self.sessions:
            raise ValueError("turns_per_session length must match sessions")
        if any(count < 1 for count in self.turns_per_session):
            raise ValueError("turns_per_session values must be >= 1")
        _require_id("ScenarioBlueprint.persona_id", self.persona_id)
        _require_id("ScenarioBlueprint.latent_arc_id", self.latent_arc_id)
        _require_unique_non_empty("ScenarioBlueprint.regime_candidates", self.regime_candidates)
        _require_id("ScenarioBlueprint.path_id", self.path_id)
        _require_id("ScenarioBlueprint.arc_spec_id", self.arc_spec_id)
        _require_json("ScenarioBlueprint.semantic_routing_json", self.semantic_routing_json)
        _require_non_empty_items("ScenarioBlueprint.observable_facts", self.observable_facts)
        _require_non_empty_items("ScenarioBlueprint.private_truth", self.private_truth)
        _require_non_empty_items("ScenarioBlueprint.response_contract", self.response_contract)
        _require_unique_non_empty(
            "ScenarioBlueprint.counterfactual_mutations",
            self.counterfactual_mutations,
        )
        _require_unique_non_empty("ScenarioBlueprint.safety_constraints", self.safety_constraints)


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: str
    corpus_id: str
    run_id: str
    generated_at: str
    generator_version: str
    git_sha: str
    scenario_package_hash: str
    generation_tier: GenerationTier
    trajectory_count: int
    split_counts: tuple[CountEntry, ...]
    family_counts: tuple[CountEntry, ...]
    model_ids: tuple[str, ...]
    prompt_hashes: tuple[KeyValue, ...]
    shard_refs: tuple[ArtifactRef, ...]
    quality: tuple[QualityRecord, ...]
    description: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("CorpusManifest.schema_version mismatch")
        _require_id("CorpusManifest.corpus_id", self.corpus_id)
        _require_id("CorpusManifest.run_id", self.run_id)
        _require_non_empty("CorpusManifest.generated_at", self.generated_at)
        _require_non_empty("CorpusManifest.generator_version", self.generator_version)
        _require_non_empty("CorpusManifest.git_sha", self.git_sha)
        _require_sha256("CorpusManifest.scenario_package_hash", self.scenario_package_hash)
        _require_non_negative("CorpusManifest.trajectory_count", self.trajectory_count)
        _require_unique_attr("split_counts", self.split_counts, "key")
        _require_unique_attr("family_counts", self.family_counts, "key")
        _require_unique_non_empty("CorpusManifest.model_ids", self.model_ids)
        _require_unique_keys("CorpusManifest.prompt_hashes", self.prompt_hashes)
        _require_unique_attr("shard_refs", self.shard_refs, "artifact_id")
        _require_unique_attr("quality", self.quality, "quality_id")
        _require_non_empty("CorpusManifest.description", self.description)
        if sum(entry.count for entry in self.split_counts) != self.trajectory_count:
            raise ValueError("split_counts must sum to trajectory_count")
        if sum(entry.count for entry in self.family_counts) != self.trajectory_count:
            raise ValueError("family_counts must sum to trajectory_count")


def _require_non_empty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_non_empty_items(field_name: str, values: tuple[str, ...]) -> None:
    for value in values:
        _require_non_empty(field_name, value)


def _require_unique_non_empty(field_name: str, values: tuple[str, ...]) -> None:
    _require_non_empty_items(field_name, values)
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} entries must be unique")


def _require_id(field_name: str, value: str) -> None:
    _require_non_empty(field_name, value)
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"{field_name} has invalid characters: {value!r}")


def _require_sha256(field_name: str, value: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _require_non_negative(field_name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_unit(field_name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")


def _require_json(field_name: str, value: str) -> None:
    try:
        json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field_name} must contain valid JSON") from error


def _require_unique_keys(field_name: str, values: tuple[KeyValue, ...]) -> None:
    keys = tuple(value.key for value in values)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} keys must be unique")


def _require_unique_attr(field_name: str, values: tuple[object, ...], attribute: str) -> None:
    resolved: list[str] = []
    for value in values:
        try:
            raw = object.__getattribute__(value, attribute)
        except AttributeError as error:
            raise ValueError(f"{field_name} item lacks required attribute {attribute!r}") from error
        if not isinstance(raw, str):
            raise ValueError(f"{field_name}.{attribute} must be a string")
        resolved.append(raw)
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"{field_name}.{attribute} values must be unique")


__all__ = [
    "SCHEMA_VERSION",
    "AdjudicationStatus",
    "AnnotationRecord",
    "AnnotationSource",
    "ArtifactRef",
    "CorpusManifest",
    "CorpusSplit",
    "CountEntry",
    "ExperienceSession",
    "ExperienceTrajectory",
    "ExperienceTurn",
    "GenerationTier",
    "KeyValue",
    "LatentTruthFrame",
    "ProvenanceRecord",
    "QualityRecord",
    "QualitySeverity",
    "ScenarioBlueprint",
    "SnapshotFrame",
    "Timescale",
    "Track",
    "TrainingUse",
    "TurnRole",
]
