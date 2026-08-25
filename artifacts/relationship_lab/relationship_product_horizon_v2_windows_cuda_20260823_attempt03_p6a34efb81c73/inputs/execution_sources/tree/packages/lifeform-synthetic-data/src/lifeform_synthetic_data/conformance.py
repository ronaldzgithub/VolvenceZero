"""Small executable conformance fixture for the v1 corpus contract."""

from __future__ import annotations

import json

from .canonical import canonical_json, stable_hash, trajectory_from_json
from .contracts import (
    SCHEMA_VERSION,
    AnnotationRecord,
    AnnotationSource,
    CorpusSplit,
    ExperienceSession,
    ExperienceTrajectory,
    ExperienceTurn,
    GenerationTier,
    KeyValue,
    LatentTruthFrame,
    ProvenanceRecord,
    QualityRecord,
    QualitySeverity,
    Timescale,
    Track,
    TrainingUse,
    TurnRole,
)


def build_conformance_trajectory() -> ExperienceTrajectory:
    scenario_ref = "conformance.relationship_continuity.001"
    scenario_hash = stable_hash({"scenario_id": scenario_ref, "version": "unified_v1"})
    annotation = AnnotationRecord(
        annotation_id="ann:conformance:001",
        target_ref="turn:0:0",
        ontology="volvence.relationship_state",
        ontology_version="1.0.0",
        label_key="continuity_acknowledged",
        label_value_json="true",
        source=AnnotationSource.GENERATOR_TRUTH,
        training_use=TrainingUse.TARGET,
        confidence=1.0,
        evidence_refs=("truth:0:0",),
        target_owner="relationship_state",
        track=Track.SELF,
        timescale=Timescale.TURN,
        scope_ids=("user:synthetic:001",),
    )
    truth_frame = LatentTruthFrame(
        frame_id="truth:0:0",
        turn_ref="turn:0:0",
        phase_id="continuity_recall",
        event_kind="user_message",
        observable_facts=(KeyValue(key="prior_topic", value="joint reading plan"),),
        private_facts=(KeyValue(key="user_goal", value="verify cross-session continuity"),),
        response_contract=(
            "acknowledge prior context without inventing details",
            "ask one scoped continuation question",
        ),
        annotation_refs=(annotation.annotation_id,),
    )
    sessions = (
        ExperienceSession(
            session_id="session:0",
            session_index=0,
            gap_days_before=0,
            turns=(
                ExperienceTurn(
                    turn_id="turn:0:0",
                    session_index=0,
                    turn_index=0,
                    role=TurnRole.USER,
                    text="我们继续上次约定的阅读计划。",
                    event_id="event:0:0",
                    latent_frame_ref=truth_frame.frame_id,
                ),
                ExperienceTurn(
                    turn_id="turn:0:1",
                    session_index=0,
                    turn_index=1,
                    role=TurnRole.ASSISTANT,
                    text="可以。我们之前把计划限定为共同阅读；你想先延续哪一部分？",
                    event_id=None,
                    latent_frame_ref=None,
                ),
            ),
        ),
    )
    return ExperienceTrajectory(
        schema_version=SCHEMA_VERSION,
        trajectory_id="trajectory:conformance:001",
        scenario_ref=scenario_ref,
        scenario_hash=scenario_hash,
        split=CorpusSplit.TRAIN,
        family="relationship_continuity",
        language="zh",
        generation_tier=GenerationTier.STRUCTURAL,
        sessions=sessions,
        truth_frames=(truth_frame,),
        snapshot_frames=(),
        annotations=(annotation,),
        artifacts=(),
        quality=(
            QualityRecord(
                quality_id="quality:conformance:001",
                check_kind="contract_round_trip",
                passed=True,
                severity=QualitySeverity.INFO,
                score=1.0,
                evidence_refs=("trajectory:conformance:001",),
                description="Built from the checked-in v1 conformance fixture.",
            ),
        ),
        provenance=ProvenanceRecord(
            run_id="run:conformance:001",
            source_kind="synthetic_fsm",
            generator_version="0.1.0",
            seed=0,
            scenario_hash=scenario_hash,
            git_sha="conformance-fixture",
            model_id=None,
            prompt_hash=None,
            created_at="2026-07-20T00:00:00Z",
            license_id="Proprietary-Synthetic-v1",
            consent_basis="fully_synthetic",
        ),
        metadata=(KeyValue(key="fixture", value="true"),),
    )


def assert_v1_conformance() -> str:
    trajectory = build_conformance_trajectory()
    payload = canonical_json(trajectory)
    reconstructed = trajectory_from_json(payload)
    if reconstructed != trajectory:
        raise AssertionError("trajectory round-trip changed the immutable value")
    if canonical_json(reconstructed) != payload:
        raise AssertionError("canonical JSON is not byte-stable after round-trip")

    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise AssertionError("canonical trajectory root must be an object")
    decoded["unknown_field"] = True
    unknown_rejected = False
    try:
        trajectory_from_json(json.dumps(decoded))
    except ValueError:
        unknown_rejected = True
    if not unknown_rejected:
        raise AssertionError("strict deserialization accepted an unknown field")
    return stable_hash(trajectory)


__all__ = ["assert_v1_conformance", "build_conformance_trajectory"]
