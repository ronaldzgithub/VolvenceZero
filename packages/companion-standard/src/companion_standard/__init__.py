# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""companion-standard: the Relationship Representation Standard.

Typed, immutable, zero-dependency schema for long-horizon human-AI
relationship state. See README.md and the public RFC
(``relationship-representation-rfc-v0.md``).
"""

from companion_standard.canonical import stable_hash, to_canonical_json, to_jsonable
from companion_standard.embedding import (
    SemanticEmbeddingBackend,
    stub_cosine_similarity,
    stub_semantic_embedding,
    stub_semantic_tokens,
)
from companion_standard.kernel import Snapshot
from companion_standard.owner_prediction import OwnerPredictionKind, OwnerPredictionSignal
from companion_standard.semantic_state import (
    SELF_SEMANTIC_OWNER_SLOTS,
    SEMANTIC_OWNER_SLOTS,
    WORLD_SEMANTIC_OWNER_SLOTS,
    BeliefAssumptionSnapshot,
    BoundaryConsentSnapshot,
    CommitmentSnapshot,
    ExecutionResultSnapshot,
    GoalValueSnapshot,
    OpenLoopSnapshot,
    PlanIntentSnapshot,
    RelationshipStateSnapshot,
    SemanticRecord,
    SemanticSnapshotValue,
    UserModelSnapshot,
)
from companion_standard.social_cognition import (
    OtherMindRecord,
    OtherMindRecordKind,
    OtherMindRecordStatus,
)
from companion_standard.trajectory import (
    SCHEMA_VERSION,
    InteractionTrajectory,
    LabelSource,
    RelationshipPhase,
    RelationshipStateLabel,
    TrajectorySession,
    TrajectorySource,
    TrajectoryTurn,
    TurnRole,
    trajectory_from_jsonable,
    trajectory_hash,
)

__all__ = [
    "SCHEMA_VERSION",
    "SELF_SEMANTIC_OWNER_SLOTS",
    "SEMANTIC_OWNER_SLOTS",
    "WORLD_SEMANTIC_OWNER_SLOTS",
    "BeliefAssumptionSnapshot",
    "BoundaryConsentSnapshot",
    "CommitmentSnapshot",
    "ExecutionResultSnapshot",
    "GoalValueSnapshot",
    "InteractionTrajectory",
    "LabelSource",
    "OpenLoopSnapshot",
    "OtherMindRecord",
    "OtherMindRecordKind",
    "OtherMindRecordStatus",
    "OwnerPredictionKind",
    "OwnerPredictionSignal",
    "PlanIntentSnapshot",
    "RelationshipPhase",
    "RelationshipStateLabel",
    "RelationshipStateSnapshot",
    "SemanticEmbeddingBackend",
    "SemanticRecord",
    "SemanticSnapshotValue",
    "Snapshot",
    "TrajectorySession",
    "TrajectorySource",
    "TrajectoryTurn",
    "TurnRole",
    "UserModelSnapshot",
    "stable_hash",
    "stub_cosine_similarity",
    "stub_semantic_embedding",
    "stub_semantic_tokens",
    "to_canonical_json",
    "to_jsonable",
    "trajectory_from_jsonable",
    "trajectory_hash",
]
