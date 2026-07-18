# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""Conformance kit — checks a producer / consumer can run against the standard.

A third party integrating with the Relationship Representation Standard
runs these checks to verify:

1. **Type conformance** — every value type in the standard is a frozen
   dataclass (immutability is normative, not advisory).
2. **Registry conformance** — the semantic owner slot registry is the
   nine canonical slots, unchanged.
3. **Trajectory conformance** — a produced trajectory document survives
   the canonical JSON round-trip with an identical stable hash, and
   validates against the schema (fail-loud, ``invalid_trajectory:``).

Usage::

    from companion_standard.conformance import (
        check_standard_self, check_trajectory_document,
    )

    check_standard_self()                       # library self-check
    check_trajectory_document(json_text)        # validate a produced doc

Each check raises ``ConformanceError`` with an actionable message on the
first violation; a clean run returns quietly.
"""

from __future__ import annotations

import dataclasses
import json

from companion_standard.canonical import stable_hash, to_canonical_json, to_jsonable
from companion_standard.kernel import Snapshot
from companion_standard.owner_prediction import OwnerPredictionSignal
from companion_standard.semantic_state import (
    SEMANTIC_OWNER_SLOTS,
    BeliefAssumptionSnapshot,
    BoundaryConsentSnapshot,
    CommitmentSnapshot,
    ExecutionResultSnapshot,
    GoalValueSnapshot,
    OpenLoopSnapshot,
    PlanIntentSnapshot,
    RelationshipStateSnapshot,
    SemanticRecord,
    UserModelSnapshot,
)
from companion_standard.social_cognition import OtherMindRecord
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


class ConformanceError(AssertionError):
    """Raised when a conformance check fails."""


_CANONICAL_SLOTS: tuple[str, ...] = (
    "plan_intent",
    "commitment",
    "open_loop",
    "user_model",
    "execution_result",
    "belief_assumption",
    "relationship_state",
    "goal_value",
    "boundary_consent",
)

STANDARD_VALUE_TYPES: tuple[type, ...] = (
    SemanticRecord,
    PlanIntentSnapshot,
    CommitmentSnapshot,
    OpenLoopSnapshot,
    UserModelSnapshot,
    ExecutionResultSnapshot,
    BeliefAssumptionSnapshot,
    RelationshipStateSnapshot,
    GoalValueSnapshot,
    BoundaryConsentSnapshot,
    OwnerPredictionSignal,
    OtherMindRecord,
    Snapshot,
    TrajectoryTurn,
    TrajectorySession,
    RelationshipStateLabel,
    InteractionTrajectory,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConformanceError(message)


def check_value_types_frozen() -> None:
    """Every standard value type must be a frozen dataclass."""
    for value_type in STANDARD_VALUE_TYPES:
        _require(
            dataclasses.is_dataclass(value_type),
            f"{value_type.__name__} is not a dataclass",
        )
        _require(
            bool(value_type.__dataclass_params__.frozen),
            f"{value_type.__name__} is not frozen; immutability is normative",
        )


def check_slot_registry() -> None:
    """The semantic owner slot registry must be the nine canonical slots."""
    _require(
        SEMANTIC_OWNER_SLOTS == _CANONICAL_SLOTS,
        "SEMANTIC_OWNER_SLOTS deviates from the canonical nine slots: "
        f"{SEMANTIC_OWNER_SLOTS!r}",
    )


def example_trajectory() -> InteractionTrajectory:
    """A minimal valid trajectory (used by the self-check and as a template)."""
    return InteractionTrajectory(
        trajectory_id="conformance-example-001",
        schema_version=SCHEMA_VERSION,
        source=TrajectorySource.SYNTHETIC_FSM,
        family="F2",
        scenario_ref="0" * 64,
        sessions=(
            TrajectorySession(
                session_index=0,
                gap_days_before=0,
                turns=(
                    TrajectoryTurn(0, TurnRole.USER, "I told her I felt invisible at the dinner."),
                    TrajectoryTurn(1, TurnRole.ASSISTANT, "That sounds painful — do you want to talk it through?"),
                ),
            ),
            TrajectorySession(
                session_index=1,
                gap_days_before=2,
                turns=(
                    TrajectoryTurn(0, TurnRole.USER, "Whatever. It does not matter."),
                    TrajectoryTurn(1, TurnRole.ASSISTANT, "Last time we spoke it clearly mattered a lot."),
                ),
            ),
        ),
        labels=(
            RelationshipStateLabel(
                session_index=0,
                turn_index=0,
                phase=RelationshipPhase.ESTABLISHING,
                trust_level=0.5,
                continuity_level=0.5,
                repair_pressure=0.0,
                source=LabelSource.FSM_GROUND_TRUTH,
            ),
            RelationshipStateLabel(
                session_index=1,
                turn_index=0,
                phase=RelationshipPhase.RUPTURED,
                trust_level=0.3,
                continuity_level=0.5,
                repair_pressure=0.8,
                source=LabelSource.FSM_GROUND_TRUTH,
            ),
        ),
    )


def check_trajectory_roundtrip(trajectory: InteractionTrajectory) -> None:
    """A trajectory must survive canonical JSON round-trip hash-identically."""
    original_hash = trajectory_hash(trajectory)
    rebuilt = trajectory_from_jsonable(json.loads(to_canonical_json(trajectory)))
    rebuilt_hash = trajectory_hash(rebuilt)
    _require(
        original_hash == rebuilt_hash,
        "canonical JSON round-trip changed the trajectory hash "
        f"({original_hash} -> {rebuilt_hash}); serialisation is lossy",
    )


def check_trajectory_document(json_text: str) -> InteractionTrajectory:
    """Validate a producer's trajectory JSON document.

    Returns the parsed trajectory on success so callers can chain checks.
    Raises ``ConformanceError`` (wrapping the schema's fail-loud
    ``ValueError``) on any violation.
    """
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ConformanceError(f"document is not valid JSON: {error}") from error
    try:
        trajectory = trajectory_from_jsonable(data)
    except (KeyError, TypeError, ValueError) as error:
        raise ConformanceError(f"document violates the trajectory schema: {error}") from error
    check_trajectory_roundtrip(trajectory)
    return trajectory


def check_standard_self() -> None:
    """Full library self-check: types, registry, example round-trip."""
    check_value_types_frozen()
    check_slot_registry()
    example = example_trajectory()
    check_trajectory_roundtrip(example)
    # jsonable output of a snapshot value must be JSON-serialisable
    json.dumps(to_jsonable(example))
    _require(
        stable_hash(example) == trajectory_hash(example),
        "trajectory_hash must equal stable_hash on the same object",
    )


__all__ = [
    "ConformanceError",
    "STANDARD_VALUE_TYPES",
    "check_slot_registry",
    "check_standard_self",
    "check_trajectory_document",
    "check_trajectory_roundtrip",
    "check_value_types_frozen",
    "example_trajectory",
]
