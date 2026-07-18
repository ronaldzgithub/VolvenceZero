# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""Canonical interaction trajectory schema.

The unit of exchange for relationship-representation models: a multi-session
human-AI interaction transcript plus per-segment relationship-state labels.
This is the input format for any encoder implementing the standard and the
output format for any trajectory generator (e.g. ``companion-trajgen``).

Label provenance rule (mirrors R12 "evaluation is read-only"):

* ``LabelSource.FSM_GROUND_TRUTH`` — labels emitted by the generating FSM at
  synthesis time. The only source allowed in open training sets.
* ``LabelSource.HUMAN_ANNOTATION`` — consented human labels.
* ``LabelSource.MODEL_PREDICTION`` — model output. Never a training label
  for the model that produced it; carried only for comparison artifacts.

Judge scores are NOT a label source — there is deliberately no enum member
for them, so a pipeline that tries to distil judge output into labels fails
at the type level rather than by convention.

Validation is fail-loud: schema violations raise ``ValueError`` with the
``invalid_trajectory:`` prefix (mirroring Companion Bench's
``invalid_scenario:`` convention) so CLIs can map them to actionable
messages. No defensive defaults for required fields.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from companion_standard.canonical import stable_hash

SCHEMA_VERSION = 1


class TurnRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class TrajectorySource(str, enum.Enum):
    """Where the trajectory came from (data-boundary provenance)."""

    SYNTHETIC_FSM = "synthetic_fsm"          # deterministic FSM simulator
    SYNTHETIC_LLM = "synthetic_llm"          # LLM user-simulator + SUT
    CONSENTED_FIRST_PARTY = "consented_first_party"  # explicit consent on record


class LabelSource(str, enum.Enum):
    """Provenance of a relationship-state label. Judge output is deliberately
    NOT representable here (see module docstring)."""

    FSM_GROUND_TRUTH = "fsm_ground_truth"
    HUMAN_ANNOTATION = "human_annotation"
    MODEL_PREDICTION = "model_prediction"


class RelationshipPhase(str, enum.Enum):
    """Closed vocabulary for the coarse relationship phase of a segment.

    Chosen to be derivable from generation-time FSM state (the Companion
    Bench action vocabulary) without judge involvement, while remaining
    meaningful for consented real-world annotation.
    """

    ESTABLISHING = "establishing"        # early pattern/preference building
    ESTABLISHED = "established"          # stable baseline rapport
    RUPTURED = "ruptured"                # unrepaired rupture is live
    REPAIR_WINDOW = "repair_window"      # user is testing a repair attempt
    REPAIRED = "repaired"                # repair landed, trust recovering
    RE_ENGAGED = "re_engaged"            # user re-engaged after repair
    DORMANT = "dormant"                  # long absence / disengagement
    BOUNDARY_TESTED = "boundary_tested"  # boundary / safety pressure active


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid_trajectory: {message}")


def _require_unit(name: str, value: float) -> None:
    _require(
        isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0,
        f"{name} must be in [0, 1], got {value!r}",
    )


@dataclass(frozen=True)
class TrajectoryTurn:
    """One utterance. ``turn_index`` is zero-based within the session."""

    turn_index: int
    role: TurnRole
    text: str

    def __post_init__(self) -> None:
        _require(self.turn_index >= 0, f"turn_index must be >= 0, got {self.turn_index}")
        _require(isinstance(self.role, TurnRole), f"role must be TurnRole, got {self.role!r}")
        _require(bool(self.text.strip()), "turn text must be non-empty")


@dataclass(frozen=True)
class TrajectorySession:
    """One session. ``gap_days_before`` is the simulated/real wall-clock gap
    since the previous session (0 for the first session)."""

    session_index: int
    gap_days_before: int
    turns: tuple[TrajectoryTurn, ...]

    def __post_init__(self) -> None:
        _require(self.session_index >= 0, f"session_index must be >= 0, got {self.session_index}")
        _require(self.gap_days_before >= 0, f"gap_days_before must be >= 0, got {self.gap_days_before}")
        _require(len(self.turns) > 0, f"session {self.session_index} has no turns")
        for position, turn in enumerate(self.turns):
            _require(
                turn.turn_index == position,
                f"session {self.session_index} turn_index {turn.turn_index} != position {position}",
            )


@dataclass(frozen=True)
class RelationshipStateLabel:
    """Relationship state at an anchor point ``(session_index, turn_index)``.

    A label applies from its anchor until the next label's anchor (or the
    trajectory end). Numeric fields are the label-compatible subset of the
    ``relationship_state`` owner snapshot readouts, all clamped to [0, 1].
    """

    session_index: int
    turn_index: int
    phase: RelationshipPhase
    trust_level: float
    continuity_level: float
    repair_pressure: float
    source: LabelSource
    evidence: str = ""

    def __post_init__(self) -> None:
        _require(self.session_index >= 0, f"label session_index must be >= 0, got {self.session_index}")
        _require(self.turn_index >= 0, f"label turn_index must be >= 0, got {self.turn_index}")
        _require(isinstance(self.phase, RelationshipPhase), f"phase must be RelationshipPhase, got {self.phase!r}")
        _require_unit("trust_level", self.trust_level)
        _require_unit("continuity_level", self.continuity_level)
        _require_unit("repair_pressure", self.repair_pressure)
        _require(isinstance(self.source, LabelSource), f"source must be LabelSource, got {self.source!r}")
        if self.source is LabelSource.HUMAN_ANNOTATION:
            _require(bool(self.evidence.strip()), "human_annotation labels must carry evidence")


@dataclass(frozen=True)
class InteractionTrajectory:
    """A complete multi-session trajectory with relationship-state labels.

    ``scenario_ref`` cites the generating scenario by stable hash (never by
    body) so held-out provenance stays auditable without disclosure.
    ``family`` is a free-form generator-defined grouping key used for
    train/val splits (whole families are assigned to exactly one split).
    """

    trajectory_id: str
    schema_version: int
    source: TrajectorySource
    family: str
    scenario_ref: str
    sessions: tuple[TrajectorySession, ...]
    labels: tuple[RelationshipStateLabel, ...]
    metadata: tuple[tuple[str, str], ...] = field(default=())

    def __post_init__(self) -> None:
        _require(bool(self.trajectory_id.strip()), "trajectory_id must be non-empty")
        _require(
            self.schema_version == SCHEMA_VERSION,
            f"schema_version must be {SCHEMA_VERSION}, got {self.schema_version}",
        )
        _require(isinstance(self.source, TrajectorySource), f"source must be TrajectorySource, got {self.source!r}")
        _require(bool(self.family.strip()), "family must be non-empty")
        _require(bool(self.scenario_ref.strip()), "scenario_ref must be non-empty")
        _require(len(self.sessions) > 0, "trajectory has no sessions")
        for position, session in enumerate(self.sessions):
            _require(
                session.session_index == position,
                f"session_index {session.session_index} != position {position}",
            )
        _require(
            self.sessions[0].gap_days_before == 0,
            "first session must have gap_days_before == 0",
        )
        turn_counts = {session.session_index: len(session.turns) for session in self.sessions}
        anchors: list[tuple[int, int]] = []
        for label in self.labels:
            _require(
                label.session_index in turn_counts,
                f"label anchored to unknown session {label.session_index}",
            )
            _require(
                label.turn_index < turn_counts[label.session_index],
                f"label anchored past end of session {label.session_index} "
                f"(turn {label.turn_index} >= {turn_counts[label.session_index]})",
            )
            anchors.append((label.session_index, label.turn_index))
        _require(
            anchors == sorted(anchors),
            "labels must be sorted by (session_index, turn_index)",
        )
        _require(
            len(anchors) == len(set(anchors)),
            "labels must have unique (session_index, turn_index) anchors",
        )


def trajectory_hash(trajectory: InteractionTrajectory) -> str:
    """Stable SHA-256 over the canonical JSON of the trajectory."""
    return stable_hash(trajectory)


def trajectory_from_jsonable(data: dict) -> InteractionTrajectory:
    """Rebuild an :class:`InteractionTrajectory` from canonical-JSON data.

    Fail-loud inverse of :func:`companion_standard.canonical.to_jsonable`:
    missing or mistyped fields raise ``KeyError`` / ``ValueError`` with the
    ``invalid_trajectory:`` prefix from the dataclass validators. Unknown
    extra keys are rejected so producer typos cannot silently pass
    conformance.
    """

    known_keys = {
        "trajectory_id", "schema_version", "source", "family",
        "scenario_ref", "sessions", "labels", "metadata",
    }
    extra = set(data) - known_keys
    _require(not extra, f"unknown trajectory keys: {sorted(extra)}")

    sessions = tuple(
        TrajectorySession(
            session_index=session["session_index"],
            gap_days_before=session["gap_days_before"],
            turns=tuple(
                TrajectoryTurn(
                    turn_index=turn["turn_index"],
                    role=TurnRole(turn["role"]),
                    text=turn["text"],
                )
                for turn in session["turns"]
            ),
        )
        for session in data["sessions"]
    )
    labels = tuple(
        RelationshipStateLabel(
            session_index=label["session_index"],
            turn_index=label["turn_index"],
            phase=RelationshipPhase(label["phase"]),
            trust_level=label["trust_level"],
            continuity_level=label["continuity_level"],
            repair_pressure=label["repair_pressure"],
            source=LabelSource(label["source"]),
            evidence=label.get("evidence", ""),
        )
        for label in data["labels"]
    )
    metadata = tuple(
        (str(key), str(value)) for key, value in data.get("metadata", ())
    )
    return InteractionTrajectory(
        trajectory_id=data["trajectory_id"],
        schema_version=data["schema_version"],
        source=TrajectorySource(data["source"]),
        family=data["family"],
        scenario_ref=data["scenario_ref"],
        sessions=sessions,
        labels=labels,
        metadata=metadata,
    )
