# Copyright 2026 Companion Standard Contributors
# Licensed under the Apache License, Version 2.0.

"""ArcRecord -> canonical InteractionTrajectory exporter.

Consumes a Companion Bench :class:`companion_bench.arc_runner.ArcRecord`
(the bench's own transcript format) and produces a
``companion_standard.InteractionTrajectory`` with FSM-ground-truth labels.

Coordinate mapping: the bench uses 1-based ``(session, turn)`` where one
"turn" is a user+assistant pair; the standard uses 0-based sessions and a
flat 0-based turn list per session where user and assistant utterances are
separate turns. A bench user turn ``t`` (1-based) anchors at trajectory
turn index ``(t - 1) * 2``.

Every exported trajectory passes the companion_standard conformance
round-trip check before it is returned — a generator that emits
non-conformant documents fails loudly at generation time, not at training
time.
"""

from __future__ import annotations

import json
import pathlib

from companion_bench.arc_runner import ArcRecord
from companion_bench.spec import ScenarioSpec

from companion_standard import (
    SCHEMA_VERSION,
    InteractionTrajectory,
    TrajectorySession,
    TrajectorySource,
    TrajectoryTurn,
    TurnRole,
    to_canonical_json,
    trajectory_hash,
)
from companion_standard.conformance import check_trajectory_roundtrip
from companion_trajgen.labeler import label_arc


def arc_record_to_trajectory(
    *,
    record: ArcRecord,
    spec: ScenarioSpec,
    source: TrajectorySource,
) -> InteractionTrajectory:
    """Convert one arc transcript + its scenario spec into a labelled trajectory."""

    sessions: list[TrajectorySession] = []
    turns_per_session: list[int] = []
    session_gaps: list[int] = []
    for session_position, arc_session in enumerate(record.sessions):
        flat_turns: list[TrajectoryTurn] = []
        for arc_turn in arc_session.turns:
            flat_turns.append(
                TrajectoryTurn(
                    turn_index=len(flat_turns),
                    role=TurnRole.USER,
                    text=arc_turn.user_text,
                )
            )
            flat_turns.append(
                TrajectoryTurn(
                    turn_index=len(flat_turns),
                    role=TurnRole.ASSISTANT,
                    text=arc_turn.assistant_text,
                )
            )
        sessions.append(
            TrajectorySession(
                session_index=session_position,
                gap_days_before=arc_session.inter_session_gap_days,
                turns=tuple(flat_turns),
            )
        )
        turns_per_session.append(len(flat_turns))
        session_gaps.append(arc_session.inter_session_gap_days)

    # FSM anchors in trajectory coordinates. Only steps that actually fired
    # (i.e. fall within the drawn turn count) appear in the transcript, so
    # we read the fired actions back from the ArcRecord itself rather than
    # the spec — the record is the ground truth of what was enacted.
    fsm_steps: list[tuple[int, int, str]] = []
    for session_position, arc_session in enumerate(record.sessions):
        for arc_turn in arc_session.turns:
            if arc_turn.fsm_action is None:
                continue
            fsm_steps.append(
                (session_position, (arc_turn.turn_index - 1) * 2, arc_turn.fsm_action)
            )

    labels = label_arc(
        fsm_steps=tuple(fsm_steps),
        session_gap_days=tuple(session_gaps),
        turns_per_session=tuple(turns_per_session),
    )

    trajectory = InteractionTrajectory(
        trajectory_id=record.arc_id,
        schema_version=SCHEMA_VERSION,
        source=source,
        family=record.family,
        scenario_ref=record.scenario_hash,
        sessions=tuple(sessions),
        labels=labels,
        metadata=(
            ("scenario_id", record.scenario_id),
            ("paraphrase_seed", str(record.paraphrase_seed)),
            ("sut_model_id", record.sut_model_id),
            ("user_simulator_model", record.user_simulator_model),
        ),
    )
    check_trajectory_roundtrip(trajectory)
    return trajectory


def write_trajectory(
    trajectory: InteractionTrajectory, out_dir: pathlib.Path | str
) -> pathlib.Path:
    """Write canonical JSON to ``<out_dir>/<trajectory_id>.trajectory.json``."""
    directory = pathlib.Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f"{trajectory.trajectory_id}.trajectory.json"
    out_path.write_text(to_canonical_json(trajectory) + "\n", encoding="utf-8")
    return out_path


def trajectory_manifest_entry(trajectory: InteractionTrajectory) -> dict:
    """Compact manifest line for dataset audit (hash-citable, body-free)."""
    return {
        "trajectory_id": trajectory.trajectory_id,
        "trajectory_hash": trajectory_hash(trajectory),
        "family": trajectory.family,
        "scenario_ref": trajectory.scenario_ref,
        "source": trajectory.source.value,
        "session_count": len(trajectory.sessions),
        "label_count": len(trajectory.labels),
    }


def write_manifest(entries: list[dict], out_dir: pathlib.Path | str) -> pathlib.Path:
    directory = pathlib.Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / "manifest.json"
    out_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


__all__ = [
    "arc_record_to_trajectory",
    "trajectory_manifest_entry",
    "write_manifest",
    "write_trajectory",
]
