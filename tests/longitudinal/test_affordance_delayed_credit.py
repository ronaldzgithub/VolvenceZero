"""Packet B (long-horizon-closure) — segment closure credit attributes to tool.

Two structural assertions on ``derive_segment_closure_credit_records``:

1. **Happy path**: when PE action context carries ``segment_id`` /
   ``affordance_name`` / ``prediction_id`` AND the temporal snapshot
   has the matching closed segment, the resulting credit record's
   ``context`` field must contain both ``affordance_name=...`` and
   ``prediction_id=...`` substrings, so reflection / replay can
   attribute the segment credit to a specific tool call by id.

2. **Mismatch safety**: when PE context references a ``segment_id``
   that is NOT in the temporal snapshot's ``closed_segments`` (e.g.
   the segment closed earlier and got rotated out), the helper MUST
   return an empty tuple, not ``None``. This is the bug fix in
   Packet B (the previous code returned ``None`` while the signature
   declared ``tuple[CreditRecord, ...]`` — consumers ``extend(...)``
   the result and would crash on ``None``).

Plus a thin chained-turn assertion via ``derive_credit_records_from_snapshots``:
the main credit pipeline must include at least one record from the
segment helper when the temporal/PE pair is consistent — it must not
silently swallow the helper's output.
"""

from __future__ import annotations

import pytest

from volvence_zero.credit.gate import (
    derive_credit_records_from_prediction_error_first,
    derive_segment_closure_credit_records,
)
from volvence_zero.dual_track import DualTrackSnapshot, TrackState
from volvence_zero.memory import Track
from volvence_zero.evaluation import EvaluationSnapshot
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.temporal import (
    ControllerState,
    TemporalAbstractionSnapshot,
    TemporalSegmentClosure,
)


def _build_pe_snapshot(
    *,
    segment_id: str,
    affordance_name: str,
    prediction_id: str,
    abstract_action_id: str = "use-tool",
    action_error: float = 0.6,
) -> PredictionErrorSnapshot:
    context = PredictionActionContext(
        segment_id=segment_id,
        abstract_action_id=abstract_action_id,
        z_t_digest=(0.3, 0.7),
        regime_id="exploration",
        affordance_name=affordance_name,
        environment_event_id="env-evt-1",
        environment_outcome_id="env-out-1",
        prediction_id=prediction_id,
    )
    actual = ActualOutcome(
        observed_turn_index=5,
        task_progress=0.4,
        relationship_delta=0.0,
        regime_stability=0.7,
        action_payoff=0.4,
        description="actual outcome after tool call",
        action_context=context,
    )
    next_pred = PredictedOutcome(
        source_turn_index=5,
        target_turn_index=6,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.0,
        predicted_regime_stability=0.7,
        predicted_action_payoff=0.5,
        confidence=0.7,
        description="prediction for next turn",
        action_context=context,
    )
    error = PredictionError(
        task_error=0.2,
        relationship_error=0.0,
        regime_error=0.1,
        action_error=action_error,
        magnitude=action_error,
        signed_reward=-action_error,
        description=f"PE for tool {affordance_name} (plan {prediction_id})",
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=None,
        actual_outcome=actual,
        next_prediction=next_pred,
        error=error,
        turn_index=5,
        bootstrap=False,
        description=f"PE snapshot for {affordance_name}",
        action_context=context,
    )


def _build_temporal_snapshot_with_segment(
    *,
    segment_id: str,
    affordance_name: str,
    abstract_action_id: str = "use-tool",
) -> TemporalAbstractionSnapshot:
    closure = TemporalSegmentClosure(
        segment_id=segment_id,
        open_turn_index=1,
        close_turn_index=4,
        abstract_action_id=abstract_action_id,
        z_t_digest=(0.3, 0.7),
        beta_open_digest=0.7,
        beta_close_digest=0.9,
        affordance_name=affordance_name,
        description=f"closed segment for {affordance_name}",
    )
    return TemporalAbstractionSnapshot(
        controller_state=ControllerState(
            code=(0.3, 0.7),
            code_dim=2,
            switch_gate=0.9,
            is_switching=True,
            steps_since_switch=0,
        ),
        active_abstract_action="next-action",
        controller_params_hash="hash-1",
        description="temporal snapshot with one closed segment",
        closed_segments=(closure,),
    )


def test_segment_credit_context_carries_affordance_name_and_prediction_id() -> None:
    """Happy path: matched closed segment -> credit context contains
    affordance_name=<x> AND prediction_id=<y> substrings.
    """
    pe = _build_pe_snapshot(
        segment_id="segment-tool-1",
        affordance_name="read_file",
        prediction_id="p-T1",
    )
    temporal = _build_temporal_snapshot_with_segment(
        segment_id="segment-tool-1",
        affordance_name="read_file",
    )

    records = derive_segment_closure_credit_records(
        prediction_error_snapshot=pe,
        temporal_snapshot=temporal,
        timestamp_ms=42,
    )

    assert len(records) == 1, "matched segment must yield exactly one record"
    record = records[0]
    assert record.level == "abstract_action_segment"
    assert record.source_event == "segment:segment-tool-1"
    assert "affordance_name=read_file" in record.context, (
        f"affordance lineage missing from context; got: {record.context!r}"
    )
    assert "prediction_id=p-T1" in record.context, (
        f"prediction_id lineage missing from context; got: {record.context!r}"
    )
    assert "abstract_action=use-tool" in record.context
    assert record.timestamp_ms == 42


def test_segment_credit_returns_empty_tuple_on_mismatch_not_none() -> None:
    """Bug fix verification: when PE context references a segment_id
    not present in temporal.closed_segments, return () not None.
    Before Packet B, this branch returned ``None`` and consumers'
    ``records.extend(...)`` would crash with TypeError.
    """
    pe = _build_pe_snapshot(
        segment_id="segment-not-in-temporal",
        affordance_name="grep",
        prediction_id="p-orphan",
    )
    # Temporal snapshot has a different segment id closed
    temporal = _build_temporal_snapshot_with_segment(
        segment_id="segment-something-else",
        affordance_name="run_test",
    )

    result = derive_segment_closure_credit_records(
        prediction_error_snapshot=pe,
        temporal_snapshot=temporal,
        timestamp_ms=0,
    )

    assert result is not None, "Packet B fix: must NOT return None"
    assert result == (), f"mismatch must yield empty tuple, got: {result!r}"
    # Critical: extend must not crash. This is the actual regression
    # vector the bug would have caused at runtime.
    bag: list = []
    bag.extend(result)
    assert bag == []


def test_main_credit_pipeline_includes_segment_record_when_matched() -> None:
    """``derive_credit_records_from_prediction_error_first`` is the
    main entry consumed by CreditModule. Its records must include at
    least one segment closure record when the PE/temporal pair
    matches — this protects against accidentally dropping the helper's
    output (e.g. by reverting the empty-tuple fix and starting to crash
    silently behind ``records.extend(None)``).
    """
    pe = _build_pe_snapshot(
        segment_id="segment-main-1",
        affordance_name="run_test",
        prediction_id="p-main-1",
    )
    temporal = _build_temporal_snapshot_with_segment(
        segment_id="segment-main-1",
        affordance_name="run_test",
    )
    dual_track = DualTrackSnapshot(
        world_track=TrackState(
            track=Track.WORLD,
            active_goals=(),
            recent_credits=(),
            controller_code=(0.3, 0.4, 0.5),
            tension_level=0.0,
        ),
        self_track=TrackState(
            track=Track.SELF,
            active_goals=(),
            recent_credits=(),
            controller_code=(0.2, 0.3, 0.4),
            tension_level=0.0,
        ),
        cross_track_tension=0.1,
        description="dual track snapshot",
    )
    evaluation = EvaluationSnapshot(
        turn_scores=(),
        session_scores=(),
        alerts=(),
        description="empty eval for credit smoke",
    )

    records = derive_credit_records_from_prediction_error_first(
        prediction_error_snapshot=pe,
        dual_track_snapshot=dual_track,
        evaluation_snapshot=evaluation,
        temporal_snapshot=temporal,
        timestamp_ms=99,
    )

    segment_records = tuple(
        r for r in records if r.level == "abstract_action_segment"
    )
    assert len(segment_records) >= 1, (
        "Main credit pipeline must surface at least one segment record "
        "when PE/temporal are consistent; got none."
    )
    record = segment_records[0]
    # Lineage must survive through the main pipeline, not just the
    # helper in isolation.
    assert "affordance_name=run_test" in record.context
    assert "prediction_id=p-main-1" in record.context


@pytest.mark.parametrize(
    "affordance_name, prediction_id",
    [
        ("write_file", "p-write-001"),
        ("grep", ""),  # empty prediction_id is legal (back-compat)
        ("", "p-no-affordance"),  # empty affordance_name is legal too
    ],
)
def test_segment_credit_context_handles_optional_lineage_fields(
    affordance_name: str, prediction_id: str,
) -> None:
    """Lineage fields are optional — the helper must still produce
    a record with the empty-string substring present, so downstream
    grep can detect "this credit had no tool / no prediction id"
    explicitly rather than missing the marker entirely.
    """
    pe = _build_pe_snapshot(
        segment_id="segment-optional",
        affordance_name=affordance_name,
        prediction_id=prediction_id,
    )
    temporal = _build_temporal_snapshot_with_segment(
        segment_id="segment-optional",
        affordance_name=affordance_name or "fallback",
    )

    records = derive_segment_closure_credit_records(
        prediction_error_snapshot=pe,
        temporal_snapshot=temporal,
        timestamp_ms=0,
    )

    assert len(records) == 1
    ctx = records[0].context
    assert f"affordance_name={affordance_name}" in ctx
    assert f"prediction_id={prediction_id}" in ctx
