"""CP-10: pre-action prediction ids are PE-owner-issued, callers only forward.

The affordance tool loop's ``plan_ref`` must come from the prediction id the
single ``prediction_error`` owner stamped on its published ``next_prediction``
— not from a caller-fabricated reference. Unknown lineage stays explicitly
``None``.
"""

from __future__ import annotations

from types import SimpleNamespace

from lifeform_affordance.tool_loop import ToolLoopOrchestrator as ToolLoopRunner


def _session_with(snapshots: dict) -> SimpleNamespace:
    return SimpleNamespace(latest_active_snapshots=snapshots)


def _pe_snapshot(prediction_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        value=SimpleNamespace(next_prediction=SimpleNamespace(prediction_id=prediction_id))
    )


def _temporal_snapshot(segment_ids: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        value=SimpleNamespace(
            closed_segments=tuple(
                SimpleNamespace(segment_id=segment_id) for segment_id in segment_ids
            )
        )
    )


def test_plan_ref_forwards_pe_owner_issued_prediction_id() -> None:
    session = _session_with(
        {
            "prediction_error": _pe_snapshot("pe:prediction_error:turn-3:next"),
            "temporal_abstraction": _temporal_snapshot(("seg-1",)),
        }
    )
    assert (
        ToolLoopRunner._plan_ref_from_snapshots(session=session)
        == "pe:prediction_error:turn-3:next"
    )


def test_plan_ref_falls_back_to_segment_then_explicit_unknown() -> None:
    # Owner-issued id empty (bootstrap) -> segment anchor.
    session = _session_with(
        {
            "prediction_error": _pe_snapshot(""),
            "temporal_abstraction": _temporal_snapshot(("seg-1", "seg-2")),
        }
    )
    assert ToolLoopRunner._plan_ref_from_snapshots(session=session) == "seg-2"
    # Nothing available -> None (explicit unknown), never a fabricated id.
    assert ToolLoopRunner._plan_ref_from_snapshots(session=_session_with({})) is None


async def test_published_next_prediction_carries_owner_issued_id() -> None:
    from volvence_zero.agent.session import AgentSessionRunner

    runner = AgentSessionRunner(rare_heavy_enabled=False)
    result = await runner.run_turn("Chart the route and check the weather window.")
    next_prediction = result.active_snapshots["prediction_error"].value.next_prediction
    assert next_prediction.prediction_id.startswith("pe:prediction_error:turn-")
