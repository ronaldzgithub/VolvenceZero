"""End-to-end: IngestionPipeline drives a real LifeformSession.

Validates that the Gap 3 slice-1 wiring actually produces kernel
turns tagged with ``trigger_kind=INGESTION`` and that the scene-end
hook fires so the R6 session-post slow loop consolidates the
ingested content.
"""

from __future__ import annotations

from lifeform_core import TurnTriggerKind
from volvence_zero.environment import EnvironmentEventKind

from lifeform_ingestion import (
    IngestionComplianceProfile,
    IngestionPipeline,
    envelope_from_task_result,
    envelope_from_text,
)


async def test_plain_text_envelope_produces_ingestion_turns() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="ingestion-e2e-plain")
    text = (
        "Continuity outweighs single-turn cleverness in relationship work.\n\n"
        "Acknowledge before solving, especially under pressure.\n\n"
        "Name the rupture before trying to reset."
    )
    envelope = envelope_from_text(
        text,
        source_uri="file:///tmp/companion-notes.md",
        uploader="test-operator",
    )
    pipeline = IngestionPipeline()
    report = await pipeline.process_envelope(envelope, session=session)
    assert report.total_chunks == 3
    assert report.processed_chunks == 3
    assert report.skipped_chunks == 0
    for record in report.turns:
        assert record.environment_event_id
        assert record.pe_action_context_environment_event_id == record.environment_event_id
        assert record.prediction_id.startswith("pe:prediction_error:turn-")
    # Every recorded turn carries the INGESTION trigger_kind.
    summaries = session.turn_summaries
    assert len(summaries) == 3
    for summary in summaries:
        assert summary.trigger_kind is TurnTriggerKind.INGESTION
    # Scene was closed by the pipeline so the R6 slow loop fired.
    assert report.ended_scene is True
    assert session.open_scene is None
    latest_trace = session.brain_session.runner.dialogue_trace_snapshot.traces[-1]
    scene_evidence = latest_trace.outcome.structured_evidence[-1]
    assert any(ref.startswith("scene_prediction:pe:prediction_error:") for ref in scene_evidence.evidence_refs)


async def test_consultative_profile_produces_user_input_turns() -> None:
    """CONSULTATIVE compliance => trigger_kind=USER_INPUT, vitals get
    per-turn baseline recharge (normal turn behaviour)."""
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="ingestion-e2e-consultative")
    envelope = envelope_from_text(
        "Here is a chunk I'd like your reaction to.",
        source_uri="inline:consultative",
        compliance_profile=IngestionComplianceProfile.CONSULTATIVE,
    )
    pipeline = IngestionPipeline()
    report = await pipeline.process_envelope(envelope, session=session)
    assert report.processed_chunks == 1
    assert session.turn_summaries[-1].trigger_kind is TurnTriggerKind.USER_INPUT
    assert report.turns[0].environment_event_kind == EnvironmentEventKind.USER_INPUT.value
    assert report.turns[0].environment_event_id
    assert report.turns[0].pe_action_context_environment_event_id == report.turns[0].environment_event_id


async def test_forced_ingestion_turns_publish_ingestion_environment_event_kind() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="ingestion-e2e-event-kind")
    envelope = envelope_from_text(
        "Canonical ingestion event kind should reach the kernel.",
        source_uri="inline:event-kind",
    )
    pipeline = IngestionPipeline()
    report = await pipeline.process_envelope(envelope, session=session)

    assert report.processed_chunks == 1
    assert session.turn_summaries[-1].trigger_kind is TurnTriggerKind.INGESTION
    assert report.turns[0].environment_event_kind == EnvironmentEventKind.INGESTION.value
    assert report.turns[0].environment_event_id
    assert report.turns[0].prediction_id.startswith("pe:prediction_error:turn-")


async def test_task_result_envelope_flows_through_ingestion() -> None:
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="ingestion-e2e-task")
    envelope = envelope_from_task_result(
        {
            "summary": "Tool crawled 3 urls.",
            "detail": "Top result points to overload-decision patterns.",
            "findings": ["pattern-a", "pattern-b"],
        },
        task_id="task-e2e-1",
    )
    pipeline = IngestionPipeline()
    report = await pipeline.process_envelope(envelope, session=session)
    # 3 known fields -> 3 chunks -> 3 turns.
    assert report.total_chunks == 3
    assert report.processed_chunks == 3
    for summary in session.turn_summaries[-3:]:
        assert summary.trigger_kind is TurnTriggerKind.INGESTION


async def test_end_scene_after_false_keeps_scene_open_for_followup_turns() -> None:
    """With ``end_scene_after=False`` the caller can continue adding
    turns (ingestion chunks or user turns) to the same scene.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="ingestion-e2e-noend")
    envelope = envelope_from_text(
        "First chunk.\n\nSecond chunk.",
        source_uri="inline:keep-open",
    )
    pipeline = IngestionPipeline()
    await pipeline.process_envelope(
        envelope,
        session=session,
        end_scene_after=False,
    )
    # Scene remains open; caller can keep going.
    assert session.open_scene is not None
    # Run a regular user turn in the same scene; trigger_kind should
    # be USER_INPUT and total turns should be 3 (2 ingestion + 1 user).
    await session.run_turn("Now a normal user question.")
    summaries = session.turn_summaries
    assert len(summaries) == 3
    assert summaries[-1].trigger_kind is TurnTriggerKind.USER_INPUT


async def test_ingestion_is_observable_via_latest_case_memory_reconcile_at_scene_end() -> None:
    """After an ingestion envelope, the scene close runs the
    case_memory provisional reconcile (Gap 4 slice 2a). Even if the
    slow loop didn't produce any provisional records, the reconcile
    result should be non-None \u2014 proving the scene-end path fired.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="ingestion-e2e-reconcile")
    envelope = envelope_from_text(
        "Ingested paragraph for reconcile integration test.",
        source_uri="inline:reconcile",
    )
    pipeline = IngestionPipeline()
    report = await pipeline.process_envelope(envelope, session=session)
    assert report.ended_scene is True
    # The scene-end wiring must have populated the reconcile result.
    from volvence_zero.application import ProvisionalReconcileResult

    reconcile = session.latest_case_memory_reconcile
    assert isinstance(reconcile, ProvisionalReconcileResult)
