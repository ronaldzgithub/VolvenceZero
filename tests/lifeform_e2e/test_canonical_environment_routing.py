"""CP-03 canonical environment routing: per-source e2e evidence.

Every trigger source that enters the kernel as a turn must arrive as a
canonical ``EnvironmentEvent`` with the correct ``EnvironmentEventKind`` and
``trigger_kind`` — including the two lifecycle sources that previously had
zero runtime call sites (``FOLLOWUP_DUE`` / ``INTERNAL_DRIVE``). This file is
also the documented host path for firing those turns: the product layer reads
``due_followups()`` / vitals drive pressure and decides to re-engage; the
lifeform never auto-fires them.

``SYSTEM_TICK`` and ``SCENE_EVENT`` intentionally do NOT appear here as kernel
turns: tick effects reach cognition only indirectly (vitals decay ->
proactive followup -> a later ``FOLLOWUP_DUE`` turn; idle close ->
``end_scene``), and scene close crosses the boundary as typed
``scene_closed_evidence`` on the dialogue trace. Both are documented
compatibility adapters per docs/specs/environment-interface.md gate 1.
"""

from __future__ import annotations

import pytest

from lifeform_core import Lifeform, LifeformConfig, TickEngineConfig
from lifeform_core.types import TurnTriggerKind
from lifeform_domain_emogpt import build_companion_vitals_bootstrap
from volvence_zero.brain import BrainConfig
from volvence_zero.environment import (
    EnvironmentActorRef,
    EnvironmentEventKind,
    EnvironmentFrame,
)


def _build_lifeform() -> Lifeform:
    config = LifeformConfig(
        brain_config=BrainConfig(rare_heavy_enabled=False),
        tick=TickEngineConfig(
            system_tick_seconds=0.0,
            energy_every_n_system_ticks=2,
            context_every_n_system_ticks=4,
        ),
        idle_close_after_system_ticks=None,
        vitals_bootstrap=build_companion_vitals_bootstrap(),
    )
    return Lifeform(config)


@pytest.mark.parametrize(
    ("trigger", "expected_kind"),
    (
        (TurnTriggerKind.USER_INPUT, EnvironmentEventKind.USER_INPUT),
        (TurnTriggerKind.INTERNAL_DRIVE, EnvironmentEventKind.INTERNAL_DRIVE),
        (TurnTriggerKind.FOLLOWUP_DUE, EnvironmentEventKind.FOLLOWUP_DUE),
        (TurnTriggerKind.INGESTION, EnvironmentEventKind.INGESTION),
        (TurnTriggerKind.APPRENTICE, EnvironmentEventKind.APPRENTICE),
    ),
)
async def test_each_trigger_reaches_kernel_as_canonical_event(
    trigger: TurnTriggerKind,
    expected_kind: EnvironmentEventKind,
) -> None:
    lifeform = _build_lifeform()
    session = lifeform.create_session(session_id=f"canon-route-{trigger.value}")

    result = await session.run_turn(
        f"Payload for a {trigger.value} turn.", trigger_kind=trigger
    )

    assert result.environment_event_kind == expected_kind.value
    assert result.environment_trigger_kind == trigger.value
    assert result.environment_event_id, "canonical event id must be non-empty"
    # Single-party compatibility frame stays explicit, never inferred downstream.
    assert result.active_speaker_id == "primary"
    assert result.audience_ids == ("self",)
    summary = session.turn_summaries[-1]
    assert summary.trigger_kind is trigger


async def test_followup_due_host_path_from_due_followup_to_canonical_turn() -> None:
    """Documented host path: due followup -> product layer fires FOLLOWUP_DUE turn."""

    lifeform = _build_lifeform()
    session = lifeform.create_session(session_id="canon-route-followup-host")

    # A real turn seeds scene state; then the metabolic clock advances far
    # enough for vitals drive deviation to surface a proactive followup.
    await session.run_turn("Let's plan the harbor inspection together.")
    await session.advance_tick(system_ticks=400, reason="idle window")

    due = session.due_followups()
    assert due, "expected at least one due followup after a long idle window"
    item = due[0]

    result = await session.run_turn(
        f"(checking in about: {item.description})",
        trigger_kind=TurnTriggerKind.FOLLOWUP_DUE,
    )
    assert result.environment_event_kind == EnvironmentEventKind.FOLLOWUP_DUE.value
    assert result.environment_trigger_kind == TurnTriggerKind.FOLLOWUP_DUE.value
    # The product layer, not the lifeform, retires the followup after acting.
    assert session.acknowledge_followup(item.followup_id) is True


async def test_product_supplied_multi_party_frame_reaches_social_owners() -> None:
    lifeform = _build_lifeform()
    session = lifeform.create_session(session_id="canon-route-multi-party")
    frame = EnvironmentFrame(
        actor=EnvironmentActorRef(actor_id="alice", display_name="Alice"),
        active_speaker_id="alice",
        addressee_ids=("self", "bob"),
        subject_ids=("alice", "bob"),
        audience_ids=("self", "bob", "cara"),
    )

    result = await session.run_turn(
        "Alice is asking Bob to review the plan while Cara observes.",
        environment_frame=frame,
    )

    assert result.active_speaker_id == "alice"
    assert result.audience_ids == ("self", "bob", "cara")
    role_snapshot = result.active_snapshots["conversational_role"].value
    assert role_snapshot.active_speaker_id == "alice"
    assert role_snapshot.addressee_ids == ("self", "bob")
    assert role_snapshot.subject_ids == ("alice", "bob")
    user_model_snapshot = result.active_snapshots["user_model"].value
    assert user_model_snapshot.interlocutor_ids == ("alice", "bob", "cara", "self")
