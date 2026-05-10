"""Slice 7.1 — typed envelope dispatch contracts.

These tests exercise :func:`dlaas_platform_api.dispatch_envelope`
directly with a fake session, so they validate the typed dispatch
table without spinning up the kernel. The point is to lock down the
invariants that ``docs/specs/dlaas-platform.md`` calls out:

1. ``interaction_type`` is the SOLE dispatch key — the handler must
   never inspect ``human_brief`` to guess the type.
2. Each typed payload (``feedback`` / ``observe`` / ``command``)
   has its own typed contract surface; missing fields surface as
   typed ``DispatchError`` codes.
3. The dispatcher never falls back to the kernel when the typed
   contract is broken — the kernel only sees fully-validated
   envelopes.

The fake session records every kernel call so each test can assert
which entry point was actually invoked. Regression of any of these
contracts means the platform-tier ``vz-* diff = 0`` promise
silently broke (a typo'd dispatch could route ``feedback`` to
``run_turn`` and contaminate user-input PE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from dlaas_platform_api import DispatchError, dispatch_envelope
from dlaas_platform_contracts import InteractionEnvelope


@dataclass
class _FakeBrainSessionRunner:
    """Minimal stand-in for ``brain_session.runner.active_snapshots()``."""

    snapshots: dict[str, Any] = field(default_factory=dict)

    def active_snapshots(self) -> dict[str, Any]:
        return self.snapshots


@dataclass
class _FakeBrainSession:
    runner: _FakeBrainSessionRunner = field(
        default_factory=_FakeBrainSessionRunner
    )


@dataclass
class _FakeAgentTurnResult:
    response_text: str = "ok"
    rationale_tags: tuple[str, ...] = ()
    active_regime: str = "calm"
    active_abstract_action: str = "respond"

    @property
    def response(self):
        class _Response:
            text = self.response_text
            rationale_tags = self.rationale_tags

        return _Response()


class _FakeSession:
    """Records every kernel call so tests can assert dispatch correctness."""

    def __init__(self) -> None:
        self.run_turn_calls: list[dict[str, Any]] = []
        self.submit_dialogue_outcome_calls: list[dict[str, Any]] = []
        self.submit_profile_event_calls: list[dict[str, Any]] = []
        self.submit_task_event_calls: list[dict[str, Any]] = []
        self.submit_reviewed_knowledge_event_calls: list[dict[str, Any]] = []
        self.submit_tool_result_calls: list[dict[str, Any]] = []
        self.end_scene_calls: list[dict[str, Any]] = []
        self._brain_session = _FakeBrainSession()

    async def run_turn(self, user_input: str, *, trigger_kind=None):
        self.run_turn_calls.append(
            {"user_input": user_input, "trigger_kind": trigger_kind}
        )
        return _FakeAgentTurnResult()

    def submit_dialogue_outcome(self, **kwargs):
        self.submit_dialogue_outcome_calls.append(kwargs)

        class _Evidence:
            evidence_id = f"evid_{len(self.submit_dialogue_outcome_calls):04d}"

        return _Evidence()

    def submit_profile_event(self, **kwargs):
        self.submit_profile_event_calls.append(kwargs)
        return (f"event:{len(self.submit_profile_event_calls):04d}",)

    def submit_task_event(self, **kwargs):
        self.submit_task_event_calls.append(kwargs)
        return (f"event:{len(self.submit_task_event_calls):04d}",)

    def submit_reviewed_knowledge_event(self, **kwargs):
        self.submit_reviewed_knowledge_event_calls.append(kwargs)
        return (f"event:{len(self.submit_reviewed_knowledge_event_calls):04d}",)

    def submit_tool_result(self, **kwargs):
        self.submit_tool_result_calls.append(kwargs)
        return (f"event:{len(self.submit_tool_result_calls):04d}",)

    async def end_scene(self, *, reason: str = "", drain_slow_loop: bool = True):
        self.end_scene_calls.append(
            {"reason": reason, "drain_slow_loop": drain_slow_loop}
        )

        class _Scene:
            scene_id = f"scene_{len(self.end_scene_calls):04d}"

        return _Scene()


def _envelope(**overrides: Any) -> InteractionEnvelope:
    base = {
        "contract_id": "ctr_test",
        "session_id": "sess_test",
        "end_user_ref": "user_test",
        "interaction_type": "chat",
        "human_brief": "hello",
    }
    base.update(overrides)
    return InteractionEnvelope.from_json(base)


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


async def test_chat_dispatches_to_run_turn_with_user_input_trigger():
    from lifeform_core.types import TurnTriggerKind

    session = _FakeSession()
    body = await dispatch_envelope(
        envelope=_envelope(human_brief="hi there"),
        session=session,
        ai_id="ai_smoke",
    )
    assert len(session.run_turn_calls) == 1
    call = session.run_turn_calls[0]
    assert call["user_input"] == "hi there"
    assert call["trigger_kind"] in (None, TurnTriggerKind.USER_INPUT)
    assert body["interaction_type"] == "chat"
    assert body["output_acts"][0]["act_type"] == "text"
    assert body["output_acts"][0]["capability"] == "text_streaming"


async def test_chat_without_human_brief_rejected():
    session = _FakeSession()
    with pytest.raises(DispatchError) as info:
        await dispatch_envelope(
            envelope=_envelope(human_brief=""),
            session=session,
            ai_id="ai_smoke",
        )
    assert info.value.code == "invalid_human_brief"
    assert info.value.status == 400
    assert session.run_turn_calls == []


# ---------------------------------------------------------------------------
# feedback (Slice 2.1)
# ---------------------------------------------------------------------------


async def test_feedback_maps_typed_valence_to_kernel_outcome_kind():
    from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind

    session = _FakeSession()
    body = await dispatch_envelope(
        envelope=_envelope(
            interaction_type="feedback",
            human_brief="thanks",
            feedback={
                "valence": "correct",
                "target_response_id": "resp_abc",
                "intensity": 0.92,
                "scope": "response",
                "evidence": "user said thanks",
            },
        ),
        session=session,
        ai_id="ai_smoke",
    )
    assert len(session.submit_dialogue_outcome_calls) == 1
    call = session.submit_dialogue_outcome_calls[0]
    assert call["kind"] is DialogueExternalOutcomeKind.HELPED
    assert pytest.approx(call["confidence"], rel=1e-6) == 0.92
    assert call["evidence_ref"] == "resp_abc"
    assert body["feedback"]["valence"] == "correct"
    assert body["feedback"]["outcome_kind"] == "helped"


@pytest.mark.parametrize(
    ("valence", "expected_kind"),
    [
        ("lead_qualified", "lead_qualified"),
        ("recommendation_made", "recommendation_made"),
        ("purchase_confirmed", "purchase_confirmed"),
        ("repurchase", "repurchase"),
        ("churned", "churned"),
    ],
)
async def test_feedback_maps_w3a_conversion_valences_to_kernel(
    valence: str, expected_kind: str
):
    """W3-A: LTV / conversion valences round-trip through the platform.

    Each new valence in the feedback envelope must reach the kernel
    via ``submit_dialogue_outcome(kind=...)`` with the matching kernel
    enum value. Missing one of these mappings would mean the platform
    silently drops a confirmed business event onto the floor; this
    contract is the only place that asserts the round-trip.
    """
    from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind

    session = _FakeSession()
    body = await dispatch_envelope(
        envelope=_envelope(
            interaction_type="feedback",
            human_brief=f"crm event: {valence}",
            feedback={
                "valence": valence,
                "target_response_id": f"resp_{valence}",
                "intensity": 0.95,
                "scope": "response",
                "evidence": f"crm webhook reported {valence}",
            },
        ),
        session=session,
        ai_id="ai_smoke",
    )
    assert len(session.submit_dialogue_outcome_calls) == 1
    call = session.submit_dialogue_outcome_calls[0]
    assert call["kind"] is DialogueExternalOutcomeKind(expected_kind)
    assert body["feedback"]["valence"] == valence
    assert body["feedback"]["outcome_kind"] == expected_kind


async def test_feedback_rejects_unknown_valence_at_edge():
    session = _FakeSession()
    with pytest.raises(DispatchError) as info:
        await dispatch_envelope(
            envelope=_envelope(
                interaction_type="feedback",
                feedback={"valence": "delicious"},
            ),
            session=session,
            ai_id="ai_smoke",
        )
    assert info.value.code == "invalid_feedback_valence"
    assert session.submit_dialogue_outcome_calls == []


async def test_feedback_without_payload_rejected():
    session = _FakeSession()
    with pytest.raises(DispatchError) as info:
        await dispatch_envelope(
            envelope=_envelope(interaction_type="feedback"),
            session=session,
            ai_id="ai_smoke",
        )
    assert info.value.code == "missing_feedback_payload"


# ---------------------------------------------------------------------------
# observe (Slice 2.2)
# ---------------------------------------------------------------------------


async def test_observe_homework_result_routes_to_submit_task_event():
    session = _FakeSession()
    body = await dispatch_envelope(
        envelope=_envelope(
            interaction_type="observe",
            human_brief="hw done",
            structured_context={
                "observation_type": "homework_result",
                "event_id": "ev1",
                "task_id": "math_hw_001",
                "status": "submitted",
                "summary": "fraction word problems",
                "detail": "10/15 correct",
            },
        ),
        session=session,
        ai_id="ai_smoke",
    )
    assert len(session.submit_task_event_calls) == 1
    call = session.submit_task_event_calls[0]
    assert call["task_id"] == "math_hw_001"
    assert call["status"] == "submitted"
    assert body["observation_type"] == "homework_result"


async def test_observe_class_note_routes_to_reviewed_knowledge():
    session = _FakeSession()
    await dispatch_envelope(
        envelope=_envelope(
            interaction_type="observe",
            human_brief="teacher note",
            structured_context={
                "observation_type": "class_note",
                "event_id": "ev2",
                "knowledge_id": "k_001",
                "summary": "fractions concept review",
                "detail": "students grasping fraction>decimal links",
                "needs_followup": True,
            },
        ),
        session=session,
        ai_id="ai_smoke",
    )
    assert len(session.submit_reviewed_knowledge_event_calls) == 1
    call = session.submit_reviewed_knowledge_event_calls[0]
    assert call["needs_followup"] is True


async def test_observe_profile_update_routes_to_submit_profile_event():
    session = _FakeSession()
    await dispatch_envelope(
        envelope=_envelope(
            interaction_type="observe",
            human_brief="profile",
            structured_context={
                "observation_type": "profile_update",
                "event_id": "ev3",
                "preferences": ["likes_visual_aids"],
                "goals": ["improve_word_problems"],
            },
        ),
        session=session,
        ai_id="ai_smoke",
    )
    assert len(session.submit_profile_event_calls) == 1
    call = session.submit_profile_event_calls[0]
    assert call["preferences"] == ("likes_visual_aids",)
    assert call["goals"] == ("improve_word_problems",)


async def test_observe_tool_result_routes_to_submit_tool_result():
    session = _FakeSession()
    await dispatch_envelope(
        envelope=_envelope(
            interaction_type="observe",
            human_brief="tool",
            structured_context={
                "observation_type": "tool_result",
                "event_id": "ev4",
                "tool_name": "grader",
                "action_id": "a1",
                "status": "ok",
                "summary": "graded 5 questions",
            },
        ),
        session=session,
        ai_id="ai_smoke",
    )
    assert len(session.submit_tool_result_calls) == 1


async def test_observe_corpus_ingest_runs_pipeline_and_emits_report():
    session = _FakeSession()
    body = await dispatch_envelope(
        envelope=_envelope(
            interaction_type="observe",
            human_brief="some corpus text",
            structured_context={
                "observation_type": "corpus_ingest",
                "corpus_text": "Paragraph one.\n\nParagraph two.",
                "source_uri": "test:smoke",
                "uploader": "tester",
            },
        ),
        session=session,
        ai_id="ai_smoke",
    )
    # IngestionPipeline drives the kernel via run_turn for each chunk.
    assert len(session.run_turn_calls) >= 1
    assert body["ingestion_report"]["processed_chunks"] >= 1
    assert body["observation_type"] == "corpus_ingest"


async def test_observe_missing_observation_type_rejected():
    session = _FakeSession()
    with pytest.raises(DispatchError) as info:
        await dispatch_envelope(
            envelope=_envelope(
                interaction_type="observe", structured_context={}
            ),
            session=session,
            ai_id="ai_smoke",
        )
    assert info.value.code == "missing_observation_type"


async def test_observe_unknown_observation_type_rejected():
    session = _FakeSession()
    with pytest.raises(DispatchError) as info:
        await dispatch_envelope(
            envelope=_envelope(
                interaction_type="observe",
                structured_context={"observation_type": "telepathy"},
            ),
            session=session,
            ai_id="ai_smoke",
        )
    assert info.value.code == "invalid_observation_type"


async def test_observe_homework_missing_required_field_rejected():
    session = _FakeSession()
    with pytest.raises(DispatchError) as info:
        await dispatch_envelope(
            envelope=_envelope(
                interaction_type="observe",
                structured_context={
                    "observation_type": "homework_result",
                    "event_id": "ev",
                    # task_id intentionally missing
                    "status": "submitted",
                    "summary": "x",
                },
            ),
            session=session,
            ai_id="ai_smoke",
        )
    assert info.value.code == "missing_field"


# ---------------------------------------------------------------------------
# teach / task (Slice 2.3)
# ---------------------------------------------------------------------------


async def test_teach_uses_apprentice_trigger():
    from lifeform_core.types import TurnTriggerKind

    session = _FakeSession()
    body = await dispatch_envelope(
        envelope=_envelope(
            interaction_type="teach",
            human_brief="先共情，再下一步行动。",
        ),
        session=session,
        ai_id="ai_smoke",
    )
    assert len(session.run_turn_calls) == 1
    assert (
        session.run_turn_calls[0]["trigger_kind"]
        is TurnTriggerKind.APPRENTICE
    )
    assert body["trigger_kind"] == TurnTriggerKind.APPRENTICE.value


async def test_task_also_uses_apprentice_trigger():
    from lifeform_core.types import TurnTriggerKind

    session = _FakeSession()
    await dispatch_envelope(
        envelope=_envelope(
            interaction_type="task",
            human_brief="提炼三条讲解策略",
        ),
        session=session,
        ai_id="ai_smoke",
    )
    assert (
        session.run_turn_calls[0]["trigger_kind"]
        is TurnTriggerKind.APPRENTICE
    )


async def test_teach_without_human_brief_rejected():
    session = _FakeSession()
    with pytest.raises(DispatchError) as info:
        await dispatch_envelope(
            envelope=_envelope(interaction_type="teach", human_brief=""),
            session=session,
            ai_id="ai_smoke",
        )
    assert info.value.code == "invalid_human_brief"


# ---------------------------------------------------------------------------
# report (Slice 2.4)
# ---------------------------------------------------------------------------


async def test_report_drains_slow_loop_via_end_scene():
    session = _FakeSession()
    body = await dispatch_envelope(
        envelope=_envelope(interaction_type="report", human_brief="weekly"),
        session=session,
        ai_id="ai_smoke",
    )
    assert len(session.end_scene_calls) == 1
    assert session.end_scene_calls[0]["drain_slow_loop"] is True
    assert body["drained"] is True
    # report_view is reserved for Slice 6+ projection — it should be
    # present in the body and start as None so consumers know to
    # poll the reflection snapshot via a separate readout.
    assert body["report_view"] is None


# ---------------------------------------------------------------------------
# command (Slice 2.4)
# ---------------------------------------------------------------------------


async def test_command_refresh_person_routes_to_profile_event():
    session = _FakeSession()
    body = await dispatch_envelope(
        envelope=_envelope(
            interaction_type="command",
            human_brief="refresh_person_context",
            target_person_ids=["student_001"],
        ),
        session=session,
        ai_id="ai_smoke",
    )
    assert len(session.submit_profile_event_calls) == 1
    call = session.submit_profile_event_calls[0]
    assert "refresh_person_context" in call["relationship_note"]
    assert body["command"] == "refresh_person_context"
    assert body["target_person_ids"] == ["student_001"]


async def test_command_end_scene_does_not_drain():
    session = _FakeSession()
    body = await dispatch_envelope(
        envelope=_envelope(
            interaction_type="command", human_brief="end_scene"
        ),
        session=session,
        ai_id="ai_smoke",
    )
    assert len(session.end_scene_calls) == 1
    assert session.end_scene_calls[0]["drain_slow_loop"] is False
    assert body["drained"] is False


async def test_command_unknown_rejected():
    session = _FakeSession()
    with pytest.raises(DispatchError) as info:
        await dispatch_envelope(
            envelope=_envelope(
                interaction_type="command", human_brief="invent_galaxy"
            ),
            session=session,
            ai_id="ai_smoke",
        )
    assert info.value.code == "invalid_command"


async def test_command_pause_session_returns_slice5_placeholder():
    session = _FakeSession()
    body = await dispatch_envelope(
        envelope=_envelope(
            interaction_type="command", human_brief="pause_session"
        ),
        session=session,
        ai_id="ai_smoke",
    )
    assert body["command"] == "pause_session"
    assert "Slice 5.1" in body["ops_pending"]
    # Pause command is recorded by the platform; it must NOT call the
    # kernel's end_scene / run_turn paths.
    assert session.run_turn_calls == []
    assert session.end_scene_calls == []


# ---------------------------------------------------------------------------
# Cross-cutting: typed enum dispatch only
# ---------------------------------------------------------------------------


async def test_command_initiate_proactive_followup_dispatches_apprentice_turn():
    """W3-B: ``initiate_proactive_followup`` runs an APPRENTICE turn.

    The dispatcher must take ``structured_context.followup_brief`` as
    the user_input for the kernel turn (the brief is the message the
    lifeform sends *to* the user; it is generated by the platform-ops
    OutboundScheduler from a vertical-supplied template, never
    inferred from chat text). The trigger kind must be APPRENTICE so
    vitals apprentice override is on for the proactive turn.
    """
    from lifeform_core.types import TurnTriggerKind

    session = _FakeSession()
    body = await dispatch_envelope(
        envelope=_envelope(
            interaction_type="command",
            human_brief="initiate_proactive_followup",
            structured_context={
                "followup_brief": "宝妈早呀，想起你家娃 5 岁啦，今天周末别太累。",
                "followup_evidence_ref": (
                    "scheduler:funnel_stage=nurturing:age_turns=5"
                ),
            },
        ),
        session=session,
        ai_id="ai_smoke",
    )
    assert len(session.run_turn_calls) == 1
    call = session.run_turn_calls[0]
    assert "宝妈早呀" in call["user_input"]
    assert call["trigger_kind"] is TurnTriggerKind.APPRENTICE
    assert body["command"] == "initiate_proactive_followup"
    assert body["trigger_kind"] == TurnTriggerKind.APPRENTICE.value
    assert body["followup_evidence_ref"] == (
        "scheduler:funnel_stage=nurturing:age_turns=5"
    )


async def test_command_initiate_proactive_followup_rejects_missing_brief():
    """Without ``followup_brief`` the dispatcher rejects at the edge.

    The kernel must never see an empty proactive followup — that
    would silently invoke a turn against an empty user_input and
    produce a degenerate response. The platform fails fast instead.
    """
    session = _FakeSession()
    with pytest.raises(DispatchError) as info:
        await dispatch_envelope(
            envelope=_envelope(
                interaction_type="command",
                human_brief="initiate_proactive_followup",
                structured_context={},
            ),
            session=session,
            ai_id="ai_smoke",
        )
    assert info.value.code == "missing_followup_brief"
    assert info.value.status == 400
    assert session.run_turn_calls == []


async def test_human_brief_does_not_affect_chat_dispatch():
    """``chat`` envelopes whose ``human_brief`` happens to match a
    command name MUST still dispatch as chat (typed enum only)."""
    session = _FakeSession()
    body = await dispatch_envelope(
        envelope=_envelope(
            interaction_type="chat", human_brief="refresh_person_context"
        ),
        session=session,
        ai_id="ai_smoke",
    )
    # run_turn was called → it's a chat. submit_profile_event was not.
    assert len(session.run_turn_calls) == 1
    assert session.submit_profile_event_calls == []
    assert body["interaction_type"] == "chat"
