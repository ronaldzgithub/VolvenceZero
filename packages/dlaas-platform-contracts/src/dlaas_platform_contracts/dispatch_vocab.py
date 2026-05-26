"""Typed dispatch vocabulary for the platform-tier router.

Slice 2 of `docs/moving forward/dlaas-platform-rollout.md` introduces
the remaining six interaction types (``feedback`` / ``observe`` /
``teach`` / ``task`` / ``report`` / ``command``). Each of those types
exposes a small typed sub-vocabulary that the platform-api dispatcher
must validate before reaching kernel entry points:

* :class:`FeedbackValence` — typed mapping from DLaaS-public labels
  (``"correct"`` / ``"incorrect"`` / etc.) to the kernel's
  :class:`volvence_zero.dialogue_trace.DialogueExternalOutcomeKind`
  vocabulary. Keeps wire-format friendliness without ever falling
  back to free-text inference (R8 + no-keyword-matching).

* :class:`ObservationType` — typed kind for the
  ``structured_context.observation_type`` slot inside an
  ``interaction_type=observe`` envelope. Each kind corresponds to
  exactly one kernel ingestion path (``IngestionPipeline.process_envelope``,
  ``submit_profile_event``, ``submit_task_event``,
  ``submit_reviewed_knowledge_event``, ``submit_tool_result``,
  ``submit_semantic_events``).

* :class:`CommandName` — typed allowlist for the
  ``interaction_type=command`` envelope's ``human_brief`` payload.
  Each value maps to ONE kernel-level action (``submit_profile_event``
  for refresh, ``end_scene`` for explicit scene closure, plus the
  ops-tier pause/resume placeholders that Slice 5.1 wires to the
  platform-ops state machine).

These enums live in the foundation wheel because they describe wire
format. The platform-api dispatcher imports them and routes each value
to the correct kernel sink. Adding a value here is a public-API
change and requires a spec update in
``docs/specs/dlaas-platform.md``.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind


class FeedbackValence(str, Enum):
    """Typed DLaaS feedback valence.

    The two outermost values (``CORRECT`` / ``INCORRECT``) are the
    user-facing labels that DLaaS public clients send (see
    EmoGPT's ``DLAAS_README.md`` §"feedback"). The remaining values
    are direct mirrors of
    :class:`DialogueExternalOutcomeKind` so platforms that already
    speak the kernel vocabulary do not need a translation table.

    Anything outside this enum is rejected at the platform-api edge
    with a typed 400 error — silent fall-through into a default kind
    would let "correct" / "wrong" / "good" / "bad" all collapse onto
    a single kernel kind without any audit trail.
    """

    CORRECT = "correct"
    INCORRECT = "incorrect"
    HELPED = "helped"
    MISSED = "missed"
    FELT_HEARD = "felt_heard"
    OVER_DIRECTIVE = "over_directive"
    DECISION_CLEARER = "decision_clearer"
    COME_BACK = "come_back"
    UNSAFE = "unsafe"
    ABANDONED = "abandoned"
    # ------------------------------------------------------------------
    # W3-A LTV / conversion-funnel valences. These mirror
    # ``DialogueExternalOutcomeKind`` so an external CRM / payments
    # integration can report a confirmed business event through the
    # same DLaaS feedback envelope (``interaction_type=feedback``,
    # ``feedback.valence="purchase_confirmed"``) without an out-of-band
    # channel.
    # ------------------------------------------------------------------
    LEAD_QUALIFIED = "lead_qualified"
    RECOMMENDATION_MADE = "recommendation_made"
    PURCHASE_CONFIRMED = "purchase_confirmed"
    REPURCHASE = "repurchase"
    CHURNED = "churned"


_FEEDBACK_VALENCE_TO_KIND: Final[dict[FeedbackValence, DialogueExternalOutcomeKind]] = {
    FeedbackValence.CORRECT: DialogueExternalOutcomeKind.HELPED,
    FeedbackValence.INCORRECT: DialogueExternalOutcomeKind.MISSED,
    FeedbackValence.HELPED: DialogueExternalOutcomeKind.HELPED,
    FeedbackValence.MISSED: DialogueExternalOutcomeKind.MISSED,
    FeedbackValence.FELT_HEARD: DialogueExternalOutcomeKind.FELT_HEARD,
    FeedbackValence.OVER_DIRECTIVE: DialogueExternalOutcomeKind.OVER_DIRECTIVE,
    FeedbackValence.DECISION_CLEARER: DialogueExternalOutcomeKind.DECISION_CLEARER,
    FeedbackValence.COME_BACK: DialogueExternalOutcomeKind.COME_BACK,
    FeedbackValence.UNSAFE: DialogueExternalOutcomeKind.UNSAFE,
    FeedbackValence.ABANDONED: DialogueExternalOutcomeKind.ABANDONED,
    # W3-A conversion-funnel valences -> kernel outcome kinds (1:1).
    FeedbackValence.LEAD_QUALIFIED: DialogueExternalOutcomeKind.LEAD_QUALIFIED,
    FeedbackValence.RECOMMENDATION_MADE: DialogueExternalOutcomeKind.RECOMMENDATION_MADE,
    FeedbackValence.PURCHASE_CONFIRMED: DialogueExternalOutcomeKind.PURCHASE_CONFIRMED,
    FeedbackValence.REPURCHASE: DialogueExternalOutcomeKind.REPURCHASE,
    FeedbackValence.CHURNED: DialogueExternalOutcomeKind.CHURNED,
}


def feedback_valence_to_outcome_kind(
    valence: FeedbackValence,
) -> DialogueExternalOutcomeKind:
    """Map a typed :class:`FeedbackValence` to its kernel-level kind.

    The mapping is total: every ``FeedbackValence`` member has exactly
    one ``DialogueExternalOutcomeKind`` target. If a future feedback
    valence cannot be expressed as a kernel kind, it must be added to
    the kernel vocabulary first (R8: kernel owns the cognitive
    taxonomy, not the platform).
    """
    try:
        return _FEEDBACK_VALENCE_TO_KIND[valence]
    except KeyError as exc:  # pragma: no cover - guarded by enum membership
        raise ValueError(
            f"FeedbackValence {valence!r} has no DialogueExternalOutcomeKind mapping; "
            f"this is a contract bug — every enum member must map."
        ) from exc


class ObservationType(str, Enum):
    """Typed observation kinds carried in ``structured_context.observation_type``.

    Each kind corresponds to a single kernel ingestion sink. The
    dispatcher validates required structured_context fields per kind
    BEFORE invoking the sink.

    * ``HOMEWORK_RESULT`` — schoolwork / training task result;
      routed to ``BrainSession.submit_task_event``.
    * ``CLASS_NOTE`` / ``TEACHER_NOTE`` — vetted human notes routed
      to ``submit_reviewed_knowledge_event``.
    * ``PROFILE_UPDATE`` — explicit user-profile changes routed
      to ``submit_profile_event``.
    * ``TOOL_RESULT`` — environment outcomes from a tool call routed
      to ``submit_tool_result``.
    * ``CORPUS_INGEST`` — long-form text material routed through the
      ``IngestionPipeline`` (FORCED compliance, INGESTION trigger).
    * ``GENERIC_SEMANTIC`` — pre-formed
      ``ExternalSemanticEventBatch`` payloads routed to
      ``submit_semantic_events`` for callers that already speak the
      kernel vocabulary.
    * ``KNOWLEDGE_RETIRED`` — explicit "this previously-ingested
      knowledge source is no longer authoritative" event. Routes
      through the ``submit_reviewed_knowledge_event`` sink as a
      CLASS_NOTE subtype so semantic owners can mark the matching
      ``knowledge_id`` stale without needing real asset deletion at
      the corpus tier. Required field: ``knowledge_id``.
    * ``PERSON_PROFILE`` — explicit third-party / counterparty
      profile snapshot. Used by digital-employee R16-R20 so the
      EmployeeTwin knows who the employee is acting toward. Routes
      through ``submit_reviewed_knowledge_event`` as a typed reviewed
      knowledge note with person attributes embedded in detail.
    """

    HOMEWORK_RESULT = "homework_result"
    CLASS_NOTE = "class_note"
    TEACHER_NOTE = "teacher_note"
    PROFILE_UPDATE = "profile_update"
    TOOL_RESULT = "tool_result"
    CORPUS_INGEST = "corpus_ingest"
    GENERIC_SEMANTIC = "generic_semantic"
    KNOWLEDGE_RETIRED = "knowledge_retired"
    PERSON_PROFILE = "person_profile"


class CommandName(str, Enum):
    """Typed allowlist for the ``interaction_type=command`` envelope.

    DLaaS clients put the command name in the ``human_brief`` slot
    (see ``DLAAS_README.md`` §"command"). The platform parses it as a
    typed enum — anything outside this allowlist is rejected at the
    edge so the kernel never sees an opaque command string.

    * ``REFRESH_PERSON_CONTEXT`` — re-ingest profile snapshot for
      the listed ``target_person_ids``; thin pass-through to
      ``submit_profile_event``.
    * ``END_SCENE`` — explicit scene closure (does NOT drain slow
      loop; the runtime can still drive R6 reflection out-of-band).
    * ``PAUSE_SESSION`` / ``RESUME_SESSION`` — operator-takeover
      placeholders. Slice 5.1 wires these to the platform-ops
      pause-state machine. In Slice 2.4 they short-circuit to a
      typed ``OutputAct`` that announces the pending implementation.
    * ``INITIATE_PROACTIVE_FOLLOWUP`` (W3-B) — trigger an outbound
      followup turn. Source: ``dlaas-platform-ops.OutboundScheduler``
      after evaluating ``relationship_state`` snapshot + cadence
      config. The dispatcher converts it into a kernel turn under
      :class:`TurnTriggerKind.APPRENTICE` (so vitals apprentice
      override is on and the proactive followup does not pollute
      slow-scale PE). The reason text travels in
      ``structured_context.followup_brief``.
    """

    REFRESH_PERSON_CONTEXT = "refresh_person_context"
    END_SCENE = "end_scene"
    PAUSE_SESSION = "pause_session"
    RESUME_SESSION = "resume_session"
    INITIATE_PROACTIVE_FOLLOWUP = "initiate_proactive_followup"


__all__ = [
    "CommandName",
    "FeedbackValence",
    "ObservationType",
    "feedback_valence_to_outcome_kind",
]
