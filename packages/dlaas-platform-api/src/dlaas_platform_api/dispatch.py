"""Typed-envelope dispatch handlers for ``/dlaas/instances/{ai_id}/interactions``.

Each ``interaction_type`` in :class:`dlaas_platform_contracts.InteractionType`
maps to exactly one async handler in this module. The handlers translate
typed envelopes into ``LifeformSession`` / ``BrainSession`` calls and
return a ready-to-serialise JSON body.

Design rules:

1. The dispatcher MUST switch on the typed enum
   (:func:`dispatch_envelope` below). It MUST NOT inspect natural-language
   fields (``human_brief``, etc.) to decide the handler — that would
   violate the no-keyword-matching invariant.
2. Each handler validates its required ``structured_context`` keys
   BEFORE calling the kernel. Missing keys raise
   :class:`DispatchError` which the caller turns into a typed 400.
3. Handlers may emit one or more :class:`OutputAct` entries plus extra
   metadata via :func:`output_acts.ok_envelope`. They never mutate the
   envelope or the session beyond the documented kernel call.
4. Handlers do NOT touch any cognitive state directly. They only read
   results returned by the kernel facade and package them into the
   wire-format response.

Slice 1 wired ``chat`` only. Slice 2 fills in the remaining six types
(``feedback`` / ``observe`` / ``teach`` / ``task`` / ``report`` /
``command``) per ``docs/moving forward/dlaas-platform-rollout.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dlaas_platform_contracts import (
    CommandName,
    FeedbackPayload,
    FeedbackValence,
    InteractionEnvelope,
    InteractionType,
    ObservationType,
    OutputAct,
    feedback_valence_to_outcome_kind,
)
from lifeform_core.types import TurnTriggerKind
from lifeform_ingestion import (
    IngestionComplianceProfile,
    IngestionPipeline,
    IngestionSourceKind,
    envelope_from_text,
)
from volvence_zero.dialogue_trace import DialogueExternalOutcomeEvidenceSource

from dlaas_platform_api.output_acts import (
    ok_envelope,
    system_act,
    text_act,
    tool_call_act,
    tool_task_act,
)


class DispatchError(Exception):
    """Raised by a handler when the envelope fails its typed contract.

    Carries an HTTP-friendly ``code`` slug (e.g. ``"missing_field"``)
    and a ``detail`` string. The route adapter converts it to a
    structured 400 response. Distinct from
    :class:`dlaas_platform_api.app._EnvelopeError`, which is raised by
    the parser BEFORE dispatch.
    """

    def __init__(self, code: str, detail: str, *, status: int = 400) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status = status


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


async def dispatch_envelope(
    *,
    envelope: InteractionEnvelope,
    session: Any,
    ai_id: str,
) -> dict[str, Any]:
    """Route a typed envelope to the matching handler.

    The single switch on :class:`InteractionType` is the SSOT for
    "which interaction goes to which kernel sink". Adding a new
    interaction type means: extend the enum, extend this switch, add
    a handler. The dispatcher refuses to fall through to a default
    path — every member must be enumerated.
    """
    kind = envelope.interaction_type
    if kind is InteractionType.CHAT:
        return await _handle_chat(envelope=envelope, session=session, ai_id=ai_id)
    if kind is InteractionType.FEEDBACK:
        return await _handle_feedback(envelope=envelope, session=session, ai_id=ai_id)
    if kind is InteractionType.OBSERVE:
        return await _handle_observe(envelope=envelope, session=session, ai_id=ai_id)
    if kind in (InteractionType.TEACH, InteractionType.TASK):
        return await _handle_apprentice(envelope=envelope, session=session, ai_id=ai_id)
    if kind is InteractionType.REPORT:
        return await _handle_report(envelope=envelope, session=session, ai_id=ai_id)
    if kind is InteractionType.COMMAND:
        return await _handle_command(envelope=envelope, session=session, ai_id=ai_id)
    raise DispatchError(  # pragma: no cover - exhaustive switch
        code="unsupported_interaction_type",
        detail=(
            f"interaction_type={kind!r} is in the typed enum but has no "
            f"handler. This is a contract bug — every InteractionType "
            f"member must have a dispatcher."
        ),
        status=500,
    )


# ---------------------------------------------------------------------------
# Slice 1 — chat
# ---------------------------------------------------------------------------


async def _handle_chat(
    *,
    envelope: InteractionEnvelope,
    session: Any,
    ai_id: str,
) -> dict[str, Any]:
    if not envelope.human_brief.strip():
        raise DispatchError(
            code="invalid_human_brief",
            detail="interaction_type=chat requires a non-empty human_brief",
        )
    native_tool_intent = _native_tool_intent(envelope.structured_context)
    loop_mode = _optional_str(
        envelope.structured_context, "tool_loop_mode", default="client"
    )
    if native_tool_intent is not None or loop_mode == "server":
        invoker = _session_invoker(session)
        if loop_mode == "server":
            from lifeform_affordance import ToolLoopOrchestrator, ToolLoopPolicy

            policy_kwargs: dict[str, Any] = {"server_side_execution": True}
            override_steps = _optional_int(
                envelope.structured_context,
                "tool_loop_max_steps",
            )
            if override_steps is not None and override_steps > 0:
                policy_kwargs["max_tool_steps"] = override_steps
            override_wall_ms = _optional_int(
                envelope.structured_context,
                "tool_loop_max_wall_ms",
            )
            if override_wall_ms is not None and override_wall_ms > 0:
                policy_kwargs["max_wall_ms"] = override_wall_ms
            orchestrator = ToolLoopOrchestrator(
                registry=invoker.registry,
                invoker=invoker,
                policy=ToolLoopPolicy(**policy_kwargs),
                contract_id=envelope.contract_id,
                granted_consents=frozenset(
                    _optional_str_sequence(
                        envelope.structured_context,
                        "granted_consents",
                    )
                ),
            )
            loop_result = await orchestrator.run(
                session=session,
                user_input=envelope.human_brief,
                initial_intents=(
                    (native_tool_intent,) if native_tool_intent is not None else ()
                ),
            )
            final_text = getattr(loop_result.final_turn_result.response, "text", "") or ""
            acts = [text_act(final_text)]
            for task_id in loop_result.async_task_ids:
                handle = invoker.get_task_handle(task_id)
                acts.append(
                    tool_task_act(
                        task_id=handle.task_id,
                        status=handle.status.value,
                        poll_after_ms=handle.poll_after_ms,
                    )
                )
            return ok_envelope(
                ai_id=ai_id,
                contract_id=envelope.contract_id,
                session_id=envelope.session_id,
                interaction_type=envelope.interaction_type.value,
                output_acts=tuple(acts),
                protocol_version=envelope.protocol_version,
                extra={
                    "tool_loop": {
                        "stop_reason": loop_result.stop_reason.value,
                        "steps": [_tool_loop_step_to_json(step) for step in loop_result.steps],
                    }
                },
            )
        if native_tool_intent is None:
            raise DispatchError(
                code="missing_tool_choice",
                detail=(
                    "structured_context.tool_loop_mode must be 'server' when "
                    "no structured_context.tool_choice is provided."
                ),
            )
        return ok_envelope(
            ai_id=ai_id,
            contract_id=envelope.contract_id,
            session_id=envelope.session_id,
            interaction_type=envelope.interaction_type.value,
            output_acts=(
                tool_call_act(
                    call_id=native_tool_intent.stable_call_id,
                    tool_name=native_tool_intent.descriptor_name,
                    arguments=dict(native_tool_intent.parameters),
                ),
            ),
            protocol_version=envelope.protocol_version,
            extra={"tool_loop": {"mode": "client"}},
        )
    result = await session.run_turn(envelope.human_brief)
    response_text = getattr(result.response, "text", "") or ""
    rationale_tags = tuple(getattr(result.response, "rationale_tags", ()) or ())
    return ok_envelope(
        ai_id=ai_id,
        contract_id=envelope.contract_id,
        session_id=envelope.session_id,
        interaction_type=envelope.interaction_type.value,
        output_acts=(text_act(response_text),),
        protocol_version=envelope.protocol_version,
        extra={
            "active_regime": getattr(result, "active_regime", None),
            "active_abstract_action": getattr(result, "active_abstract_action", None),
            "rationale_tags": list(rationale_tags),
        },
    )


def _session_invoker(session: Any) -> Any:
    try:
        return session.mcp_invoker
    except AttributeError as exc:
        raise DispatchError(
            code="tool_invoker_unavailable",
            detail="session does not expose an affordance invoker",
            status=501,
        ) from exc


def _native_tool_intent(ctx: Mapping[str, Any]) -> Any:
    raw = ctx.get("tool_choice")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise DispatchError(
            code="invalid_tool_choice",
            detail="structured_context.tool_choice must be an object",
        )
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise DispatchError(
            code="invalid_tool_choice",
            detail="structured_context.tool_choice.name must be non-empty",
        )
    arguments = raw.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise DispatchError(
            code="invalid_tool_choice",
            detail="structured_context.tool_choice.arguments must be an object",
        )
    from lifeform_affordance import ToolCallIntent

    call_id = raw.get("call_id", "")
    if call_id is not None and not isinstance(call_id, str):
        raise DispatchError(
            code="invalid_tool_choice",
            detail="structured_context.tool_choice.call_id must be a string",
        )
    return ToolCallIntent(
        descriptor_name=name,
        parameters=dict(arguments),
        call_id=call_id or "",
        plan_ref=_optional_str(ctx, "plan_ref", default=None) or None,
        source="dlaas_native",
    )


def _tool_loop_step_to_json(step: Any) -> dict[str, Any]:
    intent = step.intent
    invocation = step.invocation
    payload: dict[str, Any] = {
        "step_index": step.step_index,
        "tool_name": intent.descriptor_name,
        "arguments": dict(intent.parameters),
        "call_id": intent.stable_call_id,
        "plan_ref": intent.plan_ref,
        "status": step.status,
        "elapsed_ms": step.elapsed_ms,
    }
    if invocation is not None:
        payload["invocation"] = {
            "descriptor_name": invocation.descriptor_name,
            "status": invocation.status.value,
            "payload": dict(invocation.payload or {}),
            "error_class": invocation.error_class,
            "error_detail": invocation.error_detail,
            "tool_event_ids": list(invocation.tool_event_ids),
            "task_id": invocation.task_id,
            "kernel_summary_truncated": bool(
                getattr(invocation, "kernel_summary_truncated", False)
            ),
        }
    return payload


# ---------------------------------------------------------------------------
# Slice 2.1 — feedback
# ---------------------------------------------------------------------------


async def _handle_feedback(
    *,
    envelope: InteractionEnvelope,
    session: Any,
    ai_id: str,
) -> dict[str, Any]:
    """Translate a typed feedback envelope to ``submit_dialogue_outcome``.

    Wire format mirrors DLaaS public ``feedback`` envelope: the typed
    valence + optional ``target_response_id`` / ``intensity`` /
    ``scope`` / ``evidence`` carry directly over. The platform maps
    :class:`FeedbackValence` to the kernel's
    :class:`DialogueExternalOutcomeKind` via the explicit table in
    :func:`feedback_valence_to_outcome_kind` — never via free-text
    inference.
    """
    payload = envelope.feedback
    if payload is None:
        raise DispatchError(
            code="missing_feedback_payload",
            detail=(
                "interaction_type=feedback requires a 'feedback' object "
                "with at least 'valence'."
            ),
        )
    valence = _parse_feedback_valence(payload)
    kind = feedback_valence_to_outcome_kind(valence)
    confidence = _coerce_unit_float(
        payload.intensity, field="feedback.intensity", default=0.9
    )
    description = (payload.evidence or envelope.human_brief or "").strip()
    target_response_id = payload.target_response_id.strip() or None
    evidence = session.submit_dialogue_outcome(
        kind=kind,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=confidence,
        evidence_ref=target_response_id,
        description=description,
    )
    notice = (
        f"feedback recorded: valence={valence.value!r} → "
        f"outcome_kind={kind.value!r}"
    )
    return ok_envelope(
        ai_id=ai_id,
        contract_id=envelope.contract_id,
        session_id=envelope.session_id,
        interaction_type=envelope.interaction_type.value,
        output_acts=(system_act(notice),),
        protocol_version=envelope.protocol_version,
        extra={
            "feedback": {
                "valence": valence.value,
                "outcome_kind": kind.value,
                "scope": payload.scope,
                "target_response_id": target_response_id or "",
            },
            "evidence_id": getattr(evidence, "evidence_id", ""),
        },
    )


def _parse_feedback_valence(payload: FeedbackPayload) -> FeedbackValence:
    raw = (payload.valence or "").strip().lower()
    if not raw:
        raise DispatchError(
            code="missing_feedback_valence",
            detail="feedback.valence is required and must be non-empty.",
        )
    try:
        return FeedbackValence(raw)
    except ValueError as exc:
        allowed = ", ".join(v.value for v in FeedbackValence)
        raise DispatchError(
            code="invalid_feedback_valence",
            detail=f"feedback.valence must be one of: {allowed}",
        ) from exc


# ---------------------------------------------------------------------------
# Slice 2.2 — observe
# ---------------------------------------------------------------------------


async def _handle_observe(
    *,
    envelope: InteractionEnvelope,
    session: Any,
    ai_id: str,
) -> dict[str, Any]:
    """Route an observation to the matching kernel ingestion sink.

    The observation type is read from
    ``envelope.structured_context['observation_type']`` and parsed
    as :class:`ObservationType`. Each kind has its own required
    fields; missing fields raise a typed ``DispatchError``. There is
    no default sink — unknown observation types fail at the edge.
    """
    obs_type = _parse_observation_type(envelope)
    ctx = envelope.structured_context

    if obs_type is ObservationType.HOMEWORK_RESULT:
        return _emit_observe_response(
            envelope=envelope,
            ai_id=ai_id,
            obs_type=obs_type,
            event_ids=session.submit_task_event(
                event_id=_required_str(ctx, "event_id", obs_type),
                task_id=_required_str(ctx, "task_id", obs_type),
                status=_required_str(ctx, "status", obs_type),
                summary=_required_str(ctx, "summary", obs_type),
                detail=_optional_str(ctx, "detail"),
                due_hint=_optional_str(ctx, "due_hint", default=None),
                commitment_ref=_optional_str(ctx, "commitment_ref", default=None),
                confidence=_optional_unit_float(ctx, "confidence", default=0.75),
            ),
        )

    if obs_type in (ObservationType.CLASS_NOTE, ObservationType.TEACHER_NOTE):
        return _emit_observe_response(
            envelope=envelope,
            ai_id=ai_id,
            obs_type=obs_type,
            event_ids=session.submit_reviewed_knowledge_event(
                event_id=_required_str(ctx, "event_id", obs_type),
                knowledge_id=_required_str(ctx, "knowledge_id", obs_type),
                summary=_required_str(ctx, "summary", obs_type, default=envelope.human_brief),
                detail=_optional_str(ctx, "detail", default=envelope.human_brief),
                source_label=_optional_str(
                    ctx, "source_label", default=obs_type.value
                ),
                confidence=_optional_unit_float(ctx, "confidence", default=0.8),
                relevance_hint=_optional_str(ctx, "relevance_hint"),
                needs_followup=bool(ctx.get("needs_followup", False)),
            ),
        )

    if obs_type is ObservationType.PROFILE_UPDATE:
        return _emit_observe_response(
            envelope=envelope,
            ai_id=ai_id,
            obs_type=obs_type,
            event_ids=session.submit_profile_event(
                event_id=_required_str(ctx, "event_id", obs_type),
                source=_optional_str(ctx, "source", default=envelope.end_user_ref),
                preferences=_string_tuple(ctx.get("preferences")),
                goals=_string_tuple(ctx.get("goals")),
                consent_grants=_string_tuple(ctx.get("consent_grants")),
                consent_denials=_string_tuple(ctx.get("consent_denials")),
                relationship_note=_optional_str(ctx, "relationship_note"),
                confidence=_optional_unit_float(ctx, "confidence", default=0.75),
            ),
        )

    if obs_type is ObservationType.TOOL_RESULT:
        plan_ref = _optional_str(ctx, "plan_ref", default=None) or None
        artifact_refs = _optional_str_sequence(ctx, "artifact_refs")
        tool_result_kwargs: dict[str, Any] = {
            "event_id": _required_str(ctx, "event_id", obs_type),
            "tool_name": _required_str(ctx, "tool_name", obs_type),
            "action_id": _required_str(ctx, "action_id", obs_type),
            "status": _required_str(ctx, "status", obs_type),
            "summary": _required_str(ctx, "summary", obs_type),
            "detail": _optional_str(ctx, "detail"),
            "confidence": _optional_unit_float(ctx, "confidence", default=0.8),
        }
        if plan_ref:
            tool_result_kwargs["plan_ref"] = plan_ref
        if artifact_refs:
            tool_result_kwargs["artifact_refs"] = artifact_refs
        return _emit_observe_response(
            envelope=envelope,
            ai_id=ai_id,
            obs_type=obs_type,
            event_ids=session.submit_tool_result(**tool_result_kwargs),
        )

    if obs_type is ObservationType.CORPUS_INGEST:
        return await _handle_corpus_ingest(
            envelope=envelope, session=session, ai_id=ai_id, ctx=ctx
        )

    if obs_type is ObservationType.GENERIC_SEMANTIC:
        # GENERIC_SEMANTIC requires a typed ExternalSemanticEventBatch
        # which is non-trivial to construct from raw JSON. Slice 2 does
        # not wire this — Slice 7.x will when we have a proper
        # typed-batch parser shared with the kernel.
        raise DispatchError(
            code="not_implemented",
            detail=(
                "observation_type='generic_semantic' is reserved for Slice 7; "
                "use a more specific observation_type for now."
            ),
            status=501,
        )

    raise DispatchError(  # pragma: no cover - exhaustive
        code="unsupported_observation_type",
        detail=(
            f"observation_type={obs_type!r} has no kernel sink in Slice 2."
        ),
        status=501,
    )


async def _handle_corpus_ingest(
    *,
    envelope: InteractionEnvelope,
    session: Any,
    ai_id: str,
    ctx: Mapping[str, Any],
) -> dict[str, Any]:
    """CORPUS_INGEST → ``IngestionPipeline.process_envelope``.

    The platform builds an in-memory ``IngestionEnvelope`` from
    ``human_brief`` (or an explicit ``corpus_text``) plus the optional
    ``source_uri`` provenance and runs the standard pipeline. This
    keeps long-form material on the same code path as PDF / DOCX
    adapters — the kernel's ``IngestionPipeline`` is the single
    entry point.
    """
    text = (
        _optional_str(ctx, "corpus_text")
        or envelope.human_brief
    ).strip()
    if not text:
        raise DispatchError(
            code="missing_corpus_text",
            detail=(
                "observation_type='corpus_ingest' requires either "
                "structured_context.corpus_text or a non-empty human_brief."
            ),
        )
    source_uri = _optional_str(
        ctx, "source_uri", default=f"dlaas:{envelope.session_id}"
    )
    uploader = _optional_str(ctx, "uploader", default=envelope.end_user_ref)
    ingestion_envelope = envelope_from_text(
        text,
        source_uri=source_uri,
        uploader=uploader or "system",
        source_kind=IngestionSourceKind.CORPUS,
        compliance_profile=IngestionComplianceProfile.FORCED,
    )
    pipeline = IngestionPipeline()
    report = await pipeline.process_envelope(
        ingestion_envelope,
        session=session,
        end_scene_after=False,
    )
    notice = (
        f"corpus ingested: {report.processed_chunks}/{report.total_chunks} "
        f"chunks ok, skipped={report.skipped_chunks}"
    )
    return ok_envelope(
        ai_id=ai_id,
        contract_id=envelope.contract_id,
        session_id=envelope.session_id,
        interaction_type=envelope.interaction_type.value,
        output_acts=(system_act(notice),),
        protocol_version=envelope.protocol_version,
        extra={
            "ingestion_report": {
                "envelope_id": ingestion_envelope.envelope_id,
                "total_chunks": report.total_chunks,
                "processed_chunks": report.processed_chunks,
                "skipped_chunks": report.skipped_chunks,
                "all_succeeded": report.all_succeeded,
            },
            "observation_type": ObservationType.CORPUS_INGEST.value,
        },
    )


def _emit_observe_response(
    *,
    envelope: InteractionEnvelope,
    ai_id: str,
    obs_type: ObservationType,
    event_ids: tuple[str, ...],
) -> dict[str, Any]:
    notice = (
        f"observation recorded: type={obs_type.value!r}, "
        f"events={len(event_ids)}"
    )
    return ok_envelope(
        ai_id=ai_id,
        contract_id=envelope.contract_id,
        session_id=envelope.session_id,
        interaction_type=envelope.interaction_type.value,
        output_acts=(system_act(notice),),
        protocol_version=envelope.protocol_version,
        extra={
            "observation_type": obs_type.value,
            "event_ids": list(event_ids),
        },
    )


def _parse_observation_type(envelope: InteractionEnvelope) -> ObservationType:
    raw = envelope.structured_context.get("observation_type")
    if not isinstance(raw, str) or not raw.strip():
        raise DispatchError(
            code="missing_observation_type",
            detail=(
                "interaction_type=observe requires "
                "structured_context.observation_type (typed enum)."
            ),
        )
    try:
        return ObservationType(raw.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(t.value for t in ObservationType)
        raise DispatchError(
            code="invalid_observation_type",
            detail=f"observation_type must be one of: {allowed}",
        ) from exc


# ---------------------------------------------------------------------------
# Slice 2.3 — teach / task (apprentice)
# ---------------------------------------------------------------------------


async def _handle_apprentice(
    *,
    envelope: InteractionEnvelope,
    session: Any,
    ai_id: str,
) -> dict[str, Any]:
    """Apprentice-mode turns (``teach`` / ``task``).

    Both kinds run the regular cognition pipeline but flip the kernel's
    ``vitals.apprentice_override`` so drive deviation does not feed
    slow-scale PE for this turn (see
    :func:`lifeform_core.types.is_apprenticeship_trigger`). The two
    interaction types differ only in audit label — the wire format
    keeps them separate so an integrator can filter the ledger by
    intent later.
    """
    if not envelope.human_brief.strip():
        raise DispatchError(
            code="invalid_human_brief",
            detail=(
                f"interaction_type={envelope.interaction_type.value} "
                f"requires a non-empty human_brief."
            ),
        )
    result = await session.run_turn(
        envelope.human_brief, trigger_kind=TurnTriggerKind.APPRENTICE
    )
    response_text = getattr(result.response, "text", "") or ""
    rationale_tags = tuple(getattr(result.response, "rationale_tags", ()) or ())
    return ok_envelope(
        ai_id=ai_id,
        contract_id=envelope.contract_id,
        session_id=envelope.session_id,
        interaction_type=envelope.interaction_type.value,
        output_acts=(text_act(response_text),),
        protocol_version=envelope.protocol_version,
        extra={
            "active_regime": getattr(result, "active_regime", None),
            "active_abstract_action": getattr(result, "active_abstract_action", None),
            "rationale_tags": list(rationale_tags),
            "trigger_kind": TurnTriggerKind.APPRENTICE.value,
            "mode": envelope.mode.value,
        },
    )


# ---------------------------------------------------------------------------
# Slice 2.4 — report + command
# ---------------------------------------------------------------------------


async def _handle_report(
    *,
    envelope: InteractionEnvelope,
    session: Any,
    ai_id: str,
) -> dict[str, Any]:
    """Drain the slow loop and surface a minimal report scaffold.

    Slice 2.4 implements the kernel-side hook (``end_scene`` with
    ``drain_slow_loop=True``), which forces R6 reflection writeback to
    settle. The actual report rendering (per-week / per-person
    structured projections) is a Slice 6 / Slice 7 readout from
    reflection snapshots — by then those projections live behind a
    typed snapshot enrichment, not behind ad-hoc text formatting.
    """
    closed = await session.end_scene(
        reason=f"dlaas-report:{envelope.session_id}", drain_slow_loop=True
    )
    scene_id = getattr(closed, "scene_id", None) if closed is not None else None
    notice = (
        "report drain complete: slow loop drained; reflection snapshot ready"
    )
    return ok_envelope(
        ai_id=ai_id,
        contract_id=envelope.contract_id,
        session_id=envelope.session_id,
        interaction_type=envelope.interaction_type.value,
        output_acts=(system_act(notice),),
        protocol_version=envelope.protocol_version,
        extra={
            "drained": True,
            "scene_id": scene_id,
            "report_view": None,  # Slice 6 will populate from reflection snapshot.
        },
    )


async def _handle_command(
    *,
    envelope: InteractionEnvelope,
    session: Any,
    ai_id: str,
) -> dict[str, Any]:
    """Typed command allowlist dispatch.

    DLaaS public clients put the command identifier in ``human_brief``
    (see ``DLAAS_README.md`` §"command"). The platform parses it as a
    :class:`CommandName` enum — anything outside the allowlist is
    rejected at the edge.

    * ``REFRESH_PERSON_CONTEXT`` →
      :meth:`LifeformSession.submit_profile_event` carrying the
      caller-provided ``relationship_note`` (DLaaS doc-stated reason).
    * ``END_SCENE`` →
      :meth:`LifeformSession.end_scene` without draining the slow loop.
    * ``PAUSE_SESSION`` / ``RESUME_SESSION`` → Slice 5.1 placeholders.
      Today the platform records a ``system`` OutputAct describing
      the pending wiring; the ops state machine lands in Slice 5.1.
    """
    cmd = _parse_command_name(envelope)
    ctx = envelope.structured_context

    if cmd is CommandName.REFRESH_PERSON_CONTEXT:
        if not envelope.target_person_ids:
            raise DispatchError(
                code="missing_target_person_ids",
                detail=(
                    "command=refresh_person_context requires "
                    "target_person_ids on the envelope (or in "
                    "structured_context)."
                ),
            )
        event_ids = session.submit_profile_event(
            event_id=_optional_str(
                ctx, "event_id", default=f"refresh:{envelope.session_id}"
            ),
            source=_optional_str(ctx, "source", default=envelope.end_user_ref),
            relationship_note=_optional_str(
                ctx,
                "relationship_note",
                default=(
                    "refresh_person_context: re-pull profile snapshot for "
                    + ",".join(envelope.target_person_ids)
                ),
            ),
            confidence=_optional_unit_float(ctx, "confidence", default=0.75),
        )
        notice = (
            f"command refresh_person_context: persons="
            f"{list(envelope.target_person_ids)}"
        )
        return ok_envelope(
            ai_id=ai_id,
            contract_id=envelope.contract_id,
            session_id=envelope.session_id,
            interaction_type=envelope.interaction_type.value,
            output_acts=(system_act(notice),),
            protocol_version=envelope.protocol_version,
            extra={
                "command": cmd.value,
                "event_ids": list(event_ids),
                "target_person_ids": list(envelope.target_person_ids),
            },
        )

    if cmd is CommandName.END_SCENE:
        closed = await session.end_scene(
            reason=f"dlaas-command:{cmd.value}", drain_slow_loop=False
        )
        scene_id = getattr(closed, "scene_id", None) if closed is not None else None
        return ok_envelope(
            ai_id=ai_id,
            contract_id=envelope.contract_id,
            session_id=envelope.session_id,
            interaction_type=envelope.interaction_type.value,
            output_acts=(system_act(f"command end_scene: scene_id={scene_id!r}"),),
            protocol_version=envelope.protocol_version,
            extra={"command": cmd.value, "scene_id": scene_id, "drained": False},
        )

    if cmd in (CommandName.PAUSE_SESSION, CommandName.RESUME_SESSION):
        notice = (
            f"command {cmd.value}: ops state machine lands in Slice 5.1; "
            f"recorded but not yet active."
        )
        return ok_envelope(
            ai_id=ai_id,
            contract_id=envelope.contract_id,
            session_id=envelope.session_id,
            interaction_type=envelope.interaction_type.value,
            output_acts=(system_act(notice),),
            protocol_version=envelope.protocol_version,
            extra={"command": cmd.value, "ops_pending": "Slice 5.1"},
        )

    if cmd is CommandName.INITIATE_PROACTIVE_FOLLOWUP:
        followup_brief = _optional_str(ctx, "followup_brief", default="").strip()
        if not followup_brief:
            raise DispatchError(
                code="missing_followup_brief",
                detail=(
                    "command=initiate_proactive_followup requires "
                    "structured_context.followup_brief (the message the "
                    "lifeform should send to the user). The brief is "
                    "produced by the platform-ops OutboundScheduler "
                    "from a vertical-supplied template; the kernel does "
                    "not infer it from chat text."
                ),
            )
        followup_evidence = _optional_str(ctx, "followup_evidence_ref", default="")
        result = await session.run_turn(
            followup_brief, trigger_kind=TurnTriggerKind.APPRENTICE
        )
        response_text = getattr(result.response, "text", "") or ""
        rationale_tags = tuple(getattr(result.response, "rationale_tags", ()) or ())
        return ok_envelope(
            ai_id=ai_id,
            contract_id=envelope.contract_id,
            session_id=envelope.session_id,
            interaction_type=envelope.interaction_type.value,
            output_acts=(text_act(response_text),),
            protocol_version=envelope.protocol_version,
            extra={
                "command": cmd.value,
                "trigger_kind": TurnTriggerKind.APPRENTICE.value,
                "active_regime": getattr(result, "active_regime", None),
                "active_abstract_action": getattr(
                    result, "active_abstract_action", None
                ),
                "rationale_tags": list(rationale_tags),
                "followup_evidence_ref": followup_evidence,
            },
        )

    raise DispatchError(  # pragma: no cover - exhaustive
        code="unsupported_command",
        detail=f"command={cmd!r} has no handler.",
        status=500,
    )


def _parse_command_name(envelope: InteractionEnvelope) -> CommandName:
    raw = (envelope.human_brief or "").strip().lower()
    if not raw:
        raise DispatchError(
            code="missing_command",
            detail=(
                "interaction_type=command requires the command name in "
                "human_brief (typed CommandName)."
            ),
        )
    try:
        return CommandName(raw)
    except ValueError as exc:
        allowed = ", ".join(c.value for c in CommandName)
        raise DispatchError(
            code="invalid_command",
            detail=f"command must be one of: {allowed}",
        ) from exc


# ---------------------------------------------------------------------------
# Small typed-validation helpers
# ---------------------------------------------------------------------------


def _required_str(
    ctx: Mapping[str, Any],
    key: str,
    obs_type: ObservationType,
    *,
    default: str | None = None,
) -> str:
    value = ctx.get(key, default)
    if value is None or not str(value).strip():
        raise DispatchError(
            code="missing_field",
            detail=(
                f"observation_type={obs_type.value!r} requires "
                f"structured_context.{key} (non-empty string)."
            ),
        )
    return str(value)


def _optional_str(
    ctx: Mapping[str, Any],
    key: str,
    *,
    default: str | None = "",
) -> str:
    value = ctx.get(key, default)
    if value is None:
        return default if default is not None else ""
    return str(value)


def _optional_int(
    ctx: Mapping[str, Any],
    key: str,
) -> int | None:
    """Read an optional integer field from structured_context.

    Used by callers that want to override a default policy bound
    (e.g. ``tool_loop_max_steps`` / ``tool_loop_max_wall_ms``).
    Returns ``None`` when the key is absent or the value cannot be
    coerced to ``int`` cleanly — the caller decides the fallback.
    """
    raw = ctx.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError:
            return None
    return None


def _optional_str_sequence(ctx: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = ctx.get(key, ())
    if raw is None:
        return ()
    if isinstance(raw, str):
        raise DispatchError(
            code="invalid_field_type",
            detail=f"structured_context.{key} must be a list of strings, not a string.",
        )
    try:
        values = tuple(raw)
    except TypeError as exc:
        raise DispatchError(
            code="invalid_field_type",
            detail=f"structured_context.{key} must be a list of strings.",
        ) from exc
    out: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise DispatchError(
                code="invalid_field_type",
                detail=f"structured_context.{key} entries must be non-empty strings.",
            )
        out.append(value.strip())
    return tuple(out)


def _optional_unit_float(
    ctx: Mapping[str, Any],
    key: str,
    *,
    default: float,
) -> float:
    if key not in ctx:
        return default
    raw = ctx[key]
    return _coerce_unit_float(raw, field=f"structured_context.{key}", default=default)


def _coerce_unit_float(raw: Any, *, field: str, default: float) -> float:
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise DispatchError(
            code="invalid_field_type",
            detail=f"{field} must be a numeric value in [0,1].",
        )
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise DispatchError(
            code="invalid_field_range",
            detail=f"{field} must be in [0,1], got {value!r}.",
        )
    return value


def _string_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        raise DispatchError(
            code="invalid_field_type",
            detail=(
                "expected a list of strings, got a single string. "
                "Wrap the value in a JSON array even if it has one element."
            ),
        )
    return tuple(str(item) for item in raw)


__all__ = [
    "DispatchError",
    "OutputAct",
    "dispatch_envelope",
]
