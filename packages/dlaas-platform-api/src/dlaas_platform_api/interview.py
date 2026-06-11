"""aiohttp routes for interactive interview runs (面试 gate).

Endpoint summary (tenant auth, template-ownership checked — mirrors
the eval wheel's exam surface)::

    POST /dlaas/interview_runs                                — create
    GET  /dlaas/interview_runs/{run_id}                       — read
    GET  /dlaas/templates/{template_id}/interview_runs        — list
    POST /dlaas/interview_runs/{run_id}/turns                 — record turn
    POST /dlaas/interview_runs/{run_id}/turns/execute         — live turn
    POST /dlaas/interview_runs/{run_id}/turns/{turn_index}/score
    POST /dlaas/interview_runs/{run_id}/complete              — verdict

Flow:

* Create a run with a ``question_plan`` (questions typically come from
  the eval wheel's question generation or the app's role spec) and an
  optional live ``ai_id``.
* ``turns/execute`` asks the next planned question (or an explicit
  follow-up) against the live instance and records the response.
* Turns are scored per-turn — by an injected rubric grader (LLM
  interviewer mode) or by an operator via the score endpoint (human
  interviewer mode). Mixed mode is simply both surfaces on one run.
* ``complete`` refuses to finalize while any turn is unscored — the
  interview verdict can never be silently granted.

Layering (R12): interview scores are readouts; no learning signal
flows back into the kernel from this surface.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import (
    InterviewerKind,
    RubricEntry,
)
from dlaas_platform_launcher import (
    INSTANCE_MANAGER_APP_KEY,
    InstanceManager,
    InstanceNotFound,
)
from dlaas_platform_registry import (
    InterviewRunNotFound,
    InterviewRunStateError,
    InterviewRunStore,
    REGISTRY_APP_KEY,
    Registry,
    TemplateNotFound,
    TemplateStore,
    require_control_plane_or_service,
    require_tenant_auth,
)

_LOG = logging.getLogger("dlaas_platform_api.interview")

INTERVIEW_BUNDLE_APP_KEY = "dlaas_interview_bundle"

#: Default rubric used when an injected grader scores a live turn.
_INTERVIEW_RUBRIC = (
    RubricEntry(
        criterion="角色一致性与专业可信度",
        description=(
            "回答是否与该人物/角色的知识边界、表达风格一致，"
            "专业判断是否可信、有依据。"
        ),
        max_score=10.0,
        weight=1.0,
    ),
)


class InterviewBundle:
    """Container the api wheel reads to dispatch interview state."""

    __slots__ = ("runs", "templates", "grader")

    def __init__(self, *, registry: Registry, grader: Any | None = None) -> None:
        self.runs = InterviewRunStore(registry)
        self.templates = TemplateStore(registry)
        # Optional RubricGrader (duck-typed; see dlaas_platform_eval.grader).
        # None = operator-scored turns only.
        self.grader = grader


def attach_interview_routes(
    app: web.Application,
    *,
    registry: Registry,
    grader: Any | None = None,
) -> web.Application:
    if REGISTRY_APP_KEY not in app:
        raise ValueError(
            "attach_interview_routes requires app[REGISTRY_APP_KEY] "
            "(dlaas_platform_api.build_dlaas_app handles this)."
        )
    app[INTERVIEW_BUNDLE_APP_KEY] = InterviewBundle(
        registry=registry, grader=grader
    )
    R = app.router
    R.add_post("/dlaas/interview_runs", _handle_create)
    R.add_get("/dlaas/interview_runs/{run_id}", _handle_get)
    R.add_get(
        "/dlaas/templates/{template_id}/interview_runs",
        _handle_list_for_template,
    )
    R.add_post("/dlaas/interview_runs/{run_id}/turns", _handle_record_turn)
    R.add_post(
        "/dlaas/interview_runs/{run_id}/turns/execute", _handle_execute_turn
    )
    R.add_post(
        "/dlaas/interview_runs/{run_id}/turns/{turn_index}/score",
        _handle_score_turn,
    )
    R.add_post("/dlaas/interview_runs/{run_id}/complete", _handle_complete)
    return app


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_create(request: web.Request) -> web.Response:
    bundle = _bundle(request)
    caller = await _caller(request)
    data = await _read_json(request)
    template_id = str(data.get("template_id", "") or "")
    if not template_id:
        return _error(400, "missing_template_id", "template_id is required")
    template = await _owned_template(bundle, caller, template_id)
    if isinstance(template, web.Response):
        return template
    raw_kind = str(data.get("interviewer_kind", "operator") or "operator")
    try:
        interviewer_kind = InterviewerKind(raw_kind)
    except ValueError:
        valid = ", ".join(k.value for k in InterviewerKind)
        return _error(
            400,
            "invalid_interviewer_kind",
            f"unknown interviewer_kind {raw_kind!r}; expected one of: {valid}",
        )
    question_plan = data.get("question_plan") or []
    if not isinstance(question_plan, list) or not all(
        isinstance(q, str) and q.strip() for q in question_plan
    ):
        return _error(
            400,
            "invalid_question_plan",
            "question_plan must be a list of non-empty strings",
        )
    try:
        run = await bundle.runs.create(
            template_id=template_id,
            template_version=int(data.get("template_version", 1) or 1),
            ai_id=str(data.get("ai_id", "") or ""),
            session_id=str(data.get("session_id", "") or ""),
            interviewer_kind=interviewer_kind,
            question_plan=tuple(question_plan),
            pass_threshold=float(data.get("pass_threshold", 0.6) or 0.6),
        )
    except InterviewRunStateError as exc:
        return _error(400, "invalid_interview_run", str(exc))
    return web.json_response({"status": "ok", **run.to_json()})


async def _handle_get(request: web.Request) -> web.Response:
    bundle = _bundle(request)
    caller = await _caller(request)
    run = await _owned_run(request, bundle, caller)
    if isinstance(run, web.Response):
        return run
    return web.json_response({"status": "ok", **run.to_json()})


async def _handle_list_for_template(request: web.Request) -> web.Response:
    bundle = _bundle(request)
    caller = await _caller(request)
    template_id = request.match_info["template_id"]
    template = await _owned_template(bundle, caller, template_id)
    if isinstance(template, web.Response):
        return template
    runs = await bundle.runs.list_for_template(template_id)
    return web.json_response(
        {"status": "ok", "interview_runs": [r.to_json() for r in runs]}
    )


async def _handle_record_turn(request: web.Request) -> web.Response:
    """Record a caller-supplied turn (human interviewer / offline transcript)."""

    bundle = _bundle(request)
    caller = await _caller(request)
    run = await _owned_run(request, bundle, caller)
    if isinstance(run, web.Response):
        return run
    data = await _read_json(request)
    question = str(data.get("question", "") or "")
    if not question.strip():
        return _error(400, "missing_question", "question is required")
    raw_score = data.get("score")
    score = None if raw_score is None else float(raw_score)
    try:
        updated = await bundle.runs.append_turn(
            run_id=run.run_id,
            question=question,
            ai_response=str(data.get("ai_response", "") or ""),
            asked_by=str(data.get("asked_by", "operator") or "operator"),
            score=score,
            notes=str(data.get("notes", "") or ""),
        )
    except (InterviewRunStateError, ValueError) as exc:
        return _error(409, "invalid_turn", str(exc))
    return web.json_response({"status": "ok", **updated.to_json()})


async def _handle_execute_turn(request: web.Request) -> web.Response:
    """Ask the next question against the run's live instance.

    Body: ``{ question?, asked_by? }``. When ``question`` is omitted the
    next unasked entry of the run's ``question_plan`` is used (409
    ``plan_exhausted`` when none remain). When a rubric grader is
    configured on the bundle, the turn is auto-scored; otherwise the
    score stays empty for the operator to fill in.
    """

    bundle = _bundle(request)
    caller = await _caller(request)
    run = await _owned_run(request, bundle, caller)
    if isinstance(run, web.Response):
        return run
    if not run.ai_id:
        return _error(
            409,
            "no_live_instance",
            "this interview run has no ai_id; record turns manually via "
            "POST .../turns instead",
        )
    data = await _read_json(request, allow_empty=True)
    question = str(data.get("question", "") or "")
    asked_by = str(data.get("asked_by", "llm") or "llm")
    if not question.strip():
        asked = {t.question for t in run.turns}
        remaining = [q for q in run.question_plan if q not in asked]
        if not remaining:
            return _error(
                409,
                "plan_exhausted",
                "question_plan is exhausted; supply an explicit question",
            )
        question = remaining[0]

    session = await _interview_session(request, run=run)
    if isinstance(session, web.Response):
        return session
    from lifeform_core.types import TurnTriggerKind

    try:
        result = await session.run_turn(
            question, trigger_kind=TurnTriggerKind.APPRENTICE
        )
        response_text = result.response.text
    except (RuntimeError, ValueError) as exc:
        _LOG.warning(
            "interview execute: kernel raised on run=%s: %s", run.run_id, exc
        )
        return _error(502, "turn_failed", f"kernel turn failed: {exc}")

    score: float | None = None
    notes = ""
    if bundle.grader is not None:
        graded = bundle.grader.grade(
            rubric=_INTERVIEW_RUBRIC,
            ai_response=response_text,
            reference_answer="",
        )
        score = float(graded.weighted_score)
        notes = "auto-graded"

    try:
        updated = await bundle.runs.append_turn(
            run_id=run.run_id,
            question=question,
            ai_response=response_text,
            asked_by=asked_by,
            score=score,
            notes=notes,
        )
    except InterviewRunStateError as exc:
        return _error(409, "invalid_turn", str(exc))
    return web.json_response({"status": "ok", **updated.to_json()})


async def _handle_score_turn(request: web.Request) -> web.Response:
    bundle = _bundle(request)
    caller = await _caller(request)
    run = await _owned_run(request, bundle, caller)
    if isinstance(run, web.Response):
        return run
    data = await _read_json(request)
    raw_index = request.match_info["turn_index"]
    try:
        turn_index = int(raw_index)
    except ValueError:
        return _error(400, "invalid_turn_index", f"not an integer: {raw_index!r}")
    raw_score = data.get("score")
    if raw_score is None:
        return _error(400, "missing_score", "score is required")
    try:
        updated = await bundle.runs.score_turn(
            run_id=run.run_id,
            turn_index=turn_index,
            score=float(raw_score),
            notes=str(data.get("notes", "") or ""),
        )
    except InterviewRunStateError as exc:
        return _error(409, "invalid_score", str(exc))
    return web.json_response({"status": "ok", **updated.to_json()})


async def _handle_complete(request: web.Request) -> web.Response:
    bundle = _bundle(request)
    caller = await _caller(request)
    run = await _owned_run(request, bundle, caller)
    if isinstance(run, web.Response):
        return run
    data = await _read_json(request, allow_empty=True)
    try:
        updated = await bundle.runs.complete(
            run_id=run.run_id,
            operator_id=str(data.get("operator_id", "") or ""),
            verdict_comment=str(data.get("comment", "") or ""),
        )
    except InterviewRunStateError as exc:
        return _error(409, "incomplete_interview", str(exc))
    return web.json_response({"status": "ok", **updated.to_json()})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _bundle(request: web.Request) -> InterviewBundle:
    return request.app[INTERVIEW_BUNDLE_APP_KEY]


_OPERATOR = "operator"


async def _caller(request: web.Request):
    """Resolve the caller: operator secrets (cross-tenant) or tenant spec.

    Operator credentials (`X-Control-Plane-Secret` / `X-Service-Secret`)
    let the operator console run and score interviews for any tenant's
    persona; tenant credentials act only on templates their tenant owns.
    """

    headers = request.headers
    if "X-Control-Plane-Secret" in headers or "X-Service-Secret" in headers:
        require_control_plane_or_service(request)
        return _OPERATOR
    return await require_tenant_auth(request)


async def _owned_template(bundle: InterviewBundle, caller, template_id: str):
    try:
        template = await bundle.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)
    if caller is _OPERATOR:
        return template
    if template.tenant_id != caller.tenant_id:
        return _error(
            403,
            "tenant_mismatch",
            (
                f"authenticated tenant_id={caller.tenant_id!r} cannot act "
                f"on a template owned by tenant_id={template.tenant_id!r}"
            ),
        )
    return template


async def _owned_run(request: web.Request, bundle: InterviewBundle, caller):
    run_id = request.match_info["run_id"]
    try:
        run = await bundle.runs.get(run_id)
    except InterviewRunNotFound:
        return _error(404, "interview_run_not_found", run_id)
    template = await _owned_template(bundle, caller, run.template_id)
    if isinstance(template, web.Response):
        return template
    return run


async def _interview_session(request: web.Request, *, run):
    instance_manager = request.app.get(INSTANCE_MANAGER_APP_KEY)
    if not isinstance(instance_manager, InstanceManager):
        return _error(
            503,
            "launcher_not_available",
            "interview execution requires an InstanceManager bound to the app",
        )
    try:
        manager = instance_manager.get(run.ai_id)
    except InstanceNotFound:
        return _error(
            409,
            "instance_not_awake",
            f"ai_id={run.ai_id!r} is not awake; wake it before executing "
            "interview turns",
        )
    session_id = run.session_id or f"interview:{run.run_id}"
    try:
        return await manager.get_session(session_id)
    except LookupError:
        return await manager.create_session(session_id=session_id)


async def _read_json(
    request: web.Request, *, allow_empty: bool = False
) -> Mapping[str, Any]:
    if not request.body_exists:
        if allow_empty:
            return {}
        raise _bad_request("missing_body", "Body required")
    text = await request.text()
    if not text.strip():
        if allow_empty:
            return {}
        raise _bad_request("missing_body", "Empty body")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _bad_request("invalid_json", f"Body is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _bad_request("invalid_envelope", "Top-level body must be a JSON object")
    return data


def _bad_request(code: str, detail: str) -> web.HTTPBadRequest:
    return web.HTTPBadRequest(
        text=json.dumps({"status": "error", "error": code, "detail": detail}),
        content_type="application/json",
    )


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        {"status": "error", "error": code, "detail": detail}, status=status
    )


__all__ = [
    "INTERVIEW_BUNDLE_APP_KEY",
    "InterviewBundle",
    "attach_interview_routes",
]
