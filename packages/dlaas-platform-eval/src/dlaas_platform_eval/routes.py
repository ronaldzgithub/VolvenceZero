"""aiohttp routes for the eval gate (Slice 6.1 - 6.3).

Endpoint summary (mirrors DLaaS public surface):

* ``POST /dlaas/templates/{template_id}/audience/analyze``
* ``GET  /dlaas/templates/{template_id}/audience``
* ``GET  /dlaas/audience/{profile_id}``
* ``POST /dlaas/exam_questions``
* ``POST /dlaas/exam_questions/batch``
* ``POST /dlaas/exam_questions/generate``
* ``GET  /dlaas/exam_questions``
* ``POST /dlaas/exam_runs``
* ``GET  /dlaas/exam_runs/{run_id}``
* ``POST /dlaas/exam_runs/{run_id}/complete``
* ``POST /dlaas/exam_runs/{run_id}/execute``
* ``GET  /dlaas/templates/{template_id}/exam_runs``
* ``POST /dlaas/exam_runs/{run_id}/signoff``
* ``GET  /dlaas/templates/{template_id}/license``
* ``POST /dlaas/templates/{template_id}/license/evaluate``

All endpoints require tenant credentials. The handlers consult the
template ownership before allowing reads / writes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import (
    AudienceProfileSpec,
    ExamRunSpec,
    ExamRunStatus,
    ExamSubmissionScore,
    RubricEntry,
)
from dlaas_platform_launcher import (
    INSTANCE_MANAGER_APP_KEY,
    InstanceManager,
    InstanceNotFound,
)
from dlaas_platform_registry import (
    AudienceProfileNotFound,
    EvalStore,
    ExamQuestionNotFound,
    ExamRunNotFound,
    LaunchLicenseNotFound,
    REGISTRY_APP_KEY,
    Registry,
    TemplateNotFound,
    TemplateStore,
    assert_tenant_id_matches,
    require_tenant_auth,
)
from dlaas_platform_registry.assets import AssetNotFound, AssetStore

from dlaas_platform_eval import question_gen
from dlaas_platform_eval.audience import (
    AssetCorpusError,
    AudienceAnalysisError,
    LLMAudienceAnalyzer,
    build_audience_analyzer_from_env,
    load_asset_corpus,
)
from dlaas_platform_eval.grader import RubricGrader
from dlaas_platform_eval.llm_grader import (
    GraderResponseError,
    QuestionGenerationError,
    build_grader_from_env,
    resolve_eval_llm_config,
)

_LOG = logging.getLogger("dlaas_platform_eval")

EVAL_BUNDLE_APP_KEY = "dlaas_eval_bundle"


class EvalBundle:
    """Container the api wheel reads to dispatch eval state."""

    __slots__ = ("eval_store", "templates", "assets", "grader", "audience_analyzer")

    def __init__(
        self,
        *,
        registry: Registry,
        grader: RubricGrader | None = None,
        audience_analyzer: LLMAudienceAnalyzer | None = None,
    ) -> None:
        self.eval_store = EvalStore(registry)
        self.templates = TemplateStore(registry)
        self.assets = AssetStore(registry)
        # Default grader comes from the env-driven factory: a real
        # LLMRubricGrader when EVAL_LLM_* / PROTOCOL_LLM_* is
        # configured, else the fail-closed DefaultRubricGrader (which
        # logs its loud warning). Tests pass an explicit ``grader=``.
        self.grader = grader or build_grader_from_env()
        # Debt #14: audience analysis shares the same env contract. A
        # ``None`` analyzer keeps the honest SHADOW passthrough
        # (caller-supplied fields verbatim, evidence_stats.analyzer
        # = "none").
        self.audience_analyzer = (
            audience_analyzer or build_audience_analyzer_from_env()
        )


def attach_eval_routes(
    app: web.Application,
    *,
    registry: Registry,
    grader: RubricGrader | None = None,
    audience_analyzer: LLMAudienceAnalyzer | None = None,
) -> web.Application:
    if REGISTRY_APP_KEY not in app:
        raise ValueError(
            "attach_eval_routes requires app[REGISTRY_APP_KEY] "
            "(dlaas_platform_api.build_dlaas_app handles this)."
        )
    app[EVAL_BUNDLE_APP_KEY] = EvalBundle(
        registry=registry,
        grader=grader,
        audience_analyzer=audience_analyzer,
    )
    R = app.router
    R.add_post(
        "/dlaas/templates/{template_id}/audience/analyze",
        _handle_audience_analyze,
    )
    R.add_get(
        "/dlaas/templates/{template_id}/audience",
        _handle_list_audience_profiles,
    )
    R.add_get("/dlaas/audience/{profile_id}", _handle_get_audience_profile)

    R.add_post("/dlaas/exam_questions", _handle_create_exam_question)
    R.add_post("/dlaas/exam_questions/batch", _handle_create_exam_questions_batch)
    R.add_post(
        "/dlaas/exam_questions/generate", _handle_generate_exam_questions
    )
    R.add_get("/dlaas/exam_questions", _handle_list_exam_questions)

    R.add_post("/dlaas/exam_runs", _handle_create_exam_run)
    R.add_get("/dlaas/exam_runs/{run_id}", _handle_get_exam_run)
    R.add_post(
        "/dlaas/exam_runs/{run_id}/complete", _handle_complete_exam_run
    )
    R.add_post(
        "/dlaas/exam_runs/{run_id}/execute", _handle_execute_exam_run
    )
    R.add_get(
        "/dlaas/templates/{template_id}/exam_runs", _handle_list_template_runs
    )
    R.add_post("/dlaas/exam_runs/{run_id}/signoff", _handle_signoff_exam_run)

    R.add_get(
        "/dlaas/templates/{template_id}/license", _handle_get_license
    )
    R.add_post(
        "/dlaas/templates/{template_id}/license/evaluate",
        _handle_evaluate_license,
    )
    return app


# ---------------------------------------------------------------------------
# Audience analysis
# ---------------------------------------------------------------------------


async def _handle_audience_analyze(request: web.Request) -> web.Response:
    """Audience analysis (debt #14).

    With the eval LLM configured (or an injected analyzer), the route
    resolves each linked asset's actual content, runs the corpus
    through :class:`LLMAudienceAnalyzer`, and persists the extracted
    cohort profile — caller-supplied fields stay authoritative
    (the store's analyzer seam only fills empty slots). Without an
    analyzer the route keeps the honest SHADOW passthrough and stamps
    ``evidence_stats.analyzer = "none"``.
    """

    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    template = await _resolve_template_for_tenant(bundle, tenant, template_id)
    if isinstance(template, web.Response):
        return template
    data = await _read_json(request, allow_empty=True)
    cohort_name = str(data.get("cohort_name", "default") or "default")
    asset_ids = tuple(str(a) for a in (data.get("asset_ids") or ()))

    analyzer = bundle.audience_analyzer
    corpus_analyzer_callable = None
    evidence_stats: dict[str, Any] = {
        "asset_count": len(asset_ids),
        "analyzer": "none",
    }
    if analyzer is not None and asset_ids:
        assets = []
        for asset_id in asset_ids:
            try:
                asset = await bundle.assets.get(asset_id)
            except AssetNotFound:
                return _error(404, "asset_not_found", asset_id)
            # Assets are tenant-owned; a template must not analyse
            # another tenant's corpus.
            assert_tenant_id_matches(tenant, asset.tenant_id)
            assets.append(asset)
        try:
            corpus_chunks = load_asset_corpus(tuple(assets))
        except AssetCorpusError as exc:
            return _error(422, "asset_corpus_unreadable", str(exc))
        try:
            # Blocking LLM I/O runs off the event loop.
            enriched = await asyncio.to_thread(
                analyzer.analyze,
                cohort_name=cohort_name,
                corpus_chunks=corpus_chunks,
            )
        except AudienceAnalysisError as exc:
            _LOG.error(
                "audience analysis failed for template=%s: %s",
                template_id,
                exc,
            )
            return _error(502, "audience_analysis_error", str(exc))
        evidence_stats = {
            "asset_count": len(asset_ids),
            **dict(enriched.get("evidence_stats") or {}),
        }
        # The store's seam takes enriched.evidence_stats wholesale, so
        # fold the asset_count into it before handing over.
        precomputed = {**enriched, "evidence_stats": evidence_stats}

        def _precomputed_analyzer(**_kwargs: Any) -> Mapping[str, Any]:
            # Analysis already ran above (off-loop); the store seam
            # receives the precomputed enrichment and applies its
            # caller-fields-take-precedence merge.
            return precomputed

        corpus_analyzer_callable = _precomputed_analyzer

    profile = await bundle.eval_store.upsert_audience_profile(
        template_id=template_id,
        cohort_name=cohort_name,
        asset_ids=asset_ids,
        common_questions=tuple(
            str(q) for q in (data.get("common_questions") or ())
        ),
        communication_style=str(data.get("communication_style", "") or ""),
        emotion_triggers=tuple(
            str(t) for t in (data.get("emotion_triggers") or ())
        ),
        decision_patterns=tuple(
            str(p) for p in (data.get("decision_patterns") or ())
        ),
        evidence_stats=evidence_stats,
        corpus_analyzer_callable=corpus_analyzer_callable,
    )
    return web.json_response({"status": "ok", **profile.to_json()})


async def _handle_list_audience_profiles(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    template = await _resolve_template_for_tenant(bundle, tenant, template_id)
    if isinstance(template, web.Response):
        return template
    profiles = await bundle.eval_store.list_audience_profiles_for_template(
        template_id=template_id
    )
    return web.json_response(
        {
            "status": "ok",
            "template_id": template_id,
            "profiles": [p.to_json() for p in profiles],
        }
    )


async def _handle_get_audience_profile(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    profile_id = request.match_info["profile_id"]
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    try:
        profile: AudienceProfileSpec = await bundle.eval_store.get_audience_profile(
            profile_id
        )
    except AudienceProfileNotFound:
        return _error(404, "audience_profile_not_found", profile_id)
    template = await _resolve_template_for_tenant(
        bundle, tenant, profile.template_id
    )
    if isinstance(template, web.Response):
        return template
    return web.json_response({"status": "ok", **profile.to_json()})


# ---------------------------------------------------------------------------
# Exam questions
# ---------------------------------------------------------------------------


async def _handle_create_exam_question(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    data = await _read_json(request)
    template_id = _required_str(data, "template_id")
    template = await _resolve_template_for_tenant(bundle, tenant, template_id)
    if isinstance(template, web.Response):
        return template
    rubric = _parse_rubric(data.get("rubric") or ())
    spec = await bundle.eval_store.create_exam_question(
        template_id=template_id,
        scenario_tag=_required_str(data, "scenario_tag"),
        user_prompt=_required_str(data, "user_prompt"),
        context=data.get("context") or {},
        rubric=rubric,
        reference_answer=str(data.get("reference_answer", "") or ""),
        tags=tuple(str(t) for t in (data.get("tags") or ())),
        difficulty=str(data.get("difficulty", "medium") or "medium"),
    )
    return web.json_response({"status": "ok", **spec.to_json()})


async def _handle_create_exam_questions_batch(
    request: web.Request,
) -> web.Response:
    tenant = await require_tenant_auth(request)
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    data = await _read_json(request)
    template_id = _required_str(data, "template_id")
    template = await _resolve_template_for_tenant(bundle, tenant, template_id)
    if isinstance(template, web.Response):
        return template
    raw_questions = data.get("questions") or ()
    if not isinstance(raw_questions, list):
        return _error(400, "invalid_questions", "questions must be a list")
    created: list[dict[str, Any]] = []
    for raw in raw_questions:
        if not isinstance(raw, Mapping):
            return _error(400, "invalid_question", "each question must be an object")
        rubric = _parse_rubric(raw.get("rubric") or ())
        spec = await bundle.eval_store.create_exam_question(
            template_id=template_id,
            scenario_tag=str(raw.get("scenario_tag", "default") or "default"),
            user_prompt=str(raw.get("user_prompt", "") or ""),
            context=raw.get("context") or {},
            rubric=rubric,
            reference_answer=str(raw.get("reference_answer", "") or ""),
            tags=tuple(str(t) for t in (raw.get("tags") or ())),
            difficulty=str(raw.get("difficulty", "medium") or "medium"),
        )
        created.append(spec.to_json())
    return web.json_response({"status": "ok", "questions": created})


async def _handle_generate_exam_questions(
    request: web.Request,
) -> web.Response:
    """LLM-author scenario exam questions grounded in supplied source.

    Body: ``{ template_id, source: { topics?, corpus_excerpts?,
    signature_cases? }, count?, difficulty?, language? }``.

    Requires the eval LLM env (``EVAL_LLM_*`` falling back to
    ``PROTOCOL_LLM_*``); without it the route answers
    503 ``llm_not_configured`` — there is deliberately no stub
    question bank. Generated questions are persisted through
    ``EvalStore.create_exam_question`` (R12: exam artifacts only, no
    kernel write-back) and returned to the caller.
    """

    tenant = await require_tenant_auth(request)
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    data = await _read_json(request)
    template_id = _required_str(data, "template_id")
    template = await _resolve_template_for_tenant(bundle, tenant, template_id)
    if isinstance(template, web.Response):
        return template
    config = resolve_eval_llm_config()
    if config is None:
        return _error(
            503,
            "llm_not_configured",
            "Question generation requires the eval LLM env "
            "(EVAL_LLM_BASE_URL / EVAL_LLM_API_KEY / EVAL_LLM_MODEL, "
            "or the PROTOCOL_LLM_* equivalents).",
        )
    raw_source = data.get("source")
    if not isinstance(raw_source, Mapping):
        return _error(400, "missing_source", "source must be an object")
    try:
        source = question_gen.parse_question_source(raw_source)
    except ValueError as exc:
        return _error(400, "invalid_source", str(exc))
    if source.is_empty():
        return _error(
            400,
            "empty_source",
            "source must supply topics, corpus_excerpts, and/or "
            "signature_cases",
        )
    try:
        count = int(data.get("count", 5) or 5)
    except (TypeError, ValueError):
        return _error(400, "invalid_count", "count must be an integer")
    difficulty = str(data.get("difficulty", "medium") or "medium")
    language = str(data.get("language", "en") or "en")
    try:
        generated = await asyncio.to_thread(
            question_gen.generate_exam_questions,
            config=config,
            source=source,
            count=count,
            difficulty=difficulty,
            language=language,
        )
    except QuestionGenerationError as exc:
        _LOG.error(
            "exam question generation failed for template=%s: %s",
            template_id,
            exc,
        )
        return _error(502, "question_generation_error", str(exc))
    created: list[dict[str, Any]] = []
    for question in generated:
        spec = await bundle.eval_store.create_exam_question(
            template_id=template_id,
            scenario_tag=question.scenario_tag,
            user_prompt=question.user_prompt,
            context={"generated": True, "language": language},
            rubric=question.rubric,
            reference_answer=question.reference_answer,
            tags=question.tags,
            difficulty=question.difficulty,
        )
        created.append(spec.to_json())
    return web.json_response({"status": "ok", "questions": created})


async def _handle_list_exam_questions(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    template_id = request.query.get("template_id") or ""
    scenario_tag = request.query.get("scenario_tag") or None
    if template_id:
        template = await _resolve_template_for_tenant(
            bundle, tenant, template_id
        )
        if isinstance(template, web.Response):
            return template
    questions = await bundle.eval_store.list_exam_questions(
        template_id=template_id or None,
        scenario_tag=scenario_tag,
    )
    return web.json_response(
        {
            "status": "ok",
            "questions": [q.to_json() for q in questions],
        }
    )


# ---------------------------------------------------------------------------
# Exam runs
# ---------------------------------------------------------------------------


async def _handle_create_exam_run(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    data = await _read_json(request)
    template_id = _required_str(data, "template_id")
    template = await _resolve_template_for_tenant(bundle, tenant, template_id)
    if isinstance(template, web.Response):
        return template
    template_version = int(
        data.get("template_version", template.current_version) or 1
    )
    run_type = str(data.get("run_type", "launch_gate") or "launch_gate")
    question_ids = tuple(str(q) for q in (data.get("question_ids") or ()))
    if not question_ids:
        return _error(
            400, "missing_question_ids", "question_ids must be non-empty"
        )
    pass_threshold = float(data.get("pass_threshold", 0.6) or 0.6)
    spec = await bundle.eval_store.create_exam_run(
        template_id=template_id,
        template_version=template_version,
        run_type=run_type,
        question_ids=question_ids,
        pass_threshold=pass_threshold,
    )
    return web.json_response({"status": "ok", **spec.to_json()})


async def _handle_get_exam_run(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    run_id = request.match_info["run_id"]
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    run = await _resolve_run_for_tenant(bundle, tenant, run_id)
    if isinstance(run, web.Response):
        return run
    return web.json_response({"status": "ok", **run.to_json()})


async def _handle_list_template_runs(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    template = await _resolve_template_for_tenant(bundle, tenant, template_id)
    if isinstance(template, web.Response):
        return template
    runs = await bundle.eval_store.list_runs_for_template(
        template_id=template_id
    )
    return web.json_response(
        {
            "status": "ok",
            "template_id": template_id,
            "runs": [r.to_json() for r in runs],
        }
    )


async def _handle_complete_exam_run(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    run_id = request.match_info["run_id"]
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    run = await _resolve_run_for_tenant(bundle, tenant, run_id)
    if isinstance(run, web.Response):
        return run
    data = await _read_json(request)
    ai_responses = data.get("ai_responses") or {}
    if not isinstance(ai_responses, Mapping):
        return _error(
            400, "invalid_ai_responses", "ai_responses must be an object"
        )
    return await _finalize_run(
        request=request,
        run=run,
        ai_responses=ai_responses,
        operator_id=str(data.get("operator_id", "") or ""),
        operator_name=str(data.get("operator_name", "") or ""),
        comment=str(data.get("comment", "") or ""),
        ai_id=str(data.get("ai_id", "") or run.ai_id),
        contract_id=str(data.get("contract_id", "") or run.contract_id),
        session_id=str(data.get("session_id", "") or run.session_id),
    )


async def _handle_execute_exam_run(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    run_id = request.match_info["run_id"]
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    run = await _resolve_run_for_tenant(bundle, tenant, run_id)
    if isinstance(run, web.Response):
        return run
    data = await _read_json(request, allow_empty=True)
    ai_id = str(data.get("ai_id", "") or run.ai_id)
    if not ai_id:
        return _error(400, "missing_ai_id", "ai_id is required")
    instance_manager: InstanceManager | None = request.app.get(
        INSTANCE_MANAGER_APP_KEY
    )
    if not isinstance(instance_manager, InstanceManager):
        return _error(
            503,
            "launcher_not_available",
            "execute requires an InstanceManager bound to the app.",
        )
    try:
        manager = instance_manager.get(ai_id)
    except InstanceNotFound:
        return _error(404, "ai_id_not_found", ai_id)
    session_id = str(
        data.get("session_id", "") or f"exam:{run_id}:{ai_id}"
    )
    try:
        session = await manager.get_session(session_id)
    except LookupError:
        # ``SessionNotFoundError`` (lifeform-service) subclasses
        # ``LookupError``; the typed "session does not exist" path is
        # the only condition for which we transparently create a fresh
        # session. Real failures (network / OOM / contract violations)
        # surface to the HTTP handler.
        session = await manager.create_session(session_id=session_id)
    ai_responses: dict[str, str] = {}
    for question_id in run.question_ids:
        try:
            question = await bundle.eval_store.get_exam_question(question_id)
        except ExamQuestionNotFound:
            continue
        try:
            from lifeform_core.types import TurnTriggerKind

            result = await session.run_turn(
                question.user_prompt,
                trigger_kind=TurnTriggerKind.APPRENTICE,
            )
        except (RuntimeError, ValueError) as exc:
            _LOG.warning(
                "exam execute: kernel raised on q=%s run=%s: %s",
                question_id,
                run_id,
                exc,
            )
            ai_responses[question_id] = ""
            continue
        text = getattr(getattr(result, "response", None), "text", "") or ""
        ai_responses[question_id] = text
    return await _finalize_run(
        request=request,
        run=run,
        ai_responses=ai_responses,
        operator_id=str(data.get("operator_id", "") or ""),
        operator_name=str(data.get("operator_name", "") or ""),
        comment=str(data.get("comment", "auto-executed") or "auto-executed"),
        ai_id=ai_id,
        contract_id=str(data.get("contract_id", "") or run.contract_id),
        session_id=session_id,
    )


async def _handle_signoff_exam_run(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    run_id = request.match_info["run_id"]
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    run = await _resolve_run_for_tenant(bundle, tenant, run_id)
    if isinstance(run, web.Response):
        return run
    if run.status is not ExamRunStatus.COMPLETED:
        return _error(
            409, "run_not_completed", f"run_id={run_id!r} is not completed"
        )
    data = await _read_json(request, allow_empty=True)
    license_spec = await bundle.eval_store.upsert_launch_license(
        template_id=run.template_id,
        template_version=run.template_version,
        granted=run.passed,
        reason=str(data.get("reason", "") or "")
        or ("exam_run_passed" if run.passed else "exam_run_failed"),
        granted_by_run_id=run.run_id,
    )
    return web.json_response({"status": "ok", **license_spec.to_json()})


# ---------------------------------------------------------------------------
# Launch license
# ---------------------------------------------------------------------------


async def _handle_get_license(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    template = await _resolve_template_for_tenant(bundle, tenant, template_id)
    if isinstance(template, web.Response):
        return template
    template_version = int(
        request.query.get("template_version", template.current_version) or 1
    )
    try:
        spec = await bundle.eval_store.get_launch_license(
            template_id=template_id, template_version=template_version
        )
    except LaunchLicenseNotFound:
        return _error(
            404,
            "license_not_found",
            f"no license recorded for {template_id}@{template_version}",
        )
    return web.json_response({"status": "ok", **spec.to_json()})


async def _handle_evaluate_license(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    template_id = request.match_info["template_id"]
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    template = await _resolve_template_for_tenant(bundle, tenant, template_id)
    if isinstance(template, web.Response):
        return template
    template_version = int(
        request.query.get("template_version", template.current_version) or 1
    )
    runs = await bundle.eval_store.list_runs_for_template(template_id=template_id)
    matching_passing = [
        r
        for r in runs
        if r.template_version == template_version
        and r.status is ExamRunStatus.COMPLETED
        and r.passed
    ]
    if matching_passing:
        latest = max(matching_passing, key=lambda r: r.created_at_ms)
        spec = await bundle.eval_store.upsert_launch_license(
            template_id=template_id,
            template_version=template_version,
            granted=True,
            reason="exam_run_passed",
            granted_by_run_id=latest.run_id,
        )
    else:
        spec = await bundle.eval_store.upsert_launch_license(
            template_id=template_id,
            template_version=template_version,
            granted=False,
            reason="no_passing_run",
        )
    return web.json_response({"status": "ok", **spec.to_json()})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _finalize_run(
    *,
    request: web.Request,
    run: ExamRunSpec,
    ai_responses: Mapping[str, str],
    operator_id: str,
    operator_name: str,
    comment: str,
    ai_id: str,
    contract_id: str,
    session_id: str,
) -> web.Response:
    bundle: EvalBundle = request.app[EVAL_BUNDLE_APP_KEY]
    submissions: list[ExamSubmissionScore] = []
    wrong_set: list[str] = []
    total_score = 0.0
    counted = 0
    for question_id in run.question_ids:
        try:
            question = await bundle.eval_store.get_exam_question(question_id)
        except ExamQuestionNotFound:
            continue
        ai_response = str(ai_responses.get(question_id, "") or "")
        try:
            # ``RubricGrader.grade`` is sync by contract; the LLM judge
            # does blocking network I/O, so run it off the event loop.
            graded = await asyncio.to_thread(
                bundle.grader.grade,
                rubric=question.rubric,
                ai_response=ai_response,
                reference_answer=question.reference_answer,
            )
        except GraderResponseError as exc:
            # Fail loudly (no silent 0.5 fallback): record the run as
            # FAILED with the typed grader error in ``comment`` and
            # surface 502 to the caller. The operator ``complete`` path
            # with a working grader — or an explicit re-run — stays the
            # authoritative way to produce a passing run.
            _LOG.error(
                "grader failed on run=%s question=%s: %s",
                run.run_id,
                question_id,
                exc,
            )
            await bundle.eval_store.update_exam_run(
                run_id=run.run_id,
                status=ExamRunStatus.FAILED,
                operator_id=operator_id,
                operator_name=operator_name,
                comment=f"grader_error on {question_id}: {exc}",
                ai_id=ai_id,
                contract_id=contract_id,
                session_id=session_id,
                aggregate_score=0.0,
                passed=False,
                wrong_set=(),
                submissions=(),
            )
            return _error(502, "grader_error", str(exc))
        submission = ExamSubmissionScore(
            question_id=question_id,
            ai_response=ai_response,
            weighted_score=graded.weighted_score,
            rubric_breakdown=graded.rubric_breakdown,
        )
        submissions.append(submission)
        total_score += graded.weighted_score
        counted += 1
        if graded.weighted_score < run.pass_threshold:
            wrong_set.append(question_id)
    aggregate_score = total_score / counted if counted > 0 else 0.0
    passed = bool(submissions) and aggregate_score >= run.pass_threshold
    final_run = await bundle.eval_store.update_exam_run(
        run_id=run.run_id,
        status=ExamRunStatus.COMPLETED,
        operator_id=operator_id,
        operator_name=operator_name,
        comment=comment,
        ai_id=ai_id,
        contract_id=contract_id,
        session_id=session_id,
        aggregate_score=aggregate_score,
        passed=passed,
        wrong_set=tuple(wrong_set),
        submissions=tuple(submissions),
    )
    return web.json_response({"status": "ok", **final_run.to_json()})


async def _resolve_template_for_tenant(
    bundle: EvalBundle, tenant, template_id: str
):
    try:
        template = await bundle.templates.get(template_id)
    except TemplateNotFound:
        return _error(404, "template_not_found", template_id)
    assert_tenant_id_matches(tenant, template.tenant_id)
    return template


async def _resolve_run_for_tenant(bundle: EvalBundle, tenant, run_id: str):
    try:
        run = await bundle.eval_store.get_exam_run(run_id)
    except ExamRunNotFound:
        return _error(404, "exam_run_not_found", run_id)
    template = await _resolve_template_for_tenant(bundle, tenant, run.template_id)
    if isinstance(template, web.Response):
        return template
    return run


def _parse_rubric(raw: Any) -> tuple[RubricEntry, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "invalid_rubric",
                    "detail": "rubric must be a list of entries",
                }
            ),
            content_type="application/json",
        )
    entries: list[RubricEntry] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        try:
            entries.append(RubricEntry.from_json(item))
        except ValueError as exc:
            raise web.HTTPBadRequest(
                text=json.dumps(
                    {
                        "status": "error",
                        "error": "invalid_rubric_entry",
                        "detail": str(exc),
                    }
                ),
                content_type="application/json",
            ) from exc
    return tuple(entries)


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str) or not value.strip():
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "missing_field",
                    "detail": f"Required field {key!r} must be a non-empty string.",
                }
            ),
            content_type="application/json",
        )
    return value


async def _read_json(
    request: web.Request, *, allow_empty: bool = False
) -> Mapping[str, Any]:
    if not request.body_exists:
        if allow_empty:
            return {}
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "missing_body",
                    "detail": "Body required",
                }
            ),
            content_type="application/json",
        )
    try:
        text = await request.text()
    except (web.HTTPException, OSError):
        return {}
    if not text.strip():
        if allow_empty:
            return {}
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "missing_body",
                    "detail": "Empty body",
                }
            ),
            content_type="application/json",
        )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "invalid_json",
                    "detail": f"Body is not valid JSON: {exc}",
                }
            ),
            content_type="application/json",
        ) from exc
    if not isinstance(data, dict):
        raise web.HTTPBadRequest(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "invalid_envelope",
                    "detail": "Top-level body must be a JSON object",
                }
            ),
            content_type="application/json",
        )
    return data


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        {"status": "error", "error": code, "detail": detail}, status=status
    )


__all__ = [
    "EVAL_BUNDLE_APP_KEY",
    "EvalBundle",
    "attach_eval_routes",
]
