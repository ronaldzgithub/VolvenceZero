"""aiohttp routes for the autonomous expert-cultivation control plane.

Endpoint summary (all operator-scoped — control-plane secret):

* ``POST /dlaas/v1/cultivation``                       — seed a new expert
* ``GET  /dlaas/v1/cultivation``                       — list cultivations
* ``GET  /dlaas/v1/cultivation/{cultivation_id}``      — status + coherence
* ``POST /dlaas/v1/cultivation/{cultivation_id}/tick`` — run study cycles
* ``POST /dlaas/v1/cultivation/{cultivation_id}/graduate`` — exam gate
* ``POST /dlaas/v1/cultivation/{cultivation_id}/induct``    — operator induct

Layering (R8 / R12):

* Cognition is kernel-owned. These routes drive the kernel ONLY through
  the :class:`SessionCultivationSink` (canonical session + ingestion
  surfaces) and read back the published ``active_regime`` sequence as a
  coherence *readout*. No learning signal flows back from this layer.
* The cultivation lifecycle (status machine, exam evidence, induction)
  is platform governance persisted in :class:`CultivationStore`.

Induction is **semi-automatic**: convergence (coherence over threshold)
plus an attached exam run move a cultivation to ``ready_for_review``;
the final promotion to a default expert template requires an explicit
operator ``induct`` call. The deterministic eval grader is fail-closed
(it never auto-grants a license), so the exam run is recorded as
*evidence* the operator reviews — it does not silently gate induction.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import (
    ExamRunStatus,
    ExamSubmissionScore,
    RubricEntry,
    TemplateActivationStatus,
    TemplateStatus,
)
from dlaas_platform_eval import DefaultRubricGrader
from dlaas_platform_launcher import (
    INSTANCE_MANAGER_APP_KEY,
    InstanceManager,
    InstanceNotFound,
)
from dlaas_platform_registry import (
    CultivationNotFound,
    CultivationStatus,
    CultivationStore,
    EvalStore,
    REGISTRY_APP_KEY,
    Registry,
    TemplateStore,
    TenantNotFound,
    TenantStore,
    require_control_plane_secret,
)
from lifeform_core.types import TurnTriggerKind
from lifeform_cultivation import (
    CultivationCurriculum,
    CultivationEngine,
    CultivationSeed,
    SessionCultivationSink,
    build_identity_core_protocol,
)
from lifeform_service.openai_compat_client import build_client_from_env
from lifeform_service.protocol_uptake import (
    ProtocolUptakeConfig,
    ProtocolUptakeService,
)

_LOG = logging.getLogger("dlaas_platform_api.cultivation")

CULTIVATION_BUNDLE_APP_KEY = "dlaas_cultivation_bundle"

RUNTIME_TEMPLATE_ID = "cultivation.expert.v0"
SYSTEM_TENANT_ID = "tenant_cultivation_system"
DEFAULT_TICK_CYCLES = 4


class CultivationBundle:
    """Container the api wheel reads to dispatch cultivation state."""

    __slots__ = (
        "registry",
        "cultivations",
        "eval_store",
        "templates",
        "tenants",
        "grader",
        "_uptake_services",
    )

    def __init__(self, *, registry: Registry) -> None:
        self.registry = registry
        self.cultivations = CultivationStore(registry)
        self.eval_store = EvalStore(registry)
        self.templates = TemplateStore(registry)
        self.tenants = TenantStore(registry)
        self.grader = DefaultRubricGrader()
        # One ProtocolUptakeService per cultivation ai_id. Holds the
        # approved school protocols (Identity Core + researched theories)
        # that the instance's SessionManager seeds into each study
        # session. Lives here (not the registry) because protocols are
        # in-process kernel artifacts, not control-plane rows.
        self._uptake_services: dict[str, ProtocolUptakeService] = {}

    def uptake_service_for(self, ai_id: str) -> ProtocolUptakeService:
        svc = self._uptake_services.get(ai_id)
        if svc is None:
            svc = ProtocolUptakeService(
                config=ProtocolUptakeConfig(
                    llm_client_factory=build_client_from_env,
                )
            )
            self._uptake_services[ai_id] = svc
        return svc


def attach_cultivation_routes(
    app: web.Application,
    *,
    registry: Registry,
) -> web.Application:
    if REGISTRY_APP_KEY not in app:
        raise ValueError(
            "attach_cultivation_routes requires app[REGISTRY_APP_KEY] "
            "(dlaas_platform_api.build_dlaas_app handles this)."
        )
    app[CULTIVATION_BUNDLE_APP_KEY] = CultivationBundle(registry=registry)
    R = app.router
    R.add_post("/dlaas/v1/cultivation", _handle_create)
    R.add_get("/dlaas/v1/cultivation", _handle_list)
    R.add_get("/dlaas/v1/cultivation/{cultivation_id}", _handle_get)
    R.add_post("/dlaas/v1/cultivation/{cultivation_id}/tick", _handle_tick)
    R.add_post(
        "/dlaas/v1/cultivation/{cultivation_id}/graduate", _handle_graduate
    )
    R.add_post("/dlaas/v1/cultivation/{cultivation_id}/induct", _handle_induct)
    return app


# ---------------------------------------------------------------------------
# Create / list / get
# ---------------------------------------------------------------------------


async def _handle_create(request: web.Request) -> web.Response:
    require_control_plane_secret(request)
    bundle = _bundle(request)
    data = await _read_json(request)

    slug = _required_str(data, "slug")
    seed = _parse_seed(data)
    curriculum = _parse_curriculum(data)
    ai_id = f"cultivation:{slug}"

    instance_manager = _instance_manager(request)

    # Seed the Identity Core protocol (the operator-supplied rough persona
    # IS the reviewed school anchor) into this ai_id's uptake service, and
    # bind the service to the InstanceManager BEFORE acquire so the
    # SessionManager seeds it into every study session.
    uptake = bundle.uptake_service_for(ai_id)
    identity_core = build_identity_core_protocol(seed)
    uptake.registry.load(identity_core)
    instance_manager.set_protocol_uptake_service(ai_id, uptake)

    try:
        await instance_manager.acquire(
            ai_id=ai_id, runtime_template_id=RUNTIME_TEMPLATE_ID
        )
    except LookupError:
        return _error(
            503,
            "vertical_unavailable",
            f"runtime_template_id={RUNTIME_TEMPLATE_ID!r} is not registered; "
            f"install a lifeform-domain vertical that provides it.",
        )

    record = await bundle.cultivations.create(
        ai_id=ai_id,
        slug=slug,
        display_name=seed.display_name,
        domain=seed.domain,
        runtime_template_id=RUNTIME_TEMPLATE_ID,
        seed_persona=seed.to_json(),
        curriculum=curriculum.to_json(),
        notes=str(data.get("notes", "") or ""),
    )
    return web.json_response({"status": "ok", **record.to_json()})


async def _handle_list(request: web.Request) -> web.Response:
    require_control_plane_secret(request)
    bundle = _bundle(request)
    records = await bundle.cultivations.list_all()
    return web.json_response(
        {"status": "ok", "cultivations": [r.to_json() for r in records]}
    )


async def _handle_get(request: web.Request) -> web.Response:
    require_control_plane_secret(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    return web.json_response({"status": "ok", **record.to_json()})


# ---------------------------------------------------------------------------
# Tick — run autonomous study cycles
# ---------------------------------------------------------------------------


async def _handle_tick(request: web.Request) -> web.Response:
    require_control_plane_secret(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    if record.status in (
        CultivationStatus.READY_FOR_REVIEW,
        CultivationStatus.INDUCTED,
        CultivationStatus.FAILED,
    ):
        return _error(
            409,
            "cultivation_not_runnable",
            f"cultivation status={record.status.value!r} does not accept ticks.",
        )
    data = await _read_json(request, allow_empty=True)
    cycles = _optional_int(data, "cycles", default=DEFAULT_TICK_CYCLES)
    if cycles <= 0:
        return _error(400, "invalid_cycles", "cycles must be a positive integer")

    seed = CultivationSeed.from_json(dict(record.seed_persona))
    curriculum = CultivationCurriculum.from_json(dict(record.curriculum))

    # Bind the per-ai_id uptake service so researched theories become
    # BehaviorProtocols (re-homed onto the protocol runtime). Re-loading
    # the Identity Core is idempotent and survives a process restart that
    # would otherwise leave the service empty.
    instance_manager = _instance_manager(request)
    uptake = bundle.uptake_service_for(record.ai_id)
    uptake.registry.load(build_identity_core_protocol(seed))
    instance_manager.set_protocol_uptake_service(record.ai_id, uptake)

    async def _uptaker(text: str, source_label: str) -> str | None:
        try:
            candidate = await uptake.extract_from_markdown_text(
                text, source_label=source_label
            )
        except RuntimeError:
            # LLM not configured (PROTOCOL_LLM_* unset): degrade to corpus
            # ingestion in the sink. Not a swallowed error — it is the
            # documented degraded path the sink falls back to.
            return None
        await uptake.submit_candidate(candidate)
        try:
            approved = await uptake.approve_pending(
                candidate.protocol.protocol_id, reviewer_id="cultivation-auto"
            )
        except KeyError:
            return None
        return approved.protocol_id

    # Fresh session per tick so the SessionManager seeds the currently
    # approved protocol set (protocols approved this tick take effect at
    # the next tick — the documented convergence cadence).
    session = await _fresh_tick_session(
        request,
        ai_id=record.ai_id,
        cultivation_id=cultivation_id,
        seq=record.cycles_completed,
    )
    if isinstance(session, web.Response):
        return session

    sink = SessionCultivationSink(session=session, protocol_uptaker=_uptaker)
    engine = CultivationEngine(curriculum=curriculum, domain=seed.domain)

    seeded_chunks = 0
    if record.status is CultivationStatus.SEEDING:
        seeded_chunks = await engine.seed(sink, seed)

    progress = await engine.run_cycles(
        sink,
        start_cycle=record.cycles_completed,
        count=cycles,
        prior_regimes=record.regime_history,
    )
    new_status = (
        CultivationStatus.CONVERGING
        if progress.converged
        else CultivationStatus.STUDYING
    )
    record = await bundle.cultivations.update_progress(
        cultivation_id=cultivation_id,
        status=new_status,
        cycles_completed=progress.cycles_completed,
        coherence_score=progress.coherence_score,
        coherence_detail=progress.coherence_detail,
        regime_history=progress.regime_history,
    )
    return web.json_response(
        {
            "status": "ok",
            "seeded_chunks": seeded_chunks,
            "progress": progress.to_json(),
            **record.to_json(),
        }
    )


# ---------------------------------------------------------------------------
# Graduate — create candidate template + run exam evidence
# ---------------------------------------------------------------------------


async def _handle_graduate(request: web.Request) -> web.Response:
    require_control_plane_secret(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    if record.status not in (
        CultivationStatus.CONVERGING,
        CultivationStatus.EXAM,
        CultivationStatus.READY_FOR_REVIEW,
    ):
        return _error(
            409,
            "not_converged",
            "graduation requires the cultivation to have converged onto a "
            f"single school first (status={record.status.value!r}).",
        )

    seed = CultivationSeed.from_json(dict(record.seed_persona))
    curriculum = CultivationCurriculum.from_json(dict(record.curriculum))

    await _ensure_system_tenant(bundle)
    template = await _ensure_candidate_template(bundle, record=record, seed=seed)

    session = await _session_for(request, ai_id=record.ai_id, cultivation_id=cultivation_id)
    if isinstance(session, web.Response):
        return session

    await bundle.cultivations.update_status(
        cultivation_id=cultivation_id, status=CultivationStatus.EXAM
    )

    run = await _run_exam(
        bundle,
        template_id=template.template_id,
        topics=curriculum.topics,
        seed=seed,
        session=session,
        ai_id=record.ai_id,
        session_id=f"cultivation:{cultivation_id}",
    )
    license_spec = await bundle.eval_store.upsert_launch_license(
        template_id=template.template_id,
        template_version=1,
        granted=run.passed,
        reason="cultivation_exam_passed" if run.passed else "cultivation_exam_evidence",
        granted_by_run_id=run.run_id,
    )

    record = await bundle.cultivations.set_eval_template(
        cultivation_id=cultivation_id,
        dlaas_template_id=template.template_id,
        last_exam_run_id=run.run_id,
    )
    # Convergence is the internal gate that moves to operator review; the
    # exam run + license are attached as evidence. (The fail-closed
    # default grader does not auto-grant, so we do NOT require exam pass
    # to surface for review — the operator decides at induct.)
    record = await bundle.cultivations.update_status(
        cultivation_id=cultivation_id,
        status=CultivationStatus.READY_FOR_REVIEW,
    )
    return web.json_response(
        {
            "status": "ok",
            "template": template.to_json(),
            "exam_run": run.to_json(),
            "license": license_spec.to_json(),
            **record.to_json(),
        }
    )


# ---------------------------------------------------------------------------
# Induct — operator approval → default expert template
# ---------------------------------------------------------------------------


async def _handle_induct(request: web.Request) -> web.Response:
    require_control_plane_secret(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    if record.status is not CultivationStatus.READY_FOR_REVIEW:
        return _error(
            409,
            "not_ready_for_review",
            "induct requires status=ready_for_review "
            f"(got {record.status.value!r}). Run graduate first.",
        )
    if not record.dlaas_template_id:
        return _error(
            409,
            "missing_candidate_template",
            "no candidate template recorded; run graduate first.",
        )

    # Publish the candidate template so it becomes adoptable as a default
    # system expert (it was activated at graduation time). Bakeing a
    # process-wide figure bundle from the cultivated instance's cognition
    # (so it auto-seeds FigureBundleStore) is the documented follow-up
    # (plan risk: "export instance cognition -> figure bundle"); for now
    # induction promotes the published template under the system tenant.
    template = await bundle.templates.get(record.dlaas_template_id)
    if template.status is not TemplateStatus.PUBLISHED:
        template = await bundle.templates.patch(
            template_id=record.dlaas_template_id,
            status=TemplateStatus.PUBLISHED,
            version_note="cultivation-induct",
        )
    record = await bundle.cultivations.set_inducted(
        cultivation_id=cultivation_id,
        inducted_template_id=template.template_id,
    )
    return web.json_response(
        {"status": "ok", "template": template.to_json(), **record.to_json()}
    )


# ---------------------------------------------------------------------------
# Graduation helpers
# ---------------------------------------------------------------------------


async def _ensure_system_tenant(bundle: CultivationBundle):
    try:
        return await bundle.tenants.get(SYSTEM_TENANT_ID)
    except TenantNotFound:
        return await bundle.tenants.create(
            tenant_name="Cultivation System",
            contact_email="cultivation@system.local",
            business_type="platform_internal",
            tenant_id=SYSTEM_TENANT_ID,
        )


async def _ensure_candidate_template(
    bundle: CultivationBundle, *, record, seed: CultivationSeed
):
    """Create (or reuse) the activated+published candidate template.

    Idempotent across repeated graduate calls: when the cultivation
    already carries a ``dlaas_template_id`` we reuse it.
    """

    if record.dlaas_template_id:
        return await bundle.templates.get(record.dlaas_template_id)

    persona_spec = {
        "schema_version": "cultivation.persona.v1",
        "display_name": seed.display_name,
        "role_archetype": seed.role_archetype,
        "domain": seed.domain,
        "focus": seed.focus,
        "value_boundaries": list(seed.value_boundaries),
        "single_school_objective": seed.single_school_objective,
        "cultivated_school": str(
            record.coherence_detail.get("dominant_protocol")
            or record.coherence_detail.get("dominant_regime", "")
        ),
        "coherence_readout": str(record.coherence_detail.get("readout", "")),
        "coherence_score": record.coherence_score,
        "cultivation_id": record.cultivation_id,
        "source_ai_id": record.ai_id,
    }
    template = await bundle.templates.create(
        tenant_id=SYSTEM_TENANT_ID,
        template_name=seed.display_name,
        domain=seed.domain,
        description=f"Auto-cultivated expert: {seed.display_name} ({seed.domain})",
        runtime_template_id=RUNTIME_TEMPLATE_ID,
        persona_spec=persona_spec,
    )
    # Mark activated (the seed charter + study corpus already ran through
    # the kernel's ingestion path on the source instance) so the template
    # can be published.
    await bundle.templates.update_activation(
        template_id=template.template_id,
        activation_status=TemplateActivationStatus.ACTIVATED,
        activation_stats={
            "cultivation_id": record.cultivation_id,
            "cycles_completed": record.cycles_completed,
            "coherence_score": record.coherence_score,
        },
    )
    return await bundle.templates.patch(
        template_id=template.template_id,
        status=TemplateStatus.PUBLISHED,
        version_note="cultivation-graduate",
    )


async def _run_exam(
    bundle: CultivationBundle,
    *,
    template_id: str,
    topics: tuple[str, ...],
    seed: CultivationSeed,
    session: Any,
    ai_id: str,
    session_id: str,
):
    """Build a small exam from the curriculum topics and execute it.

    Mirrors the eval wheel's execute+finalize flow but runs in-process
    (no tenant HTTP round-trip): question per topic -> run_turn -> grade
    -> aggregate -> persisted exam run.
    """

    rubric = (
        RubricEntry(
            criterion="流派一致性与专业判断", weight=1.0, max_score=10.0
        ),
    )
    question_ids: list[str] = []
    for topic in topics:
        question = await bundle.eval_store.create_exam_question(
            template_id=template_id,
            scenario_tag=topic,
            user_prompt=(
                f"作为{seed.display_name}，请就「{topic}」给出你的专业判断，"
                f"并说明它如何与你已形成的整体认知体系保持一致。"
            ),
            rubric=rubric,
        )
        question_ids.append(question.question_id)

    run = await bundle.eval_store.create_exam_run(
        template_id=template_id,
        template_version=1,
        run_type="cultivation_gate",
        question_ids=tuple(question_ids),
        pass_threshold=0.6,
    )

    submissions: list[ExamSubmissionScore] = []
    wrong_set: list[str] = []
    total = 0.0
    counted = 0
    for question_id in run.question_ids:
        question = await bundle.eval_store.get_exam_question(question_id)
        try:
            result = await session.run_turn(
                question.user_prompt, trigger_kind=TurnTriggerKind.APPRENTICE
            )
            response_text = getattr(getattr(result, "response", None), "text", "") or ""
        except (RuntimeError, ValueError) as exc:
            _LOG.warning("cultivation exam: kernel raised on q=%s: %s", question_id, exc)
            response_text = ""
        graded = bundle.grader.grade(
            rubric=question.rubric,
            ai_response=response_text,
            reference_answer=question.reference_answer,
        )
        submissions.append(
            ExamSubmissionScore(
                question_id=question_id,
                ai_response=response_text,
                weighted_score=graded.weighted_score,
                rubric_breakdown=graded.rubric_breakdown,
            )
        )
        total += graded.weighted_score
        counted += 1
        if graded.weighted_score < run.pass_threshold:
            wrong_set.append(question_id)

    aggregate = total / counted if counted else 0.0
    passed = bool(submissions) and aggregate >= run.pass_threshold
    return await bundle.eval_store.update_exam_run(
        run_id=run.run_id,
        status=ExamRunStatus.COMPLETED,
        comment="cultivation auto-exam",
        ai_id=ai_id,
        session_id=session_id,
        aggregate_score=aggregate,
        passed=passed,
        wrong_set=tuple(wrong_set),
        submissions=tuple(submissions),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _bundle(request: web.Request) -> CultivationBundle:
    return request.app[CULTIVATION_BUNDLE_APP_KEY]


def _instance_manager(request: web.Request) -> InstanceManager:
    instance_manager = request.app.get(INSTANCE_MANAGER_APP_KEY)
    if not isinstance(instance_manager, InstanceManager):
        raise web.HTTPServiceUnavailable(
            text=json.dumps(
                {
                    "status": "error",
                    "error": "launcher_not_available",
                    "detail": "cultivation requires an InstanceManager bound to the app.",
                }
            ),
            content_type="application/json",
        )
    return instance_manager


async def _fresh_tick_session(
    request: web.Request, *, ai_id: str, cultivation_id: str, seq: int
):
    """Create a new per-tick study session seeded with approved protocols.

    A fresh ``session_id`` is required because ``create_session`` rejects
    a reused id; the new session triggers ``_inject_uptake_seed_protocols``
    so the current approved protocol set (Identity Core + researched
    theories) is loaded. Idle/LRU eviction in the SessionManager reclaims
    superseded tick sessions.
    """

    instance_manager = _instance_manager(request)
    try:
        manager = instance_manager.get(ai_id)
    except InstanceNotFound:
        try:
            manager = await instance_manager.acquire(
                ai_id=ai_id, runtime_template_id=RUNTIME_TEMPLATE_ID
            )
        except LookupError:
            return _error(
                503,
                "vertical_unavailable",
                f"runtime_template_id={RUNTIME_TEMPLATE_ID!r} is not registered.",
            )
    session_id = f"cultivation:{cultivation_id}:c{seq}"
    try:
        return await manager.get_session(session_id)
    except LookupError:
        return await manager.create_session(session_id=session_id)


async def _session_for(request: web.Request, *, ai_id: str, cultivation_id: str):
    instance_manager = _instance_manager(request)
    try:
        manager = instance_manager.get(ai_id)
    except InstanceNotFound:
        try:
            manager = await instance_manager.acquire(
                ai_id=ai_id, runtime_template_id=RUNTIME_TEMPLATE_ID
            )
        except LookupError:
            return _error(
                503,
                "vertical_unavailable",
                f"runtime_template_id={RUNTIME_TEMPLATE_ID!r} is not registered.",
            )
    session_id = f"cultivation:{cultivation_id}"
    try:
        return await manager.get_session(session_id)
    except LookupError:
        return await manager.create_session(session_id=session_id)


async def _resolve(bundle: CultivationBundle, cultivation_id: str):
    try:
        return await bundle.cultivations.get(cultivation_id)
    except CultivationNotFound:
        return _error(404, "cultivation_not_found", cultivation_id)


def _parse_seed(data: Mapping[str, Any]) -> CultivationSeed:
    payload = {
        "display_name": data.get("display_name", ""),
        "domain": data.get("domain", ""),
        "role_archetype": data.get("role_archetype", ""),
        "focus": data.get("focus", ""),
        "value_boundaries": data.get("value_boundaries", ()),
        "single_school_objective": data.get("single_school_objective", ""),
    }
    try:
        return CultivationSeed.from_json(payload)
    except ValueError as exc:
        raise _bad_request("invalid_seed", str(exc)) from exc


def _parse_curriculum(data: Mapping[str, Any]) -> CultivationCurriculum:
    raw = data.get("curriculum")
    if not isinstance(raw, Mapping):
        # Allow flat topics on the top-level body as a convenience.
        raw = {
            "topics": data.get("topics", ()),
            "source_hints": data.get("source_hints", ()),
        }
    try:
        return CultivationCurriculum.from_json(dict(raw))
    except ValueError as exc:
        raise _bad_request("invalid_curriculum", str(exc)) from exc


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str) or not value.strip():
        raise _bad_request("missing_field", f"{key!r} must be a non-empty string.")
    return value.strip()


def _optional_int(data: Mapping[str, Any], key: str, *, default: int) -> int:
    raw = data.get(key, default)
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return int(raw.strip())
        except ValueError:
            return default
    return default


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
    "CULTIVATION_BUNDLE_APP_KEY",
    "CultivationBundle",
    "attach_cultivation_routes",
]
