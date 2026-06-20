"""aiohttp routes for the autonomous expert-cultivation control plane.

Endpoint summary:

* ``POST /dlaas/v1/cultivation``                       — seed a new expert
* ``GET  /dlaas/v1/cultivation``                       — list cultivations
* ``GET  /dlaas/v1/cultivation/{cultivation_id}``      — status + coherence
* ``POST /dlaas/v1/cultivation/{cultivation_id}/tick`` — run study cycles
* ``POST /dlaas/v1/cultivation/{cultivation_id}/graduate`` — exam gate
* ``POST /dlaas/v1/cultivation/{cultivation_id}/induct``    — operator induct

Auth (dual mode): operator credentials (``X-Control-Plane-Secret`` or
``X-Service-Secret``) act cross-tenant and own *system* cultivations
(``tenant_id=""``). Tenant credentials (``X-Tenant-Api-Key`` +
``X-Tenant-Api-Secret``) create tenant-owned cultivations, see only
their own records, and graduate candidate templates under their own
``tenant_id``. Cross-tenant access is a typed 403 ``tenant_mismatch``,
never a silent read.

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
operator ``induct`` call. The exam grader comes from the eval wheel's
``build_grader_from_env`` seam (a real LLM judge when ``EVAL_LLM_*`` /
``PROTOCOL_LLM_*`` is configured, else the fail-closed deterministic
default that never auto-grants a license), so the exam run is recorded
as *evidence* the reviewer weighs — it does not silently gate induction.
"""

from __future__ import annotations

import json
import logging
import secrets
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import (
    ExamRunStatus,
    ExamSubmissionScore,
    RubricEntry,
    TemplateActivationStatus,
    TemplateStatus,
    TenantSpec,
)
from dlaas_platform_eval import build_grader_from_env
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
    TemplateNotFound,
    TemplateStore,
    TenantNotFound,
    TenantStore,
    require_control_plane_or_service,
    require_tenant_auth,
)
from lifeform_core.types import TurnTriggerKind
from lifeform_cultivation import (
    CultivationCurriculum,
    CultivationDirection,
    CultivationEngine,
    CultivationSeed,
    SessionCultivationSink,
    build_identity_core_protocol,
    parse_directions,
)
from lifeform_service.cultivation_bundle import (
    build_uptake_service_from_bundle,
    export_protocol_bundle,
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
        # Deployment seam shared with the interview/eval gates: an
        # LLMRubricGrader when EVAL_LLM_* (or PROTOCOL_LLM_*) is
        # configured, else the fail-closed DefaultRubricGrader.
        self.grader = build_grader_from_env()
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

    def hydrate_uptake_from_bundle(
        self, ai_id: str, bundle_payload: Any
    ) -> ProtocolUptakeService:
        """Seed the per-ai_id uptake service from an adopted persona bundle.

        Used by the adopted-seed cultivation path: the source template's
        ``cultivation_protocol_bundle`` (its converged school) is loaded
        so the new cultivation *continues* from that school instead of an
        empty Identity Core. The hydrated service replaces any cached one
        for this ai_id (creation-time, before acquire), then the caller
        still loads the seed Identity Core on top.
        """

        svc = build_uptake_service_from_bundle(bundle_payload)
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
    # Package routes are registered BEFORE the `{cultivation_id}` wildcard
    # so `packages` is not captured as a cultivation id.
    R.add_get("/dlaas/v1/cultivation/packages", _handle_list_packages)
    R.add_get(
        "/dlaas/v1/cultivation/packages/{package_id}", _handle_get_package
    )
    R.add_get("/dlaas/v1/cultivation/{cultivation_id}", _handle_get)
    R.add_get(
        "/dlaas/v1/cultivation/{cultivation_id}/events", _handle_events
    )
    R.add_post("/dlaas/v1/cultivation/{cultivation_id}/tick", _handle_tick)
    # Supervision (self-learning workshop): operator teach correction +
    # pause/resume + reject (drop a non-chosen school track for R15).
    R.add_post("/dlaas/v1/cultivation/{cultivation_id}/teach", _handle_teach)
    R.add_post("/dlaas/v1/cultivation/{cultivation_id}/pause", _handle_pause)
    R.add_post("/dlaas/v1/cultivation/{cultivation_id}/resume", _handle_resume)
    R.add_post("/dlaas/v1/cultivation/{cultivation_id}/reject", _handle_reject)
    R.add_post(
        "/dlaas/v1/cultivation/{cultivation_id}/graduate", _handle_graduate
    )
    R.add_post("/dlaas/v1/cultivation/{cultivation_id}/induct", _handle_induct)
    return app


# ---------------------------------------------------------------------------
# Auth (dual mode)
# ---------------------------------------------------------------------------


async def _authenticate(request: web.Request) -> TenantSpec | None:
    """Resolve the caller for the dual auth mode.

    Operator credentials (``X-Control-Plane-Secret`` or
    ``X-Service-Secret``) act cross-tenant and return ``None``; any
    other caller must present tenant credentials and gets back the
    authenticated :class:`TenantSpec`. Invalid credentials raise the
    typed 401/403 from the registry auth helpers — never a silent pass.
    """

    headers = request.headers
    if "X-Control-Plane-Secret" in headers or "X-Service-Secret" in headers:
        require_control_plane_or_service(request)
        return None
    return await require_tenant_auth(request)


def _assert_record_access(
    tenant: TenantSpec | None, record
) -> web.Response | None:
    """403 when a tenant touches a record it does not own.

    Operators (``tenant=None``) pass unconditionally. System-owned
    records (``tenant_id=""``) are operator-only, so a tenant caller is
    rejected for those too.
    """

    if tenant is None:
        return None
    if record.tenant_id != tenant.tenant_id:
        return _error(
            403,
            "tenant_mismatch",
            (
                f"authenticated tenant_id={tenant.tenant_id!r} cannot act "
                f"on cultivation {record.cultivation_id!r}"
            ),
        )
    return None


# ---------------------------------------------------------------------------
# Create / list / get
# ---------------------------------------------------------------------------


async def _handle_create(request: web.Request) -> web.Response:
    tenant = await _authenticate(request)
    bundle = _bundle(request)
    data = await _read_json(request)

    slug = _required_str(data, "slug")
    # Adopted-seed path: when ``source_template_id`` is given the seed
    # defaults are filled from the source template's persona_spec (the
    # operator can still override any field), and the template's converged
    # school is hydrated into each track at acquire time. Empty seed path
    # is unchanged (source_template_id == "").
    source_template_id = str(data.get("source_template_id", "") or "").strip()
    source_bundle_payload: Any = None
    provenance: dict[str, Any] = {}
    if source_template_id:
        resolved = await _resolve_source_template(
            bundle, source_template_id, tenant
        )
        if isinstance(resolved, web.Response):
            return resolved
        source_template, source_bundle_payload = resolved
        provenance = _source_provenance(
            data, source_template, source_bundle_payload
        )
        data = _seed_defaults_from_template(data, source_template)

    seed = _parse_seed(data)
    notes = str(data.get("notes", "") or "")
    # Operator-created cultivations stay system-owned (tenant_id="");
    # tenant-created ones carry the authenticated tenant and namespace
    # their ai_id by tenant so two tenants reusing one slug never share
    # kernel state (explicit, stable ai_id ownership).
    tenant_id = tenant.tenant_id if tenant is not None else ""
    ai_id_prefix = (
        f"cultivation:{tenant_id}" if tenant_id else "cultivation"
    )

    try:
        directions = parse_directions(data.get("directions"))
    except ValueError as exc:
        return _error(400, "invalid_directions", str(exc))

    instance_manager = _instance_manager(request)

    # Multi-direction path: one seed fans out into several self-consistent
    # school tracks, each its own ai_id + cultivation row, grouped by a
    # shared package_id. Each track converges independently so the schools
    # do not cross-contaminate.
    if directions:
        base_curriculum = _parse_curriculum(data)
        package_id = _fresh_package_id()
        track_records = []
        for direction in directions:
            track_ai_id = f"{ai_id_prefix}:{slug}:{direction.track_id}"
            acquired = await _acquire_track(
                request,
                bundle=bundle,
                instance_manager=instance_manager,
                ai_id=track_ai_id,
                seed=seed,
                tenant_id=tenant_id,
                source_bundle_payload=source_bundle_payload,
            )
            if isinstance(acquired, web.Response):
                return acquired
            track_curriculum = direction.to_curriculum(base=base_curriculum)
            record = await bundle.cultivations.create(
                ai_id=track_ai_id,
                slug=slug,
                display_name=f"{seed.display_name} · {direction.display_name}",
                domain=seed.domain,
                runtime_template_id=RUNTIME_TEMPLATE_ID,
                tenant_id=tenant_id,
                seed_persona=seed.to_json(),
                curriculum=track_curriculum.to_json(),
                notes=notes,
                package_id=package_id,
                track_id=direction.track_id,
                direction=direction.to_json(),
                source_template_id=source_template_id,
                provenance=provenance,
            )
            track_records.append(record)
        return web.json_response(
            {
                "status": "ok",
                "package_id": package_id,
                "tracks": [r.to_json() for r in track_records],
            }
        )

    # Legacy single-expert path (unchanged behaviour for operators).
    curriculum = _parse_curriculum(data)
    ai_id = f"{ai_id_prefix}:{slug}"
    acquired = await _acquire_track(
        request,
        bundle=bundle,
        instance_manager=instance_manager,
        ai_id=ai_id,
        seed=seed,
        tenant_id=tenant_id,
        source_bundle_payload=source_bundle_payload,
    )
    if isinstance(acquired, web.Response):
        return acquired

    record = await bundle.cultivations.create(
        ai_id=ai_id,
        slug=slug,
        display_name=seed.display_name,
        domain=seed.domain,
        runtime_template_id=RUNTIME_TEMPLATE_ID,
        tenant_id=tenant_id,
        seed_persona=seed.to_json(),
        curriculum=curriculum.to_json(),
        notes=notes,
        source_template_id=source_template_id,
        provenance=provenance,
    )
    return web.json_response({"status": "ok", **record.to_json()})


async def _acquire_track(
    request: web.Request,
    *,
    bundle: CultivationBundle,
    instance_manager: InstanceManager,
    ai_id: str,
    seed: CultivationSeed,
    tenant_id: str = "",
    source_bundle_payload: Any = None,
):
    """Seed the Identity Core + bind uptake + acquire one track instance.

    The operator-supplied rough persona IS the reviewed school anchor; it
    is loaded into the per-``ai_id`` uptake service and bound to the
    InstanceManager BEFORE acquire so the SessionManager seeds it into
    every study session. Returns the acquired manager, or a typed 503
    ``web.Response`` when the runtime vertical is not installed.

    Adopted-seed path: when ``source_bundle_payload`` is a converged
    cultivation protocol bundle, the uptake service is first hydrated
    from it so the new cultivation continues from the adopted persona's
    school; the seed Identity Core is then loaded on top as the anchor.
    """

    if source_bundle_payload:
        try:
            uptake = bundle.hydrate_uptake_from_bundle(
                ai_id, source_bundle_payload
            )
        except ValueError as exc:
            return _error(
                400,
                "invalid_source_bundle",
                f"source template's cultivation_protocol_bundle is malformed: {exc}",
            )
    else:
        uptake = bundle.uptake_service_for(ai_id)
    uptake.registry.load(build_identity_core_protocol(seed))
    instance_manager.set_protocol_uptake_service(ai_id, uptake)
    try:
        return await instance_manager.acquire(
            ai_id=ai_id,
            runtime_template_id=RUNTIME_TEMPLATE_ID,
            tenant_id=tenant_id,
        )
    except LookupError:
        return _error(
            503,
            "vertical_unavailable",
            f"runtime_template_id={RUNTIME_TEMPLATE_ID!r} is not registered; "
            f"install a lifeform-domain vertical that provides it.",
        )


def _fresh_package_id() -> str:
    return f"cpkg_{secrets.token_hex(4)}"


async def _handle_list(request: web.Request) -> web.Response:
    tenant = await _authenticate(request)
    bundle = _bundle(request)
    records = await bundle.cultivations.list_all(
        tenant_id=tenant.tenant_id if tenant is not None else ""
    )
    return web.json_response(
        {"status": "ok", "cultivations": [r.to_json() for r in records]}
    )


async def _handle_get(request: web.Request) -> web.Response:
    tenant = await _authenticate(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    denied = _assert_record_access(tenant, record)
    if denied is not None:
        return denied
    return web.json_response({"status": "ok", **record.to_json()})


# ---------------------------------------------------------------------------
# Packages — group sibling school tracks grown from one seed
# ---------------------------------------------------------------------------

# Cultivation lifecycle status -> package track status (mirrors the
# foundation `CultivationTrackStatus` union). `exam` rolls up to
# `converging` (it is mid-graduation), `seeding` to `studying`.
_TRACK_STATUS_BY_CULTIVATION = {
    CultivationStatus.SEEDING: "studying",
    CultivationStatus.STUDYING: "studying",
    CultivationStatus.CONVERGING: "converging",
    CultivationStatus.EXAM: "converging",
    CultivationStatus.READY_FOR_REVIEW: "ready_for_review",
    CultivationStatus.INDUCTED: "inducted",
    CultivationStatus.FAILED: "failed",
}


def _track_view(record) -> dict[str, Any]:
    detail = dict(record.coherence_detail)
    school = str(
        detail.get("dominant_protocol")
        or detail.get("dominant_regime", "")
        or ""
    )
    distinct = int(
        detail.get("distinct_schools")
        or detail.get("distinct_regimes", 0)
        or 0
    )
    template_id = record.inducted_template_id or record.dlaas_template_id
    view: dict[str, Any] = {
        "track_id": record.track_id,
        "display_name": record.display_name,
        "school": school,
        "status": _TRACK_STATUS_BY_CULTIVATION.get(record.status, "studying"),
        "coherence_score": record.coherence_score,
        "distinct_schools": distinct,
        "identity_core_present": bool(detail.get("identity_core_present", False)),
        "inducted": record.status is CultivationStatus.INDUCTED,
        "cultivation_id": record.cultivation_id,
    }
    if template_id:
        view["template_id"] = template_id
    if record.last_exam_run_id:
        view["exam_run_id"] = record.last_exam_run_id
    prov = dict(record.provenance)
    if prov.get("source_kind"):
        view["source_kind"] = str(prov["source_kind"])
    if prov.get("continuation_mode"):
        view["continuation_mode"] = str(prov["continuation_mode"])
    return view


def _build_package_view(package_id: str, records) -> dict[str, Any]:
    """Group track records into the `cultivation.package.v1` JSON shape."""

    tracks = [_track_view(r) for r in records]
    published = [
        {
            "schema_version": "persona-studio.v1",
            "artifact_kind": "template",
            "source_kind": "self_learning",
            "display_name": t["display_name"],
            "template_id": t["template_id"],
            "grounding_status": "partial",
        }
        for t in tracks
        if t.get("inducted") and t.get("template_id")
    ]
    first = records[0] if records else None
    created_at_ms = min((r.created_at_ms for r in records), default=0)
    return {
        "schema_version": "cultivation.package.v1",
        "package_id": package_id,
        "display_name": first.seed_persona.get("display_name", "")
        if first
        else "",
        "domain": first.domain if first else "",
        "seed_slug": first.slug if first else "",
        "source_kind": "self_learning",
        "tracks": tracks,
        "published": published,
        "provenance": {
            "source": "cultivation",
            "reviewed": False,
            "research_egress": "creation_time",
        },
        "created_at_ms": created_at_ms,
    }


async def _handle_list_packages(request: web.Request) -> web.Response:
    tenant = await _authenticate(request)
    bundle = _bundle(request)
    package_ids = await bundle.cultivations.list_package_ids()
    packages = []
    for package_id in package_ids:
        records = await bundle.cultivations.list_for_package(package_id)
        if tenant is not None:
            # All tracks of one package share a tenant (single create
            # call), so this filter keeps whole packages or drops them.
            records = tuple(
                r for r in records if r.tenant_id == tenant.tenant_id
            )
            if not records:
                continue
        packages.append(_build_package_view(package_id, records))
    return web.json_response({"status": "ok", "packages": packages})


async def _handle_get_package(request: web.Request) -> web.Response:
    tenant = await _authenticate(request)
    bundle = _bundle(request)
    package_id = request.match_info["package_id"]
    records = await bundle.cultivations.list_for_package(package_id)
    if not records:
        return _error(404, "package_not_found", package_id)
    denied = _assert_record_access(tenant, records[0])
    if denied is not None:
        return denied
    return web.json_response(
        {"status": "ok", **_build_package_view(package_id, records)}
    )


# ---------------------------------------------------------------------------
# Tick — run autonomous study cycles
# ---------------------------------------------------------------------------


async def _handle_tick(request: web.Request) -> web.Response:
    tenant = await _authenticate(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    denied = _assert_record_access(tenant, record)
    if denied is not None:
        return denied
    if record.status in (
        CultivationStatus.READY_FOR_REVIEW,
        CultivationStatus.INDUCTED,
        CultivationStatus.FAILED,
        CultivationStatus.PAUSED,
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
    # Durable adopted-seed continuation: re-hydrate the source school when
    # the in-process uptake service lost it (restart / eviction).
    uptake = await _ensure_source_hydration(bundle, record)
    uptake.registry.load(build_identity_core_protocol(seed))
    instance_manager.set_protocol_uptake_service(record.ai_id, uptake)

    _uptaker = _make_protocol_uptaker(uptake)

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
    # Persist the per-cycle study trail as a monitoring readout (R12):
    # what the loop researched/studied each cycle, for the supervision
    # console. Never read back as a learning signal.
    await bundle.cultivations.append_events(
        cultivation_id=cultivation_id,
        kind="cycle",
        events=tuple(e.to_json() for e in progress.events),
    )
    # Convergence timeline snapshot (one per tick): the score + dominant
    # school + distinct-school count so the console can chart how the
    # school settles over time. Pure readout (R12).
    detail = dict(progress.coherence_detail)
    await bundle.cultivations.append_events(
        cultivation_id=cultivation_id,
        kind="progress",
        events=(
            {
                "cycle_index": progress.cycles_completed,
                "coherence_score": progress.coherence_score,
                "readout_kind": progress.readout_kind,
                "dominant": str(
                    detail.get("dominant_protocol")
                    or detail.get("dominant_regime", "")
                    or ""
                ),
                "distinct_schools": int(
                    detail.get("distinct_schools")
                    or detail.get("distinct_regimes", 0)
                    or 0
                ),
                "uptaken_protocols": len(progress.uptaken_protocols),
                "converged": progress.converged,
            },
        ),
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
# Events — monitoring readout (self-learning workshop)
# ---------------------------------------------------------------------------


async def _handle_events(request: web.Request) -> web.Response:
    """Return the cultivation's per-cycle study trail + coherence readout.

    Pure readout (R12): the event log is observability for the operator
    supervision console; it is never read back as a learning signal. The
    coherence trajectory is the score history the engine published per
    tick (here the current value + accumulated regime history) so the
    console can chart convergence without owning cognition.
    """

    tenant = await _authenticate(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    denied = _assert_record_access(tenant, record)
    if denied is not None:
        return denied
    limit = _optional_int(request.query, "limit", default=500)
    all_events = await bundle.cultivations.list_events(
        cultivation_id, limit=limit
    )
    # Split the per-tick convergence ``timeline`` (kind="progress") from
    # the event log (cycles / teach corrections / supervision actions) so
    # the console can chart convergence and render the trail separately.
    timeline = [e for e in all_events if e.get("kind") == "progress"]
    events = [e for e in all_events if e.get("kind") != "progress"]
    return web.json_response(
        {
            "status": "ok",
            "cultivation_id": cultivation_id,
            "ai_id": record.ai_id,
            "cultivation_status": record.status.value,
            "cycles_completed": record.cycles_completed,
            "coherence_score": record.coherence_score,
            "coherence_detail": dict(record.coherence_detail),
            "regime_history": list(record.regime_history),
            "provenance": dict(record.provenance),
            "source_template_id": record.source_template_id,
            "events": events,
            "timeline": timeline,
        }
    )


# ---------------------------------------------------------------------------
# Supervise — operator teach correction + pause / resume / reject
# ---------------------------------------------------------------------------

# Statuses a teach correction can be injected into (in-flight study).
_TEACHABLE = (
    CultivationStatus.SEEDING,
    CultivationStatus.STUDYING,
    CultivationStatus.CONVERGING,
    CultivationStatus.EXAM,
    CultivationStatus.PAUSED,
)
# Statuses that can be paused (actively-running study only).
_PAUSABLE = (
    CultivationStatus.SEEDING,
    CultivationStatus.STUDYING,
    CultivationStatus.CONVERGING,
)


async def _handle_teach(request: web.Request) -> web.Response:
    """Inject an operator apprentice correction into a running cultivation.

    The correction text is re-homed through the same uptake service as
    autonomous research (so it competes in the active mixture on PE
    utility — never a hardcoded rule, R4) and run as one apprentice study
    turn. It does not advance the study cycle counter; the next tick
    recomputes coherence with the correction in the mixture. Recorded as
    a ``teach`` event for the supervision console.
    """

    tenant = await _authenticate(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    denied = _assert_record_access(tenant, record)
    if denied is not None:
        return denied
    if record.status not in _TEACHABLE:
        return _error(
            409,
            "cultivation_not_teachable",
            f"cultivation status={record.status.value!r} does not accept "
            "teach corrections.",
        )
    data = await _read_json(request)
    correction = _required_str(data, "text")
    source_label = str(data.get("source_label", "") or "operator:teach")

    seed = CultivationSeed.from_json(dict(record.seed_persona))
    instance_manager = _instance_manager(request)
    uptake = await _ensure_source_hydration(bundle, record)
    uptake.registry.load(build_identity_core_protocol(seed))
    instance_manager.set_protocol_uptake_service(record.ai_id, uptake)
    uptaker = _make_protocol_uptaker(uptake)

    session = await _fresh_tick_session(
        request,
        ai_id=record.ai_id,
        cultivation_id=cultivation_id,
        seq=record.cycles_completed,
    )
    if isinstance(session, web.Response):
        return session

    sink = SessionCultivationSink(session=session, protocol_uptaker=uptaker)
    protocol_id = await sink.uptake_protocol(
        corpus_text=correction, source_label=source_label
    )
    turn = await sink.study(correction)

    event = {
        "cycle_index": record.cycles_completed,
        "kind_detail": "operator_teach",
        "text": correction[:2000],
        "source_label": source_label,
        "protocol_uptaken": protocol_id or "",
        "active_regime": turn.active_regime,
        "response": turn.text[:2000],
    }
    await bundle.cultivations.append_events(
        cultivation_id=cultivation_id, kind="teach", events=(event,)
    )
    return web.json_response(
        {
            "status": "ok",
            "protocol_uptaken": protocol_id or "",
            "active_regime": turn.active_regime,
            "response": turn.text,
            **record.to_json(),
        }
    )


async def _handle_pause(request: web.Request) -> web.Response:
    tenant = await _authenticate(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    denied = _assert_record_access(tenant, record)
    if denied is not None:
        return denied
    if record.status not in _PAUSABLE:
        return _error(
            409,
            "cultivation_not_pausable",
            f"cultivation status={record.status.value!r} cannot be paused.",
        )
    record = await bundle.cultivations.update_status(
        cultivation_id=cultivation_id, status=CultivationStatus.PAUSED
    )
    await bundle.cultivations.append_events(
        cultivation_id=cultivation_id,
        kind="pause",
        events=({"cycle_index": record.cycles_completed},),
    )
    return web.json_response({"status": "ok", **record.to_json()})


async def _handle_resume(request: web.Request) -> web.Response:
    tenant = await _authenticate(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    denied = _assert_record_access(tenant, record)
    if denied is not None:
        return denied
    if record.status is not CultivationStatus.PAUSED:
        return _error(
            409,
            "cultivation_not_paused",
            f"resume requires status=paused (got {record.status.value!r}).",
        )
    # Resume to STUDYING; the next tick recomputes coherence and may move
    # the record back to CONVERGING. (We do not persist the pre-pause
    # status separately because it is a pure function of coherence.)
    record = await bundle.cultivations.update_status(
        cultivation_id=cultivation_id, status=CultivationStatus.STUDYING
    )
    await bundle.cultivations.append_events(
        cultivation_id=cultivation_id,
        kind="resume",
        events=({"cycle_index": record.cycles_completed},),
    )
    return web.json_response({"status": "ok", **record.to_json()})


async def _handle_reject(request: web.Request) -> web.Response:
    """Abandon a cultivation / school track (multi-school selection, R15).

    Used when the operator picks one self-consistent school from a
    package and retires the others. Marks the record ``failed`` with an
    optional reason; an inducted expert cannot be rejected through this
    path (retire it via template lifecycle instead).
    """

    tenant = await _authenticate(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    denied = _assert_record_access(tenant, record)
    if denied is not None:
        return denied
    if record.status is CultivationStatus.INDUCTED:
        return _error(
            409,
            "cultivation_inducted",
            "an inducted expert cannot be rejected; retire its template "
            "via the lifecycle instead.",
        )
    data = await _read_json(request, allow_empty=True)
    reason = str(data.get("reason", "") or "operator_rejected")
    record = await bundle.cultivations.update_status(
        cultivation_id=cultivation_id,
        status=CultivationStatus.FAILED,
        notes=reason,
    )
    await bundle.cultivations.append_events(
        cultivation_id=cultivation_id,
        kind="reject",
        events=({"cycle_index": record.cycles_completed, "reason": reason},),
    )
    return web.json_response({"status": "ok", **record.to_json()})


# ---------------------------------------------------------------------------
# Graduate — create candidate template + run exam evidence
# ---------------------------------------------------------------------------


async def _handle_graduate(request: web.Request) -> web.Response:
    tenant = await _authenticate(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    denied = _assert_record_access(tenant, record)
    if denied is not None:
        return denied
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

    # Durable adopted-seed continuation: ensure the source school is
    # hydrated before the candidate template exports the learned protocol
    # bundle (so a restarted adopted cultivation still carries its school).
    await _ensure_source_hydration(bundle, record)

    # Tenant-owned cultivations graduate into a candidate template owned
    # by that tenant; system cultivations keep the SYSTEM_TENANT_ID path.
    if record.tenant_id:
        owner_tenant_id = record.tenant_id
    else:
        await _ensure_system_tenant(bundle)
        owner_tenant_id = SYSTEM_TENANT_ID
    template = await _ensure_candidate_template(
        bundle, record=record, seed=seed, owner_tenant_id=owner_tenant_id
    )

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
    tenant = await _authenticate(request)
    bundle = _bundle(request)
    cultivation_id = request.match_info["cultivation_id"]
    record = await _resolve(bundle, cultivation_id)
    if isinstance(record, web.Response):
        return record
    denied = _assert_record_access(tenant, record)
    if denied is not None:
        return denied
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
    bundle: CultivationBundle,
    *,
    record,
    seed: CultivationSeed,
    owner_tenant_id: str,
):
    """Create (or reuse) the activated+published candidate template.

    Idempotent across repeated graduate calls: when the cultivation
    already carries a ``dlaas_template_id`` we reuse it.
    ``owner_tenant_id`` is the template's owning tenant —
    ``SYSTEM_TENANT_ID`` for operator (system) cultivations, the
    cultivation's own ``tenant_id`` for tenant-owned ones.
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
        # Package provenance (multi-direction cultivation). Empty when
        # this template came from the legacy single-expert path, so the
        # portal can group sibling tracks back to one seed and roll the
        # package back together (R15).
        "package_id": record.package_id,
        "track_id": record.track_id,
        # Adopted-seed provenance: which baked role this expert grew from
        # (character / author / interpreter / expert) and whether it
        # continued a real learned-state bundle or persona metadata only.
        "source_template_id": record.source_template_id,
        "source_provenance": dict(record.provenance),
    }
    # R15 portability: export the *learned* school (the converged approved
    # protocol set) into the template's seed_config so adoption re-hydrates
    # the cultivated cognition, not just the persona metadata. Empty when
    # the in-process uptake service holds no approved protocols (e.g. the
    # process restarted between tick and graduate) — then the template
    # still carries persona_spec metadata and the legacy seed path applies.
    approved = bundle.uptake_service_for(record.ai_id).loaded_approved_snapshot()
    seed_config: dict[str, Any] | None = None
    if approved:
        seed_config = {
            "cultivation_protocol_bundle": export_protocol_bundle(
                approved,
                source_ai_id=record.ai_id,
                cultivation_id=record.cultivation_id,
                package_id=record.package_id,
                track_id=record.track_id,
            )
        }
    template = await bundle.templates.create(
        tenant_id=owner_tenant_id,
        template_name=seed.display_name,
        domain=seed.domain,
        description=f"Auto-cultivated expert: {seed.display_name} ({seed.domain})",
        runtime_template_id=RUNTIME_TEMPLATE_ID,
        persona_spec=persona_spec,
        seed_config=seed_config,
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


def _make_protocol_uptaker(uptake: ProtocolUptakeService):
    """Build the (text, source_label) -> approved protocol_id|None uptaker.

    Shared by the autonomous tick loop and the operator teach correction:
    researched/taught material is extracted into a BehaviorProtocol
    candidate, submitted, and auto-approved into the uptake registry so it
    competes in the active mixture. Returns ``None`` on the documented
    degraded path (LLM not configured), which the sink turns into raw
    corpus ingestion instead.
    """

    async def _uptaker(text: str, source_label: str) -> str | None:
        try:
            candidate = await uptake.extract_from_markdown_text(
                text, source_label=source_label
            )
        except RuntimeError:
            return None
        await uptake.submit_candidate(candidate)
        try:
            approved = await uptake.approve_pending(
                candidate.protocol.protocol_id, reviewer_id="cultivation-auto"
            )
        except KeyError:
            return None
        return approved.protocol_id

    return _uptaker


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


async def _resolve_source_template(
    bundle: CultivationBundle,
    source_template_id: str,
    tenant: TenantSpec | None,
):
    """Fetch the adopted-seed source template + its protocol bundle.

    Returns ``(TemplateSpec, bundle_payload | None)`` or a typed
    ``web.Response`` on failure. ``bundle_payload`` is the source
    template's converged ``cultivation_protocol_bundle`` when present (a
    previously-cultivated/inducted expert); ``None`` for a baked persona
    that carries only ``persona_spec`` metadata, in which case the
    cultivation continues from the persona anchor via the seed Identity
    Core (documented metadata-only continuation, not an error).

    Trust boundary: a tenant caller may adopt only a template it owns or
    one that is PUBLISHED (the platform's adoptable contract); operators
    (``tenant=None``) may adopt cross-tenant. Cross-tenant adoption of a
    non-published template is a typed 403, never a silent read.
    """

    try:
        template = await bundle.templates.get(source_template_id)
    except TemplateNotFound:
        return _error(
            404,
            "source_template_not_found",
            f"source_template_id={source_template_id!r} does not exist.",
        )
    if tenant is not None:
        owns = template.tenant_id == tenant.tenant_id
        adoptable = template.status is TemplateStatus.PUBLISHED
        if not owns and not adoptable:
            return _error(
                403,
                "source_template_forbidden",
                (
                    f"tenant_id={tenant.tenant_id!r} cannot adopt template "
                    f"{source_template_id!r}: not owned and not published."
                ),
            )
    seed_config = dict(template.seed_config or {})
    payload = seed_config.get("cultivation_protocol_bundle")
    return template, payload


def _source_provenance(
    data: Mapping[str, Any], template, payload: Any
) -> dict[str, Any]:
    """Build adopted-seed provenance (role semantics + continuation mode).

    ``source_kind`` / ``source_angle`` preserve whether the adopted source
    was a character / author / interpreter / expert (inferred from the
    template's ``persona_spec`` / figure artifact, overridable by the
    caller). ``continuation_mode`` records whether the cultivation
    continues from a real learned-state bundle or only the persona anchor,
    so operators are never misled about what was carried over.
    """

    persona = dict(template.persona_spec or {})
    kind = (
        str(data.get("source_kind", "") or "").strip()
        or str(
            persona.get("source_kind")
            or persona.get("bake_angle")
            or persona.get("artifact_kind")
            or ""
        ).strip()
    )
    if not kind and template.figure_artifact_id:
        kind = "figure"
    kind = kind or "expert"
    angle = (
        str(data.get("source_angle", "") or "").strip()
        or str(persona.get("source_angle") or "").strip()
        or kind
    )
    return {
        "source_template_id": template.template_id,
        "source_kind": kind,
        "source_angle": angle,
        "continuation_mode": "protocol_bundle" if payload else "metadata_only",
    }


async def _ensure_source_hydration(bundle: CultivationBundle, record):
    """Return the per-ai_id uptake service, re-hydrating the adopted seed.

    Durability fix: the uptake service lives in process memory, so a
    restart (or LRU eviction) empties it and an adopted cultivation would
    otherwise collapse to persona-metadata-only on the next tick. When the
    record was seeded from a source template and the service currently
    holds no approved protocols, re-hydrate the source template's
    ``cultivation_protocol_bundle`` so the adopted school is restored
    before study continues. Empty-seed records are untouched.
    """

    svc = bundle.uptake_service_for(record.ai_id)
    if not record.source_template_id:
        return svc
    if svc.loaded_approved_snapshot():
        return svc
    try:
        template = await bundle.templates.get(record.source_template_id)
    except TemplateNotFound:
        return svc
    payload = dict(template.seed_config or {}).get("cultivation_protocol_bundle")
    if not payload:
        return svc
    try:
        return bundle.hydrate_uptake_from_bundle(record.ai_id, payload)
    except ValueError:
        return bundle.uptake_service_for(record.ai_id)


def _seed_defaults_from_template(
    data: Mapping[str, Any], template
) -> dict[str, Any]:
    """Fill seed fields from a source template's persona_spec.

    Operator-supplied fields win; only blanks are filled from the
    template so the adopted seed inherits the persona's identity while
    staying overridable. ``role_archetype`` falls back to a domain-based
    label so :class:`CultivationSeed` parsing never fails on a baked
    template that omitted it.
    """

    persona = dict(template.persona_spec or {})
    merged: dict[str, Any] = dict(data)

    def _fill(key: str, value: Any) -> None:
        existing = merged.get(key)
        if existing is None or (isinstance(existing, str) and not existing.strip()):
            if value not in (None, ""):
                merged[key] = value

    domain = str(persona.get("domain") or template.domain or "").strip()
    _fill("display_name", persona.get("display_name") or template.template_name)
    _fill("domain", domain)
    _fill(
        "role_archetype",
        persona.get("role_archetype") or (f"{domain}专家" if domain else ""),
    )
    _fill("focus", persona.get("focus"))
    _fill("single_school_objective", persona.get("single_school_objective"))
    if not merged.get("value_boundaries"):
        boundaries = persona.get("value_boundaries")
        if isinstance(boundaries, list):
            merged["value_boundaries"] = boundaries
    return merged


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
