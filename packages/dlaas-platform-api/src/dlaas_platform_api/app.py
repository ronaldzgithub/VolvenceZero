"""DLaaS aiohttp router and dispatch entry point.

This module owns the ``/dlaas/instances/{ai_id}/interactions`` route
plus the wiring to attach the control-plane CRUD surface (Slice 3 +
4) and the multi-ai_id launcher.

Dispatch flow:

1. Parse the JSON body into a typed
   :class:`dlaas_platform_contracts.InteractionEnvelope`.
2. If the app carries an :class:`InstanceManager`, look up the
   ``SessionManager`` for the path ``ai_id``. Fall back to the
   single-instance ``app["session_manager"]`` (Slice 1) when no
   launcher is bound or when the ``ai_id`` is not adopted yet.
3. Hand off to :func:`dlaas_platform_api.dispatch.dispatch_envelope`,
   which switches on :class:`InteractionType` and calls the matching
   kernel sink.
4. Serialise the resulting JSON body back to the client.

The Slice 1 ``attach_dlaas_routes`` entry point continues to work
without a registry — it pins every request to the single shared
``SessionManager``. ``build_dlaas_app`` is the recommended Slice 3+
entry point: it builds a registry, a launcher, and the full
control-plane surface in one call.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import (
    AdoptionConfig,
    AssetIntakeIntent,
    AssetIntakeRequest,
    AssetMediaKind,
    ArtifactKind,
    ArtifactRecord,
    AuditEvent,
    BillingEvent,
    ConsentRecord,
    DataExportJob,
    DebugAnalysisReport,
    DebugAnalysisRequest,
    DebugAppRegistration,
    DebugEventEnvelope,
    DebugFieldType,
    DebugPrivacyLevel,
    DebugSchema,
    DeletionJob,
    EvalGateDecision,
    EvalRun,
    EvalRunStatus,
    EventStreamRecord,
    ExplainTrace,
    InteractionEnvelope,
    InteractionType,
    LifeBlueprint,
    PolicySnapshot,
    ProtocolSubmission,
    PromotionDecision,
    QuotaSnapshot,
    ReadoutBundle,
    ReadoutView,
    SleepRequest,
    SnapshotExportRequest,
    TrainingJob,
    TrainingJobStatus,
    UsageRecord,
    WakeRequest,
    WebhookSubscription,
)
from dlaas_platform_launcher import (
    INSTANCE_MANAGER_APP_KEY,
    InstanceManager,
    InstanceNotFound,
)
from dlaas_platform_launcher.instance_manager import default_vertical_resolver
from dlaas_platform_eval import attach_eval_routes
from dlaas_platform_ops import (
    OPS_BUNDLE_APP_KEY,
    OpsBundle,
    attach_ops_routes,
    operator_takeover_response_body,
)
from dlaas_platform_registry import (
    GovernanceRecordNotFound,
    GovernanceStore,
    PlatformAuthBundle,
    PlatformAuthConfig,
    REGISTRY_APP_KEY,
    Registry,
    TenantStore,
)
from lifeform_service.app import create_app as create_lifeform_app
from lifeform_service.session_manager import (
    SessionAlreadyExistsError,
    SessionManager,
    SessionNotFoundError,
)

from dlaas_platform_api.control_plane import (
    CONTROL_PLANE_STORES_KEY,
    _BIND_REASON_TO_ERROR_CODE,
    _bind_failure_detail,
    attach_control_plane_routes,
    bind_figure_artifact_to_ai_id,
)
from dlaas_platform_registry import TemplateNotFound
from dlaas_platform_api.debug_analysis import build_debug_analysis
from dlaas_platform_api.dispatch import DispatchError, dispatch_envelope
from dlaas_platform_api.intake_router import resolve_intake_decision
from dlaas_platform_api.snapshot_export import snapshot_to_json

_LOG = logging.getLogger("dlaas_platform_api")

DLAAS_APP_AI_ID_KEY = "dlaas_default_ai_id"
"""``app[DLAAS_APP_AI_ID_KEY]`` — Slice 1 hardcoded ``ai_id`` fallback."""

_PROTOCOL_SUBMISSIONS_KEY = "dlaas_protocol_submissions"
_TRAINING_JOBS_KEY = "dlaas_training_jobs"
_ASSET_INTAKES_KEY = "dlaas_asset_intakes"
_AUDIT_EVENTS_KEY = "dlaas_audit_events"
_ARTIFACTS_KEY = "dlaas_artifacts"
_DATA_EXPORT_JOBS_KEY = "dlaas_data_export_jobs"
_DELETION_JOBS_KEY = "dlaas_deletion_jobs"
_EVAL_RUNS_KEY = "dlaas_eval_runs"
_WEBHOOKS_KEY = "dlaas_webhooks"
_EVENT_STREAM_KEY = "dlaas_event_stream"
_USAGE_RECORDS_KEY = "dlaas_usage_records"
_BILLING_EVENTS_KEY = "dlaas_billing_events"
_CONSENTS_KEY = "dlaas_consents"
_POLICIES_KEY = "dlaas_policies"
_GOVERNANCE_STORE_KEY = "dlaas_governance_store"
_DEBUG_APPS_KEY = "dlaas_debug_apps"
_DEBUG_SCHEMAS_KEY = "dlaas_debug_schemas"
_DEBUG_EVENTS_KEY = "dlaas_debug_events"
_DEBUG_ANALYSES_KEY = "dlaas_debug_analyses"


def attach_dlaas_routes(
    app: web.Application,
    *,
    default_ai_id: str = "ai_default",
) -> web.Application:
    """Register only the runtime ``/dlaas/instances/{ai_id}/interactions``.

    Slice 1 entry point. The app MUST already have a
    ``session_manager`` set up by ``lifeform_service.app.create_app``;
    every request — regardless of the path ``ai_id`` — is served by
    that single SessionManager. Suitable for dev / smoke testing
    before the multi-tenant control plane is wired.
    """
    if "session_manager" not in app:
        raise ValueError(
            "attach_dlaas_routes requires an aiohttp app produced by "
            "lifeform_service.app.create_app (session_manager missing)."
        )
    app[DLAAS_APP_AI_ID_KEY] = default_ai_id
    _ensure_shadow_intake_stores(app)
    app.router.add_post(
        "/dlaas/instances/{ai_id}/interactions",
        _handle_interaction,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/interactions",
        _handle_interaction,
    )
    _add_lifecycle_routes(app)
    return app


def attach_dlaas_full_stack(
    app: web.Application,
    *,
    registry: Registry,
    auth_config: PlatformAuthConfig,
    instance_manager: InstanceManager,
    default_ai_id: str = "ai_default",
    platform_endpoint: str = "",
) -> web.Application:
    """Wire registry + launcher + control plane onto an aiohttp app.

    Slice 3+ entry point. The app must already carry a
    ``session_manager`` (the Slice 1 fallback the dispatch reaches
    for when the path ``ai_id`` is not registered with the
    launcher). After this call the app exposes the runtime
    ``/dlaas/instances/{ai_id}/interactions`` route plus every
    control-plane CRUD endpoint listed in
    ``docs/specs/dlaas-platform.md``.
    """
    if "session_manager" not in app:
        raise ValueError(
            "attach_dlaas_full_stack requires an aiohttp app produced by "
            "lifeform_service.app.create_app (session_manager missing)."
        )
    app[REGISTRY_APP_KEY] = PlatformAuthBundle(
        tenant_store=TenantStore(registry),
        auth_config=auth_config,
    )
    app[INSTANCE_MANAGER_APP_KEY] = instance_manager
    app[_GOVERNANCE_STORE_KEY] = GovernanceStore(registry)
    app[DLAAS_APP_AI_ID_KEY] = default_ai_id
    _ensure_shadow_intake_stores(app)
    if platform_endpoint:
        app["dlaas_platform_endpoint"] = platform_endpoint
    app.router.add_post(
        "/dlaas/instances/{ai_id}/interactions",
        _handle_interaction,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/interactions",
        _handle_interaction,
    )
    _add_lifecycle_routes(app)
    attach_control_plane_routes(app, registry=registry)
    attach_ops_routes(app, registry=registry)
    attach_eval_routes(app, registry=registry)
    return app


def _add_lifecycle_routes(app: web.Application) -> None:
    app.router.add_get("/dlaas/v1/audit/events", _handle_audit_events)
    app.router.add_get("/dlaas/v1/audit/events/{event_id}", _handle_audit_event)
    app.router.add_get("/dlaas/v1/audit/traces/{session_id}", _handle_audit_trace)
    app.router.add_post("/dlaas/v1/debug/apps", _handle_debug_app_register)
    app.router.add_get("/dlaas/v1/debug/apps", _handle_debug_apps_list)
    app.router.add_get("/dlaas/v1/debug/apps/{app_id}", _handle_debug_app_get)
    app.router.add_post(
        "/dlaas/v1/debug/apps/{app_id}/schemas", _handle_debug_schema_register
    )
    app.router.add_get(
        "/dlaas/v1/debug/apps/{app_id}/schemas", _handle_debug_schemas_list
    )
    app.router.add_post("/dlaas/v1/debug/events", _handle_debug_event_ingest)
    app.router.add_get("/dlaas/v1/debug/events", _handle_debug_events_list)
    app.router.add_post("/dlaas/v1/debug/analysis", _handle_debug_analysis_create)
    app.router.add_get(
        "/dlaas/v1/debug/analysis/{analysis_id}", _handle_debug_analysis_get
    )
    app.router.add_get("/dlaas/v1/artifacts", _handle_artifacts_list)
    app.router.add_get("/dlaas/v1/artifacts/{artifact_id}", _handle_artifact_get)
    app.router.add_post(
        "/dlaas/v1/artifacts/{artifact_id}/promote", _handle_artifact_promote
    )
    app.router.add_post("/dlaas/v1/eval/runs", _handle_eval_run_create)
    app.router.add_get("/dlaas/v1/eval/runs/{run_id}", _handle_eval_run_get)
    app.router.add_post(
        "/dlaas/v1/eval/runs/{run_id}/approve", _handle_eval_run_approve
    )
    app.router.add_post("/dlaas/v1/webhooks", _handle_webhook_create)
    app.router.add_get("/dlaas/v1/events/stream", _handle_events_stream)
    app.router.add_get("/dlaas/v1/usage", _handle_usage)
    app.router.add_get("/dlaas/v1/quota", _handle_quota)
    app.router.add_get("/dlaas/v1/billing/events", _handle_billing_events)
    app.router.add_get("/dlaas/v1/policies", _handle_policies)
    app.router.add_post("/dlaas/v1/consents", _handle_consent_create)
    app.router.add_get(
        "/dlaas/v1/consents/{end_user_ref}", _handle_consent_get
    )
    app.router.add_get("/dlaas/v1/catalog/blueprints", _handle_catalog_blueprints)
    app.router.add_get("/dlaas/v1/catalog/verticals", _handle_catalog_verticals)
    app.router.add_get(
        "/dlaas/v1/catalog/substrate-profiles",
        _handle_catalog_substrate_profiles,
    )
    app.router.add_get("/dlaas/v1/catalog/protocols", _handle_catalog_protocols)
    app.router.add_get(
        "/dlaas/v1/catalog/tool-policies", _handle_catalog_tool_policies
    )
    app.router.add_get(
        "/dlaas/v1/catalog/training-policies",
        _handle_catalog_training_policies,
    )
    app.router.add_get(
        "/dlaas/v1/admin/instances/{ai_id}/snapshots",
        _handle_admin_snapshots,
    )
    app.router.add_get("/dlaas/v1/instances", _handle_list_instances)
    app.router.add_get("/dlaas/v1/instances/{ai_id}/status", _handle_instance_status)
    app.router.add_get("/dlaas/v1/instances/{ai_id}/readouts", _handle_readouts)
    app.router.add_get("/dlaas/v1/instances/{ai_id}/explain", _handle_explain)
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/data/export", _handle_data_export
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/data/delete", _handle_data_delete
    )
    app.router.add_get(
        "/dlaas/v1/instances/{ai_id}/data/deletion-status/{job_id}",
        _handle_deletion_status,
    )
    app.router.add_post("/dlaas/v1/instances/{ai_id}/wake", _handle_wake_instance)
    app.router.add_post("/dlaas/v1/instances/{ai_id}/sleep", _handle_sleep_instance)
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/feedback", _handle_feedback_alias
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/environment/events",
        _handle_environment_event_alias,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/environment/outcomes",
        _handle_environment_outcome_alias,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/protocols/submissions",
        _handle_protocol_submission_create,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/safety/protocols",
        _handle_safety_protocol_create,
    )
    app.router.add_get(
        "/dlaas/v1/instances/{ai_id}/safety/protocols",
        _handle_safety_protocol_list,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/safety/protocols/{submission_id}/approve",
        _handle_safety_protocol_approve,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/safety/protocols/{protocol_id}/load",
        _handle_safety_protocol_load,
    )
    app.router.add_get(
        "/dlaas/v1/instances/{ai_id}/protocols/submissions",
        _handle_protocol_submission_list,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/protocols/submissions/{submission_id}/approve",
        _handle_protocol_submission_approve,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/protocols/submissions/{submission_id}/reject",
        _handle_protocol_submission_reject,
    )
    app.router.add_get(
        "/dlaas/v1/instances/{ai_id}/protocols/library",
        _handle_protocol_library_list,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/protocols/library/{protocol_id}/load",
        _handle_protocol_library_load,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/protocols/library/{protocol_id}/unload",
        _handle_protocol_library_unload,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/training/corpus",
        _handle_training_corpus,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/assets/intake",
        _handle_asset_intake,
    )
    app.router.add_get(
        "/dlaas/v1/instances/{ai_id}/assets/intake/{asset_id}",
        _handle_asset_intake_get,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/training/jobs",
        _handle_training_job_create,
    )
    app.router.add_get(
        "/dlaas/v1/instances/{ai_id}/training/jobs/{job_id}",
        _handle_training_job_get,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/training/jobs/{job_id}/cancel",
        _handle_training_job_cancel,
    )
    app.router.add_post(
        "/dlaas/v1/instances/{ai_id}/training/jobs/{job_id}/promote",
        _handle_training_job_promote,
    )


def _ensure_shadow_intake_stores(app: web.Application) -> None:
    app.setdefault(_PROTOCOL_SUBMISSIONS_KEY, {})
    app.setdefault(_TRAINING_JOBS_KEY, {})
    app.setdefault(_ASSET_INTAKES_KEY, {})
    app.setdefault(_AUDIT_EVENTS_KEY, {})
    app.setdefault(_ARTIFACTS_KEY, {})
    app.setdefault(_DATA_EXPORT_JOBS_KEY, {})
    app.setdefault(_DELETION_JOBS_KEY, {})
    app.setdefault(_EVAL_RUNS_KEY, {})
    app.setdefault(_WEBHOOKS_KEY, {})
    app.setdefault(_EVENT_STREAM_KEY, [])
    app.setdefault(_USAGE_RECORDS_KEY, [])
    app.setdefault(_BILLING_EVENTS_KEY, [])
    app.setdefault(_CONSENTS_KEY, {})
    app.setdefault(_DEBUG_APPS_KEY, {})
    app.setdefault(_DEBUG_SCHEMAS_KEY, {})
    app.setdefault(_DEBUG_EVENTS_KEY, {})
    app.setdefault(_DEBUG_ANALYSES_KEY, {})
    app.setdefault(
        _POLICIES_KEY,
        {
            "default-data-governance": PolicySnapshot(
                policy_id="default-data-governance",
                policy_kind="data_governance",
                rules={
                    "export": "allowed",
                    "delete": "job_tracked",
                    "raw_snapshot": "admin_only",
                },
            ),
            "default-training-consent": PolicySnapshot(
                policy_id="default-training-consent",
                policy_kind="training_consent",
                rules={
                    "protocol_intake": "requires_review",
                    "corpus_intake": "requires_consent",
                    "adapter_training": "offline_gate_required",
                },
            ),
        },
    )


def build_dlaas_app(
    *,
    db_path: str | os.PathLike[str] = ":memory:",
    default_ai_id: str = "ai_default",
    control_plane_secret: str | None = None,
    service_secret: str | None = None,
    platform_endpoint: str = "",
    instance_manager: InstanceManager | None = None,
    **service_kwargs: Any,
) -> web.Application:
    """Build a lifeform-service app with the full DLaaS surface attached.

    Equivalent to:

        app = lifeform_service.app.create_app(**service_kwargs)
        registry = Registry(db_path=db_path)
        instance_manager = InstanceManager(
            vertical_resolver=default_vertical_resolver(),
            substrate_runtime=app["session_manager"].substrate_runtime,
        )
        attach_dlaas_full_stack(
            app,
            registry=registry,
            auth_config=PlatformAuthConfig(...),
            instance_manager=instance_manager,
            ...
        )

    ``control_plane_secret`` / ``service_secret`` default to
    ``$DLAAS_CONTROL_PLANE_SECRET`` / ``$DLAAS_SERVICE_SECRET``
    when the caller does not supply them explicitly. An empty value
    administratively disables that auth mode.
    """
    app = create_lifeform_app(**service_kwargs)
    registry = Registry(db_path=str(db_path))
    if instance_manager is None:
        service_manager = app["session_manager"]
        instance_manager = InstanceManager(
            vertical_resolver=default_vertical_resolver(),
            substrate_runtime=service_manager.substrate_runtime,
            alpha_identity_provider=service_manager.alpha_identity_provider,
            alpha_memory_scope_root_dir=service_manager.alpha_memory_scope_root_dir,
            attach_default_mcp_bundle=True,
        )
    auth_config = PlatformAuthConfig(
        control_plane_secret=(
            control_plane_secret
            if control_plane_secret is not None
            else os.environ.get("DLAAS_CONTROL_PLANE_SECRET", "")
        ),
        service_secret=(
            service_secret
            if service_secret is not None
            else os.environ.get("DLAAS_SERVICE_SECRET", "")
        ),
    )
    attach_dlaas_full_stack(
        app,
        registry=registry,
        auth_config=auth_config,
        instance_manager=instance_manager,
        default_ai_id=default_ai_id,
        platform_endpoint=platform_endpoint,
    )
    return app


# ---------------------------------------------------------------------------
# Runtime dispatch
# ---------------------------------------------------------------------------


async def _handle_data_export(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    data = await _read_json_or_error(request, allow_empty=True)
    if isinstance(data, web.Response):
        return data
    contract_id = str(data.get("contract_id", "") or "")
    end_user_ref = str(data.get("end_user_ref", "") or "")
    job = DataExportJob(
        job_id=_new_id("export"),
        ai_id=ai_id,
        contract_id=contract_id,
        end_user_ref=end_user_ref,
        artifact_ref=f"artifact://data-export/{ai_id}/{end_user_ref or 'all'}",
        created_at_ms=_now_ms(),
    )
    _data_export_store(request)[job.job_id] = job
    _persist_governance(request, "data_export_job", job.job_id, job.to_json(), ai_id=ai_id, contract_id=contract_id)
    _record_audit(
        request,
        event_type="data_export_requested",
        ai_id=ai_id,
        contract_id=contract_id,
        payload=job.to_json(),
    )
    _record_usage(request, ai_id=ai_id, metric="data_export", quantity=1)
    return web.json_response({"status": "ok", **job.to_json()}, status=201)


async def _handle_data_delete(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    data = await _read_json_or_error(request, allow_empty=True)
    if isinstance(data, web.Response):
        return data
    contract_id = str(data.get("contract_id", "") or "")
    end_user_ref = str(data.get("end_user_ref", "") or "")
    scopes = tuple(str(s) for s in (data.get("scopes") or ("runtime", "platform")))
    job = DeletionJob(
        job_id=_new_id("delete"),
        ai_id=ai_id,
        contract_id=contract_id,
        end_user_ref=end_user_ref,
        deleted_scopes=scopes,
        created_at_ms=_now_ms(),
    )
    _deletion_store(request)[(ai_id, job.job_id)] = job
    _persist_governance(request, "deletion_job", job.job_id, job.to_json(), ai_id=ai_id, contract_id=contract_id)
    _record_audit(
        request,
        event_type="data_delete_requested",
        ai_id=ai_id,
        contract_id=contract_id,
        payload=job.to_json(),
    )
    _record_usage(request, ai_id=ai_id, metric="data_delete", quantity=1)
    return web.json_response({"status": "ok", **job.to_json()}, status=202)


async def _handle_deletion_status(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    job_id = request.match_info.get("job_id", "")
    job = _deletion_store(request).get((ai_id, job_id))
    if job is None:
        persisted = _get_persisted_governance(request, "deletion_job", job_id)
        if persisted is not None and persisted.get("ai_id") == ai_id:
            return web.json_response({"status": "ok", **persisted})
    if job is None:
        return _json_error(status=404, error="deletion_job_not_found", detail=job_id)
    return web.json_response({"status": "ok", **job.to_json()})


async def _handle_audit_events(request: web.Request) -> web.Response:
    events_json = _list_persisted_governance(request, "audit_event")
    if events_json:
        if request.query.get("ai_id", ""):
            ai_id_filter = request.query.get("ai_id", "")
            events_json = [e for e in events_json if e.get("ai_id") == ai_id_filter]
        return web.json_response({"status": "ok", "events": events_json})
    events = list(_audit_store(request).values())
    ai_id = request.query.get("ai_id", "")
    if ai_id:
        events = [e for e in events if e.ai_id == ai_id]
    return web.json_response(
        {"status": "ok", "events": [e.to_json() for e in events]}
    )


async def _handle_audit_event(request: web.Request) -> web.Response:
    event_id = request.match_info.get("event_id", "")
    event = _audit_store(request).get(event_id)
    if event is None:
        persisted = _get_persisted_governance(request, "audit_event", event_id)
        if persisted is not None:
            return web.json_response({"status": "ok", **persisted})
    if event is None:
        return _json_error(status=404, error="audit_event_not_found", detail=event_id)
    return web.json_response({"status": "ok", **event.to_json()})


async def _handle_audit_trace(request: web.Request) -> web.Response:
    session_id = request.match_info.get("session_id", "")
    persisted_events = _list_persisted_governance(
        request, "audit_event", session_id=session_id
    )
    if persisted_events:
        return web.json_response(
            {
                "status": "ok",
                "trace_id": f"trace:{session_id}",
                "session_id": session_id,
                "event_ids": [e["event_id"] for e in persisted_events],
            }
        )
    events = [e for e in _audit_store(request).values() if e.session_id == session_id]
    trace = {
        "trace_id": f"trace:{session_id}",
        "session_id": session_id,
        "event_ids": [e.event_id for e in events],
    }
    return web.json_response({"status": "ok", **trace})


async def _handle_debug_app_register(request: web.Request) -> web.Response:
    data = await _read_json_or_error(request)
    if isinstance(data, web.Response):
        return data
    try:
        registration = DebugAppRegistration.from_json(
            data,
            created_at_ms=_now_ms(),
        )
    except ValueError as exc:
        return _json_error(status=400, error="invalid_debug_app", detail=str(exc))
    _debug_app_store(request)[registration.app_id] = registration
    _persist_governance(
        request,
        "debug_app",
        registration.app_id,
        registration.to_json(),
    )
    _record_audit(
        request,
        event_type="debug_app_registered",
        payload=registration.to_json(),
    )
    return web.json_response({"status": "ok", **registration.to_json()}, status=201)


async def _handle_debug_apps_list(request: web.Request) -> web.Response:
    persisted = _list_persisted_governance(request, "debug_app")
    apps = persisted or [app.to_json() for app in _debug_app_store(request).values()]
    return web.json_response({"status": "ok", "apps": apps})


async def _handle_debug_app_get(request: web.Request) -> web.Response:
    app_id = request.match_info.get("app_id", "")
    registration = _debug_app_store(request).get(app_id)
    if registration is None:
        persisted = _get_persisted_governance(request, "debug_app", app_id)
        if persisted is not None:
            return web.json_response({"status": "ok", **persisted})
    if registration is None:
        return _json_error(status=404, error="debug_app_not_found", detail=app_id)
    return web.json_response({"status": "ok", **registration.to_json()})


async def _handle_debug_schema_register(request: web.Request) -> web.Response:
    app_id = request.match_info.get("app_id", "")
    if not _debug_app_exists(request, app_id):
        return _json_error(status=404, error="debug_app_not_found", detail=app_id)
    data = await _read_json_or_error(request)
    if isinstance(data, web.Response):
        return data
    try:
        schema = DebugSchema.from_json(data, app_id=app_id, created_at_ms=_now_ms())
    except ValueError as exc:
        return _json_error(status=400, error="invalid_debug_schema", detail=str(exc))
    schema_key = (schema.app_id, schema.schema_version)
    _debug_schema_store(request)[schema_key] = schema
    _persist_governance(
        request,
        "debug_schema",
        _debug_schema_record_id(schema.app_id, schema.schema_version),
        schema.to_json(),
    )
    _record_audit(
        request,
        event_type="debug_schema_registered",
        payload=schema.to_json(),
    )
    return web.json_response({"status": "ok", **schema.to_json()}, status=201)


async def _handle_debug_schemas_list(request: web.Request) -> web.Response:
    app_id = request.match_info.get("app_id", "")
    persisted = [
        schema
        for schema in _list_persisted_governance(request, "debug_schema")
        if schema.get("app_id") == app_id
    ]
    schemas = persisted or [
        schema.to_json()
        for (stored_app_id, _), schema in _debug_schema_store(request).items()
        if stored_app_id == app_id
    ]
    return web.json_response({"status": "ok", "schemas": schemas})


async def _handle_debug_event_ingest(request: web.Request) -> web.Response:
    data = await _read_json_or_error(request)
    if isinstance(data, web.Response):
        return data
    event_or_response = _parse_debug_event(request, data)
    if isinstance(event_or_response, web.Response):
        return event_or_response
    event = event_or_response
    _debug_event_store(request)[event.debug_event_id] = event
    _persist_governance(
        request,
        "debug_event",
        event.debug_event_id,
        event.to_json(),
        ai_id=event.ai_id,
        session_id=event.session_id,
    )
    _record_audit(
        request,
        event_type="debug_event_ingested",
        ai_id=event.ai_id,
        session_id=event.session_id,
        actor=event.app_id,
        payload={
            "debug_event_id": event.debug_event_id,
            "app_id": event.app_id,
            "event_type": event.event_type,
            "stage": event.stage,
            "schema_version": event.schema_version,
        },
    )
    _record_usage(request, ai_id=event.ai_id, metric="debug_event", quantity=1)
    return web.json_response({"status": "ok", **event.to_json()}, status=201)


async def _handle_debug_events_list(request: web.Request) -> web.Response:
    events = _list_debug_events(request)
    total = len(events)
    offset = max(0, _int_query(request, "offset", 0))
    limit = max(1, min(_int_query(request, "limit", 100), 500))
    page = events[offset : offset + limit]
    return web.json_response(
        {
            "status": "ok",
            "events": page,
            "pagination": {
                "total": total,
                "offset": offset,
                "limit": limit,
                "returned": len(page),
            },
        }
    )


async def _handle_debug_analysis_create(request: web.Request) -> web.Response:
    data = await _read_json_or_error(request)
    if isinstance(data, web.Response):
        return data
    try:
        analysis_request = DebugAnalysisRequest.from_json(data)
    except ValueError as exc:
        return _json_error(status=400, error="invalid_debug_analysis", detail=str(exc))
    if analysis_request.include_snapshots and not _has_admin_auth(request):
        return _json_error(
            status=403,
            error="admin_auth_required",
            detail="Snapshot-backed debug analysis requires admin or service auth.",
        )
    evidence = await _build_debug_analysis_evidence(request, analysis_request)
    analysis = build_debug_analysis(
        analysis_request=analysis_request,
        evidence=evidence,
    )
    artifact_id = _new_id("debug_artifact")
    report = DebugAnalysisReport(
        analysis_id=_new_id("debug_analysis"),
        prompt=analysis_request.prompt,
        selectors=analysis_request.selectors_json(),
        evidence=evidence,
        recommendations=tuple(analysis["recommendations"]),
        version_suggestions=tuple(analysis["version_suggestions"]),
        analysis_mode=str(analysis["analysis_mode"]),
        prompt_template=str(analysis["prompt_template"]),
        artifact_id=artifact_id,
        created_at_ms=_now_ms(),
    )
    artifact = ArtifactRecord(
        artifact_id=artifact_id,
        artifact_kind=ArtifactKind.DEBUG_ANALYSIS,
        ai_id=analysis_request.ai_id,
        source_ref=f"debug-analysis://{report.analysis_id}",
        metadata={
            **report.to_json(),
            "suggestion_type": "debug_version_suggestion",
            "evidence_summary": analysis["evidence_summary"],
        },
        created_at_ms=report.created_at_ms,
    )
    _debug_analysis_store(request)[report.analysis_id] = report
    _artifact_store(request)[artifact_id] = artifact
    _persist_governance(
        request,
        "debug_analysis",
        report.analysis_id,
        report.to_json(),
        ai_id=analysis_request.ai_id,
        session_id=analysis_request.session_id,
    )
    _persist_governance(
        request,
        "artifact",
        artifact_id,
        artifact.to_json(),
        ai_id=analysis_request.ai_id,
    )
    _record_audit(
        request,
        event_type="debug_analysis_created",
        ai_id=analysis_request.ai_id,
        session_id=analysis_request.session_id,
        payload={
            "analysis_id": report.analysis_id,
            "artifact_id": artifact_id,
            "selectors": report.selectors,
        },
    )
    return web.json_response({"status": "ok", **report.to_json()}, status=201)


async def _handle_debug_analysis_get(request: web.Request) -> web.Response:
    analysis_id = request.match_info.get("analysis_id", "")
    report = _debug_analysis_store(request).get(analysis_id)
    if report is None:
        persisted = _get_persisted_governance(request, "debug_analysis", analysis_id)
        if persisted is not None:
            return web.json_response({"status": "ok", **persisted})
    if report is None:
        return _json_error(
            status=404,
            error="debug_analysis_not_found",
            detail=analysis_id,
        )
    return web.json_response({"status": "ok", **report.to_json()})


async def _handle_artifacts_list(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "artifacts": _list_persisted_governance(request, "artifact")
            or [a.to_json() for a in _artifact_store(request).values()],
        }
    )


async def _handle_artifact_get(request: web.Request) -> web.Response:
    artifact_id = request.match_info.get("artifact_id", "")
    artifact = _artifact_store(request).get(artifact_id)
    if artifact is None:
        persisted = _get_persisted_governance(request, "artifact", artifact_id)
        if persisted is not None:
            return web.json_response({"status": "ok", **persisted})
    if artifact is None:
        return _json_error(status=404, error="artifact_not_found", detail=artifact_id)
    return web.json_response({"status": "ok", **artifact.to_json()})


async def _handle_artifact_promote(request: web.Request) -> web.Response:
    artifact_id = request.match_info.get("artifact_id", "")
    artifact = _artifact_store(request).get(artifact_id)
    if artifact is None:
        return _json_error(status=404, error="artifact_not_found", detail=artifact_id)
    data = await _read_json_or_error(request, allow_empty=True)
    if isinstance(data, web.Response):
        return data
    gate_evidence = data.get("gate_evidence") or {}
    if artifact.artifact_kind is ArtifactKind.ADAPTER_CANDIDATE and not gate_evidence:
        blocked = ArtifactRecord(
            artifact_id=artifact.artifact_id,
            artifact_kind=artifact.artifact_kind,
            ai_id=artifact.ai_id,
            contract_id=artifact.contract_id,
            source_ref=artifact.source_ref,
            status="blocked",
            metadata=artifact.metadata,
            promotion_decision=PromotionDecision.BLOCK,
            created_at_ms=artifact.created_at_ms,
        )
        _artifact_store(request)[artifact_id] = blocked
        _persist_governance(
            request,
            "artifact",
            artifact_id,
            blocked.to_json(),
            ai_id=blocked.ai_id,
            contract_id=blocked.contract_id,
        )
        _record_audit(
            request,
            event_type="artifact_promotion_blocked",
            ai_id=blocked.ai_id,
            contract_id=blocked.contract_id,
            payload=blocked.to_json(),
        )
        return _json_error(
            status=409,
            error="promotion_gate_required",
            detail="adapter_candidate artifacts require gate_evidence",
            extra=blocked.to_json(),
        )
    promoted = ArtifactRecord(
        artifact_id=artifact.artifact_id,
        artifact_kind=artifact.artifact_kind,
        ai_id=artifact.ai_id,
        contract_id=artifact.contract_id,
        source_ref=artifact.source_ref,
        status="promoted",
        metadata={**dict(artifact.metadata), "gate_evidence": gate_evidence},
        promotion_decision=PromotionDecision.ALLOW,
        created_at_ms=artifact.created_at_ms,
    )
    _artifact_store(request)[artifact_id] = promoted
    _persist_governance(
        request,
        "artifact",
        artifact_id,
        promoted.to_json(),
        ai_id=promoted.ai_id,
        contract_id=promoted.contract_id,
    )
    _record_audit(
        request,
        event_type="artifact_promoted",
        ai_id=promoted.ai_id,
        contract_id=promoted.contract_id,
        payload=promoted.to_json(),
    )
    return web.json_response({"status": "promoted", **promoted.to_json()})


async def _handle_eval_run_create(request: web.Request) -> web.Response:
    data = await _read_json_or_error(request)
    if isinstance(data, web.Response):
        return data
    gate_id = str(data.get("gate_id", "") or "")
    if not gate_id:
        return _json_error(status=400, error="missing_gate_id", detail="gate_id is required")
    run = EvalRun(
        run_id=_new_id("eval"),
        gate_id=gate_id,
        ai_id=str(data.get("ai_id", "") or ""),
        contract_id=str(data.get("contract_id", "") or ""),
        score=float(data.get("score", 1.0) or 1.0),
        created_at_ms=_now_ms(),
    )
    _eval_store(request)[run.run_id] = run
    _persist_governance(request, "eval_run", run.run_id, run.to_json(), ai_id=run.ai_id, contract_id=run.contract_id)
    _record_audit(
        request,
        event_type="eval_run_created",
        ai_id=run.ai_id,
        contract_id=run.contract_id,
        payload=run.to_json(),
    )
    return web.json_response({"status": "ok", **run.to_json()}, status=201)


async def _handle_eval_run_get(request: web.Request) -> web.Response:
    run_id = request.match_info.get("run_id", "")
    run = _eval_store(request).get(run_id)
    if run is None:
        persisted = _get_persisted_governance(request, "eval_run", run_id)
        if persisted is not None:
            return web.json_response({"status": "ok", **persisted})
    if run is None:
        return _json_error(status=404, error="eval_run_not_found", detail=run_id)
    return web.json_response({"status": "ok", **run.to_json()})


async def _handle_eval_run_approve(request: web.Request) -> web.Response:
    run_id = request.match_info.get("run_id", "")
    run = _eval_store(request).get(run_id)
    if run is None:
        return _json_error(status=404, error="eval_run_not_found", detail=run_id)
    approved = EvalRun(
        run_id=run.run_id,
        gate_id=run.gate_id,
        ai_id=run.ai_id,
        contract_id=run.contract_id,
        status=EvalRunStatus.APPROVED,
        score=run.score,
        decision=EvalGateDecision(PromotionDecision.ALLOW),
        created_at_ms=run.created_at_ms,
    )
    _eval_store(request)[run_id] = approved
    _persist_governance(
        request,
        "eval_run",
        run_id,
        approved.to_json(),
        ai_id=approved.ai_id,
        contract_id=approved.contract_id,
    )
    _record_audit(
        request,
        event_type="eval_run_approved",
        ai_id=approved.ai_id,
        contract_id=approved.contract_id,
        payload=approved.to_json(),
    )
    return web.json_response({"status": "approved", **approved.to_json()})


async def _handle_webhook_create(request: web.Request) -> web.Response:
    data = await _read_json_or_error(request)
    if isinstance(data, web.Response):
        return data
    target_url = str(data.get("target_url", "") or "")
    if not target_url:
        return _json_error(status=400, error="missing_target_url", detail="target_url is required")
    webhook = WebhookSubscription(
        webhook_id=_new_id("webhook"),
        target_url=target_url,
        event_types=tuple(str(e) for e in (data.get("event_types") or ())),
        secret_ref=str(data.get("secret_ref", "") or ""),
        created_at_ms=_now_ms(),
    )
    _webhook_store(request)[webhook.webhook_id] = webhook
    _persist_governance(request, "webhook", webhook.webhook_id, webhook.to_json())
    _record_audit(
        request,
        event_type="webhook_created",
        payload=webhook.to_json(),
    )
    return web.json_response({"status": "ok", **webhook.to_json()}, status=201)


async def _handle_events_stream(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "events": _list_persisted_governance(request, "event_stream")
            or [e.to_json() for e in _event_stream(request)],
        }
    )


async def _handle_usage(request: web.Request) -> web.Response:
    persisted_usage = _list_persisted_governance(request, "usage_record")
    if persisted_usage:
        ai_id = request.query.get("ai_id", "")
        if ai_id:
            persisted_usage = [r for r in persisted_usage if r.get("ai_id") == ai_id]
        return web.json_response({"status": "ok", "usage": persisted_usage})
    records = _usage_records(request)
    ai_id = request.query.get("ai_id", "")
    if ai_id:
        records = [r for r in records if r.ai_id == ai_id]
    return web.json_response(
        {"status": "ok", "usage": [r.to_json() for r in records]}
    )


async def _handle_quota(request: web.Request) -> web.Response:
    tenant_id = request.query.get("tenant_id", "default")
    usage_count = len(_usage_records(request))
    quota = QuotaSnapshot(
        tenant_id=tenant_id,
        limits={"interactions": 100000, "training_jobs": 1000},
        usage={"records": usage_count},
    )
    return web.json_response({"status": "ok", **quota.to_json()})


async def _handle_billing_events(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "billing_events": _list_persisted_governance(request, "billing_event")
            or [e.to_json() for e in _billing_events(request)],
        }
    )


async def _handle_policies(request: web.Request) -> web.Response:
    return web.json_response(
        {"status": "ok", "policies": [p.to_json() for p in _policy_store(request).values()]}
    )


async def _handle_consent_create(request: web.Request) -> web.Response:
    data = await _read_json_or_error(request)
    if isinstance(data, web.Response):
        return data
    ai_id = str(data.get("ai_id", "") or "")
    end_user_ref = str(data.get("end_user_ref", "") or "")
    consent_type = str(data.get("consent_type", "") or "")
    if not ai_id or not end_user_ref or not consent_type:
        return _json_error(
            status=400,
            error="missing_consent_fields",
            detail="ai_id, end_user_ref and consent_type are required",
        )
    consent = ConsentRecord(
        consent_id=_new_id("consent"),
        ai_id=ai_id,
        end_user_ref=end_user_ref,
        consent_type=consent_type,
        granted=bool(data.get("granted", True)),
        evidence_ref=str(data.get("evidence_ref", "") or ""),
        created_at_ms=_now_ms(),
    )
    _consent_store(request)[(ai_id, end_user_ref, consent_type)] = consent
    _persist_governance(request, "consent", consent.consent_id, consent.to_json(), ai_id=ai_id)
    _record_audit(
        request,
        event_type="consent_recorded",
        ai_id=ai_id,
        payload=consent.to_json(),
    )
    return web.json_response({"status": "ok", **consent.to_json()}, status=201)


async def _handle_consent_get(request: web.Request) -> web.Response:
    end_user_ref = request.match_info.get("end_user_ref", "")
    ai_id = request.query.get("ai_id", "")
    consents = [
        c
        for (stored_ai, stored_user, _), c in _consent_store(request).items()
        if stored_user == end_user_ref and (not ai_id or stored_ai == ai_id)
    ]
    persisted = [
        c
        for c in _list_persisted_governance(request, "consent")
        if c.get("end_user_ref") == end_user_ref and (not ai_id or c.get("ai_id") == ai_id)
    ]
    if persisted:
        return web.json_response(
            {"status": "ok", "end_user_ref": end_user_ref, "consents": persisted}
        )
    return web.json_response(
        {"status": "ok", "end_user_ref": end_user_ref, "consents": [c.to_json() for c in consents]}
    )


async def _handle_catalog_blueprints(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "blueprints": [blueprint.to_json() for blueprint in _catalog_blueprints()],
        }
    )


async def _handle_catalog_verticals(request: web.Request) -> web.Response:
    from lifeform_service.verticals import discover_verticals

    verticals = discover_verticals()
    return web.json_response(
        {
            "status": "ok",
            "verticals": [
                {
                    "vertical_id": name,
                    "runtime_template_id": name,
                    "alpha_supported": spec.alpha_factory is not None,
                    "has_temporal_bootstrap": spec.has_temporal_bootstrap,
                    "has_regime_bootstrap": spec.has_regime_bootstrap,
                }
                for name, spec in sorted(verticals.items())
            ],
        }
    )


async def _handle_catalog_substrate_profiles(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "substrate_profiles": [
                {
                    "substrate_profile_id": "shared-frozen",
                    "mode": "shared_frozen",
                    "adapter_policy": "none",
                    "allow_rare_heavy_refresh": False,
                },
                {
                    "substrate_profile_id": "synthetic-dev",
                    "mode": "synthetic",
                    "adapter_policy": "none",
                    "allow_rare_heavy_refresh": False,
                },
            ],
        }
    )


async def _handle_catalog_protocols(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "protocols": [
                {
                    "protocol_id": "growth_advisor:cheng-laoshi",
                    "vertical_id": "growth_advisor",
                    "source": "fixture",
                    "review_status": "active",
                }
            ],
        }
    )


async def _handle_catalog_tool_policies(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "tool_policies": [
                {
                    "tool_policy_id": "default-text-only",
                    "allowed_capabilities": ["text"],
                },
                {
                    "tool_policy_id": "growth-advisor-wechat-readonly",
                    "allowed_capabilities": [
                        "text",
                        "handoff_ticket",
                        "reviewed_knowledge",
                    ],
                },
            ],
        }
    )


async def _handle_catalog_training_policies(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "training_policies": [
                {
                    "training_policy_id": "reviewed_protocol_only",
                    "allow_protocol_intake": True,
                    "allow_corpus_intake": True,
                    "allow_adapter_training": False,
                }
            ],
        }
    )


async def _handle_readouts(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    session_or_response = await _session_from_query(request, ai_id)
    if isinstance(session_or_response, web.Response):
        return session_or_response
    view_raw = request.query.get("view", "summary")
    try:
        view = ReadoutView(view_raw)
    except ValueError:
        return _json_error(
            status=400,
            error="invalid_readout_view",
            detail="view must be 'summary' or 'full'",
        )
    session = session_or_response
    snapshots = session.latest_active_snapshots
    bundle = _build_readout_bundle(
        ai_id=ai_id,
        session_id=session.session_id,
        view=view,
        snapshots=snapshots,
        request=request,
    )
    return web.json_response({"status": "ok", **bundle.to_json()})


async def _handle_admin_snapshots(request: web.Request) -> web.Response:
    if not _has_admin_auth(request):
        return _json_error(
            status=403,
            error="admin_auth_required",
            detail="Raw snapshot export requires control-plane or service auth.",
        )
    ai_id = request.match_info.get("ai_id", "")
    session_or_response = await _session_from_query(request, ai_id)
    if isinstance(session_or_response, web.Response):
        return session_or_response
    slots = tuple(request.query.getall("slot", []))
    include_shadow = request.query.get("include_shadow", "").lower() in {
        "1",
        "true",
        "yes",
    }
    try:
        export_request = SnapshotExportRequest.from_query(
            ai_id=ai_id,
            session_id=session_or_response.session_id,
            slots=slots,
            include_shadow=include_shadow,
        )
    except ValueError as exc:
        return _json_error(status=400, error="invalid_snapshot_request", detail=str(exc))
    active = session_or_response.latest_active_snapshots
    shadow = session_or_response.latest_shadow_snapshots if include_shadow else {}
    selected = export_request.slots or tuple(sorted(active))
    _record_audit(
        request,
        event_type="raw_snapshots_exported",
        ai_id=ai_id,
        session_id=session_or_response.session_id,
        payload=export_request.to_json(),
    )
    return web.json_response(
        {
            "status": "ok",
            **export_request.to_json(),
            "active": {
                slot: snapshot_to_json(active[slot], redact=True)
                for slot in selected
                if slot in active
            },
            "shadow": {
                slot: snapshot_to_json(shadow[slot], redact=True)
                for slot in selected
                if slot in shadow
            },
        }
    )


async def _handle_explain(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    session_or_response = await _session_from_query(request, ai_id)
    if isinstance(session_or_response, web.Response):
        return session_or_response
    turn_index = request.query.get("turn_index", "latest")
    snapshots = session_or_response.latest_active_snapshots
    trace = ExplainTrace(
        ai_id=ai_id,
        session_id=session_or_response.session_id,
        turn_index=turn_index,
        chain=_build_explain_chain(snapshots),
    )
    return web.json_response({"status": "ok", **trace.to_json()})


async def _handle_list_instances(request: web.Request) -> web.Response:
    launcher = request.app.get(INSTANCE_MANAGER_APP_KEY)
    if not isinstance(launcher, InstanceManager):
        return _json_error(
            status=503,
            error="launcher_not_configured",
            detail="Instance lifecycle routes require a DLaaS InstanceManager.",
        )
    return web.json_response(
        {"status": "ok", "instances": list(launcher.overview())}
    )


async def _handle_instance_status(request: web.Request) -> web.Response:
    launcher = request.app.get(INSTANCE_MANAGER_APP_KEY)
    if not isinstance(launcher, InstanceManager):
        return _json_error(
            status=503,
            error="launcher_not_configured",
            detail="Instance lifecycle routes require a DLaaS InstanceManager.",
        )
    ai_id = request.match_info.get("ai_id", "")
    try:
        status = launcher.status(ai_id)
    except InstanceNotFound:
        return _json_error(
            status=404,
            error="ai_id_not_found",
            detail=f"ai_id={ai_id!r} is not adopted on this server.",
        )
    return web.json_response({"status": "ok", **status.to_json()})


async def _handle_wake_instance(request: web.Request) -> web.Response:
    launcher = request.app.get(INSTANCE_MANAGER_APP_KEY)
    if not isinstance(launcher, InstanceManager):
        return _json_error(
            status=503,
            error="launcher_not_configured",
            detail="Instance lifecycle routes require a DLaaS InstanceManager.",
        )
    ai_id = request.match_info.get("ai_id", "")
    try:
        payload = await _read_json_object(request, allow_empty=True)
        wake = WakeRequest.from_json(payload)
        status = await launcher.wake(
            ai_id=ai_id,
            runtime_template_id=wake.runtime_template_id,
            reason=wake.reason,
        )
    except InstanceNotFound:
        return _json_error(
            status=404,
            error="ai_id_not_found",
            detail=(
                f"ai_id={ai_id!r} is not adopted; provide runtime_template_id "
                "or adopt the instance first."
            ),
        )
    except (LookupError, ValueError) as exc:
        return _json_error(status=400, error="invalid_wake_request", detail=str(exc))

    # U5 (family-memorial enabler): when the wake request names a
    # ``template_id``, resolve the template's ``figure_artifact_id`` and
    # bind the bundle to ``ai_id``'s SessionManager — same contract as
    # the adopt path (U2) but available without the full tenant
    # onboarding overhead. The bake-worker uses this so a freshly-baked
    # per-memorial bundle is bound to its memorial's ai_id immediately
    # after wake, with no second HTTP round-trip.
    #
    # Empty ``template_id`` keeps legacy wake-without-binding behaviour
    # (used by ``bootstrap-einstein`` and other operator scripts that
    # rely on the vertical factory to attach a global bundle).
    if wake.template_id:
        stores = request.app.get(CONTROL_PLANE_STORES_KEY)
        if stores is None:
            return _json_error(
                status=503,
                error="control_plane_unavailable",
                detail=(
                    "wake.template_id is set but the control plane stores "
                    "are not attached to this dlaas-platform; reach out "
                    "to platform operator."
                ),
            )
        try:
            template = await stores.templates.get(wake.template_id)
        except TemplateNotFound:
            return _json_error(
                status=404,
                error="template_not_found",
                detail=(
                    f"wake.template_id={wake.template_id!r} does not "
                    "resolve to a registered template; the bake-worker "
                    "must POST /dlaas/control/templates before /wake."
                ),
            )
        bind_result = bind_figure_artifact_to_ai_id(
            launcher, ai_id, template.figure_artifact_id
        )
        if not bind_result.bound and bind_result.reason in (
            "bundle_not_found",
            "instance_not_found",
            "lifeform_service_absent",
        ):
            return _json_error(
                status=503,
                error=_BIND_REASON_TO_ERROR_CODE[bind_result.reason],
                detail=_bind_failure_detail(
                    template.figure_artifact_id, ai_id, bind_result.reason
                ),
            )
    return web.json_response({"status": "ok", **status.to_json()})


async def _handle_sleep_instance(request: web.Request) -> web.Response:
    launcher = request.app.get(INSTANCE_MANAGER_APP_KEY)
    if not isinstance(launcher, InstanceManager):
        return _json_error(
            status=503,
            error="launcher_not_configured",
            detail="Instance lifecycle routes require a DLaaS InstanceManager.",
        )
    ai_id = request.match_info.get("ai_id", "")
    try:
        payload = await _read_json_object(request, allow_empty=True)
        sleep = SleepRequest.from_json(payload)
        status = await launcher.sleep(
            ai_id=ai_id,
            reason=sleep.reason,
            release_instance=sleep.release_instance,
        )
    except InstanceNotFound:
        return _json_error(
            status=404,
            error="ai_id_not_found",
            detail=f"ai_id={ai_id!r} is not adopted on this server.",
        )
    except ValueError as exc:
        return _json_error(status=400, error="invalid_sleep_request", detail=str(exc))
    return web.json_response({"status": "ok", **status.to_json()})


async def _handle_feedback_alias(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    try:
        data = await _read_json_object(request)
        envelope = InteractionEnvelope.from_json(
            {
                "contract_id": data.get("contract_id", ""),
                "session_id": data.get("session_id", ""),
                "end_user_ref": data.get("end_user_ref", ""),
                "interaction_type": InteractionType.FEEDBACK.value,
                "human_brief": data.get("human_brief", ""),
                "feedback": data.get("feedback") or data,
                "structured_context": data.get("structured_context") or {},
                "lang": data.get("lang", "cn"),
            }
        )
    except (ValueError, _EnvelopeError) as exc:
        return _json_error(status=400, error="invalid_feedback", detail=str(exc))
    return await _dispatch_envelope_to_instance(request, ai_id, envelope)


async def _handle_environment_event_alias(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    try:
        data = await _read_json_object(request)
        payload = data.get("payload") or {}
        if not isinstance(payload, dict):
            return _json_error(
                status=400,
                error="invalid_payload",
                detail="environment event payload must be an object",
            )
        ctx = {
            **payload,
            "observation_type": data.get("observation_type", ""),
            "event_id": data.get("event_id", ""),
        }
        envelope = InteractionEnvelope.from_json(
            {
                "contract_id": data.get("contract_id", ""),
                "session_id": data.get("session_id", ""),
                "end_user_ref": data.get("end_user_ref", ""),
                "interaction_type": InteractionType.OBSERVE.value,
                "human_brief": data.get("summary", ""),
                "structured_context": ctx,
                "target_person_ids": data.get("target_person_ids", ()),
                "lang": data.get("lang", "cn"),
            }
        )
    except (ValueError, _EnvelopeError) as exc:
        return _json_error(status=400, error="invalid_environment_event", detail=str(exc))
    return await _dispatch_envelope_to_instance(request, ai_id, envelope)


async def _handle_environment_outcome_alias(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    try:
        data = await _read_json_object(request)
        ctx = {
            "observation_type": "tool_result",
            "event_id": data.get("event_id", data.get("outcome_id", "")),
            "tool_name": data.get("tool_name", ""),
            "action_id": data.get("action_id", ""),
            "status": data.get("status", ""),
            "summary": data.get("summary", ""),
            "detail": data.get("detail", ""),
            "confidence": data.get("confidence", 0.8),
        }
        envelope = InteractionEnvelope.from_json(
            {
                "contract_id": data.get("contract_id", ""),
                "session_id": data.get("session_id", ""),
                "end_user_ref": data.get("end_user_ref", ""),
                "interaction_type": InteractionType.OBSERVE.value,
                "human_brief": data.get("summary", ""),
                "structured_context": ctx,
                "lang": data.get("lang", "cn"),
            }
        )
    except (ValueError, _EnvelopeError) as exc:
        return _json_error(
            status=400, error="invalid_environment_outcome", detail=str(exc)
        )
    return await _dispatch_envelope_to_instance(request, ai_id, envelope)


async def _handle_protocol_submission_create(request: web.Request) -> web.Response:
    try:
        data = await _read_json_object(request)
    except ValueError as exc:
        return _json_error(status=400, error="invalid_protocol_submission", detail=str(exc))
    return await _create_protocol_submission_from_data(
        request,
        data,
        pending_status="pending_review",
    )


async def _handle_safety_protocol_create(request: web.Request) -> web.Response:
    try:
        data = await _read_json_object(request)
    except ValueError as exc:
        return _json_error(status=400, error="invalid_safety_protocol", detail=str(exc))
    if not _payload_has_boundary_contract(data):
        return _json_error(
            status=400,
            error="missing_boundary_contracts",
            detail=(
                "Safety protocol submissions must include boundary_contracts "
                "or protocol.boundaries."
            ),
        )
    data.setdefault("source_type", "json_payload")
    return await _create_protocol_submission_from_data(
        request,
        data,
        pending_status="pending_safety_review",
    )


async def _handle_safety_protocol_list(request: web.Request) -> web.Response:
    return await _handle_protocol_submission_list(request)


async def _handle_safety_protocol_approve(request: web.Request) -> web.Response:
    return await _set_protocol_submission_status(
        request, review_status="active", response_status="approved"
    )


async def _handle_safety_protocol_load(request: web.Request) -> web.Response:
    return await _protocol_library_mark(request, loaded=True)


async def _create_protocol_submission_from_data(
    request: web.Request,
    data: dict[str, Any],
    *,
    pending_status: str,
) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    try:
        submission_id = f"prot_sub_{uuid.uuid4().hex[:12]}"
        submission = ProtocolSubmission.from_json(
            data, ai_id=ai_id, submission_id=submission_id
        )
    except ValueError as exc:
        return _json_error(status=400, error="invalid_protocol_submission", detail=str(exc))
    store = _protocol_store(request)
    store[(ai_id, submission_id)] = submission
    _record_audit(
        request,
        event_type="protocol_submission_created",
        ai_id=ai_id,
        contract_id=submission.contract_id,
        payload=submission.to_json(),
    )
    _record_usage(request, ai_id=ai_id, metric="protocol_submission", quantity=1)
    return web.json_response(
        {
            "status": pending_status,
            **submission.to_json(),
            "requires_review": True,
        },
        status=201,
    )


async def _handle_protocol_submission_list(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    rows = [
        submission.to_json()
        for (stored_ai_id, _), submission in _protocol_store(request).items()
        if stored_ai_id == ai_id
    ]
    return web.json_response(
        {"status": "ok", "ai_id": ai_id, "submissions": rows, "count": len(rows)}
    )


async def _handle_protocol_submission_approve(request: web.Request) -> web.Response:
    return await _set_protocol_submission_status(
        request, review_status="active", response_status="approved"
    )


async def _handle_protocol_submission_reject(request: web.Request) -> web.Response:
    return await _set_protocol_submission_status(
        request, review_status="rejected", response_status="rejected"
    )


async def _set_protocol_submission_status(
    request: web.Request,
    *,
    review_status: str,
    response_status: str,
) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    submission_id = request.match_info.get("submission_id", "")
    store = _protocol_store(request)
    existing = store.get((ai_id, submission_id))
    if existing is None:
        return _json_error(
            status=404,
            error="protocol_submission_not_found",
            detail=submission_id,
        )
    updated = ProtocolSubmission(
        submission_id=existing.submission_id,
        ai_id=existing.ai_id,
        contract_id=existing.contract_id,
        source_type=existing.source_type,
        submitted_by=existing.submitted_by,
        source_ref=existing.source_ref,
        target_vertical=existing.target_vertical,
        notes=existing.notes,
        review_level_requested=existing.review_level_requested,
        candidate_protocol_id=existing.candidate_protocol_id
        or f"{existing.ai_id}:{existing.submission_id}",
        review_status=review_status,
    )
    store[(ai_id, submission_id)] = updated
    _record_audit(
        request,
        event_type=f"protocol_submission_{response_status}",
        ai_id=ai_id,
        contract_id=updated.contract_id,
        payload=updated.to_json(),
    )
    return web.json_response({"status": response_status, **updated.to_json()})


async def _handle_protocol_library_list(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    entries = [
        submission.to_json()
        for (stored_ai_id, _), submission in _protocol_store(request).items()
        if stored_ai_id == ai_id and submission.review_status == "active"
    ]
    return web.json_response(
        {"status": "ok", "ai_id": ai_id, "entries": entries, "count": len(entries)}
    )


async def _handle_protocol_library_load(request: web.Request) -> web.Response:
    return await _protocol_library_mark(request, loaded=True)


async def _handle_protocol_library_unload(request: web.Request) -> web.Response:
    return await _protocol_library_mark(request, loaded=False)


async def _protocol_library_mark(
    request: web.Request,
    *,
    loaded: bool,
) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    protocol_id = request.match_info.get("protocol_id", "")
    matches = [
        submission
        for (stored_ai_id, _), submission in _protocol_store(request).items()
        if stored_ai_id == ai_id
        and submission.review_status == "active"
        and (
            submission.candidate_protocol_id == protocol_id
            or f"{submission.ai_id}:{submission.submission_id}" == protocol_id
        )
    ]
    if not matches:
        return _json_error(
            status=404, error="protocol_not_found", detail=protocol_id
        )
    _record_audit(
        request,
        event_type="protocol_loaded" if loaded else "protocol_unloaded",
        ai_id=ai_id,
        payload={"protocol_id": protocol_id, "loaded": loaded},
    )
    return web.json_response(
        {
            "status": "ok",
            "ai_id": ai_id,
            "protocol_id": protocol_id,
            "loaded": loaded,
            "shadow_mode": True,
            "detail": (
                "Protocol library load/unload recorded at platform layer. "
                "Runtime hot-load is delegated to ProtocolUptakeService when wired."
            ),
        }
    )


async def _handle_training_corpus(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    try:
        data = await _read_json_object(request)
        envelope = InteractionEnvelope.from_json(
            {
                "contract_id": data.get("contract_id", ""),
                "session_id": data.get("session_id", f"training-{ai_id}"),
                "end_user_ref": data.get("end_user_ref", "training"),
                "interaction_type": InteractionType.OBSERVE.value,
                "human_brief": data.get("notes", ""),
                "structured_context": {
                    "observation_type": "corpus_ingest",
                    "source_uri": data.get("source_ref", "dlaas-training-corpus"),
                    "corpus_text": data.get("text", data.get("notes", "")),
                },
            }
        )
    except (ValueError, _EnvelopeError) as exc:
        return _json_error(status=400, error="invalid_training_corpus", detail=str(exc))
    return await _dispatch_envelope_to_instance(request, ai_id, envelope)


async def _handle_asset_intake(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    try:
        data = await _read_json_object(request)
        intake_request = AssetIntakeRequest.from_json(data, ai_id=ai_id)
        manager = _resolve_session_manager(request, ai_id)
        decision = resolve_intake_decision(
            intake_request,
            provider=manager.substrate_runtime
            if intake_request.intent is AssetIntakeIntent.AUTO
            else None,
        )
    except (ValueError, _AiIdNotFoundError) as exc:
        code = exc.code if isinstance(exc, _AiIdNotFoundError) else "invalid_asset_intake"
        detail = exc.detail if isinstance(exc, _AiIdNotFoundError) else str(exc)
        status = 404 if isinstance(exc, _AiIdNotFoundError) else 400
        return _json_error(status=status, error=code, detail=detail)

    asset_id = f"asset_{uuid.uuid4().hex[:12]}"
    asset = _asset_record_from_intake(
        asset_id=asset_id,
        intake=intake_request,
        resolved_intent=decision.intent,
    )
    _asset_intake_store(request)[(ai_id, asset_id)] = asset
    _persist_governance(
        request,
        "asset_intake",
        asset_id,
        asset,
        ai_id=ai_id,
        contract_id=intake_request.contract_id,
    )

    if decision.intent is AssetIntakeIntent.STORAGE_ONLY:
        body = _asset_intake_response(asset=asset, decision_rationale=decision.rationale)
        _record_audit(
            request,
            event_type="asset_intake_stored",
            ai_id=ai_id,
            contract_id=intake_request.contract_id,
            payload=body,
        )
        return web.json_response(body, status=201)

    if decision.intent is AssetIntakeIntent.IMAGE_INTAKE:
        body = _asset_intake_response(
            asset=asset,
            decision_rationale=decision.rationale,
            extra={"image_status": "stored_pending_vision_extractor"},
        )
        _record_audit(
            request,
            event_type="asset_image_stored",
            ai_id=ai_id,
            contract_id=intake_request.contract_id,
            payload=body,
        )
        return web.json_response(body, status=202)

    if decision.intent is AssetIntakeIntent.SIMPLE_INGEST:
        envelope_or_error = _build_ingestion_envelope(asset_id, intake_request)
        if isinstance(envelope_or_error, web.Response):
            return envelope_or_error
        try:
            session = await _get_or_create_session(
                _resolve_session_manager(request, ai_id),
                intake_request.session_id,
                user_id=intake_request.end_user_ref,
            )
            from lifeform_ingestion import IngestionPipeline

            report = await IngestionPipeline().process_envelope(
                envelope_or_error, session=session
            )
        except Exception as exc:  # noqa: BLE001 - platform/kernel boundary
            return _json_error(
                status=500,
                error="asset_ingestion_failed",
                detail=str(exc),
            )
        report_json = _ingestion_report_to_json(report)
        body = _asset_intake_response(
            asset=asset,
            decision_rationale=decision.rationale,
            extra={"ingestion_report": report_json},
        )
        _record_audit(
            request,
            event_type="asset_simple_ingested",
            ai_id=ai_id,
            contract_id=intake_request.contract_id,
            payload=body,
        )
        return web.json_response(body, status=201)

    job_type = (
        "adapter_candidate"
        if decision.intent is AssetIntakeIntent.TRAINING_CANDIDATE
        else "corpus_ingestion"
    )
    job_data = {
        "contract_id": intake_request.contract_id,
        "job_type": job_type,
        "created_by": intake_request.end_user_ref,
        "source_ref": asset["asset"]["asset_id"],
        "promotion_gate": "offline_gate_required"
        if job_type == "adapter_candidate"
        else "reviewed_corpus_ingestion",
        "notes": f"asset_intake:{decision.intent.value}:{intake_request.title}",
    }
    job = TrainingJob.from_json(
        job_data, ai_id=ai_id, job_id=f"train_job_{uuid.uuid4().hex[:12]}"
    )
    status = (
        TrainingJobStatus.PENDING
        if decision.intent is AssetIntakeIntent.TRAINING_CANDIDATE
        else TrainingJobStatus.RUNNING
    )
    job = job.with_status(status)
    _training_store(request)[(ai_id, job.job_id)] = job
    body = _asset_intake_response(
        asset=asset,
        decision_rationale=decision.rationale,
        extra={"training_job": job.to_json()},
    )
    _record_audit(
        request,
        event_type="asset_training_job_created",
        ai_id=ai_id,
        contract_id=intake_request.contract_id,
        payload=body,
    )
    return web.json_response(body, status=202)


async def _handle_asset_intake_get(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    asset_id = request.match_info.get("asset_id", "")
    asset = _asset_intake_store(request).get((ai_id, asset_id))
    if asset is None:
        return _json_error(status=404, error="asset_intake_not_found", detail=asset_id)
    return web.json_response({"status": "ok", **asset})


def _asset_record_from_intake(
    *,
    asset_id: str,
    intake: AssetIntakeRequest,
    resolved_intent: AssetIntakeIntent,
) -> dict[str, Any]:
    raw_bytes = _intake_bytes(intake)
    integrity_hash = hashlib.sha256(raw_bytes).hexdigest() if raw_bytes else ""
    asset = {
        "asset_id": asset_id,
        "ai_id": intake.ai_id,
        "contract_id": intake.contract_id,
        "session_id": intake.session_id,
        "end_user_ref": intake.end_user_ref,
        "title": intake.title,
        "source_ref": intake.source_ref,
        "media_kind": intake.media_kind.value,
        "mime_type": intake.mime_type,
        "resolved_intent": resolved_intent.value,
        "integrity_hash": integrity_hash,
        "metadata": dict(intake.metadata),
        "created_at_ms": _now_ms(),
    }
    return {"asset": asset}


def _asset_intake_response(
    *,
    asset: dict[str, Any],
    decision_rationale: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "status": "ok",
        **asset,
        "decision": {
            "intent": asset["asset"]["resolved_intent"],
            "rationale": decision_rationale,
        },
    }
    if extra:
        body.update(extra)
    return body


def _build_ingestion_envelope(
    asset_id: str,
    intake: AssetIntakeRequest,
) -> Any | web.Response:
    source_uri = intake.source_ref or f"dlaas-asset:{asset_id}"
    try:
        from lifeform_ingestion import (
            IngestionSourceKind,
            envelope_from_docx_bytes,
            envelope_from_pdf_bytes,
            envelope_from_text,
        )

        if intake.media_kind in {AssetMediaKind.TEXT, AssetMediaKind.MARKDOWN, AssetMediaKind.JSON}:
            text = intake.text or _intake_bytes(intake).decode("utf-8")
            return envelope_from_text(
                text,
                source_uri=source_uri,
                uploader=intake.end_user_ref,
                envelope_id=f"asset:{asset_id}",
                source_kind=(
                    IngestionSourceKind.BOOK
                    if intake.media_kind is AssetMediaKind.MARKDOWN
                    else IngestionSourceKind.CORPUS
                ),
            )
        if intake.media_kind is AssetMediaKind.PDF:
            return envelope_from_pdf_bytes(
                _intake_bytes_required(intake),
                source_uri=source_uri,
                uploader=intake.end_user_ref,
                envelope_id=f"asset:{asset_id}",
            )
        if intake.media_kind is AssetMediaKind.DOCX:
            return envelope_from_docx_bytes(
                _intake_bytes_required(intake),
                source_uri=source_uri,
                uploader=intake.end_user_ref,
                envelope_id=f"asset:{asset_id}",
            )
    except Exception as exc:  # noqa: BLE001 - parser boundary surfaces typed error
        return _json_error(
            status=400,
            error="asset_parse_failed",
            detail=str(exc),
        )
    return _json_error(
        status=400,
        error="unsupported_asset_ingestion",
        detail=(
            f"media_kind={intake.media_kind.value!r} cannot be ingested directly; "
            "store it or route it to image_intake/training_candidate."
        ),
    )


def _intake_bytes_required(intake: AssetIntakeRequest) -> bytes:
    raw = _intake_bytes(intake)
    if not raw:
        raise ValueError("content_base64 is required for binary asset ingestion")
    return raw


def _intake_bytes(intake: AssetIntakeRequest) -> bytes:
    if intake.content_base64.strip():
        try:
            return base64.b64decode(intake.content_base64, validate=True)
        except ValueError as exc:
            raise ValueError("content_base64 must be valid base64") from exc
    if intake.text:
        return intake.text.encode("utf-8")
    return b""


def _ingestion_report_to_json(report: Any) -> dict[str, Any]:
    return {
        "envelope_id": report.envelope_id,
        "total_chunks": report.total_chunks,
        "processed_chunks": report.processed_chunks,
        "skipped_chunks": report.skipped_chunks,
        "ended_scene": report.ended_scene,
        "all_succeeded": report.all_succeeded,
        "turns": [
            {
                "chunk_id": turn.chunk_id,
                "locator": turn.locator,
                "turn_succeeded": turn.turn_succeeded,
                "skipped_reason": turn.skipped_reason,
            }
            for turn in report.turns
        ],
    }


async def _handle_training_job_create(request: web.Request) -> web.Response:
    ai_id = request.match_info.get("ai_id", "")
    try:
        data = await _read_json_object(request)
        job_id = f"train_job_{uuid.uuid4().hex[:12]}"
        job = TrainingJob.from_json(data, ai_id=ai_id, job_id=job_id)
    except ValueError as exc:
        return _json_error(status=400, error="invalid_training_job", detail=str(exc))
    _training_store(request)[(ai_id, job.job_id)] = job
    artifact_kind = (
        ArtifactKind.ADAPTER_CANDIDATE
        if job.job_type.value == "adapter_candidate"
        else ArtifactKind.TRAINING_OUTPUT
    )
    artifact = ArtifactRecord(
        artifact_id=f"artifact:{job.job_id}",
        artifact_kind=artifact_kind,
        ai_id=ai_id,
        contract_id=job.contract_id,
        source_ref=job.source_ref or job.job_id,
        status="created",
        metadata={"job_id": job.job_id, "job_type": job.job_type.value},
        created_at_ms=_now_ms(),
    )
    _artifact_store(request)[artifact.artifact_id] = artifact
    _persist_governance(
        request,
        "artifact",
        artifact.artifact_id,
        artifact.to_json(),
        ai_id=ai_id,
        contract_id=job.contract_id,
    )
    _record_audit(
        request,
        event_type="training_job_created",
        ai_id=ai_id,
        contract_id=job.contract_id,
        payload=job.to_json(),
    )
    _record_usage(request, ai_id=ai_id, metric="training_job", quantity=1)
    return web.json_response({"status": "ok", **job.to_json()}, status=201)


async def _handle_training_job_get(request: web.Request) -> web.Response:
    job = _get_training_job(request)
    if isinstance(job, web.Response):
        return job
    return web.json_response({"status": "ok", **job.to_json()})


async def _handle_training_job_cancel(request: web.Request) -> web.Response:
    job = _get_training_job(request)
    if isinstance(job, web.Response):
        return job
    updated = job.with_status(TrainingJobStatus.CANCELLED)
    _training_store(request)[(job.ai_id, job.job_id)] = updated
    _record_audit(
        request,
        event_type="training_job_cancelled",
        ai_id=job.ai_id,
        contract_id=job.contract_id,
        payload=updated.to_json(),
    )
    return web.json_response({"status": "cancelled", **updated.to_json()})


async def _handle_training_job_promote(request: web.Request) -> web.Response:
    job = _get_training_job(request)
    if isinstance(job, web.Response):
        return job
    if job.job_type.value == "adapter_candidate" and not job.gate_evidence:
        updated = job.with_status(TrainingJobStatus.BLOCKED)
        _training_store(request)[(job.ai_id, job.job_id)] = updated
        _record_audit(
            request,
            event_type="training_job_promotion_blocked",
            ai_id=job.ai_id,
            contract_id=job.contract_id,
            payload=updated.to_json(),
        )
        return _json_error(
            status=409,
            error="promotion_gate_required",
            detail="adapter_candidate jobs require gate_evidence before promotion",
            extra=updated.to_json(),
        )
    updated = job.with_status(TrainingJobStatus.PROMOTED)
    _training_store(request)[(job.ai_id, job.job_id)] = updated
    _record_audit(
        request,
        event_type="training_job_promoted",
        ai_id=job.ai_id,
        contract_id=job.contract_id,
        payload=updated.to_json(),
    )
    return web.json_response({"status": "promoted", **updated.to_json()})


async def _handle_interaction(request: web.Request) -> web.Response:
    """Adapt the HTTP request to a typed dispatch call."""
    try:
        envelope = await _parse_envelope(request)
    except _EnvelopeError as exc:
        return _json_error(status=400, error=exc.code, detail=exc.detail)

    ai_id = request.match_info.get("ai_id", "")
    if not ai_id:
        return _json_error(
            status=400,
            error="invalid_ai_id",
            detail="ai_id path segment is required",
        )

    return await _dispatch_envelope_to_instance(request, ai_id, envelope)


async def _dispatch_envelope_to_instance(
    request: web.Request,
    ai_id: str,
    envelope: InteractionEnvelope,
) -> web.Response:
    ops_bundle = request.app.get(OPS_BUNDLE_APP_KEY)
    if isinstance(ops_bundle, OpsBundle):
        if await ops_bundle.pause_store.is_paused(
            ai_id=ai_id, session_id=envelope.session_id
        ):
            body = operator_takeover_response_body(
                ai_id=ai_id,
                session_id=envelope.session_id,
                contract_id=envelope.contract_id,
                interaction_type=envelope.interaction_type.value,
            )
            return web.json_response(dict(body))

    try:
        manager = _resolve_session_manager(request, ai_id)
    except _AiIdNotFoundError as exc:
        return _json_error(
            status=404, error=exc.code, detail=exc.detail
        )

    try:
        session = await _get_or_create_session(
            manager,
            envelope.session_id,
            user_id=envelope.end_user_ref,
        )
    except SessionAlreadyExistsError as exc:  # pragma: no cover - racy
        return _json_error(
            status=409, error="session_already_exists", detail=str(exc)
        )
    except SessionNotFoundError as exc:  # pragma: no cover - get-or-create
        return _json_error(
            status=404, error="session_not_found", detail=str(exc)
        )

    try:
        body = await dispatch_envelope(
            envelope=envelope, session=session, ai_id=ai_id
        )
    except DispatchError as exc:
        return _json_error(status=exc.status, error=exc.code, detail=exc.detail)
    _record_audit(
        request,
        event_type=f"interaction_{envelope.interaction_type.value}",
        ai_id=ai_id,
        contract_id=envelope.contract_id,
        session_id=envelope.session_id,
        payload={"interaction_type": envelope.interaction_type.value},
    )
    _record_usage(request, ai_id=ai_id, metric=f"interaction.{envelope.interaction_type.value}", quantity=1)
    return web.json_response(body)


def _resolve_session_manager(
    request: web.Request, ai_id: str
) -> SessionManager:
    """Pick the SessionManager for ``ai_id``.

    Resolution order:

    1. If the launcher is bound and knows ``ai_id`` → return the
       launcher's per-ai_id ``SessionManager``.
    2. Else if the launcher is bound but does NOT know ``ai_id`` →
       reject with 404 ``ai_id_not_found``. This is the multi-tenant
       path: every ai_id must be adopted before traffic flows.
    3. Else (no launcher) fall back to ``app["session_manager"]``
       (Slice 1 single-instance path).
    """
    launcher = request.app.get(INSTANCE_MANAGER_APP_KEY)
    if isinstance(launcher, InstanceManager):
        try:
            return launcher.get(ai_id)
        except InstanceNotFound as exc:
            raise _AiIdNotFoundError(
                code="ai_id_not_found",
                detail=(
                    f"ai_id={ai_id!r} is not adopted on this server. "
                    "Call POST /dlaas/adopt with a published, activated "
                    "template before sending interactions."
                ),
            ) from exc
    return request.app["session_manager"]


def _protocol_store(request: web.Request) -> dict[tuple[str, str], ProtocolSubmission]:
    return request.app[_PROTOCOL_SUBMISSIONS_KEY]


def _training_store(request: web.Request) -> dict[tuple[str, str], TrainingJob]:
    return request.app[_TRAINING_JOBS_KEY]


def _asset_intake_store(request: web.Request) -> dict[tuple[str, str], dict[str, Any]]:
    return request.app[_ASSET_INTAKES_KEY]


def _audit_store(request: web.Request) -> dict[str, AuditEvent]:
    return request.app[_AUDIT_EVENTS_KEY]


def _artifact_store(request: web.Request) -> dict[str, ArtifactRecord]:
    return request.app[_ARTIFACTS_KEY]


def _data_export_store(request: web.Request) -> dict[str, DataExportJob]:
    return request.app[_DATA_EXPORT_JOBS_KEY]


def _deletion_store(request: web.Request) -> dict[tuple[str, str], DeletionJob]:
    return request.app[_DELETION_JOBS_KEY]


def _eval_store(request: web.Request) -> dict[str, EvalRun]:
    return request.app[_EVAL_RUNS_KEY]


def _webhook_store(request: web.Request) -> dict[str, WebhookSubscription]:
    return request.app[_WEBHOOKS_KEY]


def _event_stream(request: web.Request) -> list[EventStreamRecord]:
    return request.app[_EVENT_STREAM_KEY]


def _usage_records(request: web.Request) -> list[UsageRecord]:
    return request.app[_USAGE_RECORDS_KEY]


def _billing_events(request: web.Request) -> list[BillingEvent]:
    return request.app[_BILLING_EVENTS_KEY]


def _consent_store(
    request: web.Request,
) -> dict[tuple[str, str, str], ConsentRecord]:
    return request.app[_CONSENTS_KEY]


def _policy_store(request: web.Request) -> dict[str, PolicySnapshot]:
    return request.app[_POLICIES_KEY]


def _debug_app_store(request: web.Request) -> dict[str, DebugAppRegistration]:
    return request.app[_DEBUG_APPS_KEY]


def _debug_schema_store(
    request: web.Request,
) -> dict[tuple[str, str], DebugSchema]:
    return request.app[_DEBUG_SCHEMAS_KEY]


def _debug_event_store(request: web.Request) -> dict[str, DebugEventEnvelope]:
    return request.app[_DEBUG_EVENTS_KEY]


def _debug_analysis_store(request: web.Request) -> dict[str, DebugAnalysisReport]:
    return request.app[_DEBUG_ANALYSES_KEY]


def _debug_schema_record_id(app_id: str, schema_version: str) -> str:
    return f"{app_id}:{schema_version}"


def _debug_app_exists(request: web.Request, app_id: str) -> bool:
    if app_id in _debug_app_store(request):
        return True
    return _get_persisted_governance(request, "debug_app", app_id) is not None


def _parse_debug_event(
    request: web.Request,
    data: dict[str, Any],
) -> DebugEventEnvelope | web.Response:
    app_id = str(data.get("app_id", "") or "").strip()
    schema_version = str(data.get("schema_version", "") or "").strip()
    event_type = str(data.get("event_type", "") or "").strip()
    stage = str(data.get("stage", "") or "").strip()
    fields = data.get("fields", {})
    if not app_id:
        return _json_error(status=400, error="missing_app_id", detail="app_id is required")
    if not schema_version:
        return _json_error(
            status=400,
            error="missing_schema_version",
            detail="schema_version is required",
        )
    if not event_type:
        return _json_error(
            status=400,
            error="missing_event_type",
            detail="event_type is required",
        )
    if not stage:
        return _json_error(status=400, error="missing_stage", detail="stage is required")
    if not isinstance(fields, dict):
        return _json_error(
            status=400,
            error="invalid_debug_fields",
            detail="fields must be an object",
        )
    registration = _load_debug_app(request, app_id)
    if registration is None:
        return _json_error(status=404, error="debug_app_not_found", detail=app_id)
    ai_id = str(data.get("ai_id", "") or "")
    if registration.allowed_ai_ids and ai_id not in registration.allowed_ai_ids:
        return _json_error(
            status=403,
            error="debug_ai_id_not_allowed",
            detail=f"ai_id={ai_id!r} is not registered for app_id={app_id!r}",
        )
    if (
        registration.allowed_event_types
        and event_type not in registration.allowed_event_types
    ):
        return _json_error(
            status=403,
            error="debug_event_type_not_allowed",
            detail=f"event_type={event_type!r} is not registered for app_id={app_id!r}",
        )
    schema = _load_debug_schema(request, app_id, schema_version)
    if schema is None:
        return _json_error(
            status=404,
            error="debug_schema_not_found",
            detail=_debug_schema_record_id(app_id, schema_version),
        )
    validation_error = _validate_debug_fields(schema, event_type, fields)
    if validation_error:
        return _json_error(
            status=400,
            error="invalid_debug_fields",
            detail=validation_error,
        )
    return DebugEventEnvelope(
        debug_event_id=_new_id("debug_evt"),
        app_id=app_id,
        schema_version=schema_version,
        ai_id=ai_id,
        tenant_id=str(data.get("tenant_id", "") or ""),
        session_id=str(data.get("session_id", "") or ""),
        end_user_ref=str(data.get("end_user_ref", "") or ""),
        response_id=str(data.get("response_id", "") or ""),
        interaction_id=str(data.get("interaction_id", "") or ""),
        event_type=event_type,
        stage=stage,
        fields=fields,
        occurred_at=str(data.get("occurred_at", "") or ""),
        created_at_ms=_now_ms(),
    )


def _load_debug_app(
    request: web.Request,
    app_id: str,
) -> DebugAppRegistration | None:
    registration = _debug_app_store(request).get(app_id)
    if registration is not None:
        return registration
    persisted = _get_persisted_governance(request, "debug_app", app_id)
    if persisted is None:
        return None
    try:
        return DebugAppRegistration.from_json(
            persisted,
            created_at_ms=int(persisted.get("created_at_ms", 0) or 0),
        )
    except ValueError:
        return None


def _load_debug_schema(
    request: web.Request,
    app_id: str,
    schema_version: str,
) -> DebugSchema | None:
    schema = _debug_schema_store(request).get((app_id, schema_version))
    if schema is not None:
        return schema
    record_id = _debug_schema_record_id(app_id, schema_version)
    persisted = _get_persisted_governance(request, "debug_schema", record_id)
    if persisted is None:
        return None
    try:
        return DebugSchema.from_json(
            persisted,
            app_id=app_id,
            created_at_ms=int(persisted.get("created_at_ms", 0) or 0),
        )
    except ValueError:
        return None


def _validate_debug_fields(
    schema: DebugSchema,
    event_type: str,
    fields: dict[str, Any],
) -> str:
    if schema.event_types and event_type not in schema.event_types:
        return f"event_type={event_type!r} is not allowed by schema"
    definitions = {field.name: field for field in schema.fields}
    missing = [
        field.name
        for field in schema.fields
        if field.required and field.name not in fields
    ]
    if missing:
        return f"missing required debug fields: {', '.join(sorted(missing))}"
    if not schema.allow_extra_fields:
        extra = sorted(set(fields) - set(definitions))
        if extra:
            return f"unknown debug fields: {', '.join(extra)}"
    for name, value in fields.items():
        definition = definitions.get(name)
        if definition is None:
            continue
        if definition.privacy_level is DebugPrivacyLevel.SECRET:
            return f"field {name!r} is secret and cannot be ingested"
        type_error = _debug_field_type_error(name, definition.type, value)
        if type_error:
            return type_error
        if definition.type is DebugFieldType.ENUM and value not in definition.enum_values:
            return f"field {name!r} must be one of {list(definition.enum_values)!r}"
    return ""


def _debug_field_type_error(
    name: str,
    field_type: DebugFieldType,
    value: Any,
) -> str:
    if field_type is DebugFieldType.STRING and not isinstance(value, str):
        return f"field {name!r} must be a string"
    if field_type is DebugFieldType.NUMBER and (
        isinstance(value, bool) or not isinstance(value, int | float)
    ):
        return f"field {name!r} must be a number"
    if field_type is DebugFieldType.BOOLEAN and not isinstance(value, bool):
        return f"field {name!r} must be a boolean"
    if field_type is DebugFieldType.ENUM and not isinstance(value, str):
        return f"field {name!r} must be an enum string"
    return ""


def _list_debug_events(request: web.Request) -> list[dict[str, Any]]:
    persisted = _list_persisted_governance(
        request,
        "debug_event",
        ai_id=request.query.get("ai_id", ""),
        session_id=request.query.get("session_id", ""),
    )
    events = persisted or [
        event.to_json() for event in _debug_event_store(request).values()
    ]
    filters = {
        "app_id": request.query.get("app_id", ""),
        "ai_id": request.query.get("ai_id", ""),
        "tenant_id": request.query.get("tenant_id", ""),
        "session_id": request.query.get("session_id", ""),
        "end_user_ref": request.query.get("end_user_ref", ""),
        "event_type": request.query.get("event_type", ""),
    }
    for key, value in filters.items():
        if value:
            events = [event for event in events if event.get(key) == value]
    requested_types = {
        str(value)
        for value in (
            request.query.getall("event_type", [])
            + [
                item
                for value in request.query.getall("event_types", [])
                for item in value.split(",")
            ]
            + [
                item
                for value in request.query.getall("event_types_csv", [])
                for item in value.split(",")
            ]
        )
        if str(value).strip()
    }
    if requested_types:
        events = [
            event for event in events if str(event.get("event_type", "")) in requested_types
        ]
    created_after = _int_query(request, "created_after_ms", 0)
    created_before = _int_query(request, "created_before_ms", 0)
    if created_after:
        events = [
            event
            for event in events
            if int(event.get("created_at_ms", 0) or 0) >= created_after
        ]
    if created_before:
        events = [
            event
            for event in events
            if int(event.get("created_at_ms", 0) or 0) <= created_before
        ]
    return sorted(
        events,
        key=lambda event: int(event.get("created_at_ms", 0) or 0),
        reverse=True,
    )


async def _build_debug_analysis_evidence(
    request: web.Request,
    analysis_request: DebugAnalysisRequest,
) -> dict[str, Any]:
    debug_events = _debug_events_for_analysis(request, analysis_request)
    evidence: dict[str, Any] = {
        "debug_events": debug_events,
        "debug_event_count": len(debug_events),
    }
    if analysis_request.include_audit:
        evidence["audit_events"] = _audit_events_for_analysis(request, analysis_request)
    if analysis_request.ai_id and analysis_request.session_id:
        runtime = await _runtime_debug_evidence(request, analysis_request)
        evidence.update(runtime)
    return evidence


def _debug_events_for_analysis(
    request: web.Request,
    analysis_request: DebugAnalysisRequest,
) -> list[dict[str, Any]]:
    events = _list_persisted_governance(
        request,
        "debug_event",
        ai_id=analysis_request.ai_id,
        session_id=analysis_request.session_id,
    ) or [event.to_json() for event in _debug_event_store(request).values()]
    filters = {
        "app_id": analysis_request.app_id,
        "ai_id": analysis_request.ai_id,
        "tenant_id": analysis_request.tenant_id,
        "session_id": analysis_request.session_id,
        "end_user_ref": analysis_request.end_user_ref,
    }
    for key, value in filters.items():
        if value:
            events = [event for event in events if event.get(key) == value]
    if analysis_request.event_types:
        allowed = set(analysis_request.event_types)
        events = [event for event in events if event.get("event_type") in allowed]
    return events


def _audit_events_for_analysis(
    request: web.Request,
    analysis_request: DebugAnalysisRequest,
) -> list[dict[str, Any]]:
    events = _list_persisted_governance(
        request,
        "audit_event",
        ai_id=analysis_request.ai_id,
        session_id=analysis_request.session_id,
    ) or [event.to_json() for event in _audit_store(request).values()]
    if analysis_request.ai_id:
        events = [event for event in events if event.get("ai_id") == analysis_request.ai_id]
    if analysis_request.session_id:
        events = [
            event for event in events if event.get("session_id") == analysis_request.session_id
        ]
    return events


async def _runtime_debug_evidence(
    request: web.Request,
    analysis_request: DebugAnalysisRequest,
) -> dict[str, Any]:
    try:
        manager = _resolve_session_manager(request, analysis_request.ai_id)
        session = await manager.get_session(analysis_request.session_id)
    except (_AiIdNotFoundError, SessionNotFoundError) as exc:
        return {"runtime_error": str(exc)}
    snapshots = session.latest_active_snapshots
    runtime: dict[str, Any] = {}
    if analysis_request.include_readouts:
        runtime["readouts"] = _build_readout_bundle(
            ai_id=analysis_request.ai_id,
            session_id=analysis_request.session_id,
            view=ReadoutView.SUMMARY,
            snapshots=snapshots,
            request=request,
        ).to_json()
    if analysis_request.include_explain:
        runtime["explain"] = ExplainTrace(
            ai_id=analysis_request.ai_id,
            session_id=analysis_request.session_id,
            chain=_build_explain_chain(snapshots),
        ).to_json()
    if analysis_request.include_snapshots:
        runtime["snapshots"] = {
            slot: snapshot_to_json(snapshot, redact=True)
            for slot, snapshot in snapshots.items()
        }
    return runtime


def _debug_recommendations(evidence: dict[str, Any]) -> tuple[str, ...]:
    recommendations: list[str] = []
    debug_event_count = int(evidence.get("debug_event_count", 0) or 0)
    if debug_event_count == 0:
        recommendations.append(
            "No app-owned debug events matched the selectors; add boundary instrumentation before deeper analysis."
        )
    if evidence.get("runtime_error"):
        recommendations.append(
            "Runtime readouts were unavailable for the selected ai_id/session_id; verify wake state and session correlation."
        )
    audit_events = evidence.get("audit_events", ())
    if isinstance(audit_events, list) and not audit_events:
        recommendations.append(
            "No platform audit events matched the selectors; check auth path and correlation metadata."
        )
    if not recommendations:
        recommendations.append(
            "Evidence is available for the selected scope; compare debug events, readouts, and audit events before proposing app or DLaaS changes."
        )
    return tuple(recommendations)


def _record_audit(
    request: web.Request,
    *,
    event_type: str,
    ai_id: str = "",
    contract_id: str = "",
    session_id: str = "",
    actor: str = "",
    payload: dict[str, Any] | None = None,
) -> AuditEvent:
    event = AuditEvent(
        event_id=_new_id("audit"),
        event_type=event_type,
        ai_id=ai_id,
        contract_id=contract_id,
        session_id=session_id,
        actor=actor,
        payload=payload or {},
        created_at_ms=_now_ms(),
    )
    _audit_store(request)[event.event_id] = event
    _persist_governance(
        request,
        "audit_event",
        event.event_id,
        event.to_json(),
        ai_id=ai_id,
        contract_id=contract_id,
        session_id=session_id,
    )
    _event_stream(request).append(
        event_record := EventStreamRecord(
            event_id=event.event_id,
            event_type=event.event_type,
            payload=event.to_json(),
            created_at_ms=event.created_at_ms,
        )
    )
    _persist_governance(
        request,
        "event_stream",
        event_record.event_id,
        event_record.to_json(),
        ai_id=ai_id,
        contract_id=contract_id,
        session_id=session_id,
    )
    return event


def _record_usage(
    request: web.Request,
    *,
    ai_id: str,
    metric: str,
    quantity: float,
) -> None:
    record = UsageRecord(
        usage_id=_new_id("usage"),
        ai_id=ai_id,
        metric=metric,
        quantity=quantity,
        created_at_ms=_now_ms(),
    )
    _usage_records(request).append(record)
    _persist_governance(
        request,
        "usage_record",
        record.usage_id,
        record.to_json(),
        ai_id=ai_id,
    )
    _billing_events(request).append(
        billing := BillingEvent(
            billing_event_id=_new_id("bill"),
            ai_id=ai_id,
            amount=0.0,
            reason=metric,
            created_at_ms=record.created_at_ms,
        )
    )
    _persist_governance(
        request,
        "billing_event",
        billing.billing_event_id,
        billing.to_json(),
        ai_id=ai_id,
    )


def _governance_store(request: web.Request) -> GovernanceStore | None:
    store = request.app.get(_GOVERNANCE_STORE_KEY)
    return store if isinstance(store, GovernanceStore) else None


def _persist_governance(
    request: web.Request,
    record_kind: str,
    record_id: str,
    payload: dict[str, Any],
    *,
    ai_id: str = "",
    contract_id: str = "",
    session_id: str = "",
) -> None:
    store = _governance_store(request)
    if store is None:
        return
    store.upsert(
        record_kind=record_kind,
        record_id=record_id,
        payload=payload,
        ai_id=ai_id,
        contract_id=contract_id,
        session_id=session_id,
        created_at_ms=int(payload.get("created_at_ms", _now_ms()) or _now_ms()),
    )


def _get_persisted_governance(
    request: web.Request,
    record_kind: str,
    record_id: str,
) -> dict[str, Any] | None:
    store = _governance_store(request)
    if store is None:
        return None
    try:
        return store.get(record_kind=record_kind, record_id=record_id)
    except GovernanceRecordNotFound:
        return None


def _list_persisted_governance(
    request: web.Request,
    record_kind: str,
    *,
    ai_id: str = "",
    session_id: str = "",
) -> list[dict[str, Any]]:
    store = _governance_store(request)
    if store is None:
        return []
    return list(store.list(record_kind=record_kind, ai_id=ai_id, session_id=session_id))


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _int_query(request: web.Request, name: str, default: int) -> int:
    raw = request.query.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


async def _session_from_query(
    request: web.Request,
    ai_id: str,
) -> Any | web.Response:
    session_id = request.query.get("session_id", "").strip()
    if not session_id:
        return _json_error(
            status=400,
            error="missing_session_id",
            detail="query parameter session_id is required",
        )
    try:
        manager = _resolve_session_manager(request, ai_id)
        return await manager.get_session(session_id)
    except _AiIdNotFoundError as exc:
        return _json_error(status=404, error=exc.code, detail=exc.detail)
    except SessionNotFoundError as exc:
        return _json_error(status=404, error="session_not_found", detail=str(exc))


def _has_admin_auth(request: web.Request) -> bool:
    auth = request.app.get(REGISTRY_APP_KEY)
    config = getattr(auth, "auth_config", None)
    cp_secret = getattr(config, "control_plane_secret", "") if config is not None else ""
    svc_secret = getattr(config, "service_secret", "") if config is not None else ""
    supplied_cp = request.headers.get("X-Control-Plane-Secret", "")
    supplied_svc = request.headers.get("X-Service-Secret", "")
    return bool(
        (cp_secret and supplied_cp == cp_secret)
        or (svc_secret and supplied_svc == svc_secret)
    )


def _build_readout_bundle(
    *,
    ai_id: str,
    session_id: str,
    view: ReadoutView,
    snapshots: dict[str, Any],
    request: web.Request,
) -> ReadoutBundle:
    active_mixture = snapshots.get("active_mixture")
    response_assembly = snapshots.get("response_assembly")
    boundary_policy = snapshots.get("boundary_policy")
    prediction_error = snapshots.get("prediction_error")
    strategy_playbook = snapshots.get("strategy_playbook")
    retrieval_policy = snapshots.get("retrieval_policy")
    domain_knowledge = snapshots.get("domain_knowledge")
    case_memory = snapshots.get("case_memory")
    temporal = snapshots.get("temporal_abstraction")
    protocol_summary = _protocol_readout(active_mixture)
    return ReadoutBundle(
        ai_id=ai_id,
        session_id=session_id,
        view=view,
        body={"lifecycle": _status_json_if_known(request, ai_id)},
        cognition={
            "active_regime": _field(response_assembly, "regime_id"),
            "expression_intent": _field(response_assembly, "expression_intent"),
            "prediction_error": _snapshot_summary(prediction_error),
            "temporal": _snapshot_summary(temporal),
        },
        knowledge={
            "retrieval": _snapshot_summary(retrieval_policy),
            "domain_knowledge": _snapshot_summary(domain_knowledge),
            "case_memory": _snapshot_summary(case_memory),
        },
        strategy={
            "playbook": _snapshot_summary(strategy_playbook),
            "matched_rule_count": _len_field(strategy_playbook, "matched_rules"),
        },
        protocol=protocol_summary,
        safety={
            "boundary_policy": _snapshot_summary(boundary_policy),
            "boundary_union_ids": protocol_summary.get("boundary_union_ids", []),
        },
        training={
            "protocol_submission_count": sum(
                1 for (stored_ai, _) in _protocol_store(request) if stored_ai == ai_id
            ),
            "training_job_count": sum(
                1 for (stored_ai, _) in _training_store(request) if stored_ai == ai_id
            ),
        },
    )


def _build_explain_chain(snapshots: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    response_assembly = snapshots.get("response_assembly")
    active_mixture = snapshots.get("active_mixture")
    boundary_policy = snapshots.get("boundary_policy")
    strategy_playbook = snapshots.get("strategy_playbook")
    retrieval_policy = snapshots.get("retrieval_policy")
    prediction_error = snapshots.get("prediction_error")
    return (
        {"step": "input_event", "description": "latest LifeformSession turn"},
        {
            "step": "regime",
            "regime_id": _field(response_assembly, "regime_id"),
            "description": _description(response_assembly),
        },
        {"step": "protocol", **_protocol_readout(active_mixture)},
        {"step": "boundary", "description": _description(boundary_policy)},
        {
            "step": "strategy",
            "description": _description(strategy_playbook),
            "matched_rule_count": _len_field(strategy_playbook, "matched_rules"),
        },
        {"step": "knowledge", "description": _description(retrieval_policy)},
        {
            "step": "response",
            "expression_intent": _field(response_assembly, "expression_intent"),
            "description": _description(response_assembly),
        },
        {"step": "prediction_error", "description": _description(prediction_error)},
    )


def _protocol_readout(snapshot: Any) -> dict[str, Any]:
    value = getattr(snapshot, "value", None)
    if value is None:
        return {"active_protocols": [], "boundary_union_ids": [], "strategy_weights": []}
    active = getattr(value, "active_protocols", ())
    weights = getattr(value, "strategy_weights", ())
    return {
        "active_protocols": [
            {
                "protocol_id": getattr(item, "protocol_id", ""),
                "activation_weight": getattr(item, "activation_weight", 0.0),
                "current_phase_id": getattr(item, "current_phase_id", None),
            }
            for item in active
        ],
        "boundary_union_ids": list(getattr(value, "boundary_union_ids", ())),
        "strategy_weights": [
            {
                "protocol_id": getattr(item, "protocol_id", ""),
                "rule_id": getattr(item, "rule_id", ""),
                "weight": getattr(item, "weight", 0.0),
                "compiled_rule_id": getattr(item, "compiled_rule_id", ""),
            }
            for item in weights
        ],
        "description": getattr(value, "description", ""),
    }


def _snapshot_summary(snapshot: Any) -> dict[str, Any]:
    if snapshot is None:
        return {"present": False}
    return {
        "present": True,
        "slot_name": getattr(snapshot, "slot_name", ""),
        "owner": getattr(snapshot, "owner", ""),
        "version": getattr(snapshot, "version", 0),
        "description": _description(snapshot),
    }


def _description(snapshot: Any) -> str:
    value = getattr(snapshot, "value", None)
    return str(getattr(value, "description", "") or "")


def _field(snapshot: Any, field_name: str) -> Any:
    value = getattr(snapshot, "value", None)
    if value is None:
        return None
    return getattr(value, field_name, None)


def _len_field(snapshot: Any, field_name: str) -> int:
    value = getattr(snapshot, "value", None)
    if value is None:
        return 0
    return len(getattr(value, field_name, ()) or ())


def _status_json_if_known(request: web.Request, ai_id: str) -> dict[str, Any]:
    launcher = request.app.get(INSTANCE_MANAGER_APP_KEY)
    if not isinstance(launcher, InstanceManager):
        return {"ai_id": ai_id, "lifecycle_state": "single_instance"}
    try:
        return launcher.status(ai_id).to_json()
    except InstanceNotFound:
        return {"ai_id": ai_id, "lifecycle_state": "unknown"}


def _payload_has_boundary_contract(data: dict[str, Any]) -> bool:
    if data.get("boundary_contracts") or data.get("boundaries"):
        return True
    protocol = data.get("protocol")
    return isinstance(protocol, dict) and bool(
        protocol.get("boundary_contracts") or protocol.get("boundaries")
    )


def _catalog_blueprints() -> tuple[LifeBlueprint, ...]:
    return (
        LifeBlueprint(
            blueprint_id="companion/default/dev-v1",
            display_name="Default Companion",
            vertical=AdoptionConfig.from_json(
                {
                    "vertical": {
                        "vertical_id": "companion",
                        "runtime_template_id": "companion",
                    }
                }
            ).vertical,
            evaluation_gates=("protocol-effective",),
        ),
        LifeBlueprint(
            blueprint_id="growth-advisor/cheng-laoshi/private-domain-v1",
            display_name="Cheng Laoshi Growth Advisor",
            vertical=AdoptionConfig.from_json(
                {
                    "vertical": {
                        "vertical_id": "growth_advisor",
                        "runtime_template_id": "growth_advisor",
                        "profile_id": "cheng_laoshi",
                    },
                    "protocols": {"autoload": ["growth_advisor:cheng-laoshi"]},
                    "tools": {
                        "tool_policy_id": "growth-advisor-wechat-readonly",
                        "allowed_capabilities": [
                            "text",
                            "handoff_ticket",
                            "reviewed_knowledge",
                        ],
                    },
                    "ops": {
                        "awake_strategy": "on_demand",
                        "handoff_policy_id": "growth-advisor-standard",
                    },
                }
            ).vertical,
            substrate=AdoptionConfig.from_json({}).substrate,
            protocols=AdoptionConfig.from_json(
                {"protocols": {"autoload": ["growth_advisor:cheng-laoshi"]}}
            ).protocols,
            tools=AdoptionConfig.from_json(
                {
                    "tools": {
                        "tool_policy_id": "growth-advisor-wechat-readonly",
                        "allowed_capabilities": [
                            "text",
                            "handoff_ticket",
                            "reviewed_knowledge",
                        ],
                    }
                }
            ).tools,
            ops=AdoptionConfig.from_json(
                {
                    "ops": {
                        "awake_strategy": "on_demand",
                        "handoff_policy_id": "growth-advisor-standard",
                    }
                }
            ).ops,
            evaluation_gates=(
                "boundary-baseline",
                "protocol-effective",
                "handoff-slo",
            ),
        ),
    )


def _get_training_job(request: web.Request) -> TrainingJob | web.Response:
    ai_id = request.match_info.get("ai_id", "")
    job_id = request.match_info.get("job_id", "")
    job = _training_store(request).get((ai_id, job_id))
    if job is None:
        return _json_error(status=404, error="training_job_not_found", detail=job_id)
    return job


async def _parse_envelope(request: web.Request) -> InteractionEnvelope:
    if not request.body_exists:
        raise _EnvelopeError("invalid_envelope", "Request body is required")
    try:
        text = await request.text()
    except (web.HTTPException, OSError) as exc:
        raise _EnvelopeError(
            "invalid_body", f"Could not read body: {exc}"
        ) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _EnvelopeError(
            "invalid_json", f"Body is not valid JSON: {exc}"
        ) from exc
    try:
        return InteractionEnvelope.from_json(data)
    except ValueError as exc:
        raise _EnvelopeError("invalid_envelope", str(exc)) from exc


async def _read_json_object(
    request: web.Request,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if not request.body_exists:
        if allow_empty:
            return {}
        raise ValueError("Request body is required")
    try:
        text = await request.text()
    except (web.HTTPException, OSError) as exc:
        raise ValueError(f"Could not read body: {exc}") from exc
    if not text.strip():
        if allow_empty:
            return {}
        raise ValueError("Request body is required")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Body is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Body must be a JSON object")
    return data


async def _read_json_or_error(
    request: web.Request,
    *,
    allow_empty: bool = False,
) -> dict[str, Any] | web.Response:
    try:
        return await _read_json_object(request, allow_empty=allow_empty)
    except ValueError as exc:
        return _json_error(status=400, error="invalid_json_body", detail=str(exc))


async def _get_or_create_session(
    manager: SessionManager,
    session_id: str,
    *,
    user_id: str | None = None,
):
    """Reuse an existing session if present; otherwise create with that id."""
    try:
        return await manager.get_session(session_id)
    except SessionNotFoundError:
        return await manager.create_session(session_id=session_id, user_id=user_id)


def _json_error(
    *,
    status: int,
    error: str,
    detail: str = "",
    extra: dict[str, Any] | None = None,
) -> web.Response:
    payload: dict[str, Any] = {"status": "error", "error": error, "detail": detail}
    if extra:
        payload.update(extra)
    return web.json_response(payload, status=status)


class _EnvelopeError(Exception):
    """Raised by ``_parse_envelope`` for any 400-level parse failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class _AiIdNotFoundError(Exception):
    """Raised by :func:`_resolve_session_manager` when launcher is bound
    but the ai_id is unknown."""

    def __init__(self, *, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


__all__ = [
    "DLAAS_APP_AI_ID_KEY",
    "attach_dlaas_full_stack",
    "attach_dlaas_routes",
    "build_dlaas_app",
]
