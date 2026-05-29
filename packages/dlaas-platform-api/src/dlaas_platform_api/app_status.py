"""Application fleet status — the app-granularity health layer.

R6-R8 monitored each ai_id (a digital life). This module monitors
each *application* (a consumer deployment: einstein, coread,
repair30, family-memorial, ...). The reporting mechanism already
exists — apps register via `/dlaas/v1/debug/apps` and emit
`/dlaas/v1/debug/events` (+ ExperienceLoop). What was missing was a
per-app operational rollup: which apps are reporting, which went
silent, and which are erroring.

We reuse the exact verdict contract from `cognition_health.py`
(`status` in ok/watch/alert + a list of `signals`) so the portal can
render app status with the same `HealthBadge` / `HealthSignalList`
components. The entity differs (app_id vs ai_id); the judgment shape
is identical.

Data sources (both existing in-memory stores, no new persistence):
* `_DEBUG_APPS_KEY` — the registry (app_id / display_name / tenant_id)
* `_DEBUG_EVENTS_KEY` — the activity (app_id / event_type / stage /
  fields / created_at_ms)

Signals (derived, no new app-side reporting required):
* ``silent`` (alert) — a known app (registered or previously active)
  with zero events in the window, or whose most recent event is older
  than ``DLAAS_APP_STALE_HOURS``. A silent app is the single most
  important fleet signal: it cannot be told from a dead one without
  this.
* ``error_rate`` (watch/alert) — fraction of recent events that look
  like errors. The error test is a heuristic (``fields.ok is False``,
  ``fields.status >= 400``, or an event_type containing
  error/failed/refus) and is documented as such; calibrate per app.

No signals → ``ok`` (reporting).
"""
from __future__ import annotations

import os
import time
from typing import Any, Iterable

from aiohttp import web

_SEVERITY_ORDER = {"ok": 0, "watch": 1, "alert": 2}

_DEBUG_APPS_KEY = "dlaas_debug_apps"
_DEBUG_EVENTS_KEY = "dlaas_debug_events"

_DEFAULT_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
_MAX_WINDOW_MS = 90 * 24 * 60 * 60 * 1000

_ERROR_EVENT_TOKENS = ("error", "failed", "failure", "refus")


class AppStatusThresholds:
    def __init__(
        self,
        *,
        stale_hours: float,
        error_rate_watch: float,
        error_rate_alert: float,
        min_events_for_error: int,
    ) -> None:
        self.stale_hours = stale_hours
        self.error_rate_watch = error_rate_watch
        self.error_rate_alert = error_rate_alert
        self.min_events_for_error = min_events_for_error

    @classmethod
    def from_env(cls) -> "AppStatusThresholds":
        return cls(
            stale_hours=_env_float("DLAAS_APP_STALE_HOURS", 24.0),
            error_rate_watch=_env_float("DLAAS_APP_ERROR_RATE_WATCH", 0.1),
            error_rate_alert=_env_float("DLAAS_APP_ERROR_RATE_ALERT", 0.3),
            # Don't flag error_rate until there are enough events for the
            # ratio to mean anything.
            min_events_for_error=_env_int("DLAAS_APP_MIN_EVENTS_FOR_ERROR", 5),
        )


def is_error_event(event: dict[str, Any]) -> bool:
    """Heuristic: does this debug event look like a failure?

    Apps don't share a single error schema, so we check the common
    shapes: a falsey `ok`, an HTTP-ish `status >= 400`, or an
    event_type naming a failure.
    """
    event_type = str(event.get("event_type", "") or "").lower()
    if any(tok in event_type for tok in _ERROR_EVENT_TOKENS):
        return True
    fields = event.get("fields")
    if isinstance(fields, dict):
        if fields.get("ok") is False:
            return True
        status = fields.get("status")
        if isinstance(status, (int, float)) and not isinstance(status, bool):
            if status >= 400:
                return True
    return False


def compute_app_status(
    events: list[dict[str, Any]],
    *,
    now_ms: int,
    thresholds: AppStatusThresholds,
    app_id: str,
    registered: bool,
    display_name: str = "",
    tenant_id: str = "",
) -> dict[str, Any]:
    """Pure verdict for one app's window of events."""
    rows = sorted(events, key=lambda r: int(r.get("created_at_ms", 0) or 0))
    event_count = len(rows)
    signals: list[dict[str, Any]] = []

    last_event_ms = (
        int(rows[-1].get("created_at_ms", 0) or 0) if rows else None
    )

    event_types: list[str] = []
    for row in rows:
        et = str(row.get("event_type", "") or "")
        if et and et not in event_types:
            event_types.append(et)

    # --- silent -------------------------------------------------------
    if event_count == 0:
        # A known app (registered or previously seen) with no events in
        # the window is silent. An unregistered app with no events would
        # not appear at all, so reaching here means it is known.
        signals.append(
            _signal(
                "silent",
                "alert",
                "no events in window",
                value=0,
            )
        )
    else:
        stale_ms = now_ms - (last_event_ms or now_ms)
        stale_hours = stale_ms / (60 * 60 * 1000)
        if stale_hours >= thresholds.stale_hours:
            signals.append(
                _signal(
                    "silent",
                    "alert",
                    f"last event {stale_hours:.1f}h ago >= {thresholds.stale_hours}h",
                    value=stale_hours,
                )
            )

    # --- error_rate ---------------------------------------------------
    if event_count >= thresholds.min_events_for_error:
        error_count = sum(1 for row in rows if is_error_event(row))
        rate = error_count / event_count
        if rate >= thresholds.error_rate_alert:
            signals.append(
                _signal(
                    "error_rate",
                    "alert",
                    f"error rate {rate:.0%} >= {thresholds.error_rate_alert:.0%}",
                    value=rate,
                )
            )
        elif rate >= thresholds.error_rate_watch:
            signals.append(
                _signal(
                    "error_rate",
                    "watch",
                    f"error rate {rate:.0%} >= {thresholds.error_rate_watch:.0%}",
                    value=rate,
                )
            )

    status = "ok"
    for sig in signals:
        if _SEVERITY_ORDER[sig["severity"]] > _SEVERITY_ORDER[status]:
            status = sig["severity"]

    return {
        "app_id": app_id,
        "display_name": display_name or app_id,
        "tenant_id": tenant_id,
        "registered": registered,
        "status": status,
        "signals": signals,
        "event_count": event_count,
        "event_types": event_types,
        "last_event_ms": last_event_ms,
        "computed_at_ms": now_ms,
    }


def compute_app_overview(
    events: Iterable[dict[str, Any]],
    registry: Iterable[dict[str, Any]],
    *,
    now_ms: int,
    thresholds: AppStatusThresholds,
) -> dict[str, Any]:
    """Union the registry with event-derived apps and judge each."""
    events_by_app: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        app_id = str(ev.get("app_id", "") or "")
        if not app_id:
            continue
        events_by_app.setdefault(app_id, []).append(ev)

    registered_meta: dict[str, dict[str, str]] = {}
    for reg in registry:
        app_id = str(reg.get("app_id", "") or "")
        if not app_id:
            continue
        registered_meta[app_id] = {
            "display_name": str(reg.get("display_name", "") or app_id),
            "tenant_id": str(reg.get("tenant_id", "") or ""),
        }

    all_app_ids = set(events_by_app) | set(registered_meta)
    items: list[dict[str, Any]] = []
    counts = {"ok": 0, "watch": 0, "alert": 0}
    for app_id in all_app_ids:
        meta = registered_meta.get(app_id, {})
        rows = events_by_app.get(app_id, [])
        tenant_id = meta.get("tenant_id", "")
        if not tenant_id and rows:
            tenant_id = str(rows[-1].get("tenant_id", "") or "")
        verdict = compute_app_status(
            rows,
            now_ms=now_ms,
            thresholds=thresholds,
            app_id=app_id,
            registered=app_id in registered_meta,
            display_name=meta.get("display_name", ""),
            tenant_id=tenant_id,
        )
        counts[verdict["status"]] = counts.get(verdict["status"], 0) + 1
        items.append(verdict)

    items.sort(
        key=lambda v: (
            -_SEVERITY_ORDER.get(v["status"], 0),
            -int(v.get("last_event_ms") or 0),
        )
    )
    return {
        "status": "ok",
        "computed_at_ms": now_ms,
        "counts": counts,
        "items": items,
    }


def attach_app_status_routes(app: web.Application) -> None:
    app.router.add_get("/dlaas/v1/apps/status", _handle_app_status_overview)
    app.router.add_get(
        "/dlaas/v1/apps/status/{app_id}", _handle_app_status_detail
    )


async def _handle_app_status_overview(request: web.Request) -> web.Response:
    events = _windowed_events(request)
    registry = _registry_rows(request, tenant_id=request.query.get("tenant_id", ""))
    overview = compute_app_overview(
        events,
        registry,
        now_ms=_now_ms(),
        thresholds=AppStatusThresholds.from_env(),
    )
    return web.json_response(overview)


async def _handle_app_status_detail(request: web.Request) -> web.Response:
    app_id = request.match_info.get("app_id", "").strip()
    if not app_id:
        return web.json_response(
            {"status": "error", "error": "missing_app_id"}, status=400
        )
    events = [
        ev for ev in _windowed_events(request) if str(ev.get("app_id", "")) == app_id
    ]
    registry = {
        str(r.get("app_id", "")): r for r in _registry_rows(request, tenant_id="")
    }
    reg = registry.get(app_id)
    verdict = compute_app_status(
        events,
        now_ms=_now_ms(),
        thresholds=AppStatusThresholds.from_env(),
        app_id=app_id,
        registered=reg is not None,
        display_name=str((reg or {}).get("display_name", "") or ""),
        tenant_id=str((reg or {}).get("tenant_id", "") or ""),
    )
    # Breakdown by event_type and stage for the detail view.
    by_type: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    for ev in events:
        et = str(ev.get("event_type", "") or "unknown")
        st = str(ev.get("stage", "") or "unknown")
        by_type[et] = by_type.get(et, 0) + 1
        by_stage[st] = by_stage.get(st, 0) + 1
    return web.json_response(
        {
            "status": "ok",
            "app": verdict,
            "breakdown": {
                "by_event_type": by_type,
                "by_stage": by_stage,
            },
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _windowed_events(request: web.Request) -> list[dict[str, Any]]:
    store = request.app.get(_DEBUG_EVENTS_KEY) or {}
    rows: list[dict[str, Any]] = []
    for ev in store.values():
        if hasattr(ev, "to_json"):
            rows.append(ev.to_json())
        elif isinstance(ev, dict):
            rows.append(ev)
    tenant_id = request.query.get("tenant_id", "").strip()
    if tenant_id:
        rows = [r for r in rows if str(r.get("tenant_id", "")) == tenant_id]
    window_ms = _resolve_window_ms(request)
    cutoff = _now_ms() - window_ms
    rows = [r for r in rows if int(r.get("created_at_ms", 0) or 0) >= cutoff]
    return rows


def _registry_rows(
    request: web.Request, *, tenant_id: str
) -> list[dict[str, Any]]:
    store = request.app.get(_DEBUG_APPS_KEY) or {}
    rows: list[dict[str, Any]] = []
    for reg in store.values():
        if hasattr(reg, "to_json"):
            rows.append(reg.to_json())
        elif isinstance(reg, dict):
            rows.append(reg)
    if tenant_id:
        rows = [
            r
            for r in rows
            # Registry tenant_id may be blank (global registration); keep
            # those plus exact tenant matches.
            if not str(r.get("tenant_id", "") or "")
            or str(r.get("tenant_id", "")) == tenant_id
        ]
    return rows


def _signal(
    name: str, severity: str, detail: str, *, value: float | int
) -> dict[str, Any]:
    return {"name": name, "severity": severity, "detail": detail, "value": value}


def _resolve_window_ms(request: web.Request) -> int:
    raw = request.query.get("window", "").strip().lower()
    if not raw:
        return _DEFAULT_WINDOW_MS
    if raw.isdigit():
        return min(int(raw) * 1000, _MAX_WINDOW_MS)
    unit = raw[-1]
    try:
        amount = int(raw[:-1])
    except ValueError:
        return _DEFAULT_WINDOW_MS
    if amount <= 0:
        return _DEFAULT_WINDOW_MS
    mult = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}.get(unit)
    if mult is None:
        return _DEFAULT_WINDOW_MS
    return min(amount * mult, _MAX_WINDOW_MS)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _now_ms() -> int:
    return int(time.time() * 1000)
