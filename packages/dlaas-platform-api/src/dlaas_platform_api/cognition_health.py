"""Cognition health verdict — the normative layer over cognition snapshots.

`cognition.py` answers *what is* (descriptive readouts). This module
answers *is it OK* (normative judgment). The split is deliberate
(first-principles): observation and judgment have different change
cadences and different owners — thresholds here are an operational
policy, not a kernel fact.

Health signals (all derived from existing snapshot fields, so we add
no new cognition concept and never re-derive owner state — R12/R14):

* ``regime_instability`` — regime switches per sample over the window
  (thrash detection). A twin flipping regime almost every turn is not
  settling.
* ``pe_elevation`` — mean ``prediction_error.magnitude`` over the
  window. Persistently high PE means the kernel is continuously
  "surprised".
* ``eval_alerts`` — sum of ``eval_alert_count`` over the window.
* ``staleness`` — wall-clock gap since the most recent snapshot. A
  digital life that has gone silent for a long time may be stuck or
  abandoned.

Each signal yields a severity (``watch`` < ``alert``) or is absent.
The overall verdict is the max severity; no signals → ``ok``.

Thresholds are module constants overridable by env so operators can
calibrate against real traffic without a code change. They are first
guesses and are flagged as needing calibration in
``apps/dlaas-portal/known-debts.md`` (round 8).

Security: the overview endpoint (used by cross-tenant operators)
returns only the verdict + signal *names/details*, never the
``raw_readout`` blob, so a platform operator cannot read tenant
private readout content through the health surface.
"""
from __future__ import annotations

import os
import time
from typing import Any, Iterable

from aiohttp import web

from dlaas_platform_api.cognition import COGNITION_SNAPSHOTS_KEY

# Severity ordering for the max() roll-up.
_SEVERITY_ORDER = {"ok": 0, "watch": 1, "alert": 2}

_DEFAULT_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
_MAX_WINDOW_MS = 90 * 24 * 60 * 60 * 1000


class HealthThresholds:
    """Resolved health thresholds, env-overridable.

    Read once per request from the environment so tests can monkeypatch
    ``os.environ`` and operators can tune without a redeploy.
    """

    def __init__(
        self,
        *,
        pe_watch: float,
        pe_alert: float,
        regime_thrash: float,
        eval_alert: int,
        stale_hours: float,
    ) -> None:
        self.pe_watch = pe_watch
        self.pe_alert = pe_alert
        self.regime_thrash = regime_thrash
        self.eval_alert = eval_alert
        self.stale_hours = stale_hours

    @classmethod
    def from_env(cls) -> "HealthThresholds":
        return cls(
            pe_watch=_env_float("DLAAS_COG_PE_WATCH", 0.4),
            pe_alert=_env_float("DLAAS_COG_PE_ALERT", 0.7),
            # Alert when regime switches per (sample-1) exceeds this ratio.
            regime_thrash=_env_float("DLAAS_COG_REGIME_THRASH", 0.75),
            # Alert when summed eval_alert_count over the window >= this.
            eval_alert=_env_int("DLAAS_COG_EVAL_ALERT", 3),
            stale_hours=_env_float("DLAAS_COG_STALE_HOURS", 72.0),
        )


def compute_health(
    snapshots: list[dict[str, Any]],
    *,
    now_ms: int,
    thresholds: HealthThresholds,
    ai_id: str = "",
) -> dict[str, Any]:
    """Pure verdict for one ai_id's snapshot list.

    ``snapshots`` may be in any order; we sort by ``captured_at_ms``.
    Returns ``{ai_id, status, signals, sample_count, last_seen_ms,
    latest_session_id, computed_at_ms}`` with NO raw_readout.
    """
    rows = sorted(
        snapshots, key=lambda r: int(r.get("captured_at_ms", 0) or 0)
    )
    sample_count = len(rows)
    signals: list[dict[str, Any]] = []

    if sample_count == 0:
        return {
            "ai_id": ai_id,
            "status": "ok",
            "signals": [],
            "sample_count": 0,
            "last_seen_ms": None,
            "latest_session_id": None,
            "computed_at_ms": now_ms,
        }

    latest = rows[-1]
    last_seen_ms = int(latest.get("captured_at_ms", 0) or 0)
    latest_session_id = latest.get("session_id") or None

    # --- pe_elevation -------------------------------------------------
    pe_values = [
        _pe_magnitude(row)
        for row in rows
        if _pe_magnitude(row) is not None
    ]
    if pe_values:
        pe_mean = sum(pe_values) / len(pe_values)
        if pe_mean >= thresholds.pe_alert:
            signals.append(
                _signal(
                    "pe_elevation",
                    "alert",
                    f"mean PE magnitude {pe_mean:.3f} >= {thresholds.pe_alert}",
                    value=pe_mean,
                )
            )
        elif pe_mean >= thresholds.pe_watch:
            signals.append(
                _signal(
                    "pe_elevation",
                    "watch",
                    f"mean PE magnitude {pe_mean:.3f} >= {thresholds.pe_watch}",
                    value=pe_mean,
                )
            )

    # --- regime_instability ------------------------------------------
    # Needs at least 4 samples for a meaningful ratio.
    if sample_count >= 4:
        switches = 0
        prev = object()
        for row in rows:
            cur = row.get("regime_id")
            if prev is not object() and cur != prev:  # type: ignore[comparison-overlap]
                switches += 1
            prev = cur  # type: ignore[assignment]
        ratio = switches / max(1, sample_count - 1)
        watch_ratio = thresholds.regime_thrash * 0.66
        if ratio >= thresholds.regime_thrash:
            signals.append(
                _signal(
                    "regime_instability",
                    "alert",
                    f"regime switch ratio {ratio:.2f} >= {thresholds.regime_thrash}",
                    value=ratio,
                )
            )
        elif ratio >= watch_ratio:
            signals.append(
                _signal(
                    "regime_instability",
                    "watch",
                    f"regime switch ratio {ratio:.2f} >= {watch_ratio:.2f}",
                    value=ratio,
                )
            )

    # --- eval_alerts --------------------------------------------------
    eval_sum = sum(int(row.get("eval_alert_count", 0) or 0) for row in rows)
    if eval_sum >= thresholds.eval_alert:
        signals.append(
            _signal(
                "eval_alerts",
                "alert",
                f"{eval_sum} eval alerts in window >= {thresholds.eval_alert}",
                value=eval_sum,
            )
        )
    elif eval_sum > 0:
        signals.append(
            _signal(
                "eval_alerts",
                "watch",
                f"{eval_sum} eval alert(s) in window",
                value=eval_sum,
            )
        )

    # --- staleness ----------------------------------------------------
    stale_ms = now_ms - last_seen_ms
    stale_hours = stale_ms / (60 * 60 * 1000)
    alert_hours = thresholds.stale_hours
    watch_hours = alert_hours / 2
    if stale_hours >= alert_hours:
        signals.append(
            _signal(
                "staleness",
                "alert",
                f"no snapshot for {stale_hours:.1f}h >= {alert_hours}h",
                value=stale_hours,
            )
        )
    elif stale_hours >= watch_hours:
        signals.append(
            _signal(
                "staleness",
                "watch",
                f"no snapshot for {stale_hours:.1f}h >= {watch_hours:.1f}h",
                value=stale_hours,
            )
        )

    status = "ok"
    for sig in signals:
        if _SEVERITY_ORDER[sig["severity"]] > _SEVERITY_ORDER[status]:
            status = sig["severity"]

    return {
        "ai_id": ai_id,
        "status": status,
        "signals": signals,
        "sample_count": sample_count,
        "last_seen_ms": last_seen_ms,
        "latest_session_id": latest_session_id,
        "computed_at_ms": now_ms,
    }


def compute_overview(
    snapshots: Iterable[dict[str, Any]],
    *,
    now_ms: int,
    thresholds: HealthThresholds,
) -> dict[str, Any]:
    """Group snapshots by ai_id and compute a verdict for each.

    Returns counts by status plus a per-ai_id list. Deliberately
    excludes ``raw_readout`` (cross-tenant operator safety).
    """
    by_ai: dict[str, list[dict[str, Any]]] = {}
    tenant_of: dict[str, str] = {}
    for row in snapshots:
        ai_id = str(row.get("ai_id", "") or "")
        if not ai_id:
            continue
        by_ai.setdefault(ai_id, []).append(row)
        if ai_id not in tenant_of:
            tenant_of[ai_id] = str(row.get("tenant_id", "") or "")

    items: list[dict[str, Any]] = []
    counts = {"ok": 0, "watch": 0, "alert": 0}
    for ai_id, rows in by_ai.items():
        verdict = compute_health(
            rows, now_ms=now_ms, thresholds=thresholds, ai_id=ai_id
        )
        verdict["tenant_id"] = tenant_of.get(ai_id, "")
        counts[verdict["status"]] = counts.get(verdict["status"], 0) + 1
        items.append(verdict)

    # Sort worst-first so operators see alerts at the top.
    items.sort(
        key=lambda v: (
            -_SEVERITY_ORDER.get(v["status"], 0),
            -int(v.get("last_seen_ms") or 0),
        )
    )
    return {
        "status": "ok",
        "computed_at_ms": now_ms,
        "counts": counts,
        "items": items,
    }


def attach_cognition_health_routes(app: web.Application) -> None:
    app.router.add_get("/dlaas/v1/cognition/health", _handle_cognition_health)
    app.router.add_get(
        "/dlaas/v1/cognition/health/overview", _handle_cognition_health_overview
    )


async def _handle_cognition_health(request: web.Request) -> web.Response:
    ai_id = request.query.get("ai_id", "").strip()
    if not ai_id:
        return web.json_response(
            {"status": "error", "error": "missing_ai_id"}, status=400
        )
    rows = _filtered_rows(request, ai_id=ai_id)
    verdict = compute_health(
        rows,
        now_ms=_now_ms(),
        thresholds=HealthThresholds.from_env(),
        ai_id=ai_id,
    )
    return web.json_response({"status": "ok", "health": verdict})


async def _handle_cognition_health_overview(request: web.Request) -> web.Response:
    rows = _filtered_rows(request)
    overview = compute_overview(
        rows,
        now_ms=_now_ms(),
        thresholds=HealthThresholds.from_env(),
    )
    return web.json_response(overview)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filtered_rows(
    request: web.Request, *, ai_id: str = ""
) -> list[dict[str, Any]]:
    """Window + tenant + ai_id filtered snapshot rows from the store."""
    rows = list(request.app.get(COGNITION_SNAPSHOTS_KEY) or [])
    tenant_id = request.query.get("tenant_id", "").strip()
    if ai_id:
        rows = [r for r in rows if str(r.get("ai_id", "")) == ai_id]
    if tenant_id:
        rows = [r for r in rows if str(r.get("tenant_id", "")) == tenant_id]
    window_ms = _resolve_window_ms(request)
    cutoff = _now_ms() - window_ms
    rows = [r for r in rows if int(r.get("captured_at_ms", 0) or 0) >= cutoff]
    return rows


def _pe_magnitude(row: dict[str, Any]) -> float | None:
    pe = row.get("prediction_error")
    if not isinstance(pe, dict):
        return None
    raw = pe.get("magnitude")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _signal(
    name: str, severity: str, detail: str, *, value: float | int
) -> dict[str, Any]:
    return {
        "name": name,
        "severity": severity,
        "detail": detail,
        "value": value,
    }


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
