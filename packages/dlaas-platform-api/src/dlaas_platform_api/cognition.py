"""Cognition snapshot store and aggregation endpoints.

This module exposes the ``/dlaas/v1/cognition/*`` surface that
dlaas-portal and other apps use to visualise capability / regime /
experience trajectories.

Storage model
-------------

We mirror the existing in-memory shadow stores used by ``app.py`` —
one list keyed on the aiohttp ``web.Application`` (``app[_COGNITION_SNAPSHOTS_KEY]``).
There is no SQLAlchemy / Alembic layer: the platform API runs
single-process and persistence for cognition snapshots is a future
concern (tracked separately as round-6 open debt in
``apps/dlaas-portal/known-debts.md``).

Write path
----------

``record_cognition_snapshot`` is called from
``_dispatch_envelope_to_instance`` immediately after a successful
``dispatch_envelope``. Each interaction therefore writes one
``CognitionSnapshot`` row that captures the live readout bundle plus
the regime / prediction-error / learning-family deltas. Future
``"session_end"`` or ``"sampler"`` sources reuse the same record
helper.

Read path
---------

Five GET endpoints aggregate the in-memory list:

* ``GET /dlaas/v1/cognition/snapshots`` — raw paginated list
* ``GET /dlaas/v1/cognition/timelines/regime`` — coalesced regime
  ranges per ai_id
* ``GET /dlaas/v1/cognition/learning-family`` — 6-class sums over a
  window, suitable for a radar chart
* ``GET /dlaas/v1/cognition/experience-throughput`` — reads from the
  existing ``dlaas_debug_events`` store and groups
  experience.receipt / experience.reflection by binding and day
* ``GET /dlaas/v1/cognition/eval-trend`` — reads ``dlaas_eval_runs``
  and projects pass / fail rate by day

All five endpoints are tenant-scoped via the optional ``tenant_id``
query parameter; portal BFF already pins per-org calls so the
parameter is mainly relevant for admin / operator consoles.

Design constraints
------------------

* R14 — cognition state is owned by the kernel; this module only
  records readout projections and never re-derives cognition.
* R4 / R12 — visualisations are readouts (soft signals); we do NOT
  compute a model self-assessed capability score.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from aiohttp import web

_LOG = logging.getLogger("dlaas_platform_api.cognition")

# Public store key so other modules can reach it via ``app[<KEY>]``
COGNITION_SNAPSHOTS_KEY = "dlaas_cognition_snapshots"

# Mirrored from app.py — we deliberately avoid importing from app.py to
# keep the dependency direction one-way (app.py imports this module,
# not the reverse).
_DEBUG_EVENTS_KEY = "dlaas_debug_events"
_EVAL_RUNS_KEY = "dlaas_eval_runs"

LEARNING_FAMILY_KEYS: tuple[str, ...] = (
    "cognition",
    "knowledge",
    "strategy",
    "protocol",
    "safety",
    "training",
)

# Accepted source markers; we only enforce shape, not enum membership,
# so future writers can stamp new sources without a schema bump.
ALLOWED_SOURCES: frozenset[str] = frozenset(
    {"interaction", "session_end", "sampler", "manual"}
)

_DEFAULT_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
_MAX_WINDOW_MS = 90 * 24 * 60 * 60 * 1000
_DEFAULT_LIMIT = 200
_MAX_LIMIT = 1000


@dataclass(frozen=True)
class CognitionSnapshot:
    """One readout projection captured after an interaction.

    The raw readout bundle is stored verbatim so future endpoints can
    add new projections without re-writing history. Numeric fields are
    extracted up-front to keep the aggregation endpoints cheap.
    """

    snapshot_id: str
    tenant_id: str
    ai_id: str
    session_id: str
    source: str
    captured_at_ms: int
    regime_id: str | None
    prediction_error: dict[str, float]
    learning_family: dict[str, int]
    eval_alert_count: int
    memory_entries: int
    raw_readout: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def ensure_cognition_store(app: web.Application) -> None:
    """Idempotent store initialisation. Called from ``_ensure_shadow_intake_stores``."""
    app.setdefault(COGNITION_SNAPSHOTS_KEY, [])


def cognition_store(request: web.Request) -> list[dict[str, Any]]:
    return request.app[COGNITION_SNAPSHOTS_KEY]


def record_cognition_snapshot(
    request: web.Request,
    *,
    ai_id: str,
    session_id: str,
    snapshots: dict[str, Any],
    readout_bundle_json: dict[str, Any],
    source: str = "interaction",
    tenant_id: str = "",
) -> CognitionSnapshot:
    """Append a cognition snapshot row.

    Never raises: the dispatch path must succeed even if the readout
    bundle is incomplete or a snapshot value is missing.
    """
    if source not in ALLOWED_SOURCES:
        source = "manual"

    cognition_view = (readout_bundle_json.get("cognition") or {}) if isinstance(
        readout_bundle_json, dict
    ) else {}
    regime_id = cognition_view.get("active_regime")
    if regime_id is not None and not isinstance(regime_id, str):
        regime_id = str(regime_id)

    snapshot = CognitionSnapshot(
        snapshot_id=_new_id("cog"),
        tenant_id=tenant_id,
        ai_id=ai_id,
        session_id=session_id,
        source=source,
        captured_at_ms=_now_ms(),
        regime_id=regime_id,
        prediction_error=_extract_pe(snapshots),
        learning_family=_extract_learning_family(snapshots),
        eval_alert_count=_extract_eval_alert_count(snapshots),
        memory_entries=_extract_memory_entries(snapshots),
        raw_readout=readout_bundle_json if isinstance(readout_bundle_json, dict) else {},
    )
    cognition_store(request).append(snapshot.to_json())
    return snapshot


def attach_cognition_routes(app: web.Application) -> None:
    """Wire the five ``/dlaas/v1/cognition/*`` endpoints."""
    app.router.add_get(
        "/dlaas/v1/cognition/snapshots", _handle_cognition_snapshots_list
    )
    app.router.add_get(
        "/dlaas/v1/cognition/timelines/regime", _handle_cognition_regime_timeline
    )
    app.router.add_get(
        "/dlaas/v1/cognition/learning-family", _handle_cognition_learning_family
    )
    app.router.add_get(
        "/dlaas/v1/cognition/experience-throughput",
        _handle_cognition_experience_throughput,
    )
    app.router.add_get(
        "/dlaas/v1/cognition/eval-trend", _handle_cognition_eval_trend
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_cognition_snapshots_list(request: web.Request) -> web.Response:
    items = _filter_snapshots(request)
    limit = _bounded_int(request.query.get("limit"), default=_DEFAULT_LIMIT, maximum=_MAX_LIMIT)
    offset = max(0, _bounded_int(request.query.get("offset"), default=0, maximum=10_000))
    page = items[offset : offset + limit]
    return web.json_response(
        {
            "status": "ok",
            "count": len(items),
            "items": page,
        }
    )


async def _handle_cognition_regime_timeline(request: web.Request) -> web.Response:
    items = _filter_snapshots(request)
    items.sort(key=lambda row: int(row.get("captured_at_ms", 0) or 0))
    timeline: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in items:
        regime_id = row.get("regime_id")
        captured = int(row.get("captured_at_ms", 0) or 0)
        if current is None or current["regime_id"] != regime_id:
            if current is not None:
                current["ended_at_ms"] = captured
                current["duration_ms"] = captured - int(current["started_at_ms"])
                timeline.append(current)
            current = {
                "regime_id": regime_id,
                "started_at_ms": captured,
                "ended_at_ms": captured,
                "duration_ms": 0,
                "sample_count": 1,
            }
        else:
            current["sample_count"] += 1
            current["ended_at_ms"] = captured
            current["duration_ms"] = captured - int(current["started_at_ms"])
    if current is not None:
        timeline.append(current)
    return web.json_response(
        {
            "status": "ok",
            "count": len(timeline),
            "items": timeline,
        }
    )


async def _handle_cognition_learning_family(request: web.Request) -> web.Response:
    items = _filter_snapshots(request)
    totals: dict[str, int] = {key: 0 for key in LEARNING_FAMILY_KEYS}
    sample_count = 0
    for row in items:
        family = row.get("learning_family") or {}
        if not isinstance(family, dict):
            continue
        sample_count += 1
        for key in LEARNING_FAMILY_KEYS:
            try:
                totals[key] += int(family.get(key, 0) or 0)
            except (TypeError, ValueError):
                continue
    return web.json_response(
        {
            "status": "ok",
            "sample_count": sample_count,
            "totals": totals,
        }
    )


async def _handle_cognition_experience_throughput(
    request: web.Request,
) -> web.Response:
    events = list(_iter_debug_events(request))
    window_ms = _resolve_window_ms(request)
    cutoff = _now_ms() - window_ms
    receipt_type = "experience.receipt.v1"
    reflection_type = "experience.reflection.v1"
    by_binding_day: dict[tuple[str, str], dict[str, int]] = {}
    for event in events:
        created = int(event.get("created_at_ms", 0) or 0)
        if created < cutoff:
            continue
        event_type = str(event.get("event_type", "") or "")
        if event_type not in {receipt_type, reflection_type}:
            continue
        binding = _extract_binding(event)
        day = _ms_to_day(created)
        bucket = by_binding_day.setdefault((binding, day), {"receipts": 0, "reflections": 0})
        if event_type == receipt_type:
            bucket["receipts"] += 1
        else:
            bucket["reflections"] += 1
    items = [
        {
            "binding": binding,
            "day": day,
            "receipts": counts["receipts"],
            "reflections": counts["reflections"],
        }
        for (binding, day), counts in sorted(by_binding_day.items())
    ]
    return web.json_response(
        {
            "status": "ok",
            "count": len(items),
            "items": items,
        }
    )


async def _handle_cognition_eval_trend(request: web.Request) -> web.Response:
    runs = list(_iter_eval_runs(request))
    ai_id_filter = request.query.get("ai_id", "").strip()
    window_ms = _resolve_window_ms(request)
    cutoff = _now_ms() - window_ms
    by_day: dict[str, dict[str, float]] = {}
    for run in runs:
        created = int(run.get("created_at_ms", 0) or 0)
        if created < cutoff:
            continue
        if ai_id_filter and str(run.get("ai_id", "")) != ai_id_filter:
            continue
        score_raw = run.get("score", 0.0) or 0.0
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0
        day = _ms_to_day(created)
        bucket = by_day.setdefault(
            day, {"day": day, "runs": 0.0, "score_sum": 0.0, "pass_count": 0.0}
        )
        bucket["runs"] += 1
        bucket["score_sum"] += score
        if score >= 0.5:
            bucket["pass_count"] += 1
    items = []
    for day in sorted(by_day):
        bucket = by_day[day]
        runs_count = int(bucket["runs"])
        items.append(
            {
                "day": day,
                "runs": runs_count,
                "average_score": (bucket["score_sum"] / runs_count) if runs_count else 0.0,
                "pass_rate": (bucket["pass_count"] / runs_count) if runs_count else 0.0,
            }
        )
    return web.json_response(
        {
            "status": "ok",
            "count": len(items),
            "items": items,
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_snapshots(request: web.Request) -> list[dict[str, Any]]:
    items = list(cognition_store(request))
    ai_id = request.query.get("ai_id", "").strip()
    tenant_id = request.query.get("tenant_id", "").strip()
    session_id = request.query.get("session_id", "").strip()
    source = request.query.get("source", "").strip()
    if ai_id:
        items = [row for row in items if str(row.get("ai_id", "")) == ai_id]
    if tenant_id:
        items = [row for row in items if str(row.get("tenant_id", "")) == tenant_id]
    if session_id:
        items = [row for row in items if str(row.get("session_id", "")) == session_id]
    if source:
        items = [row for row in items if str(row.get("source", "")) == source]
    since_ms = _bounded_int(request.query.get("since_ms"), default=0, maximum=2**63 - 1)
    until_ms = _bounded_int(request.query.get("until_ms"), default=0, maximum=2**63 - 1)
    if since_ms:
        items = [
            row for row in items if int(row.get("captured_at_ms", 0) or 0) >= since_ms
        ]
    if until_ms:
        items = [
            row for row in items if int(row.get("captured_at_ms", 0) or 0) <= until_ms
        ]
    window = request.query.get("window", "").strip()
    if window and not since_ms and not until_ms:
        window_ms = _parse_window(window) or _DEFAULT_WINDOW_MS
        cutoff = _now_ms() - window_ms
        items = [
            row for row in items if int(row.get("captured_at_ms", 0) or 0) >= cutoff
        ]
    items.sort(
        key=lambda row: int(row.get("captured_at_ms", 0) or 0), reverse=True
    )
    return items


def _iter_debug_events(request: web.Request) -> Iterable[dict[str, Any]]:
    store = request.app.get(_DEBUG_EVENTS_KEY) or {}
    for event in store.values():
        if hasattr(event, "to_json"):
            yield event.to_json()
        elif isinstance(event, dict):
            yield event


def _iter_eval_runs(request: web.Request) -> Iterable[dict[str, Any]]:
    store = request.app.get(_EVAL_RUNS_KEY) or {}
    for run in store.values():
        if hasattr(run, "to_json"):
            yield run.to_json()
        elif isinstance(run, dict):
            yield run


def _extract_pe(snapshots: dict[str, Any]) -> dict[str, float]:
    """Pull the 4 PE axes from the raw prediction_error snapshot.

    Missing fields default to 0.0 — the portal sparkline renders a
    flat line in that case, which is the correct visual signal.
    """
    pe_snap = snapshots.get("prediction_error") if isinstance(snapshots, dict) else None
    value = getattr(pe_snap, "value", None) if pe_snap is not None else None
    if value is None:
        return {"magnitude": 0.0, "task": 0.0, "relationship": 0.0, "regime": 0.0, "action": 0.0}
    return {
        "magnitude": _safe_float(getattr(value, "magnitude", 0.0)),
        "task": _safe_float(getattr(value, "task", 0.0)),
        "relationship": _safe_float(getattr(value, "relationship", 0.0)),
        "regime": _safe_float(getattr(value, "regime", 0.0)),
        "action": _safe_float(getattr(value, "action", 0.0)),
    }


def _extract_learning_family(snapshots: dict[str, Any]) -> dict[str, int]:
    """Count populated slots per learning family.

    The R12 learning family is a coarse heuristic mapping kernel slot
    families to the six top-level categories. We count "1" per slot
    family that has a non-empty active snapshot.
    """
    slot_to_family = {
        "active_mixture": "cognition",
        "response_assembly": "cognition",
        "temporal_abstraction": "cognition",
        "domain_knowledge": "knowledge",
        "case_memory": "knowledge",
        "retrieval_policy": "knowledge",
        "strategy_playbook": "strategy",
        "active_protocols": "protocol",
        "boundary_policy": "safety",
        "training_corpus": "training",
    }
    counts = {key: 0 for key in LEARNING_FAMILY_KEYS}
    if not isinstance(snapshots, dict):
        return counts
    for slot, family in slot_to_family.items():
        if snapshots.get(slot) is not None:
            counts[family] += 1
    return counts


def _extract_eval_alert_count(snapshots: dict[str, Any]) -> int:
    eval_snap = snapshots.get("eval_alerts") if isinstance(snapshots, dict) else None
    value = getattr(eval_snap, "value", None) if eval_snap is not None else None
    alerts = getattr(value, "alerts", None) if value is not None else None
    if alerts is None:
        return 0
    try:
        return len(alerts)
    except TypeError:
        return 0


def _extract_memory_entries(snapshots: dict[str, Any]) -> int:
    case_snap = snapshots.get("case_memory") if isinstance(snapshots, dict) else None
    value = getattr(case_snap, "value", None) if case_snap is not None else None
    entries = getattr(value, "entries", None) if value is not None else None
    if entries is None:
        return 0
    try:
        return len(entries)
    except TypeError:
        return 0


def _extract_binding(event: dict[str, Any]) -> str:
    """Recover the experience binding name from a debug event envelope."""
    fields = event.get("fields") or {}
    if isinstance(fields, dict):
        binding = fields.get("binding") or fields.get("binding_name")
        if binding:
            return str(binding)
    app_id = event.get("app_id") or ""
    return str(app_id) if app_id else "unknown"


def _parse_window(value: str) -> int | None:
    """Accept ``7d``, ``24h``, ``30m``, ``3600s`` etc. Returns ms or None."""
    value = value.strip().lower()
    if not value:
        return None
    if value.isdigit():
        return min(int(value) * 1000, _MAX_WINDOW_MS)
    unit = value[-1]
    try:
        amount = int(value[:-1])
    except ValueError:
        return None
    if amount <= 0:
        return None
    if unit == "s":
        ms = amount * 1000
    elif unit == "m":
        ms = amount * 60 * 1000
    elif unit == "h":
        ms = amount * 60 * 60 * 1000
    elif unit == "d":
        ms = amount * 24 * 60 * 60 * 1000
    else:
        return None
    return min(ms, _MAX_WINDOW_MS)


def _resolve_window_ms(request: web.Request) -> int:
    window_raw = request.query.get("window", "").strip()
    parsed = _parse_window(window_raw) if window_raw else None
    return parsed if parsed is not None else _DEFAULT_WINDOW_MS


def _bounded_int(raw: str | None, *, default: int, maximum: int) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if value < 0:
        return default
    return min(value, maximum)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ms_to_day(ms: int) -> str:
    """ISO date for ``ms`` (UTC)."""
    return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now_ms() -> int:
    return int(time.time() * 1000)
