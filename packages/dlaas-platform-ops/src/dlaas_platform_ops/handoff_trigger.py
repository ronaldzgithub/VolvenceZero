"""Handoff escalation policy driven by ``rupture_state`` snapshots.

The platform NEVER imports the kernel's rupture_state owner — it
reads the snapshot through the public ``brain_session.runner.active_snapshots()``
surface and looks up fields by name. This keeps the platform tier
honest about R8 (single owner; consume snapshots, do not become a
second owner) and makes the trigger code resilient to internal
reshapes of ``rupture_state`` so long as the published snapshot
keys stay stable.

Threshold structure:

* If ``rupture_kind`` is ``UNSAFE`` → escalate at the first turn.
* If ``rupture_kind`` is one of the high-severity kinds
  (``MISSED`` / ``ABANDONED`` / ``BOUNDARY_CONFLICT``) AND
  ``severity`` ≥ 0.6 → escalate.
* If ``confidence_aggregate`` (computed from recent dialogue trace)
  ≤ 0.4 across 3+ recent responses → escalate.
* Otherwise → no escalation.

The escalation policy is intentionally conservative: a false
positive triggers an operator ticket (cheap), a false negative
silently fails a user (expensive).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


_HIGH_SEVERITY_KINDS: frozenset[str] = frozenset(
    {"missed", "abandoned", "boundary_conflict", "unsafe"}
)
_AUTO_ESCALATE_KINDS: frozenset[str] = frozenset({"unsafe"})

_SEVERITY_THRESHOLD = 0.6
_CONFIDENCE_THRESHOLD = 0.4
_MIN_RESPONSES_FOR_LOW_CONF = 3


@dataclass(frozen=True)
class HandoffDecision:
    """Outcome of evaluating ``rupture_state`` against escalation thresholds."""

    should_escalate: bool
    rupture_kind: str = ""
    severity: float = 0.0
    trigger_reason: str = ""
    trigger_details: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "should_escalate": self.should_escalate,
            "rupture_kind": self.rupture_kind,
            "severity": self.severity,
            "trigger_reason": self.trigger_reason,
            "trigger_details": dict(self.trigger_details),
        }


def evaluate_session(
    *,
    session: Any,
    recent_response_ids: tuple[str, ...] = (),
    confidence_aggregate: float | None = None,
) -> HandoffDecision:
    """Evaluate the handoff trigger on a live ``LifeformSession``.

    The function looks up ``rupture_state`` via the public snapshot
    surface and then applies the typed thresholds. Failure to read
    the snapshot (kernel running, but rupture_state owner not wired)
    yields ``should_escalate=False`` with a trigger_details note —
    we never escalate on missing data; the operator UI can still
    create a ticket manually.
    """
    snapshot = _read_rupture_snapshot(session)
    if snapshot is None:
        return HandoffDecision(
            should_escalate=False,
            trigger_reason="no_rupture_snapshot",
            trigger_details={
                "note": "rupture_state owner is not active for this session"
            },
        )
    rupture_kind = _safe_str(snapshot.get("rupture_kind"))
    severity = _safe_float(snapshot.get("severity"))
    if rupture_kind in _AUTO_ESCALATE_KINDS:
        return HandoffDecision(
            should_escalate=True,
            rupture_kind=rupture_kind,
            severity=severity,
            trigger_reason="rupture_unsafe",
            trigger_details={
                "rupture_state": snapshot,
                "recent_response_ids": list(recent_response_ids),
            },
        )
    if (
        rupture_kind in _HIGH_SEVERITY_KINDS
        and severity >= _SEVERITY_THRESHOLD
    ):
        return HandoffDecision(
            should_escalate=True,
            rupture_kind=rupture_kind,
            severity=severity,
            trigger_reason=f"rupture_severity:{rupture_kind}",
            trigger_details={
                "rupture_state": snapshot,
                "severity_threshold": _SEVERITY_THRESHOLD,
            },
        )
    if (
        confidence_aggregate is not None
        and len(recent_response_ids) >= _MIN_RESPONSES_FOR_LOW_CONF
        and confidence_aggregate <= _CONFIDENCE_THRESHOLD
    ):
        return HandoffDecision(
            should_escalate=True,
            rupture_kind=rupture_kind,
            severity=severity,
            trigger_reason="low_confidence_streak",
            trigger_details={
                "confidence_aggregate": confidence_aggregate,
                "recent_response_ids": list(recent_response_ids),
                "threshold": _CONFIDENCE_THRESHOLD,
            },
        )
    return HandoffDecision(
        should_escalate=False,
        rupture_kind=rupture_kind,
        severity=severity,
        trigger_reason="below_threshold",
    )


def _read_rupture_snapshot(session: Any) -> dict[str, Any] | None:
    """Pull the ``rupture_state`` snapshot via the public Brain facade.

    The platform NEVER imports kernel internals; it reaches for the
    runner's ``active_snapshots()`` mapping by name and pulls fields
    via :func:`getattr` so a kernel-internal class change does not
    break this read as long as the snapshot keys stay stable.
    """
    brain_session = getattr(session, "_brain_session", None)
    if brain_session is None:
        return None
    runner = getattr(brain_session, "runner", None)
    if runner is None:
        return None
    try:
        snapshots = runner.active_snapshots()
    except AttributeError:
        return None
    if not isinstance(snapshots, Mapping):
        return None
    snap = snapshots.get("rupture_state")
    if snap is None:
        return None
    value = getattr(snap, "value", snap)
    out: dict[str, Any] = {}
    for field_name in (
        "rupture_kind",
        "severity",
        "confidence",
        "evidence_summary",
        "is_active",
    ):
        v = getattr(value, field_name, None)
        if v is None:
            continue
        if hasattr(v, "value"):  # nested enum
            out[field_name] = v.value
        else:
            out[field_name] = v
    return out


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    if hasattr(value, "value"):
        return str(value.value).lower()
    return str(value).lower()


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


__all__ = ["HandoffDecision", "evaluate_session"]
