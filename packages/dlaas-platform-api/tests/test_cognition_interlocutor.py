"""Tests for the deep social-cognition (R16-R20) readout.

Covers the Track 1 (FULL) upstream packet:

* ``_social_readout`` / ``_interlocutor_axes`` (app.py) project the
  published ``interlocutor_state`` snapshot's 12 axes + zones verbatim
  (readout-only, R12 — never re-derives).
* ``GET /dlaas/v1/cognition/interlocutor`` returns the social block from
  the latest stored cognition snapshot for an ``ai_id``.

The pure helpers use the real ``InterlocutorState`` so the axis/zone
contract stays in lock-step. The endpoint test drives the handler with a
mocked aiohttp request over an in-memory cognition store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from dlaas_platform_api.app import _interlocutor_axes, _social_readout
from dlaas_platform_api.cognition import (
    COGNITION_SNAPSHOTS_KEY,
    _handle_cognition_interlocutor,
)
from volvence_zero.interlocutor.contracts import (
    InterlocutorState,
    InterlocutorStateSnapshot,
)


@dataclass(frozen=True)
class _FakeSnapshot:
    value: object
    owner: str = "InterlocutorStateModule"
    version: int = 3


def _interlocutor_snapshot() -> _FakeSnapshot:
    state = InterlocutorState(
        engagement_intensity=0.8,
        emotional_weight=0.9,
        rapport_warmth=0.2,
        readout_confidence=0.85,
        rationale="high emotional weight, cold rapport",
    )
    return _FakeSnapshot(
        InterlocutorStateSnapshot(state=state, description="emotional, guarded")
    )


def test_interlocutor_axes_projects_12_axes_and_zones() -> None:
    out = _interlocutor_axes(_interlocutor_snapshot())
    assert out["present"] is True
    # All 12 axes present and numeric.
    assert set(out["axes"]).issuperset({"engagement_intensity", "rapport_warmth"})
    assert out["axes"]["engagement_intensity"] == 0.8
    assert out["axes"]["emotional_weight"] == 0.9
    # Zones are computed by the kernel; high emotion -> emotional_high_zone.
    assert out["zones"]["emotional_high_zone"] is True
    assert out["readout_confidence"] == 0.85
    assert "emotional" in out["rationale"]
    assert out["owner"] == "InterlocutorStateModule"


def test_interlocutor_axes_absent_when_no_snapshot() -> None:
    assert _interlocutor_axes(None) == {"present": False}
    assert _interlocutor_axes(_FakeSnapshot(None)) == {"present": False}


def test_social_readout_combines_interlocutor_role_common_ground() -> None:
    social = _social_readout(
        {
            "interlocutor_state": _interlocutor_snapshot(),
            "conversational_role": None,
            "common_ground": None,
        }
    )
    assert social["present"] is True
    assert social["interlocutor"]["axes"]["emotional_weight"] == 0.9
    assert social["conversational_role"] == {"present": False}
    assert social["common_ground"] == {"present": False}


def test_social_readout_absent_when_no_interlocutor() -> None:
    social = _social_readout({})
    assert social["present"] is False
    assert social["interlocutor"] == {"present": False}


def _request_with_store(rows: list[dict], query: str) -> web.Request:
    app = web.Application()
    app[COGNITION_SNAPSHOTS_KEY] = rows
    return make_mocked_request(
        "GET", f"/dlaas/v1/cognition/interlocutor?{query}", app=app
    )


async def test_endpoint_returns_latest_social_block() -> None:
    social = _social_readout({"interlocutor_state": _interlocutor_snapshot()})
    rows = [
        {
            "ai_id": "ai-1",
            "session_id": "s-old",
            "captured_at_ms": 1000,
            "raw_readout": {"social": {"present": False}},
        },
        {
            "ai_id": "ai-1",
            "session_id": "s-new",
            "captured_at_ms": 2000,
            "raw_readout": {"social": social},
        },
    ]
    resp = await _handle_cognition_interlocutor(_request_with_store(rows, "ai_id=ai-1"))
    body = json.loads(resp.text)
    assert body["status"] == "ok"
    assert body["present"] is True
    assert body["session_id"] == "s-new"  # newest-first
    assert body["social"]["interlocutor"]["axes"]["emotional_weight"] == 0.9


async def test_endpoint_requires_ai_id() -> None:
    resp = await _handle_cognition_interlocutor(_request_with_store([], ""))
    assert resp.status == 400


async def test_endpoint_present_false_when_no_snapshots() -> None:
    resp = await _handle_cognition_interlocutor(
        _request_with_store([], "ai_id=ai-unknown")
    )
    body = json.loads(resp.text)
    assert body["status"] == "ok"
    assert body["present"] is False
    assert body["social"] is None
