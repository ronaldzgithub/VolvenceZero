"""Cognition snapshot API contract tests (R6).

Covers:

* The dispatch path actually writes a ``CognitionSnapshot`` for every
  successful interaction.
* The five ``/dlaas/v1/cognition/*`` endpoints honour ``ai_id`` /
  ``session_id`` / ``window`` filters and return the documented
  shape.
* The aggregation endpoints degrade gracefully on an empty store
  instead of 500-ing.
"""
from __future__ import annotations

import time

import pytest

from dlaas_platform_api import attach_dlaas_routes
from dlaas_platform_api.cognition import (
    COGNITION_SNAPSHOTS_KEY,
    LEARNING_FAMILY_KEYS,
)


@pytest.fixture
async def cognition_client(aiohttp_client):
    from lifeform_service.app import create_app
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    app = create_app(vertical=spec, max_sessions=8, idle_eviction_seconds=None)
    attach_dlaas_routes(app, default_ai_id="ai_cog")
    return await aiohttp_client(app)


def _seed_snapshot(
    app,
    *,
    ai_id: str,
    session_id: str = "sess_cog",
    regime_id: str = "regime.calm",
    captured_at_ms: int | None = None,
    tenant_id: str = "tenant_test",
    source: str = "interaction",
    learning_family: dict[str, int] | None = None,
) -> dict:
    """Append a hand-crafted snapshot row.

    Useful for the aggregation tests that need a deterministic
    timeline without spinning up real interactions.
    """
    rows = app[COGNITION_SNAPSHOTS_KEY]
    captured = captured_at_ms if captured_at_ms is not None else int(time.time() * 1000)
    family = learning_family or {key: 1 for key in LEARNING_FAMILY_KEYS}
    row = {
        "snapshot_id": f"cog_test_{len(rows):04d}",
        "tenant_id": tenant_id,
        "ai_id": ai_id,
        "session_id": session_id,
        "source": source,
        "captured_at_ms": captured,
        "regime_id": regime_id,
        "prediction_error": {
            "magnitude": 0.1,
            "task": 0.2,
            "relationship": 0.0,
            "regime": 0.0,
            "action": 0.05,
        },
        "learning_family": family,
        "eval_alert_count": 0,
        "memory_entries": 0,
        "raw_readout": {},
    }
    rows.append(row)
    return row


async def test_snapshots_empty_returns_ok(cognition_client):
    resp = await cognition_client.get("/dlaas/v1/cognition/snapshots")
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body == {"status": "ok", "count": 0, "items": []}


async def test_dispatch_writes_cognition_snapshot(cognition_client):
    """Every successful interaction must append exactly one snapshot row."""
    resp = await cognition_client.post(
        "/dlaas/v1/instances/ai_cog/interactions",
        json={
            "contract_id": "ctr_cog",
            "session_id": "sess_cog_1",
            "end_user_ref": "user_cog",
            "interaction_type": "chat",
            "human_brief": "你好，请帮我评估一下今天的状态。",
        },
    )
    assert resp.status == 200, await resp.text()

    listing = await cognition_client.get(
        "/dlaas/v1/cognition/snapshots?ai_id=ai_cog"
    )
    assert listing.status == 200, await listing.text()
    body = await listing.json()
    assert body["count"] == 1
    row = body["items"][0]
    assert row["ai_id"] == "ai_cog"
    assert row["session_id"] == "sess_cog_1"
    assert row["source"] == "interaction"
    assert set(row["prediction_error"]) >= {
        "magnitude",
        "task",
        "relationship",
        "regime",
        "action",
    }
    assert set(row["learning_family"]) == set(LEARNING_FAMILY_KEYS)
    assert "raw_readout" in row


async def test_snapshot_filters_ai_id_and_session(cognition_client):
    app = cognition_client.app
    _seed_snapshot(app, ai_id="ai_a", session_id="sess_a", regime_id="r1")
    _seed_snapshot(app, ai_id="ai_b", session_id="sess_b", regime_id="r1")
    _seed_snapshot(app, ai_id="ai_a", session_id="sess_b", regime_id="r2")

    only_ai = await cognition_client.get("/dlaas/v1/cognition/snapshots?ai_id=ai_a")
    body = await only_ai.json()
    assert body["count"] == 2

    only_session = await cognition_client.get(
        "/dlaas/v1/cognition/snapshots?ai_id=ai_a&session_id=sess_a"
    )
    body = await only_session.json()
    assert body["count"] == 1
    assert body["items"][0]["session_id"] == "sess_a"


async def test_regime_timeline_coalesces_consecutive_samples(cognition_client):
    """Three samples r1/r1/r2 must collapse into two timeline rows."""
    app = cognition_client.app
    t0 = int(time.time() * 1000)
    _seed_snapshot(app, ai_id="ai_t", regime_id="r1", captured_at_ms=t0)
    _seed_snapshot(app, ai_id="ai_t", regime_id="r1", captured_at_ms=t0 + 1000)
    _seed_snapshot(app, ai_id="ai_t", regime_id="r2", captured_at_ms=t0 + 2000)
    _seed_snapshot(app, ai_id="ai_t", regime_id="r2", captured_at_ms=t0 + 5000)

    resp = await cognition_client.get(
        "/dlaas/v1/cognition/timelines/regime?ai_id=ai_t&window=1d"
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["count"] == 2
    first, second = body["items"]
    assert first["regime_id"] == "r1"
    assert first["sample_count"] == 2
    assert second["regime_id"] == "r2"
    assert second["sample_count"] == 2
    assert second["duration_ms"] == 3000


async def test_learning_family_totals_sum_across_window(cognition_client):
    app = cognition_client.app
    _seed_snapshot(
        app,
        ai_id="ai_lf",
        learning_family={key: 2 for key in LEARNING_FAMILY_KEYS},
    )
    _seed_snapshot(
        app,
        ai_id="ai_lf",
        learning_family={key: 3 for key in LEARNING_FAMILY_KEYS},
    )
    resp = await cognition_client.get(
        "/dlaas/v1/cognition/learning-family?ai_id=ai_lf"
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["sample_count"] == 2
    assert all(body["totals"][key] == 5 for key in LEARNING_FAMILY_KEYS)


async def test_experience_throughput_empty_is_ok(cognition_client):
    """No debug events → empty items, status ok (not 500)."""
    resp = await cognition_client.get(
        "/dlaas/v1/cognition/experience-throughput"
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body == {"status": "ok", "count": 0, "items": []}


async def test_eval_trend_groups_by_day(cognition_client):
    """Two eval runs on the same day → one bucket with averaged score."""
    eval_a = await cognition_client.post(
        "/dlaas/v1/eval/runs",
        json={
            "gate_id": "gate_x",
            "ai_id": "ai_eval",
            "contract_id": "ctr_eval",
            "score": 0.9,
        },
    )
    assert eval_a.status == 201, await eval_a.text()
    eval_b = await cognition_client.post(
        "/dlaas/v1/eval/runs",
        json={
            "gate_id": "gate_x",
            "ai_id": "ai_eval",
            "contract_id": "ctr_eval",
            "score": 0.3,
        },
    )
    assert eval_b.status == 201, await eval_b.text()

    trend = await cognition_client.get(
        "/dlaas/v1/cognition/eval-trend?ai_id=ai_eval&window=7d"
    )
    assert trend.status == 200, await trend.text()
    body = await trend.json()
    assert body["count"] == 1
    bucket = body["items"][0]
    assert bucket["runs"] == 2
    assert bucket["average_score"] == pytest.approx(0.6, rel=1e-3)
    # 0.9 passes (>=0.5), 0.3 does not → 1/2 pass rate
    assert bucket["pass_rate"] == pytest.approx(0.5, rel=1e-3)


async def test_tenant_filter_isolates_rows(cognition_client):
    app = cognition_client.app
    _seed_snapshot(app, ai_id="ai_t1", tenant_id="tenant_one")
    _seed_snapshot(app, ai_id="ai_t2", tenant_id="tenant_two")
    resp = await cognition_client.get(
        "/dlaas/v1/cognition/snapshots?tenant_id=tenant_one"
    )
    body = await resp.json()
    assert body["count"] == 1
    assert body["items"][0]["tenant_id"] == "tenant_one"


async def test_window_parsing_accepts_h_m_s_units(cognition_client):
    """``window=1h`` should be honoured by the regime timeline filter."""
    app = cognition_client.app
    very_old = int(time.time() * 1000) - 24 * 60 * 60 * 1000  # 24h ago
    fresh = int(time.time() * 1000) - 30 * 60 * 1000  # 30 minutes ago
    _seed_snapshot(app, ai_id="ai_w", regime_id="r_old", captured_at_ms=very_old)
    _seed_snapshot(app, ai_id="ai_w", regime_id="r_new", captured_at_ms=fresh)

    resp = await cognition_client.get(
        "/dlaas/v1/cognition/timelines/regime?ai_id=ai_w&window=1h"
    )
    body = await resp.json()
    # 1h window drops the 24h-old r_old entry.
    assert body["count"] == 1
    assert body["items"][0]["regime_id"] == "r_new"
