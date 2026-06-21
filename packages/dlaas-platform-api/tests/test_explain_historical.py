"""Unit tests for the honest historical ``/explain`` resolution.

Covers the two new pure seams behind ``_handle_explain``:

* ``_persisted_turn_snapshots`` — reads the cognition snapshot store for one
  ``(ai_id, session_id)`` oldest-first, so an integer ``turn_index`` maps to
  a REAL recorded turn (and out-of-range becomes a 404 at the handler).
* ``_explain_chain_from_readout`` — rebuilds the explain chain from the
  persisted readout bundle JSON (a faithful summary projection), never
  re-deriving owner internals.

Lightweight fakes — no aiohttp app or kernel session required.
"""

from __future__ import annotations

from types import SimpleNamespace

from dlaas_platform_api.app import (
    _as_mapping,
    _explain_chain_from_readout,
    _persisted_turn_snapshots,
)
from dlaas_platform_api.cognition import COGNITION_SNAPSHOTS_KEY


def _row(ai_id: str, session_id: str, captured_at_ms: int, **extra: object) -> dict:
    return {
        "ai_id": ai_id,
        "session_id": session_id,
        "captured_at_ms": captured_at_ms,
        **extra,
    }


def _request_with_store(rows: list[dict]) -> object:
    return SimpleNamespace(app={COGNITION_SNAPSHOTS_KEY: list(rows)})


def test_persisted_turn_snapshots_filters_and_orders_oldest_first() -> None:
    rows = [
        _row("ai_a", "s1", 300, snapshot_id="c3"),
        _row("ai_a", "s1", 100, snapshot_id="c1"),
        _row("ai_a", "s2", 150, snapshot_id="other_session"),
        _row("ai_b", "s1", 120, snapshot_id="other_ai"),
        _row("ai_a", "s1", 200, snapshot_id="c2"),
    ]
    request = _request_with_store(rows)
    out = _persisted_turn_snapshots(request, ai_id="ai_a", session_id="s1")
    assert [r["snapshot_id"] for r in out] == ["c1", "c2", "c3"]


def test_persisted_turn_snapshots_empty_when_store_missing() -> None:
    request = SimpleNamespace(app={})
    assert _persisted_turn_snapshots(request, ai_id="ai_a", session_id="s1") == []


def test_explain_chain_from_readout_projects_summary_fields() -> None:
    readout = {
        "cognition": {
            "active_regime": "acquaintance_building",
            "expression_intent": "support-first",
            "prediction_error": {"description": "pe magnitude 0.42"},
        },
        "knowledge": {"retrieval": {"description": "2 case hits"}},
        "strategy": {
            "playbook": {"description": "warmth-led"},
            "matched_rule_count": 3,
        },
        "safety": {"boundary_policy": {"description": "no medical advice"}},
        "protocol": {
            "active_protocols": [{"protocol_id": "p1"}],
            "boundary_union_ids": ["b1"],
            "strategy_weights": [],
            "description": "proto",
        },
    }
    chain = _explain_chain_from_readout(readout)
    by_step = {step["step"]: step for step in chain}
    assert by_step["regime"]["regime_id"] == "acquaintance_building"
    assert by_step["response"]["expression_intent"] == "support-first"
    assert by_step["strategy"]["matched_rule_count"] == 3
    assert by_step["strategy"]["description"] == "warmth-led"
    assert by_step["knowledge"]["description"] == "2 case hits"
    assert by_step["boundary"]["description"] == "no medical advice"
    assert by_step["prediction_error"]["description"] == "pe magnitude 0.42"
    assert by_step["protocol"]["active_protocols"] == [{"protocol_id": "p1"}]


def test_explain_chain_from_readout_tolerates_partial_readout() -> None:
    # An empty / partial persisted readout must never raise; every field
    # falls back to a typed default.
    chain = _explain_chain_from_readout({})
    by_step = {step["step"]: step for step in chain}
    assert by_step["regime"]["regime_id"] is None
    assert by_step["strategy"]["matched_rule_count"] == 0
    assert by_step["response"]["expression_intent"] is None
    # Same step ordering as the live chain.
    assert [s["step"] for s in chain] == [
        "input_event",
        "regime",
        "protocol",
        "boundary",
        "strategy",
        "knowledge",
        "response",
        "prediction_error",
    ]


def test_as_mapping_coerces_non_mappings() -> None:
    assert _as_mapping({"a": 1}) == {"a": 1}
    assert _as_mapping(None) == {}
    assert _as_mapping("not a dict") == {}
    assert _as_mapping(5) == {}
