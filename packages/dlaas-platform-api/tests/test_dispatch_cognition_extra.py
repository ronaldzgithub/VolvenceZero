"""Unit tests for the interaction-response cognition enrichment.

Covers the "subjectivity transport gap" fix: the chat / apprentice /
proactive-followup dispatch handlers must surface the published
``expression_intent`` + a compact prediction-error projection +
the ``relationship_state`` description on the response ``extra`` block,
so the product layer can drive the visible avatar from real internal
state instead of hardcoded presets.

These exercise the pure helpers with lightweight fakes — no aiohttp app
or kernel session required.
"""

from __future__ import annotations

from dataclasses import dataclass

from dlaas_platform_api.dispatch import (
    _cognition_extra,
    _prediction_error_compact,
    _snapshot_description,
    _snapshot_value_field,
)


@dataclass(frozen=True)
class _FakeSnapshot:
    value: object


class _ResponseAssemblyValue:
    def __init__(self, expression_intent: str) -> None:
        self.expression_intent = expression_intent
        self.description = "regime=acquaintance_building; depth=brief"


class _PredictionErrorValue:
    def __init__(self, magnitude: float, relationship: float, task: float) -> None:
        self.magnitude = magnitude
        self.relationship = relationship
        self.task = task
        self.regime = 0.0
        self.action = 0.0


class _RelationshipValue:
    def __init__(self, description: str) -> None:
        self.description = description


class _FakeSession:
    def __init__(self, snapshots: dict[str, object] | None) -> None:
        self.latest_active_snapshots = snapshots


def test_cognition_extra_surfaces_expression_intent_pe_and_relationship() -> None:
    snapshots = {
        "response_assembly": _FakeSnapshot(
            _ResponseAssemblyValue(expression_intent="support-first")
        ),
        "prediction_error": _FakeSnapshot(
            _PredictionErrorValue(magnitude=0.42, relationship=0.7, task=0.1)
        ),
        "relationship_state": _FakeSnapshot(
            _RelationshipValue("warm; trust trending up; 3 open loops")
        ),
    }
    extra = _cognition_extra(_FakeSession(snapshots))
    assert extra["expression_intent"] == "support-first"
    assert extra["prediction_error"] == {
        "magnitude": 0.42,
        "relationship": 0.7,
        "task": 0.1,
    }
    assert extra["relationship_brief"] == "warm; trust trending up; 3 open loops"


def test_cognition_extra_is_empty_when_session_has_no_snapshots() -> None:
    assert _cognition_extra(_FakeSession(None)) == {}
    assert _cognition_extra(object()) == {}


def test_cognition_extra_skips_absent_slots() -> None:
    # response_assembly present, the rest absent → only expression_intent.
    snapshots = {
        "response_assembly": _FakeSnapshot(
            _ResponseAssemblyValue(expression_intent="direct-answer")
        ),
    }
    extra = _cognition_extra(_FakeSession(snapshots))
    assert extra == {"expression_intent": "direct-answer"}


def test_prediction_error_compact_only_numeric_axes() -> None:
    class _Partial:
        magnitude = 0.5
        relationship = None  # absent axis
        task = True  # booleans are not real PE scalars

    compact = _prediction_error_compact(_FakeSnapshot(_Partial()))
    assert compact == {"magnitude": 0.5}


def test_prediction_error_compact_none_when_no_value() -> None:
    assert _prediction_error_compact(None) is None
    assert _prediction_error_compact(_FakeSnapshot(None)) is None


def test_snapshot_value_field_and_description_handle_none() -> None:
    assert _snapshot_value_field(None, "expression_intent") is None
    assert _snapshot_value_field(_FakeSnapshot(None), "x") is None
    assert _snapshot_description(None) == ""
    assert _snapshot_description(_FakeSnapshot(None)) == ""
