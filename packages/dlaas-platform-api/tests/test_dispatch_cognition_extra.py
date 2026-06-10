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
    _kernel_confidence,
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


class _NextPrediction:
    """Mirrors ``PredictedOutcome``'s public ``confidence`` field."""

    def __init__(self, confidence: object) -> None:
        self.confidence = confidence


class _PEError:
    """Mirrors the nested ``PredictionError`` axes on the real snapshot."""

    def __init__(self, magnitude: float, relationship_error: float, task_error: float) -> None:
        self.magnitude = magnitude
        self.relationship_error = relationship_error
        self.task_error = task_error


class _KernelPEValue:
    """Real-shape ``PredictionErrorSnapshot`` stand-in: axes nested on
    ``error``, calibrated confidence on ``next_prediction``."""

    def __init__(
        self,
        magnitude: float,
        relationship_error: float,
        task_error: float,
        confidence: object = None,
    ) -> None:
        self.error = _PEError(magnitude, relationship_error, task_error)
        self.next_prediction = (
            _NextPrediction(confidence) if confidence is not None else None
        )


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


def test_cognition_extra_surfaces_kernel_pe_shape_and_confidence() -> None:
    """The REAL kernel snapshot shape (nested ``error.*`` axes +
    ``next_prediction.confidence``) must project into both the compact
    PE block and the D-collab-pe confidence fields."""
    snapshots = {
        "prediction_error": _FakeSnapshot(
            _KernelPEValue(
                magnitude=0.31,
                relationship_error=0.22,
                task_error=0.05,
                confidence=0.73,
            )
        ),
    }
    extra = _cognition_extra(_FakeSession(snapshots))
    assert extra["prediction_error"] == {
        "magnitude": 0.31,
        "relationship": 0.22,
        "task": 0.05,
    }
    assert extra["confidence"] == 0.73
    assert extra["confidence_origin"] == "kernel_pe"


def test_cognition_extra_omits_confidence_when_unpublished() -> None:
    """No PE snapshot / no next_prediction => the confidence fields are
    ABSENT (never fabricated)."""
    extra = _cognition_extra(
        _FakeSession(
            {
                "prediction_error": _FakeSnapshot(
                    _KernelPEValue(magnitude=0.1, relationship_error=0.0, task_error=0.0)
                )
            }
        )
    )
    assert "confidence" not in extra
    assert "confidence_origin" not in extra


def test_kernel_confidence_clamps_and_rejects_non_numeric() -> None:
    assert _kernel_confidence(None) is None
    assert _kernel_confidence(_FakeSnapshot(None)) is None
    assert (
        _kernel_confidence(
            _FakeSnapshot(_KernelPEValue(0.0, 0.0, 0.0, confidence=True))
        )
        is None
    )
    assert (
        _kernel_confidence(
            _FakeSnapshot(_KernelPEValue(0.0, 0.0, 0.0, confidence="high"))
        )
        is None
    )
    assert (
        _kernel_confidence(_FakeSnapshot(_KernelPEValue(0.0, 0.0, 0.0, confidence=1.7)))
        == 1.0
    )
    assert (
        _kernel_confidence(_FakeSnapshot(_KernelPEValue(0.0, 0.0, 0.0, confidence=-0.2)))
        == 0.0
    )


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
