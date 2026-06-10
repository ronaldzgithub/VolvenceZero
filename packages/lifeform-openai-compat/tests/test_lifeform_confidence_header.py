"""Kernel-PE confidence surface on the OpenAI-compat adapter.

Upstream half of deploy debt ``D-collab-pe``: the bridge must read the
PE owner's calibrated forward-looking confidence
(``PredictionErrorSnapshot.next_prediction.confidence``) off the turn
result's published snapshots and surface it as the
``x-lifeform-confidence`` response header — and must surface NOTHING
(no fabricated literal) when the kernel published no PE snapshot.

Pure-helper tests with lightweight fakes; no aiohttp app or kernel
session required (mirrors ``test_dispatch_cognition_extra.py`` upstream
in dlaas-platform-api).
"""

from __future__ import annotations

from dataclasses import dataclass

from lifeform_openai_compat.router import _lifeform_telemetry_headers
from lifeform_openai_compat.session_bridge import (
    LifeformCompletionResult,
    _kernel_confidence_from_result,
)


@dataclass(frozen=True)
class _FakeSnapshot:
    value: object


class _NextPrediction:
    def __init__(self, confidence: object) -> None:
        self.confidence = confidence


class _PEValue:
    def __init__(self, confidence: object = None) -> None:
        self.next_prediction = (
            _NextPrediction(confidence) if confidence is not None else None
        )


class _FakeTurnResult:
    def __init__(self, snapshots: dict[str, object] | None) -> None:
        self.active_snapshots = snapshots


class _FakeResponse:
    system_fingerprint = "lifeform:test"


class _FakeResolution:
    kind = "explicit"


def _result(confidence: float | None) -> LifeformCompletionResult:
    return LifeformCompletionResult(
        response=_FakeResponse(),  # type: ignore[arg-type]
        resolution=_FakeResolution(),  # type: ignore[arg-type]
        active_regime="casual_social",
        active_abstract_action=None,
        pe_magnitude=0.12,
        rationale_tags=(),
        confidence=confidence,
    )


def test_kernel_confidence_reads_next_prediction() -> None:
    result = _FakeTurnResult(
        {"prediction_error": _FakeSnapshot(_PEValue(confidence=0.66))}
    )
    assert _kernel_confidence_from_result(result) == 0.66


def test_kernel_confidence_none_when_unpublished() -> None:
    assert _kernel_confidence_from_result(_FakeTurnResult(None)) is None
    assert _kernel_confidence_from_result(_FakeTurnResult({})) is None
    assert (
        _kernel_confidence_from_result(
            _FakeTurnResult({"prediction_error": _FakeSnapshot(_PEValue())})
        )
        is None
    )
    assert (
        _kernel_confidence_from_result(
            _FakeTurnResult({"prediction_error": _FakeSnapshot(None)})
        )
        is None
    )


def test_kernel_confidence_clamps_and_rejects_non_numeric() -> None:
    assert (
        _kernel_confidence_from_result(
            _FakeTurnResult({"prediction_error": _FakeSnapshot(_PEValue(confidence=1.4))})
        )
        == 1.0
    )
    assert (
        _kernel_confidence_from_result(
            _FakeTurnResult(
                {"prediction_error": _FakeSnapshot(_PEValue(confidence=-0.3))}
            )
        )
        == 0.0
    )
    assert (
        _kernel_confidence_from_result(
            _FakeTurnResult(
                {"prediction_error": _FakeSnapshot(_PEValue(confidence=True))}
            )
        )
        is None
    )
    assert (
        _kernel_confidence_from_result(
            _FakeTurnResult(
                {"prediction_error": _FakeSnapshot(_PEValue(confidence="hi"))}
            )
        )
        is None
    )


def test_telemetry_headers_carry_confidence_when_present() -> None:
    headers = _lifeform_telemetry_headers(_result(confidence=0.7321))
    assert headers["x-lifeform-confidence"] == "0.7321"


def test_telemetry_headers_omit_confidence_when_absent() -> None:
    headers = _lifeform_telemetry_headers(_result(confidence=None))
    assert "x-lifeform-confidence" not in headers
    # The rest of the telemetry surface is unchanged.
    assert headers["x-lifeform-mode"] == "lifeform"
    assert headers["x-lifeform-pe-magnitude"] == "0.1200"
