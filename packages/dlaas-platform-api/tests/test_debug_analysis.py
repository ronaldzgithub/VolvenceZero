"""Tests for the debug-analysis pipeline (D24).

Covers:
- DeterministicAnalyzer still works and is the default.
- LlmAnalyzer parses a real JSON model response into recommendations +
  structured version suggestions.
- LlmAnalyzer degrades honestly (mode strings) when unconfigured, when the
  client raises, and when the client returns unusable content.
- The env gate selects the right analyzer / mode.
"""

from __future__ import annotations

from typing import Any

import pytest

from dlaas_platform_contracts import DebugAnalysisRequest

from dlaas_platform_api.debug_analysis import (
    DeterministicAnalyzer,
    LlmAnalyzer,
    build_debug_analysis,
)


def _request(prompt: str = "Why did the handoff fail for repair30?") -> DebugAnalysisRequest:
    return DebugAnalysisRequest.from_json(
        {
            "prompt": prompt,
            "selectors": {"app_id": "repair30", "event_types": ["handoff"]},
        }
    )


_EVIDENCE: dict[str, Any] = {
    "debug_events": [
        {
            "debug_event_id": "evt-1",
            "event_type": "handoff",
            "fields": {"ok": False, "handoff_trigger": "boundary_block"},
        }
    ],
    "audit_events": [],
}


class _FakeClient:
    """Records the prompts and returns a canned JSON object."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, str]] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return self.payload


class _RaisingClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise RuntimeError("upstream 503")


# ---------------------------------------------------------------------------
# Deterministic baseline
# ---------------------------------------------------------------------------


def test_deterministic_is_default_and_well_formed() -> None:
    result = build_debug_analysis(analysis_request=_request(), evidence=_EVIDENCE)
    assert result["analyzer_id"] == "deterministic"
    assert result["analysis_mode"] == "deterministic"
    assert result["recommendations"]  # non-empty
    assert result["prompt_template"] == "prompts/debug_analysis.md"


# ---------------------------------------------------------------------------
# LlmAnalyzer success path
# ---------------------------------------------------------------------------


def test_llm_analyzer_parses_structured_output() -> None:
    client = _FakeClient(
        {
            "recommendations": [
                "The boundary block fired before handoff; inspect evt-1.",
                "  ",  # blank is dropped
            ],
            "version_suggestions": [
                {
                    "issue_area": "app",
                    "evidence_refs": ["evt-1"],
                    "recommended_owner": "repair30",
                    "confidence": 0.82,
                    "proposed_next_test": "Replay session and confirm 2xx upstream.",
                },
                {
                    "issue_area": "totally-made-up",  # coerced to unknown
                    "evidence_refs": [],
                    "recommended_owner": "",  # defaults to operator
                    "confidence": 5,  # clamped to 1.0
                    "proposed_next_test": "n/a",
                },
            ],
        }
    )
    result = build_debug_analysis(
        analysis_request=_request(),
        evidence=_EVIDENCE,
        analyzer=LlmAnalyzer(client=client),
    )
    assert result["analysis_mode"] == "llm"
    assert result["analyzer_id"] == "llm"
    assert result["analyzer_version"] == "v1"
    assert result["recommendations"] == (
        "The boundary block fired before handoff; inspect evt-1.",
    )
    suggestions = result["version_suggestions"]
    assert len(suggestions) == 2
    assert suggestions[0]["issue_area"] == "app"
    assert suggestions[0]["evidence_refs"] == ["evt-1"]
    assert suggestions[0]["confidence"] == pytest.approx(0.82)
    # coercions
    assert suggestions[1]["issue_area"] == "unknown"
    assert suggestions[1]["recommended_owner"] == "operator"
    assert suggestions[1]["confidence"] == pytest.approx(1.0)
    # The client saw both centralized prompts.
    assert client.calls
    assert "DLaaS debug analyst" in client.calls[0]["system"]
    assert "Evidence summary JSON" in client.calls[0]["user"]


# ---------------------------------------------------------------------------
# LlmAnalyzer fallback paths (honest mode strings)
# ---------------------------------------------------------------------------


def test_llm_analyzer_unconfigured_falls_back() -> None:
    result = build_debug_analysis(
        analysis_request=_request(),
        evidence=_EVIDENCE,
        analyzer=LlmAnalyzer(client=None),
    )
    assert result["analysis_mode"] == "llm_unconfigured_fallback"
    assert result["recommendations"]  # deterministic content present


def test_llm_analyzer_client_error_falls_back_and_reports() -> None:
    result = build_debug_analysis(
        analysis_request=_request(),
        evidence=_EVIDENCE,
        analyzer=LlmAnalyzer(client=_RaisingClient()),
    )
    assert result["analysis_mode"] == "llm_error_fallback"
    assert "LLM debug analysis failed" in result["recommendations"][0]


def test_llm_analyzer_empty_output_falls_back() -> None:
    client = _FakeClient({"recommendations": [], "version_suggestions": []})
    result = build_debug_analysis(
        analysis_request=_request(),
        evidence=_EVIDENCE,
        analyzer=LlmAnalyzer(client=client),
    )
    assert result["analysis_mode"] == "llm_error_fallback"


def test_llm_analyzer_non_object_output_falls_back() -> None:
    class _BadClient:
        def complete_json(self, *, system_prompt: str, user_prompt: str) -> Any:
            return ["not", "a", "dict"]

    result = build_debug_analysis(
        analysis_request=_request(),
        evidence=_EVIDENCE,
        analyzer=LlmAnalyzer(client=_BadClient()),
    )
    assert result["analysis_mode"] == "llm_error_fallback"


# ---------------------------------------------------------------------------
# Env gate
# ---------------------------------------------------------------------------


def test_env_gate_selects_llm_unconfigured_when_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DLAAS_DEBUG_ANALYZER", "llm")
    for key in (
        "DLAAS_DEBUG_LLM_API_KEY",
        "DLAAS_DEBUG_LLM_BASE_URL",
        "DLAAS_DEBUG_LLM_MODEL",
        "PROTOCOL_LLM_API_KEY",
        "PROTOCOL_LLM_BASE_URL",
        "PROTOCOL_LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    result = build_debug_analysis(analysis_request=_request(), evidence=_EVIDENCE)
    assert result["analyzer_id"] == "llm"
    assert result["analysis_mode"] == "llm_unconfigured_fallback"


def test_env_gate_default_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DLAAS_DEBUG_ANALYZER", raising=False)
    result = build_debug_analysis(analysis_request=_request(), evidence=_EVIDENCE)
    assert result["analyzer_id"] == "deterministic"


def test_deterministic_analyzer_direct_use() -> None:
    analyzer = DeterministicAnalyzer()
    out = analyzer.analyze(
        analysis_request=_request(),
        evidence_summary={"debug_event_count": 0, "failed_debug_event_count": 0},
        prompt_preview="preview",
        intent_tags=(),
    )
    assert out["analysis_mode"] == "deterministic"
    assert out["recommendations"]
