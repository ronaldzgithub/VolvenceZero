"""Debug analysis pipeline for DLaaS platform evidence.

The pipeline is split into four explicit stages so the analysis boundary
stays auditable:

    resolve_evidence  -> redact_evidence -> render_prompt -> analyze

`resolve_evidence` is owned by the route handler in `app.py` (it knows how
to talk to the session manager / governance store). This module owns the
remaining stages plus a small `DebugAnalyzer` protocol so that a future
LLM-backed analyzer can be plugged in without touching the route handler
or inlining prompts.

`DeterministicAnalyzer` is the default. It is prompt-aware: it picks the
right evidence slice based on intent keywords in the user prompt and uses
field meanings (privacy_level / owner / type from the registered schema)
to produce recommendations and structured `DebugVersionSuggestion`s.

`LlmAnalyzer` (D24) is a real model-backed analyzer. It renders the
centralized prompt (see ``PROMPT_TEMPLATE_PATH`` /
``SYSTEM_PROMPT_PATH``), calls an injected ``LlmJsonClient`` in JSON
mode, and parses the structured result back into recommendations +
``DebugVersionSuggestion``s. It degrades gracefully: with no client
configured, or on any model / parse failure, it falls back to the
deterministic logic so the route handler always gets a usable report.
The ``analysis_mode`` field records which path actually ran.

Selecting an analyzer is environment-gated:
    DLAAS_DEBUG_ANALYZER=llm  -> LlmAnalyzer wired to a model client
                                  built from env (DLAAS_DEBUG_LLM_* with a
                                  fallback to PROTOCOL_LLM_*). If no
                                  credentials are present the analyzer
                                  still runs and reports the deterministic
                                  fallback path honestly.
    anything else / unset      -> DeterministicAnalyzer.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from dlaas_platform_contracts import (
    DebugAnalysisRequest,
    DebugVersionSuggestion,
)
from lifeform_core.external_llm import (
    LlmJsonClient,
    OpenAiCompatConfig,
    OpenAiCompatJsonClient,
)

_LOG = logging.getLogger("dlaas_platform_api.debug_analysis")

PROMPT_TEMPLATE_PATH = "prompts/debug_analysis.md"
SYSTEM_PROMPT_PATH = "prompts/debug_analysis_system.md"

# Issue areas the deterministic analyzer emits; the LLM is asked to stay
# within this set so downstream UI/version-suggestion consumers see a
# stable enum-like surface. Unknown values are coerced to "unknown".
_KNOWN_ISSUE_AREAS = frozenset(
    {"app", "dlaas_runtime", "prompt_template", "deployment", "unknown"}
)

_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "handoff": ("handoff", "escalation", "refer-out"),
    "boundary": ("boundary", "safety", "refer-out", "blocked"),
    "wake": ("wake", "sleep", "lifecycle", "instance"),
    "feedback": ("feedback", "valence", "rupture", "repair"),
    "chat": ("chat", "turn", "completion", "stream"),
    "training": ("training", "corpus", "sop", "asset"),
    "data_lifecycle": ("export", "delete", "forget", "account"),
}


class DebugAnalyzer(Protocol):
    analyzer_id: str
    analyzer_version: str

    def analyze(
        self,
        *,
        analysis_request: DebugAnalysisRequest,
        evidence_summary: Mapping[str, Any],
        prompt_preview: str,
        intent_tags: tuple[str, ...],
    ) -> dict[str, Any]:
        ...


def build_debug_analysis(
    *,
    analysis_request: DebugAnalysisRequest,
    evidence: Mapping[str, Any],
    analyzer: "DebugAnalyzer | None" = None,
) -> dict[str, Any]:
    """Run the full analyzer pipeline and return a dict for the route handler.

    The returned dict carries:
      - analysis_mode        : analyzer category
      - analyzer_id          : concrete analyzer name
      - analyzer_version     : analyzer version string
      - prompt_template      : centralized template path
      - prompt_preview       : rendered prompt (with redacted evidence)
      - evidence_summary     : compact summary for downstream UI
      - recommendations      : tuple of operator-facing strings
      - version_suggestions  : tuple of DebugVersionSuggestion.to_json()

    ``analyzer`` is normally selected from the environment; callers
    (and tests) may inject a concrete analyzer to bypass the env gate.
    """
    redacted = _redact_evidence(evidence)
    evidence_summary = _summarize_evidence(redacted)
    intent_tags = _intent_tags(analysis_request.prompt)
    prompt_preview = _render_prompt_preview(
        analysis_request=analysis_request,
        evidence_summary=evidence_summary,
        intent_tags=intent_tags,
    )
    if analyzer is None:
        analyzer = _select_analyzer()
    output = analyzer.analyze(
        analysis_request=analysis_request,
        evidence_summary=evidence_summary,
        prompt_preview=prompt_preview,
        intent_tags=intent_tags,
    )
    return {
        "analysis_mode": output.get("analysis_mode", "deterministic_fallback"),
        "analyzer_id": analyzer.analyzer_id,
        "analyzer_version": analyzer.analyzer_version,
        "prompt_template": PROMPT_TEMPLATE_PATH,
        "prompt_preview": prompt_preview,
        "evidence_summary": evidence_summary,
        "intent_tags": list(intent_tags),
        "recommendations": tuple(output.get("recommendations", ())),
        "version_suggestions": tuple(output.get("version_suggestions", ())),
    }


def _select_analyzer() -> DebugAnalyzer:
    requested = (os.environ.get("DLAAS_DEBUG_ANALYZER", "") or "").strip().lower()
    if requested == "llm":
        return LlmAnalyzer(client=_build_llm_client_from_env())
    return DeterministicAnalyzer()


def _env(*names: str) -> str:
    """First non-empty stripped value among ``names`` in the environment."""
    for name in names:
        value = (os.environ.get(name, "") or "").strip()
        if value:
            return value
    return ""


def _build_llm_client_from_env() -> LlmJsonClient | None:
    """Construct an OpenAI-compatible JSON client for the debug analyzer.

    Reads ``DLAAS_DEBUG_LLM_{API_KEY,BASE_URL,MODEL}`` first, then falls
    back to the shared ``PROTOCOL_LLM_*`` credentials so a deployment that
    already configured the protocol-uptake LLM gets debug analysis for
    free. Returns ``None`` (rather than raising) when credentials are
    incomplete; the caller's :class:`LlmAnalyzer` then degrades to the
    deterministic fallback and records that honestly.
    """
    api_key = _env("DLAAS_DEBUG_LLM_API_KEY", "PROTOCOL_LLM_API_KEY")
    base_url = _env("DLAAS_DEBUG_LLM_BASE_URL", "PROTOCOL_LLM_BASE_URL")
    model = _env("DLAAS_DEBUG_LLM_MODEL", "PROTOCOL_LLM_MODEL")
    if not (api_key and base_url and model):
        _LOG.info(
            "DLAAS_DEBUG_ANALYZER=llm but LLM credentials are incomplete "
            "(api_key=%s base_url=%s model=%s); debug analysis will use the "
            "deterministic fallback path.",
            bool(api_key),
            bool(base_url),
            bool(model),
        )
        return None
    timeout = _env("DLAAS_DEBUG_LLM_TIMEOUT_SECONDS")
    config_kwargs: dict[str, Any] = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
    }
    if timeout:
        try:
            config_kwargs["timeout_seconds"] = float(timeout)
        except ValueError:
            _LOG.warning(
                "Ignoring non-numeric DLAAS_DEBUG_LLM_TIMEOUT_SECONDS=%r",
                timeout,
            )
    return OpenAiCompatJsonClient(OpenAiCompatConfig(**config_kwargs))


# ---------------------------------------------------------------------------
# Evidence redaction
# ---------------------------------------------------------------------------


def _redact_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Best-effort redaction layer.

    Today the analyzer cannot reach the registered schema to know which
    fields are sensitive, so we use a conservative heuristic on event
    field NAMES: anything containing 'secret', 'token', 'api_key',
    'password', 'evidence' is masked. The route handler in `app.py`
    will later add schema-aware redaction in CP-S5; this stage is the
    backstop.
    """
    out: dict[str, Any] = {}
    for key, value in evidence.items():
        if key == "debug_events" and isinstance(value, list):
            out[key] = [_redact_event(item) for item in value]
        else:
            out[key] = value
    return out


_NAME_REDACT_PATTERNS = (
    "secret",
    "token",
    "api_key",
    "password",
    "evidence",
)


def _redact_event(event: Any) -> Any:
    if not isinstance(event, dict):
        return event
    fields = event.get("fields")
    if not isinstance(fields, dict):
        return event
    redacted_fields: dict[str, Any] = {}
    for name, val in fields.items():
        lname = name.lower()
        if any(pat in lname for pat in _NAME_REDACT_PATTERNS):
            redacted_fields[name] = "<redacted>"
        else:
            redacted_fields[name] = val
    new_event = dict(event)
    new_event["fields"] = redacted_fields
    return new_event


# ---------------------------------------------------------------------------
# Evidence summary
# ---------------------------------------------------------------------------


def _summarize_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    debug_events = _list_value(evidence.get("debug_events"))
    audit_events = _list_value(evidence.get("audit_events"))
    event_types = sorted({str(event.get("event_type", "")) for event in debug_events})
    failed_debug_events = [
        event
        for event in debug_events
        if isinstance(event.get("fields"), Mapping)
        and event["fields"].get("ok") is False
    ]
    readouts_present = isinstance(evidence.get("readouts"), Mapping)
    explain_present = isinstance(evidence.get("explain"), Mapping)
    snapshots_present = isinstance(evidence.get("snapshots"), Mapping)
    # Collect a small set of common business field highlights so the
    # analyzer can mention "app=repair30, day_index=7" etc. without
    # surfacing every raw value.
    highlights = _highlights_for(debug_events)
    return {
        "debug_event_count": len(debug_events),
        "audit_event_count": len(audit_events),
        "event_types": event_types,
        "failed_debug_event_count": len(failed_debug_events),
        "failed_debug_event_ids": [
            str(event.get("debug_event_id", "")) for event in failed_debug_events
        ],
        "runtime_error": str(evidence.get("runtime_error", "") or ""),
        "readouts_present": readouts_present,
        "explain_present": explain_present,
        "snapshots_present": snapshots_present,
        "highlights": highlights,
    }


_HIGHLIGHT_NAMES = (
    "operation",
    "agent_role",
    "company_slug",
    "tenant_slug",
    "advisor_id",
    "program_id",
    "day_index",
    "control_mode",
    "boundary_severity",
    "refer_out_required",
    "feedback_valence",
    "handoff_trigger",
    "webhook_event_type",
    "account_action",
    "lifecycle_action",
    "source_kind",
)


def _highlights_for(debug_events: list[dict[str, Any]]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for event in debug_events:
        fields = event.get("fields") if isinstance(event, dict) else None
        if not isinstance(fields, Mapping):
            continue
        for name in _HIGHLIGHT_NAMES:
            if name in fields and fields[name] not in (None, ""):
                bucket = out.setdefault(name, [])
                if fields[name] not in bucket:
                    bucket.append(fields[name])
    return out


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


def _intent_tags(prompt: str) -> tuple[str, ...]:
    lower = prompt.lower()
    return tuple(
        tag
        for tag, keywords in _INTENT_KEYWORDS.items()
        if any(kw in lower for kw in keywords)
    )


# ---------------------------------------------------------------------------
# Prompt template rendering (centralized)
# ---------------------------------------------------------------------------


def _render_prompt_preview(
    *,
    analysis_request: DebugAnalysisRequest,
    evidence_summary: Mapping[str, Any],
    intent_tags: tuple[str, ...],
) -> str:
    template = _read_package_text(PROMPT_TEMPLATE_PATH)
    return template.format(
        prompt=analysis_request.prompt,
        selectors_json=json.dumps(
            analysis_request.selectors_json(),
            ensure_ascii=False,
            sort_keys=True,
        ),
        evidence_summary_json=json.dumps(
            {**dict(evidence_summary), "intent_tags": list(intent_tags)},
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def _read_package_text(relative_path: str) -> str:
    return (Path(__file__).resolve().parent / relative_path).read_text(
        encoding="utf-8"
    )


def _list_value(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Analyzers
# ---------------------------------------------------------------------------


class DeterministicAnalyzer:
    analyzer_id = "deterministic"
    analyzer_version = "v2"

    def analyze(
        self,
        *,
        analysis_request: DebugAnalysisRequest,
        evidence_summary: Mapping[str, Any],
        prompt_preview: str,
        intent_tags: tuple[str, ...],
    ) -> dict[str, Any]:
        recommendations = self._recommendations(
            evidence_summary, intent_tags=intent_tags
        )
        suggestions = self._version_suggestions(
            evidence_summary,
            intent_tags=intent_tags,
            selectors=analysis_request.selectors_json(),
        )
        return {
            "analysis_mode": "deterministic",
            "recommendations": recommendations,
            "version_suggestions": tuple(
                suggestion.to_json() for suggestion in suggestions
            ),
        }

    def _recommendations(
        self,
        summary: Mapping[str, Any],
        *,
        intent_tags: tuple[str, ...],
    ) -> tuple[str, ...]:
        recommendations: list[str] = []
        debug_count = int(summary.get("debug_event_count", 0) or 0)
        failed_count = int(summary.get("failed_debug_event_count", 0) or 0)
        if debug_count == 0:
            recommendations.append(
                "No app-owned debug events matched the selectors; "
                "register schemas and publish boundary events before relying on analysis."
            )
        if failed_count > 0:
            recommendations.append(
                f"{failed_count} app boundary call(s) failed; inspect the failed "
                "debug event ids and upstream paths before changing app logic."
            )
        if summary.get("runtime_error"):
            recommendations.append(
                "Runtime evidence was unavailable; verify ai_id wake state and "
                "session correlation before drawing conclusions."
            )
        if not summary.get("readouts_present") and not summary.get("explain_present"):
            recommendations.append(
                "No readouts or explain trace were available; analysis is limited to "
                "app and audit facts."
            )
        if "handoff" in intent_tags:
            highlights = summary.get("highlights", {})
            triggers = highlights.get("handoff_trigger") if isinstance(highlights, Mapping) else None
            if triggers:
                recommendations.append(
                    "Handoff intent: observed trigger(s) " + ", ".join(map(str, triggers))
                )
            else:
                recommendations.append(
                    "Handoff intent: no handoff_trigger field present in the selected "
                    "events. Confirm repair30/digital-employee handoff paths actually fired."
                )
        if "boundary" in intent_tags:
            highlights = summary.get("highlights", {})
            severities = highlights.get("boundary_severity") if isinstance(highlights, Mapping) else None
            if severities:
                recommendations.append(
                    "Boundary intent: observed severity(ies) "
                    + ", ".join(map(str, severities))
                )
        if "feedback" in intent_tags:
            highlights = summary.get("highlights", {})
            valences = highlights.get("feedback_valence") if isinstance(highlights, Mapping) else None
            if valences:
                recommendations.append(
                    "Feedback intent: observed valence(s) " + ", ".join(map(str, valences))
                )
        if not recommendations:
            recommendations.append(
                "Evidence is present across app debug events and DLaaS runtime views; "
                "compare failed calls, readouts, and audit traces before changing behavior."
            )
        return tuple(recommendations)

    def _version_suggestions(
        self,
        summary: Mapping[str, Any],
        *,
        intent_tags: tuple[str, ...],
        selectors: Mapping[str, Any],
    ) -> tuple[DebugVersionSuggestion, ...]:
        failed_ids = tuple(
            str(x) for x in summary.get("failed_debug_event_ids", ())
        )
        app_id = str(selectors.get("app_id", "") or "")
        suggestions: list[DebugVersionSuggestion] = []
        if failed_ids:
            suggestions.append(
                DebugVersionSuggestion(
                    issue_area="app",
                    evidence_refs=failed_ids,
                    recommended_owner=app_id or "app",
                    confidence=0.75 if "chat" in intent_tags else 0.7,
                    proposed_next_test=(
                        "Replay the selected session and verify the failing upstream "
                        "path returns a 2xx status with the same debug schema version."
                    ),
                )
            )
        if summary.get("runtime_error"):
            suggestions.append(
                DebugVersionSuggestion(
                    issue_area="dlaas_runtime",
                    evidence_refs=(),
                    recommended_owner="dlaas-platform",
                    confidence=0.65,
                    proposed_next_test=(
                        "Wake the selected ai_id and fetch readouts for the selected "
                        "session before rerunning analysis."
                    ),
                )
            )
        if "boundary" in intent_tags or "handoff" in intent_tags:
            highlights = summary.get("highlights", {}) or {}
            refer = isinstance(highlights, Mapping) and bool(
                highlights.get("refer_out_required")
            )
            suggestions.append(
                DebugVersionSuggestion(
                    issue_area="prompt_template" if refer else "deployment",
                    evidence_refs=(),
                    recommended_owner="dlaas-platform",
                    confidence=0.5,
                    proposed_next_test=(
                        "Compare the boundary policy / handoff disclaimer rendered by "
                        "the app to the live DLaaS readouts for this session."
                    ),
                )
            )
        if not suggestions:
            suggestions.append(
                DebugVersionSuggestion(
                    issue_area="unknown",
                    evidence_refs=(),
                    recommended_owner="operator",
                    confidence=0.4,
                    proposed_next_test=(
                        "Add or select more specific app-owned fields and rerun "
                        "analysis with a narrower event_type filter."
                    ),
                )
            )
        return tuple(suggestions)


class LlmAnalyzer(DeterministicAnalyzer):
    """Model-backed analyzer (D24).

    Given an :class:`LlmJsonClient`, it sends the centralized system +
    user prompts to the model in JSON mode and parses the structured
    result into recommendations + ``DebugVersionSuggestion``s.

    Failure handling is deliberate and visible (R: no swallowed errors):

    * No client configured        -> ``analysis_mode="llm_unconfigured_fallback"``
    * Client error / bad JSON      -> ``analysis_mode="llm_error_fallback"``
                                      + a recommendation naming the failure
    * Model returned no content    -> same error-fallback path
    * Success                      -> ``analysis_mode="llm"``

    In every fallback case the deterministic analysis is returned so the
    operator still gets a usable report; the mode string tells them which
    path produced it.
    """

    analyzer_id = "llm"
    analyzer_version = "v1"

    def __init__(self, client: LlmJsonClient | None = None) -> None:
        self._client = client

    def analyze(
        self,
        *,
        analysis_request: DebugAnalysisRequest,
        evidence_summary: Mapping[str, Any],
        prompt_preview: str,
        intent_tags: tuple[str, ...],
    ) -> dict[str, Any]:
        if self._client is None:
            return self._fallback(
                analysis_request=analysis_request,
                evidence_summary=evidence_summary,
                prompt_preview=prompt_preview,
                intent_tags=intent_tags,
                mode="llm_unconfigured_fallback",
            )
        try:
            raw = self._client.complete_json(
                system_prompt=_read_package_text(SYSTEM_PROMPT_PATH),
                user_prompt=prompt_preview,
            )
            recommendations, suggestions = _parse_llm_output(raw)
        except Exception as exc:  # noqa: BLE001 - degrade, but record the reason
            _LOG.warning("LlmAnalyzer fell back to deterministic: %s", exc)
            output = self._fallback(
                analysis_request=analysis_request,
                evidence_summary=evidence_summary,
                prompt_preview=prompt_preview,
                intent_tags=intent_tags,
                mode="llm_error_fallback",
            )
            output["recommendations"] = (
                f"LLM debug analysis failed ({type(exc).__name__}); "
                "showing deterministic fallback instead.",
                *output["recommendations"],
            )
            return output
        return {
            "analysis_mode": "llm",
            "recommendations": recommendations,
            "version_suggestions": tuple(
                suggestion.to_json() for suggestion in suggestions
            ),
        }

    def _fallback(
        self,
        *,
        analysis_request: DebugAnalysisRequest,
        evidence_summary: Mapping[str, Any],
        prompt_preview: str,
        intent_tags: tuple[str, ...],
        mode: str,
    ) -> dict[str, Any]:
        output = super().analyze(
            analysis_request=analysis_request,
            evidence_summary=evidence_summary,
            prompt_preview=prompt_preview,
            intent_tags=intent_tags,
        )
        output["analysis_mode"] = mode
        return output


def _parse_llm_output(
    raw: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[DebugVersionSuggestion, ...]]:
    """Parse the model's JSON object into typed analysis output.

    Raises ``ValueError`` if the payload has no usable content so the
    caller can fall back to the deterministic analyzer instead of
    surfacing an empty (and misleading) report.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"LLM output is not a JSON object: {type(raw)!r}")
    recommendations = tuple(
        text
        for text in (
            str(item).strip() for item in _as_list(raw.get("recommendations"))
        )
        if text
    )
    suggestions = tuple(
        _suggestion_from_json(item)
        for item in _as_list(raw.get("version_suggestions"))
        if isinstance(item, Mapping)
    )
    if not recommendations and not suggestions:
        raise ValueError(
            "LLM output had neither recommendations nor version_suggestions"
        )
    return recommendations, suggestions


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _suggestion_from_json(item: Mapping[str, Any]) -> DebugVersionSuggestion:
    issue_area = str(item.get("issue_area", "") or "unknown").strip().lower()
    if issue_area not in _KNOWN_ISSUE_AREAS:
        issue_area = "unknown"
    evidence_refs = tuple(
        str(ref).strip()
        for ref in _as_list(item.get("evidence_refs"))
        if str(ref).strip()
    )
    recommended_owner = str(item.get("recommended_owner", "") or "operator").strip()
    proposed_next_test = str(item.get("proposed_next_test", "") or "").strip()
    confidence = _clamp_confidence(item.get("confidence"))
    return DebugVersionSuggestion(
        issue_area=issue_area,
        evidence_refs=evidence_refs,
        recommended_owner=recommended_owner or "operator",
        confidence=confidence,
        proposed_next_test=proposed_next_test,
    )


def _clamp_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


__all__ = [
    "DebugAnalyzer",
    "DeterministicAnalyzer",
    "LlmAnalyzer",
    "PROMPT_TEMPLATE_PATH",
    "SYSTEM_PROMPT_PATH",
    "build_debug_analysis",
]
