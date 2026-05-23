"""Debug analysis rendering for DLaaS platform evidence.

This module owns the analysis boundary. It renders a centralized prompt
template for future LLM-backed analysis and currently returns a deterministic
fallback report so the API remains useful without introducing inline prompts or
an undeclared model dependency.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dlaas_platform_contracts import DebugAnalysisRequest


def build_debug_analysis(
    *,
    analysis_request: DebugAnalysisRequest,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_summary = _summarize_evidence(evidence)
    prompt_preview = _render_prompt_preview(
        analysis_request=analysis_request,
        evidence_summary=evidence_summary,
    )
    recommendations = _recommendations(evidence_summary)
    version_suggestions = _version_suggestions(evidence_summary)
    return {
        "analysis_mode": "deterministic_fallback",
        "prompt_template": "prompts/debug_analysis.md",
        "prompt_preview": prompt_preview,
        "evidence_summary": evidence_summary,
        "recommendations": recommendations,
        "version_suggestions": version_suggestions,
    }


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
    }


def _recommendations(summary: Mapping[str, Any]) -> tuple[str, ...]:
    recommendations: list[str] = []
    if int(summary.get("debug_event_count", 0) or 0) == 0:
        recommendations.append(
            "No app-owned debug events matched the selectors; register schemas and publish boundary events before relying on analysis."
        )
    if int(summary.get("failed_debug_event_count", 0) or 0) > 0:
        recommendations.append(
            "One or more app boundary calls failed; inspect the failed debug event ids and upstream paths first."
        )
    if summary.get("runtime_error"):
        recommendations.append(
            "Runtime evidence was unavailable; verify ai_id wake state and session correlation before changing app logic."
        )
    if not summary.get("readouts_present") and not summary.get("explain_present"):
        recommendations.append(
            "No readouts or explain trace were available; analysis is limited to app/audit facts."
        )
    if not recommendations:
        recommendations.append(
            "Evidence is present across app debug events and DLaaS runtime views; compare failed calls, readouts, and audit traces before changing behavior."
        )
    return tuple(recommendations)


def _version_suggestions(summary: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if int(summary.get("failed_debug_event_count", 0) or 0) > 0:
        return (
            {
                "suggestion_type": "debug_version_suggestion",
                "issue_area": "app",
                "evidence_refs": list(summary.get("failed_debug_event_ids", ())),
                "recommended_owner": "app",
                "confidence": 0.7,
                "proposed_next_test": "Replay the selected session and verify the failing upstream path returns a 2xx status with the same debug schema version.",
            },
        )
    if summary.get("runtime_error"):
        return (
            {
                "suggestion_type": "debug_version_suggestion",
                "issue_area": "dlaas_runtime",
                "evidence_refs": [],
                "recommended_owner": "dlaas-platform",
                "confidence": 0.65,
                "proposed_next_test": "Wake the selected ai_id and fetch readouts for the selected session before rerunning analysis.",
            },
        )
    return (
        {
            "suggestion_type": "debug_version_suggestion",
            "issue_area": "unknown",
            "evidence_refs": [],
            "recommended_owner": "operator",
            "confidence": 0.4,
            "proposed_next_test": "Add or select more specific app-owned fields and rerun analysis with a narrower event type filter.",
        },
    )


def _render_prompt_preview(
    *,
    analysis_request: DebugAnalysisRequest,
    evidence_summary: Mapping[str, Any],
) -> str:
    template = _read_package_text("prompts/debug_analysis.md")
    return template.format(
        prompt=analysis_request.prompt,
        selectors_json=json.dumps(
            analysis_request.selectors_json(),
            ensure_ascii=False,
            sort_keys=True,
        ),
        evidence_summary_json=json.dumps(
            dict(evidence_summary),
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


__all__ = ["build_debug_analysis"]
