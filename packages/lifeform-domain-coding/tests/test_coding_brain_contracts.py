from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lifeform_domain_coding import (
    CodingAdviceSnapshot,
    CodingContextPackSnapshot,
    CodingContextRequest,
    CodingOutcomeKind,
    CodingOutcomeReceipt,
    CodingOutcomeReport,
    CodingOutcomeRoute,
    CodingOutcomeSource,
    CodingTaskKind,
    build_coding_lifeform,
)
from lifeform_domain_coding.coding_brain_contracts import stable_content_sha256
from volvence_zero.memory import (
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
)
from volvence_zero.runtime import WiringLevel


def _context_request() -> CodingContextRequest:
    return CodingContextRequest.from_json(
        {
            "request_id": "request-1",
            "project_id": "project-1",
            "repository_id": "repo-1",
            "task_id": "task-1",
            "task_kind": "bugfix",
            "task_summary": "Fix the state restoration regression",
            "target_paths": ["src/state.py", "tests/test_state.py"],
        }
    )


def test_context_request_is_strict_frozen_and_json_stable() -> None:
    request = _context_request()
    assert request.task_kind is CodingTaskKind.BUGFIX
    assert request.to_json()["schema_version"] == "coding-context-request.v1"
    with pytest.raises(FrozenInstanceError):
        request.task_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="unknown fields"):
        CodingContextRequest.from_json({**request.to_json(), "guessed_mode": "fast"})


@pytest.mark.parametrize(
    ("kind", "source"),
    (
        ("task_verified", "code_review"),
        ("review_approved", "ci"),
        ("merged", "test_suite"),
    ),
)
def test_outcome_kind_source_pairs_fail_closed(kind: str, source: str) -> None:
    with pytest.raises(ValueError):
        CodingOutcomeReport.from_json(
            {
                "outcome_id": "outcome-1",
                "context_pack_id": "coding-context-pack:" + "a" * 64,
                "kind": kind,
                "source": source,
                "summary": "Typed result",
                "detail": "Evidence supplied by the named source.",
                "observed_at_ms": 10,
                "evidence_ref": "ci:run-1",
            }
        )


def test_context_pack_requires_active_and_advice_requires_shadow() -> None:
    request = _context_request()
    advice = CodingAdviceSnapshot(
        advice_id="coding-advice:" + "b" * 64,
        source_turn_index=0,
        candidate_regime_id="problem_solving",
        candidate_abstract_action="inspect_before_edit",
        evidence_entry_ids=(),
        rationale="Owner-published controller readout; not applied to the pack.",
    )
    payload = {
        "request": request.to_json(),
        "rendered_context": "No prior coding outcomes were recalled.",
        "source_entry_ids": [],
    }
    digest = stable_content_sha256(payload)
    pack = CodingContextPackSnapshot(
        context_pack_id=f"coding-context-pack:{digest}",
        content_sha256=digest,
        request=request,
        generated_at_ms=100,
        source_turn_index=0,
        rendered_context=payload["rendered_context"],
        source_entry_ids=(),
        retrieval_facets=("coding-brain", "task-kind:bugfix"),
        memory_entry_count=0,
        truncated=False,
        settled_outcome_evidence_refs=(),
        pe_magnitude=0.0,
        pe_bootstrap=True,
        advice=advice,
    )
    assert pack.wiring_level is WiringLevel.ACTIVE
    assert pack.advice.wiring_level is WiringLevel.SHADOW
    assert pack.advice.applied is False
    with pytest.raises(ValueError, match="must remain WiringLevel.SHADOW"):
        CodingAdviceSnapshot(
            advice_id="coding-advice:" + "c" * 64,
            source_turn_index=0,
            candidate_regime_id="",
            candidate_abstract_action="",
            evidence_entry_ids=(),
            rationale="Invalid promotion attempt.",
            wiring_level=WiringLevel.ACTIVE,
        )


def test_receipt_requires_pe_route_only_for_deterministic_oracle() -> None:
    report = CodingOutcomeReport(
        outcome_id="outcome-1",
        context_pack_id="coding-context-pack:" + "a" * 64,
        kind=CodingOutcomeKind.TASK_REGRESSED,
        source=CodingOutcomeSource.CI,
        summary="CI failed",
        detail="tests/test_state.py::test_restore failed",
        observed_at_ms=100,
        evidence_ref="ci:run-1",
    )
    with pytest.raises(ValueError, match="deterministic outcomes must route to PE"):
        CodingOutcomeReceipt(
            receipt_id="coding-outcome-receipt:" + "d" * 64,
            content_sha256="d" * 64,
            session_id="session-1",
            project_id="project-1",
            repository_id="repo-1",
            task_id="task-1",
            report=report,
            action_turn_index=0,
            memory_entry_id="memory-1",
            memory_persisted=False,
            task_event_ids=("event-1",),
            external_outcome_evidence_id="",
            learning_route=CodingOutcomeRoute.EXECUTION_RESULT,
        )


def test_lifeform_memory_facade_preserves_owner_contract() -> None:
    lifeform = build_coding_lifeform(
        use_temporal_bootstrap=False,
        use_regime_bootstrap=False,
    )
    session = lifeform.create_session(session_id="coding-memory-facade")
    entry = session.write_memory(
        MemoryWriteRequest(
            content="CI regression in state restoration",
            track=Track.WORLD,
            stratum=MemoryStratum.EPISODIC,
            tags=("coding-brain", "outcome:task_regressed"),
            strength=0.9,
        ),
        timestamp_ms=100,
    )
    result = session.retrieve_memory(
        RetrievalQuery(
            text="state restoration regression",
            track=Track.WORLD,
            strata=(MemoryStratum.EPISODIC,),
            limit=4,
        ),
        timestamp_ms=101,
    )
    assert [item.entry_id for item in result.entries] == [entry.entry_id]
    assert session.memory_entry_count() == 1
