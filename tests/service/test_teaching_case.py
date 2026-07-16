"""TeachingCase service closure tests (runtime-ingestion spec).

Covers the I1 closure invariants:

* the envelope translation matches the spec sample (CORPUS kind,
  ``teaching-{case_id}`` envelope id, per-field chunks + locators,
  FORCED compliance),
* the service drives ingestion ONLY through
  ``IngestionPipeline.process_envelope`` (canonical turn path with
  ``trigger_kind=INGESTION``, scene closed for the R6 slow loop),
* lifecycle: duplicate submission rejected, retire is an explicit
  R15 rollback marker that survives in the audit record,
* the route layer enforces the ``ingestion-`` session-id isolation
  prefix and surfaces validation errors as typed HTTP errors.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from aiohttp import web

from lifeform_core import TurnTriggerKind

from lifeform_ingestion import IngestionComplianceProfile, IngestionSourceKind
from lifeform_service.teaching_case import (
    TeachingCase,
    TeachingCaseService,
    TeachingCaseStatus,
    register_teaching_case_routes,
    teaching_case_envelope,
)


class _FakeSession:
    """Session stub recording the canonical-turn-path calls."""

    def __init__(self) -> None:
        self.run_turn_calls: list[tuple[str, TurnTriggerKind]] = []
        self.end_scene_calls: list[str] = []

    async def run_turn(
        self,
        user_input: str,
        *,
        trigger_kind: TurnTriggerKind = TurnTriggerKind.USER_INPUT,
        environment_provenance: str | None = None,
        environment_consent_context: tuple[str, ...] = (),
    ):
        self.run_turn_calls.append((user_input, trigger_kind))
        return SimpleNamespace(
            response=SimpleNamespace(text=f"ack:{user_input[:24]}"),
            active_snapshots={},
        )

    async def end_scene(self, *, reason: str = "", drain_slow_loop: bool = True):
        self.end_scene_calls.append(reason)
        return SimpleNamespace(scene_id="scene-teaching", closed_at_tick=1)


def _case(case_id: str = "tc-1", *, rationale: str = "") -> TeachingCase:
    return TeachingCase(
        case_id=case_id,
        operator_id="operator-7",
        simulated_user_turns="User: I feel like everything is falling apart.",
        ideal_ai_response="Acknowledge the distress first, then offer one step.",
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Envelope translation
# ---------------------------------------------------------------------------


def test_teaching_case_envelope_matches_spec_shape() -> None:
    envelope = teaching_case_envelope(_case())
    assert envelope.envelope_id == "teaching-tc-1"
    assert envelope.source_kind is IngestionSourceKind.CORPUS
    assert envelope.compliance_profile is IngestionComplianceProfile.FORCED
    assert [c.chunk_id for c in envelope.chunks] == ["tc-1-user", "tc-1-ai"]
    assert envelope.chunks[0].locator == (
        "teaching_case=tc-1;field=simulated_user_turns"
    )
    assert envelope.provenance.uploader == "operator-7"
    assert envelope.provenance.source_uri == "teaching-case://tc-1"
    assert envelope.provenance.integrity_hash


def test_teaching_case_envelope_includes_rationale_chunk() -> None:
    envelope = teaching_case_envelope(
        _case(rationale="Distress must be named before structure helps.")
    )
    assert [c.chunk_id for c in envelope.chunks] == [
        "tc-1-user",
        "tc-1-ai",
        "tc-1-rationale",
    ]


def test_teaching_case_validation_fails_loudly() -> None:
    with pytest.raises(ValueError, match="case_id"):
        TeachingCase(
            case_id=" ",
            operator_id="op",
            simulated_user_turns="u",
            ideal_ai_response="a",
        )
    with pytest.raises(ValueError, match="ideal_ai_response"):
        TeachingCase(
            case_id="x",
            operator_id="op",
            simulated_user_turns="u",
            ideal_ai_response="  ",
        )


# ---------------------------------------------------------------------------
# Service lifecycle
# ---------------------------------------------------------------------------


def test_submit_ingests_through_canonical_turn_path() -> None:
    import asyncio

    service = TeachingCaseService()
    session = _FakeSession()

    record, report = asyncio.run(
        service.submit(_case(), session=session, session_id="ingestion-t1")
    )

    assert record.status is TeachingCaseStatus.INGESTED
    assert record.envelope_id == "teaching-tc-1"
    assert report.processed_chunks == 2
    # Every chunk went through run_turn with the INGESTION trigger.
    assert [kind for _, kind in session.run_turn_calls] == [
        TurnTriggerKind.INGESTION,
        TurnTriggerKind.INGESTION,
    ]
    # Scene closed so the R6 session-post slow loop consolidates.
    assert session.end_scene_calls == ["teaching-case-end"]


def test_duplicate_submission_rejected() -> None:
    import asyncio

    service = TeachingCaseService()
    session = _FakeSession()
    asyncio.run(service.submit(_case(), session=session, session_id="ingestion-t1"))
    with pytest.raises(ValueError, match="already submitted"):
        asyncio.run(
            service.submit(_case(), session=session, session_id="ingestion-t1")
        )


def test_retire_marks_record_and_blocks_double_retire() -> None:
    import asyncio

    service = TeachingCaseService()
    session = _FakeSession()
    asyncio.run(service.submit(_case(), session=session, session_id="ingestion-t1"))

    retired = service.retire("tc-1", reason="behaviour regressed")
    assert retired.status is TeachingCaseStatus.RETIRED
    assert retired.retired_reason == "behaviour regressed"
    # The envelope lineage survives retirement for R15 cleanup.
    assert retired.envelope_id == "teaching-tc-1"

    with pytest.raises(ValueError, match="already retired"):
        service.retire("tc-1", reason="again")
    with pytest.raises(KeyError, match="not in the store"):
        service.retire("missing", reason="n/a")


# ---------------------------------------------------------------------------
# Route layer
# ---------------------------------------------------------------------------


class _FakeSessionManager:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def get_session(self, session_id: str) -> _FakeSession:
        return self._session


def _build_app(session: _FakeSession) -> web.Application:
    app = web.Application()
    app["session_manager"] = _FakeSessionManager(session)
    register_teaching_case_routes(app, service=TeachingCaseService())
    return app


async def test_route_submit_and_list(aiohttp_client) -> None:
    session = _FakeSession()
    client = await aiohttp_client(_build_app(session))

    response = await client.post(
        "/v1/sessions/ingestion-op1/teaching-cases",
        json={
            "case_id": "tc-9",
            "operator_id": "operator-7",
            "simulated_user_turns": "User: how do I even start?",
            "ideal_ai_response": "Start with the smallest concrete step.",
        },
    )
    assert response.status == 200
    body = await response.json()
    assert body["record"]["status"] == "ingested"
    assert body["report"]["processed_chunks"] == 2

    listing = await (await client.get("/v1/teaching-cases")).json()
    assert listing["count"] == 1
    assert listing["teaching_cases"][0]["case_id"] == "tc-9"


async def test_route_rejects_non_ingestion_session(aiohttp_client) -> None:
    session = _FakeSession()
    client = await aiohttp_client(_build_app(session))

    response = await client.post(
        "/v1/sessions/user-session-1/teaching-cases",
        json={
            "case_id": "tc-9",
            "operator_id": "operator-7",
            "simulated_user_turns": "u",
            "ideal_ai_response": "a",
        },
    )
    assert response.status == 422
    body = await response.json()
    assert body["error"]["code"] == "not_an_ingestion_session"
    # Nothing touched the kernel.
    assert session.run_turn_calls == []


async def test_route_invalid_case_and_retire_flow(aiohttp_client) -> None:
    session = _FakeSession()
    client = await aiohttp_client(_build_app(session))

    bad = await client.post(
        "/v1/sessions/ingestion-op1/teaching-cases",
        json={"case_id": "tc-9", "operator_id": "", "simulated_user_turns": "u"},
    )
    assert bad.status == 422
    assert (await bad.json())["error"]["code"] == "invalid_teaching_case"

    ok = await client.post(
        "/v1/sessions/ingestion-op1/teaching-cases",
        json={
            "case_id": "tc-9",
            "operator_id": "operator-7",
            "simulated_user_turns": "u",
            "ideal_ai_response": "a",
        },
    )
    assert ok.status == 200

    duplicate = await client.post(
        "/v1/sessions/ingestion-op1/teaching-cases",
        json={
            "case_id": "tc-9",
            "operator_id": "operator-7",
            "simulated_user_turns": "u",
            "ideal_ai_response": "a",
        },
    )
    assert duplicate.status == 409

    retire = await client.post(
        "/v1/teaching-cases/tc-9/retire", json={"reason": "regressed"}
    )
    assert retire.status == 200
    assert (await retire.json())["record"]["status"] == "retired"

    missing = await client.post(
        "/v1/teaching-cases/nope/retire", json={"reason": "n/a"}
    )
    assert missing.status == 404
