"""TeachingCase service closure (runtime-ingestion spec, service side).

Closes the "DLaaS TeachingCase 路径" loop from
``docs/specs/runtime-ingestion.md``: an operator demonstrates "this
kind of question should be answered like this", the service stores
the case as an auditable record, translates it into an
``IngestionEnvelope`` (CORPUS kind, one chunk per teaching field),
and drives it through ``IngestionPipeline.process_envelope``.

Invariants (spec §Service-side TeachingCase 接入):

* The service layer only calls ``IngestionPipeline.process_envelope``
  — it never pokes ``LifeformSession.run_turn`` directly, and never
  reaches into any kernel owner store.
* Teaching content enters the kernel exclusively through the
  canonical turn path (``trigger_kind=INGESTION`` via the FORCED
  compliance profile); durable consolidation stays on the R6
  session-post slow loop. There is no parallel learning pipeline.
* Ingestion sessions are isolated from user sessions: the route
  layer enforces the ``ingestion-`` session-id prefix so a teaching
  case can never be replayed into a live user transcript.
* R15 rollback: every record keeps its ``envelope_id`` so reflection
  writeback lineage can be traced and the case can be retired
  (``retire``) if the taught behaviour regresses.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from enum import Enum

from aiohttp import web

from lifeform_ingestion import (
    IngestionChunk,
    IngestionComplianceProfile,
    IngestionEnvelope,
    IngestionPipeline,
    IngestionProvenance,
    IngestionReport,
    IngestionSourceKind,
)


_APP_KEY = "teaching_case_service"
_INGESTION_SESSION_PREFIX = "ingestion-"


@dataclass(frozen=True)
class TeachingCase:
    """One operator demonstration: user situation + ideal response.

    ``rationale`` is optional operator commentary ("why this is the
    right shape of answer") and, when present, becomes a third chunk
    so the lifeform ingests the reasoning alongside the exchange.
    """

    case_id: str
    operator_id: str
    simulated_user_turns: str
    ideal_ai_response: str
    rationale: str = ""
    created_ts_ms: int = 0

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("TeachingCase.case_id must be non-empty")
        if not self.operator_id.strip():
            raise ValueError("TeachingCase.operator_id must be non-empty")
        if not self.simulated_user_turns.strip():
            raise ValueError(
                "TeachingCase.simulated_user_turns must be non-empty"
            )
        if not self.ideal_ai_response.strip():
            raise ValueError("TeachingCase.ideal_ai_response must be non-empty")


class TeachingCaseStatus(str, Enum):
    """Lifecycle of a stored teaching case (audit surface)."""

    SUBMITTED = "submitted"
    INGESTED = "ingested"
    PARTIAL = "partial"
    FAILED = "failed"
    RETIRED = "retired"


@dataclass(frozen=True)
class TeachingCaseRecord:
    """Immutable audit row for one submitted case."""

    case: TeachingCase
    envelope_id: str
    session_id: str
    status: TeachingCaseStatus
    processed_chunks: int = 0
    skipped_chunks: int = 0
    ended_scene: bool = False
    retired_reason: str = ""

    def to_json(self) -> dict:
        return {
            "case_id": self.case.case_id,
            "operator_id": self.case.operator_id,
            "envelope_id": self.envelope_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "processed_chunks": self.processed_chunks,
            "skipped_chunks": self.skipped_chunks,
            "ended_scene": self.ended_scene,
            "retired_reason": self.retired_reason,
        }


def teaching_case_envelope(case: TeachingCase) -> IngestionEnvelope:
    """Translate a teaching case into the canonical ingestion envelope.

    Chunk layout follows the spec sample: one chunk for the simulated
    user turns, one for the ideal AI response (plus an optional
    rationale chunk). Locators use the
    ``teaching_case={id};field={name}`` form so the reflection
    writeback lineage points back at the exact teaching field.
    """
    ts = case.created_ts_ms or int(time.time() * 1000.0)
    combined = "\n\n".join(
        part
        for part in (
            case.simulated_user_turns,
            case.ideal_ai_response,
            case.rationale,
        )
        if part.strip()
    )
    integrity_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    chunks = [
        IngestionChunk(
            chunk_id=f"{case.case_id}-user",
            text=case.simulated_user_turns,
            locator=f"teaching_case={case.case_id};field=simulated_user_turns",
            confidence=1.0,
        ),
        IngestionChunk(
            chunk_id=f"{case.case_id}-ai",
            text=case.ideal_ai_response,
            locator=f"teaching_case={case.case_id};field=ideal_ai_response",
            confidence=1.0,
        ),
    ]
    if case.rationale.strip():
        chunks.append(
            IngestionChunk(
                chunk_id=f"{case.case_id}-rationale",
                text=case.rationale,
                locator=f"teaching_case={case.case_id};field=rationale",
                confidence=1.0,
            )
        )
    return IngestionEnvelope(
        envelope_id=f"teaching-{case.case_id}",
        source_kind=IngestionSourceKind.CORPUS,
        chunks=tuple(chunks),
        provenance=IngestionProvenance(
            uploader=case.operator_id,
            upload_ts_ms=ts,
            source_uri=f"teaching-case://{case.case_id}",
            integrity_hash=integrity_hash,
        ),
        compliance_profile=IngestionComplianceProfile.FORCED,
    )


class TeachingCaseService:
    """Stores teaching cases and drives them through the pipeline.

    In-memory store (one instance per service process): the record
    surface is a product-mirror audit log, not a second owner of any
    kernel state — everything the kernel learned lives in the normal
    owner snapshots reachable through the envelope lineage.
    """

    def __init__(self) -> None:
        self._records: dict[str, TeachingCaseRecord] = {}
        self._pipeline = IngestionPipeline()

    async def submit(
        self,
        case: TeachingCase,
        *,
        session: object,
        session_id: str,
    ) -> tuple[TeachingCaseRecord, IngestionReport]:
        """Ingest one teaching case into ``session``.

        Duplicate ``case_id`` submissions are rejected (retire the
        old case first — replaying the same id would corrupt the
        audit lineage). The scene is closed after the case so the R6
        session-post slow loop consolidates immediately.
        """
        if case.case_id in self._records:
            raise ValueError(
                f"TeachingCase {case.case_id!r} was already submitted "
                f"(status={self._records[case.case_id].status.value}); "
                f"retire it before resubmitting."
            )
        envelope = teaching_case_envelope(case)
        report = await self._pipeline.process_envelope(
            envelope,
            session=session,  # type: ignore[arg-type]
            end_scene_after=True,
            scene_end_reason="teaching-case-end",
        )
        if report.skipped_chunks == 0 and report.processed_chunks > 0:
            status = TeachingCaseStatus.INGESTED
        elif report.processed_chunks > 0:
            status = TeachingCaseStatus.PARTIAL
        else:
            status = TeachingCaseStatus.FAILED
        record = TeachingCaseRecord(
            case=case,
            envelope_id=envelope.envelope_id,
            session_id=session_id,
            status=status,
            processed_chunks=report.processed_chunks,
            skipped_chunks=report.skipped_chunks,
            ended_scene=report.ended_scene,
        )
        self._records[case.case_id] = record
        return (record, report)

    def retire(self, case_id: str, *, reason: str) -> TeachingCaseRecord:
        """Mark a case retired (R15 rollback marker).

        Retiring does not rewrite kernel memory — it flags the case
        so operators can run the standard lineage cleanup
        (``retire_case_by_lineage(envelope_id)`` on the R6 slow loop)
        and blocks accidental resubmission confusion.
        """
        record = self._records.get(case_id)
        if record is None:
            raise KeyError(f"TeachingCase {case_id!r} is not in the store.")
        if record.status is TeachingCaseStatus.RETIRED:
            raise ValueError(f"TeachingCase {case_id!r} is already retired.")
        retired = replace(
            record, status=TeachingCaseStatus.RETIRED, retired_reason=reason
        )
        self._records[case_id] = retired
        return retired

    def get(self, case_id: str) -> TeachingCaseRecord | None:
        return self._records.get(case_id)

    def list_records(self) -> tuple[TeachingCaseRecord, ...]:
        return tuple(self._records.values())


def register_teaching_case_routes(
    app: web.Application,
    *,
    service: TeachingCaseService,
) -> None:
    """Attach the teaching-case service to the app and register routes.

    * ``POST /v1/sessions/{session_id}/teaching-cases`` — submit + ingest
    * ``GET  /v1/teaching-cases``                        — list audit records
    * ``POST /v1/teaching-cases/{case_id}/retire``       — R15 rollback marker
    """
    app[_APP_KEY] = service
    app.router.add_post(
        "/v1/sessions/{session_id}/teaching-cases", _handle_submit
    )
    app.router.add_get("/v1/teaching-cases", _handle_list)
    app.router.add_post(
        "/v1/teaching-cases/{case_id}/retire", _handle_retire
    )


def _service(request: web.Request) -> TeachingCaseService:
    return request.app[_APP_KEY]


def _error(status: int, code: str, message: str) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message}}, status=status
    )


async def _handle_submit(request: web.Request) -> web.Response:
    session_id = request.match_info["session_id"]
    if not session_id.startswith(_INGESTION_SESSION_PREFIX):
        # Spec invariant: ingestion sessions are isolated from user
        # sessions; teaching content must never enter a live user
        # transcript.
        return _error(
            422,
            "not_an_ingestion_session",
            f"teaching cases require a session_id prefixed "
            f"{_INGESTION_SESSION_PREFIX!r}, got {session_id!r}. Create a "
            f"dedicated ingestion session first.",
        )
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - malformed body is a client error
        return _error(400, "invalid_json", "request body must be JSON")
    if not isinstance(payload, dict):
        return _error(400, "invalid_json", "request body must be a JSON object")
    try:
        case = TeachingCase(
            case_id=str(payload.get("case_id", "")),
            operator_id=str(payload.get("operator_id", "")),
            simulated_user_turns=str(payload.get("simulated_user_turns", "")),
            ideal_ai_response=str(payload.get("ideal_ai_response", "")),
            rationale=str(payload.get("rationale", "")),
        )
    except ValueError as exc:
        return _error(422, "invalid_teaching_case", str(exc))
    manager = request.app["session_manager"]
    session = await manager.get_session(session_id)
    try:
        record, report = await _service(request).submit(
            case, session=session, session_id=session_id
        )
    except ValueError as exc:
        return _error(409, "duplicate_teaching_case", str(exc))
    return web.json_response(
        {
            "record": record.to_json(),
            "report": {
                "envelope_id": report.envelope_id,
                "total_chunks": report.total_chunks,
                "processed_chunks": report.processed_chunks,
                "skipped_chunks": report.skipped_chunks,
                "ended_scene": report.ended_scene,
                "turns": [
                    {
                        "chunk_id": turn.chunk_id,
                        "locator": turn.locator,
                        "turn_succeeded": turn.turn_succeeded,
                        "skipped_reason": turn.skipped_reason,
                    }
                    for turn in report.turns
                ],
            },
        }
    )


async def _handle_list(request: web.Request) -> web.Response:
    records = _service(request).list_records()
    return web.json_response(
        {
            "teaching_cases": [record.to_json() for record in records],
            "count": len(records),
        }
    )


async def _handle_retire(request: web.Request) -> web.Response:
    case_id = request.match_info["case_id"]
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - empty body means no reason given
        payload = {}
    reason = str(payload.get("reason", "")) if isinstance(payload, dict) else ""
    try:
        record = _service(request).retire(case_id, reason=reason)
    except KeyError as exc:
        return _error(404, "teaching_case_not_found", str(exc))
    except ValueError as exc:
        return _error(409, "teaching_case_already_retired", str(exc))
    return web.json_response({"record": record.to_json()})


__all__ = [
    "TeachingCase",
    "TeachingCaseRecord",
    "TeachingCaseService",
    "TeachingCaseStatus",
    "register_teaching_case_routes",
    "teaching_case_envelope",
]
