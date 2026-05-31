"""Job-trace source adapter (D-de-1; huaxiaobao_get_job_trace ingestion).

This is the kernel-side ingestion contract for a long-running job's
execution trace. It consumes the JSON shape returned by the
``huaxiaobao_get_job_trace`` tool (or any future backend that emits the
same shape) and turns it into a chunked ``IngestionEnvelope`` so the
full step + tool-audit trace reaches the kernel one bounded turn at a
time, instead of being collapsed into the ~320-char kernel tool-result
summary.

Expected (provisional) trace shape::

    {
        "job_id": "...",
        "status": "completed",
        "intent": "...",
        "total_cost": ...,
        "steps": [{"step_order": 1, "tool_name": "...", "status": "...", ...}],
        "execution_evidence": [{"audit_id": "...", "tool_name": "...", ...}],
        "metadata": {...},
        "outputs": [...],
    }

Design constraints (match the other source adapters):

* Pure data in, pure data out \u2014 no kernel imports, no network, no
  dependency on the ``vz-bundle`` tool that produced the payload. The
  adapter works whether the trace came from a real backend or a stub.
* One chunk per step and per evidence row (plus a header chunk), each
  bounded by ``max_chunk_chars``; oversized records are sub-chunked via
  the shared plain-text chunker so nothing is silently truncated.
* Fails loudly when the trace carries no usable content \u2014 an empty
  trace is almost certainly a caller bug, not something to paper over
  with an empty envelope.

NOTE on the live huaxiaobao backend: the real ``GET /jobs/{id}/trace``
route does not exist in the currently-available SuperJoe backend (the
``external/huaxiaobao`` submodule is not checked out here), so a live
``huaxiaobao_get_job_trace`` call returns an ``unsupported`` payload
today. This adapter is the kernel-side half of the contract and is
exercised independently of that backend; wiring it onto a real trace is
unblocked the moment the backend route lands.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping

from lifeform_ingestion.envelope import (
    IngestionChunk,
    IngestionComplianceProfile,
    IngestionEnvelope,
    IngestionProvenance,
    IngestionSourceKind,
)
from lifeform_ingestion.sources.plain_text import (
    DEFAULT_MAX_CHUNK_CHARS,
    chunk_plain_text,
)


def envelope_from_job_trace(
    trace: Mapping[str, Any],
    *,
    job_id: str,
    uploader: str = "tool-runtime",
    upload_ts_ms: int | None = None,
    envelope_id: str | None = None,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> IngestionEnvelope:
    """Turn a job-trace dict into a chunked ``IngestionEnvelope``.

    Emits, in order: one header chunk, one chunk per ``steps`` entry,
    then one chunk per ``execution_evidence`` entry. Oversized records
    are split into ``...:part:NNNN`` sub-chunks. Raises ``ValueError``
    if ``job_id`` is empty or the trace yields no chunk content.
    """
    if not job_id.strip():
        raise ValueError("envelope_from_job_trace: job_id must be non-empty")
    if max_chunk_chars <= 0:
        raise ValueError(
            f"envelope_from_job_trace: max_chunk_chars must be > 0, "
            f"got {max_chunk_chars!r}"
        )

    serialized = json.dumps(trace, sort_keys=True, default=str)
    integrity_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if envelope_id is None:
        envelope_id = f"job-trace:{job_id}:{integrity_hash[:8]}"
    if upload_ts_ms is None:
        upload_ts_ms = int(time.time() * 1000.0)

    steps = _as_records(trace.get("steps"))
    evidence = _as_records(trace.get("execution_evidence"))

    chunks: list[IngestionChunk] = []
    header_text = _header_text(trace, job_id=job_id, step_count=len(steps), evidence_count=len(evidence))
    chunks.extend(
        _bounded_chunks(
            envelope_id=envelope_id,
            kind="header",
            index=0,
            text=header_text,
            locator=f"job_id={job_id};part=header",
            max_chunk_chars=max_chunk_chars,
        )
    )

    for index, step in enumerate(steps):
        text = _stringify_record(step)
        if not text.strip():
            continue
        chunks.extend(
            _bounded_chunks(
                envelope_id=envelope_id,
                kind="step",
                index=index,
                text=text,
                locator=f"job_id={job_id};step={index}",
                max_chunk_chars=max_chunk_chars,
            )
        )

    for index, row in enumerate(evidence):
        text = _stringify_record(row)
        if not text.strip():
            continue
        chunks.extend(
            _bounded_chunks(
                envelope_id=envelope_id,
                kind="evidence",
                index=index,
                text=text,
                locator=f"job_id={job_id};evidence={index}",
                max_chunk_chars=max_chunk_chars,
            )
        )

    if not chunks:
        raise ValueError(
            f"envelope_from_job_trace: trace for job_id={job_id!r} produced "
            f"no chunk content (no header / steps / execution_evidence). "
            f"Caller should fall back to huaxiaobao_get_job_result."
        )

    provenance = IngestionProvenance(
        uploader=uploader,
        upload_ts_ms=upload_ts_ms,
        source_uri=f"job-trace:{job_id}",
        integrity_hash=integrity_hash,
    )
    return IngestionEnvelope(
        envelope_id=envelope_id,
        source_kind=IngestionSourceKind.TASK_RESULT,
        chunks=tuple(chunks),
        provenance=provenance,
        compliance_profile=compliance_profile,
        partial_failures=(),
    )


def _header_text(
    trace: Mapping[str, Any],
    *,
    job_id: str,
    step_count: int,
    evidence_count: int,
) -> str:
    lines = [
        f"job_id: {job_id}",
        f"status: {trace.get('status', 'unknown')}",
        f"steps: {step_count}",
        f"execution_evidence: {evidence_count}",
    ]
    for optional in ("intent", "total_cost", "session_id"):
        if optional in trace and trace[optional] not in (None, ""):
            lines.append(f"{optional}: {trace[optional]}")
    return "\n".join(lines)


def _bounded_chunks(
    *,
    envelope_id: str,
    kind: str,
    index: int,
    text: str,
    locator: str,
    max_chunk_chars: int,
) -> list[IngestionChunk]:
    base_id = f"{envelope_id}:{kind}:{index:04d}"
    if len(text) <= max_chunk_chars:
        return [
            IngestionChunk(
                chunk_id=base_id,
                text=text,
                locator=locator,
                confidence=1.0,
            )
        ]
    pieces = chunk_plain_text(text, max_chunk_chars=max_chunk_chars)
    return [
        IngestionChunk(
            chunk_id=f"{base_id}:part:{part_index:04d}",
            text=segment,
            locator=f"{locator};part={part_index};offset={start}-{end}",
            confidence=1.0,
        )
        for part_index, (segment, start, end) in enumerate(pieces)
    ]


def _as_records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _stringify_record(record: Mapping[str, Any]) -> str:
    """Render a step / evidence row as stable ``key: value`` lines.

    Nested containers are JSON-encoded so a single record stays on a
    human-readable, greppable surface without recursively exploding.
    """
    lines: list[str] = []
    for key, value in record.items():
        if isinstance(value, (Mapping, list, tuple)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


__all__ = [
    "envelope_from_job_trace",
]
