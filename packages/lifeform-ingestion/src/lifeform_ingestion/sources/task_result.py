"""Task-result source adapter (Gap 3 slice 1; D-de-1 chunked ingestion).

Structured JSON / dict task results get split per-field so each
chunk carries a single piece of information (status, summary,
detail, artifact refs). The lifeform ingests each piece as its own
turn, which works well with the kernel's per-turn update granularity.

This adapter is the intended entry point for long structured tool
outputs that are bigger than a single ``submit_tool_result`` event
would comfortably carry, e.g. a web search result with a page of
excerpts: the lifeform ingests each excerpt as its own turn and the
slow loop consolidates them into durable knowledge.

D-de-1 (chunked artifact ingestion): a single ``submit_tool_result``
kernel summary truncates long detail to ~320 chars, so a big field
(a multi-page ``detail`` or ``findings`` blob) used to lose most of
its content. To preserve the full artifact, any field whose text
exceeds ``max_chunk_chars`` is now split into bounded sub-chunks
(reusing the plain-text chunker) instead of becoming one oversized
chunk. Fields that fit keep their original single ``:field:<name>``
chunk id so the change is backward-compatible.
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


DEFAULT_FIELD_ORDER: tuple[str, ...] = (
    "summary",
    "detail",
    "status",
    "findings",
    "notes",
    "artifact_refs",
)


def envelope_from_task_result(
    task_result: Mapping[str, Any],
    *,
    task_id: str,
    uploader: str = "tool-runtime",
    upload_ts_ms: int | None = None,
    envelope_id: str | None = None,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
    field_order: tuple[str, ...] = DEFAULT_FIELD_ORDER,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> IngestionEnvelope:
    """Turn a structured task_result dict into an ``IngestionEnvelope``.

    Each known field from ``field_order`` that is present AND has
    non-empty content becomes one or more chunks. Unknown fields in
    the payload are ignored (this is a keyword-agnostic structural
    adapter \u2014 we do not try to interpret arbitrary fields).

    ``chunk_id`` embeds the field name so downstream audit can tell
    "which field of the task_result produced this knowledge" without
    keeping the original payload around. A field that fits within
    ``max_chunk_chars`` keeps the canonical ``<envelope>:field:<name>``
    id; an oversized field is split into ``...:field:<name>:part:NNNN``
    chunks (D-de-1) so the full artifact survives instead of being
    truncated by the kernel summary.

    Fails loudly if NO field in ``field_order`` yielded a non-empty
    chunk \u2014 an ingestion call with no actionable content is almost
    certainly a bug (why call at all?). The empty-task-result case
    should be filtered by the caller.
    """
    if not task_id.strip():
        raise ValueError("envelope_from_task_result: task_id must be non-empty")
    if max_chunk_chars <= 0:
        raise ValueError(
            f"envelope_from_task_result: max_chunk_chars must be > 0, "
            f"got {max_chunk_chars!r}"
        )
    serialized = json.dumps(task_result, sort_keys=True, default=str)
    integrity_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if envelope_id is None:
        envelope_id = f"task-ingestion:{task_id}:{integrity_hash[:8]}"
    if upload_ts_ms is None:
        upload_ts_ms = int(time.time() * 1000.0)
    chunks: list[IngestionChunk] = []
    for field_name in field_order:
        if field_name not in task_result:
            continue
        value = task_result[field_name]
        text = _stringify(value)
        if not text.strip():
            continue
        chunks.extend(
            _chunks_for_field(
                envelope_id=envelope_id,
                task_id=task_id,
                field_name=field_name,
                text=text,
                max_chunk_chars=max_chunk_chars,
            )
        )
    if not chunks:
        raise ValueError(
            f"envelope_from_task_result: task_result for task_id={task_id!r} "
            f"produced no non-empty chunks for fields {field_order!r}. "
            f"Caller should filter empty results before invoking."
        )
    provenance = IngestionProvenance(
        uploader=uploader,
        upload_ts_ms=upload_ts_ms,
        source_uri=f"task-result:{task_id}",
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


def _chunks_for_field(
    *,
    envelope_id: str,
    task_id: str,
    field_name: str,
    text: str,
    max_chunk_chars: int,
) -> list[IngestionChunk]:
    """Produce one (small field) or many (oversized field) chunks.

    Small fields keep the canonical ``<envelope>:field:<name>`` id so
    existing consumers / audits do not change. Oversized fields are
    split with the shared plain-text chunker into deterministic
    ``...:part:NNNN`` chunks whose locators carry the byte offsets, so
    the full artifact reaches the kernel one bounded turn at a time
    instead of being truncated to the ~320-char kernel summary.
    """
    if len(text) <= max_chunk_chars:
        return [
            IngestionChunk(
                chunk_id=f"{envelope_id}:field:{field_name}",
                text=text,
                locator=f"task_id={task_id};field={field_name}",
                confidence=1.0,
            )
        ]
    pieces = chunk_plain_text(text, max_chunk_chars=max_chunk_chars)
    return [
        IngestionChunk(
            chunk_id=f"{envelope_id}:field:{field_name}:part:{index:04d}",
            text=segment,
            locator=(
                f"task_id={task_id};field={field_name};"
                f"part={index};offset={start}-{end}"
            ),
            confidence=1.0,
        )
        for index, (segment, start, end) in enumerate(pieces)
    ]


def _stringify(value: Any) -> str:
    """Turn a structured field value into a human-readable string.

    Strings pass through; iterables become bullet lists; dicts become
    ``key: value`` lists; everything else goes through ``str()``.
    Non-recursive \u2014 this adapter is deliberately shallow so it
    doesn't accidentally swallow a deeply-nested tree into one
    giant chunk.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return "\n".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {item}" for item in value)
    return str(value)


__all__ = [
    "DEFAULT_FIELD_ORDER",
    "envelope_from_task_result",
]
