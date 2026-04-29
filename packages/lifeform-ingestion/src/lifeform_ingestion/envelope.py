"""Immutable envelope shapes for Runtime Ingestion (Gap 3).

See ``docs/specs/runtime-ingestion.md`` for the full invariants. The
short version:

* ``IngestionEnvelope`` is the ONLY shape ``IngestionPipeline``
  accepts. Source adapters (plain_text / task_result / book / web)
  are pure chunkers that return one of these.
* Chunks are ordered \u2014 the pipeline replays them in sequence so
  later chunks can reference earlier facts (e.g. page N of a book
  references page N-1).
* ``parse_error`` on a chunk is non-empty only if the adapter
  FAILED to parse that chunk; the locator is still set so a human
  can trace *where* the failure occurred. Silently dropping a page
  is forbidden \u2014 partial failure must be visible via
  ``IngestionEnvelope.partial_failures``.
* Everything here is stdlib-only. No kernel imports: the envelope
  travels through ``LifeformSession.run_turn`` and never reaches
  into any owner store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IngestionSourceKind(str, Enum):
    """Where the content came from.

    Used purely as an audit label; ``IngestionPipeline`` does not
    branch on source_kind. Adapters produce envelopes with the
    appropriate kind so downstream telemetry / reflection writeback
    can key off a stable enum instead of free-form strings.
    """

    BOOK = "book"
    WEB = "web"
    TASK_RESULT = "task_result"
    CORPUS = "corpus"  # generic plain-text corpus / inline string


class IngestionComplianceProfile(str, Enum):
    """How the lifeform should treat the content.

    ``FORCED`` \u2014 operator-supplied content that should be ingested
    without the lifeform "resisting" it (pushes the vitals apprentice
    override on via the INGESTION trigger_kind, so drive deviation
    does not count as proactive PE during ingestion).

    ``CONSULTATIVE`` \u2014 the content is presented as a normal user
    turn, vitals / PE recruit as usual. Future adapters may use this
    when the user pastes something and asks the lifeform to react.
    """

    FORCED = "forced"
    CONSULTATIVE = "consultative"


@dataclass(frozen=True)
class IngestionProvenance:
    """Who uploaded this content, when, and where from.

    Stored on the envelope so reflection writeback can attribute
    ingested records back to an ``envelope_id`` + ``source_uri``;
    retired cases / wrong memories can later be reconciled by
    ``envelope_id`` in audit workflows.
    """

    uploader: str
    upload_ts_ms: int
    source_uri: str
    integrity_hash: str  # SHA256 of original source; empty string if unavailable

    def __post_init__(self) -> None:
        if not self.uploader.strip():
            raise ValueError("IngestionProvenance.uploader must be non-empty")
        if self.upload_ts_ms < 0:
            raise ValueError(
                f"IngestionProvenance.upload_ts_ms must be >= 0, "
                f"got {self.upload_ts_ms!r}"
            )
        if not self.source_uri.strip():
            raise ValueError("IngestionProvenance.source_uri must be non-empty")


@dataclass(frozen=True)
class IngestionChunk:
    """One chunk of ingestable content.

    ``locator`` is a human-readable breadcrumb like ``"page=3,offset=1024"``
    or ``"field=ideal_ai_response"``. It is surface data only \u2014 the
    pipeline passes it through to TurnSummary via the chunk_id so
    operators can trace which part of a PDF produced which turn.

    ``parse_error`` is non-empty ONLY when the source adapter failed to
    parse this chunk. A chunk with a non-empty ``parse_error`` has
    ``text`` set to a safe placeholder (e.g. ``""``) and is
    intentionally kept in the envelope so the pipeline / report can
    see that something was lost at this position \u2014 silent drops are
    forbidden (red line C: no swallowed errors).
    """

    chunk_id: str
    text: str
    locator: str
    confidence: float = 1.0
    parse_error: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ValueError("IngestionChunk.chunk_id must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"IngestionChunk.confidence must be in [0,1], "
                f"got {self.confidence!r}"
            )
        # ``text`` may be empty only when a parse_error is recorded \u2014
        # an empty chunk with no error is almost certainly a bug in the
        # adapter (why emit an empty chunk?) and we fail loudly.
        if not self.text.strip() and not self.parse_error.strip():
            raise ValueError(
                f"IngestionChunk.text is empty but no parse_error is "
                f"recorded for chunk_id={self.chunk_id!r}; silent empty "
                f"chunks are forbidden."
            )

    @property
    def has_parse_error(self) -> bool:
        return bool(self.parse_error.strip())


@dataclass(frozen=True)
class IngestionEnvelope:
    """Top-level immutable ingestion envelope.

    ``partial_failures`` is the canonical audit surface: it lists
    every ``chunk_id`` whose ``parse_error`` is non-empty so a
    caller can decide whether to retry / escalate. Keeping it as a
    separate tuple (rather than recomputing from chunks each time) is
    a stable-wire-format choice \u2014 consumers that only look at
    ``partial_failures`` do not need to traverse every chunk.
    """

    envelope_id: str
    source_kind: IngestionSourceKind
    chunks: tuple[IngestionChunk, ...]
    provenance: IngestionProvenance
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED
    partial_failures: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.envelope_id.strip():
            raise ValueError("IngestionEnvelope.envelope_id must be non-empty")
        if not self.chunks:
            raise ValueError(
                f"IngestionEnvelope.chunks must be non-empty; envelope "
                f"{self.envelope_id!r} has no chunks. An empty source "
                f"should fail at the adapter stage, not surface as an "
                f"empty envelope."
            )
        # Chunk ids must be unique within an envelope so the report /
        # audit log can name each one unambiguously.
        chunk_ids = [c.chunk_id for c in self.chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError(
                f"IngestionEnvelope.chunks must have unique chunk_ids, "
                f"got {chunk_ids!r}"
            )
        # partial_failures must be a strict subset of the chunk_ids and
        # must include every chunk that actually has a parse_error.
        declared_failures = set(self.partial_failures)
        actual_failures = {c.chunk_id for c in self.chunks if c.has_parse_error}
        if declared_failures != actual_failures:
            raise ValueError(
                f"IngestionEnvelope.partial_failures must match chunks "
                f"with non-empty parse_error exactly. declared="
                f"{sorted(declared_failures)!r} actual="
                f"{sorted(actual_failures)!r}"
            )

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def successful_chunks(self) -> tuple[IngestionChunk, ...]:
        return tuple(c for c in self.chunks if not c.has_parse_error)

    @property
    def failed_chunks(self) -> tuple[IngestionChunk, ...]:
        return tuple(c for c in self.chunks if c.has_parse_error)


__all__ = [
    "IngestionChunk",
    "IngestionComplianceProfile",
    "IngestionEnvelope",
    "IngestionProvenance",
    "IngestionSourceKind",
]
