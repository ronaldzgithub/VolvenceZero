"""Lifeform-side ingestion adapter (Gap 3 slices 1+2+2b).

Public API:

* ``IngestionEnvelope`` / ``IngestionChunk`` / ``IngestionProvenance``
* ``IngestionSourceKind`` / ``IngestionComplianceProfile``
* ``IngestionPipeline`` + ``IngestionReport`` / ``IngestionTurnRecord``
* slice 1 sources: ``chunk_plain_text`` / ``envelope_from_text`` /
  ``envelope_from_task_result``
* slice 2 sources: ``envelope_from_pdf_file`` / ``envelope_from_pdf_bytes``
  (requires the ``[pdf]`` extra) / ``envelope_from_docx_file`` /
  ``envelope_from_docx_bytes`` (requires the ``[docx]`` extra)
* slice 2b sources: ``envelope_from_url`` / ``envelope_from_html``
  (requires the ``[web]`` extra: requests + readability-lxml,
  explicit robots.txt gate, no browser automation)
* typed errors: ``PdfIngestionError`` / ``DocxIngestionError`` /
  ``WebIngestionError``

See ``docs/specs/runtime-ingestion.md`` for the spec and
``docs/implementation/13_emogpt_prd_alignment_upgrade.md`` Gap 3 for
the phased rollout plan.
"""

from lifeform_ingestion.envelope import (
    IngestionChunk,
    IngestionComplianceProfile,
    IngestionEnvelope,
    IngestionProvenance,
    IngestionSourceKind,
)
from lifeform_ingestion.pipeline import (
    IngestionPipeline,
    IngestionReport,
    IngestionTurnRecord,
)
from lifeform_ingestion.sources import (
    DEFAULT_MAX_CHUNK_CHARS,
    DEFAULT_MAX_CONTENT_BYTES,
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_PARAGRAPHS,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_S,
    DocxIngestionError,
    PdfIngestionError,
    WebIngestionError,
    chunk_plain_text,
    envelope_from_docx_bytes,
    envelope_from_docx_file,
    envelope_from_html,
    envelope_from_job_trace,
    envelope_from_pdf_bytes,
    envelope_from_pdf_file,
    envelope_from_task_result,
    envelope_from_text,
    envelope_from_url,
)

__all__ = [
    "DEFAULT_MAX_CHUNK_CHARS",
    "DEFAULT_MAX_CONTENT_BYTES",
    "DEFAULT_MAX_PAGES",
    "DEFAULT_MAX_PARAGRAPHS",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT_S",
    "DocxIngestionError",
    "IngestionChunk",
    "IngestionComplianceProfile",
    "IngestionEnvelope",
    "IngestionPipeline",
    "IngestionProvenance",
    "IngestionReport",
    "IngestionSourceKind",
    "IngestionTurnRecord",
    "PdfIngestionError",
    "WebIngestionError",
    "chunk_plain_text",
    "envelope_from_docx_bytes",
    "envelope_from_docx_file",
    "envelope_from_html",
    "envelope_from_job_trace",
    "envelope_from_pdf_bytes",
    "envelope_from_pdf_file",
    "envelope_from_task_result",
    "envelope_from_text",
    "envelope_from_url",
]
