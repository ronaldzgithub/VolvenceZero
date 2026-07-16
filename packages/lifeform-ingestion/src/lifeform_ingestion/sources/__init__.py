"""Source adapters for runtime ingestion (Gap 3).

Each adapter is a pure chunker: given external input (a string, a
file path, a JSON payload), it returns an ``IngestionEnvelope``. The
adapter NEVER touches the kernel \u2014 it only produces frozen data
that the pipeline then feeds through ``LifeformSession.run_turn``.

Slice 1 ships ``plain_text`` and ``task_result``. Slice 2 adds
``pdf`` and ``docx``. Slice 2b adds ``web`` (requests +
readability-lxml, explicit robots.txt gate, bounded fetch — no
Playwright / browser automation).

The PDF / DOCX / web adapters depend on optional extras
(``lifeform-ingestion[pdf]`` / ``[docx]`` / ``[web]``) so a thin
install that only ingests plain text / JSON pays no dependency cost.
Attempting ``envelope_from_pdf_file`` / ``envelope_from_url`` without
the extra installed raises a typed error naming the exact install
incantation.
"""

from lifeform_ingestion.sources.docx import (
    DEFAULT_MAX_PARAGRAPHS,
    DocxIngestionError,
    envelope_from_docx_bytes,
    envelope_from_docx_file,
)
from lifeform_ingestion.sources.job_trace import envelope_from_job_trace
from lifeform_ingestion.sources.pdf import (
    DEFAULT_MAX_PAGES,
    PdfIngestionError,
    envelope_from_pdf_bytes,
    envelope_from_pdf_file,
)
from lifeform_ingestion.sources.plain_text import (
    DEFAULT_MAX_CHUNK_CHARS,
    chunk_plain_text,
    envelope_from_text,
)
from lifeform_ingestion.sources.task_result import envelope_from_task_result
from lifeform_ingestion.sources.web import (
    DEFAULT_MAX_CONTENT_BYTES,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_S,
    WebIngestionError,
    envelope_from_html,
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
