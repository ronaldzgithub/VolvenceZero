"""Source adapters for runtime ingestion (Gap 3).

Each adapter is a pure chunker: given external input (a string, a
file path, a JSON payload), it returns an ``IngestionEnvelope``. The
adapter NEVER touches the kernel \u2014 it only produces frozen data
that the pipeline then feeds through ``LifeformSession.run_turn``.

Slice 1 ships ``plain_text`` and ``task_result``. PDF / DOCX / web
sources are slice 2 territory and can be added without touching the
envelope / pipeline contracts.
"""

from lifeform_ingestion.sources.plain_text import (
    chunk_plain_text,
    envelope_from_text,
)
from lifeform_ingestion.sources.task_result import envelope_from_task_result

__all__ = [
    "chunk_plain_text",
    "envelope_from_task_result",
    "envelope_from_text",
]
