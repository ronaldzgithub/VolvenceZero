"""Lifeform-side ingestion adapter (Gap 3 slice 1).

Public API:

* ``IngestionEnvelope`` / ``IngestionChunk`` / ``IngestionProvenance``
* ``IngestionSourceKind`` / ``IngestionComplianceProfile``
* ``IngestionPipeline`` + ``IngestionReport`` / ``IngestionTurnRecord``
* ``chunk_plain_text`` / ``envelope_from_text`` / ``envelope_from_task_result``

See ``docs/specs/runtime-ingestion.md`` for the spec and
``docs/implementation/13_emogpt_prd_alignment_upgrade.md`` Gap 3 for
the phased rollout plan. PDF / DOCX / web adapters are slice 2.
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
    chunk_plain_text,
    envelope_from_task_result,
    envelope_from_text,
)

__all__ = [
    "IngestionChunk",
    "IngestionComplianceProfile",
    "IngestionEnvelope",
    "IngestionPipeline",
    "IngestionProvenance",
    "IngestionReport",
    "IngestionSourceKind",
    "IngestionTurnRecord",
    "chunk_plain_text",
    "envelope_from_task_result",
    "envelope_from_text",
]
