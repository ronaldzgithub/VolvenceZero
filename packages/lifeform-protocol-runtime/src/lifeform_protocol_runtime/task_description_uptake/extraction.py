"""Packet 7.1: TaskDescriptionUptake — convert one short prompt
to a ``BehaviorProtocolCandidate``.

Reuses the document_uptake extractor by wrapping the description
into a single synthetic ``DocumentChunk``. This keeps the merge
/ schema-mapping logic DRY: the only difference between
DocumentUptake and TaskDescriptionUptake is the chunking step.
"""

from __future__ import annotations

import datetime as _dt

from volvence_zero.behavior_protocol import (
    BehaviorProtocolCandidate,
    ProtocolSourceKind,
)

from lifeform_protocol_runtime.document_uptake.extraction import (
    LlmJsonClient,
    extract_protocol_candidate,
)
from lifeform_protocol_runtime.document_uptake.ingestion import DocumentChunk


def extract_protocol_from_description(
    description: str,
    *,
    llm_client: LlmJsonClient,
    protocol_id: str,
    advisor_name: str,
    extractor_id: str = "task_description_uptake_v0",
) -> BehaviorProtocolCandidate:
    """Convert a free-text task description into a candidate protocol.

    Wraps ``description`` in a single ``DocumentChunk`` and
    delegates to the document_uptake extractor. The returned
    candidate is marked ``requires_review=True`` and uses
    :attr:`ProtocolSourceKind.TASK_DESCRIPTION` provenance so
    downstream review can distinguish it from PDF-extracted
    protocols.
    """

    if not description or not description.strip():
        raise ValueError("extract_protocol_from_description: empty description")

    chunks = (
        DocumentChunk(
            source_locator=f"task_description://{protocol_id}",
            chunk_index=0,
            source_offset=0,
            text=description.strip(),
        ),
    )
    candidate = extract_protocol_candidate(
        chunks,
        llm_client=llm_client,
        source_locator=f"task_description://{protocol_id}",
        source_kind=ProtocolSourceKind.TASK_DESCRIPTION,
        extractor_id=extractor_id,
        protocol_id_seed=protocol_id,
    )
    return candidate


__all__ = ["extract_protocol_from_description"]
