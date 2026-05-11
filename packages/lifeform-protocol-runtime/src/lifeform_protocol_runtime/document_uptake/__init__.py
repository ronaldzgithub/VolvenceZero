"""Document uptake adapters for the Behavior Protocol Runtime.

Sub-packages:

* :mod:`.ingestion` — PDF / Markdown / plain-text reading +
  deterministic chunking. No LLM dependency; pure-Python.
* :mod:`.extraction` (packet 2.3) — LLM JSON-mode extraction
  driver (chunk → ``BehaviorProtocolCandidate``).
* :mod:`.review` (packet 2.4) — ``approve_candidate`` /
  ``reject_candidate`` helpers that route candidates through
  R10 ModificationGate review.
* :mod:`.prompts` (packet 2.3) — centralized LLM prompts
  (per ``.cursor/rules/llm-prompt-centralization.mdc``).
"""

from __future__ import annotations

from lifeform_protocol_runtime.document_uptake.extraction import (
    LlmJsonClient,
    MockLlmJsonClient,
    extract_protocol_candidate,
)
from lifeform_protocol_runtime.document_uptake.ingestion import (
    DocumentChunk,
    DocumentText,
    chunk_document,
    read_markdown,
    read_pdf,
)

__all__ = [
    "DocumentChunk",
    "DocumentText",
    "LlmJsonClient",
    "MockLlmJsonClient",
    "chunk_document",
    "extract_protocol_candidate",
    "read_markdown",
    "read_pdf",
]
