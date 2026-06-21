"""Lifeform-side TaskUptake adapters for the Behavior Protocol Runtime.

This wheel hosts the lifeform-side input adapters that turn
external sources (PDFs, Markdown, free-text task descriptions,
API injections, directory scans) into
``BehaviorProtocolCandidate`` instances. The kernel-side owner
(``vz-application.protocol_runtime.ProtocolRegistryModule``)
consumes those candidates after R10 ModificationGate review.

Layering:

* ``vz-contracts`` — schema (``BehaviorProtocol``,
  ``BehaviorProtocolCandidate``, ``ProtocolProvenance``).
* ``vz-application.protocol_runtime`` — kernel owner that holds
  the registry, runs the activation loop, and applies compiled
  artifacts to application owners.
* ``lifeform-protocol-runtime`` (this wheel) — lifeform-side
  uptake adapters. Allowed to import LLM clients
  (``lifeform-openai-compat``) and PDF parsing libs.

Import boundary (enforced by
``tests/contracts/test_import_boundaries.py``):

* This wheel may import: ``vz-contracts``, ``vz-application``,
  ``lifeform-openai-compat``, third-party libs (``pypdf``).
* This wheel may NOT import: ``dlaas_platform_*`` (platform
  tier), other ``lifeform-*`` wheels (sibling lifeform-domain
  fixtures stay out of the uptake pipeline).
* Reverse imports forbidden: ``vz-*`` MUST NOT import this
  wheel (``vz-* ↛ lifeform-*`` hard rule).
"""

from __future__ import annotations

from lifeform_protocol_runtime.document_uptake import (
    DocumentChunk,
    DocumentText,
    LlmJsonClient,
    MockLlmJsonClient,
    chunk_document,
    extract_protocol_candidate,
    read_markdown,
    read_pdf,
)
from lifeform_protocol_runtime.api_injection_uptake import (
    inject_protocol_from_payload,
)
from lifeform_protocol_runtime.directory_scan_uptake import (
    DirectoryScanResult,
    scan_directory_for_protocols,
)
from lifeform_protocol_runtime.llm_clients import (
    OpenAiCompatConfig,
    OpenAiCompatJsonClient,
)
from lifeform_protocol_runtime.mentor_intake import (
    MentorIntakeApplyMode,
    MentorIntakeDecision,
    MentorIntakeKind,
    MentorIntakeRequest,
    ReviewedKnowledgeDraft,
    build_protocol_revision_proposal,
    classify_mentor_intake,
    extract_reviewed_knowledge_from_guidance,
    extract_signature_case_from_guidance,
    resolve_experience_outcome_kind,
)
from lifeform_protocol_runtime.serialization import (
    SCHEMA_VERSION as PROTOCOL_PAYLOAD_SCHEMA_VERSION,
    ProtocolPayloadSchemaError,
    protocol_from_payload,
    protocol_to_payload,
)
from lifeform_protocol_runtime.task_description_uptake import (
    extract_protocol_from_description,
)

__all__ = [
    "DirectoryScanResult",
    "DocumentChunk",
    "DocumentText",
    "LlmJsonClient",
    "MentorIntakeApplyMode",
    "MentorIntakeDecision",
    "MentorIntakeKind",
    "MentorIntakeRequest",
    "MockLlmJsonClient",
    "OpenAiCompatConfig",
    "OpenAiCompatJsonClient",
    "PROTOCOL_PAYLOAD_SCHEMA_VERSION",
    "ProtocolPayloadSchemaError",
    "ReviewedKnowledgeDraft",
    "build_protocol_revision_proposal",
    "chunk_document",
    "classify_mentor_intake",
    "extract_protocol_candidate",
    "extract_protocol_from_description",
    "extract_reviewed_knowledge_from_guidance",
    "extract_signature_case_from_guidance",
    "resolve_experience_outcome_kind",
    "inject_protocol_from_payload",
    "protocol_from_payload",
    "protocol_to_payload",
    "read_markdown",
    "read_pdf",
    "scan_directory_for_protocols",
]
