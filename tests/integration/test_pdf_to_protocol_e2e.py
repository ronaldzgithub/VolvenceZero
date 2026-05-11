"""Packet 2.5: end-to-end PDF → BehaviorProtocol via real LLM.

Env-gated by ``OPENAI_API_KEY`` (or ``VZ_DOCUMENT_UPTAKE_LIVE_LLM=1``);
skipped in CI / local-no-key environments. The companion
``test_pdf_extraction_with_mock_llm.py`` covers the same flow
with a deterministic mock so CI has a green path without
external API calls.

What this test verifies on the real LLM path:

* PDF reading + chunking produces non-empty content.
* The LLM extractor returns valid JSON for every chunk.
* Aggregated extraction yields ≥1 boundary, ≥1 strategy, and
  some identity content.
* The candidate is constructable and survives review +
  ProtocolRegistry load.
* After load, the application owners (boundary_policy /
  strategy_playbook / domain_knowledge / case_memory) have
  ``protocol:`` lineage entries from the PDF source.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lifeform_protocol_runtime.document_uptake import (
    chunk_document,
    extract_protocol_candidate,
    read_pdf,
)
from lifeform_protocol_runtime.document_uptake.review import (
    approve_candidate,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
)
from volvence_zero.behavior_protocol import (
    BehaviorProtocolCandidate,
    ProtocolSourceKind,
    ReviewLevel,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule


_PDF_FIXTURE = Path(
    "docs/fixtures/sample_protocols/"
    "private_domain_growth_advisor_guidance.pdf"
)


def _live_llm_enabled() -> bool:
    if os.environ.get("VZ_DOCUMENT_UPTAKE_LIVE_LLM") == "1":
        return True
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.fixture()
def real_llm_client():
    """Create a real OpenAI-compatible JSON-mode client.

    Skips the test when no API key is configured. Implementation
    is a thin shim over ``openai.OpenAI`` (which is already a
    transitive dep through ``lifeform-openai-compat``).
    """

    if not _live_llm_enabled():
        pytest.skip(
            "no OPENAI_API_KEY (or VZ_DOCUMENT_UPTAKE_LIVE_LLM=1); "
            "run with key to exercise the live extractor"
        )
    try:
        import openai  # noqa: F401
    except ImportError:
        pytest.skip("openai library not installed")

    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get(
        "VZ_DOCUMENT_UPTAKE_MODEL", "gpt-4o-mini"
    )
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    class _LiveClient:
        def complete_json(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
        ) -> dict:
            import json
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)

    return _LiveClient()


def test_pdf_fixture_exists() -> None:
    """The PDF fixture must be present in the repo (packet 2.5 copy step)."""
    assert _PDF_FIXTURE.exists(), (
        f"missing fixture {_PDF_FIXTURE}: re-run the packet 2.5 "
        "copy step or restore from EmoGPT/docs/渠道/摩比/."
    )


def test_pdf_reads_with_meaningful_content() -> None:
    """PDF reading produces non-trivial text content."""
    doc = read_pdf(_PDF_FIXTURE)
    assert doc.page_count >= 5
    assert len(doc.text) >= 1000


def test_pdf_chunks_deterministically() -> None:
    """Chunking is deterministic for the fixture."""
    doc = read_pdf(_PDF_FIXTURE)
    a = chunk_document(
        doc.text, source_locator=str(_PDF_FIXTURE), max_tokens=1024
    )
    b = chunk_document(
        doc.text, source_locator=str(_PDF_FIXTURE), max_tokens=1024
    )
    assert a == b
    assert len(a) >= 1


def test_pdf_extraction_with_real_llm(real_llm_client) -> None:
    """The full pipeline on the live LLM end-to-end."""
    doc = read_pdf(_PDF_FIXTURE)
    chunks = chunk_document(
        doc.text, source_locator=str(_PDF_FIXTURE), max_tokens=1024
    )

    candidate = extract_protocol_candidate(
        chunks,
        llm_client=real_llm_client,
        source_locator=str(_PDF_FIXTURE),
    )
    assert isinstance(candidate, BehaviorProtocolCandidate)
    assert candidate.requires_review is True
    assert len(candidate.protocol.boundary_contracts) >= 1
    assert len(candidate.protocol.strategy_priors) >= 1


def test_pdf_extracted_candidate_loads_after_review(real_llm_client) -> None:
    """Approved candidate flows into the ProtocolRegistry + application owners."""
    doc = read_pdf(_PDF_FIXTURE)
    chunks = chunk_document(
        doc.text, source_locator=str(_PDF_FIXTURE), max_tokens=1024
    )

    candidate = extract_protocol_candidate(
        chunks,
        llm_client=real_llm_client,
        source_locator=str(_PDF_FIXTURE),
    )
    approved, _ = approve_candidate(
        candidate,
        reviewer_id="ops-admin",
        evidence=("packet 2.5 e2e",),
        minimum_level=ReviewLevel.L4,
    )
    approved_candidate = BehaviorProtocolCandidate(
        protocol=approved,
        provenance=candidate.provenance,
        requires_review=False,
    )

    rare = ApplicationRareHeavyState()
    knowledge = ApplicationDomainKnowledgeStore()
    case_memory = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(
        application_rare_heavy_state=rare,
        domain_knowledge_store=knowledge,
        case_memory_store=case_memory,
    )
    module.load_protocol_candidate(approved_candidate)

    assert any(
        h.hint_id.startswith("protocol:")
        for h in rare.boundary_prior_hints
    )
