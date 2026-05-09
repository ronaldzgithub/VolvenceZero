"""Wave T7 contract tests — LLM-assisted profile extraction.

Uses a fake provider returning deterministic JSON to verify:

* Extraction parses well-formed LLM JSON into a typed candidate.
* Schema-mismatch on individual chunks is recorded in ``notes``
  (not raised) — best-effort across chunks.
* Total parse failure (every chunk unparseable) raises.
* ``review_profile_candidate`` enforces non-empty reviewer fields
  and at least one boundary.
* The reviewed result is a valid :class:`CharacterSoulProfile`.

Real-LLM smoke is OUT of scope (debt #10B applies); the contract
tests here exercise the wiring only.
"""

from __future__ import annotations

from dataclasses import replace as _replace

import pytest

from lifeform_domain_character import (
    CharacterBoundaryPrior,
    CharacterSoulProfile,
    ReviewedProfileCandidate,
    extract_profile_candidate,
    review_profile_candidate,
)


_VALID_JSON_PAYLOAD = (
    '{"character_name": "Test Character", '
    '"description": "A reviewed test character.", '
    '"drive_priors": [{"name": "test_drive", "target": 0.6, '
    '"homeostatic_band": [0.4, 0.8], "pe_weight": 0.5, '
    '"initial_level": 0.5}], '
    '"boundary_priors": [{"boundary_id": "test-boundary", '
    '"trigger_reasons": ["test-trigger"], '
    '"answer_depth_limit_hint": "soft", '
    '"description": "Test boundary"}], '
    '"signature_cases": [{"case_id": "test-case", '
    '"problem_pattern": "test-pattern", '
    '"description": "Test case"}], '
    '"knowledge_seeds": [{"seed_id": "test-seed", '
    '"domain": "test", "title": "Test", '
    '"summary": "Test summary", "snippet": "Test snippet"}], '
    '"strategy_priors": [{"rule_id": "test-rule", '
    '"problem_pattern": "test-pattern", '
    '"recommended_pacing": "standard", '
    '"description": "Test strategy"}], '
    '"low_confidence_fields": ["drive_priors"], '
    '"inference_confidence": 0.7}'
)


class _FakeProvider:
    def __init__(self, payloads: tuple[str, ...]) -> None:
        self._payloads = list(payloads)
        self.calls = 0

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        del prompt, max_new_tokens, temperature
        if not self._payloads:
            raise RuntimeError("FakeProvider: no payloads left")
        self.calls += 1
        return self._payloads.pop(0)


def test_extract_profile_candidate_parses_valid_json() -> None:
    provider = _FakeProvider(payloads=(_VALID_JSON_PAYLOAD,))
    candidate = extract_profile_candidate(
        novel_text="some short novel text",
        llm_runtime=provider,
        character_focus="Test Character",
        profile_id="test-extracted",
        source_title="Test Source",
        chunk_size=100,
    )
    assert isinstance(candidate, ReviewedProfileCandidate)
    assert candidate.character_name == "Test Character"
    assert len(candidate.boundary_priors) >= 1
    assert len(candidate.drive_priors) >= 1
    assert candidate.aggregate_inference_confidence > 0.0
    assert "drive_priors" in candidate.requires_review


def test_extract_profile_candidate_records_unparseable_chunk_notes() -> None:
    """If one chunk fails parsing but another succeeds, the candidate
    still comes back with notes on the failed chunk."""
    payloads = ("not actually json", _VALID_JSON_PAYLOAD)
    provider = _FakeProvider(payloads=payloads)
    candidate = extract_profile_candidate(
        novel_text="text " * 200,  # forces 2 chunks at chunk_size=400
        llm_runtime=provider,
        character_focus="Test Character",
        profile_id="test-extracted",
        source_title="Test Source",
        chunk_size=400,
    )
    assert "unparseable response" in " ".join(candidate.notes)
    assert candidate.character_name == "Test Character"


def test_extract_profile_candidate_raises_when_all_chunks_fail() -> None:
    provider = _FakeProvider(payloads=("garbage",) * 3)
    with pytest.raises(ValueError, match="no chunk yielded"):
        extract_profile_candidate(
            novel_text="text " * 600,  # 3 chunks at 1000
            llm_runtime=provider,
            character_focus="Test Character",
            profile_id="test-extracted",
            source_title="Test Source",
            chunk_size=1000,
        )


def test_extract_profile_candidate_rejects_empty_input() -> None:
    provider = _FakeProvider(payloads=())
    with pytest.raises(ValueError, match="novel_text is empty"):
        extract_profile_candidate(
            novel_text="   ",
            llm_runtime=provider,
            character_focus="Test Character",
            profile_id="test",
            source_title="Test",
        )


def test_review_profile_candidate_produces_valid_profile() -> None:
    provider = _FakeProvider(payloads=(_VALID_JSON_PAYLOAD,))
    candidate = extract_profile_candidate(
        novel_text="some short novel text",
        llm_runtime=provider,
        character_focus="Test Character",
        profile_id="test-extracted",
        source_title="Test Source",
        chunk_size=100,
    )
    profile = review_profile_candidate(
        candidate,
        reviewer="test-reviewer",
        review_locator="pr-001",
    )
    assert isinstance(profile, CharacterSoulProfile)
    assert profile.profile_id == "test-extracted"
    assert profile.reviewed_by == "test-reviewer"
    assert profile.source_uri == "pr-001"


def test_review_rejects_empty_reviewer() -> None:
    provider = _FakeProvider(payloads=(_VALID_JSON_PAYLOAD,))
    candidate = extract_profile_candidate(
        novel_text="some short novel text",
        llm_runtime=provider,
        character_focus="Test Character",
        profile_id="test",
        source_title="Test",
        chunk_size=100,
    )
    with pytest.raises(ValueError, match="reviewer"):
        review_profile_candidate(
            candidate, reviewer="   ", review_locator="loc"
        )


def test_review_rejects_empty_review_locator() -> None:
    provider = _FakeProvider(payloads=(_VALID_JSON_PAYLOAD,))
    candidate = extract_profile_candidate(
        novel_text="some short novel text",
        llm_runtime=provider,
        character_focus="Test Character",
        profile_id="test",
        source_title="Test",
        chunk_size=100,
    )
    with pytest.raises(ValueError, match="review_locator"):
        review_profile_candidate(
            candidate, reviewer="r", review_locator=""
        )


def test_review_rejects_candidate_with_zero_boundaries() -> None:
    provider = _FakeProvider(payloads=(_VALID_JSON_PAYLOAD,))
    candidate = extract_profile_candidate(
        novel_text="some short novel text",
        llm_runtime=provider,
        character_focus="Test Character",
        profile_id="test",
        source_title="Test",
        chunk_size=100,
    )
    # Force-empty the boundary list by reconstructing the candidate.
    stripped = ReviewedProfileCandidate(
        profile_id=candidate.profile_id,
        character_name=candidate.character_name,
        source_title=candidate.source_title,
        description=candidate.description,
        knowledge_seeds=candidate.knowledge_seeds,
        signature_cases=candidate.signature_cases,
        strategy_priors=candidate.strategy_priors,
        boundary_priors=(),
        drive_priors=candidate.drive_priors,
        requires_review=candidate.requires_review,
        aggregate_inference_confidence=candidate.aggregate_inference_confidence,
        provenance_chunks=candidate.provenance_chunks,
        notes=candidate.notes,
    )
    with pytest.raises(ValueError, match="zero boundary_priors"):
        review_profile_candidate(
            stripped, reviewer="r", review_locator="loc"
        )


def test_extract_profile_candidate_dedupes_per_chunk_records() -> None:
    """Two chunks emit the same boundary id; deduper keeps one."""
    provider = _FakeProvider(
        payloads=(_VALID_JSON_PAYLOAD, _VALID_JSON_PAYLOAD)
    )
    candidate = extract_profile_candidate(
        novel_text="text " * 600,  # ~3000 chars, 2 chunks at 1500
        llm_runtime=provider,
        character_focus="Test Character",
        profile_id="test-extracted",
        source_title="Test Source",
        chunk_size=1500,
    )
    boundary_ids = [b.boundary_id for b in candidate.boundary_priors]
    assert len(boundary_ids) == len(set(boundary_ids)), (
        f"deduper failed; duplicate boundary_ids: {boundary_ids}"
    )
