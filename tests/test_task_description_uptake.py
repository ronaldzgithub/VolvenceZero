"""Packet 7.1: TaskDescriptionUptake adapter tests."""

from __future__ import annotations

import pytest

from lifeform_protocol_runtime import (
    MockLlmJsonClient,
    extract_protocol_from_description,
)
from volvence_zero.behavior_protocol import ProtocolSourceKind


def _mock_client():
    return MockLlmJsonClient(
        identity={
            "advisor_name": "test-advisor",
            "description": "test advisor for task description uptake",
            "self_traits": ["patient_listener"],
            "forbidden_traits": [],
            "regimes": ["companion_track"],
        },
        boundary={
            "boundaries": [
                {
                    "boundary_id": "bd:test:no-promo",
                    "description": "no promotional content",
                    "trigger_reasons": ["promotional language detected"],
                    "blocked_topics": ["sale", "promo"],
                    "severity": "soft_remind",
                }
            ]
        },
        strategy={
            "strategies": [
                {
                    "rule_id": "rule:test:listen",
                    "problem_pattern": "user expresses confusion",
                    "recommended_ordering": ["listen", "clarify"],
                    "recommended_pacing": "slow",
                }
            ],
        },
    )


def test_extract_protocol_from_description_returns_candidate() -> None:
    desc = (
        "Tonally cheerful sales assistant for booking spas. "
        "Always polite and professional. Block promotional content."
    )
    candidate = extract_protocol_from_description(
        desc,
        llm_client=_mock_client(),
        protocol_id="test:spa-assistant",
        advisor_name="spa-assistant",
    )
    assert candidate.protocol.protocol_id  # non-empty
    assert candidate.requires_review is True
    assert (
        candidate.provenance.source_kind is ProtocolSourceKind.TASK_DESCRIPTION
    )
    assert "task_description://" in candidate.provenance.source_locator


def test_extract_protocol_from_description_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty description"):
        extract_protocol_from_description(
            "",
            llm_client=_mock_client(),
            protocol_id="test:empty",
            advisor_name="x",
        )
    with pytest.raises(ValueError, match="empty description"):
        extract_protocol_from_description(
            "   ",
            llm_client=_mock_client(),
            protocol_id="test:whitespace",
            advisor_name="x",
        )


def test_extract_protocol_provenance_carries_extractor_id() -> None:
    candidate = extract_protocol_from_description(
        "test description",
        llm_client=_mock_client(),
        protocol_id="test:p1",
        advisor_name="a1",
        extractor_id="custom_extractor_v2",
    )
    assert candidate.provenance.extractor_id == "custom_extractor_v2"
