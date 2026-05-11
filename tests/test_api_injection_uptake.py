"""Packet 7.3: APIInjectionUptake adapter tests."""

from __future__ import annotations

import pytest

from lifeform_protocol_runtime import inject_protocol_from_payload
from volvence_zero.behavior_protocol import (
    BoundarySeverity,
    ProtocolSourceKind,
    ReviewStatus,
)


def _minimal_payload(**overrides) -> dict:
    base = {
        "protocol_id": "api:test-bot",
        "advisor_name": "test-bot",
        "description": "API-injected test protocol",
    }
    base.update(overrides)
    return base


def test_inject_minimal_payload_returns_candidate() -> None:
    candidate = inject_protocol_from_payload(
        _minimal_payload(),
        request_id="req-1",
    )
    assert candidate.protocol.protocol_id == "api:test-bot"
    assert candidate.protocol.review_status is ReviewStatus.DRAFT
    assert (
        candidate.provenance.source_kind is ProtocolSourceKind.API_INJECTION
    )
    assert candidate.requires_review is True


def test_inject_missing_required_field_raises() -> None:
    with pytest.raises(ValueError, match="must include"):
        inject_protocol_from_payload({"protocol_id": "x"}, request_id="r")


def test_inject_with_full_payload_maps_all_fields() -> None:
    payload = _minimal_payload(
        identity={
            "requires_self_traits": ["warm", "patient"],
            "forbidden_self_traits": ["aggressive"],
        },
        boundaries=[
            {
                "boundary_id": "bd:api:no-medical",
                "description": "no medical advice",
                "trigger_reasons": ["medical question"],
                "blocked_topics": ["diagnosis"],
                "severity": "hard_block",
            }
        ],
        strategies=[
            {
                "rule_id": "rule:api:greet",
                "problem_pattern": "first contact",
                "recommended_ordering": ["greet", "introduce"],
                "recommended_pacing": "moderate",
            }
        ],
        success_signals=[
            {
                "signal_id": "sig:engagement-up",
                "measurable_via": "interlocutor_zone_transition",
                "description": "user engages",
            }
        ],
    )
    candidate = inject_protocol_from_payload(payload, request_id="req-2")
    p = candidate.protocol
    assert "warm" in p.identity_assertion.requires_self_traits
    assert any(b.severity is BoundarySeverity.HARD_BLOCK for b in p.boundary_contracts)
    assert any(s.rule_id == "rule:api:greet" for s in p.strategy_priors)
    assert any(s.signal_id == "sig:engagement-up" for s in p.success_signals)


def test_inject_skips_invalid_subfields() -> None:
    """Boundary missing boundary_id is skipped silently (lossy import)."""
    payload = _minimal_payload(
        boundaries=[
            {"description": "no id"},
            {
                "boundary_id": "bd:api:valid",
                "description": "ok",
                "trigger_reasons": ["x"],
            },
        ]
    )
    candidate = inject_protocol_from_payload(payload, request_id="req-3")
    assert len(candidate.protocol.boundary_contracts) == 1


def test_inject_provenance_carries_request_id() -> None:
    candidate = inject_protocol_from_payload(
        _minimal_payload(), request_id="req-abc-123"
    )
    assert "req-abc-123" in candidate.provenance.source_locator
