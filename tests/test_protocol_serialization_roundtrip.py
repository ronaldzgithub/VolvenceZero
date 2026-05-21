"""Round-trip tests for :func:`protocol_to_payload` /
:func:`protocol_from_payload`.

These guarantee that approved protocols on disk reload back into
the exact same in-memory shape — the load-bearing invariant of
the disk-backed library (debt #service-persistence). Lossy
serialisation here would mean restart-induced behaviour drift,
which is exactly the failure mode the library was built to
prevent.
"""

from __future__ import annotations

import json

import pytest

from lifeform_protocol_runtime import (
    PROTOCOL_PAYLOAD_SCHEMA_VERSION,
    ProtocolPayloadSchemaError,
    protocol_from_payload,
    protocol_to_payload,
)
from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
    BehaviorProtocolSignalSource,
    BoundaryContract,
    BoundarySeverity,
    ContextMatchSignal,
    DriveExpectation,
    FailureSignal,
    IdentityAssertion,
    KnowledgeSeed,
    ProgressionSignal,
    ProtocolRevision,
    ProtocolSourceKind,
    ReviewLevel,
    ReviewStatus,
    SignatureCase,
    StrategyPrior,
    StrategyPriorRevision,
    SuccessSignal,
    TemporalArc,
    TemporalPhase,
)


def _build_fully_populated_protocol() -> BehaviorProtocol:
    """Build a protocol that exercises every field of every nested type.

    Every optional defaults to a non-default value so a serialiser
    that silently drops a field on the to- or from- side fails
    loud here (eq-based round-trip assertion below).
    """
    return BehaviorProtocol(
        protocol_id="test:full-coverage",
        version="1.2.3",
        advisor_name="Round-Trip Advisor",
        description="Exercises every nested field for lossless I/O.",
        source_kind=ProtocolSourceKind.PDF_UPTAKE,
        source_locator="uploads/full-coverage.pdf",
        identity_assertion=IdentityAssertion(
            requires_self_traits=("warm", "patient"),
            forbidden_self_traits=("sales-pushy",),
            required_regime_compatibility=("regime:steady",),
        ),
        boundary_contracts=(
            BoundaryContract(
                boundary_id="bd:no-medical",
                description="never give medical advice",
                trigger_reasons=("medical_request",),
                blocked_topics=("diagnosis", "prescription"),
                required_disclaimers=("not a doctor",),
                refer_out_required=True,
                regime_id="regime:steady",
                answer_depth_limit_hint="short",
                clarification_required=True,
                severity=BoundarySeverity.HARD_BLOCK,
                review_level=ReviewLevel.L4,
                confidence=0.95,
            ),
            BoundaryContract(
                boundary_id="bd:no-overclaim",
                description="no overclaim",
                trigger_reasons=("certainty_claim",),
                severity=BoundarySeverity.SOFT_REMIND,
                review_level=ReviewLevel.L2,
                confidence=0.7,
            ),
        ),
        activation_conditions=ActivationConditions(
            context_match_signals=(
                ContextMatchSignal(
                    signal_id="ctx:retrieval-present",
                    measurable_via=BehaviorProtocolSignalSource.RETRIEVAL_HITS_PRESENT,
                    weight=0.8,
                    description="weight up when retrieval has hits",
                ),
            ),
            co_activation_compatible=("test:other-bot",),
            co_activation_incompatible=("test:rival-bot",),
            minimum_weight_floor=0.15,
        ),
        strategy_priors=(
            StrategyPrior(
                rule_id="rule:greet",
                problem_pattern="first contact",
                recommended_ordering=("greet", "ask-name"),
                recommended_pacing="slow",
                avoid_patterns=("avoid:sales-pitch",),
                applicability_phase=("phase:icebreaker",),
                recommended_regime="regime:steady",
                knowledge_weight_hint=0.5,
                experience_weight_hint=0.7,
                initial_weight=0.85,
                pe_decay_rate=0.01,
                pe_reinforce_rate=0.02,
                minimum_weight_floor=0.05,
                revision_history=(
                    StrategyPriorRevision(
                        revision_id="rev:rule:greet:1",
                        revised_at_tick=3,
                        delta=-0.05,
                        reason="PE-driven decay test",
                    ),
                ),
                confidence=0.8,
                description="Greet pattern",
            ),
        ),
        temporal_arc=TemporalArc(
            phases=(
                TemporalPhase(
                    phase_id="phase:icebreaker",
                    description="first turn",
                    entry_conditions=(
                        ProgressionSignal(
                            signal_id="ps:enter-ice",
                            measurable_via=BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION,
                            threshold=0.3,
                            description="entered icebreaker",
                        ),
                    ),
                    exit_conditions=(
                        ProgressionSignal(
                            signal_id="ps:exit-ice",
                            measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
                            threshold=0.1,
                            description="trust seeded",
                        ),
                    ),
                    expected_drives_state=(
                        DriveExpectation(
                            drive_name="curiosity",
                            expected_band=(0.4, 0.8),
                        ),
                    ),
                ),
            ),
            progression_signals=(
                ProgressionSignal(
                    signal_id="ps:arc-level",
                    measurable_via=BehaviorProtocolSignalSource.REGIME_TRANSITION_RECENT,
                    threshold=0.5,
                    description="arc-level transition",
                ),
            ),
        ),
        success_signals=(
            SuccessSignal(
                signal_id="ss:engaged",
                description="user keeps engaging",
                measurable_via=BehaviorProtocolSignalSource.INTERLOCUTOR_ZONE_TRANSITION,
                expected_value_range=(0.2, 0.9),
                weight_in_pe=1.2,
            ),
        ),
        failure_signals=(
            FailureSignal(
                signal_id="fs:dropout",
                description="user drops out",
                measurable_via=BehaviorProtocolSignalSource.USER_DROPOUT_OBSERVED,
                threshold=0.4,
                weight_in_pe=0.8,
            ),
        ),
        knowledge_seeds=(
            KnowledgeSeed(
                seed_id="k:fact-1",
                domain="parenting",
                title="Sleep hygiene basics",
                summary="evidence summary",
                snippet="snippet text",
                evidence_locator="docs/parenting/sleep.md#hygiene",
                confidence=0.8,
                evidence_strength="strong",
                topic_tags=("sleep", "infant"),
                source_type="reviewed-guide",
                freshness_label="current",
                jurisdiction_tags=("CN", "EU"),
                conflict_markers=("conflicts-with: old-guide",),
            ),
        ),
        signature_cases=(
            SignatureCase(
                case_id="case:bedtime",
                domain="parenting",
                problem_pattern="bedtime resistance",
                user_state_pattern="exhausted-evening",
                risk_markers=("sleep-deprivation",),
                track_tags=("track:sleep",),
                regime_tags=("regime:steady",),
                intervention_ordering=("validate", "anchor", "experiment"),
                outcome_label="resolved-after-3-days",
                confidence=0.75,
                description="case description",
                relevance_score=0.7,
                escalation_observed=True,
                repair_observed=True,
                delayed_signal_count=2,
                reconstruction_source="signature-case-curator",
            ),
        ),
        parent_protocol_id="test:parent",
        review_status=ReviewStatus.ACTIVE,
        revision_log=(
            ProtocolRevision(
                revision_id="rv:1",
                revised_at_tick=10,
                revised_by="human-reviewer",
                description="initial review pass",
                affected_field="strategy_priors",
            ),
        ),
        legacy_fixture=False,
    )


def test_round_trip_preserves_every_field() -> None:
    original = _build_fully_populated_protocol()
    payload = protocol_to_payload(original)
    restored = protocol_from_payload(payload)
    assert restored == original


def test_round_trip_through_json_string() -> None:
    """Persistence path: dump→str→load→reconstruct must equal original."""
    original = _build_fully_populated_protocol()
    text = json.dumps(protocol_to_payload(original), ensure_ascii=False)
    restored = protocol_from_payload(json.loads(text))
    assert restored == original


def test_payload_carries_schema_version() -> None:
    original = _build_fully_populated_protocol()
    payload = protocol_to_payload(original)
    assert payload["schema_version"] == PROTOCOL_PAYLOAD_SCHEMA_VERSION


def test_from_payload_rejects_missing_schema_version() -> None:
    original = _build_fully_populated_protocol()
    payload = protocol_to_payload(original)
    del payload["schema_version"]
    with pytest.raises(ProtocolPayloadSchemaError):
        protocol_from_payload(payload)


def test_from_payload_rejects_wrong_schema_version() -> None:
    original = _build_fully_populated_protocol()
    payload = protocol_to_payload(original)
    payload["schema_version"] = "9.9.future"
    with pytest.raises(ProtocolPayloadSchemaError):
        protocol_from_payload(payload)


def test_round_trip_growth_advisor_reference_protocol() -> None:
    """Production data: the Cheng Laoshi protocol must round-trip too.

    Catches regressions where the synthetic test protocol above
    omits a field that real fixture protocols depend on.
    """
    pytest.importorskip("lifeform_domain_growth_advisor")
    from lifeform_domain_growth_advisor import (
        build_cheng_laoshi_profile,
        growth_advisor_profile_to_behavior_protocol,
    )

    profile = build_cheng_laoshi_profile()
    original = growth_advisor_profile_to_behavior_protocol(profile)
    restored = protocol_from_payload(protocol_to_payload(original))
    assert restored == original
