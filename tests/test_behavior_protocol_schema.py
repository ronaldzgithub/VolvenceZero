"""Schema invariants for ``volvence_zero.behavior_protocol`` (packet 1.0).

These tests pin the contract shape and invariants for
``BehaviorProtocol`` and ``ActiveMixtureSnapshot``. They guard
against accidental schema drift before any consumer reads the
``active_mixture`` slot (consumer integration lands packet 1.2+).

What is asserted:

* The closed ``BehaviorProtocolSignalSource`` enum has the six
  packet-1.0 members and only those (extension requires synchronous
  PR additions per protocol-runtime spec §协议 → PE 映射).
* ``BehaviorProtocol`` rejects empty ``success_signals`` /
  ``failure_signals`` unless ``legacy_fixture=True``.
* ``BehaviorProtocol`` rejects duplicate ids inside its sub-tuples.
* ``BoundaryContract`` / ``StrategyPrior`` / ``SuccessSignal`` /
  ``FailureSignal`` / ``ActivationConditions`` /
  ``ActiveProtocolEntry`` enforce their per-field invariants.
* ``ActiveMixtureSnapshot`` is constructible empty and rejects
  duplicates inside ``active_protocols`` / ``boundary_union``.
"""

from __future__ import annotations

import pytest

from volvence_zero.behavior_protocol import (
    ActivationConditions,
    ActiveMixtureSnapshot,
    ActiveProtocolEntry,
    BehaviorProtocol,
    BehaviorProtocolSignalSource,
    BoundaryContract,
    BoundarySeverity,
    FailureSignal,
    IdentityAssertion,
    ProtocolSourceKind,
    ReviewLevel,
    ReviewStatus,
    StrategyPrior,
    SuccessSignal,
    TemporalArc,
    TemporalPhase,
)


# Packet 1.0 vocabulary (preserved as a snapshot of what shipped first).
_PACKET_1_0_SIGNAL_SOURCES = frozenset(
    {
        "DRIVE_HOMEOSTASIS_HOLD",
        "DRIVE_HOMEOSTASIS_BREACH",
        "BOUNDARY_VIOLATION_FIRED",
        "RUPTURE_KIND_FIRED",
        "INTERLOCUTOR_ZONE_TRANSITION",
        "USER_DROPOUT_OBSERVED",
    }
)

# Packet 1.5a' extensions: REGIME_TRANSITION_RECENT and
# RETRIEVAL_HITS_PRESENT light up the regime / retrieval direct
# context_match signal sources, fully closing ACTIVE checklist
# condition 3. Adding more members to this expected set requires
# adding a typed source AND a consumer test in the same PR (the
# closed-enum invariant — see ``BehaviorProtocolSignalSource``
# docstring).
_EXPECTED_SIGNAL_SOURCES = _PACKET_1_0_SIGNAL_SOURCES | {
    "REGIME_TRANSITION_RECENT",
    "RETRIEVAL_HITS_PRESENT",
    # Packet 7.0 — dialogue trace + commitment detectors.
    "USER_REPLY_LATENCY",
    "USER_REPLY_LENGTH",
    "USER_INITIATIVE_QUESTION",
    "COMMITMENT_FULFILLED",
    "COMMITMENT_BROKEN",
}


def _minimal_identity() -> IdentityAssertion:
    return IdentityAssertion()


def _minimal_strategy(rule_id: str = "rule-a") -> StrategyPrior:
    return StrategyPrior(
        rule_id=rule_id,
        problem_pattern="any",
        recommended_ordering=("step1",),
        recommended_pacing="paced",
    )


def _minimal_boundary(boundary_id: str = "bp-x") -> BoundaryContract:
    return BoundaryContract(
        boundary_id=boundary_id,
        description="x",
        trigger_reasons=("trigger",),
    )


def _minimal_success(signal_id: str = "ok") -> SuccessSignal:
    return SuccessSignal(
        signal_id=signal_id,
        description="ok",
        measurable_via=BehaviorProtocolSignalSource.DRIVE_HOMEOSTASIS_HOLD,
    )


def _minimal_failure(signal_id: str = "fail") -> FailureSignal:
    return FailureSignal(
        signal_id=signal_id,
        description="fail",
        measurable_via=BehaviorProtocolSignalSource.DRIVE_HOMEOSTASIS_BREACH,
    )


def _minimal_protocol(
    *,
    success_signals: tuple[SuccessSignal, ...] = (),
    failure_signals: tuple[FailureSignal, ...] = (),
    legacy_fixture: bool = False,
    protocol_id: str = "pkt:test:v1",
    boundary_contracts: tuple[BoundaryContract, ...] | None = None,
    strategy_priors: tuple[StrategyPrior, ...] | None = None,
) -> BehaviorProtocol:
    return BehaviorProtocol(
        protocol_id=protocol_id,
        version="1.0",
        advisor_name="test",
        description="test protocol",
        source_kind=ProtocolSourceKind.FIXTURE,
        source_locator="test://locator",
        identity_assertion=_minimal_identity(),
        boundary_contracts=boundary_contracts
        if boundary_contracts is not None
        else (_minimal_boundary(),),
        activation_conditions=ActivationConditions(),
        strategy_priors=strategy_priors
        if strategy_priors is not None
        else (_minimal_strategy(),),
        temporal_arc=TemporalArc(),
        success_signals=success_signals,
        failure_signals=failure_signals,
        legacy_fixture=legacy_fixture,
    )


# ---------------------------------------------------------------------------
# Signal source enum is closed at packet-1.0 membership
# ---------------------------------------------------------------------------


def test_signal_source_enum_contains_expected_members() -> None:
    actual = {member.name for member in BehaviorProtocolSignalSource}
    assert actual == _EXPECTED_SIGNAL_SOURCES, (
        "BehaviorProtocolSignalSource diverged from the expected "
        "closed vocabulary; new members must add a typed signal "
        "source AND update this gate (see closed-enum invariant in "
        f"the enum docstring). Got: {sorted(actual)}"
    )


def test_packet_1_0_signal_sources_remain_present() -> None:
    """Packet 1.5a' extends the vocabulary; it must NOT remove any
    packet-1.0 member. This test pins backward compatibility for
    the closed enum (any pre-existing protocol declaration with
    ``DRIVE_HOMEOSTASIS_*`` etc. must still parse / validate)."""

    actual = {member.name for member in BehaviorProtocolSignalSource}
    missing = _PACKET_1_0_SIGNAL_SOURCES - actual
    assert missing == set(), (
        f"packet-1.0 signal sources removed: {sorted(missing)}; "
        "removing a closed-enum member is a contract break."
    )


def test_signal_source_values_are_string_lower_snake_case() -> None:
    for member in BehaviorProtocolSignalSource:
        assert member.value == member.name.lower(), (
            f"{member.name} value {member.value!r} must equal "
            "lower-cased name (matches Enum convention used by other "
            "vz-contracts enums)"
        )


# ---------------------------------------------------------------------------
# BehaviorProtocol: PE-signal invariant
# ---------------------------------------------------------------------------


def test_behavior_protocol_requires_success_signal_unless_legacy() -> None:
    with pytest.raises(ValueError, match="success_signals"):
        _minimal_protocol(
            success_signals=(),
            failure_signals=(_minimal_failure(),),
            legacy_fixture=False,
        )


def test_behavior_protocol_requires_failure_signal_unless_legacy() -> None:
    with pytest.raises(ValueError, match="failure_signals"):
        _minimal_protocol(
            success_signals=(_minimal_success(),),
            failure_signals=(),
            legacy_fixture=False,
        )


def test_behavior_protocol_legacy_fixture_skips_pe_signal_invariant() -> None:
    protocol = _minimal_protocol(
        success_signals=(),
        failure_signals=(),
        legacy_fixture=True,
    )
    assert protocol.legacy_fixture is True
    assert protocol.success_signals == ()
    assert protocol.failure_signals == ()


def test_behavior_protocol_minimum_full_signals_construct_cleanly() -> None:
    protocol = _minimal_protocol(
        success_signals=(_minimal_success(),),
        failure_signals=(_minimal_failure(),),
    )
    assert protocol.review_status is ReviewStatus.DRAFT
    assert protocol.legacy_fixture is False


# ---------------------------------------------------------------------------
# Uniqueness invariants on sub-tuples
# ---------------------------------------------------------------------------


def test_behavior_protocol_rejects_duplicate_boundary_ids() -> None:
    with pytest.raises(ValueError, match="boundary_contracts.boundary_id"):
        _minimal_protocol(
            boundary_contracts=(
                _minimal_boundary("dup"),
                _minimal_boundary("dup"),
            ),
            success_signals=(_minimal_success(),),
            failure_signals=(_minimal_failure(),),
        )


def test_behavior_protocol_rejects_duplicate_strategy_rule_ids() -> None:
    with pytest.raises(ValueError, match="strategy_priors.rule_id"):
        _minimal_protocol(
            strategy_priors=(_minimal_strategy("dup"), _minimal_strategy("dup")),
            success_signals=(_minimal_success(),),
            failure_signals=(_minimal_failure(),),
        )


def test_behavior_protocol_rejects_duplicate_success_signal_ids() -> None:
    with pytest.raises(ValueError, match="success_signals.signal_id"):
        _minimal_protocol(
            success_signals=(_minimal_success("dup"), _minimal_success("dup")),
            failure_signals=(_minimal_failure(),),
        )


# ---------------------------------------------------------------------------
# Per-field invariants
# ---------------------------------------------------------------------------


def test_strategy_prior_rejects_invalid_initial_weight() -> None:
    with pytest.raises(ValueError, match="initial_weight"):
        StrategyPrior(
            rule_id="r",
            problem_pattern="x",
            recommended_ordering=("a",),
            recommended_pacing="p",
            initial_weight=1.5,
        )


def test_strategy_prior_rejects_floor_above_initial_weight() -> None:
    with pytest.raises(ValueError, match="minimum_weight_floor"):
        StrategyPrior(
            rule_id="r",
            problem_pattern="x",
            recommended_ordering=("a",),
            recommended_pacing="p",
            initial_weight=0.5,
            minimum_weight_floor=0.6,
        )


def test_activation_conditions_rejects_compatible_and_incompatible_overlap() -> None:
    with pytest.raises(ValueError, match="compatible AND incompatible"):
        ActivationConditions(
            co_activation_compatible=("p1",),
            co_activation_incompatible=("p1",),
        )


def test_active_protocol_entry_rejects_weight_above_one() -> None:
    with pytest.raises(ValueError, match="activation_weight"):
        ActiveProtocolEntry(
            protocol_id="p1",
            activation_weight=1.2,
        )


def test_boundary_contract_rejects_empty_boundary_id() -> None:
    with pytest.raises(ValueError, match="boundary_id"):
        BoundaryContract(
            boundary_id="",
            description="x",
            trigger_reasons=(),
            severity=BoundarySeverity.HARD_BLOCK,
            review_level=ReviewLevel.L3,
        )


# ---------------------------------------------------------------------------
# ActiveMixtureSnapshot: empty + duplicate guards
# ---------------------------------------------------------------------------


def test_active_mixture_snapshot_constructs_empty() -> None:
    snapshot = ActiveMixtureSnapshot(
        active_protocols=(),
        boundary_union_ids=(),
    )
    assert snapshot.active_protocols == ()
    assert snapshot.boundary_union_ids == ()
    assert snapshot.identity_gate_traits == ()


def test_active_mixture_snapshot_rejects_duplicate_protocol_ids() -> None:
    entry = ActiveProtocolEntry(protocol_id="p1", activation_weight=0.5)
    with pytest.raises(ValueError, match="active_protocols.protocol_id"):
        ActiveMixtureSnapshot(
            active_protocols=(entry, entry),
            boundary_union_ids=(),
        )


def test_active_mixture_snapshot_rejects_duplicate_boundary_ids() -> None:
    """Packet 1.2 (Choice A): ``boundary_union_ids`` is a tuple of
    boundary IDs (str); duplicates rejected by ``__post_init__``.
    Canonical content lives in ``ApplicationRareHeavyState``,
    populated by the protocol compile path; this snapshot
    publishes only references.
    """
    with pytest.raises(ValueError, match="boundary_union_ids"):
        ActiveMixtureSnapshot(
            active_protocols=(),
            boundary_union_ids=("bp-dup", "bp-dup"),
        )


# ---------------------------------------------------------------------------
# TemporalArc + TemporalPhase
# ---------------------------------------------------------------------------


def test_temporal_arc_rejects_duplicate_phase_ids() -> None:
    with pytest.raises(ValueError, match="phase_id duplicate"):
        TemporalArc(
            phases=(
                TemporalPhase(phase_id="p1"),
                TemporalPhase(phase_id="p1"),
            ),
        )


def test_temporal_arc_constructs_empty() -> None:
    arc = TemporalArc()
    assert arc.phases == ()
