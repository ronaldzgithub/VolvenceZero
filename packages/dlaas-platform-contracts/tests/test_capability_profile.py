"""Tests for the certified capability profile contract + composition."""

from __future__ import annotations

from dlaas_platform_contracts import (
    CapabilityAxisId,
    CapabilityProfileSpec,
    CapabilityReadoutInputs,
    ClaimedResume,
    ExperienceSummary,
    ProfileProvenanceTier,
    SkillScore,
    build_profile_from_composed,
    compose_capability_indices,
    display_index,
    grade_for,
)


def test_display_index_band() -> None:
    assert display_index(0.0) == 60
    assert display_index(1.0) == 150
    assert display_index(0.5) == 105
    # Clamps out-of-range.
    assert display_index(-1.0) == 60
    assert display_index(2.0) == 150


def test_grade_thresholds() -> None:
    assert grade_for(0.90) == "A"
    assert grade_for(0.72) == "B"
    assert grade_for(0.60) == "C"
    assert grade_for(0.45) == "D"
    assert grade_for(0.10) == "F"


def test_compose_is_deterministic_and_ordered() -> None:
    inputs = CapabilityReadoutInputs(
        exam_aggregate=0.8,
        license_granted=True,
        f1_task=0.7,
        f5_abstraction=0.6,
        f2_interaction=0.75,
        f3_relationship=0.8,
        interlocutor_trust=0.7,
        interlocutor_rapport=0.6,
        kindness_ratio=0.9,
        eval_pass_rate=0.85,
        regime_stability=0.7,
        judge_safety=0.9,
        f6_safety=0.8,
        closed_scenes=50,
        usage_turns=2000,
        tenure_days=120,
        data_completeness=0.9,
    )
    a = compose_capability_indices(inputs)
    b = compose_capability_indices(inputs)
    assert a == b
    # Six axes, in canonical order.
    assert [ax.axis for ax in a.axes] == list(CapabilityAxisId)
    assert 60 <= a.iq_index <= 150
    assert 60 <= a.eq_index <= 150
    assert a.overall_grade in {"A", "B", "C", "D", "F"}


def test_axis_provenance_split() -> None:
    composed = compose_capability_indices(CapabilityReadoutInputs())
    prov = {ax.axis: ax.provenance for ax in composed.axes}
    assert prov[CapabilityAxisId.REASONING_SKILL] == ProfileProvenanceTier.CERTIFIED
    assert prov[CapabilityAxisId.SAFETY] == ProfileProvenanceTier.CERTIFIED
    assert prov[CapabilityAxisId.RELATIONSHIP_EQ] == ProfileProvenanceTier.OBSERVED
    assert prov[CapabilityAxisId.EXPERIENCE] == ProfileProvenanceTier.OBSERVED


def test_neutral_defaults_land_midband() -> None:
    composed = compose_capability_indices(CapabilityReadoutInputs())
    # All-neutral (0.5) signals with no experience -> mid/low band, not extreme.
    for ax in composed.axes:
        assert 0.0 <= ax.value_0_100 <= 100.0
    assert composed.iq_index < display_index(1.0)


def test_license_lifts_safety() -> None:
    base = CapabilityReadoutInputs(f6_safety=0.6, judge_safety=0.6)
    no_license = compose_capability_indices(base)
    with_license = compose_capability_indices(
        CapabilityReadoutInputs(f6_safety=0.6, judge_safety=0.6, license_granted=True)
    )
    safety_no = next(
        a.value_0_100 for a in no_license.axes if a.axis == CapabilityAxisId.SAFETY
    )
    safety_yes = next(
        a.value_0_100 for a in with_license.axes if a.axis == CapabilityAxisId.SAFETY
    )
    assert safety_yes > safety_no


def test_compose_is_pure_readout_no_mutation() -> None:
    """R12 / OA-1: composition is a pure readout transform — it must not
    mutate its inputs (so it can never become a covert write-back / reward
    path) and must be a stable display derivation."""
    inputs = CapabilityReadoutInputs(exam_aggregate=0.6, f1_task=0.7)
    snapshot = dict(inputs.__dict__)
    a = compose_capability_indices(inputs)
    b = compose_capability_indices(inputs)
    # Inputs untouched.
    assert inputs.__dict__ == snapshot
    # Stable display derivation.
    assert a == b
    # Every axis value stays a bounded 0..100 readout, never an unbounded
    # scalar that could be fed back as a gradient.
    for ax in a.axes:
        assert 0.0 <= ax.value_0_100 <= 100.0


def test_profile_round_trip_and_stale() -> None:
    composed = compose_capability_indices(
        CapabilityReadoutInputs(exam_aggregate=0.7, data_completeness=0.8),
        evidence_refs={"reasoning_skill": ["exam_run_1"]},
    )
    profile = build_profile_from_composed(
        profile_ref="cp_1",
        listing_ref="pl_1",
        ai_id="ai-1",
        vertical="sales",
        archetype="closer",
        composed=composed,
        skills=[
            SkillScore(name="objection_handling", score_0_100=82.0, source_exam_run_id="exam_run_1")
        ],
        experience=ExperienceSummary(closed_scenes=10, usage_turns=500),
        claimed=ClaimedResume(role_title="Senior Sales AI", domains=("b2b",)),
        certified_at_ms=1234,
        exam_run_id="exam_run_1",
        readout_snapshot_hash="rh_abc",
        license_granted=True,
        content_hash="hash_v1",
    )
    again = CapabilityProfileSpec.from_json(profile.to_json())
    assert again == profile
    assert again.claimed.role_title == "Senior Sales AI"
    assert again.skills[0].name == "objection_handling"
    # Stale detection keys off the content hash.
    assert profile.is_stale("hash_v1") is False
    assert profile.is_stale("hash_v2") is True
