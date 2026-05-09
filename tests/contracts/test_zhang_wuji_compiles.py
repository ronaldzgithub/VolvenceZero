"""Wave C6 (Tier 1) — pin the compiled shape of the 张无忌 reference profile.

These tests are the regression guard for the shipped reference
character. They do NOT require a runtime; they only exercise the
three pure builders against the canonical profile.

Why pin specific counts:

The expected counts (knowledge_records >= 6 / case_records >= 8 /
playbook_rules >= 4 / boundary_hints >= 5 / drives == 5) are the
sub-deliverable promised in the milestone plan. If a future edit
prunes one of these tuples, the relevant evidence-chain test in
``test_character_lifeform_e2e.py`` will lose part of its assertion
surface; this test catches the change earlier and forces the
profile author to either restore the count or update the milestone
documentation explicitly.

Counts are documented as ``>= N`` rather than ``== N`` so the
profile can be enriched without breaking the test, while still
preventing accidental shrinkage.
"""

from __future__ import annotations

from lifeform_domain_character import (
    build_character_ingestion_envelope,
    build_character_package,
    build_character_vitals_bootstrap,
    build_zhang_wuji_profile,
)
from lifeform_ingestion import IngestionComplianceProfile, IngestionSourceKind
from volvence_zero.application import compile_domain_experience_package


def test_zhang_wuji_profile_constructs_and_validates() -> None:
    profile = build_zhang_wuji_profile()
    assert profile.profile_id == "zhang-wuji"
    assert profile.character_name == "张无忌"
    assert "倚天屠龙记" in profile.source_title
    # The profile is a reviewed artifact; the reviewed_by field is
    # mandatory per the schema (CharacterSoulProfile.__post_init__).
    assert profile.reviewed_by


def test_zhang_wuji_package_meets_minimum_counts() -> None:
    profile = build_zhang_wuji_profile()
    package = build_character_package(profile)

    # Spec'd minimums in the milestone plan, with ``>=`` so future
    # enrichment does not trip the test.
    assert len(package.knowledge_records) >= 6, (
        f"expected >= 6 knowledge records, got "
        f"{len(package.knowledge_records)}"
    )
    assert len(package.case_records) >= 8, (
        f"expected >= 8 case records, got {len(package.case_records)}"
    )
    assert len(package.playbook_rules) >= 4, (
        f"expected >= 4 playbook rules, got {len(package.playbook_rules)}"
    )
    assert len(package.boundary_hints) >= 5, (
        f"expected >= 5 boundary hints, got {len(package.boundary_hints)}"
    )

    # Manifest carries traceable identity so downstream telemetry
    # can attribute matches back to this package.
    assert package.manifest.package_id == "lifeform-character:zhang-wuji"
    assert package.manifest.owner == "lifeform-domain-character"


def test_zhang_wuji_package_compiles_into_application_inputs() -> None:
    profile = build_zhang_wuji_profile()
    package = build_character_package(profile)
    compiled = compile_domain_experience_package(package)

    # Compilation must validate without issues; this is the last
    # gate before the package is allowed to flow into the lifeform
    # config (R8 + character-soul-bootstrap.md).
    assert compiled.validation_report.valid


def test_zhang_wuji_vitals_bootstrap_has_signature_drives() -> None:
    profile = build_zhang_wuji_profile()
    bootstrap = build_character_vitals_bootstrap(profile)

    drive_names = {drive.name for drive in bootstrap.drives}
    expected = {
        "compassion_active",
        "decisive_under_crisis",
        "loyalty_to_kin",
        "martial_curiosity",
        "self_sacrifice_pull",
    }
    # Subset rather than equality so future drives can be added.
    missing = expected - drive_names
    assert not missing, f"missing drives in 张无忌 bootstrap: {missing}"


def test_zhang_wuji_drive_shape_diverges_from_companion_default() -> None:
    """The character vertical must NOT publish the companion drive shape.

    Companion default ships ``bond_warmth`` / ``user_engagement`` /
    ``conversation_continuity``. Zhang Wuji's drives are different by
    design (compassion_active / decisive_under_crisis / etc.), and
    if a future change accidentally folded both verticals together
    this test would catch it as a vertical-isolation regression.
    """
    profile = build_zhang_wuji_profile()
    bootstrap = build_character_vitals_bootstrap(profile)
    drive_names = {drive.name for drive in bootstrap.drives}
    companion_names = {"bond_warmth", "user_engagement", "conversation_continuity"}
    overlap = drive_names & companion_names
    assert not overlap, (
        f"Zhang Wuji drive set must not include companion drives; "
        f"overlap={overlap}"
    )


def test_zhang_wuji_decisive_under_crisis_band_is_wide() -> None:
    """Specific design property: the crisis-decisiveness drive has a
    deliberately WIDE homeostatic band, reflecting the character's
    "slow in peace, decisive in crisis" tension. A future profile
    edit that narrows this band changes the character's behavioural
    shape; we pin it so that change is intentional.
    """
    profile = build_zhang_wuji_profile()
    decisive = next(
        d for d in profile.drive_priors if d.name == "decisive_under_crisis"
    )
    band_low, band_high = decisive.homeostatic_band
    band_width = band_high - band_low
    assert band_width >= 0.5, (
        f"decisive_under_crisis band width {band_width:.2f} is too narrow; "
        f"the character's crisis-vs-peace tension requires a wide band "
        f"(plan documented >= 0.5)"
    )


def test_zhang_wuji_has_absolute_no_kill_yielding_boundary() -> None:
    """Specific design property: an ABSOLUTE boundary on harming the
    yielded / defeated. This is the most load-bearing value claim for
    the character; the boundary system must surface it as
    ``answer_depth_limit_hint='absolute'`` so downstream consumers
    can recognise it as a hard rule rather than a soft preference.
    """
    profile = build_zhang_wuji_profile()
    yield_boundaries = [
        b
        for b in profile.boundary_priors
        if any(
            "yielded" in reason or "defeated" in reason
            for reason in b.trigger_reasons
        )
    ]
    assert yield_boundaries, (
        "Zhang Wuji profile must include at least one boundary keyed "
        "to opponent-yielded / opponent-defeated trigger reasons"
    )
    absolute = [
        b for b in yield_boundaries if b.answer_depth_limit_hint == "absolute"
    ]
    assert absolute, (
        "yielded / defeated boundary must be marked absolute, not "
        "soft / strong"
    )


def test_zhang_wuji_ingestion_envelope_keyed_to_profile() -> None:
    profile = build_zhang_wuji_profile()
    excerpt = (
        "夜深，山道泥泞。\n\n他停在断桥边，听见对岸有人呼救。\n\n"
        "他没有犹豫，跨过去。"
    )
    envelope = build_character_ingestion_envelope(
        profile,
        excerpt,
        uploader="test-suite",
        upload_ts_ms=1_700_000_000_000,
        max_chunk_chars=128,
    )
    assert envelope.envelope_id == "character-ingestion:zhang-wuji"
    assert envelope.source_kind is IngestionSourceKind.BOOK
    assert envelope.compliance_profile is IngestionComplianceProfile.FORCED
    assert envelope.partial_failures == ()
    assert len(envelope.chunks) >= 2  # multi-chunk path exercised
