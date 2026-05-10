"""Smoke tests for the growth-advisor profile compilation path.

Validates:

* The Cheng Laoshi profile builds without errors and carries the
  four anchoring boundary priors that the LTV archetype requires.
* Compiling the profile through ``build_growth_advisor_package`` /
  ``build_growth_advisor_vitals_bootstrap`` produces non-empty
  records on every owner channel (knowledge / case / playbook /
  boundary / drives).
* The resulting :class:`DomainExperiencePackage` passes the
  ``vz-application`` validation contract (so the kernel will accept
  the bundle without a separate vertical-side validator).
* Required boundary IDs (``bp-no-hard-sell`` etc.) are present —
  these are the structural anti-pattern guards that distinguish the
  growth-advisor archetype from a generic companion vertical.
"""

from __future__ import annotations

import pytest

from volvence_zero.application import (
    DomainExperiencePackage,
    validate_domain_experience_package,
)

from lifeform_domain_growth_advisor import (
    GrowthAdvisorProfile,
    build_cheng_laoshi_profile,
    build_growth_advisor_package,
    build_growth_advisor_vitals_bootstrap,
)


_REQUIRED_BOUNDARY_IDS = (
    "bp-no-hard-sell",
    "bp-no-overclaim",
    "bp-no-flooding",
    "bp-no-judgmental",
)


def test_cheng_laoshi_profile_builds() -> None:
    profile = build_cheng_laoshi_profile()
    assert isinstance(profile, GrowthAdvisorProfile)
    assert profile.profile_id == "cheng-laoshi"
    assert profile.advisor_name == "谌老师"
    assert profile.knowledge_seeds, "profile must ship knowledge seeds"
    assert profile.signature_cases, "profile must ship signature cases"
    assert profile.strategy_priors, "profile must ship strategy priors"
    assert profile.boundary_priors, "profile must ship boundary priors"
    assert profile.drive_priors, "profile must ship drive priors"


def test_cheng_laoshi_profile_has_required_boundaries() -> None:
    profile = build_cheng_laoshi_profile()
    boundary_ids = {b.boundary_id for b in profile.boundary_priors}
    for required in _REQUIRED_BOUNDARY_IDS:
        assert required in boundary_ids, (
            f"growth-advisor profile must declare boundary {required!r} so "
            f"the LTV anti-sales / anti-overclaim invariants survive a "
            f"future profile rewrite"
        )


def test_cheng_laoshi_profile_has_seven_day_playbook() -> None:
    profile = build_cheng_laoshi_profile()
    rule_ids = {rule.rule_id for rule in profile.strategy_priors}
    for day in range(1, 8):
        expected_prefix = f"playbook-day{day}"
        matches = [rid for rid in rule_ids if rid.startswith(expected_prefix)]
        assert matches, (
            f"profile must declare a strategy prior for Day {day} "
            f"(prefix {expected_prefix!r}); got rule_ids={sorted(rule_ids)!r}"
        )


def test_cheng_laoshi_profile_has_four_mining_funnels() -> None:
    profile = build_cheng_laoshi_profile()
    rule_ids = {rule.rule_id for rule in profile.strategy_priors}
    for funnel in ("funnel-height", "funnel-immunity", "funnel-nutrition",
                   "funnel-vision-brain"):
        assert funnel in rule_ids, (
            f"profile must declare a need-mining funnel {funnel!r}"
        )


def test_compile_to_domain_experience_package_validates() -> None:
    profile = build_cheng_laoshi_profile()
    package = build_growth_advisor_package(profile)
    assert isinstance(package, DomainExperiencePackage)
    assert package.knowledge_records, "package must carry knowledge records"
    assert package.case_records, "package must carry case records"
    assert package.playbook_rules, "package must carry playbook rules"
    assert package.boundary_hints, "package must carry boundary hints"
    report = validate_domain_experience_package(package)
    assert report.valid, (
        f"compiled domain experience package must validate; got "
        f"errors={[issue.description for issue in report.issues]!r}"
    )


def test_vitals_bootstrap_carries_four_drives() -> None:
    profile = build_cheng_laoshi_profile()
    bootstrap = build_growth_advisor_vitals_bootstrap(profile)
    drive_names = {drive.name for drive in bootstrap.drives}
    for required in (
        "trust_building_drive",
        "empathy_response_drive",
        "restraint_against_pitch_drive",
        "kb_share_drive",
    ):
        assert required in drive_names, (
            f"vitals bootstrap must declare drive {required!r}; got "
            f"{sorted(drive_names)!r}"
        )


def test_no_hard_sell_boundary_blocks_brand_recommendation_topic() -> None:
    profile = build_cheng_laoshi_profile()
    no_sell = next(
        b for b in profile.boundary_priors if b.boundary_id == "bp-no-hard-sell"
    )
    blocked = " ".join(no_sell.blocked_topics)
    assert "brand_recommendation" in blocked, (
        "bp-no-hard-sell must block specific-brand recommendation before "
        "the trust gate; without this the LTV archetype collapses into a "
        "regular sales bot."
    )


def test_profile_post_init_rejects_empty_boundaries() -> None:
    """Calling the profile constructor without boundaries fails loudly.

    The growth-advisor archetype refuses to exist without explicit
    anti-sales boundaries; the schema enforces that.
    """
    profile = build_cheng_laoshi_profile()
    with pytest.raises(ValueError, match="boundary_priors"):
        GrowthAdvisorProfile(
            profile_id=profile.profile_id,
            advisor_name=profile.advisor_name,
            source_title=profile.source_title,
            version=profile.version,
            reviewed_by=profile.reviewed_by,
            source_uri=profile.source_uri,
            description=profile.description,
            knowledge_seeds=profile.knowledge_seeds,
            signature_cases=profile.signature_cases,
            strategy_priors=profile.strategy_priors,
            boundary_priors=(),
            drive_priors=profile.drive_priors,
        )
