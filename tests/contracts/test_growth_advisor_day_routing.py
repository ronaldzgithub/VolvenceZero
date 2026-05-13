"""Contract test: applicability_scope day-tag enum validation (debt #65).

Validates that ``GrowthAdvisorStrategyPrior.applicability_scope`` only
accepts known tags (``growth_advisor:day1`` … ``growth_advisor:day7+``
+ ``funnel:height`` etc.). A profile with an unknown day tag (e.g.
``growth_advisor:day99``) must fail-loud at construction.

SHADOW: only enum check is enforced; the actual routing path
(``compute_growth_advisor_day`` + scoped_memory.onboarding_at)
lands in #65 ACTIVE.

Refs:

* docs/specs/growth-advisor-day-counter.md
* docs/known-debts.md #65
"""

from __future__ import annotations

import pytest

from lifeform_domain_growth_advisor.profile import (
    GrowthAdvisorBoundaryPrior,
    GrowthAdvisorProfile,
    GrowthAdvisorStrategyPrior,
)


def _make_minimal_profile(scope: tuple[str, ...]) -> GrowthAdvisorProfile:
    return GrowthAdvisorProfile(
        profile_id="test-day-routing",
        advisor_name="Test",
        source_title="reviewed",
        version="v1",
        reviewed_by="reviewer",
        source_uri="internal://test",
        description="test",
        knowledge_seeds=(),
        signature_cases=(),
        strategy_priors=(
            GrowthAdvisorStrategyPrior(
                rule_id="rule-1",
                problem_pattern="test",
                recommended_regime=None,
                recommended_ordering=("ack",),
                recommended_pacing="slow",
                avoid_patterns=(),
                applicability_scope=scope,
                confidence=0.8,
                description="test rule",
            ),
        ),
        boundary_priors=(
            GrowthAdvisorBoundaryPrior(
                boundary_id="bp-test",
                regime_id=None,
                trigger_reasons=("test",),
                answer_depth_limit_hint="short",
                clarification_required=False,
                refer_out_required=False,
                blocked_topics=(),
                required_disclaimers=(),
                confidence=0.9,
                description="test",
            ),
        ),
    )


def test_known_day_tags_pass() -> None:
    """day1..day7 + day7+ all valid."""
    for n in range(1, 8):
        _make_minimal_profile(scope=(f"growth_advisor:day{n}",))
    _make_minimal_profile(scope=("growth_advisor:day7+",))


def test_known_funnel_tags_pass() -> None:
    for funnel in ("height", "immunity", "nutrition", "vision"):
        _make_minimal_profile(scope=(f"funnel:{funnel}",))


def test_combined_day_and_funnel_pass() -> None:
    _make_minimal_profile(scope=("growth_advisor:day3", "funnel:height"))


@pytest.mark.skip(
    reason=(
        "SHADOW: applicability_scope enum validation lands in "
        "GrowthAdvisorStrategyPrior.__post_init__ in #65 ACTIVE. "
        "Currently the field accepts any tuple[str, ...]; this test "
        "is the assertion contract for the future check."
    )
)
def test_unknown_day_tag_fails_loud() -> None:
    """ACTIVE: GrowthAdvisorStrategyPrior(applicability_scope=...) raises."""
    with pytest.raises(ValueError, match="unknown tags"):
        _make_minimal_profile(scope=("growth_advisor:day99",))


@pytest.mark.skip(reason="SHADOW: same as above")
def test_unknown_funnel_tag_fails_loud() -> None:
    with pytest.raises(ValueError, match="unknown tags"):
        _make_minimal_profile(scope=("funnel:made_up_funnel",))
