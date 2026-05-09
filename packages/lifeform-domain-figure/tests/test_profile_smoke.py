"""Smoke tests for the F1.1 profile schema.

Validates:

* The Einstein profile builds without errors and is fully validated.
* ``HistoricalFigureProfile.select_window`` returns the correct merge
  for both base and override paths.
* Schema validators reject known invalid inputs (empty boundary
  priors, empty domain coverage seed, inverted lifespan).
"""

from __future__ import annotations

import pytest

from lifeform_domain_figure import (
    FigureBoundaryPrior,
    FigureDrivePrior,
    FigureKnowledgeSeed,
    FigureSignatureCase,
    FigureStrategyPrior,
    HistoricalFigureProfile,
    TimeWindowedView,
    build_einstein_profile,
)


def test_einstein_profile_builds() -> None:
    profile = build_einstein_profile()
    assert profile.profile_id == "einstein"
    assert profile.figure_lifespan == (1879, 1955)
    assert profile.knowledge_seeds, "Einstein profile must have knowledge seeds"
    assert profile.signature_cases, "Einstein profile must have signature cases"
    assert profile.boundary_priors, "Einstein profile must have boundary priors"
    assert profile.drive_priors, "Einstein profile must have drive priors"
    assert "physics_foundations" in profile.domain_coverage_seed
    assert {window.window_id for window in profile.time_windows} == {
        "early-1905-1925",
        "late-1925-1955",
    }


def test_select_window_returns_self_when_unspecified() -> None:
    profile = build_einstein_profile()
    assert profile.select_window(None) is profile
    assert profile.select_window("") is profile


def test_select_window_unknown_raises() -> None:
    profile = build_einstein_profile()
    with pytest.raises(ValueError, match="window_id"):
        profile.select_window("does-not-exist")


def test_select_window_overrides_merge_by_id() -> None:
    base_seed = FigureKnowledgeSeed(
        seed_id="base-1",
        domain="d",
        title="base",
        summary="base summary",
        snippet="base",
        evidence_locator="loc",
        confidence=0.5,
        evidence_strength="medium",
    )
    override_seed = FigureKnowledgeSeed(
        seed_id="base-1",
        domain="d",
        title="override",
        summary="override summary",
        snippet="override",
        evidence_locator="loc",
        confidence=0.9,
        evidence_strength="high",
    )
    new_seed = FigureKnowledgeSeed(
        seed_id="window-only",
        domain="d",
        title="window only",
        summary="window-only summary",
        snippet="new",
        evidence_locator="loc",
        confidence=0.7,
        evidence_strength="medium",
    )
    boundary = FigureBoundaryPrior(
        boundary_id="b-1",
        regime_id=None,
        trigger_reasons=("trigger",),
        answer_depth_limit_hint="strong",
        clarification_required=False,
        refer_out_required=False,
        blocked_topics=("blocked",),
        required_disclaimers=(),
        confidence=0.8,
        description="boundary",
    )
    profile = HistoricalFigureProfile(
        profile_id="x",
        figure_name="X",
        figure_lifespan=(1900, 1980),
        version="0.1.0",
        reviewed_by="test",
        source_uri="profile:x",
        description="test",
        domain_coverage_seed=("d",),
        knowledge_seeds=(base_seed,),
        signature_cases=(),
        strategy_priors=(),
        boundary_priors=(boundary,),
        time_windows=(
            TimeWindowedView(
                window_id="late",
                year_start=1950,
                year_end=1980,
                description="late",
                knowledge_seed_overrides=(override_seed, new_seed),
            ),
        ),
    )
    windowed = profile.select_window("late")
    assert windowed.knowledge_seeds[0].title == "override"
    assert windowed.knowledge_seeds[1].seed_id == "window-only"
    assert "window:late" in windowed.version


def test_profile_rejects_empty_boundary_priors() -> None:
    with pytest.raises(ValueError, match="boundary_priors"):
        HistoricalFigureProfile(
            profile_id="x",
            figure_name="X",
            figure_lifespan=(1900, 1980),
            version="0.1.0",
            reviewed_by="test",
            source_uri="profile:x",
            description="test",
            domain_coverage_seed=("d",),
            knowledge_seeds=(),
            signature_cases=(),
            strategy_priors=(),
            boundary_priors=(),
        )


def test_profile_rejects_empty_domain_coverage_seed() -> None:
    boundary = FigureBoundaryPrior(
        boundary_id="b-1",
        regime_id=None,
        trigger_reasons=("trigger",),
        answer_depth_limit_hint="strong",
        clarification_required=False,
        refer_out_required=False,
        blocked_topics=("blocked",),
        required_disclaimers=(),
        confidence=0.8,
        description="boundary",
    )
    with pytest.raises(ValueError, match="domain_coverage_seed"):
        HistoricalFigureProfile(
            profile_id="x",
            figure_name="X",
            figure_lifespan=(1900, 1980),
            version="0.1.0",
            reviewed_by="test",
            source_uri="profile:x",
            description="test",
            domain_coverage_seed=(),
            knowledge_seeds=(),
            signature_cases=(),
            strategy_priors=(),
            boundary_priors=(boundary,),
        )


def test_profile_rejects_inverted_lifespan() -> None:
    boundary = FigureBoundaryPrior(
        boundary_id="b-1",
        regime_id=None,
        trigger_reasons=("trigger",),
        answer_depth_limit_hint="strong",
        clarification_required=False,
        refer_out_required=False,
        blocked_topics=("blocked",),
        required_disclaimers=(),
        confidence=0.8,
        description="boundary",
    )
    with pytest.raises(ValueError, match="figure_lifespan"):
        HistoricalFigureProfile(
            profile_id="x",
            figure_name="X",
            figure_lifespan=(2000, 1900),
            version="0.1.0",
            reviewed_by="test",
            source_uri="profile:x",
            description="test",
            domain_coverage_seed=("d",),
            knowledge_seeds=(),
            signature_cases=(),
            strategy_priors=(),
            boundary_priors=(boundary,),
        )
