from __future__ import annotations

import pytest

from volvence_zero.agent.longitudinal_human_anchor import (
    HumanAnchorProtocol,
    LongitudinalPersonaPlan,
    build_longitudinal_human_anchor_manifest,
)


def test_manifest_matches_plan_shape() -> None:
    manifest = build_longitudinal_human_anchor_manifest()
    assert manifest.schema_version == "longitudinal-human-anchor-manifest.v1"
    assert len(manifest.persona_plans) == 5
    assert manifest.total_sessions == 100
    for persona in manifest.persona_plans:
        assert persona.session_count == 20
        assert persona.min_turns_per_session == 8
        assert persona.max_turns_per_session == 15
        assert "relationship_continuity" in persona.tracked_metrics
        assert "default-isolation" in persona.comparison_arms
    assert manifest.human_anchor.blinded_rater_count == 3
    assert manifest.human_anchor.min_inter_rater_agreement == 0.6
    assert "volvence" in manifest.human_anchor.comparison_arms
    assert "expected_label" in manifest.human_anchor.hidden_fields


def test_persona_plan_rejects_underpowered_session_count() -> None:
    with pytest.raises(ValueError, match="session_count"):
        LongitudinalPersonaPlan(
            persona_id="too-short",
            session_count=19,
            min_turns_per_session=8,
            max_turns_per_session=15,
            comparison_arms=("a", "b"),
            tracked_metrics=("relationship_continuity",),
        )


def test_human_anchor_rejects_underpowered_raters() -> None:
    with pytest.raises(ValueError, match="blinded_rater_count"):
        HumanAnchorProtocol(
            blinded_rater_count=2,
            min_inter_rater_agreement=0.6,
            comparison_arms=("volvence", "raw"),
            hidden_fields=("profile_label",),
        )
