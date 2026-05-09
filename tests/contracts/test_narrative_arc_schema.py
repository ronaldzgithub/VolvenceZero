"""Wave T1 contract tests — NarrativeScene / NarrativeArc schema.

Pin the schema invariants so future edits cannot accidentally:

* drop required fields
* accept invalid phase / emotional_register values
* let arcs ship with duplicate scene_ids or out-of-order phase
  boundaries

Also pin the shipped 张无忌 demo arc's structural counts so a
future cleanup that prunes its scene set will fail this test
before it can break the Phase 1 e2e replay test.
"""

from __future__ import annotations

import pytest

from lifeform_domain_character import build_zhang_wuji_demo_arc
from lifeform_domain_character.narrative import NarrativeArc, NarrativeScene


def _good_scene(**overrides) -> NarrativeScene:
    base: dict[str, object] = {
        "scene_id": "scene-1",
        "phase_label": "child",
        "setting": "you stand at the bridge.",
        "decision_point": "what do you do?",
        "canonical_action": "step forward.",
        "canonical_outcome": "you cross safely.",
        "emotional_register": "calm",
        "risk_markers": ("risk-low",),
        "expected_regime": "guided_exploration",
        "evidence_locator": "test:scene-1",
    }
    base.update(overrides)
    return NarrativeScene(**base)  # type: ignore[arg-type]


def test_scene_rejects_empty_required_field() -> None:
    with pytest.raises(ValueError, match="setting"):
        _good_scene(setting="   ")
    with pytest.raises(ValueError, match="canonical_action"):
        _good_scene(canonical_action="")


def test_scene_rejects_unknown_phase_label() -> None:
    with pytest.raises(ValueError, match="phase_label"):
        _good_scene(phase_label="middle-aged")


def test_scene_rejects_unknown_emotional_register() -> None:
    with pytest.raises(ValueError, match="emotional_register"):
        _good_scene(emotional_register="frenzied")


def test_scene_accepts_none_expected_regime() -> None:
    """Some scenes legitimately do not pin a target regime; the
    schema must allow ``None`` rather than forcing a string."""
    scene = _good_scene(expected_regime=None)
    assert scene.expected_regime is None


def test_arc_rejects_too_few_scenes() -> None:
    with pytest.raises(ValueError, match=">= 5 scenes"):
        NarrativeArc(
            arc_id="arc-1",
            character_id="test-char",
            scenes=tuple(_good_scene(scene_id=f"s{i}") for i in range(3)),
            life_phase_boundaries=((0, "child"),),
            reviewed_by="r",
            source_provenance="t",
        )


def test_arc_rejects_duplicate_scene_ids() -> None:
    scenes = tuple(_good_scene(scene_id="dup") for _ in range(5))
    with pytest.raises(ValueError, match="unique scene_ids"):
        NarrativeArc(
            arc_id="arc-2",
            character_id="test-char",
            scenes=scenes,
            life_phase_boundaries=((0, "child"),),
            reviewed_by="r",
            source_provenance="t",
        )


def test_arc_rejects_out_of_order_phase_boundaries() -> None:
    scenes = tuple(_good_scene(scene_id=f"s{i}") for i in range(6))
    with pytest.raises(ValueError, match="non-decreasing in phase order"):
        NarrativeArc(
            arc_id="arc-3",
            character_id="test-char",
            scenes=scenes,
            life_phase_boundaries=(
                (0, "mature"),
                (3, "child"),  # phase regresses — should fail
            ),
            reviewed_by="r",
            source_provenance="t",
        )


def test_arc_rejects_phase_boundary_index_out_of_range() -> None:
    scenes = tuple(_good_scene(scene_id=f"s{i}") for i in range(5))
    with pytest.raises(ValueError, match="scene_index="):
        NarrativeArc(
            arc_id="arc-4",
            character_id="test-char",
            scenes=scenes,
            life_phase_boundaries=((10, "child"),),
            reviewed_by="r",
            source_provenance="t",
        )


def test_arc_rejects_empty_required_string() -> None:
    scenes = tuple(_good_scene(scene_id=f"s{i}") for i in range(5))
    with pytest.raises(ValueError, match="reviewed_by"):
        NarrativeArc(
            arc_id="arc-5",
            character_id="test-char",
            scenes=scenes,
            life_phase_boundaries=((0, "child"),),
            reviewed_by="   ",
            source_provenance="t",
        )


# ---------------------------------------------------------------------------
# Shipped 张无忌 demo arc
# ---------------------------------------------------------------------------


def test_zhang_wuji_demo_arc_constructs() -> None:
    arc = build_zhang_wuji_demo_arc()
    assert arc.arc_id == "zhang-wuji-demo-arc-v0"
    assert arc.character_id == "zhang-wuji"


def test_zhang_wuji_demo_arc_minimum_scene_count() -> None:
    """Pin scene count >= 10 so the Phase 1 replay test has enough
    decision points to drive cross-phase regime / drive evolution.
    """
    arc = build_zhang_wuji_demo_arc()
    assert len(arc.scenes) >= 10


def test_zhang_wuji_demo_arc_covers_three_life_phases() -> None:
    arc = build_zhang_wuji_demo_arc()
    seen_phases = {scene.phase_label for scene in arc.scenes}
    expected = {"child", "adolescent", "mature"}
    missing = expected - seen_phases
    assert not missing, f"demo arc missing phases: {missing}"


def test_zhang_wuji_demo_arc_phase_boundaries_monotonic() -> None:
    arc = build_zhang_wuji_demo_arc()
    phase_order = ("child", "adolescent", "mature", "elder")
    rank = {phase: i for i, phase in enumerate(phase_order)}
    last_rank = -1
    for _, label in arc.life_phase_boundaries:
        current = rank[label]
        assert current >= last_rank, (
            f"phase boundaries regressed at {label!r}; "
            f"boundaries={arc.life_phase_boundaries!r}"
        )
        last_rank = current


def test_zhang_wuji_demo_arc_scene_ids_unique() -> None:
    arc = build_zhang_wuji_demo_arc()
    ids = [scene.scene_id for scene in arc.scenes]
    assert len(set(ids)) == len(ids), f"duplicate scene_ids in demo arc: {ids}"


def test_zhang_wuji_demo_arc_keys_to_existing_profile_id() -> None:
    """The arc must target the shipped 张无忌 profile id so the
    replay driver can fail loudly when given a mismatched lifeform.
    """
    from lifeform_domain_character import build_zhang_wuji_profile

    arc = build_zhang_wuji_demo_arc()
    profile = build_zhang_wuji_profile()
    assert arc.character_id == profile.profile_id
