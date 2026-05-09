"""Narrative arc schema for experiential replay.

A :class:`NarrativeArc` is a reviewed, ordered sequence of decision-point
scenes that drive the lifeform through a "lived life" via the
:class:`ExperientialReplayDriver` (see ``replay.py``). Each scene is the
unit at which the lifeform predicts an action and the canonical novel
answer is fed back as a typed prediction-error signal — this is what
makes "experience" first-person and PE-driven rather than just
"reading about a character".

Why this schema is structured (not free-form):

* ``setting`` is rewritten to first-person ("you stand at the bridge")
  so the lifeform's inputs are framed as what HAPPENED TO IT, not what
  it READ. Any surface that lets the kernel act differently must come
  through typed fields (decision_point / canonical_action), not from
  free-form keyword matching on prose (``no-keyword-matching-hacks`` rule).
* ``canonical_action`` is the **short paraphrase** of what the
  protagonist actually did, not a verbatim novel quotation. Wheels
  that ship NarrativeArc data must paraphrase to avoid copyright; the
  test fixtures in ``arcs/zhang_wuji_demo_arc.py`` follow this rule.
* ``phase_label`` lets later phases (drive evolution, life-stage regime
  routing) key off a typed enum-ish value rather than parsing scene
  ordering.
"""

from __future__ import annotations

from dataclasses import dataclass


_VALID_PHASE_LABELS: frozenset[str] = frozenset(
    {"child", "adolescent", "mature", "elder"}
)
_VALID_EMOTIONAL_REGISTERS: frozenset[str] = frozenset(
    {
        "calm",
        "warm",
        "tense",
        "crisis",
        "grief",
        "joy",
        "shame",
        "resolve",
        "doubt",
        "wonder",
    }
)


@dataclass(frozen=True)
class NarrativeScene:
    """One reviewed scene in a character's lived arc.

    Field-level invariants:

    * ``scene_id`` must be unique within a :class:`NarrativeArc`.
    * ``setting`` must be first-person ("you ..." in English / 第二人称
      代入 in Chinese — both styles are acceptable; the
      :class:`ExperientialReplayDriver` does not parse the prose, it
      only drives it through ``run_turn``).
    * ``canonical_action`` and ``canonical_outcome`` must be reviewer-
      written paraphrases. We do not enforce length here, but the
      review checklist in ``docs/specs/character-soul-bootstrap.md``
      requires reviewers to confirm "no verbatim novel text".
    * ``phase_label`` ∈ ``{"child", "adolescent", "mature", "elder"}``.
    * ``emotional_register`` is a closed enum-ish vocabulary; we keep
      it as ``str`` to avoid an explicit Enum import in vz-contracts.
    """

    scene_id: str
    phase_label: str
    setting: str
    decision_point: str
    canonical_action: str
    canonical_outcome: str
    emotional_register: str
    risk_markers: tuple[str, ...]
    expected_regime: str | None
    evidence_locator: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("scene_id", self.scene_id),
            ("setting", self.setting),
            ("decision_point", self.decision_point),
            ("canonical_action", self.canonical_action),
            ("canonical_outcome", self.canonical_outcome),
            ("evidence_locator", self.evidence_locator),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"NarrativeScene.{field_name} must be non-empty")
        if self.phase_label not in _VALID_PHASE_LABELS:
            raise ValueError(
                f"NarrativeScene.phase_label={self.phase_label!r} not in "
                f"{sorted(_VALID_PHASE_LABELS)}"
            )
        if self.emotional_register not in _VALID_EMOTIONAL_REGISTERS:
            raise ValueError(
                f"NarrativeScene.emotional_register={self.emotional_register!r} "
                f"not in {sorted(_VALID_EMOTIONAL_REGISTERS)}"
            )


@dataclass(frozen=True)
class NarrativeArc:
    """An ordered sequence of reviewed scenes covering a character's life.

    Invariants:

    * ``character_id`` matches the :class:`CharacterSoulProfile` this
      arc belongs to (typed coupling — ExperientialReplayDriver checks
      it before running).
    * Scenes are time-ordered. The order is the order in which the
      replay driver presents them to the lifeform, so it must
      correspond to in-novel chronology (mid-life flashbacks should
      be re-anchored or split by the reviewer).
    * ``life_phase_boundaries`` lists ``(scene_index, phase_label)``
      pairs in non-decreasing phase order; the driver / evolution
      phase use these as life-stage transitions.
    * ``reviewed_by`` and ``source_provenance`` are mandatory audit
      fields per ``character-soul-bootstrap.md``.
    * Minimum 5 scenes — too few scenes makes the replay output too
      noisy to drive any drive evolution.
    """

    arc_id: str
    character_id: str
    scenes: tuple[NarrativeScene, ...]
    life_phase_boundaries: tuple[tuple[int, str], ...]
    reviewed_by: str
    source_provenance: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("arc_id", self.arc_id),
            ("character_id", self.character_id),
            ("reviewed_by", self.reviewed_by),
            ("source_provenance", self.source_provenance),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"NarrativeArc.{field_name} must be non-empty")
        if len(self.scenes) < 5:
            raise ValueError(
                f"NarrativeArc requires >= 5 scenes, got {len(self.scenes)}"
            )
        ids = [scene.scene_id for scene in self.scenes]
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"NarrativeArc.scenes must have unique scene_ids; got {ids!r}"
            )
        previous_phase_rank = -1
        phase_rank = {label: index for index, label in enumerate(_phase_order())}
        for boundary_index, (scene_index, phase_label) in enumerate(
            self.life_phase_boundaries
        ):
            if not isinstance(scene_index, int) or scene_index < 0:
                raise ValueError(
                    "NarrativeArc.life_phase_boundaries scene_index must be "
                    f">= 0, got {scene_index!r}"
                )
            if scene_index >= len(self.scenes):
                raise ValueError(
                    f"NarrativeArc.life_phase_boundaries[{boundary_index}] "
                    f"scene_index={scene_index} >= len(scenes)={len(self.scenes)}"
                )
            if phase_label not in _VALID_PHASE_LABELS:
                raise ValueError(
                    f"NarrativeArc.life_phase_boundaries phase_label="
                    f"{phase_label!r} not in {sorted(_VALID_PHASE_LABELS)}"
                )
            current_rank = phase_rank[phase_label]
            if current_rank < previous_phase_rank:
                raise ValueError(
                    "NarrativeArc.life_phase_boundaries must be "
                    f"non-decreasing in phase order; got {self.life_phase_boundaries!r}"
                )
            previous_phase_rank = current_rank


def _phase_order() -> tuple[str, ...]:
    """Canonical chronological order of life phase labels."""
    return ("child", "adolescent", "mature", "elder")


__all__ = [
    "NarrativeArc",
    "NarrativeScene",
]
