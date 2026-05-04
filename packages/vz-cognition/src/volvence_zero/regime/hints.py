"""Participation and cognitive-depth hint contracts (Gap 8).

This module owns the hint *contracts* and the scaffold derivation
tables that pre-learned-metacontroller deployments fall back to.
Consumers read hints from ``RegimeSnapshot``; they should not import
these symbols directly unless they need the derivation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Participation hint + cognitive depth hint (Gap 8)
#
# Gap 8 adds two compact advisory objects to the regime snapshot so the
# lifeform-side prompt planner can decide which sections to include
# WITHOUT re-deriving the classification itself. Both are "hints"
# rather than commands: a consumer is allowed to ignore them, but the
# spec's preferred path is "regime publishes, planner reads".
#
# The default derivation now uses the Gap 8 readout path over regime,
# dual_track, evaluation, PE, and candidate signals. The static scaffold
# remains available behind ``hint_readout_mode="scaffold"`` for rollback
# and A/B comparison. The contract surface here is stable; only the
# derivation can change.
#
# Red line A (no keyword hacks): the derivation consumes typed runtime
# signals and canonical regime ids. It is NOT string matching on user text.
# ---------------------------------------------------------------------------


class ParticipationFlowKind(str, Enum):
    """High-level conversational flow bucket for the current turn."""

    SOCIAL = "social"
    ACQUAINTANCE = "acquaintance"
    INFO = "info"
    PROBLEM = "problem"
    TASK = "task"


class ParticipationLevel(str, Enum):
    """Per-section participation level \u2014 a 3-tier gate.

    Consumers (prompt planner) use this to drop sections that should
    stay out of the prompt entirely (``SILENT``), include a minimal
    posture/placeholder (``BRIEF``), or render the full structured
    section (``STRUCTURED``). Keeping it a 3-tier enum lets a
    future metacontroller readout output a discrete probability over
    three cells rather than a continuous scalar.
    """

    SILENT = "silent"
    BRIEF = "brief"
    STRUCTURED = "structured"


class CognitiveDepth(str, Enum):
    """Compute-budget axis: how much thinking effort goes into the turn.

    Maps roughly to the EmoGPT cognitive-depth tiers. Used by the
    thinking scheduler / prompt planner as an intensity dial. The
    scheduler can also skip mid-reflection for ``REFLEXIVE`` turns
    (that integration lives in Gap 4 slice 2c).
    """

    REFLEXIVE = "reflexive"
    SHALLOW = "shallow"
    FOCUSED = "focused"
    ALERT = "alert"
    DEEP = "deep"


@dataclass(frozen=True)
class ParticipationHint:
    """Advisory from ``regime`` owner to lifeform-side prompt planner.

    Which sections / fields should be included in the rendered
    response for this turn. Every field is a typed enum so there is
    no ambiguity about "what does level 'maybe' mean". Defaults
    produce a fully-structured, info-flavoured participation so that
    consumers which don't touch ``participation_hint`` still see the
    same behaviour they saw before Gap 8 landed (back-compat).

    ``confidence`` is how sure the publisher is of this hint. A low
    value means "we guessed from regime_id only, feel free to
    override"; a high value means "metacontroller readout with
    supporting evidence, trust me". Slice 1 always publishes
    ``0.4`` because the derivation is scaffold \u2014 slice 2 will
    raise that when the learned path lands.
    """

    flow_kind: ParticipationFlowKind = ParticipationFlowKind.INFO
    panorama_level: ParticipationLevel = ParticipationLevel.STRUCTURED
    method_level: ParticipationLevel = ParticipationLevel.STRUCTURED
    task_level: ParticipationLevel = ParticipationLevel.STRUCTURED
    confidence: float = 0.4
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ParticipationHint.confidence must be in [0,1], "
                f"got {self.confidence!r}"
            )


@dataclass(frozen=True)
class CognitiveDepthHint:
    """Advisory depth tier + justification.

    Like ``ParticipationHint``, this is a hint not a command. The
    prompt planner uses it to pick a section budget; the thinking
    scheduler uses it to decide whether to queue mid-reflection
    tasks (Gap 4 slice 2c).
    """

    depth: CognitiveDepth = CognitiveDepth.FOCUSED
    rationale: str = ""
    confidence: float = 0.4

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"CognitiveDepthHint.confidence must be in [0,1], "
                f"got {self.confidence!r}"
            )


# Scaffold derivation table: regime_id -> (participation hint, depth hint).
# This is intentionally static; it exists so pre-learned-metacontroller
# deployments already get regime-sensible prompt filtering. Gap 8 slice
# 2 replaces this with a learned metacontroller readout keyed off the
# same regime_id + dual_track + vitals context. Keeping the contract
# surface stable (publisher is still regime owner, consumers still read
# ``participation_hint`` / ``depth_hint``) means only the derivation
# changes when we flip to learned weights.
#
# Red line A: the key is ``regime_id`` \u2014 a canonical categorical
# field populated via learned regime ``selection_weights``. It is NOT
# keyword-matching user text. The spec (``docs/implementation/
# 13_emogpt_prd_alignment_upgrade.md`` Gap 8) explicitly allows this
# scaffold as the starting point and requires it be labelled such.
_SCAFFOLD_PARTICIPATION: dict[str, ParticipationHint] = {
    "casual_social": ParticipationHint(
        flow_kind=ParticipationFlowKind.SOCIAL,
        panorama_level=ParticipationLevel.SILENT,
        method_level=ParticipationLevel.BRIEF,
        task_level=ParticipationLevel.SILENT,
        confidence=0.4,
        rationale="scaffold:casual_social:drop panorama and task",
    ),
    "acquaintance_building": ParticipationHint(
        flow_kind=ParticipationFlowKind.ACQUAINTANCE,
        panorama_level=ParticipationLevel.BRIEF,
        method_level=ParticipationLevel.BRIEF,
        task_level=ParticipationLevel.SILENT,
        confidence=0.4,
        rationale="scaffold:acquaintance_building:low-pressure engagement",
    ),
    "emotional_support": ParticipationHint(
        flow_kind=ParticipationFlowKind.INFO,
        panorama_level=ParticipationLevel.BRIEF,
        method_level=ParticipationLevel.BRIEF,
        task_level=ParticipationLevel.SILENT,
        confidence=0.4,
        rationale="scaffold:emotional_support:support-first, no task pressure",
    ),
    "guided_exploration": ParticipationHint(
        flow_kind=ParticipationFlowKind.PROBLEM,
        panorama_level=ParticipationLevel.BRIEF,
        method_level=ParticipationLevel.BRIEF,
        task_level=ParticipationLevel.BRIEF,
        confidence=0.4,
        rationale="scaffold:guided_exploration:explore before structure",
    ),
    "problem_solving": ParticipationHint(
        flow_kind=ParticipationFlowKind.PROBLEM,
        panorama_level=ParticipationLevel.STRUCTURED,
        method_level=ParticipationLevel.STRUCTURED,
        task_level=ParticipationLevel.STRUCTURED,
        confidence=0.4,
        rationale="scaffold:problem_solving:full structured plan",
    ),
    "repair_and_deescalation": ParticipationHint(
        flow_kind=ParticipationFlowKind.INFO,
        panorama_level=ParticipationLevel.BRIEF,
        method_level=ParticipationLevel.BRIEF,
        task_level=ParticipationLevel.SILENT,
        confidence=0.4,
        rationale="scaffold:repair_and_deescalation:name-rupture-first",
    ),
}


_SCAFFOLD_DEPTH: dict[str, CognitiveDepthHint] = {
    "casual_social": CognitiveDepthHint(
        depth=CognitiveDepth.SHALLOW,
        rationale="scaffold:casual_social:minimal compute budget",
        confidence=0.4,
    ),
    "acquaintance_building": CognitiveDepthHint(
        depth=CognitiveDepth.SHALLOW,
        rationale="scaffold:acquaintance_building:low cognitive cost",
        confidence=0.4,
    ),
    "emotional_support": CognitiveDepthHint(
        depth=CognitiveDepth.FOCUSED,
        rationale="scaffold:emotional_support:attentive but not probing",
        confidence=0.4,
    ),
    "guided_exploration": CognitiveDepthHint(
        depth=CognitiveDepth.FOCUSED,
        rationale="scaffold:guided_exploration:deliberate exploration",
        confidence=0.4,
    ),
    "problem_solving": CognitiveDepthHint(
        depth=CognitiveDepth.ALERT,
        rationale="scaffold:problem_solving:high compute for structured plan",
        confidence=0.4,
    ),
    "repair_and_deescalation": CognitiveDepthHint(
        depth=CognitiveDepth.FOCUSED,
        rationale="scaffold:repair_and_deescalation:careful, non-reactive",
        confidence=0.4,
    ),
}


def derive_participation_hint(regime_id: str) -> ParticipationHint:
    """Public scaffold derivation (Gap 8 slice 1).

    Exposed so tests, family-report consumers, and future learned
    overrides can compare. Unknown regime_ids fall back to the
    default ``ParticipationHint()`` (all-STRUCTURED) \u2014 i.e. the
    safe conservative choice of "include everything, let the
    downstream consumer decide".
    """
    hint = _SCAFFOLD_PARTICIPATION.get(regime_id)
    if hint is not None:
        return hint
    return ParticipationHint(
        rationale=f"scaffold:fallback:{regime_id!r} not in derivation table"
    )


def derive_cognitive_depth_hint(regime_id: str) -> CognitiveDepthHint:
    """Public scaffold derivation for cognitive-depth hints."""
    hint = _SCAFFOLD_DEPTH.get(regime_id)
    if hint is not None:
        return hint
    return CognitiveDepthHint(
        rationale=f"scaffold:fallback:{regime_id!r} not in derivation table"
    )
