"""Lifeform-side snapshot shape for affordance candidates.

This is the public surface the prompt planner / response
synthesizer reads when deciding what to render into the prompt.
The snapshot is a frozen dataclass so consumers cannot mutate it;
the publisher (an ``AffordanceModule`` wrapper around the registry
\u2014 slice 2) rebuilds it per turn.

Slice 1 ships the shape + a synthesiser that turns a registry into
a default snapshot where every registered affordance is a
candidate with a neutral score. Slice 2 replaces the synthesiser
with a metacontroller-driven scorer keyed off regime +
temporal_abstraction + boundary_consent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from volvence_zero.affordance import (
    AffordanceCost,
    AffordanceDescriptor,
)

from lifeform_affordance.registry import AffordanceRegistry


@dataclass(frozen=True)
class AffordanceCandidate:
    """One candidate the lifeform could invoke this turn.

    ``score`` in ``[0, 1]`` \u2014 higher means "more likely to help
    with this turn". Slice 1 sets everything to 0.5 (neutral).
    ``blocked_reason`` non-empty means the safety_model / boundary
    policy has ruled this candidate out; the prompt renderer will
    typically hide it but it stays in the snapshot for audit.
    """

    descriptor_name: str
    score: float
    rationale: str
    expected_cost: AffordanceCost
    blocked_reason: str = ""
    # G3: report-only learned scorer candidate. ``None`` when no learner
    # is wired (default snapshots stay byte-compatible); NEVER consumed
    # by live selection while the learner is SHADOW.
    shadow_learned_score: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"AffordanceCandidate.score must be in [0,1], "
                f"got {self.score!r} for {self.descriptor_name!r}"
            )
        if not self.descriptor_name.strip():
            raise ValueError(
                "AffordanceCandidate.descriptor_name must be non-empty"
            )

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked_reason.strip())


@dataclass(frozen=True)
class AffordanceSnapshot:
    """The lifeform's per-turn affordance advisory.

    ``available`` is the full descriptor set the registry exposes
    (modulo ``excluded_from_runtime_selection``). ``candidates_for_turn``
    is the subset the lifeform thinks is actually worth surfacing
    this turn, with per-candidate score / rationale / blocked_reason.
    ``selected`` is what the metacontroller (slice 2) or an explicit
    rule chose; None means no affordance was picked.
    """

    available: tuple[AffordanceDescriptor, ...]
    candidates_for_turn: tuple[AffordanceCandidate, ...]
    selected: AffordanceCandidate | None
    description: str = ""

    def __post_init__(self) -> None:
        available_names = {d.name for d in self.available}
        for candidate in self.candidates_for_turn:
            if candidate.descriptor_name not in available_names:
                raise ValueError(
                    f"AffordanceSnapshot.candidates_for_turn references "
                    f"{candidate.descriptor_name!r} which is not in the "
                    f"available set; candidates must be drawn from "
                    f"available descriptors."
                )
        if self.selected is not None and self.selected.descriptor_name not in {
            c.descriptor_name for c in self.candidates_for_turn
        }:
            raise ValueError(
                f"AffordanceSnapshot.selected={self.selected.descriptor_name!r} "
                f"must be one of the candidates_for_turn."
            )


def build_neutral_snapshot(
    registry: AffordanceRegistry,
    *,
    include_excluded_from_runtime_selection: bool = False,
) -> AffordanceSnapshot:
    """Slice-1 synthesiser: turn a registry into an all-neutral snapshot.

    Every non-excluded descriptor becomes a candidate with
    ``score=0.5`` and a neutral rationale. No safety gating is
    applied here; a future ``AffordanceModule`` will hook in
    ``boundary_consent`` + ``regime.blocked_in_regimes`` to set
    ``blocked_reason`` appropriately.

    ``selected`` is None: slice 1 has no selection logic. A caller
    that wants to force a selection (e.g. a test) can build the
    snapshot manually.
    """
    available: list[AffordanceDescriptor] = []
    candidates: list[AffordanceCandidate] = []
    for d in registry.all_descriptors():
        if (
            not include_excluded_from_runtime_selection
            and d.excluded_from_runtime_selection
        ):
            continue
        available.append(d)
        candidates.append(
            AffordanceCandidate(
                descriptor_name=d.name,
                score=0.5,
                rationale="neutral scaffold (slice 1): all candidates weighted equally",
                expected_cost=d.cost_model,
            )
        )
    return AffordanceSnapshot(
        available=tuple(available),
        candidates_for_turn=tuple(candidates),
        selected=None,
        description=(
            f"Neutral affordance snapshot; {len(available)} candidate(s) "
            f"(slice 1 scaffold)."
        ),
    )


__all__ = [
    "AffordanceCandidate",
    "AffordanceSnapshot",
    "build_neutral_snapshot",
]
