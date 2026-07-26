"""The decision workspace owner.

Holds the *structure* of a decision under consideration — which options
are live, which dimensions they are being weighed on, which unknowns the
ranking hangs on, what evidence has been cited, and whether a conclusion
has been reached. Nothing else.

Two properties do most of the architectural work here:

**It does not decide when it exists.** Activation is read from
``regime``'s ``participation_hint.panorama_level`` — the single gate
(see ``docs/specs/cognitive-regime.md``). A second activation judgement
living in this owner would be exactly the scenario-rule sprawl the gate
exists to avoid, so the module has no notion of stakes, topic, or
urgency at all. It reads one enum.

**It holds references, not copies.** The semantic facts belong to the
owners that already publish them: what the user values is
``goal_value``, what is still unresolved is ``open_loop``, what needs
verifying is ``belief_assumption``, which plans are candidates is
``plan_intent``. This owner records *that option A is on the table and
corresponds to plan-ref X* — never A's summary text. That is enforced
structurally rather than by discipline: none of the records below has a
field that could hold prose. Two writers of the same fact drift apart
the moment a session is rehydrated.

**Not part of the semantic spine.** ``semantic_spine_coverage`` is
computed over the five core semantic owners; this slot must stay out of
that denominator, or every historical paper-suite and companion readout
shifts underneath itself for reasons that have nothing to do with the
relationship state those metrics track.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from volvence_zero.regime import ParticipationLevel, RegimeSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel

# Conclusion states. Deliberately three, and deliberately not a
# confidence number: the distinction that matters downstream is whether
# the system is entitled to *state* a conclusion, not how sure it feels.
CONCLUSION_NONE = "none"
# A working answer that is explicitly contingent on unresolved unknowns.
# Rendering must stay hedged while any unknown remains open.
CONCLUSION_PROVISIONAL = "provisional"
# Every unknown that could reorder the options has been resolved.
CONCLUSION_SETTLED = "settled"


@dataclass(frozen=True)
class DecisionOption:
    """One live path.

    ``plan_ref`` points into ``plan_intent``; the option's content lives
    there. There is deliberately no ``label`` / ``summary`` field — the
    absence is the ownership contract.
    """

    option_id: str
    plan_ref: str
    status: str = "live"


@dataclass(frozen=True)
class DecisionUnknown:
    """Something the ranking waits on.

    At least one ref must be set; an unknown with no source is an
    unknown this owner invented.
    """

    unknown_id: str
    open_loop_ref: str | None = None
    belief_ref: str | None = None

    def __post_init__(self) -> None:
        if self.open_loop_ref is None and self.belief_ref is None:
            raise ValueError(
                f"DecisionUnknown {self.unknown_id!r} has no source ref; "
                f"unknowns are held by open_loop / belief_assumption and "
                f"referenced here, never originated here"
            )


@dataclass(frozen=True)
class DecisionWorkspaceSnapshot:
    """Published decision structure for the current turn."""

    engagement: ParticipationLevel = ParticipationLevel.SILENT
    engagement_rationale: str = ""
    options: tuple[DecisionOption, ...] = ()
    # Ids into ``goal_value.value_priorities``: the axes options are
    # weighed on. The weights themselves stay in goal_value.
    dimension_refs: tuple[str, ...] = ()
    unknowns: tuple[DecisionUnknown, ...] = ()
    # Ids of evidence records backing any stated claim. A conclusion
    # that cites nothing is a conclusion with no provenance.
    evidence_refs: tuple[str, ...] = ()
    conclusion_state: str = CONCLUSION_NONE
    description: str = ""

    @property
    def instantiated(self) -> bool:
        return self.engagement is not ParticipationLevel.SILENT


def _regime_snapshot(upstream: Mapping[str, Snapshot[Any]]) -> RegimeSnapshot | None:
    snapshot = upstream.get("regime")
    value = getattr(snapshot, "value", None)
    return value if isinstance(value, RegimeSnapshot) else None


def _typed(upstream: Mapping[str, Snapshot[Any]], slot: str, expected: type) -> Any:
    """Read a slot's value, rejecting inactive owners' placeholders."""
    value = getattr(upstream.get(slot), "value", None)
    return value if isinstance(value, expected) else None


class DecisionWorkspaceModule(RuntimeModule[DecisionWorkspaceSnapshot]):
    slot_name = "decision_workspace"
    owner = "DecisionWorkspaceModule"
    value_type = DecisionWorkspaceSnapshot
    dependencies = (
        "regime",
        "plan_intent",
        "goal_value",
        "open_loop",
        "belief_assumption",
    )
    default_wiring_level = WiringLevel.SHADOW

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[DecisionWorkspaceSnapshot]:
        from companion_standard.semantic_state import (
            BeliefAssumptionSnapshot,
            GoalValueSnapshot,
            OpenLoopSnapshot,
            PlanIntentSnapshot,
        )

        regime = _regime_snapshot(upstream)
        hint = regime.participation_hint if regime is not None else None
        engagement = (
            hint.panorama_level if hint is not None else ParticipationLevel.SILENT
        )
        rationale = hint.rationale if hint is not None else "no-regime-snapshot"

        # SILENT: the workspace is not instantiated. Note this is a real
        # early return, not a filter applied to a populated structure —
        # a turn the gate closed leaves no decision structure behind at
        # all, so nothing downstream can quietly read it anyway.
        if engagement is ParticipationLevel.SILENT:
            return self.publish(
                DecisionWorkspaceSnapshot(
                    engagement=engagement,
                    engagement_rationale=rationale,
                    description="decision_workspace: not instantiated (panorama silent)",
                )
            )

        plan_intent = _typed(upstream, "plan_intent", PlanIntentSnapshot)
        goal_value = _typed(upstream, "goal_value", GoalValueSnapshot)
        open_loop = _typed(upstream, "open_loop", OpenLoopSnapshot)
        belief = _typed(upstream, "belief_assumption", BeliefAssumptionSnapshot)

        options = self._options(plan_intent)
        unknowns = self._unknowns(open_loop, belief)

        # BRIEF: keep the option set and the unknowns alive so the next
        # turn can escalate from an observed structure rather than from
        # scratch, but publish no dimensions and no conclusion. Nothing
        # here reaches the prompt at this tier.
        if engagement is ParticipationLevel.BRIEF:
            return self.publish(
                DecisionWorkspaceSnapshot(
                    engagement=engagement,
                    engagement_rationale=rationale,
                    options=options,
                    unknowns=unknowns,
                    description=(
                        f"decision_workspace: tracking {len(options)} option(s), "
                        f"{len(unknowns)} unknown(s); structure withheld (brief)"
                    ),
                )
            )

        dimension_refs = (
            tuple(record.record_id for record in goal_value.value_priorities)
            if goal_value is not None
            else ()
        )
        evidence_refs = self._evidence_refs(belief)
        conclusion_state = (
            CONCLUSION_SETTLED if not unknowns else CONCLUSION_PROVISIONAL
        )
        return self.publish(
            DecisionWorkspaceSnapshot(
                engagement=engagement,
                engagement_rationale=rationale,
                options=options,
                dimension_refs=dimension_refs,
                unknowns=unknowns,
                evidence_refs=evidence_refs,
                conclusion_state=conclusion_state,
                description=(
                    f"decision_workspace: {len(options)} option(s) over "
                    f"{len(dimension_refs)} dimension(s), {len(unknowns)} "
                    f"open unknown(s), conclusion={conclusion_state}"
                ),
            )
        )

    # ------------------------------------------------------------------
    # Reference extraction. Each helper only ever copies record ids.
    # ------------------------------------------------------------------

    @staticmethod
    def _options(plan_intent: Any) -> tuple[DecisionOption, ...]:
        if plan_intent is None:
            return ()
        options: list[DecisionOption] = []
        for record in plan_intent.candidate_plans:
            options.append(
                DecisionOption(
                    option_id=f"option:{record.record_id}",
                    plan_ref=record.record_id,
                    status="chosen"
                    if record.record_id == plan_intent.active_plan_id
                    else "live",
                )
            )
        for record in plan_intent.deferred_intents:
            options.append(
                DecisionOption(
                    option_id=f"option:{record.record_id}",
                    plan_ref=record.record_id,
                    status="deferred",
                )
            )
        return tuple(options)

    @staticmethod
    def _unknowns(open_loop: Any, belief: Any) -> tuple[DecisionUnknown, ...]:
        unknowns: list[DecisionUnknown] = []
        if belief is not None:
            for record in belief.verification_needs:
                unknowns.append(
                    DecisionUnknown(
                        unknown_id=f"unknown:{record.record_id}",
                        belief_ref=record.record_id,
                    )
                )
        if open_loop is not None:
            for record in open_loop.unresolved_loops:
                unknowns.append(
                    DecisionUnknown(
                        unknown_id=f"unknown:{record.record_id}",
                        open_loop_ref=record.record_id,
                    )
                )
        return tuple(unknowns)

    @staticmethod
    def _evidence_refs(belief: Any) -> tuple[str, ...]:
        if belief is None:
            return ()
        return tuple(record.record_id for record in belief.beliefs)
