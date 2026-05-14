"""ExperientialReplayDriver — drive a Lifeform through a NarrativeArc.

Each :class:`NarrativeScene` is processed as a 4-step micro-loop:

1. **Setting turn** — present the first-person scene framing as an
   APPRENTICE turn. The APPRENTICE trigger activates the vitals
   apprentice override for this turn only (see
   ``lifeform_core.types.is_apprenticeship_trigger``), so vitals do
   not generate "resistance PE" against being told what's happening
   — the kernel absorbs the situation as if living it.
2. **Decision turn** — present the typed ``decision_point`` as a
   USER_INPUT turn. The kernel runs its full ETA / regime / planner /
   memory pipeline and produces a response text — the lifeform's
   *predicted* action.
3. **Outcome submission** — call
   ``session.submit_dialogue_outcome`` with a typed
   :class:`DialogueExternalOutcomeKind` derived from the scene's
   emotional register and an ``evidence_ref`` carrying the canonical
   action. This is the load-bearing PE回流 step: the rupture / repair
   chain reads this typed evidence on subsequent turns and the
   prediction error owner produces a typed gap signal.
4. **End scene** — ``end_scene(drain_slow_loop=True)`` so the R6
   session-post slow loop fires before the next scene, consolidating
   any durable lessons.

What this driver IS:

* A pure orchestrator over the existing canonical session API.
* Stateless between arcs (construct fresh per arc).

What this driver is NOT:

* A new kernel owner. It does not write to any owner's private
  state; it only calls ``run_turn`` / ``submit_dialogue_outcome`` /
  ``end_scene`` — same surface a human dialogue user would use.
* An LLM-driven scene generator. The arc is a reviewed structured
  artifact; the driver does not paraphrase or extend it at runtime.
* A keyword-matching engine. The mapping from
  ``emotional_register`` to ``DialogueExternalOutcomeKind`` is a
  reviewed lookup table (typed enum-to-enum), not a free-text
  classifier.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from lifeform_core import (
    Lifeform,
    LifeformSession,
    TurnTriggerKind,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.prediction import PredictionErrorSnapshot
from volvence_zero.regime import RegimeSnapshot

from lifeform_domain_character.narrative import NarrativeArc, NarrativeScene


# Reviewed mapping from emotional register to the typed outcome kind
# that best characterises the scene's external signal. The mapping
# is **closed** — every value in ``_VALID_EMOTIONAL_REGISTERS`` (see
# ``narrative.py``) appears here so a runtime lookup never falls
# through to a default.
_REGISTER_TO_OUTCOME: dict[str, DialogueExternalOutcomeKind] = {
    "calm": DialogueExternalOutcomeKind.HELPED,
    "warm": DialogueExternalOutcomeKind.FELT_HEARD,
    "tense": DialogueExternalOutcomeKind.DECISION_CLEARER,
    "crisis": DialogueExternalOutcomeKind.DECISION_CLEARER,
    "grief": DialogueExternalOutcomeKind.COME_BACK,
    "joy": DialogueExternalOutcomeKind.HELPED,
    "shame": DialogueExternalOutcomeKind.MISSED,
    "resolve": DialogueExternalOutcomeKind.HELPED,
    "doubt": DialogueExternalOutcomeKind.DECISION_CLEARER,
    "wonder": DialogueExternalOutcomeKind.FELT_HEARD,
}


@dataclass(frozen=True)
class SceneReplayRecord:
    """Per-scene audit record returned in :class:`ReplayReport`.

    All fields are typed primitives so callers (Phase 4 evolution
    pipeline, Phase 5 demo) can reason about each scene without
    re-parsing kernel snapshots.
    """

    scene_id: str
    phase_label: str
    predicted_action_snippet: str
    canonical_action: str
    outcome_kind: str
    pe_magnitude: float
    active_regime: str | None
    drive_level_after: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ReplayReport:
    """Aggregate result of running an arc through the driver."""

    arc_id: str
    character_id: str
    scenes_processed: int
    per_scene: tuple[SceneReplayRecord, ...]
    total_pe_signal: float
    drive_drift: tuple[tuple[str, float], ...]  # name -> end - start
    regime_sequence_payoff_growth: int
    final_vitals: tuple[tuple[str, float], ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


def _drive_levels_from_session(session: LifeformSession) -> tuple[tuple[str, float], ...]:
    snapshot = session.vitals_snapshot
    if snapshot is None:
        return ()
    return tuple((d.name, float(d.level)) for d in snapshot.drive_levels)


def _regime_sequence_payoff_count(active_snapshots: dict[str, Any]) -> int:
    # ``RegimeSnapshot.sequence_payoffs`` is the typed field name (plural);
    # direct access keeps schema drift loud per R8 / SSOT.
    snap = active_snapshots.get("regime")
    if snap is None or not isinstance(snap.value, RegimeSnapshot):
        return 0
    return len(snap.value.sequence_payoffs)


def _pe_magnitude(active_snapshots: dict[str, Any]) -> float:
    snap = active_snapshots.get("prediction_error")
    if snap is None or not isinstance(snap.value, PredictionErrorSnapshot):
        return 0.0
    return float(snap.value.error.magnitude)


def _truncate(text: str, *, max_chars: int = 240) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


class ExperientialReplayDriver:
    """Drive a NarrativeArc through a Lifeform session.

    Construct one driver per arc / session combination; reuse for a
    single ``run_arc`` call. The driver is intentionally stateless
    between calls (no instance attributes carry over) so accidental
    reuse cannot leak observations between arcs.
    """

    def __init__(
        self,
        *,
        scene_end_drains_slow_loop: bool = True,
        evidence_confidence: float = 0.9,
        evidence_source: DialogueExternalOutcomeEvidenceSource = (
            DialogueExternalOutcomeEvidenceSource.HUMAN_REVIEW
        ),
    ) -> None:
        self._scene_end_drains_slow_loop = scene_end_drains_slow_loop
        if not 0.0 <= evidence_confidence <= 1.0:
            raise ValueError(
                "evidence_confidence must be in [0, 1], got "
                f"{evidence_confidence!r}"
            )
        self._evidence_confidence = float(evidence_confidence)
        self._evidence_source = evidence_source

    async def run_arc_async(
        self,
        *,
        arc: NarrativeArc,
        lifeform: Lifeform,
        session_id_prefix: str = "replay",
    ) -> ReplayReport:
        """Run every scene in the arc against a fresh session.

        A fresh session per arc keeps scene-to-scene state inside the
        memory store + ETA + regime owners (which is what we want for
        cross-scene accumulation), while resetting transient state
        like the open scene boundary. This is the same pattern the
        longitudinal benchmark uses when sharing a memory store
        across rounds.
        """
        if arc.character_id != _profile_id_from_lifeform(lifeform):
            raise ValueError(
                "ExperientialReplayDriver: arc.character_id="
                f"{arc.character_id!r} does not match the lifeform's "
                f"profile id {_profile_id_from_lifeform(lifeform)!r}; "
                "the wrong character is about to live the wrong life. "
                "Pass the correct lifeform or build a different arc."
            )
        session = lifeform.create_session(
            session_id=f"{session_id_prefix}-{arc.arc_id}"
        )
        start_drives = dict(_drive_levels_from_session(session))
        # Establish a baseline snapshot via one bootstrap turn so
        # subsequent regime / PE deltas have something to compare to.
        bootstrap_result = await session.run_turn(
            f"开始经历 {arc.character_id} 的一段经历。",
            trigger_kind=TurnTriggerKind.USER_INPUT,
        )
        baseline_regime_payoff = _regime_sequence_payoff_count(
            bootstrap_result.active_snapshots
        )
        per_scene: list[SceneReplayRecord] = []
        total_pe = 0.0
        for scene in arc.scenes:
            record = await self._replay_one_scene(scene=scene, session=session)
            per_scene.append(record)
            total_pe += record.pe_magnitude
        # Final vitals + regime payoff after all scenes.
        final_drives = _drive_levels_from_session(session)
        final_drive_dict = dict(final_drives)
        drive_drift = tuple(
            (name, final_drive_dict.get(name, 0.0) - start_drives.get(name, 0.0))
            for name in sorted(set(start_drives) | set(final_drive_dict))
        )
        # Pull final regime sequence_payoff via one read-only turn so
        # we can compare against the bootstrap baseline.
        sentinel = await session.run_turn(
            "经历结束，回望刚才。",
            trigger_kind=TurnTriggerKind.USER_INPUT,
        )
        final_regime_payoff = _regime_sequence_payoff_count(
            sentinel.active_snapshots
        )
        return ReplayReport(
            arc_id=arc.arc_id,
            character_id=arc.character_id,
            scenes_processed=len(per_scene),
            per_scene=tuple(per_scene),
            total_pe_signal=total_pe,
            drive_drift=drive_drift,
            regime_sequence_payoff_growth=(
                final_regime_payoff - baseline_regime_payoff
            ),
            final_vitals=final_drives,
        )

    def run_arc(
        self,
        *,
        arc: NarrativeArc,
        lifeform: Lifeform,
        session_id_prefix: str = "replay",
    ) -> ReplayReport:
        """Synchronous convenience wrapper for callers outside an
        existing event loop."""
        return asyncio.run(
            self.run_arc_async(
                arc=arc,
                lifeform=lifeform,
                session_id_prefix=session_id_prefix,
            )
        )

    async def _replay_one_scene(
        self,
        *,
        scene: NarrativeScene,
        session: LifeformSession,
    ) -> SceneReplayRecord:
        # Step 1: setting as APPRENTICE turn so vitals override absorbs it.
        await session.run_turn(
            scene.setting,
            trigger_kind=TurnTriggerKind.APPRENTICE,
        )
        # Step 2: decision turn — lifeform predicts.
        decision_result = await session.run_turn(
            scene.decision_point,
            trigger_kind=TurnTriggerKind.USER_INPUT,
        )
        predicted_text = _truncate(decision_result.response.text)
        active_regime = decision_result.active_regime
        pe_magnitude = _pe_magnitude(decision_result.active_snapshots)
        # Step 3: typed outcome回流.
        outcome_kind = _REGISTER_TO_OUTCOME[scene.emotional_register]
        try:
            session.submit_dialogue_outcome(
                kind=outcome_kind,
                source=self._evidence_source,
                confidence=self._evidence_confidence,
                evidence_ref=f"replay:{scene.scene_id}:{scene.canonical_action[:80]}",
                description=(
                    f"Replay outcome for scene {scene.scene_id} (phase="
                    f"{scene.phase_label}); canonical: {scene.canonical_outcome[:160]}"
                ),
            )
        except RuntimeError as exc:
            # ``submit_dialogue_outcome`` raises when the brain config
            # disallows the configured source (e.g. LLM_PROPOSAL with
            # the flag off). In replay we use HUMAN_REVIEW which is
            # always allowed, so a RuntimeError here is a legitimate
            # configuration regression and we re-raise.
            raise RuntimeError(
                f"submit_dialogue_outcome failed on scene {scene.scene_id!r} "
                f"with source={self._evidence_source!r}: {exc}"
            ) from exc
        # Step 4: end scene with slow-loop drain so durable lessons
        # consolidate before the next scene.
        await session.end_scene(
            reason=f"replay-scene-end:{scene.scene_id}",
            drain_slow_loop=self._scene_end_drains_slow_loop,
        )
        drives_after = _drive_levels_from_session(session)
        return SceneReplayRecord(
            scene_id=scene.scene_id,
            phase_label=scene.phase_label,
            predicted_action_snippet=predicted_text,
            canonical_action=scene.canonical_action,
            outcome_kind=outcome_kind.value,
            pe_magnitude=pe_magnitude,
            active_regime=active_regime,
            drive_level_after=drives_after,
        )


def _profile_id_from_lifeform(lifeform: Lifeform) -> str:
    """Pull the character profile id from the lifeform's compiled
    DomainExperiencePackage. Used by the driver for the
    arc-vs-character integrity check.
    """
    pkgs = lifeform.config.brain_config.domain_experience_packages
    for pkg in pkgs:
        package_id = pkg.manifest.package_id
        if package_id.startswith("lifeform-character:"):
            return package_id.split(":", 1)[1]
    raise ValueError(
        "lifeform does not appear to be a character vertical "
        "(no lifeform-character:* package in its config). "
        "ExperientialReplayDriver requires a character lifeform."
    )


__all__ = [
    "ExperientialReplayDriver",
    "ReplayReport",
    "SceneReplayRecord",
]
