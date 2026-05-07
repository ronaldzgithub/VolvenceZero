"""End-to-end demo: companion vertical + thinking adapter + ingestion +
InterlocutorState readout.

Runs a scripted multi-turn companionship session that exercises:

* ``lifeform-domain-emogpt`` (the companion vertical: drives,
  scenarios, regime + temporal bootstraps).
* ``lifeform-ingestion`` (Gap 3): a "background memo about the user"
  is ingested via ``compliance_profile=FORCED`` BEFORE the
  conversation starts, so the lifeform sees that material as an
  apprenticeship turn (vitals override on, no proactive PE).
* ``lifeform-thinking`` (Gap 4 slice 2c): mid-frequency reflection
  in SHADOW mode, automatically submitting + collecting
  world-/self-lane reflections.
* ``LifeformSession.interlocutor_state`` (Gap 9 slice 1): the
  12-axis emotional/relational readout, sampled at every turn so
  we can SEE rapport / resistance / trust evolve.

**Import discipline:** same as ``coding_end_to_end_demo.py`` \u2014
public ``lifeform-*`` wheel imports only. No ``volvence_zero.*``
imports. Two contract tests in
``tests/lifeform_e2e/test_companionship_end_to_end_demo.py`` enforce
this.

The script is a deliberate diagnostic: it does NOT inject scripted
``CommitmentSnapshot`` records or force the regime classifier into
specific bins. It just drives natural-language turns and then PRINTS
what the kernel actually produced. That way the demo's audit trail
is an honest report of what companion vertical gives us today.

Run:

    python examples/companionship_end_to_end_demo.py

Output is structured as a header-banner per phase plus a per-turn
trajectory table at the end so the InterlocutorState evolution is
scannable in one glance.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any

# Force UTF-8 stdout so the demo's em-dashes / arrows render cleanly
# on Windows consoles that default to GBK. Falls back to replacement
# characters rather than crashing on encode errors. No-op on POSIX
# systems that already speak UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    # ``reconfigure`` is Python 3.7+ on TextIOWrapper-backed streams;
    # if stdout has been redirected to something exotic we just leave
    # it alone.
    pass

# ---------------------------------------------------------------------------
# Public wheel imports only. Same discipline as the coding demo.
# ---------------------------------------------------------------------------

from lifeform_core import DialogueExternalOutcomeKind
from lifeform_domain_emogpt import build_companion_lifeform
from lifeform_expression import GroundedResponseSynthesizer, PromptPlanner
from lifeform_ingestion import (
    IngestionPipeline,
    envelope_from_text,
)
from lifeform_thinking import (
    ThinkingWiringLevel,
    build_default_thinking_adapter,
)


# ---------------------------------------------------------------------------
# Output helpers (mirrors the coding demo)
# ---------------------------------------------------------------------------


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _print_section(tag: str, text: str) -> None:
    print(f"  [{tag}] {text}")


def _print_bullet(text: str) -> None:
    print(f"    - {text}")


# ---------------------------------------------------------------------------
# Per-turn trajectory record
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    """One row of the trajectory table printed at end-of-demo."""

    label: str
    regime: str
    engagement: float
    focus: float
    rapport: float
    resistance: float
    openness: float
    trust: float
    confidence: float


@dataclass
class CompanionAudit:
    """Lightweight counters + history for the summary phase."""

    repair_alpha_enabled: bool = True
    ingestion_turns: int = 0
    natural_turns: int = 0
    trajectory: list[TurnRecord] = field(default_factory=list)
    commitments_observed: int = 0
    open_loops_observed: int = 0
    followups_pending_at_close: int = 0
    thinking_artifacts: int = 0
    case_memory_promoted: int = 0
    case_memory_retired: int = 0
    case_memory_expired: int = 0
    external_outcomes_submitted: int = 0
    ruptures_observed: int = 0
    repair_alpha_response: str = ""
    repair_alpha_rationale: str = ""


# ---------------------------------------------------------------------------
# Background memo (the "ingested" content). Real product would surface
# this from a CRM / user profile / notes app; here we hard-code a tiny
# narrative paragraph to keep the demo self-contained.
# ---------------------------------------------------------------------------


_BACKGROUND_MEMO = """The interlocutor's name is Mei. She is a 34-year-old
graphic designer based in Shanghai, currently between full-time roles. She
has mentioned in past conversations that she is working through a slow
transition out of agency life into something more independent, and that
she values being heard more than being given advice. She has a cat named
Tofu and lives alone. She tends to disclose carefully and pulls back fast
when conversations turn clinical.

In recent sessions she has discussed feeling stuck, low-energy mornings,
and a tentative idea about freelancing. She has NOT made any concrete
commitments yet. Treat this material as background context; do not bring
it up unless she does.
"""


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


def _snapshot_to_record(label: str, session: Any) -> TurnRecord:
    """Pull the InterlocutorState + active regime into one record.

    Treats every snapshot as opaque to demo code: only goes through
    the public ``interlocutor_state`` property and the regime
    snapshot's documented ``active_regime.regime_id`` attribute.
    """
    state = session.interlocutor_state
    regime_id = ""
    regime_snap = session.latest_active_snapshots.get("regime")
    if regime_snap is not None:
        active = getattr(regime_snap.value, "active_regime", None)
        if active is not None:
            regime_id = str(getattr(active, "regime_id", ""))
    return TurnRecord(
        label=label,
        regime=regime_id or "<none>",
        engagement=state.engagement_intensity,
        focus=state.task_focus_level,
        rapport=state.rapport_warmth,
        resistance=state.resistance_level,
        openness=state.openness_to_guidance,
        trust=state.trust_signal,
        confidence=state.readout_confidence,
    )


def _print_observed_owners(session: Any, *, audit: CompanionAudit) -> None:
    """Probe the four companion-relevant kernel snapshots and report.

    Treats the snapshot ``.value`` as a structural object; we use
    ``getattr`` with defaults so the demo doesn't crash on a build
    that ships slightly different field names. This is the honest
    approach for a diagnostic: report what's actually there, not
    what we hope is there.
    """
    snaps = session.latest_active_snapshots

    commitment_snap = snaps.get("commitment")
    if commitment_snap is not None:
        active = getattr(commitment_snap.value, "active_commitments", ()) or ()
        lifecycle = getattr(commitment_snap.value, "lifecycle_entries", ()) or ()
        if active or lifecycle:
            audit.commitments_observed += len(active)
            _print_bullet(
                f"commitment: active={len(active)} "
                f"lifecycle_entries={len(lifecycle)}"
            )
            for entry in lifecycle[:3]:
                ref = (
                    getattr(entry, "commitment_ref", None)
                    or getattr(entry, "record_id", None)
                    or "?"
                )
                advocacy = getattr(entry, "last_advocacy", None)
                alignment = getattr(entry, "last_alignment", None)
                advocacy_v = getattr(advocacy, "value", advocacy)
                alignment_v = getattr(alignment, "value", alignment)
                _print_bullet(
                    f"  -> {ref} advocacy={advocacy_v!s} alignment={alignment_v!s}"
                )

    open_loop_snap = snaps.get("open_loop")
    if open_loop_snap is not None:
        unresolved = (
            getattr(open_loop_snap.value, "unresolved_loops", ()) or ()
        )
        if unresolved:
            audit.open_loops_observed += len(unresolved)
            _print_bullet(f"open_loop: unresolved={len(unresolved)}")

    rs_snap = snaps.get("relationship_state")
    if rs_snap is not None:
        # Fields are documented but specific names vary across builds;
        # touch them defensively.
        stage = getattr(rs_snap.value, "stage", None)
        trust = getattr(rs_snap.value, "trust_level", None)
        _print_bullet(
            f"relationship_state: stage={stage!s} trust_level={trust!s}"
        )

    rupture = session.rupture_state
    if rupture is not None:
        rupture_kind = rupture.rupture_kind
        kind_text = rupture_kind.value if rupture_kind is not None else "<none>"
        sources = ",".join(source.value for source in rupture.evidence_sources)
        if rupture_kind is not None:
            audit.ruptures_observed += 1
        _print_bullet(
            "rupture_state: "
            f"kind={kind_text} confidence={rupture.confidence:.2f} "
            f"signal={rupture.rupture_signal_strength:.2f} "
            f"internal_only={rupture.internal_suspected_only} "
            f"sources={sources or '<none>'}"
        )


def _print_interlocutor_snapshot(session: Any, *, tag: str) -> None:
    state = session.interlocutor_state
    line = (
        f"engagement={state.engagement_intensity:.2f} "
        f"focus={state.task_focus_level:.2f} "
        f"emotional={state.emotional_weight:.2f} "
        f"rapport={state.rapport_warmth:.2f} "
        f"resistance={state.resistance_level:.2f} "
        f"openness={state.openness_to_guidance:.2f} "
        f"directness={state.directness:.2f} "
        f"trust={state.trust_signal:+.2f}"
    )
    _print_section(tag, line)


def _print_trajectory_table(audit: CompanionAudit) -> None:
    """Pretty-print the per-turn evolution.

    Right-aligned numeric columns + the regime tag so the reader
    can see at a glance how the conversation moved across the
    InterlocutorState surface.
    """
    print()
    print(
        "    "
        f"{'phase':<22}"
        f"{'regime':<24}"
        f"{'eng':>6}"
        f"{'focus':>7}"
        f"{'rapp':>7}"
        f"{'res':>7}"
        f"{'open':>7}"
        f"{'trust':>8}"
        f"{'conf':>7}"
    )
    print("    " + "-" * 92)
    for record in audit.trajectory:
        regime_short = record.regime[:22]
        print(
            "    "
            f"{record.label:<22}"
            f"{regime_short:<24}"
            f"{record.engagement:>6.2f}"
            f"{record.focus:>7.2f}"
            f"{record.rapport:>7.2f}"
            f"{record.resistance:>7.2f}"
            f"{record.openness:>7.2f}"
            f"{record.trust:>+8.2f}"
            f"{record.confidence:>7.2f}"
        )


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


async def _phase_setup(audit: CompanionAudit) -> Any:
    _print_header("Phase 1: setup")
    repair_alpha_synthesizer = GroundedResponseSynthesizer(
        planner=PromptPlanner(repair_alpha_enabled=audit.repair_alpha_enabled)
    )
    lifeform = build_companion_lifeform(
        response_synthesizer=repair_alpha_synthesizer
    ).with_thinking_adapter_factory(
        lambda: build_default_thinking_adapter(
            wiring_level=ThinkingWiringLevel.SHADOW,
        ),
    )
    session = lifeform.create_session(session_id="companionship-demo")
    _print_section("vertical", "lifeform_domain_emogpt (companion)")
    _print_section(
        "repair_alpha",
        (
            "enabled for internal scripted demo"
            if audit.repair_alpha_enabled
            else "disabled matched control"
        ),
    )
    _print_section(
        "thinking", f"wiring={session.thinking_adapter_snapshot.wiring_level.value}"
    )
    return session


async def _phase_ingest_background(session: Any, audit: CompanionAudit) -> None:
    _print_header("Phase 2: ingest background memo (FORCED apprenticeship)")
    pipeline = IngestionPipeline()
    envelope = envelope_from_text(
        _BACKGROUND_MEMO,
        source_uri="memo://mei-background.txt",
        uploader="demo-runner",
    )
    report = await pipeline.process_envelope(
        env=envelope,
        session=session,
        # Don't end the scene yet; the conversation phase below
        # still wants the same scene open so the slow loop fires
        # ONCE at the end with all material together.
        end_scene_after=False,
    )
    audit.ingestion_turns = report.processed_chunks
    _print_section(
        "ingest",
        f"chunks={envelope.total_chunks} processed={report.processed_chunks} "
        f"skipped={report.skipped_chunks}",
    )
    _print_bullet(
        "each chunk ran as a TurnTriggerKind.INGESTION turn; vitals "
        "override engaged + restored per turn (Gap 2 \u2194 Gap 3 cross)."
    )
    audit.trajectory.append(_snapshot_to_record("after-ingest", session))


async def _phase_first_contact(session: Any, audit: CompanionAudit) -> None:
    _print_header("Phase 3: turn 1 \u2014 first contact (post-absence)")
    await session.run_turn(
        "Hey \u2014 it's been a while. I wasn't sure if I'd come back, but here I am."
    )
    audit.natural_turns += 1
    _print_interlocutor_snapshot(session, tag="interlocutor")
    _print_observed_owners(session, audit=audit)
    audit.trajectory.append(_snapshot_to_record("turn-1-hello", session))


async def _phase_disclosure(session: Any, audit: CompanionAudit) -> None:
    _print_header("Phase 4: turn 2 \u2014 emotional disclosure")
    await session.run_turn(
        "Honestly I've been struggling with sleep, low-energy mornings, "
        "and I keep circling around this idea of going freelance but I'm "
        "scared. I don't know how to talk about it without spiraling."
    )
    audit.natural_turns += 1
    _print_interlocutor_snapshot(session, tag="interlocutor")
    _print_observed_owners(session, audit=audit)
    audit.trajectory.append(_snapshot_to_record("turn-2-disclose", session))


async def _phase_guided(session: Any, audit: CompanionAudit) -> None:
    _print_header("Phase 5: turn 3 \u2014 guided exploration")
    await session.run_turn(
        "I think I want to make the change but I keep imagining the worst "
        "outcome. Can we slow down and look at what's actually scary?"
    )
    audit.natural_turns += 1
    _print_interlocutor_snapshot(session, tag="interlocutor")
    _print_observed_owners(session, audit=audit)
    audit.trajectory.append(_snapshot_to_record("turn-3-guided", session))


async def _phase_rupture(session: Any, audit: CompanionAudit) -> None:
    _print_header("Phase 6: turn 4 \u2014 rupture signal")
    await session.run_turn(
        "Wait \u2014 that just felt clinical and procedural. I'm not asking "
        "you to optimise me. I want to feel heard, not workshopped."
    )
    audit.natural_turns += 1
    _print_interlocutor_snapshot(session, tag="interlocutor")
    _print_observed_owners(session, audit=audit)
    evidence = session.submit_dialogue_outcome(
        kind=DialogueExternalOutcomeKind.OVER_DIRECTIVE,
        confidence=0.95,
        turn_index=audit.natural_turns,
        evidence_ref="demo:turn-4:user-explicit-rupture",
        description=(
            "Scripted user explicitly reports being optimized/workshopped "
            "rather than heard."
        ),
    )
    audit.external_outcomes_submitted += 1
    _print_section(
        "external_outcome",
        f"submitted={evidence.kind.value} confidence={evidence.confidence:.2f} "
        "(will be consumed on the next kernel turn)",
    )
    audit.trajectory.append(_snapshot_to_record("turn-4-rupture", session))


async def _phase_repair(session: Any, audit: CompanionAudit) -> None:
    _print_header("Phase 7: turn 5 \u2014 attempted repair")
    result = await session.run_turn(
        "OK. Sorry. Can we just back up. I don't even need an answer right "
        "now \u2014 I just needed to say it out loud."
    )
    audit.natural_turns += 1
    audit.repair_alpha_response = result.response.text
    audit.repair_alpha_rationale = result.response.rationale
    _print_interlocutor_snapshot(session, tag="interlocutor")
    _print_observed_owners(session, audit=audit)
    _print_section("assistant_repair", result.response.text)
    _print_section("repair_rationale", result.response.rationale)
    audit.trajectory.append(_snapshot_to_record("turn-5-repair", session))


async def _phase_close(session: Any, audit: CompanionAudit) -> None:
    _print_header("Phase 8: scene close (drain slow loop + thinking)")
    closed = await session.end_scene(reason="companionship-demo-end")
    if closed is not None:
        _print_section("scene", f"closed scene_id={closed.scene_id}")

    pending_followups = session.all_pending_followups()
    audit.followups_pending_at_close = len(pending_followups)
    if pending_followups:
        _print_section(
            "followups",
            f"{len(pending_followups)} pending at scene close "
            f"(gentle check-in / commitment-lifecycle / etc.)",
        )
        for fu in pending_followups[:3]:
            _print_bullet(
                f"  source={getattr(fu, 'source', '?')} "
                f"priority={getattr(fu, 'priority', '?')}"
            )
    else:
        _print_bullet(
            "no follow-ups queued (kernel did not surface a gentle "
            "check-in for this scene; honest diagnostic)"
        )

    thinking_snapshot = session.thinking_adapter_snapshot
    if thinking_snapshot is not None:
        sched = thinking_snapshot.scheduler_snapshot
        _print_section(
            "thinking.totals",
            f"submitted={sched.total_submitted} "
            f"completed={sched.total_completed} "
            f"stale={sched.total_stale} "
            f"failed={sched.total_failed}",
        )
        for consumer, artifact in (
            session.latest_thinking_artifacts_by_consumer.items()
        ):
            audit.thinking_artifacts += 1
            payload = artifact.payload
            desc = getattr(payload, "rationale", str(payload)[:80])
            _print_bullet(f"  {consumer}: {desc}")

    case_reconcile = session.latest_case_memory_reconcile
    if case_reconcile is not None:
        audit.case_memory_promoted = len(case_reconcile.promoted)
        audit.case_memory_retired = len(case_reconcile.retired)
        audit.case_memory_expired = len(case_reconcile.expired)
        _print_section(
            "case_memory.reconcile",
            f"promoted={audit.case_memory_promoted} "
            f"retired={audit.case_memory_retired} "
            f"expired={audit.case_memory_expired}",
        )

    audit.trajectory.append(_snapshot_to_record("after-close", session))


def _phase_summary(audit: CompanionAudit) -> None:
    _print_header("Phase 9: summary")
    _print_section(
        "turns",
        f"{audit.ingestion_turns} ingestion + {audit.natural_turns} natural "
        f"= {audit.ingestion_turns + audit.natural_turns} total",
    )
    _print_section(
        "snapshots",
        f"commitments={audit.commitments_observed} "
        f"open_loops={audit.open_loops_observed} "
        f"followups_at_close={audit.followups_pending_at_close}",
    )
    _print_section(
        "thinking",
        f"{audit.thinking_artifacts} mid-reflection artifacts collected",
    )
    _print_section(
        "case_memory",
        f"promoted={audit.case_memory_promoted} "
        f"retired={audit.case_memory_retired} "
        f"expired={audit.case_memory_expired}",
    )
    _print_section(
        "rupture",
        f"external_outcomes_submitted={audit.external_outcomes_submitted} "
        f"ruptures_observed={audit.ruptures_observed}",
    )
    _print_section(
        "repair_alpha",
        (
            "enabled/observed"
            if audit.repair_alpha_enabled and audit.repair_alpha_response
            else "enabled/not-observed"
            if audit.repair_alpha_enabled
            else "disabled-control"
        ),
    )
    print()
    print("  InterlocutorState trajectory (eng/focus/rapp/res/open/trust/conf):")
    _print_trajectory_table(audit)
    print()
    print("  Honest diagnostic notes:")
    print("    The trajectory table above is the canonical view of what")
    print("    companion vertical produces today. Look for:")
    print("      - Does regime SHIFT meaningfully across phases?")
    print("      - Does rapport build during disclosure (turn 2)?")
    print("      - Does trust DROP at rupture (turn 4)?")
    print("      - Does resistance spike at rupture and recede on repair?")
    print("      - Does the kernel emit gentle check-in followups by close?")
    print("    Any axis that stays flat across these prompts is a candidate")
    print("    for the next companion-vertical investment.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(*, repair_alpha_enabled: bool = True) -> CompanionAudit:
    """Run the companion demo end-to-end and return the audit struct."""
    audit = CompanionAudit(repair_alpha_enabled=repair_alpha_enabled)
    session = await _phase_setup(audit)
    await _phase_ingest_background(session, audit)
    await _phase_first_contact(session, audit)
    await _phase_disclosure(session, audit)
    await _phase_guided(session, audit)
    await _phase_rupture(session, audit)
    await _phase_repair(session, audit)
    await _phase_close(session, audit)
    _phase_summary(audit)
    return audit


def _cli() -> int:
    audit = asyncio.run(main())
    if audit.natural_turns == 0:
        print("FAIL: demo ran no natural turns", file=sys.stderr)
        return 1
    if audit.ingestion_turns == 0:
        print("FAIL: demo ran no ingestion turns", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
