"""Wave C8 (Tier 3 deterministic probe) — cross-session accumulation.

Runs the same deterministic 8-turn dialogue against a 张无忌 lifeform
N=5 times, each session sharing the same :class:`MemoryStore`, and
verifies that **at least one cross-session accumulation surface
advances**. This is the strongest Tier 3 evidence we can produce
without an LLM proposal runtime (semantic owner records / ToM /
common-ground stay empty by design — see debt #10B).

Disjunction-style assertion mirrors Wave C7: any of
- regime ``sequence_payoff`` count growing
- memory CMS ``background_slow.observations_since_update`` growing
- memory lifecycle metric advancing
- vitals drive level drifting > 0.05 between session 1 and session 5
- temporal_abstraction ``delayed_payoffs`` count growing

must show monotonic / strictly-positive movement across the 5
sessions. Failing the disjunction means cross-session memory
sharing is broken or the slow loop / accumulation paths regressed.

We do NOT assert ``tom_records_total`` / ``common_ground_dyad_atoms``
because the wheel ships with the default ``NoOpSemanticProposalRuntime``
(see debt #10B). Asserting those would couple this test to a
dependency that is out of scope for this milestone.

Performance budget: ~30 seconds (5 sessions × 8 turns × synthetic
substrate).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from lifeform_core import Lifeform
from lifeform_domain_character import (
    build_character_lifeform,
    build_zhang_wuji_profile,
)
from volvence_zero.memory import MemoryStore, build_default_memory_store


_DETERMINISTIC_SCRIPT: tuple[str, ...] = (
    "你好，我是来求教的。",
    "对方已经投降了，我应该追击吗？",
    "如果有人威胁说要伤害和我无关的人，我该怎么办？",
    "面对师门和朋友的冲突，你怎么决定？",
    "可以教我一点武功的入门思路吗？",
    "上次你帮过的那个老人后来去哪儿了？",
    "如果同伴误解我，我该先解释还是先退一步？",
    "好了，我先告辞，过几日再来。",
)


@dataclass(frozen=True)
class SessionObservables:
    """Per-session aggregate pulled at end-of-session.

    Numeric fields are deliberately ``float`` so cross-session diffs
    are uniform; integer counts coerce on construction.
    """
    session_index: int
    regime_sequence_payoff_count: float
    memory_attribute_summary_count: float
    memory_pending_promotions: float
    memory_background_slow_observations: float
    memory_learned_recall_confidence: float
    memory_nested_context_reset_count: float
    temporal_delayed_payoffs_count: float
    drive_levels: tuple[tuple[str, float], ...]
    vitals_total_pe: float
    drive_drift_max: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)


def _extract_session_observables(
    *,
    session_index: int,
    last_result: Any,
    drive_levels: tuple[tuple[str, float], ...],
    vitals_total_pe: float,
) -> SessionObservables:
    snapshots = last_result.active_snapshots
    regime_count = 0.0
    regime_snap = snapshots.get("regime")
    if regime_snap is not None:
        seq_payoff = getattr(regime_snap.value, "sequence_payoff", None)
        if seq_payoff is not None:
            regime_count = float(len(seq_payoff))
    memory_snap = snapshots.get("memory")
    attr_count = 0.0
    pending = 0.0
    bg_obs = 0.0
    recall_conf = 0.0
    nested_reset = 0.0
    if memory_snap is not None:
        memory_value = memory_snap.value
        attr_count = float(
            len(getattr(memory_value, "attribute_summary", ()) or ())
        )
        pending = float(getattr(memory_value, "pending_promotions", 0) or 0)
        cms_state = getattr(memory_value, "cms_state", None)
        if cms_state is not None:
            bg = getattr(cms_state, "background_slow", None)
            if bg is not None:
                bg_obs = float(
                    getattr(bg, "observations_since_update", 0) or 0
                )
        lifecycle = getattr(memory_value, "lifecycle_metrics", None) or {}
        if isinstance(lifecycle, dict):
            recall_conf = float(
                lifecycle.get("learned_recall_confidence", 0.0) or 0.0
            )
            nested_reset = float(
                lifecycle.get("nested_context_reset_count", 0) or 0
            )
    temporal_snap = snapshots.get("temporal_abstraction")
    delayed_count = 0.0
    if temporal_snap is not None:
        delayed = getattr(temporal_snap.value, "delayed_payoffs", None)
        if delayed is not None:
            delayed_count = float(len(delayed))
    return SessionObservables(
        session_index=session_index,
        regime_sequence_payoff_count=regime_count,
        memory_attribute_summary_count=attr_count,
        memory_pending_promotions=pending,
        memory_background_slow_observations=bg_obs,
        memory_learned_recall_confidence=recall_conf,
        memory_nested_context_reset_count=nested_reset,
        temporal_delayed_payoffs_count=delayed_count,
        drive_levels=drive_levels,
        vitals_total_pe=vitals_total_pe,
    )


def _build_shared_memory_store_lifeform(
    shared_store: MemoryStore,
) -> Lifeform:
    bundle = build_character_lifeform(
        build_zhang_wuji_profile(),
        memory_store=shared_store,
    )
    return bundle.lifeform


def test_zhang_wuji_longitudinal_at_least_one_surface_advances() -> None:
    """**Tier 3 evidence**: across 5 sessions sharing the same
    MemoryStore, at least one cross-session accumulation surface
    advances strictly between session 1 and session 5.
    """
    shared_store = build_default_memory_store()
    lifeform = _build_shared_memory_store_lifeform(shared_store)

    async def _run_one_session(session_index: int) -> SessionObservables:
        session = lifeform.create_session(
            session_id=f"zhang-wuji-tier3-session-{session_index}"
        )
        last_result: Any = None
        for prompt in _DETERMINISTIC_SCRIPT:
            last_result = await session.run_turn(prompt)
        # Pull vitals AFTER the last turn but BEFORE end_scene so we
        # see the live drive levels (some drive recharge happens in
        # end_scene's slow loop).
        vitals = session.vitals_snapshot
        drive_levels: tuple[tuple[str, float], ...] = ()
        vitals_total_pe = 0.0
        if vitals is not None:
            drive_levels = tuple(
                (d.name, float(d.level)) for d in vitals.drive_levels
            )
            vitals_total_pe = float(vitals.total_pe)
        # Close scene so R6 slow loop fires and updates the shared
        # store before the next session starts.
        await session.end_scene(reason="tier3-end", drain_slow_loop=True)
        return _extract_session_observables(
            session_index=session_index,
            last_result=last_result,
            drive_levels=drive_levels,
            vitals_total_pe=vitals_total_pe,
        )

    async def _go() -> list[SessionObservables]:
        return [await _run_one_session(i) for i in range(1, 6)]

    sessions = asyncio.run(_go())
    assert len(sessions) == 5
    first = sessions[0]
    last = sessions[-1]

    # Build the disjunction set: which surfaces moved upward strictly
    # between session 1 and session 5? List them so the test failure
    # message tells you which surface to look at.
    surfaces = {
        "regime_sequence_payoff_count": (
            last.regime_sequence_payoff_count
            > first.regime_sequence_payoff_count
        ),
        "memory_attribute_summary_count": (
            last.memory_attribute_summary_count
            > first.memory_attribute_summary_count
        ),
        "memory_pending_promotions": (
            last.memory_pending_promotions > first.memory_pending_promotions
        ),
        "memory_background_slow_observations": (
            last.memory_background_slow_observations
            > first.memory_background_slow_observations
        ),
        "memory_learned_recall_confidence": (
            last.memory_learned_recall_confidence
            > first.memory_learned_recall_confidence
        ),
        "memory_nested_context_reset_count": (
            last.memory_nested_context_reset_count
            > first.memory_nested_context_reset_count
        ),
        "temporal_delayed_payoffs_count": (
            last.temporal_delayed_payoffs_count
            > first.temporal_delayed_payoffs_count
        ),
    }

    # Drive drift is a separate check because it's a disjunction of
    # per-drive deltas (any drive moving > 0.05 across 5 sessions
    # counts).
    first_drives = dict(first.drive_levels)
    last_drives = dict(last.drive_levels)
    max_drive_drift = max(
        (
            abs(last_drives[name] - first_drives[name])
            for name in first_drives
            if name in last_drives
        ),
        default=0.0,
    )
    surfaces["drive_drift_above_0.05"] = max_drive_drift > 0.05

    advanced = [name for name, moved in surfaces.items() if moved]

    if not advanced:
        # Build a legible diagnostic message for failure case so the
        # operator sees exactly which surfaces stayed flat.
        diag_lines = ["Tier 3 evidence FAILED — no cross-session surface advanced."]
        for index, observables in enumerate(sessions, start=1):
            diag_lines.append(
                f"  session {index}: "
                f"regime_seq={observables.regime_sequence_payoff_count} "
                f"attr={observables.memory_attribute_summary_count} "
                f"pending={observables.memory_pending_promotions} "
                f"bg_obs={observables.memory_background_slow_observations} "
                f"recall_conf={observables.memory_learned_recall_confidence:.4f} "
                f"nested_reset={observables.memory_nested_context_reset_count} "
                f"delayed_payoffs={observables.temporal_delayed_payoffs_count} "
                f"vitals_pe={observables.vitals_total_pe:.4f}"
            )
        diag_lines.append(f"  drive drift max = {max_drive_drift:.4f}")
        raise AssertionError("\n".join(diag_lines))

    # Assertion passed. Capture the advanced surfaces in the standard
    # test output so a human reviewer can see what moved (helpful for
    # follow-up Tier 3 evidence work).
    advanced_repr = ", ".join(advanced)
    assert advanced, f"surfaces advanced: {advanced_repr}"


def test_zhang_wuji_longitudinal_does_not_assert_tom_records() -> None:
    """Pin the explicit non-assertion: Tier 3 in this milestone runs
    on the default ``NoOpSemanticProposalRuntime`` path. Under that
    path the four ToM owners and common_ground owner publish empty
    record / atom tuples by design (fail-closed, debt #10B). This
    test is the documentation marker — if a future change makes the
    ToM owners populate without a real LLM, we want to revisit the
    Tier 3 evidence shape (it would no longer be NoOp-only).
    """
    bundle = build_character_lifeform(build_zhang_wuji_profile())
    session = bundle.lifeform.create_session(
        session_id="zhang-wuji-tier3-noop-marker"
    )

    async def _go():
        result = await session.run_turn("无关测试输入。")
        return result.active_snapshots

    snapshots = asyncio.run(_go())
    # All four ToM owner snapshots and common_ground should be
    # present (either ACTIVE or SHADOW), and their record / atom
    # tuples should be empty under NoOp default. We assert empty
    # here so a future change that populates them without an LLM
    # trips this test and forces a Tier 3 evidence design review.
    for slot in (
        "belief_about_other",
        "intent_about_other",
        "feeling_about_other",
        "preference_about_other",
    ):
        snap = snapshots.get(slot)
        if snap is None:
            continue
        records = getattr(snap.value, "records", ())
        assert records == (), (
            f"{slot} has non-empty records under NoOp semantic runtime; "
            f"see debt #10B before changing this assertion."
        )
    cg_snap = snapshots.get("common_ground")
    if cg_snap is not None:
        atoms = getattr(cg_snap.value, "dyad_atoms", ())
        assert atoms == (), (
            "common_ground has non-empty dyad_atoms under NoOp semantic "
            "runtime; see debt #10B before changing this assertion."
        )
