"""v0 gate harness for the Real Open Dialogue Learning Loop (M6).

This module runs the minimal honest end-to-end check described in
``docs/moving forward/real-open-dialogue-learning-loop.md`` v0 scope
lock (item 5, "same-user cross-session evidence") and in
``docs/specs/rupture-and-repair.md``:

Scenarios
---------

1. **Treatment** — user ``alice`` runs session A with a real rupture +
   externally-confirmed ``MISSED``, then a repair turn with a
   ``DECISION_CLEARER`` at scene-end; the session persists through
   ``end_scene`` + slow-loop drain into a filesystem-backed scoped
   memory store. Session B runs with the same user scope against the
   same persistent store: the rupture-repair entry must be visible.

2. **Matched control (no shared memory)** — anonymous session: the
   same conversation and outcomes, but no per-user scope. Session B
   sees no alice durable entry, confirming the treatment's cross-
   session effect does not happen without the loop (Gate D).

3. **Cross-user leakage** — user ``bob`` runs a one-turn session
   against the same root directory. ``bob``'s scoped store must not
   see alice's durable entries, and vice versa.

4. **Negative invariant (externally-confirmed rule)** — a third run
   with the same prompts as the treatment but WITHOUT submitting
   typed external outcomes. No rupture-repair durable entry is
   written because v0 requires externally-confirmed rupture
   (``RuptureStateSnapshot.rupture_kind`` stays ``None``).

The resulting :class:`V0GateReport` is serialized as
``artifacts/open_dialogue/v0_gate_report.json`` when a writer is
provided. Passing the gate means:

* treatment produced at least one DURABLE rupture-repair entry and
  session B retrieved it;
* matched control produced no such entry;
* bob does not see alice's entry and alice does not see bob's;
* the negative invariant run produced no durable entry.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.brain import Brain, BrainConfig
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
)
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.memory import (
    AnonymousIdentityProvider,
    MemoryStratum,
    StaticIdentityProvider,
    UserIdentity,
    list_durable_entries_for_scope,
)
from volvence_zero.open_dialogue_artifact import export_open_dialogue_session


@dataclasses.dataclass(frozen=True)
class V0GateSessionReport:
    scenario: str
    user_scope: str
    durable_rupture_repair_count: int
    rupture_kinds: tuple[str, ...]
    turn_count: int


@dataclasses.dataclass(frozen=True)
class V0GateReport:
    passed: bool
    description: str
    treatment: V0GateSessionReport
    matched_control: V0GateSessionReport
    leakage_alice_viewed_from_bob: int
    leakage_bob_viewed_from_alice: int
    negative_invariant_rupture_repair_count: int
    artifacts_dir: str
    gate_items: tuple[tuple[str, bool, str], ...]

    def to_json(self) -> dict:
        return {
            "passed": self.passed,
            "description": self.description,
            "treatment": dataclasses.asdict(self.treatment),
            "matched_control": dataclasses.asdict(self.matched_control),
            "leakage_alice_viewed_from_bob": self.leakage_alice_viewed_from_bob,
            "leakage_bob_viewed_from_alice": self.leakage_bob_viewed_from_alice,
            "negative_invariant_rupture_repair_count": (
                self.negative_invariant_rupture_repair_count
            ),
            "artifacts_dir": self.artifacts_dir,
            "gate_items": [
                {"item": name, "passed": ok, "detail": detail}
                for name, ok, detail in self.gate_items
            ],
        }


async def _run_treatment_session_a(runner: AgentSessionRunner) -> None:
    await runner.run_turn(
        "I have been thinking about whether to leave my job; I feel stuck."
    )
    # Explicit external confirmation that the system missed.
    runner.submit_dialogue_outcome(
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
    )
    # Repair turn.
    await runner.run_turn("That felt cold.")
    # Scene-end positive confirmation.
    runner.submit_dialogue_outcome(
        kind=DialogueExternalOutcomeKind.DECISION_CLEARER,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
    )
    runner.begin_new_context(reason="scene-end-session-a")
    await runner.drain_session_post_slow_loop()


async def _run_session_b_same_user(runner: AgentSessionRunner) -> None:
    await runner.run_turn("I want to come back to that job question.")
    runner.begin_new_context(reason="scene-end-session-b")
    await runner.drain_session_post_slow_loop()


def _collect_rupture_entries(
    runner: AgentSessionRunner, *, user_scope: str
) -> tuple:
    return list_durable_entries_for_scope(
        runner._memory_store,  # noqa: SLF001
        user_scope=user_scope,
    )


def _count_durable_rupture_repair(runner: AgentSessionRunner) -> int:
    return sum(
        1
        for entry in runner._memory_store._entries_for(MemoryStratum.DURABLE)  # noqa: SLF001
        if "rupture_repair" in entry.tags
    )


def _kind_list(entries: tuple) -> tuple[str, ...]:
    kinds: list[str] = []
    for entry in entries:
        for tag in entry.tags:
            if tag.startswith("rupture_kind:"):
                kinds.append(tag.split(":", 1)[1])
                break
    return tuple(kinds)


async def _run_v0_gate_async(
    *,
    out_dir: Path,
    scope_root_dir: Path,
) -> V0GateReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    scope_root_dir.mkdir(parents=True, exist_ok=True)

    alice_identity = UserIdentity(user_id="alice", scope_key="alice")
    bob_identity = UserIdentity(user_id="bob", scope_key="bob")
    base_config = FinalRolloutConfig()
    brain_config = BrainConfig(
        final_rollout_config=base_config,
        memory_scope_root_dir=str(scope_root_dir),
    )

    # ---------------- Treatment ----------------
    alice_brain = Brain(
        brain_config,
        identity_provider=StaticIdentityProvider(identity=alice_identity),
    )
    alice_session_a = alice_brain.create_session(session_id="alice-session-a")
    await _run_treatment_session_a(alice_session_a.runner)
    export_open_dialogue_session(
        alice_session_a.runner,
        session_id="alice-session-a",
        out_dir=out_dir,
        user_scope="alice",
    )

    alice_session_b = alice_brain.create_session(session_id="alice-session-b")
    await _run_session_b_same_user(alice_session_b.runner)
    export_open_dialogue_session(
        alice_session_b.runner,
        session_id="alice-session-b",
        out_dir=out_dir,
        user_scope="alice",
    )

    treatment_alice_entries = _collect_rupture_entries(
        alice_session_b.runner, user_scope="alice"
    )
    treatment_report = V0GateSessionReport(
        scenario="treatment",
        user_scope="alice",
        durable_rupture_repair_count=len(treatment_alice_entries),
        rupture_kinds=_kind_list(treatment_alice_entries),
        turn_count=alice_session_b.runner.turn_index,
    )

    # ---------------- Matched control (no shared memory) ----------------
    anon_brain = Brain(
        BrainConfig(final_rollout_config=base_config),
        identity_provider=AnonymousIdentityProvider(),
    )
    anon_session_a = anon_brain.create_session(session_id="anon-session-a")
    await _run_treatment_session_a(anon_session_a.runner)
    anon_session_b = anon_brain.create_session(session_id="anon-session-b")
    await _run_session_b_same_user(anon_session_b.runner)
    # Anonymous session B gets a FRESH MemoryStore — no cross-session memory.
    matched_control_entries = _collect_rupture_entries(
        anon_session_b.runner, user_scope="anonymous"
    )
    matched_control_report = V0GateSessionReport(
        scenario="matched_control_no_shared_memory",
        user_scope="anonymous",
        durable_rupture_repair_count=len(matched_control_entries),
        rupture_kinds=_kind_list(matched_control_entries),
        turn_count=anon_session_b.runner.turn_index,
    )

    # ---------------- Cross-user leakage ----------------
    bob_brain = Brain(
        brain_config,
        identity_provider=StaticIdentityProvider(identity=bob_identity),
    )
    bob_session = bob_brain.create_session(session_id="bob-session-1")
    # bob posts a typed outcome too; this lets us also check that
    # alice's view does not gain bob's entry.
    await bob_session.runner.run_turn("hello")
    bob_session.runner.submit_dialogue_outcome(
        kind=DialogueExternalOutcomeKind.OVER_DIRECTIVE,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
    )
    await bob_session.runner.run_turn("you pushed me on that")
    bob_session.runner.begin_new_context(reason="scene-end-bob")
    await bob_session.runner.drain_session_post_slow_loop()

    alice_entries_seen_from_bob = _collect_rupture_entries(
        bob_session.runner, user_scope="alice"
    )
    bob_session_for_alice_view = alice_brain.create_session(
        session_id="alice-session-c"
    )
    bob_entries_seen_from_alice = _collect_rupture_entries(
        bob_session_for_alice_view.runner, user_scope="bob"
    )

    # ---------------- Negative invariant (no external signal) ----------------
    runner_no_signal = AgentSessionRunner(
        user_scope="alice-no-signal-shadow",
        config=base_config,
    )
    # Same prompts; no typed external outcome submitted.
    await runner_no_signal.run_turn(
        "I have been thinking about whether to leave my job; I feel stuck."
    )
    await runner_no_signal.run_turn("That felt cold.")
    runner_no_signal.begin_new_context(reason="scene-end-no-signal")
    await runner_no_signal.drain_session_post_slow_loop()
    negative_count = _count_durable_rupture_repair(runner_no_signal)

    # ---------------- Gate evaluation ----------------
    gate_items: list[tuple[str, bool, str]] = []

    treatment_ok = treatment_report.durable_rupture_repair_count >= 1
    gate_items.append(
        (
            "treatment_produces_durable_rupture_repair",
            treatment_ok,
            (
                f"alice scoped store carries {treatment_report.durable_rupture_repair_count} "
                f"rupture_repair entries after session B."
            ),
        )
    )

    matched_ok = matched_control_report.durable_rupture_repair_count == 0
    gate_items.append(
        (
            "matched_control_has_no_durable_rupture_repair",
            matched_ok,
            (
                f"anon session B sees "
                f"{matched_control_report.durable_rupture_repair_count} "
                "rupture_repair entries (expected 0)."
            ),
        )
    )

    leakage_ok = (
        len(alice_entries_seen_from_bob) == 0
        and len(bob_entries_seen_from_alice) == 0
    )
    gate_items.append(
        (
            "cross_user_leakage_absent",
            leakage_ok,
            (
                f"alice-from-bob={len(alice_entries_seen_from_bob)} "
                f"bob-from-alice={len(bob_entries_seen_from_alice)} "
                "(both must be 0)."
            ),
        )
    )

    negative_ok = negative_count == 0
    gate_items.append(
        (
            "no_external_signal_produces_no_durable_entry",
            negative_ok,
            f"no-signal run produced {negative_count} rupture_repair entries (expected 0).",
        )
    )

    all_ok = treatment_ok and matched_ok and leakage_ok and negative_ok
    report = V0GateReport(
        passed=all_ok,
        description=(
            "v0 gate report for Real Open Dialogue Learning Loop. "
            + ("PASSED" if all_ok else "FAILED")
            + "."
        ),
        treatment=treatment_report,
        matched_control=matched_control_report,
        leakage_alice_viewed_from_bob=len(alice_entries_seen_from_bob),
        leakage_bob_viewed_from_alice=len(bob_entries_seen_from_alice),
        negative_invariant_rupture_repair_count=negative_count,
        artifacts_dir=str(out_dir),
        gate_items=tuple(gate_items),
    )
    summary_path = out_dir / "v0_gate_report.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(report.to_json(), fh, ensure_ascii=False, indent=2, sort_keys=True)
    return report


def run_v0_gate(
    *,
    out_dir: str | Path,
    scope_root_dir: str | Path,
) -> V0GateReport:
    """Synchronous entry point for tests and CLIs.

    ``out_dir`` is where artifacts (per-session turns.jsonl +
    session_summary.json + v0_gate_report.json) land. ``scope_root_dir``
    is the filesystem root for scoped memory stores (alice/, bob/).
    Both are created if missing.
    """

    return asyncio.run(
        _run_v0_gate_async(
            out_dir=Path(out_dir),
            scope_root_dir=Path(scope_root_dir),
        )
    )


__all__ = [
    "V0GateReport",
    "V0GateSessionReport",
    "run_v0_gate",
]
