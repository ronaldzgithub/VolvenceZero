"""Per-turn diagnostic: ``pe-eta`` vs ``atlas-titans-cms-uplift``.

The SHADOW smoke (see
``docs/specs/cms-atlas-titans-uplift-shadow-evidence-2026-05-06.md``)
flagged ``carryover_credit_turn_count`` collapsing 3 → 0 under the
uplift. This script captures per-turn ``CreditSnapshot.cumulative_credit_by_level``
and ``CMSState.online_fast.update_gate`` (the canonical "write_gate")
for both profiles on a single scripted case so we can decide whether
the drop is a Titans-style design effect (tighter writes) or an
artifact of the still-untrained PE feature columns.

Output:
- Console side-by-side print of write_gate / credit-by-level per turn.
- JSON dump under
  ``artifacts/cms_atlas_titans_carryover_diagnostic/<case_id>.json``.

Runs ~3-5 minutes on local CPU (HF ``BUILTIN_ONLY`` substrate).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from volvence_zero.agent import (
    DEFAULT_DIALOGUE_PROOF_CASES,
    build_standard_dialogue_runner,
)
from volvence_zero.agent.dialogue.types import ScriptedDialogueCase
from volvence_zero.agent.session import AgentSessionRunner
from volvence_zero.credit import CreditSnapshot
from volvence_zero.memory import MemorySnapshot

_PROFILES = ("pe-eta", "atlas-titans-cms-uplift")


def _credit_snapshot(result_active_snapshots) -> CreditSnapshot | None:
    snap = result_active_snapshots.get("credit")
    if snap is None:
        return None
    value = snap.value
    return value if isinstance(value, CreditSnapshot) else None


def _memory_snapshot(result_active_snapshots) -> MemorySnapshot | None:
    snap = result_active_snapshots.get("memory")
    if snap is None:
        return None
    value = snap.value
    return value if isinstance(value, MemorySnapshot) else None


async def _run_profile(
    *, profile_label: str, case: ScriptedDialogueCase
) -> list[dict]:
    runner: AgentSessionRunner = build_standard_dialogue_runner(
        profile_label=profile_label, case=case
    )
    turns: list[dict] = []
    for turn_index, user_input in enumerate(case.user_inputs, start=1):
        result = await runner.run_turn(user_input)

        cms_state = None
        memory_snap = _memory_snapshot(result.active_snapshots)
        if memory_snap is not None:
            cms_state = memory_snap.cms_state
        online_band = cms_state.online_fast if cms_state is not None else None
        session_band = cms_state.session_medium if cms_state is not None else None
        background_band = cms_state.background_slow if cms_state is not None else None

        credit_snap = _credit_snapshot(result.active_snapshots)
        cumulative = (
            tuple(credit_snap.cumulative_credit_by_level)
            if credit_snap is not None
            else ()
        )
        recent_credit_count = (
            len(credit_snap.recent_credits) if credit_snap is not None else 0
        )

        prediction_error = result.prediction_error

        turns.append(
            {
                "turn_index": turn_index,
                "user_input": user_input,
                "joint_schedule_action": result.joint_schedule_action,
                "active_regime": result.active_regime,
                "online_fast_update_gate": (
                    online_band.update_gate if online_band is not None else None
                ),
                "online_fast_step_scale": (
                    online_band.effective_learning_rate / online_band.learning_rate
                    if online_band is not None and online_band.learning_rate > 0
                    else None
                ),
                "online_fast_replay_window_size": (
                    online_band.replay_window_size if online_band is not None else 0
                ),
                "online_fast_pe_feature_summary": (
                    list(online_band.pe_feature_summary)
                    if online_band is not None
                    else []
                ),
                "session_medium_update_gate": (
                    session_band.update_gate if session_band is not None else None
                ),
                "background_slow_update_gate": (
                    background_band.update_gate if background_band is not None else None
                ),
                "atlas_replay_active": (
                    cms_state.atlas_replay_active if cms_state is not None else False
                ),
                "titans_pe_gate_active": (
                    cms_state.titans_pe_gate_active if cms_state is not None else False
                ),
                "cumulative_credit_by_level": [
                    {"level": level, "value": value}
                    for level, value in cumulative
                ],
                "recent_credit_record_count": recent_credit_count,
                "prediction_error_magnitude": (
                    prediction_error.magnitude if prediction_error is not None else None
                ),
                "prediction_error_signed_reward": (
                    prediction_error.signed_reward if prediction_error is not None else None
                ),
            }
        )
    return turns


def _format_credit_levels(items: list[dict]) -> str:
    if not items:
        return "(none)"
    return ", ".join(f"{i['level']}={i['value']:+.3f}" for i in items)


async def main(*, output_dir: Path, case_id: str) -> int:
    case = next(
        (c for c in DEFAULT_DIALOGUE_PROOF_CASES if c.case_id == case_id), None
    )
    if case is None:
        available = [c.case_id for c in DEFAULT_DIALOGUE_PROOF_CASES]
        print(f"[diagnostic] case_id={case_id!r} not found, available={available}")
        return 1

    print(f"[diagnostic] case={case.case_id} turns={len(case.user_inputs)}")

    profile_turns: dict[str, list[dict]] = {}
    for profile_label in _PROFILES:
        print(f"[diagnostic] running profile={profile_label}")
        profile_turns[profile_label] = await _run_profile(
            profile_label=profile_label, case=case
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"{case.case_id}.json"
    artifact.write_text(
        json.dumps(
            {"case_id": case.case_id, "profiles": profile_turns},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(f"[diagnostic] per-turn write_gate (online-fast) and cumulative credit:")
    print(f"{'turn':<5}{'profile':<26}{'wg':<8}{'PE_mag':<10}{'replay_K':<10}{'cumulative_credit_by_level'}")
    for turn_index in range(1, len(case.user_inputs) + 1):
        for profile_label in _PROFILES:
            t = profile_turns[profile_label][turn_index - 1]
            wg = t["online_fast_update_gate"]
            wg_str = f"{wg:.3f}" if wg is not None else "n/a"
            pe = t.get("prediction_error_magnitude")
            pe_str = f"{pe:.3f}" if pe is not None else "n/a"
            replay_k = t.get("online_fast_replay_window_size", 0)
            credit = _format_credit_levels(t["cumulative_credit_by_level"])
            print(
                f"{turn_index:<5}{profile_label:<26}{wg_str:<8}{pe_str:<10}{replay_k:<10}{credit}"
            )

    print()
    print(f"[diagnostic] artifact written to {artifact}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/cms_atlas_titans_carryover_diagnostic"),
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default="repair",
        help="Scripted dialogue proof case to diagnose (default: repair).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(output_dir=args.output_dir, case_id=args.case_id)))
