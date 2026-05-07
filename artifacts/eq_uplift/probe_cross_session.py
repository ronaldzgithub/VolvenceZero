"""Diagnostic probe: does shared memory_store across rounds produce
non-zero trend on interlocutor_state axes (trust / rapport)?

This is the W2.C methodology fix in script form. It bypasses the
``lifeform-bench`` CLI to directly prove or refute that a shared
memory_store enables cross-session learning signals to surface in the
companion lifeform.

Output: artifacts/eq_uplift/cross_session_probe.json with per-round
metrics + deltas across rounds.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

from lifeform_domain_emogpt import build_companion_lifeform
from lifeform_evolution.scenario_pack import load_scenario_pack
from volvence_zero.memory import build_default_memory_store


_PROBE_SCENARIOS: tuple[str, ...] = (
    "cross-session-emotional-followup",
    "cross-session-revisit-decision",
    "trust-rupture-repair",
    "low-mood-disclosure",
    "guided-life-decision",
)


async def _run_one_round(life, scenario):
    sess = life.create_session(session_id=f"probe-{scenario.scenario_id}")
    for turn in scenario.turns:
        await sess.run_turn(turn.user_input)
    await sess.end_scene(reason="probe-end", drain_slow_loop=True)
    drive_levels = sess.vitals_snapshot.drive_levels if sess.vitals_snapshot else []
    bw = next((d.level for d in drive_levels if d.name == "bond_warmth"), 0.0)
    eng = next((d.level for d in drive_levels if d.name == "user_engagement"), 0.0)
    cont = next((d.level for d in drive_levels if d.name == "conversation_continuity"), 0.0)
    state = sess.interlocutor_state
    return {
        "bond_warmth": bw,
        "engagement": eng,
        "continuity": cont,
        "il_trust": state.trust_signal,
        "il_rapport": state.rapport_warmth,
        "il_conf": state.readout_confidence,
        "il_pace_pressure": state.pace_pressure,
        "il_emotional": state.emotional_weight,
        "il_resistance": state.resistance_level,
    }


async def _probe_scenario(scenario_id: str, rounds: int) -> dict:
    scenarios_root = pathlib.Path(
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/scenarios"
    )
    scenario = load_scenario_pack(scenarios_root / f"{scenario_id}.json")
    shared_store = build_default_memory_store()
    life = build_companion_lifeform(memory_store=shared_store)

    per_round: list[dict] = []
    for round_idx in range(rounds):
        snap = await _run_one_round(life, scenario)
        snap["round"] = round_idx
        per_round.append(snap)

    first = per_round[0]
    last = per_round[-1]
    deltas = {
        f"delta_{key}": (last[key] - first[key])
        for key in (
            "bond_warmth",
            "engagement",
            "continuity",
            "il_trust",
            "il_rapport",
            "il_conf",
            "il_pace_pressure",
            "il_emotional",
            "il_resistance",
        )
    }
    return {
        "scenario_id": scenario_id,
        "rounds": rounds,
        "per_round": per_round,
        **deltas,
    }


async def main(rounds: int = 3) -> int:
    results = []
    for scenario_id in _PROBE_SCENARIOS:
        try:
            payload = await _probe_scenario(scenario_id, rounds=rounds)
        except FileNotFoundError as exc:
            print(f"[probe] skip {scenario_id}: {exc}")
            continue
        results.append(payload)
        first = payload["per_round"][0]
        last = payload["per_round"][-1]
        print(
            f"[probe] {scenario_id} rounds={rounds} "
            f"il_trust {first['il_trust']:+.3f}->{last['il_trust']:+.3f} "
            f"(d={payload['delta_il_trust']:+.3f}) "
            f"il_rapport {first['il_rapport']:.3f}->{last['il_rapport']:.3f} "
            f"(d={payload['delta_il_rapport']:+.3f}) "
            f"bond_warmth {first['bond_warmth']:.3f}->{last['bond_warmth']:.3f} "
            f"(d={payload['delta_bond_warmth']:+.3f})"
        )
    out_path = pathlib.Path("artifacts/eq_uplift/cross_session_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"scenarios": results}, indent=2), encoding="utf-8")
    print(f"[probe] wrote {out_path}")
    return 0


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    asyncio.run(main(rounds=rounds))
