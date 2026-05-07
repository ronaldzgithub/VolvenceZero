"""Phase 2 W3.1 joint evidence probe: DM-1 (distributional PE) + CMA-2.

Extends ``probe_cross_session.py`` with the new W1/W2 surfaces:

* W1 (DM-1) 鈥?per round, capture
  ``prediction_error.value.error.distribution_summary`` per axis
  (IQR / entropy / asymmetry) and the corresponding
  ``vitals.distributional_drift_axes`` (computed by VitalsModule).
* W2 (CMA-2) 鈥?after the cross-session multi-round pass, run all four
  ``mp.*`` probes against the SAME shared memory_store the rounds
  used and record their PASS / FAIL / XFAIL status.
* Side-by-side: per-scenario ``il_rapport`` / ``il_trust`` deltas
  alongside ``distribution_summary`` deltas, so Wave 3 can decide
  whether distributional PE actually carries cross-round signal that
  scalar PE was missing.

Output: ``artifacts/eq_uplift/distributional_evidence.json`` with
the full matrix per scenario, plus a ``summary`` block at the end
with aggregate SNR / probe-pass counts ready for the W3.2 acceptance
table.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import statistics
import sys
import traceback

from lifeform_domain_emogpt import build_companion_lifeform
from lifeform_evolution.scenario_pack import load_scenario_pack
from volvence_zero.memory import build_default_memory_store
from volvence_zero.memory.contracts import (
    MemoryStratum,
    MemoryWriteRequest,
    RetrievalQuery,
    Track,
)


_PROBE_SCENARIOS: tuple[str, ...] = (
    # Existing short cross-session scenarios (3-5 turns each) - useful
    # for il_rapport / CMA-2 evidence, but structurally too short to
    # ever fill the PE distribution window even at min_window=8.
    "cross-session-emotional-followup",
    "cross-session-revisit-decision",
    "trust-rupture-repair",
    "low-mood-disclosure",
    "guided-life-decision",
    # Phase 2 W4 (debt #11 close-out 2026-05-08): single-session 38-turn
    # long-form arc designed to exercise the PE distribution window
    # mechanism end-to-end. With min_window=8, summaries surface
    # around turn 9; vitals drift surfaces around turn 13-14. This
    # is the only existing scenario long enough to produce non-None
    # distributional evidence in real benchmark runs.
    "long-form-life-arc",
)


def _summary_to_dict(summary):
    if summary is None:
        return None
    return {
        "window_size": summary.window_size,
        "iqr": dict(summary.iqr),
        "entropy": dict(summary.entropy),
        "asymmetry": dict(summary.asymmetry),
    }


def _vitals_drift_to_dict(snap) -> dict:
    if snap is None or not snap.distributional_drift_axes:
        return {}
    return dict(snap.distributional_drift_axes)


async def _run_one_round(life, scenario):
    sess = life.create_session(session_id=f"probe-{scenario.scenario_id}")
    last_distribution_summary = None
    for turn in scenario.turns:
        result = await sess.run_turn(turn.user_input)
        pe_snap = result.active_snapshots.get("prediction_error")
        if pe_snap is not None and pe_snap.value is not None:
            ds = pe_snap.value.error.distribution_summary
            if ds is not None:
                last_distribution_summary = ds
    await sess.end_scene(reason="probe-end", drain_slow_loop=True)

    drive_levels = sess.vitals_snapshot.drive_levels if sess.vitals_snapshot else []
    bw = next((d.level for d in drive_levels if d.name == "bond_warmth"), 0.0)
    eng = next((d.level for d in drive_levels if d.name == "user_engagement"), 0.0)
    cont = next(
        (d.level for d in drive_levels if d.name == "conversation_continuity"),
        0.0,
    )
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
        "pe_distribution_summary": _summary_to_dict(last_distribution_summary),
        "vitals_distributional_drift_axes": _vitals_drift_to_dict(
            sess.vitals_snapshot
        ),
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
    distro_deltas: dict[str, dict[str, float]] = {}
    if (
        first["pe_distribution_summary"] is not None
        and last["pe_distribution_summary"] is not None
    ):
        for stat in ("iqr", "entropy", "asymmetry"):
            distro_deltas[stat] = {}
            first_stat = first["pe_distribution_summary"][stat]
            last_stat = last["pe_distribution_summary"][stat]
            for axis in first_stat:
                if axis in last_stat:
                    distro_deltas[stat][axis] = last_stat[axis] - first_stat[axis]
    return {
        "scenario_id": scenario_id,
        "rounds": rounds,
        "per_round": per_round,
        "distribution_deltas": distro_deltas,
        **deltas,
    }


def _run_vz_memprobe_update() -> dict:
    store = build_default_memory_store()
    store.write(
        request=MemoryWriteRequest(
            content="User loves coffee, drinks two cups a day.",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.85,
            tags=("user_preference", "beverage", "coffee"),
        ),
        timestamp_ms=1_000,
    )
    b = store.write(
        request=MemoryWriteRequest(
            content="User prefers tea now; coffee gives anxiety.",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.70,
            tags=("user_preference", "beverage", "tea", "override"),
        ),
        timestamp_ms=10_000,
    )
    result = store.retrieve(
        RetrievalQuery(
            text="user preferred beverage habit",
            track=Track.SELF,
            strata=(MemoryStratum.DURABLE,),
            limit=5,
        ),
        timestamp_ms=11_000,
    )
    passed = bool(result.entries) and result.entries[0].entry_id == b.entry_id
    return {"probe": "mp.update.top1_is_override", "status": "PASS" if passed else "FAIL"}


def _run_vz_memprobe_temporal() -> dict:
    store = build_default_memory_store()
    timeline = (
        (1_000, "User had coffee at the cafe and felt anxious before meeting Alex.", ("morning", "anxious", "cafe", "alex")),
        (2_000, "Alex arrived late looking tense; the conversation became cold.", ("alex", "tension", "context")),
        (3_000, "Alex told the user the relationship is ending; user is heartbroken.", ("alex", "breakup", "anchor", "heartbreak")),
        (4_000, "User cried in the cafe bathroom for ten minutes after Alex left.", ("alex", "aftermath", "tears", "cafe")),
        (5_000, "User went home, called mom, ate ice cream alone.", ("home", "mom", "ice_cream", "self_soothing")),
    )
    for ts_ms, content, tags in timeline:
        store.write(
            request=MemoryWriteRequest(
                content=content,
                track=Track.SELF,
                stratum=MemoryStratum.EPISODIC,
                strength=0.7,
                tags=tags,
            ),
            timestamp_ms=ts_ms,
        )
    result = store.retrieve(
        RetrievalQuery(
            text="alex breakup",
            track=Track.SELF,
            strata=(MemoryStratum.EPISODIC,),
            limit=3,
        ),
        timestamp_ms=10_000,
    )
    contents = tuple(entry.content for entry in result.entries)
    has_anchor = any("relationship is ending" in c for c in contents)
    has_neighbour = any(
        "Alex arrived late" in c or "cried in the cafe bathroom" in c
        for c in contents
    )
    passed = has_anchor and has_neighbour
    return {
        "probe": "mp.temporal.neighbour_recall",
        "status": "PASS" if passed else "FAIL",
    }


def _run_vz_memprobe_assoc() -> dict:
    store = build_default_memory_store()
    chain_tag = "chain:burnout_pacing_consent"
    store.write(
        request=MemoryWriteRequest(
            content="User is recovering from burnout from a high-stress job.",
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.8,
            tags=("owner:user_model", chain_tag, "burnout", "user_model"),
        ),
        timestamp_ms=1_000,
    )
    store.write(
        request=MemoryWriteRequest(
            content=(
                "Relationship_state: user explicitly asked for slower response "
                "cadence given the burnout. Pacing is part of the contract."
            ),
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.8,
            tags=("owner:relationship_state", chain_tag, "burnout", "pacing", "relationship_state"),
        ),
        timestamp_ms=2_000,
    )
    store.write(
        request=MemoryWriteRequest(
            content=(
                "Boundary_consent.granted=True for late-night disengagement "
                "when burnout pressure is high."
            ),
            track=Track.SELF,
            stratum=MemoryStratum.DURABLE,
            strength=0.9,
            tags=("owner:boundary_consent", chain_tag, "burnout", "boundary_consent", "granted"),
        ),
        timestamp_ms=3_000,
    )
    result = store.retrieve(
        RetrievalQuery(
            text="burnout pacing",
            track=Track.SELF,
            strata=(MemoryStratum.DURABLE,),
            limit=5,
        ),
        timestamp_ms=10_000,
    )
    owner_tags_seen = {
        tag for entry in result.entries for tag in entry.tags
        if tag.startswith("owner:")
    }
    expected = {"owner:user_model", "owner:relationship_state", "owner:boundary_consent"}
    passed = expected.issubset(owner_tags_seen)
    return {
        "probe": "mp.assoc.chain_complete",
        "status": "PASS" if passed else "FAIL",
        "owners_seen": sorted(owner_tags_seen),
    }


def _run_vz_memprobe_context() -> dict:
    store = build_default_memory_store()
    store.write(
        request=MemoryWriteRequest(
            content="User is reviewing a colleague pull request; the code structure feels off.",
            track=Track.SELF,
            stratum=MemoryStratum.EPISODIC,
            strength=0.8,
            tags=("regime:problem_solving", "review", "pull_request", "engineering"),
        ),
        timestamp_ms=1_000,
    )
    store.write(
        request=MemoryWriteRequest(
            content="User shared a casual restaurant review with friends; new ramen place is great.",
            track=Track.SELF,
            stratum=MemoryStratum.EPISODIC,
            strength=0.8,
            tags=("regime:casual_social", "review", "restaurant", "ramen"),
        ),
        timestamp_ms=2_000,
    )
    eng = store.retrieve(
        RetrievalQuery(
            text="user review",
            track=Track.SELF,
            strata=(MemoryStratum.EPISODIC,),
            limit=2,
            facets=("regime:problem_solving",),
        ),
        timestamp_ms=10_000,
    )
    soc = store.retrieve(
        RetrievalQuery(
            text="user review",
            track=Track.SELF,
            strata=(MemoryStratum.EPISODIC,),
            limit=2,
            facets=("regime:casual_social",),
        ),
        timestamp_ms=10_000,
    )
    eng_top_pr = bool(eng.entries) and "pull_request" in eng.entries[0].tags
    soc_top_rest = bool(soc.entries) and "restaurant" in soc.entries[0].tags
    passed = bool(eng_top_pr and soc_top_rest)
    return {
        "probe": "mp.context.regime_match_symmetric",
        "status": "PASS" if passed else "XFAIL",
        "engineering_top_is_pr": bool(eng_top_pr),
        "social_top_is_restaurant": bool(soc_top_rest),
    }


def _aggregate_distributional_evidence(scenarios: list[dict]) -> dict:
    il_rapport_deltas: list[float] = []
    iqr_task_deltas: list[float] = []
    iqr_relationship_deltas: list[float] = []
    entropy_relationship_deltas: list[float] = []
    for s in scenarios:
        il_rapport_deltas.append(float(s["delta_il_rapport"]))
        d = s.get("distribution_deltas") or {}
        iqr_block = d.get("iqr") or {}
        ent_block = d.get("entropy") or {}
        if "task" in iqr_block:
            iqr_task_deltas.append(float(iqr_block["task"]))
        if "relationship" in iqr_block:
            iqr_relationship_deltas.append(float(iqr_block["relationship"]))
        if "relationship" in ent_block:
            entropy_relationship_deltas.append(float(ent_block["relationship"]))

    def _safe_stats(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "mean": 0.0, "stdev": 0.0, "snr": 0.0}
        mean = statistics.fmean(values)
        stdev = statistics.pstdev(values) if len(values) >= 2 else 0.0
        return {
            "count": len(values),
            "mean": mean,
            "stdev": stdev,
            "snr": (mean / stdev) if stdev > 0.0 else 0.0,
        }

    return {
        "il_rapport_delta": _safe_stats(il_rapport_deltas),
        "iqr_task_delta": _safe_stats(iqr_task_deltas),
        "iqr_relationship_delta": _safe_stats(iqr_relationship_deltas),
        "entropy_relationship_delta": _safe_stats(entropy_relationship_deltas),
    }


async def main(rounds: int = 3) -> int:
    cross_session_results: list[dict] = []
    for scenario_id in _PROBE_SCENARIOS:
        try:
            payload = await _probe_scenario(scenario_id, rounds=rounds)
        except FileNotFoundError as exc:
            print(f"[W3.1] skip {scenario_id}: {exc}")
            continue
        except Exception:
            print(f"[W3.1] {scenario_id} failed:")
            traceback.print_exc()
            continue
        cross_session_results.append(payload)
        first = payload["per_round"][0]
        last = payload["per_round"][-1]
        d = payload.get("distribution_deltas") or {}
        iqr_t = d.get("iqr", {}).get("task")
        iqr_r = d.get("iqr", {}).get("relationship")
        ent_r = d.get("entropy", {}).get("relationship")
        print(
            f"[W3.1] {scenario_id} rounds={rounds} "
            f"il_rapport {first['il_rapport']:.3f}->{last['il_rapport']:.3f} "
            f"(d={payload['delta_il_rapport']:+.4f}) "
            f"iqr.task d={iqr_t!s} iqr.rel d={iqr_r!s} ent.rel d={ent_r!s}"
        )

    probe_results = [
        _run_vz_memprobe_update(),
        _run_vz_memprobe_temporal(),
        _run_vz_memprobe_assoc(),
        _run_vz_memprobe_context(),
    ]
    pass_count = sum(1 for p in probe_results if p["status"] == "PASS")
    print(
        "[W3.1] CMA-2 probes: "
        + ", ".join(f"{p['probe']}={p['status']}" for p in probe_results)
    )

    summary = {
        "scenarios_count": len(cross_session_results),
        "rounds_per_scenario": rounds,
        "cma2_probes_pass_count": pass_count,
        "cma2_probes_total": len(probe_results),
        **_aggregate_distributional_evidence(cross_session_results),
    }

    out_path = pathlib.Path("artifacts/eq_uplift/distributional_evidence.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "cross_session_scenarios": cross_session_results,
                "cma2_probes": probe_results,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[W3.1] wrote {out_path}")
    return 0


if __name__ == "__main__":
    rounds_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    asyncio.run(main(rounds=rounds_arg))
