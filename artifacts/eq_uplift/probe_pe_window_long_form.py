"""Phase 2 W4 (debt #11) — long-form mechanism-validation probe.

Single scenario, single session, single round. Captures the kernel's
PE distribution window and vitals distributional drift readouts on
EVERY turn, plus the raw 4-axis PE values so we can post-hoc replay
the same data through ``_PEDistributionWindow`` instances at
different ``min_window`` settings to assess statistical stability.

Output: ``artifacts/eq_uplift/pe_window_long_form.json`` with:
* ``per_turn``: turn-by-turn timeline of PE summary / drift presence
  + raw axis errors
* ``summary``: ``first_summary_turn`` / ``first_drift_turn`` /
  ``final_distribution_summary``
* ``simulated_min_window_curves``: post-hoc IQR / entropy / asymmetry
  per axis at ``min_window`` in {6, 8, 10, 12, 16}, sampled at turn
  25 and at the final turn, plus a stability ratio
  ``iqr_8_over_iqr_16`` per axis to feed Wave B's decision.

Usage: ``python artifacts/eq_uplift/probe_pe_window_long_form.py``
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import traceback

from lifeform_domain_emogpt import build_companion_lifeform
from lifeform_evolution.scenario_pack import load_scenario_pack
from volvence_zero.memory import build_default_memory_store
from volvence_zero.prediction.error import _PEDistributionWindow, PredictionError


_SCENARIO_ID = "long-form-life-arc"
_SCENARIO_PATH = pathlib.Path(
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/scenarios"
) / f"{_SCENARIO_ID}.json"
_OUTPUT_PATH = pathlib.Path("artifacts/eq_uplift/pe_window_long_form.json")
_SIMULATED_MIN_WINDOWS: tuple[int, ...] = (6, 8, 10, 12, 16)
_SIMULATED_SAMPLE_TURNS: tuple[int, ...] = (25,)


def _summary_to_dict(summary):
    if summary is None:
        return None
    return {
        "window_size": summary.window_size,
        "iqr": dict(summary.iqr),
        "entropy": dict(summary.entropy),
        "asymmetry": dict(summary.asymmetry),
    }


def _drift_to_dict(snap) -> dict:
    if snap is None or not snap.distributional_drift_axes:
        return {}
    return dict(snap.distributional_drift_axes)


def _raw_axes_from_pe_value(pe_value) -> dict[str, float]:
    if pe_value is None:
        return {}
    err = pe_value.error
    return {
        "task": float(err.task_error),
        "relationship": float(err.relationship_error),
        "regime": float(err.regime_error),
        "action": float(err.action_error),
        "magnitude": float(err.magnitude),
        "bootstrap": bool(pe_value.bootstrap),
    }


async def _run_long_form() -> dict:
    scenario = load_scenario_pack(_SCENARIO_PATH)
    shared_store = build_default_memory_store()
    life = build_companion_lifeform(memory_store=shared_store)
    sess = life.create_session(session_id=f"probe-{_SCENARIO_ID}")

    per_turn: list[dict] = []
    first_summary_turn: int | None = None
    first_drift_turn: int | None = None
    final_summary = None

    for index, turn in enumerate(scenario.turns):
        result = await sess.run_turn(turn.user_input)
        pe_snap = result.active_snapshots.get("prediction_error")
        pe_value = pe_snap.value if pe_snap is not None else None
        ds = pe_value.error.distribution_summary if pe_value is not None else None
        if ds is not None:
            final_summary = ds
            if first_summary_turn is None:
                first_summary_turn = index + 1

        drift_axes = _drift_to_dict(sess.vitals_snapshot)
        if drift_axes and first_drift_turn is None:
            first_drift_turn = index + 1

        per_turn.append(
            {
                "turn": index + 1,
                "raw": _raw_axes_from_pe_value(pe_value),
                "pe_summary_present": ds is not None,
                "pe_summary_window_size": ds.window_size if ds is not None else None,
                "drift_axes_present": bool(drift_axes),
                "drift_axes": drift_axes,
            }
        )

    await sess.end_scene(reason="long-form-probe-end", drain_slow_loop=True)

    return {
        "scenario_id": _SCENARIO_ID,
        "total_turns": len(per_turn),
        "first_summary_turn": first_summary_turn,
        "first_drift_turn": first_drift_turn,
        "final_distribution_summary": _summary_to_dict(final_summary),
        "per_turn": per_turn,
    }


def _replay_window_at(
    raw_errors: list[dict[str, float]],
    *,
    min_window: int,
    target_turns: tuple[int, ...],
) -> dict:
    """Feed raw PE errors through a fresh ``_PEDistributionWindow`` and
    capture summary at each ``target_turn`` (1-indexed).

    Bootstrap turns (``bootstrap=True`` on the original PE value) are
    skipped to mirror what the production owner does in
    ``_advance``.
    """
    window = _PEDistributionWindow(min_window=min_window, max_window=64)
    captured: dict[int, dict | None] = {turn: None for turn in target_turns}
    captured["final"] = None  # type: ignore[index]
    for index, raw in enumerate(raw_errors):
        turn = index + 1
        if not raw or raw.get("bootstrap", True):
            # Bootstrap or pre-PE turn — mirror the owner's skip rule.
            if turn in captured:
                summary = window.summarise()
                captured[turn] = _summary_to_dict(summary)
            continue
        err = PredictionError(
            task_error=raw["task"],
            relationship_error=raw["relationship"],
            regime_error=raw["regime"],
            action_error=raw["action"],
            magnitude=raw["magnitude"],
            signed_reward=0.0,
            description="replay",
        )
        window.update(err)
        if turn in captured:
            captured[turn] = _summary_to_dict(window.summarise())
    captured["final"] = _summary_to_dict(window.summarise())  # type: ignore[index]
    return captured


def _simulate_min_window_curves(per_turn: list[dict]) -> dict:
    """Post-hoc replay raw errors at multiple ``min_window`` settings.

    For each ``min_window`` in {6, 8, 10, 12, 16}, replay the same
    sample stream and report the summary at turn ``_SIMULATED_SAMPLE_TURNS``
    plus the final turn. Also compute per-axis stability ratios
    ``iqr_<n>_over_iqr_16_final`` to feed Wave B's decision.
    """
    raw_errors = [t["raw"] for t in per_turn]

    captures: dict[int, dict] = {}
    for mw in _SIMULATED_MIN_WINDOWS:
        captures[mw] = _replay_window_at(
            raw_errors,
            min_window=mw,
            target_turns=_SIMULATED_SAMPLE_TURNS,
        )

    # Compute stability ratios at the final turn — focus on min_window=8
    # vs min_window=16 since that is the parameter Wave B's decision
    # will land on by default.
    stability: dict[str, float | None] = {}
    final_n8 = captures[8]["final"] if captures[8].get("final") else None
    final_n16 = captures[16]["final"] if captures[16].get("final") else None
    if final_n8 is not None and final_n16 is not None:
        for axis in ("task", "relationship", "regime", "action"):
            iqr_8 = final_n8["iqr"].get(axis)
            iqr_16 = final_n16["iqr"].get(axis)
            if iqr_8 is None or iqr_16 is None:
                stability[axis] = None
                continue
            denom = max(abs(iqr_16), 1e-6)
            stability[axis] = iqr_8 / denom

    return {
        "captures": {str(mw): captures[mw] for mw in _SIMULATED_MIN_WINDOWS},
        "iqr_8_over_iqr_16_final": stability,
        "stability_judgement": {
            axis: (
                "STABLE"
                if (ratio is not None and 0.5 <= ratio <= 2.0)
                else ("MISSING" if ratio is None else "UNSTABLE")
            )
            for axis, ratio in stability.items()
        },
    }


def _verdict(top_level: dict) -> dict:
    """Produce the Wave A.3 acceptance verdict.

    PASS criteria (per plan):
    - first_summary_turn <= 18
    - first_drift_turn <= 23
    - all 4 axes are STABLE at iqr_8/iqr_16 ratio in [0.5, 2.0]
    """
    fs = top_level.get("first_summary_turn")
    fd = top_level.get("first_drift_turn")
    sim = top_level.get("simulated_min_window_curves") or {}
    judgement = sim.get("stability_judgement") or {}
    cond_summary = fs is not None and fs <= 18
    cond_drift = fd is not None and fd <= 23
    cond_stable = all(v == "STABLE" for v in judgement.values()) and len(judgement) == 4
    overall = cond_summary and cond_drift and cond_stable
    return {
        "first_summary_turn_le_18": cond_summary,
        "first_drift_turn_le_23": cond_drift,
        "all_axes_stable_n8_vs_n16": cond_stable,
        "overall_pass": overall,
    }


async def main() -> int:
    try:
        run_payload = await _run_long_form()
    except Exception:  # noqa: BLE001 - probe diagnostic, must record traceback
        traceback.print_exc()
        return 1

    sim = _simulate_min_window_curves(run_payload["per_turn"])
    run_payload["simulated_min_window_curves"] = sim
    run_payload["verdict"] = _verdict(run_payload)

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    fs = run_payload["first_summary_turn"]
    fd = run_payload["first_drift_turn"]
    verdict = run_payload["verdict"]
    print(f"[W4-probe] scenario={_SCENARIO_ID} total_turns={run_payload['total_turns']}")
    print(f"[W4-probe] first_summary_turn={fs} first_drift_turn={fd}")
    print(
        "[W4-probe] iqr_8_over_iqr_16 (final): "
        + ", ".join(
            f"{axis}={v:.3f}" if v is not None else f"{axis}=None"
            for axis, v in (sim["iqr_8_over_iqr_16_final"] or {}).items()
        )
    )
    print(f"[W4-probe] stability_judgement={sim['stability_judgement']}")
    print(f"[W4-probe] verdict={verdict}")
    print(f"[W4-probe] wrote {_OUTPUT_PATH}")
    return 0 if verdict["overall_pass"] else 0  # always 0; verdict is in JSON


if __name__ == "__main__":
    asyncio.run(main())
