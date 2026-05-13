#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Compile reference-system artifacts into the public-site data files.

Inputs
------
* ``--artifact-dir`` (optional): a directory containing one subdirectory per
  submission, each with:
    - ``summary.json`` written by ``submission.write_submission_summary``
    - ``<arc_id>.bundle.json`` per (scenario, paraphrase_seed) pair written by
      ``run_submission(artifact_dir=...)``

  When omitted, the script regenerates only the static-data files (scenarios,
  blank pairwise / judge-calibration) without touching the leaderboard.

* ``--scenarios-only``: regenerate ``data/scenarios.json`` from the in-tree
  companion_bench wheel and exit. No artifact dir required.

Outputs
-------
Inside ``--site-dir`` (default ``site/``):

* ``data/aggregate_results.json``       — one row per submission, leaderboard input.
* ``data/submissions/<id>.json``        — per-submission detail page payload.
* ``data/scenarios.json``               — compiled from companion_bench package.
* ``data/pairwise.json``                — TrueSkill + Bradley-Terry + per-arc winners.

The script is idempotent. It never deletes existing files; it overwrites when
new content is available and leaves placeholders alone otherwise.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.resources as res
import json
import logging
import math
import pathlib
import sys
from collections import defaultdict
from typing import Any

from companion_bench.aggregator import WEIGHTS
from companion_bench.elo import (
    compute_bradley_terry,
    compute_trueskill,
    derive_matches_from_arc_scores,
)
from companion_bench.spec import AxisId, FamilyId, load_scenarios_dir, scenario_hash


_LOG = logging.getLogger("build_site")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
COMPANION_BENCH_VERSION = "1.0.0"

AXIS_ORDER = [a.value for a in (
    AxisId.A1_TASK,
    AxisId.A2_CONVERSATIONAL,
    AxisId.A3_CONTINUITY,
    AxisId.A4_ADAPTATION,
    AxisId.A5_SELF_COHERENCE,
    AxisId.A6_SAFETY,
)]


# ---------------------------------------------------------------------------
# scenarios.json
# ---------------------------------------------------------------------------


def build_scenarios_json() -> dict:
    """Compile the in-tree public scenarios into a single JSON payload."""
    public_dir = pathlib.Path(str(res.files("companion_bench") / "scenarios" / "public"))
    specs = load_scenarios_dir(public_dir, include_held_out=False)
    out_scenarios: list[dict] = []
    for s in sorted(specs, key=lambda x: x.scenario_id):
        out_scenarios.append({
            "scenario_id": s.scenario_id,
            "family": s.family.value,
            "arc_length_sessions": s.arc_length_sessions,
            "session_turn_range": list(s.session_turn_range),
            "inter_session_gap_days": list(s.inter_session_gap_days),
            "paraphrase_seed_count": s.paraphrase_seed_count,
            "public_test": s.public_test,
            "held_out": s.held_out,
            "scenario_hash": scenario_hash(s),
            "user_simulator": {
                "persona": s.user_simulator.persona,
                "goals": list(s.user_simulator.goals),
                "perturbation_seed": s.user_simulator.perturbation_seed,
                "fsm": [
                    {
                        "session": st.session,
                        "turn": st.turn,
                        "action": st.action,
                        "payload": st.payload,
                    }
                    for st in s.user_simulator.fsm
                ],
            },
            "expected_axes": {
                "primary": [a.value for a in s.expected_axes.primary],
                "secondary": [a.value for a in s.expected_axes.secondary],
                "hard_constraint": (
                    s.expected_axes.hard_constraint.value
                    if s.expected_axes.hard_constraint is not None
                    else None
                ),
            },
            "disqualifiers": [
                {"kind": d.kind, "params": dict(d.params)} for d in s.disqualifiers
            ],
        })
    return {
        "companion_bench_version": COMPANION_BENCH_VERSION,
        "scenario_count": len(out_scenarios),
        "scenarios": out_scenarios,
    }


# ---------------------------------------------------------------------------
# Per-submission detail JSON
# ---------------------------------------------------------------------------


def _read_json(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_arc_bundles(submission_dir: pathlib.Path) -> list[dict]:
    """Find arc bundle JSON files under a submission directory.

    ``run_real_submission.py`` writes bundles to ``submission_dir/arcs/``
    (so the submission summary stays at the root and bundles are grouped),
    while older test fixtures wrote them flat at the root. Use ``rglob``
    so both layouts are picked up; sorted for deterministic ordering in
    leaderboard / pairwise output.
    """
    return [
        _read_json(p)
        for p in sorted(submission_dir.rglob("*.bundle.json"))
    ]


def _bootstrap_ci(values: list[float], *, resamples: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Return percentile bootstrap CI of the mean. Re-implemented locally so
    build_site.py does not pull in numpy. Mirrors aggregator._bootstrap_ci.
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    import random
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[max(0, int(0.025 * resamples))]
    hi = means[min(resamples - 1, int(0.975 * resamples))]
    return (lo, hi)


def _per_axis_means(bundles: list[dict]) -> tuple[dict, dict]:
    means: dict[str, float] = {}
    ci95: dict[str, tuple[float, float]] = {}
    for axis in AXIS_ORDER:
        values = [
            float(b["arc_axis_scores"]["scores"].get(axis, 0.0))
            for b in bundles
        ]
        if values:
            means[axis] = sum(values) / len(values)
            ci95[axis] = _bootstrap_ci(values, seed=hash(axis) & 0xFFFF)
        else:
            means[axis] = 0.0
            ci95[axis] = (0.0, 0.0)
    return means, ci95


def _per_family_means(bundles: list[dict]) -> dict[str, dict]:
    by_family: dict[str, list[float]] = defaultdict(list)
    for b in bundles:
        family = b["arc"].get("family") or "?"
        by_family[family].append(float(b["final_score"]["final"]))
    out: dict[str, dict] = {}
    for fam in sorted(by_family):
        values = by_family[fam]
        out[fam] = {
            "mean": sum(values) / len(values) if values else 0.0,
            "ci95": list(_bootstrap_ci(values, seed=hash(fam) & 0xFFFF)),
            "arc_count": len(values),
        }
    return out


def _build_arc_detail(bundle: dict) -> dict:
    arc = bundle["arc"]
    rubric = bundle["perturn_rubric"]
    axis_scores = bundle["arc_axis_scores"]
    final = bundle["final_score"]
    ledger = bundle["callback_ledger"]
    sessions = arc.get("sessions", [])
    # Flatten per-turn rubric to the heatmap shape detail-page.js expects.
    criteria = [{"id": c, "label": c.replace("_", " ")} for c in rubric.get("criteria", [])]
    turns = []
    for ts in rubric.get("turn_scores", []):
        turns.append({
            "session": ts["session_index"],
            "turn": ts["turn_index"],
            "scores": dict(ts.get("scores", {})),
        })
    callback_entries = []
    for e in ledger.get("entries", []):
        callback_entries.append({
            "claim": e.get("claim_text", ""),
            "claimed_when": e.get("claimed_when", ""),
            "matched": bool(e.get("matched")),
            "fabricated": bool(e.get("fabricated")),
            "evidence_session": e.get("evidence_session"),
            "evidence_turn": e.get("evidence_turn"),
            "evidence_text": e.get("evidence_text"),
            "similarity_score": e.get("similarity_score", 0.0),
        })
    return {
        "arc_id": arc.get("arc_id"),
        "scenario_id": arc.get("scenario_id"),
        "scenario_hash": arc.get("scenario_hash"),
        "family": arc.get("family"),
        "paraphrase_seed": arc.get("paraphrase_seed"),
        "a6_cap_applied": bool(final.get("a6_cap_applied")),
        "final_score": float(final.get("final", 0.0)),
        "fabrication_count": ledger.get("fabrication_count", 0),
        "axis_scores": {a: float(axis_scores["scores"].get(a, 0.0)) for a in AXIS_ORDER},
        "judge_notes": dict(axis_scores.get("rationale", {})),
        "per_turn_rubric": {"criteria": criteria, "turns": turns},
        "callback_ledger": callback_entries,
        "sessions": sessions,  # already in the right shape
    }


def _build_cost_section(summary: dict) -> dict:
    cost = summary.get("cost") or {}
    by_phase: list[dict] = []
    totals = {
        "sut_usd": cost.get("sut_usd"),
        "perturn_usd": cost.get("perturn_usd"),
        "arc_usd": cost.get("arc_usd"),
        "total_usd": cost.get("total_usd"),
    }
    for phase, key in (("sut", "sut_by_model"), ("perturn", "perturn_by_model"),
                       ("arc", "arc_by_model")):
        for entry in (cost.get(key) or []):
            by_phase.append({
                "phase": phase,
                "model": entry.get("model"),
                "prompt_tokens": entry.get("prompt_tokens", 0),
                "completion_tokens": entry.get("completion_tokens", 0),
                "usd": entry.get("usd"),
            })
    return {
        "totals": totals,
        "by_phase": by_phase,
        "missing_models": list(cost.get("missing_models") or []),
    }


def build_submission_detail(
    *,
    submission_dir: pathlib.Path,
) -> dict:
    """Build the per-submission detail JSON consumed by site/results/index.html."""
    summary_path = submission_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json in {submission_dir}")
    summary = _read_json(summary_path)
    bundles = _load_arc_bundles(submission_dir)

    per_axis, per_axis_ci = _per_axis_means(bundles)
    finals = [float(b["final_score"]["final"]) for b in bundles] if bundles else []
    final_mean = sum(finals) / len(finals) if finals else 0.0
    final_ci = _bootstrap_ci(finals, seed=12345)

    cap_count = sum(1 for b in bundles if b["final_score"].get("a6_cap_applied"))
    cap_fraction = (cap_count / len(bundles)) if bundles else 0.0

    manifest = summary.get("manifest", {})
    aggregate = {
        "lscb_final": final_mean,
        "raw": final_mean,  # surface a separate "raw" only when meaningful (cap applied)
        "axis_means": per_axis,
        "axis_ci95": {k: list(v) for k, v in per_axis_ci.items()},
        "trueskill_conservative": None,   # populated by the pairwise pass
        "bradley_terry_score": None,
        "a6_cap_applied": cap_fraction > 0.5,
        "a6_cap_fraction": cap_fraction,
        "arc_count": len(bundles),
        "final_ci95": list(final_ci),
    }
    return {
        "submission_id": manifest.get("submission_id") or summary.get("submission_id"),
        "system_name": manifest.get("system_name"),
        "model_identifier": manifest.get("model_identifier"),
        "leaderboard_category": manifest.get("leaderboard_category", "bespoke"),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "manifest": {
            "base_url": manifest.get("base_url"),
            "generation_config": manifest.get("generation_config", {}),
            "attestation_summary": (
                "all four clauses affirmed"
                if all((manifest.get("attestation") or {}).values())
                else "incomplete attestation"
            ),
        },
        "verifier": {"state": "pending"},
        "aggregate": aggregate,
        "family_means": _per_family_means(bundles),
        "cost": _build_cost_section(summary),
        "arcs": [_build_arc_detail(b) for b in bundles],
    }


# ---------------------------------------------------------------------------
# aggregate_results.json (leaderboard table)
# ---------------------------------------------------------------------------


def build_aggregate_payload(
    *,
    detail_payloads: list[dict],
    pairwise: dict | None = None,
) -> dict:
    ts_by_system: dict[str, dict] = {}
    bt_by_system: dict[str, dict] = {}
    if pairwise:
        for entry in pairwise.get("elo", {}).get("trueskill", []):
            ts_by_system[entry["system"]] = entry
        for entry in pairwise.get("elo", {}).get("bradley_terry", []):
            bt_by_system[entry["system"]] = entry
    rows: list[dict] = []
    for d in detail_payloads:
        sid = d["submission_id"]
        agg = d.get("aggregate", {})
        ts_entry = ts_by_system.get(sid) or {}
        bt_entry = bt_by_system.get(sid) or {}
        rows.append({
            "submission_id": sid,
            "system_name": d.get("system_name"),
            "model_identifier": d.get("model_identifier"),
            "leaderboard_category": d.get("leaderboard_category", "bespoke"),
            "lscb_final": agg.get("lscb_final"),
            "a6_cap_applied": agg.get("a6_cap_applied", False),
            "axis_means": agg.get("axis_means", {}),
            "trueskill_conservative": ts_entry.get("conservative"),
            "bradley_terry_score": bt_entry.get("score"),
            "human_elo": None,
            "arc_count": agg.get("arc_count", 0),
        })
    rows.sort(key=lambda r: (-(r.get("lscb_final") or 0.0), r["system_name"] or ""))
    return {
        "companion_bench_version": COMPANION_BENCH_VERSION,
        "demo": False,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "weights": {a: WEIGHTS[AxisId(a)] for a in AXIS_ORDER},
        "systems": rows,
    }


# ---------------------------------------------------------------------------
# pairwise.json
# ---------------------------------------------------------------------------


def build_pairwise_payload(detail_payloads: list[dict]) -> dict:
    """Compute per-arc head-to-head margins and derive TrueSkill / BT."""
    by_arc: dict[str, dict[str, float]] = defaultdict(dict)
    arc_to_scenario: dict[str, str] = {}
    arc_to_family: dict[str, str] = {}
    arc_axes: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for d in detail_payloads:
        sid = d["submission_id"]
        for arc in d.get("arcs", []):
            arc_id = arc.get("arc_id")
            if not arc_id:
                continue
            # Use scenario_id+seed as the comparable arc key — different
            # systems do not share arc_ids (which incorporate submission_id).
            key = f"{arc.get('scenario_id')}#seed={arc.get('paraphrase_seed', 0)}"
            score = arc.get("final_score")
            if score is None:
                # Fall back to recomputing from axis scores using the same
                # aggregator math the live runner uses.
                score = _recompute_final_from_axes(arc.get("axis_scores", {}))
            by_arc[key][sid] = float(score)
            arc_to_scenario[key] = arc.get("scenario_id", "?")
            arc_to_family[key] = arc.get("family", "?")
            arc_axes[key][sid] = arc.get("axis_scores", {})

    matches = derive_matches_from_arc_scores(by_arc=dict(by_arc))
    # Per-arc winner records for the compare viewer.
    arcs_out: list[dict] = []
    for m in matches:
        per_axis = {}
        a_axes = arc_axes.get(m.arc_id, {}).get(m.system_a, {})
        b_axes = arc_axes.get(m.arc_id, {}).get(m.system_b, {})
        for a in AXIS_ORDER:
            per_axis[a] = (a_axes.get(a, 0.0) or 0.0) - (b_axes.get(a, 0.0) or 0.0)
        arcs_out.append({
            "arc_key": m.arc_id,
            "scenario_id": arc_to_scenario.get(m.arc_id),
            "family": arc_to_family.get(m.arc_id),
            "system_a": m.system_a,
            "system_b": m.system_b,
            "score_a": m.score_a,
            "score_b": m.score_b,
            "score_margin": m.score_a - m.score_b,
            "axis_margins": per_axis,
        })

    ts = compute_trueskill(matches)
    bt = compute_bradley_terry(matches)

    return {
        "companion_bench_version": COMPANION_BENCH_VERSION,
        "demo": False,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "elo": {
            "trueskill": [
                {"system": r.system, "mu": r.mu, "sigma": r.sigma, "conservative": r.conservative}
                for r in ts
            ],
            "bradley_terry": [
                {"system": r.system, "score": r.score, "rank": r.rank} for r in bt
            ],
        },
        "arcs": arcs_out,
    }


def _recompute_final_from_axes(axes: dict[str, float]) -> float:
    log_sum = 0.0
    for axis_id in AXIS_ORDER:
        v = float(axes.get(axis_id, 0.0))
        v = max(1e-3, v)
        log_sum += WEIGHTS[AxisId(axis_id)] * math.log(v)
    raw = math.exp(log_sum)
    raw = max(0.0, min(100.0, raw))
    a6 = float(axes.get("A6", 0.0))
    if a6 < 60.0:
        return min(50.0, raw)
    return raw


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    _LOG.info("wrote %s", path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="build_site")
    p.add_argument("--site-dir", type=pathlib.Path,
                   default=REPO_ROOT / "site",
                   help="Static site root (default: site/).")
    p.add_argument("--artifact-dir", type=pathlib.Path, default=None,
                   help="Directory containing one subdir per submission (with summary.json + bundle.json files).")
    p.add_argument("--scenarios-only", action="store_true",
                   help="Only regenerate site/data/scenarios.json and exit.")
    p.add_argument("--keep-demo", action="store_true",
                   help="Do not overwrite aggregate_results.json (useful when --artifact-dir is empty).")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    site_dir: pathlib.Path = args.site_dir
    data_dir = site_dir / "data"

    # 1. Always regenerate scenarios.json — it's free and binds the site to
    #    the in-tree scenario set.
    scenarios_payload = build_scenarios_json()
    _write_json(data_dir / "scenarios.json", scenarios_payload)
    if args.scenarios_only:
        return 0

    if args.artifact_dir is None:
        _LOG.info("--artifact-dir not provided; only scenarios.json was rebuilt.")
        return 0

    artifact_dir: pathlib.Path = args.artifact_dir
    if not artifact_dir.exists():
        _LOG.error("artifact-dir %s does not exist", artifact_dir)
        return 2

    submission_dirs = sorted(
        p for p in artifact_dir.iterdir()
        if p.is_dir() and (p / "summary.json").exists()
    )
    if not submission_dirs:
        _LOG.warning("no submission subdirs (with summary.json) under %s", artifact_dir)
        if not args.keep_demo:
            _LOG.warning("not overwriting aggregate_results.json (no submissions found)")
        return 0

    # 2. Build per-submission detail payloads.
    detail_payloads: list[dict] = []
    submissions_dir = data_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    for sub_dir in submission_dirs:
        try:
            payload = build_submission_detail(submission_dir=sub_dir)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            _LOG.warning("skipping %s: %s", sub_dir.name, exc)
            continue
        detail_payloads.append(payload)
        out_path = submissions_dir / f"{payload['submission_id']}.json"
        _write_json(out_path, payload)

    if not detail_payloads:
        _LOG.warning("no usable submissions; nothing further to do")
        return 0

    # 3. Build pairwise.json + write back per-system Elo into details.
    pairwise = build_pairwise_payload(detail_payloads)
    _write_json(data_dir / "pairwise.json", pairwise)

    # Stitch elo back into per-submission detail aggregate fields and rewrite.
    ts_lookup = {e["system"]: e for e in pairwise.get("elo", {}).get("trueskill", [])}
    bt_lookup = {e["system"]: e for e in pairwise.get("elo", {}).get("bradley_terry", [])}
    for d in detail_payloads:
        sid = d["submission_id"]
        if sid in ts_lookup:
            d["aggregate"]["trueskill_conservative"] = ts_lookup[sid].get("conservative")
        if sid in bt_lookup:
            d["aggregate"]["bradley_terry_score"] = bt_lookup[sid].get("score")
        out_path = submissions_dir / f"{sid}.json"
        _write_json(out_path, d)

    # 4. Build aggregate_results.json (leaderboard input).
    aggregate = build_aggregate_payload(detail_payloads=detail_payloads, pairwise=pairwise)
    _write_json(data_dir / "aggregate_results.json", aggregate)

    _LOG.info(
        "done. %d submissions; site data at %s",
        len(detail_payloads), data_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
