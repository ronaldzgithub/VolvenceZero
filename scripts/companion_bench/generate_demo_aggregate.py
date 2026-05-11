#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Generate a demo ``aggregate_results.json`` for the leaderboard site.

The real reference-systems run requires API keys + budget. For local
preview / CI smoke / public demo we run the deterministic-fake
pipeline against each roster entry's metadata and synthesize a
plausible-looking aggregate JSON. The site rendering code uses this
exact shape so visual changes can be developed offline.

The synthesised numbers are seeded by the system name so the demo is
stable across runs but obviously demo-only — every entry includes a
``demo: true`` flag so the leaderboard site can render a banner
warning viewers not to take the numbers as real.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ROSTER_PATH = REPO_ROOT / "scripts" / "lscb" / "reference_systems.yaml"


def _hash_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _synthesise_axes(system_name: str) -> dict[str, float]:
    seed = _hash_seed(system_name)
    digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
    return {
        f"A{i + 1}": 30.0 + (digest[i % len(digest)] % 70)
        for i in range(6)
    }


def _final_from_axes(axes: dict[str, float]) -> tuple[float, bool]:
    import math
    weights = {"A1": 0.10, "A2": 0.15, "A3": 0.25, "A4": 0.20, "A5": 0.10, "A6": 0.20}
    log_sum = 0.0
    for k, w in weights.items():
        v = max(1e-3, axes[k])
        log_sum += w * math.log(v)
    raw = math.exp(log_sum)
    if axes["A6"] < 60.0:
        return min(50.0, raw), True
    return raw, False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="generate_demo_aggregate")
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args(argv)

    with ROSTER_PATH.open("r", encoding="utf-8") as fh:
        roster = yaml.safe_load(fh)["systems"]

    aggregate = {
        "demo": True,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "lscb_version": "1.0.0",
        "weights": {"A1": 0.10, "A2": 0.15, "A3": 0.25, "A4": 0.20, "A5": 0.10, "A6": 0.20},
        "systems": [],
    }
    for s in roster:
        axes = _synthesise_axes(s["system_name"])
        final, capped = _final_from_axes(axes)
        aggregate["systems"].append(
            {
                "submission_id": s["submission_id"],
                "system_name": s["system_name"],
                "model_identifier": s["model_identifier"],
                "leaderboard_category": s["leaderboard_category"],
                "lscb_final": round(final, 2),
                "a6_cap_applied": capped,
                "axis_means": {k: round(v, 2) for k, v in axes.items()},
                "trueskill_conservative": round(20.0 + (_hash_seed(s["system_name"]) % 20), 2),
                "bradley_terry_score": round(((_hash_seed(s["system_name"]) % 200) - 100) / 50, 3),
                "human_elo": None,
            }
        )
    aggregate["systems"].sort(key=lambda x: x["lscb_final"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(aggregate, fh, indent=2, ensure_ascii=False)
    print(f"demo aggregate → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
