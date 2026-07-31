#!/usr/bin/env python3
"""Generate the read-only ecology alignment-formation attribution artifact."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from volvence_ant.experiments.alignment_attribution import (
    build_alignment_formation_attribution,
)


_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (_ROOT / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--station1-report",
        type=Path,
        default=Path(
            "research/ant/results/ecology_recovery/same_physics_baseline/"
            "ecology_same_physics_station1.seed0.20260731T052300Z.json"
        ),
    )
    parser.add_argument(
        "--review-report",
        type=Path,
        default=Path(
            "research/ant/results/ecology_recovery/same_physics_baseline/"
            "ecology_same_physics_alignment_review.seed0.20260731T053814Z.json"
        ),
    )
    parser.add_argument("--station1-progress-dir", type=Path, required=True)
    parser.add_argument("--review-progress-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


async def _main() -> int:
    args = _parser().parse_args()
    output_path = (
        _resolve(args.json_out) if args.json_out is not None else None
    )
    if output_path is not None and output_path.exists():
        raise FileExistsError(
            "refusing to overwrite immutable attribution artifact: "
            f"{output_path}"
        )
    payload = await build_alignment_formation_attribution(
        station1_report_path=_resolve(args.station1_report),
        review_report_path=_resolve(args.review_report),
        station1_progress_dir=_resolve(args.station1_progress_dir),
        review_progress_dir=_resolve(args.review_progress_dir),
        seed=args.seed,
        source_root=_ROOT,
    )
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if output_path is None:
        print(rendered, end="")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
