#!/usr/bin/env python3
"""Run the read-only L1-B formation-protection checkpoint precheck."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from volvence_ant.experiments.alignment_formation_protection import (
    build_alignment_formation_protection_precheck,
)


_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (_ROOT / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-report",
        type=Path,
        default=Path(
            "research/ant/results/ecology_recovery/same_physics_baseline/"
            "ecology_same_physics_alignment_review.seed0.20260731T053814Z.json"
        ),
    )
    parser.add_argument("--review-progress-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-seed", type=int, default=700_003)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


async def _main() -> int:
    args = _parser().parse_args()
    output_path = (
        _resolve(args.json_out) if args.json_out is not None else None
    )
    if output_path is not None and output_path.exists():
        raise FileExistsError(
            "refusing to overwrite immutable L1-B precheck artifact: "
            f"{output_path}"
        )
    payload = await build_alignment_formation_protection_precheck(
        review_report_path=_resolve(args.review_report),
        review_progress_dir=_resolve(args.review_progress_dir),
        seed=args.seed,
        probe_seed=args.probe_seed,
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
