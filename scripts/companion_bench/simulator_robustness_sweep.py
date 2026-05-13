"""P5 #53: user-simulator robustness sweep across LLM families.

Validates that swapping the user-simulator LLM (GPT / Claude /
Qwen / DeepSeek) on the same SUT pool does not produce bias-driven
ranking flips. Forces simulator-family rotation in v1.x release
runs.

Output:
* per-SUT × per-axis variance across simulator families
* simulator-family bias direction (e.g. "GPT-5 simulator favors
  verbose SUT outputs by Y points on A2")
* recommended simulator rotation policy

Run::

    python scripts/companion_bench/simulator_robustness_sweep.py --dry-run

Refs:
    docs/moving forward/companion-bench-public-launch-packet.md §2.3
    docs/external/companion-bench-simulator-robustness-v0.md
    docs/known-debts.md #53
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


_DEFAULT_SIMULATOR_FAMILIES = (
    "gpt5",
    "claude47",
    "qwen3",
    "deepseek4",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulator-families",
        type=lambda s: tuple(s.split(",")),
        default=_DEFAULT_SIMULATOR_FAMILIES,
    )
    parser.add_argument(
        "--reference-sut",
        type=lambda s: tuple(s.split(",")),
        default=("vz", "gpt5", "claude47", "qwen3", "deepseek4"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts/companion_bench")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _emit_placeholder_artifact(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    out_path = out_dir / f"simulator_robustness_sweep-{today}.json"
    payload = {
        "scaffold_status": "SHADOW",
        "sweep_kind": "simulator_robustness",
        "simulator_families": list(args.simulator_families),
        "reference_sut": list(args.reference_sut),
        "seed": args.seed,
        "dry_run": True,
        "per_sut_axis_variance": None,
        "per_simulator_family_bias_direction": None,
        "rotation_policy_recommendation": None,
        "notes": (
            "P5 #53 SHADOW. Sweeps "
            f"{len(args.simulator_families)} sim × "
            f"{len(args.reference_sut)} SUT × scenarios. Quarterly "
            "rotation policy lands in spec §4.x simulator family "
            "rotation."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dry_run:
        sys.stderr.write("P5 #53 SHADOW: real sweep not wired. Use --dry-run.\n")
        return 2
    out_path = _emit_placeholder_artifact(args)
    print(f"wrote SHADOW placeholder: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
