#!/usr/bin/env python3
"""Run or emit a capacity→gain ladder schedule.

The full factorial manifest is intentionally large. This runner gives operators
one reproducible entry point for a bounded slice: select n_z values, backend
combo and turn count, then either emit the exact commands (dry-run) or execute
the learned-shadow soak lane for each arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from volvence_zero.agent.capacity_ladder import (
    build_capacity_ladder_manifest,
)


def _csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _csv_strings(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-z", type=_csv_ints, default=(3, 16, 64, 256))
    parser.add_argument("--turns", type=_csv_ints, default=(500,))
    parser.add_argument(
        "--backend-combos",
        type=_csv_strings,
        default=("runtime+ssl+internal-rl+cms-torch",),
    )
    parser.add_argument("--substrates", type=_csv_strings, default=("qwen-1.5b-main",))
    parser.add_argument("--seeds", type=_csv_ints, default=(0,))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/capacity_ladder"),
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_capacity_ladder_manifest(
        n_z_values=args.n_z,
        backend_combos=args.backend_combos,
        trace_turns=args.turns,
        substrates=args.substrates,
        seeds=args.seeds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "capacity_ladder_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": manifest.schema_version,
                "description": manifest.description,
                "arms": [arm.__dict__ for arm in manifest.arms],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    commands: list[list[str]] = []
    for arm in manifest.arms:
        arm_output = args.output_dir / arm.arm_id
        commands.append(
            [
                sys.executable,
                "-u",
                "scripts/run_learned_shadow_soak.py",
                "--turns",
                str(arm.trace_turns),
                "--output-dir",
                str(arm_output),
                "--temporal-latent-dim",
                str(arm.temporal_latent_dim),
                "--backend-combo",
                arm.backend_combo,
            ]
        )

    schedule_path = args.output_dir / "capacity_ladder_commands.json"
    schedule_path.write_text(
        json.dumps({"commands": commands}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote manifest: {manifest_path}")
    print(f"wrote command schedule: {schedule_path}")
    if not args.execute:
        return 0
    unsupported_execute = tuple(
        arm.backend_combo
        for arm in manifest.arms
        if arm.backend_combo != "runtime+ssl+internal-rl+cms-torch"
    )
    if unsupported_execute:
        raise SystemExit(
            "execute mode currently supports only the full four-backend "
            "learned-shadow collector; unsupported combos: "
            + ", ".join(sorted(set(unsupported_execute)))
        )

    for command in commands:
        print("[capacity] " + " ".join(command))
        rc = subprocess.run(command, check=False).returncode
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
