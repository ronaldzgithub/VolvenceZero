# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Emit the canonical scenario hash table to stdout.

Used by CI to detect drift between docs/external/companion-bench-public-scenario-hashes.txt
and the actual public scenario set. Held-out hashes are emitted in the
same format but to a separate file (kept in the private submodule) when
``--include-heldout`` is passed.

Usage:

    python scripts/companion_bench/emit_scenario_hashes.py
    python scripts/companion_bench/emit_scenario_hashes.py --include-heldout > heldout.txt
"""

from __future__ import annotations

import argparse
import importlib.resources as res
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="emit_scenario_hashes")
    parser.add_argument(
        "--include-heldout",
        action="store_true",
        help=(
            "Include held-out scenarios from external/companionbench-heldout/. "
            "By default the script only emits public hashes."
        ),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional output file. Defaults to stdout. The repo file "
            "lives at docs/external/companion-bench-public-scenario-hashes.txt."
        ),
    )
    args = parser.parse_args(argv)

    from companion_bench.spec import load_scenarios_dir, scenario_hash

    public_dir = pathlib.Path(str(res.files("companion_bench") / "scenarios" / "public"))
    specs = list(load_scenarios_dir(public_dir, include_held_out=False))

    if args.include_heldout:
        heldout_dir = REPO_ROOT / "external" / "companionbench-heldout" / "scenarios"
        if heldout_dir.exists():
            specs.extend(load_scenarios_dir(heldout_dir, include_held_out=True))

    specs.sort(key=lambda s: s.scenario_id)
    lines = ["# Companion Bench scenario hash manifest"]
    for s in specs:
        lines.append(f"{s.scenario_id}\t{scenario_hash(s)}")

    body = "\n".join(lines) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
