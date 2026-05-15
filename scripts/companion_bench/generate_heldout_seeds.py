# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Generate the initial held-out scenario seed set (96 YAML files).

This tool is run **once** by an organiser to populate the private
``companionbench/heldout`` repository. The output is NOT committed
to the public monorepo; the user moves the resulting tree into the
private repo and pushes it there.

The seeds are derived from the 24 public scenarios by:

* Cycling through 4 distinct surface-form variants per public scenario
  (so each public family of 4 → 16 held-out scenarios).
* Variants change persona, identity context, and FSM payload wording
  while preserving the family-level probe (rupture, callback,
  long-absence, etc.).
* All variants get ``held_out: true``, ``public_test: false``.

Total: 24 public × 4 variants = 96 held-out, matching RFC §10's
"4× public" proposal.

Usage::

    python scripts/companion_bench/generate_heldout_seeds.py \\
        --output external/companionbench-heldout/scenarios

Then push that tree into the private repo:

    cd external/companionbench-heldout
    git init && git add . && git commit -m "v1.0 held-out seed"
    git remote add origin git@github.com:companionbench/heldout.git
    git push -u origin main
"""

from __future__ import annotations

import argparse
import importlib.resources as res
import pathlib
import sys

import yaml


# Variant suffix → (persona override, payload prefix override). Four
# variants per public scenario produce 96 total. The variants stay
# topically aligned to the family probe but shift the surface so a
# system overfit on the public set shows a measurable gap.
_VARIANT_PROFILES: tuple[tuple[str, str, str], ...] = (
    (
        "v01",
        "the same probe with a different cultural background",
        "different-locale-",
    ),
    (
        "v02",
        "the same probe with a different age cohort and life stage",
        "different-cohort-",
    ),
    (
        "v03",
        "the same probe with a different occupation surface",
        "different-occupation-",
    ),
    (
        "v04",
        "the same probe with a different concrete contextual detail",
        "different-context-",
    ),
)


def _load_public_yaml(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _make_variant(
    public_doc: dict,
    *,
    variant_suffix: str,
    persona_note: str,
    payload_prefix: str,
) -> dict:
    """Apply variant transformations to a public scenario YAML doc.

    The transformations preserve every structural field (FSM action
    coordinates, expected_axes, disqualifiers, arc length) so the
    held-out set probes the same axis with a different surface.
    """

    out = {
        "scenario_id": f"{public_doc['scenario_id']}-{variant_suffix}-heldout",
        "family": public_doc["family"],
        "arc_length_sessions": public_doc["arc_length_sessions"],
        "session_turn_range": list(public_doc["session_turn_range"]),
        "inter_session_gap_days": list(public_doc["inter_session_gap_days"]),
        "user_simulator": {
            "persona": (
                f"{public_doc['user_simulator']['persona']} "
                f"(held-out variant: {persona_note})"
            ),
            "goals": list(public_doc["user_simulator"]["goals"]),
            "perturbation_seed": int(public_doc["user_simulator"]["perturbation_seed"])
            + _hash_offset(variant_suffix),
            "fsm": [
                {
                    "session": int(step["session"]),
                    "turn": int(step["turn"]),
                    "action": step["action"],
                    **(
                        {"payload": f"{payload_prefix}{step['payload']}"}
                        if step.get("payload")
                        else {}
                    ),
                }
                for step in public_doc["user_simulator"]["fsm"]
            ],
        },
        "expected_axes": dict(public_doc["expected_axes"]),
        "disqualifiers": list(public_doc["disqualifiers"]),
        "public_test": False,
        "held_out": True,
        "paraphrase_seed_count": int(public_doc.get("paraphrase_seed_count", 3)),
    }
    return out


def _hash_offset(suffix: str) -> int:
    return sum(ord(c) for c in suffix) % 1009


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generate_heldout_seeds")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Directory to write 96 held-out YAML files into.",
    )
    parser.add_argument(
        "--public-dir",
        type=pathlib.Path,
        default=None,
        help="Override path to the public scenarios dir (defaults to packaged location).",
    )
    args = parser.parse_args(argv)

    if args.public_dir is None:
        args.public_dir = pathlib.Path(
            str(res.files("companion_bench") / "scenarios" / "public")
        )

    public_files = sorted(args.public_dir.glob("*.yaml"))
    if len(public_files) != 24:
        print(
            f"warning: expected 24 public scenarios, found {len(public_files)}; "
            f"continuing anyway",
            file=sys.stderr,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in public_files:
        public_doc = _load_public_yaml(path)
        for suffix, persona_note, payload_prefix in _VARIANT_PROFILES:
            variant = _make_variant(
                public_doc,
                variant_suffix=suffix,
                persona_note=persona_note,
                payload_prefix=payload_prefix,
            )
            out_path = args.output / f"{variant['scenario_id']}.yaml"
            with out_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(
                    variant, fh, sort_keys=False, allow_unicode=True, width=120
                )
            written += 1

    print(f"wrote {written} held-out scenarios to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
