#!/usr/bin/env python3
"""Packet T1: freeze the fallible observational advisor table.

Builds ``credit_expert_actions`` over a calibration run's settled
trajectories (Packet T0) and freezes the result as the ADVISOR artifact
for the timing-v2 line. The advisor is observational and known to carry
survivorship/difficulty confounding (coding-lab spec §7.5) — that
fallibility is the experimental object, so recommendations are recorded
as-is, never filtered by prior plausibility.

Outputs (create-only):
* ``<output>``            — advisor artifact JSON (cells + full table + lineage)
* ``<output>.sha256``     — artifact digest, pinned by the T2 prereg
* ``<output-cells>``      — bare ``[[state_key, action], ...]`` list, the
  ``--directed-cells-json`` input of the Packet 3.5 directed runner
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _package_src in sorted(_REPO_ROOT.glob("packages/*/src")):
    if str(_package_src) not in sys.path:
        sys.path.insert(0, str(_package_src))

from lifeform_domain_coding.lab.junctions import (  # noqa: E402
    build_action_outcome_table,
    collect_junctions,
    credit_expert_actions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="freeze the observational advisor table")
    parser.add_argument("--calibration-run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-cells", required=True)
    parser.add_argument("--min-action-support", type=int, default=5)
    parser.add_argument("--min-pass-rate-margin", type=float, default=0.10)
    args = parser.parse_args(argv)

    run_dir = pathlib.Path(args.calibration_run_dir).resolve()
    trajectory_paths = tuple(sorted(run_dir.glob("chains/chain-*/trajectories/episode-*.jsonl")))
    if not trajectory_paths:
        raise FileNotFoundError(f"no settled trajectories under {run_dir!s}")
    records = collect_junctions(trajectory_paths)
    table = build_action_outcome_table(records)
    advisor = credit_expert_actions(
        records,
        min_action_support=args.min_action_support,
        min_pass_rate_margin=args.min_pass_rate_margin,
    )
    if not advisor:
        raise ValueError(
            "observational corpus yields no advisor cells at the frozen support/"
            "margin thresholds; enlarge the T0 corpus before freezing"
        )
    corpus_digest = hashlib.sha256(
        "\n".join(hashlib.sha256(p.read_bytes()).hexdigest() for p in trajectory_paths).encode()
    ).hexdigest()

    artifact = {
        "artifact": "coding-lab-advisor-table.v1",
        "source_calibration_run": run_dir.name,
        "trajectory_count": len(trajectory_paths),
        "corpus_sha256": corpus_digest,
        "junction_records": len(records),
        "min_action_support": args.min_action_support,
        "min_pass_rate_margin": args.min_pass_rate_margin,
        "advisor_cells": [[key, advisor[key]] for key in sorted(advisor)],
        "observational_table": {
            state_key: [
                {
                    "action": stat.action,
                    "trials": stat.trials,
                    "passes": stat.passes,
                    "pass_rate": round(stat.pass_rate, 6),
                }
                for stat in stats
            ]
            for state_key, stats in sorted(table.items())
        },
        "honest_boundaries": {
            "observational": True,
            "confounding_registered": "docs/specs/coding-lab.md §7.5 (survivorship/difficulty)",
            "recommendations_unfiltered": True,
            "note": (
                "This table is the FALLIBLE ADVISOR of the timing-v2 line. Its "
                "recommendations get causally priced by the directed Packet 3.5 "
                "RCT; nothing here is a capability claim or a ground truth."
            ),
        },
    }
    output = pathlib.Path(args.output)
    cells_output = pathlib.Path(args.output_cells)
    for path in (output, cells_output):
        if path.exists():
            raise FileExistsError(f"create-only output already exists: {path}")
    raw = json.dumps(artifact, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    output.write_text(raw, encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    cells_output.write_text(
        json.dumps(artifact["advisor_cells"], ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "advisor_path": str(output),
                "sha256": digest,
                "advisor_cells": artifact["advisor_cells"],
                "junction_records": len(records),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
