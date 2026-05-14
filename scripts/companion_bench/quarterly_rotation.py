#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Quarterly governance rotation automation (debt #35).

Quarterly tasks (RFC §11):

1. **Held-out paraphrase rotation** — propose 8 new paraphrases
   per held-out scenario (24 scenarios × 8 paraphrases = 192
   proposals). Output: ``artifacts/governance/heldout_rotation_<quarter>.jsonl``
   for the maintainer to review + commit to the private heldout
   submodule.
2. **Lexicon rotation proposal** — surface 5 high-cost vocabulary
   items the user-simulator should rotate to keep the lexicon
   from being memorised by submitter SUTs. Output:
   ``artifacts/governance/lexicon_rotation_<quarter>.txt``.
3. **Judge family rotation log** — append the current quarter's
   judge family choice + a 1-line rationale to
   ``docs/external/companion-bench-judge-rotation-log.md``.

The script only **proposes**; an organisation maintainer reviews
the outputs and decides what lands in the next release.

Usage::

    python scripts/companion_bench/quarterly_rotation.py \\
        --quarter 2026Q3 \\
        --judge-family openai \\
        --judge-rationale "Anthropic blocked over capacity; OpenAI rotation kept."
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_GOVERNANCE_LOG = (
    _REPO_ROOT / "docs" / "external" / "companion-bench-judge-rotation-log.md"
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quarterly_rotation")
    p.add_argument(
        "--quarter",
        required=True,
        help="Quarter identifier (e.g. 2026Q3).",
    )
    p.add_argument(
        "--judge-family",
        choices=("anthropic", "openai", "deepseek", "qwen", "google"),
        required=True,
        help="LLM judge family for this quarter.",
    )
    p.add_argument(
        "--judge-rationale",
        required=True,
        help="One-line reason for this quarter's judge-family choice.",
    )
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_REPO_ROOT / "artifacts" / "governance",
        help="Where to write the rotation proposal artifacts.",
    )
    p.add_argument(
        "--paraphrases-per-scenario",
        type=int,
        default=8,
        help="How many paraphrase proposals per held-out scenario (default 8).",
    )
    p.add_argument(
        "--lexicon-rotation-count",
        type=int,
        default=5,
        help="How many vocabulary items to propose rotating (default 5).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing artifacts.",
    )
    return p


# Lightweight deterministic vocabulary tracker. Production wires this
# to a real n-gram count over user-simulator outputs across the last
# 4 quarters; SHADOW keeps a curated rotation queue so the contract
# test can lock the schema. The queue is a const here; later versions
# would generate it.
_ROTATION_VOCAB_QUEUE: tuple[str, ...] = (
    "absolutely",
    "wonderful",
    "amazing",
    "honestly",
    "literally",
    "basically",
    "definitely",
    "actually",
    "totally",
    "perfectly",
)


def _emit_paraphrase_proposals(
    *, quarter: str, scenarios_per_family: int, paraphrases_per_scenario: int,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """Write per-scenario paraphrase rotation proposals.

    SHADOW: deterministic placeholder seed templates. The maintainer
    reviews + replaces the proposals with real reviewer-curated
    paraphrases before committing.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"heldout_rotation_{quarter}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for family in ("F1", "F2", "F3", "F4", "F5", "F6"):
            for idx in range(1, scenarios_per_family + 1):
                scenario_id = f"{family}-heldout-{idx:03d}"
                for p_idx in range(paraphrases_per_scenario):
                    fh.write(
                        json.dumps(
                            {
                                "quarter": quarter,
                                "scenario_id": scenario_id,
                                "paraphrase_index": p_idx,
                                "proposal_status": "needs_reviewer",
                                "shadow_template": (
                                    f"[SHADOW seed paraphrase {p_idx} for "
                                    f"{scenario_id}]"
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    return out


def _emit_lexicon_rotation(
    *, quarter: str, count: int, output_dir: pathlib.Path,
) -> pathlib.Path:
    """Write deterministic lexicon rotation proposal."""

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"lexicon_rotation_{quarter}.txt"
    proposals = _ROTATION_VOCAB_QUEUE[:count]
    lines = [
        f"# Lexicon rotation proposal — {quarter}",
        "# Vocabulary items to rotate OUT of the user-simulator's",
        "# lexicon for this quarter (debt #35). Maintainer picks",
        "# replacements + commits to companion-bench/scenarios.",
        "",
    ] + list(proposals)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _append_judge_rotation_log(
    *, quarter: str, family: str, rationale: str,
) -> pathlib.Path:
    """Append a one-line judge-family entry to the rotation log."""

    if not _GOVERNANCE_LOG.exists():
        _GOVERNANCE_LOG.parent.mkdir(parents=True, exist_ok=True)
        _GOVERNANCE_LOG.write_text(
            "# Companion Bench: Judge Family Rotation Log\n\n"
            "> Maintained by `scripts/companion_bench/quarterly_rotation.py` (debt #35).\n"
            "> Each row: quarter | judge family | rationale | timestamp.\n\n"
            "| Quarter | Judge family | Rationale | Timestamp (UTC) |\n"
            "|---|---|---|---|\n",
            encoding="utf-8",
        )
    timestamp_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with _GOVERNANCE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {quarter} | {family} | {rationale.replace('|', '/')}"
            f" | {timestamp_iso} |\n"
        )
    return _GOVERNANCE_LOG


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dry_run:
        print(
            f"DRY RUN: quarter={args.quarter!r}, family={args.judge_family!r}, "
            f"output_dir={args.output_dir!r}"
        )
        return 0
    paraphrase_path = _emit_paraphrase_proposals(
        quarter=args.quarter,
        scenarios_per_family=4,
        paraphrases_per_scenario=args.paraphrases_per_scenario,
        output_dir=args.output_dir,
    )
    print(f"paraphrase proposals → {paraphrase_path}")
    lexicon_path = _emit_lexicon_rotation(
        quarter=args.quarter,
        count=args.lexicon_rotation_count,
        output_dir=args.output_dir,
    )
    print(f"lexicon rotation → {lexicon_path}")
    log_path = _append_judge_rotation_log(
        quarter=args.quarter,
        family=args.judge_family,
        rationale=args.judge_rationale,
    )
    print(f"judge rotation log appended → {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
