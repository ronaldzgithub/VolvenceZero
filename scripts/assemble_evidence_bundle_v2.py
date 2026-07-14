"""One-command unified evidence bundle v2 assembly (CP-00 / GAP-10).

Aggregates the frozen sub-bundles the operator requests into a single
``evidence_bundle_v2.json`` with shared provenance and per-input sha256
fingerprints. Missing / provenance-incomplete inputs fail loudly.

Example (aggregate a learned-shadow soak + an EQ bundle):

    python scripts/assemble_evidence_bundle_v2.py \
        --learned-shadow artifacts/learned_shadow_soak/learned_shadow_soak.json \
        --eq-longitudinal artifacts/eq_bundle/evidence_bundle.json \
        --output artifacts/evidence_bundle_v2.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from volvence_zero.agent.evidence_bundle_v2 import (
    EvidenceBundleV2Error,
    assemble_evidence_bundle_v2,
)

_LANE_FLAGS = (
    ("--dialogue-paper-suite", "dialogue_paper_suite"),
    ("--eta-paper-suite", "eta_paper_suite"),
    ("--eq-longitudinal", "eq_longitudinal"),
    ("--learned-shadow", "learned_shadow"),
    ("--companion-p1-manifest", "companion_p1_manifest"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, _lane in _LANE_FLAGS:
        parser.add_argument(flag, type=pathlib.Path, default=None)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/evidence_bundle_v2.json"),
    )
    parser.add_argument("--bundle-id", default="evidence-bundle-v2")
    args = parser.parse_args(argv)

    inputs = {
        lane: getattr(args, lane)
        for _flag, lane in _LANE_FLAGS
        if getattr(args, lane) is not None
    }
    try:
        bundle = assemble_evidence_bundle_v2(
            inputs=inputs, bundle_id=args.bundle_id
        )
    except EvidenceBundleV2Error as exc:
        print(f"[evidence-bundle-v2] FAILED: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[evidence-bundle-v2] wrote {args.output} "
        f"(lanes: {', '.join(bundle['lane_names'])})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
