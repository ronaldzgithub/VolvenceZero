"""Validate completed Gate 8/11 pilot ratings and freeze formal power."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from volvence_zero.agent.gate811_human_anchor_analysis import (
    analyze_gate811_pilot_ratings,
    export_gate811_pilot_analysis,
    validate_gate811_analysis_preregistration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--human-anchor-preregistration", type=Path, required=True)
    parser.add_argument("--analysis-preregistration", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--rater-roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    human_bytes = args.human_anchor_preregistration.read_bytes()
    human = json.loads(human_bytes)
    analysis_bytes = args.analysis_preregistration.read_bytes()
    analysis = json.loads(analysis_bytes)
    validate_gate811_analysis_preregistration(
        analysis,
        repo_root=args.repo_root,
    )
    packet_path = args.packet_dir / "pilot_packet_blinded.json"
    key_path = args.packet_dir / "pilot_key_internal.json"
    template_path = args.packet_dir / "pilot_rating_template.csv"
    manifest_path = args.packet_dir / "manifest.json"
    packet_bytes = packet_path.read_bytes()
    key_bytes = key_path.read_bytes()
    rater_roster_bytes = args.rater_roster.read_bytes()
    report = analyze_gate811_pilot_ratings(
        human_anchor_preregistration=human,
        human_anchor_preregistration_bytes=human_bytes,
        human_anchor_preregistration_sha256=hashlib.sha256(
            human_bytes
        ).hexdigest(),
        analysis_preregistration=analysis,
        analysis_preregistration_bytes=analysis_bytes,
        analysis_preregistration_sha256=hashlib.sha256(
            analysis_bytes
        ).hexdigest(),
        packet=json.loads(packet_bytes),
        packet_bytes=packet_bytes,
        internal_key=json.loads(key_bytes),
        internal_key_bytes=key_bytes,
        packet_manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        rating_template_csv=template_path.read_text(encoding="utf-8"),
        rating_csv=args.ratings.read_text(encoding="utf-8"),
        rater_roster=json.loads(rater_roster_bytes),
        rater_roster_bytes=rater_roster_bytes,
    )
    manifest = export_gate811_pilot_analysis(
        report=report,
        output_path=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
