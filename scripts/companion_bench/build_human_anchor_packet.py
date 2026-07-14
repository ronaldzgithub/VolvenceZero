#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Build a blinded human-anchor review packet from ablation transcripts.

Plan §14 / frozen registry evidence gate: at least 3 blinded raters review
matched transcripts from the comparison arms with system identity hidden.
This tool takes per-track transcript directories, strips every identity /
label field the protocol hides, shuffles items deterministically, and emits:

* ``human_anchor_packet.json``  the blinded items raters see (no track
  names, no profile labels, no expected labels), plus the rating protocol
  block (rater count, agreement threshold, hidden fields).
* ``human_anchor_key.json``     the unblinding key (item id -> track /
  source path). Keep this OUT of the rater-facing distribution.

Human ratings are evaluation readouts ONLY (R12): they never feed a reward
or learning path.

    python scripts/companion_bench/build_human_anchor_packet.py \
        --track volvence=artifacts/companion-ablation/<date>/volvence/transcripts \
        --track raw=artifacts/companion-ablation/<date>/raw/transcripts \
        --seed 7 --output-dir artifacts/companion-ablation/<date>/human-anchor
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys

_SCHEMA_VERSION = "human-anchor-packet.v1"

# Identity / label keys hidden from raters (protocol hidden_fields plus the
# transport-level fields that would leak the same information).
_HIDDEN_KEYS = frozenset(
    {
        "profile_label",
        "system_identity",
        "expected_label",
        "track",
        "system_name",
        "submission_id",
        "model_identifier",
        "vertical",
        "base_url",
    }
)


def _strip_identity(value):
    """Recursively remove hidden keys from a JSON payload."""

    if isinstance(value, dict):
        return {
            key: _strip_identity(item)
            for key, item in value.items()
            if key not in _HIDDEN_KEYS
        }
    if isinstance(value, list):
        return [_strip_identity(item) for item in value]
    return value


def _parse_track(raw: str) -> tuple[str, pathlib.Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--track expects NAME=TRANSCRIPTS_DIR, got {raw!r}"
        )
    name, _, path = raw.partition("=")
    return name.strip(), pathlib.Path(path.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        action="append",
        type=_parse_track,
        required=True,
        metavar="NAME=DIR",
        help="Comparison arm name and its transcript directory (repeatable).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-items-per-track", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/human_anchor"),
    )
    parser.add_argument("--blinded-rater-count", type=int, default=3)
    parser.add_argument("--min-inter-rater-agreement", type=float, default=0.6)
    args = parser.parse_args(argv)

    if len(args.track) < 2:
        print(
            "error: a human-anchor packet needs at least two comparison arms",
            file=sys.stderr,
        )
        return 2
    if args.blinded_rater_count < 3:
        print("error: protocol requires >= 3 blinded raters", file=sys.stderr)
        return 2

    items: list[dict] = []
    key_entries: list[dict] = []
    for track_name, transcripts_dir in args.track:
        if not transcripts_dir.is_dir():
            print(
                f"error: transcripts dir for {track_name!r} does not exist: "
                f"{transcripts_dir}",
                file=sys.stderr,
            )
            return 2
        paths = sorted(transcripts_dir.glob("*.json"))
        if not paths:
            print(
                f"error: no *.json transcripts under {transcripts_dir} "
                f"for track {track_name!r}",
                file=sys.stderr,
            )
            return 2
        if args.max_items_per_track is not None:
            paths = paths[: args.max_items_per_track]
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            item_id = hashlib.sha256(
                f"{args.seed}:{track_name}:{path.name}".encode("utf-8")
            ).hexdigest()[:16]
            items.append(
                {
                    "item_id": item_id,
                    "transcript": _strip_identity(payload),
                }
            )
            key_entries.append(
                {
                    "item_id": item_id,
                    "track": track_name,
                    "source_path": str(path),
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(items)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet = {
        "schema_version": _SCHEMA_VERSION,
        "protocol": {
            "blinded_rater_count": args.blinded_rater_count,
            "min_inter_rater_agreement": args.min_inter_rater_agreement,
            "hidden_fields": sorted(_HIDDEN_KEYS),
            "readout_only": True,
            "note": (
                "Human ratings are evaluation readouts only (R12); they are "
                "never a reward or learning source."
            ),
        },
        "seed": args.seed,
        "item_count": len(items),
        "items": items,
    }
    packet_path = args.output_dir / "human_anchor_packet.json"
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    key_path = args.output_dir / "human_anchor_key.json"
    key_path.write_text(
        json.dumps(
            {
                "schema_version": _SCHEMA_VERSION,
                "seed": args.seed,
                "entries": key_entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[human-anchor] blinded packet -> {packet_path} ({len(items)} items)")
    print(f"[human-anchor] unblinding key -> {key_path} (do NOT ship to raters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
