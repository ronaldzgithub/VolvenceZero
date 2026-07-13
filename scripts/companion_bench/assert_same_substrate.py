#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Substrate-consistency guard for the same-substrate Companion Bench ablation.

The whole causal claim of the ablation rests on one invariant: every track
(``raw`` / ``ref-harness`` / ``camel`` / ``volvence-cold`` / ``volvence``) runs
on **byte-identical substrate weights**. If two tracks silently run on different
weights (e.g. a different revision pulled on a restart), a score delta is no
longer attributable to the cognitive/agent layer.

This script makes the invariant auditable. Each serve wrapper writes a tiny
``substrate_fingerprint.json`` for its track::

    {"track": "raw", "substrate_model_id": "Qwen/Qwen2.5-7B-Instruct",
     "weights_sha256": "<optional hash>", "served_at": "..."}

and this guard asserts the ``substrate_model_id`` (and ``weights_sha256`` when
present) are identical across all tracks. It fails loud on any mismatch, exactly
like ``compare_ablation.py`` refuses to emit a verdict when the substrate id is
inconsistent.

Usage::

    # From recorded fingerprint files (one per track):
    python scripts/companion_bench/assert_same_substrate.py \\
        --fingerprint-file raw=artifacts/.../raw/substrate_fingerprint.json \\
        --fingerprint-file ref-harness=artifacts/.../ref-harness/substrate_fingerprint.json \\
        ...

    # Or inline (CI smoke / quick check):
    python scripts/companion_bench/assert_same_substrate.py \\
        --fingerprint raw=Qwen/Qwen2.5-7B-Instruct \\
        --fingerprint camel=Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Any


@dataclasses.dataclass(frozen=True)
class TrackFingerprint:
    track: str
    substrate_model_id: str
    weights_sha256: str | None = None

    def key(self) -> tuple[str, str | None]:
        return (self.substrate_model_id, self.weights_sha256)


class SubstrateMismatchError(RuntimeError):
    """Raised when tracks do not share the same substrate."""


def parse_pair(spec: str) -> tuple[str, str]:
    """Parse a ``name=value`` pair (value may itself contain ``=``)."""

    if "=" not in spec:
        raise ValueError(f"expected name=value, got {spec!r}")
    name, value = spec.split("=", 1)
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise ValueError(f"both sides of name=value must be non-empty: {spec!r}")
    return name, value


def fingerprint_from_inline(track: str, model_id: str) -> TrackFingerprint:
    return TrackFingerprint(track=track, substrate_model_id=model_id)


def fingerprint_from_file(track: str, path: pathlib.Path) -> TrackFingerprint:
    if not path.exists():
        raise FileNotFoundError(f"fingerprint file not found for {track!r}: {path}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    model_id = data.get("substrate_model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(
            f"{path}: missing/invalid 'substrate_model_id' (was this written by a serve wrapper?)"
        )
    sha = data.get("weights_sha256")
    sha_str = str(sha) if isinstance(sha, str) and sha.strip() else None
    return TrackFingerprint(
        track=track, substrate_model_id=model_id.strip(), weights_sha256=sha_str,
    )


def assert_consistent(
    fingerprints: list[TrackFingerprint],
    *,
    require_weights_sha256: bool = False,
) -> TrackFingerprint:
    """Assert every fingerprint shares the same substrate. Returns the canonical.

    Raises :class:`SubstrateMismatchError` on any divergence, listing the
    offending tracks so the operator can see exactly which serve wrapper is on
    the wrong weights.
    """

    if not fingerprints:
        raise SubstrateMismatchError("no fingerprints supplied")
    if require_weights_sha256:
        missing_hashes = tuple(
            fingerprint.track
            for fingerprint in fingerprints
            if fingerprint.weights_sha256 is None
        )
        if missing_hashes:
            raise SubstrateMismatchError(
                "weights_sha256 is required for P1; missing for: "
                + ", ".join(missing_hashes)
            )
    base = fingerprints[0]
    mismatches: list[str] = []
    for fp in fingerprints[1:]:
        if fp.substrate_model_id != base.substrate_model_id:
            mismatches.append(
                f"{fp.track}: substrate_model_id={fp.substrate_model_id!r} "
                f"!= {base.track}={base.substrate_model_id!r}"
            )
            continue
        # Only compare hashes when BOTH sides recorded one (hash is optional).
        if (
            fp.weights_sha256 is not None
            and base.weights_sha256 is not None
            and fp.weights_sha256 != base.weights_sha256
        ):
            mismatches.append(
                f"{fp.track}: weights_sha256={fp.weights_sha256!r} "
                f"!= {base.track}={base.weights_sha256!r}"
            )
    if mismatches:
        raise SubstrateMismatchError(
            "same-substrate invariant VIOLATED — the ablation is not comparable:\n  "
            + "\n  ".join(mismatches)
        )
    return base


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="assert_same_substrate",
        description=(
            "Assert every ablation track shares byte-identical substrate weights. "
            "Fails loud on any mismatch."
        ),
    )
    p.add_argument(
        "--fingerprint",
        action="append",
        default=[],
        metavar="TRACK=MODEL_ID",
        help="inline fingerprint: track name = substrate model id (repeatable).",
    )
    p.add_argument(
        "--fingerprint-file",
        action="append",
        default=[],
        metavar="TRACK=PATH",
        help="path to a substrate_fingerprint.json for a track (repeatable).",
    )
    p.add_argument(
        "--require-weights-sha256",
        action="store_true",
        help="reject every track that lacks a byte-level weight digest",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    fingerprints: list[TrackFingerprint] = []
    try:
        for spec in args.fingerprint:
            track, model_id = parse_pair(spec)
            fingerprints.append(fingerprint_from_inline(track, model_id))
        for spec in args.fingerprint_file:
            track, path_str = parse_pair(spec)
            fingerprints.append(fingerprint_from_file(track, pathlib.Path(path_str)))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if len(fingerprints) < 2:
        print(
            "error: supply at least two tracks (--fingerprint / --fingerprint-file)",
            file=sys.stderr,
        )
        return 2

    try:
        canonical = assert_consistent(
            fingerprints,
            require_weights_sha256=args.require_weights_sha256,
        )
    except SubstrateMismatchError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("OK: all tracks share the same substrate")
    print(f"  substrate_model_id: {canonical.substrate_model_id}")
    if canonical.weights_sha256:
        print(f"  weights_sha256: {canonical.weights_sha256}")
    for fp in fingerprints:
        print(f"  - {fp.track}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
