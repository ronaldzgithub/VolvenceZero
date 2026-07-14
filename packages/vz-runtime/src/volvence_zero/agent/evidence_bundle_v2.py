"""Unified evidence bundle v2 aggregator (CP-00 / GAP-10).

Before this module the repo had four parallel frozen-bundle entrypoints
(dialogue paper-suite, ETA paper-suite, EQ longitudinal, learned-shadow
smoke/soak) plus the CompanionBench P1 run manifest — each with its own
provenance shape and no single artifact an external reviewer could recompute
from. ``assemble_evidence_bundle_v2`` aggregates the sub-bundles the operator
requests into ONE manifest with:

* a shared git/runtime provenance block,
* per-input sha256 + size fingerprints,
* fail-loud validation: a REQUESTED input that is missing on disk, is not
  valid JSON, or lacks its lane's required provenance keys raises
  ``EvidenceBundleV2Error`` instead of emitting a partial bundle.

This module aggregates and validates; it never re-derives verdicts from the
sub-bundles (each lane's owner keeps its own gate semantics).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib
import platform as _platform
import subprocess
import sys
from typing import Any

EVIDENCE_BUNDLE_V2_SCHEMA_VERSION = "evidence-bundle.v2"

#: Known input lanes and the top-level keys their payloads must carry for
#: the aggregate to count as provenance-complete. A lane missing its keys
#: fails loudly — schema drift must not be papered over.
_LANE_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "dialogue_paper_suite": ("provenance",),
    "eta_paper_suite": ("provenance",),
    "eq_longitudinal": ("artifact_provenance", "provenance"),
    "learned_shadow": ("schema_version", "backend_wiring"),
    "companion_p1_manifest": ("git_sha", "tracks"),
}


class EvidenceBundleV2Error(RuntimeError):
    """A requested input is missing, unreadable, or provenance-incomplete."""


def _git_output(args: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            ("git",) + args, check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _collect_provenance() -> dict[str, Any]:
    status = _git_output(("status", "--porcelain"))
    return {
        "git_sha": _git_output(("rev-parse", "HEAD")),
        "git_branch": _git_output(("branch", "--show-current")),
        "working_tree_dirty": status not in {"", "unknown"},
        "python_version": sys.version.split()[0],
        "platform": _platform.platform(),
    }


def _file_fingerprint(path: pathlib.Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _load_lane_payload(lane: str, path: pathlib.Path) -> dict[str, Any]:
    if lane not in _LANE_REQUIRED_KEYS:
        raise EvidenceBundleV2Error(
            f"unknown evidence lane {lane!r}; known lanes: "
            f"{sorted(_LANE_REQUIRED_KEYS)}"
        )
    if not path.is_file():
        raise EvidenceBundleV2Error(
            f"requested {lane} input does not exist: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceBundleV2Error(
            f"{lane} input at {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvidenceBundleV2Error(
            f"{lane} input at {path} must be a JSON object, got "
            f"{type(payload).__name__}"
        )
    missing = tuple(
        key for key in _LANE_REQUIRED_KEYS[lane] if key not in payload
    )
    if missing:
        raise EvidenceBundleV2Error(
            f"{lane} input at {path} lacks required provenance keys "
            f"{missing}; refusing to aggregate a provenance-incomplete lane."
        )
    return payload


def assemble_evidence_bundle_v2(
    *,
    inputs: dict[str, pathlib.Path],
    bundle_id: str = "evidence-bundle-v2",
) -> dict[str, Any]:
    """Aggregate the requested lane artifacts into one v2 manifest.

    ``inputs`` maps lane name -> artifact path. At least one lane is
    required; every requested lane is validated fail-loud. The manifest
    embeds each lane's payload verbatim (no verdict re-derivation) plus a
    fingerprint, so a reviewer can verify byte-level integrity and re-open
    any lane's own artifact.
    """

    if not inputs:
        raise EvidenceBundleV2Error(
            "evidence bundle v2 requires at least one input lane"
        )
    lanes: dict[str, Any] = {}
    for lane, path in sorted(inputs.items()):
        path = pathlib.Path(path)
        payload = _load_lane_payload(lane, path)
        lanes[lane] = {
            "fingerprint": _file_fingerprint(path),
            "payload": payload,
        }
    return {
        "schema_version": EVIDENCE_BUNDLE_V2_SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "provenance": _collect_provenance(),
        "lanes": lanes,
        "lane_names": sorted(lanes),
        "description": (
            "Unified evidence bundle v2: byte-fingerprinted aggregation of "
            "the requested evidence lanes under one git/runtime provenance "
            "block. Lane verdicts are owned by the lane artifacts; this "
            "manifest never re-derives them."
        ),
    }


__all__ = [
    "EVIDENCE_BUNDLE_V2_SCHEMA_VERSION",
    "EvidenceBundleV2Error",
    "assemble_evidence_bundle_v2",
]
