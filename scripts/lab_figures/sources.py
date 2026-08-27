"""Frozen-artifact sources for evidence figures.

Every figure number is read from an artifact at render time. Missing or moved
artifacts must fail loudly rather than fall back to a literal, so that a figure
can never drift away from the evidence it claims to show.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

CODING_PACKET1 = "artifacts/coding_lab/coding_lab_observer_scripted_20260812/report.json"
CODING_PACKET2 = (
    "artifacts/coding_lab/coding_lab_packet2_formal_v2_qwen3codernext_20260813/report.json"
)
CODING_PACKET3 = "artifacts/coding_lab/coding_lab_packet3_s3e_formal_20260813/report.json"
CODING_PACKET4 = "artifacts/coding_lab/coding_lab_packet4_formal_20260813/review.json"
RELATIONSHIP_CAMPAIGN = (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_development_campaign_20260827_pbc4c0882c6b0_c73e9ed7ac298/"
    "report.json"
)
RELATIONSHIP_SOURCE_V4 = (
    "artifacts/relationship_lab/"
    "relationship_product_horizon_source_v4_admission_20260826_b3988b21/"
    "source/source_protocol.json"
)
SCENARIO_V3_TRUTH = (
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/"
    "scenario_packages/relationship_transfer_v3/generator_truth.json"
)
SCENARIO_V3_PUBLIC = (
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/"
    "scenario_packages/relationship_transfer_v3/rendered_observations.json"
)


def resolve(relative_path: str) -> pathlib.Path:
    """Resolve a repo-relative source path, failing loudly when absent."""

    path = REPO_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"evidence source missing: {relative_path!r} (looked in {path!s}). "
            "Figures must be rendered from frozen artifacts; refusing to invent numbers."
        )
    return path


def load(relative_path: str) -> dict:
    """Load one JSON artifact."""

    path = resolve(relative_path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{relative_path} must contain a JSON object")
    return payload


def digest(relative_path: str) -> str:
    """sha256 of one source artifact, for the figure sidecar."""

    return hashlib.sha256(resolve(relative_path).read_bytes()).hexdigest()


def provenance(*relative_paths: str) -> list[dict[str, str]]:
    """Sidecar provenance block: which artifacts a figure was computed from."""

    return [
        {"path": relative_path, "sha256": digest(relative_path)}
        for relative_path in relative_paths
    ]


__all__ = [
    "CODING_PACKET1",
    "CODING_PACKET2",
    "CODING_PACKET3",
    "CODING_PACKET4",
    "RELATIONSHIP_CAMPAIGN",
    "RELATIONSHIP_SOURCE_V4",
    "REPO_ROOT",
    "SCENARIO_V3_PUBLIC",
    "SCENARIO_V3_TRUTH",
    "digest",
    "load",
    "provenance",
    "resolve",
]
