"""CP-00 / GAP-10: unified evidence bundle v2 aggregation contract."""

from __future__ import annotations

import json
import pathlib

import pytest

from volvence_zero.agent.evidence_bundle_v2 import (
    EVIDENCE_BUNDLE_V2_SCHEMA_VERSION,
    EvidenceBundleV2Error,
    assemble_evidence_bundle_v2,
)


def _write(path: pathlib.Path, payload: dict) -> pathlib.Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregates_requested_lanes_with_fingerprints(tmp_path: pathlib.Path) -> None:
    shadow = _write(
        tmp_path / "learned_shadow_soak.json",
        {
            "schema_version": "learned-shadow-soak.v1",
            "backend_wiring": {"temporal_ssl_backend": "shadow"},
        },
    )
    eq = _write(
        tmp_path / "evidence_bundle.json",
        {
            "artifact_provenance": [],
            "provenance": {"git_sha": "abc"},
        },
    )

    bundle = assemble_evidence_bundle_v2(
        inputs={"learned_shadow": shadow, "eq_longitudinal": eq}
    )

    assert bundle["schema_version"] == EVIDENCE_BUNDLE_V2_SCHEMA_VERSION
    assert bundle["lane_names"] == ["eq_longitudinal", "learned_shadow"]
    assert set(bundle["provenance"]) >= {
        "git_sha",
        "git_branch",
        "working_tree_dirty",
    }
    for lane in bundle["lane_names"]:
        fingerprint = bundle["lanes"][lane]["fingerprint"]
        assert len(fingerprint["sha256"]) == 64
        assert fingerprint["size_bytes"] > 0


def test_missing_input_fails_loudly(tmp_path: pathlib.Path) -> None:
    with pytest.raises(EvidenceBundleV2Error, match="does not exist"):
        assemble_evidence_bundle_v2(
            inputs={"learned_shadow": tmp_path / "nope.json"}
        )


def test_provenance_incomplete_lane_fails_loudly(tmp_path: pathlib.Path) -> None:
    bad = _write(tmp_path / "bad.json", {"schema_version": "x"})  # no backend_wiring
    with pytest.raises(EvidenceBundleV2Error, match="required provenance keys"):
        assemble_evidence_bundle_v2(inputs={"learned_shadow": bad})


def test_unknown_lane_and_empty_inputs_fail_loudly(tmp_path: pathlib.Path) -> None:
    with pytest.raises(EvidenceBundleV2Error, match="at least one input"):
        assemble_evidence_bundle_v2(inputs={})
    some = _write(tmp_path / "x.json", {"a": 1})
    with pytest.raises(EvidenceBundleV2Error, match="unknown evidence lane"):
        assemble_evidence_bundle_v2(inputs={"mystery_lane": some})
