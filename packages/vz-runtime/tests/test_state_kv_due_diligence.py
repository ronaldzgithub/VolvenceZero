from __future__ import annotations

import json
from pathlib import Path

import pytest

from volvence_zero.state_kv_due_diligence import (
    FREEZE_MANIFEST_SCHEMA_VERSION,
    build_due_diligence_report,
    build_freeze_manifest,
    freeze_manifest_from_json,
    verify_frozen_evidence,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prefix_manifest(root: Path) -> str:
    path = root / "prefix.json"
    _write_json(
        path,
        {
            "model_id": "frozen-model",
            "weights_sha256": "a" * 64,
            "artifact_id": "prefix-artifact",
        },
    )
    return "prefix.json"


def _evidence(root: Path) -> dict[str, str]:
    payloads = {
        "retention": {"schema_version": "retention.v1", "gate_state": "pass"},
        "cost": {"schema_version": "cost.v1", "gate_state": "pass"},
        "judge_court": {"schema_version": "court.v1", "gate_state": "pass"},
        "quality_noninferiority": {
            "schema_version": "quality.v1",
            "gate_state": "pass",
            "claims": [
                {
                    "claim": "claim_quality_noninferior_to_bprime",
                    "state": "pass",
                }
            ],
        },
        "carrier_diagnostics": {
            "schema_version": "carrier.v1",
            "prefix_artifact_id": "prefix-artifact",
            "carrier_is_live": False,
            "claims": [
                {
                    "claim": "claim_slot_attention_read",
                    "state": "fail",
                }
            ],
        },
        "temporal_causal": {
            "schema_version": "temporal.v1",
            "gate_state": "pass",
        },
        "control_dim": {
            "schema_version": "control.v1",
            "gate_state": "insufficient_data",
        },
        "credit_longitudinal": {
            "schema_version": "credit-longitudinal.v1",
            "gate_state": "mechanism_supported",
            "claims": [
                {
                    "claim": "claim_credit_feedback_applied_increment_grows",
                    "state": "pass",
                },
                {
                    "claim": "claim_credit_feedback_improves_matched_outcome",
                    "state": "insufficient_data",
                },
            ],
        },
        "bank_gain": {
            "schema_version": "bank.v1",
            "gate_state": "insufficient_data",
        },
        "deployment": {
            "schema_version": "deployment.v1",
            "gate_state": "pass",
        },
        "generation_seed": {
            "schema_version": "seed.v1",
            "gate_state": "pass",
        },
        "safety_negatives": {
            "schema_version": "safety-negatives.v1",
            "gate_state": "pass",
            "claims": [
                {
                    "claim": "claim_stale_conditioning_is_inert",
                    "state": "pass",
                },
                {
                    "claim": "claim_latent_state_resists_output_extraction",
                    "state": "pass",
                },
            ],
        },
        "identification": {
            "schema_version": "identification.v1",
            "verdict_state": "retain-strict",
            "candidate_arm": "state-kv-arm-g-prefix-pure",
            "matching": [
                {
                    "arm": "state-kv-arm-g-prefix-pure",
                    "accuracy": 1.0,
                },
                {
                    "arm": "state-kv-arm-bprime",
                    "accuracy": 0.5,
                },
            ],
        },
        "five_arm_identification": {
            "schema_version": "identification.v1",
            "verdict_state": "retain-strict",
            "candidate_arm": "state-kv-arm-g-prefix-pure",
            "matching": [
                {
                    "arm": "state-kv-arm-e-pure",
                    "accuracy": 0.5,
                    "ci_low": 0.25,
                    "ci_high": 0.75,
                }
            ],
        },
    }
    result = {}
    for evidence_id, payload in payloads.items():
        relative = f"{evidence_id}.json"
        _write_json(root / relative, payload)
        result[evidence_id] = relative
    return result


def _manifest(root: Path):
    return build_freeze_manifest(
        repo_root=root,
        prefix_manifest_path=_prefix_manifest(root),
        evidence_paths=_evidence(root),
        profile_labels=("arm-a", "arm-g"),
        generation_seeds=(1, 2, 3),
        scenario_sets=("heldout",),
        metric_definitions=("matching-ci",),
        judge_panel=("judge-a", "judge-b"),
        experiment_config={
            "generation": {"temperature": 0.0, "max_new_tokens": 8},
            "resolved_profiles": {"arm-a": {"personal_conditioning": "shadow"}},
        },
    )


def test_freeze_manifest_validates_against_json_schema(
    tmp_path: Path,
) -> None:
    import jsonschema

    manifest = _manifest(tmp_path)
    schema_path = (
        Path(__file__).parents[1]
        / "src/volvence_zero/schemas/state_kv_freeze_manifest.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    jsonschema.validate(manifest.as_json_dict(), schema)
    assert (
        manifest.as_json_dict()["schema_version"]
        == FREEZE_MANIFEST_SCHEMA_VERSION
    )


def test_frozen_evidence_fingerprint_detects_mutation(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    retention = tmp_path / "retention.json"
    retention.write_text('{"schema_version":"changed.v1"}', encoding="utf-8")

    with pytest.raises(ValueError, match="frozen evidence changed"):
        verify_frozen_evidence(repo_root=tmp_path, manifest=manifest)


def test_loaded_manifest_rejects_tampered_canonical_payload(
    tmp_path: Path,
) -> None:
    payload = _manifest(tmp_path).as_json_dict()
    payload["scenario_sets"] = ["changed"]

    with pytest.raises(ValueError, match="freeze_id"):
        freeze_manifest_from_json(payload)


def test_loaded_manifest_rejects_non_object_evidence_entry(
    tmp_path: Path,
) -> None:
    payload = _manifest(tmp_path).as_json_dict()
    payload["evidence"].append("broken")

    with pytest.raises(TypeError, match="evidence entries"):
        freeze_manifest_from_json(payload)


def test_due_diligence_marks_only_directly_supported_claim_proven(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    report = build_due_diligence_report(
        repo_root=tmp_path,
        manifest=manifest,
    )

    states = {
        conclusion.conclusion_id: conclusion.state
        for conclusion in report.conclusions
    }
    assert report.gate_state == "partial"
    assert states["C2"] == "proven"
    assert states["C1"] == "not-yet-proven"
    assert states["C3"] == "not-yet-proven"
    assert states["C6"] == "proven"
    assert states["C7"] == "not-yet-proven"


def test_due_diligence_c6_fails_when_extraction_negative_fails(
    tmp_path: Path,
) -> None:
    prefix_manifest_path = _prefix_manifest(tmp_path)
    evidence_paths = _evidence(tmp_path)
    safety_path = tmp_path / evidence_paths["safety_negatives"]
    payload = json.loads(safety_path.read_text(encoding="utf-8"))
    payload["gate_state"] = "fail"
    payload["claims"][1]["state"] = "fail"
    _write_json(safety_path, payload)
    manifest = build_freeze_manifest(
        repo_root=tmp_path,
        prefix_manifest_path=prefix_manifest_path,
        evidence_paths=evidence_paths,
        profile_labels=("arm-a", "arm-g"),
        generation_seeds=(1, 2, 3),
        scenario_sets=("heldout",),
        metric_definitions=("matching-ci",),
        judge_panel=("judge-a", "judge-b"),
        experiment_config={"generation": {"max_new_tokens": 8}},
    )

    report = build_due_diligence_report(repo_root=tmp_path, manifest=manifest)

    states = {
        conclusion.conclusion_id: conclusion.state
        for conclusion in report.conclusions
    }
    assert states["C6"] == "not-yet-proven"


def test_due_diligence_c3_requires_live_matching_prefix_and_five_arm_control(
    tmp_path: Path,
) -> None:
    prefix_manifest_path = _prefix_manifest(tmp_path)
    evidence_paths = _evidence(tmp_path)
    carrier_path = tmp_path / evidence_paths["carrier_diagnostics"]
    _write_json(
        carrier_path,
        {
            "schema_version": "carrier.v1",
            "prefix_artifact_id": "prefix-artifact",
            "carrier_is_live": True,
            "claims": [
                {
                    "claim": "claim_slot_attention_read",
                    "state": "pass",
                }
            ],
        },
    )
    manifest = build_freeze_manifest(
        repo_root=tmp_path,
        prefix_manifest_path=prefix_manifest_path,
        evidence_paths=evidence_paths,
        profile_labels=("arm-a", "arm-e", "arm-g"),
        generation_seeds=(1, 2, 3),
        scenario_sets=("heldout",),
        metric_definitions=("matching-ci",),
        judge_panel=("judge-a", "judge-b"),
        experiment_config={"generation": {"max_new_tokens": 8}},
    )

    report = build_due_diligence_report(
        repo_root=tmp_path,
        manifest=manifest,
    )

    states = {
        conclusion.conclusion_id: conclusion.state
        for conclusion in report.conclusions
    }
    assert states["C3"] == "proven"
