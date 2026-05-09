"""Wave E5 evidence bundle assembler contract tests.

Pins the typed verdict schema produced by
``lifeform_evolution.evidence_bundle.assemble_bundle``: a JSON-
serialisable dict with ``bundle_id``, ``gates`` (list of typed
gate verdicts), ``artifact_provenance`` (sha256 + size), and
``overall_passed``.

These tests do NOT require a real LLM run. They build synthetic
longitudinal artifacts on disk, exercise the assembler, and
verify the resulting manifest's typed shape and gate logic.
"""

from __future__ import annotations

import json
import pathlib

from lifeform_evolution.evidence_bundle import assemble_bundle


def _write_long_form_artifact(
    path: pathlib.Path,
    *,
    scenario_id: str,
    tom_records_total_last: int,
    cg_dyad_atoms_total_last: int,
    il_rapport_trend_snr_mean: float,
    pe_window_filled_scenario_ratio: float,
) -> None:
    payload = {
        "scenarios": [
            {
                "scenario_id": scenario_id,
                "tom_records_total_last": tom_records_total_last,
                "common_ground_dyad_atoms_total_last": cg_dyad_atoms_total_last,
                "rounds": 5,
                "passed": True,
            }
        ],
        "cross_scenario_summary": {
            "scenario_count": 1,
            "il_rapport_trend_snr_mean": il_rapport_trend_snr_mean,
            "pe_window_filled_scenario_ratio": pe_window_filled_scenario_ratio,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_assemble_bundle_with_no_artifacts_reports_failure(tmp_path):
    bundle = assemble_bundle(bundle_dir=tmp_path)
    assert bundle["overall_passed"] is False
    assert bundle["bundle_id"] == "eq_evidence_chain_closure"
    assert bundle["long_form_scenarios_present"] == []
    gate_ids = {gate["gate_id"] for gate in bundle["gates"]}
    assert "debt_10b_item3" in gate_ids
    assert "debt_10c_il_rapport_snr" in gate_ids
    assert "debt_11_long_form_coverage" in gate_ids


def test_assemble_bundle_closes_debt_10b_when_records_present(tmp_path):
    _write_long_form_artifact(
        tmp_path / "long-form-life-arc_longitudinal.json",
        scenario_id="long-form-life-arc",
        tom_records_total_last=8,
        cg_dyad_atoms_total_last=3,
        il_rapport_trend_snr_mean=2.1,
        pe_window_filled_scenario_ratio=0.75,
    )
    bundle = assemble_bundle(bundle_dir=tmp_path)
    by_id = {gate["gate_id"]: gate for gate in bundle["gates"]}
    assert by_id["debt_10b_item3"]["passed"] is True
    assert by_id["debt_10c_il_rapport_snr"]["passed"] is True
    assert by_id["debt_11_long_form_coverage"]["passed"] is True


def test_assemble_bundle_keeps_debt_open_when_thresholds_unmet(tmp_path):
    _write_long_form_artifact(
        tmp_path / "long-form-life-arc_longitudinal.json",
        scenario_id="long-form-life-arc",
        tom_records_total_last=0,  # debt 10B stays open
        cg_dyad_atoms_total_last=0,
        il_rapport_trend_snr_mean=0.40,  # debt 10C stays open
        pe_window_filled_scenario_ratio=0.20,  # debt 11 stays open
    )
    bundle = assemble_bundle(
        bundle_dir=tmp_path,
        snr_threshold=1.5,
        pe_window_ratio_threshold=0.5,
    )
    by_id = {gate["gate_id"]: gate for gate in bundle["gates"]}
    assert by_id["debt_10b_item3"]["passed"] is False
    assert by_id["debt_10c_il_rapport_snr"]["passed"] is False
    assert by_id["debt_11_long_form_coverage"]["passed"] is False
    assert bundle["overall_passed"] is False


def test_assemble_bundle_records_artifact_sha256(tmp_path):
    artifact_path = tmp_path / "long-form-task-arc_longitudinal.json"
    _write_long_form_artifact(
        artifact_path,
        scenario_id="long-form-task-arc",
        tom_records_total_last=5,
        cg_dyad_atoms_total_last=2,
        il_rapport_trend_snr_mean=1.6,
        pe_window_filled_scenario_ratio=0.5,
    )
    bundle = assemble_bundle(bundle_dir=tmp_path)
    provenance = {
        record["scenario_id"]: record for record in bundle["artifact_provenance"]
    }
    assert "long-form-task-arc" in provenance
    assert "sha256" in provenance["long-form-task-arc"]
    # 64-char hex digest
    assert len(provenance["long-form-task-arc"]["sha256"]) == 64
    assert provenance["long-form-task-arc"]["size_bytes"] > 0


def test_rollback_drill_gates_track_test_file_presence(tmp_path):
    bundle = assemble_bundle(bundle_dir=tmp_path)
    by_id = {gate["gate_id"]: gate for gate in bundle["gates"]}
    # The rollback drill test ships in this very repo, so the gate
    # passes regardless of the bundle dir.
    assert by_id["debt_6_rewarding_state_head_promotion"]["passed"] is True
    assert by_id["debt_7_pe_critic_head_promotion"]["passed"] is True
