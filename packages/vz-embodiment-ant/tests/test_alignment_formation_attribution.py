"""Contract tests for the L1-A alignment-formation attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from volvence_ant.experiments.alignment_attribution import (
    AlignmentFormationAttributionError,
    classify_alignment_hypotheses,
    summarize_body_food_exposure,
)


_ROOT = Path(__file__).resolve().parents[3]


def _alignment_rows(*, failed_body_id: int, failed_margin: float) -> list[dict]:
    return [
        {
            "body_id": body_id,
            "target_aligned": body_id != failed_body_id,
            "margin_over_threshold": (
                failed_margin if body_id == failed_body_id else 0.1
            ),
        }
        for body_id in range(4)
    ]


def _geometry(*, failed_distance: float) -> dict:
    return {
        "per_body": [
            {
                "body_id": body_id,
                "distance_to_aligned_centroid": (
                    failed_distance if body_id == 2 else 0.2
                ),
                "world_policy_update_step": 10 + body_id,
            }
            for body_id in range(4)
        ]
    }


def _probe_matrix(*, failed_aligned: bool = False) -> list[dict]:
    return [
        {
            "probe_seed": seed,
            "rows": [
                {
                    "body_id": body_id,
                    "target_aligned": (
                        failed_aligned if body_id == 2 else True
                    ),
                }
                for body_id in range(4)
            ],
        }
        for seed in (700_003, 700_004)
    ]


def _exposure() -> list[dict]:
    return [
        {
            "body_id": body_id,
            "combined_encountered_food_events": (25 if body_id == 2 else 12),
            "combined_pickup_events": (11 if body_id == 2 else 5),
        }
        for body_id in range(4)
    ]


def test_selects_h1_when_failure_is_stable_exposed_and_parameter_outlier() -> None:
    result = classify_alignment_hypotheses(
        station_rows=_alignment_rows(failed_body_id=2, failed_margin=-0.4),
        review_rows=_alignment_rows(failed_body_id=2, failed_margin=-0.2),
        combined_exposure=_exposure(),
        station_geometry=_geometry(failed_distance=1.0),
        review_geometry=_geometry(failed_distance=1.2),
        station_probe_matrix=_probe_matrix(),
        review_probe_matrix=_probe_matrix(),
    )

    assert result["selected_hypothesis"] == "H1_learning_state_divergence"
    assert result["measurement_noise_kill_triggered"] is False
    assert (
        result["hypotheses"]["H2_curriculum_exposure_imbalance"]["verdict"]
        == "not-supported"
    )
    assert (
        result["hypotheses"]["H3_gradient_interference"]["verdict"]
        == "inconclusive"
    )


def test_seed_crossing_triggers_measurement_semantics_redesign() -> None:
    result = classify_alignment_hypotheses(
        station_rows=_alignment_rows(failed_body_id=2, failed_margin=-0.4),
        review_rows=_alignment_rows(failed_body_id=2, failed_margin=-0.2),
        combined_exposure=_exposure(),
        station_geometry=_geometry(failed_distance=1.0),
        review_geometry=_geometry(failed_distance=1.2),
        station_probe_matrix=_probe_matrix(failed_aligned=True),
        review_probe_matrix=_probe_matrix(),
    )

    assert result["measurement_noise_kill_triggered"] is True
    assert result["selected_hypothesis"] == "INCONCLUSIVE_REDESIGN_REQUIRED"


def test_exposure_summary_fails_loudly_on_missing_body_lineage() -> None:
    with pytest.raises(
        AlignmentFormationAttributionError,
        match="four body-lineage rows",
    ):
        summarize_body_food_exposure(
            {
                "episodes": [
                    {
                        "body_lineage": [
                            {
                                "body_id": 0,
                                "stage": "butter",
                                "tier": "near",
                                "applied_distance": 1.0,
                            }
                        ]
                    }
                ]
            }
        )


def test_formal_artifact_binds_sources_and_freezes_h1_decision() -> None:
    artifact_path = (
        _ROOT
        / "research/ant/results/ecology_recovery/same_physics_baseline/"
        "alignment_formation_attribution.v1.json"
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "alignment-formation-attribution.v1"
    assert payload["read_only"] is True
    assert payload["training_or_journal_write_performed"] is False
    assert payload["decision"]["failed_body_id"] == 2
    assert payload["decision"]["selected_hypothesis"] == (
        "H1_learning_state_divergence"
    )
    assert payload["decision"]["measurement_noise_kill_triggered"] is False
    assert payload["decision"]["station2_remains_unauthorized"] is True
    h3 = payload["decision"]["hypotheses"]["H3_gradient_interference"]
    assert h3["verdict"] == "inconclusive"
    assert h3["direct_per_update_gradient_cosine_available"] is False
    assert "failed_body_margin_change_during_butter_near_review" in h3
    assert "failed_body_margin_change_during_frozen_food_review" not in h3
    assert payload["thresholds"]["probe_seeds"] == [
        700_003,
        700_004,
        700_005,
        700_006,
        700_007,
    ]
    for source_name in ("station1_report", "review_report"):
        source = payload["sources"][source_name]
        source_path = _ROOT / source["path"]
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source[
            "sha256"
        ]
