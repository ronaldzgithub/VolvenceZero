from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from volvence_zero.agent.gate811_human_anchor import (
    GATE811_HUMAN_ANCHOR_SCHEMA_VERSION,
    build_gate811_human_anchor_preregistration,
    validate_gate811_human_anchor_preregistration,
    write_gate811_human_anchor_preregistration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _payload() -> dict[str, object]:
    return build_gate811_human_anchor_preregistration(
        repo_root=REPO_ROOT,
        created_at_unix_ms=1_785_520_000_000,
    )


def test_preregistration_binds_gate8_and_gate11_matched_arms() -> None:
    payload = _payload()
    assert payload["schema_version"] == GATE811_HUMAN_ANCHOR_SCHEMA_VERSION
    sources = payload["source_bindings"]
    assert isinstance(sources, list)
    assert [(row["gate_id"], row["experimental_arm"], row["control_arm"]) for row in sources] == [
        (11, "correct-user-state", "stateless"),
        (8, "sleep-consolidation", "no-sleep"),
    ]
    assert all(len(row["manifest_sha256"]) == 64 for row in sources)


def test_preregistration_freezes_blinding_power_and_readout_discipline() -> None:
    payload = _payload()
    assert payload["pilot"]["pairs_per_contrast"] == 24
    assert payload["pilot"]["pilot_rows_excluded_from_formal"] is True
    assert payload["formal"]["minimum_pairs_per_contrast"] == 60
    assert payload["formal"]["maximum_pairs_per_contrast"] == 300
    assert payload["formal"]["minimum_preference_win_rate"] == 0.6
    assert payload["formal"]["minimum_composite_likert_delta"] == 0.35
    assert payload["authorization"]["rating_may_enter_reward_or_credit"] is False
    assert payload["authorization"]["production_promotion_authorized"] is False
    hidden = payload["blinding"]["hidden_fields"]
    assert "arm_label" in hidden
    assert "expected_winner" in hidden
    assert payload["blinding"]["llm_judge_is_human_anchor"] is False


def test_validation_rejects_posthoc_threshold_change() -> None:
    payload = deepcopy(_payload())
    payload["formal"]["minimum_preference_win_rate"] = 0.5
    with pytest.raises(ValueError, match="preregistration drift"):
        validate_gate811_human_anchor_preregistration(
            payload,
            repo_root=REPO_ROOT,
        )


def test_write_round_trip_binds_preregistration_hash(tmp_path: Path) -> None:
    payload = _payload()
    output = tmp_path / "gate811_human_anchor_prereg.json"
    manifest = write_gate811_human_anchor_preregistration(
        payload=payload,
        output_path=output,
    )
    restored = json.loads(output.read_text(encoding="utf-8"))
    validate_gate811_human_anchor_preregistration(
        restored,
        repo_root=REPO_ROOT,
    )
    assert len(manifest["preregistration_sha256"]) == 64
    assert manifest["production_promotion_authorized"] is False
