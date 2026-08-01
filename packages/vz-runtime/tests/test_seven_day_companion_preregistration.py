from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from volvence_zero.agent.seven_day_companion_preregistration import (
    build_seven_day_companion_preregistration,
    validate_seven_day_companion_preregistration,
    write_seven_day_companion_preregistration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _payload() -> dict[str, object]:
    return build_seven_day_companion_preregistration(
        repo_root=REPO_ROOT,
        created_at_unix_ms=1_786_032_000_000,
    )


def test_preregistration_freezes_six_scenarios_six_arms_and_real_models() -> None:
    payload = _payload()
    assert len(payload["scenario_ids"]) == 6
    assert payload["formal_run"]["run_count"] == 36
    assert payload["formal_run"]["session_count"] == 252
    assert payload["formal_run"]["exchange_count"] == 1260
    assert payload["source_requirements"][
        "deterministic_fake_allowed_in_formal"
    ] is False
    assert payload["authorization"][
        "production_promotion_authorized"
    ] is False
    source_snapshot = payload["execution_source_snapshot"]
    assert source_snapshot["file_count"] > len(payload["code_manifest"])
    assert len(source_snapshot["tree_sha256"]) == 64
    assert "packages/*/src" in source_snapshot["roots"]
    assert "scripts/run_gate811_simulated_capture.py" in source_snapshot["roots"]
    validate_seven_day_companion_preregistration(
        payload,
        repo_root=REPO_ROOT,
    )


def test_preregistration_drift_fails_loudly() -> None:
    payload = deepcopy(_payload())
    payload["minimum_effects"][
        "callback_hit_rate_gain"
    ] = 0.0
    with pytest.raises(ValueError, match="drift"):
        validate_seven_day_companion_preregistration(
            payload,
            repo_root=REPO_ROOT,
        )


def test_execution_source_tree_drift_fails_loudly() -> None:
    payload = deepcopy(_payload())
    payload["execution_source_snapshot"]["tree_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="drift"):
        validate_seven_day_companion_preregistration(
            payload,
            repo_root=REPO_ROOT,
        )


def test_preregistration_is_immutable_once_written(tmp_path: Path) -> None:
    path = tmp_path / "prereg.json"
    digest = write_seven_day_companion_preregistration(
        payload=_payload(),
        output_path=path,
    )
    assert len(digest) == 64
    with pytest.raises(FileExistsError, match="immutable"):
        write_seven_day_companion_preregistration(
            payload=_payload(),
            output_path=path,
        )
