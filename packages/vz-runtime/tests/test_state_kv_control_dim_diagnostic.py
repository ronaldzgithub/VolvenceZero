from __future__ import annotations

import pytest

from volvence_zero.state_kv_control_dim_diagnostic import (
    CONTROL_DIM_SCHEMA_VERSION,
    ControlDimSample,
    build_control_dim_verdict,
)


def _samples(delta: float) -> tuple[ControlDimSample, ...]:
    return tuple(
        ControlDimSample(
            sample_id=f"sample-{index}",
            full_code=(0.1, 0.2, 0.3, 0.4, 0.5),
            full_outcome=0.5 + delta,
            rank3_outcome=0.5,
            dynamic_off_outcome=0.45,
        )
        for index in range(8)
    )


def _verdict(samples: tuple[ControlDimSample, ...]):
    return build_control_dim_verdict(
        samples=samples,
        artifact_id="control-dim-test",
        source_artifacts=("matched-run",),
    )


def test_matched_full_dimension_increment_opens_d1_gate() -> None:
    verdict = _verdict(_samples(0.03))

    assert verdict.gate_state == "pass"
    assert verdict.bottleneck_proven is True
    assert verdict.p5d_decision == "proceed-to-full-dimension-artifact"
    assert verdict.as_json_dict()["schema_version"] == CONTROL_DIM_SCHEMA_VERSION


def test_missing_matched_arm_waits_for_evidence() -> None:
    verdict = _verdict(())

    assert verdict.gate_state == "insufficient_data"
    assert verdict.bottleneck_proven is False
    assert (
        verdict.p5d_decision
        == "await-matched-full-dimension-evidence"
    )


def test_subthreshold_full_dimension_increment_closes_gate() -> None:
    verdict = _verdict(_samples(0.01))

    assert verdict.gate_state == "fail"
    assert verdict.bottleneck_proven is False
    assert verdict.p5d_decision == "retain-rank3-stop-d1-d2"


def test_full_code_must_be_wider_than_rank3() -> None:
    with pytest.raises(ValueError, match="wider than 3"):
        ControlDimSample(
            sample_id="broken",
            full_code=(0.1, 0.2, 0.3),
            full_outcome=0.5,
            rank3_outcome=0.4,
            dynamic_off_outcome=0.3,
        )


def test_partial_three_arm_observation_is_rejected() -> None:
    with pytest.raises(ValueError, match="all three"):
        ControlDimSample(
            sample_id="partial",
            full_code=(0.1, 0.2, 0.3, 0.4),
            full_outcome=0.5,
            rank3_outcome=0.4,
        )
