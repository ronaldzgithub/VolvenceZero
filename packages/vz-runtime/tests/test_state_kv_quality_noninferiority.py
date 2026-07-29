from __future__ import annotations

import pytest

from volvence_zero.state_kv_quality_noninferiority import (
    QualityPair,
    build_quality_noninferiority_verdict,
)


def _pairs(delta: float) -> tuple[QualityPair, ...]:
    return tuple(
        QualityPair(
            experiment_id=f"exp-{index}",
            scenario_id=f"scenario-{index % 2}",
            sampling_seed=1701 + index % 3,
            judge_model_id=f"judge-{index % 2}",
            substrate_fingerprint="frozen-model@abc",
            candidate_accuracy=0.5 + delta,
            bprime_accuracy=0.5,
        )
        for index in range(8)
    )


def test_multi_axis_noninferiority_passes() -> None:
    verdict = build_quality_noninferiority_verdict(pairs=_pairs(0.1))

    assert verdict.gate_state == "pass"
    assert verdict.delta_ci == pytest.approx((0.1, 0.1))


def test_missing_judge_coverage_is_insufficient() -> None:
    pairs = tuple(
        QualityPair(
            experiment_id=pair.experiment_id,
            scenario_id=pair.scenario_id,
            sampling_seed=pair.sampling_seed,
            judge_model_id="one-judge",
            substrate_fingerprint=pair.substrate_fingerprint,
            candidate_accuracy=pair.candidate_accuracy,
            bprime_accuracy=pair.bprime_accuracy,
        )
        for pair in _pairs(0.1)
    )

    verdict = build_quality_noninferiority_verdict(pairs=pairs)

    assert verdict.gate_state == "insufficient_data"


def test_cross_substrate_pairs_fail_loudly() -> None:
    pairs = list(_pairs(0.1))
    pairs[0] = QualityPair(
        experiment_id=pairs[0].experiment_id,
        scenario_id=pairs[0].scenario_id,
        sampling_seed=pairs[0].sampling_seed,
        judge_model_id=pairs[0].judge_model_id,
        substrate_fingerprint="different",
        candidate_accuracy=pairs[0].candidate_accuracy,
        bprime_accuracy=pairs[0].bprime_accuracy,
    )

    with pytest.raises(ValueError, match="one frozen substrate"):
        build_quality_noninferiority_verdict(pairs=pairs)
