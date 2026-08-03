from __future__ import annotations

import math
import statistics

import pytest

from volvence_zero.agent.evidence_statistics import paired_student_t_ci95


def test_paired_student_t_ci_has_no_single_pair_interval() -> None:
    assert paired_student_t_ci95(()) is None
    assert paired_student_t_ci95((0.25,)) is None


def test_paired_student_t_ci_uses_df17_critical_value() -> None:
    values = tuple(float(index) / 100.0 for index in range(18))
    observed = paired_student_t_ci95(values)
    assert observed is not None
    mean = statistics.fmean(values)
    half_width = 2.110 * statistics.stdev(values) / math.sqrt(18)
    assert observed == pytest.approx(
        (mean - half_width, mean + half_width),
        rel=1e-12,
    )


def test_paired_student_t_ci_fails_closed_beyond_frozen_table() -> None:
    with pytest.raises(ValueError, match="at most 31"):
        paired_student_t_ci95(tuple(float(index) for index in range(32)))
