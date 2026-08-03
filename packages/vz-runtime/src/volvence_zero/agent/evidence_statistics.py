"""Small-sample statistics shared by confirmatory companion evidence.

The seven-day campaigns are intentionally small (currently six or eighteen
matched pairs), so a normal 1.96 multiplier is not an admissible substitute
for the preregistered two-sided Student-t interval.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence


# Two-sided 95% Student-t critical values, t_(0.975, df).  Every currently
# preregistered seven-day matrix has df <= 17.  Keeping the complete 1..30
# table makes the implementation deterministic and dependency-free while
# leaving room for a larger matched matrix.
_T_975_BY_DF = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)

PAIRED_STUDENT_T_95_METHOD = (
    "paired two-sided Student-t 95% interval; fixed t_(0.975, df) table; "
    "n<2 has no interval"
)


def paired_student_t_ci95(
    values: Sequence[float],
) -> tuple[float, float] | None:
    """Return a paired two-sided 95% Student-t interval.

    One observation cannot estimate sampling variance and therefore has no
    confidence interval.  Values beyond the frozen table are rejected rather
    than silently changing statistical methods.
    """

    count = len(values)
    if count < 2:
        return None
    degrees_of_freedom = count - 1
    if degrees_of_freedom > len(_T_975_BY_DF):
        raise ValueError(
            "paired Student-t CI supports at most 31 observations; "
            f"got {count}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("paired Student-t CI values must be finite")
    mean = statistics.fmean(values)
    critical = _T_975_BY_DF[degrees_of_freedom - 1]
    half_width = critical * statistics.stdev(values) / math.sqrt(count)
    return (mean - half_width, mean + half_width)


__all__ = ("PAIRED_STUDENT_T_95_METHOD", "paired_student_t_ci95")
