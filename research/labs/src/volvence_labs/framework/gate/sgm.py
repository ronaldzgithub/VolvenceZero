"""SGM e-value gate: sequentially-valid statistical test for promotion.

Implements an anytime-valid e-process (Ramdas & Wang 2023) that accumulates
evidence across seeds/runs. The e-value grows when the probe consistently
outperforms baseline; it shrinks (toward 1) when evidence is weak.

Key property: at any stopping time, if e-value > 1/α, we can reject H0
(probe is no better than baseline) at level α. This is valid regardless
of when we choose to stop — no p-hacking possible.

References:
- Ramdas, Grünwald, Vovk, Shafer (2023) "Game-Theoretic Statistics and Safe Anytime-Valid Inference"
- Howard, Ramdas, McAuliffe, Sekhon (2021) "Time-uniform, nonparametric, nonasymptotic confidence sequences"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EValue:
    """Running e-value state.

    value: current e-value (starts at 1.0)
    n_updates: number of evidence updates incorporated
    log_value: log(e-value) for numerical stability
    history: list of (update_idx, log_increment, cumulative_log_value)
    """
    value: float = 1.0
    n_updates: int = 0
    log_value: float = 0.0
    history: list[tuple[int, float, float]] = field(default_factory=list)

    def update(self, log_increment: float) -> None:
        """Incorporate new evidence. log_increment > 0 means evidence for H1."""
        self.log_value += log_increment
        self.value = math.exp(min(self.log_value, 700))  # prevent overflow
        self.n_updates += 1
        self.history.append((self.n_updates, log_increment, self.log_value))

    def exceeds_threshold(self, alpha: float = 0.05) -> bool:
        """Check if e-value exceeds 1/alpha (reject H0 at level alpha)."""
        return self.value > 1.0 / alpha

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "log_value": self.log_value,
            "n_updates": self.n_updates,
            "history_len": len(self.history),
        }


class SGMGate:
    """Sequential e-value gate using sub-Gaussian mixture (SGM) martingale.

    For each new (probe_on, baseline) pair of observations, computes a
    likelihood ratio under H1 (probe is better by at least delta) vs H0
    (probe is same or worse). The product of these ratios is the e-value.

    The SGM uses a Hoeffding-style sub-Gaussian bound:
        log_increment = lambda * (X - mu0) - lambda^2 * sigma^2 / 2

    where:
        X = probe_on_value - baseline_value (observed difference)
        mu0 = 0 (null: no difference)
        lambda = tuning parameter (set to delta / sigma^2 for optimal power)
        sigma = assumed sub-Gaussian parameter (range / 2 for bounded data)
    """

    def __init__(
        self,
        *,
        delta: float = 0.01,
        sigma: float = 1.0,
        alpha: float = 0.05,
        lambda_scale: float = 1.0,
    ):
        """
        Args:
            delta: minimum meaningful effect size under H1
            sigma: sub-Gaussian parameter (controls how much we trust each observation)
            alpha: significance level (e-value must exceed 1/alpha to reject H0)
            lambda_scale: multiplier for the optimal lambda (1.0 = optimal, <1 = conservative)
        """
        self.delta = delta
        self.sigma = sigma
        self.alpha = alpha
        # Optimal lambda for detecting effect of size delta
        self._lambda = lambda_scale * delta / (sigma ** 2)
        self._e_value = EValue()

    @property
    def e_value(self) -> EValue:
        return self._e_value

    def update(self, probe_on_value: float, baseline_value: float) -> float:
        """Incorporate one (probe_on, baseline) observation pair.

        Returns the log-increment (positive = evidence for H1).
        """
        diff = probe_on_value - baseline_value
        # SGM log-likelihood ratio increment
        log_inc = self._lambda * diff - (self._lambda ** 2) * (self.sigma ** 2) / 2.0
        self._e_value.update(log_inc)
        return log_inc

    def update_batch(self, probe_on_values: list[float], baseline_values: list[float]) -> None:
        """Incorporate multiple paired observations."""
        for p, b in zip(probe_on_values, baseline_values):
            self.update(p, b)

    def decision(self) -> str:
        """Current decision: 'approve' if e-value > 1/alpha, else 'hold'."""
        if self._e_value.exceeds_threshold(self.alpha):
            return "approve"
        return "hold"

    def reset(self) -> None:
        """Reset e-value to 1.0 (start fresh)."""
        self._e_value = EValue()

    def summary(self) -> dict[str, Any]:
        return {
            "e_value": self._e_value.value,
            "log_e_value": self._e_value.log_value,
            "n_updates": self._e_value.n_updates,
            "threshold": 1.0 / self.alpha,
            "decision": self.decision(),
            "delta": self.delta,
            "sigma": self.sigma,
            "lambda": self._lambda,
        }


class HoeffdingBound:
    """Hoeffding anytime-valid confidence bound.

    Provides a confidence sequence for the mean difference (probe - baseline).
    At any time n, the bound is:
        |mean_diff - true_diff| <= sqrt(log(2/alpha) / (2n)) * range

    This is tighter than Chebyshev for bounded random variables.
    """

    def __init__(self, *, range_bound: float = 2.0, alpha: float = 0.05):
        self.range_bound = range_bound
        self.alpha = alpha
        self._diffs: list[float] = []

    def update(self, probe_on_value: float, baseline_value: float) -> None:
        self._diffs.append(probe_on_value - baseline_value)

    @property
    def n(self) -> int:
        return len(self._diffs)

    @property
    def mean_diff(self) -> float:
        if not self._diffs:
            return 0.0
        return sum(self._diffs) / len(self._diffs)

    def confidence_width(self) -> float:
        """Current confidence interval half-width."""
        if self.n == 0:
            return float("inf")
        return self.range_bound * math.sqrt(math.log(2.0 / self.alpha) / (2.0 * self.n))

    def lower_bound(self) -> float:
        """Lower confidence bound on true mean difference."""
        return self.mean_diff - self.confidence_width()

    def upper_bound(self) -> float:
        """Upper confidence bound on true mean difference."""
        return self.mean_diff + self.confidence_width()

    def is_significant(self, delta: float = 0.0) -> bool:
        """Is the lower bound above delta? (i.e., probe is significantly better)"""
        return self.lower_bound() > delta
