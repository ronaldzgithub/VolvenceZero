"""School-coherence readout.

The product requirement is that the cultivated expert converges onto a
single coherent school ("流派") rather than a mixture of schools. In the
kernel's terms (R14) a school is a persistent *regime* identity, not a
prompt label. We therefore measure convergence as the **concentration**
of the kernel's observed ``active_regime`` over the cultivation's study
turns: a high concentration on one regime means the expert is thinking
from one coherent stance; a flat distribution means it is still a
mixture.

This is strictly a readout (R12): it is computed from the regime labels
the kernel already publishes per turn and never flows back as a learning
signal. It uses no keyword matching — only the typed regime identifiers
emitted by the metacontroller.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CoherenceAssessment:
    score: float
    dominant_regime: str
    dominant_share: float
    distinct_regimes: int
    total_observations: int
    normalized_entropy: float
    distribution: dict[str, int]

    def to_json(self) -> dict[str, object]:
        return {
            "score": self.score,
            "dominant_regime": self.dominant_regime,
            "dominant_share": self.dominant_share,
            "distinct_regimes": self.distinct_regimes,
            "total_observations": self.total_observations,
            "normalized_entropy": self.normalized_entropy,
            "distribution": dict(self.distribution),
        }


def assess_coherence(regime_history: Sequence[str]) -> CoherenceAssessment:
    """Compute the school-coherence assessment from a regime sequence.

    ``score`` is the share of study turns spent in the single most
    common regime (in ``[0,1]``). An empty history yields a zero score
    (a cultivation with no observed cognition has not converged).
    """

    cleaned = [r for r in regime_history if r]
    total = len(cleaned)
    if total == 0:
        return CoherenceAssessment(
            score=0.0,
            dominant_regime="",
            dominant_share=0.0,
            distinct_regimes=0,
            total_observations=0,
            normalized_entropy=0.0,
            distribution={},
        )
    counts = Counter(cleaned)
    dominant_regime, dominant_count = counts.most_common(1)[0]
    dominant_share = dominant_count / total
    distinct = len(counts)

    # Normalised Shannon entropy over the regime distribution: 0 when all
    # turns share one regime, 1 when uniformly spread. Surfaced alongside
    # the headline concentration so the operator can see *how* mixed the
    # tail is, not just the dominant share.
    if distinct <= 1:
        normalized_entropy = 0.0
    else:
        entropy = -sum(
            (c / total) * math.log(c / total) for c in counts.values()
        )
        normalized_entropy = entropy / math.log(distinct)

    return CoherenceAssessment(
        score=dominant_share,
        dominant_regime=dominant_regime,
        dominant_share=dominant_share,
        distinct_regimes=distinct,
        total_observations=total,
        normalized_entropy=normalized_entropy,
        distribution=dict(counts),
    )


__all__ = ["CoherenceAssessment", "assess_coherence"]
