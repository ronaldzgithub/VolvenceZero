"""School-coherence readout.

The product requirement is that the cultivated expert converges onto a
single coherent school ("流派") rather than a mixture of schools. In
Volvence Zero a school is NOT a regime label (regime is orthogonal — an
interaction mode); a school is the **active mixture of BehaviorProtocols**
the kernel maintains. The correct convergence signal is therefore the
concentration of the protocol ``active_mixture``: among the researched
theory-protocols, does one dominate (a converged school), or are many
co-active with spread weight (still a mixture)?

:func:`assess_protocol_coherence` is the primary readout. The legacy
:func:`assess_coherence` (regime-label concentration) is retained only as
a degraded fallback for environments where the protocol runtime is not
publishing ``active_mixture``; it measures interaction-mode stickiness,
which is a weak proxy and explicitly NOT the school.

Both are strictly readouts (R12): computed from what the kernel already
publishes, never fed back as a learning signal, and never keyword-based.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lifeform_cultivation.protocols import is_identity_core


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


@dataclass(frozen=True)
class ProtocolCoherenceAssessment:
    """School-coherence computed from the protocol active_mixture.

    ``score`` is the activation-weight share of the single dominant
    researched school-protocol (the Identity Core anchor is excluded
    from the competition, since it is always present by floor). A score
    near 1.0 means the mixture has converged onto one school; a low score
    means many theory-protocols remain co-active (still a mixture).
    """

    score: float
    dominant_protocol: str
    dominant_share: float
    distinct_schools: int
    total_protocols: int
    identity_core_present: bool
    boundary_union_size: int
    distribution: dict[str, float]

    def to_json(self) -> dict[str, object]:
        return {
            "score": self.score,
            "dominant_protocol": self.dominant_protocol,
            "dominant_share": self.dominant_share,
            "distinct_schools": self.distinct_schools,
            "total_protocols": self.total_protocols,
            "identity_core_present": self.identity_core_present,
            "boundary_union_size": self.boundary_union_size,
            "distribution": dict(self.distribution),
            "readout": "protocol_active_mixture",
        }


def assess_protocol_coherence(active_mixture: Any) -> ProtocolCoherenceAssessment:
    """Assess school convergence from an ``ActiveMixtureSnapshot``.

    ``active_mixture`` is the published snapshot value (carries
    ``active_protocols`` with ``protocol_id`` + ``activation_weight``,
    and ``boundary_union_ids``). ``None`` (protocol runtime not
    publishing) yields a zero-converged assessment so the caller can
    fall back to the legacy regime readout.
    """

    if active_mixture is None:
        return ProtocolCoherenceAssessment(
            score=0.0,
            dominant_protocol="",
            dominant_share=0.0,
            distinct_schools=0,
            total_protocols=0,
            identity_core_present=False,
            boundary_union_size=0,
            distribution={},
        )

    entries = tuple(active_mixture.active_protocols)
    identity_present = any(is_identity_core(e.protocol_id) for e in entries)
    boundary_union_size = len(tuple(active_mixture.boundary_union_ids))

    # The school competition is among the non-identity (researched)
    # protocols; the Identity Core anchor is always there by floor.
    schools = [e for e in entries if not is_identity_core(e.protocol_id)]
    distribution = {e.protocol_id: float(e.activation_weight) for e in schools}
    total_weight = sum(distribution.values())

    if not schools or total_weight <= 0.0:
        return ProtocolCoherenceAssessment(
            score=0.0,
            dominant_protocol="",
            dominant_share=0.0,
            distinct_schools=len(schools),
            total_protocols=len(entries),
            identity_core_present=identity_present,
            boundary_union_size=boundary_union_size,
            distribution=distribution,
        )

    dominant_protocol = max(distribution, key=distribution.get)
    dominant_share = distribution[dominant_protocol] / total_weight
    return ProtocolCoherenceAssessment(
        score=dominant_share,
        dominant_protocol=dominant_protocol,
        dominant_share=dominant_share,
        distinct_schools=len(schools),
        total_protocols=len(entries),
        identity_core_present=identity_present,
        boundary_union_size=boundary_union_size,
        distribution=distribution,
    )


__all__ = [
    "CoherenceAssessment",
    "ProtocolCoherenceAssessment",
    "assess_coherence",
    "assess_protocol_coherence",
]
