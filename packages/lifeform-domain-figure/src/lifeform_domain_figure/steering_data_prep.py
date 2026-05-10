"""Convert reviewed contrast pairs into steering training plans.

P5.1 sits between :mod:`lifeform_domain_figure.contrast_set` and
:mod:`lifeform_domain_figure.steering_bake`:

* The contrast set declares **what** the figure disagrees about with
  whom, in reviewer-paraphrased text form.
* The steering bake (P5.2) consumes a **typed training plan**:
  positive residual vectors (the figure's stance) and negative
  residual vectors (the opponent's stance), plus per-pair labels and
  reviewer confidences.

This module is the deterministic, dependency-free bridge: it
embeds each paraphrase into the same 256-dim hashing-embedding
space the retrieval index uses, so the steering bake can train a
linear contrastive readout without taking a hard dependency on
``vz-substrate`` or ``torch``. The embedding is intentionally
identical to :func:`lifeform_domain_figure.retrieval_index._hashing_embedding`
so that retrieval-time queries and steering-time residuals live in
the same coordinate system; this is what makes the contrastive
direction inferred at bake time meaningful at retrieval time.

The plan is a frozen dataclass with an integrity hash so the bake
output (P5.2) is reproducible byte-for-byte (R15).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from lifeform_domain_figure.contrast_set import (
    FigureContrastPair,
    FigureContrastSet,
)
from lifeform_domain_figure.retrieval_index import (
    _hashing_embedding,
    _tokenize,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SteeringTrainingPair:
    """One (positive, negative) residual training pair for steering.

    Fields:

    * ``pair_id``                 — mirrors the originating
                                    :class:`FigureContrastPair.pair_id`.
    * ``axis``                    — disagreement axis label (e.g.
                                    ``"locality"``).
    * ``opponent_id``             — opponent normalised id.
    * ``positive_residual``       — embedding of the figure's stance.
    * ``negative_residual``       — embedding of the opponent stance.
    * ``confidence``              — reviewer confidence in the pair
                                    (passed through as a per-sample
                                    weight by the bake job).
    * ``evidence_locator``        — passes through to the steering
                                    artifact's audit trail.
    """

    pair_id: str
    axis: str
    opponent_id: str
    positive_residual: tuple[float, ...]
    negative_residual: tuple[float, ...]
    confidence: float
    evidence_locator: str

    def __post_init__(self) -> None:
        if not self.pair_id.strip():
            raise ValueError("SteeringTrainingPair.pair_id must be non-empty")
        if len(self.positive_residual) != len(self.negative_residual):
            raise ValueError(
                "SteeringTrainingPair: positive and negative residuals "
                "must share dimensionality "
                f"(positive={len(self.positive_residual)}, "
                f"negative={len(self.negative_residual)})"
            )
        if len(self.positive_residual) == 0:
            raise ValueError(
                "SteeringTrainingPair: residuals must be non-empty "
                f"(pair_id={self.pair_id!r})"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"SteeringTrainingPair.confidence must be in [0,1], "
                f"got {self.confidence!r}"
            )


@dataclass(frozen=True)
class FigureSteeringTrainingPlan:
    """Frozen training plan for the steering bake (P5.2).

    The plan carries enough information to make a steering bake
    reproducible byte-for-byte: same contrast set + same plan
    parameters → same training pairs → same baked steering artifact.
    The :attr:`integrity_hash` is what the F5 ``ModificationGate``
    proposal pins so a roll-forward / rollback can verify the bake's
    provenance (R15).
    """

    schema_version: int
    figure_id: str
    embedding_dim: int
    pairs: tuple[SteeringTrainingPair, ...]
    contrast_set_description: str
    integrity_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"FigureSteeringTrainingPlan.schema_version mismatch: "
                f"got {self.schema_version!r}, expected {SCHEMA_VERSION!r}"
            )
        if not self.figure_id.strip():
            raise ValueError(
                "FigureSteeringTrainingPlan.figure_id must be non-empty"
            )
        if not self.pairs:
            raise ValueError(
                "FigureSteeringTrainingPlan.pairs must be non-empty; "
                "refusing to build a steering plan with no training pairs."
            )
        if not self.integrity_hash.strip():
            raise ValueError(
                "FigureSteeringTrainingPlan.integrity_hash must be non-empty"
            )
        for pair in self.pairs:
            if len(pair.positive_residual) != self.embedding_dim:
                raise ValueError(
                    f"FigureSteeringTrainingPlan: pair {pair.pair_id!r} "
                    f"residual dim {len(pair.positive_residual)} does not "
                    f"match plan dim {self.embedding_dim}"
                )

    @property
    def total_pairs(self) -> int:
        return len(self.pairs)

    @property
    def axes(self) -> tuple[str, ...]:
        """Sorted unique axes covered by the plan."""
        return tuple(sorted({pair.axis for pair in self.pairs}))


def build_steering_training_plan(
    contrast_set: FigureContrastSet,
) -> FigureSteeringTrainingPlan:
    """Embed a contrast set into a deterministic training plan.

    Each :class:`FigureContrastPair` becomes one
    :class:`SteeringTrainingPair`:

    * ``positive_residual`` = hashing embedding of
      ``pair.figure_stance``.
    * ``negative_residual`` = hashing embedding of
      ``pair.opponent_stance``.

    The embedding function is the same one
    :class:`FigureRetrievalIndex` uses, so the steering direction
    inferred from these vectors lives in the same coordinate system
    as runtime retrieval. The function is pure: same contrast_set
    in → same plan out (including the same integrity hash).
    """

    if not contrast_set.pairs:
        raise ValueError(
            "build_steering_training_plan: contrast_set.pairs is empty; "
            "this should have been caught by FigureContrastSet.__post_init__"
        )
    training_pairs: list[SteeringTrainingPair] = []
    for pair in contrast_set.pairs:
        training_pairs.append(_pair_to_training_pair(pair))
    integrity_hash = _compute_plan_integrity_hash(
        figure_id=contrast_set.figure_id,
        pairs=tuple(training_pairs),
    )
    return FigureSteeringTrainingPlan(
        schema_version=SCHEMA_VERSION,
        figure_id=contrast_set.figure_id,
        embedding_dim=len(training_pairs[0].positive_residual),
        pairs=tuple(training_pairs),
        contrast_set_description=contrast_set.description,
        integrity_hash=integrity_hash,
    )


def _pair_to_training_pair(pair: FigureContrastPair) -> SteeringTrainingPair:
    positive_tokens = _tokenize(pair.figure_stance)
    negative_tokens = _tokenize(pair.opponent_stance)
    if not positive_tokens:
        raise ValueError(
            f"build_steering_training_plan: figure_stance produced zero "
            f"tokens for pair {pair.pair_id!r}; refusing to build a "
            f"degenerate residual."
        )
    if not negative_tokens:
        raise ValueError(
            f"build_steering_training_plan: opponent_stance produced zero "
            f"tokens for pair {pair.pair_id!r}; refusing to build a "
            f"degenerate residual."
        )
    positive = _hashing_embedding(positive_tokens)
    negative = _hashing_embedding(negative_tokens)
    return SteeringTrainingPair(
        pair_id=pair.pair_id,
        axis=pair.axis,
        opponent_id=pair.opponent_id,
        positive_residual=positive,
        negative_residual=negative,
        confidence=pair.confidence,
        evidence_locator=pair.evidence_locator,
    )


def _compute_plan_integrity_hash(
    *,
    figure_id: str,
    pairs: tuple[SteeringTrainingPair, ...],
) -> str:
    payload = (
        SCHEMA_VERSION,
        figure_id,
        tuple(
            (
                pair.pair_id,
                pair.axis,
                pair.opponent_id,
                tuple(round(v, 6) for v in pair.positive_residual),
                tuple(round(v, 6) for v in pair.negative_residual),
                round(pair.confidence, 6),
                pair.evidence_locator,
            )
            for pair in pairs
        ),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "FigureSteeringTrainingPlan",
    "SteeringTrainingPair",
    "build_steering_training_plan",
]
