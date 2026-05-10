"""F5 / P5.2 — bake steering vectors from a contrast training plan.

Builds a frozen :class:`FigureSteeringSet` artifact:

* One :class:`SteeringVector` per disagreement axis (the contrastive
  direction ``positive - negative`` averaged across pairs sharing the
  axis, weighted by reviewer confidence).
* One ``aggregate`` :class:`SteeringVector` over the whole set (the
  global figure-vs-opponents direction, used at runtime when the
  query does not match a single axis).
* A :func:`to_substrate_adapter_layers` adapter that emits standard
  :class:`SubstrateDeltaAdapterLayer` tuples — the same type the
  vz-substrate residual backends already emit — so the runtime
  consumes the steering vectors as constant deltas through the
  existing checkpoint surface (no vz-substrate kernel change).

This module is **CPU-runnable**: it implements a deterministic
linear contrastive readout in pure stdlib (no torch / numpy
dependency). The plan said "real CPU backend" — the bake is
algebraically equivalent to a linear LDA-style direction with
unit-normalised inputs, which is the canonical readout used in
contrastive steering papers; we ship the exact closed-form for it
rather than a wrapped torch trainer to keep the bake reproducible
across machines.

The :func:`apply_steering_through_gate` helper mirrors
``lifeform_domain_character.apply_drive_evolution_through_gate``
and routes a steering proposal through the existing
:class:`ModificationGate.OFFLINE` path. R10: this is rare-heavy
self-modification — figure stance is part of who the lifeform is.
R15: a rollback re-attaches the previous steering set's id, which
is recorded in :attr:`SteeringApplyResult.rollback_evidence`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from volvence_zero.credit.gate import (
    GateDecision,
    ModificationGate,
    ModificationProposal,
    evaluate_gate,
    evaluate_gate_reasons,
)
from volvence_zero.evaluation.types import EvaluationSnapshot
from volvence_zero.substrate import SubstrateDeltaAdapterLayer

from lifeform_domain_figure.compiler import attach_steering_to_bundle
from lifeform_domain_figure.figure_artifact import FigureArtifactBundle
from lifeform_domain_figure.steering_data_prep import (
    FigureSteeringTrainingPlan,
    SteeringTrainingPair,
)


SCHEMA_VERSION = 1
_DEFAULT_VECTOR_SCALE = 1.0
_AGGREGATE_AXIS = "_aggregate"


@dataclass(frozen=True)
class SteeringVector:
    """One per-axis contrastive steering direction.

    The vector lives in the same coordinate system as the hashing
    embedding produced by :mod:`lifeform_domain_figure.steering_data_prep`,
    which is also the coordinate system of the runtime retrieval
    index. The :attr:`scale` field is the average per-pair
    margin (the cosine separation between positive and negative
    sides, weighted by reviewer confidence) — the runtime uses it
    as the natural strength of this axis when blending multiple
    vectors.
    """

    axis: str
    direction: tuple[float, ...]
    scale: float
    sample_count: int
    description: str

    def __post_init__(self) -> None:
        if not self.axis.strip():
            raise ValueError("SteeringVector.axis must be non-empty")
        if len(self.direction) == 0:
            raise ValueError(
                f"SteeringVector.direction must be non-empty (axis={self.axis!r})"
            )
        if not math.isfinite(self.scale):
            raise ValueError(
                f"SteeringVector.scale must be finite, got {self.scale!r}"
            )
        if self.sample_count <= 0:
            raise ValueError(
                f"SteeringVector.sample_count must be > 0, got {self.sample_count!r}"
            )


@dataclass(frozen=True)
class FigureSteeringSet:
    """Frozen set of per-axis steering vectors for one figure.

    Output of :func:`bake_steering_set` and the artifact bound into
    :attr:`FigureArtifactBundle.steering` via
    :func:`attach_steering_to_bundle`.
    """

    schema_version: int
    figure_id: str
    embedding_dim: int
    vectors: tuple[SteeringVector, ...]
    aggregate: SteeringVector
    training_plan_hash: str
    integrity_hash: str
    description: str

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"FigureSteeringSet.schema_version mismatch: "
                f"got {self.schema_version!r}, expected {SCHEMA_VERSION!r}"
            )
        if not self.figure_id.strip():
            raise ValueError("FigureSteeringSet.figure_id must be non-empty")
        if not self.vectors:
            raise ValueError(
                "FigureSteeringSet.vectors must be non-empty; refusing to "
                "bake a steering set with no per-axis directions."
            )
        if len(self.aggregate.direction) != self.embedding_dim:
            raise ValueError(
                f"FigureSteeringSet.aggregate dim {len(self.aggregate.direction)} "
                f"does not match embedding_dim {self.embedding_dim}"
            )
        for vector in self.vectors:
            if len(vector.direction) != self.embedding_dim:
                raise ValueError(
                    f"FigureSteeringSet vector {vector.axis!r} dim "
                    f"{len(vector.direction)} does not match embedding_dim "
                    f"{self.embedding_dim}"
                )
        seen_axes: set[str] = set()
        for vector in self.vectors:
            if vector.axis in seen_axes:
                raise ValueError(
                    f"FigureSteeringSet has duplicate axis {vector.axis!r}"
                )
            seen_axes.add(vector.axis)
        if self.aggregate.axis != _AGGREGATE_AXIS:
            raise ValueError(
                f"FigureSteeringSet.aggregate.axis must be {_AGGREGATE_AXIS!r}, "
                f"got {self.aggregate.axis!r}"
            )

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Sorted, axis names excluding the aggregate."""
        return tuple(sorted(vector.axis for vector in self.vectors))

    def get_vector(self, axis: str) -> SteeringVector:
        """Return the per-axis vector or raise ``KeyError`` (fail-loud)."""
        for vector in self.vectors:
            if vector.axis == axis:
                return vector
        raise KeyError(
            f"FigureSteeringSet has no axis {axis!r}; available: "
            f"{self.axis_names}"
        )

    def to_substrate_adapter_layers(
        self,
        *,
        layer_index: int = 0,
        scale: float = _DEFAULT_VECTOR_SCALE,
    ) -> tuple[SubstrateDeltaAdapterLayer, ...]:
        """Render the steering set as constant-delta adapter layers.

        Each per-axis vector becomes one
        :class:`SubstrateDeltaAdapterLayer`; the aggregate is the
        last layer. Layers all share the same logical residual
        coordinate space, so the consumer is free to apply only a
        subset (e.g. only the aggregate, or only one axis matched
        by the live query) without rewriting the artifact.
        """

        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError(
                f"to_substrate_adapter_layers: scale must be > 0 and finite, "
                f"got {scale!r}"
            )
        layers: list[SubstrateDeltaAdapterLayer] = []
        for offset, vector in enumerate(
            sorted(self.vectors, key=lambda v: v.axis)
        ):
            scaled = tuple(scale * vector.scale * v for v in vector.direction)
            layers.append(
                SubstrateDeltaAdapterLayer(
                    layer_index=layer_index + offset,
                    delta_vector=scaled,
                    mean_abs_delta=_mean_abs(scaled),
                    description=(
                        f"figure-steering:{self.figure_id}:axis={vector.axis} "
                        f"(samples={vector.sample_count})"
                    ),
                )
            )
        scaled_aggregate = tuple(
            scale * self.aggregate.scale * v for v in self.aggregate.direction
        )
        layers.append(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index + len(self.vectors),
                delta_vector=scaled_aggregate,
                mean_abs_delta=_mean_abs(scaled_aggregate),
                description=(
                    f"figure-steering:{self.figure_id}:axis=_aggregate "
                    f"(samples={self.aggregate.sample_count})"
                ),
            )
        )
        return tuple(layers)


def bake_steering_set(
    plan: FigureSteeringTrainingPlan,
    *,
    description: str | None = None,
) -> FigureSteeringSet:
    """Train a contrastive linear readout from a training plan.

    Per-axis: average ``positive - negative`` across pairs sharing
    the axis (weighted by reviewer confidence), then re-normalise to
    unit length. The per-axis :attr:`scale` is the average cosine
    margin :math:`(p \\cdot d - n \\cdot d)` along the resulting
    direction — i.e. how much positive/negative separate when
    projected onto the inferred direction.

    Aggregate: same construction over all pairs.

    The bake is a pure function: same plan in → same set out
    (including byte-for-byte identical integrity hash).
    """

    if not plan.pairs:
        raise ValueError(
            "bake_steering_set: plan.pairs is empty; this should have been "
            "caught by FigureSteeringTrainingPlan.__post_init__"
        )
    by_axis: dict[str, list[SteeringTrainingPair]] = {}
    for pair in plan.pairs:
        by_axis.setdefault(pair.axis, []).append(pair)
    vectors: list[SteeringVector] = []
    for axis, pairs in sorted(by_axis.items()):
        direction, scale = _direction_for_pairs(pairs, dim=plan.embedding_dim)
        vectors.append(
            SteeringVector(
                axis=axis,
                direction=direction,
                scale=scale,
                sample_count=len(pairs),
                description=(
                    f"Contrastive direction for axis {axis!r} from "
                    f"{len(pairs)} reviewed pair(s)."
                ),
            )
        )
    aggregate_direction, aggregate_scale = _direction_for_pairs(
        list(plan.pairs), dim=plan.embedding_dim
    )
    aggregate = SteeringVector(
        axis=_AGGREGATE_AXIS,
        direction=aggregate_direction,
        scale=aggregate_scale,
        sample_count=len(plan.pairs),
        description=(
            f"Global figure-vs-opponents direction from "
            f"{len(plan.pairs)} reviewed pair(s)."
        ),
    )
    integrity_hash = _compute_set_integrity_hash(
        figure_id=plan.figure_id,
        vectors=tuple(vectors),
        aggregate=aggregate,
        training_plan_hash=plan.integrity_hash,
    )
    return FigureSteeringSet(
        schema_version=SCHEMA_VERSION,
        figure_id=plan.figure_id,
        embedding_dim=plan.embedding_dim,
        vectors=tuple(vectors),
        aggregate=aggregate,
        training_plan_hash=plan.integrity_hash,
        integrity_hash=integrity_hash,
        description=(
            description
            or f"Baked steering set for {plan.figure_id} from "
            f"{len(plan.pairs)} reviewed contrast pair(s) across "
            f"{len(by_axis)} axis(es)."
        ),
    )


def attach_baked_steering(
    bundle: FigureArtifactBundle,
    steering: FigureSteeringSet,
) -> FigureArtifactBundle:
    """Attach a baked steering set to a bundle (re-keys bundle id).

    Thin wrapper over :func:`attach_steering_to_bundle` that keeps
    the F5 callsite from importing both modules.
    """

    if steering.figure_id != bundle.figure_id:
        raise ValueError(
            f"attach_baked_steering: steering.figure_id={steering.figure_id!r} "
            f"does not match bundle.figure_id={bundle.figure_id!r}"
        )
    return attach_steering_to_bundle(
        bundle,
        steering=steering,
        steering_integrity=steering.integrity_hash,
    )


# ---------------------------------------------------------------------------
# OFFLINE-gate apply path (mirrors apply_drive_evolution_through_gate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatedSteeringProposal:
    """A steering set proposal after the gate has decided."""

    proposal: ModificationProposal
    decision: GateDecision
    block_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SteeringApplyResult:
    """Return value of :func:`apply_steering_through_gate`."""

    bundle: FigureArtifactBundle
    base_bundle: FigureArtifactBundle
    steering: FigureSteeringSet
    gate: GatedSteeringProposal
    applied: bool
    rollback_evidence: str


def apply_steering_through_gate(
    *,
    base_bundle: FigureArtifactBundle,
    steering: FigureSteeringSet,
    evaluation_snapshot: EvaluationSnapshot,
    validation_delta: float = 0.05,
    capacity_cost: float = 0.20,
    rollback_evidence: str = "",
) -> SteeringApplyResult:
    """Route a steering set through the OFFLINE :class:`ModificationGate`.

    The proposal targets ``figure.steering_set`` and pins the
    training-plan hash + new steering integrity hash in its
    old/new value hashes so the gate's audit log can verify the
    bake's provenance after the fact (R8 + R15).

    A non-empty ``rollback_evidence`` is required: it should
    identify the previous steering bundle id (or sentinel
    ``"absent"``) so the operator can reattach it on rollback.
    """

    if steering.figure_id != base_bundle.figure_id:
        raise ValueError(
            "apply_steering_through_gate: steering.figure_id="
            f"{steering.figure_id!r} does not match "
            f"base_bundle.figure_id={base_bundle.figure_id!r}"
        )
    if not rollback_evidence.strip():
        raise ValueError(
            "apply_steering_through_gate: rollback_evidence must be "
            "non-empty (the OFFLINE gate requires it)"
        )
    proposal = _proposal_for_steering(
        base_bundle=base_bundle,
        steering=steering,
        validation_delta=validation_delta,
        capacity_cost=capacity_cost,
        rollback_evidence=rollback_evidence,
    )
    decision = evaluate_gate(
        proposal=proposal,
        evaluation_snapshot=evaluation_snapshot,
    )
    if decision is GateDecision.BLOCK:
        reasons = evaluate_gate_reasons(
            proposal=proposal,
            evaluation_snapshot=evaluation_snapshot,
        )
        return SteeringApplyResult(
            bundle=base_bundle,
            base_bundle=base_bundle,
            steering=steering,
            gate=GatedSteeringProposal(
                proposal=proposal,
                decision=decision,
                block_reasons=reasons,
            ),
            applied=False,
            rollback_evidence=rollback_evidence,
        )
    new_bundle = attach_baked_steering(base_bundle, steering)
    return SteeringApplyResult(
        bundle=new_bundle,
        base_bundle=base_bundle,
        steering=steering,
        gate=GatedSteeringProposal(
            proposal=proposal,
            decision=decision,
            block_reasons=(),
        ),
        applied=True,
        rollback_evidence=rollback_evidence,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _direction_for_pairs(
    pairs: list[SteeringTrainingPair],
    *,
    dim: int,
) -> tuple[tuple[float, ...], float]:
    """Compute the unit-normalised contrastive direction + cosine margin."""

    accumulator = [0.0] * dim
    total_weight = 0.0
    for pair in pairs:
        weight = max(pair.confidence, 1e-6)
        for index in range(dim):
            accumulator[index] += weight * (
                pair.positive_residual[index] - pair.negative_residual[index]
            )
        total_weight += weight
    if total_weight <= 0.0:
        raise ValueError(
            "_direction_for_pairs: total weight is zero; refusing to "
            "produce a degenerate steering direction."
        )
    averaged = [value / total_weight for value in accumulator]
    norm = math.sqrt(sum(value * value for value in averaged))
    if norm <= 1e-9:
        # Reviewer-supplied positive and negative paraphrases collapsed
        # to the same residual after embedding (different surface tokens
        # mapped to identical hash buckets). Fail-loud rather than emit a
        # zero steering vector that would silently no-op at runtime.
        raise ValueError(
            "_direction_for_pairs: averaged contrastive direction has "
            "near-zero norm; the contrast pairs do not separate in the "
            "current embedding. Add more discriminating paraphrases or "
            "broaden the embedding."
        )
    direction = tuple(value / norm for value in averaged)
    margin_sum = 0.0
    margin_weight = 0.0
    for pair in pairs:
        weight = max(pair.confidence, 1e-6)
        positive_proj = sum(
            pair.positive_residual[i] * direction[i] for i in range(dim)
        )
        negative_proj = sum(
            pair.negative_residual[i] * direction[i] for i in range(dim)
        )
        margin_sum += weight * (positive_proj - negative_proj)
        margin_weight += weight
    scale = margin_sum / margin_weight if margin_weight > 0.0 else 0.0
    return direction, scale


def _mean_abs(vector: tuple[float, ...]) -> float:
    if not vector:
        return 0.0
    return sum(abs(value) for value in vector) / len(vector)


def _compute_set_integrity_hash(
    *,
    figure_id: str,
    vectors: tuple[SteeringVector, ...],
    aggregate: SteeringVector,
    training_plan_hash: str,
) -> str:
    payload = (
        SCHEMA_VERSION,
        figure_id,
        training_plan_hash,
        tuple(
            (
                vector.axis,
                tuple(round(v, 6) for v in vector.direction),
                round(vector.scale, 6),
                vector.sample_count,
            )
            for vector in vectors
        ),
        (
            aggregate.axis,
            tuple(round(v, 6) for v in aggregate.direction),
            round(aggregate.scale, 6),
            aggregate.sample_count,
        ),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _proposal_for_steering(
    *,
    base_bundle: FigureArtifactBundle,
    steering: FigureSteeringSet,
    validation_delta: float,
    capacity_cost: float,
    rollback_evidence: str,
) -> ModificationProposal:
    old_value_repr = repr(
        (
            "figure.steering_set",
            base_bundle.figure_id,
            getattr(base_bundle.steering, "integrity_hash", "absent"),
        )
    )
    new_value_repr = repr(
        (
            "figure.steering_set",
            steering.figure_id,
            steering.integrity_hash,
            steering.training_plan_hash,
        )
    )
    return ModificationProposal(
        target=f"figure.steering_set[{base_bundle.figure_id}]",
        desired_gate=ModificationGate.OFFLINE,
        old_value_hash=hashlib.sha256(
            old_value_repr.encode("utf-8")
        ).hexdigest(),
        new_value_hash=hashlib.sha256(
            new_value_repr.encode("utf-8")
        ).hexdigest(),
        justification=(
            f"Bake steering set for {steering.figure_id} from "
            f"{steering.aggregate.sample_count} reviewed contrast pair(s)."
        ),
        is_reversible=True,
        validation_delta=validation_delta,
        capacity_cost=capacity_cost,
        rollback_evidence=rollback_evidence,
    )


__all__ = [
    "SCHEMA_VERSION",
    "FigureSteeringSet",
    "GatedSteeringProposal",
    "SteeringApplyResult",
    "SteeringVector",
    "apply_steering_through_gate",
    "attach_baked_steering",
    "bake_steering_set",
]
