"""Shared bounded policy math for vertical Brain adapters.

Vertical owners publish typed numeric features and retain ownership of state,
candidate meaning, evidence, authority, outcome facts, PE settlement and
checkpoint persistence.  This module only implements deterministic bounded
ranking, intervention timing and exact-credit parameter updates.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping


def _require_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        scale = math.exp(-value)
        return 1.0 / (1.0 + scale)
    scale = math.exp(value)
    return scale / (1.0 + scale)


def _softmax(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        raise ValueError("bounded policy softmax requires at least one score")
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    denominator = math.fsum(exponentials)
    return tuple(value / denominator for value in exponentials)


@dataclass(frozen=True)
class BoundedPolicyCandidate:
    """Adapter-owned candidate projected onto the shared policy surface."""

    candidate_id: str
    action_key: str
    feature_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not isinstance(self.action_key, str) or not self.action_key:
            raise ValueError("action_key must be non-empty")
        if not isinstance(self.feature_values, tuple) or not self.feature_values:
            raise ValueError("feature_values must be a non-empty tuple")
        for value in self.feature_values:
            _require_finite("feature value", value)


@dataclass(frozen=True)
class BoundedPolicyRankedCandidate:
    candidate_id: str
    action_key: str
    rank: int
    policy_score: float
    selection_probability: float
    feature_values: tuple[float, ...]


@dataclass(frozen=True)
class BoundedPolicyDecision:
    ranked_candidates: tuple[BoundedPolicyRankedCandidate, ...]
    recommended_candidate_id: str
    selected_candidate_id: str
    intervention_probability: float
    intervenes: bool


@dataclass(frozen=True)
class BoundedPolicyUpdate:
    action_weights: tuple[tuple[str, tuple[float, ...]], ...]
    intervention_weights: tuple[float, ...]
    intervention_bias: float
    parameter_delta_l2: float


def _validated_weight_rows(
    action_weights: tuple[tuple[str, tuple[float, ...]], ...],
    *,
    feature_count: int,
) -> dict[str, tuple[float, ...]]:
    if not isinstance(action_weights, tuple) or not action_weights:
        raise ValueError("action_weights must be a non-empty tuple")
    rows: dict[str, tuple[float, ...]] = {}
    for action_key, weights in action_weights:
        if not isinstance(action_key, str) or not action_key:
            raise ValueError("action weight key must be non-empty")
        if action_key in rows:
            raise ValueError("action weight keys must be unique")
        if not isinstance(weights, tuple) or len(weights) != feature_count:
            raise ValueError("action weight rows must match feature geometry")
        rows[action_key] = tuple(
            _require_finite("action weight", value) for value in weights
        )
    return rows


def rank_and_gate_bounded_policy(
    *,
    candidates: tuple[BoundedPolicyCandidate, ...],
    action_weights: tuple[tuple[str, tuple[float, ...]], ...],
    intervention_weights: tuple[float, ...],
    intervention_bias: float,
    maximum_candidates: int,
    intervention_enabled: bool = True,
    intervention_threshold: float = 0.5,
) -> BoundedPolicyDecision:
    """Rank typed candidates and choose whether the bounded policy intervenes."""

    if not isinstance(candidates, tuple) or not candidates:
        raise ValueError("bounded policy requires at least one candidate")
    if (
        isinstance(maximum_candidates, bool)
        or not isinstance(maximum_candidates, int)
        or maximum_candidates < 1
    ):
        raise ValueError("maximum_candidates must be a positive integer")
    threshold = _require_finite(
        "intervention_threshold",
        intervention_threshold,
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("intervention_threshold must be in [0, 1]")
    feature_count = len(candidates[0].feature_values)
    candidate_ids = tuple(item.candidate_id for item in candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ids must be unique")
    if any(len(item.feature_values) != feature_count for item in candidates):
        raise ValueError("candidate feature geometry must be stable")
    rows = _validated_weight_rows(
        action_weights,
        feature_count=feature_count,
    )
    if (
        not isinstance(intervention_weights, tuple)
        or len(intervention_weights) != feature_count
    ):
        raise ValueError("intervention weights must match feature geometry")
    timing_weights = tuple(
        _require_finite("intervention weight", value)
        for value in intervention_weights
    )
    timing_bias = _require_finite("intervention_bias", intervention_bias)

    scored: list[tuple[BoundedPolicyCandidate, float]] = []
    for candidate in candidates:
        try:
            row = rows[candidate.action_key]
        except KeyError as exc:
            raise ValueError(
                f"missing action weights for {candidate.action_key!r}"
            ) from exc
        score = math.fsum(
            weight * feature
            for weight, feature in zip(
                row,
                candidate.feature_values,
                strict=True,
            )
        )
        scored.append((candidate, score))
    scored.sort(key=lambda item: (-item[1], item[0].candidate_id))
    selected_pool = tuple(scored[:maximum_candidates])
    probabilities = _softmax(tuple(score for _, score in selected_pool))
    ranked = tuple(
        BoundedPolicyRankedCandidate(
            candidate_id=candidate.candidate_id,
            action_key=candidate.action_key,
            rank=index,
            policy_score=score,
            selection_probability=probability,
            feature_values=candidate.feature_values,
        )
        for index, ((candidate, score), probability) in enumerate(
            zip(selected_pool, probabilities, strict=True),
            start=1,
        )
    )
    recommended = ranked[0].candidate_id
    intervention_probability = _sigmoid(
        math.fsum(
            weight * feature
            for weight, feature in zip(
                timing_weights,
                ranked[0].feature_values,
                strict=True,
            )
        )
        + timing_bias
    )
    intervenes = intervention_enabled and intervention_probability > threshold
    return BoundedPolicyDecision(
        ranked_candidates=ranked,
        recommended_candidate_id=recommended,
        selected_candidate_id=recommended if intervenes else "",
        intervention_probability=intervention_probability,
        intervenes=intervenes,
    )


def apply_bounded_policy_credit(
    *,
    action_weights: tuple[tuple[str, tuple[float, ...]], ...],
    intervention_weights: tuple[float, ...],
    intervention_bias: float,
    decision: BoundedPolicyDecision,
    credited_candidate_id: str,
    noop_candidate_id: str,
    signed_credit: float,
    learning_rate: float,
    max_abs_parameter: float,
) -> BoundedPolicyUpdate:
    """Apply one adapter-settled PE credit to an immutable parameter snapshot."""

    if not decision.ranked_candidates:
        raise ValueError("bounded policy decision has no ranked candidates")
    feature_count = len(decision.ranked_candidates[0].feature_values)
    rows = _validated_weight_rows(
        action_weights,
        feature_count=feature_count,
    )
    if (
        not isinstance(intervention_weights, tuple)
        or len(intervention_weights) != feature_count
    ):
        raise ValueError("intervention weights must match feature geometry")
    timing_weights = tuple(
        _require_finite("intervention weight", value)
        for value in intervention_weights
    )
    timing_bias = _require_finite("intervention_bias", intervention_bias)
    credit = _require_finite("signed_credit", signed_credit)
    rate = _require_finite("learning_rate", learning_rate)
    cap = _require_finite("max_abs_parameter", max_abs_parameter)
    if rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if cap <= 0.0:
        raise ValueError("max_abs_parameter must be positive")

    ranked_ids = tuple(item.candidate_id for item in decision.ranked_candidates)
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked candidate ids must be unique")
    if any(
        len(item.feature_values) != feature_count
        for item in decision.ranked_candidates
    ):
        raise ValueError("ranked candidate feature geometry must be stable")
    noop_credit = credited_candidate_id == noop_candidate_id
    if noop_credit:
        if decision.intervenes:
            raise ValueError("NOOP credit requires a NOOP policy decision")
        selected_features = decision.ranked_candidates[0].feature_values
        action_indicator = 0.0
    else:
        if not decision.intervenes:
            raise ValueError("candidate credit requires an intervention decision")
        if credited_candidate_id != decision.selected_candidate_id:
            raise ValueError("credited candidate does not match selected candidate")
        selected = next(
            (
                item
                for item in decision.ranked_candidates
                if item.candidate_id == credited_candidate_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("credited candidate is absent from ranked candidates")
        selected_features = selected.feature_values
        action_indicator = 1.0

    next_rows = {name: list(weights) for name, weights in rows.items()}
    if not noop_credit:
        gradients: dict[str, list[float]] = defaultdict(
            lambda: [0.0] * feature_count
        )
        for candidate in decision.ranked_candidates:
            if candidate.action_key not in rows:
                raise ValueError(
                    f"missing action weights for {candidate.action_key!r}"
                )
            indicator = 1.0 if candidate.candidate_id == credited_candidate_id else 0.0
            scale = rate * credit * (
                indicator - candidate.selection_probability
            )
            for index, feature in enumerate(candidate.feature_values):
                gradients[candidate.action_key][index] += scale * feature
        for action_key, deltas in gradients.items():
            next_rows[action_key] = [
                max(-cap, min(cap, value + delta))
                for value, delta in zip(
                    next_rows[action_key],
                    deltas,
                    strict=True,
                )
            ]

    intervention_scale = rate * credit * (
        action_indicator - decision.intervention_probability
    )
    next_timing_weights = tuple(
        max(-cap, min(cap, weight + intervention_scale * feature))
        for weight, feature in zip(
            timing_weights,
            selected_features,
            strict=True,
        )
    )
    next_timing_bias = max(
        -cap,
        min(cap, timing_bias + intervention_scale),
    )
    ordered_next_rows = tuple(
        (action_key, tuple(next_rows[action_key]))
        for action_key, _ in action_weights
    )
    old_parameters = tuple(
        value for _, weights in action_weights for value in weights
    ) + (*timing_weights, timing_bias)
    new_parameters = tuple(
        value for _, weights in ordered_next_rows for value in weights
    ) + (*next_timing_weights, next_timing_bias)
    parameter_delta_l2 = math.sqrt(
        math.fsum(
            (after - before) ** 2
            for before, after in zip(
                old_parameters,
                new_parameters,
                strict=True,
            )
        )
    )
    return BoundedPolicyUpdate(
        action_weights=ordered_next_rows,
        intervention_weights=next_timing_weights,
        intervention_bias=next_timing_bias,
        parameter_delta_l2=parameter_delta_l2,
    )


__all__ = (
    "BoundedPolicyCandidate",
    "BoundedPolicyDecision",
    "BoundedPolicyRankedCandidate",
    "BoundedPolicyUpdate",
    "apply_bounded_policy_credit",
    "rank_and_gate_bounded_policy",
)
