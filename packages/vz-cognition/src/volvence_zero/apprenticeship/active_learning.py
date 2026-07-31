"""Owner-side bounded selection primitives for Gate 4 evidence.

These primitives operate only on typed numeric readouts.  They do not
interpret natural-language topics or write semantic state; the runtime evidence
harness remains responsible for replay order and artifact aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

from volvence_zero.apprenticeship.core import _random_feedback_sample


GATE4_ACTIVE_POLICIES = (
    "segment-aware-active",
    "turn-level-active",
    "random-feedback",
    "no-feedback",
    "shuffled-segment-boundary",
)


@dataclass(frozen=True)
class Gate4FeedbackCandidate:
    transition_id: str
    guidance_text: str
    turn_index: int
    predictor_features: tuple[float, ...]
    pe_magnitude: float
    segment_features: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise ValueError("Gate 4 candidate transition_id must be non-empty")
        if not self.predictor_features:
            raise ValueError("Gate 4 predictor features must be non-empty")
        if not self.segment_features:
            raise ValueError("Gate 4 segment features must be non-empty")
        values = (
            *self.predictor_features,
            self.pe_magnitude,
            *self.segment_features,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Gate 4 candidate features must be finite")
        if self.pe_magnitude < 0.0:
            raise ValueError("Gate 4 PE magnitude must be non-negative")


class BoundedBinaryAlignmentReadout:
    """Small deterministic logistic readout used identically by every arm."""

    def __init__(
        self,
        *,
        feature_count: int,
        learning_rate: float = 0.18,
        l2_penalty: float = 0.02,
        epoch_count: int = 160,
    ) -> None:
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        if learning_rate <= 0.0 or l2_penalty < 0.0 or epoch_count < 1:
            raise ValueError("invalid bounded readout optimizer configuration")
        self._feature_count = feature_count
        self._learning_rate = learning_rate
        self._l2_penalty = l2_penalty
        self._epoch_count = epoch_count
        self._weights = (0.0,) * feature_count
        self._bias = 0.0

    @property
    def parameters(self) -> tuple[float, ...]:
        return (*self._weights, self._bias)

    def fit(
        self,
        examples: Sequence[tuple[tuple[float, ...], bool]],
    ) -> None:
        self._weights = (0.0,) * self._feature_count
        self._bias = 0.0
        if not examples:
            return
        for features, _ in examples:
            self._validate_features(features)
        weights = list(self._weights)
        bias = self._bias
        scale = 1.0 / len(examples)
        for _ in range(self._epoch_count):
            weight_gradients = [0.0] * self._feature_count
            bias_gradient = 0.0
            for features, label in examples:
                probability = _sigmoid(
                    bias
                    + sum(
                        weight * value
                        for weight, value in zip(
                            weights,
                            features,
                            strict=True,
                        )
                    )
                )
                residual = probability - float(label)
                bias_gradient += residual
                for index, value in enumerate(features):
                    weight_gradients[index] += residual * value
            for index, gradient in enumerate(weight_gradients):
                update = self._learning_rate * (
                    gradient * scale
                    + self._l2_penalty * weights[index]
                )
                weights[index] = _clamp(weights[index] - update, -4.0, 4.0)
            bias = _clamp(
                bias - self._learning_rate * bias_gradient * scale,
                -4.0,
                4.0,
            )
        self._weights = tuple(weights)
        self._bias = bias

    def predict_probability(self, features: tuple[float, ...]) -> float:
        self._validate_features(features)
        return _sigmoid(
            self._bias
            + sum(
                weight * value
                for weight, value in zip(
                    self._weights,
                    features,
                    strict=True,
                )
            )
        )

    def _validate_features(self, features: tuple[float, ...]) -> None:
        if len(features) != self._feature_count:
            raise ValueError(
                "Gate 4 predictor feature width drifted: "
                f"expected {self._feature_count}, got {len(features)}"
            )
        if not all(math.isfinite(value) for value in features):
            raise ValueError("Gate 4 predictor features must be finite")


class BoundedLabelUtilityReadout:
    """Bounded owner-side estimate of the value of requesting one label."""

    def __init__(
        self,
        *,
        feature_count: int,
        minimum_observations: int = 4,
        learning_rate: float = 0.08,
        l2_penalty: float = 0.03,
        epoch_count: int = 120,
        utility_scale: float = 0.10,
    ) -> None:
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        if minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        if (
            learning_rate <= 0.0
            or l2_penalty < 0.0
            or epoch_count < 1
            or utility_scale <= 0.0
        ):
            raise ValueError("invalid bounded utility optimizer configuration")
        self._feature_count = feature_count
        self._minimum_observations = minimum_observations
        self._learning_rate = learning_rate
        self._l2_penalty = l2_penalty
        self._epoch_count = epoch_count
        self._utility_scale = utility_scale
        self._weights = (0.0,) * feature_count
        self._bias = 0.0
        self._observations: list[tuple[tuple[float, ...], float]] = []

    @property
    def feature_count(self) -> int:
        return self._feature_count

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    @property
    def ready(self) -> bool:
        return self.observation_count >= self._minimum_observations

    @property
    def parameters(self) -> tuple[float, ...]:
        return (*self._weights, self._bias)

    def observe_loss_delta(
        self,
        *,
        features: tuple[float, ...],
        loss_before: float,
        loss_after: float,
    ) -> float:
        self._validate_features(features)
        if not math.isfinite(loss_before) or not math.isfinite(loss_after):
            raise ValueError("Gate 4 utility losses must be finite")
        utility = loss_before - loss_after
        bounded_utility = math.tanh(utility / self._utility_scale)
        self._observations.append((features, bounded_utility))
        self._fit()
        return utility

    def predict_utility(self, features: tuple[float, ...]) -> float:
        self._validate_features(features)
        if not self.ready:
            raise RuntimeError(
                "Gate 4 utility readout is still in cold start"
            )
        return math.tanh(
            self._bias
            + sum(
                weight * value
                for weight, value in zip(
                    self._weights,
                    features,
                    strict=True,
                )
            )
        )

    def _fit(self) -> None:
        weights = [0.0] * self._feature_count
        bias = 0.0
        scale = 1.0 / len(self._observations)
        for _ in range(self._epoch_count):
            weight_gradients = [0.0] * self._feature_count
            bias_gradient = 0.0
            for features, target in self._observations:
                prediction = math.tanh(
                    bias
                    + sum(
                        weight * value
                        for weight, value in zip(
                            weights,
                            features,
                            strict=True,
                        )
                    )
                )
                residual = prediction - target
                derivative = 1.0 - prediction * prediction
                bias_gradient += residual * derivative
                for index, value in enumerate(features):
                    weight_gradients[index] += (
                        residual * derivative * value
                    )
            for index, gradient in enumerate(weight_gradients):
                update = self._learning_rate * (
                    gradient * scale
                    + self._l2_penalty * weights[index]
                )
                weights[index] = _clamp(weights[index] - update, -4.0, 4.0)
            bias = _clamp(
                bias - self._learning_rate * bias_gradient * scale,
                -4.0,
                4.0,
            )
        self._weights = tuple(weights)
        self._bias = bias

    def _validate_features(self, features: tuple[float, ...]) -> None:
        if len(features) != self._feature_count:
            raise ValueError(
                "Gate 4 utility feature width drifted: "
                f"expected {self._feature_count}, got {len(features)}"
            )
        if not all(math.isfinite(value) for value in features):
            raise ValueError("Gate 4 utility features must be finite")


def label_utility_features(
    *,
    candidate: Gate4FeedbackCandidate,
    probability: float,
    selected_segment_features: Sequence[tuple[float, ...]],
) -> tuple[float, ...]:
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Gate 4 candidate probability must be in [0, 1]")
    segment_novelty = _segment_novelty(
        candidate.segment_features,
        selected_segment_features,
    )
    beta_open = candidate.segment_features[-3]
    beta_close = candidate.segment_features[-2]
    segment_length = candidate.segment_features[-1]
    boundary_strength = _clamp(
        (
            abs(beta_close - beta_open)
            + min(abs(segment_length) / 4.0, 1.0)
        )
        / 2.0,
        0.0,
        1.0,
    )
    return (
        1.0 - abs(2.0 * probability - 1.0),
        candidate.pe_magnitude / (1.0 + candidate.pe_magnitude),
        segment_novelty,
        boundary_strength,
        1.0 - segment_novelty,
        *(
            value / (1.0 + abs(value))
            for value in candidate.segment_features
        ),
    )


def random_feedback_order(
    candidates: Sequence[Gate4FeedbackCandidate],
) -> tuple[int, ...]:
    """Return the reproducible random-control order.

    Candidates selected by the production matched-control predicate come
    first in registry order.  The deterministic hash tail is only a fail-loud
    completion path when an unusually small corpus yields fewer samples than
    the frozen budget.
    """

    sampled = tuple(
        index
        for index, candidate in enumerate(candidates)
        if _random_feedback_sample(
            guidance_text=candidate.guidance_text,
            turn_index=candidate.turn_index,
        )
    )
    sampled_set = set(sampled)
    remainder = tuple(
        sorted(
            (
                index
                for index in range(len(candidates))
                if index not in sampled_set
            ),
            key=lambda index: (
                _random_rank(candidates[index]),
                candidates[index].transition_id,
            ),
        )
    )
    return sampled + remainder


def select_active_candidate(
    *,
    candidates: Sequence[Gate4FeedbackCandidate],
    probabilities: Sequence[float],
    selected_indices: frozenset[int],
    policy: str,
    utility_readout: BoundedLabelUtilityReadout | None = None,
) -> int:
    if policy not in {
        "segment-aware-active",
        "turn-level-active",
        "shuffled-segment-boundary",
    }:
        raise ValueError(f"Unsupported Gate 4 active policy {policy!r}")
    if len(candidates) != len(probabilities):
        raise ValueError("Gate 4 candidate/probability length mismatch")
    if len(selected_indices) >= len(candidates):
        raise ValueError("Gate 4 active selector has no remaining candidate")
    selected_segment_features = tuple(
        candidates[index].segment_features
        for index in sorted(selected_indices)
    )
    scored: list[tuple[float, str, int]] = []
    for index, (candidate, probability) in enumerate(
        zip(candidates, probabilities, strict=True)
    ):
        if index in selected_indices:
            continue
        uncertainty = 1.0 - abs(2.0 * probability - 1.0)
        pe_strength = candidate.pe_magnitude / (
            1.0 + candidate.pe_magnitude
        )
        if policy == "turn-level-active":
            score = 0.65 * uncertainty + 0.35 * pe_strength
        elif utility_readout is not None and utility_readout.ready:
            score = utility_readout.predict_utility(
                label_utility_features(
                    candidate=candidate,
                    probability=probability,
                    selected_segment_features=selected_segment_features,
                )
            )
        else:
            score = uncertainty
        scored.append((score, candidate.transition_id, index))
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def _segment_novelty(
    features: tuple[float, ...],
    selected: Sequence[tuple[float, ...]],
) -> float:
    if not selected:
        return 1.0
    width = len(features)
    if any(len(item) != width for item in selected):
        raise ValueError("Gate 4 segment feature width drifted")
    distance = min(
        math.sqrt(
            sum(
                (left - right) ** 2
                for left, right in zip(features, item, strict=True)
            )
            / width
        )
        for item in selected
    )
    return _clamp(distance, 0.0, 1.0)


def _random_rank(candidate: Gate4FeedbackCandidate) -> int:
    payload = (
        f"{candidate.turn_index}:{candidate.guidance_text}"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        "big",
    )


def _sigmoid(value: float) -> float:
    bounded = _clamp(value, -30.0, 30.0)
    return 1.0 / (1.0 + math.exp(-bounded))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


__all__ = [
    "GATE4_ACTIVE_POLICIES",
    "BoundedBinaryAlignmentReadout",
    "BoundedLabelUtilityReadout",
    "Gate4FeedbackCandidate",
    "label_utility_features",
    "random_feedback_order",
    "select_active_candidate",
]
