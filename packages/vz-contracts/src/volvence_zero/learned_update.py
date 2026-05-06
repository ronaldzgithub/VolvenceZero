from __future__ import annotations

import math
from dataclasses import dataclass


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _mean_abs(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _rms(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _target_signature(target_id: str) -> tuple[float, float, float]:
    if not target_id:
        return (0.0, 0.0, 0.0)
    code_points = tuple(ord(char) for char in target_id)
    total = sum(code_points)
    mean_code = total / len(code_points)
    spread = max(code_points) - min(code_points)
    parity = sum((index + 1) * value for index, value in enumerate(code_points))
    return (
        _clamp_signed((mean_code / 127.0) * 2.0 - 1.0),
        _clamp_signed((spread / 127.0) * 2.0 - 1.0),
        _clamp_signed(((parity % 997) / 498.5) - 1.0),
    )


def _init_weight(size: int, *, scale: float, seed: int) -> tuple[float, ...]:
    return tuple(
        scale * (((seed + index * 2654435761 + 17) % 65537) / 32768.5 - 1.0)
        for index in range(size)
    )


# Feature layout used by ATLAS / Titans uplift in vz-memory CMS.
# See docs/specs/cms-atlas-titans-uplift.md §4 / §5.
#
# - BASE_FEATURE_DIM (12): legacy CMS decision features (current/target/delta
#   norms, hyperparameters). Pre-uplift LearnedUpdateRuleState has these only.
# - PE_FEATURE_DIM (4): added by Titans surprise gating
#   (|task_error|, |relationship_error|, |regime_error|, |action_error|).
#
# A rule whose feature_dim >= BASE + PE is treated as PE-aware on restore: the
# trailing PE_FEATURE_DIM columns are zero-padded when an older
# (feature_version=1) state is loaded, to preserve numerical equivalence with
# pre-uplift behavior whenever PE inputs are zero.
LEARNED_UPDATE_BASE_FEATURE_DIM = 12
LEARNED_UPDATE_PE_FEATURE_DIM = 4
LEARNED_UPDATE_PE_AWARE_FEATURE_DIM = (
    LEARNED_UPDATE_BASE_FEATURE_DIM + LEARNED_UPDATE_PE_FEATURE_DIM
)


@dataclass(frozen=True)
class LearnedUpdateDecision:
    target_id: str
    write_gate: float
    step_scale: float
    momentum_gate: float
    slow_mix: float
    reset_mix: float
    bias_delta: float
    confidence: float
    guard_applied: bool = False
    guard_reason: str = ""
    description: str = ""


@dataclass(frozen=True)
class LearnedUpdateRuleState:
    rule_id: str
    feature_dim: int
    hidden_dim: int
    update_count: int
    last_feature_norm: float
    last_improvement: float
    last_guard_reason: str
    input_projection: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_projection: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, ...]
    last_decisions: tuple[LearnedUpdateDecision, ...] = ()
    base_learning_rate: float = 0.0
    last_effective_learning_rate: float = 0.0
    last_reward: float = 0.0
    last_stability: float = 0.0
    last_write_gate: float = 0.0
    last_step_scale: float = 0.0
    last_momentum_gate: float = 0.0
    last_slow_mix: float = 0.0
    last_reset_mix: float = 0.0
    last_confidence: float = 0.0
    description: str = ""
    # ATLAS / Titans uplift: feature layout version.
    # 1 = legacy (no PE features in trailing 4 columns).
    # 2 = PE-aware (trailing 4 columns are PE magnitudes when feature_dim >= 16).
    # Default 1 keeps legacy serialized states deserializable as-is.
    feature_version: int = 1


class LearnedUpdateRule:
    """Small bounded meta-updater shared by temporal and memory owners."""

    _OUTPUT_DIM = 7

    def __init__(
        self,
        *,
        rule_id: str,
        feature_dim: int,
        hidden_dim: int = 8,
        learning_rate: float = 0.08,
    ) -> None:
        self._rule_id = rule_id
        self._feature_dim = max(1, feature_dim)
        self._hidden_dim = max(4, hidden_dim)
        self._learning_rate = max(0.01, min(learning_rate, 0.25))
        self._input_projection = tuple(
            _init_weight(self._feature_dim, scale=0.12, seed=11 + row * 7)
            for row in range(self._hidden_dim)
        )
        self._hidden_bias = tuple(0.0 for _ in range(self._hidden_dim))
        self._output_projection = tuple(
            _init_weight(self._hidden_dim, scale=0.10, seed=97 + row * 13)
            for row in range(self._OUTPUT_DIM)
        )
        self._output_bias = tuple(0.0 for _ in range(self._OUTPUT_DIM))
        self._update_count = 0
        self._last_feature_norm = 0.0
        self._last_improvement = 0.0
        self._last_guard_reason = ""
        self._last_decisions: tuple[LearnedUpdateDecision, ...] = ()
        self._last_effective_learning_rate = 0.0
        self._last_reward = 0.0
        self._last_stability = 0.0
        self._last_write_gate = 0.0
        self._last_step_scale = 0.0
        self._last_momentum_gate = 0.0
        self._last_slow_mix = 0.0
        self._last_reset_mix = 0.0
        self._last_confidence = 0.0

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def zero_input_columns(self, *, start: int, end: int) -> None:
        """Reset input_projection columns ``[start:end)`` to zero.

        Used by callers that semantically partition the input feature space
        (e.g. CMS's ATLAS/Titans uplift, where columns reserved for PE
        features must start clean when restoring a legacy state).
        """
        if start < 0 or end <= start or start >= self._feature_dim:
            return
        end = min(end, self._feature_dim)
        self._input_projection = tuple(
            tuple(
                0.0 if start <= index < end else value
                for index, value in enumerate(row)
            )
            for row in self._input_projection
        )

    def restore_state(self, state: LearnedUpdateRuleState) -> None:
        if state.feature_dim != self._feature_dim or state.hidden_dim != self._hidden_dim:
            input_projection = tuple(
                tuple(
                    row[index % len(row)] if row else 0.0
                    for index in range(self._feature_dim)
                )
                for row in (
                    state.input_projection[row_index % len(state.input_projection)]
                    if state.input_projection
                    else tuple(0.0 for _ in range(state.feature_dim))
                    for row_index in range(self._hidden_dim)
                )
            )
            hidden_bias = tuple(
                state.hidden_bias[index % len(state.hidden_bias)] if state.hidden_bias else 0.0
                for index in range(self._hidden_dim)
            )
            output_projection = tuple(
                tuple(
                    row[index % len(row)] if row else 0.0
                    for index in range(self._hidden_dim)
                )
                for row in (
                    state.output_projection[row_index % len(state.output_projection)]
                    if state.output_projection
                    else tuple(0.0 for _ in range(state.hidden_dim))
                    for row_index in range(self._OUTPUT_DIM)
                )
            )
        else:
            input_projection = state.input_projection
            hidden_bias = state.hidden_bias
            output_projection = state.output_projection
        self._input_projection = input_projection
        self._hidden_bias = hidden_bias
        self._output_projection = output_projection
        self._output_bias = state.output_bias
        self._update_count = state.update_count
        self._last_feature_norm = state.last_feature_norm
        self._last_improvement = state.last_improvement
        self._last_guard_reason = state.last_guard_reason
        self._last_decisions = state.last_decisions
        self._last_effective_learning_rate = state.last_effective_learning_rate
        self._last_reward = state.last_reward
        self._last_stability = state.last_stability
        self._last_write_gate = state.last_write_gate
        self._last_step_scale = state.last_step_scale
        self._last_momentum_gate = state.last_momentum_gate
        self._last_slow_mix = state.last_slow_mix
        self._last_reset_mix = state.last_reset_mix
        self._last_confidence = state.last_confidence

    def export_state(self) -> LearnedUpdateRuleState:
        return LearnedUpdateRuleState(
            rule_id=self._rule_id,
            feature_dim=self._feature_dim,
            hidden_dim=self._hidden_dim,
            update_count=self._update_count,
            last_feature_norm=self._last_feature_norm,
            last_improvement=self._last_improvement,
            last_guard_reason=self._last_guard_reason,
            input_projection=self._input_projection,
            hidden_bias=self._hidden_bias,
            output_projection=self._output_projection,
            output_bias=self._output_bias,
            last_decisions=self._last_decisions,
            base_learning_rate=self._learning_rate,
            last_effective_learning_rate=self._last_effective_learning_rate,
            last_reward=self._last_reward,
            last_stability=self._last_stability,
            last_write_gate=self._last_write_gate,
            last_step_scale=self._last_step_scale,
            last_momentum_gate=self._last_momentum_gate,
            last_slow_mix=self._last_slow_mix,
            last_reset_mix=self._last_reset_mix,
            last_confidence=self._last_confidence,
            description=(
                f"Learned update rule {self._rule_id} dim={self._feature_dim} hidden={self._hidden_dim} "
                f"updates={self._update_count} improvement={self._last_improvement:.3f} "
                f"eff_lr={self._last_effective_learning_rate:.3f} reward={self._last_reward:.3f} "
                f"guard={self._last_guard_reason or 'clear'}."
            ),
            # New rules write feature_version=2 only when they use the
            # canonical PE-aware feature layout (16 dims, trailing 4 reserved
            # for PE). Other rules (e.g. metacontroller) keep version=1.
            feature_version=(
                2
                if self._feature_dim == LEARNED_UPDATE_PE_AWARE_FEATURE_DIM
                else 1
            ),
        )

    def decide(self, *, target_id: str, features: tuple[float, ...]) -> LearnedUpdateDecision:
        aligned = self._align_features(features)
        feature_norm = _mean_abs(aligned)
        self._last_feature_norm = feature_norm
        target_signature = _target_signature(target_id)
        previous_target_decision = next(
            (decision for decision in self._last_decisions if decision.target_id == target_id),
            None,
        )
        target_gate_bias = (
            previous_target_decision.write_gate - 0.5
            if previous_target_decision is not None
            else target_signature[0] * 0.12
        )
        target_step_bias = (
            previous_target_decision.step_scale - 0.5
            if previous_target_decision is not None
            else target_signature[1] * 0.10
        )
        target_slow_bias = (
            previous_target_decision.slow_mix - 0.5
            if previous_target_decision is not None
            else target_signature[2] * 0.08
        )
        momentum_memory = self._last_momentum_gate - 0.5
        historical_reward = self._last_reward
        historical_stability = self._last_stability
        trend_signal = _clamp_signed(
            self._last_improvement * 0.55
            + historical_reward * 0.25
            + (historical_stability - 0.5) * 0.20
        )
        hidden = tuple(
            math.tanh(
                sum(weight * value for weight, value in zip(row, aligned, strict=True)) + bias
            )
            for row, bias in zip(self._input_projection, self._hidden_bias, strict=True)
        )
        raw = tuple(
            sum(weight * value for weight, value in zip(row, hidden, strict=True)) + bias
            for row, bias in zip(self._output_projection, self._output_bias, strict=True)
        )
        guard_applied = False
        guard_reason = ""
        overload_pressure = _clamp_unit(
            (feature_norm - (0.72 + historical_stability * 0.08)) / 0.30
        )
        instability_pressure = _clamp_unit(
            (0.55 - historical_stability) / 0.45
        )
        write_gate = _sigmoid(
            raw[0]
            + trend_signal * 0.45
            + target_gate_bias * 0.80
            - overload_pressure * 1.35
        )
        step_scale = _sigmoid(
            raw[1]
            + trend_signal * 0.55
            + target_step_bias * 0.85
            - overload_pressure * 1.20
            - instability_pressure * 0.45
        )
        momentum_gate = _sigmoid(
            raw[2]
            + momentum_memory * 0.65
            + trend_signal * 0.25
            - overload_pressure * 0.55
        )
        slow_mix = _sigmoid(
            raw[3]
            + target_slow_bias * 0.90
            + max(-historical_reward, 0.0) * 0.35
            + instability_pressure * 0.30
        )
        reset_mix = _sigmoid(
            raw[4]
            + overload_pressure * 0.95
            + instability_pressure * 0.60
            - target_gate_bias * 0.25
        )
        bias_delta = math.tanh(
            raw[5]
            + trend_signal * 0.30
            + target_signature[0] * 0.18
            - overload_pressure * 0.20
        )
        confidence = _sigmoid(
            raw[6]
            + historical_stability * 0.70
            + max(self._last_improvement, 0.0) * 0.35
            - overload_pressure * 0.85
            - instability_pressure * 0.45
        )
        if overload_pressure > 0.55 or (feature_norm > 0.92 and historical_stability < 0.45):
            guard_applied = True
            if overload_pressure > 0.55:
                guard_reason = "feature-overload"
            else:
                guard_reason = "instability-trend"
            suppression = 1.0 - max(overload_pressure * 0.45, instability_pressure * 0.30)
            step_scale *= suppression
            write_gate *= 1.0 - overload_pressure * 0.35
            momentum_gate *= 1.0 - overload_pressure * 0.18
            bias_delta *= 1.0 - max(overload_pressure, instability_pressure) * 0.40
            slow_mix *= 1.0 - overload_pressure * 0.20
            reset_mix = _clamp_unit(reset_mix + overload_pressure * 0.18 + instability_pressure * 0.10)
            confidence *= 1.0 - max(overload_pressure, instability_pressure) * 0.32
        decision = LearnedUpdateDecision(
            target_id=target_id,
            write_gate=_clamp_unit(write_gate),
            step_scale=_clamp_unit(step_scale),
            momentum_gate=_clamp_unit(momentum_gate),
            slow_mix=_clamp_unit(slow_mix),
            reset_mix=_clamp_unit(reset_mix),
            bias_delta=_clamp_signed(bias_delta),
            confidence=_clamp_unit(confidence),
            guard_applied=guard_applied,
            guard_reason=guard_reason,
            description=(
                f"{target_id} gate={write_gate:.2f} step={step_scale:.2f} momentum={momentum_gate:.2f} "
                f"slow_mix={slow_mix:.2f} reset_mix={reset_mix:.2f} bias={bias_delta:.2f} "
                f"confidence={confidence:.2f}"
            ),
        )
        self._last_guard_reason = guard_reason
        self._last_decisions = tuple(dec for dec in self._last_decisions if dec.target_id != target_id) + (decision,)
        self._last_write_gate = decision.write_gate
        self._last_step_scale = decision.step_scale
        self._last_momentum_gate = decision.momentum_gate
        self._last_slow_mix = decision.slow_mix
        self._last_reset_mix = decision.reset_mix
        self._last_confidence = decision.confidence
        self._last_effective_learning_rate = self._learning_rate * decision.step_scale * max(decision.write_gate, 0.05)
        return decision

    def learn(
        self,
        *,
        features: tuple[float, ...],
        decision: LearnedUpdateDecision,
        improvement: float,
        stability: float,
    ) -> None:
        aligned = self._align_features(features)
        hidden = tuple(
            math.tanh(
                sum(weight * value for weight, value in zip(row, aligned, strict=True)) + bias
            )
            for row, bias in zip(self._input_projection, self._hidden_bias, strict=True)
        )
        stability = _clamp_unit(stability)
        improvement_trend = improvement * 0.70 + self._last_improvement * 0.30
        stability_memory = stability * 0.65 + self._last_stability * 0.35
        reward = _clamp_signed(
            improvement_trend * 2.40
            + stability_memory * 0.55
            + self._last_reward * 0.20
            - max(self._last_feature_norm - 0.82, 0.0) * 0.80
        )
        self._last_reward = reward
        self._last_stability = stability_memory
        target_signature = _target_signature(decision.target_id)
        target_adaptation = 0.55 + abs(target_signature[0]) * 0.20 + abs(target_signature[1]) * 0.15
        recover_pressure = _clamp_unit((0.58 - stability_memory) / 0.58)
        desired_gate = _clamp_unit(
            decision.write_gate
            + reward * 0.16 * target_adaptation
            - recover_pressure * 0.06
        )
        desired_step = _clamp_unit(
            decision.step_scale
            + reward * 0.24 * target_adaptation
            + (1.0 - stability_memory) * 0.08
            - max(self._last_feature_norm - 0.90, 0.0) * 0.35
        )
        desired_momentum = _clamp_unit(
            decision.momentum_gate
            + reward * 0.10
            + stability_memory * 0.08
        )
        desired_slow_mix = _clamp_unit(
            decision.slow_mix
            + max(-reward, 0.0) * 0.12
            + recover_pressure * 0.20
            + abs(target_signature[2]) * 0.06
        )
        desired_reset_mix = _clamp_unit(
            decision.reset_mix
            + recover_pressure * 0.24
            + max(-reward, 0.0) * 0.08
        )
        desired_bias = _clamp_signed(
            decision.bias_delta + reward * 0.16 + target_signature[0] * 0.05
        )
        desired_confidence = _clamp_unit(
            decision.confidence
            + reward * 0.18
            + stability_memory * 0.12
            - recover_pressure * 0.10
        )
        targets = (
            desired_gate,
            desired_step,
            desired_momentum,
            desired_slow_mix,
            desired_reset_mix,
            desired_bias,
            desired_confidence,
        )
        current = (
            decision.write_gate,
            decision.step_scale,
            decision.momentum_gate,
            decision.slow_mix,
            decision.reset_mix,
            decision.bias_delta,
            decision.confidence,
        )
        output_errors = tuple(targets[index] - current[index] for index in range(self._OUTPUT_DIM))
        adaptive_learning_rate = self._learning_rate * (
            0.45
            + decision.confidence * 0.30
            + decision.write_gate * 0.15
            + stability_memory * 0.10
        )
        adaptive_learning_rate *= 1.0 - max(self._last_feature_norm - 0.88, 0.0) * 0.45
        adaptive_learning_rate = max(self._learning_rate * 0.25, min(self._learning_rate * 1.35, adaptive_learning_rate))
        new_output_bias = []
        new_output_projection: list[tuple[float, ...]] = []
        for row_index, row in enumerate(self._output_projection):
            delta = output_errors[row_index]
            new_output_bias.append(_clamp_signed(self._output_bias[row_index] + delta * adaptive_learning_rate * 0.6))
            new_output_projection.append(
                tuple(
                    _clamp_signed(weight + delta * hidden[col_index] * adaptive_learning_rate)
                    for col_index, weight in enumerate(row)
                )
            )
        hidden_feedback = []
        for hidden_index in range(self._hidden_dim):
            routed = sum(
                output_errors[row_index] * self._output_projection[row_index][hidden_index]
                for row_index in range(self._OUTPUT_DIM)
            )
            hidden_feedback.append(routed * (1.0 - hidden[hidden_index] * hidden[hidden_index]))
        new_hidden_bias = []
        new_input_projection: list[tuple[float, ...]] = []
        for row_index, row in enumerate(self._input_projection):
            delta = hidden_feedback[row_index]
            new_hidden_bias.append(_clamp_signed(self._hidden_bias[row_index] + delta * adaptive_learning_rate * 0.5))
            new_input_projection.append(
                tuple(
                    _clamp_signed(weight + delta * aligned[col_index] * adaptive_learning_rate * 0.7)
                    for col_index, weight in enumerate(row)
                )
            )
        self._output_bias = tuple(new_output_bias)
        self._output_projection = tuple(new_output_projection)
        self._hidden_bias = tuple(new_hidden_bias)
        self._input_projection = tuple(new_input_projection)
        self._update_count += 1
        self._last_improvement = improvement_trend
        self._last_effective_learning_rate = adaptive_learning_rate

    def _align_features(self, features: tuple[float, ...]) -> tuple[float, ...]:
        if not features:
            return tuple(0.0 for _ in range(self._feature_dim))
        if len(features) == self._feature_dim:
            aligned = tuple(float(value) for value in features)
        else:
            aligned = tuple(float(features[index % len(features)]) for index in range(self._feature_dim))
        mean_value = _mean(aligned)
        centered = tuple(value - mean_value for value in aligned)
        scale = max(_rms(centered), 0.35 + self._last_feature_norm * 0.40)
        normalized = tuple(_clamp_signed(value / scale) for value in centered)
        if all(abs(value) < 1e-9 for value in normalized):
            return tuple(_clamp_signed(value) for value in aligned)
        return normalized
