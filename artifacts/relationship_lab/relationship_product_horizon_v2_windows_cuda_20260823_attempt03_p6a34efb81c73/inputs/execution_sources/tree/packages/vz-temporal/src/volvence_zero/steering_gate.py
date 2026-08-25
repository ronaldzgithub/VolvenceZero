"""Temporal owner for the bounded {noop, steer} policy decision."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
import statistics
from typing import Mapping

from volvence_zero.credit.gate import CreditRecord, CreditSnapshot
from volvence_zero.prediction import PredictionErrorSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.steering_contracts import (
    STEERING_CONDITION_BELIEF_SLOT,
    STEERING_GATE_DECISION_SLOT,
    STEERING_GATE_CHECKPOINT_SCHEMA_VERSION,
    SteeringConditionBelief,
    SteeringGateAction,
    SteeringGateArtifact,
    SteeringGateCheckpoint,
    SteeringGateDecision,
)


_SUPPORTED_FEATURES = frozenset(
    {
        "belief_margin",
        "fresh_margin",
        "belief_disagrees_fresh",
        "base_action_entropy",
        "prediction_error_magnitude",
        "staleness_proxy",
    }
)


def _softmax_pair(left: float, right: float) -> tuple[float, float]:
    maximum = max(left, right)
    left_exp = math.exp(left - maximum)
    right_exp = math.exp(right - maximum)
    total = left_exp + right_exp
    return left_exp / total, right_exp / total


@dataclass(frozen=True)
class SteeringGateLearningReport:
    consumed_record_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    mean_terminal_credit: float
    mean_directional_terminal_credit: float
    old_policy_version: int
    new_policy_version: int
    update_applied: bool
    description: str


class SteeringGateModule(RuntimeModule[SteeringGateDecision]):
    """Publish a frozen-policy gate decision over owner-published PE proxies."""

    slot_name = STEERING_GATE_DECISION_SLOT
    owner = "SteeringGateModule"
    value_type = SteeringGateDecision
    dependencies = (STEERING_CONDITION_BELIEF_SLOT, "prediction_error")
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        artifact: SteeringGateArtifact,
        learning_rate: float = 0.05,
        learning_enabled: bool = True,
        decision_mode: str = "frozen-policy-argmax",
        exploration_seed: int = 0,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        unsupported = tuple(
            name for name in artifact.feature_names if name not in _SUPPORTED_FEATURES
        )
        if unsupported:
            raise ValueError(
                f"unsupported steering gate feature names: {unsupported!r}"
            )
        if not math.isfinite(learning_rate) or not 0.0 < learning_rate <= 0.2:
            raise ValueError("steering gate learning_rate must be within (0, 0.2]")
        self._artifact = artifact
        self._learning_rate = learning_rate
        self._learning_enabled = bool(learning_enabled)
        if decision_mode not in {
            "frozen-policy-argmax",
            "evidence-stochastic",
        }:
            raise ValueError("steering gate decision_mode is unsupported")
        if exploration_seed < 0:
            raise ValueError("steering gate exploration_seed must be non-negative")
        if (
            decision_mode == "evidence-stochastic"
            and self.wiring_level is WiringLevel.ACTIVE
        ):
            raise ValueError(
                "evidence-stochastic steering gate is forbidden in ACTIVE wiring"
            )
        self._decision_mode = decision_mode
        self._exploration_seed = exploration_seed
        self._rng = random.Random(exploration_seed)
        self._decision_count = 0
        self._pending_decisions: dict[str, SteeringGateDecision] = {}
        self._consumed_credit_record_ids: set[str] = set()

    @property
    def artifact(self) -> SteeringGateArtifact:
        return self._artifact

    def install_artifact(self, artifact: SteeringGateArtifact) -> None:
        if artifact.feature_names != self._artifact.feature_names:
            raise ValueError("gate artifact feature schema cannot change in-place")
        if artifact.policy_version <= self._artifact.policy_version:
            raise ValueError("gate artifact policy_version must increase")
        self._artifact = artifact

    def set_learning_enabled(self, enabled: bool) -> None:
        self._learning_enabled = bool(enabled)

    def export_checkpoint(self, *, checkpoint_id: str) -> SteeringGateCheckpoint:
        return SteeringGateCheckpoint(
            schema_version=STEERING_GATE_CHECKPOINT_SCHEMA_VERSION,
            checkpoint_id=checkpoint_id,
            artifact=self._artifact,
            learning_rate=self._learning_rate,
            learning_enabled=self._learning_enabled,
            decision_mode=self._decision_mode,
            exploration_seed=self._exploration_seed,
            decision_count=self._decision_count,
            pending_decisions=tuple(self._pending_decisions.values()),
            consumed_credit_record_ids=tuple(
                sorted(self._consumed_credit_record_ids)
            ),
            description=(
                "Complete steering gate owner state for exact rollback and "
                "continuation."
            ),
        )

    def restore_checkpoint(self, checkpoint: SteeringGateCheckpoint) -> None:
        if not isinstance(checkpoint, SteeringGateCheckpoint):
            raise TypeError("restore_checkpoint requires SteeringGateCheckpoint")
        if checkpoint.learning_rate != self._learning_rate:
            raise ValueError("steering gate checkpoint learning-rate drift")
        if checkpoint.decision_mode != self._decision_mode:
            raise ValueError("steering gate checkpoint decision-mode drift")
        if checkpoint.exploration_seed != self._exploration_seed:
            raise ValueError("steering gate checkpoint exploration-seed drift")
        pending = {
            decision.decision_id: decision
            for decision in checkpoint.pending_decisions
        }
        consumed = set(checkpoint.consumed_credit_record_ids)
        rng = random.Random(self._exploration_seed)
        if self._decision_mode == "evidence-stochastic":
            for _ in range(checkpoint.decision_count):
                rng.random()
        self._artifact = checkpoint.artifact
        self._learning_enabled = checkpoint.learning_enabled
        self._decision_count = checkpoint.decision_count
        self._pending_decisions = pending
        self._consumed_credit_record_ids = consumed
        self._rng = rng

    def settle_terminal_credit(
        self,
        credit_snapshot: CreditSnapshot,
    ) -> SteeringGateLearningReport:
        records = tuple(
            record
            for record in credit_snapshot.recent_credits
            if record.level == "steering_terminal_prediction_error"
            and record.record_id not in self._consumed_credit_record_ids
        )
        old_version = self._artifact.policy_version
        if not records:
            return SteeringGateLearningReport(
                consumed_record_ids=(),
                episode_ids=(),
                mean_terminal_credit=0.0,
                mean_directional_terminal_credit=0.0,
                old_policy_version=old_version,
                new_policy_version=old_version,
                update_applied=False,
                description="No new steering terminal credit records.",
            )
        missing = tuple(
            record.prediction_id
            for record in records
            if record.prediction_id not in self._pending_decisions
        )
        if missing:
            raise ValueError(
                "steering terminal credit references unknown decisions: "
                f"{missing!r}"
            )
        by_episode: dict[str, list[CreditRecord]] = {}
        for record in records:
            if not record.environment_outcome_id:
                raise ValueError("steering terminal credit requires episode lineage")
            by_episode.setdefault(record.environment_outcome_id, []).append(record)

        weights = [list(row) for row in self._artifact.weights]
        bias = list(self._artifact.bias)
        update_applied = False
        directional_credits: list[float] = []
        for episode_id in sorted(by_episode):
            episode_records = by_episode[episode_id]
            gradients = [
                [0.0, 0.0] for _ in self._artifact.feature_names
            ]
            bias_gradient = [0.0, 0.0]
            for record in episode_records:
                decision = self._pending_decisions[record.prediction_id]
                directional_credit = (
                    record.credit_value
                    if decision.action is SteeringGateAction.STEER
                    else -record.credit_value
                )
                directional_credits.append(directional_credit)
                if not self._learning_enabled or abs(directional_credit) <= 1e-12:
                    continue
                probability = decision.steer_probability
                action_index = (
                    1 if decision.action is SteeringGateAction.STEER else 0
                )
                action_probabilities = (1.0 - probability, probability)
                observations = dict(decision.observations)
                for output_index in range(2):
                    score = (
                        float(output_index == action_index)
                        - action_probabilities[output_index]
                    )
                    bias_gradient[output_index] += directional_credit * score
                    for feature_index, name in enumerate(
                        self._artifact.feature_names
                    ):
                        gradients[feature_index][output_index] += (
                            directional_credit * observations[name] * score
                        )
            divisor = float(len(episode_records))
            if not self._learning_enabled or not any(
                abs(value) > 1e-12
                for value in directional_credits[-len(episode_records) :]
            ):
                continue
            for feature_index in range(len(weights)):
                for output_index in range(2):
                    candidate = weights[feature_index][output_index] + (
                        self._learning_rate
                        * gradients[feature_index][output_index]
                        / divisor
                    )
                    weights[feature_index][output_index] = max(
                        -8.0, min(8.0, candidate)
                    )
            for output_index in range(2):
                candidate = bias[output_index] + (
                    self._learning_rate
                    * bias_gradient[output_index]
                    / divisor
                )
                bias[output_index] = max(-8.0, min(8.0, candidate))
            update_applied = True

        if update_applied:
            next_version = old_version + 1
            base_artifact_id = self._artifact.artifact_id.split(":online:", 1)[0]
            self._artifact = replace(
                self._artifact,
                artifact_id=f"{base_artifact_id}:online:{next_version}",
                weights=tuple(tuple(row) for row in weights),
                bias=tuple(bias),
                policy_version=next_version,
                description=(
                    "Online-fast steering gate updated only from PE-owned "
                    "terminal counterfactual credit."
                ),
            )
        for record in records:
            self._consumed_credit_record_ids.add(record.record_id)
            self._pending_decisions.pop(record.prediction_id)
        return SteeringGateLearningReport(
            consumed_record_ids=tuple(record.record_id for record in records),
            episode_ids=tuple(sorted(by_episode)),
            mean_terminal_credit=statistics.fmean(
                record.credit_value for record in records
            ),
            mean_directional_terminal_credit=statistics.fmean(
                directional_credits
            ),
            old_policy_version=old_version,
            new_policy_version=self._artifact.policy_version,
            update_applied=update_applied,
            description=(
                "Steering gate terminal credit consumed from Credit owner; "
                "evaluation readouts were not inputs."
            ),
        )

    def _decide_observations(
        self,
        *,
        observations: tuple[tuple[str, float], ...],
    ) -> SteeringGateDecision:
        if tuple(name for name, _ in observations) != self._artifact.feature_names:
            raise ValueError("steering gate observation feature schema drift")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for _, value in observations
        ):
            raise ValueError("steering gate observations must be within [0, 1]")
        logits = tuple(
            self._artifact.bias[action_index]
            + sum(
                value * self._artifact.weights[index][action_index]
                for index, (_, value) in enumerate(observations)
            )
            for action_index in range(2)
        )
        _, steer_probability = _softmax_pair(*logits)
        if self._decision_mode == "evidence-stochastic":
            action = (
                SteeringGateAction.STEER
                if self._rng.random() < steer_probability
                else SteeringGateAction.NOOP
            )
        else:
            action = (
                SteeringGateAction.STEER
                if logits[1] > logits[0]
                else SteeringGateAction.NOOP
            )
        self._decision_count += 1
        decision = SteeringGateDecision(
            decision_id=(
                f"{self._artifact.artifact_id}:decision:{self._decision_count}"
            ),
            action=action,
            steer_probability=steer_probability,
            observations=observations,
            policy_artifact_id=self._artifact.artifact_id,
            policy_version=self._artifact.policy_version,
            terminal_credit_pending=True,
            decision_mode=self._decision_mode,
            description=(
                "Bounded steering gate decision from owner-published "
                "belief/PE proxy observations; no evaluation readout consumed."
            ),
        )
        self._pending_decisions[decision.decision_id] = decision
        if len(self._pending_decisions) > 2048:
            oldest = next(iter(self._pending_decisions))
            del self._pending_decisions[oldest]
        return decision

    def replay_observations(
        self,
        observations: tuple[tuple[str, float], ...],
    ) -> Snapshot[SteeringGateDecision]:
        """Replay owner-published observations for preregistered evidence.

        The replay surface consumes the exact immutable observation tuple from
        a prior SHADOW decision.  It never reconstructs a PE or sensor state
        from raw dialogue text.
        """

        return self.publish(self._decide_observations(observations=observations))

    def _decide(
        self,
        *,
        belief: SteeringConditionBelief,
        prediction_error: PredictionErrorSnapshot,
    ) -> SteeringGateDecision:
        feature_values = {
            "belief_margin": belief.belief_margin,
            "fresh_margin": belief.fresh_margin,
            "belief_disagrees_fresh": float(belief.belief_disagrees_fresh),
            "base_action_entropy": belief.base_action_entropy,
            "prediction_error_magnitude": min(
                1.0,
                max(0.0, prediction_error.error.magnitude / 4.0),
            ),
            "staleness_proxy": belief.staleness_proxy,
        }
        observations = tuple(
            (name, feature_values[name]) for name in self._artifact.feature_names
        )
        return self._decide_observations(observations=observations)

    async def process(
        self,
        upstream: Mapping[str, Snapshot[object]],
    ) -> Snapshot[SteeringGateDecision]:
        belief_snapshot = upstream[STEERING_CONDITION_BELIEF_SLOT]
        prediction_snapshot = upstream["prediction_error"]
        if not isinstance(belief_snapshot.value, SteeringConditionBelief):
            raise TypeError("steering gate requires SteeringConditionBelief")
        if not isinstance(prediction_snapshot.value, PredictionErrorSnapshot):
            raise TypeError("steering gate requires PredictionErrorSnapshot")
        return self.publish(
            self._decide(
                belief=belief_snapshot.value,
                prediction_error=prediction_snapshot.value,
            )
        )

    async def process_standalone(
        self,
        **kwargs: object,
    ) -> Snapshot[SteeringGateDecision]:
        belief = kwargs.get("belief")
        prediction_error = kwargs.get("prediction_error")
        if not isinstance(belief, SteeringConditionBelief):
            raise TypeError("process_standalone requires belief")
        if not isinstance(prediction_error, PredictionErrorSnapshot):
            raise TypeError("process_standalone requires prediction_error")
        return self.publish(
            self._decide(belief=belief, prediction_error=prediction_error)
        )


__all__ = ("SteeringGateLearningReport", "SteeringGateModule")
