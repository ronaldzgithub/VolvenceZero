from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.credit import CreditRecord
from volvence_zero.memory import Track
from volvence_zero.substrate import SubstrateSnapshot
from volvence_zero.temporal import (
    ControllerState,
    LearnedLiteTemporalPolicy,
    TemporalAbstractionSnapshot,
)


@dataclass(frozen=True)
class ZTransition:
    step_index: int
    abstract_action: str
    controller_state: ControllerState
    reward: float


@dataclass(frozen=True)
class ZRollout:
    rollout_id: str
    transitions: tuple[ZTransition, ...]
    total_reward: float
    description: str


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


class InternalRLSandbox:
    """Minimal z-space rollout sandbox for abstract-action RL experiments."""

    def __init__(self, *, policy: LearnedLiteTemporalPolicy | None = None) -> None:
        self._policy = policy or LearnedLiteTemporalPolicy()

    @property
    def policy(self) -> LearnedLiteTemporalPolicy:
        return self._policy

    def rollout(
        self,
        *,
        rollout_id: str,
        substrate_steps: tuple[SubstrateSnapshot, ...],
    ) -> ZRollout:
        previous_snapshot: TemporalAbstractionSnapshot | None = None
        transitions: list[ZTransition] = []
        for step_index, substrate_snapshot in enumerate(substrate_steps):
            temporal_step = self._policy.step(
                substrate_snapshot=substrate_snapshot,
                previous_snapshot=previous_snapshot,
                memory_snapshot=None,
                reflection_snapshot=None,
            )
            reward = self._reward_for_step(temporal_step.controller_state)
            transitions.append(
                ZTransition(
                    step_index=step_index,
                    abstract_action=temporal_step.active_abstract_action,
                    controller_state=temporal_step.controller_state,
                    reward=reward,
                )
            )
            previous_snapshot = TemporalAbstractionSnapshot(
                controller_state=temporal_step.controller_state,
                active_abstract_action=temporal_step.active_abstract_action,
                controller_params_hash=temporal_step.controller_params_hash,
                description=temporal_step.description,
            )
        total_reward = sum(transition.reward for transition in transitions)
        return ZRollout(
            rollout_id=rollout_id,
            transitions=tuple(transitions),
            total_reward=total_reward,
            description=(
                f"Internal RL rollout over {len(transitions)} abstract actions with total reward "
                f"{total_reward:.2f}."
            ),
        )

    def optimize(self, rollout: ZRollout) -> None:
        if not rollout.transitions:
            return
        average_reward = rollout.total_reward / len(rollout.transitions)
        reward_scale = max(average_reward, 0.05)
        self._policy.fit_from_signals(
            residual_strength=max(0.4, reward_scale),
            memory_strength=0.25,
            reflection_strength=0.35 if average_reward > 0.4 else 0.2,
        )

    def _reward_for_step(self, controller_state: ControllerState) -> float:
        reward = sum(controller_state.code) / max(len(controller_state.code), 1)
        if controller_state.is_switching:
            reward += 0.1
        reward -= controller_state.steps_since_switch * 0.02
        return _clamp(reward)


def derive_abstract_action_credit(
    *,
    rollout: ZRollout,
    timestamp_ms: int,
) -> tuple[CreditRecord, ...]:
    records: list[CreditRecord] = []
    for transition in rollout.transitions:
        records.append(
            CreditRecord(
                record_id=f"{rollout.rollout_id}:{transition.step_index}",
                level="abstract_action",
                track=Track.SHARED,
                source_event=transition.abstract_action,
                credit_value=_clamp(transition.reward),
                context=rollout.description,
                timestamp_ms=timestamp_ms + transition.step_index,
            )
        )
    return tuple(records)
