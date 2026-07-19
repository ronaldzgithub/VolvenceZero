"""Small end-to-end PPO baseline for fair digital-ant comparisons.

This baseline intentionally bypasses ``z_t``: the MLP maps the same frozen
``sense_encode`` vector directly to bounded motor commands.  It is never used
by the VZ learning arm and may read sparse delivery facts only inside its own
offline trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from volvence_ant.env.ant_world import AntWorld
from volvence_ant.substrate.navigator import AntNavigator
from volvence_ant.substrate.sense_encode import SENSE_CHANNELS, sense_encode


@dataclass(frozen=True)
class PPOConfig:
    episodes: int = 8
    ticks_per_episode: int = 96
    update_epochs: int = 4
    learning_rate: float = 3e-3
    gamma: float = 0.98
    clip_ratio: float = 0.2
    entropy_weight: float = 0.01


@dataclass(frozen=True)
class E2EEvaluation:
    food_pickups: int
    food_delivered: int
    positions: tuple[tuple[float, float], ...]


class E2ERLAnt:
    """Torch PPO policy with direct observation-to-motor control."""

    def __init__(self, *, seed: int, hidden_dim: int = 32) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("E2ERLAnt requires the 'torch' extra") from exc

        torch.manual_seed(seed)
        self._torch = torch
        self._policy = torch.nn.Sequential(
            torch.nn.Linear(len(SENSE_CHANNELS), hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 3),
        )
        self._log_std = torch.nn.Parameter(torch.full((2,), -0.5))

    def parameter_digest(self) -> tuple[float, ...]:
        torch = self._torch
        with torch.no_grad():
            flat = torch.cat(
                [
                    *(parameter.detach().flatten() for parameter in self._policy.parameters()),
                    self._log_std.detach().flatten(),
                ]
            )
        return tuple(float(value) for value in flat[:16])

    def _forward(self, observation: np.ndarray) -> tuple[object, object]:
        torch = self._torch
        output = self._policy(torch.as_tensor(observation, dtype=torch.float32))
        return output[:2], output[2]

    @staticmethod
    def _command(raw_action: np.ndarray, world: AntWorld) -> tuple[float, float]:
        turn = float(np.tanh(raw_action[0]) * world.config.max_turn_rate)
        step = float((np.tanh(raw_action[1]) + 1.0) * 0.5 * world.config.step_size)
        return turn, step

    def train(
        self,
        *,
        world_factory: Callable[[int], AntWorld],
        seed: int,
        config: PPOConfig | None = None,
    ) -> None:
        cfg = config or PPOConfig()
        torch = self._torch
        optimizer = torch.optim.Adam(
            (*self._policy.parameters(), self._log_std),
            lr=cfg.learning_rate,
        )
        for episode in range(cfg.episodes):
            world = world_factory(seed + episode)
            navigator = AntNavigator(
                step_size=world.config.step_size,
                heading_noise=0.0,
                step_noise=0.0,
                seed=seed + episode,
            )
            navigator.reset(initial_heading=world.observe().eval_true_heading)
            observations: list[np.ndarray] = []
            actions: list[object] = []
            old_log_probs: list[object] = []
            values: list[object] = []
            rewards: list[float] = []
            for _ in range(cfg.ticks_per_episode):
                observation = world.observe()
                encoded = sense_encode(
                    observation,
                    navigator.state,
                    turn_command_scale=world.config.max_turn_rate,
                )
                mean, value = self._forward(encoded)
                distribution = torch.distributions.Normal(mean, self._log_std.exp())
                raw_action = distribution.sample()
                log_prob = distribution.log_prob(raw_action).sum()
                turn, step = self._command(raw_action.detach().numpy(), world)
                navigator.update(turn_command=turn, step_command=step)
                world.act(turn_command=turn, step_command=step)
                transition = world.last_transition()
                observations.append(encoded)
                actions.append(raw_action.detach())
                old_log_probs.append(log_prob.detach())
                values.append(value.detach())
                rewards.append(1.0 if transition.delivered else 0.0)
            returns: list[float] = []
            running = 0.0
            for reward in reversed(rewards):
                running = reward + cfg.gamma * running
                returns.append(running)
            returns.reverse()
            obs_tensor = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
            action_tensor = torch.stack(actions)
            old_log_prob_tensor = torch.stack(old_log_probs)
            return_tensor = torch.as_tensor(returns, dtype=torch.float32)
            value_tensor = torch.stack(values)
            advantages = return_tensor - value_tensor
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            for _ in range(cfg.update_epochs):
                output = self._policy(obs_tensor)
                distribution = torch.distributions.Normal(
                    output[:, :2], self._log_std.exp()
                )
                log_probs = distribution.log_prob(action_tensor).sum(dim=-1)
                ratio = (log_probs - old_log_prob_tensor).exp()
                clipped = torch.clamp(
                    ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio
                )
                policy_loss = -torch.minimum(
                    ratio * advantages, clipped * advantages
                ).mean()
                value_loss = 0.5 * (output[:, 2] - return_tensor).square().mean()
                entropy = distribution.entropy().sum(dim=-1).mean()
                loss = policy_loss + value_loss - cfg.entropy_weight * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def evaluate(self, *, world: AntWorld, ticks: int, seed: int) -> E2EEvaluation:
        torch = self._torch
        navigator = AntNavigator(
            step_size=world.config.step_size,
            heading_noise=0.0,
            step_noise=0.0,
            seed=seed,
        )
        navigator.reset(initial_heading=world.observe().eval_true_heading)
        positions: list[tuple[float, float]] = []
        with torch.no_grad():
            for _ in range(ticks):
                encoded = sense_encode(
                    world.observe(),
                    navigator.state,
                    turn_command_scale=world.config.max_turn_rate,
                )
                mean, _ = self._forward(encoded)
                turn, step = self._command(mean.numpy(), world)
                navigator.update(turn_command=turn, step_command=step)
                world.act(turn_command=turn, step_command=step)
                positions.append(world.body().position)
        return E2EEvaluation(
            food_pickups=world.food_pickups,
            food_delivered=world.food_delivered,
            positions=tuple(positions),
        )


__all__ = ["E2EEvaluation", "E2ERLAnt", "PPOConfig"]
