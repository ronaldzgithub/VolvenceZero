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
from volvence_ant.substrate.sense_encode import (
    AntSenseSchema,
    sense_channels,
    sense_encode,
)

# Frozen-substrate body sensors, shared by EVERY matched-control arm.
#
# ``docs/specs/digital-ant-embodiment.md`` §3 requires the sky compass and the
# proprioceptive noise to be identical across arms: "罗盘是所有导航共用的
# substrate 传感器 ... 同一 frozen substrate 在 matched-control 各臂间一致".
# ``AntSessionConfig`` (kernel arms) and ``FixedRuleConfig`` (scripted arm)
# declare exactly these values; ``RandomAnt`` (the floor arm) reads them from
# here. This baseline used to run a noiseless, compass-less navigator, i.e. it
# was compared on a *different body* — a strictly easier path-integration
# problem than the arms it is meant to bound; ``RandomAnt`` had the same defect
# until 2026-07-27. ``tests/test_frozen_functions.py`` pins EVERY arm named in
# ``research/ant/05_ecology_p0_p1_p2_plan.md`` §5.4 (P2-B) to the same numbers.
SHARED_HEADING_NOISE: float = 0.01
SHARED_STEP_NOISE: float = 0.01
SHARED_COMPASS_GAIN: float = 0.85
SHARED_COMPASS_NOISE: float = 0.007


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

    def __init__(
        self,
        *,
        seed: int,
        hidden_dim: int = 32,
        sense_schema: AntSenseSchema = AntSenseSchema.V1,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("E2ERLAnt requires the 'torch' extra") from exc

        torch.manual_seed(seed)
        self._torch = torch
        self._sense_schema = sense_schema
        self._policy = torch.nn.Sequential(
            torch.nn.Linear(len(sense_channels(sense_schema)), hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 3),
        )
        self._log_std = torch.nn.Parameter(torch.full((2,), -0.5))
        self._navigators: dict[int, AntNavigator] = {}

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

    def _encode(
        self,
        world: AntWorld,
        navigator: AntNavigator,
        *,
        body_id: int,
    ) -> np.ndarray:
        return sense_encode(
            world.observe(body_id),
            navigator.state,
            turn_command_scale=world.config.max_turn_rate,
            schema=self._sense_schema,
        )

    def _episode_navigator(
        self,
        world: AntWorld,
        *,
        body_id: int,
        seed: int,
        synchronize: bool,
    ) -> AntNavigator:
        navigator = AntNavigator(
            step_size=world.config.step_size,
            heading_noise=SHARED_HEADING_NOISE,
            step_noise=SHARED_STEP_NOISE,
            compass_gain=SHARED_COMPASS_GAIN,
            compass_noise=SHARED_COMPASS_NOISE,
            seed=seed,
        )
        navigator.reset(
            initial_heading=world.observe(body_id).eval_true_heading
        )
        if synchronize:
            # Forced-start curriculum episodes reposition bodies away from the
            # nest. The kernel arms resynchronise path integration there, so
            # the baseline must receive the same home vector or its egocentric
            # home channels are corrupted and the comparison is unfair.
            body = world.body(body_id)
            navigator.sync_to(
                x=body.x,
                y=body.y,
                heading=body.heading,
                nest=world.nest,
            )
        return navigator

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
        episode_keys: tuple[int, ...] | None = None,
        n_bodies: int = 1,
        synchronize_navigators: bool = False,
    ) -> None:
        """Run offline PPO over ``episodes`` fresh worlds.

        ``episode_keys`` replays an externally frozen training schedule: the
        factory receives each key instead of ``seed + episode``, so a matched
        formal run can hand every arm the identical layout sequence. ``None``
        preserves the historical ``seed + episode`` walk.
        """

        cfg = config or PPOConfig()
        if n_bodies < 1:
            raise ValueError("n_bodies must be >= 1")
        if episode_keys is not None and not episode_keys:
            raise ValueError("episode_keys must be non-empty when provided")
        torch = self._torch
        optimizer = torch.optim.Adam(
            (*self._policy.parameters(), self._log_std),
            lr=cfg.learning_rate,
        )
        episodes = (
            len(episode_keys) if episode_keys is not None else cfg.episodes
        )
        for episode in range(episodes):
            world = world_factory(
                episode_keys[episode]
                if episode_keys is not None
                else seed + episode
            )
            navigators = {
                body_id: self._episode_navigator(
                    world,
                    body_id=body_id,
                    seed=seed + episode + body_id * 7919,
                    synchronize=synchronize_navigators,
                )
                for body_id in range(n_bodies)
            }
            observations: list[np.ndarray] = []
            actions: list[object] = []
            old_log_probs: list[object] = []
            values: list[object] = []
            # Per-body reward traces: discounted returns must not leak credit
            # across bodies that only share the policy, not a trajectory.
            body_rewards: dict[int, list[float]] = {
                body_id: [] for body_id in range(n_bodies)
            }
            body_slots: dict[int, list[int]] = {
                body_id: [] for body_id in range(n_bodies)
            }
            for _ in range(cfg.ticks_per_episode):
                for body_id in range(n_bodies):
                    navigator = navigators[body_id]
                    encoded = self._encode(world, navigator, body_id=body_id)
                    mean, value = self._forward(encoded)
                    distribution = torch.distributions.Normal(
                        mean, self._log_std.exp()
                    )
                    raw_action = distribution.sample()
                    log_prob = distribution.log_prob(raw_action).sum()
                    turn, step = self._command(
                        raw_action.detach().numpy(), world
                    )
                    # Act first, then integrate: the shared substrate fuses the
                    # sky-compass reading of the POST-move absolute heading,
                    # exactly as AntSession / FixedRuleAnt do.
                    moved = world.act(
                        turn_command=turn,
                        step_command=step,
                        body_id=body_id,
                    )
                    navigator.update(
                        turn_command=turn,
                        step_command=step,
                        true_heading=moved.eval_true_heading,
                    )
                    transition = world.last_transition(body_id)
                    body_slots[body_id].append(len(observations))
                    observations.append(encoded)
                    actions.append(raw_action.detach())
                    old_log_probs.append(log_prob.detach())
                    values.append(value.detach())
                    body_rewards[body_id].append(
                        1.0 if transition.delivered else 0.0
                    )
            returns_by_slot = [0.0] * len(observations)
            for body_id in range(n_bodies):
                running = 0.0
                for slot, reward in zip(
                    reversed(body_slots[body_id]),
                    reversed(body_rewards[body_id]),
                    strict=True,
                ):
                    running = reward + cfg.gamma * running
                    returns_by_slot[slot] = running
            returns = returns_by_slot
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
        navigator = self._episode_navigator(
            world, body_id=0, seed=seed, synchronize=False
        )
        positions: list[tuple[float, float]] = []
        with torch.no_grad():
            for _ in range(ticks):
                encoded = self._encode(world, navigator, body_id=0)
                mean, _ = self._forward(encoded)
                turn, step = self._command(mean.numpy(), world)
                moved = world.act(turn_command=turn, step_command=step)
                navigator.update(
                    turn_command=turn,
                    step_command=step,
                    true_heading=moved.eval_true_heading,
                )
                positions.append(world.body().position)
        return E2EEvaluation(
            food_pickups=world.food_pickups,
            food_delivered=world.food_delivered,
            positions=tuple(positions),
        )

    def attach(
        self,
        world: AntWorld,
        *,
        body_id: int = 0,
        seed: int = 0,
        synchronize_navigator: bool = False,
    ) -> None:
        """Bind one body of ``world`` so it can be driven with ``step``.

        Colony evaluation drives every body through the same shared policy,
        matching how ``FixedRuleAnt``/``RandomAnt`` baselines are stepped by
        the ecology harnesses.
        """

        self._navigators[body_id] = self._episode_navigator(
            world,
            body_id=body_id,
            seed=seed,
            synchronize=synchronize_navigator,
        )

    @property
    def navigator(self) -> AntNavigator:
        """The frozen-substrate navigator of body 0.

        ``AntSession`` / ``FixedRuleAnt`` / ``RandomAnt`` all expose
        ``navigator`` as a plain attribute, so duck-typed cross-arm access
        (matched-control parity guards, colony diagnostics) must find the same
        shape here. Multi-body callers use :meth:`navigator_for`.
        """

        return self.navigator_for(0)

    def navigator_for(self, body_id: int) -> AntNavigator:
        """Return the frozen-substrate navigator bound to ``body_id``.

        Read-only accessor so matched-control guards can compare the sensor
        parameters this arm actually constructed against the other arms.
        """

        if body_id not in self._navigators:
            raise RuntimeError(
                f"E2ERLAnt body {body_id} was never attached to a world"
            )
        return self._navigators[body_id]

    def step(self, world: AntWorld, *, body_id: int = 0) -> None:
        """Apply one deterministic (mean-action) command to an attached body."""

        navigator = self.navigator_for(body_id)
        torch = self._torch
        with torch.no_grad():
            encoded = self._encode(world, navigator, body_id=body_id)
            mean, _ = self._forward(encoded)
            turn, step = self._command(mean.numpy(), world)
        moved = world.act(
            turn_command=turn, step_command=step, body_id=body_id
        )
        navigator.update(
            turn_command=turn,
            step_command=step,
            true_heading=moved.eval_true_heading,
        )


__all__ = [
    "SHARED_COMPASS_GAIN",
    "SHARED_COMPASS_NOISE",
    "SHARED_HEADING_NOISE",
    "SHARED_STEP_NOISE",
    "E2EEvaluation",
    "E2ERLAnt",
    "PPOConfig",
]
