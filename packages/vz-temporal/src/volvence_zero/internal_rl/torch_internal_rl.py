"""Real-autograd Internal RL on z_t (NL/ETA migration, Phase 2).

Paper-grade counterpart to the heuristic PPO-shaped loop in
:mod:`volvence_zero.internal_rl.sandbox`. This implements ETA's Internal RL
literally on the latent controller code ``z_t``:

- The action space is ``z_t`` (low-dimensional), not raw tokens.
- The causal policy ``pi(z_t | e_{1:t})`` is a torch ``nn``-style module with a
  state-dependent Gaussian head; the value head is a real critic.
- Optimization is genuine PPO: GAE advantages, a clipped surrogate objective,
  value regression, and an entropy bonus, all via ``torch.autograd`` (not the
  ``math.sin`` pseudo-noise + analytic step of the legacy sandbox).

The environment is a hierarchical, **sparse / delayed-reward** proof episode:
the agent must emit the right abstract action (a ``z_t`` near a phase-specific
target) across phases, and the reward is delivered only at the terminal step.
This is exactly the setting where token-level RL fails and ETA's latent-space
RL is supposed to win, so it is the right place to prove the mechanism.

``torch`` is imported lazily and this module is NOT re-exported from the
``volvence_zero.temporal`` / ``internal_rl`` facade, keeping them torch-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded at call sites
        raise ImportError(
            "Torch Internal RL requires torch. Install the vz-temporal "
            "'[torch]' extra."
        ) from exc
    return torch


@dataclass(frozen=True)
class TorchInternalRLConfig:
    n_z: int = 4
    hidden_dim: int = 16
    phases: int = 3
    steps_per_phase: int = 2
    learning_rate: float = 0.02
    clip_epsilon: float = 0.2
    gamma: float = 0.97
    gae_lambda: float = 0.95
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    ppo_epochs: int = 4
    init_log_std: float = -0.5
    seed: int = 1234


@dataclass(frozen=True)
class TorchRLReport:
    iterations: int
    mean_return_before: float
    mean_return_after: float
    return_improvement: float
    final_policy_loss: float
    final_value_loss: float
    mean_clip_fraction: float
    mean_approx_kl: float
    parameters_changed: int
    parameter_change_rate: float
    optimized: bool
    description: str = ""


class TorchCausalZPolicy:
    """Causal Gaussian policy over z_t plus a value critic (real autograd)."""

    def __init__(self, config: TorchInternalRLConfig) -> None:
        self._torch = _require_torch()
        torch = self._torch
        self.config = config
        self._dtype = torch.float64
        # Causal observation: [last_z (n_z), phase_scalar, bias].
        self._obs_dim = config.n_z + 2
        g = torch.Generator(device="cpu")
        g.manual_seed(config.seed)

        def lin(out_dim: int, in_dim: int, *, off: int) -> list[Any]:
            gg = torch.Generator(device="cpu")
            gg.manual_seed(config.seed + off)
            scale = 1.0 / math.sqrt(max(in_dim, 1))
            w = torch.empty(out_dim, in_dim, dtype=self._dtype)
            w.normal_(0.0, scale, generator=gg)
            w.requires_grad_(True)
            b = torch.zeros(out_dim, dtype=self._dtype, requires_grad=True)
            return [w, b]

        h = config.hidden_dim
        self._pi1 = lin(h, self._obs_dim, off=1)
        self._pi_mean = lin(config.n_z, h, off=2)
        self._log_std = torch.full(
            (config.n_z,), float(config.init_log_std), dtype=self._dtype, requires_grad=True
        )
        self._v1 = lin(h, self._obs_dim, off=3)
        self._v2 = lin(1, h, off=4)
        self._gen = g

    def parameters(self) -> list[Any]:
        return [
            *self._pi1, *self._pi_mean, self._log_std, *self._v1, *self._v2,
        ]

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def _mlp(self, x: Any, layer1: list[Any], layer2: list[Any]) -> Any:
        torch = self._torch
        h = torch.tanh(torch.matmul(layer1[0], x) + layer1[1])
        return torch.matmul(layer2[0], h) + layer2[1]

    def policy_mean(self, obs: Any) -> Any:
        return self._mlp(obs, self._pi1, self._pi_mean)

    def value(self, obs: Any) -> Any:
        return self._mlp(obs, self._v1, self._v2)[0]

    def std(self) -> Any:
        return self._torch.exp(self._log_std)

    def act(self, obs: Any, *, deterministic: bool = False) -> tuple[Any, Any, Any]:
        """Sample z_t; return (action, log_prob, value)."""

        torch = self._torch
        mean = self.policy_mean(obs)
        std = self.std()
        if deterministic:
            action = mean
        else:
            eps = torch.randn(self.config.n_z, dtype=self._dtype, generator=self._gen)
            action = mean + std * eps
        log_prob = self._log_prob(mean, std, action)
        value = self.value(obs)
        return action.detach(), log_prob.detach(), value.detach()

    def _log_prob(self, mean: Any, std: Any, action: Any) -> Any:
        torch = self._torch
        var = std.pow(2)
        return torch.sum(
            -0.5 * ((action - mean).pow(2) / (var + 1e-8))
            - torch.log(std + 1e-8)
            - 0.5 * math.log(2.0 * math.pi)
        )

    def evaluate(self, obs: Any, action: Any) -> tuple[Any, Any, Any]:
        """Recompute (log_prob, entropy, value) for a stored (obs, action)."""

        torch = self._torch
        mean = self.policy_mean(obs)
        std = self.std()
        log_prob = self._log_prob(mean, std, action)
        entropy = torch.sum(0.5 * math.log(2.0 * math.pi * math.e) + torch.log(std + 1e-8))
        value = self.value(obs)
        return log_prob, entropy, value


class TorchInternalRLEnvironment:
    """Hierarchical sparse-reward episode over z_t (delayed terminal reward)."""

    def __init__(self, config: TorchInternalRLConfig) -> None:
        self._torch = _require_torch()
        self.config = config
        torch = self._torch
        # Fixed per-phase target abstract actions (the "subgoals"): the policy
        # must learn to emit z_t near the right target for the active phase.
        g = torch.Generator(device="cpu")
        g.manual_seed(config.seed + 777)
        self._phase_targets = [
            torch.empty(config.n_z, dtype=torch.float64).uniform_(-1.0, 1.0, generator=g)
            for _ in range(config.phases)
        ]

    @property
    def horizon(self) -> int:
        return self.config.phases * self.config.steps_per_phase

    def phase_of(self, step: int) -> int:
        return min(step // self.config.steps_per_phase, self.config.phases - 1)

    def observation(self, *, last_z: Any, step: int) -> Any:
        torch = self._torch
        phase_scalar = torch.tensor(
            [self.phase_of(step) / max(self.config.phases - 1, 1)], dtype=torch.float64
        )
        bias = torch.ones(1, dtype=torch.float64)
        return torch.cat([last_z, phase_scalar, bias])

    def step_reward(self, *, action: Any, step: int) -> float:
        """Per-step closeness to the active subgoal target (kept for delayed sum)."""

        torch = self._torch
        target = self._phase_targets[self.phase_of(step)]
        dist = float(torch.sqrt(torch.sum((action - target).pow(2))))
        # closeness in [0, 1]; 1.0 == exact match
        return max(0.0, 1.0 - dist / (2.0 * math.sqrt(self.config.n_z)))


@dataclass(frozen=True)
class _Transition:
    obs: Any
    action: Any
    log_prob: float
    value: float
    reward: float


def _collect_rollout(
    policy: TorchCausalZPolicy,
    env: TorchInternalRLEnvironment,
    *,
    deterministic: bool = False,
) -> tuple[list[_Transition], float]:
    torch = policy._torch
    last_z = torch.zeros(policy.config.n_z, dtype=torch.float64)
    transitions: list[_Transition] = []
    closeness_sum = 0.0
    horizon = env.horizon
    with torch.no_grad():
        for step in range(horizon):
            obs = env.observation(last_z=last_z, step=step)
            action, log_prob, value = policy.act(obs, deterministic=deterministic)
            closeness = env.step_reward(action=action, step=step)
            closeness_sum += closeness
            # Sparse / delayed: reward is delivered only at the terminal step.
            reward = 0.0
            transitions.append(
                _Transition(
                    obs=obs, action=action,
                    log_prob=float(log_prob), value=float(value), reward=reward,
                )
            )
            last_z = action
    # Terminal delayed reward == total closeness accumulated across the episode.
    if transitions:
        term = transitions[-1]
        transitions[-1] = _Transition(
            obs=term.obs, action=term.action, log_prob=term.log_prob,
            value=term.value, reward=closeness_sum,
        )
    return transitions, closeness_sum


def _compute_gae(
    transitions: list[_Transition], *, gamma: float, lam: float
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(transitions)
    last_adv = 0.0
    for t in reversed(range(len(transitions))):
        next_value = transitions[t + 1].value if t + 1 < len(transitions) else 0.0
        delta = transitions[t].reward + gamma * next_value - transitions[t].value
        last_adv = delta + gamma * lam * last_adv
        advantages[t] = last_adv
    returns = [adv + tr.value for adv, tr in zip(advantages, transitions)]
    return advantages, returns


class TorchInternalRLTrainer:
    """PPO trainer for the causal z-policy (real autograd, offline)."""

    def __init__(self, config: TorchInternalRLConfig | None = None) -> None:
        self._torch = _require_torch()
        self.config = config or TorchInternalRLConfig()
        self.policy = TorchCausalZPolicy(self.config)
        self.env = TorchInternalRLEnvironment(self.config)
        self._opt = self._torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )

    def _mean_return(self, *, rollouts: int, deterministic: bool) -> float:
        total = 0.0
        for _ in range(rollouts):
            _, closeness = _collect_rollout(self.policy, self.env, deterministic=deterministic)
            total += closeness
        return total / max(rollouts, 1)

    def train(
        self, *, iterations: int = 30, rollouts_per_iter: int = 8, optimize: bool = True
    ) -> TorchRLReport:
        torch = self._torch
        before = self._mean_return(rollouts=16, deterministic=True)
        params_before = [p.detach().clone() for p in self.policy.parameters()]

        last_policy_loss = 0.0
        last_value_loss = 0.0
        clip_fractions: list[float] = []
        approx_kls: list[float] = []

        for _ in range(max(iterations, 1)):
            batch: list[_Transition] = []
            adv: list[float] = []
            ret: list[float] = []
            for _r in range(rollouts_per_iter):
                transitions, _ = _collect_rollout(self.policy, self.env)
                a, rr = _compute_gae(
                    transitions, gamma=self.config.gamma, lam=self.config.gae_lambda
                )
                batch.extend(transitions)
                adv.extend(a)
                ret.extend(rr)
            if not optimize or not batch:
                continue
            adv_t = torch.tensor(adv, dtype=torch.float64)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            ret_t = torch.tensor(ret, dtype=torch.float64)
            old_log_probs = torch.tensor([tr.log_prob for tr in batch], dtype=torch.float64)

            for _epoch in range(self.config.ppo_epochs):
                new_log_probs = []
                entropies = []
                values = []
                for tr in batch:
                    lp, ent, val = self.policy.evaluate(tr.obs, tr.action)
                    new_log_probs.append(lp)
                    entropies.append(ent)
                    values.append(val)
                new_lp = torch.stack(new_log_probs)
                ent_t = torch.stack(entropies)
                val_t = torch.stack(values)

                ratio = torch.exp(new_lp - old_log_probs)
                unclipped = ratio * adv_t
                clipped = torch.clamp(
                    ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon
                ) * adv_t
                policy_loss = -torch.mean(torch.min(unclipped, clipped))
                value_loss = torch.mean((ret_t - val_t).pow(2))
                entropy_bonus = torch.mean(ent_t)
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_bonus
                )
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()

                with torch.no_grad():
                    clip_fractions.append(
                        float(((ratio - 1.0).abs() > self.config.clip_epsilon).to(torch.float64).mean())
                    )
                    approx_kls.append(float(torch.mean(old_log_probs - new_lp)))
                last_policy_loss = float(policy_loss.detach())
                last_value_loss = float(value_loss.detach())

        after = self._mean_return(rollouts=16, deterministic=True)
        params_after = self.policy.parameters()
        changed = 0
        total = 0
        for b, a in zip(params_before, params_after):
            diff = (a.detach() - b).abs()
            changed += int((diff > 1e-12).sum())
            total += int(diff.numel())
        return TorchRLReport(
            iterations=iterations,
            mean_return_before=before,
            mean_return_after=after,
            return_improvement=after - before,
            final_policy_loss=last_policy_loss,
            final_value_loss=last_value_loss,
            mean_clip_fraction=sum(clip_fractions) / max(len(clip_fractions), 1),
            mean_approx_kl=sum(approx_kls) / max(len(approx_kls), 1),
            parameters_changed=changed,
            parameter_change_rate=changed / max(total, 1),
            optimized=optimize,
            description=(
                f"torch internal RL: return {before:.3f} -> {after:.3f} "
                f"(optimize={optimize})"
            ),
        )


@dataclass(frozen=True)
class InternalRLProofReport:
    full: TorchRLReport
    no_optimize: TorchRLReport
    full_improves: bool
    control_does_not_improve: bool
    full_beats_control: bool
    description: str = ""


def run_internal_rl_proof(
    *, config: TorchInternalRLConfig | None = None, iterations: int = 40
) -> InternalRLProofReport:
    """Matched control: real PPO (optimize) vs no-optimize on the same task.

    Phase 2 exit evidence. ``full`` runs PPO; ``no_optimize`` collects identical
    rollouts but never updates the policy. The latent-space RL claim holds only
    if the optimized policy improves the sparse delayed return while the
    no-optimize control does not.
    """

    cfg = config or TorchInternalRLConfig()
    full = TorchInternalRLTrainer(cfg).train(iterations=iterations, optimize=True)
    control = TorchInternalRLTrainer(cfg).train(iterations=iterations, optimize=False)
    full_improves = full.return_improvement > 1e-3 and full.parameters_changed > 0
    control_flat = abs(control.return_improvement) <= 1e-6 and control.parameters_changed == 0
    return InternalRLProofReport(
        full=full,
        no_optimize=control,
        full_improves=full_improves,
        control_does_not_improve=control_flat,
        full_beats_control=full.mean_return_after > control.mean_return_after + 1e-3,
        description=(
            f"internal RL proof: full improves={full_improves}, "
            f"control flat={control_flat}, "
            f"full beats control={full.mean_return_after > control.mean_return_after}"
        ),
    )
