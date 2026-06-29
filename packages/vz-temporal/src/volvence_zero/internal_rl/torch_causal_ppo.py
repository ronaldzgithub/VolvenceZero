"""Owner-mainline torch PPO for the causal z-policy (autograd-owner-integration Phase C).

The standalone Phase-2 proof (`torch_internal_rl.py`) trains a separate toy
policy on a toy environment. This module instead performs a **real torch
autograd PPO step on the live policy parameters** (`track_weights` + critic),
consuming the **real `ZTransition` batch** produced by `InternalRLSandbox`
rollouts. It writes the refined float weights back into the same parameter
store / policy critic the runtime and checkpoint path use.

The PPO step is self-consistent: the importance ratio compares the torch policy
evaluated at the post-update parameters against a frozen copy of the same torch
policy at the pre-update parameters (not the heuristic pure policy that
generated the rollout), so the surrogate is a valid PPO objective on the live
parameters using the stored advantages / returns.

Gating (in `CausalZPolicy.optimize`):
- DISABLED: not called (pure heuristic update is the live writer).
- SHADOW: run on a copy, return evidence, do not write back.
- ACTIVE: write refined track weights + critic back to the store/policy.

``torch`` is imported lazily; this module is not in the internal_rl facade.
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
            "Torch causal PPO requires torch. Install the vz-temporal '[torch]' extra."
        ) from exc
    return torch


@dataclass(frozen=True)
class TorchPPOReport:
    backend: str
    transition_count: int
    policy_loss: float
    value_loss: float
    approx_kl: float
    clip_fraction: float
    entropy: float
    parameters_changed: int
    parameter_change_rate: float
    wrote_back: bool
    description: str = ""


def torch_causal_ppo_update(
    *,
    parameter_store: Any,
    value_weights: dict,
    value_bias: dict,
    track: Any,
    transitions: Sequence[Any],
    n_z: int,
    write_back: bool,
    ppo_epochs: int = 4,
    clip_epsilon: float = 0.2,
    learning_rate: float = 0.02,
    entropy_coef: float = 0.005,
    value_coef: float = 0.5,
) -> TorchPPOReport:
    """One real-autograd PPO update over a live ZTransition batch.

    ``value_weights`` / ``value_bias`` are the CausalZPolicy critic dicts (keyed
    by track). On ACTIVE write-back this updates ``parameter_store.track_weights``
    and the critic dicts in place.
    """

    torch = _require_torch()
    usable = [
        t for t in transitions
        if t.observation_signature and t.policy_action
    ]
    if not usable:
        return TorchPPOReport(
            backend="active" if write_back else "shadow", transition_count=0,
            policy_loss=0.0, value_loss=0.0, approx_kl=0.0, clip_fraction=0.0,
            entropy=0.0, parameters_changed=0, parameter_change_rate=0.0,
            wrote_back=False, description="no usable transitions",
        )

    dtype = torch.float64

    def vec(values: Sequence[float], length: int) -> Any:
        data = list(values)[:length] + [0.0] * max(0, length - len(values))
        return torch.tensor(data, dtype=dtype)

    obs = torch.stack([vec(t.observation_signature, n_z) for t in usable])
    actions = torch.stack([vec(t.policy_action, n_z) for t in usable])
    advantages = torch.tensor([float(t.advantage_estimate) for t in usable], dtype=dtype)
    returns = torch.tensor([float(t.return_estimate) for t in usable], dtype=dtype)
    if advantages.abs().sum() == 0:
        advantages = torch.tensor([float(t.reward) for t in usable], dtype=dtype)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Live policy params as torch leaves.
    base_weights = list(parameter_store.track_weights[track])[:n_z]
    base_weights += [0.0] * max(0, n_z - len(base_weights))
    w = torch.tensor(base_weights, dtype=dtype, requires_grad=True)
    log_std = torch.full((n_z,), math.log(0.1), dtype=dtype, requires_grad=True)
    cw_base = list(value_weights.get(track, tuple(0.0 for _ in range(n_z))))[:n_z]
    cw_base += [0.0] * max(0, n_z - len(cw_base))
    cw = torch.tensor(cw_base, dtype=dtype, requires_grad=True)
    cb = torch.tensor([float(value_bias.get(track, 0.0))], dtype=dtype, requires_grad=True)

    def policy_mean(weights: Any) -> Any:
        # elementwise parametric mean in [0,1]; matches the obs/action arity.
        return torch.clamp(weights.unsqueeze(0) * obs, 0.0, 1.0)

    def log_prob(mean: Any, ls: Any) -> Any:
        std = torch.exp(ls)
        var = std.pow(2)
        return torch.sum(
            -0.5 * ((actions - mean.expand_as(actions)).pow(2) / (var + 1e-8))
            - ls - 0.5 * math.log(2.0 * math.pi),
            dim=1,
        )

    with torch.no_grad():
        old_mean = policy_mean(w)
        old_log_prob = log_prob(old_mean, log_std)

    opt = torch.optim.Adam([w, log_std, cw, cb], lr=learning_rate)
    before = [w.detach().clone(), log_std.detach().clone(), cw.detach().clone(), cb.detach().clone()]
    last_policy_loss = last_value_loss = last_kl = last_clip = last_entropy = 0.0
    for _ in range(ppo_epochs):
        mean = policy_mean(w)
        new_lp = log_prob(mean, log_std)
        ratio = torch.exp(new_lp - old_log_prob)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.mean(torch.min(unclipped, clipped))
        values = torch.matmul(obs, cw) + cb
        value_loss = torch.mean((returns - values).pow(2))
        entropy = torch.mean(torch.sum(log_std + 0.5 * math.log(2.0 * math.pi * math.e), dim=0))
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            last_policy_loss = float(policy_loss)
            last_value_loss = float(value_loss)
            last_kl = float(torch.mean(old_log_prob - new_lp))
            last_clip = float(((ratio - 1.0).abs() > clip_epsilon).to(dtype).mean())
            last_entropy = float(entropy)

    after = [w, log_std, cw, cb]
    changed = 0
    total = 0
    for b, a in zip(before, after):
        diff = (a.detach() - b).abs()
        changed += int((diff > 1e-12).sum())
        total += int(diff.numel())

    if write_back:
        # Renormalize track weights to the non-negative simplex like the pure
        # path (sum to 1) so downstream consumers see a coherent mixture.
        w_pos = [max(0.0, float(v)) for v in w.detach().tolist()]
        s = sum(w_pos)
        if s > 1e-9:
            normalized = tuple(v / s for v in w_pos)
        else:
            normalized = tuple(1.0 / n_z for _ in range(n_z))
        parameter_store.track_weights[track] = normalized
        value_weights[track] = tuple(float(v) for v in cw.detach().tolist())
        value_bias[track] = float(cb.detach()[0])
        parameter_store.align_temporal_from_tracks()

    return TorchPPOReport(
        backend="active" if write_back else "shadow",
        transition_count=len(usable),
        policy_loss=last_policy_loss,
        value_loss=last_value_loss,
        approx_kl=last_kl,
        clip_fraction=last_clip,
        entropy=last_entropy,
        parameters_changed=changed,
        parameter_change_rate=changed / max(total, 1),
        wrote_back=write_back,
        description=(
            f"torch causal PPO: transitions={len(usable)} "
            f"policy_loss={last_policy_loss:.4f} value_loss={last_value_loss:.4f} "
            f"changed={changed} wrote_back={write_back}"
        ),
    )
