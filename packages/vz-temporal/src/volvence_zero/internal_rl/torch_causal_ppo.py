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
from dataclasses import dataclass, replace
from typing import Any, Sequence

from volvence_zero.temporal.causal_action_projection import (
    normalize_causal_action_head_contrast_pairs,
)


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


def _transition_source(transition: Any) -> str:
    """Read the source while preserving legacy synthetic fixture compatibility."""

    try:
        return str(transition.transition_source)
    except AttributeError:
        # Pre-runtime-replay callers supplied the original synthetic transition
        # shape, which had no explicit source field.
        return "synthetic"


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
    runtime_track_modulation_strength: float = 0.0,
    causal_action_head_enabled: bool = False,
    causal_action_head_strength: float = 0.0,
    causal_action_head_effective_dims: tuple[int, ...] | None = None,
    causal_action_head_contrast_pairs: (
        tuple[tuple[int, int], ...] | None
    ) = None,
    causal_action_head_exclusive_steering: bool = False,
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
    transition_sources = {_transition_source(t) for t in usable}
    if len(transition_sources) != 1:
        raise ValueError(
            "torch Internal-RL cannot mix transition sources, got "
            f"{tuple(sorted(transition_sources))}"
        )
    transition_source = next(iter(transition_sources))
    if transition_source not in {"synthetic", "runtime-replay"}:
        raise ValueError(
            f"unsupported torch Internal-RL transition source {transition_source!r}"
        )
    runtime_replay = transition_source == "runtime-replay"
    if not 0.0 <= causal_action_head_strength <= 1.0:
        raise ValueError(
            "causal_action_head_strength must be within [0, 1], "
            f"got {causal_action_head_strength!r}"
        )
    effective_dims = (
        tuple(range(n_z))
        if causal_action_head_effective_dims is None
        else causal_action_head_effective_dims
    )
    if (
        not effective_dims
        or len(set(effective_dims)) != len(effective_dims)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < n_z
            for index in effective_dims
        )
    ):
        raise ValueError(
            "causal_action_head_effective_dims must be unique integer z "
            f"indices within [0, {n_z}), got {effective_dims!r}"
        )
    contrast_pairs = normalize_causal_action_head_contrast_pairs(
        causal_action_head_contrast_pairs,
        n_z=n_z,
        effective_dims=effective_dims,
    )
    if causal_action_head_exclusive_steering and not contrast_pairs:
        raise ValueError(
            "exclusive steering requires non-empty contrast_pairs"
        )

    dtype = torch.float64
    effective_dim_mask = torch.tensor(
        tuple(1.0 if index in effective_dims else 0.0 for index in range(n_z)),
        dtype=dtype,
    )

    def vec(values: Sequence[float], length: int) -> Any:
        data = list(values)[:length] + [0.0] * max(0, length - len(values))
        return torch.tensor(data, dtype=dtype)

    obs = torch.stack([vec(t.observation_signature, n_z) for t in usable])
    actions = torch.stack([vec(t.policy_action, n_z) for t in usable])
    hidden = (
        torch.stack([vec(t.hidden_state, n_z) for t in usable])
        if (
            runtime_track_modulation_strength > 0.0
            and not runtime_replay
        )
        or (causal_action_head_enabled and not runtime_replay)
        else None
    )
    if causal_action_head_enabled and runtime_replay:
        invalid_action_states = tuple(
            len(t.runtime_action_head_state)
            for t in usable
            if len(t.runtime_action_head_state) != n_z
        )
        if invalid_action_states:
            raise ValueError(
                "torch causal action head runtime state dimension mismatch: "
                f"expected={n_z}, actual={invalid_action_states}"
            )
        action_head_state = torch.stack(
            [
                vec(t.runtime_action_head_state, n_z)
                for t in usable
            ]
        )
    else:
        action_head_state = hidden
    runtime_base_mean = (
        torch.stack([vec(t.runtime_base_mean, n_z) for t in usable])
        if runtime_replay
        else None
    )
    runtime_base_std = (
        torch.stack([vec(t.runtime_base_std, n_z) for t in usable])
        if runtime_replay
        else None
    )
    runtime_previous_code = (
        torch.stack([vec(t.runtime_previous_code, n_z) for t in usable])
        if runtime_replay
        else None
    )
    runtime_beta_t = (
        torch.stack(
            [
                vec(
                    t.runtime_beta_t
                    if isinstance(t.runtime_beta_t, tuple)
                    else tuple(float(t.runtime_beta_t) for _ in range(n_z)),
                    n_z,
                )
                for t in usable
            ]
        )
        if runtime_replay
        else None
    )
    runtime_posterior_sample_scale = (
        torch.tensor(
            [
                [float(t.runtime_posterior_sample_scale)]
                for t in usable
            ],
            dtype=dtype,
        )
        if runtime_replay
        else None
    )
    if (
        runtime_posterior_sample_scale is not None
        and (
            torch.any(runtime_posterior_sample_scale <= 0.0)
            or torch.any(runtime_posterior_sample_scale > 1.0)
        )
    ):
        raise ValueError(
            "torch runtime posterior sample scale must be within (0, 1]"
        )
    runtime_other_track_sum = (
        torch.stack([vec(t.runtime_other_track_sum, n_z) for t in usable])
        if runtime_replay
        else None
    )
    advantages = torch.tensor([float(t.advantage_estimate) for t in usable], dtype=dtype)
    returns = torch.tensor([float(t.return_estimate) for t in usable], dtype=dtype)
    if advantages.abs().sum() == 0:
        advantages = torch.tensor([float(t.reward) for t in usable], dtype=dtype)
    if not runtime_replay:
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
    head_parameters = (
        parameter_store.causal_action_head_parameters(track=track)
        if causal_action_head_enabled
        else None
    )
    head_input = (
        torch.tensor(
            head_parameters.input_factors,
            dtype=dtype,
            requires_grad=True,
        )
        if head_parameters is not None
        else None
    )
    head_output = (
        torch.tensor(
            head_parameters.output_factors,
            dtype=dtype,
            requires_grad=True,
        )
        if head_parameters is not None
        else None
    )
    head_bias = (
        torch.tensor(
            head_parameters.bias,
            dtype=dtype,
            requires_grad=True,
        )
        if head_parameters is not None
        else None
    )

    def action_head_residual() -> Any:
        if not causal_action_head_enabled:
            return 0.0
        if (
            action_head_state is None
            or head_input is None
            or head_output is None
            or head_bias is None
        ):
            raise RuntimeError(
                "causal action head requires state features and parameters"
            )
        basis = torch.tanh(
            torch.matmul(
                action_head_state,
                head_input.transpose(0, 1),
            )
            / math.sqrt(max(n_z, 1))
        )
        residual = causal_action_head_strength * torch.tanh(
            torch.matmul(basis, head_output.transpose(0, 1))
            + head_bias.unsqueeze(0)
        ) * effective_dim_mask.unsqueeze(0)
        if not contrast_pairs:
            return residual
        columns = [residual[:, index] for index in range(n_z)]
        for left, right in contrast_pairs:
            contrast = 0.5 * (columns[left] - columns[right])
            columns[left] = contrast
            columns[right] = -contrast
        return torch.stack(columns, dim=1)

    def base_off_contrast(candidate: Any) -> Any:
        # In-graph mirror of the serving-side exclusive-steering projection:
        # the deterministic base mean keeps only each pair's common mode, so
        # autograd routes all contrast-axis credit to the head parameters.
        if not causal_action_head_exclusive_steering:
            return candidate
        columns = [candidate[:, index] for index in range(n_z)]
        for left, right in contrast_pairs:
            common = 0.5 * (columns[left] + columns[right])
            columns[left] = common
            columns[right] = common
        return torch.stack(columns, dim=1)

    def policy_mean(weights: Any) -> Any:
        if runtime_replay:
            if (
                runtime_base_mean is None
                or runtime_previous_code is None
                or runtime_beta_t is None
                or runtime_other_track_sum is None
            ):
                raise RuntimeError(
                    "runtime replay requires captured posterior/beta/track context"
                )
            aggregate_weights = (
                weights.unsqueeze(0) + runtime_other_track_sum
            ) / 3.0
            gain = 1.0 + runtime_track_modulation_strength * (
                aggregate_weights * n_z - 1.0
            )
            gain = torch.clamp(gain, 0.5, 1.5)
            modulated_mean = base_off_contrast(
                torch.clamp(
                    runtime_base_mean * gain,
                    -1.0,
                    1.0,
                )
            )
            if causal_action_head_enabled:
                modulated_mean = torch.clamp(
                    modulated_mean + action_head_residual(),
                    -1.0,
                    1.0,
                )
            return torch.clamp(
                runtime_beta_t * modulated_mean
                + (1.0 - runtime_beta_t) * runtime_previous_code,
                -1.0,
                1.0,
            )
        if runtime_track_modulation_strength <= 0.0:
            # Byte-compatible historical rollback lane.
            candidate = base_off_contrast(weights.unsqueeze(0) * obs)
            if causal_action_head_enabled:
                candidate = candidate + action_head_residual()
            return torch.clamp(candidate, 0.0, 1.0)

        # Match CausalZPolicy._policy_mean and the live ndim forward:
        # construct an unmodulated causal candidate, then apply the exact same
        # aggregate track gain. Only ``track`` is trainable in this PPO step;
        # the other two owner-local track vectors are frozen context.
        if hidden is None:
            raise RuntimeError(
                "runtime track modulation requires transition hidden_state"
            )
        base_candidate = torch.clamp(
            hidden * 0.50 + obs * 0.30 + actions * 0.20,
            0.0,
            1.0,
        )
        other_tracks = [
            other_track
            for other_track in parameter_store.track_weights
            if other_track != track
        ]
        if len(other_tracks) != 2:
            raise RuntimeError(
                "runtime track modulation requires exactly world/self/shared "
                f"tracks, got {tuple(parameter_store.track_weights)}"
            )
        other_sum = (
            vec(parameter_store.track_weights[other_tracks[0]], n_z)
            + vec(parameter_store.track_weights[other_tracks[1]], n_z)
        )
        mean_weights = (weights + other_sum) / 3.0
        gain = 1.0 + runtime_track_modulation_strength * (
            mean_weights * n_z - 1.0
        )
        gain = torch.clamp(gain, 0.5, 1.5)
        candidate = base_off_contrast(base_candidate * gain.unsqueeze(0))
        if causal_action_head_enabled:
            candidate = candidate + action_head_residual()
        return torch.clamp(candidate, 0.0, 1.0)

    def policy_std(weights: Any, ls: Any) -> Any:
        if runtime_replay:
            if (
                runtime_base_std is None
                or runtime_beta_t is None
                or runtime_other_track_sum is None
                or runtime_posterior_sample_scale is None
            ):
                raise RuntimeError(
                    "runtime replay requires captured posterior standard deviation"
                )
            aggregate_weights = (
                weights.unsqueeze(0) + runtime_other_track_sum
            ) / 3.0
            gain = torch.clamp(
                1.0
                + runtime_track_modulation_strength
                * (aggregate_weights * n_z - 1.0),
                0.5,
                1.5,
            )
            return torch.clamp(
                torch.abs(
                    runtime_beta_t
                    * runtime_base_std
                    * gain
                    * runtime_posterior_sample_scale
                ),
                0.02,
                0.5,
            )
        return torch.exp(ls).unsqueeze(0).expand_as(actions)

    def log_prob(mean: Any, ls: Any, weights: Any) -> Any:
        std = policy_std(weights, ls)
        if runtime_replay:
            variance = torch.clamp(std.pow(2), min=1e-6)
            return torch.sum(
                -0.5
                * (
                    (actions - mean.expand_as(actions)).pow(2) / variance
                    + torch.log(2.0 * math.pi * variance)
                ),
                dim=1,
            )
        var = std.pow(2)
        return torch.sum(
            -0.5 * ((actions - mean.expand_as(actions)).pow(2) / (var + 1e-8))
            - torch.log(std) - 0.5 * math.log(2.0 * math.pi),
            dim=1,
        )

    with torch.no_grad():
        old_mean = policy_mean(w)
        old_log_prob = (
            torch.tensor(
                [float(t.log_prob) for t in usable],
                dtype=dtype,
            )
            if runtime_replay
            else log_prob(old_mean, log_std, w)
        )

    optimizer_parameters = [w, log_std, cw, cb]
    if (
        head_input is not None
        and head_output is not None
        and head_bias is not None
    ):
        optimizer_parameters.extend(
            [head_input, head_output, head_bias]
        )
    opt = torch.optim.Adam(optimizer_parameters, lr=learning_rate)
    before = [
        parameter.detach().clone()
        for parameter in optimizer_parameters
    ]
    last_policy_loss = last_value_loss = last_kl = last_clip = last_entropy = 0.0
    for _ in range(ppo_epochs):
        mean = policy_mean(w)
        new_lp = log_prob(mean, log_std, w)
        ratio = torch.exp(new_lp - old_log_prob)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.mean(torch.min(unclipped, clipped))
        values = torch.matmul(obs, cw) + cb
        value_loss = torch.mean((returns - values).pow(2))
        current_std = policy_std(w, log_std)
        entropy = torch.mean(
            torch.sum(
                torch.log(current_std)
                + 0.5 * math.log(2.0 * math.pi * math.e),
                dim=1,
            )
        )
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

    after = optimizer_parameters
    changed = 0
    total = 0
    for b, a in zip(before, after, strict=True):
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
        if (
            head_parameters is not None
            and head_input is not None
            and head_output is not None
            and head_bias is not None
        ):
            parameter_store.restore_causal_action_head_parameters(
                replace(
                    head_parameters,
                    input_factors=tuple(
                        tuple(float(value) for value in row)
                        for row in head_input.detach().tolist()
                    ),
                    output_factors=tuple(
                        tuple(float(value) for value in row)
                        for row in head_output.detach().tolist()
                    ),
                    bias=tuple(
                        float(value)
                        for value in head_bias.detach().tolist()
                    ),
                    update_step=head_parameters.update_step + 1,
                )
            )
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
