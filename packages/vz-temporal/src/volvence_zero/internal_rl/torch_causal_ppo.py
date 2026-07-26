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
from volvence_zero.temporal.interface import (
    LATENT_CODE_BOUNDS,
    causal_action_head_update_scales,
    project_causal_action_head_update,
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


# Historical signed bound for the runtime-replay reconstruction. It is the
# exact rollback baseline, NOT a latent-code contract: the live ndim forward
# bounds ``code`` with the temporal owner's ``LATENT_CODE_BOUNDS``. Reward and
# advantage clamps are a different convention and stay signed in both lanes.
_HISTORICAL_SIGNED_LATENT_BOUNDS: tuple[float, float] = (-1.0, 1.0)


def resolve_latent_code_bounds(*, latent_unit_clamp: bool) -> tuple[float, float]:
    """Select the replay reconstruction bound from the declared contract.

    Mirrors ``InternalRLSandbox``'s ``latent_unit_clamp`` seam one-for-one:
    ``True`` reconstructs on the live metacontroller's unit latent range,
    ``False`` keeps the historical signed bound and is the exact rollback.
    """

    return (
        LATENT_CODE_BOUNDS
        if latent_unit_clamp
        else _HISTORICAL_SIGNED_LATENT_BOUNDS
    )


def torch_runtime_replay_latent_mean(
    torch: Any,
    *,
    base_mean: Any,
    gain: Any,
    previous_code: Any,
    beta_t: Any,
    head_residual: Any | None,
    project_base: Any,
    bounds: tuple[float, float],
) -> Any:
    """Rebuild the replayed runtime policy mean under the declared bound.

    On a saturated contrast axis the signed rollback bound lets the
    reconstructed mean leave the plant's range, so ``(action - mean)`` is
    signed against the action actually taken and the head gradient points the
    wrong way. Under ``LATENT_CODE_BOUNDS`` this lane reconstructs exactly what
    the live forward could have emitted. The clamp positions match the pure
    lane's ``runtime_replay_policy_distribution`` one-for-one.
    """

    lower, upper = bounds
    modulated_mean = project_base(
        torch.clamp(base_mean * gain, lower, upper)
    )
    if head_residual is not None:
        modulated_mean = torch.clamp(
            modulated_mean + head_residual,
            lower,
            upper,
        )
    return torch.clamp(
        beta_t * modulated_mean + (1.0 - beta_t) * previous_code,
        lower,
        upper,
    )


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
    causal_action_head_mirror_equivariance: bool = False,
    latent_unit_clamp: bool = False,
) -> TorchPPOReport:
    """One real-autograd PPO update over a live ZTransition batch.

    ``value_weights`` / ``value_bias`` are the CausalZPolicy critic dicts (keyed
    by track). On ACTIVE write-back this updates ``parameter_store.track_weights``
    and the critic dicts in place.

    ``latent_unit_clamp`` is the same contract the pure lane declares, and it
    is scoped to exactly one thing: the bound used to reconstruct the
    **runtime-replay** policy mean. ``False`` keeps the historical signed
    bound (exact rollback); both lanes must be given the same value or they
    reconstruct different means for the same batch, which is what
    ``assert_runtime_replay_latent_bounds_agree`` polices.

    It does **not** reach the synthetic branches: both of those have always
    reconstructed on the temporal owner's ``LATENT_CODE_BOUNDS`` regardless of
    this flag. That is deliberate, and it is why the cross-lane guard is
    scoped to replay -- see the comment in ``policy_mean`` and
    docs/specs/temporal-abstraction.md.
    """

    torch = _require_torch()
    latent_code_bounds = resolve_latent_code_bounds(
        latent_unit_clamp=latent_unit_clamp
    )
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
    if (
        causal_action_head_mirror_equivariance
        and not causal_action_head_exclusive_steering
    ):
        raise ValueError(
            "causal action head mirror equivariance requires exclusive steering"
        )
    if causal_action_head_mirror_equivariance and not runtime_replay:
        raise ValueError(
            "causal action head mirror equivariance requires runtime-replay "
            "transitions with owner-published mirror states"
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
        if causal_action_head_mirror_equivariance:
            invalid_mirror_states = tuple(
                len(t.runtime_action_head_mirror_state)
                for t in usable
                if len(t.runtime_action_head_mirror_state) != n_z
            )
            if invalid_mirror_states:
                raise ValueError(
                    "torch causal action head runtime mirror-state dimension "
                    f"mismatch: expected={n_z}, actual={invalid_mirror_states}"
                )
            action_head_mirror_state = torch.stack(
                [
                    vec(t.runtime_action_head_mirror_state, n_z)
                    for t in usable
                ]
            )
        else:
            action_head_mirror_state = None
    else:
        action_head_state = hidden
        action_head_mirror_state = None
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

    def raw_action_head_residual(state: Any) -> Any:
        if (
            state is None
            or head_input is None
            or head_output is None
            or head_bias is None
        ):
            raise RuntimeError(
                "causal action head requires state features and parameters"
            )
        basis = torch.tanh(
            torch.matmul(
                state,
                head_input.transpose(0, 1),
            )
            / math.sqrt(max(n_z, 1))
        )
        residual = causal_action_head_strength * torch.tanh(
            torch.matmul(basis, head_output.transpose(0, 1))
            + head_bias.unsqueeze(0)
        ) * effective_dim_mask.unsqueeze(0)
        return residual

    def action_head_residual() -> Any:
        if not causal_action_head_enabled:
            return 0.0
        residual = raw_action_head_residual(action_head_state)
        if causal_action_head_mirror_equivariance:
            mirrored_residual = raw_action_head_residual(
                action_head_mirror_state
            )
            columns = [
                0.5 * (
                    residual[:, index] + mirrored_residual[:, index]
                )
                for index in range(n_z)
            ]
            for left, right in contrast_pairs:
                direct_contrast = 0.5 * (
                    residual[:, left] - residual[:, right]
                )
                mirrored_contrast = 0.5 * (
                    mirrored_residual[:, left]
                    - mirrored_residual[:, right]
                )
                equivariant = 0.5 * (
                    direct_contrast - mirrored_contrast
                )
                columns[left] = equivariant
                columns[right] = -equivariant
            return torch.stack(columns, dim=1)
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
            return torch_runtime_replay_latent_mean(
                torch,
                base_mean=runtime_base_mean,
                gain=gain,
                previous_code=runtime_previous_code,
                beta_t=runtime_beta_t,
                head_residual=(
                    action_head_residual()
                    if causal_action_head_enabled
                    else None
                ),
                project_base=base_off_contrast,
                bounds=latent_code_bounds,
            )
        # The two synthetic branches are OUTSIDE the ``latent_unit_clamp``
        # seam by construction: they have always reconstructed on the temporal
        # owner's own latent range, which is what ``latent_unit_clamp=True``
        # asks the replay lane to adopt. Routing the literal through the owner
        # constant keeps the arithmetic byte-identical while removing the
        # second declaration of the latent range this module is forbidden to
        # own. Selecting ``resolve_latent_code_bounds`` here instead would move
        # the historical default OFF the frozen plant's range, not onto it.
        synthetic_lower, synthetic_upper = LATENT_CODE_BOUNDS
        if runtime_track_modulation_strength <= 0.0:
            # Byte-compatible historical rollback lane.
            candidate = base_off_contrast(weights.unsqueeze(0) * obs)
            if causal_action_head_enabled:
                candidate = candidate + action_head_residual()
            return torch.clamp(candidate, synthetic_lower, synthetic_upper)

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
            synthetic_lower,
            synthetic_upper,
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
        return torch.clamp(candidate, synthetic_lower, synthetic_upper)

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

    # The causal action head is deliberately NOT an Adam parameter. Adam
    # normalises each element's step to ~lr, so folding the head into the same
    # optimizer and then clamping to the owner envelope produced a bang-bang
    # maximum-step controller: measured on this repo's batch, the written step
    # equalled the per-step cap exactly (bias 0.010000, output 0.050000) at
    # lr=0.02 and at lr=0.5, and at a 1000x-smaller gradient -- i.e. the update
    # carried only the SIGN of the gradient, and 10 successive updates pinned
    # bias at its absolute ceiling. A pinned bias IS the cross-state fixed
    # steering intercept the envelope exists to prevent
    # (docs/specs/digital-ant-embodiment.md). The head therefore takes the pure
    # owner's PROPORTIONAL step (``causal_action_head_update_scales``) and the
    # envelope is a guard rail on top of it, not the step size.
    optimizer_parameters = [w, log_std, cw, cb]
    head_leaves = (
        [head_input, head_output, head_bias]
        if head_parameters is not None
        else []
    )
    opt = torch.optim.Adam(optimizer_parameters, lr=learning_rate)
    # Resolved only when the head is declared, so a domain that declares none
    # of this contract reaches byte-identical code to the pre-envelope lane.
    head_scales = (
        causal_action_head_update_scales(
            learning_rate=learning_rate,
            batch_size=len(usable),
        )
        if head_parameters is not None
        else None
    )
    # Change accounting still spans the head leaves, exactly as when they were
    # optimizer members, so ``parameter_change_rate`` keeps its old meaning.
    tracked_parameters = optimizer_parameters + head_leaves

    def current_head_parameters() -> Any:
        return replace(
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
                float(value) for value in head_bias.detach().tolist()
            ),
        )

    def project_head_onto_owner_envelope() -> None:
        """Pull the current head leaves back into the frozen envelope.

        ``head_parameters`` is this call's baseline: the write-back bumps
        ``update_step`` by exactly one, so the whole call is ONE owner update
        and its total displacement must satisfy the per-step bounds. Projecting
        after every head step (not only before write-back) also keeps the
        remaining PPO epochs' forward pass inside the envelope. The bounds
        themselves live in ``temporal.interface``; this lane owns none of them.
        """

        if head_parameters is None:
            return
        projected = project_causal_action_head_update(
            baseline=head_parameters,
            candidate=current_head_parameters(),
        )
        with torch.no_grad():
            head_input.copy_(
                torch.tensor(projected.input_factors, dtype=dtype)
            )
            head_output.copy_(
                torch.tensor(projected.output_factors, dtype=dtype)
            )
            head_bias.copy_(torch.tensor(projected.bias, dtype=dtype))

    def step_head_within_owner_envelope() -> None:
        """One proportional, owner-scaled gradient step on the head leaves.

        The scales come from the temporal owner (``learning_rate / batch``, and
        ``* bias_learning_rate_ratio * bias_state_path_scale`` for the bias), so
        a small gradient produces a small step strictly inside the envelope and
        a large gradient still saturates the cap. No threshold or scale literal
        is duplicated here.
        """

        if head_parameters is None:
            return
        for name, leaf in (
            ("input_factors", head_input),
            ("output_factors", head_output),
            ("bias", head_bias),
        ):
            if leaf.grad is None:
                # The head is wired into ``policy_mean`` whenever it is
                # enabled, so a missing gradient means the surrogate silently
                # detached it -- a no-learning head, not a fallback.
                raise RuntimeError(
                    "causal action head parameter received no gradient from "
                    f"the PPO surrogate: {name}"
                )
        with torch.no_grad():
            head_input.add_(
                head_input.grad,
                alpha=-head_scales.factor_learning_rate,
            )
            head_output.add_(
                head_output.grad,
                alpha=-head_scales.factor_learning_rate,
            )
            head_bias.add_(
                head_bias.grad,
                alpha=-head_scales.bias_signal_learning_rate,
            )
        project_head_onto_owner_envelope()

    before = [
        parameter.detach().clone()
        for parameter in tracked_parameters
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
        for leaf in head_leaves:
            # The head leaves are outside ``opt``; without this their ``.grad``
            # would accumulate across PPO epochs.
            leaf.grad = None
        loss.backward()
        opt.step()
        step_head_within_owner_envelope()
        with torch.no_grad():
            last_policy_loss = float(policy_loss)
            last_value_loss = float(value_loss)
            last_kl = float(torch.mean(old_log_prob - new_lp))
            last_clip = float(((ratio - 1.0).abs() > clip_epsilon).to(dtype).mean())
            last_entropy = float(entropy)

    after = tracked_parameters
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
        if head_parameters is not None:
            # The leaves were already projected after every head step;
            # re-project so the written value is bounded even when
            # ``ppo_epochs == 0``, and let the owner re-validate on install so
            # an escape fails loudly instead of persisting silently.
            parameter_store.restore_causal_action_head_parameters(
                replace(
                    project_causal_action_head_update(
                        baseline=head_parameters,
                        candidate=current_head_parameters(),
                    ),
                    update_step=head_parameters.update_step + 1,
                ),
                enforce_envelope=True,
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
