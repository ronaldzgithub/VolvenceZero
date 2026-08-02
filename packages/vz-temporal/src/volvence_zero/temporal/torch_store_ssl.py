"""Owner-mainline torch autograd SSL over the live MetacontrollerParameterStore.

Phase A of the autograd-owner-integration plan. Unlike the standalone Phase-1
proof (`torch_metacontroller.py`, which owns a separate fused module), this
trainer operates **directly on the store's `Ndim*Parameters`**: it reads the
exact float weights the runtime metacontroller consumes, runs a faithful torch
forward of the ndim encoder/switch/decoder, and optimizes rate-distortion with
real backprop. A trace carrying ``ExpertActionTarget`` uses ETA Eq.3 action
distortion; an untargeted trace retains the explicitly reported multi-horizon
next-residual proxy. Both paths use posterior KL and a Bernoulli switch-rate
prior through a hard straight-through gate, then write updated floats back into
the same store.

This is the bridge that makes `MetacontrollerSSLTrainer` capable of genuine
autograd learning of the runtime weights, gated by `WiringLevel`:

- DISABLED: not used (pure heuristic path is the live writer / rollback base).
- SHADOW: train a COPY seeded from the store; compare; do NOT write back.
- ACTIVE: train seeded from the store and write the refined weights back.

`torch` is imported lazily; this module is not re-exported from the temporal
facade. Weights cross the store boundary only as float tuples (R8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from volvence_zero.substrate import TrainingTrace
from volvence_zero.temporal.metacontroller_components import (
    NDIM_POSTERIOR_CURRENT_WEIGHT,
    NDIM_POSTERIOR_HISTORY_WEIGHT,
    NDIM_POSTERIOR_RECURRENT_WEIGHT,
    POSTERIOR_PARAMETERIZATION_LEGACY,
    POSTERIOR_PARAMETERIZATION_SMOOTH,
    POSTERIOR_STD_SMOOTH_FLOOR,
    _POSTERIOR_PARAMETERIZATIONS,
    NdimDecoderParameters,
    NdimEncoderParameters,
    NdimFFNParams,
    NdimGRUParams,
    NdimSwitchParameters,
    _current_observation_signal,
    _fold_residual_to_ndim,
    _project_to_ndim,
)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded at call sites
        raise ImportError(
            "Torch store SSL requires torch. Install the vz-temporal '[torch]' extra."
        ) from exc
    return torch


class SteeredActionNLLScorer(Protocol):
    """Substrate-owned differentiable action likelihood through the frozen model.

    ETA Eq.3 distortion contract: the scorer injects per-step control deltas
    into the (frozen) base model's residual stream and returns the
    differentiable NLL of the expert action per step, so the metacontroller
    is trained *through* the controlled model instead of regressing the
    action vector directly. Implemented by
    ``volvence_zero.substrate.TransformersSteeredActionScorer``; wired by the
    orchestration layer (vz-runtime), never constructed here.
    """

    @property
    def hidden_size(self) -> int: ...

    def action_index(self, action_id: str) -> int: ...

    def action_nll(
        self,
        *,
        source_texts: tuple[str, ...],
        control_deltas: Any,
        action_indices: tuple[int, ...],
    ) -> Any: ...

    def trainable_parameters(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class StoreSSLReport:
    trace_id: str
    prediction_loss: float
    kl_loss: float
    total_loss: float
    trained_steps: int
    switch_sparsity: float
    binary_switch_ratio: float
    grad_norm: float
    parameters_changed: int
    parameter_change_rate: float
    wrote_back: bool
    trajectory_count: int = 1
    optimizer_step: int = 1
    optimizer_state_reused: bool = False
    switch_rate_loss: float = 0.0
    switch_binary_loss: float = 0.0
    switch_group_loss: float = 0.0
    gate_choice_loss: float = 0.0
    keep_prediction_loss: float = 0.0
    switch_prediction_loss: float = 0.0
    mean_switch_probability: float = 0.0
    prediction_horizon: int = 1
    distortion_target: str = "absolute"
    supervision_target: str = "next-residual-summary-absolute-proxy"
    expert_action_supervision: bool = False
    expert_action_boundary_f1: float = 0.0
    boundary_switch_probability: float = 0.0
    continuation_switch_probability: float = 0.0
    boundary_switch_preference: float = 0.0
    continuation_switch_preference: float = 0.0
    target_variance: float = 0.0
    switch_threshold_before: float = 0.55
    switch_threshold_after: float = 0.55
    # CP-05 (GAP-09): the refined weights as an exportable CANDIDATE
    # checkpoint. Under SHADOW (wrote_back=False) the live store is
    # untouched but the candidate is retained so promotion can restore
    # exactly what the SHADOW pass trained instead of retraining blind.
    # Under ACTIVE these equal what was written back.
    candidate_encoder_parameters: NdimEncoderParameters | None = None
    candidate_switch_parameters: NdimSwitchParameters | None = None
    candidate_decoder_parameters: NdimDecoderParameters | None = None
    description: str = ""


@dataclass(frozen=True)
class StoreSSLEvaluationReport:
    """No-grad readout of the SSL objective on held-out traces.

    ``kl_rate`` is the mean per-dimension posterior KL (the rate axis of the
    ETA rate-distortion criterion); ``distortion`` is the mean prediction
    loss under the session's supervision mode (action NLL in steered mode).
    """

    batch_id: str
    trace_count: int
    step_count: int
    distortion: float
    kl_rate: float
    mean_switch_probability: float
    hard_switch_frequency: float
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    boundary_switch_probability: float
    continuation_switch_probability: float
    switch_threshold: float
    supervision_target: str
    description: str = ""


def _rate_calibrated_threshold(
    probabilities: tuple[float, ...],
    *,
    target_rate: float,
    fallback: float,
) -> float:
    """Choose an unlabeled decision threshold matching the aggregate switch rate."""

    if not probabilities:
        return fallback
    ordered = tuple(sorted(max(0.0, min(1.0, value)) for value in probabilities))
    positive_count = max(1, min(len(ordered) - 1, round(target_rate * len(ordered))))
    split = len(ordered) - positive_count
    lower = ordered[split - 1]
    upper = ordered[split]
    return max(0.05, min(0.95, (lower + upper) / 2.0))


def _step_input_vectors(trace: TrainingTrace, n_input: int) -> list[tuple[float, ...]]:
    """Mirror `_summarize_substrate_ndim`: per-step n_input vector from a trace."""

    vectors: list[tuple[float, ...]] = []
    for step in trace.steps:
        raw: list[float] = []
        for act in step.residual_activations:
            raw.extend(act.activation)
        if not raw:
            for feat in step.feature_surface:
                raw.extend(feat.values)
        if not raw:
            raw = [0.0]
        vectors.append(_fold_residual_to_ndim(tuple(raw), n_input))
    return vectors


class _TorchNdimMetacontroller:
    """Torch mirror of the ndim encoder/switch/decoder, seeded from store floats."""

    def __init__(self, *, n_z: int, encoder: NdimEncoderParameters,
                 switch: NdimSwitchParameters, decoder: NdimDecoderParameters) -> None:
        torch = _require_torch()
        self._torch = torch
        self.n_z = n_z
        self.n_input = encoder.n_input
        self._dtype = torch.float64

        def mat(m) -> Any:
            return torch.tensor([list(row) for row in m], dtype=self._dtype, requires_grad=True)

        def vec(v) -> Any:
            return torch.tensor(list(v), dtype=self._dtype, requires_grad=True)

        g = encoder.gru
        self.W_z, self.U_z, self.b_z = mat(g.W_z), mat(g.U_z), vec(g.b_z)
        self.W_r, self.U_r, self.b_r = mat(g.W_r), mat(g.U_r), vec(g.b_r)
        self.W_h, self.U_h, self.b_h = mat(g.W_h), mat(g.U_h), vec(g.b_h)
        self.posterior_proj = mat(encoder.posterior_proj)
        self.current_proj = mat(encoder.current_proj)
        self.posterior_std_proj = mat(encoder.posterior_std_proj)
        self.sw_W1, self.sw_b1 = mat(switch.gate_ffn.W1), vec(switch.gate_ffn.b1)
        self.sw_W2, self.sw_b2 = mat(switch.gate_ffn.W2), vec(switch.gate_ffn.b2)
        self.dec_W1, self.dec_b1 = mat(decoder.decoder_ffn.W1), vec(decoder.decoder_ffn.b1)
        self.dec_W2, self.dec_b2 = mat(decoder.decoder_ffn.W2), vec(decoder.decoder_ffn.b2)

    def parameters(self) -> list[Any]:
        return [
            self.W_z, self.U_z, self.b_z, self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h, self.posterior_proj, self.current_proj,
            self.posterior_std_proj,
            self.sw_W1, self.sw_b1, self.sw_W2, self.sw_b2,
            self.dec_W1, self.dec_b1, self.dec_W2, self.dec_b2,
        ]

    def _gru_step(self, x: Any, h: Any) -> Any:
        torch = self._torch
        z = torch.sigmoid(torch.matmul(self.W_z, x) + torch.matmul(self.U_z, h) + self.b_z)
        r = torch.sigmoid(torch.matmul(self.W_r, x) + torch.matmul(self.U_r, h) + self.b_r)
        h_cand = torch.tanh(torch.matmul(self.W_h, x) + torch.matmul(self.U_h, r * h) + self.b_h)
        return (1.0 - z) * h + z * h_cand

    def _decode(self, latent_code: Any) -> Any:
        torch = self._torch
        dec_hidden = torch.tanh(
            torch.matmul(self.dec_W1, latent_code) + self.dec_b1
        )
        decoder_output = torch.matmul(
            self.dec_W2,
            dec_hidden,
        ) + self.dec_b2
        raw_control = 0.65 * latent_code + 0.35 * decoder_output
        bounded_control = torch.clamp(raw_control, 0.0, 1.0)
        return (
            raw_control
            + (bounded_control - raw_control).detach()
        )

    def rollout(
        self,
        step_inputs: Sequence[tuple[float, ...]],
        *,
        switch_threshold: float,
        generator: Any | None = None,
        gate_mode: str = "hard-st",
        posterior_parameterization: str = POSTERIOR_PARAMETERIZATION_LEGACY,
    ):
        """Run the training forward with one segment gate per step.

        The switch network still emits an ndim proposal so its stored shape
        remains runtime-compatible. Its mean is the scalar segment hazard.

        ``generator``: when provided, ``z_tilde`` is a genuine
        reparameterized sample ``mu + sigma * eps`` with seeded Gaussian
        noise, so ``alpha * KL`` acts as a real information-rate constraint
        (ETA Eq.3). When ``None`` (evaluation), the posterior mean is used —
        the standard deterministic readout the runtime consumes.

        ``gate_mode``:

        - ``"hard-st"`` (legacy / control arm): hard straight-through gate on
          the latent transition.
        - ``"continuous"`` (Eq.3 SSL relaxation, paper §E.1): the scalar
          switch probability itself interpolates ``z_tilde`` with the
          previous code; hard switches are threshold telemetry only and do
          not enter the transition.

        ``posterior_parameterization``:

        - ``"legacy"``: ``mu = clamp(.., 0, 1)``, ``sigma = clamp(|W h|,
          0.05, 0.95)``, and ``z_tilde``/``latent_code`` clamped to [0,1].
          The ``abs`` is non-differentiable at 0 and every clamp zeroes the
          gradient at its boundary, which made the Gate-1 rate axis noisy and
          non-monotonic (bimodal seeds).
        - ``"smooth"``: ``mu`` is unbounded and ``sigma = softplus(W h) +
          1e-4`` is smooth and strictly positive, so the KL/rate responds
          smoothly to ``alpha``. The decoder's straight-through clamp still
          bounds the applied control, so the code stays runtime-compatible.

        The KL formula in the objective is unchanged: it reads ``mu`` and
        ``sigma`` directly, so ``0.5*(mu^2 + sigma^2 - 1 - 2 log sigma)`` keeps
        its meaning under either parameterization.
        """

        if gate_mode not in ("hard-st", "continuous"):
            raise ValueError(
                f"Unsupported gate_mode {gate_mode!r}; expected 'hard-st' or "
                "'continuous'."
            )
        if posterior_parameterization not in _POSTERIOR_PARAMETERIZATIONS:
            raise ValueError(
                "Unsupported posterior_parameterization "
                f"{posterior_parameterization!r}; expected one of "
                f"{_POSTERIOR_PARAMETERIZATIONS}."
            )
        smooth_posterior = (
            posterior_parameterization == POSTERIOR_PARAMETERIZATION_SMOOTH
        )
        torch = self._torch
        h = torch.zeros(self.n_z, dtype=self._dtype)
        prev_code = torch.zeros(self.n_z, dtype=self._dtype)
        hidden_sum = torch.zeros(self.n_z, dtype=self._dtype)
        controls: list[Any] = []
        means: list[Any] = []
        stds: list[Any] = []
        betas: list[Any] = []
        switch_probabilities: list[Any] = []
        hard_switches: list[Any] = []
        keep_controls: list[Any] = []
        switch_controls: list[Any] = []
        count = 0
        for raw in step_inputs:
            x = torch.tensor(list(raw), dtype=self._dtype)
            current_signal = torch.tensor(
                list(_current_observation_signal(raw, self.n_z)),
                dtype=self._dtype,
            )
            h = self._gru_step(x, h)
            count += 1
            hidden_sum = hidden_sum + h
            avg_hidden = hidden_sum / count
            mean_linear = (
                NDIM_POSTERIOR_RECURRENT_WEIGHT
                * torch.matmul(self.posterior_proj, h)
                + NDIM_POSTERIOR_HISTORY_WEIGHT * avg_hidden
                + NDIM_POSTERIOR_CURRENT_WEIGHT
                * torch.matmul(self.current_proj, current_signal)
            )
            std_raw = torch.matmul(self.posterior_std_proj, h)
            if smooth_posterior:
                posterior_mean = mean_linear
                posterior_std = (
                    torch.nn.functional.softplus(std_raw)
                    + POSTERIOR_STD_SMOOTH_FLOOR
                )
            else:
                posterior_mean = torch.clamp(mean_linear, 0.0, 1.0)
                posterior_std = torch.clamp(torch.abs(std_raw), 0.05, 0.95)
            if generator is None:
                z_tilde = posterior_mean
            else:
                sample_noise = torch.randn(
                    self.n_z, generator=generator, dtype=self._dtype
                )
                z_tilde = posterior_mean + posterior_std * sample_noise
                if not smooth_posterior:
                    z_tilde = torch.clamp(z_tilde, 0.0, 1.0)
            # switch gate: the pure ndim code uses ``gate_input = delta + z_tilde``
            # as tuple CONCATENATION (2*n_z dims), matching the n_z*2-column W1.
            gate_input = torch.cat([z_tilde - prev_code, z_tilde])
            sw_hidden = torch.tanh(torch.matmul(self.sw_W1, gate_input) + self.sw_b1)
            raw_gate = torch.matmul(self.sw_W2, sw_hidden) + self.sw_b2
            beta_cont = torch.sigmoid(raw_gate)
            switch_probability = torch.mean(beta_cont)
            hard_switch = (
                torch.ones((), dtype=self._dtype)
                if count == 1
                else (switch_probability >= switch_threshold).to(self._dtype)
            )
            if gate_mode == "continuous":
                effective_gate = (
                    torch.ones((), dtype=self._dtype)
                    if count == 1
                    else switch_probability
                )
                mixed_code = (
                    effective_gate * z_tilde
                    + (1.0 - effective_gate) * prev_code
                )
                latent_code = (
                    mixed_code
                    if smooth_posterior
                    else torch.clamp(mixed_code, 0.0, 1.0)
                )
            else:
                straight_through_gate = (
                    hard_switch.detach()
                    - switch_probability.detach()
                    + switch_probability
                )
                mixed_code = (
                    straight_through_gate * z_tilde
                    + (1.0 - straight_through_gate) * prev_code
                )
                latent_code = (
                    mixed_code
                    if smooth_posterior
                    else torch.clamp(mixed_code, 0.0, 1.0)
                )
            applied_control = self._decode(latent_code)
            controls.append(applied_control)
            keep_controls.append(self._decode(prev_code))
            switch_controls.append(self._decode(z_tilde))
            means.append(posterior_mean)
            stds.append(posterior_std)
            betas.append(beta_cont)
            switch_probabilities.append(switch_probability)
            hard_switches.append(hard_switch)
            prev_code = latent_code
        return {
            "controls": controls,
            "means": means,
            "stds": stds,
            "betas": betas,
            "switch_probabilities": switch_probabilities,
            "hard_switches": hard_switches,
            "keep_controls": keep_controls,
            "switch_controls": switch_controls,
        }

    # --- write-back to the ndim parameter dataclasses (float tuples) ---

    def _m(self, t: Any) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(float(v) for v in row) for row in t.detach().tolist())

    def _v(self, t: Any) -> tuple[float, ...]:
        return tuple(float(v) for v in t.detach().tolist())

    def to_encoder_params(self, n_input: int) -> NdimEncoderParameters:
        return NdimEncoderParameters(
            n_input=n_input,
            gru=NdimGRUParams(
                W_z=self._m(self.W_z), U_z=self._m(self.U_z), b_z=self._v(self.b_z),
                W_r=self._m(self.W_r), U_r=self._m(self.U_r), b_r=self._v(self.b_r),
                W_h=self._m(self.W_h), U_h=self._m(self.U_h), b_h=self._v(self.b_h),
            ),
            posterior_proj=self._m(self.posterior_proj),
            current_proj=self._m(self.current_proj),
            posterior_std_proj=self._m(self.posterior_std_proj),
        )

    def to_switch_params(self) -> NdimSwitchParameters:
        return NdimSwitchParameters(
            gate_ffn=NdimFFNParams(
                W1=self._m(self.sw_W1), b1=self._v(self.sw_b1),
                W2=self._m(self.sw_W2), b2=self._v(self.sw_b2),
            )
        )

    def to_decoder_params(self) -> NdimDecoderParameters:
        return NdimDecoderParameters(
            decoder_ffn=NdimFFNParams(
                W1=self._m(self.dec_W1), b1=self._v(self.dec_b1),
                W2=self._m(self.dec_W2), b2=self._v(self.dec_b2),
            )
        )


class StoreSSLTrainingSession:
    """Persistent optimizer bound to one metacontroller parameter store."""

    def __init__(
        self,
        *,
        n_z: int,
        alpha: float = 0.1,
        learning_rate: float = 0.02,
        switch_prior: float = 0.10,
        # Rate-distortion packet (2026-08): the switch regularizers and the
        # manufactured gate supervision are OFF by default. They were
        # stability patches for the never-switch collapse, but the ETA paper
        # treats that collapse as a diagnostic readout; suppressing the
        # symptom destroys the diagnosis. Non-zero values remain supported as
        # the explicit legacy/control arm.
        switch_rate_weight: float = 0.0,
        switch_binary_weight: float = 0.0,
        switch_group_weight: float = 0.0,
        proposal_prediction_weight: float = 0.50,
        gate_choice_weight: float = 0.0,
        gate_choice_temperature: float = 0.02,
        prediction_horizon: int = 3,
        distortion_target: str = "innovation",
        action_scorer: SteeredActionNLLScorer | None = None,
        reparam_seed: int = 20260801,
        substrate_learning_rate: float = 1e-4,
        posterior_parameterization: str = POSTERIOR_PARAMETERIZATION_LEGACY,
    ) -> None:
        if n_z <= 3:
            raise RuntimeError(
                "StoreSSLTrainingSession requires ndim parameters (n_z > 3)."
            )
        if not 0.0 < switch_prior < 1.0:
            raise ValueError("switch_prior must be strictly between 0 and 1.")
        if any(
            value < 0.0
            for value in (
                alpha,
                learning_rate,
                switch_rate_weight,
                switch_binary_weight,
                switch_group_weight,
                proposal_prediction_weight,
                gate_choice_weight,
            )
        ):
            raise ValueError("SSL loss weights and learning_rate must be non-negative.")
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1.")
        if gate_choice_temperature <= 0.0:
            raise ValueError("gate_choice_temperature must be positive.")
        if distortion_target not in {"absolute", "innovation"}:
            raise ValueError(
                "distortion_target must be 'absolute' or 'innovation'."
            )
        if action_scorer is not None and any(
            value != 0.0
            for value in (
                switch_rate_weight,
                switch_binary_weight,
                switch_group_weight,
                proposal_prediction_weight,
                gate_choice_weight,
            )
        ):
            # Eq.3 has exactly two loss terms: action NLL + alpha * KL. The
            # steered-action mode exists to test emergence; direct switch
            # supervision would contaminate the instrument.
            raise ValueError(
                "Steered-action SSL (ETA Eq.3) forbids switch regularizers "
                "and gate/proposal supervision; set switch_rate_weight, "
                "switch_binary_weight, switch_group_weight, "
                "proposal_prediction_weight and gate_choice_weight to 0."
            )
        self._n_z = n_z
        self._alpha = alpha
        self._learning_rate = learning_rate
        self._switch_prior = switch_prior
        self._switch_rate_weight = switch_rate_weight
        self._switch_binary_weight = switch_binary_weight
        self._switch_group_weight = switch_group_weight
        self._proposal_prediction_weight = proposal_prediction_weight
        self._gate_choice_weight = gate_choice_weight
        self._gate_choice_temperature = gate_choice_temperature
        self._prediction_horizon = prediction_horizon
        self._distortion_target = distortion_target
        self._action_scorer = action_scorer
        self._reparam_seed = reparam_seed
        if posterior_parameterization not in _POSTERIOR_PARAMETERIZATIONS:
            raise ValueError(
                "posterior_parameterization must be one of "
                f"{_POSTERIOR_PARAMETERIZATIONS}, got "
                f"{posterior_parameterization!r}."
            )
        self._posterior_parameterization = posterior_parameterization
        if substrate_learning_rate <= 0.0:
            raise ValueError("substrate_learning_rate must be positive.")
        self._substrate_learning_rate = substrate_learning_rate
        self._generator: Any = None
        self._steer_W: Any = None
        self._steer_b: Any = None
        self._module: _TorchNdimMetacontroller | None = None
        self._optimizer_parameters: tuple[Any, ...] = ()
        self._optimizer: Any = None
        self._store_identity: int | None = None
        self._optimizer_step = 0
        self._last_written_parameters: tuple[
            NdimEncoderParameters,
            NdimSwitchParameters,
            NdimDecoderParameters,
        ] | None = None

    @property
    def optimizer_step(self) -> int:
        return self._optimizer_step

    @property
    def steered_action_mode(self) -> bool:
        return self._action_scorer is not None

    def _ensure_initialized(self, *, store: Any) -> None:
        if (
            store.ndim_encoder_parameters is None
            or store.ndim_switch_parameters is None
            or store.ndim_decoder_parameters is None
        ):
            raise RuntimeError(
                "torch store SSL requires ndim parameters (n_z > 3)."
            )
        if self._store_identity is not None and self._store_identity != id(store):
            raise RuntimeError(
                "StoreSSLTrainingSession is bound to one parameter store and "
                "cannot be reused with another."
            )
        if self._module is not None:
            return
        torch = _require_torch()
        self._module = _TorchNdimMetacontroller(
            n_z=self._n_z,
            encoder=store.ndim_encoder_parameters,
            switch=store.ndim_switch_parameters,
            decoder=store.ndim_decoder_parameters,
        )
        self._generator = torch.Generator()
        self._generator.manual_seed(self._reparam_seed)
        optimizer_parameters = list(self._module.parameters())
        if self._action_scorer is not None:
            hidden_size = int(self._action_scorer.hidden_size)
            # Steering expansion head (part of the controller phi): maps the
            # bounded n_z control to a residual-stream-width delta. Small
            # non-zero init so gradient reaches the controller from step one.
            self._steer_W = (
                0.05
                * torch.randn(
                    (hidden_size, self._n_z),
                    generator=self._generator,
                    dtype=torch.float64,
                )
            ).requires_grad_(True)
            self._steer_b = torch.zeros(
                hidden_size, dtype=torch.float64, requires_grad=True
            )
            optimizer_parameters.append(self._steer_W)
            optimizer_parameters.append(self._steer_b)
        substrate_parameters = (
            list(self._action_scorer.trainable_parameters())
            if self._action_scorer is not None
            else []
        )
        parameter_groups = [
            {"params": optimizer_parameters, "lr": self._learning_rate}
        ]
        if substrate_parameters:
            # Joint (non-frozen) validity-control arm: the substrate's upper
            # blocks co-adapt at a conventional fine-tuning rate, not the
            # controller rate.
            parameter_groups.append(
                {
                    "params": substrate_parameters,
                    "lr": self._substrate_learning_rate,
                }
            )
        self._optimizer = torch.optim.Adam(parameter_groups)
        self._optimizer_parameters = tuple(
            optimizer_parameters + substrate_parameters
        )
        self._store_identity = id(store)

    def _all_parameters(self) -> tuple[Any, ...]:
        if self._module is None:
            raise RuntimeError("Store SSL module is not initialized.")
        return self._optimizer_parameters

    def _assert_active_store_not_mutated(self, *, store: Any) -> None:
        if self._last_written_parameters is None:
            return
        current = (
            store.ndim_encoder_parameters,
            store.ndim_switch_parameters,
            store.ndim_decoder_parameters,
        )
        if current != self._last_written_parameters:
            changed = tuple(
                name
                for name, actual, expected in zip(
                    ("encoder", "switch", "decoder"),
                    current,
                    self._last_written_parameters,
                    strict=True,
                )
                if actual != expected
            )
            raise RuntimeError(
                "ACTIVE torch SSL store parameters changed outside the bound "
                f"persistent optimizer session: changed={changed}."
            )

    def _trace_objective(
        self,
        *,
        trace: TrainingTrace,
        switch_threshold: float,
        use_sampling: bool = False,
    ) -> dict[str, Any]:
        if self._module is None:
            raise RuntimeError("Store SSL module is not initialized.")
        if len(trace.steps) < 2:
            raise ValueError(
                f"Store SSL trace {trace.trace_id!r} must contain at least two steps."
            )
        if self._action_scorer is not None:
            return self._trace_objective_steered(
                trace=trace,
                switch_threshold=switch_threshold,
                use_sampling=use_sampling,
            )
        torch = _require_torch()
        inputs = _step_input_vectors(trace, self._module.n_input)
        residual_targets = tuple(_step_input_vectors(trace, self._n_z))
        action_targets = tuple(
            step.expert_action_target for step in trace.steps
        )
        has_expert_action = any(target is not None for target in action_targets)
        if has_expert_action and not all(
            target is not None for target in action_targets
        ):
            raise ValueError(
                f"Store SSL trace {trace.trace_id!r} mixes expert-targeted "
                "and untargeted steps."
            )
        level_values = tuple(
            value for target in residual_targets for value in target
        )
        level_scale = max(
            math.sqrt(
                sum(value * value for value in level_values)
                / max(len(level_values), 1)
            ),
            1e-4,
        )
        adjacent_deltas = tuple(
            future - current
            for current_target, future_target in zip(
                residual_targets,
                residual_targets[1:],
                strict=False,
            )
            for current, future in zip(
                current_target,
                future_target,
                strict=True,
            )
        )
        innovation_scale = max(
            math.sqrt(
                sum(delta * delta for delta in adjacent_deltas)
                / max(len(adjacent_deltas), 1)
            ),
            1e-4,
        )

        def prediction_target(
            *,
            current: tuple[float, ...],
            future: tuple[float, ...],
        ) -> tuple[float, ...]:
            if self._distortion_target == "absolute":
                return tuple(
                    0.5 + 0.5 * math.tanh(value / level_scale)
                    for value in future
                )
            return tuple(
                0.5
                + 0.5
                * math.tanh((future_value - current_value) / innovation_scale)
                for current_value, future_value in zip(
                    current,
                    future,
                    strict=True,
                )
            )

        if has_expert_action:
            step_targets = tuple(
                (
                    (
                        _project_to_ndim(target.values, self._n_z),
                        1.0,
                    ),
                )
                for target in action_targets
                if target is not None
            )
            rollout_inputs = inputs
            supervision_target = "expert-action-vector"
            effective_prediction_horizon = 1
        else:
            step_targets = tuple(
                tuple(
                    (
                        prediction_target(
                            current=residual_targets[step_index],
                            future=residual_targets[step_index + horizon],
                        ),
                        1.0 / horizon,
                    )
                    for horizon in range(
                        1,
                        min(
                            self._prediction_horizon,
                            len(residual_targets) - step_index - 1,
                        )
                        + 1,
                    )
                )
                for step_index in range(len(residual_targets) - 1)
            )
            rollout_inputs = inputs[:-1]
            supervision_target = (
                f"next-residual-summary-{self._distortion_target}-proxy"
            )
            effective_prediction_horizon = self._prediction_horizon

        out = self._module.rollout(
            rollout_inputs,
            switch_threshold=switch_threshold,
            generator=self._generator if use_sampling else None,
            gate_mode="hard-st",
            posterior_parameterization=self._posterior_parameterization,
        )

        def distortion(control: Any, target: Any) -> Any:
            mse = torch.mean((control - target).pow(2))
            if not has_expert_action:
                return mse
            cosine = torch.sum(control * target) / (
                torch.linalg.vector_norm(control).clamp_min(1e-6)
                * torch.linalg.vector_norm(target).clamp_min(1e-6)
            )
            return mse + 0.5 * (1.0 - cosine)

        prediction_terms: list[Any] = []
        keep_terms: list[Any] = []
        switch_terms: list[Any] = []
        kl_terms: list[Any] = []
        target_values: list[float] = []
        for step_index, control in enumerate(out["controls"]):
            horizon_terms: list[Any] = []
            keep_horizon_terms: list[Any] = []
            switch_horizon_terms: list[Any] = []
            horizon_weights: list[float] = []
            for target_vector, target_weight in step_targets[step_index]:
                target_values.extend(target_vector)
                target = torch.tensor(
                    list(target_vector),
                    dtype=torch.float64,
                )
                horizon_terms.append(distortion(control, target))
                keep_horizon_terms.append(
                    distortion(out["keep_controls"][step_index], target)
                )
                switch_horizon_terms.append(
                    distortion(out["switch_controls"][step_index], target)
                )
                horizon_weights.append(target_weight)
            weight_total = sum(horizon_weights)
            prediction_terms.append(
                sum(
                    term * weight
                    for term, weight in zip(
                        horizon_terms,
                        horizon_weights,
                        strict=True,
                    )
                )
                / weight_total
            )
            keep_terms.append(
                sum(
                    term * weight
                    for term, weight in zip(
                        keep_horizon_terms,
                        horizon_weights,
                        strict=True,
                    )
                )
                / weight_total
            )
            switch_terms.append(
                sum(
                    term * weight
                    for term, weight in zip(
                        switch_horizon_terms,
                        horizon_weights,
                        strict=True,
                    )
                )
                / weight_total
            )
            mean = out["means"][step_index]
            std = out["stds"][step_index]
            kl_terms.append(
                0.5
                * torch.mean(
                    mean.pow(2)
                    + std.pow(2)
                    - 1.0
                    - 2.0 * torch.log(std)
                )
            )

        eligible_probabilities = out["switch_probabilities"][1:]
        eligible_betas = out["betas"][1:]
        if eligible_probabilities:
            epsilon = 1e-8
            prior = torch.tensor(self._switch_prior, dtype=torch.float64)
            binary_terms = []
            group_terms = []
            for probability, beta_vector in zip(
                eligible_probabilities,
                eligible_betas,
                strict=True,
            ):
                probability = torch.clamp(
                    probability,
                    epsilon,
                    1.0 - epsilon,
                )
                binary_terms.append(probability * (1.0 - probability))
                group_terms.append(
                    torch.mean((beta_vector - probability).pow(2))
                )
            mean_probability = torch.clamp(
                torch.stack(eligible_probabilities).mean(),
                epsilon,
                1.0 - epsilon,
            )
            switch_rate_loss = (
                mean_probability * torch.log(mean_probability / prior)
                + (1.0 - mean_probability)
                * torch.log((1.0 - mean_probability) / (1.0 - prior))
            )
            switch_binary_loss = torch.stack(binary_terms).mean()
            switch_group_loss = torch.stack(group_terms).mean()
        else:
            zero = prediction_terms[0] * 0.0
            switch_rate_loss = zero
            switch_binary_loss = zero
            switch_group_loss = zero
        gate_preferences = tuple(
            torch.sigmoid(
                (keep_loss.detach() - switch_loss.detach())
                / self._gate_choice_temperature
            )
            for keep_loss, switch_loss in zip(
                keep_terms[1:],
                switch_terms[1:],
                strict=True,
            )
        )
        gate_choice_terms = tuple(
            -(
                preference
                * torch.log(torch.clamp(probability, 1e-8, 1.0 - 1e-8))
                + (1.0 - preference)
                * torch.log(
                    torch.clamp(1.0 - probability, 1e-8, 1.0 - 1e-8)
                )
            )
            for probability, preference in zip(
                out["switch_probabilities"][1:],
                gate_preferences,
                strict=True,
            )
        )
        gate_choice_loss = (
            torch.stack(gate_choice_terms).mean()
            if gate_choice_terms
            else prediction_terms[0] * 0.0
        )
        return {
            "prediction_loss": torch.stack(prediction_terms).mean(),
            "keep_prediction_loss": torch.stack(keep_terms).mean(),
            "switch_prediction_loss": torch.stack(switch_terms).mean(),
            "kl_loss": torch.stack(kl_terms).mean(),
            "switch_rate_loss": switch_rate_loss,
            "switch_binary_loss": switch_binary_loss,
            "switch_group_loss": switch_group_loss,
            "gate_choice_loss": gate_choice_loss,
            "trained_steps": len(prediction_terms),
            "switch_probabilities": eligible_probabilities,
            "hard_switches": out["hard_switches"][1:],
            "gate_preferences": gate_preferences,
            "supervision_target": supervision_target,
            "expert_action_supervision": has_expert_action,
            "prediction_horizon": effective_prediction_horizon,
            "expert_action_switch_labels": (
                tuple(
                    float(
                        action_targets[index].action_id
                        != action_targets[index - 1].action_id
                    )
                    for index in range(1, len(action_targets))
                )
                if has_expert_action
                else ()
            ),
            "target_variance": (
                sum(
                    (value - sum(target_values) / len(target_values)) ** 2
                    for value in target_values
                )
                / len(target_values)
                if target_values
                else 0.0
            ),
        }

    def _trace_objective_steered(
        self,
        *,
        trace: TrainingTrace,
        switch_threshold: float,
        use_sampling: bool,
    ) -> dict[str, Any]:
        """ETA Eq.3 objective: distortion is the expert-action NLL through
        the steered frozen model, so the target dimension is the action
        space (decoupled from n_z) and gradient reaches the controller only
        via the controlled substrate forward."""

        torch = _require_torch()
        scorer = self._action_scorer
        if scorer is None or self._module is None:
            raise RuntimeError(
                "Steered objective requires an action scorer and an "
                "initialized module."
            )
        if self._steer_W is None or self._steer_b is None:
            raise RuntimeError(
                "Steered objective requires the initialized steering head."
            )
        action_targets = tuple(
            step.expert_action_target for step in trace.steps
        )
        if any(target is None for target in action_targets):
            raise ValueError(
                f"Steered-action SSL trace {trace.trace_id!r} requires an "
                "ExpertActionTarget on every step."
            )
        observation_texts = tuple(
            step.observation_text for step in trace.steps
        )
        if any(not text.strip() for text in observation_texts):
            raise ValueError(
                f"Steered-action SSL trace {trace.trace_id!r} requires a "
                "nonempty observation_text on every step."
            )
        inputs = _step_input_vectors(trace, self._module.n_input)
        out = self._module.rollout(
            inputs,
            switch_threshold=switch_threshold,
            generator=self._generator if use_sampling else None,
            gate_mode="continuous",
            posterior_parameterization=self._posterior_parameterization,
        )
        control_stack = torch.stack(out["controls"])
        deltas = (
            torch.matmul(control_stack, self._steer_W.transpose(0, 1))
            + self._steer_b
        )
        action_indices = tuple(
            scorer.action_index(target.action_id)
            for target in action_targets
        )
        nll = scorer.action_nll(
            source_texts=observation_texts,
            control_deltas=deltas,
            action_indices=action_indices,
        )
        prediction_terms = list(torch.unbind(nll))
        kl_terms: list[Any] = []
        for mean, std in zip(out["means"], out["stds"], strict=True):
            kl_terms.append(
                0.5
                * torch.mean(
                    mean.pow(2) + std.pow(2) - 1.0 - 2.0 * torch.log(std)
                )
            )
        eligible_probabilities = out["switch_probabilities"][1:]
        zero = prediction_terms[0].detach() * 0.0
        # Switch-rate telemetry (never enters the Eq.3 loss; the session
        # constructor forces all switch weights to zero in this mode).
        if eligible_probabilities:
            epsilon = 1e-8
            prior = torch.tensor(self._switch_prior, dtype=torch.float64)
            mean_probability = torch.clamp(
                torch.stack(
                    [p.detach() for p in eligible_probabilities]
                ).mean(),
                epsilon,
                1.0 - epsilon,
            )
            switch_rate_loss = (
                mean_probability * torch.log(mean_probability / prior)
                + (1.0 - mean_probability)
                * torch.log((1.0 - mean_probability) / (1.0 - prior))
            )
        else:
            switch_rate_loss = zero
        gate_preferences = tuple(
            torch.zeros((), dtype=torch.float64)
            for _ in eligible_probabilities
        )
        target_values: list[float] = []
        for target in action_targets:
            target_values.extend(target.values)
        return {
            "prediction_loss": torch.stack(prediction_terms).mean(),
            "keep_prediction_loss": zero,
            "switch_prediction_loss": zero,
            "kl_loss": torch.stack(kl_terms).mean(),
            "switch_rate_loss": switch_rate_loss,
            "switch_binary_loss": zero,
            "switch_group_loss": zero,
            "gate_choice_loss": zero,
            "trained_steps": len(prediction_terms),
            "switch_probabilities": eligible_probabilities,
            "hard_switches": out["hard_switches"][1:],
            "gate_preferences": gate_preferences,
            "supervision_target": "steered-action-nll",
            "expert_action_supervision": True,
            "prediction_horizon": 1,
            "expert_action_switch_labels": tuple(
                float(
                    action_targets[index].action_id
                    != action_targets[index - 1].action_id
                )
                for index in range(1, len(action_targets))
            ),
            "target_variance": (
                sum(
                    (value - sum(target_values) / len(target_values)) ** 2
                    for value in target_values
                )
                / len(target_values)
                if target_values
                else 0.0
            ),
        }

    def evaluate_batch(
        self,
        *,
        store: Any,
        traces: tuple[TrainingTrace, ...],
        batch_id: str,
        switch_threshold: float = 0.55,
    ) -> StoreSSLEvaluationReport:
        """Deterministic no-grad readout of rate/distortion on given traces.

        Uses the posterior mean (no sampling) so the readout is the same
        deterministic path the runtime would consume. Never mutates the
        store or the optimizer state.
        """

        if not traces:
            raise ValueError("Store SSL evaluation requires at least one trace.")
        self._ensure_initialized(store=store)
        torch = _require_torch()
        with torch.no_grad():
            objectives = tuple(
                self._trace_objective(
                    trace=trace,
                    switch_threshold=switch_threshold,
                    use_sampling=False,
                )
                for trace in traces
            )
        supervision_targets = {
            str(objective["supervision_target"]) for objective in objectives
        }
        if len(supervision_targets) != 1:
            raise ValueError(
                "Store SSL evaluation cannot mix supervision modes: "
                f"{tuple(sorted(supervision_targets))}."
            )
        total_steps = sum(
            int(objective["trained_steps"]) for objective in objectives
        )
        distortion = float(
            sum(
                float(objective["prediction_loss"])
                * int(objective["trained_steps"])
                for objective in objectives
            )
            / total_steps
        )
        kl_rate = float(
            sum(
                float(objective["kl_loss"]) * int(objective["trained_steps"])
                for objective in objectives
            )
            / total_steps
        )
        probabilities = tuple(
            float(probability)
            for objective in objectives
            for probability in objective["switch_probabilities"]
        )
        hard_switches = tuple(
            float(hard)
            for objective in objectives
            for hard in objective["hard_switches"]
        )
        rows = tuple(
            (
                float(probability),
                float(hard),
                expected,
            )
            for objective in objectives
            if objective["expert_action_switch_labels"]
            for probability, hard, expected in zip(
                objective["switch_probabilities"],
                objective["hard_switches"],
                objective["expert_action_switch_labels"],
                strict=True,
            )
        )
        true_positive = sum(
            1 for _p, predicted, expected in rows
            if predicted >= 0.5 and expected >= 0.5
        )
        predicted_positive = sum(
            1 for _p, predicted, _e in rows if predicted >= 0.5
        )
        expected_positive = sum(
            1 for _p, _predicted, expected in rows if expected >= 0.5
        )
        precision = true_positive / max(predicted_positive, 1)
        recall = true_positive / max(expected_positive, 1)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        boundary_probabilities = tuple(
            probability for probability, _h, expected in rows
            if expected >= 0.5
        )
        continuation_probabilities = tuple(
            probability for probability, _h, expected in rows
            if expected < 0.5
        )
        return StoreSSLEvaluationReport(
            batch_id=batch_id,
            trace_count=len(traces),
            step_count=total_steps,
            distortion=distortion,
            kl_rate=kl_rate,
            mean_switch_probability=(
                sum(probabilities) / len(probabilities)
                if probabilities
                else 0.0
            ),
            hard_switch_frequency=(
                sum(hard_switches) / len(hard_switches)
                if hard_switches
                else 0.0
            ),
            boundary_precision=precision,
            boundary_recall=recall,
            boundary_f1=f1,
            boundary_switch_probability=(
                sum(boundary_probabilities) / len(boundary_probabilities)
                if boundary_probabilities
                else 0.0
            ),
            continuation_switch_probability=(
                sum(continuation_probabilities)
                / len(continuation_probabilities)
                if continuation_probabilities
                else 0.0
            ),
            switch_threshold=switch_threshold,
            supervision_target=next(iter(supervision_targets)),
            description=(
                f"store SSL evaluation batch={batch_id} "
                f"traces={len(traces)} steps={total_steps} "
                f"distortion={distortion:.4f} kl_rate={kl_rate:.4f}"
            ),
        )

    def train_batch(
        self,
        *,
        store: Any,
        traces: tuple[TrainingTrace, ...],
        batch_id: str,
        switch_threshold: float = 0.55,
        write_back: bool,
    ) -> StoreSSLReport:
        """Optimize one unordered trajectory batch with persistent Adam state."""

        if not traces:
            raise ValueError("Store SSL batch requires at least one trace.")
        if write_back and self._action_scorer is not None:
            # The steering head and any joint-arm substrate parameters have
            # no representation in the pure ndim store; writing back only the
            # ndim floats would silently drop half the trained controller.
            raise RuntimeError(
                "Steered-action SSL sessions cannot write back to the pure "
                "ndim parameter store; run with write_back=False."
            )
        self._ensure_initialized(store=store)
        if write_back:
            self._assert_active_store_not_mutated(store=store)
        if self._module is None or self._optimizer is None:
            raise RuntimeError("Store SSL session failed to initialize.")
        torch = _require_torch()
        before = [
            parameter.detach().clone()
            for parameter in self._all_parameters()
        ]
        self._optimizer.zero_grad()
        objectives = tuple(
            self._trace_objective(
                trace=trace,
                switch_threshold=switch_threshold,
                use_sampling=True,
            )
            for trace in traces
        )
        supervision_targets = {
            str(objective["supervision_target"])
            for objective in objectives
        }
        if len(supervision_targets) != 1:
            raise ValueError(
                "Store SSL batch cannot mix expert-action and residual-proxy "
                f"supervision: {tuple(sorted(supervision_targets))}."
            )
        supervision_target = next(iter(supervision_targets))
        expert_action_supervision = all(
            bool(objective["expert_action_supervision"])
            for objective in objectives
        )
        effective_prediction_horizon = max(
            int(objective["prediction_horizon"])
            for objective in objectives
        )
        total_steps = sum(
            int(objective["trained_steps"]) for objective in objectives
        )

        def weighted_mean(name: str) -> Any:
            return sum(
                objective[name] * int(objective["trained_steps"])
                for objective in objectives
            ) / total_steps

        prediction_loss = weighted_mean("prediction_loss")
        keep_prediction_loss = weighted_mean("keep_prediction_loss")
        switch_prediction_loss = weighted_mean("switch_prediction_loss")
        kl_loss = weighted_mean("kl_loss")
        switch_rate_loss = weighted_mean("switch_rate_loss")
        switch_binary_loss = weighted_mean("switch_binary_loss")
        switch_group_loss = weighted_mean("switch_group_loss")
        gate_choice_loss = weighted_mean("gate_choice_loss")
        target_variance = sum(
            float(objective["target_variance"])
            * int(objective["trained_steps"])
            for objective in objectives
        ) / total_steps
        total = (
            prediction_loss
            + self._proposal_prediction_weight * switch_prediction_loss
            + self._gate_choice_weight * gate_choice_loss
            + self._alpha * kl_loss
            + self._switch_rate_weight * switch_rate_loss
            + self._switch_binary_weight * switch_binary_loss
            + self._switch_group_weight * switch_group_loss
        )
        total.backward()
        grad_norm = math.sqrt(
            sum(
                float(parameter.grad.pow(2).sum())
                for parameter in self._all_parameters()
                if parameter.grad is not None
            )
        )
        self._optimizer.step()
        self._optimizer_step += 1

        changed = 0
        total_params = 0
        for old, current in zip(
            before,
            self._all_parameters(),
            strict=True,
        ):
            difference = (current.detach().to(old.device) - old).abs()
            changed += int((difference > 1e-12).sum())
            total_params += int(difference.numel())

        with torch.no_grad():
            post_update_objectives = tuple(
                self._trace_objective(
                    trace=trace,
                    switch_threshold=switch_threshold,
                )
                for trace in traces
            )
        post_update_probabilities = tuple(
            float(probability.detach())
            for objective in post_update_objectives
            for probability in objective["switch_probabilities"]
        )
        calibrated_threshold = (
            store.calibrate_beta_threshold(
                post_update_probabilities,
                target_rate=self._switch_prior,
            )
            if write_back
            else _rate_calibrated_threshold(
                post_update_probabilities,
                target_rate=self._switch_prior,
                fallback=switch_threshold,
            )
        )
        if self._action_scorer is None:
            with torch.no_grad():
                diagnostic_objectives = tuple(
                    self._trace_objective(
                        trace=trace,
                        switch_threshold=calibrated_threshold,
                    )
                    for trace in traces
                )
        else:
            # Steered mode: hard switches are pure threshold telemetry on the
            # continuous gate, so re-thresholding the post-update pass is
            # exact and avoids a second full pass through the frozen model.
            diagnostic_objectives = tuple(
                {
                    **objective,
                    "hard_switches": tuple(
                        (
                            probability.detach() >= calibrated_threshold
                        ).to(torch.float64)
                        for probability in objective["switch_probabilities"]
                    ),
                }
                for objective in post_update_objectives
            )
        switch_probabilities = tuple(
            float(probability.detach())
            for objective in diagnostic_objectives
            for probability in objective["switch_probabilities"]
        )
        hard_switches = tuple(
            float(hard_switch.detach())
            for objective in diagnostic_objectives
            for hard_switch in objective["hard_switches"]
        )
        mean_probability = (
            sum(switch_probabilities) / len(switch_probabilities)
            if switch_probabilities
            else 0.0
        )
        hard_switch_ratio = (
            sum(hard_switches) / len(hard_switches)
            if hard_switches
            else 0.0
        )
        expert_switch_rows = tuple(
            (
                float(probability.detach()),
                float(hard_switch.detach()),
                float(preference.detach()),
                expected,
            )
            for objective in diagnostic_objectives
            if objective["expert_action_switch_labels"]
            for probability, hard_switch, preference, expected in zip(
                objective["switch_probabilities"],
                objective["hard_switches"],
                objective["gate_preferences"],
                objective["expert_action_switch_labels"],
                strict=True,
            )
        )
        true_positive = sum(
            1
            for _probability, predicted, _preference, expected in expert_switch_rows
            if predicted >= 0.5 and expected >= 0.5
        )
        predicted_positive = sum(
            1
            for _probability, predicted, _preference, _expected in expert_switch_rows
            if predicted >= 0.5
        )
        expected_positive = sum(
            1
            for _probability, _predicted, _preference, expected in expert_switch_rows
            if expected >= 0.5
        )
        boundary_precision = true_positive / max(predicted_positive, 1)
        boundary_recall = true_positive / max(expected_positive, 1)
        expert_action_boundary_f1 = (
            2.0
            * boundary_precision
            * boundary_recall
            / (boundary_precision + boundary_recall)
            if boundary_precision + boundary_recall > 0.0
            else 0.0
        )
        boundary_probabilities = tuple(
            probability
            for probability, _predicted, _preference, expected in expert_switch_rows
            if expected >= 0.5
        )
        continuation_probabilities = tuple(
            probability
            for probability, _predicted, _preference, expected in expert_switch_rows
            if expected < 0.5
        )
        boundary_preferences = tuple(
            preference
            for _probability, _predicted, preference, expected in expert_switch_rows
            if expected >= 0.5
        )
        continuation_preferences = tuple(
            preference
            for _probability, _predicted, preference, expected in expert_switch_rows
            if expected < 0.5
        )

        candidate_encoder = self._module.to_encoder_params(
            self._module.n_input
        )
        candidate_switch = self._module.to_switch_params()
        candidate_decoder = self._module.to_decoder_params()
        candidate_parameters = (
            candidate_encoder,
            candidate_switch,
            candidate_decoder,
        )
        if write_back:
            (
                store.ndim_encoder_parameters,
                store.ndim_switch_parameters,
                store.ndim_decoder_parameters,
            ) = candidate_parameters
            self._last_written_parameters = candidate_parameters

        return StoreSSLReport(
            trace_id=batch_id,
            prediction_loss=float(prediction_loss.detach()),
            kl_loss=float(kl_loss.detach()),
            total_loss=float(total.detach()),
            trained_steps=total_steps,
            switch_sparsity=1.0 - mean_probability,
            binary_switch_ratio=hard_switch_ratio,
            grad_norm=grad_norm,
            parameters_changed=changed,
            parameter_change_rate=changed / max(total_params, 1),
            wrote_back=write_back,
            trajectory_count=len(traces),
            optimizer_step=self._optimizer_step,
            optimizer_state_reused=self._optimizer_step > 1,
            switch_rate_loss=float(switch_rate_loss.detach()),
            switch_binary_loss=float(switch_binary_loss.detach()),
            switch_group_loss=float(switch_group_loss.detach()),
            gate_choice_loss=float(gate_choice_loss.detach()),
            keep_prediction_loss=float(keep_prediction_loss.detach()),
            switch_prediction_loss=float(switch_prediction_loss.detach()),
            mean_switch_probability=mean_probability,
            prediction_horizon=effective_prediction_horizon,
            distortion_target=(
                "steered-action-nll"
                if self._action_scorer is not None
                else self._distortion_target
            ),
            supervision_target=supervision_target,
            expert_action_supervision=expert_action_supervision,
            expert_action_boundary_f1=expert_action_boundary_f1,
            boundary_switch_probability=(
                sum(boundary_probabilities) / len(boundary_probabilities)
                if boundary_probabilities
                else 0.0
            ),
            continuation_switch_probability=(
                sum(continuation_probabilities)
                / len(continuation_probabilities)
                if continuation_probabilities
                else 0.0
            ),
            boundary_switch_preference=(
                sum(boundary_preferences) / len(boundary_preferences)
                if boundary_preferences
                else 0.0
            ),
            continuation_switch_preference=(
                sum(continuation_preferences) / len(continuation_preferences)
                if continuation_preferences
                else 0.0
            ),
            target_variance=target_variance,
            switch_threshold_before=switch_threshold,
            switch_threshold_after=calibrated_threshold,
            candidate_encoder_parameters=candidate_encoder,
            candidate_switch_parameters=candidate_switch,
            candidate_decoder_parameters=candidate_decoder,
            description=(
                f"store SSL batch={batch_id} trajectories={len(traces)} "
                f"optimizer_step={self._optimizer_step} "
                f"pred={float(prediction_loss.detach()):.4f} "
                f"kl={float(kl_loss.detach()):.4f} "
                f"switch_p={mean_probability:.4f} "
                f"threshold={switch_threshold:.4f}->{calibrated_threshold:.4f} "
                f"switch_rate={float(switch_rate_loss.detach()):.4f} "
                f"changed={changed} wrote_back={write_back}"
            ),
        )


def train_store_ssl(
    *,
    store: Any,
    trace: TrainingTrace,
    n_z: int,
    alpha: float = 0.1,
    learning_rate: float = 0.02,
    switch_threshold: float = 0.55,
    switch_prior: float = 0.10,
    switch_rate_weight: float = 0.0,
    switch_binary_weight: float = 0.0,
    switch_group_weight: float = 0.0,
    proposal_prediction_weight: float = 0.50,
    gate_choice_weight: float = 0.0,
    gate_choice_temperature: float = 0.02,
    prediction_horizon: int = 3,
    distortion_target: str = "innovation",
    write_back: bool,
) -> StoreSSLReport:
    """Run one standalone SSL step; persistent callers own a session."""

    session = StoreSSLTrainingSession(
        n_z=n_z,
        alpha=alpha,
        learning_rate=learning_rate,
        switch_prior=switch_prior,
        switch_rate_weight=switch_rate_weight,
        switch_binary_weight=switch_binary_weight,
        switch_group_weight=switch_group_weight,
        proposal_prediction_weight=proposal_prediction_weight,
        gate_choice_weight=gate_choice_weight,
        gate_choice_temperature=gate_choice_temperature,
        prediction_horizon=prediction_horizon,
        distortion_target=distortion_target,
    )
    return session.train_batch(
        store=store,
        traces=(trace,),
        batch_id=trace.trace_id,
        switch_threshold=switch_threshold,
        write_back=write_back,
    )
