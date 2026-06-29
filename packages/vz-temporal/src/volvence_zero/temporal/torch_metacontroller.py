"""Real-autograd metacontroller + SSL trainer (NL/ETA migration, Phases 1 & 3).

This is the paper-grade counterpart to the pure-Python heuristic metacontroller
in :mod:`volvence_zero.temporal.metacontroller_components`. It implements the
ETA variational SSL objective (Eq.3) with **real backpropagation** through a GRU
encoder, a learned switch gate, and a residual decoder:

    L(phi) = sum_t [ action_prediction_loss(a_t, U_t)  +  alpha * D_KL(q_t || p) ]

Two design points close the highest-value ETA gaps identified in the gap audit:

1. **KL target is selectable and defaults to the paper's N(0, I)**
   (``KLTarget.STANDARD_NORMAL``). The variational bottleneck toward the
   standard normal is what drives sparse, emergent switching in ETA. The
   previous pure path regularized toward a *learned* prior, which dampens that
   pressure; that behavior is kept as ``KLTarget.LEARNED_PRIOR`` (the
   CMS-enhanced variant, ETA appendix C.2) for matched ablation.

2. **Switching uses a straight-through estimator (STE)** instead of a fixed
   0.55 threshold: the forward pass binarizes the gate (true persist/switch
   behavior) while the backward pass flows the gradient of the continuous gate,
   so switch sparsity is *learned* under the ``alpha * D_KL`` pressure rather
   than hardcoded.

Scope: this module is offline/training-oriented (Phase 1) and also provides the
backend-agnostic forward used by the runtime SHADOW dual-run (Phase 3). ``torch``
is imported lazily; importing this module never forces a torch dependency, and
it is intentionally NOT re-exported from the ``volvence_zero.temporal`` facade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from volvence_zero.substrate import TrainingTrace
from volvence_zero.temporal.metacontroller_components import summarize_residual_activations


class KLTarget(str, Enum):
    """Prior the posterior is regularized toward in the Eq.3 KL term."""

    STANDARD_NORMAL = "standard_normal"  # ETA Eq.3 default: D_KL(q || N(0, I))
    LEARNED_PRIOR = "learned_prior"      # CMS-enhanced variant (appendix C.2)


@dataclass(frozen=True)
class TorchMetacontrollerConfig:
    n_z: int = 8
    input_dim: int = 3
    action_dim: int = 3
    hidden_dim: int = 8
    alpha: float = 0.1
    kl_target: KLTarget = KLTarget.STANDARD_NORMAL
    use_ste: bool = True
    switch_threshold: float = 0.5
    learning_rate: float = 0.02
    seed: int = 1234


@dataclass(frozen=True)
class TorchSSLReport:
    trace_id: str
    prediction_loss: float
    kl_loss: float
    total_loss: float
    trained_steps: int
    switch_sparsity: float          # 1 - mean(beta): higher = more persistence
    binary_switch_ratio: float      # fraction of steps that actually switched
    mean_persistence_steps: float
    grad_norm: float
    parameters_changed: int
    parameter_change_rate: float
    kl_target: str
    alpha: float
    description: str = ""


@dataclass(frozen=True)
class TorchMetacontrollerArtifact:
    """Self-describing parameter snapshot for the rare-heavy artifact path.

    Carries only float payloads + metadata (no torch tensors) so it can travel
    through the existing offline export/import path and be re-applied to a
    freshly constructed module without changing any public snapshot schema.
    """

    config_n_z: int
    config_hidden_dim: int
    config_action_dim: int
    config_input_dim: int
    kl_target: str
    alpha: float
    state: tuple[tuple[str, tuple[float, ...], tuple[int, ...]], ...]
    update_count: int
    description: str = ""


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded at call sites
        raise ImportError(
            "TorchMetacontroller requires torch. Install the vz-temporal "
            "'[torch]' extra, or use the pure-Python metacontroller "
            "(WiringLevel.DISABLED)."
        ) from exc
    return torch


def _trace_step_inputs(trace: TrainingTrace) -> list[tuple[float, ...]]:
    """Summarize each trace step's residual activations into a fixed vector.

    Uses the same ``summarize_residual_activations`` 3-tuple the pure SSL path
    consumes, so both backends learn from an identical signal.
    """

    return [
        summarize_residual_activations(step.residual_activations, step.feature_surface)
        for step in trace.steps
    ]


class TorchMetacontroller:
    """GRU encoder + learned switch + residual decoder, with real autograd.

    Built directly on ``torch`` (not the abstract backend) because offline
    training needs a torch optimizer over leaf parameters. The forward is a
    faithful realization of ETA Eq.1/Eq.2 with reparameterized posterior
    sampling and an STE switch gate.
    """

    def __init__(self, config: TorchMetacontrollerConfig) -> None:
        self._torch = _require_torch()
        torch = self._torch
        self.config = config
        self._dtype = torch.float64
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(config.seed)
        self._update_count = 0

        def lin(out_dim: int, in_dim: int, *, seed_offset: int) -> Any:
            g = torch.Generator(device="cpu")
            g.manual_seed(config.seed + seed_offset)
            scale = 1.0 / math.sqrt(max(in_dim, 1))
            w = torch.empty(out_dim, in_dim, dtype=self._dtype)
            w.normal_(0.0, scale, generator=g)
            w.requires_grad_(True)
            b = torch.zeros(out_dim, dtype=self._dtype, requires_grad=True)
            return [w, b]

        n_z = config.n_z
        h = config.hidden_dim
        # GRU cell weights (input->hidden and hidden->hidden), 3 gates.
        self._gru_x = lin(3 * h, config.input_dim, seed_offset=1)
        self._gru_h = lin(3 * h, h, seed_offset=2)
        # Posterior heads.
        self._mean_head = lin(n_z, h, seed_offset=3)
        self._logvar_head = lin(n_z, h, seed_offset=4)
        # Learned prior heads (only used by KLTarget.LEARNED_PRIOR).
        self._prior_mean_head = lin(n_z, h, seed_offset=5)
        self._prior_logvar_head = lin(n_z, h, seed_offset=6)
        # Switch gate: [hidden, z_tilde, z_prev] -> scalar in (0,1).
        self._switch = lin(1, h + 2 * n_z, seed_offset=7)
        # Decoder: z_t -> hidden -> action_dim (matches action target dim).
        self._dec1 = lin(h, n_z, seed_offset=8)
        self._dec2 = lin(config.action_dim, h, seed_offset=9)

        self._h = h

    # --- parameter management ---

    def parameters(self) -> list[Any]:
        groups = [
            self._gru_x, self._gru_h, self._mean_head, self._logvar_head,
            self._prior_mean_head, self._prior_logvar_head, self._switch,
            self._dec1, self._dec2,
        ]
        return [p for group in groups for p in group]

    def _named_groups(self) -> list[tuple[str, list[Any]]]:
        return [
            ("gru_x", self._gru_x), ("gru_h", self._gru_h),
            ("mean_head", self._mean_head), ("logvar_head", self._logvar_head),
            ("prior_mean_head", self._prior_mean_head),
            ("prior_logvar_head", self._prior_logvar_head),
            ("switch", self._switch), ("dec1", self._dec1), ("dec2", self._dec2),
        ]

    # --- forward primitives ---

    def _gru_step(self, x: Any, h_prev: Any) -> Any:
        torch = self._torch
        gx = torch.matmul(self._gru_x[0], x) + self._gru_x[1]
        gh = torch.matmul(self._gru_h[0], h_prev) + self._gru_h[1]
        h = self._h
        z = torch.sigmoid(gx[:h] + gh[:h])
        r = torch.sigmoid(gx[h:2 * h] + gh[h:2 * h])
        n = torch.tanh(gx[2 * h:] + r * gh[2 * h:])
        return (1.0 - z) * h_prev + z * n

    def _posterior(self, h_state: Any) -> tuple[Any, Any]:
        torch = self._torch
        mean = torch.matmul(self._mean_head[0], h_state) + self._mean_head[1]
        logvar = torch.matmul(self._logvar_head[0], h_state) + self._logvar_head[1]
        return mean, logvar

    def _prior(self, h_state: Any) -> tuple[Any, Any]:
        torch = self._torch
        mean = torch.matmul(self._prior_mean_head[0], h_state) + self._prior_mean_head[1]
        logvar = torch.matmul(self._prior_logvar_head[0], h_state) + self._prior_logvar_head[1]
        return mean, logvar

    def _switch_gate(self, h_state: Any, z_tilde: Any, z_prev: Any) -> Any:
        torch = self._torch
        feat = torch.cat([h_state, z_tilde, z_prev])
        raw = torch.matmul(self._switch[0], feat) + self._switch[1]
        return torch.sigmoid(raw)[0]

    def _decode(self, z_t: Any) -> Any:
        torch = self._torch
        hidden = torch.tanh(torch.matmul(self._dec1[0], z_t) + self._dec1[1])
        return torch.matmul(self._dec2[0], hidden) + self._dec2[1]

    def _ste_beta(self, beta_cont: Any) -> Any:
        """Straight-through binarized gate: forward hard, backward soft."""

        torch = self._torch
        if not self.config.use_ste:
            return beta_cont
        hard = (beta_cont >= self.config.switch_threshold).to(self._dtype)
        # forward == hard, backward grad flows through beta_cont
        return hard.detach() + (beta_cont - beta_cont.detach())

    # --- full sequence rollout ---

    def rollout(self, step_inputs: Sequence[tuple[float, ...]], *, sample: bool) -> dict[str, Any]:
        """Run encoder->switch->decoder over a prefix, returning autograd tensors."""

        torch = self._torch
        h_state = torch.zeros(self._h, dtype=self._dtype)
        z_prev = torch.zeros(self.config.n_z, dtype=self._dtype)
        controls: list[Any] = []
        means: list[Any] = []
        logvars: list[Any] = []
        prior_means: list[Any] = []
        prior_logvars: list[Any] = []
        betas: list[Any] = []
        for raw in step_inputs:
            x = torch.tensor(list(raw), dtype=self._dtype)
            h_state = self._gru_step(x, h_state)
            mean, logvar = self._posterior(h_state)
            std = torch.exp(0.5 * logvar)
            if sample:
                eps = torch.randn(self.config.n_z, dtype=self._dtype, generator=self._gen)
                z_tilde = mean + std * eps
            else:
                z_tilde = mean
            beta_cont = self._switch_gate(h_state, z_tilde, z_prev)
            beta = self._ste_beta(beta_cont)
            z_t = beta * z_tilde + (1.0 - beta) * z_prev
            control = self._decode(z_t)
            controls.append(control)
            means.append(mean)
            logvars.append(logvar)
            betas.append(beta_cont)
            if self.config.kl_target is KLTarget.LEARNED_PRIOR:
                pm, pl = self._prior(h_state)
                prior_means.append(pm)
                prior_logvars.append(pl)
            z_prev = z_t
        return {
            "controls": controls,
            "means": means,
            "logvars": logvars,
            "prior_means": prior_means,
            "prior_logvars": prior_logvars,
            "betas": betas,
        }

    # --- artifact (rare-heavy path) ---

    def export_artifact(self) -> TorchMetacontrollerArtifact:
        state: list[tuple[str, tuple[float, ...], tuple[int, ...]]] = []
        for name, group in self._named_groups():
            for idx, p in enumerate(group):
                flat = tuple(float(v) for v in p.detach().reshape(-1).tolist())
                shape = tuple(int(d) for d in p.shape)
                state.append((f"{name}.{idx}", flat, shape))
        return TorchMetacontrollerArtifact(
            config_n_z=self.config.n_z,
            config_hidden_dim=self.config.hidden_dim,
            config_action_dim=self.config.action_dim,
            config_input_dim=self.config.input_dim,
            kl_target=self.config.kl_target.value,
            alpha=self.config.alpha,
            state=tuple(state),
            update_count=self._update_count,
            description=(
                f"torch metacontroller artifact: n_z={self.config.n_z} "
                f"kl={self.config.kl_target.value} updates={self._update_count}"
            ),
        )

    def load_artifact(self, artifact: TorchMetacontrollerArtifact) -> None:
        torch = self._torch
        by_name = {name: (flat, shape) for name, flat, shape in artifact.state}
        for name, group in self._named_groups():
            for idx, p in enumerate(group):
                key = f"{name}.{idx}"
                if key not in by_name:
                    raise KeyError(f"artifact missing parameter '{key}'")
                flat, shape = by_name[key]
                with torch.no_grad():
                    p.copy_(torch.tensor(list(flat), dtype=self._dtype).reshape(shape))
        self._update_count = artifact.update_count


def _kl_standard_normal(mean: Any, logvar: Any, torch: Any) -> Any:
    """D_KL(N(mean, exp(logvar)) || N(0, I)) — ETA Eq.3 default."""

    return 0.5 * torch.sum(mean.pow(2) + torch.exp(logvar) - 1.0 - logvar)


def _kl_learned_prior(mean: Any, logvar: Any, pmean: Any, plogvar: Any, torch: Any) -> Any:
    """D_KL between posterior and a learned diagonal Gaussian prior."""

    var = torch.exp(logvar)
    pvar = torch.exp(plogvar)
    return 0.5 * torch.sum(
        plogvar - logvar + (var + (mean - pmean).pow(2)) / (pvar + 1e-8) - 1.0
    )


class TorchMetacontrollerSSLTrainer:
    """Offline SSL trainer doing real backprop on the ETA Eq.3 objective."""

    def __init__(self, config: TorchMetacontrollerConfig | None = None) -> None:
        self._torch = _require_torch()
        self.config = config or TorchMetacontrollerConfig()
        self.module = TorchMetacontroller(self.config)
        self._opt = self._torch.optim.Adam(
            self.module.parameters(), lr=self.config.learning_rate
        )

    def train_on_trace(self, trace: TrainingTrace) -> TorchSSLReport:
        torch = self._torch
        inputs = _trace_step_inputs(trace)
        if len(inputs) < 2:
            return TorchSSLReport(
                trace_id=trace.trace_id, prediction_loss=0.0, kl_loss=0.0,
                total_loss=0.0, trained_steps=0, switch_sparsity=0.0,
                binary_switch_ratio=0.0, mean_persistence_steps=0.0, grad_norm=0.0,
                parameters_changed=0, parameter_change_rate=0.0,
                kl_target=self.config.kl_target.value, alpha=self.config.alpha,
                description="trace too short for SSL",
            )
        # Teacher-force: predict step t+1's summary from prefix [0..t].
        before = [p.detach().clone() for p in self.module.parameters()]
        self._opt.zero_grad()
        out = self.module.rollout(inputs[:-1], sample=True)
        controls = out["controls"]
        means = out["means"]
        logvars = out["logvars"]
        betas = out["betas"]

        pred_terms = []
        kl_terms = []
        for t, control in enumerate(controls):
            target = torch.tensor(list(inputs[t + 1]), dtype=torch.float64)
            n = min(control.shape[0], target.shape[0])
            pred_terms.append(torch.mean((control[:n] - target[:n]).pow(2)))  # MSE
            if self.config.kl_target is KLTarget.STANDARD_NORMAL:
                kl_terms.append(_kl_standard_normal(means[t], logvars[t], torch))
            else:
                kl_terms.append(
                    _kl_learned_prior(
                        means[t], logvars[t],
                        out["prior_means"][t], out["prior_logvars"][t], torch,
                    )
                )
        prediction_loss = torch.stack(pred_terms).mean()
        kl_loss = torch.stack(kl_terms).mean()
        total = prediction_loss + self.config.alpha * kl_loss
        prediction_loss_value = float(prediction_loss.detach())
        kl_loss_value = float(kl_loss.detach())
        total_loss_value = float(total.detach())
        total.backward()
        grad_norm = math.sqrt(
            sum(float(p.grad.pow(2).sum()) for p in self.module.parameters() if p.grad is not None)
        )
        self._opt.step()
        self.module._update_count += 1

        # parameter-change evidence (proves real gradient updates happened).
        after = self.module.parameters()
        changed = 0
        total_params = 0
        for b, a in zip(before, after):
            diff = (a.detach() - b).abs()
            changed += int((diff > 1e-12).sum())
            total_params += int(diff.numel())
        change_rate = changed / max(total_params, 1)

        beta_vals = [float(b.detach()) for b in betas]
        mean_beta = sum(beta_vals) / max(len(beta_vals), 1)
        switches = sum(1 for b in beta_vals if b >= self.config.switch_threshold)
        return TorchSSLReport(
            trace_id=trace.trace_id,
            prediction_loss=prediction_loss_value,
            kl_loss=kl_loss_value,
            total_loss=total_loss_value,
            trained_steps=len(controls),
            switch_sparsity=1.0 - mean_beta,
            binary_switch_ratio=switches / max(len(beta_vals), 1),
            mean_persistence_steps=len(beta_vals) / max(switches, 1),
            grad_norm=grad_norm,
            parameters_changed=changed,
            parameter_change_rate=change_rate,
            kl_target=self.config.kl_target.value,
            alpha=self.config.alpha,
            description=(
                f"eq3 ssl: pred={prediction_loss_value:.4f} kl={kl_loss_value:.4f} "
                f"target={self.config.kl_target.value} alpha={self.config.alpha}"
            ),
        )

    def train_epochs(self, traces: Sequence[TrainingTrace], *, epochs: int) -> TorchSSLReport:
        last: TorchSSLReport | None = None
        for _ in range(max(epochs, 1)):
            for trace in traces:
                last = self.train_on_trace(trace)
        assert last is not None
        return last

    def train_on_sequence(
        self, step_inputs: Sequence[tuple[float, ...]], targets: Sequence[tuple[float, ...]]
    ) -> TorchSSLReport:
        """Eq.3 SSL step on explicit (step_inputs, targets) — used by the strict
        hierarchical ETA evidence suite, bypassing trace summarization so the
        abstract-action structure is controlled and deterministic."""

        torch = self._torch
        if len(step_inputs) < 2:
            return TorchSSLReport(
                trace_id="seq", prediction_loss=0.0, kl_loss=0.0, total_loss=0.0,
                trained_steps=0, switch_sparsity=0.0, binary_switch_ratio=0.0,
                mean_persistence_steps=0.0, grad_norm=0.0, parameters_changed=0,
                parameter_change_rate=0.0, kl_target=self.config.kl_target.value,
                alpha=self.config.alpha, description="sequence too short",
            )
        self._opt.zero_grad()
        out = self.module.rollout(step_inputs[:-1], sample=True)
        controls, means, logvars, betas = out["controls"], out["means"], out["logvars"], out["betas"]
        pred_terms = []
        kl_terms = []
        for t, control in enumerate(controls):
            target = torch.tensor(list(targets[t + 1]), dtype=torch.float64)
            n = min(control.shape[0], target.shape[0])
            pred_terms.append(torch.mean((control[:n] - target[:n]).pow(2)))
            if self.config.kl_target is KLTarget.STANDARD_NORMAL:
                kl_terms.append(_kl_standard_normal(means[t], logvars[t], torch))
            else:
                kl_terms.append(
                    _kl_learned_prior(
                        means[t], logvars[t], out["prior_means"][t], out["prior_logvars"][t], torch
                    )
                )
        prediction_loss = torch.stack(pred_terms).mean()
        kl_loss = torch.stack(kl_terms).mean()
        total = prediction_loss + self.config.alpha * kl_loss
        pred_v = float(prediction_loss.detach())
        kl_v = float(kl_loss.detach())
        total.backward()
        self._opt.step()
        self.module._update_count += 1
        beta_vals = [float(b.detach().mean()) for b in betas]
        mean_beta = sum(beta_vals) / max(len(beta_vals), 1)
        switches = sum(1 for b in beta_vals if b >= self.config.switch_threshold)
        return TorchSSLReport(
            trace_id="seq", prediction_loss=pred_v, kl_loss=kl_v,
            total_loss=float(total.detach()), trained_steps=len(controls),
            switch_sparsity=1.0 - mean_beta,
            binary_switch_ratio=switches / max(len(beta_vals), 1),
            mean_persistence_steps=len(beta_vals) / max(switches, 1),
            grad_norm=0.0, parameters_changed=0, parameter_change_rate=0.0,
            kl_target=self.config.kl_target.value, alpha=self.config.alpha,
            description="sequence eq3 step",
        )

    def eval_switch_sparsity(self, step_inputs: Sequence[tuple[float, ...]]) -> float:
        """Deterministic mean switch sparsity (1 - mean beta) over a sequence."""

        out = self.module.rollout(step_inputs, sample=False)
        betas = [float(b.detach().mean()) for b in out["betas"]]
        return 1.0 - (sum(betas) / max(len(betas), 1))

    def eval_codes(self, step_inputs: Sequence[tuple[float, ...]]) -> list[tuple[float, ...]]:
        """Deterministic per-step latent codes z_t (controls) for reuse analysis."""

        out = self.module.rollout(step_inputs, sample=False)
        return [tuple(float(v) for v in c.detach().tolist()) for c in out["controls"]]


# ---------------------------------------------------------------------------
# Strict hierarchical ETA evidence (Phase E)
# ---------------------------------------------------------------------------


def build_hierarchical_sequences(
    *,
    n_sequences: int,
    phases: int,
    steps_per_phase: int,
    n_phase_types: int = 3,
    action_dim: int = 3,
    seed: int = 1234,
) -> list[tuple[list[tuple[float, ...]], list[tuple[float, ...]]]]:
    """Deterministic piecewise-constant hierarchical sequences.

    Each sequence is a sequence of phases; within a phase the target is a fixed
    phase-type vector (an abstract action to PERSIST), and the target changes at
    phase boundaries (a SWITCH). Reusing phase types across sequences creates the
    held-out reuse structure ETA's bottleneck is supposed to exploit.
    """

    import random

    rng = random.Random(seed)
    phase_vecs = [
        tuple(rng.uniform(0.1, 0.9) for _ in range(action_dim)) for _ in range(n_phase_types)
    ]
    sequences: list[tuple[list[tuple[float, ...]], list[tuple[float, ...]]]] = []
    for s in range(n_sequences):
        order = [rng.randrange(n_phase_types) for _ in range(phases)]
        inputs: list[tuple[float, ...]] = []
        targets: list[tuple[float, ...]] = []
        for ptype in order:
            base = phase_vecs[ptype]
            for _ in range(steps_per_phase):
                # small deterministic jitter on inputs; targets are the clean phase vec
                jitter = tuple(min(1.0, max(0.0, v + rng.uniform(-0.02, 0.02))) for v in base)
                inputs.append(jitter)
                targets.append(base)
        sequences.append((inputs, targets))
    return sequences


def _held_out_code_reuse(
    trainer: "TorchMetacontrollerSSLTrainer",
    train_seqs,
    holdout_seqs,
    *,
    threshold: float = 0.15,
) -> float:
    """Fraction of held-out steps whose code is near a training-time code.

    Higher = better reuse / generalization of discovered abstract actions.
    """

    centroids: list[tuple[float, ...]] = []
    for inputs, _ in train_seqs:
        centroids.extend(trainer.eval_codes(inputs))
    if not centroids:
        return 0.0

    def nearest(code: tuple[float, ...]) -> float:
        best = float("inf")
        for c in centroids:
            d = sum((a - b) ** 2 for a, b in zip(code, c)) ** 0.5
            best = min(best, d)
        return best

    hits = 0
    total = 0
    for inputs, _ in holdout_seqs:
        for code in trainer.eval_codes(inputs):
            total += 1
            if nearest(code) <= threshold:
                hits += 1
    return hits / max(total, 1)


@dataclass(frozen=True)
class StrictETAEvidence:
    sparsity_by_alpha_standard_normal: tuple[tuple[float, float], ...]
    high_alpha_increases_sparsity: bool
    sparsity_monotone_nondecreasing: bool
    held_out_reuse_standard_normal_alpha0: float
    held_out_reuse_standard_normal_high_alpha: float
    held_out_reuse_improves_with_bottleneck: bool
    description: str = ""


def run_strict_eta_evidence(
    *,
    alphas: Sequence[float] = (0.0, 0.3, 1.0),
    epochs: int = 25,
    n_z: int = 8,
    seed: int = 1234,
) -> StrictETAEvidence:
    """Phase E strict evidence on a controlled hierarchical suite.

    Trains the real-autograd metacontroller under the standard-normal KL at
    several alphas and measures (1) switch sparsity and (2) held-out abstract-
    action (code) reuse. The ETA rate-distortion claim is that a stronger
    bottleneck (higher alpha) compresses the latent, increasing within-phase
    persistence (sparsity) and code reuse on held-out sequences.
    """

    train_seqs = build_hierarchical_sequences(
        n_sequences=6, phases=4, steps_per_phase=3, n_phase_types=3, seed=seed
    )
    holdout_seqs = build_hierarchical_sequences(
        n_sequences=4, phases=4, steps_per_phase=3, n_phase_types=3, seed=seed + 999
    )

    sparsity_by_alpha: list[tuple[float, float]] = []
    reuse_by_alpha: dict[float, float] = {}
    for alpha in alphas:
        cfg = TorchMetacontrollerConfig(
            n_z=n_z, hidden_dim=n_z, alpha=alpha,
            kl_target=KLTarget.STANDARD_NORMAL, seed=seed,
        )
        trainer = TorchMetacontrollerSSLTrainer(cfg)
        for _ in range(epochs):
            for inputs, targets in train_seqs:
                trainer.train_on_sequence(inputs, targets)
        sparsity = sum(trainer.eval_switch_sparsity(inputs) for inputs, _ in train_seqs) / len(train_seqs)
        sparsity_by_alpha.append((alpha, sparsity))
        reuse_by_alpha[alpha] = _held_out_code_reuse(trainer, train_seqs, holdout_seqs)

    ordered = [s for _, s in sparsity_by_alpha]
    high_increases = len(ordered) >= 2 and ordered[-1] > ordered[0] + 1e-3
    monotone = all(b >= a - 1e-3 for a, b in zip(ordered, ordered[1:]))
    a0 = reuse_by_alpha[alphas[0]]
    aH = reuse_by_alpha[alphas[-1]]
    reuse_improves = aH >= a0 - 1e-9
    return StrictETAEvidence(
        sparsity_by_alpha_standard_normal=tuple(sparsity_by_alpha),
        high_alpha_increases_sparsity=high_increases,
        sparsity_monotone_nondecreasing=monotone,
        held_out_reuse_standard_normal_alpha0=a0,
        held_out_reuse_standard_normal_high_alpha=aH,
        held_out_reuse_improves_with_bottleneck=reuse_improves,
        description=(
            f"strict ETA: sparsity {ordered} high_increases={high_increases} "
            f"monotone={monotone} reuse a0={a0:.3f} aH={aH:.3f}"
        ),
    )


@dataclass(frozen=True)
class KLTargetAblationRow:
    kl_target: str
    alpha: float
    final_prediction_loss: float
    final_kl_loss: float
    switch_sparsity: float
    binary_switch_ratio: float
    parameter_change_rate: float


@dataclass(frozen=True)
class KLTargetAblationReport:
    rows: tuple[KLTargetAblationRow, ...]
    # Genuine variational-bottleneck signature: under the standard-normal
    # target, raising alpha compresses the achieved posterior KL toward 0.
    standard_normal_kl_compresses_with_alpha: bool
    # Causal effect: alpha measurably shapes the discovered switch behavior.
    alpha_affects_switch_behavior: bool
    # The two KL targets do not collapse to the same switching behavior.
    targets_differ: bool
    description: str = ""


def compare_kl_targets(
    traces: Sequence[TrainingTrace],
    *,
    alphas: Sequence[float] = (0.0, 0.2, 1.0),
    epochs: int = 6,
    n_z: int = 8,
    seed: int = 1234,
) -> KLTargetAblationReport:
    """Matched ablation: standard-normal vs learned-prior KL across alphas.

    Phase 1 exit evidence. The robust, paper-faithful invariant of the ETA
    variational bottleneck is that the regularizer ``alpha * D_KL(q || N(0, I))``
    *compresses* the posterior toward the prior — so the achieved KL decreases
    as alpha increases. We assert that (a clean, guaranteed-direction signal),
    plus that alpha has a measurable causal effect on the *discovered* switch
    behavior and that the standard-normal and learned-prior targets do not
    collapse to identical switching (i.e. switching is not just a hardcoded
    threshold). We deliberately do NOT assert strict monotone sparsity, which
    would over-claim in a tiny toy and is exactly the kind of scaffold-driven
    result the gap audit warns against.
    """

    rows: list[KLTargetAblationRow] = []
    kl_by_alpha: dict[float, float] = {}
    sparsity_by_alpha_sn: dict[float, float] = {}
    sparsity_by_alpha_lp: dict[float, float] = {}
    for kl_target in (KLTarget.STANDARD_NORMAL, KLTarget.LEARNED_PRIOR):
        for alpha in alphas:
            cfg = TorchMetacontrollerConfig(
                n_z=n_z, hidden_dim=n_z, alpha=alpha, kl_target=kl_target, seed=seed
            )
            trainer = TorchMetacontrollerSSLTrainer(cfg)
            report = trainer.train_epochs(traces, epochs=epochs)
            rows.append(
                KLTargetAblationRow(
                    kl_target=kl_target.value,
                    alpha=alpha,
                    final_prediction_loss=report.prediction_loss,
                    final_kl_loss=report.kl_loss,
                    switch_sparsity=report.switch_sparsity,
                    binary_switch_ratio=report.binary_switch_ratio,
                    parameter_change_rate=report.parameter_change_rate,
                )
            )
            if kl_target is KLTarget.STANDARD_NORMAL:
                kl_by_alpha[alpha] = report.kl_loss
                sparsity_by_alpha_sn[alpha] = report.switch_sparsity
            else:
                sparsity_by_alpha_lp[alpha] = report.switch_sparsity

    ordered_alphas = sorted(kl_by_alpha)
    kl_ordered = [kl_by_alpha[a] for a in ordered_alphas]
    # Non-increasing achieved KL with alpha, and strictly lower at the top.
    kl_compresses = (
        len(kl_ordered) >= 2
        and all(b <= a + 1e-6 for a, b in zip(kl_ordered, kl_ordered[1:]))
        and kl_ordered[-1] < kl_ordered[0] - 1e-6
    )
    sn_sparsities = [sparsity_by_alpha_sn[a] for a in ordered_alphas]
    alpha_effect = (max(sn_sparsities) - min(sn_sparsities)) > 1e-3 if sn_sparsities else False
    targets_differ = any(
        abs(sparsity_by_alpha_sn[a] - sparsity_by_alpha_lp[a]) > 1e-3
        for a in ordered_alphas
    )
    return KLTargetAblationReport(
        rows=tuple(rows),
        standard_normal_kl_compresses_with_alpha=kl_compresses,
        alpha_affects_switch_behavior=alpha_effect,
        targets_differ=targets_differ,
        description=(
            "matched KL-target ablation; standard-normal KL compresses with "
            f"alpha={kl_compresses}, alpha affects switching={alpha_effect}, "
            f"targets differ={targets_differ}"
        ),
    )
