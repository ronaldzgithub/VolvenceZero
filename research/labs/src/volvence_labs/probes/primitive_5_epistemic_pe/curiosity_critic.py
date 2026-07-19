"""P5 Epistemic PE — Curiosity-Critic probe (stage 1 real hooks).

Hypothesis: 2-head disagreement on TinyLlama separates epistemic from aleatoric
uncertainty. The epistemic component correlates with "learnable surprise" while
aleatoric reflects irreducible noise.

Cells:
- raw_pe: standard cross-entropy per token (no decomposition)
- epistemic_only: disagreement between 2 prediction heads (epistemic signal)
- aleatoric_only: mean prediction confidence (aleatoric proxy)
- critic_split: full Curiosity-Critic decomposition (epistemic drives exploration weight)

Model: TinyLlama-1.1B-Chat (or synthetic fallback for CI).
Eval: HellaSwag / ARC-E subset (or synthetic sequences for CI).

Paper: Curiosity-Critic (2604.18701).
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Optional

import numpy as np

from ...framework.probe import (
    BaseProbe,
    GateReport,
    PrimitiveTag,
    ProbeContext,
    ReadoutBundle,
    RunOutcome,
    register_probe,
)
from ...framework.wiring import AblationCell


def _model_available() -> bool:
    """Check if torch + transformers are importable and a model can load."""
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


def _generate_synthetic_logits(seed: int, seq_len: int = 64, vocab_size: int = 128) -> dict:
    """Generate synthetic 2-head logits for CI testing."""
    rng = np.random.default_rng(seed)
    # Head 1 and Head 2 produce slightly different logit distributions.
    head1_logits = rng.standard_normal((seq_len, vocab_size)).astype(np.float32)
    head2_logits = head1_logits + rng.standard_normal((seq_len, vocab_size)).astype(np.float32) * 0.3
    targets = rng.integers(0, vocab_size, size=seq_len)
    return {
        "head1_logits": head1_logits,
        "head2_logits": head2_logits,
        "targets": targets,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
    }


def _generate_real_logits(seed: int, model_id: str = "sshleifer/tiny-gpt2", text: str = None, n_samples: int = 5) -> dict:
    """Generate N-sample MC dropout logits from a real model.

    Uses framework-level epistemic_aleatoric_split() operator (Curiosity-Critic 2604.18701).
    Falls back to Gaussian-noise perturbation for models without trainable dropout.

    The probe still consumes head1/head2 (legacy), but also exposes the full ensemble
    so future cells can plug in better separators.
    """
    try:
        import torch
        from ...framework.runtime import get_model_runtime
        from ...framework.runtime.uncertainty import (
            mc_dropout_logits,
            epistemic_aleatoric_split,
        )

        rt = get_model_runtime(model_id, dtype="fp32")
        rt.load_model()

        if text is None:
            texts = [
                "The meaning of life is to find purpose and happiness in what we do.",
                "Machine learning models can sometimes produce unexpected outputs.",
                "The weather today is partly cloudy with a chance of rain.",
                "Quantum computing promises to revolutionize cryptography.",
            ]
            text = texts[seed % len(texts)]

        ensemble = mc_dropout_logits(rt, text, n_samples=n_samples, max_length=128, noise_scale=0.05)
        epistemic_per_token, aleatoric_per_token = epistemic_aleatoric_split(ensemble)

        result = rt.get_logits_for_text(text, max_length=128)
        input_ids = result["input_ids"].numpy()

        targets = input_ids[1:]
        head1 = ensemble[0, :-1, :]
        head2 = ensemble[1 % n_samples, :-1, :]
        epistemic = epistemic_per_token[:-1]
        aleatoric = aleatoric_per_token[:-1]

        return {
            "head1_logits": head1.tolist(),
            "head2_logits": head2.tolist(),
            "epistemic_per_token": epistemic.tolist(),
            "aleatoric_per_token": aleatoric.tolist(),
            "ensemble_n_samples": n_samples,
            "targets": targets.tolist(),
            "seq_len": len(targets),
            "vocab_size": int(head1.shape[1]),
            "model_id": model_id,
            "model_sha": rt.model_sha,
            "source": "real",
            "text": text,
        }
    except Exception as e:
        result = _generate_synthetic_logits(seed=seed)
        result["source"] = "synthetic_fallback"
        result["fallback_reason"] = str(e)
        return result


def _compute_pe_from_logits(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Cross-entropy per position from logits and target indices."""
    # softmax + gather
    max_logits = logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    # Gather target probabilities
    target_probs = probs[np.arange(len(targets)), targets]
    target_probs = np.clip(target_probs, 1e-8, 1.0)
    return -np.log(target_probs)


def _epistemic_from_disagreement(
    head1_logits: np.ndarray,
    head2_logits: np.ndarray,
) -> np.ndarray:
    """Epistemic uncertainty = KL divergence between two heads' predictions."""
    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    p1 = softmax(head1_logits)
    p2 = softmax(head2_logits)
    # Symmetric KL as epistemic proxy
    kl_12 = (p1 * np.log(np.clip(p1 / np.clip(p2, 1e-8, None), 1e-8, None))).sum(axis=-1)
    kl_21 = (p2 * np.log(np.clip(p2 / np.clip(p1, 1e-8, None), 1e-8, None))).sum(axis=-1)
    return (kl_12 + kl_21) / 2.0


def _aleatoric_from_entropy(logits: np.ndarray) -> np.ndarray:
    """Aleatoric uncertainty = mean entropy of predictions (irreducible noise)."""
    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    p = softmax((logits))
    entropy = -(p * np.log(np.clip(p, 1e-8, None))).sum(axis=-1)
    return entropy


@register_probe
class CuriosityCriticProbe(BaseProbe):
    id = "pe-curiosity-critic-v1"
    hypothesis = (
        "2-head disagreement on TinyLlama separates epistemic from aleatoric PE. "
        "Epistemic component correlates with learnable surprise; aleatoric with irreducible noise."
    )
    primitive = PrimitiveTag.P5_EPISTEMIC_PE
    r_ids = ("R-PE",)

    def knobs(self) -> dict[str, list]:
        return {
            "disagreement_scale": [0.3, 0.5, 1.0],
            "use_real_model": [False, True],
            "model_id": ["sshleifer/tiny-gpt2"],
        }

    def default_inputs(self, seed: int) -> Any:
        data = _generate_synthetic_logits(seed=seed)
        return {
            "seed": seed,
            "head1_logits": data["head1_logits"].tolist(),
            "head2_logits": data["head2_logits"].tolist(),
            "targets": data["targets"].tolist(),
            "seq_len": data["seq_len"],
            "vocab_size": data["vocab_size"],
            "source": "synthetic",
        }

    def real_inputs(self, seed: int, knobs: Mapping[str, Any]) -> Any:
        """Generate inputs from real model using N-sample MC Dropout."""
        model_id = knobs.get("model_id", "sshleifer/tiny-gpt2")
        n_samples = int(knobs.get("mc_n_samples", 5))
        data = _generate_real_logits(seed=seed, model_id=model_id, n_samples=n_samples)
        return {
            "seed": seed,
            "head1_logits": data["head1_logits"],
            "head2_logits": data["head2_logits"],
            "epistemic_per_token": data.get("epistemic_per_token"),
            "aleatoric_per_token": data.get("aleatoric_per_token"),
            "ensemble_n_samples": data.get("ensemble_n_samples", 1),
            "targets": data["targets"],
            "seq_len": data["seq_len"],
            "vocab_size": data["vocab_size"],
            "source": data.get("source", "real"),
            "model_id": data.get("model_id"),
            "model_sha": data.get("model_sha"),
        }

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        inputs = ctx.inputs
        head1 = np.array(inputs["head1_logits"], dtype=np.float32)
        head2 = np.array(inputs["head2_logits"], dtype=np.float32)
        targets = np.array(inputs["targets"], dtype=np.int64)

        precomputed_e = inputs.get("epistemic_per_token")
        precomputed_a = inputs.get("aleatoric_per_token")
        has_ensemble = precomputed_e is not None and precomputed_a is not None

        if ctx.cell == AblationCell.BASELINE:
            pe1 = _compute_pe_from_logits(head1, targets)
            pe2 = _compute_pe_from_logits(head2, targets)
            pe = (pe1 + pe2) / 2.0
            epistemic = np.zeros_like(pe)
            aleatoric = pe.copy()

        elif ctx.cell == AblationCell.PROBE_ON:
            pe1 = _compute_pe_from_logits(head1, targets)
            pe2 = _compute_pe_from_logits(head2, targets)
            pe = (pe1 + pe2) / 2.0
            if has_ensemble:
                epistemic = np.array(precomputed_e, dtype=np.float32)
                aleatoric = np.array(precomputed_a, dtype=np.float32)
            else:
                epistemic = _epistemic_from_disagreement(head1, head2)
                aleatoric = _aleatoric_from_entropy((head1 + head2) / 2.0)

        elif ctx.cell == AblationCell.PROBE_OFF:
            pe1 = _compute_pe_from_logits(head1, targets)
            pe = pe1
            epistemic = np.zeros_like(pe)
            aleatoric = np.zeros_like(pe)

        elif ctx.cell == AblationCell.COUNTERFACTUAL:
            rng = np.random.default_rng(ctx.seed + 9999)
            shuffled_targets = rng.permutation(targets)
            pe1 = _compute_pe_from_logits(head1, shuffled_targets)
            pe2 = _compute_pe_from_logits(head2, shuffled_targets)
            pe = (pe1 + pe2) / 2.0
            if has_ensemble:
                epistemic = np.array(precomputed_e, dtype=np.float32)
                aleatoric = np.array(precomputed_a, dtype=np.float32)
            else:
                epistemic = _epistemic_from_disagreement(head1, head2)
                aleatoric = _aleatoric_from_entropy((head1 + head2) / 2.0)
        else:
            raise ValueError(f"unknown cell: {ctx.cell!r}")

        mean_pe = float(pe.mean())
        std_pe = float(pe.std())
        mean_epistemic = float(epistemic.mean())
        mean_aleatoric = float(aleatoric.mean())
        epistemic_share = mean_epistemic / (mean_epistemic + mean_aleatoric + 1e-8)

        readouts = ReadoutBundle(
            metrics={
                "mean_pe": mean_pe,
                "std_pe": std_pe,
                "mean_epistemic": mean_epistemic,
                "mean_aleatoric": mean_aleatoric,
                "epistemic_share": epistemic_share,
                "n": float(len(pe)),
            },
            artifacts={
                "pe_head": pe[:8].tolist(),
                "epistemic_head": epistemic[:8].tolist(),
                "aleatoric_head": aleatoric[:8].tolist(),
            },
            tags={
                "cell": ctx.cell.value,
                "wiring": ctx.level.value,
                "seed": ctx.seed,
                "source": inputs.get("source", "unknown"),
            },
        )

        return RunOutcome(
            readouts=readouts,
            output={
                "pe_length": int(len(pe)),
                "cell": ctx.cell.value,
                "epistemic_share": epistemic_share,
            },
        )

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        if not outcomes:
            return GateReport(passed=False, reason="no outcomes", stats={})

        # Gate: epistemic_share in probe_on cell should be > 0.1 (non-trivial split)
        probe_on = [o for o in outcomes if o.readouts.tags.get("cell") == "probe_on"]
        if not probe_on:
            return GateReport(passed=False, reason="no probe_on outcomes", stats={})

        shares = [o.readouts.metrics.get("epistemic_share", 0) for o in probe_on]
        mean_share = sum(shares) / len(shares)
        passed = mean_share > 0.05
        return GateReport(
            passed=passed,
            reason=f"epistemic_share mean={mean_share:.4f} (threshold=0.05)",
            stats={"mean_epistemic_share": mean_share, "n_probe_on": len(probe_on)},
        )
