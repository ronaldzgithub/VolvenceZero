"""Epistemic / Aleatoric uncertainty separation operator.

Implements the Curiosity-Critic decomposition (Pathak 2024, 2604.18701):
    PE_total ≈ PE_epistemic + PE_aleatoric

where:
- Epistemic: reducible via more data/learning. Estimated by ensemble disagreement.
- Aleatoric: irreducible noise. Estimated by mean entropy of predictions.

This module is framework-level (no probe-specific assumptions) so other probes
(F1 BPC, P3 CPD, P5 PE-baseline) can reuse the same operator.

Usage:
    from volvence_labs.framework.runtime.uncertainty import (
        mc_dropout_logits,
        epistemic_aleatoric_split,
    )

    samples = mc_dropout_logits(rt, "Hello world", n_samples=5)  # (N, T, V)
    e, a = epistemic_aleatoric_split(samples)  # per-token epistemic / aleatoric

References:
- Kendall & Gal (2017) "What Uncertainties Do We Need in Bayesian Deep Learning?"
- Pathak et al. (2024) "Curiosity-Critic" arxiv 2604.18701
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def mc_dropout_logits(
    rt,
    text: str,
    n_samples: int = 5,
    max_length: int = 128,
    noise_scale: float = 0.0,
) -> np.ndarray:
    """Generate N forward passes with dropout enabled (or Gaussian noise fallback).

    Returns:
        ndarray of shape (n_samples, seq_len, vocab) — logits per sample.

    For models without trainable dropout (e.g. TinyLlama frozen), we add a small
    amount of Gaussian noise to the input embeddings to simulate stochastic forward
    passes. ``noise_scale`` controls this fallback amplitude.
    """
    import torch

    rt.load_model()
    model = rt.model
    tokenizer = rt.tokenizer

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(rt.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(rt.device)

    # Detect if model has any active dropout modules
    has_dropout = any(
        isinstance(m, torch.nn.Dropout) and m.p > 0
        for m in model.modules()
    )

    samples = []
    for i in range(n_samples):
        with torch.no_grad():
            if has_dropout and i > 0:
                # Enable dropout for samples >= 1
                model.train()
            else:
                model.eval()

            # Optional: input embedding noise (works even when dropout is absent)
            if not has_dropout and noise_scale > 0 and i > 0:
                # Inject noise via embedding layer hook
                embed = model.get_input_embeddings()
                input_embeds = embed(input_ids)
                noise = torch.randn_like(input_embeds) * noise_scale
                input_embeds = input_embeds + noise
                kwargs = {"inputs_embeds": input_embeds}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                out = model(**kwargs)
            else:
                kwargs = {"input_ids": input_ids}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                out = model(**kwargs)

            logits = out.logits[0].cpu().float().numpy()
            samples.append(logits)

    model.eval()
    return np.stack(samples, axis=0)  # (N, T, V)


def epistemic_aleatoric_split(
    sample_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose total predictive uncertainty into epistemic + aleatoric.

    Args:
        sample_logits: (N, T, V) where N = number of MC samples.

    Returns:
        epistemic: (T,) per-token epistemic uncertainty (mutual information).
        aleatoric: (T,) per-token aleatoric uncertainty (mean entropy).

    Decomposition (Kendall & Gal 2017):
        H[E[p]] = H_total = aleatoric + epistemic
        aleatoric = E[H[p]]   (mean of per-sample entropies)
        epistemic = H_total - aleatoric  (mutual information between params and y)
    """
    N, T, V = sample_logits.shape
    # Per-sample probabilities
    probs = softmax(sample_logits, axis=-1)  # (N, T, V)
    # Mean prediction across samples
    mean_probs = probs.mean(axis=0)  # (T, V)
    # Total predictive entropy
    h_total = -(mean_probs * np.log(np.clip(mean_probs, 1e-8, None))).sum(axis=-1)  # (T,)
    # Mean per-sample entropy = aleatoric
    h_per_sample = -(probs * np.log(np.clip(probs, 1e-8, None))).sum(axis=-1)  # (N, T)
    aleatoric = h_per_sample.mean(axis=0)  # (T,)
    # Epistemic = mutual information = H[E[p]] - E[H[p]]
    epistemic = h_total - aleatoric
    epistemic = np.maximum(epistemic, 0.0)  # numerical floor
    return epistemic, aleatoric


def cross_entropy_per_token(
    logits: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Per-token cross-entropy (PE) given logits and target indices.

    Args:
        logits: (T, V) logits.
        targets: (T,) integer target token ids.

    Returns:
        (T,) cross-entropy per position.
    """
    probs = softmax(logits, axis=-1)
    target_probs = probs[np.arange(len(targets)), targets]
    return -np.log(np.clip(target_probs, 1e-8, 1.0))
