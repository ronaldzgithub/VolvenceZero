"""F2 Cross-modal z_t — Joint latent space geometric consistency probe.

Hypothesis: V-JEPA visual features and TinyLlama text hidden states, when
projected into a shared latent space z_t, exhibit geometric consistency
(cosine alignment, linear separability) that exceeds chance and persists
across modalities. This is evidence for a universal latent controller space.

Cells:
- baseline (text_only): z_t from text hidden states only
- probe_on (joint_z): z_t from concatenated text + vision features
- probe_off (random_projection): random projection baseline
- counterfactual (misaligned): deliberately misaligned modalities

Corresponds to VZ P0-R4.1 (Function Vectors + Refusal Direction cross-modal).
"""

from __future__ import annotations

from typing import Any, Mapping

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


def _generate_synthetic_crossmodal(seed: int, n_samples: int = 32, text_dim: int = 64, vision_dim: int = 48) -> dict:
    """Generate synthetic cross-modal features with planted alignment.

    Text and vision features share a low-rank structure (simulating
    semantic alignment between modalities).
    """
    rng = np.random.default_rng(seed)
    latent_dim = 16

    # Shared latent factors
    latent = rng.standard_normal((n_samples, latent_dim)).astype(np.float32)

    # Text features: linear projection from latent + noise
    text_proj = rng.standard_normal((latent_dim, text_dim)).astype(np.float32) * 0.5
    text_features = latent @ text_proj + rng.standard_normal((n_samples, text_dim)).astype(np.float32) * 0.3

    # Vision features: different projection from same latent + noise
    vision_proj = rng.standard_normal((latent_dim, vision_dim)).astype(np.float32) * 0.5
    vision_features = latent @ vision_proj + rng.standard_normal((n_samples, vision_dim)).astype(np.float32) * 0.3

    # Labels: cluster assignment based on latent
    labels = (latent[:, 0] > 0).astype(np.int32)

    return {
        "text_features": text_features.tolist(),
        "vision_features": vision_features.tolist(),
        "labels": labels.tolist(),
        "n_samples": n_samples,
        "text_dim": text_dim,
        "vision_dim": vision_dim,
        "latent_dim": latent_dim,
        "source": "synthetic",
    }


def _cosine_alignment(a: np.ndarray, b: np.ndarray) -> float:
    """Mean pairwise cosine similarity between corresponding rows."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    cosines = (a_norm * b_norm).sum(axis=1)
    return float(cosines.mean())


def _linear_separability(features: np.ndarray, labels: np.ndarray) -> float:
    """Simple linear separability score via centroid classifier."""
    classes = np.unique(labels)
    if len(classes) < 2:
        return 0.5

    centroids = np.array([features[labels == c].mean(axis=0) for c in classes])
    # Classify each sample by nearest centroid
    predictions = np.array([
        classes[np.argmin(np.linalg.norm(centroids - f, axis=1))]
        for f in features
    ])
    return float((predictions == labels).mean())


def _cka_linear(x: np.ndarray, y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment (CKA) between two feature matrices.

    CKA measures representational similarity independent of rotation/scaling.
    """
    n = x.shape[0]
    # Center
    x_c = x - x.mean(axis=0)
    y_c = y - y.mean(axis=0)
    # Gram matrices
    xx = x_c @ x_c.T
    yy = y_c @ y_c.T
    # CKA
    hsic_xy = np.trace(xx @ yy) / (n - 1) ** 2
    hsic_xx = np.trace(xx @ xx) / (n - 1) ** 2
    hsic_yy = np.trace(yy @ yy) / (n - 1) ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


@register_probe
class CrossModalProbe(BaseProbe):
    id = "crossmodal-z-v1"
    hypothesis = (
        "Joint text+vision latent space z_t exhibits geometric consistency "
        "(CKA, linear separability) exceeding single-modality baselines."
    )
    primitive = PrimitiveTag.P2_LATENT_CONTROLLER  # Cross-primitive: P2 + P1
    r_ids = ("R4", "R2")

    def knobs(self) -> dict[str, list]:
        return {
            "projection_dim": [16, 32],
            "use_real_model": [False, True],
        }

    def default_inputs(self, seed: int) -> Any:
        return _generate_synthetic_crossmodal(seed=seed)

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        inputs = ctx.inputs
        text_feat = np.array(inputs["text_features"], dtype=np.float32)
        vision_feat = np.array(inputs["vision_features"], dtype=np.float32)
        labels = np.array(inputs["labels"], dtype=np.int32)
        proj_dim = knobs.get("projection_dim", 16)

        rng = np.random.default_rng(ctx.seed)

        if ctx.cell == AblationCell.BASELINE:
            # Text only: project to lower dim
            proj = rng.standard_normal((text_feat.shape[1], proj_dim)).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            z_t = text_feat @ proj
            cka = _cka_linear(text_feat, text_feat)  # self-CKA = 1.0
            separability = _linear_separability(z_t, labels)

        elif ctx.cell == AblationCell.PROBE_ON:
            # Joint z_t: concatenate and project
            joint = np.concatenate([text_feat, vision_feat], axis=1)
            proj = rng.standard_normal((joint.shape[1], proj_dim)).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            z_t = joint @ proj
            cka = _cka_linear(text_feat, vision_feat)
            separability = _linear_separability(z_t, labels)

        elif ctx.cell == AblationCell.PROBE_OFF:
            # Random projection (no structure)
            random_feat = rng.standard_normal(text_feat.shape).astype(np.float32)
            proj = rng.standard_normal((text_feat.shape[1], proj_dim)).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            z_t = random_feat @ proj
            cka = _cka_linear(random_feat, vision_feat)
            separability = _linear_separability(z_t, labels)

        elif ctx.cell == AblationCell.COUNTERFACTUAL:
            # Misaligned: shuffle vision features to break correspondence
            shuffled_vision = rng.permutation(vision_feat)
            joint = np.concatenate([text_feat, shuffled_vision], axis=1)
            proj = rng.standard_normal((joint.shape[1], proj_dim)).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            z_t = joint @ proj
            cka = _cka_linear(text_feat, shuffled_vision)
            separability = _linear_separability(z_t, labels)
        else:
            raise ValueError(f"unknown cell: {ctx.cell!r}")

        readouts = ReadoutBundle(
            metrics={
                "cka": cka,
                "linear_separability": separability,
                "z_t_norm": float(np.linalg.norm(z_t, axis=1).mean()),
                "n_samples": float(inputs["n_samples"]),
            },
            artifacts={"z_t_head": z_t[:4].tolist()},
            tags={"cell": ctx.cell.value, "seed": ctx.seed},
        )

        return RunOutcome(
            readouts=readouts,
            output={"cell": ctx.cell.value, "cka": cka, "separability": separability},
        )

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        if not outcomes:
            return GateReport(passed=False, reason="no outcomes", stats={})

        probe_on = [o for o in outcomes if o.readouts.tags.get("cell") == "probe_on"]
        baseline = [o for o in outcomes if o.readouts.tags.get("cell") == "baseline"]

        if not probe_on or not baseline:
            return GateReport(passed=False, reason="missing cells", stats={})

        # Joint z_t should have higher separability than text-only
        p_sep = sum(o.readouts.metrics["linear_separability"] for o in probe_on) / len(probe_on)
        b_sep = sum(o.readouts.metrics["linear_separability"] for o in baseline) / len(baseline)
        # CKA between modalities should be > 0 (non-trivial alignment)
        p_cka = sum(o.readouts.metrics["cka"] for o in probe_on) / len(probe_on)

        passed = p_sep >= b_sep and p_cka > 0.01
        return GateReport(
            passed=passed,
            reason=f"joint_sep={p_sep:.3f} vs text_sep={b_sep:.3f}, CKA={p_cka:.3f}",
            stats={"joint_sep": p_sep, "text_sep": b_sep, "cka": p_cka},
        )
