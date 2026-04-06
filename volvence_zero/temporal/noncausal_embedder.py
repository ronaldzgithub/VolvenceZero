"""Non-Causal Sequence Embedder — s(e_{1:T}) (ETA Appendix B.3).

This module implements the full-sequence bidirectional encoder used during
training-time posterior inference. It creates the key information asymmetry
in the metacontroller variational framework:

    - Training time: posterior q(z_t | e_{1:T}) conditioned on full sequence
    - Runtime: causal policy π(z_t | e_{1:t}) conditioned only on past

The non-causal embedder produces a fixed-size summary vector that enriches
the posterior mean/std, enabling tighter variational bounds during SSL.
At runtime, the causal encoder must match this posterior WITHOUT access
to future tokens — this is the core challenge that drives representation
learning.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.substrate import SubstrateSnapshot
from volvence_zero.temporal.metacontroller_components import (
    NdimSequenceEncoder,
    _project_to_ndim,
    _summarize_substrate_ndim,
)
from volvence_zero.temporal.tensor_ops import (
    Mat,
    Vec,
    ffn_2layer,
    gru_cell,
    init_ffn_params,
    init_gru_params,
    mat_vec,
    vec_add,
    vec_clamp,
    vec_mean,
    vec_mul,
    vec_norm,
    vec_scale,
    vec_sub,
    zeros,
)


@dataclass(frozen=True)
class NonCausalEmbedding:
    """Output of the bidirectional sequence embedder."""

    summary_vector: Vec
    forward_final: Vec
    backward_final: Vec
    sequence_length: int
    information_content: float
    description: str


@dataclass(frozen=True)
class PosteriorEnrichment:
    """Enriched posterior from combining causal encoder with non-causal embedding."""

    enriched_mean: Vec
    enriched_std: Vec
    kl_tightening: float
    description: str


class NonCausalSequenceEmbedder:
    """Bidirectional GRU encoder that processes the full sequence.

    Architecture:
        forward GRU:  e_1 → e_2 → ... → e_T  →  h_fwd
        backward GRU: e_T → e_{T-1} → ... → e_1  →  h_bwd
        summary = FFN(concat(h_fwd, h_bwd))

    The summary s(e_{1:T}) is only available during training. At runtime,
    the causal encoder must approximate the same posterior without it.
    """

    def __init__(self, *, n_z: int = 16, seed: int = 200) -> None:
        self._n_z = n_z
        fwd_params = init_gru_params(n_z, n_z, seed=seed)
        self._fwd_gru = _GRUParams(
            W_z=fwd_params["W_z"], U_z=fwd_params["U_z"], b_z=fwd_params["b_z"],
            W_r=fwd_params["W_r"], U_r=fwd_params["U_r"], b_r=fwd_params["b_r"],
            W_h=fwd_params["W_h"], U_h=fwd_params["U_h"], b_h=fwd_params["b_h"],
        )
        bwd_params = init_gru_params(n_z, n_z, seed=seed + 10)
        self._bwd_gru = _GRUParams(
            W_z=bwd_params["W_z"], U_z=bwd_params["U_z"], b_z=bwd_params["b_z"],
            W_r=bwd_params["W_r"], U_r=bwd_params["U_r"], b_r=bwd_params["b_r"],
            W_h=bwd_params["W_h"], U_h=bwd_params["U_h"], b_h=bwd_params["b_h"],
        )
        self._merge_ffn = init_ffn_params(n_z * 2, n_z, n_z, seed=seed + 20)

    @property
    def n_z(self) -> int:
        return self._n_z

    def embed(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
    ) -> NonCausalEmbedding:
        """Encode the full sequence bidirectionally."""
        step_vectors = _summarize_substrate_ndim(substrate_snapshot, self._n_z)
        n = self._n_z
        h_fwd = zeros(n)
        for vec in step_vectors:
            h_fwd = gru_cell(
                x=vec, h_prev=h_fwd,
                W_z=self._fwd_gru.W_z, U_z=self._fwd_gru.U_z, b_z=self._fwd_gru.b_z,
                W_r=self._fwd_gru.W_r, U_r=self._fwd_gru.U_r, b_r=self._fwd_gru.b_r,
                W_h=self._fwd_gru.W_h, U_h=self._fwd_gru.U_h, b_h=self._fwd_gru.b_h,
            )
        h_bwd = zeros(n)
        for vec in reversed(step_vectors):
            h_bwd = gru_cell(
                x=vec, h_prev=h_bwd,
                W_z=self._bwd_gru.W_z, U_z=self._bwd_gru.U_z, b_z=self._bwd_gru.b_z,
                W_r=self._bwd_gru.W_r, U_r=self._bwd_gru.U_r, b_r=self._bwd_gru.b_r,
                W_h=self._bwd_gru.W_h, U_h=self._bwd_gru.U_h, b_h=self._bwd_gru.b_h,
            )
        concat = h_fwd + h_bwd
        summary = ffn_2layer(
            x=concat,
            W1=self._merge_ffn["W1"], b1=self._merge_ffn["b1"],
            W2=self._merge_ffn["W2"], b2=self._merge_ffn["b2"],
        )
        info_content = vec_norm(summary) / max(self._n_z ** 0.5, 1.0)
        return NonCausalEmbedding(
            summary_vector=summary,
            forward_final=h_fwd,
            backward_final=h_bwd,
            sequence_length=len(step_vectors),
            information_content=min(info_content, 1.0),
            description=(
                f"non_causal_embed n_z={n} len={len(step_vectors)} "
                f"info={info_content:.3f} fwd_norm={vec_norm(h_fwd):.3f} "
                f"bwd_norm={vec_norm(h_bwd):.3f}"
            ),
        )

    def embed_from_steps(
        self,
        *,
        step_vectors: tuple[Vec, ...],
    ) -> NonCausalEmbedding:
        """Encode from pre-computed step vectors (for SSL training)."""
        n = self._n_z
        projected = tuple(_project_to_ndim(v, n) for v in step_vectors) if step_vectors else (zeros(n),)
        h_fwd = zeros(n)
        for vec in projected:
            h_fwd = gru_cell(
                x=vec, h_prev=h_fwd,
                W_z=self._fwd_gru.W_z, U_z=self._fwd_gru.U_z, b_z=self._fwd_gru.b_z,
                W_r=self._fwd_gru.W_r, U_r=self._fwd_gru.U_r, b_r=self._fwd_gru.b_r,
                W_h=self._fwd_gru.W_h, U_h=self._fwd_gru.U_h, b_h=self._fwd_gru.b_h,
            )
        h_bwd = zeros(n)
        for vec in reversed(projected):
            h_bwd = gru_cell(
                x=vec, h_prev=h_bwd,
                W_z=self._bwd_gru.W_z, U_z=self._bwd_gru.U_z, b_z=self._bwd_gru.b_z,
                W_r=self._bwd_gru.W_r, U_r=self._bwd_gru.U_r, b_r=self._bwd_gru.b_r,
                W_h=self._bwd_gru.W_h, U_h=self._bwd_gru.U_h, b_h=self._bwd_gru.b_h,
            )
        concat = h_fwd + h_bwd
        summary = ffn_2layer(
            x=concat,
            W1=self._merge_ffn["W1"], b1=self._merge_ffn["b1"],
            W2=self._merge_ffn["W2"], b2=self._merge_ffn["b2"],
        )
        info_content = vec_norm(summary) / max(n ** 0.5, 1.0)
        return NonCausalEmbedding(
            summary_vector=summary,
            forward_final=h_fwd,
            backward_final=h_bwd,
            sequence_length=len(projected),
            information_content=min(info_content, 1.0),
            description=(
                f"non_causal_embed n_z={n} len={len(projected)} "
                f"info={info_content:.3f}"
            ),
        )

    def enrich_posterior(
        self,
        *,
        causal_mean: Vec,
        causal_std: Vec,
        embedding: NonCausalEmbedding,
        blend_weight: float = 0.3,
    ) -> PosteriorEnrichment:
        """Combine causal encoder posterior with non-causal summary.

        The enriched posterior is tighter than the causal-only posterior,
        measured by kl_tightening (expected KL reduction).
        """
        n = min(len(causal_mean), self._n_z)
        summary = embedding.summary_vector[:n]
        enriched_mean = vec_clamp(
            vec_add(
                vec_scale(causal_mean[:n], 1.0 - blend_weight),
                vec_scale(summary, blend_weight),
            ),
            0.0, 1.0,
        )
        std_modulation = tuple(
            max(0.05, causal_std[i] * (1.0 - blend_weight * abs(summary[i])))
            for i in range(n)
        )
        causal_variance = sum(s ** 2 for s in causal_std[:n]) / max(n, 1)
        enriched_variance = sum(s ** 2 for s in std_modulation) / max(n, 1)
        kl_tightening = max(0.0, causal_variance - enriched_variance) / max(causal_variance, 1e-6)
        return PosteriorEnrichment(
            enriched_mean=enriched_mean,
            enriched_std=std_modulation,
            kl_tightening=kl_tightening,
            description=(
                f"posterior_enrichment blend={blend_weight:.2f} "
                f"kl_tightening={kl_tightening:.3f} "
                f"info={embedding.information_content:.3f}"
            ),
        )


@dataclass(frozen=True)
class _GRUParams:
    W_z: Mat
    U_z: Mat
    b_z: Vec
    W_r: Mat
    U_r: Mat
    b_r: Vec
    W_h: Mat
    U_h: Mat
    b_h: Vec
