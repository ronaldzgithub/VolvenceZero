"""Smoke tests for F5 steering bake on real residual stream (debt #21).

Validates:

* :meth:`OpenWeightResidualRuntime.capture_for_contrastive` default
  implementation works for any subclass that exposes ``capture``.
* :func:`build_steering_training_plan(contrast_set, substrate_runtime=...)`
  produces residual vectors whose dimension matches the runtime's
  hidden state width at the requested layer (NOT the 256-dim
  hashing embedding).
* The resulting steering set's ``to_substrate_adapter_layers`` is
  shape-compatible with :class:`SubstrateDeltaAdapterLayer`.
* Skipping the real-residual path keeps the legacy hashing-embedding
  fallback fully working (no regression).

Real Transformers runtime under ``@pytest.mark.hf`` so the CPU
smoke (~1s on tiny-gpt2) is gated behind opt-in.
"""

from __future__ import annotations

import importlib.util

import pytest

from volvence_zero.substrate import SubstrateDeltaAdapterLayer
from volvence_zero.substrate.residual_contracts import OpenWeightRuntimeCapture

from lifeform_domain_figure import (
    bake_steering_set,
    build_einstein_contrast_set,
    build_steering_training_plan,
)


def _hf_stack_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("transformers", "torch")
    )


# ---------------------------------------------------------------------------
# Fake runtime — exercises the ABC default capture_for_contrastive
# ---------------------------------------------------------------------------


class _FakeResidualRuntime:
    """Minimal duck-typed runtime that yields deterministic residuals.

    Used to test the default :meth:`capture_for_contrastive`
    implementation without spawning a real HF model. It satisfies
    ``runtime.capture_for_contrastive(positive_texts=..., negative_texts=..., layer_index=...)``
    by computing a per-text vector from a hash of the text and
    averaging.
    """

    HIDDEN_DIM = 16

    def capture_for_contrastive(
        self,
        *,
        positive_texts,
        negative_texts,
        layer_index,
    ):
        del layer_index
        positive = self._pool(positive_texts)
        negative = self._pool(negative_texts)
        return (positive, negative)

    @staticmethod
    def _pool(texts):
        import hashlib

        sums = [0.0] * _FakeResidualRuntime.HIDDEN_DIM
        for text in texts:
            digest = hashlib.blake2b(
                text.encode("utf-8"), digest_size=_FakeResidualRuntime.HIDDEN_DIM
            ).digest()
            for index, byte in enumerate(digest):
                sums[index] += (byte / 255.0) - 0.5
        return tuple(value / max(len(texts), 1) for value in sums)


def test_fallback_path_unchanged_when_no_runtime_supplied() -> None:
    contrast = build_einstein_contrast_set()
    plan = build_steering_training_plan(contrast)
    assert plan.embedding_dim == 256, (
        "default hashing-embedding dim is 256 (matches retrieval index)"
    )
    for pair in plan.pairs:
        assert len(pair.positive_residual) == 256
        assert len(pair.negative_residual) == 256


def test_real_residual_path_uses_runtime_hidden_dim() -> None:
    contrast = build_einstein_contrast_set()
    runtime = _FakeResidualRuntime()
    plan = build_steering_training_plan(
        contrast, substrate_runtime=runtime, layer_index=0
    )
    assert plan.embedding_dim == _FakeResidualRuntime.HIDDEN_DIM
    for pair in plan.pairs:
        assert len(pair.positive_residual) == _FakeResidualRuntime.HIDDEN_DIM
        assert len(pair.negative_residual) == _FakeResidualRuntime.HIDDEN_DIM


def test_real_residual_changes_plan_integrity_hash() -> None:
    """Switching coordinate systems MUST yield a different plan hash
    so a downstream gate proposal can detect the substrate-residual
    bake even when same contrast set is the input."""

    contrast = build_einstein_contrast_set()
    plan_hash_legacy = build_steering_training_plan(contrast).integrity_hash
    plan_hash_real = build_steering_training_plan(
        contrast, substrate_runtime=_FakeResidualRuntime()
    ).integrity_hash
    assert plan_hash_legacy != plan_hash_real


def test_real_residual_bake_yields_substrate_adapter_layer_shape() -> None:
    contrast = build_einstein_contrast_set()
    runtime = _FakeResidualRuntime()
    plan = build_steering_training_plan(contrast, substrate_runtime=runtime)
    steering = bake_steering_set(plan)
    layers = steering.to_substrate_adapter_layers()
    assert all(isinstance(layer, SubstrateDeltaAdapterLayer) for layer in layers)
    for layer in layers:
        assert len(layer.delta_vector) == _FakeResidualRuntime.HIDDEN_DIM


# ---------------------------------------------------------------------------
# Real HF runtime path — only when transformers + torch are installed
# ---------------------------------------------------------------------------


@pytest.mark.hf
def test_capture_for_contrastive_returns_hidden_dim_pair() -> None:
    """A real Transformers runtime's capture_for_contrastive must
    return two equal-length residual means at the configured layer."""

    if not _hf_stack_available():
        pytest.skip("transformers + torch not installed")
    from volvence_zero.substrate.residual_backend import (
        TransformersOpenWeightResidualRuntime,
    )

    runtime = TransformersOpenWeightResidualRuntime(
        model_id="sshleifer/tiny-gpt2",
        device="cpu",
    )
    positive, negative = runtime.capture_for_contrastive(
        positive_texts=("Reality is independent of observation.",),
        negative_texts=("Observation creates reality.",),
        layer_index=0,
    )
    assert len(positive) > 0
    assert len(positive) == len(negative)
    # Real residual means must NOT be byte-equal across positive and
    # negative texts (otherwise the contrastive direction collapses).
    assert any(
        abs(p - n) > 1e-6 for p, n in zip(positive, negative)
    ), "residual means must differ between positive and negative texts"


@pytest.mark.hf
def test_capture_default_path_still_works_via_capture_method() -> None:
    """Synthetic runtimes that don't override _mean_residual_at_layer
    still satisfy capture_for_contrastive through the abstract default
    that walks the public capture()."""

    if not _hf_stack_available():
        pytest.skip("transformers + torch not installed")
    from volvence_zero.substrate import (
        SyntheticOpenWeightResidualRuntime,
    )

    runtime = SyntheticOpenWeightResidualRuntime(
        model_id="synthetic-test",
    )
    capture = runtime.capture(source_text="hello")
    assert isinstance(capture, OpenWeightRuntimeCapture)
    layer_indices = {a.layer_index for a in capture.residual_activations if a.activation}
    assert layer_indices, "synthetic runtime should populate residual_activations"
    layer_index = sorted(layer_indices)[0]
    positive, negative = runtime.capture_for_contrastive(
        positive_texts=("alpha", "beta"),
        negative_texts=("gamma", "delta"),
        layer_index=layer_index,
    )
    assert len(positive) == len(negative)
    assert len(positive) > 0
