"""Smoke tests for the LoRAAwareResidualRuntime Protocol + hot-swap.

Closes debt #20: ``PersonaLoRAPool.activate(figure_id, runtime=...)``
must invoke the runtime's ``activate_lora`` for the lifetime of
the context, and must restore byte-identical forward behaviour
on exit.

Three layers of coverage:

1. **Protocol structural typing**: any runtime exposing
   ``activate_lora(layers) -> ContextManager`` satisfies the
   Protocol; the pool happily routes through it.
2. **ABC default no-op**: synthetic runtimes inherit the no-op
   ``activate_lora`` and pass the protocol check; the activate
   context still yields the registered record.
3. **Real-forward override**: a real
   :class:`TransformersOpenWeightResidualRuntime` adds the LoRA
   delta to the residual stream during the context and removes
   it on exit. Asserts the base ``state_dict`` hash stays
   byte-identical (R2: frozen base never mutates).
"""

from __future__ import annotations

import hashlib
import importlib.util

import pytest

from volvence_zero.substrate import (
    LoRAAwareResidualRuntime,
    PersonaLoRAPool,
    SubstrateDeltaAdapterLayer,
    SyntheticOpenWeightResidualRuntime,
)


def _hf_stack_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("transformers", "torch")
    )


def _persona_layers() -> tuple[SubstrateDeltaAdapterLayer, ...]:
    return (
        SubstrateDeltaAdapterLayer(
            layer_index=0,
            delta_vector=tuple(0.05 for _ in range(32)),
            mean_abs_delta=0.05,
            description="test-persona-layer-0",
        ),
        SubstrateDeltaAdapterLayer(
            layer_index=1,
            delta_vector=tuple(-0.02 for _ in range(32)),
            mean_abs_delta=0.02,
            description="test-persona-layer-1",
        ),
    )


def _register(pool: PersonaLoRAPool, *, figure_id: str = "test-figure") -> str:
    return pool.register(
        figure_id=figure_id,
        source_bundle_id="contract-bundle",
        backend_id="synthetic-v1",
        training_plan_hash="x" * 64,
        adapter_layers=_persona_layers(),
        parameter_count=64,
    )


# ---------------------------------------------------------------------------
# Protocol typing
# ---------------------------------------------------------------------------


def test_synthetic_runtime_satisfies_lora_aware_protocol() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-test")
    assert isinstance(runtime, LoRAAwareResidualRuntime)


def test_protocol_rejects_runtime_without_activate_lora() -> None:
    class _NoActivate:
        pass

    runtime = _NoActivate()
    assert not isinstance(runtime, LoRAAwareResidualRuntime)


# ---------------------------------------------------------------------------
# Pool activate path
# ---------------------------------------------------------------------------


def test_pool_activate_without_runtime_yields_record_unchanged() -> None:
    """Legacy passthrough: when runtime is None the activate context
    still yields the record (no real forward effect)."""

    pool = PersonaLoRAPool()
    record_id = _register(pool)
    with pool.activate("test-figure") as record:
        assert record.record_id == record_id
        assert record.adapter_layers == _persona_layers()


def test_pool_activate_with_synthetic_runtime_uses_default_noop() -> None:
    """Synthetic runtimes implement the ABC default activate_lora;
    nesting two activates raises so persona conflicts are loud."""

    pool = PersonaLoRAPool()
    _register(pool)
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-test")
    with pool.activate("test-figure", runtime=runtime) as record:
        assert record.figure_id == "test-figure"
        # Nesting must fail loud per Protocol contract.
        with pytest.raises(RuntimeError, match="nested activation"):
            with pool.activate("test-figure", runtime=runtime):
                pass


def test_pool_activate_rejects_non_lora_aware_runtime() -> None:
    pool = PersonaLoRAPool()
    _register(pool)

    class _Stub:
        pass

    with pytest.raises(TypeError, match="LoRAAwareResidualRuntime"):
        with pool.activate("test-figure", runtime=_Stub()):
            pass


# ---------------------------------------------------------------------------
# Real-forward effect on Transformers runtime (hf-only)
# ---------------------------------------------------------------------------


def _state_dict_hash(model) -> str:
    """Deterministic hash over a torch model's state_dict.

    Used to assert the frozen base is not mutated across an
    activate context. We hash tensor shapes + bytes so a single
    byte flip on any parameter would change the digest.
    """

    parts: list[str] = []
    for name, tensor in sorted(model.state_dict().items()):
        parts.append(name)
        parts.append(repr(tuple(tensor.shape)))
        parts.append(
            hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        )
    return hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()


def _aggressive_persona_layers(width: int = 64) -> tuple[SubstrateDeltaAdapterLayer, ...]:
    """Persona deltas with non-zero variance for tiny-gpt2 logit shift.

    GPT-2 blocks contain LayerNorm modules that subtract the mean
    of the residual stream; a **constant** delta gets absorbed by
    that normalisation and never propagates to logits. We shape
    the delta as alternating + / - values so the per-dim variance
    is non-zero and survives LayerNorm. The values are large
    (0.7) so the logit shift on tiny-gpt2 (hidden_dim=2) is
    robustly measurable.

    This is **not** a representative production persona — it's a
    forcing function so the hot-swap contract test can detect the
    forward-effect deterministically.
    """

    layer_0 = tuple(0.7 if index % 2 == 0 else -0.7 for index in range(width))
    layer_1 = tuple(-0.5 if index % 2 == 0 else 0.5 for index in range(width))
    return (
        SubstrateDeltaAdapterLayer(
            layer_index=0,
            delta_vector=layer_0,
            mean_abs_delta=0.7,
            description="aggressive-test-layer-0",
        ),
        SubstrateDeltaAdapterLayer(
            layer_index=1,
            delta_vector=layer_1,
            mean_abs_delta=0.5,
            description="aggressive-test-layer-1",
        ),
    )


@pytest.mark.hf
def test_transformers_runtime_activate_lora_changes_logits() -> None:
    """Activating a non-zero persona LoRA must shift the logits
    of the same source text; deactivating must restore the
    pre-activation logits byte-identically."""

    if not _hf_stack_available():
        pytest.skip("transformers + torch not installed")
    from volvence_zero.substrate.residual_backend import (
        TransformersOpenWeightResidualRuntime,
    )

    runtime = TransformersOpenWeightResidualRuntime(
        model_id="sshleifer/tiny-gpt2",
        device="cpu",
    )
    base_capture = runtime.capture(source_text="reality is")
    base_logits = base_capture.token_logits
    persona = _aggressive_persona_layers()
    pool = PersonaLoRAPool()
    pool.register(
        figure_id="test-figure-aggressive",
        source_bundle_id="bundle-real",
        backend_id="synthetic-v1",
        training_plan_hash="y" * 64,
        adapter_layers=persona,
        parameter_count=64,
    )
    with pool.activate("test-figure-aggressive", runtime=runtime):
        active_capture = runtime.capture(source_text="reality is")
    after_capture = runtime.capture(source_text="reality is")
    # Assert: activating shifted logits. Threshold is set well above
    # the FP32 noise floor but conservative enough to survive on
    # tiny-gpt2 (hidden_dim=2) where the per-logit delta is small.
    max_diff = max(
        abs(a - b)
        for a, b in zip(active_capture.token_logits, base_logits)
    )
    assert max_diff > 1e-9, (
        f"activate_lora should shift logits; max_diff={max_diff!r}"
    )
    # ...and deactivating restored them byte-identically.
    assert tuple(after_capture.token_logits) == tuple(base_logits)


@pytest.mark.hf
def test_transformers_runtime_activate_lora_does_not_mutate_base() -> None:
    """R2: persona LoRA activation must NOT mutate the frozen base
    model's state_dict. Hash before / inside / after the context;
    all three must match byte-identically."""

    if not _hf_stack_available():
        pytest.skip("transformers + torch not installed")
    from volvence_zero.substrate.residual_backend import (
        TransformersOpenWeightResidualRuntime,
    )

    runtime = TransformersOpenWeightResidualRuntime(
        model_id="sshleifer/tiny-gpt2",
        device="cpu",
    )
    base_hash_before = _state_dict_hash(runtime._model)  # noqa: SLF001 — test access
    persona = _persona_layers()
    pool = PersonaLoRAPool()
    pool.register(
        figure_id="test-figure",
        source_bundle_id="bundle-frozen",
        backend_id="synthetic-v1",
        training_plan_hash="z" * 64,
        adapter_layers=persona,
        parameter_count=64,
    )
    with pool.activate("test-figure", runtime=runtime):
        base_hash_during = _state_dict_hash(runtime._model)  # noqa: SLF001
        # Run a forward to ensure the hook fires while the base is checked.
        runtime.capture(source_text="frozen check")
    base_hash_after = _state_dict_hash(runtime._model)  # noqa: SLF001
    assert base_hash_before == base_hash_during == base_hash_after, (
        "frozen base state_dict drifted across activate_lora context"
    )
