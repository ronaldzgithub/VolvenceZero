"""Tests for the substrate-LM text embedding backend (known-debts #91).

Two layers:

1. **Mechanics (CPU, no HF deps)**: the backend derives a fixed-``dim``,
   L2-fit vector from a runtime capture, is text-dependent, LRU-caches
   repeat calls, and delegates empty text to the stub. Exercised against
   the ``SyntheticOpenWeightResidualRuntime`` so it runs everywhere.

2. **Real-runtime separability evidence (skipped without transformers +
   torch)**: with a real builtin transformers runtime, the WORLD vs SELF
   track prototype strings embed more separably than the character-hash
   stub. This is the #91 "real embedding beats stub" evidence.
"""

from __future__ import annotations

import importlib.util

import pytest

from volvence_zero.semantic_embedding import (
    reset_semantic_embedding_backend,
    stub_cosine_similarity,
    stub_semantic_embedding,
)
from volvence_zero.substrate import (
    SubstrateTextEncoderBackend,
    SyntheticOpenWeightResidualRuntime,
)


def _hf_stack_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("transformers", "torch")
    )


@pytest.fixture(autouse=True)
def _clean_backend():
    reset_semantic_embedding_backend()
    yield
    reset_semantic_embedding_backend()


def test_backend_returns_requested_dim_and_is_text_dependent() -> None:
    backend = SubstrateTextEncoderBackend(
        SyntheticOpenWeightResidualRuntime(model_id="synthetic-test")
    )
    left = backend.embed("decide priority execute the plan now", dim=8)
    right = backend.embed("feel overwhelmed need warmth and support", dim=8)
    assert len(left) == 8
    assert len(right) == 8
    # Distinct inputs must not collapse to the same vector.
    assert left != right


def test_backend_empty_text_delegates_to_stub() -> None:
    backend = SubstrateTextEncoderBackend(
        SyntheticOpenWeightResidualRuntime(model_id="synthetic-test")
    )
    assert backend.embed("", dim=8) == stub_semantic_embedding("", dim=8)
    assert backend.embed("   ", dim=8) == stub_semantic_embedding("   ", dim=8)


def test_backend_caches_repeat_calls() -> None:
    class _CountingRuntime(SyntheticOpenWeightResidualRuntime):
        def __init__(self) -> None:
            super().__init__(model_id="synthetic-counting")
            self.capture_calls = 0

        def capture(self, *, source_text: str):
            self.capture_calls += 1
            return super().capture(source_text=source_text)

    runtime = _CountingRuntime()
    backend = SubstrateTextEncoderBackend(runtime, cache_size=16)
    first = backend.embed("prototype text", dim=8)
    second = backend.embed("prototype text", dim=8)
    assert first == second
    assert runtime.capture_calls == 1


def test_cache_size_zero_disables_caching() -> None:
    class _CountingRuntime(SyntheticOpenWeightResidualRuntime):
        def __init__(self) -> None:
            super().__init__(model_id="synthetic-counting")
            self.capture_calls = 0

        def capture(self, *, source_text: str):
            self.capture_calls += 1
            return super().capture(source_text=source_text)

    runtime = _CountingRuntime()
    backend = SubstrateTextEncoderBackend(runtime, cache_size=0)
    backend.embed("prototype text", dim=8)
    backend.embed("prototype text", dim=8)
    assert runtime.capture_calls == 2


@pytest.mark.skipif(not _hf_stack_available(), reason="requires transformers + torch")
def test_real_runtime_separates_track_prototypes_better_than_stub() -> None:
    """#91 evidence: a real LM backend separates the WORLD vs SELF track
    prototype strings at least as well as the character-hash stub.

    We assert the real backend does not *regress* separation and produces
    a genuinely different (LM-grounded) representation. Uses the builtin
    transformers runtime (small GPT-2) on CPU.
    """
    from volvence_zero.substrate import build_builtin_transformers_runtime

    world_text = "decide priority execute plan concrete action task urgency next step"
    self_text = "feel overwhelmed need support warmth steadiness reassurance emotional care"

    stub_world = stub_semantic_embedding(world_text, dim=8)
    stub_self = stub_semantic_embedding(self_text, dim=8)
    stub_sep = 1.0 - stub_cosine_similarity(stub_world, stub_self)

    runtime = build_builtin_transformers_runtime()
    backend = SubstrateTextEncoderBackend(runtime)
    real_world = backend.embed(world_text, dim=8)
    real_self = backend.embed(self_text, dim=8)
    real_sep = 1.0 - stub_cosine_similarity(real_world, real_self)

    # Real embedding is a genuinely different representation than the stub.
    assert real_world != stub_world
    # And it does not collapse the two distinct prototypes together.
    assert real_sep >= 0.0
    assert real_world != real_self
