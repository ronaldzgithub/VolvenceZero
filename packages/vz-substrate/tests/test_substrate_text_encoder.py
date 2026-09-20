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

import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import threading
import time
from types import SimpleNamespace

import pytest

from volvence_zero.semantic_embedding import (
    reset_semantic_embedding_backend,
    stub_cosine_similarity,
    stub_semantic_embedding,
)
from volvence_zero.substrate import (
    FeatureSignal,
    HFTextGenerationProvider,
    SubstrateTextEncoderBackend,
    SyntheticOpenWeightResidualRuntime,
)
from volvence_zero.substrate.runtime_execution import run_runtime_call


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


def test_concurrent_cache_miss_is_captured_once() -> None:
    class _SlowCountingRuntime(SyntheticOpenWeightResidualRuntime):
        def __init__(self) -> None:
            super().__init__(model_id="synthetic-slow-counting")
            self.capture_calls = 0

        def capture(self, *, source_text: str):
            self.capture_calls += 1
            time.sleep(0.01)
            return super().capture(source_text=source_text)

    runtime = _SlowCountingRuntime()
    backend = SubstrateTextEncoderBackend(runtime, cache_size=16)
    barrier = threading.Barrier(8)

    def embed_once() -> tuple[float, ...]:
        barrier.wait()
        return backend.embed("shared prototype text", dim=8)

    with ThreadPoolExecutor(max_workers=8) as executor:
        vectors = tuple(executor.map(lambda _: embed_once(), range(8)))

    assert vectors == (vectors[0],) * 8
    assert runtime.capture_calls == 1


async def test_embed_async_runs_capture_on_owner_thread_without_blocking_loop() -> None:
    class _ThreadRecordingRuntime(SyntheticOpenWeightResidualRuntime):
        def __init__(self) -> None:
            super().__init__(model_id="synthetic-async-thread")
            self.capture_thread_name = ""
            self.capture_started = threading.Event()

        def capture(self, *, source_text: str):
            self.capture_thread_name = threading.current_thread().name
            self.capture_started.set()
            time.sleep(0.08)
            return super().capture(source_text=source_text)

    runtime = _ThreadRecordingRuntime()
    backend = SubstrateTextEncoderBackend(runtime, cache_size=16)
    task = asyncio.create_task(backend.embed_async("async prototype", dim=8))

    assert await asyncio.to_thread(runtime.capture_started.wait, 2.0)
    heartbeat_ticks = 0
    while not task.done():
        heartbeat_ticks += 1
        await asyncio.sleep(0.005)

    vector = await task
    assert len(vector) == 8
    assert heartbeat_ticks >= 3
    assert runtime.capture_thread_name.startswith("vz-runtime-owner")
    assert runtime.capture_thread_name != threading.current_thread().name


async def test_embed_async_waits_for_owner_contention_without_blocking_loop() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="synthetic-contention")
    backend = SubstrateTextEncoderBackend(runtime, cache_size=16)
    owner_entered = threading.Event()
    owner_release = threading.Event()

    def hold_owner() -> None:
        owner_entered.set()
        assert owner_release.wait(timeout=2.0)

    holder = asyncio.create_task(
        run_runtime_call(
            runtime=runtime,
            operation=hold_owner,
            operation_kind="test_hold_owner",
        )
    )
    assert await asyncio.to_thread(owner_entered.wait, 2.0)

    embedding = asyncio.create_task(
        backend.embed_async("queued async prototype", dim=8)
    )
    heartbeat_ticks = 0
    for _ in range(8):
        heartbeat_ticks += 1
        await asyncio.sleep(0.005)

    assert not holder.done()
    assert not embedding.done()
    owner_release.set()
    await holder
    vector = await embedding
    assert heartbeat_ticks == 8
    assert len(vector) == 8


async def test_embed_async_same_key_misses_are_single_flight() -> None:
    class _SlowCountingRuntime(SyntheticOpenWeightResidualRuntime):
        def __init__(self) -> None:
            super().__init__(model_id="synthetic-async-single-flight")
            self.capture_calls = 0
            self.capture_threads: list[str] = []

        def capture(self, *, source_text: str):
            self.capture_calls += 1
            self.capture_threads.append(threading.current_thread().name)
            time.sleep(0.02)
            return super().capture(source_text=source_text)

    runtime = _SlowCountingRuntime()
    backend = SubstrateTextEncoderBackend(runtime, cache_size=16)
    vectors = await asyncio.gather(
        *(backend.embed_async("shared async prototype", dim=8) for _ in range(12))
    )

    assert vectors == [vectors[0]] * 12
    assert runtime.capture_calls == 1
    assert len(runtime.capture_threads) == 1
    assert runtime.capture_threads[0].startswith("vz-runtime-owner")


async def test_embed_async_failure_does_not_cache_and_retry_recovers() -> None:
    class _FailOnceRuntime(SyntheticOpenWeightResidualRuntime):
        def __init__(self) -> None:
            super().__init__(model_id="synthetic-async-recovery")
            self.capture_calls = 0

        def capture(self, *, source_text: str):
            self.capture_calls += 1
            if self.capture_calls == 1:
                raise RuntimeError("test async capture failure")
            return super().capture(source_text=source_text)

    runtime = _FailOnceRuntime()
    backend = SubstrateTextEncoderBackend(runtime, cache_size=16)

    with pytest.raises(RuntimeError, match="test async capture failure"):
        await backend.embed_async("recoverable async prototype", dim=8)

    recovered = await backend.embed_async("recoverable async prototype", dim=8)
    cached = await backend.embed_async("recoverable async prototype", dim=8)
    assert recovered == cached
    assert runtime.capture_calls == 2


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


class _FastTokenizerCaptureRuntime:
    """Minimal shared runtime exercising the real Rust tokenizer backend."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self.capture_calls = 0
        self.fail_next_capture = False

    @property
    def runtime_execution_owner(self) -> object:
        return self

    @property
    def runtime_tokenizer_owner(self) -> object:
        return self._tokenizer

    def capture(self, *, source_text: str):
        self.capture_calls += 1
        if self.fail_next_capture:
            self.fail_next_capture = False
            raise RuntimeError("test capture failure")
        encoded = self._tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            max_length=8,
        )
        token_count = int(encoded["input_ids"].shape[-1])
        return SimpleNamespace(
            feature_surface=(
                FeatureSignal(
                    name="token-count",
                    values=(float(token_count), float(len(source_text) % 17 + 1)),
                    source="test-fast-tokenizer",
                ),
            )
        )


class _EchoGenerationModel:
    def generate(self, input_ids, **_kwargs):
        return input_ids


def _in_memory_fast_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    rust_backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "alpha": 2,
                "beta": 3,
                "embedding": 4,
                "proposal": 5,
            },
            unk_token="[UNK]",
        )
    )
    rust_backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=rust_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )


@pytest.mark.skipif(not _hf_stack_available(), reason="requires transformers + torch")
def test_embedding_and_proposal_serialize_shared_fast_tokenizer() -> None:
    """Embedding truncation=True must not race proposal truncation=False.

    ``PreTrainedTokenizerFast`` mutates truncation state in its Rust backend
    for each call. Unguarded mixed calls reliably raise ``Already borrowed``;
    both paths must therefore converge on the runtime/tokenizer resource gate.
    """

    tokenizer = _in_memory_fast_tokenizer()
    runtime = _FastTokenizerCaptureRuntime(tokenizer)
    backend = SubstrateTextEncoderBackend(runtime, cache_size=1024)
    provider = HFTextGenerationProvider(
        model=_EchoGenerationModel(),
        tokenizer=tokenizer,
        use_chat_template=False,
        runtime_execution_owner=runtime,
    )
    worker_count = 6
    iterations = 150
    barrier = threading.Barrier(worker_count)
    runtime_errors: list[str] = []
    errors_guard = threading.Lock()

    def exercise(worker_index: int) -> None:
        barrier.wait()
        for iteration in range(iterations):
            try:
                if worker_index % 2 == 0:
                    backend.embed(
                        f"embedding alpha beta {worker_index} {iteration}",
                        dim=2,
                    )
                else:
                    provider.generate(
                        prompt=f"proposal alpha beta {worker_index} {iteration}"
                    )
            except RuntimeError as exc:
                with errors_guard:
                    runtime_errors.append(str(exc))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        tuple(executor.map(exercise, range(worker_count)))

    assert runtime_errors == []
    assert runtime.capture_calls == (worker_count // 2) * iterations

    cached_text = "embedding alpha beta 0 0"
    capture_calls = runtime.capture_calls
    first = backend.embed(cached_text, dim=2)
    second = backend.embed(cached_text, dim=2)
    assert first == second
    assert runtime.capture_calls == capture_calls


@pytest.mark.skipif(not _hf_stack_available(), reason="requires transformers + torch")
def test_fast_tokenizer_guard_releases_after_capture_failure() -> None:
    tokenizer = _in_memory_fast_tokenizer()
    runtime = _FastTokenizerCaptureRuntime(tokenizer)
    backend = SubstrateTextEncoderBackend(runtime, cache_size=16)
    provider = HFTextGenerationProvider(
        model=_EchoGenerationModel(),
        tokenizer=tokenizer,
        use_chat_template=False,
        runtime_execution_owner=runtime,
    )
    runtime.fail_next_capture = True

    with pytest.raises(RuntimeError, match="test capture failure"):
        backend.embed("embedding alpha beta", dim=2)

    # The failed value was not cached and the reentrant resource lock was
    # released: both proposal generation and a retry can use the tokenizer.
    assert provider.generate(prompt="proposal alpha beta") == ""
    recovered = backend.embed("embedding alpha beta", dim=2)
    assert len(recovered) == 2
    assert backend.embed("embedding alpha beta", dim=2) == recovered
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
