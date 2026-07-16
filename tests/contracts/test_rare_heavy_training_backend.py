"""S1 contract tests: injectable rare-heavy adapter training backend.

Covers the substrate seam that replaces the heuristic-only rare-heavy
checkpoint path with a first-class real-training backend:

* request / result contracts fail loudly on empty or malformed payloads,
* an injected backend takes over ``train_rare_heavy`` (the built-in
  adapter-delta loop stays the uninjected fallback),
* ``clone_for_rare_heavy`` propagates the injected backend so the
  ``session_training_phase`` orchestration trains through it unchanged,
* backend failures propagate (no silent fallback to the heuristic), and
* Brain config wiring: ``rare_heavy_training_backend="peft-lora"`` on a
  non-transformers substrate fails loudly; the default leaves the
  synthetic runtime untouched (rollback target).

The real-PEFT end-to-end loop is exercised by the hf-gated test at the
bottom (skips cleanly without torch/peft/transformers or the tiny-gpt2
checkpoint).
"""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

from volvence_zero.brain import Brain, BrainConfig
from volvence_zero.substrate import (
    OpenWeightResidualStreamSubstrateAdapter,
    PeftLoraRareHeavyBackend,
    RARE_HEAVY_PEFT_TRAINING_MODE,
    RareHeavyAdapterTrainingBackend,
    RareHeavyAdapterTrainingResult,
    RareHeavyTrainingRequest,
    SubstrateDeltaAdapterLayer,
    SyntheticOpenWeightResidualRuntime,
    build_training_trace,
)


def _traces() -> tuple:
    return tuple(
        build_training_trace(trace_id=f"s1-trace-{index}", source_text=text)
        for index, text in enumerate(
            (
                "calm reflective collaboration",
                "steady repair before planning",
            )
        )
    )


def _substrate_batches(runtime: SyntheticOpenWeightResidualRuntime, traces) -> tuple:
    adapter = OpenWeightResidualStreamSubstrateAdapter(runtime=runtime)
    return tuple(
        (asyncio.run(adapter.capture(source_text=trace.source_text)),)
        for trace in traces
    )


class _FakeBackend:
    """Deterministic in-test backend implementing the training protocol."""

    backend_id = "fake-rare-heavy-v1"

    def __init__(self) -> None:
        self.requests: list[RareHeavyTrainingRequest] = []

    def train(self, request: RareHeavyTrainingRequest) -> RareHeavyAdapterTrainingResult:
        self.requests.append(request)
        layers = tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index,
                delta_vector=tuple(0.01 for _ in range(request.hidden_size)),
                mean_abs_delta=0.01,
                description=f"fake delta layer={layer_index}",
            )
            for layer_index in request.layer_indices
        )
        return RareHeavyAdapterTrainingResult(
            training_mode="fake-rare-heavy-v1",
            adapter_layers=layers,
            training_loss=0.25,
            initial_loss=1.0,
            steps_taken=4,
            description="fake backend trained.",
        )


class _ExplodingBackend:
    backend_id = "exploding"

    def train(self, request: RareHeavyTrainingRequest) -> RareHeavyAdapterTrainingResult:
        raise RuntimeError("backend exploded")


def test_request_and_result_contracts_fail_loudly() -> None:
    with pytest.raises(ValueError, match="layer_indices"):
        RareHeavyTrainingRequest(
            model_id="m", hidden_size=8, layer_indices=(), device="cpu", traces=()
        )
    with pytest.raises(ValueError, match="hidden_size"):
        RareHeavyTrainingRequest(
            model_id="m", hidden_size=0, layer_indices=(1,), device="cpu", traces=()
        )
    with pytest.raises(ValueError, match="adapter_layers"):
        RareHeavyAdapterTrainingResult(
            training_mode="x",
            adapter_layers=(),
            training_loss=0.0,
            initial_loss=0.0,
            steps_taken=0,
            description="",
        )


def test_fake_backend_satisfies_protocol() -> None:
    assert isinstance(_FakeBackend(), RareHeavyAdapterTrainingBackend)
    assert isinstance(PeftLoraRareHeavyBackend(), RareHeavyAdapterTrainingBackend)


def test_injected_backend_takes_over_synthetic_train_rare_heavy() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="s1-synthetic")
    backend = _FakeBackend()
    runtime.set_rare_heavy_training_backend(backend)
    traces = _traces()
    batches = _substrate_batches(runtime, traces)

    checkpoint = runtime.clone_for_rare_heavy().train_rare_heavy(
        traces=traces,
        substrate_steps_per_trace=batches,
    )

    assert checkpoint.training_mode == "fake-rare-heavy-v1"
    assert checkpoint.adapter_layers
    assert checkpoint.adapter_training_loss == pytest.approx(0.25)
    # The clone (not the original) received the delegated request —
    # proof that clone_for_rare_heavy propagates the injected backend.
    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.model_id == "s1-synthetic"
    assert request.hidden_size > 0
    assert request.layer_indices
    assert request.traces == traces

    # The delegated checkpoint round-trips through the normal import path.
    fresh = SyntheticOpenWeightResidualRuntime(
        model_id="s1-synthetic",
        allow_live_substrate_mutation=True,
    )
    operations = fresh.import_rare_heavy_state(checkpoint)
    assert "rare-heavy:substrate-import" in operations


def test_uninjected_runtime_keeps_builtin_adapter_delta_path() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="s1-synthetic-builtin")
    traces = _traces()
    batches = _substrate_batches(runtime, traces)

    checkpoint = runtime.clone_for_rare_heavy().train_rare_heavy(
        traces=traces,
        substrate_steps_per_trace=batches,
    )

    assert checkpoint.training_mode == "adapter-delta-v2"


def test_backend_failure_propagates_without_heuristic_fallback() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="s1-synthetic-explode")
    runtime.set_rare_heavy_training_backend(_ExplodingBackend())
    traces = _traces()
    batches = _substrate_batches(runtime, traces)

    with pytest.raises(RuntimeError, match="backend exploded"):
        runtime.clone_for_rare_heavy().train_rare_heavy(
            traces=traces,
            substrate_steps_per_trace=batches,
        )


def test_clearing_backend_restores_builtin_path() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="s1-synthetic-clear")
    runtime.set_rare_heavy_training_backend(_FakeBackend())
    runtime.set_rare_heavy_training_backend(None)
    traces = _traces()
    batches = _substrate_batches(runtime, traces)

    checkpoint = runtime.clone_for_rare_heavy().train_rare_heavy(
        traces=traces,
        substrate_steps_per_trace=batches,
    )

    assert checkpoint.training_mode == "adapter-delta-v2"


def test_peft_backend_config_validation_fails_loudly() -> None:
    with pytest.raises(ValueError, match="target_modules"):
        PeftLoraRareHeavyBackend(target_modules=())
    with pytest.raises(ValueError, match="rank/alpha"):
        PeftLoraRareHeavyBackend(rank=0)
    with pytest.raises(ValueError, match="max_steps"):
        PeftLoraRareHeavyBackend(max_steps=0)


def test_brain_config_peft_lora_requires_transformers_runtime() -> None:
    brain = Brain(BrainConfig(rare_heavy_training_backend="peft-lora"))
    with pytest.raises(ValueError, match="peft-lora"):
        brain.create_session(session_id="s1-misconfigured")


def test_brain_default_leaves_synthetic_runtime_uninjected() -> None:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="s1-brain-default")
    brain = Brain(
        BrainConfig(substrate_mode="injected"),
        substrate_runtime=runtime,
    )
    brain.create_session(session_id="s1-default")
    assert runtime._rare_heavy_training_backend is None


def _tiny_gpt2_cached() -> bool:
    """True when tiny-gpt2 is fully present in the local HF cache."""

    import transformers

    try:
        transformers.AutoTokenizer.from_pretrained(
            "sshleifer/tiny-gpt2", local_files_only=True
        )
        transformers.AutoModelForCausalLM.from_pretrained(
            "sshleifer/tiny-gpt2", local_files_only=True
        )
    except OSError:
        return False
    return True


@pytest.mark.hf
def test_peft_lora_backend_trains_real_adapter_on_tiny_gpt2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not all(
        importlib.util.find_spec(name) is not None
        for name in ("transformers", "torch", "peft")
    ):
        pytest.skip("transformers + torch + peft not installed")
    if not _tiny_gpt2_cached():
        pytest.skip("sshleifer/tiny-gpt2 not in local HF cache")
    # The cache guard above proved the checkpoint is fully local; force
    # offline so a flaky hub (e.g. 504) can't fail a purely-local test.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    backend = PeftLoraRareHeavyBackend(max_steps=2, rank=2, alpha=4)
    request = RareHeavyTrainingRequest(
        model_id="sshleifer/tiny-gpt2",
        hidden_size=8,
        layer_indices=(0, 1),
        device="cpu",
        traces=_traces(),
    )

    result = backend.train(request)

    assert result.training_mode == RARE_HEAVY_PEFT_TRAINING_MODE
    assert result.steps_taken == 2
    assert len(result.adapter_layers) == 2
    for layer in result.adapter_layers:
        assert len(layer.delta_vector) == 8
    # Determinism of the projection: the same trained weights always
    # produce non-empty finite vectors.
    assert all(
        all(value == value for value in layer.delta_vector)  # no NaN
        for layer in result.adapter_layers
    )


@pytest.mark.hf
def test_peft_lora_backend_rejects_empty_traces() -> None:
    if not all(
        importlib.util.find_spec(name) is not None
        for name in ("transformers", "torch", "peft")
    ):
        pytest.skip("transformers + torch + peft not installed")
    backend = PeftLoraRareHeavyBackend()
    request = RareHeavyTrainingRequest(
        model_id="sshleifer/tiny-gpt2",
        hidden_size=8,
        layer_indices=(0,),
        device="cpu",
        traces=(),
    )
    with pytest.raises(ValueError, match="no non-empty trace texts"):
        backend.train(request)
