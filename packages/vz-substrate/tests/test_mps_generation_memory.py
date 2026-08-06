"""MPS generation memory guards must not change CUDA behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from volvence_zero.substrate.residual_backend import (
    TransformersOpenWeightResidualRuntime,
)


class _FakeTensor:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = values
        self.device: str | None = None

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.values))

    def __getitem__(self, key: tuple[object, slice]) -> "_FakeTensor":
        _, token_slice = key
        return _FakeTensor(self.values[token_slice])

    def to(self, device: str) -> "_FakeTensor":
        self.device = device
        return self


class _FakeMPS:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.synchronize_calls = 0
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self.available

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _CaptureTensor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cpu_calls = 0

    def detach(self) -> "_CaptureTensor":
        return self

    def cpu(self) -> "_CaptureTensor":
        self.cpu_calls += 1
        return self


class _StackedCaptureTensor:
    def __init__(self, values: tuple[_CaptureTensor, ...]) -> None:
        self.values = values
        self.cpu_calls = 0

    def cpu(self) -> "_StackedCaptureTensor":
        self.cpu_calls += 1
        return self

    def __getitem__(self, index: int) -> _CaptureTensor:
        return self.values[index]


class _CaptureTorch:
    Tensor = _CaptureTensor

    def __init__(self) -> None:
        self.stack_calls = 0
        self.last_stacked: _StackedCaptureTensor | None = None

    def stack(
        self,
        values: tuple[_CaptureTensor, ...],
        *,
        dim: int,
    ) -> _StackedCaptureTensor:
        assert dim == 0
        self.stack_calls += 1
        self.last_stacked = _StackedCaptureTensor(values)
        return self.last_stacked


def _runtime(*, device: str, mps: _FakeMPS) -> TransformersOpenWeightResidualRuntime:
    runtime = object.__new__(TransformersOpenWeightResidualRuntime)
    runtime.model_id = "test-runtime"
    runtime._device = device
    runtime._mps_generation_max_input_tokens = 4
    runtime._torch = SimpleNamespace(Tensor=_FakeTensor, mps=mps)
    return runtime


def test_prepare_model_inputs_caps_only_mps_context() -> None:
    mps = _FakeMPS()
    encoded = {
        "input_ids": _FakeTensor((1, 2, 3, 4, 5, 6)),
        "attention_mask": _FakeTensor((1, 1, 1, 1, 1, 1)),
    }

    prepared = _runtime(device="mps", mps=mps)._prepare_model_inputs(encoded=encoded)

    assert prepared["input_ids"].values == (3, 4, 5, 6)
    assert prepared["attention_mask"].values == (1, 1, 1, 1)
    assert prepared["input_ids"].device == "mps"


def test_prepare_model_inputs_leaves_cuda_context_unchanged() -> None:
    mps = _FakeMPS()
    encoded = {
        "input_ids": _FakeTensor((1, 2, 3, 4, 5, 6)),
        "attention_mask": _FakeTensor((1, 1, 1, 1, 1, 1)),
    }

    prepared = _runtime(device="cuda", mps=mps)._prepare_model_inputs(encoded=encoded)

    assert prepared["input_ids"].values == (1, 2, 3, 4, 5, 6)
    assert prepared["attention_mask"].values == (1, 1, 1, 1, 1, 1)
    assert prepared["input_ids"].device == "cuda"


def test_release_generation_cache_is_mps_only() -> None:
    mps = _FakeMPS()
    _runtime(device="mps", mps=mps)._release_mps_generation_cache()

    assert mps.synchronize_calls == 1
    assert mps.empty_cache_calls == 1

    _runtime(device="cuda", mps=mps)._release_mps_generation_cache()

    assert mps.synchronize_calls == 1
    assert mps.empty_cache_calls == 1


def test_generation_hook_materializes_only_final_layer_capture() -> None:
    runtime = object.__new__(TransformersOpenWeightResidualRuntime)
    runtime.model_id = "test-runtime"
    capture_torch = _CaptureTorch()
    runtime._torch = capture_torch
    runtime._personal_conditioning_layer_gains = {1: 0.0}
    runtime._rare_heavy_adapter_deltas = {}
    runtime._online_fast_adapter_deltas = {}
    runtime._rare_heavy_adapter_scale = 0.0
    runtime._online_fast_delta_scale = 0.0
    captured: dict[int, object] = {}
    hook = runtime._make_capture_hook(
        layer_index=1,
        captured_layers=captured,
        control_delta=None,
        capture_residuals=True,
        defer_cpu_capture=True,
    )
    first = _CaptureTensor("first-token")
    final = _CaptureTensor("final-token")

    hook(None, (), first)
    hook(None, (), final)

    assert captured == {1: final}
    assert first.cpu_calls == 0
    assert final.cpu_calls == 0

    materialized = runtime._materialize_captured_layers(captured)

    assert materialized == {1: final}
    assert first.cpu_calls == 0
    assert final.cpu_calls == 0
    assert capture_torch.stack_calls == 1
    assert capture_torch.last_stacked is not None
    assert capture_torch.last_stacked.cpu_calls == 1


def test_generation_capture_extracts_only_first_raw_logit_step() -> None:
    torch = pytest.importorskip("torch")
    runtime = object.__new__(TransformersOpenWeightResidualRuntime)
    runtime.model_id = "test-runtime"
    runtime._torch = torch
    first = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    second = torch.full((2, 4), 99.0)

    extracted = runtime._extract_first_generation_logits(
        SimpleNamespace(logits=(first, second))
    )

    assert extracted.shape == (2, 1, 4)
    assert torch.equal(extracted[:, 0, :], first)


def test_generation_capture_rejects_missing_raw_logits() -> None:
    torch = pytest.importorskip("torch")
    runtime = object.__new__(TransformersOpenWeightResidualRuntime)
    runtime.model_id = "test-runtime"
    runtime._torch = torch

    with pytest.raises(TypeError, match="first-step raw logits"):
        runtime._extract_first_generation_logits(SimpleNamespace(logits=()))


def test_runtime_attests_actual_loaded_model_dtype() -> None:
    torch = pytest.importorskip("torch")
    runtime = object.__new__(TransformersOpenWeightResidualRuntime)
    runtime.model_id = "test-runtime"
    runtime._model = SimpleNamespace(dtype=torch.bfloat16)

    assert runtime.model_dtype == "bfloat16"


def test_residual_semantic_readout_stays_on_materialized_cpu_snapshot() -> None:
    torch = pytest.importorskip("torch")
    runtime = object.__new__(TransformersOpenWeightResidualRuntime)
    runtime._torch = torch
    runtime._device = "not-a-real-device"
    runtime._layer_indices = (1, 2)
    runtime._semantic_basis_cpu = torch.eye(4, dtype=torch.float32)
    captured = {
        1: torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]],
            dtype=torch.bfloat16,
        ),
        2: torch.tensor(
            [[[2.0, 1.0, 0.0, -1.0], [3.0, 2.0, 1.0, 0.0]]],
            dtype=torch.bfloat16,
        ),
    }

    profile = runtime._residual_semantic_profile(
        captured_layers=captured,
    )

    assert len(profile) == 4
    assert all(torch.isfinite(torch.tensor(profile)))
    assert sum(value * value for value in profile) == pytest.approx(1.0)
