"""MPS generation memory guards must not change CUDA behavior."""

from __future__ import annotations

from types import SimpleNamespace

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
