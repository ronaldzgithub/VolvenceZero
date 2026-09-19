from __future__ import annotations

from types import SimpleNamespace

import pytest

from volvence_zero.substrate import TransformersOpenWeightResidualRuntime
from volvence_zero.substrate import residual_backend as residual_backend_module


class _FakeAutoModel:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object):
        cls.calls.append((model_id, kwargs))
        return object()


def _runtime_shell(*, device: str = "cuda") -> TransformersOpenWeightResidualRuntime:
    runtime = object.__new__(TransformersOpenWeightResidualRuntime)
    runtime._transformers = SimpleNamespace(AutoModelForCausalLM=_FakeAutoModel)
    runtime._torch = SimpleNamespace(float16="float16", bfloat16="bfloat16")
    runtime._device = device
    runtime._requested_model_dtype = None
    runtime._execution_profile = None
    return runtime


def test_model_loader_uses_accelerate_low_memory_device_map(monkeypatch):
    _FakeAutoModel.calls.clear()
    monkeypatch.setattr(residual_backend_module, "_accelerate_available", lambda: True)

    runtime = _runtime_shell()
    runtime._load_model(model_id="local-snapshot", local_files_only=True)

    _, kwargs = _FakeAutoModel.calls[-1]
    assert kwargs["local_files_only"] is True
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["device_map"] == "cuda"
    assert runtime._model_loaded_with_device_map is True


def test_model_loader_does_not_pass_accelerate_flags_when_optional_dependency_missing(
    monkeypatch,
):
    _FakeAutoModel.calls.clear()
    monkeypatch.setattr(residual_backend_module, "_accelerate_available", lambda: False)

    runtime = _runtime_shell(device="cpu")
    runtime._load_model(model_id="local-snapshot", local_files_only=True)

    _, kwargs = _FakeAutoModel.calls[-1]
    assert "low_cpu_mem_usage" not in kwargs
    assert "device_map" not in kwargs
    assert runtime._model_loaded_with_device_map is False


def test_strict_profile_fails_closed_without_accelerate(monkeypatch):
    monkeypatch.setattr(residual_backend_module, "_accelerate_available", lambda: False)
    runtime = _runtime_shell()
    runtime._execution_profile = object()

    with pytest.raises(RuntimeError, match="requires accelerate"):
        runtime._load_model(model_id="local-snapshot", local_files_only=True)


def test_windows_cuda_fails_closed_without_accelerate(monkeypatch):
    monkeypatch.setattr(residual_backend_module, "_accelerate_available", lambda: False)
    monkeypatch.setattr(residual_backend_module.os, "name", "nt")
    runtime = _runtime_shell(device="cuda")

    with pytest.raises(RuntimeError, match="requires accelerate"):
        runtime._load_model(model_id="local-snapshot", local_files_only=True)


def test_prepare_model_does_not_recopy_a_device_mapped_model():
    runtime = _runtime_shell()
    to_calls: list[str] = []
    runtime._model = SimpleNamespace(
        to=lambda device: to_calls.append(device),
        eval=lambda: None,
        parameters=lambda: (),
    )
    runtime._model_loaded_with_device_map = True

    runtime._prepare_model()

    assert to_calls == []
