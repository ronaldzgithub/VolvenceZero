"""Tests for the vLLM multi-LoRA router + runtime (no vllm dependency).

A fake engine + fake LoRARequest factory exercise per-request LoRA
routing, stable id assignment, the activation-context path, and the
fail-loud residual surface — all without importing vllm.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from volvence_zero.substrate import (
    VLLMLoRARouter,
    VLLMOpenWeightResidualRuntime,
)


@dataclass
class _FakeLoRARequest:
    name: str
    lora_id: int
    path: str


def _fake_factory(name, lora_id, path):
    return _FakeLoRARequest(name=name, lora_id=lora_id, path=path)


@dataclass
class _FakeCompletion:
    text: str
    token_ids: tuple[int, ...]


@dataclass
class _FakeOutput:
    outputs: list


class _FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def generate(self, prompts, sampling_params, lora_request=None):
        prompt = prompts[0]
        self.calls.append((prompt, lora_request))
        suffix = f"|lora={lora_request.lora_id}" if lora_request else "|base"
        return [_FakeOutput(outputs=[_FakeCompletion(text=prompt + suffix, token_ids=(1, 2, 3))])]


def _fake_sampling(max_tokens, temperature):
    return {"max_tokens": max_tokens, "temperature": temperature}


def _runtime() -> VLLMOpenWeightResidualRuntime:
    return VLLMOpenWeightResidualRuntime(
        model_id="fake/model",
        engine=_FakeEngine(),
        lora_request_factory=_fake_factory,
        sampling_params_factory=_fake_sampling,
    )


def test_router_assigns_stable_ids() -> None:
    router = VLLMLoRARouter(_fake_factory)
    r1 = router.request_for("/ck/a")
    r2 = router.request_for("/ck/b")
    r1_again = router.request_for("/ck/a")
    assert r1.lora_id == 1
    assert r2.lora_id == 2
    assert r1_again is r1
    assert router.resident_count == 2
    assert router.id_for("/ck/a") == 1
    assert router.id_for("/ck/missing") is None


def test_router_rejects_empty_path() -> None:
    router = VLLMLoRARouter(_fake_factory)
    with pytest.raises(ValueError):
        router.request_for("")


def test_generate_base_has_no_lora() -> None:
    runtime = _runtime()
    result = runtime.generate(prompt="hello")
    assert result.text == "hello|base"
    assert result.token_count == 3


def test_activate_peft_adapter_routes_lora() -> None:
    runtime = _runtime()
    with runtime.activate_peft_adapter("/ck/persona"):
        result = runtime.generate(prompt="hi")
    assert "lora=1" in result.text
    # After the context, generation is back to base.
    assert runtime.generate(prompt="hi").text == "hi|base"


def test_nested_activation_raises() -> None:
    runtime = _runtime()
    with runtime.activate_peft_adapter("/ck/a"):
        with pytest.raises(RuntimeError, match="nested activation"):
            with runtime.activate_peft_adapter("/ck/b"):
                pass


def test_generate_for_request_routes_per_call() -> None:
    runtime = _runtime()
    # Concurrent path: each call carries its own adapter without a
    # shared activation context.
    a = runtime.generate_for_request(prompt="a", lora_checkpoint_dir="/ck/a")
    b = runtime.generate_for_request(prompt="b", lora_checkpoint_dir="/ck/b")
    base = runtime.generate_for_request(prompt="c")
    assert "lora=1" in a.text
    assert "lora=2" in b.text
    assert base.text == "c|base"
    assert runtime.lora_router.resident_count == 2


def test_activate_lora_hook_path_unsupported() -> None:
    runtime = _runtime()
    with pytest.raises(NotImplementedError, match="activate_peft_adapter"):
        with runtime.activate_lora(()):
            pass


def test_capture_and_apply_control_fail_loud() -> None:
    runtime = _runtime()
    with pytest.raises(NotImplementedError, match="residual"):
        runtime.capture(source_text="x")
    with pytest.raises(NotImplementedError, match="residual"):
        runtime.apply_control()


def test_activation_is_task_local_for_concurrency() -> None:
    import asyncio

    runtime = _runtime()

    async def turn(figure_dir: str, expect_id: int) -> str:
        with runtime.activate_peft_adapter(figure_dir):
            # Yield control so both tasks interleave inside their
            # respective activation contexts.
            await asyncio.sleep(0)
            return runtime.generate(prompt="p").text

    async def main() -> list[str]:
        return await asyncio.gather(
            turn("/ck/a", 1),
            turn("/ck/b", 2),
        )

    results = asyncio.run(main())
    # Each task routed its OWN persona despite interleaving — no
    # cross-task clobber of the active checkpoint.
    assert any("lora=1" in r for r in results)
    assert any("lora=2" in r for r in results)
    assert all("base" not in r for r in results)
