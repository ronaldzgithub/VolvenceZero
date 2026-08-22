from __future__ import annotations

import contextlib
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from volvence_zero.substrate import (
    GenerationContextBudgetAttestation,
    GenerationResult,
    LocalSubstrateRuntimeMode,
    SubstrateFallbackMode,
    OpenWeightRuntimeCapture,
    TransformersExecutionAttestation,
    TransformersOpenWeightResidualRuntime,
    WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
    build_transformers_runtime_with_fallback,
    fingerprint_transformers_execution_assets,
)
from volvence_zero.substrate import residual_backend as residual_backend_module
from volvence_zero.substrate import (
    steered_action_scoring as steered_action_scoring_module,
)


_SHA = "a" * 64
_ASSETS_SHA = "c" * 64
_REVISION = "b" * 40


def _canonical_snapshot_path(
    tmp_path,
    *,
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    revision: str = _REVISION,
):
    snapshot = (
        tmp_path
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    return snapshot


def _attestation() -> TransformersExecutionAttestation:
    profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    return TransformersExecutionAttestation(
        profile_id=profile.profile_id,
        preset_name=profile.preset_name,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_revision=_REVISION,
        model_weights_sha256=_SHA,
        execution_assets_sha256=_ASSETS_SHA,
        runtime_origin="hf-local",
        platform_system="Windows",
        platform_release="11",
        device="cuda:0",
        device_name="NVIDIA GeForce RTX 4090",
        python_version="3.11.15",
        torch_version="2.12.0+cu126",
        transformers_version="5.9.0",
        cuda_version="12.6",
        cudnn_version=91002,
        device_compute_capability=(8, 9),
        attention_implementation="sdpa",
        sdpa_backend="cudnn",
        sdpa_backend_policy="exclusive-cudnn",
        sdpa_backend_exclusive=True,
        generation_use_cache=True,
        require_generation_chat_template=True,
        generation_capture_strategy="first-full-prompt-set-once",
        capture_failure_mode="raise",
        context_window_tokens=32768,
        local_files_only=True,
        fallback_mode="deny",
        fail_on_truncation=True,
        model_dtype="bfloat16",
        hidden_size=1536,
        model_max_position_embeddings=32768,
        hook_layer_indices=(14,),
    )


def _runtime_shell() -> TransformersOpenWeightResidualRuntime:
    return object.__new__(TransformersOpenWeightResidualRuntime)


def _patch_strict_windows_cuda_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(residual_backend_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        TransformersOpenWeightResidualRuntime,
        "_resolve_device",
        lambda self, *, device: "cuda",
    )


def test_strict_profile_and_attestation_are_frozen_and_content_addressed() -> None:
    profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    attestation = _attestation()

    assert len(profile.profile_id) == 64
    assert profile.require_generation_chat_template is True
    assert attestation.require_generation_chat_template is True
    assert len(attestation.attestation_id) == 64
    assert attestation.attestation_id == _attestation().attestation_id
    assert "pretrained_source" not in attestation.to_payload()
    assert "resolved_model_snapshot" not in attestation.to_payload()
    with pytest.raises(FrozenInstanceError):
        profile.generation_use_cache = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        attestation.device = "cpu"  # type: ignore[misc]


def test_generation_result_keeps_legacy_attestation_fields_empty() -> None:
    result = GenerationResult(
        text="legacy",
        token_count=1,
        capture=None,
        description="legacy profile",
    )

    assert result.execution_attestation_id == ""
    assert result.context_budget is None


def _context_budget(
    *, execution_attestation_id: str = _SHA
) -> GenerationContextBudgetAttestation:
    return GenerationContextBudgetAttestation(
        execution_attestation_id=execution_attestation_id,
        input_mode="chat-template",
        input_token_count=3,
        prefix_slot_count=0,
        effective_max_new_tokens=2,
        combined_token_count=5,
        context_window_tokens=32768,
        remaining_token_count=32763,
    )


def test_profile_and_attestation_reject_type_and_cross_field_attacks() -> None:
    profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    attestation = _attestation()

    with pytest.raises(TypeError, match="exact bool"):
        replace(profile, generation_use_cache=1)
    with pytest.raises(ValueError, match="profile_id is not canonical"):
        replace(attestation, profile_id="d" * 64)
    with pytest.raises(ValueError, match="profile facts drifted"):
        replace(attestation, generation_use_cache=False)
    with pytest.raises(ValueError, match="model_revision"):
        replace(attestation, model_revision=f"snapshots/{_REVISION}")


@pytest.mark.parametrize(
    "hook_layers",
    (
        [7],
        (7, 7),
        (8, 7),
        (False,),
    ),
)
def test_attestation_rejects_noncanonical_hook_layers(hook_layers) -> None:
    with pytest.raises((TypeError, ValueError), match="hook layers"):
        replace(_attestation(), hook_layer_indices=hook_layers)


def test_context_budget_rejects_bool_token_count() -> None:
    with pytest.raises(ValueError, match="token counts"):
        replace(_context_budget(), input_token_count=True)


def test_context_budget_rejects_plain_tokenizer_claim() -> None:
    with pytest.raises(ValueError, match="input_mode"):
        replace(_context_budget(), input_mode="plain-tokenizer")


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("input_token_count", 0, "must be positive"),
        ("effective_max_new_tokens", 0, "must be positive"),
        ("context_window_tokens", 32767, "canonical 32768"),
    ),
)
def test_context_budget_v1_rejects_noncanonical_positive_budget(
    field_name: str,
    value: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_context_budget(), **{field_name: value})


def test_generation_result_requires_bidirectional_matching_lineage() -> None:
    budget = _context_budget()

    with pytest.raises(ValueError, match="present or absent together"):
        GenerationResult(
            text="strict",
            token_count=1,
            capture=None,
            description="missing context",
            execution_attestation_id=_SHA,
        )
    with pytest.raises(ValueError, match="present or absent together"):
        GenerationResult(
            text="strict",
            token_count=1,
            capture=None,
            description="missing id",
            context_budget=budget,
        )
    with pytest.raises(ValueError, match="lineage mismatch"):
        GenerationResult(
            text="strict",
            token_count=1,
            capture=None,
            description="mismatched id",
            execution_attestation_id="d" * 64,
            context_budget=budget,
        )


def test_combined_context_budget_accepts_limit_and_rejects_plus_one() -> None:
    torch = pytest.importorskip("torch")
    runtime = _runtime_shell()
    runtime._execution_profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    runtime._execution_attestation = _attestation()
    prefix_pairs = [
        (
            torch.zeros((1, 1, 4, 1)),
            torch.zeros((1, 1, 4, 1)),
        )
    ]

    at_limit = runtime._build_generation_context_budget(
        input_token_count=32760,
        prefix_pairs=prefix_pairs,
        effective_max_new_tokens=4,
        input_mode="chat-template",
    )

    assert isinstance(at_limit, GenerationContextBudgetAttestation)
    assert at_limit.combined_token_count == 32768
    assert at_limit.remaining_token_count == 0
    with pytest.raises(ValueError, match="context budget exceeded"):
        runtime._build_generation_context_budget(
            input_token_count=32761,
            prefix_pairs=prefix_pairs,
            effective_max_new_tokens=4,
            input_mode="chat-template",
        )


class _StrictChatTokenizer:
    def __init__(self, *, fail_tokenize: bool = False) -> None:
        self.calls: list[bool] = []
        self.messages: list[tuple[tuple[str, str], ...]] = []
        self.fail_tokenize = fail_tokenize

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        del kwargs
        self.calls.append(bool(tokenize))
        self.messages.append(
            tuple((item["role"], item["content"]) for item in messages)
        )
        if tokenize:
            if self.fail_tokenize:
                raise ValueError("template tokenize failure")
            torch = pytest.importorskip("torch")
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }
        return "<|system|>strict<|assistant|>"

    def __call__(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("strict chat path bypassed apply_chat_template")


class _SizedChatTokenizer:
    eos_token_id = 0

    def __init__(self, *, token_count: int) -> None:
        self.token_count = token_count
        self.messages: list[tuple[tuple[str, str], ...]] = []

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        del kwargs
        self.messages.append(
            tuple((item["role"], item["content"]) for item in messages)
        )
        if not tokenize:
            return "<|user|>hello<|assistant|>"
        torch = pytest.importorskip("torch")
        return {
            "input_ids": torch.ones((1, self.token_count), dtype=torch.long),
            "attention_mask": torch.ones(
                (1, self.token_count), dtype=torch.long
            ),
        }


class _HookHandle:
    def __init__(self, hooks: list[object], hook: object) -> None:
        self._hooks = hooks
        self._hook = hook

    def remove(self) -> None:
        self._hooks.remove(self._hook)


class _FakeBlock:
    def __init__(self) -> None:
        self.hooks: list[object] = []
        self.registration_count = 0

    def register_forward_hook(self, hook):
        self.registration_count += 1
        self.hooks.append(hook)
        return _HookHandle(self.hooks, hook)

    def fire(self, hidden):
        adjusted = hidden
        for hook in tuple(self.hooks):
            replacement = hook(self, (), adjusted)
            if replacement is not None:
                adjusted = replacement
        return adjusted


class _FakeGenerateModel:
    def __init__(self, *, block: _FakeBlock) -> None:
        self.block = block
        self.generate_count = 0
        self.generate_kwargs: dict[str, object] = {}

    def generate(self, **kwargs):
        torch = pytest.importorskip("torch")
        self.generate_count += 1
        self.generate_kwargs = dict(kwargs)
        input_ids = kwargs["input_ids"]
        self.block.fire(torch.zeros((1, int(input_ids.shape[-1]), 2)))
        self.block.fire(torch.zeros((1, 1, 2)))
        sequences = torch.cat(
            (input_ids, torch.tensor([[7]], dtype=input_ids.dtype)), dim=-1
        )
        return SimpleNamespace(
            sequences=sequences,
            logits=(torch.zeros((1, 1, 8)),),
        )


def _generation_runtime(
    *,
    strict: bool,
    input_token_count: int,
) -> tuple[TransformersOpenWeightResidualRuntime, _FakeBlock, _FakeGenerateModel, list[int]]:
    torch = pytest.importorskip("torch")
    runtime = _runtime_shell()
    block = _FakeBlock()
    model = _FakeGenerateModel(block=block)
    captured_lengths: list[int] = []
    runtime._execution_profile = (
        WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1 if strict else None
    )
    runtime._execution_attestation = _attestation() if strict else None
    runtime._torch = torch
    runtime._tokenizer = _SizedChatTokenizer(token_count=input_token_count)
    runtime._model = model
    runtime._device = "cpu"
    runtime._mps_generation_max_input_tokens = 32768
    runtime.model_id = "test-model"
    runtime._loaded_base_model_weights_sha256 = _SHA
    runtime._layer_indices = (0,)
    runtime._block_modules = (block,)
    runtime._hidden_size = 2
    runtime._character_prefix_registry = None
    runtime._character_prefix_pairs = None
    runtime._character_prefix_package = None
    runtime._character_residual_deltas = {}
    runtime._character_residual_adapter_id = ""
    runtime._online_fast_adapter_deltas = {}
    runtime._rare_heavy_adapter_deltas = {}
    runtime._personal_conditioning_layer_gains = {}
    runtime._adapter_delta_for_layer = lambda *, layer_index: None
    runtime._extract_hidden_tensor = lambda *, output: output
    runtime._build_personal_conditioning_delta = lambda *, conditioning: None
    runtime._decode_generated_text = lambda *, token_ids: "generated"
    runtime._strict_sdpa_context = lambda: contextlib.nullcontext()

    def build_capture(**kwargs):
        captured_lengths.append(
            int(next(iter(kwargs["captured_layers"].values())).shape[-2])
        )
        return OpenWeightRuntimeCapture(
            token_logits=(1.0,),
            feature_surface=(),
            residual_activations=(),
            residual_sequence=(),
            description="fake generation capture",
        )

    runtime._build_runtime_capture = build_capture
    return runtime, block, model, captured_lengths


def test_strict_chat_messages_must_use_template_tokenize_true() -> None:
    torch = pytest.importorskip("torch")
    runtime = _runtime_shell()
    tokenizer = _StrictChatTokenizer()
    runtime._execution_profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    runtime._tokenizer = tokenizer
    runtime._torch = torch
    runtime._device = "cpu"
    runtime._mps_generation_max_input_tokens = 1024

    rendered, inputs = runtime._build_generation_inputs(
        prompt="ignored",
        system_context="ignored",
        chat_messages=(("system", "strict"), ("user", "hello")),
    )

    assert rendered == "<|system|>strict<|assistant|>"
    assert tokenizer.calls == [False, True]
    assert tuple(inputs["input_ids"].shape) == (1, 3)


def test_strict_chat_template_failure_cannot_fall_back_to_role_text() -> None:
    torch = pytest.importorskip("torch")
    runtime = _runtime_shell()
    runtime._execution_profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    runtime._tokenizer = _StrictChatTokenizer(fail_tokenize=True)
    runtime._torch = torch
    runtime._device = "cpu"
    runtime._mps_generation_max_input_tokens = 1024

    with pytest.raises(RuntimeError, match="chat-template tokenization failed"):
        runtime._build_generation_inputs(
            prompt="ignored",
            system_context="ignored",
            chat_messages=(("user", "hello"),),
        )


def test_strict_plain_prompt_is_normalized_through_chat_template() -> None:
    torch = pytest.importorskip("torch")
    runtime = _runtime_shell()
    tokenizer = _StrictChatTokenizer()
    runtime._execution_profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    runtime._tokenizer = tokenizer
    runtime._torch = torch
    runtime._device = "cpu"
    runtime._mps_generation_max_input_tokens = 1024

    rendered, _ = runtime._build_generation_inputs(
        prompt="hello",
        system_context="strict system",
        chat_messages=(),
    )

    assert rendered == "<|system|>strict<|assistant|>"
    assert tokenizer.calls == [False, True]
    assert tokenizer.messages == [
        (("system", "strict system"), ("user", "hello")),
        (("system", "strict system"), ("user", "hello")),
    ]


def test_strict_generate_rejects_32769_before_hook_or_model_forward() -> None:
    runtime, block, model, _ = _generation_runtime(
        strict=True,
        input_token_count=32768,
    )

    with pytest.raises(ValueError, match="context budget exceeded"):
        runtime.generate(
            prompt="ignored",
            chat_messages=(("user", "hello"),),
            max_new_tokens=1,
            temperature=0.0,
        )

    assert block.registration_count == 0
    assert model.generate_count == 0


def test_successful_strict_generate_publishes_execution_and_budget_lineage() -> None:
    runtime, _, model, captured_lengths = _generation_runtime(
        strict=True,
        input_token_count=3,
    )

    result = runtime.generate(
        prompt="ignored",
        chat_messages=(("user", "hello"),),
        max_new_tokens=2,
        temperature=0.0,
    )

    assert runtime.execution_attestation is runtime._execution_attestation
    assert result.execution_attestation_id == runtime.execution_attestation.attestation_id
    assert result.context_budget is not None
    assert (
        result.context_budget.execution_attestation_id
        == runtime.execution_attestation.attestation_id
    )
    assert result.context_budget.input_token_count == 3
    assert result.context_budget.effective_max_new_tokens == 2
    assert result.context_budget.combined_token_count == 5
    assert model.generate_kwargs["use_cache"] is True
    assert captured_lengths == [3]


def test_strict_plain_generate_attests_chat_template_input_mode() -> None:
    runtime, _, _, _ = _generation_runtime(
        strict=True,
        input_token_count=3,
    )

    result = runtime.generate(
        prompt="hello",
        system_context="strict system",
        max_new_tokens=2,
        temperature=0.0,
    )

    assert result.context_budget is not None
    assert result.context_budget.input_mode == "chat-template"
    assert runtime._tokenizer.messages == [
        (("system", "strict system"), ("user", "hello")),
        (("system", "strict system"), ("user", "hello")),
    ]


def test_legacy_windows_generate_keeps_no_cache_and_latest_step_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _, model, captured_lengths = _generation_runtime(
        strict=False,
        input_token_count=3,
    )
    monkeypatch.setattr(
        residual_backend_module,
        "os",
        SimpleNamespace(name="nt"),
    )

    result = runtime.generate(
        prompt="ignored",
        chat_messages=(("user", "hello"),),
        max_new_tokens=2,
        temperature=0.0,
    )

    assert model.generate_kwargs["use_cache"] is False
    assert captured_lengths == [1]
    assert result.execution_attestation_id == ""
    assert result.context_budget is None


def test_cached_hook_captures_full_prompt_once_but_applies_delta_every_step() -> None:
    torch = pytest.importorskip("torch")
    runtime = _runtime_shell()
    runtime._torch = torch
    runtime._personal_conditioning_layer_gains = {}
    runtime._adapter_delta_for_layer = lambda *, layer_index: None
    runtime._extract_hidden_tensor = lambda *, output: output
    captured: dict[int, object] = {}
    fires: dict[int, int] = {}
    hook = runtime._make_capture_hook(
        layer_index=3,
        captured_layers=captured,
        control_delta=torch.ones(2),
        capture_first_full_prompt=True,
        hook_fire_counts=fires,
    )

    prompt_output = hook(None, (), torch.zeros((1, 3, 2)))
    decode_one = hook(None, (), torch.zeros((1, 1, 2)))
    decode_two = hook(None, (), torch.zeros((1, 1, 2)))

    assert tuple(captured[3].shape) == (1, 3, 2)
    assert torch.equal(captured[3], torch.ones((1, 3, 2)))
    assert torch.equal(prompt_output, torch.ones((1, 3, 2)))
    assert torch.equal(decode_one, torch.ones((1, 1, 2)))
    assert torch.equal(decode_two, torch.ones((1, 1, 2)))
    assert fires == {3: 3}


def test_strict_capture_failure_is_raised_while_legacy_returns_none() -> None:
    runtime = _runtime_shell()
    runtime.model_id = "test-model"
    runtime._device = "cuda"
    runtime._build_runtime_capture = lambda **kwargs: (_ for _ in ()).throw(
        ValueError("bad capture")
    )
    runtime._execution_profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1

    with pytest.raises(RuntimeError, match="strict transformers residual capture"):
        runtime._finalize_generation_capture(
            source_text="x",
            input_ids=object(),
            logits=object(),
            captured_layers={0: object()},
            control_applied=False,
        )

    runtime._execution_profile = None
    assert (
        runtime._finalize_generation_capture(
            source_text="x",
            input_ids=object(),
            logits=object(),
            captured_layers={0: object()},
            control_applied=False,
        )
        is None
    )


def test_strict_sdpa_context_selects_singleton_exclusive_cudnn() -> None:
    calls: list[tuple[object, bool]] = []
    cudnn_backend = object()

    class _Attention:
        SDPBackend = SimpleNamespace(CUDNN_ATTENTION=cudnn_backend)

        @staticmethod
        def sdpa_kernel(backend, *, set_priority):
            calls.append((backend, set_priority))
            return contextlib.nullcontext()

    runtime = _runtime_shell()
    runtime._execution_profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    runtime._torch = SimpleNamespace(
        nn=SimpleNamespace(attention=_Attention()),
    )

    with runtime._strict_sdpa_context():
        pass

    assert calls == [(cudnn_backend, True)]


def test_runtime_scorer_receives_strict_exclusive_sdpa_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, bool]] = []
    captured: dict[str, object] = {}
    cudnn_backend = object()
    scorer_sentinel = object()

    class _Attention:
        SDPBackend = SimpleNamespace(CUDNN_ATTENTION=cudnn_backend)

        @staticmethod
        def sdpa_kernel(backend, *, set_priority):
            calls.append((backend, set_priority))
            return contextlib.nullcontext()

    def build_scorer(**kwargs):
        captured.update(kwargs)
        return scorer_sentinel

    monkeypatch.setattr(
        steered_action_scoring_module,
        "TransformersSteeredActionScorer",
        build_scorer,
    )
    runtime = _runtime_shell()
    runtime._execution_profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    runtime._torch = SimpleNamespace(
        nn=SimpleNamespace(attention=_Attention()),
    )
    runtime._model = object()
    runtime._tokenizer = object()
    runtime._block_modules = (object(),)
    runtime._layer_indices = (0,)
    runtime._hidden_size = 8
    runtime._device = "cuda"
    runtime._fail_on_truncation = True
    runtime.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    runtime._resolve_final_norm_module = lambda: object()

    scorer = runtime.build_steered_action_scorer(
        action_options=(object(), object()),
    )
    context_factory = captured["forward_context_factory"]
    with context_factory():
        pass

    assert scorer is scorer_sentinel
    assert captured["fail_on_truncation"] is True
    assert calls == [(cudnn_backend, True)]


def test_strict_capture_bypasses_windows_pool_and_wraps_forward_in_sdpa() -> None:
    torch = pytest.importorskip("torch")
    runtime = _runtime_shell()
    block = _FakeBlock()
    sentinel = OpenWeightRuntimeCapture(
        token_logits=(1.0,),
        feature_surface=(),
        residual_activations=(),
        residual_sequence=(),
        description="strict capture",
    )
    events: list[str] = []

    @contextlib.contextmanager
    def strict_context():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def make_capture_hook(**kwargs):
        def hook(module, args, output):
            del module, args
            kwargs["captured_layers"][0] = output
            return None

        return hook

    class _CaptureModel:
        def __call__(self, **kwargs):
            del kwargs
            assert events == ["enter"]
            hidden = torch.zeros((1, 2, 2))
            block.fire(hidden)
            return SimpleNamespace(logits=torch.zeros((1, 2, 8)))

    runtime._execution_profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    runtime._torch = torch
    runtime._tokenize = lambda *, source_text: {
        "input_ids": torch.tensor([[1, 2]])
    }
    runtime._layer_indices = (0,)
    runtime._block_modules = (block,)
    runtime._make_capture_hook = make_capture_hook
    runtime._model = _CaptureModel()
    runtime._strict_sdpa_context = strict_context
    runtime._materialize_captured_layers = lambda captured: captured
    runtime._extract_logits = lambda *, outputs: outputs.logits
    runtime._build_runtime_capture = lambda **kwargs: sentinel
    runtime._capture_pooled_summary = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("strict capture used Windows pooled fallback")
    )

    assert runtime.capture(source_text="strict") is sentinel
    assert events == ["enter", "exit"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"local_files_only": False}, "requires STRICT_LOCAL"),
        ({"fallback_mode": SubstrateFallbackMode.ALLOW_BUILTIN}, "fallback DENY"),
        ({"expected_model_weights_sha256": "bad"}, "weights SHA-256"),
        ({"verified_model_revision": "snapshots/bad"}, "model revision"),
        ({"expected_execution_assets_sha256": "bad"}, "execution-assets"),
        ({"max_length": 32767}, "arguments drifted"),
    ),
)
def test_strict_factory_rejects_non_strict_inputs(
    overrides: dict[str, object],
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "device": "cuda",
        "local_files_only": True,
        "runtime_mode": LocalSubstrateRuntimeMode.STRICT_LOCAL,
        "fallback_mode": SubstrateFallbackMode.DENY,
        "max_length": 32768,
        "fail_on_truncation": True,
        "model_dtype": "bfloat16",
        "expected_model_weights_sha256": _SHA,
        "verified_model_revision": _REVISION,
        "expected_execution_assets_sha256": _ASSETS_SHA,
        "execution_profile": WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_transformers_runtime_with_fallback(**kwargs)


def test_strict_factory_rejects_actual_weight_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    snapshot_root = _canonical_snapshot_path(tmp_path)
    monkeypatch.setattr(
        residual_backend_module,
        "fingerprint_model_weight_files",
        lambda path: "c" * 64,
    )

    with pytest.raises(ValueError, match="weight digest does not match"):
        build_transformers_runtime_with_fallback(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            model_source=str(snapshot_root),
            device="cuda",
            local_files_only=True,
            runtime_mode=LocalSubstrateRuntimeMode.STRICT_LOCAL,
            fallback_mode=SubstrateFallbackMode.DENY,
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            expected_model_weights_sha256=_SHA,
            verified_model_revision=_REVISION,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        )


def test_execution_asset_fingerprint_binds_all_nonweight_snapshot_files(
    tmp_path,
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        '{"chat_template":"v1"}', encoding="utf-8"
    )
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text('{"weight_map":{}}', encoding="utf-8")
    custom_code = tmp_path / "modeling_custom.py"
    custom_code.write_text("VERSION = 1\n", encoding="utf-8")
    weight_path = tmp_path / "model.safetensors"
    weight_path.write_bytes(b"weight-v1")

    first = fingerprint_transformers_execution_assets(tmp_path)
    index_path.write_text('{"weight_map":{"a":"shard"}}', encoding="utf-8")
    second = fingerprint_transformers_execution_assets(tmp_path)
    custom_code.write_text("VERSION = 2\n", encoding="utf-8")
    third = fingerprint_transformers_execution_assets(tmp_path)
    weight_path.write_bytes(b"weight-v2")
    fourth = fingerprint_transformers_execution_assets(tmp_path)

    assert len(first) == 64
    assert first != second
    assert second != third
    assert third == fourth


def test_strict_factory_passes_verified_revision_to_snapshot_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    hub = pytest.importorskip("huggingface_hub")
    snapshot_root = (
        tmp_path
        / "models--Qwen--Qwen2.5-1.5B-Instruct"
        / "snapshots"
        / _REVISION
    )
    snapshot_root.mkdir(parents=True)
    download_calls: list[dict[str, object]] = []
    runtime_calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs):
        download_calls.append(dict(kwargs))
        return str(snapshot_root)

    def build_runtime(**kwargs):
        runtime_calls.append(dict(kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(
        residual_backend_module,
        "fingerprint_model_weight_files",
        lambda path: _SHA,
    )
    monkeypatch.setattr(
        residual_backend_module,
        "fingerprint_transformers_execution_assets",
        lambda path: _ASSETS_SHA,
    )
    monkeypatch.setattr(
        residual_backend_module,
        "TransformersOpenWeightResidualRuntime",
        build_runtime,
    )

    build_transformers_runtime_with_fallback(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        device="cuda",
        local_files_only=True,
        runtime_mode=LocalSubstrateRuntimeMode.STRICT_LOCAL,
        fallback_mode=SubstrateFallbackMode.DENY,
        max_length=32768,
        fail_on_truncation=True,
        model_dtype="bfloat16",
        expected_model_weights_sha256=_SHA,
        verified_model_revision=_REVISION,
        expected_execution_assets_sha256=_ASSETS_SHA,
        execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
    )

    assert download_calls == [
        {
            "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "local_files_only": True,
            "revision": _REVISION,
        }
    ]
    assert runtime_calls[0]["verified_model_revision"] == _REVISION
    assert (
        runtime_calls[0]["expected_execution_assets_sha256"]
        == _ASSETS_SHA
    )


def test_strict_factory_rejects_logical_source_alias() -> None:
    with pytest.raises(ValueError, match="different from model_id"):
        build_transformers_runtime_with_fallback(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            model_source="other/repository",
            device="cuda",
            local_files_only=True,
            runtime_mode=LocalSubstrateRuntimeMode.STRICT_LOCAL,
            fallback_mode=SubstrateFallbackMode.DENY,
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            expected_model_weights_sha256=_SHA,
            verified_model_revision=_REVISION,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        )


def test_strict_local_copy_requires_explicit_verified_revision(tmp_path) -> None:
    with pytest.raises(ValueError, match="copied local snapshot"):
        build_transformers_runtime_with_fallback(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            model_source=str(tmp_path),
            device="cuda",
            local_files_only=True,
            runtime_mode=LocalSubstrateRuntimeMode.STRICT_LOCAL,
            fallback_mode=SubstrateFallbackMode.DENY,
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            expected_model_weights_sha256=_SHA,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        )


def test_strict_factory_rejects_arbitrary_local_copy_with_claimed_revision(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="cannot be bound"):
        build_transformers_runtime_with_fallback(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            model_source=str(tmp_path),
            device="cuda",
            local_files_only=True,
            runtime_mode=LocalSubstrateRuntimeMode.STRICT_LOCAL,
            fallback_mode=SubstrateFallbackMode.DENY,
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            expected_model_weights_sha256=_SHA,
            verified_model_revision=_REVISION,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        )


def test_strict_constructor_rejects_arbitrary_local_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_strict_windows_cuda_host(monkeypatch)

    with pytest.raises(ValueError, match="cannot be bound"):
        TransformersOpenWeightResidualRuntime(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            pretrained_source=str(tmp_path),
            device="cuda",
            local_files_only=True,
            runtime_origin="hf-local",
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            loaded_base_model_weights_sha256=_SHA,
            verified_model_revision=_REVISION,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        )


def test_strict_constructor_rejects_snapshot_repo_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_strict_windows_cuda_host(monkeypatch)
    snapshot_root = _canonical_snapshot_path(tmp_path)

    with pytest.raises(ValueError, match="cannot be bound"):
        TransformersOpenWeightResidualRuntime(
            model_id="other/repository",
            pretrained_source=str(snapshot_root),
            device="cuda",
            local_files_only=True,
            runtime_origin="hf-local",
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            loaded_base_model_weights_sha256=_SHA,
            verified_model_revision=_REVISION,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        )


def test_strict_constructor_rejects_snapshot_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_strict_windows_cuda_host(monkeypatch)
    snapshot_root = _canonical_snapshot_path(
        tmp_path,
        revision="d" * 40,
    )

    with pytest.raises(ValueError, match="revision does not match"):
        TransformersOpenWeightResidualRuntime(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            pretrained_source=str(snapshot_root),
            device="cuda",
            local_files_only=True,
            runtime_origin="hf-local",
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            loaded_base_model_weights_sha256=_SHA,
            verified_model_revision=_REVISION,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        )


@pytest.mark.parametrize("injected_name", ("model", "tokenizer"))
def test_strict_constructor_rejects_injected_model_or_tokenizer(
    injected_name: str,
) -> None:
    injected = {injected_name: object()}
    with pytest.raises(ValueError, match="forbids injected"):
        TransformersOpenWeightResidualRuntime(
            model_id="test-model",
            device="cuda",
            local_files_only=True,
            runtime_origin="hf-local",
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            loaded_base_model_weights_sha256=_SHA,
            verified_model_revision=_REVISION,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
            **injected,
        )


def test_strict_factory_rejects_execution_asset_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    snapshot_root = _canonical_snapshot_path(tmp_path)
    monkeypatch.setattr(
        residual_backend_module,
        "fingerprint_model_weight_files",
        lambda path: _SHA,
    )
    monkeypatch.setattr(
        residual_backend_module,
        "fingerprint_transformers_execution_assets",
        lambda path: "d" * 64,
    )

    with pytest.raises(ValueError, match="execution-assets digest"):
        build_transformers_runtime_with_fallback(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            model_source=str(snapshot_root),
            device="cuda",
            local_files_only=True,
            runtime_mode=LocalSubstrateRuntimeMode.STRICT_LOCAL,
            fallback_mode=SubstrateFallbackMode.DENY,
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            expected_model_weights_sha256=_SHA,
            verified_model_revision=_REVISION,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        )


def test_strict_constructor_recomputes_and_rejects_execution_asset_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    snapshot_root = _canonical_snapshot_path(
        tmp_path,
        model_id="test-model",
    )
    _patch_strict_windows_cuda_host(monkeypatch)
    monkeypatch.setattr(
        residual_backend_module,
        "fingerprint_model_weight_files",
        lambda path: _SHA,
    )
    monkeypatch.setattr(
        residual_backend_module,
        "fingerprint_transformers_execution_assets",
        lambda path: "d" * 64,
    )

    with pytest.raises(ValueError, match="assets SHA-256 mismatch"):
        TransformersOpenWeightResidualRuntime(
            model_id="test-model",
            pretrained_source=str(snapshot_root),
            device="cuda",
            local_files_only=True,
            runtime_origin="hf-local",
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            loaded_base_model_weights_sha256=_SHA,
            verified_model_revision=_REVISION,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        )


def test_strict_factory_rejects_snapshot_download_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    hub = pytest.importorskip("huggingface_hub")
    different_revision = "d" * 40
    snapshot_root = (
        tmp_path
        / "models--Qwen--Qwen2.5-1.5B-Instruct"
        / "snapshots"
        / different_revision
    )
    snapshot_root.mkdir(parents=True)
    monkeypatch.setattr(
        hub,
        "snapshot_download",
        lambda **kwargs: str(snapshot_root),
    )

    with pytest.raises(ValueError, match="different from the requested"):
        build_transformers_runtime_with_fallback(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            device="cuda",
            local_files_only=True,
            runtime_mode=LocalSubstrateRuntimeMode.STRICT_LOCAL,
            fallback_mode=SubstrateFallbackMode.DENY,
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            expected_model_weights_sha256=_SHA,
            verified_model_revision=_REVISION,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        )


def test_strict_constructor_rejects_non_windows_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(residual_backend_module.platform, "system", lambda: "Linux")

    with pytest.raises(RuntimeError, match="requires Windows"):
        TransformersOpenWeightResidualRuntime(
            model_id="test-model",
            device="cuda",
            local_files_only=True,
            runtime_origin="hf-local",
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            loaded_base_model_weights_sha256=_SHA,
            verified_model_revision=_REVISION,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        )


def test_strict_constructor_rejects_cpu_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(residual_backend_module.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        TransformersOpenWeightResidualRuntime(
            model_id="test-model",
            device="cpu",
            local_files_only=True,
            runtime_origin="hf-local",
            max_length=32768,
            fail_on_truncation=True,
            model_dtype="bfloat16",
            loaded_base_model_weights_sha256=_SHA,
            verified_model_revision=_REVISION,
            expected_execution_assets_sha256=_ASSETS_SHA,
            execution_profile=WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1,
        )


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(device: str) -> str:
        del device
        return "RTX"

    @staticmethod
    def get_device_capability(device: str) -> tuple[int, int]:
        del device
        return (8, 9)


class _FakeCudnn:
    @staticmethod
    def version() -> int:
        return 91002


def _attestation_runtime(
    *,
    max_positions: int,
    attention_implementation: str = "sdpa",
) -> TransformersOpenWeightResidualRuntime:
    runtime = _runtime_shell()
    runtime._execution_profile = WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1
    runtime._torch = SimpleNamespace(
        __version__="2.12.0+cu126",
        cuda=_FakeCuda(),
        backends=SimpleNamespace(cudnn=_FakeCudnn()),
        version=SimpleNamespace(cuda="12.6"),
    )
    runtime._transformers = SimpleNamespace(__version__="5.9.0")
    runtime._model = SimpleNamespace(
        dtype="torch.bfloat16",
        config=SimpleNamespace(
            _attn_implementation=attention_implementation,
            _commit_hash=_REVISION,
            max_position_embeddings=max_positions,
        ),
    )
    runtime._pretrained_source = f"models/snapshots/{_REVISION}"
    runtime._max_length = 32768
    runtime._loaded_base_model_weights_sha256 = _SHA
    runtime._verified_model_revision = _REVISION
    runtime._execution_assets_sha256 = _ASSETS_SHA
    runtime._device = "cuda:0"
    runtime._layer_indices = (7,)
    runtime._hidden_size = 1536
    runtime.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    runtime.runtime_origin = "hf-local"
    return runtime


def test_execution_attestation_rejects_model_context_drift() -> None:
    runtime = _attestation_runtime(max_positions=32767)

    with pytest.raises(RuntimeError, match="native context window drift"):
        runtime._build_execution_attestation()


def test_execution_attestation_rejects_attention_drift() -> None:
    runtime = _attestation_runtime(
        max_positions=32768,
        attention_implementation="eager",
    )

    with pytest.raises(RuntimeError, match="did not load SDPA"):
        runtime._build_execution_attestation()


def test_execution_attestation_rejects_model_config_revision_mismatch() -> None:
    runtime = _attestation_runtime(max_positions=32768)
    runtime._model.config._commit_hash = "d" * 40

    with pytest.raises(RuntimeError, match="config revision"):
        runtime._build_execution_attestation()


def test_execution_attestation_contains_resolved_runtime_facts() -> None:
    runtime = _attestation_runtime(max_positions=32768)

    attestation = runtime._build_execution_attestation()

    assert attestation.model_revision == _REVISION
    assert attestation.execution_assets_sha256 == _ASSETS_SHA
    assert attestation.device_compute_capability == (8, 9)
    assert attestation.hidden_size == 1536
    assert attestation.hook_layer_indices == (7,)
    assert attestation.model_max_position_embeddings == 32768
    assert attestation.attention_implementation == "sdpa"
    assert attestation.generation_use_cache is True
    with pytest.raises(AttributeError):
        runtime.execution_attestation = attestation
