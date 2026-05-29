"""vLLM-backed generation runtime with native multi-LoRA serving.

The transformers runtime decodes one request at a time and loads/swaps a
single LoRA per forward; it is the right backend for residual capture +
steering bake but caps throughput at "one decode in flight". For
high-concurrency multi-tenant serving the intended production backend is
vLLM, which batches many requests and attaches a **per-request**
``LoRARequest`` so N tenants with N different personas decode
concurrently on one frozen base (S-LoRA style).

Scope of this runtime:

* It is a **generation / serving** backend. vLLM does not expose the
  residual stream, so :meth:`capture` / :meth:`apply_control` fail loud
  — residual capture + steering bake stay on the transformers runtime.
* It implements per-request LoRA routing via :class:`VLLMLoRARouter`
  (pure, unit-testable) and the persona-pool activation contract via
  :meth:`activate_peft_adapter` (sets the active adapter for generates
  inside the context).
* ``activate_lora`` (the projected-summary hook path) is **not**
  supported — vLLM requires real adapter checkpoints, so the DLaaS
  adopt path (which now carries ``peft_checkpoint_dir``) routes through
  :meth:`activate_peft_adapter`.

vLLM is an **optional** dependency (``vz-substrate[vllm]``); the module
imports cleanly without it and only requires it when a real engine is
constructed. Tests inject a fake engine + ``LoRARequest`` factory.

R2: vLLM holds the base weights frozen; LoRA adapters are additive
per-request overlays, never a base-weight update.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
from typing import Any, Callable, Iterator

from volvence_zero.substrate.peft_adapter_cache import adapter_name_for
from volvence_zero.substrate.residual_contracts import GenerationResult
from volvence_zero.substrate.residual_interfaces import (
    OpenWeightResidualRuntime,
)

# A LoRARequest factory takes (adapter_name, int_id, checkpoint_dir) and
# returns whatever object the engine's ``generate`` accepts as
# ``lora_request`` (vllm.lora.request.LoRARequest in production).
LoRARequestFactory = Callable[[str, int, str], Any]


class VLLMLoRARouter:
    """Assigns stable integer ids + ``LoRARequest`` objects per adapter.

    vLLM identifies a resident LoRA by a unique positive integer id; the
    router hands out monotonically increasing ids keyed by checkpoint
    directory so repeated requests for the same persona reuse the same
    id (and vLLM keeps it warm in its adapter slots). Pure logic — no
    vllm import — so the routing can be unit-tested with a fake factory.
    """

    def __init__(self, lora_request_factory: LoRARequestFactory) -> None:
        self._factory = lora_request_factory
        self._by_path: dict[str, tuple[int, Any]] = {}
        self._next_id = 1

    @property
    def resident_count(self) -> int:
        return len(self._by_path)

    def request_for(self, checkpoint_dir: str) -> Any:
        """Return the ``LoRARequest`` for ``checkpoint_dir`` (stable id)."""

        if not checkpoint_dir:
            raise ValueError(
                "VLLMLoRARouter.request_for: checkpoint_dir must be non-empty"
            )
        existing = self._by_path.get(checkpoint_dir)
        if existing is not None:
            return existing[1]
        lora_id = self._next_id
        self._next_id += 1
        name = adapter_name_for(checkpoint_dir)
        request = self._factory(name, lora_id, checkpoint_dir)
        self._by_path[checkpoint_dir] = (lora_id, request)
        return request

    def id_for(self, checkpoint_dir: str) -> int | None:
        entry = self._by_path.get(checkpoint_dir)
        return entry[0] if entry is not None else None


def _default_lora_request_factory() -> LoRARequestFactory:
    """Real factory building ``vllm.lora.request.LoRARequest``."""

    module = importlib.import_module("vllm.lora.request")
    lora_request_cls = module.LoRARequest

    def factory(name: str, lora_id: int, checkpoint_dir: str) -> Any:
        return lora_request_cls(name, lora_id, checkpoint_dir)

    return factory


class VLLMOpenWeightResidualRuntime(OpenWeightResidualRuntime):
    """vLLM serving runtime with per-request multi-LoRA.

    Construct with a real engine by passing ``model_id`` (lazily imports
    vllm and builds ``vllm.LLM(enable_lora=True, ...)``), or inject an
    ``engine`` + ``lora_request_factory`` for tests.
    """

    def __init__(
        self,
        *,
        model_id: str,
        max_loras: int = 4,
        max_lora_rank: int = 16,
        dtype: str = "auto",
        engine: Any | None = None,
        lora_request_factory: LoRARequestFactory | None = None,
        sampling_params_factory: Callable[[int, float], Any] | None = None,
        **engine_kwargs: Any,
    ) -> None:
        if not model_id.strip():
            raise ValueError("VLLMOpenWeightResidualRuntime.model_id must be non-empty")
        self.model_id = model_id
        self.is_frozen = True
        self.runtime_origin = "vllm"
        self.supports_live_substrate_mutation = False
        self.supports_offline_substrate_training = False
        self._max_loras = max(1, max_loras)
        if engine is not None:
            self._engine = engine
        else:
            vllm = importlib.import_module("vllm")
            self._engine = vllm.LLM(
                model=model_id,
                enable_lora=True,
                max_loras=self._max_loras,
                max_lora_rank=max_lora_rank,
                dtype=dtype,
                **engine_kwargs,
            )
        # Use the injected factory when provided (tests / custom
        # transports); otherwise build the real vllm-backed factory
        # lazily (only reached when no factory was injected).
        self._lora_request_factory = (
            lora_request_factory or _default_lora_request_factory()
        )
        self._sampling_params_factory = sampling_params_factory
        self._router = VLLMLoRARouter(self._lora_request_factory)
        # Active adapter is tracked per async task (not per instance) so
        # concurrent tenant turns each route their own persona without
        # clobbering a shared attribute — unlike the transformers
        # runtime, vLLM batches requests so the serial-decode assumption
        # does not hold here.
        self._active_checkpoint_var: contextvars.ContextVar[str | None] = (
            contextvars.ContextVar(
                f"vllm_active_checkpoint_{id(self)}", default=None
            )
        )

    # -- residual surface: not available on a serving backend ----------

    def capture(self, *, source_text: str):
        raise NotImplementedError(
            "VLLMOpenWeightResidualRuntime is a generation/serving backend "
            "and does not expose the residual stream. Use "
            "TransformersOpenWeightResidualRuntime for residual capture / "
            "steering bake."
        )

    def apply_control(self, **kwargs):
        raise NotImplementedError(
            "VLLMOpenWeightResidualRuntime does not support residual "
            "control application; route control through the transformers "
            "runtime."
        )

    # -- LoRA activation contract --------------------------------------

    @property
    def lora_router(self) -> VLLMLoRARouter:
        return self._router

    @contextlib.contextmanager
    def activate_peft_adapter(self, checkpoint_dir: str) -> Iterator[None]:
        """Make ``checkpoint_dir`` the active adapter for generates in scope.

        vLLM attaches the adapter per request, so the context just pins
        which checkpoint subsequent :meth:`generate` calls route through
        the :class:`VLLMLoRARouter`. The active checkpoint is stored in a
        :class:`contextvars.ContextVar` so concurrent tenant turns (each
        its own async task) do not clobber one another. Nesting *within
        the same task* is rejected so persona conflicts are loud; two
        different tasks activating different personas concurrently is the
        supported multi-tenant path, not an error.
        """

        if not str(checkpoint_dir).strip():
            raise ValueError(
                "activate_peft_adapter: checkpoint_dir must be non-empty"
            )
        if self._active_checkpoint_var.get() is not None:
            raise RuntimeError(
                "activate_peft_adapter: nested activation detected in this "
                "task; exit the outer adapter context before activating "
                "another."
            )
        # Warm the router so the id is assigned up front (vLLM will load
        # the adapter into a slot on first use).
        self._router.request_for(str(checkpoint_dir))
        token = self._active_checkpoint_var.set(str(checkpoint_dir))
        try:
            yield
        finally:
            self._active_checkpoint_var.reset(token)

    def activate_lora(self, layers):
        raise NotImplementedError(
            "VLLMOpenWeightResidualRuntime does not support the projected "
            "adapter_layers hook path; vLLM requires a real PEFT checkpoint. "
            "Bake a checkpoint and route through activate_peft_adapter "
            "(the DLaaS adopt path now carries peft_checkpoint_dir)."
        )

    # -- generation ----------------------------------------------------

    def generate(
        self,
        *,
        prompt: str,
        system_context: str = "",
        chat_messages: tuple[tuple[str, str], ...] = (),
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        control_parameters: tuple[float, ...] = (),
        control_scale: float = 0.0,
        generation_constraints: Any | None = None,
    ) -> GenerationResult:
        del system_context, chat_messages, control_parameters, control_scale
        del generation_constraints
        return self._generate_with_lora(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            lora_checkpoint_dir=self._active_checkpoint_var.get(),
        )

    def generate_for_request(
        self,
        *,
        prompt: str,
        lora_checkpoint_dir: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Per-request generate that routes its own adapter (concurrent path).

        Unlike :meth:`generate` (which reads the activation context),
        this attaches the adapter for *this request only* so many
        tenants can decode concurrently with different personas in one
        batched engine.
        """

        return self._generate_with_lora(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            lora_checkpoint_dir=lora_checkpoint_dir,
        )

    def _generate_with_lora(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        lora_checkpoint_dir: str | None,
    ) -> GenerationResult:
        lora_request = None
        if lora_checkpoint_dir:
            lora_request = self._router.request_for(lora_checkpoint_dir)
        sampling_params = self._build_sampling_params(
            max_new_tokens=max_new_tokens, temperature=temperature
        )
        outputs = self._engine.generate(
            [prompt], sampling_params, lora_request=lora_request
        )
        text, token_count = _read_first_output(outputs)
        return GenerationResult(
            text=text,
            token_count=token_count,
            capture=None,
            description=(
                f"vllm:{self.model_id}"
                + (f" lora={adapter_name_for(lora_checkpoint_dir)}" if lora_checkpoint_dir else "")
            ),
        )

    def _build_sampling_params(self, *, max_new_tokens: int, temperature: float) -> Any:
        if self._sampling_params_factory is not None:
            return self._sampling_params_factory(
                max(1, max_new_tokens), max(0.0, temperature)
            )
        vllm = importlib.import_module("vllm")
        return vllm.SamplingParams(
            max_tokens=max(1, max_new_tokens),
            temperature=max(0.0, temperature),
        )


def _read_first_output(outputs: Any) -> tuple[str, int]:
    """Extract ``(text, token_count)`` from a vLLM generate result."""

    if not outputs:
        raise RuntimeError("vLLM generate returned no outputs")
    first = outputs[0]
    completions = first.outputs
    if not completions:
        raise RuntimeError("vLLM generate returned an empty completion list")
    completion = completions[0]
    text = completion.text
    token_ids = getattr(completion, "token_ids", ())
    return text, len(tuple(token_ids))


__all__ = [
    "LoRARequestFactory",
    "VLLMLoRARouter",
    "VLLMOpenWeightResidualRuntime",
]
