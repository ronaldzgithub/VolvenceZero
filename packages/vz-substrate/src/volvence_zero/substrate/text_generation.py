"""Lightweight text-generation provider over a HuggingFace transformers model.

Used by ``LLMSemanticProposalRuntime`` (kernel-side semantic event
extractor) to issue prompts at the AI's underlying frozen substrate
model and read back a short structured response. Designed so the
SAME loaded model that drives the substrate's residual capture can
also serve text generation, avoiding double-loading Qwen weights
into RAM.

Two design decisions worth knowing:

1. **Pure text in / text out.** The provider does NOT expose the
   model object; consumers only see strings. That keeps semantic
   event extraction decoupled from substrate's residual-stream
   internals: a future swap to a remote / API-only LLM is a
   provider replacement, not an architecture rewrite.
2. **Greedy decoding by default.** Semantic event extraction is a
   classification task; nucleus / temperature sampling would add
   noise. Callers wanting variety pass ``temperature > 0``.

This module imports torch + transformers eagerly; importing it
without those installed will fail loud. The module-level docstring
spells that out so callers can defer the import behind their own
optional-dep gate (mirrors the ``transformers`` extra pattern in
``vz-substrate``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from volvence_zero.substrate.runtime_execution import (
    run_runtime_call,
    runtime_resource_guard,
)


class TextGenerationProvider(Protocol):
    """Minimal text-in / text-out interface.

    Lives behind a Protocol so the kernel-side ``LLMSemanticProposalRuntime``
    can be unit-tested against a fake provider that returns canned
    strings, with no transformers dependency in the tests.
    """

    def generate(
        self, *, prompt: str, max_new_tokens: int = 16, temperature: float = 0.0
    ) -> str: ...


@dataclass(frozen=True)
class _GenerationConfig:
    max_new_tokens: int
    temperature: float


class HFTextGenerationProvider:
    """``TextGenerationProvider`` impl backed by a HuggingFace causal LM.

    Constructed with a pre-loaded ``model`` + ``tokenizer`` (the
    same instances the substrate runtime uses) so loading Qwen
    1.5B doesn't happen twice. The provider runs each call inside
    ``torch.no_grad()`` and returns the raw decoded string with
    leading whitespace trimmed. Caller is responsible for parsing.

    On generate-time failure (CUDA OOM, tokenizer error, model
    raise) the exception bubbles up so callers can fall back to a
    NoOp pathway. We deliberately do NOT swallow errors here \u2014
    ``LLMSemanticProposalRuntime`` is the right place to decide
    fallback behaviour.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str = "cpu",
        default_max_new_tokens: int = 16,
        use_chat_template: bool = True,
        runtime_execution_owner: object | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._default_max_new_tokens = default_max_new_tokens
        self._use_chat_template = (
            use_chat_template
            and getattr(tokenizer, "apply_chat_template", None) is not None
        )
        # Multiple lifeforms may wrap the same loaded residual runtime in
        # distinct provider objects. Bind their generation calls back to that
        # shared owner so the process-local execution gate serializes access to
        # the one underlying HF model. Standalone/test providers bind directly
        # to the injected model identity, so separate wrappers still converge.
        self._runtime_execution_owner = (
            model if runtime_execution_owner is None else runtime_execution_owner
        )
        # Lazy import torch so non-substrate callers don't pay the
        # import cost just because the module file exists.
        import torch  # noqa: F401  - imported for side-effect availability check
        self._torch = torch

    @property
    def runtime_execution_owner(self) -> object:
        """Shared identity used by the async model-execution gate."""

        return self._runtime_execution_owner

    @property
    def runtime_tokenizer_owner(self) -> object:
        """Tokenizer identity used by the process-local tokenizer gate."""

        return self._tokenizer

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
    ) -> str:
        # Some offline/apprenticeship extractors call the provider directly
        # rather than through ``run_runtime_call``. Keep the same loaded model
        # and tokenizer protected for that path too. The guard is reentrant,
        # so async callers already running under the boundary do not deadlock.
        with runtime_resource_guard(self):
            return self._generate_guarded(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

    def _generate_guarded(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        budget = max_new_tokens or self._default_max_new_tokens
        if self._use_chat_template:
            # Two-step: render the chat template to a string, then
            # tokenise that string. Newer transformers releases
            # return a ``BatchEncoding`` from ``apply_chat_template``
            # under some kwarg combinations, which would surprise
            # ``model.generate``; the two-step idiom always yields
            # a plain ``input_ids`` tensor we can call ``.to`` on.
            messages = [{"role": "user", "content": prompt}]
            formatted = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = self._tokenizer(
                formatted, return_tensors="pt"
            ).input_ids
        else:
            input_ids = self._tokenizer(
                prompt, return_tensors="pt"
            ).input_ids
        if hasattr(input_ids, "to"):
            input_ids = input_ids.to(self._device)
        eos_token_id = (
            self._tokenizer.eos_token_id
            if getattr(self._tokenizer, "eos_token_id", None) is not None
            else None
        )
        pad_token_id = (
            self._tokenizer.pad_token_id
            if getattr(self._tokenizer, "pad_token_id", None) is not None
            else eos_token_id
        )
        do_sample = temperature > 0.0
        with self._torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=budget,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        # Strip the prompt prefix so the caller only sees the
        # generated continuation.
        prefix_len = input_ids.shape[-1]
        new_tokens = output_ids[0][prefix_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


async def generate_text_async(
    provider: TextGenerationProvider,
    *,
    prompt: str,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    operation_kind: str = "text_generation",
) -> str:
    """Run a sync text provider on its canonical per-owner executor.

    This is the shared async adapter for live consumers whose public provider
    contract must remain sync-compatible. ``run_runtime_call`` resolves the
    provider's runtime/model/tokenizer owner identities; it is therefore not
    equivalent to a naked ``asyncio.to_thread`` call.
    """

    return await run_runtime_call(
        runtime=provider,
        operation=lambda: provider.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ),
        operation_kind=operation_kind,
    )


__all__ = [
    "HFTextGenerationProvider",
    "TextGenerationProvider",
    "generate_text_async",
]
