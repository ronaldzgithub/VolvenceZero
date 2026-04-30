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
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._default_max_new_tokens = default_max_new_tokens
        self._use_chat_template = (
            use_chat_template
            and getattr(tokenizer, "apply_chat_template", None) is not None
        )
        # Lazy import torch so non-substrate callers don't pay the
        # import cost just because the module file exists.
        import torch  # noqa: F401  - imported for side-effect availability check
        self._torch = torch

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
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


__all__ = [
    "HFTextGenerationProvider",
    "TextGenerationProvider",
]
