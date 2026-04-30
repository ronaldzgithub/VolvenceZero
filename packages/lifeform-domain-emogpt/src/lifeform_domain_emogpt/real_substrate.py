"""Helper for building a companion lifeform on a real HF substrate.

This module is the **first-principles fix** for the diagnostics
surfaced by ``examples/companionship_end_to_end_demo.py``:

1. With the synthetic substrate, the regime classifier never leaves
   ``acquaintance_building`` and the 12-axis InterlocutorState
   barely moves. Root cause: the synthetic ``feature_surface``
   pulls (task / support / repair / exploration / directive) are
   content-agnostic, so dual_track's ``controller_code`` stays
   flat across emotionally-different prompts.
2. Even on a real substrate, the default ``NoOpSemanticProposalRuntime``
   only emits OBSERVE proposals \u2014 commitments never get
   CREATE / COMPLETE / BLOCK / DEFER lifecycle events, so the
   commitment owner stays inert.

This file fixes both:

- It defaults to **Qwen 2.5 1.5B Instruct** (slice 2a phase B).
  The 1.5B model produces visibly more discriminative anchor pulls
  than 0.5B and gives the regime soft-prior calibrator something
  to actually grip; loaded in bfloat16 it fits in <4 GB RAM.
- It exposes ``with_llm_semantic_runtime=True`` (slice 2a phase A).
  When enabled the function pre-loads the HF model + tokenizer
  ONCE, hands them to ``TransformersOpenWeightResidualRuntime``
  for residual capture, and wraps them in
  ``HFTextGenerationProvider`` + ``LLMSemanticProposalRuntime``
  so the commitment owner sees real CREATE / COMPLETE / BLOCK /
  DEFER classifications. Both subsystems share one set of weights
  in RAM \u2014 no double loading.

Public API:

* ``CompanionLifeformBundle`` \u2014 small frozen dataclass with
  ``lifeform``, ``runtime``, ``runtime_origin``, ``model_id``,
  ``is_real_substrate``, ``llm_semantic_runtime``.
* ``build_companion_lifeform_with_real_substrate(...)`` \u2014 returns
  a ``CompanionLifeformBundle``. Tries to load the requested HF
  model; falls back to the builtin (random-weights GPT-2) on
  network / dependency failure when ``fallback_to_builtin=True``
  (default), with ``is_real_substrate=False`` so the caller can
  signal degraded mode in their UI / demo output. When fallback
  engages the LLM semantic runtime is silently dropped (it would
  classify against random weights = meaningless).

Why a separate module rather than extending ``build_companion_lifeform``:
keeping it separate means hosts that don't need a real substrate
(unit tests, the existing companion demo) pay no import cost; the
``transformers`` / ``torch`` import is deferred to the first call.

See Gap 9 / companion-vertical follow-ups in
``docs/implementation/13_emogpt_prd_alignment_upgrade.md`` for the
phased rollout. Slice 2a phases B + A land here together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Default model. Qwen 2.5 1.5B Instruct (slice 2a phase B):
# * ~3 GB on disk after first download,
# * loaded in bfloat16 the in-memory footprint is ~2.9 GB,
# * decoder-only with a standard ``model.model.layers`` block path
#   that ``TransformersOpenWeightResidualRuntime._resolve_transformer_blocks``
#   already supports,
# * instruction-tuned, which yields more semantically-differentiated
#   activations on conversational input than the base or 0.5B model
#   (relevant for the 5 anchor pulls task / support / repair /
#   exploration / directive).
DEFAULT_REAL_MODEL_SOURCE: str = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_REAL_MODEL_ID: str = "qwen2.5-1.5b-instruct"


@dataclass(frozen=True)
class CompanionLifeformBundle:
    """A built companion lifeform plus the runtimes that drive it,
    plus audit metadata callers need to honestly report which mode
    they are in.

    ``is_real_substrate`` is the canonical flag for the residual
    pathway. ``llm_semantic_runtime`` is the canonical handle for
    the commitment-classification pathway: ``None`` means the
    OBSERVE-only NoOp runtime is wired (either because the caller
    didn't ask for it or because real-substrate fallback engaged).

    The whole point of slice 2a is that "synthetic / random-weights
    / NoOp semantic" output is acknowledged-bad, not silently
    labelled "real" \u2014 callers who report status should surface
    BOTH flags.
    """

    lifeform: Any
    runtime: Any
    runtime_origin: str
    model_id: str
    is_real_substrate: bool
    llm_semantic_runtime: Any = None

    @property
    def status_label(self) -> str:
        if self.is_real_substrate:
            base = f"real:{self.model_id}"
        else:
            base = f"degraded:{self.runtime_origin}:{self.model_id}"
        if self.llm_semantic_runtime is not None:
            return f"{base}+llm-semantic"
        return base


def _build_real_substrate_with_optional_llm(
    *,
    model_source: str,
    model_id: str,
    device: str,
    fallback_to_builtin: bool,
    use_llm_semantic_runtime: bool,
    torch_dtype: str | None,
    local_files_only: bool,
) -> tuple[Any, str, bool, Any]:
    """Load the HF model exactly once and wire residual + LLM runtimes.

    Returns ``(substrate_runtime, runtime_origin, is_real_substrate,
    llm_semantic_runtime_or_none)``. On ``ImportError`` or HF load
    failure: if ``fallback_to_builtin`` is True we fall through to
    the builtin random-weights GPT-2 runtime and ``llm_semantic_runtime``
    is ``None`` (random weights cannot classify anything meaningful).
    Otherwise the original exception bubbles up.

    The LLM semantic runtime SHARES the loaded model + tokenizer
    instance with the residual runtime. That is the whole reason
    we side-step ``build_transformers_runtime_with_fallback``: it
    loads the model internally and exposes no public accessor, so
    we'd have to load Qwen twice to wire generation. Here we
    pre-load with the right dtype, then hand the instances to BOTH
    consumers.
    """
    from volvence_zero.substrate import (
        HFTextGenerationProvider,
        SubstrateFallbackMode,
        TransformersOpenWeightResidualRuntime,
        build_builtin_transformers_runtime,
        resolve_substrate_fallback_mode,
    )

    fallback_mode = resolve_substrate_fallback_mode(
        fallback_mode=(
            SubstrateFallbackMode.ALLOW_BUILTIN
            if fallback_to_builtin
            else SubstrateFallbackMode.DENY
        ),
        fallback_to_builtin=fallback_to_builtin,
    )

    runtime_origin = "hf-local" if local_files_only else "hf-pretrained"

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if torch_dtype is not None:
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            if torch_dtype not in dtype_map:
                raise ValueError(
                    f"Unsupported torch_dtype={torch_dtype!r}; "
                    f"choose one of {sorted(dtype_map)}."
                )
            load_kwargs["torch_dtype"] = dtype_map[torch_dtype]
            # ``low_cpu_mem_usage`` materialises the model layer-by-layer
            # rather than a single fp32 buffer + cast pass, which is what
            # caused the 1.5B fp32 OOM-then-segfault on the diagnostics
            # box. Required when we explicitly pick a non-default dtype.
            load_kwargs["low_cpu_mem_usage"] = True

        tokenizer = AutoTokenizer.from_pretrained(
            model_source, local_files_only=local_files_only
        )
        model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)

        substrate_runtime = TransformersOpenWeightResidualRuntime(
            model_id=model_id,
            pretrained_source=model_source,
            device=device,
            model=model,
            tokenizer=tokenizer,
            local_files_only=local_files_only,
            runtime_origin=runtime_origin,
        )
        # Pin the runtime origin attribute (the constructor accepts
        # it but ``build_transformers_runtime_with_fallback`` is the
        # canonical setter; we mirror its behaviour explicitly here).
        substrate_runtime.runtime_origin = runtime_origin

        llm_semantic_runtime: Any = None
        if use_llm_semantic_runtime:
            from volvence_zero.semantic_state.llm_runtime import (
                LLMSemanticProposalRuntime,
            )

            provider = HFTextGenerationProvider(
                model=model,
                tokenizer=tokenizer,
                device=substrate_runtime._device,
            )
            llm_semantic_runtime = LLMSemanticProposalRuntime(provider=provider)

        return substrate_runtime, runtime_origin, True, llm_semantic_runtime
    except Exception:
        if fallback_mode is not SubstrateFallbackMode.ALLOW_BUILTIN:
            raise
        # Fall back to the builtin random-weights runtime. We do
        # NOT carry the LLM semantic runtime forward: classifying
        # against untrained weights would be theatre.
        runtime = build_builtin_transformers_runtime(
            model_id="builtin-transformers-runtime",
            device=device,
        )
        return runtime, getattr(runtime, "runtime_origin", "builtin-fallback"), False, None


def build_companion_lifeform_with_real_substrate(
    *,
    model_source: str = DEFAULT_REAL_MODEL_SOURCE,
    model_id: str = DEFAULT_REAL_MODEL_ID,
    device: str = "cpu",
    fallback_to_builtin: bool = True,
    use_temporal_bootstrap: bool = True,
    use_regime_bootstrap: bool = True,
    use_vitals_bootstrap: bool = True,
    use_llm_semantic_runtime: bool = False,
    torch_dtype: str | None = "bfloat16",
    local_files_only: bool = False,
    config: Any = None,
) -> CompanionLifeformBundle:
    """Build a companion lifeform backed by a real HF transformer.

    Pipeline:

    1. Load the HF model + tokenizer ONCE, in ``torch_dtype``
       precision (``bfloat16`` by default to keep the 1.5B model
       under 4 GB RAM).
    2. Wrap them in ``TransformersOpenWeightResidualRuntime`` for
       residual capture.
    3. If ``use_llm_semantic_runtime=True``: wrap the SAME model
       in ``HFTextGenerationProvider`` + ``LLMSemanticProposalRuntime``
       so the commitment owner sees lifecycle events extracted
       from the user turn (CREATE / COMPLETE / BLOCK / DEFER /
       OBSERVE). The runtime call adds ~one short generation per
       turn (8 tokens, greedy) on top of the residual capture.
    4. Inject both runtimes into ``build_companion_lifeform`` which
       forces ``substrate_mode="injected"`` so all sessions share
       the one loaded model.
    5. On HF load failure with ``fallback_to_builtin=True``: drop
       to the builtin random-weights GPT-2 runtime AND silently
       drop the LLM semantic runtime. Returned bundle exposes
       both signals (``is_real_substrate`` + ``llm_semantic_runtime``)
       so the caller can surface degraded status honestly.

    Defaults to ``device="cpu"`` because Qwen 1.5B at bfloat16 fits
    comfortably in laptop RAM and saves the demo from CUDA / hardware
    bring-up. Hosts with a GPU pass ``device="cuda"`` (or
    ``"auto"``); the runtime forwards it to the underlying torch
    model.

    Lifeform-construction kwargs (``use_*_bootstrap``, ``config``)
    are forwarded to ``build_companion_lifeform`` so callers don't
    have to chain two factories.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    runtime, runtime_origin, is_real_substrate, llm_semantic_runtime = (
        _build_real_substrate_with_optional_llm(
            model_source=model_source,
            model_id=model_id,
            device=device,
            fallback_to_builtin=fallback_to_builtin,
            use_llm_semantic_runtime=use_llm_semantic_runtime,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
    )

    lifeform = build_companion_lifeform(
        config=config,
        use_temporal_bootstrap=use_temporal_bootstrap,
        use_regime_bootstrap=use_regime_bootstrap,
        use_vitals_bootstrap=use_vitals_bootstrap,
        substrate_runtime=runtime,
        semantic_proposal_runtime=llm_semantic_runtime,
    )
    return CompanionLifeformBundle(
        lifeform=lifeform,
        runtime=runtime,
        runtime_origin=runtime_origin or "unknown",
        model_id=model_id,
        is_real_substrate=is_real_substrate,
        llm_semantic_runtime=llm_semantic_runtime,
    )


__all__ = [
    "CompanionLifeformBundle",
    "DEFAULT_REAL_MODEL_ID",
    "DEFAULT_REAL_MODEL_SOURCE",
    "build_companion_lifeform_with_real_substrate",
]
