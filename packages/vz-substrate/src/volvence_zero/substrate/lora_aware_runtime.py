"""Substrate-side Protocol for runtimes that support persona LoRA hot-swap.

This module defines :class:`LoRAAwareResidualRuntime`, the Protocol
the persona-LoRA pool calls into to activate / deactivate a
specific persona's adapter delta on top of the frozen base.

Why a separate Protocol module:

* :class:`OpenWeightResidualRuntime` (in ``residual_interfaces.py``)
  is the **abstract base** every runtime extends. It defines
  capture / apply_control / generate / rare-heavy-export / etc.
  Adding LoRA hot-swap as another abstract method would force
  every existing runtime (synthetic / transformers / future
  vLLM) to implement it on day one.
* The Protocol here is **structurally typed**: any runtime that
  defines ``activate_lora(layers) -> AbstractContextManager`` is
  a :class:`LoRAAwareResidualRuntime`, regardless of inheritance.
  This lets the persona-LoRA pool fall through to a no-op when
  the runtime does not implement the Protocol — and lets a future
  S-LoRA / vLLM runtime opt in by simply adding the method.

R2 contract: ``activate_lora`` MUST mutate **only** the controller
layer (LoRA delta), never the frozen base. The corresponding
contract test (``test_lora_activate_does_not_mutate_base.py``)
asserts the base ``state_dict`` hash is byte-identical before and
after the activate context.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from volvence_zero.substrate.residual_contracts import (
    SubstrateDeltaAdapterLayer,
)


@runtime_checkable
class LoRAAwareResidualRuntime(Protocol):
    """Runtimes that can activate a LoRA delta over their frozen base.

    Implementations:

    * :class:`TransformersOpenWeightResidualRuntime` (and
      :class:`OpenWeightResidualRuntime` ABC default) implement
      this by registering a forward-hook that adds the LoRA
      delta to the residual stream at the configured layers.
    * The synthetic runtime implements a no-op / accounting
      version so test code can call ``pool.activate(...,
      runtime=synthetic_runtime)`` without crashing — the
      capture path returns the same residuals with or without
      activation, which is the synthetic backend's documented
      semantics (no real forward to mutate).
    * A future S-LoRA / vLLM backend overrides this to push
      adapter weights onto the GPU-resident frozen base for the
      duration of the context.

    The context-manager idiom guarantees every activate is
    paired with a deactivate, even on exception, so a session
    cannot leak persona LoRA into a subsequent session.
    """

    def activate_lora(
        self,
        layers: tuple[SubstrateDeltaAdapterLayer, ...],
    ) -> AbstractContextManager[None]:
        """Activate ``layers`` over the frozen base for the context.

        Implementations must:

        1. Add the layer-indexed deltas to the runtime's forward
           pass (real runtimes via forward hooks; synthetic via
           accounting).
        2. On context exit (normal OR exception) restore the
           pre-activation state byte-identically. The frozen
           base must not be mutated.
        3. Be idempotent on re-entry: nesting two activations is
           an error (the implementation must raise
           ``RuntimeError``) so persona conflicts are loud, not
           silently merged.
        """
        ...


__all__ = [
    "LoRAAwareResidualRuntime",
]
