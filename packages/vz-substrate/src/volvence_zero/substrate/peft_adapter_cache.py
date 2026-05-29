"""Bounded LRU cache of resident PEFT adapters for a frozen base model.

Without a cache, ``activate_peft_adapter`` re-runs
``peft.PeftModel.from_pretrained`` (disk read + module surgery) on every
turn and ``unload()`` on exit, so a busy multi-tenant process pays the
full adapter load cost per request. This module keeps up to
``VZ_LORA_CACHE_MAX`` adapters resident in a single
:class:`peft.PeftModel` (peft's native multi-adapter support) and just
switches the active adapter per turn.

R2 invariant: the **frozen base weights are never mutated**. peft injects
additive LoRA sub-modules; on context exit the cache disables all
adapter layers so the next base / different-persona turn is clean. A
forward with adapters disabled is byte-identical to the pre-injection
base forward (the LoRA layers short-circuit).

The torch / peft operations are injected through :class:`PeftAdapterOps`
so the cache logic (LRU, hit/miss accounting, eviction) is unit-testable
without a real model.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from collections import OrderedDict
from typing import Any, Iterator, Protocol

CACHE_MAX_ENV = "VZ_LORA_CACHE_MAX"
_DEFAULT_CACHE_MAX = 4


def peft_cache_max() -> int:
    """Return the configured resident-adapter cap (``VZ_LORA_CACHE_MAX``)."""

    raw = os.environ.get(CACHE_MAX_ENV, "").strip()
    if not raw:
        return _DEFAULT_CACHE_MAX
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{CACHE_MAX_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if value < 1:
        raise ValueError(
            f"{CACHE_MAX_ENV} must be >= 1, got {value!r}"
        )
    return value


def adapter_name_for(checkpoint_dir: str) -> str:
    """Deterministic peft adapter name for a checkpoint directory."""

    digest = hashlib.sha256(checkpoint_dir.encode("utf-8")).hexdigest()
    return f"persona_{digest[:16]}"


class PeftAdapterOps(Protocol):
    """Torch/peft operations the cache delegates to.

    Implemented for real by :class:`DefaultPeftAdapterOps`; tests inject
    a fake to exercise the LRU / accounting without torch.
    """

    def create(self, base_model: Any, checkpoint_dir: str, name: str) -> Any:
        """Create a PeftModel wrapping ``base_model`` with adapter ``name``."""

    def load(self, peft_model: Any, checkpoint_dir: str, name: str) -> None:
        """Load an additional adapter ``name`` into ``peft_model``."""

    def set_active(self, peft_model: Any, name: str) -> None:
        """Make ``name`` the active adapter."""

    def enable(self, peft_model: Any) -> None:
        """Enable adapter layers (persona overlay on)."""

    def disable(self, peft_model: Any) -> None:
        """Disable adapter layers (clean frozen-base forward)."""

    def delete(self, peft_model: Any, name: str) -> None:
        """Evict adapter ``name`` from ``peft_model``."""


class DefaultPeftAdapterOps:
    """Real peft-backed :class:`PeftAdapterOps`."""

    def __init__(self, peft_module: Any) -> None:
        self._peft = peft_module

    def create(self, base_model: Any, checkpoint_dir: str, name: str) -> Any:
        model = self._peft.PeftModel.from_pretrained(
            base_model, checkpoint_dir, adapter_name=name
        )
        model.eval()
        return model

    def load(self, peft_model: Any, checkpoint_dir: str, name: str) -> None:
        peft_model.load_adapter(checkpoint_dir, adapter_name=name)

    def set_active(self, peft_model: Any, name: str) -> None:
        peft_model.set_adapter(name)

    def enable(self, peft_model: Any) -> None:
        # ``enable_adapter_layers`` / ``disable_adapter_layers`` are
        # defined on the LoRA tuner (``PeftModel.base_model``), the
        # canonical owner; calling there avoids depending on
        # ``PeftModel.__getattr__`` forwarding (not present on the
        # class surface in peft 0.19.x).
        peft_model.base_model.enable_adapter_layers()

    def disable(self, peft_model: Any) -> None:
        peft_model.base_model.disable_adapter_layers()

    def delete(self, peft_model: Any, name: str) -> None:
        peft_model.delete_adapter(name)


class PeftAdapterCache:
    """Bounded LRU of resident PEFT adapters over one frozen base."""

    def __init__(self, *, ops: PeftAdapterOps, max_adapters: int) -> None:
        if max_adapters < 1:
            raise ValueError("PeftAdapterCache.max_adapters must be >= 1")
        self._ops = ops
        self._max = max_adapters
        # checkpoint_dir -> adapter_name, ordered oldest -> newest.
        self._order: "OrderedDict[str, str]" = OrderedDict()
        self._peft_model: Any = None
        self.hits = 0
        self.misses = 0

    @property
    def resident_count(self) -> int:
        return len(self._order)

    @property
    def peft_model(self) -> Any:
        return self._peft_model

    @contextlib.contextmanager
    def activate(self, *, base_model: Any, checkpoint_dir: str) -> Iterator[Any]:
        """Activate ``checkpoint_dir``'s adapter, loading/evicting as needed.

        Yields the resident :class:`peft.PeftModel` with the requested
        adapter active. On exit the adapters are disabled so the base
        model returns to its frozen forward path (R2).
        """

        name = adapter_name_for(checkpoint_dir)
        if checkpoint_dir in self._order:
            self.hits += 1
            self._order.move_to_end(checkpoint_dir)
        else:
            self.misses += 1
            if self._peft_model is None:
                self._peft_model = self._ops.create(
                    base_model, checkpoint_dir, name
                )
            else:
                self._ops.load(self._peft_model, checkpoint_dir, name)
            self._order[checkpoint_dir] = name
            self._evict_to_capacity()
        self._ops.set_active(self._peft_model, name)
        self._ops.enable(self._peft_model)
        try:
            yield self._peft_model
        finally:
            self._ops.disable(self._peft_model)

    def _evict_to_capacity(self) -> None:
        # The just-inserted adapter is newest (at the end); evict from
        # the front (oldest) so we never drop the adapter we are about
        # to activate.
        while len(self._order) > self._max:
            old_dir, old_name = next(iter(self._order.items()))
            self._order.popitem(last=False)
            self._ops.delete(self._peft_model, old_name)


def build_default_peft_adapter_cache(
    peft_module: Any,
    *,
    max_adapters: int | None = None,
) -> PeftAdapterCache:
    """Construct a cache backed by real peft ops."""

    return PeftAdapterCache(
        ops=DefaultPeftAdapterOps(peft_module),
        max_adapters=max_adapters if max_adapters is not None else peft_cache_max(),
    )


__all__ = [
    "CACHE_MAX_ENV",
    "DefaultPeftAdapterOps",
    "PeftAdapterCache",
    "PeftAdapterOps",
    "adapter_name_for",
    "build_default_peft_adapter_cache",
    "peft_cache_max",
]
