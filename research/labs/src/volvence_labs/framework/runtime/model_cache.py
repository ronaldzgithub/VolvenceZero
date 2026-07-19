"""Process-level cache for ModelRuntime instances.

Avoids reloading the same model 32 times per probe (4 cells × 8 seeds in
the shadow profile). Cache key = (model_id, revision, device, dtype).

Usage:
    from volvence_labs.framework.runtime import get_model_runtime

    rt = get_model_runtime("TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype="fp32")
    rt.load_model()  # only loads first time; subsequent calls are no-op

The cache is process-local (multiprocessing workers each get their own).
SequentialRunner reuses across units automatically.

Test isolation: call ``clear_model_cache()`` in test setUp/tearDown if needed.
"""

from __future__ import annotations

from typing import Optional

from .model import ModelRuntime


_CACHE: dict[tuple, ModelRuntime] = {}


def get_model_runtime(
    model_id: Optional[str] = None,
    *,
    device: str = "auto",
    dtype: str = "auto",
    revision: str = "main",
) -> ModelRuntime:
    """Return a cached ModelRuntime, or create+cache a new one.

    Same arguments yield the same instance (loaded once, reused).
    """
    key = (model_id or "default", revision, device, dtype)
    rt = _CACHE.get(key)
    if rt is None:
        rt = ModelRuntime(model_id, device=device, dtype=dtype, revision=revision)
        _CACHE[key] = rt
    return rt


def clear_model_cache() -> None:
    """Unload all cached runtimes and clear the cache.

    Call between large probe sweeps to free GPU memory.
    """
    global _CACHE
    for rt in _CACHE.values():
        try:
            rt.unload()
        except Exception:
            pass
    _CACHE = {}


def cache_size() -> int:
    """Number of cached runtimes (for testing)."""
    return len(_CACHE)
