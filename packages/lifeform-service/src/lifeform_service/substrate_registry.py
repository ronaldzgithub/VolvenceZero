"""Pluggable Qwen-runtime selection for the browser-chat service.

This module owns the "which open-weight runtime is active in this
service process?" state. The browser-chat UI lets a developer
switch base models at runtime (e.g. drop the active 7B Qwen and
load a 1.5B variant for fast iteration); doing that safely requires
exactly one place that knows:

* what runtime is currently bound,
* what model ids the operator has authorised,
* how to atomically replace the runtime without leaving in-flight
  sessions in a half-swapped state.

R2 + R8 posture (read this before changing anything):

1. **R2 — substrate is frozen during a session.** A swap is a
   *process-level* event: every active session is closed before the
   new runtime is loaded. No session ever observes its substrate
   change mid-turn. The post-swap state is equivalent to having
   restarted the service with a different ``MODEL_ID``.
2. **R8 — single owner.** :class:`SubstrateRuntimeProvider` is the
   sole owner of the current substrate. ``SessionManager`` reads
   ``provider.current_runtime`` per ``create_session`` call rather
   than caching a runtime reference. The provider serialises swaps
   under an ``asyncio.Lock`` so two concurrent swap requests can
   never interleave half-loaded states.
3. **R2 — substrate stays frozen.** Every runtime the provider
   loads is rejected if it advertises ``supports_live_substrate_mutation``
   — the same gate ``create_app`` enforces today, just centralised
   here so swap-loaded runtimes get the same check.

DLaaS (multi-tenant production lane) is intentionally **not** wired
to this provider: ``dlaas_platform_api.build_dlaas_app`` continues
to construct ``create_app`` with the legacy ``substrate_runtime=...``
argument, which produces a fixed (non-swappable) provider. The
``/v1/admin/substrate`` route then reports ``swap_supported=false``
and refuses swap calls with HTTP 503. That keeps production
multi-tenant deployments out of the hot-swap blast radius.

Memory strategy on swap:

The default flow is **unload-first, load-second**: drop the current
runtime reference, force a ``gc.collect()`` and best-effort
``torch.cuda.empty_cache()``, then build the new runtime. This
costs 1× peak VRAM at the cost of a small "no runtime active"
window if the new load fails (in which case the swap raises and
the provider stays in a no-runtime state until a successful
follow-up swap). For a developer-facing browser-chat tool the
trade-off lands on memory safety; production paths that need
warm-swap behaviour can subclass and override
:meth:`SubstrateRuntimeProvider._reload_runtime`.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from volvence_zero.substrate import (
    SubstrateFallbackMode,
    build_transformers_runtime_with_fallback,
)


if TYPE_CHECKING:
    from volvence_zero.substrate import OpenWeightResidualRuntime


_LOG = logging.getLogger("lifeform_service.substrate_registry")


@dataclass(frozen=True)
class SubstrateModelSpec:
    """Operator-curated description of a switchable base model.

    Fields are deliberately UI-friendly primitives — the chat
    browser's model dropdown renders ``display_name`` plus
    ``size_label`` and tooltips ``notes``. Production callers that
    need richer metadata (e.g. capability flags) extend this in a
    subclass.

    ``device_hint`` is only a recommendation surfaced in the UI;
    the actual runtime device is whatever the provider was
    constructed with (which mirrors the ``DEVICE`` env handed to
    ``start_browser_chat_qwen.*``).
    """

    model_id: str
    display_name: str
    family: str
    size_label: str
    device_hint: str = ""
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "family": self.family,
            "size_label": self.size_label,
            "device_hint": self.device_hint,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SubstrateSwapResult:
    """Outcome of a successful :meth:`SubstrateRuntimeProvider.swap`.

    ``previous_model_id`` is empty when the provider had no active
    runtime before the swap (e.g. lazy first-load).
    """

    model_id: str
    previous_model_id: str
    runtime_origin: str
    closed_session_count: int
    duration_seconds: float


class SubstrateSwapError(RuntimeError):
    """Raised when a swap fails. Wraps the underlying loader error
    plus context on which model was being loaded so callers can
    surface a useful HTTP error.
    """

    def __init__(
        self,
        *,
        target_model_id: str,
        previous_model_id: str,
        cause: BaseException,
    ) -> None:
        super().__init__(
            f"Failed to swap substrate to {target_model_id!r} "
            f"(previous={previous_model_id!r}): {cause}"
        )
        self.target_model_id = target_model_id
        self.previous_model_id = previous_model_id
        self.__cause__ = cause


class UnknownSubstrateModelError(LookupError):
    """Raised when a swap requests a model_id outside the allowlist.

    The provider refuses to load arbitrary model ids by default —
    in a multi-user environment this is a security boundary
    (loading any HF id can fetch large blobs and execute model
    code via ``trust_remote_code``); even for single-user dev it
    catches typos before they trigger a multi-minute download.
    """


@dataclass
class _ProviderState:
    """Internal mutable state guarded by ``SubstrateRuntimeProvider._lock``."""

    current_runtime: "OpenWeightResidualRuntime | None"
    current_model_id: str
    swap_count: int = 0
    last_swap_error: str = ""


class SubstrateRuntimeProvider:
    """SSOT for the service's open-weight base-model selection.

    Construct once per service process. ``current_runtime`` may be
    ``None`` if the provider was created without an initial runtime
    (lazy-first-load path); :meth:`current_runtime_or_raise` is the
    convenience accessor that fails loudly when callers expect a
    runtime to exist.
    """

    def __init__(
        self,
        *,
        initial_runtime: "OpenWeightResidualRuntime | None",
        initial_model_id: str,
        available: Sequence[SubstrateModelSpec],
        runtime_loader: "Callable[[str], OpenWeightResidualRuntime]",
        swap_supported: bool = True,
        on_pre_swap: "Callable[[], int] | None" = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if initial_runtime is not None and not initial_model_id:
            raise ValueError(
                "initial_runtime supplied without initial_model_id"
            )
        if initial_runtime is not None:
            _enforce_frozen_for_sharing(initial_runtime)
        if not available:
            raise ValueError("SubstrateRuntimeProvider needs a non-empty allowlist")
        seen: set[str] = set()
        deduped: list[SubstrateModelSpec] = []
        for spec in available:
            if spec.model_id in seen:
                continue
            seen.add(spec.model_id)
            deduped.append(spec)
        self._available: tuple[SubstrateModelSpec, ...] = tuple(deduped)
        if initial_runtime is not None and initial_model_id not in seen:
            raise ValueError(
                f"initial_model_id={initial_model_id!r} is not in the "
                f"allowlist: {sorted(seen)!r}"
            )
        self._loader = runtime_loader
        self._swap_supported = swap_supported
        self._on_pre_swap = on_pre_swap
        self._clock = clock or _default_clock
        self._lock = asyncio.Lock()
        self._state = _ProviderState(
            current_runtime=initial_runtime,
            current_model_id=initial_model_id if initial_runtime is not None else "",
        )

    # ------------------------------------------------------------------
    # Read API (no lock — current state is monotonic between swaps and
    # readers tolerate a slightly-stale runtime: any caller that already
    # captured a runtime reference can keep using it; the provider's
    # invariants only require freshness at session-creation time).
    # ------------------------------------------------------------------

    @property
    def swap_supported(self) -> bool:
        return self._swap_supported

    @property
    def current_runtime(self) -> "OpenWeightResidualRuntime | None":
        return self._state.current_runtime

    @property
    def current_model_id(self) -> str:
        return self._state.current_model_id

    @property
    def swap_count(self) -> int:
        return self._state.swap_count

    @property
    def last_swap_error(self) -> str:
        return self._state.last_swap_error

    @property
    def available(self) -> tuple[SubstrateModelSpec, ...]:
        return self._available

    def is_known_model(self, model_id: str) -> bool:
        return any(spec.model_id == model_id for spec in self._available)

    def current_runtime_or_raise(self) -> "OpenWeightResidualRuntime":
        runtime = self._state.current_runtime
        if runtime is None:
            raise RuntimeError(
                "SubstrateRuntimeProvider has no current runtime — "
                "call ensure_loaded() or swap() before accessing it."
            )
        return runtime

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def set_pre_swap_callback(
        self, callback: "Callable[[], int] | None"
    ) -> None:
        """Install / replace the pre-swap callback.

        Wired by ``create_app`` after :class:`SessionManager` is
        constructed: the manager's ``close_all_sessions_locked`` is
        the canonical pre-swap action. Returning the count of
        closed sessions lets the swap result expose useful UI
        feedback ("closed 4 sessions before loading new model").
        """
        self._on_pre_swap = callback

    async def swap(self, model_id: str) -> SubstrateSwapResult:
        """Atomically replace the current runtime with ``model_id``.

        Steps under the lock:

        1. Validate ``model_id`` is in the allowlist.
        2. Run the pre-swap callback (typically: close all
           sessions). If the callback raises, the swap is aborted
           and the old runtime stays active.
        3. Drop the old runtime reference; force ``gc.collect()``
           and best-effort ``torch.cuda.empty_cache()``.
        4. Load the new runtime via the loader callable. The
           loader is responsible for enforcing freshness flags
           (e.g. ``allow_live_substrate_mutation=False``) — the
           provider double-checks via
           :func:`_enforce_frozen_for_sharing`.
        5. On success: update state, return :class:`SubstrateSwapResult`.
        6. On loader failure: leave state with no runtime (safer
           than half-loaded), record ``last_swap_error``, and
           re-raise as :class:`SubstrateSwapError`.
        """
        if not self._swap_supported:
            raise RuntimeError(
                "SubstrateRuntimeProvider was constructed with "
                "swap_supported=False; the runtime is fixed."
            )
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("swap(model_id): model_id must be a non-empty string")
        target = model_id.strip()
        if not self.is_known_model(target):
            raise UnknownSubstrateModelError(
                f"model_id={target!r} is not in the substrate allowlist; "
                f"available: {[spec.model_id for spec in self._available]!r}"
            )

        async with self._lock:
            previous_model_id = self._state.current_model_id
            if (
                target == previous_model_id
                and self._state.current_runtime is not None
            ):
                # Idempotent: no-op when the requested model is already loaded.
                return SubstrateSwapResult(
                    model_id=target,
                    previous_model_id=previous_model_id,
                    runtime_origin=getattr(
                        self._state.current_runtime, "runtime_origin", "unknown"
                    ),
                    closed_session_count=0,
                    duration_seconds=0.0,
                )
            started = self._clock()
            closed_sessions = 0
            if self._on_pre_swap is not None:
                closed_sessions = self._on_pre_swap()
            self._unload_locked()
            try:
                runtime = self._loader(target)
            except Exception as exc:  # noqa: BLE001 — wrapped + re-raised
                # Loader failure: clear current_model_id too so
                # /v1/info and /v1/models honestly report "no
                # active runtime". Operators read last_swap_error
                # for the previous-model context.
                self._state.current_model_id = ""
                self._state.last_swap_error = (
                    f"load failed for {target!r}: {exc}"
                )
                raise SubstrateSwapError(
                    target_model_id=target,
                    previous_model_id=previous_model_id,
                    cause=exc,
                ) from exc
            _enforce_frozen_for_sharing(runtime)
            self._state.current_runtime = runtime
            self._state.current_model_id = target
            self._state.swap_count += 1
            self._state.last_swap_error = ""
            duration = self._clock() - started
            _LOG.info(
                "substrate_swap target=%s previous=%s closed_sessions=%d duration=%.2fs",
                target,
                previous_model_id,
                closed_sessions,
                duration,
            )
            return SubstrateSwapResult(
                model_id=target,
                previous_model_id=previous_model_id,
                runtime_origin=getattr(runtime, "runtime_origin", "unknown"),
                closed_session_count=closed_sessions,
                duration_seconds=duration,
            )

    def _unload_locked(self) -> None:
        """Drop the current runtime ref + force GC + empty cuda cache.

        Called only with ``self._lock`` held. Best-effort:
        ``torch.cuda.empty_cache`` is wrapped in try/except because
        not every install has torch / cuda available, and the
        provider must keep working in a synthetic-only environment.
        We catch the *specific* Import / Attribute errors from a
        missing torch installation; truly unexpected failures are
        re-raised so the operator sees them.
        """
        self._state.current_runtime = None
        gc.collect()
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            return
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return
        try:
            if cuda.is_available():
                cuda.empty_cache()
        except (RuntimeError, AttributeError) as exc:
            _LOG.warning("torch.cuda.empty_cache failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_clock() -> float:
    import time

    return time.monotonic()


def _enforce_frozen_for_sharing(runtime: "OpenWeightResidualRuntime") -> None:
    """R2 invariant — refuse runtimes that allow live mutation.

    Mirrors the check ``lifeform_service.app._enforce_frozen_for_sharing``
    runs at app construction; centralising it here means swap-loaded
    runtimes get the same gate as the initial runtime.
    """
    if getattr(runtime, "supports_live_substrate_mutation", False):
        raise ValueError(
            "Cannot share a runtime that has supports_live_substrate_mutation=True "
            "across sessions: per-session adapter-delta updates would corrupt other "
            "sessions' weights. Build the runtime with "
            "allow_live_substrate_mutation=False (the default) when sharing."
        )


# ---------------------------------------------------------------------------
# Curated Qwen allowlist
# ---------------------------------------------------------------------------


# Reference: the sizing table maintained in start_browser_chat_qwen.{ps1,sh}.
# Only the variants that fit on a typical dev workstation are included; the
# 32B / 72B / MoE families are documented but excluded from the allowlist
# because they require remote inference. Operators that need a different
# variant set can override via the MODEL_ID_ALLOWLIST env var.
DEFAULT_QWEN_MODEL_SPECS: tuple[SubstrateModelSpec, ...] = (
    SubstrateModelSpec(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        display_name="Qwen2.5 1.5B Instruct",
        family="qwen2.5",
        size_label="1.5B",
        device_hint="cpu / mps / cuda",
        notes=(
            "Smallest variant — fast to load (~1-2 GB), but tends to "
            "collapse into single-character or off-topic replies once "
            "the kernel's plan/ordering instructions stack on top of "
            "the user turn. Useful for plumbing smoke tests, not for "
            "behaviour evaluation."
        ),
    ),
    SubstrateModelSpec(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        display_name="Qwen2.5 3B Instruct",
        family="qwen2.5",
        size_label="3B",
        device_hint="cpu / mps / cuda",
        notes=(
            "~6-8 GB load. Borderline-coherent on the kernel's "
            "structured prompt; works for fast iteration loops but "
            "shows obvious instruction-follow drift on long turns."
        ),
    ),
    SubstrateModelSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        display_name="Qwen2.5 7B Instruct",
        family="qwen2.5",
        size_label="7B",
        device_hint="cuda 16 GB / Mac M-series 24 GB",
        notes=(
            "Recommended default. Smallest base model that reliably "
            "follows the VZ structured system prompt AND keeps "
            "multi-turn coherence on short follow-ups."
        ),
    ),
    SubstrateModelSpec(
        model_id="Qwen/Qwen2.5-14B-Instruct",
        display_name="Qwen2.5 14B Instruct",
        family="qwen2.5",
        size_label="14B",
        device_hint="cuda 16 GB (Q4) / 32 GB (bf16)",
        notes=(
            "bf16 needs ~28 GB; 4-bit quantization fits a 16 GB GPU. "
            "Marked best-effort on this dev path because the Q4 "
            "loader configuration depends on the operator's runtime "
            "build options."
        ),
    ),
)


def parse_model_id_allowlist(
    raw: str | None,
    *,
    fallback: Sequence[SubstrateModelSpec] = DEFAULT_QWEN_MODEL_SPECS,
) -> tuple[SubstrateModelSpec, ...]:
    """Parse the ``MODEL_ID_ALLOWLIST`` env var into typed specs.

    Format: comma-separated HuggingFace ids
    (``Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct``). Unknown
    ids that don't match the curated table are still accepted but
    surfaced with a generic display name; callers that want richer
    metadata should extend :data:`DEFAULT_QWEN_MODEL_SPECS`.

    Empty / unset string returns ``fallback`` unchanged so the
    operator's default experience does not depend on env wiring.
    """
    if not raw or not raw.strip():
        return tuple(fallback)
    requested = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if not requested:
        return tuple(fallback)
    by_id = {spec.model_id: spec for spec in DEFAULT_QWEN_MODEL_SPECS}
    out: list[SubstrateModelSpec] = []
    for model_id in requested:
        spec = by_id.get(model_id)
        if spec is not None:
            out.append(spec)
            continue
        # Unknown id: synthesize a placeholder so the operator can
        # still reach it. ``family`` / ``size_label`` left blank so
        # the UI can render "(unknown)" labels.
        out.append(
            SubstrateModelSpec(
                model_id=model_id,
                display_name=model_id.rsplit("/", 1)[-1],
                family="unknown",
                size_label="",
                device_hint="",
                notes="Operator-supplied via MODEL_ID_ALLOWLIST.",
            )
        )
    return tuple(out)


def merge_initial_into_allowlist(
    *,
    initial_model_id: str,
    allowlist: Sequence[SubstrateModelSpec],
) -> tuple[SubstrateModelSpec, ...]:
    """Ensure ``initial_model_id`` is always in the allowlist.

    The browser-chat startup honours ``MODEL_ID`` regardless of the
    allowlist (otherwise an operator could lock themselves out of
    their own startup model). Returns a tuple where ``initial_model_id``
    is first, deduped against the rest.
    """
    if not initial_model_id.strip():
        return tuple(allowlist)
    existing = {spec.model_id: spec for spec in allowlist}
    initial = existing.get(initial_model_id) or SubstrateModelSpec(
        model_id=initial_model_id,
        display_name=initial_model_id.rsplit("/", 1)[-1],
        family="initial",
        size_label="",
        device_hint="",
        notes="Initial model from MODEL_ID env.",
    )
    rest = [spec for spec in allowlist if spec.model_id != initial_model_id]
    return (initial, *rest)


def build_qwen_runtime_loader(
    *,
    device: str,
    local_files_only: bool,
    fallback_mode: SubstrateFallbackMode,
) -> "Callable[[str], OpenWeightResidualRuntime]":
    """Return a closure that builds a runtime for a given model_id.

    Captures the device / fallback policy once at startup so swap
    requests don't need to re-thread those parameters. Matches the
    invocation pattern in ``start_browser_chat_qwen.{ps1,sh}``.
    """

    def _load(model_id: str) -> "OpenWeightResidualRuntime":
        runtime = build_transformers_runtime_with_fallback(
            model_id=model_id,
            device=device,
            local_files_only=local_files_only,
            fallback_mode=fallback_mode,
            allow_live_substrate_mutation=False,
        )
        runtime_origin = getattr(runtime, "runtime_origin", "unknown")
        if runtime_origin == "builtin-fallback":
            raise RuntimeError(
                f"Expected a real HF runtime for model_id={model_id!r}, got "
                "builtin-fallback. Check HF connectivity / local cache."
            )
        return runtime

    return _load


def build_substrate_provider_from_env(
    *,
    initial_runtime: "OpenWeightResidualRuntime | None",
    initial_model_id: str,
    runtime_loader: "Callable[[str], OpenWeightResidualRuntime]",
    allowlist_env: str | None = None,
) -> SubstrateRuntimeProvider:
    """Convenience for the start scripts.

    ``allowlist_env`` is the value of ``MODEL_ID_ALLOWLIST`` (or
    ``None``); leaving it unset gets the default Qwen lineup.
    The initial model is always merged in so a startup with a
    custom ``MODEL_ID`` is never locked out.
    """
    allowlist = parse_model_id_allowlist(allowlist_env)
    full = merge_initial_into_allowlist(
        initial_model_id=initial_model_id, allowlist=allowlist
    )
    return SubstrateRuntimeProvider(
        initial_runtime=initial_runtime,
        initial_model_id=initial_model_id,
        available=full,
        runtime_loader=runtime_loader,
        swap_supported=True,
    )


def fixed_provider_from_runtime(
    runtime: "OpenWeightResidualRuntime | None",
) -> SubstrateRuntimeProvider | None:
    """Wrap a single pre-built runtime in a non-swappable provider.

    Used by ``create_app(substrate_runtime=...)`` (DLaaS / legacy
    callers) so the rest of the service can uniformly consume a
    provider. Returns ``None`` for ``runtime=None`` so the caller
    can keep "no shared runtime" as a distinct state from "fixed
    runtime / no swap".
    """
    if runtime is None:
        return None
    model_id = getattr(runtime, "model_id", "")
    if not isinstance(model_id, str) or not model_id:
        model_id = "fixed-runtime"
    spec = SubstrateModelSpec(
        model_id=model_id,
        display_name=model_id,
        family="fixed",
        size_label="",
        device_hint="",
        notes="Service started with a fixed substrate runtime; swap is disabled.",
    )

    def _refuse(_: str) -> "OpenWeightResidualRuntime":
        raise RuntimeError("Fixed provider does not support runtime loads")

    return SubstrateRuntimeProvider(
        initial_runtime=runtime,
        initial_model_id=model_id,
        available=(spec,),
        runtime_loader=_refuse,
        swap_supported=False,
    )


__all__ = [
    "DEFAULT_QWEN_MODEL_SPECS",
    "SubstrateModelSpec",
    "SubstrateRuntimeProvider",
    "SubstrateSwapError",
    "SubstrateSwapResult",
    "UnknownSubstrateModelError",
    "build_qwen_runtime_loader",
    "build_substrate_provider_from_env",
    "fixed_provider_from_runtime",
    "merge_initial_into_allowlist",
    "parse_model_id_allowlist",
]
