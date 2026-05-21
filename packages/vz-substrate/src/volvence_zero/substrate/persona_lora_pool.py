"""Persona LoRA pool — substrate-side additive interface.

A small process-level registry that holds N LoRA adapter artifacts
keyed by ``(figure_id, bundle_id)`` so the runtime can hot-swap
which persona's delta is overlaid on top of the same frozen base.

Why this lives in ``vz-substrate``:

* The substrate is the layer that owns the frozen base model and
  emits :class:`SubstrateDeltaAdapterLayer` tuples through its own
  rare-heavy / online-fast checkpoint paths. Adding a multi-LoRA
  hot-swap here keeps the adapter ownership in one wheel.
* Putting the pool in the substrate also means the lifeform tier
  does NOT take a runtime dependency on a substrate-internal mutable
  registry; it consumes the pool through its public symbols
  (:func:`default_pool`, :class:`PersonaLoRAPool`).

What this module is **not**:

* It is **not** a real S-LoRA / vLLM multi-LoRA inference path. The
  in-memory pool stores adapter layers as immutable
  :class:`SubstrateDeltaAdapterLayer` tuples; a future packet wires
  the actual GPU-resident hot-swap (the pool's ``register`` /
  ``activate`` calls are the contract surface that future code
  will satisfy).
* It does **not** import any ``lifeform_*`` symbols. The pool only
  knows about :class:`SubstrateDeltaAdapterLayer`; the figure-vertical
  pushes its baked artifact's ``adapter_layers`` into the pool
  through :meth:`register` and reads them back through
  :meth:`lookup_layers`.

R8 / R15 contract:

* Records are immutable; updates produce a fresh record id.
* :meth:`register` returns the record id so callers can pin it in
  their proposal's ``new_value_hash`` (this is what makes a
  rollback observable).
"""

from __future__ import annotations

import contextlib
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Iterator

from volvence_zero.substrate.lora_aware_runtime import (
    LoRAAwareResidualRuntime,
)
from volvence_zero.substrate.residual_contracts import (
    SubstrateDeltaAdapterLayer,
)


@dataclass(frozen=True)
class PersonaLoRARecord:
    """One registered persona LoRA in the pool.

    Fields:

    * ``record_id``         — pool-level identity
                              (``"figure:einstein:bundle-id-prefix"``).
    * ``figure_id``         — caller-supplied figure id.
    * ``source_bundle_id``  — figure-vertical bundle id the
                              artifact was baked from.
    * ``backend_id``        — backend identifier from the artifact
                              (``"synthetic-v1"`` etc.).
    * ``training_plan_hash``— hash of the LoRA training plan.
    * ``adapter_layers``    — frozen tuple of delta layers.
    * ``parameter_count``   — caller-reported total parameter count
                              (used as a capacity-cost signal at the
                              gate and as a sanity check at activate).
    * ``description``       — short human-readable description.
    * ``peft_checkpoint_dir`` — on-disk path to a saved
                                ``peft.save_pretrained`` snapshot
                                (debt #40 closure path); empty
                                string when the artifact has no
                                real PEFT checkpoint (synthetic
                                backend, or PEFT bake with the
                                checkpoint explicitly disabled).
                                When non-empty AND the runtime
                                supports :meth:`activate_peft_adapter`,
                                :meth:`activate` prefers it over
                                the projected ``adapter_layers``
                                summary forward hook (which is the
                                LayerNorm-eaten fallback path).
    """

    record_id: str
    figure_id: str
    source_bundle_id: str
    backend_id: str
    training_plan_hash: str
    adapter_layers: tuple[SubstrateDeltaAdapterLayer, ...]
    parameter_count: int
    description: str
    peft_checkpoint_dir: str = ""

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("PersonaLoRARecord.record_id must be non-empty")
        if not self.figure_id.strip():
            raise ValueError("PersonaLoRARecord.figure_id must be non-empty")
        if not self.source_bundle_id.strip():
            raise ValueError(
                "PersonaLoRARecord.source_bundle_id must be non-empty"
            )
        if not self.backend_id.strip():
            raise ValueError("PersonaLoRARecord.backend_id must be non-empty")
        if not self.training_plan_hash.strip():
            raise ValueError(
                "PersonaLoRARecord.training_plan_hash must be non-empty"
            )
        if not self.adapter_layers:
            raise ValueError(
                "PersonaLoRARecord.adapter_layers must be non-empty; the "
                "pool refuses to hold a degenerate record."
            )
        if self.parameter_count <= 0:
            raise ValueError(
                f"PersonaLoRARecord.parameter_count must be > 0, "
                f"got {self.parameter_count!r}"
            )


class PersonaLoRANotFound(LookupError):
    """Raised when a requested LoRA record id is not registered."""


class PersonaLoRAPool:
    """Process-level registry of persona LoRA records.

    Thread-safe through a single :class:`threading.Lock`; calls are
    infrequent compared to chat traffic so the lock is not a hot
    path.

    Future-S-LoRA contract:

    A real multi-LoRA inference backend will subclass this pool and
    override :meth:`activate` to push the selected record's adapter
    layers onto the GPU-resident frozen base. Until that lands,
    :meth:`activate` is a no-op that returns the record so callers
    can take read-only delta layers from it.
    """

    def __init__(self) -> None:
        self._records: dict[str, PersonaLoRARecord] = {}
        self._lock = threading.Lock()

    def register(
        self,
        *,
        figure_id: str,
        source_bundle_id: str,
        backend_id: str,
        training_plan_hash: str,
        adapter_layers: tuple[SubstrateDeltaAdapterLayer, ...],
        parameter_count: int,
        description: str = "",
        peft_checkpoint_dir: str = "",
    ) -> str:
        """Register a persona LoRA and return its ``record_id``.

        Re-registering with the same ``(figure_id, source_bundle_id)``
        replaces the previous record. The pool DOES NOT version-pin
        records — versioning is the figure-vertical's responsibility
        through :class:`FigureLoRAArtifact.integrity_hash`. Callers
        that need rollback must reattach the previous bundle by id.
        """

        record_id = _record_id_for(
            figure_id=figure_id, source_bundle_id=source_bundle_id
        )
        record = PersonaLoRARecord(
            record_id=record_id,
            figure_id=figure_id,
            source_bundle_id=source_bundle_id,
            backend_id=backend_id,
            training_plan_hash=training_plan_hash,
            adapter_layers=adapter_layers,
            parameter_count=parameter_count,
            description=(
                description
                or (
                    f"Persona LoRA for figure {figure_id} "
                    f"(bundle={source_bundle_id}, backend={backend_id})"
                )
            ),
            peft_checkpoint_dir=peft_checkpoint_dir,
        )
        with self._lock:
            self._records[record_id] = record
            self._records[figure_id] = record
        return record_id

    def lookup(self, record_or_figure_id: str) -> PersonaLoRARecord:
        """Return the record for ``record_id`` or ``figure_id``.

        Raises :class:`PersonaLoRANotFound` if no matching record is
        registered. Fail-loud per
        ``no-swallow-errors-no-hasattr-abuse.mdc``.
        """

        if not record_or_figure_id:
            raise ValueError(
                "PersonaLoRAPool.lookup: id must be non-empty"
            )
        record = self._records.get(record_or_figure_id)
        if record is None:
            raise PersonaLoRANotFound(
                f"PersonaLoRAPool: no record registered for "
                f"{record_or_figure_id!r}"
            )
        return record

    def has(self, record_or_figure_id: str) -> bool:
        """Whether a record can be resolved (id or figure_id)."""

        return bool(record_or_figure_id) and record_or_figure_id in self._records

    def lookup_layers(
        self, record_or_figure_id: str
    ) -> tuple[SubstrateDeltaAdapterLayer, ...]:
        """Convenience: return the adapter layers of a registered record."""

        return self.lookup(record_or_figure_id).adapter_layers

    def activate(
        self,
        record_or_figure_id: str,
        *,
        runtime: LoRAAwareResidualRuntime | None = None,
    ) -> AbstractContextManager[PersonaLoRARecord]:
        """Activate ``record_or_figure_id``'s persona LoRA on ``runtime``.

        Returns a **context manager** that yields the
        :class:`PersonaLoRARecord` for the duration of the
        activation. On context exit, the runtime restores the
        pre-activation forward path byte-identically (the LoRA
        delta is no longer applied to subsequent forward calls).

        When ``runtime`` is ``None``, the context manager still
        yields the record so calling code is uniform: the legacy
        passthrough behaviour is preserved (registered LoRA in
        memory, no real forward effect — useful for SHADOW /
        diagnostic paths).

        When ``runtime`` is supplied, it MUST satisfy
        :class:`LoRAAwareResidualRuntime` (have an ``activate_lora``
        method). The pool calls
        ``runtime.activate_lora(record.adapter_layers)`` and
        re-yields the record inside that context. This is the
        load-bearing path for debt #20: the persona's adapter
        delta is added to the runtime's residual stream for the
        whole context, and removed on exit.

        Implementations are responsible for guaranteeing that the
        frozen base is not mutated; tests assert byte-identical
        ``state_dict`` before / after the context.
        """

        record = self.lookup(record_or_figure_id)
        return _persona_lora_activation_context(
            record=record, runtime=runtime
        )

    def deregister(self, record_or_figure_id: str) -> None:
        """Remove a record from the pool (rollback path).

        No-op if the record is not present (rollback is idempotent).
        """

        if not record_or_figure_id:
            return
        with self._lock:
            removed = self._records.pop(record_or_figure_id, None)
            if removed is None:
                return
            secondary_keys = [
                key for key, value in self._records.items() if value is removed
            ]
            for key in secondary_keys:
                self._records.pop(key, None)

    def keys(self) -> tuple[str, ...]:
        """Snapshot of registered ids (for diagnostics only)."""

        with self._lock:
            return tuple(self._records.keys())


_DEFAULT_POOL: PersonaLoRAPool | None = None
_DEFAULT_POOL_LOCK = threading.Lock()


def default_pool() -> PersonaLoRAPool:
    """Return the process-wide default :class:`PersonaLoRAPool`.

    The default pool is lazily initialised on first access and
    starts empty. The lifeform-service adopt path is the only
    intended caller of :meth:`PersonaLoRAPool.register`; tests that
    want isolation should construct their own pool instance instead
    of mutating the default.
    """

    global _DEFAULT_POOL
    if _DEFAULT_POOL is not None:
        return _DEFAULT_POOL
    with _DEFAULT_POOL_LOCK:
        if _DEFAULT_POOL is not None:
            return _DEFAULT_POOL
        _DEFAULT_POOL = PersonaLoRAPool()
    return _DEFAULT_POOL


def _record_id_for(*, figure_id: str, source_bundle_id: str) -> str:
    suffix = source_bundle_id
    if ":" in source_bundle_id:
        suffix = source_bundle_id.split(":")[-1][:16]
    return f"persona-lora:{figure_id}:{suffix}"


@contextlib.contextmanager
def _persona_lora_activation_context(
    *,
    record: PersonaLoRARecord,
    runtime: LoRAAwareResidualRuntime | None,
) -> Iterator[PersonaLoRARecord]:
    """Yield ``record`` for the lifetime of an optional runtime LoRA activation.

    Two activation backends, picked at runtime:

    1. **Real PEFT adapter** (debt #40 closure path): when ``record``
       has a non-empty ``peft_checkpoint_dir`` AND ``runtime`` exposes
       :meth:`activate_peft_adapter`, the pool loads the saved peft
       checkpoint into the base model for the context. The trained
       A/B matrices are applied through the target_module linears'
       forward, so ``BUNDLE`` and ``BUNDLE_LORA`` produce observably
       different outputs in real Qwen.

    2. **Projected adapter_layers summary** (legacy path): when the
       record has no checkpoint OR the runtime does not implement
       ``activate_peft_adapter``, the pool falls back to
       :meth:`activate_lora`'s forward hook that adds the projected
       ``delta_vector`` to the residual stream. This path is the one
       LayerNorm zeroes out — kept for back-compat with the synthetic
       backend and with bundles baked before the checkpoint field
       existed.

    The ``runtime is None`` path stays a pure passthrough — it
    preserves the legacy behaviour where a process-level pool was
    just an artifact registry.
    """

    if runtime is None:
        yield record
        return
    peft_checkpoint_dir = record.peft_checkpoint_dir
    activate_peft = getattr(runtime, "activate_peft_adapter", None)
    if peft_checkpoint_dir and callable(activate_peft):
        with activate_peft(peft_checkpoint_dir):
            yield record
        return
    activate = getattr(runtime, "activate_lora", None)
    if not callable(activate):
        raise TypeError(
            "PersonaLoRAPool.activate: runtime does not satisfy "
            "LoRAAwareResidualRuntime (no callable activate_lora). "
            f"Got {type(runtime).__name__!r}."
        )
    with activate(record.adapter_layers):
        yield record


__all__ = [
    "PersonaLoRANotFound",
    "PersonaLoRAPool",
    "PersonaLoRARecord",
    "default_pool",
]
