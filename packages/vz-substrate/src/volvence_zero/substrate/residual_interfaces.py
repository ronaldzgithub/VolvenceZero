"""Substrate residual-backend ABCs.

Defines :class:`OpenWeightResidualRuntime` (the frozen-LLM residual
capture + intervention contract every concrete runtime implements) and
:class:`ResidualInterventionBackend` (the intervention strategy that
runtimes plug in to apply controller-code deltas to the residual
stream). Both are abstract and pure-interface; concrete implementations
live in sibling modules.

Slice S.3 (2026-05-04): extracted from the previous monolithic
``residual_backend.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from volvence_zero.substrate.adapter import (
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
)

from volvence_zero.substrate.residual_contracts import (
    GenerationResult,
    OpenWeightRuntimeCapture,
    ResidualControlApplication,
    SubstrateDeltaAdapterLayer,
    SubstrateOnlineFastCheckpoint,
    SubstrateRareHeavyCheckpoint,
    TrainingTrace,
)


class OpenWeightResidualRuntime(ABC):
    """Hook-ready runtime contract for frozen open-weight residual access."""

    model_id: str
    is_frozen: bool
    runtime_origin: str = "unknown"
    supports_live_substrate_mutation: bool = False
    supports_offline_substrate_training: bool = False

    @abstractmethod
    def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        """Capture a frozen-model residual snapshot for the given source text."""

    def capture_for_contrastive(
        self,
        *,
        positive_texts: tuple[str, ...],
        negative_texts: tuple[str, ...],
        layer_index: int = 0,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Mean-pool residuals at ``layer_index`` for two text groups.

        Used by the figure-vertical F5 steering bake (debt #21
        closure) to derive a contrastive direction in the
        substrate's actual residual coordinate system rather than
        in a hash-derived embedding space. The default
        implementation walks ``capture(source_text=...)`` for each
        text in each group and averages the activation vector at
        ``layer_index``; concrete subclasses are free to override
        with a more efficient batched path (the transformers
        runtime does so via ``_capture_hidden_state_means``).

        Returns a ``(positive_mean, negative_mean)`` pair of
        ``tuple[float, ...]`` whose dimensionality matches the
        runtime's residual activation width at the requested
        layer. Both texts groups must be non-empty; both means
        must match in dimension (the runtime is responsible for
        producing consistent activation widths across captures).
        """

        if not positive_texts:
            raise ValueError(
                "capture_for_contrastive: positive_texts must be non-empty"
            )
        if not negative_texts:
            raise ValueError(
                "capture_for_contrastive: negative_texts must be non-empty"
            )
        positive_mean = self._mean_residual_at_layer(
            texts=positive_texts, layer_index=layer_index
        )
        negative_mean = self._mean_residual_at_layer(
            texts=negative_texts, layer_index=layer_index
        )
        if len(positive_mean) != len(negative_mean):
            raise RuntimeError(
                "capture_for_contrastive: positive and negative residual "
                f"means have different widths ({len(positive_mean)} vs "
                f"{len(negative_mean)}); the runtime produced "
                "inconsistent activation shapes."
            )
        return (positive_mean, negative_mean)

    def _mean_residual_at_layer(
        self,
        *,
        texts: tuple[str, ...],
        layer_index: int,
    ) -> tuple[float, ...]:
        """Average the residual activation at ``layer_index`` across ``texts``.

        Subclasses MAY override for batched / GPU-resident paths;
        the default uses the public :meth:`capture` so synthetic
        runtimes work without further wiring.
        """

        sums: list[float] | None = None
        sample_count = 0
        for text in texts:
            capture = self.capture(source_text=text)
            activation = _extract_activation_at_layer(
                capture=capture, layer_index=layer_index
            )
            if sums is None:
                sums = list(activation)
            else:
                if len(activation) != len(sums):
                    raise RuntimeError(
                        "capture_for_contrastive: activation width drifted "
                        f"across texts ({len(activation)} vs {len(sums)}); "
                        "the runtime produced inconsistent activation shapes."
                    )
                for index, value in enumerate(activation):
                    sums[index] += value
            sample_count += 1
        if not sums or sample_count == 0:
            raise RuntimeError(
                "capture_for_contrastive: produced empty activation pool; "
                "either every text yielded zero activations or the runtime "
                "is misconfigured."
            )
        return tuple(value / sample_count for value in sums)

    @abstractmethod
    def apply_control(
        self,
        *,
        source_text: str,
        substrate_snapshot: SubstrateSnapshot,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> ResidualControlApplication:
        """Apply bounded residual intervention through the runtime."""

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
        generation_constraints: "GenerationConstraints | None" = None,
        capture_residuals: bool = True,
    ) -> GenerationResult:
        """Generate text using the underlying model.

        Subclasses that hold a real model override this to run
        autoregressive decoding.  The default implementation returns
        a placeholder so synthetic runtimes remain functional.

        ``capture_residuals=False`` lets pass-through callers (the raw
        ablation track) skip building the runtime residual capture; real
        backends may honour it to avoid the expensive post-generate
        re-forward. The default placeholder ignores it.
        """
        del generation_constraints, capture_residuals
        return GenerationResult(
            text=f"[generation not supported by {self.model_id}]",
            token_count=0,
            capture=None,
            description=f"{self.model_id} does not support generation",
        )

    @property
    def fallback_active(self) -> bool:
        return self.runtime_origin in {"builtin-fallback", "synthetic-open-weight"}

    @property
    def capture_source(self) -> str:
        return "real" if not self.fallback_active else "fallback"

    @property
    def experimental_live_mutation_enabled(self) -> bool:
        return self.supports_live_substrate_mutation

    @property
    def live_mutation_mode(self) -> str:
        return "experimental-live" if self.experimental_live_mutation_enabled else "frozen-review-only"

    def require_live_substrate_mutation(self, *, operation: str) -> None:
        if not self.supports_live_substrate_mutation:
            raise RuntimeError(
                f"{type(self).__name__} blocks {operation} because the default frozen-substrate doctrine "
                "does not allow live substrate mutation. Keep the runtime on capture/control/generate only, "
                "or opt into explicit experimental live mutation."
            )

    def require_offline_substrate_training(self, *, operation: str) -> None:
        if not self.supports_offline_substrate_training:
            raise RuntimeError(
                f"{type(self).__name__} blocks {operation} because this runtime is not marked as an "
                "offline substrate-artifact owner. Clone the runtime for rare-heavy/offline training first."
            )

    def require_substrate_artifact_import(self, *, operation: str) -> None:
        if self.supports_live_substrate_mutation or self.supports_offline_substrate_training:
            return
        raise RuntimeError(
            f"{type(self).__name__} blocks {operation} because importing substrate artifacts into the live "
            "runtime is disabled under the frozen-substrate doctrine. Use an offline clone or explicit "
            "experimental live mutation mode."
        )

    def export_rare_heavy_state(self, *, checkpoint_id: str | None = None) -> SubstrateRareHeavyCheckpoint:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rare-heavy substrate export."
        )

    def import_rare_heavy_state(self, checkpoint: SubstrateRareHeavyCheckpoint) -> tuple[str, ...]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rare-heavy substrate import."
        )

    def restore_rare_heavy_state(self, checkpoint: SubstrateRareHeavyCheckpoint) -> tuple[str, ...]:
        return self.import_rare_heavy_state(checkpoint)

    def train_rare_heavy(
        self,
        *,
        traces: tuple[TrainingTrace, ...] = (),
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
        checkpoint_id: str | None = None,
    ) -> SubstrateRareHeavyCheckpoint:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rare-heavy substrate training."
        )

    def clone_for_rare_heavy(self) -> "OpenWeightResidualRuntime":
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rare-heavy runtime cloning."
        )

    def export_online_fast_state(
        self,
        *,
        checkpoint_id: str | None = None,
    ) -> SubstrateOnlineFastCheckpoint:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement online-fast substrate export."
        )

    def apply_online_fast_state(self, checkpoint: SubstrateOnlineFastCheckpoint) -> tuple[str, ...]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement online-fast substrate apply."
        )

    def restore_online_fast_state(self, checkpoint: SubstrateOnlineFastCheckpoint) -> tuple[str, ...]:
        return self.apply_online_fast_state(checkpoint)

    @contextmanager
    def activate_lora(
        self,
        layers: tuple[SubstrateDeltaAdapterLayer, ...],
    ) -> Iterator[None]:
        """Activate persona LoRA ``layers`` for the duration of context.

        Default implementation is a no-op + accounting only: it
        records the activation on the runtime instance for tests
        / audit and yields without mutating the forward path.
        Concrete real-forward runtimes (e.g.,
        :class:`TransformersOpenWeightResidualRuntime`) override
        this to register forward hooks that add the deltas to the
        residual stream and remove them on exit.

        The default implementation enforces:

        1. ``layers`` is a non-empty tuple of
           :class:`SubstrateDeltaAdapterLayer`.
        2. Re-entry is rejected (raises ``RuntimeError``) so
           persona conflicts are loud rather than silently merged.
        3. On exit (normal OR exception) the activation flag is
           cleared.

        R2: this never mutates the frozen base. Synthetic
        runtimes have no forward to mutate; real runtimes apply
        a delta in the controller layer only.
        """

        if not layers:
            raise ValueError(
                "activate_lora: layers tuple must be non-empty (refusing "
                "to activate an empty persona)."
            )
        if getattr(self, "_lora_activation_in_flight", False):
            raise RuntimeError(
                "activate_lora: nested activation detected; the runtime "
                "is already inside an activate_lora context. Exit the "
                "outer context before activating a different persona."
            )
        # Use object.__setattr__ so frozen / dataclass-mixin runtimes
        # do not reject the bookkeeping write; this is intentional —
        # the activation flag is runtime state, not configuration.
        try:
            object.__setattr__(self, "_lora_activation_in_flight", True)
            object.__setattr__(self, "_lora_activation_layers", tuple(layers))
            yield
        finally:
            object.__setattr__(self, "_lora_activation_in_flight", False)
            object.__setattr__(self, "_lora_activation_layers", ())


class ResidualInterventionBackend(ABC):
    """Bounded owner-side backend for residual control application."""

    name: str

    @abstractmethod
    def apply_control(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> ResidualControlApplication:
        """Apply bounded residual control and return the resulting effect."""


def _extract_activation_at_layer(
    *,
    capture: OpenWeightRuntimeCapture,
    layer_index: int,
) -> tuple[float, ...]:
    """Pull the activation tuple at ``layer_index`` from a capture.

    Used by :meth:`OpenWeightResidualRuntime.capture_for_contrastive`
    to walk the captured residuals and average them. We prefer the
    primary :attr:`OpenWeightRuntimeCapture.residual_activations`
    list and fall back to the per-step ``residual_sequence``
    representation when needed (synthetic captures populate the
    sequence first; transformers captures populate both).
    """

    for activation in capture.residual_activations:
        if activation.layer_index == layer_index and activation.activation:
            return tuple(activation.activation)
    if capture.residual_sequence:
        latest_step = capture.residual_sequence[-1]
        for activation in latest_step.residual_activations:
            if activation.layer_index == layer_index and activation.activation:
                return tuple(activation.activation)
    raise RuntimeError(
        f"capture_for_contrastive: no activation tuple at layer_index="
        f"{layer_index!r}; the runtime did not capture this layer."
    )


