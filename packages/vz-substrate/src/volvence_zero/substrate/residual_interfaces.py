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
from typing import Any, Sequence

from volvence_zero.substrate.adapter import (
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
)

from volvence_zero.substrate.residual_contracts import (
    GenerationResult,
    OpenWeightRuntimeCapture,
    ResidualControlApplication,
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
    ) -> GenerationResult:
        """Generate text using the underlying model.

        Subclasses that hold a real model override this to run
        autoregressive decoding.  The default implementation returns
        a placeholder so synthetic runtimes remain functional.
        """
        del generation_constraints
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


