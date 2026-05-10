"""F6 / P6.2 — PEFT-backed LoRA bake backend (interface stub).

Pinned :class:`LoRABakeBackend` implementation that **does not yet
run**. The contract surface is fixed (input: :class:`LoRATrainingPlan`,
output: :class:`FigureLoRAArtifact` with kernel-shape adapter
layers) so a future packet can land the actual GPU-backed PEFT
training without rippling changes through the figure vertical's
public surface.

Until the real wire-up lands, :meth:`bake` raises
``NotImplementedError`` with a precise message naming the
upstream packet that will activate it. This is intentional: the
SHADOW deployment path uses :class:`SyntheticLoRABakeBackend` from
:mod:`lifeform_domain_figure.lora_bake_synthetic`, so the absence
of a real PEFT bake never silently degrades behaviour — it only
prevents an operator from accidentally configuring a backend that
cannot run.

Why a separate file rather than a flag in the synthetic backend:

* Keeps the synthetic backend's surface free of "real-or-not" forks
  that violate ``no-keyword-matching-hacks.mdc``.
* Makes the future PEFT packet's diff small and reviewable: it
  replaces this file's :meth:`bake` body with the real
  implementation, leaves the dataclass fields and ``backend_id``
  in place, and the rest of the figure vertical does not move.
"""

from __future__ import annotations

from dataclasses import dataclass

from lifeform_domain_figure.lora_artifact import (
    FigureLoRAArtifact,
    LoRABakeBackend,
)
from lifeform_domain_figure.lora_data_prep import LoRATrainingPlan


_BACKEND_ID = "peft-v1-stub"


@dataclass(frozen=True)
class PEFTLoRABakeBackend(LoRABakeBackend):
    """Interface-pinned PEFT-backed LoRA bake backend.

    Future fields (kept in mind for the real wire-up):

    * ``model_id``       — HuggingFace model id of the frozen base.
    * ``peft_config``    — PEFT LoRAConfig serialised as a typed
                           dataclass (target_modules / r / alpha /
                           dropout).
    * ``runtime_device`` — explicit ``"cpu"`` / ``"cuda"`` switch.
    * ``checkpoint_dir`` — out-of-process artifact staging dir.

    None of these are present yet — adding them as public fields
    before the real implementation lands risks freezing the wrong
    surface. The future packet introduces them together with the
    real :meth:`bake` body in one reviewable change.
    """

    model_id: str = ""

    @property
    def backend_id(self) -> str:
        return _BACKEND_ID

    def bake(self, plan: LoRATrainingPlan) -> FigureLoRAArtifact:
        raise NotImplementedError(
            "PEFTLoRABakeBackend.bake is not yet wired up. The interface is "
            "pinned so the figure vertical's public surface is stable; the "
            "real GPU-backed implementation lands in a future F6.X packet "
            "(see docs/specs/figure-vertical.md). For SHADOW deployments "
            "and tests, use SyntheticLoRABakeBackend from "
            "lifeform_domain_figure.lora_bake_synthetic. plan.figure_id="
            f"{plan.figure_id!r} was about to be baked."
        )


__all__ = [
    "PEFTLoRABakeBackend",
]
