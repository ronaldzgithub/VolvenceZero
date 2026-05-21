"""F6 LoRA artifact schema + abstract backend interface.

The figure vertical produces persona LoRA artifacts through one of
several backends:

* :class:`lifeform_domain_figure.lora_bake_synthetic.SyntheticLoRABakeBackend`
  — deterministic, dependency-free, byte-for-byte reproducible
  synthetic delta. Used in SHADOW mode and in tests; the bake job
  runs in milliseconds with no torch / GPU dependency.
* :class:`lifeform_domain_figure.lora_bake_peft.PEFTLoRABakeBackend`
  — interface shell pinned for a future real PEFT-backed
  implementation; raises ``NotImplementedError`` until the F6.X
  wire-up packet lands.

Both backends consume a :class:`LoRATrainingPlan` (P6.1) and emit a
frozen :class:`FigureLoRAArtifact` whose ``adapter_layers`` are
shape-identical to the kernel's
:class:`volvence_zero.substrate.SubstrateDeltaAdapterLayer`. This
identical shape is what lets the runtime persona-LoRA pool (P6.3)
swap artifacts on the same frozen base without invasive kernel
changes (R2 — adaptation in the controller layer, base frozen).

R8 / R15: the artifact carries an ``integrity_hash`` that a
:class:`PersonaLoRAProposal` (P6.1) pins as ``new_value_hash``;
rolling back means re-attaching the previous artifact's id.
"""

from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass

from volvence_zero.substrate import SubstrateDeltaAdapterLayer

from lifeform_domain_figure.lora_data_prep import LoRATrainingPlan


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FigureLoRAArtifact:
    """Frozen, integrity-hashed persona LoRA artifact.

    Produced by a :class:`LoRABakeBackend` and bound into
    :attr:`FigureArtifactBundle.lora` via
    :func:`attach_baked_lora`. The artifact is the single object
    that crosses the wheel boundary; the persona-LoRA pool (P6.3)
    consumes its :attr:`adapter_layers` and never reaches into
    backend-internal state.

    ``peft_checkpoint_dir`` (optional, default ``""``): on-disk path
    to a real ``peft.save_pretrained(...)`` snapshot — populated by
    :class:`lifeform_domain_figure.lora_bake_peft.PEFTLoRABakeBackend`
    when a real PEFT loop ran. The runtime's
    :meth:`LoRAAwareResidualRuntime.activate_peft_adapter` consumes
    this path to apply the trained LoRA via
    ``peft.PeftModel.from_pretrained`` at inference time (debt #40
    closure: the projected ``adapter_layers`` summary vector is
    eaten by LayerNorm in real Qwen forward; loading the actual
    A/B matrices through peft restores the trained delta).

    The checkpoint path is **not** included in
    :func:`compute_lora_integrity_hash` because the path is
    platform-specific while the artifact's logical identity
    (figure, backend, plan, projected layers) is portable. Two
    machines that bake the same plan emit byte-identical
    ``integrity_hash`` even when they save the checkpoint to
    different local directories.
    """

    schema_version: int
    figure_id: str
    backend_id: str
    rank: int
    target_layer_index: int
    adapter_layers: tuple[SubstrateDeltaAdapterLayer, ...]
    training_plan_hash: str
    integrity_hash: str
    parameter_count: int
    description: str
    peft_checkpoint_dir: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"FigureLoRAArtifact.schema_version mismatch: "
                f"got {self.schema_version!r}, expected {SCHEMA_VERSION!r}"
            )
        if not self.figure_id.strip():
            raise ValueError("FigureLoRAArtifact.figure_id must be non-empty")
        if not self.backend_id.strip():
            raise ValueError("FigureLoRAArtifact.backend_id must be non-empty")
        if self.rank <= 0:
            raise ValueError(
                f"FigureLoRAArtifact.rank must be > 0, got {self.rank!r}"
            )
        if self.target_layer_index < 0:
            raise ValueError(
                f"FigureLoRAArtifact.target_layer_index must be >= 0, "
                f"got {self.target_layer_index!r}"
            )
        if not self.adapter_layers:
            raise ValueError(
                "FigureLoRAArtifact.adapter_layers must be non-empty; "
                "refusing to ship a degenerate LoRA artifact."
            )
        if not self.training_plan_hash.strip():
            raise ValueError(
                "FigureLoRAArtifact.training_plan_hash must be non-empty"
            )
        if not self.integrity_hash.strip():
            raise ValueError(
                "FigureLoRAArtifact.integrity_hash must be non-empty"
            )
        if self.parameter_count <= 0:
            raise ValueError(
                f"FigureLoRAArtifact.parameter_count must be > 0, "
                f"got {self.parameter_count!r}"
            )

    @property
    def total_layers(self) -> int:
        return len(self.adapter_layers)


class LoRABakeBackend(abc.ABC):
    """Abstract base for figure persona LoRA bake backends.

    Concrete backends are responsible for:

    * Reading a :class:`LoRATrainingPlan`.
    * Producing a :class:`FigureLoRAArtifact` with adapter layers
      shape-identical to
      :class:`volvence_zero.substrate.SubstrateDeltaAdapterLayer`.
    * Pinning ``training_plan_hash`` to ``plan.integrity_hash`` so
      the gate-side proposal (:class:`PersonaLoRAProposal`) can
      re-verify provenance.
    """

    @property
    @abc.abstractmethod
    def backend_id(self) -> str:
        """A short stable identifier (``"synthetic-v1"`` etc.)."""

    @abc.abstractmethod
    def bake(self, plan: LoRATrainingPlan) -> FigureLoRAArtifact:
        """Bake a :class:`FigureLoRAArtifact` from a training plan."""


def compute_lora_integrity_hash(
    *,
    figure_id: str,
    backend_id: str,
    training_plan_hash: str,
    adapter_layers: tuple[SubstrateDeltaAdapterLayer, ...],
) -> str:
    """Deterministic SHA-256 over the LoRA artifact's identity fields."""

    payload = (
        SCHEMA_VERSION,
        figure_id,
        backend_id,
        training_plan_hash,
        tuple(
            (
                layer.layer_index,
                tuple(round(v, 6) for v in layer.delta_vector),
                round(layer.mean_abs_delta, 6),
                layer.description,
            )
            for layer in adapter_layers
        ),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "FigureLoRAArtifact",
    "LoRABakeBackend",
    "compute_lora_integrity_hash",
]
