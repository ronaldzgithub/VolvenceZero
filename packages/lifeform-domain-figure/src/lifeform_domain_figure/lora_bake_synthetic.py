"""F6 / P6.2 — synthetic LoRA bake backend.

Deterministic, dependency-free implementation of
:class:`LoRABakeBackend`. The backend converts a
:class:`LoRATrainingPlan` into a :class:`FigureLoRAArtifact` whose
``adapter_layers`` carry hash-derived delta vectors that:

* Are shape-identical to
  :class:`volvence_zero.substrate.SubstrateDeltaAdapterLayer`
  (so the persona-LoRA pool consumes them through the same
  surface as kernel-emitted layers).
* Are reproducible byte-for-byte from the same training plan
  (R15 rollback contract).
* Carry per-layer descriptions that include the figure id,
  backend id, plan hash prefix, and rank — every layer is
  self-describing for audit purposes.

The synthetic backend does NOT attempt to learn anything from the
training plan; its purpose is to give SHADOW deployments and tests
a real artifact whose identity the gate / pool / adopt path can
exercise end-to-end. The real PEFT-backed backend
(``lora_bake_peft``) is a separate concern; the contract surface
they share is :class:`LoRABakeBackend`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from volvence_zero.substrate import SubstrateDeltaAdapterLayer

from lifeform_domain_figure.compiler import attach_lora_to_bundle
from lifeform_domain_figure.figure_artifact import FigureArtifactBundle
from lifeform_domain_figure.lora_artifact import (
    SCHEMA_VERSION,
    FigureLoRAArtifact,
    LoRABakeBackend,
    compute_lora_integrity_hash,
)
from lifeform_domain_figure.lora_data_prep import LoRATrainingPlan


_BACKEND_ID = "synthetic-v1"
_DEFAULT_VECTOR_DIM = 32


@dataclass(frozen=True)
class SyntheticLoRABakeBackend(LoRABakeBackend):
    """Deterministic synthetic LoRA bake backend.

    Optional fields:

    * ``vector_dim`` — width of each emitted delta vector (default
      :data:`_DEFAULT_VECTOR_DIM`). The kernel's
      :class:`SubstrateDeltaAdapterLayer` shape is intentionally a
      ``tuple[float, ...]`` of arbitrary length, so the synthetic
      backend picks a small, fixed dimension that is large enough
      to be non-trivial in tests but cheap in storage.
    * ``layer_count_per_rank`` — number of adapter layers emitted per
      rank unit (default ``1``). Synthetic deltas are interpretable
      as one layer per rank dimension; real PEFT artifacts will
      typically emit one layer per attention block.
    """

    vector_dim: int = _DEFAULT_VECTOR_DIM
    layer_count_per_rank: int = 1

    def __post_init__(self) -> None:
        if self.vector_dim <= 0:
            raise ValueError(
                f"SyntheticLoRABakeBackend.vector_dim must be > 0, "
                f"got {self.vector_dim!r}"
            )
        if self.layer_count_per_rank <= 0:
            raise ValueError(
                f"SyntheticLoRABakeBackend.layer_count_per_rank must be > 0, "
                f"got {self.layer_count_per_rank!r}"
            )

    @property
    def backend_id(self) -> str:
        return _BACKEND_ID

    def bake(self, plan: LoRATrainingPlan) -> FigureLoRAArtifact:
        """Synthesise a deterministic :class:`FigureLoRAArtifact`.

        The bake derives every adapter layer from a SHA-256 hash
        seeded by ``(plan.integrity_hash, layer_offset)``, so the
        same plan in produces the same artifact bytes out
        regardless of host / Python build / wall clock.
        """

        layers: list[SubstrateDeltaAdapterLayer] = []
        total_layer_count = plan.rank * self.layer_count_per_rank
        for offset in range(total_layer_count):
            delta_vector = self._derive_delta_vector(
                plan.integrity_hash, offset
            )
            layers.append(
                SubstrateDeltaAdapterLayer(
                    layer_index=plan.target_layer_index + offset,
                    delta_vector=delta_vector,
                    mean_abs_delta=_mean_abs(delta_vector),
                    description=(
                        f"figure-persona-lora:{plan.figure_id}:"
                        f"backend={_BACKEND_ID}:plan="
                        f"{plan.integrity_hash[:8]}:offset={offset}"
                    ),
                )
            )
        adapter_layers = tuple(layers)
        integrity_hash = compute_lora_integrity_hash(
            figure_id=plan.figure_id,
            backend_id=_BACKEND_ID,
            training_plan_hash=plan.integrity_hash,
            adapter_layers=adapter_layers,
        )
        return FigureLoRAArtifact(
            schema_version=SCHEMA_VERSION,
            figure_id=plan.figure_id,
            backend_id=_BACKEND_ID,
            rank=plan.rank,
            target_layer_index=plan.target_layer_index,
            adapter_layers=adapter_layers,
            training_plan_hash=plan.integrity_hash,
            integrity_hash=integrity_hash,
            parameter_count=total_layer_count * self.vector_dim,
            description=(
                f"Synthetic persona LoRA for {plan.figure_id} "
                f"({total_layer_count} layer(s) of width {self.vector_dim}); "
                f"deterministic bake from training plan "
                f"{plan.integrity_hash[:8]}."
            ),
        )

    def _derive_delta_vector(
        self, plan_hash: str, layer_offset: int
    ) -> tuple[float, ...]:
        """Derive a ``vector_dim``-dimensional delta from a hash seed."""

        seed = f"{plan_hash}:{layer_offset}".encode("utf-8")
        digest = hashlib.shake_256(seed).digest(self.vector_dim * 2)
        values: list[float] = []
        for index in range(self.vector_dim):
            byte_pair = int.from_bytes(
                digest[index * 2 : index * 2 + 2], "big", signed=False
            )
            value = (byte_pair / 65535.0) - 0.5
            values.append(value)
        return tuple(values)


def attach_baked_lora(
    bundle: FigureArtifactBundle,
    artifact: FigureLoRAArtifact,
) -> FigureArtifactBundle:
    """Attach a baked LoRA artifact to a bundle (re-keys bundle id).

    Thin wrapper over :func:`attach_lora_to_bundle` that keeps the
    F6 callsite from importing both modules.
    """

    if artifact.figure_id != bundle.figure_id:
        raise ValueError(
            f"attach_baked_lora: artifact.figure_id={artifact.figure_id!r} "
            f"does not match bundle.figure_id={bundle.figure_id!r}"
        )
    return attach_lora_to_bundle(
        bundle,
        lora=artifact,
        lora_integrity=artifact.integrity_hash,
    )


def _mean_abs(vector: tuple[float, ...]) -> float:
    if not vector:
        return 0.0
    return sum(abs(value) for value in vector) / len(vector)


__all__ = [
    "SyntheticLoRABakeBackend",
    "attach_baked_lora",
]
