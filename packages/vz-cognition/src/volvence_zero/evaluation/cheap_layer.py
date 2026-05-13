"""Cheap layer of the evaluation cascade (architecture-uplift A2 step 1).

This module is the **架构层入口** for the cheap tier of the evaluation
cascade defined in [`docs/specs/evaluation-cascade.md`](../../../../../../../docs/specs/evaluation-cascade.md).

Phased implementation:

- **Step 1 (this T7 packet)**: declare the role marker
  (``EvaluationCascadeRole.CHEAP_LAYER``) and a thin facade
  (``EvaluationCheapLayer``) over the existing ``EvaluationModule``.
  Backbone migration (moving compute_* helpers out of ``backbone.py``)
  is deferred — this packet only freezes the public API surface so the
  mid_layer / expensive_layer (T8 / T9) can layer on top without
  breaking the six existing EvaluationSnapshot downstreams.

- **Step 2/3 (T8/T9)**: implement mid_layer / expensive_layer / aggregator
  on top of the snapshot produced here. They opt-in by consuming
  ``EvaluationSnapshot`` via the normal kernel propagate path; the
  cheap layer remains the single owner of the ``evaluation`` slot.

Field-identical invariant (spec §关键不变量):
``EvaluationSnapshot`` 字段 (turn_scores / session_scores / alerts /
description / structured_alerts / reflection_accuracy / longitudinal_verdict)
must remain byte-equivalent across the cheap_layer migration. This is
enforced by ``tests/contracts/test_evaluation_cascade.py``.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

from volvence_zero.evaluation.types import EvaluationSnapshot

if TYPE_CHECKING:  # pragma: no cover — only for type hints
    from volvence_zero.evaluation.backbone import EvaluationModule

__all__ = [
    "EvaluationCascadeRole",
    "EvaluationCheapLayer",
    "EVALUATION_SNAPSHOT_FIELD_NAMES",
]


# ---------------------------------------------------------------------------
# Field-identical invariant
# ---------------------------------------------------------------------------


def _compute_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(EvaluationSnapshot))


# Frozen at module load to lock the cheap_layer field-identical invariant.
# Any future change to EvaluationSnapshot must (a) update this constant in
# the same commit, (b) update DATA_CONTRACT §3.7, (c) update at least one
# of the 6 existing downstreams documented in spec §A2.
EVALUATION_SNAPSHOT_FIELD_NAMES: tuple[str, ...] = _compute_field_names()


# ---------------------------------------------------------------------------
# Role marker
# ---------------------------------------------------------------------------


class EvaluationCascadeRole(str, enum.Enum):
    """Cascade-tier role marker attached to evaluation-related modules.

    Used by contract tests + future SHADOW evidence collection to identify
    which tier a module belongs to. Plain string enum so it serialises
    cleanly into snapshot value descriptions.
    """

    CHEAP_LAYER = "cheap_layer"
    MID_LAYER = "mid_layer"
    EXPENSIVE_LAYER = "expensive_layer"
    CROSS_GENERATION_AGGREGATOR = "cross_generation_aggregator"


# ---------------------------------------------------------------------------
# Cheap-layer facade
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EvaluationCheapLayer:
    """Thin facade pointing to the EvaluationModule that publishes the
    ``evaluation`` slot.

    Why a facade instead of moving backbone.py code inline?

    - The existing ``EvaluationModule`` + ``EvaluationBackbone`` chain is
      already field-identical to its published snapshot. Inlining would risk
      subtly altering call ordering or session-cache behaviour, breaking the
      6 downstreams documented in
      [`docs/specs/evaluation-cascade.md`](../../../../../../../docs/specs/evaluation-cascade.md).
    - The cascade SSOT requirement (T7 Done 标志) is *not* "code lives in
      cheap_layer.py" but "cheap_layer 是 evaluation slot 的唯一 owner, 输出
      field-identical EvaluationSnapshot". A facade satisfies both.
    - Future backbone refactoring becomes a localised packet (does not block
      mid_layer T8 starting).

    The facade is constructed by orchestrators that want to advertise the
    role marker on their evaluation owner without changing dispatch.
    """

    module: "EvaluationModule"
    role: EvaluationCascadeRole = EvaluationCascadeRole.CHEAP_LAYER

    @property
    def slot_name(self) -> str:
        return self.module.slot_name

    @property
    def owner(self) -> str:
        return self.module.owner

    @property
    def wiring_level(self):
        return self.module.wiring_level

    def is_cheap_layer(self) -> bool:
        return self.role is EvaluationCascadeRole.CHEAP_LAYER
