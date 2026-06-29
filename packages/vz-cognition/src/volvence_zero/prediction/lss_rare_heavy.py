"""LSS rare-heavy checkpoint (autograd-owner-integration Phase F).

Connects the real-gradient Local Surprise Signal (`torch_lss.py`) into the
existing rare-heavy artifact path as a first-class, auditable OFFLINE owner
artifact, WITHOUT changing the runtime ``prediction_error`` snapshot schema.

The checkpoint is float-only (no torch tensors), so it travels through the same
``RareHeavyArtifact`` bundle as the temporal / memory / substrate / application
checkpoints. The runtime keeps publishing the bounded semantic
``PredictionErrorSnapshot`` every turn; the gradient LSS lives only in this
offline artifact and, on import, adjusts prediction-owner internal calibration
(never the published snapshot).

A grounding gate (fail-closed) requires every entry to satisfy the identity
``runtime PE == -LSS`` before a checkpoint is accepted, so the bounded runtime
proxy stays provably tied to the true gradient surprise.

This module is torch-free at import time (it only imports ``torch_lss`` lazily
inside the builder), so it is safe to re-export from the prediction facade and
import from the vz-temporal rare-heavy pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LSSEntry:
    axes: tuple[str, ...]
    local_surprise: tuple[float, ...]
    loss: float
    magnitude: float
    grounded: bool


@dataclass(frozen=True)
class LSSRareHeavyCheckpoint:
    checkpoint_id: str
    entries: tuple[LSSEntry, ...]
    mean_magnitude: float
    all_grounded: bool
    description: str = ""


def build_lss_rare_heavy_checkpoint(
    *,
    samples: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...],
    checkpoint_id: str,
) -> LSSRareHeavyCheckpoint:
    """Compute a float-only LSS checkpoint from (predicted, actual) PE axis tuples.

    Each sample is ``(predicted_axes, actual_axes)`` (typically the four
    PredictionError axes: task / relationship / regime / action). Uses real
    torch autograd to compute ``dL/doutput`` and the runtime-PE grounding bridge.
    Fail-closed: raises if any entry's grounding identity does not hold.
    """

    from volvence_zero.prediction.torch_lss import (
        bridge_runtime_pe_to_lss,
        compute_gradient_lss,
    )

    entries: list[LSSEntry] = []
    total_mag = 0.0
    all_grounded = True
    for predicted, actual in samples:
        art = compute_gradient_lss(predicted, actual)
        bridge = bridge_runtime_pe_to_lss(predicted=predicted, actual=actual)
        grounded = bridge.proxy_is_negative_lss and bridge.magnitude_correlates
        if not grounded:
            raise ValueError(
                "LSS grounding gate failed: runtime PE != -LSS for a sample; "
                "refusing to build an ungrounded rare-heavy LSS checkpoint."
            )
        entries.append(
            LSSEntry(
                axes=art.axes,
                local_surprise=art.local_surprise,
                loss=art.loss,
                magnitude=art.magnitude,
                grounded=grounded,
            )
        )
        total_mag += art.magnitude
        all_grounded = all_grounded and grounded
    mean_mag = total_mag / max(len(entries), 1)
    return LSSRareHeavyCheckpoint(
        checkpoint_id=checkpoint_id,
        entries=tuple(entries),
        mean_magnitude=mean_mag,
        all_grounded=all_grounded,
        description=(
            f"LSS rare-heavy checkpoint: {len(entries)} entries, "
            f"mean|LSS|={mean_mag:.4f}, grounded={all_grounded}"
        ),
    )
