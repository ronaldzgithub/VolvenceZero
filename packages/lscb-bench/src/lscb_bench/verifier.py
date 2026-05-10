# Copyright 2026 LSCB Contributors
# Licensed under the Apache License, Version 2.0.

"""Reproducibility verifier (RFC §7.3).

Per the RFC, organisers re-run **one random public-test arc** per
submission. If results deviate beyond seed variance (>5% per axis)
the submission is flagged. We model this as:

1. Pick a deterministic-but-input-derived ``arc_id`` from the
   submission's public arc set (so the choice is auditable, not
   manipulable by the submitter).
2. Re-run that arc in the same harness configuration.
3. Compare per-axis scores between the original run and the re-run;
   flag if any axis differs by more than ``per_axis_threshold``
   (default 5.0 absolute points on the 0-100 scale, equal to RFC
   "5% per axis").

The verifier does NOT itself produce LSCB scores — it consumes them
from the orchestrator. It is a thin comparison + reporting layer.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Mapping

from lscb_bench.spec import AxisId


@dataclasses.dataclass(frozen=True)
class VerifierResult:
    """One verification record."""

    submission_id: str
    arc_id: str
    original_axis_scores: dict[AxisId, float]
    rerun_axis_scores: dict[AxisId, float]
    per_axis_delta: dict[AxisId, float]
    per_axis_threshold: float
    flagged: bool
    flag_reasons: tuple[str, ...]

    def to_json(self) -> dict:
        return {
            "submission_id": self.submission_id,
            "arc_id": self.arc_id,
            "original_axis_scores": {a.value: self.original_axis_scores.get(a, 0.0) for a in AxisId},
            "rerun_axis_scores": {a.value: self.rerun_axis_scores.get(a, 0.0) for a in AxisId},
            "per_axis_delta": {a.value: self.per_axis_delta.get(a, 0.0) for a in AxisId},
            "per_axis_threshold": self.per_axis_threshold,
            "flagged": self.flagged,
            "flag_reasons": list(self.flag_reasons),
        }


def pick_verification_arc(
    *,
    submission_id: str,
    public_arc_ids: list[str],
    salt: str = "",
) -> str:
    """Pick a deterministic-but-uniform arc id from the public set.

    Uses SHA-256 over ``(submission_id, salt, sorted public_arc_ids)``
    so the same submission always picks the same verification arc but
    submitters cannot predict it before ``public_arc_ids`` is published.
    """
    if not public_arc_ids:
        raise ValueError("pick_verification_arc requires at least one public arc id")
    sorted_ids = sorted(public_arc_ids)
    payload = f"{submission_id}|{salt}|{'|'.join(sorted_ids)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(sorted_ids)
    return sorted_ids[idx]


def compare_axis_scores(
    *,
    submission_id: str,
    arc_id: str,
    original: Mapping[AxisId, float],
    rerun: Mapping[AxisId, float],
    per_axis_threshold: float = 5.0,
) -> VerifierResult:
    """Compute per-axis deltas and flag if any exceeds the threshold."""

    deltas: dict[AxisId, float] = {}
    flag_reasons: list[str] = []
    for axis in AxisId:
        a = float(original.get(axis, 0.0))
        b = float(rerun.get(axis, 0.0))
        d = abs(a - b)
        deltas[axis] = d
        if d > per_axis_threshold:
            flag_reasons.append(
                f"axis {axis.value} differs by {d:.2f} > threshold {per_axis_threshold:.2f} "
                f"(original={a:.2f}, rerun={b:.2f})"
            )
    return VerifierResult(
        submission_id=submission_id,
        arc_id=arc_id,
        original_axis_scores=dict(original),
        rerun_axis_scores=dict(rerun),
        per_axis_delta=deltas,
        per_axis_threshold=per_axis_threshold,
        flagged=bool(flag_reasons),
        flag_reasons=tuple(flag_reasons),
    )
