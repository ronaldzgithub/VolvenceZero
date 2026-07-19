"""Workstream E: matched-control proofs for the digital ant.

Two layers:

- ``latent_proofs`` wraps the substrate-agnostic torch proofs the kernel itself
  uses as exit evidence (optimize-vs-no-optimize Internal RL on ``z_t``, and the
  strict-ETA information-bottleneck ladder). These are the authoritative
  "learning is real, not hardcoded" evidence and run in seconds.
- ``matched_control`` runs behavioural ant arms (learned / PE-off / fixed-rule /
  random) on one shared environment + seed and reports directional metrics.
"""

from __future__ import annotations

from volvence_ant.proofs.latent_proofs import AntLatentProofReport, run_ant_latent_proofs
from volvence_ant.proofs.matched_control import (
    ArmAggregate,
    ArmMetrics,
    MatchedControlReport,
    MultiSeedMatchedControlReport,
    aggregate_matched_control_reports,
    run_behavioral_matched_control,
    run_multiseed_matched_control,
    run_single_seed_matched_control,
)

__all__ = [
    "AntLatentProofReport",
    "ArmAggregate",
    "ArmMetrics",
    "MatchedControlReport",
    "MultiSeedMatchedControlReport",
    "aggregate_matched_control_reports",
    "run_ant_latent_proofs",
    "run_behavioral_matched_control",
    "run_multiseed_matched_control",
    "run_single_seed_matched_control",
]
