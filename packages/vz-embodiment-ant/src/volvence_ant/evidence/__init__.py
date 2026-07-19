"""Workstream F: ant-active-evidence lane.

Reuses the substrate-agnostic ``evaluate_learned_active_candidate`` gate from
the vz-runtime facade, feeding it evidence collected on the DIGITAL-ANT
substrate instead of the HF/LLM substrate. The gate stays intentionally
conservative (real_trace_turns >= 500, validation_delta >= 0.02, strict ETA,
PE-off / ETA-off controls, rollback, latency, safety), so at toy scale it
correctly BLOCKS promotion — which is the honest, rigorous outcome.
"""

from __future__ import annotations

from volvence_ant.evidence.ant_active_evidence import (
    AntActiveEvidenceBundle,
    collect_ant_active_evidence,
)
from volvence_ant.evidence.provenance import (
    AntArtifactIntegrityError,
    AntRunProvenance,
    collect_ant_provenance,
    verify_ant_artifact_manifest,
    write_ant_artifact_bundle,
)

__all__ = [
    "AntActiveEvidenceBundle",
    "AntArtifactIntegrityError",
    "AntRunProvenance",
    "collect_ant_provenance",
    "collect_ant_active_evidence",
    "verify_ant_artifact_manifest",
    "write_ant_artifact_bundle",
]
