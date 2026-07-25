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
    atomic_write_json,
    collect_ant_provenance,
    stable_json_digest,
    verify_ant_artifact_manifest,
    write_ant_artifact_bundle,
)
from volvence_ant.evidence.resume import (
    AntResumeStateError,
    SeedPartialStore,
    ant_stage_fingerprint,
)
from volvence_ant.evidence.runtime_profile import (
    ANT_CAUSAL_ACTION_HEAD_CONTRAST_PAIRS,
    ANT_CAUSAL_ACTION_HEAD_EFFECTIVE_DIMS,
    ANT_CAUSAL_ACTION_HEAD_RANK,
    ANT_CAUSAL_ACTION_HEAD_STRENGTH,
    ANT_RUNTIME_BATCH_TRANSITION_SIZE,
    ANT_RUNTIME_EXPLORATION_STRENGTH,
    ANT_RUNTIME_MODULATION_STRENGTH,
    ANT_RUNTIME_SEGMENT_MAX_STEPS,
    ant_runtime_replay_rollout_config,
)

__all__ = [
    "AntActiveEvidenceBundle",
    "AntArtifactIntegrityError",
    "AntResumeStateError",
    "AntRunProvenance",
    "ANT_CAUSAL_ACTION_HEAD_EFFECTIVE_DIMS",
    "ANT_CAUSAL_ACTION_HEAD_CONTRAST_PAIRS",
    "ANT_CAUSAL_ACTION_HEAD_RANK",
    "ANT_CAUSAL_ACTION_HEAD_STRENGTH",
    "ANT_RUNTIME_BATCH_TRANSITION_SIZE",
    "ANT_RUNTIME_EXPLORATION_STRENGTH",
    "ANT_RUNTIME_MODULATION_STRENGTH",
    "ANT_RUNTIME_SEGMENT_MAX_STEPS",
    "SeedPartialStore",
    "ant_stage_fingerprint",
    "ant_runtime_replay_rollout_config",
    "atomic_write_json",
    "collect_ant_provenance",
    "collect_ant_active_evidence",
    "stable_json_digest",
    "verify_ant_artifact_manifest",
    "write_ant_artifact_bundle",
]
