"""Frozen CMS contract types (bands, tower profile, checkpoint state).

All pure data; no behaviour. The runtime logic lives in ``cms``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from volvence_zero.learned_update import LearnedUpdateRuleState


class CMSVariant(str, Enum):
    SEQUENTIAL = "sequential"
    INDEPENDENT = "independent"
    NESTED = "nested"


@dataclass(frozen=True)
class CMSBandState:
    name: str
    vector: tuple[float, ...]
    last_update_ms: int
    cadence_interval: int
    observations_since_update: int
    pending_signal: tuple[float, ...]
    learning_rate: float = 0.0
    effective_learning_rate: float = 0.0
    momentum: tuple[float, ...] = ()
    anti_forgetting_strength: float = 0.0
    update_gate: float = 0.0
    slow_mix: float = 0.0
    reset_mix: float = 0.0
    confidence: float = 0.0
    update_summary: str = ""
    mode: str = "vector"
    mlp_param_count: int = 0


@dataclass(frozen=True)
class CMSTowerLevelState:
    level_id: str
    role: str
    vector: tuple[float, ...]
    cadence_interval: int
    source_level_ids: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class CMSTowerProfile:
    profile_id: str
    levels: tuple[CMSTowerLevelState, ...]
    readout_vector: tuple[float, ...]
    description: str


@dataclass(frozen=True)
class CMSContinuumBand:
    band_id: str
    role: str
    vector: tuple[float, ...]
    cadence_interval: int
    update_frequency: float
    persistence_bias: float
    retrieval_weight: float
    pending_signal: tuple[float, ...] = ()
    source_band_ids: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class CMSContinuumReconstructionEdge:
    edge_id: str
    source_band_id: str
    target_band_id: str
    transfer_kind: str
    strength: float
    description: str


@dataclass(frozen=True)
class CMSContinuumProfile:
    profile_id: str
    bands: tuple[CMSContinuumBand, ...]
    reconstruction_edges: tuple[CMSContinuumReconstructionEdge, ...]
    readout_band_id: str
    description: str


@dataclass(frozen=True)
class CMSHopeSelfModificationState:
    enabled: bool
    update_count: int
    last_target_id: str
    generated_learning_rate: float
    generated_decay_rate: float
    generated_reset_rate: float
    last_improvement: float
    last_stability: float
    last_reward: float
    guarded: bool
    guard_reason: str = ""
    description: str = ""


@dataclass(frozen=True)
class CMSState:
    online_fast: CMSBandState
    session_medium: CMSBandState
    background_slow: CMSBandState
    total_observations: int
    total_reflections: int
    description: str
    variant: str = "sequential"
    tower_profile: CMSTowerProfile | None = None
    tower_depth: int = 0
    continuum_profile: CMSContinuumProfile | None = None
    update_rule_state: LearnedUpdateRuleState | None = None
    hope_self_modification_state: CMSHopeSelfModificationState | None = None


@dataclass(frozen=True)
class CMSCheckpointState:
    online_fast: tuple[float, ...]
    session_medium: tuple[float, ...]
    background_slow: tuple[float, ...]
    last_update_ms: int
    total_observations: int
    total_reflections: int
    session_observations_since_update: int
    background_observations_since_update: int
    session_pending_signal: tuple[float, ...]
    background_pending_signal: tuple[float, ...]
    mode: str = "vector"
    mlp_params: tuple[tuple[tuple[float, ...], ...], ...] = ()
    nested_session_init_target: tuple[float, ...] = ()
    nested_online_init_target: tuple[float, ...] = ()
    tower_meta_levels: tuple[tuple[str, tuple[float, ...]], ...] = ()
    update_rule_state: LearnedUpdateRuleState | None = None
    hope_self_modification_state: CMSHopeSelfModificationState | None = None


@dataclass(frozen=True)
class CMSTowerConsolidationUpdate:
    online_signal: tuple[float, ...] = ()
    session_signal: tuple[float, ...] = ()
    background_signal: tuple[float, ...] = ()
    decay_pressure: float = 0.0
    reset_fast_context: bool = False
    description: str = ""
