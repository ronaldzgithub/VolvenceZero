"""Evidence-only companion startup profiles.

These profiles are process-start configuration, not product modes.  They
exist so a preregistered matched experiment can prove that the intended
owner-level intervention was load-bearing while preserving the normal
``companion`` defaults byte-for-byte.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

from volvence_zero.brain import BrainConfig
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.joint_loop import JointLoopSchedule
from volvence_zero.runtime import WiringLevel


GATE1_PE_TEMPORAL_ON = "gate1-pe-temporal-on-v1"
GATE1_PE_TEMPORAL_OFF = "gate1-pe-temporal-off-v1"
GATE4_ACTIVE_SELECTOR = "gate4-active-selector-v1"
GATE4_RANDOM_FEEDBACK = "gate4-random-feedback-v1"
GATE5_MULTIFREQUENCY_CMS = "gate5-multifrequency-cms-v1"
GATE5_SINGLE_TIMESCALE = "gate5-single-timescale-v1"
GATE6_CONDITIONED_META_INIT = "gate6-conditioned-meta-init-v1"
GATE6_COPY_INIT = "gate6-copy-init-v1"
GATE7_SSL_RL_FULL = "gate7-ssl-rl-full-v1"
GATE7_NO_SSL = "gate7-no-ssl-v1"
GATE7_NO_RL = "gate7-no-rl-v1"
GATE9_M3_SLOW_ON = "gate9-m3-slow-on-v1"
GATE9_M3_SLOW_OFF = "gate9-m3-slow-off-v1"
GATE10_RARE_HEAVY_IMPORT = "gate10-rare-heavy-import-v1"
GATE10_RARE_HEAVY_REVIEW = "gate10-rare-heavy-review-v1"
MSC_RUNTIME_COLLECTOR = "msc-runtime-collector-v1"
MSC_STEERING_SHADOW_COLLECTOR = "msc-steering-shadow-collector-v1"
MSC_RUNTIME_PROFILE_NAMES = frozenset(
    {MSC_RUNTIME_COLLECTOR, MSC_STEERING_SHADOW_COLLECTOR}
)
COMPANION_EVIDENCE_PROFILE_NAMES = (
    GATE1_PE_TEMPORAL_ON,
    GATE1_PE_TEMPORAL_OFF,
    GATE4_ACTIVE_SELECTOR,
    GATE4_RANDOM_FEEDBACK,
    GATE5_MULTIFREQUENCY_CMS,
    GATE5_SINGLE_TIMESCALE,
    GATE6_CONDITIONED_META_INIT,
    GATE6_COPY_INIT,
    GATE7_SSL_RL_FULL,
    GATE7_NO_SSL,
    GATE7_NO_RL,
    GATE9_M3_SLOW_ON,
    GATE9_M3_SLOW_OFF,
    GATE10_RARE_HEAVY_IMPORT,
    GATE10_RARE_HEAVY_REVIEW,
    MSC_RUNTIME_COLLECTOR,
    MSC_STEERING_SHADOW_COLLECTOR,
)


@dataclass(frozen=True)
class CompanionEvidenceProfile:
    """Frozen process-level intervention contract for one evidence arm."""

    name: str
    gate_id: int
    arm_role: str
    brain_overrides: tuple[tuple[str, object], ...] = ()
    rollout_overrides: tuple[tuple[str, object], ...] = ()
    turn_trigger_kind: str = "user_input"
    allow_single_session_live_substrate_mutation: bool = False
    allow_typed_observation_frame: bool = False
    publish_runtime_context: bool = False

    def apply(self, base: BrainConfig) -> BrainConfig:
        rollout = base.final_rollout_config or FinalRolloutConfig()
        rollout = replace(rollout, **dict(self.rollout_overrides))
        return replace(
            base,
            **dict(self.brain_overrides),
            final_rollout_config=rollout,
        )

    def intervention_contract(self) -> dict[str, object]:
        contract = {
            "gate_id": self.gate_id,
            "arm_role": self.arm_role,
            "brain_overrides": {key: _jsonable(value) for key, value in self.brain_overrides},
            "rollout_overrides": {key: _jsonable(value) for key, value in self.rollout_overrides},
            "turn_trigger_kind": self.turn_trigger_kind,
            "allow_single_session_live_substrate_mutation": (self.allow_single_session_live_substrate_mutation),
            "allow_typed_observation_frame": self.allow_typed_observation_frame,
            "publish_runtime_context": self.publish_runtime_context,
            "prediction_error_publication": "active-in-both-arms",
            "sut_generation_temperature": 0.0,
            "production_default_changed": False,
        }
        if self.gate_id == 1:
            brain = dict(self.brain_overrides)
            rollout = dict(self.rollout_overrides)
            contract.update(
                {
                    "external_prediction_error_drive": brain["external_prediction_error_drive"],
                    "prediction_error_readout_only": brain["prediction_error_readout_only"],
                    "primary_prediction_error_dominance_enabled": brain["primary_prediction_error_dominance_enabled"],
                    "prediction_error_temporal_learning_enabled": brain["external_prediction_error_drive"],
                    "prediction_error_temporal_switch": rollout["prediction_error_temporal_switch"].value,
                    "prediction_error_runtime_modulation": rollout["prediction_error_runtime_modulation"].value,
                }
            )
        return contract


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


_DEFAULT_SCHEDULE = JointLoopSchedule()
_NO_RL_SCHEDULE = replace(_DEFAULT_SCHEDULE, rl_interval=0)
_RARE_HEAVY_SCHEDULE = replace(
    _DEFAULT_SCHEDULE,
    pe_rare_heavy_threshold=0.0,
)


_PROFILES = {
    GATE1_PE_TEMPORAL_ON: CompanionEvidenceProfile(
        name=GATE1_PE_TEMPORAL_ON,
        gate_id=1,
        arm_role="treatment",
        brain_overrides=(
            ("external_prediction_error_drive", True),
            ("prediction_error_readout_only", False),
            ("primary_prediction_error_dominance_enabled", True),
        ),
        rollout_overrides=(
            ("prediction_error_temporal_switch", WiringLevel.ACTIVE),
            ("prediction_error_runtime_modulation", WiringLevel.ACTIVE),
        ),
    ),
    GATE1_PE_TEMPORAL_OFF: CompanionEvidenceProfile(
        name=GATE1_PE_TEMPORAL_OFF,
        gate_id=1,
        arm_role="rollback-control",
        brain_overrides=(
            ("external_prediction_error_drive", False),
            ("prediction_error_readout_only", True),
            ("primary_prediction_error_dominance_enabled", False),
        ),
        rollout_overrides=(
            ("prediction_error_temporal_switch", WiringLevel.ACTIVE),
            ("prediction_error_runtime_modulation", WiringLevel.ACTIVE),
        ),
    ),
    GATE4_ACTIVE_SELECTOR: CompanionEvidenceProfile(
        name=GATE4_ACTIVE_SELECTOR,
        gate_id=4,
        arm_role="treatment",
        brain_overrides=(("apprenticeship_feedback_policy", "owner"),),
        rollout_overrides=(("apprenticeship_alignment", WiringLevel.ACTIVE),),
        turn_trigger_kind="apprentice",
    ),
    GATE4_RANDOM_FEEDBACK: CompanionEvidenceProfile(
        name=GATE4_RANDOM_FEEDBACK,
        gate_id=4,
        arm_role="matched-random-control",
        brain_overrides=(("apprenticeship_feedback_policy", "random"),),
        rollout_overrides=(("apprenticeship_alignment", WiringLevel.ACTIVE),),
        turn_trigger_kind="apprentice",
    ),
    GATE5_MULTIFREQUENCY_CMS: CompanionEvidenceProfile(
        name=GATE5_MULTIFREQUENCY_CMS,
        gate_id=5,
        arm_role="treatment",
        brain_overrides=(
            ("cms_variant", "nested"),
            ("cms_session_cadence", 2),
            ("cms_background_cadence", 4),
            ("cms_pe_features_enabled", True),
            ("cms_replay_window_size", 8),
        ),
    ),
    GATE5_SINGLE_TIMESCALE: CompanionEvidenceProfile(
        name=GATE5_SINGLE_TIMESCALE,
        gate_id=5,
        arm_role="rollback-control",
        brain_overrides=(
            ("cms_variant", "independent"),
            ("cms_session_cadence", 1),
            ("cms_background_cadence", 1),
            ("cms_pe_features_enabled", True),
            ("cms_replay_window_size", 8),
        ),
    ),
    GATE6_CONDITIONED_META_INIT: CompanionEvidenceProfile(
        name=GATE6_CONDITIONED_META_INIT,
        gate_id=6,
        arm_role="treatment",
        brain_overrides=(
            ("cms_variant", "nested"),
            ("cms_context_conditioned_meta_init", True),
            ("nested_context_reset_mode", "meta-init"),
        ),
    ),
    GATE6_COPY_INIT: CompanionEvidenceProfile(
        name=GATE6_COPY_INIT,
        gate_id=6,
        arm_role="rollback-control",
        brain_overrides=(
            ("cms_variant", "nested"),
            ("cms_context_conditioned_meta_init", False),
            ("nested_context_reset_mode", "copy-init"),
        ),
    ),
    GATE7_SSL_RL_FULL: CompanionEvidenceProfile(
        name=GATE7_SSL_RL_FULL,
        gate_id=7,
        arm_role="treatment",
        brain_overrides=(
            ("temporal_profile", "learned-ndim"),
            ("joint_schedule", _DEFAULT_SCHEDULE),
        ),
        rollout_overrides=(
            ("temporal_ssl_backend", WiringLevel.ACTIVE),
            ("temporal_runtime_backend", WiringLevel.ACTIVE),
            ("internal_rl_backend", WiringLevel.ACTIVE),
            ("internal_rl_runtime_replay", WiringLevel.ACTIVE),
            ("internal_rl_runtime_modulation_strength", 0.3),
        ),
    ),
    GATE7_NO_SSL: CompanionEvidenceProfile(
        name=GATE7_NO_SSL,
        gate_id=7,
        arm_role="no-ssl-control",
        brain_overrides=(
            ("temporal_profile", "learned-ndim"),
            ("joint_schedule", _DEFAULT_SCHEDULE),
            ("joint_apply_ssl_optimization", False),
        ),
        rollout_overrides=(
            ("temporal_ssl_backend", WiringLevel.ACTIVE),
            ("temporal_runtime_backend", WiringLevel.ACTIVE),
            ("internal_rl_backend", WiringLevel.ACTIVE),
            ("internal_rl_runtime_replay", WiringLevel.ACTIVE),
            ("internal_rl_runtime_modulation_strength", 0.3),
        ),
    ),
    GATE7_NO_RL: CompanionEvidenceProfile(
        name=GATE7_NO_RL,
        gate_id=7,
        arm_role="no-rl-control",
        brain_overrides=(
            ("temporal_profile", "learned-ndim"),
            ("joint_schedule", _DEFAULT_SCHEDULE),
            ("joint_apply_policy_optimization", False),
        ),
        rollout_overrides=(
            ("temporal_ssl_backend", WiringLevel.ACTIVE),
            ("temporal_runtime_backend", WiringLevel.ACTIVE),
            ("internal_rl_backend", WiringLevel.ACTIVE),
            ("internal_rl_runtime_replay", WiringLevel.ACTIVE),
            ("internal_rl_runtime_modulation_strength", 0.3),
        ),
    ),
    GATE9_M3_SLOW_ON: CompanionEvidenceProfile(
        name=GATE9_M3_SLOW_ON,
        gate_id=9,
        arm_role="treatment",
        brain_overrides=(
            ("temporal_profile", "learned-ndim"),
            ("joint_schedule", _NO_RL_SCHEDULE),
        ),
        rollout_overrides=(
            ("temporal_ssl_backend", WiringLevel.ACTIVE),
            ("temporal_runtime_backend", WiringLevel.ACTIVE),
            ("temporal_ssl_m3_slow_gain", 1.0),
        ),
    ),
    GATE9_M3_SLOW_OFF: CompanionEvidenceProfile(
        name=GATE9_M3_SLOW_OFF,
        gate_id=9,
        arm_role="rollback-control",
        brain_overrides=(
            ("temporal_profile", "learned-ndim"),
            ("joint_schedule", _NO_RL_SCHEDULE),
        ),
        rollout_overrides=(
            ("temporal_ssl_backend", WiringLevel.ACTIVE),
            ("temporal_runtime_backend", WiringLevel.ACTIVE),
            ("temporal_ssl_m3_slow_gain", 0.0),
        ),
    ),
    GATE10_RARE_HEAVY_IMPORT: CompanionEvidenceProfile(
        name=GATE10_RARE_HEAVY_IMPORT,
        gate_id=10,
        arm_role="treatment",
        brain_overrides=(
            ("rare_heavy_enabled", True),
            ("rare_heavy_trace_window", 5),
            ("rare_heavy_min_traces", 4),
            ("rare_heavy_cooldown_turns", 7),
            ("allow_live_substrate_mutation", True),
            ("joint_schedule", _RARE_HEAVY_SCHEDULE),
        ),
        allow_single_session_live_substrate_mutation=True,
    ),
    GATE10_RARE_HEAVY_REVIEW: CompanionEvidenceProfile(
        name=GATE10_RARE_HEAVY_REVIEW,
        gate_id=10,
        arm_role="review-only-control",
        brain_overrides=(
            ("rare_heavy_enabled", True),
            ("rare_heavy_trace_window", 5),
            ("rare_heavy_min_traces", 4),
            ("rare_heavy_cooldown_turns", 7),
            ("allow_live_substrate_mutation", False),
            ("joint_schedule", _RARE_HEAVY_SCHEDULE),
        ),
    ),
    MSC_RUNTIME_COLLECTOR: CompanionEvidenceProfile(
        name=MSC_RUNTIME_COLLECTOR,
        gate_id=0,
        arm_role="formal-volvence-runtime-collector",
        allow_typed_observation_frame=True,
        publish_runtime_context=True,
    ),
    MSC_STEERING_SHADOW_COLLECTOR: CompanionEvidenceProfile(
        name=MSC_STEERING_SHADOW_COLLECTOR,
        gate_id=0,
        arm_role="formal-dialogue-steering-shadow-collector",
        rollout_overrides=(
            ("steering_sensor", WiringLevel.SHADOW),
            ("steering_executor", WiringLevel.SHADOW),
            ("steering_gate", WiringLevel.SHADOW),
            ("steering_shadow_hook", True),
        ),
        allow_typed_observation_frame=True,
        publish_runtime_context=True,
    ),
}


def resolve_companion_evidence_profile(
    name: str,
) -> CompanionEvidenceProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown companion evidence profile {name!r}; expected one of {COMPANION_EVIDENCE_PROFILE_NAMES!r}"
        ) from exc


def _cuda_attestation(device: str) -> dict[str, object] | None:
    if not device.startswith("cuda"):
        return None
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on deployment
        raise RuntimeError("CUDA evidence profile requires torch for hardware attestation") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA evidence profile requested {device!r}, but CUDA is unavailable")
    index = 0
    if ":" in device:
        raw_index = device.split(":", 1)[1]
        if not raw_index.isdigit():
            raise ValueError(f"invalid CUDA device {device!r}")
        index = int(raw_index)
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {index} is unavailable; device_count={torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(index)
    return {
        "device_index": index,
        "device_name": torch.cuda.get_device_name(index),
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": (int(torch.backends.cudnn.version()) if torch.backends.cudnn.version() is not None else None),
    }


def write_companion_evidence_profile_attestation(
    *,
    output_dir: Path,
    profile: CompanionEvidenceProfile,
    substrate_model_id: str,
    substrate_device: str,
    temporal_n_z: int | None = None,
    steering_bundle_id: str | None = None,
    steering_bundle_sha256: str | None = None,
) -> Path:
    """Write an immutable startup attestation under the isolated evidence root."""

    if profile.name in MSC_RUNTIME_PROFILE_NAMES:
        if temporal_n_z not in {3, 16, 64, 256}:
            raise ValueError(
                "MSC runtime profile attestation requires temporal_n_z 3/16/64/256"
            )
    elif temporal_n_z is not None:
        raise ValueError(
            "temporal_n_z attestation is only valid for an MSC runtime profile"
        )
    if profile.name == MSC_STEERING_SHADOW_COLLECTOR:
        if not steering_bundle_id or steering_bundle_sha256 is None:
            raise ValueError(
                "MSC steering profile attestation requires bundle lineage"
            )
        if len(steering_bundle_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in steering_bundle_sha256
        ):
            raise ValueError("MSC steering bundle SHA-256 is invalid")
    elif steering_bundle_id is not None or steering_bundle_sha256 is not None:
        raise ValueError(
            "steering bundle attestation is only valid for the steering profile"
        )

    payload: dict[str, Any] = {
        "schema_version": "companion-evidence-runtime-profile.v1",
        "profile": profile.name,
        "scope": "evidence-only",
        "substrate_model_id": substrate_model_id,
        "substrate_device": substrate_device,
        "intervention": profile.intervention_contract(),
        "rollback": {
            "method": "restart-without---companion-evidence-profile",
            "production_default": "all-new-gates-disabled",
        },
        "cuda": _cuda_attestation(substrate_device),
    }
    if temporal_n_z is not None:
        payload["temporal_n_z"] = temporal_n_z
    if steering_bundle_id is not None:
        payload["steering_bundle_id"] = steering_bundle_id
        payload["steering_bundle_sha256"] = steering_bundle_sha256
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["attestation_sha256"] = hashlib.sha256(canonical_payload).hexdigest()
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "companion_evidence_runtime_profile.json"
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"companion evidence profile attestation drift: {path}")
        return path
    path.write_bytes(encoded)
    return path


__all__ = [
    "COMPANION_EVIDENCE_PROFILE_NAMES",
    "GATE1_PE_TEMPORAL_OFF",
    "GATE1_PE_TEMPORAL_ON",
    "GATE4_ACTIVE_SELECTOR",
    "GATE4_RANDOM_FEEDBACK",
    "GATE5_MULTIFREQUENCY_CMS",
    "GATE5_SINGLE_TIMESCALE",
    "GATE6_CONDITIONED_META_INIT",
    "GATE6_COPY_INIT",
    "GATE7_NO_RL",
    "GATE7_NO_SSL",
    "GATE7_SSL_RL_FULL",
    "GATE9_M3_SLOW_OFF",
    "GATE9_M3_SLOW_ON",
    "GATE10_RARE_HEAVY_IMPORT",
    "GATE10_RARE_HEAVY_REVIEW",
    "MSC_RUNTIME_COLLECTOR",
    "MSC_RUNTIME_PROFILE_NAMES",
    "MSC_STEERING_SHADOW_COLLECTOR",
    "CompanionEvidenceProfile",
    "resolve_companion_evidence_profile",
    "write_companion_evidence_profile_attestation",
]
