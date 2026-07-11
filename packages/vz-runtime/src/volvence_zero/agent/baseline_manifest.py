from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from typing import Any

from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.joint_loop import JointLoopSchedule
from volvence_zero.runtime import WiringLevel


@dataclass(frozen=True)
class DefaultBehaviorBaselineManifest:
    schema_version: str
    baseline_id: str
    product_brain_defaults: tuple[tuple[str, str], ...]
    dialogue_runner_defaults: tuple[tuple[str, str], ...]
    rollout_wiring: tuple[tuple[str, str], ...]
    capability_wiring_digest: str
    learned_coverage_spec: str
    learned_coverage_version: str
    description: str

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


def _stringify(value: Any) -> str:
    if isinstance(value, WiringLevel):
        return value.value
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _rollout_wiring_snapshot(config: FinalRolloutConfig) -> tuple[tuple[str, str], ...]:
    return tuple(
        (field.name, value.value)
        for field in fields(config)
        if isinstance((value := getattr(config, field.name)), WiringLevel)
    )


def _capability_wiring_digest(config: FinalRolloutConfig) -> str:
    normalized = {
        owner: {
            capability: level.value
            for capability, level in sorted(capabilities.items())
        }
        for owner, capabilities in sorted(config.capability_wirings.items())
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def build_default_behavior_baseline_manifest() -> DefaultBehaviorBaselineManifest:
    """Freeze product and paper-suite defaults without claiming equivalence."""

    # Local import avoids making the generic agent package import the Brain
    # facade during module initialization.
    from volvence_zero.brain import BrainConfig

    brain = BrainConfig()
    rollout = FinalRolloutConfig()
    product_schedule = JointLoopSchedule()
    dialogue_schedule = JointLoopSchedule(
        ssl_interval=1,
        rl_interval=2,
        pe_rare_heavy_threshold=0.4,
    )
    product_defaults = (
        ("substrate_mode", brain.substrate_mode),
        ("substrate_model_id", brain.substrate_model_id),
        ("substrate_local_files_only", _stringify(brain.substrate_local_files_only)),
        ("substrate_fallback_mode", _stringify(brain.substrate_fallback_mode)),
        ("temporal_latent_dim", str(brain.temporal_latent_dim)),
        ("memory_scope_root_dir", _stringify(brain.memory_scope_root_dir)),
        ("application_persistence_dir", _stringify(brain.application_persistence_dir)),
        ("owner_hydration_wiring", brain.owner_hydration_wiring.value),
        ("semantic_embedding_backend_wiring", brain.semantic_embedding_backend_wiring.value),
        ("allow_live_substrate_mutation", "false"),
        ("runner_schedule", json.dumps(asdict(product_schedule), sort_keys=True, default=str)),
    )
    dialogue_defaults = (
        ("profile_label", "pe-eta"),
        ("profile_dispatch_mode", "legacy-if-elif"),
        ("substrate_runtime_mode", "builtin-only"),
        ("allow_live_substrate_mutation", "true"),
        ("runner_schedule", json.dumps(asdict(dialogue_schedule), sort_keys=True, default=str)),
    )
    return DefaultBehaviorBaselineManifest(
        schema_version="default-behavior-baseline.v1",
        baseline_id="volvence-defaults-2026-07",
        product_brain_defaults=product_defaults,
        dialogue_runner_defaults=dialogue_defaults,
        rollout_wiring=_rollout_wiring_snapshot(rollout),
        capability_wiring_digest=_capability_wiring_digest(rollout),
        learned_coverage_spec="docs/specs/learned-vs-heuristic-coverage.md",
        learned_coverage_version="v0.1@2026-07-04",
        description=(
            "Frozen comparison surface for product Brain defaults and the "
            "dialogue pe-eta paper baseline. The two surfaces are intentionally "
            "recorded separately because their schedules and substrate mutation "
            "settings are not equivalent."
        ),
    )
