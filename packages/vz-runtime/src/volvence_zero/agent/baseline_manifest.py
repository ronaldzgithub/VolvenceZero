from __future__ import annotations

import time
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


@dataclass(frozen=True)
class RuntimeBehaviorBaseline:
    """Frozen runtime-metric baseline for the default profile (CP-01 / GAP-10).

    Complements ``DefaultBehaviorBaselineManifest`` (config surface only)
    with the RUNTIME readouts the plan requires frozen before any model
    upgrade: turn latency, the PE four-axis values, evaluation family
    means, credit closure and temporal switch readouts. Deterministic
    inputs (fixed synthetic turn texts, default config, rare-heavy off);
    latency fields are environment-dependent and therefore excluded from
    the digest.
    """

    schema_version: str
    baseline_id: str
    turn_count: int
    mean_turn_seconds: float
    max_turn_seconds: float
    pe_task_error: float
    pe_relationship_error: float
    pe_regime_error: float
    pe_action_error: float
    pe_magnitude: float
    evaluation_family_means: tuple[tuple[str, float], ...]
    credit_cumulative_by_level: tuple[tuple[str, float], ...]
    credit_delayed_ledger_size: int
    regime_id: str
    temporal_switch_gate: float
    temporal_steps_since_switch: int
    description: str

    def digest(self) -> str:
        """Environment-independent digest (latency fields excluded)."""

        payload = asdict(self)
        payload.pop("mean_turn_seconds")
        payload.pop("max_turn_seconds")
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


_BASELINE_TURN_TEXTS: tuple[str, ...] = (
    "Walk me through the harbor plan for tomorrow.",
    "The tide tables changed; adjust the schedule.",
    "Now summarize what we committed to.",
    "One more check: anything still open?",
)


async def build_runtime_behavior_baseline(
    *, turn_texts: tuple[str, ...] = _BASELINE_TURN_TEXTS
) -> RuntimeBehaviorBaseline:
    """Run the default profile on a fixed synthetic script and freeze readouts.

    Same seed + same profile must replay a stable digest (CP-01 exit
    condition); a digest change is a behaviour-baseline change that must be
    reviewed before model upgrades proceed.
    """

    from volvence_zero.agent.session import AgentSessionRunner

    if not turn_texts:
        raise ValueError("turn_texts must be non-empty")
    runner = AgentSessionRunner(rare_heavy_enabled=False)
    durations: list[float] = []
    result = None
    for text in turn_texts:
        started = time.perf_counter()
        result = await runner.run_turn(text)
        durations.append(time.perf_counter() - started)
    assert result is not None

    pe_value = result.active_snapshots["prediction_error"].value
    evaluation_value = result.active_snapshots["evaluation"].value
    credit_value = result.active_snapshots["credit"].value
    regime_value = result.active_snapshots["regime"].value
    temporal_value = result.active_snapshots["temporal_abstraction"].value

    family_sums: dict[str, list[float]] = {}
    for score in evaluation_value.turn_scores:
        family_sums.setdefault(score.family, []).append(score.value)
    family_means = tuple(
        sorted(
            (family, round(sum(values) / len(values), 6))
            for family, values in family_sums.items()
        )
    )

    return RuntimeBehaviorBaseline(
        schema_version="runtime-behavior-baseline.v1",
        baseline_id="volvence-defaults-runtime-2026-07",
        turn_count=len(turn_texts),
        mean_turn_seconds=round(sum(durations) / len(durations), 4),
        max_turn_seconds=round(max(durations), 4),
        pe_task_error=round(pe_value.error.task_error, 6),
        pe_relationship_error=round(pe_value.error.relationship_error, 6),
        pe_regime_error=round(pe_value.error.regime_error, 6),
        pe_action_error=round(pe_value.error.action_error, 6),
        pe_magnitude=round(pe_value.error.magnitude, 6),
        evaluation_family_means=family_means,
        credit_cumulative_by_level=tuple(
            (level, round(value, 6))
            for level, value in credit_value.cumulative_credit_by_level
        ),
        credit_delayed_ledger_size=credit_value.delayed_ledger_size,
        regime_id=regime_value.active_regime.regime_id,
        temporal_switch_gate=round(
            temporal_value.controller_state.switch_gate, 6
        ),
        temporal_steps_since_switch=(
            temporal_value.controller_state.steps_since_switch
        ),
        description=(
            f"Runtime baseline over {len(turn_texts)} deterministic synthetic "
            "turns under production defaults (rare-heavy off). Latency fields "
            "are informational; the digest freezes behaviour only."
        ),
    )


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
        learned_coverage_version="v0.1@2026-07-14",
        description=(
            "Frozen comparison surface for product Brain defaults and the "
            "dialogue pe-eta paper baseline. The two surfaces are intentionally "
            "recorded separately because their schedules and substrate mutation "
            "settings are not equivalent."
        ),
    )
