"""Relationship-conditioned Gate 2 longitudinal evidence lane.

This lane is intentionally independent of the historical v35/v36/v37
longitudinal sources.  The temporal selector consumes a full residual state
plus the public Relationship owner readout.  Synthetic relationship inputs
are typed fixtures passed through the real cognition owner; text is never
parsed to reconstruct the condition.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Protocol, Sequence

from companion_standard.semantic_state import SemanticRecord

from volvence_zero.agent.gate2_longitudinal_capture import (
    GATE2_V35_CONTROL_BASIS_FINGERPRINT,
    Gate2CandidateControlContract,
)
from volvence_zero.conditioning_bank_contracts import ConditioningBankReadout
from volvence_zero.internal_rl import (
    CounterfactualActionExample,
    KernelResidualActionSelectorArtifact,
    RELATIONSHIP_CONDITIONED_SELECTOR_FEATURE_SCHEMA_VERSION,
    fit_kernel_residual_action_selector,
    relationship_conditioned_selector_state_vector,
    selector_artifact_from_payload,
    selector_artifact_to_payload,
)
from volvence_zero.relationship_conditioning import (
    RELATIONSHIP_CONDITIONING_READOUT_LABELS,
    RelationshipConditioningModule,
)
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    BoundaryConsentSnapshot,
    RelationshipStateSnapshot,
)
from volvence_zero.substrate import SubstrateSnapshot


GATE2_CONDITIONED_PREREG_SCHEMA_VERSION = (
    "eta-gate2-longitudinal-conditioned.v1"
)
GATE2_CONDITIONED_SOURCE_SCHEMA_VERSION = (
    "gate2-longitudinal-conditioned-source.v1"
)
GATE2_CONDITIONED_SELECTOR_WRAPPER_SCHEMA_VERSION = (
    "eta-gate2-relationship-conditioned-selector.v1"
)
GATE2_CONDITIONED_CAPTURE_SCHEMA_VERSION = (
    "gate2-longitudinal-conditioned-capture.v1"
)
GATE2_CONDITIONED_TRAINING_SEED = 1291
GATE2_CONDITIONED_EVALUATION_SEEDS = (1301, 1313, 1327)
GATE2_CONDITIONED_TRAINING_COUNT = 64
GATE2_CONDITIONED_EVALUATION_COUNT = 510
GATE2_CONDITIONED_SESSION_SIZE = 10
GATE2_CONDITIONED_MIN_EFFECT = 0.02
GATE2_CONDITIONED_MIN_SESSION_POSITIVE_RATE = 0.60
GATE2_CONDITIONED_RIDGE_STRENGTH = 1.0
GATE2_CONDITIONED_TRACK_SCALE = (0.7, 0.7, 0.7)


@dataclass(frozen=True)
class Gate2ConditionedSourcePlan:
    schema_version: str
    transition_id: str
    seed: int
    global_index: int
    consumer_session_index: int
    relationship_profile_id: str
    opaque_observation_ref: str
    prediction_turn: str
    settlement_turn: str

    def __post_init__(self) -> None:
        if self.schema_version != GATE2_CONDITIONED_SOURCE_SCHEMA_VERSION:
            raise ValueError("unsupported Gate 2 conditioned source schema")
        if not self.transition_id or not self.relationship_profile_id:
            raise ValueError("conditioned source identity must be non-empty")
        if self.global_index < 0 or self.consumer_session_index < 0:
            raise ValueError("conditioned source indexes must be non-negative")
        if not self.prediction_turn.strip() or not self.settlement_turn.strip():
            raise ValueError("conditioned source turns must be non-empty")


@dataclass(frozen=True)
class Gate2RelationshipProfile:
    profile_id: str
    trust_level: float
    cumulative_trust_level: float
    continuity_level: float
    repair_pressure: float
    emotional_load: float
    stabilization_need: float
    trust_recovery_signal: float
    unresolved_tension_count: int
    recent_repair_count: int
    relationship_age_turns: int
    compliance_score: float
    consent_clarity: float
    denied_boundary: bool = False

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("relationship profile_id must be non-empty")
        bounded = (
            self.trust_level,
            self.cumulative_trust_level,
            self.continuity_level,
            self.repair_pressure,
            self.emotional_load,
            self.stabilization_need,
            self.trust_recovery_signal,
            self.compliance_score,
            self.consent_clarity,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("relationship profile values must be in [0, 1]")
        if (
            self.unresolved_tension_count < 0
            or self.recent_repair_count < 0
            or self.relationship_age_turns < 1
        ):
            raise ValueError("relationship profile counts are invalid")


GATE2_RELATIONSHIP_PROFILES = (
    Gate2RelationshipProfile(
        profile_id="rupture-repair",
        trust_level=0.25,
        cumulative_trust_level=0.35,
        continuity_level=0.45,
        repair_pressure=0.90,
        emotional_load=0.80,
        stabilization_need=0.85,
        trust_recovery_signal=0.20,
        unresolved_tension_count=3,
        recent_repair_count=0,
        relationship_age_turns=16,
        compliance_score=0.90,
        consent_clarity=0.85,
    ),
    Gate2RelationshipProfile(
        profile_id="steady-trust",
        trust_level=0.88,
        cumulative_trust_level=0.84,
        continuity_level=0.90,
        repair_pressure=0.10,
        emotional_load=0.15,
        stabilization_need=0.12,
        trust_recovery_signal=0.75,
        unresolved_tension_count=0,
        recent_repair_count=2,
        relationship_age_turns=24,
        compliance_score=0.95,
        consent_clarity=0.95,
    ),
    Gate2RelationshipProfile(
        profile_id="boundary-sensitive",
        trust_level=0.52,
        cumulative_trust_level=0.58,
        continuity_level=0.62,
        repair_pressure=0.55,
        emotional_load=0.58,
        stabilization_need=0.65,
        trust_recovery_signal=0.35,
        unresolved_tension_count=2,
        recent_repair_count=1,
        relationship_age_turns=18,
        compliance_score=0.40,
        consent_clarity=0.30,
        denied_boundary=True,
    ),
    Gate2RelationshipProfile(
        profile_id="recovering-continuity",
        trust_level=0.58,
        cumulative_trust_level=0.48,
        continuity_level=0.68,
        repair_pressure=0.42,
        emotional_load=0.40,
        stabilization_need=0.38,
        trust_recovery_signal=0.88,
        unresolved_tension_count=1,
        recent_repair_count=3,
        relationship_age_turns=22,
        compliance_score=0.88,
        consent_clarity=0.82,
    ),
)

_SOURCE_TEMPLATES = (
    "A follow-up is due after the last exchange. Choose one bounded response.",
    "The next turn should preserve the relationship while moving one step.",
    "A decision point has arrived. Select the most appropriate next response.",
    "The dialogue is continuing after a meaningful update. Respond carefully.",
    "One concise next action is needed without inventing new personal facts.",
    "The prior interaction changed what a calibrated response should do next.",
    "Continue the exchange with one auditable and reversible next step.",
    "The current state calls for a relationship-aware response selection.",
    "Select a next response that fits the established interaction state.",
    "A bounded continuation is required before any further external action.",
)

_SETTLEMENT_BY_PROFILE = {
    "rupture-repair": (
        "Acknowledge the rupture, slow the pace, and offer a reversible repair step."
    ),
    "steady-trust": (
        "Continue directly with the agreed next step and state how progress will be checked."
    ),
    "boundary-sensitive": (
        "Pause external action, restate the boundary, and request explicit consent before proceeding."
    ),
    "recovering-continuity": (
        "Recognize the partial recovery, keep the next step small, and preserve an easy revision path."
    ),
}


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(record_id: str, *, source_turn: int) -> SemanticRecord:
    return SemanticRecord(
        record_id=record_id,
        summary="typed synthetic relationship evidence",
        detail="fixed evidence fixture for the preregistered Gate 2 lane",
        confidence=1.0,
        status="active",
        source_turn=source_turn,
        evidence="preregistered typed relationship profile",
    )


def compile_gate2_relationship_readout(
    profile: Gate2RelationshipProfile,
) -> ConditioningBankReadout:
    """Compile a frozen typed profile through the real cognition owner."""

    rapport = _record(f"{profile.profile_id}:rapport", source_turn=1)
    tensions = tuple(
        _record(f"{profile.profile_id}:tension:{index}", source_turn=index + 2)
        for index in range(profile.unresolved_tension_count)
    )
    denied = (
        (_record(f"{profile.profile_id}:boundary", source_turn=2),)
        if profile.denied_boundary
        else ()
    )
    relationship = RelationshipStateSnapshot(
        trust_level=profile.trust_level,
        continuity_level=profile.continuity_level,
        repair_pressure=profile.repair_pressure,
        rapport_signals=(rapport,),
        relational_tensions=tensions,
        control_signal=0.0,
        description="preregistered relationship owner fixture",
        emotional_load=profile.emotional_load,
        stabilization_need=profile.stabilization_need,
        recent_repair_count=profile.recent_repair_count,
        unresolved_tension_count=profile.unresolved_tension_count,
        trust_recovery_signal=profile.trust_recovery_signal,
        relationship_continuity_score=profile.continuity_level,
        cumulative_trust_level=profile.cumulative_trust_level,
        relationship_age_turns=profile.relationship_age_turns,
    )
    boundary = BoundaryConsentSnapshot(
        granted_consents=(
            _record(f"{profile.profile_id}:consent", source_turn=1),
        ),
        missing_consents=(),
        denied_boundaries=denied,
        memory_consent="granted",
        external_action_consent=(
            "denied" if profile.denied_boundary else "granted"
        ),
        compliance_score=profile.compliance_score,
        control_signal=0.0,
        description="preregistered boundary owner fixture",
        consent_clarity=profile.consent_clarity,
        denial_count=int(profile.denied_boundary),
        external_action_blocked=profile.denied_boundary,
    )
    upstream = {
        "relationship_state": Snapshot(
            slot_name="relationship_state",
            owner="RelationshipStateModule",
            version=1,
            timestamp_ms=0,
            value=relationship,
        ),
        "boundary_consent": Snapshot(
            slot_name="boundary_consent",
            owner="BoundaryConsentModule",
            version=1,
            timestamp_ms=0,
            value=boundary,
        ),
    }
    module = RelationshipConditioningModule(
        wiring_level=WiringLevel.ACTIVE,
        credit_feedback_level=WiringLevel.DISABLED,
    )
    return asyncio.run(module.process(upstream)).value


def gate2_relationship_readouts() -> dict[str, ConditioningBankReadout]:
    readouts = {
        profile.profile_id: compile_gate2_relationship_readout(profile)
        for profile in GATE2_RELATIONSHIP_PROFILES
    }
    if any(
        readout.readout_labels != RELATIONSHIP_CONDITIONING_READOUT_LABELS
        for readout in readouts.values()
    ):
        raise RuntimeError("Relationship owner label contract drifted")
    if len({readout.source_fingerprint for readout in readouts.values()}) != len(
        readouts
    ):
        raise RuntimeError("Relationship owner profiles collapsed")
    return readouts


def build_gate2_conditioned_source_plans(
    *,
    seed: int,
    count: int,
) -> tuple[Gate2ConditionedSourcePlan, ...]:
    allowed = (GATE2_CONDITIONED_TRAINING_SEED,) + (
        GATE2_CONDITIONED_EVALUATION_SEEDS
    )
    if seed not in allowed:
        raise ValueError(f"unregistered Gate 2 conditioned seed {seed}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("conditioned source count must be positive")
    seed_rank = allowed.index(seed)
    plans = []
    for index in range(count):
        matched_group_index = index // len(GATE2_RELATIONSHIP_PROFILES)
        profile = GATE2_RELATIONSHIP_PROFILES[
            (index + seed_rank) % len(GATE2_RELATIONSHIP_PROFILES)
        ]
        opaque_ref = hashlib.sha256(
            f"gate2-conditioned:{seed}:{matched_group_index}".encode("utf-8")
        ).hexdigest()[:12]
        source = _SOURCE_TEMPLATES[
            (matched_group_index + seed_rank) % len(_SOURCE_TEMPLATES)
        ]
        plans.append(
            Gate2ConditionedSourcePlan(
                schema_version=GATE2_CONDITIONED_SOURCE_SCHEMA_VERSION,
                transition_id=f"gate2-cond-s{seed}-t{index:04d}",
                seed=seed,
                global_index=index,
                consumer_session_index=index // GATE2_CONDITIONED_SESSION_SIZE,
                relationship_profile_id=profile.profile_id,
                opaque_observation_ref=opaque_ref,
                prediction_turn=f"Observation {opaque_ref}. {source}",
                settlement_turn=(
                    f"Observation {opaque_ref}. "
                    f"{_SETTLEMENT_BY_PROFILE[profile.profile_id]}"
                ),
            )
        )
    return tuple(plans)


def gate2_conditioned_plan_digest(
    plans: Sequence[Gate2ConditionedSourcePlan],
) -> str:
    return hashlib.sha256(
        _canonical_bytes(tuple(asdict(plan) for plan in plans))
    ).hexdigest()


def gate2_conditioned_permutation_action_index(
    *,
    seed: int,
    global_index: int,
) -> int:
    if seed not in GATE2_CONDITIONED_EVALUATION_SEEDS:
        raise ValueError(f"unregistered Gate 2 evaluation seed {seed}")
    if global_index < 0:
        raise ValueError("permutation index must be non-negative")
    seed_rank = GATE2_CONDITIONED_EVALUATION_SEEDS.index(seed)
    return (global_index + seed_rank * 7) % 22


def gate2_conditioned_permuted_profile_id(
    *,
    seed: int,
    relationship_profile_id: str,
) -> str:
    profile_ids = tuple(
        profile.profile_id for profile in GATE2_RELATIONSHIP_PROFILES
    )
    if seed not in GATE2_CONDITIONED_EVALUATION_SEEDS:
        raise ValueError(f"unregistered Gate 2 evaluation seed {seed}")
    if relationship_profile_id not in profile_ids:
        raise ValueError("unknown relationship profile")
    seed_rank = GATE2_CONDITIONED_EVALUATION_SEEDS.index(seed)
    current = profile_ids.index(relationship_profile_id)
    return profile_ids[(current + seed_rank + 1) % len(profile_ids)]


class _ContinuationScoringRuntime(Protocol):
    def score_continuation(
        self,
        *,
        source_text: str,
        continuation_text: str,
        applied_control: tuple[float, float, float],
        track_scale: tuple[float, float, float],
    ) -> Any: ...


def build_gate2_conditioned_training_example(
    *,
    plan: Gate2ConditionedSourcePlan,
    snapshot: SubstrateSnapshot,
    relationship_readout: ConditioningBankReadout,
    runtime: _ContinuationScoringRuntime,
    candidate_contract: Gate2CandidateControlContract,
) -> CounterfactualActionExample:
    state = relationship_conditioned_selector_state_vector(
        snapshot,
        relationship_readout,
    )
    zero_nll = runtime.score_continuation(
        source_text=plan.prediction_turn,
        continuation_text=plan.settlement_turn,
        applied_control=candidate_contract.controls[0],
        track_scale=GATE2_CONDITIONED_TRACK_SCALE,
    ).mean_negative_log_likelihood
    deltas = []
    for control in candidate_contract.controls:
        nll = runtime.score_continuation(
            source_text=plan.prediction_turn,
            continuation_text=plan.settlement_turn,
            applied_control=control,
            track_scale=GATE2_CONDITIONED_TRACK_SCALE,
        ).mean_negative_log_likelihood
        deltas.append(float(zero_nll - nll))
    return CounterfactualActionExample(
        example_id=plan.transition_id,
        group_id=f"session-{plan.consumer_session_index}",
        split="train",
        state_features=state,
        candidate_raw_deltas=tuple(deltas),
    )


def fit_gate2_relationship_conditioned_selector(
    examples: Sequence[CounterfactualActionExample],
) -> KernelResidualActionSelectorArtifact:
    return fit_kernel_residual_action_selector(
        tuple(examples),
        ridge_strength=GATE2_CONDITIONED_RIDGE_STRENGTH,
    )


def gate2_conditioned_selector_wrapper(
    *,
    selector: KernelResidualActionSelectorArtifact,
    training_plan_digest: str,
    candidate_contract: Gate2CandidateControlContract,
) -> dict[str, object]:
    expected_dim = 8076 + len(RELATIONSHIP_CONDITIONING_READOUT_LABELS)
    if selector.input_dim != expected_dim:
        raise ValueError(
            "conditioned selector input shape drift: "
            f"expected={expected_dim}, actual={selector.input_dim}"
        )
    if selector.action_count != len(candidate_contract.controls):
        raise ValueError("conditioned selector action count drift")
    return {
        "schema_version": GATE2_CONDITIONED_SELECTOR_WRAPPER_SCHEMA_VERSION,
        "feature_schema_version": (
            RELATIONSHIP_CONDITIONED_SELECTOR_FEATURE_SCHEMA_VERSION
        ),
        "relationship_readout_labels": list(
            RELATIONSHIP_CONDITIONING_READOUT_LABELS
        ),
        "relationship_owner_schema": "relationship-conditioning.v2",
        "training_seed": GATE2_CONDITIONED_TRAINING_SEED,
        "training_count": GATE2_CONDITIONED_TRAINING_COUNT,
        "training_plan_digest": training_plan_digest,
        "ridge_strength": GATE2_CONDITIONED_RIDGE_STRENGTH,
        "control_basis_fingerprint": GATE2_V35_CONTROL_BASIS_FINGERPRINT,
        "candidate_mapping_fingerprint": candidate_contract.mapping_fingerprint,
        "artifact": selector_artifact_to_payload(selector),
    }


def load_gate2_conditioned_selector_wrapper(
    payload: Mapping[str, object],
) -> KernelResidualActionSelectorArtifact:
    if (
        payload.get("schema_version")
        != GATE2_CONDITIONED_SELECTOR_WRAPPER_SCHEMA_VERSION
        or payload.get("feature_schema_version")
        != RELATIONSHIP_CONDITIONED_SELECTOR_FEATURE_SCHEMA_VERSION
        or tuple(payload.get("relationship_readout_labels", ()))
        != RELATIONSHIP_CONDITIONING_READOUT_LABELS
        or payload.get("control_basis_fingerprint")
        != GATE2_V35_CONTROL_BASIS_FINGERPRINT
    ):
        raise ValueError("conditioned selector wrapper contract drift")
    artifact_payload = payload.get("artifact")
    if not isinstance(artifact_payload, Mapping):
        raise ValueError("conditioned selector wrapper lacks artifact")
    selector = selector_artifact_from_payload(artifact_payload)
    if not isinstance(selector, KernelResidualActionSelectorArtifact):
        raise TypeError("conditioned selector must use kernel ridge")
    expected_dim = 8076 + len(RELATIONSHIP_CONDITIONING_READOUT_LABELS)
    if selector.input_dim != expected_dim or selector.action_count != 22:
        raise ValueError("conditioned selector artifact shape drift")
    return selector


def summarize_gate2_conditioned_seed(
    *,
    seed: int,
    rows: Sequence[Mapping[str, object]],
    expected_count: int = GATE2_CONDITIONED_EVALUATION_COUNT,
) -> dict[str, object]:
    selected_minus_permutation = tuple(
        float(row["selected_minus_action_permutation"]) for row in rows
    )
    selected_minus_zero = tuple(
        float(row["selected_minus_zero"]) for row in rows
    )
    selected_minus_condition_permutation = tuple(
        float(row["selected_minus_condition_permutation"]) for row in rows
    )
    by_session: dict[int, list[float]] = {}
    for row in rows:
        by_session.setdefault(int(row["consumer_session_index"]), []).append(
            float(row["selected_minus_condition_permutation"])
        )
    session_means = tuple(
        statistics.fmean(values)
        for _, values in sorted(by_session.items())
    )
    complete = len(rows) == expected_count
    means = {
        "selector_minus_action_permutation_mean": (
            statistics.fmean(selected_minus_permutation)
            if selected_minus_permutation
            else 0.0
        ),
        "selector_minus_zero_mean": (
            statistics.fmean(selected_minus_zero)
            if selected_minus_zero
            else 0.0
        ),
        "selector_minus_condition_permutation_mean": (
            statistics.fmean(selected_minus_condition_permutation)
            if selected_minus_condition_permutation
            else 0.0
        ),
    }
    session_positive_rate = (
        sum(value > 0.0 for value in session_means) / len(session_means)
        if session_means
        else 0.0
    )
    gates = {
        "count_at_least_500": complete and len(rows) >= 500,
        "selector_minus_action_permutation_at_least_0_02": (
            complete
            and means["selector_minus_action_permutation_mean"]
            >= GATE2_CONDITIONED_MIN_EFFECT
        ),
        "selector_minus_zero_at_least_0_02": (
            complete
            and means["selector_minus_zero_mean"]
            >= GATE2_CONDITIONED_MIN_EFFECT
        ),
        "selector_minus_condition_permutation_at_least_0_02": (
            complete
            and means["selector_minus_condition_permutation_mean"]
            >= GATE2_CONDITIONED_MIN_EFFECT
        ),
        "condition_session_positive_rate_at_least_0_60": (
            complete
            and session_positive_rate
            >= GATE2_CONDITIONED_MIN_SESSION_POSITIVE_RATE
        ),
    }
    return {
        "seed": seed,
        "row_count": len(rows),
        "consumer_session_count": len(session_means),
        **means,
        "condition_session_positive_rate": session_positive_rate,
        "gates": gates,
        "complete": complete,
        "single_seed_stoploss_passed": all(gates.values()),
    }


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


__all__ = [
    "GATE2_CONDITIONED_CAPTURE_SCHEMA_VERSION",
    "GATE2_CONDITIONED_EVALUATION_COUNT",
    "GATE2_CONDITIONED_EVALUATION_SEEDS",
    "GATE2_CONDITIONED_MIN_EFFECT",
    "GATE2_CONDITIONED_MIN_SESSION_POSITIVE_RATE",
    "GATE2_CONDITIONED_PREREG_SCHEMA_VERSION",
    "GATE2_CONDITIONED_RIDGE_STRENGTH",
    "GATE2_CONDITIONED_SELECTOR_WRAPPER_SCHEMA_VERSION",
    "GATE2_CONDITIONED_SOURCE_SCHEMA_VERSION",
    "GATE2_CONDITIONED_TRAINING_COUNT",
    "GATE2_CONDITIONED_TRAINING_SEED",
    "GATE2_RELATIONSHIP_PROFILES",
    "Gate2ConditionedSourcePlan",
    "Gate2RelationshipProfile",
    "build_gate2_conditioned_source_plans",
    "build_gate2_conditioned_training_example",
    "compile_gate2_relationship_readout",
    "fit_gate2_relationship_conditioned_selector",
    "gate2_conditioned_permutation_action_index",
    "gate2_conditioned_permuted_profile_id",
    "gate2_conditioned_plan_digest",
    "gate2_conditioned_selector_wrapper",
    "gate2_relationship_readouts",
    "load_gate2_conditioned_selector_wrapper",
    "summarize_gate2_conditioned_seed",
]
