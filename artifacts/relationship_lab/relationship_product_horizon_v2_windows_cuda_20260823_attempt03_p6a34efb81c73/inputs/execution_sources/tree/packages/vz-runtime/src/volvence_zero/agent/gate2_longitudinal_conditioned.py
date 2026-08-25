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
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Protocol, Sequence

from volvence_zero.agent.gate2_longitudinal_capture import (
    GATE2_V35_CONTROL_BASIS_FINGERPRINT,
    Gate2CandidateControlContract,
    build_gate2_longitudinal_capture_runtime,
    load_gate2_candidate_control_contract,
)
from volvence_zero.agent.shared_settled_trace import (
    shared_trace_runtime_fingerprint,
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
    SemanticRecord,
)
from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind


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
_T_CRITICAL_95_DF2 = 4.302652729749
GATE2_CONDITIONED_REQUIRED_FILES = (
    "manifest.json",
    "selector_artifact.json",
    "outcomes.jsonl",
    "prediction_errors.jsonl",
    "segments.jsonl",
    "action_selection.jsonl",
    "ablation_results.json",
    "promotion_verdict.json",
    "rollback_evidence.json",
    "report.md",
)
GATE2_CONDITIONED_CODE_PATHS = (
    "packages/vz-temporal/src/volvence_zero/internal_rl/counterfactual_selector.py",
    "packages/vz-temporal/src/volvence_zero/internal_rl/__init__.py",
    "packages/vz-cognition/src/volvence_zero/relationship_conditioning.py",
    "packages/vz-contracts/src/volvence_zero/conditioning_bank_contracts.py",
    "packages/vz-runtime/src/volvence_zero/agent/gate2_longitudinal_conditioned.py",
    "packages/vz-runtime/src/volvence_zero/agent/gate2_longitudinal_capture.py",
    "packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py",
    "packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py",
    "scripts/preregister_gate2_longitudinal_conditioned.py",
    "scripts/run_gate2_longitudinal_conditioned.py",
)


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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row["record_sha256"] = hashlib.sha256(_canonical_bytes(row)).hexdigest()
    with path.open("ab") as handle:
        handle.write(_canonical_bytes(row) + b"\n")
        handle.flush()


def _load_jsonl(path: Path, *, id_field: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"blank conditioned row at {path}:{line_number}")
        payload = json.loads(line)
        actual_digest = payload.pop("record_sha256", None)
        expected_digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        payload["record_sha256"] = actual_digest
        if actual_digest != expected_digest:
            raise ValueError(
                f"conditioned row digest mismatch at {path}:{line_number}"
            )
        row_id = payload.get(id_field)
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"conditioned row lacks {id_field}")
        if row_id in rows:
            raise ValueError(f"duplicate conditioned row {row_id!r}")
        rows[row_id] = payload
    return rows


def gate2_conditioned_code_manifest(repo_root: str | Path) -> dict[str, str]:
    root = Path(repo_root)
    manifest = {}
    for relative in GATE2_CONDITIONED_CODE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Gate 2 conditioned code binding lacks {relative}"
            )
        manifest[relative] = _sha256_file(path)
    return manifest


def gate2_conditioned_code_tree_sha256(
    manifest: Mapping[str, str],
) -> str:
    return hashlib.sha256(_canonical_bytes(dict(sorted(manifest.items())))).hexdigest()


def build_gate2_conditioned_preregistration(
    *,
    repo_root: str | Path,
    candidate_artifact_path: str | Path,
    substrate_fingerprint: str,
) -> dict[str, object]:
    """Freeze the complete L3-B contract before selector fitting or eval."""

    root = Path(repo_root)
    candidate_path = Path(candidate_artifact_path)
    if not candidate_path.is_absolute():
        candidate_path = root / candidate_path
    candidate_contract = load_gate2_candidate_control_contract(candidate_path)
    code_manifest = gate2_conditioned_code_manifest(root)
    readouts = gate2_relationship_readouts()
    training_plans = build_gate2_conditioned_source_plans(
        seed=GATE2_CONDITIONED_TRAINING_SEED,
        count=GATE2_CONDITIONED_TRAINING_COUNT,
    )
    evaluation_plan_digests = {
        str(seed): gate2_conditioned_plan_digest(
            build_gate2_conditioned_source_plans(
                seed=seed,
                count=GATE2_CONDITIONED_EVALUATION_COUNT,
            )
        )
        for seed in GATE2_CONDITIONED_EVALUATION_SEEDS
    }
    return {
        "schema_version": GATE2_CONDITIONED_PREREG_SCHEMA_VERSION,
        "created_at_unix_ms": int(time.time() * 1000),
        "mechanism": {
            "feature_schema_version": (
                RELATIONSHIP_CONDITIONED_SELECTOR_FEATURE_SCHEMA_VERSION
            ),
            "historical_v35_unconditioned_shape": 8076,
            "conditioned_input_shape": (
                8076 + len(RELATIONSHIP_CONDITIONING_READOUT_LABELS)
            ),
            "relationship_readout_labels": list(
                RELATIONSHIP_CONDITIONING_READOUT_LABELS
            ),
            "relationship_profile_source_fingerprints": {
                profile_id: readout.source_fingerprint
                for profile_id, readout in sorted(readouts.items())
            },
            "silent_unconditioned_fallback_allowed": False,
            "live_injection_enabled": False,
        },
        "substrate": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "cpu",
            "strict_local": True,
            "activation_width": 896,
            "hook_layers": [20, 21, 22],
            "fingerprint": substrate_fingerprint,
            "control_basis_fingerprint": (
                GATE2_V35_CONTROL_BASIS_FINGERPRINT
            ),
            "track_scale": list(GATE2_CONDITIONED_TRACK_SCALE),
        },
        "training": {
            "seed": GATE2_CONDITIONED_TRAINING_SEED,
            "count": GATE2_CONDITIONED_TRAINING_COUNT,
            "plan_digest": gate2_conditioned_plan_digest(training_plans),
            "fit_split": "train-only",
            "ridge_strength": GATE2_CONDITIONED_RIDGE_STRENGTH,
            "candidate_action_count": len(candidate_contract.controls),
            "candidate_artifact_path": str(
                candidate_path.relative_to(root)
            ),
            "candidate_artifact_sha256": _sha256_file(candidate_path),
            "candidate_mapping_fingerprint": (
                candidate_contract.mapping_fingerprint
            ),
        },
        "evaluation": {
            "seeds": list(GATE2_CONDITIONED_EVALUATION_SEEDS),
            "count_per_seed": GATE2_CONDITIONED_EVALUATION_COUNT,
            "plan_digests": evaluation_plan_digests,
            "consumer_session_size": GATE2_CONDITIONED_SESSION_SIZE,
            "action_permutation_schedule": (
                "(global_index+seed_rank*7)%22"
            ),
            "condition_permutation_schedule": (
                "(profile_index+seed_rank+1)%4"
            ),
            "min_effect": GATE2_CONDITIONED_MIN_EFFECT,
            "min_condition_session_positive_rate": (
                GATE2_CONDITIONED_MIN_SESSION_POSITIVE_RATE
            ),
            "single_seed_stoploss_seed": (
                GATE2_CONDITIONED_EVALUATION_SEEDS[0]
            ),
            "single_seed_stoploss_blocks_later_seeds": True,
            "cross_seed_ci": "two-sided-95%-t-ci-lower>=0.02",
        },
        "forbidden_reuse": {
            "historical_seeds": [1201, 1213, 1223],
            "historical_selector_fingerprint": (
                "ef360e0e72e00d235e7fc0df39b249178e080bf2065c6443dad801dfd77f4293"
            ),
            "historical_routes_reused": False,
            "threshold_relaxation_allowed": False,
            "post_result_refit_allowed": False,
        },
        "authorization": {
            "training_and_seed_1301": True,
            "later_seeds_require_seed_1301_stoploss_pass": True,
            "production_promotion_before_full_pass": False,
            "failure_terminally_closes_gate2_longitudinal_claim": True,
        },
        "code_manifest": code_manifest,
        "code_tree_sha256": gate2_conditioned_code_tree_sha256(code_manifest),
    }


def validate_gate2_conditioned_preregistration(
    payload: Mapping[str, object],
    *,
    repo_root: str | Path,
) -> None:
    if payload.get("schema_version") != GATE2_CONDITIONED_PREREG_SCHEMA_VERSION:
        raise ValueError("unsupported Gate 2 conditioned preregistration")
    current_manifest = gate2_conditioned_code_manifest(repo_root)
    if payload.get("code_manifest") != current_manifest:
        raise ValueError("Gate 2 conditioned code manifest drift")
    if payload.get("code_tree_sha256") != gate2_conditioned_code_tree_sha256(
        current_manifest
    ):
        raise ValueError("Gate 2 conditioned code tree digest drift")
    training = payload.get("training")
    evaluation = payload.get("evaluation")
    substrate = payload.get("substrate")
    authorization = payload.get("authorization")
    if not all(
        isinstance(value, Mapping)
        for value in (training, evaluation, substrate, authorization)
    ):
        raise ValueError("conditioned preregistration sections are incomplete")
    assert isinstance(training, Mapping)
    assert isinstance(evaluation, Mapping)
    assert isinstance(substrate, Mapping)
    assert isinstance(authorization, Mapping)
    candidate_path = Path(repo_root) / str(training["candidate_artifact_path"])
    if _sha256_file(candidate_path) != training.get("candidate_artifact_sha256"):
        raise ValueError("conditioned candidate artifact drift")
    candidate_contract = load_gate2_candidate_control_contract(candidate_path)
    if (
        candidate_contract.mapping_fingerprint
        != training.get("candidate_mapping_fingerprint")
    ):
        raise ValueError("conditioned candidate mapping drift")
    training_plans = build_gate2_conditioned_source_plans(
        seed=GATE2_CONDITIONED_TRAINING_SEED,
        count=GATE2_CONDITIONED_TRAINING_COUNT,
    )
    if gate2_conditioned_plan_digest(training_plans) != training.get(
        "plan_digest"
    ):
        raise ValueError("conditioned training plan drift")
    expected_eval = {
        str(seed): gate2_conditioned_plan_digest(
            build_gate2_conditioned_source_plans(
                seed=seed,
                count=GATE2_CONDITIONED_EVALUATION_COUNT,
            )
        )
        for seed in GATE2_CONDITIONED_EVALUATION_SEEDS
    }
    if evaluation.get("plan_digests") != expected_eval:
        raise ValueError("conditioned evaluation plan drift")
    if (
        substrate.get("control_basis_fingerprint")
        != GATE2_V35_CONTROL_BASIS_FINGERPRINT
        or authorization.get("training_and_seed_1301") is not True
        or authorization.get("production_promotion_before_full_pass") is not False
    ):
        raise ValueError("conditioned authorization contract drift")


def write_gate2_conditioned_preregistration(
    *,
    payload: Mapping[str, object],
    output_path: str | Path,
) -> dict[str, object]:
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"conditioned preregistration exists: {target}")
    _write_json(target, payload)
    manifest = {
        "schema_version": "gate2-conditioned-prereg-manifest.v1",
        "artifact_path": str(target),
        "artifact_sha256": _sha256_file(target),
        "artifact_size": target.stat().st_size,
    }
    manifest_path = target.with_suffix(".manifest.json")
    _write_json(manifest_path, manifest)
    return manifest


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


def _capture_conditioned_source_snapshot(
    runtime: Any,
    *,
    source_text: str,
) -> SubstrateSnapshot:
    capture = runtime.capture(source_text=source_text)
    if runtime.fallback_active:
        raise RuntimeError("Gate 2 conditioned capture entered fallback")
    if not capture.residual_sequence or not capture.residual_activations:
        raise RuntimeError("Gate 2 conditioned capture lacks real residuals")
    return SubstrateSnapshot(
        model_id=runtime.model_id,
        is_frozen=runtime.is_frozen,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=capture.token_logits,
        feature_surface=capture.feature_surface,
        residual_activations=capture.residual_activations,
        residual_sequence=capture.residual_sequence,
        unavailable_fields=(),
        description=(
            f"{capture.description} Relationship-conditioned Gate 2 readout."
        ),
    )


def _strict_progress_prefix(
    rows: Mapping[str, Mapping[str, Any]],
    plans: Sequence[Gate2ConditionedSourcePlan],
) -> None:
    expected = tuple(plan.transition_id for plan in plans[: len(rows)])
    if tuple(rows) != expected:
        raise ValueError("Gate 2 conditioned progress is not a strict prefix")


def _training_example_from_row(
    row: Mapping[str, object],
) -> CounterfactualActionExample:
    features = row.get("state_features")
    deltas = row.get("candidate_raw_deltas")
    if not isinstance(features, list) or not isinstance(deltas, list):
        raise ValueError("conditioned training row lacks numeric arrays")
    resolved_features = tuple(
        _finite(value, field="state_features") for value in features
    )
    resolved_deltas = tuple(
        _finite(value, field="candidate_raw_deltas") for value in deltas
    )
    if len(resolved_features) != 8076 + len(
        RELATIONSHIP_CONDITIONING_READOUT_LABELS
    ):
        raise ValueError("conditioned training feature shape drift")
    if len(resolved_deltas) != 22:
        raise ValueError("conditioned training action shape drift")
    return CounterfactualActionExample(
        example_id=str(row["transition_id"]),
        group_id=str(row["group_id"]),
        split="train",
        state_features=resolved_features,
        candidate_raw_deltas=resolved_deltas,
    )


def _confidence_interval_95(
    values: Sequence[float],
) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if values else 0.0
        return (value, value)
    mean = statistics.fmean(values)
    half_width = (
        _T_CRITICAL_95_DF2
        * statistics.stdev(values)
        / math.sqrt(len(values))
    )
    return (mean - half_width, mean + half_width)


def _export_gate2_conditioned_bundle(
    *,
    output_root: Path,
    prereg_path: Path,
    prereg_payload: Mapping[str, object],
    selector_wrapper: Mapping[str, object],
    seed_summaries: Sequence[Mapping[str, object]],
    basis_provenance: Mapping[str, str],
    source_hashes_before: Mapping[str, str],
) -> dict[str, object]:
    complete_summaries = tuple(
        summary for summary in seed_summaries if bool(summary["complete"])
    )
    all_complete = len(complete_summaries) == len(
        GATE2_CONDITIONED_EVALUATION_SEEDS
    )
    all_seed_gates = all(
        bool(summary["single_seed_stoploss_passed"])
        for summary in complete_summaries
    ) and all_complete
    primary_means = tuple(
        float(summary["selector_minus_condition_permutation_mean"])
        for summary in complete_summaries
    )
    confidence_interval = _confidence_interval_95(primary_means)
    cross_seed_ci_gate = (
        all_complete
        and confidence_interval[0] >= GATE2_CONDITIONED_MIN_EFFECT
    )
    readout_supported = all_seed_gates and cross_seed_ci_gate
    first_summary = next(
        (
            summary
            for summary in seed_summaries
            if int(summary["seed"])
            == GATE2_CONDITIONED_EVALUATION_SEEDS[0]
        ),
        None,
    )
    single_seed_stoploss_failed = bool(
        first_summary is not None
        and first_summary["complete"]
        and not first_summary["single_seed_stoploss_passed"]
    )
    status = (
        "longitudinal-supported"
        if readout_supported
        else "single-seed-stoploss"
        if single_seed_stoploss_failed
        else "not-supported"
        if all_complete
        else "capture-in-progress"
    )
    source_hashes_after = {
        path: _sha256_file(Path(path)) for path in source_hashes_before
    }
    if source_hashes_after != dict(source_hashes_before):
        raise RuntimeError("Gate 2 conditioned run mutated a frozen input")
    manifest = {
        "schema_version": GATE2_CONDITIONED_CAPTURE_SCHEMA_VERSION,
        "preregistration_path": str(prereg_path),
        "preregistration_sha256": _sha256_file(prereg_path),
        "code_tree_sha256": prereg_payload["code_tree_sha256"],
        "selector_fingerprint": selector_wrapper["artifact"][
            "model_fingerprint"
        ],
        "feature_schema_version": (
            RELATIONSHIP_CONDITIONED_SELECTOR_FEATURE_SCHEMA_VERSION
        ),
        "relationship_readout_labels": list(
            RELATIONSHIP_CONDITIONING_READOUT_LABELS
        ),
        "basis_provenance": dict(basis_provenance),
        "required_files": list(GATE2_CONDITIONED_REQUIRED_FILES),
    }
    _write_json(output_root / "manifest.json", manifest)
    with (output_root / "outcomes.jsonl").open("w", encoding="utf-8") as handle:
        for summary in seed_summaries:
            handle.write(_canonical_bytes(summary).decode("utf-8") + "\n")
    with (output_root / "prediction_errors.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for summary in seed_summaries:
            row = {
                "seed": summary["seed"],
                "selector_minus_action_permutation_mean": summary[
                    "selector_minus_action_permutation_mean"
                ],
                "selector_minus_zero_mean": summary[
                    "selector_minus_zero_mean"
                ],
                "selector_minus_condition_permutation_mean": summary[
                    "selector_minus_condition_permutation_mean"
                ],
            }
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")
    with (output_root / "segments.jsonl").open("w", encoding="utf-8") as handle:
        for summary in seed_summaries:
            row = {
                "seed": summary["seed"],
                "consumer_session_count": summary["consumer_session_count"],
                "condition_session_positive_rate": summary[
                    "condition_session_positive_rate"
                ],
            }
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")
    action_rows = []
    for seed in GATE2_CONDITIONED_EVALUATION_SEEDS:
        rows = _load_jsonl(
            output_root / f"seed_{seed}" / "matched_outcomes.jsonl",
            id_field="transition_id",
        )
        counts: dict[int, int] = {}
        conditioned_differences = 0
        for row in rows.values():
            selected = int(row["selected_action_index"])
            counts[selected] = counts.get(selected, 0) + 1
            conditioned_differences += int(
                selected != int(row["condition_permutation_action_index"])
            )
        action_rows.append(
            {
                "seed": seed,
                "selected_action_counts": counts,
                "selected_action_coverage": len(counts),
                "conditioned_action_difference_rate": (
                    conditioned_differences / len(rows) if rows else 0.0
                ),
            }
        )
    with (output_root / "action_selection.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in action_rows:
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")
    ablation = {
        "schema_version": GATE2_CONDITIONED_CAPTURE_SCHEMA_VERSION,
        "seed_summaries": list(seed_summaries),
        "primary_seed_means": list(primary_means),
        "primary_confidence_interval_95": list(confidence_interval),
        "cross_seed_ci_gate": cross_seed_ci_gate,
        "readout_supported": readout_supported,
    }
    _write_json(output_root / "ablation_results.json", ablation)
    verdict = {
        "schema_version": GATE2_CONDITIONED_CAPTURE_SCHEMA_VERSION,
        "status": status,
        "single_seed_stoploss_failed": single_seed_stoploss_failed,
        "all_seeds_complete": all_complete,
        "cross_seed_ci_gate": cross_seed_ci_gate,
        "longitudinal_readout_supported": readout_supported,
        "official_gate2_longitudinal_verdict": (
            "longitudinal-supported" if readout_supported else "not-supported"
        ),
        "inherited_gate2_evidence_level": "causal-supported",
        "promotion_allowed": readout_supported,
        "production_live_promotion_authorized": False,
    }
    _write_json(output_root / "promotion_verdict.json", verdict)
    rollback = {
        "schema_version": GATE2_CONDITIONED_CAPTURE_SCHEMA_VERSION,
        "source_hashes_before": dict(source_hashes_before),
        "source_hashes_after": source_hashes_after,
        "source_unchanged": True,
        "selector_installed_live": False,
        "substrate_weights_updated": False,
        "runtime_owner_state_written": False,
        "rollback": "delete this isolated evidence directory",
    }
    _write_json(output_root / "rollback_evidence.json", rollback)
    first_primary = (
        float(first_summary["selector_minus_condition_permutation_mean"])
        if first_summary is not None
        else 0.0
    )
    report = (
        "# Gate 2 relationship-conditioned longitudinal capture\n\n"
        f"- status: `{status}`\n"
        f"- official Gate 2 longitudinal verdict: "
        f"`{verdict['official_gate2_longitudinal_verdict']}`\n"
        f"- seed 1301 conditioned selector−condition permutation: "
        f"`{first_primary:.9f}`\n"
        f"- all seeds complete: `{all_complete}`\n"
        f"- cross-seed 95% CI: `{confidence_interval}`\n\n"
        "The selector consumed the public Relationship owner readout. No "
        "selector or residual control was installed into a live session.\n"
    )
    (output_root / "report.md").write_text(report, encoding="utf-8")
    freeze_manifest = {
        name: _sha256_file(output_root / name)
        for name in GATE2_CONDITIONED_REQUIRED_FILES
        if (output_root / name).is_file()
    }
    for relative in ("training_examples.jsonl",):
        path = output_root / relative
        if path.is_file():
            freeze_manifest[relative] = _sha256_file(path)
    for seed in GATE2_CONDITIONED_EVALUATION_SEEDS:
        path = output_root / f"seed_{seed}" / "matched_outcomes.jsonl"
        if path.is_file():
            freeze_manifest[str(path.relative_to(output_root))] = _sha256_file(
                path
            )
    _write_json(output_root / "freeze_manifest.json", freeze_manifest)
    return verdict


def run_gate2_conditioned_evidence(
    *,
    repo_root: str | Path,
    preregistration_path: str | Path,
    output_root: str | Path,
    max_training_records: int | None = None,
    max_evaluation_records: int | None = None,
) -> dict[str, object]:
    """Fit once, then run fresh seeds under the frozen single-seed stoploss."""

    root = Path(repo_root)
    prereg_path = Path(preregistration_path)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    validate_gate2_conditioned_preregistration(prereg, repo_root=root)
    training_contract = prereg["training"]
    assert isinstance(training_contract, Mapping)
    candidate_path = root / str(training_contract["candidate_artifact_path"])
    candidate_contract = load_gate2_candidate_control_contract(candidate_path)
    source_hashes_before = {
        str(prereg_path): _sha256_file(prereg_path),
        str(candidate_path): _sha256_file(candidate_path),
    }
    for relative, digest in prereg["code_manifest"].items():
        path = root / relative
        if _sha256_file(path) != digest:
            raise ValueError(f"conditioned source drift before run: {relative}")
        source_hashes_before[str(path)] = digest
    runtime, basis_provenance = build_gate2_longitudinal_capture_runtime()
    substrate_fingerprint = shared_trace_runtime_fingerprint(runtime)
    substrate_contract = prereg["substrate"]
    assert isinstance(substrate_contract, Mapping)
    if substrate_fingerprint != substrate_contract["fingerprint"]:
        raise ValueError(
            "Gate 2 conditioned substrate fingerprint drift: "
            f"expected={substrate_contract['fingerprint']}, "
            f"actual={substrate_fingerprint}"
        )
    readouts = gate2_relationship_readouts()
    training_plans = build_gate2_conditioned_source_plans(
        seed=GATE2_CONDITIONED_TRAINING_SEED,
        count=GATE2_CONDITIONED_TRAINING_COUNT,
    )
    training_path = output / "training_examples.jsonl"
    training_rows = _load_jsonl(training_path, id_field="transition_id")
    _strict_progress_prefix(training_rows, training_plans)
    training_limit = (
        min(GATE2_CONDITIONED_TRAINING_COUNT, max_training_records)
        if max_training_records is not None
        else GATE2_CONDITIONED_TRAINING_COUNT
    )
    for plan in training_plans[len(training_rows) : training_limit]:
        snapshot = _capture_conditioned_source_snapshot(
            runtime,
            source_text=plan.prediction_turn,
        )
        readout = readouts[plan.relationship_profile_id]
        example = build_gate2_conditioned_training_example(
            plan=plan,
            snapshot=snapshot,
            relationship_readout=readout,
            runtime=runtime,
            candidate_contract=candidate_contract,
        )
        row = {
            "schema_version": GATE2_CONDITIONED_SOURCE_SCHEMA_VERSION,
            "transition_id": plan.transition_id,
            "group_id": example.group_id,
            "relationship_profile_id": plan.relationship_profile_id,
            "relationship_source_fingerprint": readout.source_fingerprint,
            "state_features": list(example.state_features),
            "candidate_raw_deltas": list(example.candidate_raw_deltas),
            "substrate_fingerprint": substrate_fingerprint,
            "substrate_mutation_applied": False,
        }
        _append_jsonl(training_path, row)
        training_rows[plan.transition_id] = row
    if len(training_rows) < GATE2_CONDITIONED_TRAINING_COUNT:
        return {
            "status": "training-in-progress",
            "training_count": len(training_rows),
            "promotion_allowed": False,
        }
    examples = tuple(
        _training_example_from_row(row) for row in training_rows.values()
    )
    selector = fit_gate2_relationship_conditioned_selector(examples)
    selector_wrapper = gate2_conditioned_selector_wrapper(
        selector=selector,
        training_plan_digest=str(training_contract["plan_digest"]),
        candidate_contract=candidate_contract,
    )
    selector_path = output / "selector_artifact.json"
    if selector_path.is_file():
        existing_wrapper = json.loads(selector_path.read_text(encoding="utf-8"))
        load_gate2_conditioned_selector_wrapper(existing_wrapper)
        if existing_wrapper != selector_wrapper:
            raise ValueError("conditioned selector refit was not reproducible")
    else:
        _write_json(selector_path, selector_wrapper)
    seed_summaries = []
    for seed in GATE2_CONDITIONED_EVALUATION_SEEDS:
        if seed != GATE2_CONDITIONED_EVALUATION_SEEDS[0]:
            first = seed_summaries[0]
            if not bool(first["single_seed_stoploss_passed"]):
                break
        plans = build_gate2_conditioned_source_plans(
            seed=seed,
            count=GATE2_CONDITIONED_EVALUATION_COUNT,
        )
        outcome_path = output / f"seed_{seed}" / "matched_outcomes.jsonl"
        rows = _load_jsonl(outcome_path, id_field="transition_id")
        _strict_progress_prefix(rows, plans)
        evaluation_limit = (
            min(GATE2_CONDITIONED_EVALUATION_COUNT, max_evaluation_records)
            if max_evaluation_records is not None
            else GATE2_CONDITIONED_EVALUATION_COUNT
        )
        for plan in plans[len(rows) : evaluation_limit]:
            snapshot = _capture_conditioned_source_snapshot(
                runtime,
                source_text=plan.prediction_turn,
            )
            correct_readout = readouts[plan.relationship_profile_id]
            permuted_profile_id = gate2_conditioned_permuted_profile_id(
                seed=seed,
                relationship_profile_id=plan.relationship_profile_id,
            )
            correct_state = relationship_conditioned_selector_state_vector(
                snapshot,
                correct_readout,
            )
            permuted_state = relationship_conditioned_selector_state_vector(
                snapshot,
                readouts[permuted_profile_id],
            )
            selected_values = selector.predict_action_values(correct_state)
            permuted_values = selector.predict_action_values(permuted_state)
            selected_index = max(
                range(22), key=lambda index: (selected_values[index], -index)
            )
            condition_permutation_index = max(
                range(22), key=lambda index: (permuted_values[index], -index)
            )
            action_permutation_index = gate2_conditioned_permutation_action_index(
                seed=seed,
                global_index=plan.global_index,
            )
            indices = {
                0,
                selected_index,
                condition_permutation_index,
                action_permutation_index,
            }
            nll_by_index = {
                index: runtime.score_continuation(
                    source_text=plan.prediction_turn,
                    continuation_text=plan.settlement_turn,
                    applied_control=candidate_contract.controls[index],
                    track_scale=GATE2_CONDITIONED_TRACK_SCALE,
                ).mean_negative_log_likelihood
                for index in indices
            }
            selected_nll = nll_by_index[selected_index]
            row = {
                "schema_version": GATE2_CONDITIONED_CAPTURE_SCHEMA_VERSION,
                "transition_id": plan.transition_id,
                "seed": seed,
                "global_index": plan.global_index,
                "consumer_session_index": plan.consumer_session_index,
                "relationship_profile_id": plan.relationship_profile_id,
                "condition_permutation_profile_id": permuted_profile_id,
                "relationship_source_fingerprint": (
                    correct_readout.source_fingerprint
                ),
                "selected_action_index": selected_index,
                "condition_permutation_action_index": (
                    condition_permutation_index
                ),
                "action_permutation_index": action_permutation_index,
                "selected_minus_action_permutation": (
                    nll_by_index[action_permutation_index] - selected_nll
                ),
                "selected_minus_zero": nll_by_index[0] - selected_nll,
                "selected_minus_condition_permutation": (
                    nll_by_index[condition_permutation_index] - selected_nll
                ),
                "selector_fingerprint": selector.model_fingerprint,
                "feature_schema_version": (
                    RELATIONSHIP_CONDITIONED_SELECTOR_FEATURE_SCHEMA_VERSION
                ),
                "capture_source": "real",
                "fallback_active": False,
                "substrate_mutation_applied": False,
                "live_selector_installed": False,
            }
            _append_jsonl(outcome_path, row)
            rows[plan.transition_id] = row
        summary = summarize_gate2_conditioned_seed(seed=seed, rows=tuple(rows.values()))
        seed_summaries.append(summary)
        _write_json(output / f"seed_{seed}" / "summary.json", summary)
        if summary["complete"] and not summary["single_seed_stoploss_passed"]:
            break
    return _export_gate2_conditioned_bundle(
        output_root=output,
        prereg_path=prereg_path,
        prereg_payload=prereg,
        selector_wrapper=selector_wrapper,
        seed_summaries=seed_summaries,
        basis_provenance=basis_provenance,
        source_hashes_before=source_hashes_before,
    )


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


__all__ = [
    "GATE2_CONDITIONED_CAPTURE_SCHEMA_VERSION",
    "GATE2_CONDITIONED_CODE_PATHS",
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
    "build_gate2_conditioned_preregistration",
    "compile_gate2_relationship_readout",
    "fit_gate2_relationship_conditioned_selector",
    "gate2_conditioned_code_manifest",
    "gate2_conditioned_code_tree_sha256",
    "gate2_conditioned_permutation_action_index",
    "gate2_conditioned_permuted_profile_id",
    "gate2_conditioned_plan_digest",
    "gate2_conditioned_selector_wrapper",
    "gate2_relationship_readouts",
    "load_gate2_conditioned_selector_wrapper",
    "summarize_gate2_conditioned_seed",
    "validate_gate2_conditioned_preregistration",
    "write_gate2_conditioned_preregistration",
]
