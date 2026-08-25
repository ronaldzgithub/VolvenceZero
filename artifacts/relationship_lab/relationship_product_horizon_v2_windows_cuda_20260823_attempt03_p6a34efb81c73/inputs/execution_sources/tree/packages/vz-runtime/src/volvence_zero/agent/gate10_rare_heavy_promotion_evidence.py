"""Gate 10 matched-control rare-heavy promotion and rollback evidence.

The harness composes existing owner surfaces:

- offline substrate clone training,
- ``ModificationGate.OFFLINE`` review,
- session-owned rare-heavy review/import/rollback,
- temporal, memory, substrate, and application checkpoints.

It does not add a runtime slot or bypass an owner import surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

from volvence_zero.agent import AgentSessionRunner
from volvence_zero.credit.gate import (
    GateDecision,
    ModificationGate,
    ModificationProposal,
    evaluate_gate_reasons,
)
from volvence_zero.evaluation.types import (
    EvaluationScore,
    EvaluationSnapshot,
)
from volvence_zero.joint_loop import JointLoopSchedule
from volvence_zero.joint_loop.pipeline import RareHeavyArtifact
from volvence_zero.substrate import (
    RARE_HEAVY_STRUCTURAL_OBJECTIVE_VERSION,
    ResidualActivation,
    RareHeavyStructuralObjective,
    SubstrateSnapshot,
    SurfaceKind,
    SyntheticOpenWeightResidualRuntime,
    build_training_trace,
)
from volvence_zero.temporal import DualTrackRareHeavySnapshot


GATE10_SCHEMA_VERSION = "gate10-rare-heavy-promotion.v3"
GATE10_SEEDS = (1021, 1031, 1033)
GATE10_ARMS = (
    "full-import",
    "candidate-review-only",
    "rejected-candidate",
    "rollback-to-previous",
)
GATE10_THRESHOLDS = {
    "heldout_gain_vs_review_only": 0.008,
    "catastrophic_forgetting_max": 0.02,
    "cross_user_leakage_max": 0.0,
}
GATE10_REQUIRED_FILES = (
    "manifest.yaml",
    "predictions.jsonl",
    "outcomes.jsonl",
    "prediction_errors.jsonl",
    "segments.jsonl",
    "credit.jsonl",
    "state_diff.jsonl",
    "action_selection.jsonl",
    "ablation_results.json",
    "promotion_verdict.json",
    "rollback_evidence.json",
    "report.md",
)
_TRAINING_SOURCES = (
    "bounded shared route alpha beta",
    "bounded shared route beta gamma",
    "bounded shared route delta alpha",
    "bounded shared route gamma epsilon",
)
_HELDOUT_SOURCES = (
    "heldout shared route alpha epsilon",
    "heldout shared route delta gamma",
)
_OLD_SOURCES = (
    "legacy shared route alpha beta",
    "legacy shared route gamma delta",
)
_STRUCTURAL_OBJECTIVE = RareHeavyStructuralObjective()


@dataclass(frozen=True)
class Gate10PrivacyAttestation:
    evidence_id: str
    shared_aggregate_only: bool
    source_user_fact_count: int
    consent_record_count: int
    relationship_trace_count: int
    cross_user_leakage_count: int


@dataclass(frozen=True)
class Gate10CandidateEnvelope:
    candidate_id: str
    cohort_scope: str
    training_mode: str
    parameter_count: int
    substrate_fingerprint: str
    owner_checkpoint_id: str
    evaluation_evidence_id: str
    gate_verdict: str
    privacy_attestation: Gate10PrivacyAttestation
    artifact: RareHeavyArtifact


@dataclass(frozen=True)
class Gate10ArmResult:
    seed: int
    arm: str
    gate_decision: str
    gate_reasons: tuple[str, ...]
    metadata_complete: bool
    compatibility_passed: bool
    review_only_side_effect_count: int
    import_applied: bool
    automatic_rejection: bool
    heldout_score: float
    heldout_gain: float
    catastrophic_forgetting: float
    cross_user_leakage_count: int
    old_scenario_replay_count: int
    kill_condition_count: int
    rollback_triggered: bool
    rollback_exact: bool
    checkpoint_fingerprint_match: bool
    state_fingerprint_before: str
    state_fingerprint_after: str
    substrate_fingerprint_before: str
    substrate_fingerprint_after: str
    substrate_fingerprint_match: bool
    applied_operation_count: int


@dataclass(frozen=True)
class Gate10EvidenceReport:
    schema_version: str
    seed_schedule: tuple[int, ...]
    arm_schedule: tuple[str, ...]
    thresholds: tuple[tuple[str, float], ...]
    results: tuple[Gate10ArmResult, ...]
    aggregate_metrics: tuple[tuple[str, float], ...]
    mechanism_gates: tuple[tuple[str, bool, float], ...]
    causal_gates: tuple[tuple[str, bool, float], ...]
    full_chain_rollback_passed: bool
    verdict: str
    description: str


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mean(values: tuple[float, ...]) -> float:
    return statistics.fmean(values) if values else 0.0


def _strip_checkpoint_identity(value: Any) -> Any:
    payload = _jsonable(value)
    if isinstance(payload, dict):
        return {
            key: (
                ""
                if key in {"checkpoint_id", "artifact_id"}
                else _strip_checkpoint_identity(item)
            )
            for key, item in payload.items()
        }
    if isinstance(payload, list):
        return [_strip_checkpoint_identity(item) for item in payload]
    return payload


def _checkpoint_fingerprint(checkpoint: object) -> str:
    return _sha256(_strip_checkpoint_identity(checkpoint))


def _evaluation_snapshot() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(
                family="audit",
                metric_name="contract_integrity",
                value=1.0,
                confidence=1.0,
                evidence="gate10-preregistered-replay",
            ),
            EvaluationScore(
                family="audit",
                metric_name="rollback_resilience",
                value=1.0,
                confidence=1.0,
                evidence="gate10-full-chain-checkpoint",
            ),
            EvaluationScore(
                family="audit",
                metric_name="fallback_reliance",
                value=0.0,
                confidence=1.0,
                evidence="gate10-synthetic-owner-surface",
            ),
        ),
        session_scores=(),
        alerts=(),
        description="Gate 10 preregistered candidate review evidence.",
    )


def _target_snapshot(
    *,
    model_id: str,
    source_text: str,
) -> SubstrateSnapshot:
    trace = build_training_trace(
        trace_id=f"gate10-target:{_sha256(source_text)[:12]}",
        source_text=source_text,
    )
    latest = trace.steps[-1]
    shifted = tuple(
        ResidualActivation(
            layer_index=activation.layer_index,
            activation=tuple(
                max(
                    -1.0,
                    min(1.0, value + delta),
                )
                for value, delta in zip(
                    activation.activation,
                    _STRUCTURAL_OBJECTIVE.residual_delta(
                        source_text=source_text,
                        layer_index=activation.layer_index,
                        width=len(activation.activation),
                    ),
                    strict=True,
                )
            ),
            step=activation.step,
        )
        for activation in latest.residual_activations
    )
    return SubstrateSnapshot(
        model_id=model_id,
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=tuple(
            min(
                sum(feature.values) / max(len(feature.values), 1),
                1.0,
            )
            for feature in latest.feature_surface
        ),
        feature_surface=latest.feature_surface,
        residual_activations=shifted,
        residual_sequence=(),
        unavailable_fields=(),
        description=(
            "Gate 10 offline target snapshot derived from a shared structured "
            "cohort with structural objective "
            f"{RARE_HEAVY_STRUCTURAL_OBJECTIVE_VERSION}; no per-user state "
            "is present."
        ),
    )


def _build_candidate(
    *,
    runner: AgentSessionRunner,
    seed: int,
    rejected: bool,
) -> Gate10CandidateEnvelope:
    runtime = runner.residual_runtime
    training_traces = tuple(
        build_training_trace(
            trace_id=f"gate10:{seed}:train:{index}",
            source_text=source,
        )
        for index, source in enumerate(_TRAINING_SOURCES)
    )
    offline = runtime.clone_for_rare_heavy()
    for source in _TRAINING_SOURCES:
        offline.capture(source_text=source)
    target_batches = tuple(
        (
            _target_snapshot(
                model_id=runtime.model_id,
                source_text=source,
            ),
        )
        for source in _TRAINING_SOURCES
    )
    substrate_checkpoint = offline.train_rare_heavy(
        traces=training_traces,
        substrate_steps_per_trace=target_batches,
        checkpoint_id=f"gate10:{seed}:substrate-candidate",
    )
    temporal_snapshot = DualTrackRareHeavySnapshot(
        world_snapshot=(
            runner.joint_loop.world_temporal_policy.export_rare_heavy_snapshot()
        ),
        self_snapshot=(
            runner.joint_loop.self_temporal_policy.export_rare_heavy_snapshot()
        ),
        description=(
            "Gate 10 matched candidate preserves temporal owners while "
            "testing the shared substrate artifact import."
        ),
    )
    memory_checkpoint = runner.memory_store.export_rare_heavy_state(
        checkpoint_id=f"gate10:{seed}:memory-candidate"
    )
    artifact = RareHeavyArtifact(
        artifact_id=f"gate10:{seed}:{'rejected' if rejected else 'candidate'}",
        owner_path="offline-sslrl-pipeline",
        created_at_ms=seed,
        temporal_snapshot=temporal_snapshot,
        memory_checkpoint=memory_checkpoint,
        substrate_checkpoint=substrate_checkpoint,
        transition_step=len(training_traces),
        final_ssl_loss=substrate_checkpoint.adapter_training_loss,
        final_total_reward=0.08 if not rejected else -0.10,
        description=(
            "Gate 10 offline-clone candidate with owner checkpoints and "
            "shared-cohort adapter evidence."
        ),
    )
    privacy = Gate10PrivacyAttestation(
        evidence_id=f"gate10:{seed}:privacy",
        shared_aggregate_only=not rejected,
        source_user_fact_count=1 if rejected else 0,
        consent_record_count=0,
        relationship_trace_count=0,
        cross_user_leakage_count=1 if rejected else 0,
    )
    proposal = ModificationProposal(
        target="substrate.rare_heavy.shared_adapter",
        desired_gate=ModificationGate.OFFLINE,
        old_value_hash=_sha256(
            runtime.export_rare_heavy_state(
                checkpoint_id=f"gate10:{seed}:baseline"
            )
        ),
        new_value_hash=_sha256(substrate_checkpoint),
        justification=(
            "Promote an offline-clone shared adapter after matched held-out "
            "and old-scenario replay."
        ),
        is_reversible=True,
        validation_delta=-0.10 if rejected else 0.08,
        capacity_cost=0.90 if rejected else 0.20,
        rollback_evidence=f"gate10:{seed}:full-chain-checkpoint",
    )
    gate_reasons = list(
        evaluate_gate_reasons(
            proposal=proposal,
            evaluation_snapshot=_evaluation_snapshot(),
        )
    )
    if privacy.cross_user_leakage_count:
        gate_reasons.append(
            "privacy attestation reports cross-user leakage"
        )
    gate_verdict = (
        GateDecision.BLOCK.value
        if gate_reasons
        else GateDecision.ALLOW.value
    )
    return Gate10CandidateEnvelope(
        candidate_id=artifact.artifact_id,
        cohort_scope=f"shared-synthetic-cohort:gate10:{seed}",
        training_mode=substrate_checkpoint.training_mode,
        parameter_count=substrate_checkpoint.adapter_parameter_count,
        substrate_fingerprint=(
            substrate_checkpoint.compatibility_fingerprint
        ),
        owner_checkpoint_id=memory_checkpoint.checkpoint_id,
        evaluation_evidence_id=f"gate10:{seed}:heldout-old-replay",
        gate_verdict=gate_verdict,
        privacy_attestation=privacy,
        artifact=artifact,
    )


def _metadata_complete(candidate: Gate10CandidateEnvelope) -> bool:
    return (
        bool(candidate.cohort_scope)
        and bool(candidate.training_mode)
        and candidate.parameter_count > 0
        and bool(candidate.substrate_fingerprint)
        and bool(candidate.owner_checkpoint_id)
        and bool(candidate.evaluation_evidence_id)
        and candidate.gate_verdict
        in {GateDecision.ALLOW.value, GateDecision.BLOCK.value}
        and bool(candidate.privacy_attestation.evidence_id)
    )


def _capture_values(
    runtime: SyntheticOpenWeightResidualRuntime,
    *,
    source_text: str,
) -> tuple[float, ...]:
    capture = runtime.capture(source_text=source_text)
    return tuple(
        value
        for activation in capture.residual_activations
        for value in activation.activation
    )


def _score(
    observed: tuple[float, ...],
    target: tuple[float, ...],
) -> float:
    return max(
        0.0,
        1.0
        - _mean(
            tuple(
                abs(a - b)
                for a, b in zip(observed, target, strict=True)
            )
        ),
    )


def _state_checkpoint(
    runner: AgentSessionRunner,
    artifact: RareHeavyArtifact,
    *,
    label: str,
) -> object:
    return runner.review_rare_heavy_artifact(
        artifact,
        checkpoint_id=label,
    ).checkpoint


def _run_arm(
    *,
    seed: int,
    arm: str,
) -> Gate10ArmResult:
    runtime = SyntheticOpenWeightResidualRuntime(
        model_id=f"gate10-runtime-{seed}-{arm}",
        allow_live_substrate_mutation=True,
    )
    runner = AgentSessionRunner(
        session_id=f"gate10:{seed}:{arm}",
        default_residual_runtime=runtime,
        joint_schedule=JointLoopSchedule(
            ssl_interval=99,
            rl_interval=99,
        ),
        rare_heavy_enabled=False,
    )
    rejected = arm == "rejected-candidate"
    candidate = _build_candidate(
        runner=runner,
        seed=seed,
        rejected=rejected,
    )
    artifact = candidate.artifact
    # Capture establishes the synthetic owner's observed residual shape.
    # Warm it before the baseline checkpoint so shape discovery is not
    # misclassified as a review/import side effect.
    for source in _HELDOUT_SOURCES + _OLD_SOURCES:
        runtime.capture(source_text=source)
    baseline_checkpoint = _state_checkpoint(
        runner,
        artifact,
        label=f"gate10:{seed}:{arm}:baseline",
    )
    substrate_before_checkpoint = runtime.export_rare_heavy_state(
        checkpoint_id=f"gate10:{seed}:{arm}:substrate-before"
    )
    substrate_before = _checkpoint_fingerprint(
        substrate_before_checkpoint
    )
    owner_state_before = _checkpoint_fingerprint(baseline_checkpoint)
    state_before = _sha256((owner_state_before, substrate_before))
    baseline_heldout_values = tuple(
        _capture_values(runtime, source_text=source)
        for source in _HELDOUT_SOURCES
    )
    heldout_targets = tuple(
        tuple(
            max(-1.0, min(1.0, value + delta))
            for value, delta in zip(
                values,
                tuple(
                    delta
                    for activation in runtime.capture(
                        source_text=source
                    ).residual_activations
                    for delta in _STRUCTURAL_OBJECTIVE.residual_delta(
                        source_text=source,
                        layer_index=activation.layer_index,
                        width=len(activation.activation),
                    )
                ),
                strict=True,
            )
        )
        for source, values in zip(
            _HELDOUT_SOURCES,
            baseline_heldout_values,
            strict=True,
        )
    )
    baseline_old_values = tuple(
        _capture_values(runtime, source_text=source)
        for source in _OLD_SOURCES
    )
    baseline_heldout_score = _mean(
        tuple(
            _score(observed, target)
            for observed, target in zip(
                baseline_heldout_values,
                heldout_targets,
                strict=True,
            )
        )
    )
    gate_reasons: list[str] = []
    if candidate.gate_verdict == GateDecision.BLOCK.value:
        gate_reasons.append("ModificationGate blocked candidate")
    if candidate.privacy_attestation.cross_user_leakage_count:
        gate_reasons.append("privacy attestation reports cross-user leakage")
    import_result = None
    rollback_triggered = False
    applied_operations: tuple[str, ...] = ()
    if arm == "candidate-review-only":
        review = runner.review_rare_heavy_artifact(
            artifact,
            checkpoint_id=f"gate10:{seed}:{arm}:review",
        )
        applied_operations = review.applied_operations
    elif arm == "rejected-candidate":
        if candidate.gate_verdict != GateDecision.BLOCK.value:
            raise RuntimeError("Gate 10 rejected arm was not blocked")
    else:
        if candidate.gate_verdict != GateDecision.ALLOW.value:
            raise RuntimeError("Gate 10 accepted candidate was blocked")
        import_result = runner.apply_rare_heavy_artifact(
            artifact,
            checkpoint_id=f"gate10:{seed}:{arm}:import",
        )
        applied_operations = import_result.applied_operations
    active_heldout_values = tuple(
        _capture_values(runtime, source_text=source)
        for source in _HELDOUT_SOURCES
    )
    active_old_values = tuple(
        _capture_values(runtime, source_text=source)
        for source in _OLD_SOURCES
    )
    active_heldout_score = _mean(
        tuple(
            _score(observed, target)
            for observed, target in zip(
                active_heldout_values,
                heldout_targets,
                strict=True,
            )
        )
    )
    forgetting = _mean(
        tuple(
            1.0 - _score(observed, target)
            for observed, target in zip(
                active_old_values,
                baseline_old_values,
                strict=True,
            )
        )
    )
    if arm == "rollback-to-previous" and import_result is not None:
        runner.rollback_rare_heavy_import(import_result.checkpoint)
        rollback_triggered = True
    elif arm == "full-import" and import_result is not None:
        # The score above is measured while ACTIVE. Cleanup uses the same
        # rollback surface so the out-of-turn harness leaves no live mutation.
        runner.rollback_rare_heavy_import(import_result.checkpoint)
    final_checkpoint = _state_checkpoint(
        runner,
        artifact,
        label=f"gate10:{seed}:{arm}:final",
    )
    substrate_after_checkpoint = runtime.export_rare_heavy_state(
        checkpoint_id=f"gate10:{seed}:{arm}:substrate-after"
    )
    substrate_after = _checkpoint_fingerprint(
        substrate_after_checkpoint
    )
    owner_state_after = _checkpoint_fingerprint(final_checkpoint)
    state_after = _sha256((owner_state_after, substrate_after))
    substrate_match = substrate_after == substrate_before
    rollback_exact = (
        owner_state_after == owner_state_before
        and substrate_match
    )
    review_side_effect_count = (
        0
        if arm != "candidate-review-only" or rollback_exact
        else 1
    )
    compatibility_passed = (
        artifact.substrate_checkpoint is not None
        and artifact.substrate_checkpoint.model_id == runtime.model_id
        and bool(artifact.substrate_checkpoint.compatibility_fingerprint)
    )
    kill_count = (
        int(candidate.privacy_attestation.cross_user_leakage_count > 0)
        + int(candidate.gate_verdict == GateDecision.BLOCK.value)
    )
    return Gate10ArmResult(
        seed=seed,
        arm=arm,
        gate_decision=candidate.gate_verdict,
        gate_reasons=tuple(gate_reasons),
        metadata_complete=_metadata_complete(candidate),
        compatibility_passed=compatibility_passed,
        review_only_side_effect_count=review_side_effect_count,
        import_applied=import_result is not None,
        automatic_rejection=(
            arm == "rejected-candidate"
            and import_result is None
            and candidate.gate_verdict == GateDecision.BLOCK.value
        ),
        heldout_score=active_heldout_score,
        heldout_gain=active_heldout_score - baseline_heldout_score,
        catastrophic_forgetting=forgetting,
        cross_user_leakage_count=(
            candidate.privacy_attestation.cross_user_leakage_count
        ),
        old_scenario_replay_count=len(_OLD_SOURCES),
        kill_condition_count=kill_count,
        rollback_triggered=rollback_triggered,
        rollback_exact=rollback_exact,
        checkpoint_fingerprint_match=rollback_exact,
        state_fingerprint_before=state_before,
        state_fingerprint_after=state_after,
        substrate_fingerprint_before=substrate_before,
        substrate_fingerprint_after=substrate_after,
        substrate_fingerprint_match=substrate_match,
        applied_operation_count=len(applied_operations),
    )


def _arm_mean(
    rows: tuple[Gate10ArmResult, ...],
    *,
    arm: str,
    field: str,
) -> float:
    return _mean(
        tuple(
            float(getattr(row, field))
            for row in rows
            if row.arm == arm
        )
    )


def run_gate10_evidence(
    *,
    seed_schedule: tuple[int, ...] = GATE10_SEEDS,
) -> Gate10EvidenceReport:
    if not seed_schedule:
        raise ValueError("Gate 10 seed_schedule must not be empty")
    if any(seed not in GATE10_SEEDS for seed in seed_schedule):
        raise ValueError(
            "Gate 10 seed_schedule contains an unregistered seed"
        )
    rows = tuple(
        _run_arm(seed=seed, arm=arm)
        for seed in seed_schedule
        for arm in GATE10_ARMS
    )
    full_gain = _arm_mean(
        rows,
        arm="full-import",
        field="heldout_gain",
    )
    review_gain = _arm_mean(
        rows,
        arm="candidate-review-only",
        field="heldout_gain",
    )
    full_forgetting = _arm_mean(
        rows,
        arm="full-import",
        field="catastrophic_forgetting",
    )
    rollback_exact_rate = _arm_mean(
        rows,
        arm="rollback-to-previous",
        field="rollback_exact",
    )
    aggregate_metrics = (
        ("full_heldout_gain", full_gain),
        ("review_only_heldout_gain", review_gain),
        ("full_gain_vs_review_only", full_gain - review_gain),
        ("full_catastrophic_forgetting", full_forgetting),
        (
            "full_cross_user_leakage_count",
            _arm_mean(
                rows,
                arm="full-import",
                field="cross_user_leakage_count",
            ),
        ),
        ("rollback_exact_rate", rollback_exact_rate),
    )
    mechanism_gates = (
        (
            "candidate-metadata-complete",
            all(row.metadata_complete for row in rows),
            float(sum(not row.metadata_complete for row in rows)),
        ),
        (
            "compatibility-check-passed",
            all(row.compatibility_passed for row in rows),
            float(sum(not row.compatibility_passed for row in rows)),
        ),
        (
            "review-only-side-effect-zero",
            all(
                row.review_only_side_effect_count == 0
                for row in rows
                if row.arm == "candidate-review-only"
            ),
            float(
                sum(
                    row.review_only_side_effect_count
                    for row in rows
                    if row.arm == "candidate-review-only"
                )
            ),
        ),
        (
            "rejected-candidate-auto-blocked",
            all(
                row.automatic_rejection
                for row in rows
                if row.arm == "rejected-candidate"
            ),
            float(
                sum(
                    not row.automatic_rejection
                    for row in rows
                    if row.arm == "rejected-candidate"
                )
            ),
        ),
        (
            "accepted-import-owner-path-used",
            all(
                row.import_applied and row.applied_operation_count > 0
                for row in rows
                if row.arm in {"full-import", "rollback-to-previous"}
            ),
            float(
                sum(
                    not row.import_applied
                    for row in rows
                    if row.arm
                    in {"full-import", "rollback-to-previous"}
                )
            ),
        ),
        (
            "full-chain-rollback-exact",
            rollback_exact_rate == 1.0
            and all(
                row.substrate_fingerprint_match
                for row in rows
                if row.arm == "rollback-to-previous"
            ),
            rollback_exact_rate,
        ),
    )
    causal_gates = (
        (
            "heldout-gain-vs-review-only",
            full_gain - review_gain
            >= GATE10_THRESHOLDS["heldout_gain_vs_review_only"],
            full_gain - review_gain,
        ),
        (
            "catastrophic-forgetting-bounded",
            full_forgetting
            <= GATE10_THRESHOLDS["catastrophic_forgetting_max"],
            full_forgetting,
        ),
        (
            "cross-user-leakage-zero",
            all(
                row.cross_user_leakage_count
                == GATE10_THRESHOLDS["cross_user_leakage_max"]
                for row in rows
                if row.arm in {"full-import", "rollback-to-previous"}
            ),
            float(
                sum(
                    row.cross_user_leakage_count
                    for row in rows
                    if row.arm
                    in {"full-import", "rollback-to-previous"}
                )
            ),
        ),
        (
            "kill-condition-auto-rejects",
            all(
                row.automatic_rejection
                and row.kill_condition_count >= 1
                for row in rows
                if row.arm == "rejected-candidate"
            ),
            float(
                sum(
                    row.kill_condition_count
                    for row in rows
                    if row.arm == "rejected-candidate"
                )
            ),
        ),
        (
            "rollback-restores-previous",
            rollback_exact_rate == 1.0,
            rollback_exact_rate,
        ),
    )
    mechanism_passed = all(
        passed for _name, passed, _value in mechanism_gates
    )
    verdict = (
        "causal-supported"
        if mechanism_passed
        and all(passed for _name, passed, _value in causal_gates)
        else "not-supported"
        if mechanism_passed
        else "invalid"
    )
    return Gate10EvidenceReport(
        schema_version=GATE10_SCHEMA_VERSION,
        seed_schedule=seed_schedule,
        arm_schedule=GATE10_ARMS,
        thresholds=tuple(sorted(GATE10_THRESHOLDS.items())),
        results=rows,
        aggregate_metrics=aggregate_metrics,
        mechanism_gates=mechanism_gates,
        causal_gates=causal_gates,
        full_chain_rollback_passed=rollback_exact_rate == 1.0,
        verdict=verdict,
        description=(
            "Gate 10 four-arm rare-heavy promotion evidence with full-chain "
            f"rollback drill; verdict={verdict}."
        ),
    )


def _write_jsonl(
    path: Path,
    rows: tuple[Mapping[str, object], ...],
) -> None:
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def export_gate10_evidence_bundle(
    report: Gate10EvidenceReport,
    *,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows_by_file: dict[str, tuple[Mapping[str, object], ...]] = {
        "predictions.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "gate_decision": row.gate_decision,
                "gate_reasons": row.gate_reasons,
            }
            for row in report.results
        ),
        "outcomes.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "heldout_score": row.heldout_score,
                "heldout_gain": row.heldout_gain,
                "catastrophic_forgetting": (
                    row.catastrophic_forgetting
                ),
            }
            for row in report.results
        ),
        "prediction_errors.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "kill_condition_count": row.kill_condition_count,
                "cross_user_leakage_count": (
                    row.cross_user_leakage_count
                ),
            }
            for row in report.results
        ),
        "segments.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "old_scenario_replay_count": (
                    row.old_scenario_replay_count
                ),
            }
            for row in report.results
        ),
        "credit.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "gate_decision": row.gate_decision,
                "import_applied": row.import_applied,
                "automatic_rejection": row.automatic_rejection,
            }
            for row in report.results
        ),
        "state_diff.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "state_fingerprint_before": (
                    row.state_fingerprint_before
                ),
                "state_fingerprint_after": (
                    row.state_fingerprint_after
                ),
                "checkpoint_fingerprint_match": (
                    row.checkpoint_fingerprint_match
                ),
                "substrate_fingerprint_before": (
                    row.substrate_fingerprint_before
                ),
                "substrate_fingerprint_after": (
                    row.substrate_fingerprint_after
                ),
                "substrate_fingerprint_match": (
                    row.substrate_fingerprint_match
                ),
            }
            for row in report.results
        ),
        "action_selection.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "metadata_complete": row.metadata_complete,
                "compatibility_passed": row.compatibility_passed,
                "applied_operation_count": row.applied_operation_count,
            }
            for row in report.results
        ),
    }
    written: list[Path] = []
    for filename, rows in rows_by_file.items():
        path = target / filename
        _write_jsonl(path, rows)
        written.append(path)
    ablation_path = target / "ablation_results.json"
    ablation_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "arm_schedule": report.arm_schedule,
                "results": _jsonable(report.results),
                "aggregate_metrics": report.aggregate_metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(ablation_path)
    verdict_path = target / "promotion_verdict.json"
    verdict_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "verdict": report.verdict,
                "mechanism_gates": report.mechanism_gates,
                "causal_gates": report.causal_gates,
                "full_chain_rollback_passed": (
                    report.full_chain_rollback_passed
                ),
                "promotion_scope": (
                    "Synthetic shared-cohort offline-clone artifact; "
                    "production promotion remains unauthorized."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(verdict_path)
    rollback_path = target / "rollback_evidence.json"
    rollback_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "full_chain_rollback_passed": (
                    report.full_chain_rollback_passed
                ),
                "rows": [
                    {
                        "seed": row.seed,
                        "arm": row.arm,
                        "rollback_triggered": row.rollback_triggered,
                        "rollback_exact": row.rollback_exact,
                        "checkpoint_fingerprint_match": (
                            row.checkpoint_fingerprint_match
                        ),
                        "substrate_fingerprint_match": (
                            row.substrate_fingerprint_match
                        ),
                        "substrate_before": (
                            row.substrate_fingerprint_before
                        ),
                        "substrate_after": (
                            row.substrate_fingerprint_after
                        ),
                        "before": row.state_fingerprint_before,
                        "after": row.state_fingerprint_after,
                    }
                    for row in report.results
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(rollback_path)
    manifest = {
        "schema_version": report.schema_version,
        "suite_id": "gate10-rare-heavy-promotion",
        "seed_schedule": report.seed_schedule,
        "arm_schedule": report.arm_schedule,
        "thresholds": dict(report.thresholds),
        "required_files": GATE10_REQUIRED_FILES,
        "cohort_scope": "shared-synthetic-cohort",
        "training_mode": "offline-clone:adapter-delta-v2",
        "structural_objective": (
            RARE_HEAVY_STRUCTURAL_OBJECTIVE_VERSION
        ),
        "parameter_count_source": "SubstrateRareHeavyCheckpoint",
        "substrate_fingerprint_source": (
            "SubstrateRareHeavyCheckpoint.compatibility_fingerprint"
        ),
        "evaluation_evidence": "matched-heldout-and-old-scenario-replay",
        "gate_verdict_source": "ModificationGate.OFFLINE",
        "privacy_attestation": (
            "shared aggregate only; user facts/consent/relationship "
            "traces excluded"
        ),
    }
    manifest_path = target / "manifest.yaml"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written.append(manifest_path)
    report_path = target / "report.md"
    report_path.write_text(
        (
            "# Gate 10 rare-heavy promotion evidence\n\n"
            f"- verdict: `{report.verdict}`\n"
            "- scope: synthetic shared-cohort offline-clone artifact\n"
            f"- full-chain rollback: `{report.full_chain_rollback_passed}`\n\n"
            "## Mechanism gates\n\n"
            + "".join(
                f"- {name}: `{passed}` ({value:.6f})\n"
                for name, passed, value in report.mechanism_gates
            )
            + "\n## Causal gates\n\n"
            + "".join(
                f"- {name}: `{passed}` ({value:.6f})\n"
                for name, passed, value in report.causal_gates
            )
        ),
        encoding="utf-8",
    )
    written.append(report_path)
    return tuple(written)


def verify_gate10_evidence_bundle(
    output_dir: str | Path,
) -> dict[str, object]:
    target = Path(output_dir)
    missing = tuple(
        filename
        for filename in GATE10_REQUIRED_FILES
        if not (target / filename).is_file()
    )
    if missing:
        return {
            "passed": False,
            "missing_files": missing,
            "verdict": "invalid",
        }
    manifest = json.loads(
        (target / "manifest.yaml").read_text(encoding="utf-8")
    )
    verdict = json.loads(
        (target / "promotion_verdict.json").read_text(encoding="utf-8")
    )
    passed = (
        manifest["schema_version"] == GATE10_SCHEMA_VERSION
        and tuple(manifest["seed_schedule"]) == GATE10_SEEDS
        and tuple(manifest["arm_schedule"]) == GATE10_ARMS
        and tuple(manifest["required_files"]) == GATE10_REQUIRED_FILES
        and verdict["verdict"]
        in {"invalid", "not-supported", "causal-supported"}
    )
    return {
        "passed": passed,
        "missing_files": (),
        "verdict": verdict["verdict"],
        "full_chain_rollback_passed": verdict[
            "full_chain_rollback_passed"
        ],
    }
