"""Development-only causal isolation of PE-credit learning in the P4 gate.

The frozen P1m named reader, P4.1 subjects, owner loop, Lab authorization and
reactive environment are identical in both arms.  Both arms use the same
zero-initialized LEARNED gate.  The sole intervention is whether exact
settlement -> social PE -> dedicated credit is applied to the gate checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import os
import pathlib
import tempfile

from lifeform_domain_emogpt.lab import (
    P4CanaryMechanismRun,
    load_relationship_p4_longitudinal_canary_evaluator_bundle,
    load_relationship_p4_longitudinal_canary_view,
    load_relationship_transfer_dataset,
    relationship_p4_lab_active_authorization,
    run_relationship_p4_subject_mechanism,
    sha256_json,
)
from lifeform_domain_emogpt.lab.environment import (
    ReactiveRelationshipEnvironment,
)
from lifeform_domain_emogpt.relationship_action_gate import (
    RELATIONSHIP_ACTION_CREDIT_LEVEL,
    RelationshipActionGateMode,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    PrototypeRelationshipPreferenceForecastRuntime,
    RelationshipTextEmbedder,
)
from lifeform_evolution.relationship_lab_p4_named_reader import (
    RelationshipP4NamedReaderTransmissionReport,
)
from lifeform_evolution.relationship_lab_packet1m_qualification import (
    RelationshipP1mQualificationProtocol,
)


P4_PE_LEARNING_REPORT_SCHEMA_VERSION = (
    "relationship-p4-pe-credit-learning-report.v1"
)
_CLAIM_BOUNDARY = (
    "This post-selected development report can show that exact owner "
    "settlement -> social PE -> dedicated credit changes a bounded learned "
    "gate checkpoint and later typed actions/outcomes under process recovery. "
    "It uses the already-seen two-subject P4.1 fixture and a named reader "
    "selected after P1m, so it cannot repair P1m, establish formal Learnable, "
    "conditional held-out gate generalization, Volvence advantage, residual "
    "Steerable, production ACTIVE, or the complete four-able claim."
)


class P4PeLearningArm(str, Enum):
    NAMED_LEARNED_NO_CREDIT = "named_reader_learned_no_credit"
    NAMED_LEARNED_PE_CREDIT = "named_reader_learned_pe_credit"


@dataclass(frozen=True)
class P4PeLearningArmRun:
    arm: P4PeLearningArm
    reader_artifact_id: str
    mechanism: P4CanaryMechanismRun

    def __post_init__(self) -> None:
        _require_sha256(self.reader_artifact_id, "P4.4 reader artifact")
        if self.mechanism.gate_mode is not RelationshipActionGateMode.LEARNED:
            raise ValueError("P4.4 requires LEARNED gate in both arms")
        expected_credit = self.arm is P4PeLearningArm.NAMED_LEARNED_PE_CREDIT
        if self.mechanism.credit_applied_to_gate != expected_credit:
            raise ValueError("P4.4 arm/credit intervention drift")
        if any(
            readout is None
            or readout.reader_artifact_id != self.reader_artifact_id
            for readout in self.mechanism.condition_readouts
        ):
            raise ValueError("P4.4 named reader lineage drift")
        if any(
            audit.credit_level != RELATIONSHIP_ACTION_CREDIT_LEVEL
            or audit.credit_applied_to_gate != expected_credit
            for audit in self.mechanism.gate_audits
        ):
            raise ValueError("P4.4 PE-credit audit lineage drift")
        _validate_checkpoint_chain(self.mechanism)

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "reader_artifact_id": self.reader_artifact_id,
            "subject_scope": self.mechanism.subject_scope,
            "gate_mode": self.mechanism.gate_mode.value,
            "credit_applied_to_gate": self.mechanism.credit_applied_to_gate,
            "traces": [item.to_payload() for item in self.mechanism.traces],
            "gate_audits": [
                item.to_payload() for item in self.mechanism.gate_audits
            ],
            "condition_readouts": [
                {
                    "condition_label": item.condition_label,
                    "confidence": item.confidence,
                    "normalized_margin": item.normalized_margin,
                    "reader_artifact_id": item.reader_artifact_id,
                    "source_observation_sha256": (
                        item.source_observation_sha256
                    ),
                }
                for item in self.mechanism.condition_readouts
                if item is not None
            ],
            "positive_outcome_count": self.mechanism.positive_outcome_count,
            "preferred_action_match_count": (
                self.mechanism.preferred_action_match_count
            ),
            "reversal_opportunity_count": (
                self.mechanism.reversal_opportunity_count
            ),
            "reversal_match_count": self.mechanism.reversal_match_count,
            "gate_update_count": self.mechanism.gate_update_count,
            "process_restart_count": self.mechanism.process_restart_count,
        }


@dataclass(frozen=True)
class P4PeLearningArmSummary:
    arm: P4PeLearningArm
    subject_count: int
    decision_count: int
    steer_count: int
    positive_outcome_count: int
    preferred_action_match_count: int
    reversal_opportunity_count: int
    reversal_match_count: int
    credit_applied_count: int
    parameter_change_count: int
    final_update_count: int
    mean_steer_probability: float
    process_restart_count: int

    def __post_init__(self) -> None:
        if self.subject_count != 2 or self.decision_count != 16:
            raise ValueError("P4.4 development summary shape drift")
        for value in (
            self.steer_count,
            self.positive_outcome_count,
            self.preferred_action_match_count,
            self.reversal_opportunity_count,
            self.reversal_match_count,
            self.credit_applied_count,
            self.parameter_change_count,
            self.final_update_count,
            self.process_restart_count,
        ):
            if value < 0:
                raise ValueError("P4.4 summary count cannot be negative")
        if not 0.0 <= self.mean_steer_probability <= 1.0:
            raise ValueError("P4.4 mean steer probability is invalid")
        expected_credit = self.arm is P4PeLearningArm.NAMED_LEARNED_PE_CREDIT
        if expected_credit:
            if self.credit_applied_count != 16 or self.final_update_count != 16:
                raise ValueError("P4.4 PE arm did not apply all exact credits")
        elif any(
            (
                self.credit_applied_count,
                self.parameter_change_count,
                self.final_update_count,
            )
        ):
            raise ValueError("P4.4 no-credit arm changed learned state")

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "subject_count": self.subject_count,
            "decision_count": self.decision_count,
            "steer_count": self.steer_count,
            "positive_outcome_count": self.positive_outcome_count,
            "preferred_action_match_count": self.preferred_action_match_count,
            "reversal_opportunity_count": self.reversal_opportunity_count,
            "reversal_match_count": self.reversal_match_count,
            "credit_applied_count": self.credit_applied_count,
            "parameter_change_count": self.parameter_change_count,
            "final_update_count": self.final_update_count,
            "mean_steer_probability": self.mean_steer_probability,
            "process_restart_count": self.process_restart_count,
        }


@dataclass(frozen=True)
class RelationshipP4PeLearningReport:
    p4_named_reader_report_artifact_id: str
    p1m_protocol_id: str
    p1m_reader_artifact_id: str
    p4_protocol_sha256: str
    p4_public_plan_sha256: str
    runs: tuple[P4PeLearningArmRun, ...]
    summaries: tuple[P4PeLearningArmSummary, ...]
    matched_probability_change_count: int
    matched_action_change_count: int
    matched_outcome_change_count: int
    causal_next_pulse_probability_change_count: int
    causal_next_pulse_action_change_count: int
    steer_count_gain: int
    preferred_action_match_gain: int
    positive_outcome_gain: int
    reader_frozen_across_arms: bool
    exact_pe_credit_only: bool
    evaluation_feedback_to_learning: bool
    component_selected_after_p1m_observation: bool
    seen_fixture_only: bool
    formal_evidence_authorized: bool
    verdict: str
    claim_boundary: str = _CLAIM_BOUNDARY
    schema_version: str = P4_PE_LEARNING_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != P4_PE_LEARNING_REPORT_SCHEMA_VERSION:
            raise ValueError("P4.4 report schema drift")
        for value in (
            self.p4_named_reader_report_artifact_id,
            self.p1m_protocol_id,
            self.p1m_reader_artifact_id,
            self.p4_protocol_sha256,
            self.p4_public_plan_sha256,
        ):
            _require_sha256(value, "P4.4 report lineage")
        if len(self.runs) != 4:
            raise ValueError("P4.4 report requires four matched runs")
        _validate_matched_run_shape(self.runs)
        expected_summaries = tuple(
            _summarize(self.runs, arm) for arm in P4PeLearningArm
        )
        if self.summaries != expected_summaries:
            raise ValueError("P4.4 derived summary drift")
        comparison = _compare_runs(self.runs)
        observed_comparison = (
            self.matched_probability_change_count,
            self.matched_action_change_count,
            self.matched_outcome_change_count,
            self.causal_next_pulse_probability_change_count,
            self.causal_next_pulse_action_change_count,
        )
        if observed_comparison != comparison:
            raise ValueError("P4.4 matched causal metric drift")
        cold, learned = self.summaries
        expected_gains = (
            learned.steer_count - cold.steer_count,
            learned.preferred_action_match_count
            - cold.preferred_action_match_count,
            learned.positive_outcome_count - cold.positive_outcome_count,
        )
        if (
            self.steer_count_gain,
            self.preferred_action_match_gain,
            self.positive_outcome_gain,
        ) != expected_gains:
            raise ValueError("P4.4 derived gain drift")
        if not (
            self.reader_frozen_across_arms
            and self.exact_pe_credit_only
            and self.component_selected_after_p1m_observation
            and self.seen_fixture_only
        ):
            raise ValueError("P4.4 intervention/claim boundary drift")
        if self.evaluation_feedback_to_learning or self.formal_evidence_authorized:
            raise ValueError("P4.4 evidence firewall is open")
        expected_verdict = _verdict(
            learned=learned,
            causal_probability_changes=(
                self.causal_next_pulse_probability_change_count
            ),
            causal_action_changes=self.causal_next_pulse_action_change_count,
            preferred_gain=self.preferred_action_match_gain,
            positive_gain=self.positive_outcome_gain,
        )
        if self.verdict != expected_verdict:
            raise ValueError("P4.4 verdict drift")
        if self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("P4.4 claim boundary drift")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "p4_named_reader_report_artifact_id": (
                    self.p4_named_reader_report_artifact_id
                ),
                "p1m_protocol_id": self.p1m_protocol_id,
                "p1m_reader_artifact_id": self.p1m_reader_artifact_id,
                "p4_protocol_sha256": self.p4_protocol_sha256,
                "p4_public_plan_sha256": self.p4_public_plan_sha256,
            },
            "intervention": {
                "reader_frozen_across_arms": self.reader_frozen_across_arms,
                "gate_mode_both_arms": RelationshipActionGateMode.LEARNED.value,
                "sole_difference": "apply_exact_pe_derived_credit_to_gate",
                "exact_pe_credit_only": self.exact_pe_credit_only,
                "evaluation_feedback_to_learning": (
                    self.evaluation_feedback_to_learning
                ),
            },
            "runs": [item.to_payload() for item in self.runs],
            "summaries": [item.to_payload() for item in self.summaries],
            "matched_probability_change_count": (
                self.matched_probability_change_count
            ),
            "matched_action_change_count": self.matched_action_change_count,
            "matched_outcome_change_count": self.matched_outcome_change_count,
            "causal_next_pulse_probability_change_count": (
                self.causal_next_pulse_probability_change_count
            ),
            "causal_next_pulse_action_change_count": (
                self.causal_next_pulse_action_change_count
            ),
            "steer_count_gain": self.steer_count_gain,
            "preferred_action_match_gain": self.preferred_action_match_gain,
            "positive_outcome_gain": self.positive_outcome_gain,
            "component_selected_after_p1m_observation": (
                self.component_selected_after_p1m_observation
            ),
            "seen_fixture_only": self.seen_fixture_only,
            "formal_evidence_authorized": self.formal_evidence_authorized,
            "verdict": self.verdict,
            "claim_boundary": self.claim_boundary,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())


async def run_relationship_p4_pe_credit_learning(
    *,
    p1m_protocol: RelationshipP1mQualificationProtocol,
    source_report: RelationshipP4NamedReaderTransmissionReport,
    embedder: RelationshipTextEmbedder,
) -> RelationshipP4PeLearningReport:
    _validate_source(protocol=p1m_protocol, report=source_report)
    view = load_relationship_p4_longitudinal_canary_view()
    evaluator = load_relationship_p4_longitudinal_canary_evaluator_bundle()
    authorization = relationship_p4_lab_active_authorization(view.contract)
    dataset = load_relationship_transfer_dataset(
        package_name=view.contract.source_package_name
    )
    environment = ReactiveRelationshipEnvironment(dataset)
    runs: list[P4PeLearningArmRun] = []
    for subject in view.subjects:
        for arm in P4PeLearningArm:
            runtime = PrototypeRelationshipPreferenceForecastRuntime(
                artifact=p1m_protocol.reader_artifact,
                embedder=embedder,
            )
            mechanism = await run_relationship_p4_subject_mechanism(
                subject=subject,
                evaluator=evaluator,
                authorization=authorization,
                environment=environment,
                forecast_runtime=runtime,
                gate_mode=RelationshipActionGateMode.LEARNED,
                apply_credit_to_gate=(
                    arm is P4PeLearningArm.NAMED_LEARNED_PE_CREDIT
                ),
            )
            runs.append(
                P4PeLearningArmRun(
                    arm=arm,
                    reader_artifact_id=p1m_protocol.reader_artifact.artifact_id,
                    mechanism=mechanism,
                )
            )
    frozen_runs = tuple(runs)
    summaries = tuple(
        _summarize(frozen_runs, arm) for arm in P4PeLearningArm
    )
    (
        probability_changes,
        action_changes,
        outcome_changes,
        next_probability_changes,
        next_action_changes,
    ) = _compare_runs(frozen_runs)
    cold, learned = summaries
    preferred_gain = (
        learned.preferred_action_match_count
        - cold.preferred_action_match_count
    )
    positive_gain = (
        learned.positive_outcome_count - cold.positive_outcome_count
    )
    return RelationshipP4PeLearningReport(
        p4_named_reader_report_artifact_id=source_report.artifact_id,
        p1m_protocol_id=p1m_protocol.protocol_id,
        p1m_reader_artifact_id=p1m_protocol.reader_artifact.artifact_id,
        p4_protocol_sha256=view.contract.protocol_sha256,
        p4_public_plan_sha256=view.public_plan_sha256,
        runs=frozen_runs,
        summaries=summaries,
        matched_probability_change_count=probability_changes,
        matched_action_change_count=action_changes,
        matched_outcome_change_count=outcome_changes,
        causal_next_pulse_probability_change_count=next_probability_changes,
        causal_next_pulse_action_change_count=next_action_changes,
        steer_count_gain=learned.steer_count - cold.steer_count,
        preferred_action_match_gain=preferred_gain,
        positive_outcome_gain=positive_gain,
        reader_frozen_across_arms=True,
        exact_pe_credit_only=True,
        evaluation_feedback_to_learning=False,
        component_selected_after_p1m_observation=True,
        seen_fixture_only=True,
        formal_evidence_authorized=False,
        verdict=_verdict(
            learned=learned,
            causal_probability_changes=next_probability_changes,
            causal_action_changes=next_action_changes,
            preferred_gain=preferred_gain,
            positive_gain=positive_gain,
        ),
    )


def render_relationship_p4_pe_learning_markdown(
    report: RelationshipP4PeLearningReport,
) -> str:
    rows = [
        "# P4 PE-credit learning canary",
        "",
        f"- verdict: `{report.verdict}`",
        f"- artifact: `{report.artifact_id}`",
        (
            "- matched probability/action/outcome changes: "
            f"`{report.matched_probability_change_count}/16`, "
            f"`{report.matched_action_change_count}/16`, "
            f"`{report.matched_outcome_change_count}/16`"
        ),
        (
            "- causal next-pulse probability/action changes: "
            f"`{report.causal_next_pulse_probability_change_count}`, "
            f"`{report.causal_next_pulse_action_change_count}`"
        ),
        "",
        (
            "| arm | steer | action match | positive | reversal | "
            "credit | parameter change |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.summaries:
        rows.append(
            f"| {item.arm.value} | {item.steer_count}/16 | "
            f"{item.preferred_action_match_count}/16 | "
            f"{item.positive_outcome_count}/16 | "
            f"{item.reversal_match_count}/{item.reversal_opportunity_count} | "
            f"{item.credit_applied_count}/16 | "
            f"{item.parameter_change_count}/16 |"
        )
    rows.extend(("", report.claim_boundary, ""))
    return "\n".join(rows)


def write_relationship_p4_pe_learning_report(
    report: RelationshipP4PeLearningReport,
    *,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "pe_credit_learning_report.json"
    markdown_path = output / "pe_credit_learning_report.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(
            "P4.4 PE-learning artifacts are create-only"
        )
    payload = {**report.to_payload(), "artifact_id": report.artifact_id}
    _atomic_create_text(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    try:
        _atomic_create_text(
            markdown_path,
            render_relationship_p4_pe_learning_markdown(report),
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "P4.4 markdown collision left the create-only JSON intact"
        ) from exc
    return json_path, markdown_path


def validate_relationship_p4_pe_learning_report_files(
    report: RelationshipP4PeLearningReport,
    *,
    json_path: pathlib.Path,
    markdown_path: pathlib.Path,
) -> None:
    expected = {**report.to_payload(), "artifact_id": report.artifact_id}
    actual = json.loads(pathlib.Path(json_path).read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("P4.4 PE-learning report artifact drift")
    if pathlib.Path(markdown_path).read_text(
        encoding="utf-8"
    ) != render_relationship_p4_pe_learning_markdown(report):
        raise ValueError("P4.4 PE-learning markdown artifact drift")


def _validate_source(
    *,
    protocol: RelationshipP1mQualificationProtocol,
    report: RelationshipP4NamedReaderTransmissionReport,
) -> None:
    if (
        report.verdict
        != "named_reader_transmission_observed_development_only"
        or report.formal_evidence_authorized
        or not report.seen_fixture_only
        or not report.component_selected_after_p1m_observation
    ):
        raise ValueError("P4.4 requires terminal development P4.3 source")
    if (
        report.p1m_protocol_id != protocol.protocol_id
        or report.p1m_reader_artifact_id
        != protocol.reader_artifact.artifact_id
    ):
        raise ValueError("P4.4 reader/protocol source lineage drift")


def _validate_checkpoint_chain(mechanism: P4CanaryMechanismRun) -> None:
    for index, audit in enumerate(mechanism.gate_audits):
        if index == 0:
            if audit.pre_update_count != 0:
                raise ValueError("P4.4 gate did not start from zero state")
            continue
        previous = mechanism.gate_audits[index - 1]
        if (
            audit.pre_update_state_sha256
            != previous.post_update_state_sha256
            or audit.pre_update_weights != previous.post_update_weights
            or audit.pre_update_bias != previous.post_update_bias
            or audit.pre_update_count != previous.post_update_count
        ):
            raise ValueError("P4.4 gate checkpoint failed process recovery")


def _validate_matched_run_shape(
    runs: tuple[P4PeLearningArmRun, ...],
) -> None:
    scopes_by_arm = {
        arm: tuple(
            run.mechanism.subject_scope for run in runs if run.arm is arm
        )
        for arm in P4PeLearningArm
    }
    if any(
        len(scopes) != 2 or len(set(scopes)) != 2
        for scopes in scopes_by_arm.values()
    ) or set(scopes_by_arm[P4PeLearningArm.NAMED_LEARNED_NO_CREDIT]) != set(
        scopes_by_arm[P4PeLearningArm.NAMED_LEARNED_PE_CREDIT]
    ):
        raise ValueError("P4.4 matched subject shape drift")
    by_key = {(run.arm, run.mechanism.subject_scope): run for run in runs}
    for scope in scopes_by_arm[P4PeLearningArm.NAMED_LEARNED_NO_CREDIT]:
        cold = by_key[(P4PeLearningArm.NAMED_LEARNED_NO_CREDIT, scope)]
        learned = by_key[(P4PeLearningArm.NAMED_LEARNED_PE_CREDIT, scope)]
        if (
            cold.reader_artifact_id != learned.reader_artifact_id
            or cold.mechanism.gate_audits[0].pre_update_state_sha256
            != learned.mechanism.gate_audits[0].pre_update_state_sha256
        ):
            raise ValueError("P4.4 matched arm initialization drift")


def _summarize(
    runs: tuple[P4PeLearningArmRun, ...],
    arm: P4PeLearningArm,
) -> P4PeLearningArmSummary:
    selected = tuple(run for run in runs if run.arm is arm)
    audits = tuple(
        audit for run in selected for audit in run.mechanism.gate_audits
    )
    traces = tuple(
        trace for run in selected for trace in run.mechanism.traces
    )
    return P4PeLearningArmSummary(
        arm=arm,
        subject_count=len(selected),
        decision_count=len(traces),
        steer_count=sum(audit.gate_action == "steer" for audit in audits),
        positive_outcome_count=sum(
            run.mechanism.positive_outcome_count for run in selected
        ),
        preferred_action_match_count=sum(
            run.mechanism.preferred_action_match_count for run in selected
        ),
        reversal_opportunity_count=sum(
            run.mechanism.reversal_opportunity_count for run in selected
        ),
        reversal_match_count=sum(
            run.mechanism.reversal_match_count for run in selected
        ),
        credit_applied_count=sum(
            audit.credit_applied_to_gate for audit in audits
        ),
        parameter_change_count=sum(audit.parameter_changed for audit in audits),
        final_update_count=sum(
            run.mechanism.gate_update_count for run in selected
        ),
        mean_steer_probability=math.fsum(
            audit.steer_probability for audit in audits
        )
        / len(audits),
        process_restart_count=sum(
            run.mechanism.process_restart_count for run in selected
        ),
    )


def _compare_runs(
    runs: tuple[P4PeLearningArmRun, ...],
) -> tuple[int, int, int, int, int]:
    by_key = {(run.arm, run.mechanism.subject_scope): run for run in runs}
    scopes = tuple(
        run.mechanism.subject_scope
        for run in runs
        if run.arm is P4PeLearningArm.NAMED_LEARNED_NO_CREDIT
    )
    probability_changes = 0
    action_changes = 0
    outcome_changes = 0
    next_probability_changes = 0
    next_action_changes = 0
    for scope in scopes:
        cold = by_key[(P4PeLearningArm.NAMED_LEARNED_NO_CREDIT, scope)]
        learned = by_key[(P4PeLearningArm.NAMED_LEARNED_PE_CREDIT, scope)]
        for index, (cold_trace, learned_trace, cold_audit, learned_audit) in enumerate(
            zip(
                cold.mechanism.traces,
                learned.mechanism.traces,
                cold.mechanism.gate_audits,
                learned.mechanism.gate_audits,
                strict=True,
            )
        ):
            if (
                cold_trace.session_id != learned_trace.session_id
                or cold_audit.session_id != learned_audit.session_id
            ):
                raise ValueError("P4.4 matched session drift")
            probability_differs = not math.isclose(
                cold_audit.steer_probability,
                learned_audit.steer_probability,
                abs_tol=1e-15,
            )
            action_differs = (
                cold_trace.exposed_action_id != learned_trace.exposed_action_id
            )
            probability_changes += int(probability_differs)
            action_changes += int(action_differs)
            outcome_changes += int(
                cold_trace.observed_outcome_id
                != learned_trace.observed_outcome_id
            )
            if index > 0 and learned.mechanism.gate_audits[
                index - 1
            ].parameter_changed:
                next_probability_changes += int(probability_differs)
                next_action_changes += int(action_differs)
    return (
        probability_changes,
        action_changes,
        outcome_changes,
        next_probability_changes,
        next_action_changes,
    )


def _verdict(
    *,
    learned: P4PeLearningArmSummary,
    causal_probability_changes: int,
    causal_action_changes: int,
    preferred_gain: int,
    positive_gain: int,
) -> str:
    parameter_learning = (
        learned.parameter_change_count > 0
        and causal_probability_changes > 0
    )
    behavioral_transmission = parameter_learning and causal_action_changes > 0
    beneficial = behavioral_transmission and preferred_gain > 0 and positive_gain > 0
    if beneficial:
        return "pe_credit_learning_transmission_observed_development_only"
    if behavioral_transmission:
        return "pe_credit_changed_behavior_without_positive_gain"
    if parameter_learning:
        return "pe_credit_changed_parameters_without_action_transmission"
    return "pe_credit_learning_not_observed"


def _atomic_create_text(path: pathlib.Path, content: str) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = pathlib.Path(handle.name)
    try:
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _require_sha256(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


__all__ = [
    "P4_PE_LEARNING_REPORT_SCHEMA_VERSION",
    "P4PeLearningArm",
    "P4PeLearningArmRun",
    "P4PeLearningArmSummary",
    "RelationshipP4PeLearningReport",
    "render_relationship_p4_pe_learning_markdown",
    "run_relationship_p4_pe_credit_learning",
    "validate_relationship_p4_pe_learning_report_files",
    "write_relationship_p4_pe_learning_report",
]
