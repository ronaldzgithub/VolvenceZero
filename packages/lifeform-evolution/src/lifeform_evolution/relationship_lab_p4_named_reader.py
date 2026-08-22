"""Post-P1m development test of named-reader transmission into P4 actions.

This lane deliberately runs after the failed P1m instrument qualification.  It
does not rescue that verdict.  Instead, it asks one narrower causal question on
the already-seen P4.1 fixture: when the only changed collaborator is the frozen
P1m named condition reader, does the owner-authored forecast change the typed
action and reactive outcome after repeated process restarts?

Both arms use an ALWAYS gate and never apply PE-derived credit, so this package
isolates Readable transmission.  Learnable and residual Steerable remain for
later convergence packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
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
    RelationshipActionGateMode,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    PrototypeRelationshipPreferenceForecastRuntime,
    RelationshipTextEmbedder,
)
from lifeform_domain_emogpt.relationship_forecast import (
    BoundedRelationshipPreferenceForecastRuntime,
)
from lifeform_evolution.relationship_lab_packet1m_qualification import (
    RELATIONSHIP_P1M_PAIR_FLIP_WILSON_LOWER_EXCLUSIVE,
    RelationshipP1mQualificationArm,
    RelationshipP1mQualificationProtocol,
    RelationshipP1mQualificationReport,
    RelationshipP1mQualificationVerdict,
)
from volvence_zero.social_cognition import RelationshipConditionReadout


P4_NAMED_READER_REPORT_SCHEMA_VERSION = (
    "relationship-p4-named-reader-transmission-report.v1"
)
_CLAIM_BOUNDARY = (
    "This post-P1m development report uses the already-seen P4.1 v3 fixture and "
    "selects the sole P1m named reader after observing its terminal directional "
    "result. It can show owner/readout-to-action-to-environment transmission "
    "under repeated hydration. It cannot repair the failed P1m baseline, "
    "authorize P2 formal, establish Volvence advantage or complete Readable, "
    "Learnable, residual Steerable, production ACTIVE, or four-able claims."
)


class P4NamedReaderArm(str, Enum):
    LEGACY_READER_ALWAYS = "legacy_reader_always"
    P1M_NAMED_READER_ALWAYS = "p1m_named_reader_always"


@dataclass(frozen=True)
class P4NamedReaderArmRun:
    arm: P4NamedReaderArm
    reader_artifact_id: str | None
    mechanism: P4CanaryMechanismRun

    def __post_init__(self) -> None:
        if self.mechanism.gate_mode is not RelationshipActionGateMode.ALWAYS:
            raise ValueError("P4 named-reader isolation requires ALWAYS gate")
        if self.mechanism.credit_applied_to_gate:
            raise ValueError("P4 named-reader isolation cannot update the gate")
        if self.mechanism.gate_update_count != 0:
            raise ValueError("P4 named-reader isolation changed gate parameters")
        if self.arm is P4NamedReaderArm.LEGACY_READER_ALWAYS:
            if self.reader_artifact_id is not None or any(
                item is not None for item in self.mechanism.condition_readouts
            ):
                raise ValueError("legacy reader unexpectedly published named state")
        else:
            if self.reader_artifact_id is None:
                raise ValueError("named reader arm requires artifact lineage")
            _require_sha256(self.reader_artifact_id, "reader_artifact_id")
            if any(
                item is None or item.reader_artifact_id != self.reader_artifact_id
                for item in self.mechanism.condition_readouts
            ):
                raise ValueError("named reader arm condition lineage drift")

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "reader_artifact_id": self.reader_artifact_id,
            "subject_scope": self.mechanism.subject_scope,
            "gate_mode": self.mechanism.gate_mode.value,
            "credit_applied_to_gate": self.mechanism.credit_applied_to_gate,
            "traces": [item.to_payload() for item in self.mechanism.traces],
            "condition_readouts": [
                _condition_payload(item)
                for item in self.mechanism.condition_readouts
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
class P4NamedReaderArmSummary:
    arm: P4NamedReaderArm
    subject_count: int
    decision_count: int
    named_readout_count: int
    positive_outcome_count: int
    preferred_action_match_count: int
    reversal_opportunity_count: int
    reversal_match_count: int
    process_restart_count: int

    def __post_init__(self) -> None:
        if self.subject_count != 2 or self.decision_count != 16:
            raise ValueError("P4 named-reader development shape drift")
        if self.arm is P4NamedReaderArm.LEGACY_READER_ALWAYS:
            if self.named_readout_count != 0:
                raise ValueError("legacy reader summary contains named readouts")
        elif self.named_readout_count != self.decision_count:
            raise ValueError("named reader summary is missing readouts")
        for value in (
            self.positive_outcome_count,
            self.preferred_action_match_count,
            self.reversal_opportunity_count,
            self.reversal_match_count,
            self.process_restart_count,
        ):
            if value < 0:
                raise ValueError("P4 named-reader summary count is negative")

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "subject_count": self.subject_count,
            "decision_count": self.decision_count,
            "named_readout_count": self.named_readout_count,
            "positive_outcome_count": self.positive_outcome_count,
            "preferred_action_match_count": self.preferred_action_match_count,
            "reversal_opportunity_count": self.reversal_opportunity_count,
            "reversal_match_count": self.reversal_match_count,
            "process_restart_count": self.process_restart_count,
        }


@dataclass(frozen=True)
class RelationshipP4NamedReaderTransmissionReport:
    p1m_report_artifact_id: str
    p1m_protocol_id: str
    p1m_reader_artifact_id: str
    p4_protocol_sha256: str
    p4_public_plan_sha256: str
    runs: tuple[P4NamedReaderArmRun, ...]
    summaries: tuple[P4NamedReaderArmSummary, ...]
    matched_action_change_count: int
    matched_outcome_change_count: int
    preferred_action_match_gain: int
    positive_outcome_gain: int
    component_selected_after_p1m_observation: bool
    seen_fixture_only: bool
    evaluation_feedback_to_learning: bool
    p2_formal_authorized: bool
    formal_evidence_authorized: bool
    verdict: str
    claim_boundary: str = _CLAIM_BOUNDARY
    schema_version: str = P4_NAMED_READER_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != P4_NAMED_READER_REPORT_SCHEMA_VERSION:
            raise ValueError("P4 named-reader report schema mismatch")
        for value in (
            self.p1m_report_artifact_id,
            self.p1m_protocol_id,
            self.p1m_reader_artifact_id,
            self.p4_protocol_sha256,
            self.p4_public_plan_sha256,
        ):
            _require_sha256(value, "P4 named-reader report lineage")
        if len(self.runs) != 4:
            raise ValueError("P4 named-reader report requires four matched runs")
        scopes_by_arm = {
            arm: tuple(
                item.mechanism.subject_scope
                for item in self.runs
                if item.arm is arm
            )
            for arm in P4NamedReaderArm
        }
        if any(
            len(scopes) != 2 or len(set(scopes)) != 2
            for scopes in scopes_by_arm.values()
        ) or set(scopes_by_arm[P4NamedReaderArm.LEGACY_READER_ALWAYS]) != set(
            scopes_by_arm[P4NamedReaderArm.P1M_NAMED_READER_ALWAYS]
        ):
            raise ValueError("P4 named-reader matched subject shape drift")
        if any(
            item.reader_artifact_id != self.p1m_reader_artifact_id
            for item in self.runs
            if item.arm is P4NamedReaderArm.P1M_NAMED_READER_ALWAYS
        ):
            raise ValueError("P4 named-reader source artifact lineage drift")
        if tuple(item.arm for item in self.summaries) != tuple(P4NamedReaderArm):
            raise ValueError("P4 named-reader summary order drift")
        expected_summaries = tuple(
            _summarize(self.runs, arm) for arm in P4NamedReaderArm
        )
        if self.summaries != expected_summaries:
            raise ValueError("P4 named-reader derived summary drift")
        expected_action_changes, expected_outcome_changes = (
            _matched_change_counts(self.runs)
        )
        if (
            self.matched_action_change_count != expected_action_changes
            or self.matched_outcome_change_count != expected_outcome_changes
        ):
            raise ValueError("P4 named-reader matched change metric drift")
        for value in (
            self.matched_action_change_count,
            self.matched_outcome_change_count,
        ):
            if not 0 <= value <= 16:
                raise ValueError("P4 named-reader matched change count is invalid")
        legacy_summary, named_summary = self.summaries
        if self.preferred_action_match_gain != (
            named_summary.preferred_action_match_count
            - legacy_summary.preferred_action_match_count
        ) or self.positive_outcome_gain != (
            named_summary.positive_outcome_count
            - legacy_summary.positive_outcome_count
        ):
            raise ValueError("P4 named-reader derived gain drift")
        if not (
            self.component_selected_after_p1m_observation
            and self.seen_fixture_only
        ):
            raise ValueError("P4 named-reader post-selection boundary drift")
        if (
            self.evaluation_feedback_to_learning
            or self.p2_formal_authorized
            or self.formal_evidence_authorized
        ):
            raise ValueError("P4 named-reader evidence firewall is open")
        observed = (
            self.matched_action_change_count > 0
            and self.preferred_action_match_gain > 0
        )
        expected_verdict = (
            "named_reader_transmission_observed_development_only"
            if observed
            else "named_reader_transmission_not_observed"
        )
        if self.verdict != expected_verdict:
            raise ValueError("P4 named-reader report verdict drift")
        if self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("P4 named-reader claim boundary drift")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "p1m_report_artifact_id": self.p1m_report_artifact_id,
                "p1m_protocol_id": self.p1m_protocol_id,
                "p1m_reader_artifact_id": self.p1m_reader_artifact_id,
                "p4_protocol_sha256": self.p4_protocol_sha256,
                "p4_public_plan_sha256": self.p4_public_plan_sha256,
            },
            "runs": [item.to_payload() for item in self.runs],
            "summaries": [item.to_payload() for item in self.summaries],
            "matched_action_change_count": self.matched_action_change_count,
            "matched_outcome_change_count": self.matched_outcome_change_count,
            "preferred_action_match_gain": self.preferred_action_match_gain,
            "positive_outcome_gain": self.positive_outcome_gain,
            "component_selected_after_p1m_observation": (
                self.component_selected_after_p1m_observation
            ),
            "seen_fixture_only": self.seen_fixture_only,
            "evaluation_feedback_to_learning": (
                self.evaluation_feedback_to_learning
            ),
            "p2_formal_authorized": self.p2_formal_authorized,
            "formal_evidence_authorized": self.formal_evidence_authorized,
            "verdict": self.verdict,
            "claim_boundary": self.claim_boundary,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())


async def run_relationship_p4_named_reader_transmission(
    *,
    p1m_protocol: RelationshipP1mQualificationProtocol,
    p1m_report: RelationshipP1mQualificationReport,
    embedder: RelationshipTextEmbedder,
    embedding_model_id: str,
    embedding_weights_sha256: str,
) -> RelationshipP4NamedReaderTransmissionReport:
    _validate_p1m_source(
        protocol=p1m_protocol,
        report=p1m_report,
        embedding_model_id=embedding_model_id,
        embedding_weights_sha256=embedding_weights_sha256,
    )
    view = load_relationship_p4_longitudinal_canary_view()
    evaluator = load_relationship_p4_longitudinal_canary_evaluator_bundle()
    authorization = relationship_p4_lab_active_authorization(view.contract)
    dataset = load_relationship_transfer_dataset(
        package_name=view.contract.source_package_name
    )
    environment = ReactiveRelationshipEnvironment(dataset)
    runs: list[P4NamedReaderArmRun] = []
    for subject in view.subjects:
        legacy = await run_relationship_p4_subject_mechanism(
            subject=subject,
            evaluator=evaluator,
            authorization=authorization,
            environment=environment,
            forecast_runtime=BoundedRelationshipPreferenceForecastRuntime(),
            gate_mode=RelationshipActionGateMode.ALWAYS,
            apply_credit_to_gate=False,
        )
        runs.append(
            P4NamedReaderArmRun(
                arm=P4NamedReaderArm.LEGACY_READER_ALWAYS,
                reader_artifact_id=None,
                mechanism=legacy,
            )
        )
        named_runtime = PrototypeRelationshipPreferenceForecastRuntime(
            artifact=p1m_protocol.reader_artifact,
            embedder=embedder,
        )
        named = await run_relationship_p4_subject_mechanism(
            subject=subject,
            evaluator=evaluator,
            authorization=authorization,
            environment=environment,
            forecast_runtime=named_runtime,
            gate_mode=RelationshipActionGateMode.ALWAYS,
            apply_credit_to_gate=False,
        )
        runs.append(
            P4NamedReaderArmRun(
                arm=P4NamedReaderArm.P1M_NAMED_READER_ALWAYS,
                reader_artifact_id=p1m_protocol.reader_artifact.artifact_id,
                mechanism=named,
            )
        )
    frozen_runs = tuple(runs)
    summaries = tuple(
        _summarize(frozen_runs, arm) for arm in P4NamedReaderArm
    )
    action_changes, outcome_changes = _matched_change_counts(frozen_runs)
    legacy_summary, named_summary = summaries
    preferred_gain = (
        named_summary.preferred_action_match_count
        - legacy_summary.preferred_action_match_count
    )
    positive_gain = (
        named_summary.positive_outcome_count
        - legacy_summary.positive_outcome_count
    )
    verdict = (
        "named_reader_transmission_observed_development_only"
        if action_changes > 0 and preferred_gain > 0
        else "named_reader_transmission_not_observed"
    )
    return RelationshipP4NamedReaderTransmissionReport(
        p1m_report_artifact_id=p1m_report.artifact_id,
        p1m_protocol_id=p1m_protocol.protocol_id,
        p1m_reader_artifact_id=p1m_protocol.reader_artifact.artifact_id,
        p4_protocol_sha256=view.contract.protocol_sha256,
        p4_public_plan_sha256=view.public_plan_sha256,
        runs=frozen_runs,
        summaries=summaries,
        matched_action_change_count=action_changes,
        matched_outcome_change_count=outcome_changes,
        preferred_action_match_gain=preferred_gain,
        positive_outcome_gain=positive_gain,
        component_selected_after_p1m_observation=True,
        seen_fixture_only=True,
        evaluation_feedback_to_learning=False,
        p2_formal_authorized=False,
        formal_evidence_authorized=False,
        verdict=verdict,
    )


def write_relationship_p4_named_reader_report(
    report: RelationshipP4NamedReaderTransmissionReport,
    *,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "named_reader_transmission_report.json"
    payload = {**report.to_payload(), "artifact_id": report.artifact_id}
    _atomic_create_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    return path


def write_relationship_p4_named_reader_markdown(
    report: RelationshipP4NamedReaderTransmissionReport,
    *,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "named_reader_transmission_report.md"
    _atomic_create_text(
        path,
        render_relationship_p4_named_reader_markdown(report),
    )
    return path


def validate_relationship_p4_named_reader_report_files(
    report: RelationshipP4NamedReaderTransmissionReport,
    *,
    report_path: pathlib.Path,
    markdown_path: pathlib.Path,
) -> None:
    expected_payload = {**report.to_payload(), "artifact_id": report.artifact_id}
    actual_payload = json.loads(
        pathlib.Path(report_path).read_text(encoding="utf-8")
    )
    if actual_payload != expected_payload:
        raise ValueError("P4 named-reader report artifact drift")
    if (
        actual_payload.get("artifact_id") != report.artifact_id
        or sha256_json(report.to_payload()) != report.artifact_id
    ):
        raise ValueError("P4 named-reader report artifact id mismatch")
    expected_markdown = render_relationship_p4_named_reader_markdown(report)
    if pathlib.Path(markdown_path).read_text(encoding="utf-8") != expected_markdown:
        raise ValueError("P4 named-reader markdown artifact drift")


def render_relationship_p4_named_reader_markdown(
    report: RelationshipP4NamedReaderTransmissionReport,
) -> str:
    rows = [
        "# P4 named-reader transmission canary",
        "",
        f"- verdict: `{report.verdict}`",
        f"- artifact: `{report.artifact_id}`",
        f"- matched action changes: `{report.matched_action_change_count}/16`",
        f"- matched outcome changes: `{report.matched_outcome_change_count}/16`",
        "",
        "| arm | action match | positive outcome | reversal match | named readout |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report.summaries:
        rows.append(
            f"| {item.arm.value} | {item.preferred_action_match_count}/16 | "
            f"{item.positive_outcome_count}/16 | "
            f"{item.reversal_match_count}/{item.reversal_opportunity_count} | "
            f"{item.named_readout_count}/16 |"
        )
    rows.extend(("", report.claim_boundary, ""))
    return "\n".join(rows)


def _validate_p1m_source(
    *,
    protocol: RelationshipP1mQualificationProtocol,
    report: RelationshipP1mQualificationReport,
    embedding_model_id: str,
    embedding_weights_sha256: str,
) -> None:
    if (
        report.protocol_id != protocol.protocol_id
        or report.plan_artifact_id != protocol.plan_artifact_id
        or report.dataset_fingerprint != protocol.dataset_fingerprint
    ):
        raise ValueError("P4 named-reader P1m lineage drift")
    if (
        report.verdict
        is not RelationshipP1mQualificationVerdict.BASELINE_TOO_WEAK
        or report.qualification_passed
        or not report.scenario_versioning_closed
    ):
        raise ValueError("P4 named-reader requires the terminal failed P1m path")
    structured = next(
        item
        for item in report.arm_metrics
        if item.arm is RelationshipP1mQualificationArm.STRUCTURED_STATE
    )
    if (
        structured.valid_decisions != 48
        or structured.pair_flip_wilson_lower
        <= RELATIONSHIP_P1M_PAIR_FLIP_WILSON_LOWER_EXCLUSIVE
    ):
        raise ValueError("P4 named-reader source lacks directional readout signal")
    if embedding_model_id != protocol.reader_artifact.embedding_model_id:
        raise ValueError("P4 named-reader embedding model lineage drift")
    _require_sha256(embedding_weights_sha256, "embedding_weights_sha256")
    if (
        embedding_weights_sha256
        != protocol.reader_artifact.embedding_weights_sha256
    ):
        raise ValueError("P4 named-reader embedding weights lineage drift")


def _summarize(
    runs: tuple[P4NamedReaderArmRun, ...],
    arm: P4NamedReaderArm,
) -> P4NamedReaderArmSummary:
    selected = tuple(item for item in runs if item.arm is arm)
    return P4NamedReaderArmSummary(
        arm=arm,
        subject_count=len(selected),
        decision_count=sum(len(item.mechanism.traces) for item in selected),
        named_readout_count=sum(
            sum(readout is not None for readout in item.mechanism.condition_readouts)
            for item in selected
        ),
        positive_outcome_count=sum(
            item.mechanism.positive_outcome_count for item in selected
        ),
        preferred_action_match_count=sum(
            item.mechanism.preferred_action_match_count for item in selected
        ),
        reversal_opportunity_count=sum(
            item.mechanism.reversal_opportunity_count for item in selected
        ),
        reversal_match_count=sum(
            item.mechanism.reversal_match_count for item in selected
        ),
        process_restart_count=sum(
            item.mechanism.process_restart_count for item in selected
        ),
    )


def _matched_change_counts(
    runs: tuple[P4NamedReaderArmRun, ...],
) -> tuple[int, int]:
    by_key = {
        (run.arm, run.mechanism.subject_scope): run for run in runs
    }
    scopes = tuple(
        run.mechanism.subject_scope
        for run in runs
        if run.arm is P4NamedReaderArm.LEGACY_READER_ALWAYS
    )
    action_changes = 0
    outcome_changes = 0
    for subject_scope in scopes:
        legacy = by_key[
            (P4NamedReaderArm.LEGACY_READER_ALWAYS, subject_scope)
        ]
        named = by_key[
            (P4NamedReaderArm.P1M_NAMED_READER_ALWAYS, subject_scope)
        ]
        for legacy_trace, named_trace in zip(
            legacy.mechanism.traces,
            named.mechanism.traces,
            strict=True,
        ):
            if legacy_trace.session_id != named_trace.session_id:
                raise ValueError("P4 named-reader matched session drift")
            action_changes += int(
                legacy_trace.exposed_action_id != named_trace.exposed_action_id
            )
            outcome_changes += int(
                legacy_trace.observed_outcome_id
                != named_trace.observed_outcome_id
            )
    return action_changes, outcome_changes


def _condition_payload(
    readout: RelationshipConditionReadout | None,
) -> dict[str, object] | None:
    if readout is None:
        return None
    return {
        "condition_label": readout.condition_label,
        "confidence": readout.confidence,
        "normalized_margin": readout.normalized_margin,
        "candidate_scores": [
            {"label": label, "score": score}
            for label, score in readout.candidate_scores
        ],
        "reader_artifact_id": readout.reader_artifact_id,
        "source_observation_sha256": readout.source_observation_sha256,
    }


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
    "P4_NAMED_READER_REPORT_SCHEMA_VERSION",
    "P4NamedReaderArm",
    "P4NamedReaderArmRun",
    "P4NamedReaderArmSummary",
    "RelationshipP4NamedReaderTransmissionReport",
    "render_relationship_p4_named_reader_markdown",
    "run_relationship_p4_named_reader_transmission",
    "validate_relationship_p4_named_reader_report_files",
    "write_relationship_p4_named_reader_markdown",
    "write_relationship_p4_named_reader_report",
]
