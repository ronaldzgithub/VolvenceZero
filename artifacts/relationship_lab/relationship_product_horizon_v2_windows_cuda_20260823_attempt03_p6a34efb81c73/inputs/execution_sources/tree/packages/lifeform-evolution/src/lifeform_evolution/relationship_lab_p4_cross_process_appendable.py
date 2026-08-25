"""Real child-process owner-hydration preflight for the seen P4 fixture.

This package isolates one narrow Appendable question.  Every public pulse is
executed by a fresh ``sys.executable`` child.  The child receives only the
current public session plus a filesystem state root; historical relationship
state can reach the owner only through ``OwnerHydrationStore``.

The three matched arms stage the target's exact prior boundary, an empty prior
boundary, or the paired subject's exact same-stage prior boundary.  No model,
evaluator, environment, PE, credit, gate update, residual, or expression path
is part of this preflight.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Mapping

from lifeform_domain_emogpt.lab import (
    P4CanaryOnboardingSession,
    append_relationship_p4_onboarding_session,
    load_relationship_p4_longitudinal_canary_view,
    sha256_json,
)
from lifeform_domain_emogpt.relationship_forecast import (
    BoundedRelationshipPreferenceForecastRuntime,
)
from volvence_zero.memory.persistence import FileSystemPersistenceBackend
from volvence_zero.owner_hydration_store import OwnerHydrationStore
from volvence_zero.runtime import WiringLevel
from volvence_zero.social import (
    PreferenceAboutOtherModule,
    PreferenceActionForecastRequest,
    SocialRecordStore,
)
from volvence_zero.social_cognition import PreferenceAboutOtherSnapshot


P4_CROSS_PROCESS_PROTOCOL_SCHEMA_VERSION = (
    "relationship-p4-cross-process-owner-hydration-preflight.v1"
)
P4_CROSS_PROCESS_REQUEST_SCHEMA_VERSION = (
    "relationship-p4-cross-process-owner-hydration-request.v1"
)
P4_CROSS_PROCESS_RECEIPT_SCHEMA_VERSION = (
    "relationship-p4-cross-process-owner-hydration-receipt.v1"
)
P4_CROSS_PROCESS_REPORT_SCHEMA_VERSION = (
    "relationship-p4-cross-process-owner-hydration-report.v1"
)
P4_CROSS_PROCESS_ONBOARDING_SCHEMA_VERSION = (
    "relationship-p4-canary-onboarding-session.v1"
)
P4_CROSS_PROCESS_DECISION_SCHEMA_VERSION = (
    "relationship-p4-canary-decision-session.v1"
)

_OWNER_NAME = "social_record_store"
_OWNER_KEY = "owner_hydration/social_record_store"
_OWNER_SAFE_KEY = "owner_hydration__social_record_store"
_INTERLOCUTOR_ID = "primary"
_EXPECTED_SUBJECT_COUNT = 2
_EXPECTED_ONBOARDING_COUNT = 4
_EXPECTED_DECISION_COUNT = 8
_EXPECTED_PULSE_COUNT = _EXPECTED_ONBOARDING_COUNT + _EXPECTED_DECISION_COUNT
_EXPECTED_INVOCATION_COUNT = (
    _EXPECTED_SUBJECT_COUNT * _EXPECTED_PULSE_COUNT * 3
)
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DEFAULT_PROTOCOL_PATH = (
    pathlib.Path(__file__).resolve().parent
    / "protocols"
    / "relationship_p4_cross_process_appendable_preflight_v1.json"
)
_DEFAULT_WORKER_SCRIPT = (
    _REPO_ROOT / "scripts" / "run_relationship_lab_p4_cross_process_appendable.py"
)

_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "invocation_nonce",
        "arm",
        "subject_id",
        "subject_scope",
        "pulse_index",
        "output_boundary",
        "source_subject_id",
        "source_boundary",
        "source_state_sha256",
        "state_root",
        "session_path",
        "parent_pid",
        "max_versions",
    }
)
_ONBOARDING_KEYS = frozenset(
    {
        "schema_version",
        "pulse_kind",
        "session_id",
        "session_index",
        "event_id",
        "observation_summary",
        "action_id",
        "observed_outcome_id",
        "reaction_summary",
        "observation_ref",
    }
)
_DECISION_KEYS = frozenset(
    {
        "schema_version",
        "pulse_kind",
        "session_id",
        "decision_index",
        "action_turn_index",
        "decision_id",
        "phase_id",
        "probe_surface_family",
        "current_observation",
        "observation_ref",
        "candidate_action_ids",
        "typed_outcome_ids",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "arm",
        "subject_id",
        "subject_scope",
        "pulse_index",
        "output_boundary",
        "source_subject_id",
        "source_boundary",
        "invocation_nonce",
        "child_pid",
        "parent_pid",
        "python_executable",
        "python_version",
        "platform_system",
        "session_kind",
        "session_id",
        "request_sha256",
        "session_sha256",
        "source_state_sha256",
        "pre_state_sha256",
        "post_state_sha256",
        "backend",
        "hydration_store",
        "wiring_level",
        "owner_name",
        "owner_key",
        "max_versions",
        "load_attempted",
        "owner_loaded",
        "pre_backend_version",
        "post_backend_version",
        "pre_backend_bytes_sha256",
        "post_backend_bytes_sha256",
        "pre_raw_file_sha256",
        "post_raw_file_sha256",
        "pre_owner_payload_sha256",
        "post_owner_payload_sha256",
        "forecast_status",
        "forecast_count",
        "forecast_id",
        "recommended_action_id",
        "forecast_source_record_count",
        "owner_record_count",
        "owner_action_outcome_count",
        "owner_forecast_count",
        "gate_invoked",
        "environment_settled",
        "prediction_error_computed",
        "credit_computed",
        "learning_applied",
        "raw_history_replayed",
        "model_output_count",
        "qwen_output_count",
        "residual_intervention_count",
        "formal_evidence_authorized",
    }
)
_FORBIDDEN_REQUEST_KEYS = frozenset(
    {
        "history",
        "records",
        "action_outcomes",
        "owner_payload",
        "owner_snapshot",
        "persistence_snapshot",
        "forecast_scores",
        "preferred_action_id",
        "environment_seed",
        "evaluator",
    }
)


class P4CrossProcessArm(str, Enum):
    CORRECT_PRIOR_STATE = "correct_prior_state"
    EMPTY_PRIOR_STATE = "empty_prior_state"
    SWAPPED_SUBJECT_PRIOR_STATE = "swapped_subject_prior_state"


@dataclass(frozen=True)
class P4CrossProcessProtocol:
    protocol_id: str
    p4_protocol_sha256: str
    p4_public_plan_sha256: str
    subject_ids: tuple[str, ...]
    donor_pairs: tuple[tuple[str, str], ...]
    onboarding_pulses_per_subject: int
    decision_probes_per_subject: int
    max_versions: int
    claim_boundary: str

    def __post_init__(self) -> None:
        for value in (
            self.protocol_id,
            self.p4_protocol_sha256,
            self.p4_public_plan_sha256,
        ):
            _require_sha256(value, "P4 cross-process protocol lineage")
        if len(self.subject_ids) != _EXPECTED_SUBJECT_COUNT:
            raise ValueError("P4 cross-process protocol requires two subjects")
        if len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError("P4 cross-process protocol subject IDs repeat")
        if self.onboarding_pulses_per_subject != _EXPECTED_ONBOARDING_COUNT:
            raise ValueError("P4 cross-process onboarding shape drift")
        if self.decision_probes_per_subject != _EXPECTED_DECISION_COUNT:
            raise ValueError("P4 cross-process decision shape drift")
        donors = dict(self.donor_pairs)
        if set(donors) != set(self.subject_ids):
            raise ValueError("P4 cross-process donor map shape drift")
        if any(
            donor not in self.subject_ids or donor == target
            for target, donor in donors.items()
        ):
            raise ValueError("P4 cross-process donor pairing is invalid")
        if self.max_versions < _EXPECTED_PULSE_COUNT:
            raise ValueError("P4 cross-process backend must retain every boundary")
        if not self.claim_boundary.strip():
            raise ValueError("P4 cross-process claim boundary is empty")

    @property
    def donor_by_subject(self) -> dict[str, str]:
        return dict(self.donor_pairs)


@dataclass(frozen=True)
class P4CrossProcessPulseEvidence:
    arm: P4CrossProcessArm
    subject_id: str
    subject_scope: str
    pulse_index: int
    output_boundary: int
    session_kind: str
    session_id: str
    invocation_nonce: str
    host_child_pid: int
    receipt_child_pid: int
    receipt_parent_pid: int
    source_subject_id: str | None
    source_boundary: int
    source_state_sha256: str
    pre_state_sha256: str
    post_state_sha256: str
    owner_loaded: bool
    pre_backend_version: int | None
    post_backend_version: int
    pre_raw_file_sha256: str | None
    post_raw_file_sha256: str
    pre_backend_bytes_sha256: str | None
    post_backend_bytes_sha256: str
    pre_owner_payload_sha256: str
    post_owner_payload_sha256: str
    forecast_status: str
    forecast_count: int
    recommended_action_id: str | None
    forecast_source_record_count: int
    owner_record_count: int
    owner_action_outcome_count: int
    owner_forecast_count: int
    state_root: str
    session_path: str
    request_path: str
    receipt_path: str
    session_sha256: str
    request_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.subject_scope, "subject_scope")
        for value in (
            self.source_state_sha256,
            self.pre_state_sha256,
            self.post_state_sha256,
            self.post_raw_file_sha256,
            self.post_backend_bytes_sha256,
            self.pre_owner_payload_sha256,
            self.post_owner_payload_sha256,
            self.session_sha256,
            self.request_sha256,
            self.receipt_sha256,
        ):
            _require_sha256(value, "P4 cross-process pulse digest")
        for optional_value in (
            self.pre_raw_file_sha256,
            self.pre_backend_bytes_sha256,
        ):
            if optional_value is not None:
                _require_sha256(optional_value, "P4 cross-process prior digest")
        if not 0 <= self.pulse_index < _EXPECTED_PULSE_COUNT:
            raise ValueError("P4 cross-process pulse index is invalid")
        if self.output_boundary != self.pulse_index + 1:
            raise ValueError("P4 cross-process output boundary drift")
        if self.host_child_pid != self.receipt_child_pid:
            raise ValueError("P4 cross-process host/receipt PID mismatch")
        if self.receipt_child_pid == self.receipt_parent_pid:
            raise ValueError("P4 cross-process worker did not leave the parent process")
        if self.pre_state_sha256 != self.source_state_sha256:
            raise ValueError("P4 cross-process staged state digest drift")
        if self.post_backend_version <= 0:
            raise ValueError("P4 cross-process post version must be positive")
        if self.owner_loaded != (self.pre_backend_version is not None):
            raise ValueError("P4 cross-process hydration/version mismatch")
        if self.owner_loaded != (self.pre_raw_file_sha256 is not None):
            raise ValueError("P4 cross-process hydration/raw-file mismatch")
        if self.owner_loaded != (self.pre_backend_bytes_sha256 is not None):
            raise ValueError("P4 cross-process hydration/backend-bytes mismatch")
        if self.session_kind == "onboarding":
            if self.forecast_status != "not_requested" or self.forecast_count != 0:
                raise ValueError("onboarding pulse unexpectedly ran a forecast")
        elif self.session_kind == "decision_probe":
            if self.forecast_count not in (0, 1):
                raise ValueError("decision probe forecast cardinality is invalid")
            expected_status = (
                "published" if self.forecast_count == 1 else "absent_no_owner_evidence"
            )
            if self.forecast_status != expected_status:
                raise ValueError("decision probe forecast status drift")
            if (self.recommended_action_id is None) != (self.forecast_count == 0):
                raise ValueError("decision probe action/forecast mismatch")
        else:
            raise ValueError("P4 cross-process session kind is invalid")
        for path_value in (
            self.state_root,
            self.session_path,
            self.request_path,
            self.receipt_path,
        ):
            if not path_value or "\\" in path_value or pathlib.PurePosixPath(path_value).is_absolute():
                raise ValueError("P4 cross-process report paths must be relative POSIX paths")

    def to_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "subject_id": self.subject_id,
            "subject_scope": self.subject_scope,
            "pulse_index": self.pulse_index,
            "output_boundary": self.output_boundary,
            "session_kind": self.session_kind,
            "session_id": self.session_id,
            "invocation_nonce": self.invocation_nonce,
            "host_child_pid": self.host_child_pid,
            "receipt_child_pid": self.receipt_child_pid,
            "receipt_parent_pid": self.receipt_parent_pid,
            "source_subject_id": self.source_subject_id,
            "source_boundary": self.source_boundary,
            "source_state_sha256": self.source_state_sha256,
            "pre_state_sha256": self.pre_state_sha256,
            "post_state_sha256": self.post_state_sha256,
            "owner_loaded": self.owner_loaded,
            "pre_backend_version": self.pre_backend_version,
            "post_backend_version": self.post_backend_version,
            "pre_raw_file_sha256": self.pre_raw_file_sha256,
            "post_raw_file_sha256": self.post_raw_file_sha256,
            "pre_backend_bytes_sha256": self.pre_backend_bytes_sha256,
            "post_backend_bytes_sha256": self.post_backend_bytes_sha256,
            "pre_owner_payload_sha256": self.pre_owner_payload_sha256,
            "post_owner_payload_sha256": self.post_owner_payload_sha256,
            "forecast_status": self.forecast_status,
            "forecast_count": self.forecast_count,
            "recommended_action_id": self.recommended_action_id,
            "forecast_source_record_count": self.forecast_source_record_count,
            "owner_record_count": self.owner_record_count,
            "owner_action_outcome_count": self.owner_action_outcome_count,
            "owner_forecast_count": self.owner_forecast_count,
            "state_root": self.state_root,
            "session_path": self.session_path,
            "request_path": self.request_path,
            "receipt_path": self.receipt_path,
            "session_sha256": self.session_sha256,
            "request_sha256": self.request_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True)
class RelationshipP4CrossProcessAppendableReport:
    protocol_id: str
    p4_protocol_sha256: str
    p4_public_plan_sha256: str
    python_executable: str
    python_version: str
    platform_system: str
    platform_release: str
    parent_pid: int
    pulses: tuple[P4CrossProcessPulseEvidence, ...]
    correct_empty_forecast_presence_change_count: int
    correct_swapped_recommended_action_change_count: int
    mechanical_cross_process_chain_observed: bool
    seen_fixture_only: bool
    independent_subject_count: int
    raw_history_replayed: bool
    evaluator_loaded: bool
    learning_applied: bool
    model_output_count: int
    qwen_output_count: int
    residual_intervention_count: int
    formal_evidence_authorized: bool
    verdict: str
    claim_boundary: str
    schema_version: str = P4_CROSS_PROCESS_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != P4_CROSS_PROCESS_REPORT_SCHEMA_VERSION:
            raise ValueError("P4 cross-process report schema drift")
        for value in (
            self.protocol_id,
            self.p4_protocol_sha256,
            self.p4_public_plan_sha256,
        ):
            _require_sha256(value, "P4 cross-process report lineage")
        if len(self.pulses) != _EXPECTED_INVOCATION_COUNT:
            raise ValueError("P4 cross-process invocation count drift")
        if len({item.invocation_nonce for item in self.pulses}) != len(self.pulses):
            raise ValueError("P4 cross-process invocation nonce repeated")
        if any(item.receipt_parent_pid != self.parent_pid for item in self.pulses):
            raise ValueError("P4 cross-process child parent PID drift")
        _validate_state_chains(self.pulses)
        expected_presence, expected_swapped = _matched_forecast_changes(self.pulses)
        if self.correct_empty_forecast_presence_change_count != expected_presence:
            raise ValueError("P4 cross-process correct/empty metric drift")
        if self.correct_swapped_recommended_action_change_count != expected_swapped:
            raise ValueError("P4 cross-process correct/swapped metric drift")
        if not self.mechanical_cross_process_chain_observed:
            raise ValueError("P4 cross-process mechanical chain was not observed")
        if (
            not self.seen_fixture_only
            or self.independent_subject_count != 0
            or self.raw_history_replayed
            or self.evaluator_loaded
            or self.learning_applied
            or self.model_output_count != 0
            or self.qwen_output_count != 0
            or self.residual_intervention_count != 0
            or self.formal_evidence_authorized
        ):
            raise ValueError("P4 cross-process evidence firewall is open")
        effect_observed = expected_presence > 0 or expected_swapped > 0
        expected_verdict = (
            "cross_process_owner_hydration_forecast_effect_observed_development_only"
            if effect_observed
            else "cross_process_owner_hydration_observed_without_forecast_effect"
        )
        if self.verdict != expected_verdict:
            raise ValueError("P4 cross-process report verdict drift")
        if not self.claim_boundary.strip():
            raise ValueError("P4 cross-process report claim boundary is empty")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "protocol_id": self.protocol_id,
                "p4_protocol_sha256": self.p4_protocol_sha256,
                "p4_public_plan_sha256": self.p4_public_plan_sha256,
            },
            "runtime": {
                "python_executable": self.python_executable,
                "python_version": self.python_version,
                "platform_system": self.platform_system,
                "platform_release": self.platform_release,
                "parent_pid": self.parent_pid,
            },
            "pulses": [item.to_payload() for item in self.pulses],
            "metrics": {
                "invocation_count": len(self.pulses),
                "correct_empty_forecast_presence_change_count": (
                    self.correct_empty_forecast_presence_change_count
                ),
                "correct_swapped_recommended_action_change_count": (
                    self.correct_swapped_recommended_action_change_count
                ),
                "mechanical_cross_process_chain_observed": (
                    self.mechanical_cross_process_chain_observed
                ),
            },
            "firewall": {
                "seen_fixture_only": self.seen_fixture_only,
                "independent_subject_count": self.independent_subject_count,
                "raw_history_replayed": self.raw_history_replayed,
                "evaluator_loaded": self.evaluator_loaded,
                "learning_applied": self.learning_applied,
                "model_output_count": self.model_output_count,
                "qwen_output_count": self.qwen_output_count,
                "residual_intervention_count": self.residual_intervention_count,
                "formal_evidence_authorized": self.formal_evidence_authorized,
            },
            "verdict": self.verdict,
            "claim_boundary": self.claim_boundary,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())


def relationship_p4_cross_process_protocol_path() -> pathlib.Path:
    return _DEFAULT_PROTOCOL_PATH


def load_relationship_p4_cross_process_protocol(
    path: pathlib.Path | None = None,
) -> P4CrossProcessProtocol:
    protocol_path = pathlib.Path(path or _DEFAULT_PROTOCOL_PATH)
    raw = _load_json_object(protocol_path)
    _require_exact_keys(
        raw,
        {"schema_version", "source", "experiment", "owner_contract", "firewall", "claim_boundary"},
        "P4 cross-process protocol",
    )
    if raw["schema_version"] != P4_CROSS_PROCESS_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("P4 cross-process protocol schema mismatch")
    source = _require_mapping(raw["source"], "source")
    experiment = _require_mapping(raw["experiment"], "experiment")
    owner = _require_mapping(raw["owner_contract"], "owner_contract")
    firewall = _require_mapping(raw["firewall"], "firewall")
    expected_arms = tuple(item.value for item in P4CrossProcessArm)
    if tuple(str(item) for item in experiment["arms"]) != expected_arms:
        raise ValueError("P4 cross-process arm order drift")
    if not experiment["fresh_os_child_per_pulse"]:
        raise ValueError("P4 cross-process protocol disabled fresh children")
    if source["independent_subject_count"] != 0:
        raise ValueError("P4 cross-process protocol invented independent subjects")
    expected_firewall = {
        "parent_passes_owner_payload": False,
        "parent_passes_history": False,
        "raw_history_replayed": False,
        "evaluator_loaded": False,
        "environment_settled": False,
        "prediction_error_computed": False,
        "credit_computed": False,
        "learning_applied": False,
        "qwen_output_count": 0,
        "model_output_count": 0,
        "residual_intervention_count": 0,
        "expression_authorized": False,
        "production_active_authorized": False,
        "formal_evidence_authorized": False,
    }
    if dict(firewall) != expected_firewall:
        raise ValueError("P4 cross-process protocol firewall drift")
    if (
        owner["owner_name"] != _OWNER_NAME
        or owner["storage_key"] != _OWNER_KEY
        or owner["backend"] != "FileSystemPersistenceBackend"
        or owner["hydration_store"] != "OwnerHydrationStore"
        or owner["wiring_level"] != WiringLevel.ACTIVE.value
        or owner["reader_runtime"]
        != BoundedRelationshipPreferenceForecastRuntime.runtime_id
        or owner["decision_probe_gate_invoked"]
    ):
        raise ValueError("P4 cross-process owner contract drift")
    subject_ids = tuple(str(item) for item in source["subject_ids"])
    donor_raw = _require_mapping(experiment["paired_donor"], "paired_donor")
    return P4CrossProcessProtocol(
        protocol_id=sha256_json(raw),
        p4_protocol_sha256=str(source["p4_protocol_sha256"]),
        p4_public_plan_sha256=str(source["p4_public_plan_sha256"]),
        subject_ids=subject_ids,
        donor_pairs=tuple((subject, str(donor_raw[subject])) for subject in subject_ids),
        onboarding_pulses_per_subject=int(experiment["onboarding_pulses_per_subject"]),
        decision_probes_per_subject=int(experiment["decision_probes_per_subject"]),
        max_versions=int(owner["max_versions"]),
        claim_boundary=str(raw["claim_boundary"]),
    )


def _canonical_public_session_payloads(subject: Any) -> tuple[dict[str, object], ...]:
    return tuple(
        {**item.to_sut_payload(), "pulse_kind": "onboarding"}
        for item in subject.onboarding_sessions
    ) + tuple(
        {**item.to_sut_payload(), "pulse_kind": "decision_probe"}
        for item in subject.decision_sessions
    )


def run_relationship_p4_cross_process_appendable_preflight(
    *,
    output_dir: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
    worker_script: pathlib.Path | None = None,
    python_executable: str | None = None,
) -> RelationshipP4CrossProcessAppendableReport:
    """Run 72 one-shot child processes over three matched state interventions."""

    protocol = load_relationship_p4_cross_process_protocol(protocol_path)
    view = load_relationship_p4_longitudinal_canary_view()
    if (
        view.contract.protocol_sha256 != protocol.p4_protocol_sha256
        or view.public_plan_sha256 != protocol.p4_public_plan_sha256
    ):
        raise ValueError("P4 cross-process source protocol lineage drift")
    subjects_by_id = {item.subject_id: item for item in view.subjects}
    if tuple(subjects_by_id) != protocol.subject_ids:
        raise ValueError("P4 cross-process source subject order drift")

    root = pathlib.Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"P4 cross-process output already exists: {root}")
    root.mkdir(parents=True)
    sessions_root = root / "sessions"
    requests_root = root / "requests"
    receipts_root = root / "receipts"
    state_root = root / "state"
    for directory in (sessions_root, requests_root, receipts_root, state_root):
        directory.mkdir()

    session_files: dict[tuple[str, int], pathlib.Path] = {}
    for subject in view.subjects:
        subject_dir = sessions_root / subject.subject_id
        subject_dir.mkdir()
        public_sessions = _canonical_public_session_payloads(subject)
        if len(public_sessions) != _EXPECTED_PULSE_COUNT:
            raise ValueError("P4 cross-process public session shape drift")
        for pulse_index, payload in enumerate(public_sessions):
            session_path = subject_dir / f"pulse-{pulse_index:03d}.json"
            _write_create_json(session_path, payload)
            session_files[(subject.subject_id, pulse_index)] = session_path

    executable = str(python_executable or sys.executable)
    script = pathlib.Path(worker_script or _DEFAULT_WORKER_SCRIPT).resolve()
    if not pathlib.Path(executable).is_file():
        raise FileNotFoundError(f"P4 cross-process Python executable missing: {executable}")
    if not script.is_file():
        raise FileNotFoundError(f"P4 cross-process worker script missing: {script}")

    pulses: list[P4CrossProcessPulseEvidence] = []
    correct_boundaries: dict[tuple[str, int], pathlib.Path] = {}
    for subject in view.subjects:
        boundary_zero = (
            state_root
            / P4CrossProcessArm.CORRECT_PRIOR_STATE.value
            / subject.subject_id
            / "boundary-000"
        )
        boundary_zero.mkdir(parents=True)
        correct_boundaries[(subject.subject_id, 0)] = boundary_zero
        for pulse_index in range(_EXPECTED_PULSE_COUNT):
            source = correct_boundaries[(subject.subject_id, pulse_index)]
            target = source.parent / f"boundary-{pulse_index + 1:03d}"
            shutil.copytree(source, target)
            pulses.append(
                _launch_worker(
                    root=root,
                    protocol=protocol,
                    arm=P4CrossProcessArm.CORRECT_PRIOR_STATE,
                    subject_id=subject.subject_id,
                    subject_scope=subject.subject_scope,
                    pulse_index=pulse_index,
                    source_subject_id=subject.subject_id,
                    source_boundary=pulse_index,
                    state_root=target,
                    session_path=session_files[(subject.subject_id, pulse_index)],
                    requests_root=requests_root,
                    receipts_root=receipts_root,
                    executable=executable,
                    script=script,
                )
            )
            correct_boundaries[(subject.subject_id, pulse_index + 1)] = target

    for subject in view.subjects:
        arm_root = state_root / P4CrossProcessArm.EMPTY_PRIOR_STATE.value / subject.subject_id
        for pulse_index in range(_EXPECTED_PULSE_COUNT):
            target = arm_root / f"boundary-{pulse_index + 1:03d}"
            target.mkdir(parents=True)
            pulses.append(
                _launch_worker(
                    root=root,
                    protocol=protocol,
                    arm=P4CrossProcessArm.EMPTY_PRIOR_STATE,
                    subject_id=subject.subject_id,
                    subject_scope=subject.subject_scope,
                    pulse_index=pulse_index,
                    source_subject_id=None,
                    source_boundary=0,
                    state_root=target,
                    session_path=session_files[(subject.subject_id, pulse_index)],
                    requests_root=requests_root,
                    receipts_root=receipts_root,
                    executable=executable,
                    script=script,
                )
            )

    donor_by_subject = protocol.donor_by_subject
    for subject in view.subjects:
        donor_id = donor_by_subject[subject.subject_id]
        arm_root = (
            state_root
            / P4CrossProcessArm.SWAPPED_SUBJECT_PRIOR_STATE.value
            / subject.subject_id
        )
        for pulse_index in range(_EXPECTED_PULSE_COUNT):
            source = correct_boundaries[(donor_id, pulse_index)]
            target = arm_root / f"boundary-{pulse_index + 1:03d}"
            shutil.copytree(source, target)
            pulses.append(
                _launch_worker(
                    root=root,
                    protocol=protocol,
                    arm=P4CrossProcessArm.SWAPPED_SUBJECT_PRIOR_STATE,
                    subject_id=subject.subject_id,
                    subject_scope=subject.subject_scope,
                    pulse_index=pulse_index,
                    source_subject_id=donor_id,
                    source_boundary=pulse_index,
                    state_root=target,
                    session_path=session_files[(subject.subject_id, pulse_index)],
                    requests_root=requests_root,
                    receipts_root=receipts_root,
                    executable=executable,
                    script=script,
                )
            )

    frozen_pulses = tuple(pulses)
    child_runtimes = {
        (
            _require_text_value(receipt["python_executable"], "child executable"),
            _require_text_value(receipt["python_version"], "child Python version"),
            _require_text_value(receipt["platform_system"], "child platform system"),
        )
        for pulse in frozen_pulses
        for receipt in (
            _load_json_object(
                root.joinpath(*pathlib.PurePosixPath(pulse.receipt_path).parts)
            ),
        )
    }
    if len(child_runtimes) != 1:
        raise ValueError("P4 cross-process child runtime attestation drift")
    child_executable, child_python_version, child_platform_system = next(
        iter(child_runtimes)
    )
    presence_changes, swapped_changes = _matched_forecast_changes(frozen_pulses)
    verdict = (
        "cross_process_owner_hydration_forecast_effect_observed_development_only"
        if presence_changes > 0 or swapped_changes > 0
        else "cross_process_owner_hydration_observed_without_forecast_effect"
    )
    report = RelationshipP4CrossProcessAppendableReport(
        protocol_id=protocol.protocol_id,
        p4_protocol_sha256=protocol.p4_protocol_sha256,
        p4_public_plan_sha256=protocol.p4_public_plan_sha256,
        python_executable=child_executable,
        python_version=child_python_version,
        platform_system=child_platform_system,
        platform_release=platform.release(),
        parent_pid=os.getpid(),
        pulses=frozen_pulses,
        correct_empty_forecast_presence_change_count=presence_changes,
        correct_swapped_recommended_action_change_count=swapped_changes,
        mechanical_cross_process_chain_observed=True,
        seen_fixture_only=True,
        independent_subject_count=0,
        raw_history_replayed=False,
        evaluator_loaded=False,
        learning_applied=False,
        model_output_count=0,
        qwen_output_count=0,
        residual_intervention_count=0,
        formal_evidence_authorized=False,
        verdict=verdict,
        claim_boundary=protocol.claim_boundary,
    )
    write_relationship_p4_cross_process_report(report, output_dir=root)
    return report


def _launch_worker(
    *,
    root: pathlib.Path,
    protocol: P4CrossProcessProtocol,
    arm: P4CrossProcessArm,
    subject_id: str,
    subject_scope: str,
    pulse_index: int,
    source_subject_id: str | None,
    source_boundary: int,
    state_root: pathlib.Path,
    session_path: pathlib.Path,
    requests_root: pathlib.Path,
    receipts_root: pathlib.Path,
    executable: str,
    script: pathlib.Path,
) -> P4CrossProcessPulseEvidence:
    invocation_nonce = uuid.uuid4().hex
    request_dir = requests_root / arm.value / subject_id
    receipt_dir = receipts_root / arm.value / subject_id
    request_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / f"pulse-{pulse_index:03d}.json"
    receipt_path = receipt_dir / f"pulse-{pulse_index:03d}.json"
    source_state_sha256 = _state_directory_sha256(state_root)
    request_payload = {
        "schema_version": P4_CROSS_PROCESS_REQUEST_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "invocation_nonce": invocation_nonce,
        "arm": arm.value,
        "subject_id": subject_id,
        "subject_scope": subject_scope,
        "pulse_index": pulse_index,
        "output_boundary": pulse_index + 1,
        "source_subject_id": source_subject_id,
        "source_boundary": source_boundary,
        "source_state_sha256": source_state_sha256,
        "state_root": _relative_posix(root, state_root),
        "session_path": _relative_posix(root, session_path),
        "parent_pid": os.getpid(),
        "max_versions": protocol.max_versions,
    }
    request_sha256 = _write_create_json(request_path, request_payload)
    child_environment = os.environ.copy()
    child_environment["PYTHONNOUSERSITE"] = "1"
    child_environment["PYTHONHASHSEED"] = "0"
    process = subprocess.Popen(
        [
            executable,
            str(script),
            "worker-pulse",
            "--request",
            str(request_path),
            "--receipt",
            str(receipt_path),
            "--run-root",
            str(root),
        ],
        cwd=str(_REPO_ROOT),
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise TimeoutError(
            f"P4 cross-process worker timed out: arm={arm.value}, "
            f"subject={subject_id}, pulse={pulse_index}"
        ) from exc
    if process.returncode != 0:
        raise RuntimeError(
            "P4 cross-process worker failed: "
            f"arm={arm.value}, subject={subject_id}, pulse={pulse_index}, "
            f"exit={process.returncode}, stdout={stdout!r}, stderr={stderr!r}"
        )
    if not receipt_path.is_file():
        raise FileNotFoundError("P4 cross-process worker omitted its receipt")
    receipt = _load_json_object(receipt_path)
    session = _load_json_object(session_path)
    _validate_session_payload(session, pulse_index=pulse_index)
    _validate_worker_receipt(
        receipt,
        request=request_payload,
        request_sha256=request_sha256,
        host_child_pid=process.pid,
        session=session,
    )
    return P4CrossProcessPulseEvidence(
        arm=arm,
        subject_id=subject_id,
        subject_scope=subject_scope,
        pulse_index=pulse_index,
        output_boundary=pulse_index + 1,
        session_kind=str(receipt["session_kind"]),
        session_id=str(receipt["session_id"]),
        invocation_nonce=invocation_nonce,
        host_child_pid=process.pid,
        receipt_child_pid=int(receipt["child_pid"]),
        receipt_parent_pid=int(receipt["parent_pid"]),
        source_subject_id=source_subject_id,
        source_boundary=source_boundary,
        source_state_sha256=source_state_sha256,
        pre_state_sha256=str(receipt["pre_state_sha256"]),
        post_state_sha256=str(receipt["post_state_sha256"]),
        owner_loaded=bool(receipt["owner_loaded"]),
        pre_backend_version=_optional_int(receipt["pre_backend_version"]),
        post_backend_version=int(receipt["post_backend_version"]),
        pre_raw_file_sha256=_optional_text(receipt["pre_raw_file_sha256"]),
        post_raw_file_sha256=str(receipt["post_raw_file_sha256"]),
        pre_backend_bytes_sha256=_optional_text(
            receipt["pre_backend_bytes_sha256"]
        ),
        post_backend_bytes_sha256=str(receipt["post_backend_bytes_sha256"]),
        pre_owner_payload_sha256=str(receipt["pre_owner_payload_sha256"]),
        post_owner_payload_sha256=str(receipt["post_owner_payload_sha256"]),
        forecast_status=str(receipt["forecast_status"]),
        forecast_count=int(receipt["forecast_count"]),
        recommended_action_id=_optional_text(receipt["recommended_action_id"]),
        forecast_source_record_count=int(receipt["forecast_source_record_count"]),
        owner_record_count=int(receipt["owner_record_count"]),
        owner_action_outcome_count=int(receipt["owner_action_outcome_count"]),
        owner_forecast_count=int(receipt["owner_forecast_count"]),
        state_root=_relative_posix(root, state_root),
        session_path=_relative_posix(root, session_path),
        request_path=_relative_posix(root, request_path),
        receipt_path=_relative_posix(root, receipt_path),
        session_sha256=str(receipt["session_sha256"]),
        request_sha256=request_sha256,
        receipt_sha256=_sha256_file(receipt_path),
    )


def run_relationship_p4_cross_process_worker(
    *,
    request_path: pathlib.Path,
    receipt_path: pathlib.Path,
    run_root: pathlib.Path,
) -> None:
    request_file = pathlib.Path(request_path).resolve()
    receipt_file = pathlib.Path(receipt_path).resolve()
    request = _load_json_object(request_file)
    _validate_worker_request(request, expected_parent_pid=os.getppid())
    invocation_nonce = request["invocation_nonce"]
    pulse_index = request["pulse_index"]
    max_versions = request["max_versions"]

    resolved_run_root = pathlib.Path(run_root).resolve()
    state_root = _resolve_run_relative_path(
        resolved_run_root,
        request["state_root"],
        "state_root",
    )
    session_path = _resolve_run_relative_path(
        resolved_run_root,
        request["session_path"],
        "session_path",
    )
    if not state_root.is_dir():
        raise FileNotFoundError(f"P4 cross-process state root missing: {state_root}")
    session = _load_json_object(session_path)
    _validate_session_payload(session, pulse_index=pulse_index)
    session_kind = session["pulse_kind"]

    request_sha256 = _sha256_file(request_file)
    session_sha256 = _sha256_file(session_path)
    pre_state_sha256 = _state_directory_sha256(state_root)
    if pre_state_sha256 != request["source_state_sha256"]:
        raise ValueError("P4 cross-process staged state changed before worker load")

    backend = FileSystemPersistenceBackend(
        base_dir=str(state_root),
        max_versions=max_versions,
    )
    hydration = OwnerHydrationStore(
        backend=backend,
        wiring_level=WiringLevel.ACTIVE,
    )
    store = SocialRecordStore()
    pre_backend = backend.load_checkpoint(key=_OWNER_KEY)
    owner_loaded = hydration.hydrate_owner_if_present(store, _OWNER_NAME)
    if owner_loaded != (pre_backend is not None):
        raise RuntimeError("P4 cross-process owner/backend hydration disagreement")
    pre_version = None if pre_backend is None else pre_backend[1]
    pre_backend_sha256 = (
        None if pre_backend is None else _sha256_bytes(pre_backend[0])
    )
    pre_raw_sha256 = (
        None
        if pre_version is None
        else _sha256_file(_checkpoint_path(state_root, pre_version))
    )
    pre_owner_snapshot = store.export_persistence_snapshot()
    pre_owner_sha256 = sha256_json(pre_owner_snapshot.payload)

    if session_kind == "onboarding":
        onboarding = P4CanaryOnboardingSession(
            subject_id=str(request["subject_id"]),
            session_id=str(session["session_id"]),
            session_index=int(session["session_index"]),
            event_id=str(session["event_id"]),
            observation_summary=str(session["observation_summary"]),
            action_id=str(session["action_id"]),
            observed_outcome_id=str(session["observed_outcome_id"]),
            reaction_summary=str(session["reaction_summary"]),
            observation_ref=str(session["observation_ref"]),
        )
        asyncio.run(
            append_relationship_p4_onboarding_session(
                store=store,
                session=onboarding,
            )
        )
        forecast_status = "not_requested"
        current_forecasts: tuple[Any, ...] = ()
    else:
        forecast_request = PreferenceActionForecastRequest(
            decision_id=str(session["decision_id"]),
            interlocutor_id=_INTERLOCUTOR_ID,
            current_observation=str(session["current_observation"]),
            observation_ref=str(session["observation_ref"]),
            candidate_action_ids=tuple(str(item) for item in session["candidate_action_ids"]),
            outcome_ids=tuple(str(item) for item in session["typed_outcome_ids"]),
            turn_index=int(session["action_turn_index"]),
            session_scope=str(request["subject_scope"]),
        )
        owner = PreferenceAboutOtherModule(
            turn_index=forecast_request.turn_index,
            wiring_level=WiringLevel.SHADOW,
            record_store=store,
            action_forecast_runtime=BoundedRelationshipPreferenceForecastRuntime(),
            action_forecast_request=forecast_request,
        )
        published = asyncio.run(owner.process({})).value
        if not isinstance(published, PreferenceAboutOtherSnapshot):
            raise TypeError("P4 cross-process forecast owner published unexpected snapshot")
        current_forecasts = tuple(
            item
            for item in published.action_forecasts
            if item.decision_id == forecast_request.decision_id
        )
        if len(current_forecasts) > 1:
            raise RuntimeError("P4 cross-process owner published duplicate forecasts")
        forecast_status = (
            "published" if current_forecasts else "absent_no_owner_evidence"
        )

    post_owner_snapshot = hydration.export_and_save_owner(store, _OWNER_NAME)
    post_backend = backend.load_checkpoint(key=_OWNER_KEY)
    if post_backend is None:
        raise RuntimeError("P4 cross-process backend omitted saved owner snapshot")
    post_backend_bytes, post_version = post_backend
    if post_version != (1 if pre_version is None else pre_version + 1):
        raise RuntimeError("P4 cross-process backend version did not advance")
    post_raw_path = _checkpoint_path(state_root, post_version)
    post_state_sha256 = _state_directory_sha256(state_root)
    payload = post_owner_snapshot.payload
    tom_records = _require_mapping(payload["tom_records"], "tom_records")
    receipt = {
        "schema_version": P4_CROSS_PROCESS_RECEIPT_SCHEMA_VERSION,
        "protocol_id": request["protocol_id"],
        "arm": request["arm"],
        "subject_id": request["subject_id"],
        "subject_scope": request["subject_scope"],
        "pulse_index": pulse_index,
        "output_boundary": request["output_boundary"],
        "source_subject_id": request["source_subject_id"],
        "source_boundary": request["source_boundary"],
        "invocation_nonce": invocation_nonce,
        "child_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "session_kind": session_kind,
        "session_id": session["session_id"],
        "request_sha256": request_sha256,
        "session_sha256": session_sha256,
        "source_state_sha256": request["source_state_sha256"],
        "pre_state_sha256": pre_state_sha256,
        "post_state_sha256": post_state_sha256,
        "backend": "FileSystemPersistenceBackend",
        "hydration_store": "OwnerHydrationStore",
        "wiring_level": WiringLevel.ACTIVE.value,
        "owner_name": _OWNER_NAME,
        "owner_key": _OWNER_KEY,
        "max_versions": max_versions,
        "load_attempted": True,
        "owner_loaded": owner_loaded,
        "pre_backend_version": pre_version,
        "post_backend_version": post_version,
        "pre_backend_bytes_sha256": pre_backend_sha256,
        "post_backend_bytes_sha256": _sha256_bytes(post_backend_bytes),
        "pre_raw_file_sha256": pre_raw_sha256,
        "post_raw_file_sha256": _sha256_file(post_raw_path),
        "pre_owner_payload_sha256": pre_owner_sha256,
        "post_owner_payload_sha256": sha256_json(post_owner_snapshot.payload),
        "forecast_status": forecast_status,
        "forecast_count": len(current_forecasts),
        "forecast_id": (
            None if not current_forecasts else current_forecasts[0].forecast_id
        ),
        "recommended_action_id": (
            None
            if not current_forecasts
            else current_forecasts[0].recommended_action_id
        ),
        "forecast_source_record_count": (
            0
            if not current_forecasts
            else len(current_forecasts[0].source_record_ids)
        ),
        "owner_record_count": sum(len(records) for records in tom_records.values()),
        "owner_action_outcome_count": len(payload["preference_action_outcomes"]),
        "owner_forecast_count": len(payload["preference_action_forecasts"]),
        "gate_invoked": False,
        "environment_settled": False,
        "prediction_error_computed": False,
        "credit_computed": False,
        "learning_applied": False,
        "raw_history_replayed": False,
        "model_output_count": 0,
        "qwen_output_count": 0,
        "residual_intervention_count": 0,
        "formal_evidence_authorized": False,
    }
    _write_create_json(receipt_file, receipt)


def write_relationship_p4_cross_process_report(
    report: RelationshipP4CrossProcessAppendableReport,
    *,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    output = pathlib.Path(output_dir)
    report_path = output / "cross_process_owner_hydration_report.json"
    markdown_path = output / "cross_process_owner_hydration_report.md"
    _write_create_json(
        report_path,
        {**report.to_payload(), "artifact_id": report.artifact_id},
    )
    _write_create_bytes(
        markdown_path,
        render_relationship_p4_cross_process_markdown(report).encode("utf-8"),
    )
    return report_path, markdown_path


def render_relationship_p4_cross_process_markdown(
    report: RelationshipP4CrossProcessAppendableReport,
) -> str:
    return "\n".join(
        (
            "# P4 cross-process owner hydration preflight",
            "",
            f"- verdict: `{report.verdict}`",
            f"- artifact: `{report.artifact_id}`",
            f"- real child invocations: `{len(report.pulses)}`",
            "- independent subjects: `0`",
            "- formal evidence authorized: `false`",
            "- model / Qwen / residual outputs: `0 / 0 / 0`",
            (
                "- correct vs empty forecast-presence changes: "
                f"`{report.correct_empty_forecast_presence_change_count}/16`"
            ),
            (
                "- correct vs swapped recommended-action changes: "
                f"`{report.correct_swapped_recommended_action_change_count}/16`"
            ),
            "",
            report.claim_boundary,
            "",
        )
    )


def validate_relationship_p4_cross_process_report_files(
    *,
    output_dir: pathlib.Path,
) -> None:
    output = pathlib.Path(output_dir).resolve()
    report_path = output / "cross_process_owner_hydration_report.json"
    markdown_path = output / "cross_process_owner_hydration_report.md"
    payload = _load_json_object(report_path)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "source",
            "runtime",
            "pulses",
            "metrics",
            "firewall",
            "verdict",
            "claim_boundary",
            "artifact_id",
        },
        "P4 cross-process report",
    )
    artifact_id = str(payload.pop("artifact_id"))
    if sha256_json(payload) != artifact_id:
        raise ValueError("P4 cross-process report artifact drift")
    if payload["schema_version"] != P4_CROSS_PROCESS_REPORT_SCHEMA_VERSION:
        raise ValueError("P4 cross-process report schema mismatch")
    pulse_payloads = payload["pulses"]
    if not isinstance(pulse_payloads, list) or len(pulse_payloads) != _EXPECTED_INVOCATION_COUNT:
        raise ValueError("P4 cross-process report pulse shape drift")
    frozen_pulses = tuple(
        _pulse_from_payload(_require_mapping(item, "report pulse"))
        for item in pulse_payloads
    )
    source = _require_mapping(payload["source"], "report source")
    runtime = _require_mapping(payload["runtime"], "report runtime")
    metrics = _require_mapping(payload["metrics"], "report metrics")
    firewall = _require_mapping(payload["firewall"], "report firewall")
    _require_exact_keys(
        source,
        {"protocol_id", "p4_protocol_sha256", "p4_public_plan_sha256"},
        "P4 cross-process report source",
    )
    _require_exact_keys(
        runtime,
        {
            "python_executable",
            "python_version",
            "platform_system",
            "platform_release",
            "parent_pid",
        },
        "P4 cross-process report runtime",
    )
    _require_exact_keys(
        metrics,
        {
            "invocation_count",
            "correct_empty_forecast_presence_change_count",
            "correct_swapped_recommended_action_change_count",
            "mechanical_cross_process_chain_observed",
        },
        "P4 cross-process report metrics",
    )
    _require_exact_keys(
        firewall,
        {
            "seen_fixture_only",
            "independent_subject_count",
            "raw_history_replayed",
            "evaluator_loaded",
            "learning_applied",
            "model_output_count",
            "qwen_output_count",
            "residual_intervention_count",
            "formal_evidence_authorized",
        },
        "P4 cross-process report firewall",
    )
    reconstructed = RelationshipP4CrossProcessAppendableReport(
        protocol_id=_require_text_value(source["protocol_id"], "protocol_id"),
        p4_protocol_sha256=_require_text_value(
            source["p4_protocol_sha256"], "p4_protocol_sha256"
        ),
        p4_public_plan_sha256=_require_text_value(
            source["p4_public_plan_sha256"], "p4_public_plan_sha256"
        ),
        python_executable=_require_text_value(
            runtime["python_executable"], "python_executable"
        ),
        python_version=_require_text_value(runtime["python_version"], "python_version"),
        platform_system=_require_text_value(
            runtime["platform_system"], "platform_system"
        ),
        platform_release=_require_text_value(
            runtime["platform_release"], "platform_release"
        ),
        parent_pid=_require_int_value(runtime["parent_pid"], "parent_pid"),
        pulses=frozen_pulses,
        correct_empty_forecast_presence_change_count=_require_int_value(
            metrics["correct_empty_forecast_presence_change_count"],
            "correct_empty_forecast_presence_change_count",
        ),
        correct_swapped_recommended_action_change_count=_require_int_value(
            metrics["correct_swapped_recommended_action_change_count"],
            "correct_swapped_recommended_action_change_count",
        ),
        mechanical_cross_process_chain_observed=_require_bool_value(
            metrics["mechanical_cross_process_chain_observed"],
            "mechanical_cross_process_chain_observed",
        ),
        seen_fixture_only=_require_bool_value(
            firewall["seen_fixture_only"], "seen_fixture_only"
        ),
        independent_subject_count=_require_int_value(
            firewall["independent_subject_count"], "independent_subject_count"
        ),
        raw_history_replayed=_require_bool_value(
            firewall["raw_history_replayed"], "raw_history_replayed"
        ),
        evaluator_loaded=_require_bool_value(
            firewall["evaluator_loaded"], "evaluator_loaded"
        ),
        learning_applied=_require_bool_value(
            firewall["learning_applied"], "learning_applied"
        ),
        model_output_count=_require_int_value(
            firewall["model_output_count"], "model_output_count"
        ),
        qwen_output_count=_require_int_value(
            firewall["qwen_output_count"], "qwen_output_count"
        ),
        residual_intervention_count=_require_int_value(
            firewall["residual_intervention_count"],
            "residual_intervention_count",
        ),
        formal_evidence_authorized=_require_bool_value(
            firewall["formal_evidence_authorized"],
            "formal_evidence_authorized",
        ),
        verdict=_require_text_value(payload["verdict"], "verdict"),
        claim_boundary=_require_text_value(payload["claim_boundary"], "claim_boundary"),
    )
    if reconstructed.to_payload() != payload or reconstructed.artifact_id != artifact_id:
        raise ValueError("P4 cross-process reconstructed report drift")
    protocol = load_relationship_p4_cross_process_protocol()
    if (
        reconstructed.protocol_id != protocol.protocol_id
        or reconstructed.p4_protocol_sha256 != protocol.p4_protocol_sha256
        or reconstructed.p4_public_plan_sha256 != protocol.p4_public_plan_sha256
        or reconstructed.claim_boundary != protocol.claim_boundary
    ):
        raise ValueError("P4 cross-process report protocol lineage drift")
    canonical_view = load_relationship_p4_longitudinal_canary_view()
    canonical_subjects = {item.subject_id: item for item in canonical_view.subjects}
    canonical_sessions = {
        (subject.subject_id, pulse_index): session
        for subject in canonical_view.subjects
        for pulse_index, session in enumerate(
            _canonical_public_session_payloads(subject)
        )
    }

    for pulse, pulse_map in zip(frozen_pulses, pulse_payloads, strict=True):
        for path_key, digest_key in (
            ("session_path", "session_sha256"),
            ("request_path", "request_sha256"),
            ("receipt_path", "receipt_sha256"),
        ):
            relative = pathlib.PurePosixPath(str(pulse_map[path_key]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("P4 cross-process report path escapes output")
            actual_path = output.joinpath(*relative.parts)
            if _sha256_file(actual_path) != pulse_map[digest_key]:
                raise ValueError("P4 cross-process referenced artifact drift")
        session_path = output.joinpath(
            *pathlib.PurePosixPath(pulse.session_path).parts
        )
        request_path = output.joinpath(
            *pathlib.PurePosixPath(pulse.request_path).parts
        )
        receipt_path = output.joinpath(
            *pathlib.PurePosixPath(pulse.receipt_path).parts
        )
        state_path = output.joinpath(*pathlib.PurePosixPath(pulse.state_root).parts)
        request = _load_json_object(request_path)
        _validate_worker_request(
            request,
            expected_parent_pid=reconstructed.parent_pid,
        )
        session = _load_json_object(session_path)
        _validate_session_payload(session, pulse_index=pulse.pulse_index)
        canonical_subject = canonical_subjects.get(pulse.subject_id)
        if canonical_subject is None:
            raise ValueError("P4 cross-process report subject is not canonical")
        if pulse.subject_scope != canonical_subject.subject_scope:
            raise ValueError("P4 cross-process report subject scope drift")
        if session != canonical_sessions[(pulse.subject_id, pulse.pulse_index)]:
            raise ValueError("P4 cross-process public session lineage drift")
        receipt = _load_json_object(receipt_path)
        _validate_worker_receipt(
            receipt,
            request=request,
            request_sha256=pulse.request_sha256,
            host_child_pid=pulse.host_child_pid,
            session=session,
        )
        _validate_report_pulse_lineage(
            report=reconstructed,
            pulse=pulse,
            request=request,
            receipt=receipt,
            session=session,
            protocol_max_versions=protocol.max_versions,
        )
        if (
            _resolve_run_relative_path(output, request["state_root"], "state_root")
            != state_path.resolve()
        ):
            raise ValueError("P4 cross-process request state path drift")
        if (
            _resolve_run_relative_path(output, request["session_path"], "session_path")
            != session_path.resolve()
        ):
            raise ValueError("P4 cross-process request session path drift")
        if _state_directory_sha256(state_path) != pulse.post_state_sha256:
            raise ValueError("P4 cross-process persisted state directory drift")
        post_owner_payload = _validate_checkpoint_against_pulse(
            state_path=state_path,
            version=pulse.post_backend_version,
            expected_raw_sha256=pulse.post_raw_file_sha256,
            expected_backend_sha256=pulse.post_backend_bytes_sha256,
            expected_owner_payload_sha256=pulse.post_owner_payload_sha256,
            label="post",
        )
        _validate_checkpoint_semantics(
            pulse=pulse,
            receipt=receipt,
            session=session,
            owner_payload=post_owner_payload,
        )
        if pulse.pre_backend_version is not None:
            if (
                pulse.pre_raw_file_sha256 is None
                or pulse.pre_backend_bytes_sha256 is None
            ):
                raise ValueError("P4 cross-process retained prior digest omitted")
            _validate_checkpoint_against_pulse(
                state_path=state_path,
                version=pulse.pre_backend_version,
                expected_raw_sha256=pulse.pre_raw_file_sha256,
                expected_backend_sha256=pulse.pre_backend_bytes_sha256,
                expected_owner_payload_sha256=pulse.pre_owner_payload_sha256,
                label="pre",
            )
        _validate_source_state_archive(output, pulse)
    if any(
        b"\r\n" in path.read_bytes()
        for path in output.rglob("*")
        if path.is_file() and path.suffix in {".json", ".md"}
    ):
        raise ValueError("P4 cross-process artifacts contain CRLF drift")
    if markdown_path.read_text(encoding="utf-8") != render_relationship_p4_cross_process_markdown(
        reconstructed
    ):
        raise ValueError("P4 cross-process markdown artifact drift")


def _validate_worker_receipt(
    receipt: Mapping[str, object],
    *,
    request: Mapping[str, object],
    request_sha256: str,
    host_child_pid: int,
    session: Mapping[str, object],
) -> None:
    _require_exact_keys(receipt, _RECEIPT_KEYS, "P4 cross-process worker receipt")
    _validate_worker_request(request, expected_parent_pid=None)
    _validate_session_payload(session, pulse_index=request["pulse_index"])
    if receipt["schema_version"] != P4_CROSS_PROCESS_RECEIPT_SCHEMA_VERSION:
        raise ValueError("P4 cross-process worker receipt schema mismatch")
    for key in (
        "protocol_id",
        "subject_id",
        "subject_scope",
        "python_executable",
        "python_version",
        "platform_system",
        "session_kind",
        "session_id",
        "backend",
        "hydration_store",
        "wiring_level",
        "owner_name",
        "owner_key",
        "forecast_status",
    ):
        _require_text_value(receipt[key], f"receipt {key}")
    for key in (
        "protocol_id",
        "subject_scope",
        "request_sha256",
        "session_sha256",
        "source_state_sha256",
        "pre_state_sha256",
        "post_state_sha256",
        "post_backend_bytes_sha256",
        "post_raw_file_sha256",
        "pre_owner_payload_sha256",
        "post_owner_payload_sha256",
    ):
        _require_sha256_value(receipt[key], f"receipt {key}")
    for key in (
        "pre_backend_bytes_sha256",
        "pre_raw_file_sha256",
    ):
        optional_digest = _optional_text_value(receipt[key], f"receipt {key}")
        if optional_digest is not None:
            _require_sha256(optional_digest, f"receipt {key}")
    _optional_text_value(receipt["source_subject_id"], "receipt source_subject_id")
    _optional_text_value(receipt["forecast_id"], "receipt forecast_id")
    _optional_text_value(
        receipt["recommended_action_id"], "receipt recommended_action_id"
    )
    for key in (
        "pulse_index",
        "output_boundary",
        "source_boundary",
        "child_pid",
        "parent_pid",
        "max_versions",
        "post_backend_version",
        "forecast_count",
        "forecast_source_record_count",
        "owner_record_count",
        "owner_action_outcome_count",
        "owner_forecast_count",
        "model_output_count",
        "qwen_output_count",
        "residual_intervention_count",
    ):
        if _require_int_value(receipt[key], f"receipt {key}") < 0:
            raise ValueError(f"P4 cross-process receipt {key} must be non-negative")
    pre_version = _optional_int_value(
        receipt["pre_backend_version"], "receipt pre_backend_version"
    )
    if pre_version is not None and pre_version <= 0:
        raise ValueError("P4 cross-process receipt pre version must be positive")
    for key in (
        "load_attempted",
        "owner_loaded",
        "gate_invoked",
        "environment_settled",
        "prediction_error_computed",
        "credit_computed",
        "learning_applied",
        "raw_history_replayed",
        "formal_evidence_authorized",
    ):
        _require_bool_value(receipt[key], f"receipt {key}")
    for key in (
        "protocol_id",
        "arm",
        "subject_id",
        "subject_scope",
        "pulse_index",
        "output_boundary",
        "source_subject_id",
        "source_boundary",
        "invocation_nonce",
    ):
        if receipt[key] != request[key]:
            raise ValueError(f"P4 cross-process receipt request drift: {key}")
    _require_sha256(request_sha256, "request_sha256")
    if receipt["request_sha256"] != request_sha256:
        raise ValueError("P4 cross-process receipt request digest drift")
    if receipt["child_pid"] != host_child_pid:
        raise ValueError("P4 cross-process receipt host PID drift")
    if receipt["parent_pid"] != request["parent_pid"]:
        raise ValueError("P4 cross-process receipt parent PID drift")
    if (
        receipt["backend"] != "FileSystemPersistenceBackend"
        or receipt["hydration_store"] != "OwnerHydrationStore"
        or receipt["wiring_level"] != WiringLevel.ACTIVE.value
        or receipt["owner_name"] != _OWNER_NAME
        or receipt["owner_key"] != _OWNER_KEY
        or receipt["load_attempted"] is not True
    ):
        raise ValueError("P4 cross-process receipt owner contract drift")
    if receipt["max_versions"] != request["max_versions"]:
        raise ValueError("P4 cross-process receipt retention drift")
    if receipt["source_state_sha256"] != receipt["pre_state_sha256"]:
        raise ValueError("P4 cross-process receipt staged state drift")
    owner_loaded = receipt["owner_loaded"]
    if owner_loaded != (pre_version is not None):
        raise ValueError("P4 cross-process receipt hydration/version drift")
    if owner_loaded != (receipt["pre_raw_file_sha256"] is not None):
        raise ValueError("P4 cross-process receipt hydration/raw drift")
    if owner_loaded != (receipt["pre_backend_bytes_sha256"] is not None):
        raise ValueError("P4 cross-process receipt hydration/backend drift")
    expected_post_version = 1 if pre_version is None else pre_version + 1
    if receipt["post_backend_version"] != expected_post_version:
        raise ValueError("P4 cross-process receipt backend version drift")
    if receipt["pre_raw_file_sha256"] != receipt["pre_backend_bytes_sha256"]:
        raise ValueError("P4 cross-process receipt prior raw/backend bytes drift")
    if receipt["post_raw_file_sha256"] != receipt["post_backend_bytes_sha256"]:
        raise ValueError("P4 cross-process receipt post raw/backend bytes drift")
    if (
        receipt["session_kind"] != session["pulse_kind"]
        or receipt["session_id"] != session["session_id"]
    ):
        raise ValueError("P4 cross-process receipt session lineage drift")
    if receipt["session_kind"] == "onboarding":
        if (
            receipt["forecast_status"] != "not_requested"
            or receipt["forecast_count"] != 0
            or receipt["forecast_id"] is not None
            or receipt["recommended_action_id"] is not None
            or receipt["forecast_source_record_count"] != 0
        ):
            raise ValueError("P4 cross-process onboarding receipt forecast drift")
    elif receipt["forecast_count"] == 0:
        if (
            receipt["forecast_status"] != "absent_no_owner_evidence"
            or receipt["forecast_id"] is not None
            or receipt["recommended_action_id"] is not None
            or receipt["forecast_source_record_count"] != 0
        ):
            raise ValueError("P4 cross-process absent forecast receipt drift")
    elif receipt["forecast_count"] == 1:
        if (
            receipt["forecast_status"] != "published"
            or _optional_text_value(receipt["forecast_id"], "receipt forecast_id")
            is None
            or _optional_text_value(
                receipt["recommended_action_id"], "receipt recommended_action_id"
            )
            is None
            or receipt["forecast_source_record_count"] <= 0
        ):
            raise ValueError("P4 cross-process published forecast receipt drift")
    else:
        raise ValueError("P4 cross-process receipt forecast cardinality drift")
    for flag in (
        "gate_invoked",
        "environment_settled",
        "prediction_error_computed",
        "credit_computed",
        "learning_applied",
        "raw_history_replayed",
        "formal_evidence_authorized",
    ):
        if receipt[flag] is not False:
            raise ValueError(f"P4 cross-process receipt firewall open: {flag}")
    for count in (
        "model_output_count",
        "qwen_output_count",
        "residual_intervention_count",
    ):
        if receipt[count] != 0:
            raise ValueError(f"P4 cross-process receipt count drift: {count}")


def _validate_worker_request(
    request: Mapping[str, object],
    *,
    expected_parent_pid: int | None,
) -> None:
    _require_exact_keys(request, _REQUEST_KEYS, "P4 cross-process worker request")
    if _FORBIDDEN_REQUEST_KEYS.intersection(request):
        raise ValueError("P4 cross-process request contains forbidden history state")
    if request["schema_version"] != P4_CROSS_PROCESS_REQUEST_SCHEMA_VERSION:
        raise ValueError("P4 cross-process request schema mismatch")
    arm = P4CrossProcessArm(_require_text_value(request["arm"], "request arm"))
    _require_sha256_value(request["protocol_id"], "request protocol_id")
    subject_id = _require_text_value(request["subject_id"], "request subject_id")
    _require_sha256_value(request["subject_scope"], "request subject_scope")
    _require_sha256_value(
        request["source_state_sha256"], "request source_state_sha256"
    )
    source_subject_id = _optional_text_value(
        request["source_subject_id"], "request source_subject_id"
    )
    nonce = _require_text_value(request["invocation_nonce"], "request nonce")
    if len(nonce) != 32 or any(
        character not in "0123456789abcdef" for character in nonce
    ):
        raise ValueError("P4 cross-process invocation nonce is invalid")
    pulse_index = _require_int_value(request["pulse_index"], "request pulse_index")
    output_boundary = _require_int_value(
        request["output_boundary"], "request output_boundary"
    )
    source_boundary = _require_int_value(
        request["source_boundary"], "request source_boundary"
    )
    parent_pid = _require_int_value(request["parent_pid"], "request parent_pid")
    max_versions = _require_int_value(request["max_versions"], "request max_versions")
    if parent_pid <= 0:
        raise ValueError("P4 cross-process request parent PID must be positive")
    if expected_parent_pid is not None and parent_pid != expected_parent_pid:
        raise ValueError("P4 cross-process worker parent PID mismatch")
    if not 0 <= pulse_index < _EXPECTED_PULSE_COUNT:
        raise ValueError("P4 cross-process request pulse index is invalid")
    if output_boundary != pulse_index + 1:
        raise ValueError("P4 cross-process request boundary drift")
    if source_boundary < 0 or source_boundary >= output_boundary:
        raise ValueError("P4 cross-process request leaks a future boundary")
    if max_versions < _EXPECTED_PULSE_COUNT:
        raise ValueError("P4 cross-process request retention is too small")
    _require_relative_posix_value(request["state_root"], "request state_root")
    _require_relative_posix_value(request["session_path"], "request session_path")
    if arm is P4CrossProcessArm.CORRECT_PRIOR_STATE:
        if source_subject_id != subject_id or source_boundary != pulse_index:
            raise ValueError("P4 correct request source lineage drift")
    elif arm is P4CrossProcessArm.EMPTY_PRIOR_STATE:
        if source_subject_id is not None or source_boundary != 0:
            raise ValueError("P4 empty request source lineage drift")
    elif (
        source_subject_id is None
        or source_subject_id == subject_id
        or source_boundary != pulse_index
    ):
        raise ValueError("P4 swapped request source lineage drift")


def _validate_session_payload(
    session: Mapping[str, object],
    *,
    pulse_index: int,
) -> None:
    if "pulse_kind" not in session:
        raise ValueError("P4 cross-process session omitted pulse_kind")
    session_kind = _require_text_value(session["pulse_kind"], "session pulse_kind")
    if session_kind == "onboarding":
        _require_exact_keys(session, _ONBOARDING_KEYS, "P4 onboarding pulse")
        if session["schema_version"] != P4_CROSS_PROCESS_ONBOARDING_SCHEMA_VERSION:
            raise ValueError("P4 onboarding session schema drift")
        if _require_int_value(session["session_index"], "session_index") != pulse_index:
            raise ValueError("P4 onboarding session index drift")
        text_keys = _ONBOARDING_KEYS - {"session_index"}
    elif session_kind == "decision_probe":
        _require_exact_keys(session, _DECISION_KEYS, "P4 decision pulse")
        if session["schema_version"] != P4_CROSS_PROCESS_DECISION_SCHEMA_VERSION:
            raise ValueError("P4 decision session schema drift")
        if _require_int_value(
            session["decision_index"], "decision_index"
        ) != pulse_index - _EXPECTED_ONBOARDING_COUNT:
            raise ValueError("P4 decision session index drift")
        if _require_int_value(session["action_turn_index"], "action_turn_index") < 0:
            raise ValueError("P4 decision turn index must be non-negative")
        for key in ("candidate_action_ids", "typed_outcome_ids"):
            values = session[key]
            if not isinstance(values, list) or not values:
                raise TypeError(f"session {key} must be a non-empty list")
            texts = tuple(
                _require_text_value(value, f"session {key} item") for value in values
            )
            if len(set(texts)) != len(texts):
                raise ValueError(f"session {key} contains duplicates")
        text_keys = _DECISION_KEYS - {
            "decision_index",
            "action_turn_index",
            "candidate_action_ids",
            "typed_outcome_ids",
        }
    else:
        raise ValueError("P4 cross-process session kind is invalid")
    if _FORBIDDEN_REQUEST_KEYS.intersection(session):
        raise ValueError("P4 cross-process session contains forbidden history state")
    for key in text_keys:
        _require_text_value(session[key], f"session {key}")


def _validate_report_pulse_lineage(
    *,
    report: RelationshipP4CrossProcessAppendableReport,
    pulse: P4CrossProcessPulseEvidence,
    request: Mapping[str, object],
    receipt: Mapping[str, object],
    session: Mapping[str, object],
    protocol_max_versions: int,
) -> None:
    request_expected = {
        "protocol_id": report.protocol_id,
        "arm": pulse.arm.value,
        "subject_id": pulse.subject_id,
        "subject_scope": pulse.subject_scope,
        "pulse_index": pulse.pulse_index,
        "output_boundary": pulse.output_boundary,
        "source_subject_id": pulse.source_subject_id,
        "source_boundary": pulse.source_boundary,
        "source_state_sha256": pulse.source_state_sha256,
        "state_root": pulse.state_root,
        "session_path": pulse.session_path,
        "parent_pid": report.parent_pid,
        "max_versions": protocol_max_versions,
    }
    for key, expected in request_expected.items():
        if request[key] != expected:
            raise ValueError(f"P4 cross-process report/request drift: {key}")
    receipt_expected = {
        "protocol_id": report.protocol_id,
        "arm": pulse.arm.value,
        "subject_id": pulse.subject_id,
        "subject_scope": pulse.subject_scope,
        "pulse_index": pulse.pulse_index,
        "output_boundary": pulse.output_boundary,
        "session_kind": pulse.session_kind,
        "session_id": pulse.session_id,
        "invocation_nonce": pulse.invocation_nonce,
        "child_pid": pulse.receipt_child_pid,
        "parent_pid": pulse.receipt_parent_pid,
        "source_subject_id": pulse.source_subject_id,
        "source_boundary": pulse.source_boundary,
        "source_state_sha256": pulse.source_state_sha256,
        "pre_state_sha256": pulse.pre_state_sha256,
        "post_state_sha256": pulse.post_state_sha256,
        "owner_loaded": pulse.owner_loaded,
        "pre_backend_version": pulse.pre_backend_version,
        "post_backend_version": pulse.post_backend_version,
        "pre_raw_file_sha256": pulse.pre_raw_file_sha256,
        "post_raw_file_sha256": pulse.post_raw_file_sha256,
        "pre_backend_bytes_sha256": pulse.pre_backend_bytes_sha256,
        "post_backend_bytes_sha256": pulse.post_backend_bytes_sha256,
        "pre_owner_payload_sha256": pulse.pre_owner_payload_sha256,
        "post_owner_payload_sha256": pulse.post_owner_payload_sha256,
        "forecast_status": pulse.forecast_status,
        "forecast_count": pulse.forecast_count,
        "recommended_action_id": pulse.recommended_action_id,
        "forecast_source_record_count": pulse.forecast_source_record_count,
        "owner_record_count": pulse.owner_record_count,
        "owner_action_outcome_count": pulse.owner_action_outcome_count,
        "owner_forecast_count": pulse.owner_forecast_count,
        "session_sha256": pulse.session_sha256,
        "request_sha256": pulse.request_sha256,
    }
    for key, expected in receipt_expected.items():
        if receipt[key] != expected:
            raise ValueError(f"P4 cross-process report/receipt drift: {key}")
    if pulse.host_child_pid != receipt["child_pid"]:
        raise ValueError("P4 cross-process report host/receipt PID drift")
    if (
        report.python_executable != receipt["python_executable"]
        or report.python_version != receipt["python_version"]
        or report.platform_system != receipt["platform_system"]
    ):
        raise ValueError("P4 cross-process report child runtime drift")
    if (
        session["pulse_kind"] != pulse.session_kind
        or session["session_id"] != pulse.session_id
    ):
        raise ValueError("P4 cross-process report/session lineage drift")


def _validate_checkpoint_against_pulse(
    *,
    state_path: pathlib.Path,
    version: int,
    expected_raw_sha256: str,
    expected_backend_sha256: str,
    expected_owner_payload_sha256: str,
    label: str,
) -> Mapping[str, Any]:
    checkpoint_path = _checkpoint_path(state_path, version)
    raw_bytes = checkpoint_path.read_bytes()
    try:
        # FileSystemPersistenceBackend.load_checkpoint reads in text mode and
        # then UTF-8 encodes; reproduce that contract independently here.
        backend_equivalent_bytes = checkpoint_path.read_text(
            encoding="utf-8"
        ).encode("utf-8")
        checkpoint = json.loads(backend_equivalent_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"P4 cross-process {label} checkpoint is invalid JSON") from exc
    if raw_bytes != backend_equivalent_bytes:
        raise ValueError(f"P4 cross-process {label} raw/backend bytes differ")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"P4 cross-process {label} checkpoint must be an object")
    _require_exact_keys(
        checkpoint,
        {"owner_name", "schema_version", "payload", "description"},
        f"P4 cross-process {label} checkpoint",
    )
    if checkpoint["owner_name"] != _OWNER_NAME:
        raise ValueError(f"P4 cross-process {label} checkpoint owner drift")
    if _require_int_value(
        checkpoint["schema_version"], f"{label} checkpoint schema_version"
    ) <= 0:
        raise ValueError(f"P4 cross-process {label} checkpoint schema is invalid")
    if not isinstance(checkpoint["description"], str):
        raise TypeError(f"P4 cross-process {label} checkpoint description must be text")
    owner_payload = _require_mapping(
        checkpoint["payload"], f"P4 cross-process {label} owner payload"
    )
    canonical_backend_bytes = json.dumps(checkpoint, sort_keys=True).encode("utf-8")
    if backend_equivalent_bytes != canonical_backend_bytes:
        raise ValueError(f"P4 cross-process {label} checkpoint encoding drift")
    if _sha256_bytes(raw_bytes) != expected_raw_sha256:
        raise ValueError(f"P4 cross-process {label} raw checkpoint drift")
    if _sha256_bytes(backend_equivalent_bytes) != expected_backend_sha256:
        raise ValueError(f"P4 cross-process {label} backend checkpoint drift")
    if expected_raw_sha256 != expected_backend_sha256:
        raise ValueError(f"P4 cross-process {label} recorded raw/backend digest drift")
    if sha256_json(owner_payload) != expected_owner_payload_sha256:
        raise ValueError(f"P4 cross-process {label} owner payload drift")
    return owner_payload


def _validate_checkpoint_semantics(
    *,
    pulse: P4CrossProcessPulseEvidence,
    receipt: Mapping[str, object],
    session: Mapping[str, object],
    owner_payload: Mapping[str, Any],
) -> None:
    tom_records = _require_mapping(owner_payload["tom_records"], "tom_records")
    owner_record_count = 0
    for records in tom_records.values():
        if not isinstance(records, list):
            raise TypeError("P4 cross-process tom record collection must be a list")
        owner_record_count += len(records)
    outcomes = owner_payload["preference_action_outcomes"]
    forecasts = owner_payload["preference_action_forecasts"]
    if not isinstance(outcomes, list) or not isinstance(forecasts, list):
        raise TypeError("P4 cross-process owner forecast collections must be lists")
    derived: dict[str, object] = {
        "owner_record_count": owner_record_count,
        "owner_action_outcome_count": len(outcomes),
        "owner_forecast_count": len(forecasts),
    }
    if pulse.session_kind == "onboarding":
        derived.update(
            forecast_status="not_requested",
            forecast_count=0,
            forecast_id=None,
            recommended_action_id=None,
            forecast_source_record_count=0,
        )
    else:
        decision_id = session["decision_id"]
        current = []
        for item in forecasts:
            forecast = _require_mapping(item, "owner forecast")
            if forecast.get("decision_id") == decision_id:
                current.append(forecast)
        if len(current) > 1:
            raise ValueError("P4 cross-process checkpoint has duplicate current forecasts")
        if not current:
            derived.update(
                forecast_status="absent_no_owner_evidence",
                forecast_count=0,
                forecast_id=None,
                recommended_action_id=None,
                forecast_source_record_count=0,
            )
        else:
            forecast = current[0]
            source_record_ids = forecast["source_record_ids"]
            if not isinstance(source_record_ids, list):
                raise TypeError("P4 cross-process forecast source IDs must be a list")
            derived.update(
                forecast_status="published",
                forecast_count=1,
                forecast_id=_require_text_value(
                    forecast["forecast_id"], "checkpoint forecast_id"
                ),
                recommended_action_id=_require_text_value(
                    forecast["recommended_action_id"],
                    "checkpoint recommended_action_id",
                ),
                forecast_source_record_count=len(source_record_ids),
            )
    for key, expected in derived.items():
        if receipt[key] != expected:
            raise ValueError(f"P4 cross-process receipt/checkpoint drift: {key}")


def _validate_state_chains(
    pulses: tuple[P4CrossProcessPulseEvidence, ...],
) -> None:
    by_key = {(item.arm, item.subject_id, item.pulse_index): item for item in pulses}
    if len(by_key) != len(pulses):
        raise ValueError("P4 cross-process matched pulse key repeated")
    subjects = tuple(
        dict.fromkeys(
            item.subject_id
            for item in pulses
            if item.arm is P4CrossProcessArm.CORRECT_PRIOR_STATE
        )
    )
    if len(subjects) != _EXPECTED_SUBJECT_COUNT:
        raise ValueError("P4 cross-process matched subject shape drift")
    donor_by_subject = {subjects[0]: subjects[1], subjects[1]: subjects[0]}
    for subject in subjects:
        for pulse_index in range(_EXPECTED_PULSE_COUNT):
            correct = by_key[
                (P4CrossProcessArm.CORRECT_PRIOR_STATE, subject, pulse_index)
            ]
            empty = by_key[
                (P4CrossProcessArm.EMPTY_PRIOR_STATE, subject, pulse_index)
            ]
            swapped = by_key[
                (
                    P4CrossProcessArm.SWAPPED_SUBJECT_PRIOR_STATE,
                    subject,
                    pulse_index,
                )
            ]
            expected_version = pulse_index + 1
            if correct.post_backend_version != expected_version:
                raise ValueError("P4 correct state version chain drift")
            if correct.owner_loaded != (pulse_index > 0):
                raise ValueError("P4 correct state hydration chain drift")
            if empty.owner_loaded or empty.post_backend_version != 1:
                raise ValueError("P4 empty state arm carried a prior boundary")
            if swapped.post_backend_version != expected_version:
                raise ValueError("P4 swapped state version chain drift")
            if swapped.owner_loaded != (pulse_index > 0):
                raise ValueError("P4 swapped state hydration chain drift")
            if correct.source_subject_id != subject:
                raise ValueError("P4 correct state source subject drift")
            if empty.source_subject_id is not None or empty.source_boundary != 0:
                raise ValueError("P4 empty state source lineage drift")
            if (
                swapped.source_subject_id != donor_by_subject[subject]
                or swapped.source_boundary != pulse_index
                or swapped.source_boundary >= swapped.output_boundary
            ):
                raise ValueError("P4 swapped state leaked a future/wrong donor boundary")
            if pulse_index > 0:
                prior_correct = by_key[
                    (
                        P4CrossProcessArm.CORRECT_PRIOR_STATE,
                        subject,
                        pulse_index - 1,
                    )
                ]
                donor_correct = by_key[
                    (
                        P4CrossProcessArm.CORRECT_PRIOR_STATE,
                        donor_by_subject[subject],
                        pulse_index - 1,
                    )
                ]
                if correct.pre_raw_file_sha256 != prior_correct.post_raw_file_sha256:
                    raise ValueError("P4 correct state raw checkpoint chain drift")
                if swapped.pre_raw_file_sha256 != donor_correct.post_raw_file_sha256:
                    raise ValueError("P4 swapped state donor checkpoint drift")


def _matched_forecast_changes(
    pulses: tuple[P4CrossProcessPulseEvidence, ...],
) -> tuple[int, int]:
    by_key = {(item.arm, item.subject_id, item.pulse_index): item for item in pulses}
    keys = tuple(
        (item.subject_id, item.pulse_index)
        for item in pulses
        if item.arm is P4CrossProcessArm.CORRECT_PRIOR_STATE
        and item.session_kind == "decision_probe"
    )
    if len(keys) != _EXPECTED_SUBJECT_COUNT * _EXPECTED_DECISION_COUNT:
        raise ValueError("P4 cross-process decision comparison shape drift")
    presence_changes = 0
    swapped_changes = 0
    for subject_id, pulse_index in keys:
        correct = by_key[
            (P4CrossProcessArm.CORRECT_PRIOR_STATE, subject_id, pulse_index)
        ]
        empty = by_key[
            (P4CrossProcessArm.EMPTY_PRIOR_STATE, subject_id, pulse_index)
        ]
        swapped = by_key[
            (
                P4CrossProcessArm.SWAPPED_SUBJECT_PRIOR_STATE,
                subject_id,
                pulse_index,
            )
        ]
        presence_changes += int(correct.forecast_count != empty.forecast_count)
        swapped_changes += int(
            correct.recommended_action_id != swapped.recommended_action_id
        )
    return presence_changes, swapped_changes


def _pulse_from_payload(
    payload: Mapping[str, object],
) -> P4CrossProcessPulseEvidence:
    _require_exact_keys(
        payload,
        {field.name for field in fields(P4CrossProcessPulseEvidence)},
        "P4 cross-process report pulse",
    )
    return P4CrossProcessPulseEvidence(
        arm=P4CrossProcessArm(
            _require_text_value(payload["arm"], "pulse arm")
        ),
        subject_id=_require_text_value(payload["subject_id"], "subject_id"),
        subject_scope=_require_text_value(
            payload["subject_scope"], "subject_scope"
        ),
        pulse_index=_require_int_value(payload["pulse_index"], "pulse_index"),
        output_boundary=_require_int_value(
            payload["output_boundary"], "output_boundary"
        ),
        session_kind=_require_text_value(
            payload["session_kind"], "session_kind"
        ),
        session_id=_require_text_value(payload["session_id"], "session_id"),
        invocation_nonce=_require_text_value(
            payload["invocation_nonce"], "invocation_nonce"
        ),
        host_child_pid=_require_int_value(
            payload["host_child_pid"], "host_child_pid"
        ),
        receipt_child_pid=_require_int_value(
            payload["receipt_child_pid"], "receipt_child_pid"
        ),
        receipt_parent_pid=_require_int_value(
            payload["receipt_parent_pid"], "receipt_parent_pid"
        ),
        source_subject_id=_optional_text_value(
            payload["source_subject_id"], "source_subject_id"
        ),
        source_boundary=_require_int_value(
            payload["source_boundary"], "source_boundary"
        ),
        source_state_sha256=_require_text_value(
            payload["source_state_sha256"], "source_state_sha256"
        ),
        pre_state_sha256=_require_text_value(
            payload["pre_state_sha256"], "pre_state_sha256"
        ),
        post_state_sha256=_require_text_value(
            payload["post_state_sha256"], "post_state_sha256"
        ),
        owner_loaded=_require_bool_value(
            payload["owner_loaded"], "owner_loaded"
        ),
        pre_backend_version=_optional_int_value(
            payload["pre_backend_version"], "pre_backend_version"
        ),
        post_backend_version=_require_int_value(
            payload["post_backend_version"], "post_backend_version"
        ),
        pre_raw_file_sha256=_optional_text_value(
            payload["pre_raw_file_sha256"], "pre_raw_file_sha256"
        ),
        post_raw_file_sha256=_require_text_value(
            payload["post_raw_file_sha256"], "post_raw_file_sha256"
        ),
        pre_backend_bytes_sha256=_optional_text_value(
            payload["pre_backend_bytes_sha256"],
            "pre_backend_bytes_sha256",
        ),
        post_backend_bytes_sha256=_require_text_value(
            payload["post_backend_bytes_sha256"],
            "post_backend_bytes_sha256",
        ),
        pre_owner_payload_sha256=_require_text_value(
            payload["pre_owner_payload_sha256"],
            "pre_owner_payload_sha256",
        ),
        post_owner_payload_sha256=_require_text_value(
            payload["post_owner_payload_sha256"],
            "post_owner_payload_sha256",
        ),
        forecast_status=_require_text_value(
            payload["forecast_status"], "forecast_status"
        ),
        forecast_count=_require_int_value(
            payload["forecast_count"], "forecast_count"
        ),
        recommended_action_id=_optional_text_value(
            payload["recommended_action_id"], "recommended_action_id"
        ),
        forecast_source_record_count=_require_int_value(
            payload["forecast_source_record_count"],
            "forecast_source_record_count",
        ),
        owner_record_count=_require_int_value(
            payload["owner_record_count"], "owner_record_count"
        ),
        owner_action_outcome_count=_require_int_value(
            payload["owner_action_outcome_count"],
            "owner_action_outcome_count",
        ),
        owner_forecast_count=_require_int_value(
            payload["owner_forecast_count"], "owner_forecast_count"
        ),
        state_root=_require_text_value(payload["state_root"], "state_root"),
        session_path=_require_text_value(
            payload["session_path"], "session_path"
        ),
        request_path=_require_text_value(
            payload["request_path"], "request_path"
        ),
        receipt_path=_require_text_value(
            payload["receipt_path"], "receipt_path"
        ),
        session_sha256=_require_text_value(
            payload["session_sha256"], "session_sha256"
        ),
        request_sha256=_require_text_value(
            payload["request_sha256"], "request_sha256"
        ),
        receipt_sha256=_require_text_value(
            payload["receipt_sha256"], "receipt_sha256"
        ),
    )


def _validate_source_state_archive(
    output: pathlib.Path,
    pulse: P4CrossProcessPulseEvidence,
) -> None:
    if pulse.arm is P4CrossProcessArm.EMPTY_PRIOR_STATE:
        expected = sha256_json(())
    else:
        if pulse.source_subject_id is None:
            raise ValueError("P4 cross-process stateful arm omitted source subject")
        source_relative = pathlib.PurePosixPath(
            "state",
            P4CrossProcessArm.CORRECT_PRIOR_STATE.value,
            pulse.source_subject_id,
            f"boundary-{pulse.source_boundary:03d}",
        )
        source_path = output.joinpath(*source_relative.parts)
        expected = _state_directory_sha256(source_path)
    if expected != pulse.source_state_sha256:
        raise ValueError("P4 cross-process immutable source archive drift")


def _checkpoint_path(state_root: pathlib.Path, version: int) -> pathlib.Path:
    path = state_root / f"{_OWNER_SAFE_KEY}_v{version}.json"
    if not path.is_file():
        raise FileNotFoundError(f"P4 cross-process raw checkpoint missing: {path}")
    return path


def _state_directory_sha256(root: pathlib.Path) -> str:
    directory = pathlib.Path(root)
    manifest = tuple(
        (
            path.relative_to(directory).as_posix(),
            path.stat().st_size,
            _sha256_file(path),
        )
        for path in sorted(
            (item for item in directory.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(directory).as_posix(),
        )
    )
    return sha256_json(manifest)


def _write_create_json(path: pathlib.Path, payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    return _write_create_bytes(path, f"{serialized}\n".encode("utf-8"))


def _write_create_bytes(path: pathlib.Path, payload: bytes) -> str:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(payload)


def _load_json_object(path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = json.loads(pathlib.Path(path).read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return raw


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str] | frozenset[str],
    label: str,
) -> None:
    actual = set(payload)
    if actual != set(expected):
        raise ValueError(
            f"{label} keys drifted: missing={sorted(set(expected) - actual)!r}, "
            f"extra={sorted(actual - set(expected))!r}"
        )


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _require_sha256_value(value: object, label: str) -> str:
    text = _require_text_value(value, label)
    _require_sha256(text, label)
    return text


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_posix(root: pathlib.Path, path: pathlib.Path) -> str:
    return pathlib.Path(path).resolve().relative_to(root.resolve()).as_posix()


def _resolve_run_relative_path(
    run_root: pathlib.Path,
    value: object,
    label: str,
) -> pathlib.Path:
    relative_text = _require_text_value(value, label)
    relative = pathlib.PurePosixPath(relative_text)
    if relative.is_absolute() or ".." in relative.parts or "\\" in relative_text:
        raise ValueError(f"{label} must be a contained relative POSIX path")
    resolved = pathlib.Path(run_root).resolve().joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(pathlib.Path(run_root).resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the run root") from exc
    return resolved


def _require_relative_posix_value(value: object, label: str) -> str:
    text = _require_text_value(value, label)
    relative = pathlib.PurePosixPath(text)
    if relative.is_absolute() or ".." in relative.parts or "\\" in text:
        raise ValueError(f"{label} must be a contained relative POSIX path")
    return text


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _require_text_value(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be non-empty text")
    return value


def _optional_text_value(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_text_value(value, label)


def _require_int_value(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _optional_int_value(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _require_int_value(value, label)


def _require_bool_value(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


__all__ = [
    "P4_CROSS_PROCESS_PROTOCOL_SCHEMA_VERSION",
    "P4_CROSS_PROCESS_RECEIPT_SCHEMA_VERSION",
    "P4_CROSS_PROCESS_REPORT_SCHEMA_VERSION",
    "P4_CROSS_PROCESS_REQUEST_SCHEMA_VERSION",
    "P4CrossProcessArm",
    "P4CrossProcessProtocol",
    "P4CrossProcessPulseEvidence",
    "RelationshipP4CrossProcessAppendableReport",
    "load_relationship_p4_cross_process_protocol",
    "relationship_p4_cross_process_protocol_path",
    "render_relationship_p4_cross_process_markdown",
    "run_relationship_p4_cross_process_appendable_preflight",
    "run_relationship_p4_cross_process_worker",
    "validate_relationship_p4_cross_process_report_files",
    "write_relationship_p4_cross_process_report",
]
