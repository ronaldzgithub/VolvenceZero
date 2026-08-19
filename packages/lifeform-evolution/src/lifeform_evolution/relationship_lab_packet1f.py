"""Relationship Lab P1f: audit public evidence before another model run.

P1f is a development-only solvability check for ``relationship_transfer_v3``.
It asks one frozen semantic encoder whether every public history and unseen
probe is closer to its sealed abstract-condition summary than to the competing
summary.  The audit never exposes those labels to a system under test and does
not create PE, credit, learning, steering, or runtime state.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from companion_ref_harness.embed import Embedder, cosine
from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME,
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RelationshipHistoryEvent,
    RelationshipPublicEvidenceContract,
    RelationshipTransferDataset,
    sha256_json,
)
from lifeform_evolution.relationship_lab_packet1e import (
    RelationshipP1eReport,
    RelationshipP1eVerdict,
)


RELATIONSHIP_P1F_REPORT_SCHEMA_VERSION = "relationship-p1f-public-evidence-audit-report.v1"
_EVIDENCE_KINDS = frozenset({"history", "probe"})
_HEX_DIGITS = frozenset("0123456789abcdef")
_REPORT_CLAIM_BOUNDARY = (
    "P1f establishes only development-set semantic legibility of every public "
    "relationship_transfer_v3 history and probe under one frozen BGE-M3 "
    "auditor. It does not prove human readability, Qwen transfer, Volvence "
    "advantage, Appendable/Readable/Learnable/Steerable, formal held-out "
    "superiority, or product value. Human anchors remain pending and audit "
    "labels never feed learning or steering."
)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX_DIGITS for char in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")
    return value


def _require_timestamp(value: object, field_name: str) -> str:
    text = _require_text(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return text


def _require_exact_object(
    value: object,
    *,
    field_name: str,
    expected_fields: set[str],
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{field_name} fields do not match the frozen schema")
    return value


def _require_metric(value: object, field_name: str, *, signed: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    lower = -2.0 if signed else 0.0
    if not math.isfinite(result) or not lower <= result <= 2.0:
        raise ValueError(f"{field_name} is outside its valid range")
    return result


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_relationship_p1f_embedder_name(
    contract: RelationshipPublicEvidenceContract,
) -> str:
    return (
        f"relationship-lab/{contract.semantic_audit_embedder}:"
        f"{contract.semantic_audit_model_source}:sha256:"
        f"{contract.semantic_audit_weights_sha256}"
    )


@dataclass(frozen=True)
class RelationshipP1fEvidenceUnit:
    evidence_kind: str
    source_id: str
    public_text_sha256: str
    expected_anchor_sha256: str
    predicted_anchor_sha256: str
    expected_score: float
    competitor_score: float
    correct_anchor_margin: float
    correct: bool

    def __post_init__(self) -> None:
        if self.evidence_kind not in _EVIDENCE_KINDS:
            raise ValueError("P1f evidence_kind is invalid")
        _require_text(self.source_id, "source_id")
        for field_name, value in (
            ("public_text_sha256", self.public_text_sha256),
            ("expected_anchor_sha256", self.expected_anchor_sha256),
            ("predicted_anchor_sha256", self.predicted_anchor_sha256),
        ):
            _require_sha256(value, field_name)
        expected = _require_metric(self.expected_score, "expected_score", signed=True)
        competitor = _require_metric(
            self.competitor_score,
            "competitor_score",
            signed=True,
        )
        margin = _require_metric(
            self.correct_anchor_margin,
            "correct_anchor_margin",
            signed=True,
        )
        if margin != round(expected - competitor, 12):
            raise ValueError("P1f evidence margin does not match its scores")
        if not isinstance(self.correct, bool) or self.correct is not (margin > 0.0):
            raise ValueError("P1f evidence correctness does not match its margin")
        predicted_is_expected = self.predicted_anchor_sha256 == self.expected_anchor_sha256
        if predicted_is_expected is not self.correct:
            raise ValueError("P1f predicted anchor does not match correctness")

    def to_payload(self) -> dict[str, object]:
        return {
            "evidence_kind": self.evidence_kind,
            "source_id": self.source_id,
            "public_text_sha256": self.public_text_sha256,
            "expected_anchor_sha256": self.expected_anchor_sha256,
            "predicted_anchor_sha256": self.predicted_anchor_sha256,
            "expected_score": self.expected_score,
            "competitor_score": self.competitor_score,
            "correct_anchor_margin": self.correct_anchor_margin,
            "correct": self.correct,
        }

    @classmethod
    def from_payload(
        cls,
        value: object,
        *,
        field_name: str,
    ) -> "RelationshipP1fEvidenceUnit":
        payload = _require_exact_object(
            value,
            field_name=field_name,
            expected_fields={
                "competitor_score",
                "correct",
                "correct_anchor_margin",
                "evidence_kind",
                "expected_anchor_sha256",
                "expected_score",
                "predicted_anchor_sha256",
                "public_text_sha256",
                "source_id",
            },
        )
        return cls(
            evidence_kind=payload["evidence_kind"],
            source_id=payload["source_id"],
            public_text_sha256=payload["public_text_sha256"],
            expected_anchor_sha256=payload["expected_anchor_sha256"],
            predicted_anchor_sha256=payload["predicted_anchor_sha256"],
            expected_score=payload["expected_score"],
            competitor_score=payload["competitor_score"],
            correct_anchor_margin=payload["correct_anchor_margin"],
            correct=payload["correct"],
        )


class RelationshipP1fVerdict(str, Enum):
    CONSUMER_PROTOCOL_FREEZE_CANDIDATE = "consumer_protocol_freeze_candidate"
    REWRITE_PUBLIC_EVIDENCE_CONTRACT_AGAIN = "rewrite_public_evidence_contract_again"


_NEXT_ACTIONS = {
    RelationshipP1fVerdict.CONSUMER_PROTOCOL_FREEZE_CANDIDATE: (
        "Freeze the v3 Qwen consumer protocol in P1g before producing any v3 "
        "Qwen output; keep the formal hidden test closed."
    ),
    RelationshipP1fVerdict.REWRITE_PUBLIC_EVIDENCE_CONTRACT_AGAIN: (
        "Revise the public evidence contract again; do not tune a Qwen consumer "
        "against failed v3 outputs or add PE learning/steering."
    ),
}


@dataclass(frozen=True)
class RelationshipP1fReport:
    created_at_iso: str
    source_p1e_report_artifact_id: str
    public_evidence_contract_sha256: str
    package_name: str
    dataset_fingerprint: str
    semantic_auditor_version: str
    semantic_audit_method: str
    semantic_similarity: str
    embedder_name: str
    weights_sha256: str
    score_precision_decimals: int
    required_evidence_units: int
    required_top1_accuracy: float
    required_minimum_correct_anchor_margin: float
    required_minimum_mean_correct_anchor_margin: float
    evidence_units: tuple[RelationshipP1fEvidenceUnit, ...]
    correct_count: int
    top1_accuracy: float
    minimum_correct_anchor_margin: float
    mean_correct_anchor_margin: float
    human_anchor_status: str
    verdict: RelationshipP1fVerdict
    schema_version: str = RELATIONSHIP_P1F_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1F_REPORT_SCHEMA_VERSION:
            raise ValueError("P1f report schema_version mismatch")
        _require_timestamp(self.created_at_iso, "created_at_iso")
        for field_name, value in (
            ("source_p1e_report_artifact_id", self.source_p1e_report_artifact_id),
            (
                "public_evidence_contract_sha256",
                self.public_evidence_contract_sha256,
            ),
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("weights_sha256", self.weights_sha256),
        ):
            _require_sha256(value, field_name)
        if self.package_name != RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
            raise ValueError("P1f report package mismatch")
        if self.semantic_auditor_version != "relationship-public-evidence-auditor.v1":
            raise ValueError("P1f semantic auditor version mismatch")
        if self.semantic_similarity != "cosine":
            raise ValueError("P1f similarity method mismatch")
        _require_text(self.semantic_audit_method, "semantic_audit_method")
        _require_text(self.embedder_name, "embedder_name")
        if self.score_precision_decimals != 12:
            raise ValueError("P1f score precision mismatch")
        if (
            type(self.required_evidence_units) is not int
            or self.required_evidence_units <= 0
            or len(self.evidence_units) != self.required_evidence_units
        ):
            raise ValueError("P1f evidence units do not meet the frozen count")
        for field_name, value in (
            ("required_top1_accuracy", self.required_top1_accuracy),
            (
                "required_minimum_correct_anchor_margin",
                self.required_minimum_correct_anchor_margin,
            ),
            (
                "required_minimum_mean_correct_anchor_margin",
                self.required_minimum_mean_correct_anchor_margin,
            ),
            ("top1_accuracy", self.top1_accuracy),
        ):
            metric = _require_metric(value, field_name)
            if metric > 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        _require_metric(
            self.minimum_correct_anchor_margin,
            "minimum_correct_anchor_margin",
            signed=True,
        )
        _require_metric(
            self.mean_correct_anchor_margin,
            "mean_correct_anchor_margin",
            signed=True,
        )
        if type(self.correct_count) is not int:
            raise ValueError("P1f correct_count must be an integer")
        expected_count = sum(unit.correct for unit in self.evidence_units)
        expected_accuracy = round(expected_count / len(self.evidence_units), 12)
        margins = tuple(unit.correct_anchor_margin for unit in self.evidence_units)
        expected_minimum = min(margins)
        expected_mean = round(sum(margins) / len(margins), 12)
        if (
            self.correct_count != expected_count
            or self.top1_accuracy != expected_accuracy
            or self.minimum_correct_anchor_margin != expected_minimum
            or self.mean_correct_anchor_margin != expected_mean
        ):
            raise ValueError("P1f aggregate metrics diverge from evidence units")
        if self.human_anchor_status != "pending_before_formal":
            raise ValueError("P1f cannot claim a completed human anchor")
        passed = (
            self.correct_count == self.required_evidence_units
            and self.top1_accuracy >= self.required_top1_accuracy
            and self.minimum_correct_anchor_margin >= self.required_minimum_correct_anchor_margin
            and self.mean_correct_anchor_margin >= self.required_minimum_mean_correct_anchor_margin
        )
        expected_verdict = (
            RelationshipP1fVerdict.CONSUMER_PROTOCOL_FREEZE_CANDIDATE
            if passed
            else RelationshipP1fVerdict.REWRITE_PUBLIC_EVIDENCE_CONTRACT_AGAIN
        )
        if self.verdict is not expected_verdict:
            raise ValueError("P1f verdict diverges from frozen thresholds")

    @property
    def next_action(self) -> str:
        return _NEXT_ACTIONS[self.verdict]

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "source_p1e_report_artifact_id": self.source_p1e_report_artifact_id,
            "public_evidence_contract_sha256": (self.public_evidence_contract_sha256),
            "package_name": self.package_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "semantic_auditor_version": self.semantic_auditor_version,
            "semantic_audit_method": self.semantic_audit_method,
            "semantic_similarity": self.semantic_similarity,
            "embedder_name": self.embedder_name,
            "weights_sha256": self.weights_sha256,
            "score_precision_decimals": self.score_precision_decimals,
            "thresholds": {
                "required_evidence_units": self.required_evidence_units,
                "required_top1_accuracy": self.required_top1_accuracy,
                "required_minimum_correct_anchor_margin": (self.required_minimum_correct_anchor_margin),
                "required_minimum_mean_correct_anchor_margin": (self.required_minimum_mean_correct_anchor_margin),
            },
            "evidence_units": [unit.to_payload() for unit in self.evidence_units],
            "metrics": {
                "correct_count": self.correct_count,
                "top1_accuracy": self.top1_accuracy,
                "minimum_correct_anchor_margin": (self.minimum_correct_anchor_margin),
                "mean_correct_anchor_margin": self.mean_correct_anchor_margin,
            },
            "human_anchor_status": self.human_anchor_status,
            "verdict": self.verdict.value,
            "next_action": self.next_action,
            "claim_boundary": _REPORT_CLAIM_BOUNDARY,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self._canonical_payload())

    def to_json(self) -> str:
        payload = self._canonical_payload()
        payload["artifact_id"] = self.artifact_id
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "RelationshipP1fReport":
        raw = json.loads(encoded)
        payload = _require_exact_object(
            raw,
            field_name="P1f report",
            expected_fields={
                "artifact_id",
                "claim_boundary",
                "created_at_iso",
                "dataset_fingerprint",
                "embedder_name",
                "evidence_units",
                "human_anchor_status",
                "metrics",
                "next_action",
                "package_name",
                "public_evidence_contract_sha256",
                "schema_version",
                "score_precision_decimals",
                "semantic_auditor_version",
                "semantic_audit_method",
                "semantic_similarity",
                "source_p1e_report_artifact_id",
                "thresholds",
                "verdict",
                "weights_sha256",
            },
        )
        raw_units = payload["evidence_units"]
        if not isinstance(raw_units, list):
            raise ValueError("P1f evidence_units must be an array")
        units = tuple(
            RelationshipP1fEvidenceUnit.from_payload(
                item,
                field_name=f"P1f evidence_units[{index}]",
            )
            for index, item in enumerate(raw_units)
        )
        thresholds = _require_exact_object(
            payload["thresholds"],
            field_name="P1f thresholds",
            expected_fields={
                "required_evidence_units",
                "required_minimum_correct_anchor_margin",
                "required_minimum_mean_correct_anchor_margin",
                "required_top1_accuracy",
            },
        )
        metrics = _require_exact_object(
            payload["metrics"],
            field_name="P1f metrics",
            expected_fields={
                "correct_count",
                "mean_correct_anchor_margin",
                "minimum_correct_anchor_margin",
                "top1_accuracy",
            },
        )
        try:
            verdict = RelationshipP1fVerdict(payload["verdict"])
        except (TypeError, ValueError) as exc:
            raise ValueError("P1f verdict is invalid") from exc
        report = cls(
            schema_version=payload["schema_version"],
            created_at_iso=payload["created_at_iso"],
            source_p1e_report_artifact_id=(payload["source_p1e_report_artifact_id"]),
            public_evidence_contract_sha256=(payload["public_evidence_contract_sha256"]),
            package_name=payload["package_name"],
            dataset_fingerprint=payload["dataset_fingerprint"],
            semantic_auditor_version=payload["semantic_auditor_version"],
            semantic_audit_method=payload["semantic_audit_method"],
            semantic_similarity=payload["semantic_similarity"],
            embedder_name=payload["embedder_name"],
            weights_sha256=payload["weights_sha256"],
            score_precision_decimals=payload["score_precision_decimals"],
            required_evidence_units=thresholds["required_evidence_units"],
            required_top1_accuracy=thresholds["required_top1_accuracy"],
            required_minimum_correct_anchor_margin=(thresholds["required_minimum_correct_anchor_margin"]),
            required_minimum_mean_correct_anchor_margin=(thresholds["required_minimum_mean_correct_anchor_margin"]),
            evidence_units=units,
            correct_count=metrics["correct_count"],
            top1_accuracy=metrics["top1_accuracy"],
            minimum_correct_anchor_margin=(metrics["minimum_correct_anchor_margin"]),
            mean_correct_anchor_margin=metrics["mean_correct_anchor_margin"],
            human_anchor_status=payload["human_anchor_status"],
            verdict=verdict,
        )
        artifact_id = payload["artifact_id"]
        _require_sha256(artifact_id, "artifact_id")
        if (
            payload["claim_boundary"] != _REPORT_CLAIM_BOUNDARY
            or payload["next_action"] != report.next_action
            or artifact_id != report.artifact_id
        ):
            raise ValueError("P1f report derived fields or artifact_id mismatch")
        return report


def _validate_vector(
    vector: tuple[float, ...],
    *,
    expected_dim: int,
    field_name: str,
) -> tuple[float, ...]:
    if len(vector) != expected_dim:
        raise ValueError(f"{field_name} dimension mismatch")
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{field_name} must be finite and non-empty")
    if math.sqrt(sum(value * value for value in vector)) == 0.0:
        raise ValueError(f"{field_name} cannot be the zero vector")
    return vector


def _history_audit_text(
    history: RelationshipHistoryEvent,
    *,
    joiner: str,
) -> str:
    return joiner.join((history.user_utterance, history.user_reaction))


def _audit_unit(
    *,
    evidence_kind: str,
    source_id: str,
    text: str,
    expected_condition_id: str,
    condition_vectors: dict[str, tuple[float, ...]],
    condition_anchor_hashes: dict[str, str],
    embedder: Embedder,
    precision: int,
) -> RelationshipP1fEvidenceUnit:
    vector = _validate_vector(
        embedder.embed(text),
        expected_dim=embedder.dim,
        field_name=f"P1f public evidence {source_id}",
    )
    if expected_condition_id not in condition_vectors:
        raise ValueError(f"P1f evidence {source_id} references an unknown condition")
    competitor_ids = tuple(condition_id for condition_id in condition_vectors if condition_id != expected_condition_id)
    if len(competitor_ids) != 1:
        raise ValueError("P1f requires exactly one competing abstract condition")
    competitor_id = competitor_ids[0]
    expected_score = round(
        cosine(vector, condition_vectors[expected_condition_id]),
        precision,
    )
    competitor_score = round(
        cosine(vector, condition_vectors[competitor_id]),
        precision,
    )
    margin = round(expected_score - competitor_score, precision)
    correct = margin > 0.0
    predicted_id = expected_condition_id if correct else competitor_id
    return RelationshipP1fEvidenceUnit(
        evidence_kind=evidence_kind,
        source_id=source_id,
        public_text_sha256=_text_sha256(text),
        expected_anchor_sha256=condition_anchor_hashes[expected_condition_id],
        predicted_anchor_sha256=condition_anchor_hashes[predicted_id],
        expected_score=expected_score,
        competitor_score=competitor_score,
        correct_anchor_margin=margin,
        correct=correct,
    )


def assess_relationship_packet1f(
    *,
    dataset: RelationshipTransferDataset,
    source_p1e_report: RelationshipP1eReport,
    embedder: Embedder,
    weights_sha256: str,
    created_at_iso: str | None = None,
) -> RelationshipP1fReport:
    if dataset.package_name != RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
        raise ValueError("P1f requires relationship_transfer_v3")
    contract = dataset.public_evidence_contract
    if contract is None:
        raise ValueError("P1f dataset has no public evidence contract")
    if (
        source_p1e_report.package_name != RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME
        or source_p1e_report.artifact_id != contract.source_p1e_report_artifact_id
        or source_p1e_report.verdict.value != contract.source_required_verdict
        or source_p1e_report.verdict is not RelationshipP1eVerdict.REWRITE_PUBLIC_EVIDENCE_CONTRACT
    ):
        raise ValueError("P1f source P1e report does not satisfy the frozen trigger")
    _require_sha256(weights_sha256, "weights_sha256")
    if weights_sha256 != contract.semantic_audit_weights_sha256:
        raise ValueError("P1f embedder weights diverge from the public evidence contract")
    expected_embedder_name = canonical_relationship_p1f_embedder_name(contract)
    if embedder.name != expected_embedder_name:
        raise ValueError("P1f embedder identity diverges from the frozen contract")
    if embedder.dim <= 0:
        raise ValueError("P1f embedder dimension must be positive")

    conditions = {item.condition_id: item for item in dataset.abstract_conditions}
    if len(conditions) != 2:
        raise ValueError("P1f requires exactly two sealed condition anchors")
    anchor_hashes = {
        condition_id: _text_sha256(condition.hidden_summary) for condition_id, condition in conditions.items()
    }
    if len(set(anchor_hashes.values())) != len(anchor_hashes):
        raise ValueError("P1f sealed condition anchors must be distinct")
    condition_vectors = {
        condition_id: _validate_vector(
            embedder.embed(condition.hidden_summary),
            expected_dim=embedder.dim,
            field_name=f"P1f condition anchor {condition_id}",
        )
        for condition_id, condition in conditions.items()
    }
    history_bindings = dict(dataset.history_condition_bindings)
    units: list[RelationshipP1fEvidenceUnit] = []
    for observation in sorted(dataset.observations, key=lambda item: item.scene_id):
        for history in observation.histories:
            units.append(
                _audit_unit(
                    evidence_kind="history",
                    source_id=history.event_id,
                    text=_history_audit_text(
                        history,
                        joiner=contract.history_text_joiner,
                    ),
                    expected_condition_id=history_bindings[history.event_id],
                    condition_vectors=condition_vectors,
                    condition_anchor_hashes=anchor_hashes,
                    embedder=embedder,
                    precision=contract.score_precision_decimals,
                )
            )
        dynamic = dataset.dynamic_for_scene(observation.scene_id)
        if dynamic.probe_condition_id is None:
            raise ValueError("P1f probe has no sealed condition binding")
        units.append(
            _audit_unit(
                evidence_kind="probe",
                source_id=observation.scene_id,
                text=observation.current_input,
                expected_condition_id=dynamic.probe_condition_id,
                condition_vectors=condition_vectors,
                condition_anchor_hashes=anchor_hashes,
                embedder=embedder,
                precision=contract.score_precision_decimals,
            )
        )
    frozen_units = tuple(units)
    correct_count = sum(unit.correct for unit in frozen_units)
    accuracy = round(correct_count / len(frozen_units), 12)
    margins = tuple(unit.correct_anchor_margin for unit in frozen_units)
    minimum_margin = min(margins)
    mean_margin = round(sum(margins) / len(margins), 12)
    passed = (
        len(frozen_units) == contract.required_evidence_units
        and accuracy >= contract.required_top1_accuracy
        and minimum_margin >= contract.minimum_correct_anchor_margin
        and mean_margin >= contract.minimum_mean_correct_anchor_margin
    )
    verdict = (
        RelationshipP1fVerdict.CONSUMER_PROTOCOL_FREEZE_CANDIDATE
        if passed
        else RelationshipP1fVerdict.REWRITE_PUBLIC_EVIDENCE_CONTRACT_AGAIN
    )
    return RelationshipP1fReport(
        created_at_iso=created_at_iso or datetime.now(timezone.utc).isoformat(),
        source_p1e_report_artifact_id=source_p1e_report.artifact_id,
        public_evidence_contract_sha256=contract.contract_sha256,
        package_name=dataset.package_name,
        dataset_fingerprint=dataset.dataset_fingerprint,
        semantic_auditor_version=contract.semantic_auditor_version,
        semantic_audit_method=contract.semantic_audit_method,
        semantic_similarity=contract.semantic_similarity,
        embedder_name=embedder.name,
        weights_sha256=weights_sha256,
        score_precision_decimals=contract.score_precision_decimals,
        required_evidence_units=contract.required_evidence_units,
        required_top1_accuracy=contract.required_top1_accuracy,
        required_minimum_correct_anchor_margin=(contract.minimum_correct_anchor_margin),
        required_minimum_mean_correct_anchor_margin=(contract.minimum_mean_correct_anchor_margin),
        evidence_units=frozen_units,
        correct_count=correct_count,
        top1_accuracy=accuracy,
        minimum_correct_anchor_margin=minimum_margin,
        mean_correct_anchor_margin=mean_margin,
        human_anchor_status=contract.human_anchor_status,
        verdict=verdict,
    )


def write_relationship_packet1f_report(
    report: RelationshipP1fReport,
    *,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    target = pathlib.Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "packet1f_report.json"
    markdown_path = target / "packet1f_report.md"
    existing = tuple(path for path in (json_path, markdown_path) if path.exists())
    if existing:
        raise FileExistsError(f"P1f report files already exist: {existing}")
    json_path.write_text(report.to_json(), encoding="utf-8")
    history_count = sum(unit.evidence_kind == "history" for unit in report.evidence_units)
    probe_count = len(report.evidence_units) - history_count
    lines = [
        "# Relationship Lab P1f public evidence audit",
        "",
        f"- artifact_id: `{report.artifact_id}`",
        f"- source P1e: `{report.source_p1e_report_artifact_id}`",
        f"- dataset_fingerprint: `{report.dataset_fingerprint}`",
        f"- public evidence contract: `{report.public_evidence_contract_sha256}`",
        f"- auditor: `{report.semantic_auditor_version}` / `{report.semantic_similarity}`",
        f"- evidence units: **{len(report.evidence_units)}** ({history_count} histories + {probe_count} probes)",
        f"- correct anchors: **{report.correct_count}/{len(report.evidence_units)}**",
        f"- top-1 accuracy: **{report.top1_accuracy:.3f}**",
        f"- minimum margin: **{report.minimum_correct_anchor_margin:.6f}**",
        f"- mean margin: **{report.mean_correct_anchor_margin:.6f}**",
        f"- human anchor: **{report.human_anchor_status}**",
        f"- verdict: **{report.verdict.value}**",
        "",
        "## Required next action",
        "",
        report.next_action,
        "",
        "## Claim boundary",
        "",
        _REPORT_CLAIM_BOUNDARY,
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def load_relationship_packet1f_report(path: pathlib.Path) -> RelationshipP1fReport:
    file_path = pathlib.Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return RelationshipP1fReport.from_json(file_path.read_text(encoding="utf-8"))


__all__ = [
    "RELATIONSHIP_P1F_REPORT_SCHEMA_VERSION",
    "RelationshipP1fEvidenceUnit",
    "RelationshipP1fReport",
    "RelationshipP1fVerdict",
    "assess_relationship_packet1f",
    "canonical_relationship_p1f_embedder_name",
    "load_relationship_packet1f_report",
    "write_relationship_packet1f_report",
]
