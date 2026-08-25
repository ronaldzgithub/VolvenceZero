"""Relationship Lab P1L: blinded human-anchor packet for v3 public evidence.

P1f proved a frozen BGE-M3 auditor can separate every public history and probe
from the competing sealed condition summary.  That is not human readability.
P1L freezes a rater-facing packet that shows only public text and two unlabeled
summaries, plus an evaluator-only answer key.  Ratings cannot enter memory, PE,
credit, reward, controller, or steering, and they cannot rewrite the v3
public-evidence contract until a separate scored report says so.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME,
    RelationshipTransferDataset,
    canonical_json,
    sha256_json,
)


RELATIONSHIP_P1L_PROTOCOL_SCHEMA_VERSION = "relationship-p1l-human-anchor-protocol.v1"
RELATIONSHIP_P1L_PACKET_SCHEMA_VERSION = "relationship-p1l-human-anchor-packet.v1"
RELATIONSHIP_P1L_REPORT_SCHEMA_VERSION = "relationship-p1l-human-anchor-report.v1"
RELATIONSHIP_P1L_SHUFFLE_SEED = "relationship-p1l-option-shuffle.v1"
RELATIONSHIP_P1L_MINIMUM_RATERS = 3
RELATIONSHIP_P1L_MINIMUM_MAJORITY_AGREEMENT = 0.8
RELATIONSHIP_P1L_MINIMUM_MAJORITY_ACCURACY = 0.8
RELATIONSHIP_P1L_REQUIRED_UNITS = 60
RELATIONSHIP_P1L_PREPARED_NEXT_ACTION = "collect_three_blinded_independent_ratings"

_HEX_DIGITS = frozenset("0123456789abcdef")
_PROTOCOL_CLAIM_BOUNDARY = (
    "P1L freezes a blinded human-anchor packet over the already-audited v3 "
    "public evidence surface. Raters never see condition ids, preferred "
    "actions, or evaluator keys. The packet is not Qwen transfer evidence, "
    "Readable/Learnable/Steerable evidence, consumer qualification, or a "
    "four-capability claim, and it cannot feed PE, credit, reward, or steering."
)
_REPORT_CLAIM_BOUNDARY = (
    "P1L scores blinded human majority labels against the sealed v3 condition "
    "anchors. A pass only retires the pending human-readability gate for this "
    "development package. It does not rewrite the public-evidence contract in "
    "place, does not qualify a consumer, and does not authorize P2 or learning."
)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in _HEX_DIGITS for char in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 digest")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
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


def _require_object(
    value: object,
    expected_keys: set[str],
    *,
    field_name: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"{field_name} keys do not match the frozen schema")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_text(path: pathlib.Path, content: str) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        temporary = pathlib.Path(handle.name)
    temporary.replace(target)


def _history_text(utterance: str, reaction: str, joiner: str) -> str:
    return joiner.join((utterance, reaction))


def _option_order(unit_id: str) -> tuple[str, str]:
    digest = _sha256_text(f"{RELATIONSHIP_P1L_SHUFFLE_SEED}:{unit_id}")
    if digest[0] in "01234567":
        return ("A", "B")
    return ("B", "A")


@dataclass(frozen=True)
class RelationshipP1lUnit:
    unit_id: str
    evidence_kind: str
    source_id: str
    public_text: str
    option_a: str
    option_b: str
    expected_option: str
    expected_anchor_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.unit_id, "P1L unit id")
        if self.evidence_kind not in {"history", "probe"}:
            raise ValueError("P1L evidence_kind is invalid")
        _require_text(self.source_id, "P1L source id")
        _require_text(self.public_text, "P1L public text")
        _require_text(self.option_a, "P1L option A")
        _require_text(self.option_b, "P1L option B")
        if self.expected_option not in {"A", "B"}:
            raise ValueError("P1L expected option must be A or B")
        _require_sha256(self.expected_anchor_sha256, "P1L expected anchor")
        if self.option_a == self.option_b:
            raise ValueError("P1L options must be distinct")

    def rater_payload(self) -> dict[str, str]:
        return {
            "unit_id": self.unit_id,
            "evidence_kind": self.evidence_kind,
            "source_id": self.source_id,
            "public_text": self.public_text,
            "option_a": self.option_a,
            "option_b": self.option_b,
        }

    def sealed_payload(self) -> dict[str, str]:
        return {
            "unit_id": self.unit_id,
            "public_text_sha256": _sha256_text(self.public_text),
            "expected_option": self.expected_option,
            "expected_anchor_sha256": self.expected_anchor_sha256,
        }


def build_relationship_p1l_units(
    dataset: RelationshipTransferDataset,
) -> tuple[RelationshipP1lUnit, ...]:
    if dataset.package_name != RELATIONSHIP_TRANSFER_V3_PACKAGE_NAME:
        raise ValueError("P1L requires relationship_transfer_v3")
    contract = dataset.public_evidence_contract
    if contract is None:
        raise ValueError("P1L dataset has no public evidence contract")
    conditions = tuple(
        sorted(dataset.abstract_conditions, key=lambda item: item.condition_id)
    )
    if len(conditions) != 2:
        raise ValueError("P1L requires exactly two sealed condition summaries")
    summaries = {item.condition_id: item.hidden_summary for item in conditions}
    anchors = {
        item.condition_id: _sha256_text(item.hidden_summary) for item in conditions
    }
    bindings = dict(dataset.history_condition_bindings)
    units: list[RelationshipP1lUnit] = []
    for observation in sorted(dataset.observations, key=lambda item: item.scene_id):
        for history in observation.histories:
            expected_id = bindings[history.event_id]
            units.append(
                _unit(
                    evidence_kind="history",
                    source_id=history.event_id,
                    public_text=_history_text(
                        history.user_utterance,
                        history.user_reaction,
                        contract.history_text_joiner,
                    ),
                    expected_condition_id=expected_id,
                    summaries=summaries,
                    anchors=anchors,
                )
            )
        dynamic = dataset.dynamic_for_scene(observation.scene_id)
        if dynamic.probe_condition_id is None:
            raise ValueError("P1L probe has no sealed condition binding")
        units.append(
            _unit(
                evidence_kind="probe",
                source_id=observation.scene_id,
                public_text=observation.current_input,
                expected_condition_id=dynamic.probe_condition_id,
                summaries=summaries,
                anchors=anchors,
            )
        )
    if len(units) != RELATIONSHIP_P1L_REQUIRED_UNITS:
        raise ValueError("P1L unit count diverges from the public-evidence contract")
    unit_ids = tuple(item.unit_id for item in units)
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("P1L unit ids must be unique")
    return tuple(units)


def _unit(
    *,
    evidence_kind: str,
    source_id: str,
    public_text: str,
    expected_condition_id: str,
    summaries: dict[str, str],
    anchors: dict[str, str],
) -> RelationshipP1lUnit:
    unit_id = f"{evidence_kind}:{source_id}"
    ordered_ids = tuple(sorted(summaries))
    labels = _option_order(unit_id)
    option_by_label = {
        labels[0]: summaries[ordered_ids[0]],
        labels[1]: summaries[ordered_ids[1]],
    }
    expected_label = next(
        label
        for label, summary in option_by_label.items()
        if summary == summaries[expected_condition_id]
    )
    return RelationshipP1lUnit(
        unit_id=unit_id,
        evidence_kind=evidence_kind,
        source_id=source_id,
        public_text=public_text,
        option_a=option_by_label["A"],
        option_b=option_by_label["B"],
        expected_option=expected_label,
        expected_anchor_sha256=anchors[expected_condition_id],
    )


@dataclass(frozen=True)
class RelationshipP1lProtocol:
    frozen_at_iso: str
    training_dataset_fingerprint: str
    public_evidence_contract_sha256: str
    shuffle_seed: str
    required_units: int
    minimum_independent_raters: int
    minimum_majority_agreement: float
    minimum_majority_accuracy: float
    rater_packet_sha256: str
    sealed_key_sha256: str
    claim_boundary: str = _PROTOCOL_CLAIM_BOUNDARY
    next_action: str = RELATIONSHIP_P1L_PREPARED_NEXT_ACTION
    schema_version: str = RELATIONSHIP_P1L_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1L_PROTOCOL_SCHEMA_VERSION:
            raise ValueError("P1L protocol schema_version mismatch")
        _require_timestamp(self.frozen_at_iso, "P1L protocol timestamp")
        _require_sha256(
            self.training_dataset_fingerprint, "P1L dataset fingerprint"
        )
        _require_sha256(
            self.public_evidence_contract_sha256, "P1L public evidence contract"
        )
        _require_sha256(self.rater_packet_sha256, "P1L rater packet hash")
        _require_sha256(self.sealed_key_sha256, "P1L sealed key hash")
        if self.shuffle_seed != RELATIONSHIP_P1L_SHUFFLE_SEED:
            raise ValueError("P1L shuffle seed is not frozen")
        if self.required_units != RELATIONSHIP_P1L_REQUIRED_UNITS:
            raise ValueError("P1L required unit count is not frozen")
        if self.minimum_independent_raters != RELATIONSHIP_P1L_MINIMUM_RATERS:
            raise ValueError("P1L rater count is not frozen")
        if self.minimum_majority_agreement != RELATIONSHIP_P1L_MINIMUM_MAJORITY_AGREEMENT:
            raise ValueError("P1L majority-agreement threshold is not frozen")
        if self.minimum_majority_accuracy != RELATIONSHIP_P1L_MINIMUM_MAJORITY_ACCURACY:
            raise ValueError("P1L majority-accuracy threshold is not frozen")
        if self.claim_boundary != _PROTOCOL_CLAIM_BOUNDARY:
            raise ValueError("P1L protocol claim boundary is not frozen")
        if self.next_action != RELATIONSHIP_P1L_PREPARED_NEXT_ACTION:
            raise ValueError("P1L protocol next action is not frozen")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frozen_at_iso": self.frozen_at_iso,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "public_evidence_contract_sha256": self.public_evidence_contract_sha256,
            "shuffle_seed": self.shuffle_seed,
            "required_units": self.required_units,
            "minimum_independent_raters": self.minimum_independent_raters,
            "minimum_majority_agreement": self.minimum_majority_agreement,
            "minimum_majority_accuracy": self.minimum_majority_accuracy,
            "rater_packet_sha256": self.rater_packet_sha256,
            "sealed_key_sha256": self.sealed_key_sha256,
            "experiment_guards": {
                "labels_available_to_raters": False,
                "evaluation_feedback_to_pe_credit_reward_or_steering": False,
                "rewrites_public_evidence_contract_in_place": False,
                "p2_enabled": False,
            },
            "claim_boundary": self.claim_boundary,
            "next_action": self.next_action,
        }

    @property
    def protocol_id(self) -> str:
        return sha256_json(self.to_payload())


def _rater_packet_bytes(units: tuple[RelationshipP1lUnit, ...]) -> str:
    return canonical_json([item.rater_payload() for item in units]) + "\n"


def _sealed_key_bytes(units: tuple[RelationshipP1lUnit, ...]) -> str:
    return canonical_json([item.sealed_payload() for item in units]) + "\n"


def freeze_relationship_p1l_protocol(
    *,
    dataset: RelationshipTransferDataset,
    frozen_at_iso: str | None = None,
) -> tuple[RelationshipP1lProtocol, tuple[RelationshipP1lUnit, ...]]:
    units = build_relationship_p1l_units(dataset)
    contract = dataset.public_evidence_contract
    if contract is None:
        raise ValueError("P1L dataset has no public evidence contract")
    protocol = RelationshipP1lProtocol(
        frozen_at_iso=frozen_at_iso or datetime.now(timezone.utc).isoformat(),
        training_dataset_fingerprint=dataset.dataset_fingerprint,
        public_evidence_contract_sha256=contract.contract_sha256,
        shuffle_seed=RELATIONSHIP_P1L_SHUFFLE_SEED,
        required_units=RELATIONSHIP_P1L_REQUIRED_UNITS,
        minimum_independent_raters=RELATIONSHIP_P1L_MINIMUM_RATERS,
        minimum_majority_agreement=RELATIONSHIP_P1L_MINIMUM_MAJORITY_AGREEMENT,
        minimum_majority_accuracy=RELATIONSHIP_P1L_MINIMUM_MAJORITY_ACCURACY,
        rater_packet_sha256=_sha256_text(_rater_packet_bytes(units)),
        sealed_key_sha256=_sha256_text(_sealed_key_bytes(units)),
    )
    return protocol, units


def write_relationship_p1l_packet(
    *,
    protocol: RelationshipP1lProtocol,
    units: tuple[RelationshipP1lUnit, ...],
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    target = pathlib.Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    protocol_path = target / "packet1l_protocol.json"
    rater_json = target / "rater_packet.json"
    rater_csv = target / "rater_packet.csv"
    sealed_key = target / "sealed_answer_key.json"
    existing = tuple(
        path
        for path in (protocol_path, rater_json, rater_csv, sealed_key)
        if path.exists()
    )
    if existing:
        raise FileExistsError(f"P1L packet already exists: {existing}")
    protocol_payload = protocol.to_payload()
    protocol_payload["protocol_id"] = protocol.protocol_id
    _atomic_write_text(
        protocol_path,
        json.dumps(protocol_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    rater_bytes = _rater_packet_bytes(units)
    sealed_bytes = _sealed_key_bytes(units)
    if _sha256_text(rater_bytes) != protocol.rater_packet_sha256:
        raise RuntimeError("P1L rater packet hash mismatch")
    if _sha256_text(sealed_bytes) != protocol.sealed_key_sha256:
        raise RuntimeError("P1L sealed key hash mismatch")
    _atomic_write_text(rater_json, rater_bytes)
    _atomic_write_text(sealed_key, sealed_bytes)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "rater_id",
            "unit_id",
            "evidence_kind",
            "source_id",
            "public_text",
            "option_a",
            "option_b",
            "chosen_option",
        ),
    )
    writer.writeheader()
    for unit in units:
        writer.writerow({**unit.rater_payload(), "rater_id": "", "chosen_option": ""})
    _atomic_write_text(rater_csv, buffer.getvalue())
    return protocol_path, rater_json, rater_csv, sealed_key


def load_relationship_p1l_protocol(path: pathlib.Path) -> RelationshipP1lProtocol:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("P1L protocol must be a JSON object")
    protocol_id = _require_sha256(raw.pop("protocol_id"), "P1L protocol id")
    guards = raw.pop("experiment_guards")
    if any(
        _require_bool(value, f"P1L guard {name}") for name, value in guards.items()
    ):
        raise ValueError("P1L protocol guards must all be false")
    protocol = RelationshipP1lProtocol(
        schema_version=_require_text(raw["schema_version"], "P1L protocol schema"),
        frozen_at_iso=raw["frozen_at_iso"],
        training_dataset_fingerprint=raw["training_dataset_fingerprint"],
        public_evidence_contract_sha256=raw["public_evidence_contract_sha256"],
        shuffle_seed=raw["shuffle_seed"],
        required_units=_require_int(raw["required_units"], "P1L required units"),
        minimum_independent_raters=_require_int(
            raw["minimum_independent_raters"], "P1L raters"
        ),
        minimum_majority_agreement=_require_number(
            raw["minimum_majority_agreement"], "P1L agreement"
        ),
        minimum_majority_accuracy=_require_number(
            raw["minimum_majority_accuracy"], "P1L accuracy"
        ),
        rater_packet_sha256=raw["rater_packet_sha256"],
        sealed_key_sha256=raw["sealed_key_sha256"],
        claim_boundary=raw["claim_boundary"],
        next_action=raw["next_action"],
    )
    if protocol.protocol_id != protocol_id:
        raise ValueError("P1L protocol id mismatch")
    return protocol


def load_relationship_p1l_sealed_key(path: pathlib.Path) -> tuple[dict[str, str], ...]:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("P1L sealed key must be a JSON array")
    rows = []
    for item in raw:
        parsed = _require_object(
            item,
            {
                "unit_id",
                "public_text_sha256",
                "expected_option",
                "expected_anchor_sha256",
            },
            field_name="P1L sealed key row",
        )
        rows.append(
            {
                "unit_id": _require_text(parsed["unit_id"], "P1L key unit"),
                "public_text_sha256": _require_sha256(
                    parsed["public_text_sha256"], "P1L key text hash"
                ),
                "expected_option": _require_text(
                    parsed["expected_option"], "P1L key option"
                ),
                "expected_anchor_sha256": _require_sha256(
                    parsed["expected_anchor_sha256"], "P1L key anchor"
                ),
            }
        )
    return tuple(rows)


class RelationshipP1lVerdict(str, Enum):
    RATINGS_PENDING = "human_anchor_ratings_pending"
    PASSED = "human_anchor_passed_development"
    FAILED = "human_anchor_failed_development"


_NEXT_ACTIONS = {
    RelationshipP1lVerdict.RATINGS_PENDING: RELATIONSHIP_P1L_PREPARED_NEXT_ACTION,
    RelationshipP1lVerdict.PASSED: "retire_pending_human_anchor_status_in_a_new_contract_revision",
    RelationshipP1lVerdict.FAILED: "rewrite_public_language_before_claiming_human_readability",
}


@dataclass(frozen=True)
class RelationshipP1lRating:
    rater_id: str
    unit_id: str
    chosen_option: str

    def __post_init__(self) -> None:
        _require_text(self.rater_id, "P1L rater id")
        _require_text(self.unit_id, "P1L rating unit")
        if self.chosen_option not in {"A", "B"}:
            raise ValueError("P1L chosen option must be A or B")


def load_relationship_p1l_ratings(path: pathlib.Path) -> tuple[RelationshipP1lRating, ...]:
    file_path = pathlib.Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix == ".json":
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise ValueError("P1L ratings JSON must be an array")
        return tuple(
            RelationshipP1lRating(
                rater_id=_require_text(item["rater_id"], "P1L rating rater"),
                unit_id=_require_text(item["unit_id"], "P1L rating unit"),
                chosen_option=_require_text(item["chosen_option"], "P1L rating option"),
            )
            for item in raw
        )
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for item in reader:
        chosen = item["chosen_option"].strip()
        if not chosen:
            continue
        rows.append(
            RelationshipP1lRating(
                rater_id=_require_text(item.get("rater_id", ""), "P1L rating rater")
                if item.get("rater_id")
                else _require_text(file_path.stem, "P1L rating rater from filename"),
                unit_id=_require_text(item["unit_id"], "P1L rating unit"),
                chosen_option=_require_text(chosen, "P1L rating option"),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class RelationshipP1lReport:
    created_at_iso: str
    protocol_id: str
    training_dataset_fingerprint: str
    public_evidence_contract_sha256: str
    rater_count: int
    unit_count: int
    majority_agreement: float
    majority_accuracy: float
    verdict: RelationshipP1lVerdict
    claim_boundary: str = _REPORT_CLAIM_BOUNDARY
    schema_version: str = RELATIONSHIP_P1L_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RELATIONSHIP_P1L_REPORT_SCHEMA_VERSION:
            raise ValueError("P1L report schema_version mismatch")
        _require_timestamp(self.created_at_iso, "P1L report timestamp")
        _require_sha256(self.protocol_id, "P1L report protocol id")
        _require_sha256(
            self.training_dataset_fingerprint, "P1L report dataset fingerprint"
        )
        _require_sha256(
            self.public_evidence_contract_sha256, "P1L report contract"
        )
        if self.unit_count != RELATIONSHIP_P1L_REQUIRED_UNITS:
            raise ValueError("P1L report unit count is not frozen")
        if self.claim_boundary != _REPORT_CLAIM_BOUNDARY:
            raise ValueError("P1L report claim boundary is not frozen")

    @property
    def next_action(self) -> str:
        return _NEXT_ACTIONS[self.verdict]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_iso": self.created_at_iso,
            "protocol_id": self.protocol_id,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "public_evidence_contract_sha256": self.public_evidence_contract_sha256,
            "rater_count": self.rater_count,
            "unit_count": self.unit_count,
            "majority_agreement": self.majority_agreement,
            "majority_accuracy": self.majority_accuracy,
            "verdict": self.verdict.value,
            "next_action": self.next_action,
            "experiment_guards": {
                "evaluation_feedback_to_pe_credit_reward_or_steering": False,
                "rewrites_public_evidence_contract_in_place": False,
                "p2_enabled": False,
            },
            "claim_boundary": self.claim_boundary,
        }

    @property
    def artifact_id(self) -> str:
        return sha256_json(self.to_payload())


def assess_relationship_p1l_ratings(
    *,
    protocol: RelationshipP1lProtocol,
    units: tuple[RelationshipP1lUnit, ...],
    ratings: tuple[RelationshipP1lRating, ...],
    created_at_iso: str | None = None,
) -> RelationshipP1lReport:
    if _sha256_text(_rater_packet_bytes(units)) != protocol.rater_packet_sha256:
        raise ValueError("P1L units diverge from the frozen rater packet")
    if not ratings:
        return RelationshipP1lReport(
            created_at_iso=created_at_iso or datetime.now(timezone.utc).isoformat(),
            protocol_id=protocol.protocol_id,
            training_dataset_fingerprint=protocol.training_dataset_fingerprint,
            public_evidence_contract_sha256=protocol.public_evidence_contract_sha256,
            rater_count=0,
            unit_count=len(units),
            majority_agreement=0.0,
            majority_accuracy=0.0,
            verdict=RelationshipP1lVerdict.RATINGS_PENDING,
        )
    raters = tuple(sorted({item.rater_id for item in ratings}))
    if len(raters) < protocol.minimum_independent_raters:
        return RelationshipP1lReport(
            created_at_iso=created_at_iso or datetime.now(timezone.utc).isoformat(),
            protocol_id=protocol.protocol_id,
            training_dataset_fingerprint=protocol.training_dataset_fingerprint,
            public_evidence_contract_sha256=protocol.public_evidence_contract_sha256,
            rater_count=len(raters),
            unit_count=len(units),
            majority_agreement=0.0,
            majority_accuracy=0.0,
            verdict=RelationshipP1lVerdict.RATINGS_PENDING,
        )
    expected = {item.unit_id: item.expected_option for item in units}
    by_unit: dict[str, list[str]] = {item.unit_id: [] for item in units}
    seen: set[tuple[str, str]] = set()
    for rating in ratings:
        identity = (rating.rater_id, rating.unit_id)
        if identity in seen:
            raise ValueError("P1L duplicate rater/unit rating")
        seen.add(identity)
        if rating.unit_id not in by_unit:
            raise ValueError("P1L rating references an unknown unit")
        by_unit[rating.unit_id].append(rating.chosen_option)
    agreed = 0
    correct = 0
    for unit_id, choices in by_unit.items():
        if len(choices) < protocol.minimum_independent_raters:
            raise ValueError("P1L unit is missing a complete rater set")
        counts = {"A": choices.count("A"), "B": choices.count("B")}
        majority = "A" if counts["A"] > counts["B"] else "B" if counts["B"] > counts["A"] else None
        if majority is not None:
            agreed += 1
            correct += int(majority == expected[unit_id])
    agreement = agreed / len(units)
    accuracy = correct / len(units)
    passed = (
        agreement >= protocol.minimum_majority_agreement
        and accuracy >= protocol.minimum_majority_accuracy
    )
    return RelationshipP1lReport(
        created_at_iso=created_at_iso or datetime.now(timezone.utc).isoformat(),
        protocol_id=protocol.protocol_id,
        training_dataset_fingerprint=protocol.training_dataset_fingerprint,
        public_evidence_contract_sha256=protocol.public_evidence_contract_sha256,
        rater_count=len(raters),
        unit_count=len(units),
        majority_agreement=agreement,
        majority_accuracy=accuracy,
        verdict=(
            RelationshipP1lVerdict.PASSED if passed else RelationshipP1lVerdict.FAILED
        ),
    )


def write_relationship_p1l_report(
    *,
    report: RelationshipP1lReport,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    path = pathlib.Path(output_dir) / "packet1l_report.json"
    if path.exists():
        raise FileExistsError(f"P1L report already exists: {path}")
    payload = report.to_payload()
    payload["artifact_id"] = report.artifact_id
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return path


__all__ = [
    "RELATIONSHIP_P1L_MINIMUM_MAJORITY_ACCURACY",
    "RELATIONSHIP_P1L_MINIMUM_MAJORITY_AGREEMENT",
    "RELATIONSHIP_P1L_MINIMUM_RATERS",
    "RELATIONSHIP_P1L_PREPARED_NEXT_ACTION",
    "RELATIONSHIP_P1L_REQUIRED_UNITS",
    "RelationshipP1lProtocol",
    "RelationshipP1lRating",
    "RelationshipP1lReport",
    "RelationshipP1lUnit",
    "RelationshipP1lVerdict",
    "assess_relationship_p1l_ratings",
    "build_relationship_p1l_units",
    "freeze_relationship_p1l_protocol",
    "load_relationship_p1l_protocol",
    "load_relationship_p1l_ratings",
    "load_relationship_p1l_sealed_key",
    "write_relationship_p1l_packet",
    "write_relationship_p1l_report",
]
