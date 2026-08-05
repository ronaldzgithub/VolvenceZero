"""Validation-only expert anchor for dialogue steering decisions.

The C2 anchor is deliberately outside the learning DAG.  It consumes
already-deidentified, explicitly consented material and a private key derived
from the PE owner's terminal settlement.  Human ratings are readouts: this
module cannot authorize credit, policy updates, or production promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Mapping, Sequence

from volvence_zero.steering_contracts import SteeringTerminalPredictionError


STEERING_HUMAN_ANCHOR_SCHEMA_VERSION = "steering-human-anchor-pilot.v1"
STEERING_HUMAN_ANCHOR_CONSENT_SCOPE = (
    "steering-human-anchor-validation-only.v1"
)
STEERING_HUMAN_ANCHOR_PILOT_UNITS = 48
STEERING_HUMAN_ANCHOR_RATERS_PER_UNIT = 2
STEERING_HUMAN_ANCHOR_MIN_EXACT_AGREEMENT = 0.75
STEERING_HUMAN_ANCHOR_MIN_COHEN_KAPPA = 0.60
STEERING_HUMAN_ANCHOR_MAX_UNRATABLE_RATE = 0.10
STEERING_HUMAN_ANCHOR_MIN_RESOLVED_DIRECTIONS = 24
STEERING_HUMAN_ANCHOR_ALIGNMENT_REVIEW_FLOOR = 0.60
STEERING_HUMAN_ANCHOR_PE_DEADZONE = 0.02


def _require_nonempty(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _require_sha256(value: str, *, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256 digest") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"unsupported canonical JSON value {type(value).__name__}")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
        + b"\n"
    )


class SteeringAnchorArm(str, Enum):
    A = "a"
    B = "b"


class SteeringAnchorPreference(str, Enum):
    A = "a"
    B = "b"
    TIE = "tie"
    UNRATABLE = "unratable"


class SteeringAnchorUnratableReason(str, Enum):
    NONE = "none"
    MISSING_CONTEXT = "missing-context"
    MATERIAL_CORRUPTION = "material-corruption"
    PRIVACY_CONCERN = "privacy-concern"


@dataclass(frozen=True)
class SteeringAnchorProtocol:
    schema_version: str = STEERING_HUMAN_ANCHOR_SCHEMA_VERSION
    pilot_units: int = STEERING_HUMAN_ANCHOR_PILOT_UNITS
    raters_per_unit: int = STEERING_HUMAN_ANCHOR_RATERS_PER_UNIT
    minimum_exact_agreement: float = (
        STEERING_HUMAN_ANCHOR_MIN_EXACT_AGREEMENT
    )
    minimum_cohen_kappa: float = STEERING_HUMAN_ANCHOR_MIN_COHEN_KAPPA
    maximum_unratable_rate: float = (
        STEERING_HUMAN_ANCHOR_MAX_UNRATABLE_RATE
    )
    minimum_resolved_directions: int = (
        STEERING_HUMAN_ANCHOR_MIN_RESOLVED_DIRECTIONS
    )
    pe_deadzone: float = STEERING_HUMAN_ANCHOR_PE_DEADZONE
    raw_source_retention_days: int = 30
    deidentified_packet_retention_days: int = 180
    audit_tombstone_retention_days: int = 730
    validation_anchor_only: bool = True
    learning_use_authorized: bool = False
    production_promotion_authorized: bool = False


@dataclass(frozen=True)
class SteeringAnchorConsentAttestation:
    consent_record_sha256: str
    consent_document_sha256: str
    subject_ref_sha256: str
    scope: str
    active_at_capture: bool
    learning_use_authorized: bool
    withdrawal_channel_sha256: str
    retention_deadline_unix_ms: int

    def __post_init__(self) -> None:
        for field in (
            "consent_record_sha256",
            "consent_document_sha256",
            "subject_ref_sha256",
            "withdrawal_channel_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.scope != STEERING_HUMAN_ANCHOR_CONSENT_SCOPE:
            raise ValueError("consent scope does not authorize the C2 anchor")
        if not self.active_at_capture:
            raise ValueError("consent must be active at capture")
        if self.learning_use_authorized:
            raise ValueError("C2 consent must not authorize learning use")
        if self.retention_deadline_unix_ms <= 0:
            raise ValueError("retention deadline must be positive")


@dataclass(frozen=True)
class SteeringAnchorPrivacyAttestation:
    review_artifact_sha256: str
    reviewer_role: str
    human_review_completed: bool
    direct_identifiers_removed: bool
    quasi_identifiers_generalized: bool
    third_party_content_cleared: bool
    raw_source_excluded: bool

    def __post_init__(self) -> None:
        _require_sha256(
            self.review_artifact_sha256,
            field="review_artifact_sha256",
        )
        _require_nonempty(self.reviewer_role, field="reviewer_role")
        required = (
            self.human_review_completed,
            self.direct_identifiers_removed,
            self.quasi_identifiers_generalized,
            self.third_party_content_cleared,
            self.raw_source_excluded,
        )
        if not all(required):
            raise ValueError(
                "all typed deidentification attestations must be true"
            )


@dataclass(frozen=True)
class SteeringAnchorTurn:
    speaker: str
    text: str

    def __post_init__(self) -> None:
        if self.speaker not in {"user", "assistant"}:
            raise ValueError("turn speaker must be user or assistant")
        _require_nonempty(self.text, field="turn.text")


@dataclass(frozen=True)
class SteeringAnchorCaptureUnit:
    unit_id: str
    episode_ref_sha256: str
    decision_ref_sha256: str
    capture_source_sha256: str
    captured_at_unix_ms: int
    context_turns: tuple[SteeringAnchorTurn, ...]
    arm_a_response: str
    arm_b_response: str
    consent: SteeringAnchorConsentAttestation
    privacy: SteeringAnchorPrivacyAttestation

    def __post_init__(self) -> None:
        _require_sha256(self.unit_id, field="unit_id")
        for field in (
            "episode_ref_sha256",
            "decision_ref_sha256",
            "capture_source_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.captured_at_unix_ms <= 0:
            raise ValueError("captured_at_unix_ms must be positive")
        if not self.context_turns:
            raise ValueError("anchor capture requires deidentified context")
        _require_nonempty(self.arm_a_response, field="arm_a_response")
        _require_nonempty(self.arm_b_response, field="arm_b_response")
        if self.arm_a_response == self.arm_b_response:
            raise ValueError("anchor comparison arms must differ")
        if self.consent.retention_deadline_unix_ms <= self.captured_at_unix_ms:
            raise ValueError("consent retention deadline must follow capture")


@dataclass(frozen=True)
class SteeringAnchorInternalKeyEntry:
    unit_id: str
    decision_ref_sha256: str
    steered_arm: SteeringAnchorArm
    terminal_pe_sha256: str
    relative_mse_improvement: float
    policy_version: int

    def __post_init__(self) -> None:
        _require_sha256(self.unit_id, field="unit_id")
        _require_sha256(
            self.decision_ref_sha256,
            field="decision_ref_sha256",
        )
        _require_sha256(self.terminal_pe_sha256, field="terminal_pe_sha256")
        if not isinstance(self.steered_arm, SteeringAnchorArm):
            raise ValueError("steered_arm must be SteeringAnchorArm")
        if (
            not math.isfinite(self.relative_mse_improvement)
            or not -1.0 <= self.relative_mse_improvement <= 1.0
        ):
            raise ValueError("relative_mse_improvement must be within [-1, 1]")
        if self.policy_version < 1:
            raise ValueError("policy_version must be positive")

    @classmethod
    def from_terminal_prediction_error(
        cls,
        *,
        unit_id: str,
        decision_id: str,
        steered_arm: SteeringAnchorArm,
        policy_version: int,
        settlement: SteeringTerminalPredictionError,
    ) -> SteeringAnchorInternalKeyEntry:
        if decision_id not in settlement.decision_ids:
            raise ValueError("decision is not bound to terminal PE settlement")
        return cls(
            unit_id=unit_id,
            decision_ref_sha256=_sha256_text(decision_id),
            steered_arm=steered_arm,
            terminal_pe_sha256=hashlib.sha256(
                _canonical_bytes(asdict(settlement))
            ).hexdigest(),
            relative_mse_improvement=settlement.relative_mse_improvement,
            policy_version=policy_version,
        )


@dataclass(frozen=True)
class SteeringAnchorRaterAttestation:
    rater_id: str
    eligibility_review_sha256: str
    human_expert_attested: bool
    domain_expertise_attested: bool
    independent_from_policy_training_attested: bool

    def __post_init__(self) -> None:
        _require_nonempty(self.rater_id, field="rater_id")
        _require_sha256(
            self.eligibility_review_sha256,
            field="eligibility_review_sha256",
        )
        if not all(
            (
                self.human_expert_attested,
                self.domain_expertise_attested,
                self.independent_from_policy_training_attested,
            )
        ):
            raise ValueError("rater must satisfy all typed eligibility checks")


@dataclass(frozen=True)
class SteeringAnchorRating:
    unit_id: str
    rater_id: str
    eligibility_review_sha256: str
    preferred_arm: SteeringAnchorPreference
    arm_a_relationship_support: int | None
    arm_b_relationship_support: int | None
    arm_a_boundary_respect: int | None
    arm_b_boundary_respect: int | None
    arm_a_task_preservation: int | None
    arm_b_task_preservation: int | None
    unratable_reason: SteeringAnchorUnratableReason

    def __post_init__(self) -> None:
        _require_sha256(self.unit_id, field="unit_id")
        _require_nonempty(self.rater_id, field="rater_id")
        _require_sha256(
            self.eligibility_review_sha256,
            field="eligibility_review_sha256",
        )
        if not isinstance(self.preferred_arm, SteeringAnchorPreference):
            raise ValueError("preferred_arm must be SteeringAnchorPreference")
        if not isinstance(
            self.unratable_reason,
            SteeringAnchorUnratableReason,
        ):
            raise ValueError(
                "unratable_reason must be SteeringAnchorUnratableReason"
            )
        scores = (
            self.arm_a_relationship_support,
            self.arm_b_relationship_support,
            self.arm_a_boundary_respect,
            self.arm_b_boundary_respect,
            self.arm_a_task_preservation,
            self.arm_b_task_preservation,
        )
        if self.preferred_arm is SteeringAnchorPreference.UNRATABLE:
            if self.unratable_reason is SteeringAnchorUnratableReason.NONE:
                raise ValueError("unratable ratings require a typed reason")
            if any(score is not None for score in scores):
                raise ValueError("unratable ratings must not fabricate scores")
            return
        if self.unratable_reason is not SteeringAnchorUnratableReason.NONE:
            raise ValueError("ratable rows must use unratable_reason=none")
        if any(
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 1 <= score <= 5
            for score in scores
        ):
            raise ValueError("ratable dimension scores must be integers 1..5")


@dataclass(frozen=True)
class SteeringAnchorWithdrawal:
    consent_record_sha256: str
    requested_at_unix_ms: int
    completed_at_unix_ms: int
    raw_source_deleted: bool
    capture_removed: bool
    ratings_removed: bool
    reidentification_mapping_deleted: bool

    def __post_init__(self) -> None:
        _require_sha256(
            self.consent_record_sha256,
            field="consent_record_sha256",
        )
        if self.requested_at_unix_ms <= 0:
            raise ValueError("withdrawal request timestamp must be positive")
        if self.completed_at_unix_ms < self.requested_at_unix_ms:
            raise ValueError("withdrawal completion precedes request")
        if not all(
            (
                self.raw_source_deleted,
                self.capture_removed,
                self.ratings_removed,
                self.reidentification_mapping_deleted,
            )
        ):
            raise ValueError("withdrawal must complete every deletion surface")


@dataclass(frozen=True)
class SteeringAnchorPacketBundle:
    public_packet: Mapping[str, object]
    internal_key: Mapping[str, object]
    rating_template: tuple[Mapping[str, object], ...]
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class SteeringAnchorPilotReport:
    unit_count: int
    rating_count: int
    exact_agreement: float
    cohen_kappa: float | None
    unratable_rate: float
    agreement_gate_passed: bool
    resolved_direction_count: int
    directional_alignment_rate: float | None
    c1_alignment_review_required: bool
    expansion_admissible: bool
    learning_use_authorized: bool
    production_promotion_authorized: bool
    description: str


def build_steering_human_anchor_protocol() -> SteeringAnchorProtocol:
    return SteeringAnchorProtocol()


def apply_steering_anchor_withdrawals(
    *,
    captures: Sequence[SteeringAnchorCaptureUnit],
    withdrawals: Sequence[SteeringAnchorWithdrawal],
) -> tuple[
    tuple[SteeringAnchorCaptureUnit, ...],
    tuple[Mapping[str, object], ...],
]:
    """Remove every unit covered by completed withdrawals.

    The returned tombstones contain only digests/timestamps, never transcript
    text or a reversible identity mapping.  Packet/rating artifacts must be
    rebuilt from the returned capture set.
    """

    withdrawn = {
        withdrawal.consent_record_sha256: withdrawal
        for withdrawal in withdrawals
    }
    if len(withdrawn) != len(withdrawals):
        raise ValueError("duplicate withdrawal consent record")
    retained = tuple(
        capture
        for capture in captures
        if capture.consent.consent_record_sha256 not in withdrawn
    )
    tombstones = tuple(
        {
            "consent_record_sha256": consent_record_sha256,
            "completed_at_unix_ms": withdrawal.completed_at_unix_ms,
            "content_retained": False,
            "reidentification_mapping_retained": False,
        }
        for consent_record_sha256, withdrawal in sorted(withdrawn.items())
    )
    return retained, tombstones


def build_steering_human_anchor_pilot_packet(
    *,
    captures: Sequence[SteeringAnchorCaptureUnit],
    internal_keys: Sequence[SteeringAnchorInternalKeyEntry],
    created_at_unix_ms: int,
) -> SteeringAnchorPacketBundle:
    protocol = build_steering_human_anchor_protocol()
    if created_at_unix_ms <= 0:
        raise ValueError("created_at_unix_ms must be positive")
    if len(captures) != protocol.pilot_units:
        raise ValueError(
            f"C2 pilot requires exactly {protocol.pilot_units} units"
        )
    capture_by_id = {capture.unit_id: capture for capture in captures}
    key_by_id = {entry.unit_id: entry for entry in internal_keys}
    if len(capture_by_id) != len(captures):
        raise ValueError("duplicate steering anchor capture unit")
    if len(key_by_id) != len(internal_keys):
        raise ValueError("duplicate steering anchor key entry")
    if set(capture_by_id) != set(key_by_id):
        raise ValueError("capture/internal-key unit coverage drift")
    for unit_id, capture in capture_by_id.items():
        key = key_by_id[unit_id]
        if capture.decision_ref_sha256 != key.decision_ref_sha256:
            raise ValueError("capture/internal-key decision lineage drift")
        if capture.consent.retention_deadline_unix_ms <= created_at_unix_ms:
            raise ValueError("pilot packet would outlive consent retention")

    public_units = [
        {
            "unit_id": capture.unit_id,
            "context_turns": [
                {"speaker": turn.speaker, "text": turn.text}
                for turn in capture.context_turns
            ],
            "arm_a_response": capture.arm_a_response,
            "arm_b_response": capture.arm_b_response,
            "rating_dimensions": [
                "relationship_support",
                "boundary_respect",
                "task_preservation",
            ],
            "preferred_arm_values": ["a", "b", "tie", "unratable"],
        }
        for capture in sorted(captures, key=lambda item: item.unit_id)
    ]
    private_entries = [
        {
            "unit_id": entry.unit_id,
            "decision_ref_sha256": entry.decision_ref_sha256,
            "steered_arm": entry.steered_arm.value,
            "terminal_pe_sha256": entry.terminal_pe_sha256,
            "relative_mse_improvement": entry.relative_mse_improvement,
            "policy_version": entry.policy_version,
        }
        for entry in sorted(internal_keys, key=lambda item: item.unit_id)
    ]
    public_packet: dict[str, object] = {
        "schema_version": protocol.schema_version,
        "created_at_unix_ms": created_at_unix_ms,
        "pilot_only": True,
        "validation_anchor_only": True,
        "learning_use_authorized": False,
        "production_promotion_authorized": False,
        "unit_count": len(public_units),
        "units": public_units,
    }
    internal_key: dict[str, object] = {
        "schema_version": protocol.schema_version,
        "created_at_unix_ms": created_at_unix_ms,
        "do_not_distribute_to_raters": True,
        "entries": private_entries,
    }
    rating_template = tuple(
        {
            "rater_slot": rater_slot,
            "unit_id": capture.unit_id,
            "preferred_arm": "",
            "arm_a_relationship_support": "",
            "arm_b_relationship_support": "",
            "arm_a_boundary_respect": "",
            "arm_b_boundary_respect": "",
            "arm_a_task_preservation": "",
            "arm_b_task_preservation": "",
            "unratable_reason": "none",
        }
        for capture in sorted(captures, key=lambda item: item.unit_id)
        for rater_slot in range(1, protocol.raters_per_unit + 1)
    )
    hashes = {
        "public_packet_sha256": hashlib.sha256(
            _canonical_bytes(public_packet)
        ).hexdigest(),
        "internal_key_sha256": hashlib.sha256(
            _canonical_bytes(internal_key)
        ).hexdigest(),
        "rating_template_sha256": hashlib.sha256(
            _canonical_bytes(rating_template)
        ).hexdigest(),
    }
    manifest: dict[str, object] = {
        "schema_version": protocol.schema_version,
        "created_at_unix_ms": created_at_unix_ms,
        "hashes": hashes,
        "pilot_only": True,
        "validation_anchor_only": True,
        "learning_use_authorized": False,
        "production_promotion_authorized": False,
        "raw_transcript_in_manifest": False,
    }
    return SteeringAnchorPacketBundle(
        public_packet=public_packet,
        internal_key=internal_key,
        rating_template=rating_template,
        manifest=manifest,
    )


def _verify_packet_bundle(bundle: SteeringAnchorPacketBundle) -> None:
    hashes = bundle.manifest.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("steering anchor manifest lacks hashes")
    observed = {
        "public_packet_sha256": hashlib.sha256(
            _canonical_bytes(bundle.public_packet)
        ).hexdigest(),
        "internal_key_sha256": hashlib.sha256(
            _canonical_bytes(bundle.internal_key)
        ).hexdigest(),
        "rating_template_sha256": hashlib.sha256(
            _canonical_bytes(bundle.rating_template)
        ).hexdigest(),
    }
    if dict(hashes) != observed:
        raise ValueError("steering anchor packet hash drift")
    for payload in (bundle.public_packet, bundle.manifest):
        if payload.get("learning_use_authorized") is not False:
            raise ValueError("C2 packet cannot authorize learning use")
        if payload.get("production_promotion_authorized") is not False:
            raise ValueError("C2 packet cannot authorize promotion")


def _cohen_kappa(
    left: Sequence[SteeringAnchorPreference],
    right: Sequence[SteeringAnchorPreference],
) -> tuple[float, float | None]:
    if not left or len(left) != len(right):
        raise ValueError("Cohen kappa requires two aligned non-empty lanes")
    observed = sum(a is b for a, b in zip(left, right, strict=True)) / len(
        left
    )
    categories = tuple(SteeringAnchorPreference)
    expected = sum(
        (sum(value is category for value in left) / len(left))
        * (sum(value is category for value in right) / len(right))
        for category in categories
    )
    if 1.0 - expected <= 1e-12:
        return observed, None
    return observed, (observed - expected) / (1.0 - expected)


def _score_for_arm(
    rating: SteeringAnchorRating,
    *,
    arm: SteeringAnchorArm,
    dimension: str,
) -> int:
    value = getattr(rating, f"arm_{arm.value}_{dimension}")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("ratable steering score is missing")
    return value


def _canonical_vote(
    rating: SteeringAnchorRating,
    key: SteeringAnchorInternalKeyEntry,
) -> int:
    if rating.preferred_arm in {
        SteeringAnchorPreference.TIE,
        SteeringAnchorPreference.UNRATABLE,
    }:
        return 0
    preferred = SteeringAnchorArm(rating.preferred_arm.value)
    noop_arm = (
        SteeringAnchorArm.B
        if key.steered_arm is SteeringAnchorArm.A
        else SteeringAnchorArm.A
    )
    relationship_delta = _score_for_arm(
        rating,
        arm=key.steered_arm,
        dimension="relationship_support",
    ) - _score_for_arm(
        rating,
        arm=noop_arm,
        dimension="relationship_support",
    )
    boundary_delta = _score_for_arm(
        rating,
        arm=key.steered_arm,
        dimension="boundary_respect",
    ) - _score_for_arm(
        rating,
        arm=noop_arm,
        dimension="boundary_respect",
    )
    task_delta = _score_for_arm(
        rating,
        arm=key.steered_arm,
        dimension="task_preservation",
    ) - _score_for_arm(
        rating,
        arm=noop_arm,
        dimension="task_preservation",
    )
    if boundary_delta < -1 or task_delta < -1:
        return -1
    if preferred is key.steered_arm and relationship_delta > 0:
        return 1
    if preferred is noop_arm:
        return -1
    return 0


def analyze_steering_human_anchor_pilot(
    *,
    bundle: SteeringAnchorPacketBundle,
    raters: Sequence[SteeringAnchorRaterAttestation],
    ratings: Sequence[SteeringAnchorRating],
) -> SteeringAnchorPilotReport:
    """Analyze rubric reliability and read-only C1 directional alignment."""

    _verify_packet_bundle(bundle)
    protocol = build_steering_human_anchor_protocol()
    if len(raters) != protocol.raters_per_unit:
        raise ValueError("C2 pilot requires exactly two expert raters")
    rater_by_id = {rater.rater_id: rater for rater in raters}
    if len(rater_by_id) != len(raters):
        raise ValueError("duplicate steering anchor rater")
    public_units = bundle.public_packet.get("units")
    private_entries = bundle.internal_key.get("entries")
    if not isinstance(public_units, list) or not isinstance(
        private_entries, list
    ):
        raise ValueError("steering anchor packet structure drift")
    unit_ids = tuple(unit["unit_id"] for unit in public_units)
    if len(unit_ids) != protocol.pilot_units or len(set(unit_ids)) != len(
        unit_ids
    ):
        raise ValueError("steering anchor pilot unit coverage drift")
    keys = {
        entry["unit_id"]: SteeringAnchorInternalKeyEntry(
            unit_id=entry["unit_id"],
            decision_ref_sha256=entry["decision_ref_sha256"],
            steered_arm=SteeringAnchorArm(entry["steered_arm"]),
            terminal_pe_sha256=entry["terminal_pe_sha256"],
            relative_mse_improvement=entry["relative_mse_improvement"],
            policy_version=entry["policy_version"],
        )
        for entry in private_entries
    }
    if set(keys) != set(unit_ids):
        raise ValueError("analysis key coverage drift")

    rating_by_unit_rater: dict[tuple[str, str], SteeringAnchorRating] = {}
    for rating in ratings:
        rater = rater_by_id.get(rating.rater_id)
        if rater is None:
            raise ValueError("rating references an unregistered rater")
        if rating.eligibility_review_sha256 != (
            rater.eligibility_review_sha256
        ):
            raise ValueError("rating/rater eligibility lineage drift")
        if rating.unit_id not in keys:
            raise ValueError("rating references an unknown unit")
        identity = (rating.unit_id, rating.rater_id)
        if identity in rating_by_unit_rater:
            raise ValueError("duplicate steering anchor rating row")
        rating_by_unit_rater[identity] = rating
    expected_rating_count = protocol.pilot_units * protocol.raters_per_unit
    if len(rating_by_unit_rater) != expected_rating_count:
        raise ValueError("steering anchor rating coverage is incomplete")

    ordered_raters = tuple(sorted(rater_by_id))
    left = tuple(
        rating_by_unit_rater[(unit_id, ordered_raters[0])].preferred_arm
        for unit_id in unit_ids
    )
    right = tuple(
        rating_by_unit_rater[(unit_id, ordered_raters[1])].preferred_arm
        for unit_id in unit_ids
    )
    exact_agreement, kappa = _cohen_kappa(left, right)
    unratable_count = sum(
        rating.preferred_arm is SteeringAnchorPreference.UNRATABLE
        for rating in ratings
    )
    unratable_rate = unratable_count / len(ratings)
    agreement_gate = (
        exact_agreement >= protocol.minimum_exact_agreement
        and kappa is not None
        and kappa >= protocol.minimum_cohen_kappa
        and unratable_rate <= protocol.maximum_unratable_rate
    )

    resolved = 0
    aligned = 0
    for unit_id in unit_ids:
        key = keys[unit_id]
        votes = tuple(
            _canonical_vote(
                rating_by_unit_rater[(unit_id, rater_id)],
                key,
            )
            for rater_id in ordered_raters
        )
        if votes[0] == 0 or votes[0] != votes[1]:
            continue
        if key.relative_mse_improvement > protocol.pe_deadzone:
            pe_direction = 1
        elif key.relative_mse_improvement < -protocol.pe_deadzone:
            pe_direction = -1
        else:
            continue
        resolved += 1
        aligned += int(votes[0] == pe_direction)
    alignment_rate = aligned / resolved if resolved else None
    review_required = (
        resolved >= protocol.minimum_resolved_directions
        and alignment_rate is not None
        and alignment_rate < STEERING_HUMAN_ANCHOR_ALIGNMENT_REVIEW_FLOOR
    )
    return SteeringAnchorPilotReport(
        unit_count=len(unit_ids),
        rating_count=len(ratings),
        exact_agreement=exact_agreement,
        cohen_kappa=kappa,
        unratable_rate=unratable_rate,
        agreement_gate_passed=agreement_gate,
        resolved_direction_count=resolved,
        directional_alignment_rate=alignment_rate,
        c1_alignment_review_required=review_required,
        expansion_admissible=agreement_gate,
        learning_use_authorized=False,
        production_promotion_authorized=False,
        description=(
            "C2 validation-only expert anchor. Agreement controls rubric "
            "expansion; C1 directional alignment is a readout and cannot "
            "update credit, policy, or promotion state."
        ),
    )


__all__ = (
    "STEERING_HUMAN_ANCHOR_ALIGNMENT_REVIEW_FLOOR",
    "STEERING_HUMAN_ANCHOR_CONSENT_SCOPE",
    "STEERING_HUMAN_ANCHOR_SCHEMA_VERSION",
    "SteeringAnchorArm",
    "SteeringAnchorCaptureUnit",
    "SteeringAnchorConsentAttestation",
    "SteeringAnchorInternalKeyEntry",
    "SteeringAnchorPacketBundle",
    "SteeringAnchorPilotReport",
    "SteeringAnchorPreference",
    "SteeringAnchorPrivacyAttestation",
    "SteeringAnchorProtocol",
    "SteeringAnchorRaterAttestation",
    "SteeringAnchorRating",
    "SteeringAnchorTurn",
    "SteeringAnchorUnratableReason",
    "SteeringAnchorWithdrawal",
    "analyze_steering_human_anchor_pilot",
    "apply_steering_anchor_withdrawals",
    "build_steering_human_anchor_pilot_packet",
    "build_steering_human_anchor_protocol",
)
