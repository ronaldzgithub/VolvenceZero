"""Matched-ablation gate for State-KV substrate-to-temporal causality."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from volvence_zero.substrate import SubstrateSnapshot
from volvence_zero.temporal import TemporalAbstractionSnapshot

TEMPORAL_CAUSAL_SCHEMA_VERSION = "state-kv-temporal-causal.v1"


@dataclass(frozen=True)
class TemporalCausalArm:
    label: str
    substrate: SubstrateSnapshot
    temporal: TemporalAbstractionSnapshot


@dataclass(frozen=True)
class TemporalCausalClaim:
    claim: str
    state: str
    detail: str


@dataclass(frozen=True)
class TemporalCausalVerdict:
    artifact_id: str
    substrate_fingerprint: str
    source_text_sha256: str
    residual_tolerance: float
    temporal_tolerance: float
    claims: tuple[TemporalCausalClaim, ...]
    distances: tuple[tuple[str, float], ...]
    arms: tuple[tuple[str, object], ...]

    @property
    def gate_state(self) -> str:
        return (
            "pass"
            if self.claims and all(claim.state == "pass" for claim in self.claims)
            else "fail"
        )

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": TEMPORAL_CAUSAL_SCHEMA_VERSION,
            "gate_state": self.gate_state,
            "artifact_id": self.artifact_id,
            "substrate_fingerprint": self.substrate_fingerprint,
            "source_text_sha256": self.source_text_sha256,
            "residual_tolerance": self.residual_tolerance,
            "temporal_tolerance": self.temporal_tolerance,
            "claims": [
                {
                    "claim": claim.claim,
                    "state": claim.state,
                    "detail": claim.detail,
                }
                for claim in self.claims
            ],
            "distances": dict(self.distances),
            "arms": dict(self.arms),
        }


def build_temporal_causal_verdict(
    *,
    baseline: TemporalCausalArm,
    correct_state: TemporalCausalArm,
    wrong_user: TemporalCausalArm,
    revoked: TemporalCausalArm,
    artifact_id: str,
    substrate_fingerprint: str,
    source_text: str,
    residual_tolerance: float = 1e-8,
    temporal_tolerance: float = 1e-8,
) -> TemporalCausalVerdict:
    """Evaluate the four-arm physical capture and temporal matched ablation."""

    arms = (baseline, correct_state, wrong_user, revoked)
    expected_labels = ("baseline", "correct-state", "wrong-user", "revoked")
    if tuple(arm.label for arm in arms) != expected_labels:
        raise ValueError(
            "temporal causal arms must be ordered and labelled "
            f"{expected_labels!r}."
        )
    if residual_tolerance < 0 or temporal_tolerance < 0:
        raise ValueError("causal tolerances must be non-negative.")

    residuals = {
        arm.label: _flatten_residual_sequence(arm.substrate) for arm in arms
    }
    codes = {
        arm.label: arm.temporal.controller_state.code for arm in arms
    }
    betas = {
        arm.label: arm.temporal.controller_state.switch_gate for arm in arms
    }
    residual_correct = _mean_abs_distance(
        residuals["baseline"], residuals["correct-state"]
    )
    residual_wrong = _mean_abs_distance(
        residuals["baseline"], residuals["wrong-user"]
    )
    residual_between_states = _mean_abs_distance(
        residuals["correct-state"], residuals["wrong-user"]
    )
    residual_revoked = _mean_abs_distance(
        residuals["baseline"], residuals["revoked"]
    )
    z_correct = _mean_abs_distance(codes["baseline"], codes["correct-state"])
    z_wrong = _mean_abs_distance(codes["baseline"], codes["wrong-user"])
    z_between_states = _mean_abs_distance(
        codes["correct-state"], codes["wrong-user"]
    )
    z_revoked = _mean_abs_distance(codes["baseline"], codes["revoked"])
    beta_correct = abs(betas["correct-state"] - betas["baseline"])
    beta_wrong = abs(betas["wrong-user"] - betas["baseline"])
    beta_revoked = abs(betas["revoked"] - betas["baseline"])

    attestation_passed = (
        not baseline.substrate.personal_conditioning_applied
        and correct_state.substrate.personal_conditioning_applied
        and wrong_user.substrate.personal_conditioning_applied
        and not revoked.substrate.personal_conditioning_applied
    )
    residual_passed = (
        residual_correct > residual_tolerance
        and residual_wrong > residual_tolerance
        and residual_between_states > residual_tolerance
        and residual_revoked <= residual_tolerance
    )
    z_passed = (
        z_correct > temporal_tolerance
        and z_wrong > temporal_tolerance
        and z_between_states > temporal_tolerance
        and z_revoked <= temporal_tolerance
    )
    beta_passed = (
        max(beta_correct, beta_wrong) > temporal_tolerance
        and beta_revoked <= temporal_tolerance
    )
    lineage_passed = (
        baseline.substrate.conditioning_lineage is None
        and revoked.substrate.conditioning_lineage is None
        and correct_state.substrate.conditioning_lineage is not None
        and wrong_user.substrate.conditioning_lineage is not None
        and correct_state.substrate.conditioning_lineage.carrier == "prefix_kv"
        and wrong_user.substrate.conditioning_lineage.carrier == "prefix_kv"
        and correct_state.temporal.conditioning_lineage_refs
        == (correct_state.substrate.conditioning_lineage,)
        and wrong_user.temporal.conditioning_lineage_refs
        == (wrong_user.substrate.conditioning_lineage,)
        and not baseline.temporal.conditioning_lineage_refs
        and not revoked.temporal.conditioning_lineage_refs
    )

    claims = (
        TemporalCausalClaim(
            claim="claim_capture_conditioning_attested",
            state="pass" if attestation_passed else "fail",
            detail=(
                "correct-state and wrong-user captures report applied=true; "
                "baseline and revoked report applied=false"
            ),
        ),
        TemporalCausalClaim(
            claim="claim_residual_causality",
            state="pass" if residual_passed else "fail",
            detail=(
                f"baseline→correct={residual_correct:.12g}, "
                f"baseline→wrong={residual_wrong:.12g}, "
                f"correct→wrong={residual_between_states:.12g}, "
                f"baseline→revoked={residual_revoked:.12g}"
            ),
        ),
        TemporalCausalClaim(
            claim="claim_temporal_code_causality",
            state="pass" if z_passed else "fail",
            detail=(
                f"z baseline→correct={z_correct:.12g}, "
                f"baseline→wrong={z_wrong:.12g}, "
                f"correct→wrong={z_between_states:.12g}, "
                f"baseline→revoked={z_revoked:.12g}"
            ),
        ),
        TemporalCausalClaim(
            claim="claim_temporal_switch_causality",
            state="pass" if beta_passed else "fail",
            detail=(
                f"beta baseline→correct={beta_correct:.12g}, "
                f"baseline→wrong={beta_wrong:.12g}, "
                f"baseline→revoked={beta_revoked:.12g}"
            ),
        ),
        TemporalCausalClaim(
            claim="claim_conditioning_lineage_alignment",
            state="pass" if lineage_passed else "fail",
            detail=(
                "only conditioned arms publish prefix_kv lineage and temporal "
                "preserves the exact substrate reference"
            ),
        ),
    )
    distances = (
        ("residual_baseline_to_correct", residual_correct),
        ("residual_baseline_to_wrong", residual_wrong),
        ("residual_correct_to_wrong", residual_between_states),
        ("residual_baseline_to_revoked", residual_revoked),
        ("z_baseline_to_correct", z_correct),
        ("z_baseline_to_wrong", z_wrong),
        ("z_correct_to_wrong", z_between_states),
        ("z_baseline_to_revoked", z_revoked),
        ("beta_baseline_to_correct", beta_correct),
        ("beta_baseline_to_wrong", beta_wrong),
        ("beta_baseline_to_revoked", beta_revoked),
    )
    arm_payloads = tuple(
        (
            arm.label,
            {
                "personal_conditioning_applied": (
                    arm.substrate.personal_conditioning_applied
                ),
                "conditioning_carrier": (
                    arm.substrate.conditioning_lineage.carrier
                    if arm.substrate.conditioning_lineage is not None
                    else ""
                ),
                "residual_sha256": _float_vector_digest(
                    residuals[arm.label]
                ),
                "z_t": list(codes[arm.label]),
                "beta_t": betas[arm.label],
            },
        )
        for arm in arms
    )
    return TemporalCausalVerdict(
        artifact_id=artifact_id,
        substrate_fingerprint=substrate_fingerprint,
        source_text_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        residual_tolerance=residual_tolerance,
        temporal_tolerance=temporal_tolerance,
        claims=claims,
        distances=distances,
        arms=arm_payloads,
    )


def _flatten_residual_sequence(snapshot: SubstrateSnapshot) -> tuple[float, ...]:
    values = tuple(
        value
        for step in snapshot.residual_sequence
        for activation in step.residual_activations
        for value in activation.activation
    )
    if not values:
        raise ValueError(
            f"substrate snapshot {snapshot.model_id!r} has no residual sequence."
        )
    return values


def _mean_abs_distance(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"matched vectors differ in width: {len(left)} vs {len(right)}."
        )
    if not left:
        raise ValueError("matched vectors must be non-empty.")
    return sum(
        abs(left_value - right_value)
        for left_value, right_value in zip(left, right, strict=True)
    ) / len(left)


def _float_vector_digest(values: tuple[float, ...]) -> str:
    payload = json.dumps(
        [float(value).hex() for value in values],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
