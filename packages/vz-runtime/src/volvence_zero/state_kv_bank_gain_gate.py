"""Read-only State KV P4-c per-bank gain and negative-control gate."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Sequence

BANK_GAIN_SCHEMA_VERSION = "state-kv-bank-gain.v3"
BANK_GAIN_PANEL_SCHEMA_VERSION = "state-kv-bank-gain.v4"
BANK_GAIN_PROFILE_LABELS = (
    "state-kv-bank-none",
    "state-kv-bank-personal-only",
    "state-kv-bank-relationship-only",
    "state-kv-bank-dual",
)
SUPPORTED_BANKS = ("personal", "relationship")


@dataclass(frozen=True)
class PairedBankGainSample:
    """Dual-bank output paired with the same turn after one bank is removed."""

    probe_id: str
    bank_type: str
    dual_output: str
    ablated_output: str
    dual_match_correct: bool | None
    ablated_match_correct: bool | None

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("bank-gain sample requires probe_id")
        if self.bank_type not in SUPPORTED_BANKS:
            raise ValueError(
                f"unsupported bank_type {self.bank_type!r}; "
                f"expected one of {SUPPORTED_BANKS!r}"
            )
        if not self.dual_output.strip() or not self.ablated_output.strip():
            raise ValueError("bank-gain paired outputs must be non-empty")
        judge_states = (self.dual_match_correct, self.ablated_match_correct)
        if (judge_states[0] is None) != (judge_states[1] is None):
            raise ValueError(
                "paired bank-gain sample must provide both blind-judge "
                "outcomes or neither"
            )


@dataclass(frozen=True)
class IrrelevantBankControlSample:
    """A turn where adding one semantically irrelevant bank should not help."""

    probe_id: str
    bank_type: str
    router_score: float
    without_bank_match_correct: bool | None
    with_bank_match_correct: bool | None

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("irrelevant-bank sample requires probe_id")
        if self.bank_type not in SUPPORTED_BANKS:
            raise ValueError(
                f"unsupported bank_type {self.bank_type!r}; "
                f"expected one of {SUPPORTED_BANKS!r}"
            )
        if not 0.0 <= self.router_score <= 1.0:
            raise ValueError("irrelevant-bank router_score must be in [0, 1]")
        judge_states = (
            self.without_bank_match_correct,
            self.with_bank_match_correct,
        )
        if (judge_states[0] is None) != (judge_states[1] is None):
            raise ValueError(
                "irrelevant-bank sample must provide both blind-judge "
                "outcomes or neither"
            )


@dataclass(frozen=True)
class NonBankPersonaControlSample:
    """Persona match result from the arm where no conditioning bank is live."""

    probe_id: str
    bank_type: str
    match_correct: bool | None

    def __post_init__(self) -> None:
        if not self.probe_id:
            raise ValueError("non-bank persona control requires probe_id")
        if self.bank_type not in SUPPORTED_BANKS:
            raise ValueError(
                f"unsupported bank_type {self.bank_type!r}; "
                f"expected one of {SUPPORTED_BANKS!r}"
            )


@dataclass(frozen=True)
class BankGainClaim:
    claim: str
    state: str
    detail: str


@dataclass(frozen=True)
class BankGainMetric:
    bank_type: str
    sample_count: int
    judged_sample_count: int
    output_divergence_rate: float
    blind_match_gain: float | None
    blind_match_gain_ci: tuple[float, float] | None


@dataclass(frozen=True)
class NonBankPersonaMetric:
    bank_type: str
    sample_count: int
    judged_sample_count: int
    blind_match_accuracy: float | None
    blind_match_accuracy_ci: tuple[float, float] | None


@dataclass(frozen=True)
class BankPersonaContrast:
    """Pre-treatment proof that persona-specific bank state did not collapse."""

    bank_type: str
    probe_count: int
    material_contrast_count: int
    fingerprint_contrast_count: int

    def __post_init__(self) -> None:
        if self.bank_type not in SUPPORTED_BANKS:
            raise ValueError(
                f"unsupported bank_type {self.bank_type!r}; "
                f"expected one of {SUPPORTED_BANKS!r}"
            )
        if self.probe_count < 0:
            raise ValueError("bank persona contrast probe_count must be >= 0")
        if not 0 <= self.material_contrast_count <= self.probe_count:
            raise ValueError(
                "material_contrast_count must be within [0, probe_count]"
            )
        if not 0 <= self.fingerprint_contrast_count <= self.probe_count:
            raise ValueError(
                "fingerprint_contrast_count must be within [0, probe_count]"
            )

    @property
    def passed(self) -> bool:
        return (
            self.probe_count > 0
            and self.material_contrast_count == self.probe_count
            and self.fingerprint_contrast_count == self.probe_count
        )


@dataclass(frozen=True)
class BankGainVerdict:
    artifact_id: str
    substrate_fingerprint: str
    router_version: str
    bootstrap_seed: int
    minimum_samples: int
    irrelevant_router_score_ceiling: float
    non_bank_chance_accuracy: float
    judge_model_id: str
    judge_family: str
    judge_material_kind: str
    observation_artifact_sha256: str
    semantic_backend: str
    persona_contrasts: tuple[BankPersonaContrast, ...]
    metrics: tuple[BankGainMetric, ...]
    non_bank_persona_metrics: tuple[NonBankPersonaMetric, ...]
    claims: tuple[BankGainClaim, ...]
    bank_count_frozen: bool
    freeze_reason: str

    @property
    def gate_state(self) -> str:
        states = {claim.state for claim in self.claims}
        if states == {"pass"}:
            return "pass"
        if "fail" in states:
            return "fail"
        return "insufficient_data"

    def as_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": BANK_GAIN_SCHEMA_VERSION,
            "gate_state": self.gate_state,
            "artifact_id": self.artifact_id,
            "substrate_fingerprint": self.substrate_fingerprint,
            "router_version": self.router_version,
            "profile_labels": list(BANK_GAIN_PROFILE_LABELS),
            "bootstrap_seed": self.bootstrap_seed,
            "minimum_samples": self.minimum_samples,
            "irrelevant_router_score_ceiling": (
                self.irrelevant_router_score_ceiling
            ),
            "non_bank_chance_accuracy": self.non_bank_chance_accuracy,
            "judge": {
                "model_id": self.judge_model_id,
                "family": self.judge_family,
                "material_kind": self.judge_material_kind,
            },
            "observation_artifact_sha256": self.observation_artifact_sha256,
            "semantic_backend": self.semantic_backend,
            "persona_contrasts": [
                {
                    "bank_type": contrast.bank_type,
                    "probe_count": contrast.probe_count,
                    "material_contrast_count": (
                        contrast.material_contrast_count
                    ),
                    "fingerprint_contrast_count": (
                        contrast.fingerprint_contrast_count
                    ),
                    "passed": contrast.passed,
                }
                for contrast in self.persona_contrasts
            ],
            "metrics": [
                {
                    "bank_type": metric.bank_type,
                    "sample_count": metric.sample_count,
                    "judged_sample_count": metric.judged_sample_count,
                    "output_divergence_rate": metric.output_divergence_rate,
                    "blind_match_gain": metric.blind_match_gain,
                    "blind_match_gain_ci": (
                        list(metric.blind_match_gain_ci)
                        if metric.blind_match_gain_ci is not None
                        else None
                    ),
                }
                for metric in self.metrics
            ],
            "non_bank_persona_metrics": [
                {
                    "bank_type": metric.bank_type,
                    "sample_count": metric.sample_count,
                    "judged_sample_count": metric.judged_sample_count,
                    "blind_match_accuracy": metric.blind_match_accuracy,
                    "blind_match_accuracy_ci": (
                        list(metric.blind_match_accuracy_ci)
                        if metric.blind_match_accuracy_ci is not None
                        else None
                    ),
                }
                for metric in self.non_bank_persona_metrics
            ],
            "claims": [
                {
                    "claim": claim.claim,
                    "state": claim.state,
                    "detail": claim.detail,
                }
                for claim in self.claims
            ],
            "bank_count_frozen": self.bank_count_frozen,
            "freeze_reason": self.freeze_reason,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_json_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


@dataclass(frozen=True)
class BankGainPanelVerdict:
    """Two-judge confirmation over one frozen observation artifact."""

    preregistration_sha256: str
    observation_artifact_sha256: str
    judge_verdicts: tuple[BankGainVerdict, ...]
    claims: tuple[BankGainClaim, ...]

    @property
    def gate_state(self) -> str:
        states = {claim.state for claim in self.claims}
        if states == {"pass"}:
            return "pass"
        if "fail" in states:
            return "fail"
        return "insufficient_data"

    @property
    def bank_count_frozen(self) -> bool:
        freeze_eligible = {
            "claim_personal_independent_gain",
            "claim_relationship_independent_gain",
            "claim_irrelevant_bank_negative_control",
        }
        return any(
            claim.claim in freeze_eligible and claim.state == "fail"
            for claim in self.claims
        )

    def as_json_dict(self) -> dict[str, object]:
        first = self.judge_verdicts[0]
        return {
            "schema_version": BANK_GAIN_PANEL_SCHEMA_VERSION,
            "gate_state": self.gate_state,
            "artifact_id": first.artifact_id,
            "substrate_fingerprint": first.substrate_fingerprint,
            "router_version": first.router_version,
            "profile_labels": list(BANK_GAIN_PROFILE_LABELS),
            "preregistration_sha256": self.preregistration_sha256,
            "observation_artifact_sha256": (
                self.observation_artifact_sha256
            ),
            "judge_panel": [
                verdict.as_json_dict() for verdict in self.judge_verdicts
            ],
            "claims": [
                {
                    "claim": claim.claim,
                    "state": claim.state,
                    "detail": claim.detail,
                }
                for claim in self.claims
            ],
            "bank_count_frozen": self.bank_count_frozen,
            "freeze_reason": (
                "At least one fully observed panel-confirmed independent-"
                "gain or irrelevant-bank control failed; freeze the bank "
                "count at Personal + Relationship."
                if self.bank_count_frozen
                else ""
            ),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_json_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )


def build_bank_gain_panel_verdict(
    *,
    judge_verdicts: Sequence[BankGainVerdict],
    preregistration_sha256: str,
) -> BankGainPanelVerdict:
    """Require two distinct frozen judges over byte-identical observations."""

    verdicts = tuple(judge_verdicts)
    if len(verdicts) < 2:
        raise ValueError("bank-gain v4 requires at least two judge verdicts")
    judge_ids = tuple(verdict.judge_model_id for verdict in verdicts)
    if any(not judge_id for judge_id in judge_ids):
        raise ValueError("bank-gain v4 requires non-empty judge model ids")
    if len(set(judge_ids)) != len(judge_ids):
        raise ValueError("bank-gain v4 requires distinct judge model ids")
    if not preregistration_sha256:
        raise ValueError("bank-gain v4 requires preregistration SHA-256")
    observation_ids = {
        verdict.observation_artifact_sha256 for verdict in verdicts
    }
    if "" in observation_ids or len(observation_ids) != 1:
        raise ValueError(
            "bank-gain v4 judge verdicts must share one observation artifact"
        )
    artifact_ids = {verdict.artifact_id for verdict in verdicts}
    substrates = {verdict.substrate_fingerprint for verdict in verdicts}
    routers = {verdict.router_version for verdict in verdicts}
    if len(artifact_ids) != 1 or len(substrates) != 1 or len(routers) != 1:
        raise ValueError(
            "bank-gain v4 judge verdicts must share artifact, substrate, and router"
        )
    claim_names = tuple(claim.claim for claim in verdicts[0].claims)
    if any(
        tuple(claim.claim for claim in verdict.claims) != claim_names
        for verdict in verdicts[1:]
    ):
        raise ValueError("bank-gain v4 judge claim sets do not match")
    panel_claims = []
    for index, claim_name in enumerate(claim_names):
        states = tuple(verdict.claims[index].state for verdict in verdicts)
        if all(state == "pass" for state in states):
            state = "pass"
        elif any(state == "fail" for state in states):
            state = "fail"
        else:
            state = "insufficient_data"
        panel_claims.append(
            BankGainClaim(
                claim=claim_name,
                state=state,
                detail="; ".join(
                    f"{judge_id}={judge_state}"
                    for judge_id, judge_state in zip(
                        judge_ids, states, strict=True
                    )
                ),
            )
        )
    return BankGainPanelVerdict(
        preregistration_sha256=preregistration_sha256,
        observation_artifact_sha256=next(iter(observation_ids)),
        judge_verdicts=verdicts,
        claims=tuple(panel_claims),
    )


def _paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    seed: int,
    samples: int = 4000,
) -> tuple[float, float]:
    if not deltas:
        raise ValueError("paired bootstrap requires at least one delta")
    rng = random.Random(seed)
    count = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return (
        means[int(0.025 * (samples - 1))],
        means[int(0.975 * (samples - 1))],
    )


def _bank_metric(
    bank_type: str,
    samples: Sequence[PairedBankGainSample],
    *,
    bootstrap_seed: int,
) -> BankGainMetric:
    matching = [sample for sample in samples if sample.bank_type == bank_type]
    judged = [
        sample
        for sample in matching
        if sample.dual_match_correct is not None
    ]
    deltas = [
        float(sample.dual_match_correct)
        - float(sample.ablated_match_correct)
        for sample in judged
    ]
    return BankGainMetric(
        bank_type=bank_type,
        sample_count=len(matching),
        judged_sample_count=len(judged),
        output_divergence_rate=(
            sum(
                sample.dual_output != sample.ablated_output
                for sample in matching
            )
            / len(matching)
            if matching
            else 0.0
        ),
        blind_match_gain=(
            sum(deltas) / len(deltas) if deltas else None
        ),
        blind_match_gain_ci=(
            _paired_bootstrap_ci(
                deltas,
                seed=bootstrap_seed + SUPPORTED_BANKS.index(bank_type),
            )
            if deltas
            else None
        ),
    )


def _non_bank_persona_metric(
    bank_type: str,
    samples: Sequence[NonBankPersonaControlSample],
    *,
    bootstrap_seed: int,
) -> NonBankPersonaMetric:
    matching = [sample for sample in samples if sample.bank_type == bank_type]
    judged = [sample for sample in matching if sample.match_correct is not None]
    outcomes = [float(sample.match_correct) for sample in judged]
    return NonBankPersonaMetric(
        bank_type=bank_type,
        sample_count=len(matching),
        judged_sample_count=len(judged),
        blind_match_accuracy=(
            sum(outcomes) / len(outcomes) if outcomes else None
        ),
        blind_match_accuracy_ci=(
            _paired_bootstrap_ci(
                outcomes,
                seed=bootstrap_seed + 31 + SUPPORTED_BANKS.index(bank_type),
            )
            if outcomes
            else None
        ),
    )


def build_bank_gain_verdict(
    *,
    paired_samples: Sequence[PairedBankGainSample],
    irrelevant_controls: Sequence[IrrelevantBankControlSample],
    non_bank_persona_controls: Sequence[NonBankPersonaControlSample],
    persona_contrasts: Sequence[BankPersonaContrast],
    artifact_id: str,
    substrate_fingerprint: str,
    router_version: str,
    minimum_samples: int = 8,
    irrelevant_router_score_ceiling: float = 0.2,
    non_bank_chance_accuracy: float = 0.5,
    bootstrap_seed: int = 7301,
    judge_model_id: str = "",
    judge_family: str = "",
    judge_material_kind: str = "",
    observation_artifact_sha256: str = "",
    semantic_backend: str = "",
) -> BankGainVerdict:
    """Build a frozen verdict without mutating any runtime owner."""

    if not artifact_id:
        raise ValueError("bank-gain verdict requires artifact_id")
    if not router_version:
        raise ValueError("bank-gain verdict requires router_version")
    if minimum_samples < 2:
        raise ValueError("minimum_samples must be >= 2")
    if not 0.0 <= irrelevant_router_score_ceiling <= 1.0:
        raise ValueError(
            "irrelevant_router_score_ceiling must be in [0, 1]"
        )
    if not 0.0 < non_bank_chance_accuracy < 1.0:
        raise ValueError("non_bank_chance_accuracy must be within (0, 1)")

    metrics = tuple(
        _bank_metric(
            bank,
            paired_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for bank in SUPPORTED_BANKS
    )
    non_bank_metrics = tuple(
        _non_bank_persona_metric(
            bank,
            non_bank_persona_controls,
            bootstrap_seed=bootstrap_seed,
        )
        for bank in SUPPORTED_BANKS
    )
    non_bank_metric_by_bank = {
        metric.bank_type: metric for metric in non_bank_metrics
    }
    contrast_by_bank = {
        contrast.bank_type: contrast for contrast in persona_contrasts
    }
    if len(contrast_by_bank) != len(persona_contrasts):
        raise ValueError("persona_contrasts must contain unique bank types")
    claims: list[BankGainClaim] = []
    for metric in metrics:
        contrast = contrast_by_bank.get(metric.bank_type)
        contrast_state = (
            "pass"
            if contrast is not None and contrast.passed
            else "insufficient_data"
        )
        claims.append(
            BankGainClaim(
                claim=f"claim_{metric.bank_type}_state_contrast",
                state=contrast_state,
                detail=(
                    "missing bank-state contrast"
                    if contrast is None
                    else (
                        f"probes={contrast.probe_count}, "
                        "material_contrasts="
                        f"{contrast.material_contrast_count}, "
                        "fingerprint_contrasts="
                        f"{contrast.fingerprint_contrast_count}"
                    )
                ),
            )
        )
        non_bank_metric = non_bank_metric_by_bank[metric.bank_type]
        if (
            non_bank_metric.sample_count < minimum_samples
            or non_bank_metric.judged_sample_count < minimum_samples
            or non_bank_metric.blind_match_accuracy_ci is None
        ):
            isolation_state = "insufficient_data"
        elif (
            non_bank_metric.blind_match_accuracy_ci[0]
            <= non_bank_chance_accuracy
        ):
            isolation_state = "pass"
        else:
            # This is an experiment-validity failure, not evidence that the
            # bank itself lacks gain. The semantic spine already exposes the
            # persona without any bank, so the marginal-gain contrast has no
            # isolated treatment and must remain inconclusive.
            isolation_state = "insufficient_data"
        claims.append(
            BankGainClaim(
                claim=f"claim_{metric.bank_type}_non_bank_isolation",
                state=isolation_state,
                detail=(
                    f"n={non_bank_metric.sample_count}, judged="
                    f"{non_bank_metric.judged_sample_count}, accuracy="
                    f"{non_bank_metric.blind_match_accuracy!r}, ci="
                    f"{non_bank_metric.blind_match_accuracy_ci!r}, "
                    f"chance={non_bank_chance_accuracy:.3f}"
                ),
            )
        )
        claim_name = f"claim_{metric.bank_type}_independent_gain"
        if contrast_state != "pass" or isolation_state != "pass":
            state = "insufficient_data"
        elif (
            metric.sample_count < minimum_samples
            or metric.judged_sample_count < minimum_samples
            or metric.blind_match_gain_ci is None
        ):
            state = "insufficient_data"
        elif (
            metric.output_divergence_rate > 0.0
            and metric.blind_match_gain_ci[0] > 0.0
        ):
            state = "pass"
        else:
            state = "fail"
        claims.append(
            BankGainClaim(
                claim=claim_name,
                state=state,
                detail=(
                    f"n={metric.sample_count}, judged="
                    f"{metric.judged_sample_count}, divergence="
                    f"{metric.output_divergence_rate:.3f}, match_gain="
                    f"{metric.blind_match_gain!r}, "
                    f"ci={metric.blind_match_gain_ci!r}"
                ),
            )
        )

    judged_controls = [
        sample
        for sample in irrelevant_controls
        if sample.with_bank_match_correct is not None
    ]
    control_deltas = [
        float(sample.with_bank_match_correct)
        - float(sample.without_bank_match_correct)
        for sample in judged_controls
    ]
    control_ci = (
        _paired_bootstrap_ci(control_deltas, seed=bootstrap_seed + 101)
        if control_deltas
        else None
    )
    if (
        len(irrelevant_controls) < minimum_samples
        or len(judged_controls) < minimum_samples
        or control_ci is None
    ):
        control_state = "insufficient_data"
    elif (
        max(sample.router_score for sample in irrelevant_controls)
        <= irrelevant_router_score_ceiling
        and control_ci[1] <= 0.0
    ):
        control_state = "pass"
    else:
        control_state = "fail"
    claims.append(
        BankGainClaim(
            claim="claim_irrelevant_bank_negative_control",
            state=control_state,
            detail=(
                f"n={len(irrelevant_controls)}, judged="
                f"{len(judged_controls)}, max_router_score="
                f"{max((sample.router_score for sample in irrelevant_controls), default=None)!r}, "
                f"match_gain_ci={control_ci!r}, ceiling="
                f"{irrelevant_router_score_ceiling:.3f}"
            ),
        )
    )

    # Missing observations are not evidence of marginal-gain decay. The
    # programme-level stop condition may freeze expansion only after a
    # measured failure, never merely because the experiment has not run.
    freeze_eligible_claims = {
        "claim_personal_independent_gain",
        "claim_relationship_independent_gain",
        "claim_irrelevant_bank_negative_control",
    }
    bank_count_frozen = any(
        claim.claim in freeze_eligible_claims and claim.state == "fail"
        for claim in claims
    )
    freeze_reason = (
        "At least one fully observed independent-gain or irrelevant-bank "
        "control failed; freeze the bank count at Personal + Relationship."
        if bank_count_frozen
        else ""
    )
    return BankGainVerdict(
        artifact_id=artifact_id,
        substrate_fingerprint=substrate_fingerprint,
        router_version=router_version,
        bootstrap_seed=bootstrap_seed,
        minimum_samples=minimum_samples,
        irrelevant_router_score_ceiling=irrelevant_router_score_ceiling,
        non_bank_chance_accuracy=non_bank_chance_accuracy,
        judge_model_id=judge_model_id,
        judge_family=judge_family,
        judge_material_kind=judge_material_kind,
        observation_artifact_sha256=observation_artifact_sha256,
        semantic_backend=semantic_backend,
        persona_contrasts=tuple(persona_contrasts),
        metrics=metrics,
        non_bank_persona_metrics=non_bank_metrics,
        claims=tuple(claims),
        bank_count_frozen=bank_count_frozen,
        freeze_reason=freeze_reason,
    )


__all__ = [
    "BANK_GAIN_PROFILE_LABELS",
    "BANK_GAIN_SCHEMA_VERSION",
    "BANK_GAIN_PANEL_SCHEMA_VERSION",
    "BankGainClaim",
    "BankGainMetric",
    "BankPersonaContrast",
    "BankGainVerdict",
    "BankGainPanelVerdict",
    "IrrelevantBankControlSample",
    "NonBankPersonaControlSample",
    "NonBankPersonaMetric",
    "PairedBankGainSample",
    "build_bank_gain_verdict",
    "build_bank_gain_panel_verdict",
]
