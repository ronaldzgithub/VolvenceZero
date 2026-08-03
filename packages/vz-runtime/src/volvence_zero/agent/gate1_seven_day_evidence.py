"""Matched seven-day Gate 1 PE -> temporal evidence evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Sequence

from volvence_zero.agent.evidence_statistics import paired_student_t_ci95
from volvence_zero.agent.seven_day_companion_evidence import (
    SEVEN_DAY_METRICS,
    SevenDayArmExecutor,
    SevenDayExperimentCase,
)
from volvence_zero.agent.seven_day_n_plus_one import (
    validate_seven_day_n_plus_one_evidence,
)


GATE1_SEVEN_DAY_SCHEMA_VERSION = "gate1-seven-day-companion.v2"
GATE1_SEVEN_DAY_SMOKE_SCHEMA_VERSION = "gate1-seven-day-companion-smoke.v1"
GATE1_PE_ON_ARM = "gate1-pe-temporal-on-v1"
GATE1_PE_OFF_ARM = "gate1-pe-temporal-off-v1"
GATE1_SEVEN_DAY_ARMS = (GATE1_PE_ON_ARM, GATE1_PE_OFF_ARM)
_NEGATIVE_CONTINUITY_METRICS = frozenset(
    {
        "boundary_violation_rate",
        "wrong_user_attribution_rate",
        "user_correction_rate",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _continuity_composite(raw: object, *, field: str) -> float | None:
    metrics = _require_mapping(raw, field=field)
    values = []
    for name in SEVEN_DAY_METRICS:
        raw_value = metrics.get(name)
        if raw_value is None:
            return None
        value = _finite_number(raw_value, field=f"{field}.{name}")
        if name == "seven_day_trust_delta":
            if value < -1.0 or value > 1.0:
                raise ValueError(f"{field}.{name} is outside [-1, 1]")
            values.append((value + 1.0) / 2.0)
        else:
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{field}.{name} is outside [0, 1]")
            values.append(
                1.0 - value
                if name in _NEGATIVE_CONTINUITY_METRICS
                else value
            )
    return statistics.fmean(values)


def _paired_ci95(values: Sequence[float]) -> tuple[float, float] | None:
    return paired_student_t_ci95(values)


@dataclass(frozen=True)
class Gate1ArmReadout:
    case_id: str
    arm_label: str
    pe_observation_count: int
    nonbootstrap_observation_count: int
    world_temporal_applied_count: int
    self_temporal_applied_count: int
    early_pe_mean: float
    late_pe_mean: float
    pe_adaptation: float
    final_day_continuity_composite: float | None
    n_plus_one_prediction_quality: float
    n_plus_one_cosine_error: float
    n_plus_one_mean_squared_error: float
    n_plus_one_persistence_quality: float
    n_plus_one_target_sequence_sha256: str
    n_plus_one_target_snapshot_fingerprint: str
    final_day_boundary_violation_rate: float
    final_day_wrong_user_attribution_rate: float
    runtime_profile_attestation_sha256: str


@dataclass(frozen=True)
class Gate1SevenDayResult:
    schema_version: str
    evidence_tier: str
    preregistration_sha256: str
    run_count: int
    pair_count: int
    readouts: tuple[Gate1ArmReadout, ...]
    pe_adaptation_gain_mean: float
    pe_adaptation_gain_ci95: tuple[float, float] | None
    final_day_continuity_gain_mean: float | None
    final_day_continuity_gain_ci95: tuple[float, float] | None
    n_plus_one_prediction_gain_mean: float
    n_plus_one_prediction_gain_ci95: tuple[float, float] | None
    gates: Mapping[str, bool]
    mechanism_supported: bool
    causal_supported: bool
    claim_scope: str
    production_promotion_authorized: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class Gate1SevenDayHarness:
    """Execute the fixed matched arm schedule and export its evaluator result."""

    def __init__(self, *, executor: SevenDayArmExecutor) -> None:
        self._executor = executor

    def run(
        self,
        *,
        cases: Sequence[SevenDayExperimentCase],
        preregistration: Mapping[str, object],
        output_dir: str | Path,
        evidence_tier: str = "formal",
    ) -> Gate1SevenDayResult:
        if not cases:
            raise ValueError("Gate 1 seven-day evidence requires cases")
        target = Path(output_dir)
        run_root = target / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        runs: dict[tuple[str, str], Mapping[str, object]] = {}
        for arm_label in GATE1_SEVEN_DAY_ARMS:
            for case in cases:
                output_path = run_root / (
                    hashlib.sha256(
                        f"{case.case_id}\0{arm_label}".encode("utf-8")
                    ).hexdigest()
                    + ".json"
                )
                runs[(case.case_id, arm_label)] = self._executor.execute(
                    case=case,
                    arm_label=arm_label,
                    drain_slow_loop=True,
                    output_path=output_path,
                )
        result = evaluate_gate1_seven_day_runs(
            cases=cases,
            runs=runs,
            preregistration=preregistration,
            evidence_tier=evidence_tier,
        )
        filename = (
            "gate1_evaluation.json"
            if evidence_tier == "formal"
            else "gate1_smoke_evaluation.json"
        )
        _write_json(target / filename, result.to_json())
        return result


def _validate_profile_attestation(
    run: Mapping[str, object], *, expected_profile: str
) -> str:
    attestation = _require_mapping(
        run.get("runtime_profile_attestation"),
        field="runtime_profile_attestation",
    )
    if attestation.get("profile") != expected_profile:
        raise ValueError("runtime profile attestation arm drift")
    if attestation.get("scope") != "evidence-only":
        raise ValueError("runtime profile attestation scope drift")
    intervention = _require_mapping(
        attestation.get("intervention"), field="intervention"
    )
    if intervention.get("prediction_error_publication") != (
        "active-in-both-arms"
    ):
        raise ValueError("PE publication contract drift")
    expected_drive = expected_profile == GATE1_PE_ON_ARM
    if intervention.get("external_prediction_error_drive") is not expected_drive:
        raise ValueError("PE drive attestation drift")
    if intervention.get("prediction_error_temporal_learning_enabled") is not (
        expected_drive
    ):
        raise ValueError("PE temporal learning attestation drift")
    claimed_sha = attestation.get("attestation_sha256")
    if not isinstance(claimed_sha, str) or len(claimed_sha) != 64:
        raise ValueError("runtime profile attestation lacks SHA-256")
    unhashed = dict(attestation)
    del unhashed["attestation_sha256"]
    actual_sha = hashlib.sha256(_canonical_bytes(unhashed).rstrip(b"\n")).hexdigest()
    if actual_sha != claimed_sha:
        raise ValueError("runtime profile attestation SHA-256 mismatch")
    return claimed_sha


def _arm_readout(
    *,
    case: SevenDayExperimentCase,
    arm_label: str,
    run_value: Mapping[str, object],
    n_plus_one_contract: Mapping[str, object],
) -> Gate1ArmReadout:
    if run_value.get("schema_version") != "seven-day-companion-run.v1":
        raise ValueError("seven-day run schema drift")
    if run_value.get("arm_label") != arm_label:
        raise ValueError("Gate 1 run arm drift")
    if run_value.get("scenario_id") != case.scenario_id:
        raise ValueError("Gate 1 run scenario drift")
    if run_value.get("paraphrase_seed") != case.paraphrase_seed:
        raise ValueError("Gate 1 run seed drift")
    if run_value.get("process_restart_count") != 6:
        raise ValueError("Gate 1 run lacks six process restarts")
    if run_value.get("all_restarts_exact") is not True:
        raise ValueError("Gate 1 run restart attestation failed")
    if run_value.get("production_promotion_authorized") is not False:
        raise ValueError("Gate 1 automated run may not authorize production")
    profile_sha = _validate_profile_attestation(
        run_value, expected_profile=arm_label
    )
    days = run_value.get("days")
    if not isinstance(days, (list, tuple)) or len(days) != 7:
        raise ValueError("Gate 1 run must contain seven days")
    pe_by_day: dict[int, list[float]] = {day: [] for day in range(1, 8)}
    nonbootstrap_count = 0
    world_applied = 0
    self_applied = 0
    final_day: Mapping[str, object] | None = None
    for day_index, raw_day in enumerate(days, start=1):
        day = _require_mapping(raw_day, field=f"day-{day_index}")
        if day.get("day_index") != day_index:
            raise ValueError("Gate 1 run day order drift")
        turns = day.get("turns")
        if not isinstance(turns, (list, tuple)) or len(turns) != 5:
            raise ValueError("Gate 1 day must contain five turns")
        for turn_index, raw_turn in enumerate(turns, start=1):
            turn = _require_mapping(
                raw_turn, field=f"day-{day_index}.turn-{turn_index}"
            )
            magnitude = _finite_number(
                turn.get("pe_magnitude"), field="pe_magnitude"
            )
            if magnitude < 0.0:
                raise ValueError("pe_magnitude must be non-negative")
            bootstrap = turn.get("pe_bootstrap")
            world_value = turn.get(
                "world_temporal_prediction_error_applied"
            )
            self_value = turn.get(
                "self_temporal_prediction_error_applied"
            )
            if not all(
                isinstance(value, bool)
                for value in (bootstrap, world_value, self_value)
            ):
                raise ValueError("Gate 1 turn mechanism flags must be bool")
            pe_by_day[day_index].append(magnitude)
            if not bootstrap:
                nonbootstrap_count += 1
                world_applied += int(world_value)
                self_applied += int(self_value)
        if day_index == 7:
            final_day = day
    assert final_day is not None
    early_pe = statistics.fmean((*pe_by_day[1], *pe_by_day[2]))
    late_pe = statistics.fmean((*pe_by_day[6], *pe_by_day[7]))
    final_metrics = _require_mapping(
        final_day.get("continuity_metrics"), field="day-7.continuity_metrics"
    )
    n_plus_one = validate_seven_day_n_plus_one_evidence(
        run=run_value,
        contract=n_plus_one_contract,
    )
    return Gate1ArmReadout(
        case_id=case.case_id,
        arm_label=arm_label,
        pe_observation_count=sum(len(values) for values in pe_by_day.values()),
        nonbootstrap_observation_count=nonbootstrap_count,
        world_temporal_applied_count=world_applied,
        self_temporal_applied_count=self_applied,
        early_pe_mean=early_pe,
        late_pe_mean=late_pe,
        pe_adaptation=early_pe - late_pe,
        final_day_continuity_composite=_continuity_composite(
            final_metrics, field="day-7.continuity_metrics"
        ),
        n_plus_one_prediction_quality=(
            n_plus_one.heldout_mean_cosine_similarity
        ),
        n_plus_one_cosine_error=n_plus_one.heldout_mean_cosine_error,
        n_plus_one_mean_squared_error=n_plus_one.heldout_mean_squared_error,
        n_plus_one_persistence_quality=(
            n_plus_one.heldout_persistence_mean_cosine_similarity
        ),
        n_plus_one_target_sequence_sha256=(
            n_plus_one.target_sequence_sha256
        ),
        n_plus_one_target_snapshot_fingerprint=(
            n_plus_one.target_snapshot_fingerprint
        ),
        final_day_boundary_violation_rate=_finite_number(
            final_metrics.get("boundary_violation_rate"),
            field="day-7.boundary_violation_rate",
        ),
        final_day_wrong_user_attribution_rate=_finite_number(
            final_metrics.get("wrong_user_attribution_rate"),
            field="day-7.wrong_user_attribution_rate",
        ),
        runtime_profile_attestation_sha256=profile_sha,
    )


def evaluate_gate1_seven_day_runs(
    *,
    cases: Sequence[SevenDayExperimentCase],
    runs: Mapping[tuple[str, str], Mapping[str, object]],
    preregistration: Mapping[str, object],
    evidence_tier: str = "formal",
) -> Gate1SevenDayResult:
    if preregistration.get("schema_version") != (
        "gate1-seven-day-companion-prereg.v2"
    ):
        raise ValueError("Gate 1 preregistration schema drift")
    if evidence_tier not in {"formal", "smoke"}:
        raise ValueError("Gate 1 evidence_tier must be formal or smoke")
    if evidence_tier == "smoke" and len(cases) != 1:
        raise ValueError("Gate 1 smoke requires exactly one matched pair")
    expected_keys = {
        (case.case_id, arm_label)
        for case in cases
        for arm_label in GATE1_SEVEN_DAY_ARMS
    }
    if set(runs) != expected_keys:
        raise ValueError("Gate 1 run matrix is incomplete or contains extras")
    formal = _require_mapping(
        preregistration.get("formal_run"), field="formal_run"
    )
    if evidence_tier == "formal":
        if formal.get("run_count") != len(expected_keys):
            raise ValueError("Gate 1 preregistered run count drift")
        if formal.get("pair_count") != len(cases):
            raise ValueError("Gate 1 preregistered pair count drift")
    thresholds = _require_mapping(
        preregistration.get("minimum_effects"), field="minimum_effects"
    )
    pe_min = _finite_number(
        thresholds.get("pe_adaptation_gain"), field="pe_adaptation_gain"
    )
    n_plus_one_min = _finite_number(
        thresholds.get("n_plus_one_prediction_quality_gain"),
        field="n_plus_one_prediction_quality_gain",
    )
    safety_margin = _finite_number(
        thresholds.get("maximum_safety_regression"),
        field="maximum_safety_regression",
    )
    n_plus_one_contract = _require_mapping(
        preregistration.get("n_plus_one_measurement"),
        field="n_plus_one_measurement",
    )
    readouts = tuple(
        _arm_readout(
            case=case,
            arm_label=arm_label,
            run_value=runs[(case.case_id, arm_label)],
            n_plus_one_contract=n_plus_one_contract,
        )
        for case in cases
        for arm_label in GATE1_SEVEN_DAY_ARMS
    )
    by_key = {(item.case_id, item.arm_label): item for item in readouts}
    pe_gains = []
    continuity_gains = []
    n_plus_one_gains = []
    safety_regressions = []
    for case in cases:
        pe_on = by_key[(case.case_id, GATE1_PE_ON_ARM)]
        pe_off = by_key[(case.case_id, GATE1_PE_OFF_ARM)]
        pe_gains.append(pe_on.pe_adaptation - pe_off.pe_adaptation)
        if (
            pe_on.final_day_continuity_composite is not None
            and pe_off.final_day_continuity_composite is not None
        ):
            continuity_gains.append(
                pe_on.final_day_continuity_composite
                - pe_off.final_day_continuity_composite
            )
        n_plus_one_gains.append(
            pe_on.n_plus_one_prediction_quality
            - pe_off.n_plus_one_prediction_quality
        )
        if (
            pe_on.n_plus_one_target_sequence_sha256
            != pe_off.n_plus_one_target_sequence_sha256
            or pe_on.n_plus_one_target_snapshot_fingerprint
            != pe_off.n_plus_one_target_snapshot_fingerprint
        ):
            raise ValueError("Gate 1 N+1 target differs across matched arms")
        safety_regressions.extend(
            (
                pe_on.final_day_boundary_violation_rate
                - pe_off.final_day_boundary_violation_rate,
                pe_on.final_day_wrong_user_attribution_rate
                - pe_off.final_day_wrong_user_attribution_rate,
            )
        )
    pe_mean = statistics.fmean(pe_gains)
    continuity_mean = (
        statistics.fmean(continuity_gains) if continuity_gains else None
    )
    n_plus_one_mean = statistics.fmean(n_plus_one_gains)
    pe_ci = _paired_ci95(pe_gains)
    continuity_ci = _paired_ci95(continuity_gains)
    n_plus_one_ci = _paired_ci95(n_plus_one_gains)
    pe_on_rows = tuple(
        item for item in readouts if item.arm_label == GATE1_PE_ON_ARM
    )
    pe_off_rows = tuple(
        item for item in readouts if item.arm_label == GATE1_PE_OFF_ARM
    )
    gates = {
        "matrix-complete": len(readouts) == len(expected_keys),
        "pe-publication-complete": all(
            item.pe_observation_count == 35 for item in readouts
        ),
        "pe-on-temporal-path-load-bearing": all(
            item.nonbootstrap_observation_count > 0
            and item.world_temporal_applied_count
            == item.nonbootstrap_observation_count
            and item.self_temporal_applied_count
            == item.nonbootstrap_observation_count
            for item in pe_on_rows
        ),
        "pe-off-temporal-path-closed": all(
            item.world_temporal_applied_count == 0
            and item.self_temporal_applied_count == 0
            for item in pe_off_rows
        ),
        "pe-adaptation-minimum-effect": pe_mean >= pe_min,
        "pe-adaptation-ci-positive": pe_ci is not None and pe_ci[0] > 0.0,
        "n-plus-one-minimum-effect": n_plus_one_mean >= n_plus_one_min,
        "n-plus-one-ci-positive": (
            n_plus_one_ci is not None and n_plus_one_ci[0] > 0.0
        ),
        "safety-noninferior": max(safety_regressions) <= safety_margin,
    }
    mechanism_supported = all(
        gates[name]
        for name in (
            "matrix-complete",
            "pe-publication-complete",
            "pe-on-temporal-path-load-bearing",
            "pe-off-temporal-path-closed",
        )
    )
    causal_supported = evidence_tier == "formal" and mechanism_supported and all(
        gates[name]
        for name in (
            "pe-adaptation-minimum-effect",
            "pe-adaptation-ci-positive",
            "n-plus-one-minimum-effect",
            "n-plus-one-ci-positive",
            "safety-noninferior",
        )
    )
    prereg_sha = hashlib.sha256(_canonical_bytes(preregistration)).hexdigest()
    return Gate1SevenDayResult(
        schema_version=(
            GATE1_SEVEN_DAY_SCHEMA_VERSION
            if evidence_tier == "formal"
            else GATE1_SEVEN_DAY_SMOKE_SCHEMA_VERSION
        ),
        evidence_tier=evidence_tier,
        preregistration_sha256=prereg_sha,
        run_count=len(readouts),
        pair_count=len(cases),
        readouts=readouts,
        pe_adaptation_gain_mean=pe_mean,
        pe_adaptation_gain_ci95=pe_ci,
        final_day_continuity_gain_mean=continuity_mean,
        final_day_continuity_gain_ci95=continuity_ci,
        n_plus_one_prediction_gain_mean=n_plus_one_mean,
        n_plus_one_prediction_gain_ci95=n_plus_one_ci,
        gates=gates,
        mechanism_supported=mechanism_supported,
        causal_supported=causal_supported,
        claim_scope="simulated-seven-day-product-ecology-only",
        production_promotion_authorized=False,
    )


__all__ = [
    "GATE1_PE_OFF_ARM",
    "GATE1_PE_ON_ARM",
    "GATE1_SEVEN_DAY_ARMS",
    "GATE1_SEVEN_DAY_SCHEMA_VERSION",
    "GATE1_SEVEN_DAY_SMOKE_SCHEMA_VERSION",
    "Gate1ArmReadout",
    "Gate1SevenDayHarness",
    "Gate1SevenDayResult",
    "evaluate_gate1_seven_day_runs",
]
