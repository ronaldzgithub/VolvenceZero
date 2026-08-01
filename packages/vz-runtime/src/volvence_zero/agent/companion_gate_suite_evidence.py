"""Matched seven-day product-path evidence for Gates 4/5/6/7/9/10.

The evaluator consumes only frozen HTTP run artifacts.  It never writes to a
runtime owner and it never authorizes production promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Protocol, Sequence

from volvence_zero.agent.seven_day_companion_evidence import (
    SEVEN_DAY_METRICS,
    SevenDayExperimentCase,
)


GATE_SUITE_SCHEMA_VERSION = "companion-gate-suite-seven-day.v1"
GATE_ARM_SCHEDULES: Mapping[int, tuple[str, ...]] = {
    4: ("gate4-active-selector-v1", "gate4-random-feedback-v1"),
    5: ("gate5-multifrequency-cms-v1", "gate5-single-timescale-v1"),
    6: ("gate6-conditioned-meta-init-v1", "gate6-copy-init-v1"),
    7: ("gate7-ssl-rl-full-v1", "gate7-no-ssl-v1", "gate7-no-rl-v1"),
    9: ("gate9-m3-slow-on-v1", "gate9-m3-slow-off-v1"),
    10: ("gate10-rare-heavy-import-v1", "gate10-rare-heavy-review-v1"),
}
GATE_TREATMENT_ARMS: Mapping[int, str] = {gate_id: arms[0] for gate_id, arms in GATE_ARM_SCHEDULES.items()}
GATE_PRIMARY_MINIMUMS: Mapping[int, float] = {
    4: 0.02,
    5: 0.02,
    6: 0.02,
    7: 0.02,
    9: 0.02,
    10: 0.02,
}

_NEGATIVE_CONTINUITY_METRICS = frozenset(
    {
        "boundary_violation_rate",
        "wrong_user_attribution_rate",
        "user_correction_rate",
    }
)


class GateSuiteArmExecutor(Protocol):
    def execute(
        self,
        *,
        case: SevenDayExperimentCase,
        arm_label: str,
        drain_slow_loop: bool,
        output_path: Path,
    ) -> Mapping[str, object]: ...


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


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _pairs(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be key/value pairs")
    output: dict[str, object] = {}
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[0], str):
            raise ValueError(f"{field} contains an invalid pair")
        if item[0] in output:
            raise ValueError(f"{field} contains duplicate key {item[0]!r}")
        output[item[0]] = item[1]
    return output


def _continuity_composite(value: object, *, field: str) -> float:
    metrics = _mapping(value, field=field)
    normalized = []
    for name in SEVEN_DAY_METRICS:
        item = _number(metrics.get(name), field=f"{field}.{name}")
        if name == "seven_day_trust_delta":
            if not -1.0 <= item <= 1.0:
                raise ValueError(f"{field}.{name} is outside [-1, 1]")
            normalized.append((item + 1.0) / 2.0)
        else:
            if not 0.0 <= item <= 1.0:
                raise ValueError(f"{field}.{name} is outside [0, 1]")
            normalized.append(1.0 - item if name in _NEGATIVE_CONTINUITY_METRICS else item)
    return statistics.fmean(normalized)


def _ci95(values: Sequence[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return (mean - half, mean + half)


def _profile_sha(
    run: Mapping[str, object],
    *,
    expected_arm: str,
    expected_contract: Mapping[str, object],
    gate_id: int,
) -> str:
    profile = _mapping(
        run.get("runtime_profile_attestation"),
        field="runtime_profile_attestation",
    )
    if profile.get("profile") != expected_arm:
        raise ValueError("runtime profile arm drift")
    intervention = _mapping(profile.get("intervention"), field="profile.intervention")
    if intervention.get("gate_id") != gate_id:
        raise ValueError("runtime profile gate drift")
    if dict(intervention) != dict(expected_contract):
        raise ValueError("runtime profile intervention differs from preregistration")
    claimed = profile.get("attestation_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("runtime profile lacks SHA-256")
    unhashed = dict(profile)
    del unhashed["attestation_sha256"]
    actual = hashlib.sha256(_canonical_bytes(unhashed).rstrip(b"\n")).hexdigest()
    if actual != claimed:
        raise ValueError("runtime profile SHA-256 mismatch")
    return claimed


@dataclass(frozen=True)
class GateSuiteArmReadout:
    case_id: str
    arm_label: str
    primary_score: float
    final_day_continuity_composite: float
    boundary_violation_rate: float
    wrong_user_attribution_rate: float
    mechanism_counts: tuple[tuple[str, int], ...]
    runtime_profile_attestation_sha256: str


@dataclass(frozen=True)
class GateSuiteComparison:
    control_arm: str
    primary_gain_mean: float
    primary_gain_ci95: tuple[float, float] | None
    continuity_gain_mean: float
    continuity_gain_ci95: tuple[float, float] | None
    safety_regression_max: float


@dataclass(frozen=True)
class GateSuiteResult:
    schema_version: str
    gate_id: int
    preregistration_sha256: str
    arm_schedule: tuple[str, ...]
    run_count: int
    pair_count: int
    readouts: tuple[GateSuiteArmReadout, ...]
    comparisons: tuple[GateSuiteComparison, ...]
    gates: Mapping[str, bool]
    mechanism_supported: bool
    causal_supported: bool
    claim_scope: str
    production_promotion_authorized: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _gate4_score(
    turns: Sequence[tuple[int, Mapping[str, object], Mapping[str, object]]],
) -> tuple[float, dict[str, int]]:
    requests = 0
    useful = 0
    active = 0
    apprentice = 0
    utility = 0.0
    for _day, turn, telemetry in turns:
        is_request = telemetry.get("feedback_requested") is True
        tags = turn.get("event_tags")
        if not isinstance(tags, (list, tuple)):
            raise ValueError("Gate 4 turn event_tags are invalid")
        opportunity = bool({"boundary", "callback"}.intersection(tags))
        requests += int(is_request)
        useful += int(is_request and opportunity)
        active += int(telemetry.get("apprenticeship_active") is True)
        apprentice += int(telemetry.get("environment_trigger_kind") == "apprentice")
        if is_request:
            utility += 1.0 if opportunity else -0.25
    return utility / len(turns), {
        "feedback_requests": requests,
        "useful_feedback_requests": useful,
        "apprenticeship_active_turns": active,
        "apprentice_trigger_turns": apprentice,
    }


def _gate5_score(
    turns: Sequence[tuple[int, Mapping[str, object], Mapping[str, object]]],
) -> tuple[float, dict[str, int]]:
    late_scores = []
    counts = {
        "nested_variant_turns": 0,
        "independent_variant_turns": 0,
        "atlas_active_turns": 0,
        "pe_gate_active_turns": 0,
        "positive_observation_turns": 0,
    }
    for day, _turn, telemetry in turns:
        variant = telemetry.get("cms_variant")
        counts["nested_variant_turns"] += int(variant == "nested")
        counts["independent_variant_turns"] += int(variant == "independent")
        counts["atlas_active_turns"] += int(telemetry.get("cms_atlas_replay_active") is True)
        counts["pe_gate_active_turns"] += int(telemetry.get("cms_pe_gate_active") is True)
        observations = telemetry.get("cms_total_observations")
        counts["positive_observation_turns"] += int(isinstance(observations, int) and observations > 0)
        if day >= 6:
            absorption = _number(
                telemetry.get("cms_new_knowledge_absorption"),
                field="cms_new_knowledge_absorption",
            )
            retention = _number(
                telemetry.get("cms_old_knowledge_retention"),
                field="cms_old_knowledge_retention",
            )
            late_scores.append((absorption + retention) / 2.0)
    return statistics.fmean(late_scores), counts


def _gate6_score(
    days: Sequence[Mapping[str, object]],
    turns: Sequence[tuple[int, Mapping[str, object], Mapping[str, object]]],
) -> tuple[float, dict[str, int]]:
    first_turn_pe = []
    reset_count = 0
    positive_alignment = 0
    for day_index, day in enumerate(days, start=1):
        end = _pairs(
            day.get("end_scene_gate_telemetry"),
            field=f"day-{day_index}.end_scene_gate_telemetry",
        )
        reset_count += int(end.get("nested_context_reset_applied") is True)
        positive_alignment += int(
            _number(
                end.get("slow_to_fast_target_alignment_gain"),
                field="slow_to_fast_target_alignment_gain",
            )
            > 0.0
        )
    for day, turn, _telemetry in turns:
        if day >= 2 and turn.get("exchange_index") == 1:
            first_turn_pe.append(_number(turn.get("pe_magnitude"), field="pe_magnitude"))
    return -statistics.fmean(first_turn_pe), {
        "nested_context_resets": reset_count,
        "positive_target_alignment_resets": positive_alignment,
    }


def _gate7_score(
    turns: Sequence[tuple[int, Mapping[str, object], Mapping[str, object]]],
) -> tuple[float, dict[str, int]]:
    rewards: dict[str, list[float]] = {"early": [], "late": []}
    counts = {
        "joint_cycles": 0,
        "ssl_rollbacks": 0,
        "policy_updates": 0,
        "runtime_replay_active_turns": 0,
        "runtime_replay_transition_turns": 0,
    }
    for day, _turn, telemetry in turns:
        if telemetry.get("joint_cycle_executed") is True:
            counts["joint_cycles"] += 1
            counts["ssl_rollbacks"] += int(telemetry.get("ssl_rollback_applied") is True)
            counts["policy_updates"] += int(telemetry.get("internal_rl_policy_update_applied") is True)
            reward = _number(
                telemetry.get("internal_rl_total_reward"),
                field="internal_rl_total_reward",
            )
            if day <= 2:
                rewards["early"].append(reward)
            if day >= 6:
                rewards["late"].append(reward)
        counts["runtime_replay_active_turns"] += int(telemetry.get("runtime_replay_wiring") == "active")
        transitions = telemetry.get("runtime_replay_transition_count")
        counts["runtime_replay_transition_turns"] += int(isinstance(transitions, int) and transitions > 0)
    if not rewards["early"] or not rewards["late"]:
        raise ValueError("Gate 7 lacks early/late joint-cycle rewards")
    return (statistics.fmean(rewards["late"]) - statistics.fmean(rewards["early"])), counts


def _gate9_score(
    turns: Sequence[tuple[int, Mapping[str, object], Mapping[str, object]]],
) -> tuple[float, dict[str, int]]:
    losses: dict[str, list[float]] = {"early": [], "late": []}
    gains = set()
    positive_slow_signal = 0
    cycles = 0
    for day, _turn, telemetry in turns:
        gains.add(
            _number(
                telemetry.get("ssl_m3_slow_gain"),
                field="ssl_m3_slow_gain",
            )
        )
        positive_slow_signal += int(
            _number(
                telemetry.get("ssl_m3_slow_momentum_norm"),
                field="ssl_m3_slow_momentum_norm",
            )
            > 0.0
        )
        if telemetry.get("joint_cycle_executed") is True:
            cycles += 1
            loss = _number(
                telemetry.get("ssl_prediction_loss"),
                field="ssl_prediction_loss",
            )
            if day <= 2:
                losses["early"].append(loss)
            if day >= 6:
                losses["late"].append(loss)
    if not losses["early"] or not losses["late"]:
        raise ValueError("Gate 9 lacks early/late SSL losses")
    gain_x1000 = int(round(next(iter(gains)) * 1000)) if len(gains) == 1 else -1
    return (statistics.fmean(losses["early"]) - statistics.fmean(losses["late"])), {
        "joint_cycles": cycles,
        "positive_slow_signal_turns": positive_slow_signal,
        "configured_slow_gain_x1000": gain_x1000,
    }


def _gate10_score(
    turns: Sequence[tuple[int, Mapping[str, object], Mapping[str, object]]],
) -> tuple[float, dict[str, int]]:
    pe: dict[str, list[float]] = {"early": [], "late": []}
    counts = {
        "rare_heavy_recommendations": 0,
        "rare_heavy_imports": 0,
        "rare_heavy_review_only": 0,
        "pre_import_passes": 0,
    }
    for day, turn, telemetry in turns:
        magnitude = _number(turn.get("pe_magnitude"), field="pe_magnitude")
        if day <= 2:
            pe["early"].append(magnitude)
        if day >= 6:
            pe["late"].append(magnitude)
        counts["rare_heavy_recommendations"] += int(telemetry.get("rare_heavy_recommended") is True)
        counts["rare_heavy_imports"] += int(telemetry.get("rare_heavy_applied") is True)
        counts["rare_heavy_review_only"] += int(telemetry.get("rare_heavy_import_decision") == "blocked-by-doctrine")
        counts["pre_import_passes"] += int(telemetry.get("rare_heavy_pre_import_passed") is True)
    return statistics.fmean(pe["early"]) - statistics.fmean(pe["late"]), counts


def _arm_readout(
    *,
    gate_id: int,
    case: SevenDayExperimentCase,
    arm: str,
    run: Mapping[str, object],
    profile_contract: Mapping[str, object],
) -> GateSuiteArmReadout:
    if run.get("schema_version") != "seven-day-companion-run.v1":
        raise ValueError("seven-day run schema drift")
    if (
        run.get("arm_label") != arm
        or run.get("scenario_id") != case.scenario_id
        or run.get("paraphrase_seed") != case.paraphrase_seed
    ):
        raise ValueError("seven-day gate run identity drift")
    if (
        run.get("process_restart_count") != 6
        or run.get("all_restarts_exact") is not True
        or run.get("production_promotion_authorized") is not False
    ):
        raise ValueError("seven-day gate lifecycle/authorization drift")
    sha = _profile_sha(
        run,
        expected_arm=arm,
        expected_contract=profile_contract,
        gate_id=gate_id,
    )
    raw_days = run.get("days")
    if not isinstance(raw_days, (list, tuple)) or len(raw_days) != 7:
        raise ValueError("seven-day gate run must contain seven days")
    days = tuple(_mapping(day, field=f"day-{index}") for index, day in enumerate(raw_days, start=1))
    turns: list[tuple[int, Mapping[str, object], Mapping[str, object]]] = []
    for day_index, day in enumerate(days, start=1):
        raw_turns = day.get("turns")
        if not isinstance(raw_turns, (list, tuple)) or len(raw_turns) != 5:
            raise ValueError("seven-day gate day must contain five turns")
        for turn_index, raw_turn in enumerate(raw_turns, start=1):
            turn = _mapping(raw_turn, field=f"day-{day_index}.turn-{turn_index}")
            telemetry = _pairs(
                turn.get("gate_telemetry"),
                field=f"day-{day_index}.turn-{turn_index}.gate_telemetry",
            )
            turns.append((day_index, turn, telemetry))
    if gate_id == 4:
        primary, counts = _gate4_score(turns)
    elif gate_id == 5:
        primary, counts = _gate5_score(turns)
    elif gate_id == 6:
        primary, counts = _gate6_score(days, turns)
    elif gate_id == 7:
        primary, counts = _gate7_score(turns)
    elif gate_id == 9:
        primary, counts = _gate9_score(turns)
    elif gate_id == 10:
        primary, counts = _gate10_score(turns)
    else:  # pragma: no cover - guarded by the public entry point
        raise ValueError(f"unsupported gate {gate_id}")
    final_metrics = _mapping(
        days[-1].get("continuity_metrics"),
        field="day-7.continuity_metrics",
    )
    return GateSuiteArmReadout(
        case_id=case.case_id,
        arm_label=arm,
        primary_score=primary,
        final_day_continuity_composite=_continuity_composite(final_metrics, field="day-7.continuity_metrics"),
        boundary_violation_rate=_number(
            final_metrics.get("boundary_violation_rate"),
            field="boundary_violation_rate",
        ),
        wrong_user_attribution_rate=_number(
            final_metrics.get("wrong_user_attribution_rate"),
            field="wrong_user_attribution_rate",
        ),
        mechanism_counts=tuple(sorted(counts.items())),
        runtime_profile_attestation_sha256=sha,
    )


def _mechanism_passed(*, gate_id: int, arm: str, counts: Mapping[str, int]) -> bool:
    if gate_id == 4:
        return (
            counts["apprenticeship_active_turns"] == 35
            and counts["apprentice_trigger_turns"] == 35
            and counts["feedback_requests"] > 0
        )
    if gate_id == 5:
        expected = "nested_variant_turns" if arm == GATE_TREATMENT_ARMS[5] else "independent_variant_turns"
        return (
            counts[expected] == 35
            and counts["atlas_active_turns"] == 35
            and counts["pe_gate_active_turns"] == 35
            and counts["positive_observation_turns"] > 0
        )
    if gate_id == 6:
        if counts["nested_context_resets"] != 7:
            return False
        return (
            counts["positive_target_alignment_resets"] > 0
            if arm == GATE_TREATMENT_ARMS[6]
            else counts["positive_target_alignment_resets"] == 0
        )
    if gate_id == 7:
        if counts["joint_cycles"] == 0:
            return False
        if arm == "gate7-no-ssl-v1":
            return counts["ssl_rollbacks"] > 0
        if arm == "gate7-no-rl-v1":
            return counts["policy_updates"] == 0
        return (
            counts["policy_updates"] > 0
            and counts["runtime_replay_active_turns"] > 0
            and counts["runtime_replay_transition_turns"] > 0
        )
    if gate_id == 9:
        expected_gain = 1000 if arm == GATE_TREATMENT_ARMS[9] else 0
        return (
            counts["joint_cycles"] > 0
            and counts["positive_slow_signal_turns"] > 0
            and counts["configured_slow_gain_x1000"] == expected_gain
        )
    if gate_id == 10:
        if counts["rare_heavy_recommendations"] == 0:
            return False
        if arm == GATE_TREATMENT_ARMS[10]:
            return counts["rare_heavy_imports"] > 0
        return counts["rare_heavy_imports"] == 0 and counts["rare_heavy_review_only"] > 0
    return False


def evaluate_companion_gate_suite(
    *,
    gate_id: int,
    cases: Sequence[SevenDayExperimentCase],
    runs: Mapping[tuple[str, str], Mapping[str, object]],
    preregistration: Mapping[str, object],
) -> GateSuiteResult:
    if gate_id not in GATE_ARM_SCHEDULES:
        raise ValueError(f"unsupported seven-day gate {gate_id}")
    if (
        preregistration.get("schema_version") != ("companion-gate-suite-seven-day-prereg.v1")
        or preregistration.get("gate_id") != gate_id
    ):
        raise ValueError("gate-suite preregistration drift")
    arms = GATE_ARM_SCHEDULES[gate_id]
    expected_keys = {(case.case_id, arm) for case in cases for arm in arms}
    if set(runs) != expected_keys:
        raise ValueError("gate-suite run matrix incomplete or contains extras")
    profile_contracts = _mapping(preregistration.get("profile_contracts"), field="profile_contracts")
    if set(profile_contracts) != set(arms):
        raise ValueError("gate-suite profile contract schedule drift")
    readouts = tuple(
        _arm_readout(
            gate_id=gate_id,
            case=case,
            arm=arm,
            run=runs[(case.case_id, arm)],
            profile_contract=_mapping(profile_contracts[arm], field=f"profile_contracts.{arm}"),
        )
        for case in cases
        for arm in arms
    )
    by_key = {(row.case_id, row.arm_label): row for row in readouts}
    treatment = GATE_TREATMENT_ARMS[gate_id]
    comparisons = []
    for control in arms[1:]:
        primary = [
            by_key[(case.case_id, treatment)].primary_score - by_key[(case.case_id, control)].primary_score
            for case in cases
        ]
        continuity = [
            by_key[(case.case_id, treatment)].final_day_continuity_composite
            - by_key[(case.case_id, control)].final_day_continuity_composite
            for case in cases
        ]
        safety = [
            max(
                by_key[(case.case_id, treatment)].boundary_violation_rate
                - by_key[(case.case_id, control)].boundary_violation_rate,
                by_key[(case.case_id, treatment)].wrong_user_attribution_rate
                - by_key[(case.case_id, control)].wrong_user_attribution_rate,
            )
            for case in cases
        ]
        comparisons.append(
            GateSuiteComparison(
                control_arm=control,
                primary_gain_mean=statistics.fmean(primary),
                primary_gain_ci95=_ci95(primary),
                continuity_gain_mean=statistics.fmean(continuity),
                continuity_gain_ci95=_ci95(continuity),
                safety_regression_max=max(safety),
            )
        )
    mechanism = all(
        _mechanism_passed(
            gate_id=gate_id,
            arm=row.arm_label,
            counts=dict(row.mechanism_counts),
        )
        for row in readouts
    )
    minimum = _number(
        _mapping(
            preregistration.get("minimum_effects"),
            field="minimum_effects",
        ).get("primary_gain"),
        field="minimum_effects.primary_gain",
    )
    continuity_minimum = _number(
        _mapping(
            preregistration.get("minimum_effects"),
            field="minimum_effects",
        ).get("continuity_gain"),
        field="minimum_effects.continuity_gain",
    )
    gates = {
        "matrix-complete": len(readouts) == len(expected_keys),
        "mechanism-load-bearing": mechanism,
        "primary-minimum-effect-all-controls": all(row.primary_gain_mean >= minimum for row in comparisons),
        "primary-ci-positive-all-controls": all(
            row.primary_gain_ci95 is not None and row.primary_gain_ci95[0] > 0.0 for row in comparisons
        ),
        "continuity-minimum-effect-all-controls": all(
            row.continuity_gain_mean >= continuity_minimum for row in comparisons
        ),
        "continuity-ci-positive-all-controls": all(
            row.continuity_gain_ci95 is not None and row.continuity_gain_ci95[0] > 0.0 for row in comparisons
        ),
        "safety-noninferior-all-controls": all(row.safety_regression_max <= 0.0 for row in comparisons),
    }
    causal = mechanism and all(gates.values())
    return GateSuiteResult(
        schema_version=GATE_SUITE_SCHEMA_VERSION,
        gate_id=gate_id,
        preregistration_sha256=hashlib.sha256(_canonical_bytes(preregistration)).hexdigest(),
        arm_schedule=arms,
        run_count=len(readouts),
        pair_count=len(cases),
        readouts=readouts,
        comparisons=tuple(comparisons),
        gates=gates,
        mechanism_supported=mechanism,
        causal_supported=causal,
        claim_scope="simulated-seven-day-product-ecology-only",
        production_promotion_authorized=False,
    )


class CompanionGateSuiteHarness:
    def __init__(self, *, gate_id: int, executor: GateSuiteArmExecutor) -> None:
        if gate_id not in GATE_ARM_SCHEDULES:
            raise ValueError(f"unsupported seven-day gate {gate_id}")
        self._gate_id = gate_id
        self._executor = executor

    def run(
        self,
        *,
        cases: Sequence[SevenDayExperimentCase],
        preregistration: Mapping[str, object],
        output_dir: str | Path,
    ) -> GateSuiteResult:
        if not cases:
            raise ValueError("gate-suite evidence requires cases")
        target = Path(output_dir)
        run_root = target / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        runs: dict[tuple[str, str], Mapping[str, object]] = {}
        for arm in GATE_ARM_SCHEDULES[self._gate_id]:
            for case in cases:
                output_path = run_root / (
                    hashlib.sha256(f"{case.case_id}\0{arm}".encode("utf-8")).hexdigest() + ".json"
                )
                runs[(case.case_id, arm)] = self._executor.execute(
                    case=case,
                    arm_label=arm,
                    drain_slow_loop=True,
                    output_path=output_path,
                )
        result = evaluate_companion_gate_suite(
            gate_id=self._gate_id,
            cases=cases,
            runs=runs,
            preregistration=preregistration,
        )
        (target / f"gate{self._gate_id}_evaluation.json").write_bytes(_canonical_bytes(result.to_json()))
        return result


__all__ = [
    "GATE_ARM_SCHEDULES",
    "GATE_PRIMARY_MINIMUMS",
    "GATE_SUITE_SCHEMA_VERSION",
    "GATE_TREATMENT_ARMS",
    "CompanionGateSuiteHarness",
    "GateSuiteArmReadout",
    "GateSuiteComparison",
    "GateSuiteResult",
    "evaluate_companion_gate_suite",
]
