"""Read-only evidence harness for seven-day simulated companion ablations.

The executor port produces transcripts through the public product lifecycle.
This module freezes arm scheduling, validates exact matching, computes daily
readouts, and exports an evaluation-only bundle.  It owns no runtime state and
never writes ratings or evaluation results back into learning owners.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Protocol, Sequence


SEVEN_DAY_ABLATION_SCHEMA_VERSION = "seven-day-companion-ablation.v1"
SEVEN_DAY_PREREG_SCHEMA_VERSION = "seven-day-companion-simulated.v1"
SEVEN_DAY_STATE_ARMS = (
    "correct-user-state",
    "stateless",
    "swapped-user-state",
    "shuffled-history",
)
SEVEN_DAY_SLEEP_ARMS = ("sleep-consolidation", "no-sleep")
SEVEN_DAY_ALL_ARMS = (*SEVEN_DAY_STATE_ARMS, *SEVEN_DAY_SLEEP_ARMS)
SEVEN_DAY_METRICS = (
    "callback_hit_rate",
    "boundary_violation_rate",
    "wrong_user_attribution_rate",
    "open_loop_closure_rate",
    "user_correction_rate",
    "remembered_item_usefulness",
    "seven_day_trust_delta",
)
_NEGATIVE_METRICS = frozenset(
    {
        "boundary_violation_rate",
        "wrong_user_attribution_rate",
        "user_correction_rate",
    }
)
_STATE_POLICY_BY_ARM = {
    "correct-user-state": "correct-user-state",
    "stateless": "stateless",
    "swapped-user-state": "swapped-user-state",
    "shuffled-history": "shuffled-history",
    "sleep-consolidation": "correct-user-state",
    "no-sleep": "correct-user-state",
}


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


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value))


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value


def _metric(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric or null")
    result = float(value)
    lower = -1.0 if field.endswith("seven_day_trust_delta") else 0.0
    if not math.isfinite(result) or result < lower or result > 1.0:
        raise ValueError(f"{field} is outside its declared range")
    return result


@dataclass(frozen=True)
class SevenDayExperimentCase:
    scenario_id: str
    paraphrase_seed: int

    def __post_init__(self) -> None:
        _require_string(self.scenario_id, field="scenario_id")
        if self.paraphrase_seed < 0:
            raise ValueError("paraphrase_seed must be non-negative")

    @property
    def case_id(self) -> str:
        return f"{self.scenario_id}:seed-{self.paraphrase_seed}"


@dataclass(frozen=True)
class SevenDayRunEnvelope:
    case: SevenDayExperimentCase
    arm_label: str
    run: Mapping[str, object]


@dataclass(frozen=True)
class SevenDayDailyReadout:
    case_id: str
    arm_label: str
    day_index: int
    phase: str
    callback_opportunity: bool
    metrics: Mapping[str, float | None]
    continuity_composite: float | None
    fsm_probe_pass_rate: float | None


@dataclass(frozen=True)
class SevenDayComparison:
    contrast_id: str
    experimental_arm: str
    control_arm: str
    expected_pair_count: int
    complete_composite_pair_count: int
    complete_callback_pair_count: int
    complete_cold_start_pair_count: int
    final_day_composite_gain_mean: float | None
    final_day_composite_gain_ci95: tuple[float, float] | None
    callback_gain_mean: float | None
    callback_gain_ci95: tuple[float, float] | None
    cold_start_composite_gain_mean: float | None
    cold_start_composite_gain_ci95: tuple[float, float] | None
    fsm_probe_pass_gain_mean: float | None


@dataclass(frozen=True)
class SevenDayAblationResult:
    schema_version: str
    preregistration_sha256: str
    run_count: int
    case_count: int
    daily_readouts: tuple[SevenDayDailyReadout, ...]
    comparisons: tuple[SevenDayComparison, ...]
    gates: Mapping[str, bool]
    passed: bool
    claim_scope: str
    production_promotion_authorized: bool
    evaluation_writeback_allowed: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class SevenDayArmExecutor(Protocol):
    """Product-path executor implemented outside the evaluation owner."""

    def execute(
        self,
        *,
        case: SevenDayExperimentCase,
        arm_label: str,
        drain_slow_loop: bool,
        output_path: Path,
    ) -> Mapping[str, object]: ...


class SevenDayCompanionAblationHarness:
    """Run the frozen six-arm schedule, then evaluate the emitted artifacts."""

    def __init__(self, *, executor: SevenDayArmExecutor) -> None:
        self._executor = executor

    def run(
        self,
        *,
        cases: Sequence[SevenDayExperimentCase],
        preregistration: Mapping[str, object],
        output_dir: str | Path,
    ) -> SevenDayAblationResult:
        if not cases:
            raise ValueError("seven-day ablation requires at least one case")
        target = Path(output_dir)
        run_root = target / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        envelopes = []
        # Arm-major order is part of the state-control protocol: every
        # correct-state archive must exist before swapped/shuffled controls
        # stage their matched donor/reference snapshots.
        for arm in SEVEN_DAY_ALL_ARMS:
            for case in cases:
                output_path = run_root / (
                    hashlib.sha256(
                        f"{case.case_id}\0{arm}".encode("utf-8")
                    ).hexdigest()
                    + ".json"
                )
                payload = self._executor.execute(
                    case=case,
                    arm_label=arm,
                    drain_slow_loop=(arm != "no-sleep"),
                    output_path=output_path,
                )
                envelopes.append(
                    SevenDayRunEnvelope(
                        case=case,
                        arm_label=arm,
                        run=payload,
                    )
                )
        return export_seven_day_ablation_bundle(
            runs=tuple(envelopes),
            preregistration=preregistration,
            output_dir=target,
        )


def _normalized_composite(
    metrics: Mapping[str, float | None],
) -> float | None:
    if any(metrics[name] is None for name in SEVEN_DAY_METRICS):
        return None
    values = []
    for name in SEVEN_DAY_METRICS:
        value = metrics[name]
        assert value is not None
        if name == "seven_day_trust_delta":
            values.append((value + 1.0) / 2.0)
        elif name in _NEGATIVE_METRICS:
            values.append(1.0 - value)
        else:
            values.append(value)
    return statistics.fmean(values)


def _readout(
    *,
    case_id: str,
    arm_label: str,
    day_index: int,
    phase: str,
    raw_metrics: object,
    turns: Sequence[Mapping[str, object]],
) -> SevenDayDailyReadout:
    payload = _require_mapping(raw_metrics, field=f"day {day_index} {phase}")
    metrics = {
        name: _metric(
            payload.get(name), field=f"day {day_index} {phase}.{name}"
        )
        for name in SEVEN_DAY_METRICS
    }
    callback_opportunity = any(
        "callback" in tuple(turn.get("event_tags", ())) for turn in turns
    )
    probe_values = tuple(
        turn.get("fsm_probe_passed")
        for turn in turns
        if turn.get("fsm_probe_passed") is not None
    )
    if any(not isinstance(value, bool) for value in probe_values):
        raise ValueError("fsm_probe_passed must be bool or null")
    probe_rate = (
        sum(bool(value) for value in probe_values) / len(probe_values)
        if probe_values
        else None
    )
    return SevenDayDailyReadout(
        case_id=case_id,
        arm_label=arm_label,
        day_index=day_index,
        phase=phase,
        callback_opportunity=callback_opportunity,
        metrics=metrics,
        continuity_composite=_normalized_composite(metrics),
        fsm_probe_pass_rate=probe_rate,
    )


def _validate_run(
    envelope: SevenDayRunEnvelope,
) -> tuple[SevenDayDailyReadout, ...]:
    run = _require_mapping(envelope.run, field="run")
    if run.get("schema_version") != "seven-day-companion-run.v1":
        raise ValueError("seven-day run schema drift")
    if run.get("arm_label") != envelope.arm_label:
        raise ValueError("seven-day run arm label drift")
    if run.get("scenario_id") != envelope.case.scenario_id:
        raise ValueError("seven-day run scenario drift")
    if run.get("paraphrase_seed") != envelope.case.paraphrase_seed:
        raise ValueError("seven-day run paraphrase seed drift")
    if run.get("process_restart_count") != 6 or run.get(
        "all_restarts_exact"
    ) is not True:
        raise ValueError("seven-day run lacks six exact process restarts")
    if run.get("simulated_longitudinal_only") is not True:
        raise ValueError("seven-day run claim scope drift")
    if run.get("external_human_value_claim_allowed") is not False:
        raise ValueError("automated run may not claim external human value")
    if run.get("production_promotion_authorized") is not False:
        raise ValueError("automated run may not authorize production")
    attestation = _require_mapping(
        run.get("source_attestation"), field="source_attestation"
    )
    simulator_family = _require_string(
        attestation.get("simulator_model_family"),
        field="simulator_model_family",
    ).lower()
    sut_family = _require_string(
        attestation.get("sut_model_family"), field="sut_model_family"
    ).lower()
    if simulator_family == sut_family:
        raise ValueError("simulator and SUT families must differ")
    days = run.get("days")
    if not isinstance(days, (list, tuple)) or len(days) != 7:
        raise ValueError("seven-day run must contain exactly seven days")
    readouts = []
    for expected_day, raw_day in enumerate(days, start=1):
        day = _require_mapping(raw_day, field="day")
        if day.get("day_index") != expected_day:
            raise ValueError("seven-day run day order drift")
        turns = day.get("turns")
        if not isinstance(turns, (list, tuple)) or len(turns) != 5:
            raise ValueError("seven-day run must contain five exchanges per day")
        typed_turns = tuple(
            _require_mapping(turn, field="turn") for turn in turns
        )
        drained = day.get("end_scene_slow_loop_drained")
        expected_drained = envelope.arm_label != "no-sleep"
        if drained is not expected_drained:
            raise ValueError("sleep arm end-scene intervention drift")
        restart = day.get("restart_after_day")
        if expected_day < 7:
            restart_payload = _require_mapping(
                restart, field="restart_after_day"
            )
            intervention = _require_mapping(
                restart_payload.get("state_intervention"),
                field="state_intervention",
            )
            expected_policy = _STATE_POLICY_BY_ARM[envelope.arm_label]
            if (
                intervention.get("experiment_arm_label")
                != envelope.arm_label
                or intervention.get("state_loading_policy")
                != expected_policy
                or intervention.get("after_day_index") != expected_day
            ):
                raise ValueError("state intervention attestation drift")
            archived_digest = intervention.get("archived_state_sha256")
            if not isinstance(archived_digest, str) or len(archived_digest) != 64:
                raise ValueError("state archive digest is missing")
            source_day = intervention.get("next_day_source_day_index")
            loaded_digest = intervention.get("next_day_loaded_state_sha256")
            if expected_policy == "stateless":
                if source_day is not None or loaded_digest is not None:
                    raise ValueError("stateless arm staged prior state")
            elif (
                isinstance(source_day, bool)
                or not isinstance(source_day, int)
                or source_day < 1
                or source_day > expected_day
                or not isinstance(loaded_digest, str)
                or len(loaded_digest) != 64
            ):
                raise ValueError("stateful arm source attestation drift")
        elif restart is not None:
            raise ValueError("day seven may not contain restart evidence")
        readouts.extend(
            (
                _readout(
                    case_id=envelope.case.case_id,
                    arm_label=envelope.arm_label,
                    day_index=expected_day,
                    phase="cold_start",
                    raw_metrics=day.get("cold_start_continuity_metrics"),
                    turns=typed_turns,
                ),
                _readout(
                    case_id=envelope.case.case_id,
                    arm_label=envelope.arm_label,
                    day_index=expected_day,
                    phase="end_of_day",
                    raw_metrics=day.get("continuity_metrics"),
                    turns=typed_turns,
                ),
            )
        )
    return tuple(readouts)


def _user_turn_digest(run: Mapping[str, object]) -> str:
    days = run["days"]
    assert isinstance(days, (list, tuple))
    texts = []
    for day in days:
        assert isinstance(day, Mapping)
        turns = day["turns"]
        assert isinstance(turns, (list, tuple))
        for turn in turns:
            assert isinstance(turn, Mapping)
            texts.append(turn["user_text"])
    return _sha256(texts)


def _paired_summary(
    values: Sequence[float],
) -> tuple[float | None, tuple[float, float] | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, (mean, mean)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, (mean - half, mean + half)


def _mean_present(values: Sequence[float | None]) -> float | None:
    present = tuple(value for value in values if value is not None)
    return statistics.fmean(present) if present else None


def _comparison(
    *,
    contrast_id: str,
    experimental_arm: str,
    control_arm: str,
    case_ids: Sequence[str],
    by_key: Mapping[tuple[str, str, int, str], SevenDayDailyReadout],
) -> SevenDayComparison:
    composite_gains = []
    callback_gains = []
    cold_start_gains = []
    probe_gains = []
    for case_id in case_ids:
        final_treatment = by_key[(case_id, experimental_arm, 7, "end_of_day")]
        final_control = by_key[(case_id, control_arm, 7, "end_of_day")]
        if (
            final_treatment.continuity_composite is not None
            and final_control.continuity_composite is not None
        ):
            composite_gains.append(
                final_treatment.continuity_composite
                - final_control.continuity_composite
            )
        treatment_callbacks = []
        control_callbacks = []
        treatment_probes = []
        control_probes = []
        for day_index in range(1, 8):
            treatment = by_key[
                (case_id, experimental_arm, day_index, "end_of_day")
            ]
            control = by_key[
                (case_id, control_arm, day_index, "end_of_day")
            ]
            if treatment.callback_opportunity:
                treatment_callbacks.append(
                    treatment.metrics["callback_hit_rate"]
                )
                control_callbacks.append(control.metrics["callback_hit_rate"])
            treatment_probes.append(treatment.fsm_probe_pass_rate)
            control_probes.append(control.fsm_probe_pass_rate)
        treatment_callback = _mean_present(treatment_callbacks)
        control_callback = _mean_present(control_callbacks)
        if treatment_callback is not None and control_callback is not None:
            callback_gains.append(treatment_callback - control_callback)
        treatment_probe = _mean_present(treatment_probes)
        control_probe = _mean_present(control_probes)
        if treatment_probe is not None and control_probe is not None:
            probe_gains.append(treatment_probe - control_probe)
        treatment_cold = _mean_present(
            [
                by_key[
                    (case_id, experimental_arm, day, "cold_start")
                ].continuity_composite
                for day in range(2, 8)
            ]
        )
        control_cold = _mean_present(
            [
                by_key[(case_id, control_arm, day, "cold_start")]
                .continuity_composite
                for day in range(2, 8)
            ]
        )
        if treatment_cold is not None and control_cold is not None:
            cold_start_gains.append(treatment_cold - control_cold)
    composite_mean, composite_ci = _paired_summary(composite_gains)
    callback_mean, callback_ci = _paired_summary(callback_gains)
    cold_mean, cold_ci = _paired_summary(cold_start_gains)
    probe_mean, _probe_ci = _paired_summary(probe_gains)
    return SevenDayComparison(
        contrast_id=contrast_id,
        experimental_arm=experimental_arm,
        control_arm=control_arm,
        expected_pair_count=len(case_ids),
        complete_composite_pair_count=len(composite_gains),
        complete_callback_pair_count=len(callback_gains),
        complete_cold_start_pair_count=len(cold_start_gains),
        final_day_composite_gain_mean=composite_mean,
        final_day_composite_gain_ci95=composite_ci,
        callback_gain_mean=callback_mean,
        callback_gain_ci95=callback_ci,
        cold_start_composite_gain_mean=cold_mean,
        cold_start_composite_gain_ci95=cold_ci,
        fsm_probe_pass_gain_mean=probe_mean,
    )


def evaluate_seven_day_ablation(
    *,
    runs: Sequence[SevenDayRunEnvelope],
    preregistration: Mapping[str, object],
) -> SevenDayAblationResult:
    """Validate exact matching and evaluate all preregistered contrasts."""

    if preregistration.get("schema_version") != SEVEN_DAY_PREREG_SCHEMA_VERSION:
        raise ValueError("seven-day preregistration schema drift")
    prereg_sha = _sha256(preregistration)
    scenario_ids = preregistration.get("scenario_ids")
    formal_run = _require_mapping(
        preregistration.get("formal_run"), field="formal_run"
    )
    seeds = formal_run.get("paraphrase_seeds")
    if not isinstance(scenario_ids, list) or not all(
        isinstance(item, str) and item for item in scenario_ids
    ):
        raise ValueError("preregistration scenario_ids are missing")
    if not isinstance(seeds, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in seeds
    ):
        raise ValueError("preregistration paraphrase seeds are missing")
    thresholds = _require_mapping(
        preregistration.get("minimum_effects"), field="minimum_effects"
    )
    composite_min = _metric(
        thresholds.get("final_day_continuity_composite_gain"),
        field="final_day_continuity_composite_gain",
    )
    callback_min = _metric(
        thresholds.get("callback_hit_rate_gain"),
        field="callback_hit_rate_gain",
    )
    cold_min = _metric(
        thresholds.get("cold_start_continuity_composite_gain"),
        field="cold_start_continuity_composite_gain",
    )
    if None in (composite_min, callback_min, cold_min):
        raise ValueError("minimum effects may not be null")
    case_ids = tuple(sorted({envelope.case.case_id for envelope in runs}))
    planned_case_ids = {
        SevenDayExperimentCase(scenario_id, seed).case_id
        for scenario_id in scenario_ids
        for seed in seeds
    }
    if set(case_ids) != planned_case_ids:
        raise ValueError("formal run case matrix drifted from preregistration")
    expected_keys = {
        (case_id, arm) for case_id in case_ids for arm in SEVEN_DAY_ALL_ARMS
    }
    keyed = {(item.case.case_id, item.arm_label): item for item in runs}
    if len(keyed) != len(runs) or set(keyed) != expected_keys:
        raise ValueError("seven-day arm/case matrix is incomplete or duplicated")
    readouts = []
    for envelope in runs:
        readouts.extend(_validate_run(envelope))
    for case_id in case_ids:
        case_runs = [keyed[(case_id, arm)].run for arm in SEVEN_DAY_ALL_ARMS]
        if len({_user_turn_digest(run) for run in case_runs}) != 1:
            raise ValueError("matched arms do not share exact user turns")
        attestations = [run["source_attestation"] for run in case_runs]
        assert all(isinstance(item, Mapping) for item in attestations)
        matched_fields = (
            "simulator_model_id",
            "simulator_model_family",
            "sut_model_id",
            "sut_model_family",
            "model_and_adapter_fingerprint",
        )
        for field in matched_fields:
            if len({item[field] for item in attestations}) != 1:
                raise ValueError(f"matched arm source {field} drift")
        timestamps = []
        for run in case_runs:
            days = run["days"]
            assert isinstance(days, (list, tuple))
            timestamps.append(
                tuple(day["virtual_observed_at_ms"] for day in days)
            )
        if len(set(timestamps)) != 1:
            raise ValueError("matched arm virtual calendar drift")
    by_readout = {
        (item.case_id, item.arm_label, item.day_index, item.phase): item
        for item in readouts
    }
    comparisons = []
    for control in SEVEN_DAY_STATE_ARMS[1:]:
        comparisons.append(
            _comparison(
                contrast_id=f"correct-user-state-vs-{control}",
                experimental_arm="correct-user-state",
                control_arm=control,
                case_ids=case_ids,
                by_key=by_readout,
            )
        )
    comparisons.append(
        _comparison(
            contrast_id="sleep-consolidation-vs-no-sleep",
            experimental_arm="sleep-consolidation",
            control_arm="no-sleep",
            case_ids=case_ids,
            by_key=by_readout,
        )
    )
    gates: dict[str, bool] = {}
    for comparison in comparisons:
        complete = comparison.expected_pair_count
        gates[f"{comparison.contrast_id}:metric-coverage"] = all(
            count == complete
            for count in (
                comparison.complete_composite_pair_count,
                comparison.complete_callback_pair_count,
                comparison.complete_cold_start_pair_count,
            )
        )
        if comparison.experimental_arm == "sleep-consolidation":
            mean = comparison.cold_start_composite_gain_mean
            interval = comparison.cold_start_composite_gain_ci95
            minimum = cold_min
        else:
            mean = comparison.final_day_composite_gain_mean
            interval = comparison.final_day_composite_gain_ci95
            minimum = composite_min
        gates[f"{comparison.contrast_id}:primary-effect"] = bool(
            mean is not None
            and interval is not None
            and mean >= minimum
            and interval[0] > 0.0
        )
        gates[f"{comparison.contrast_id}:callback-effect"] = bool(
            comparison.callback_gain_mean is not None
            and comparison.callback_gain_ci95 is not None
            and comparison.callback_gain_mean >= callback_min
            and comparison.callback_gain_ci95[0] > 0.0
        )
    return SevenDayAblationResult(
        schema_version=SEVEN_DAY_ABLATION_SCHEMA_VERSION,
        preregistration_sha256=prereg_sha,
        run_count=len(runs),
        case_count=len(case_ids),
        daily_readouts=tuple(readouts),
        comparisons=tuple(comparisons),
        gates=gates,
        passed=all(gates.values()),
        claim_scope="simulated-user-real-lifecycle-only",
        production_promotion_authorized=False,
        evaluation_writeback_allowed=False,
    )


def export_seven_day_ablation_bundle(
    *,
    runs: Sequence[SevenDayRunEnvelope],
    preregistration: Mapping[str, object],
    output_dir: str | Path,
) -> SevenDayAblationResult:
    result = evaluate_seven_day_ablation(
        runs=runs,
        preregistration=preregistration,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    result_payload = result.to_json()
    _write_json(target / "ablation_results.json", result_payload)
    daily_path = target / "daily_metrics.jsonl"
    with daily_path.open("wb") as handle:
        for readout in result.daily_readouts:
            handle.write(_canonical_bytes(asdict(readout)))
    verdict = {
        "schema_version": "seven-day-companion-verdict.v1",
        "passed": result.passed,
        "claim_scope": result.claim_scope,
        "external_human_value_claim_allowed": False,
        "production_promotion_authorized": False,
        "evaluation_writeback_allowed": False,
        "failed_gates": [
            name for name, passed in result.gates.items() if not passed
        ],
    }
    _write_json(target / "promotion_verdict.json", verdict)
    manifest = {
        "schema_version": SEVEN_DAY_ABLATION_SCHEMA_VERSION,
        "preregistration_sha256": result.preregistration_sha256,
        "arm_schedule": list(SEVEN_DAY_ALL_ARMS),
        "case_count": result.case_count,
        "run_count": result.run_count,
        "required_files": [
            "ablation_results.json",
            "daily_metrics.jsonl",
            "promotion_verdict.json",
            "report.md",
        ],
        "claim_scope": result.claim_scope,
    }
    _write_json(target / "manifest.json", manifest)
    report = [
        "# 七天陪伴模拟纵向消融",
        "",
        f"- verdict: `{result.passed}`",
        f"- cases: `{result.case_count}`; runs: `{result.run_count}`",
        "- claim: simulated-user + real lifecycle only",
        "- human value / production promotion: not authorized",
        "- evaluation writeback: forbidden",
        "",
        "## Contrasts",
        "",
    ]
    for comparison in result.comparisons:
        report.append(
            f"- `{comparison.contrast_id}`: final composite "
            f"`{comparison.final_day_composite_gain_mean}`, callback "
            f"`{comparison.callback_gain_mean}`, cold-start "
            f"`{comparison.cold_start_composite_gain_mean}`"
        )
    (target / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def load_seven_day_run_envelopes(
    run_dir: str | Path,
) -> tuple[SevenDayRunEnvelope, ...]:
    """Load the formal ``runs/*.json`` matrix without inferring metadata."""

    root = Path(run_dir)
    paths = tuple(sorted(root.glob("*.json")))
    if not paths:
        raise FileNotFoundError(f"no seven-day run JSON files in {root}")
    envelopes = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = _require_mapping(payload, field=str(path))
        if run.get("schema_version") != "seven-day-companion-run.v1":
            raise ValueError(f"non-run JSON found in formal run directory: {path}")
        scenario_id = _require_string(
            run.get("scenario_id"), field="scenario_id"
        )
        seed = run.get("paraphrase_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("formal run paraphrase_seed is invalid")
        arm_label = _require_string(run.get("arm_label"), field="arm_label")
        envelopes.append(
            SevenDayRunEnvelope(
                case=SevenDayExperimentCase(scenario_id, seed),
                arm_label=arm_label,
                run=run,
            )
        )
    return tuple(envelopes)


__all__ = [
    "SEVEN_DAY_ABLATION_SCHEMA_VERSION",
    "SEVEN_DAY_ALL_ARMS",
    "SEVEN_DAY_METRICS",
    "SEVEN_DAY_PREREG_SCHEMA_VERSION",
    "SEVEN_DAY_SLEEP_ARMS",
    "SEVEN_DAY_STATE_ARMS",
    "SevenDayAblationResult",
    "SevenDayArmExecutor",
    "SevenDayCompanionAblationHarness",
    "SevenDayDailyReadout",
    "SevenDayExperimentCase",
    "SevenDayRunEnvelope",
    "evaluate_seven_day_ablation",
    "export_seven_day_ablation_bundle",
    "load_seven_day_run_envelopes",
]
