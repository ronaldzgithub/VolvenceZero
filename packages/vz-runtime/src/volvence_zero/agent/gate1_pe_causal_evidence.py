"""Gate 1 matched PE-drive causal evidence.

The benchmark behavior remains owned by the existing dialogue harness.  This
module only freezes the Gate 1 arms, held-out split, seed schedule and verdict
calculation, then exports the common twelve-file evidence packet.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

from volvence_zero.agent.dialogue import (
    DEFAULT_OPEN_DIALOGUE_SCENARIOS,
    OpenDialogueBenchmarkComparisonReport,
    build_real_dialogue_comprehensive_runner_factories,
    run_open_dialogue_ablation_benchmark,
)
from volvence_zero.substrate import LocalSubstrateRuntimeMode


GATE1_PE_CAUSAL_SCHEMA_VERSION = "gate1-pe-causal.v1"
GATE1_PE_CAUSAL_PROFILES = ("pe-eta", "pe-drive-off")
GATE1_PE_CAUSAL_SEEDS = (101, 211, 307)
GATE1_PE_CAUSAL_MIN_EFFECT = 0.25
GATE1_PE_CAUSAL_PRIMARY_CHECK = (
    "late-episode-stabilization-or-improvement"
)
GATE1_PE_CAUSAL_REQUIRED_FILES = (
    "manifest.yaml",
    "predictions.jsonl",
    "outcomes.jsonl",
    "prediction_errors.jsonl",
    "segments.jsonl",
    "credit.jsonl",
    "state_diff.jsonl",
    "action_selection.jsonl",
    "ablation_results.json",
    "promotion_verdict.json",
    "rollback_evidence.json",
    "report.md",
)


@dataclass(frozen=True)
class Gate1PECausalCaseResult:
    seed: int
    profile_label: str
    scenario_id: str
    split: str
    primary_check_passed: bool
    open_case_passed: bool
    reasons: tuple[str, ...]
    acceptance_checks: tuple[tuple[str, bool], ...]
    delayed_improvement_observed: bool
    late_episode_stability_score: float
    prediction_chain_turn_count: int
    pe_triggered_turn_count: int
    online_learning_turn_count: int
    temporal_change_count: int
    turn_count: int


@dataclass(frozen=True)
class Gate1PECausalSeedResult:
    seed: int
    profile_labels: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    substrate_fingerprint: str
    runtime_origin: str
    full_learning_success_rate: float
    no_pe_drive_learning_success_rate: float
    heldout_learning_gain: float
    full_open_pass_rate: float
    no_pe_drive_open_pass_rate: float
    passed: bool
    case_results: tuple[Gate1PECausalCaseResult, ...]


def gate1_pe_causal_heldout_scenarios():
    scenarios = tuple(
        scenario
        for scenario in DEFAULT_OPEN_DIALOGUE_SCENARIOS
        if scenario.split == "open_heldout"
    )
    expected_ids = (
        "open_repair_heldout",
        "open_clarification_heldout",
        "open_failure_loop_heldout",
        "open_goal_shift_heldout",
    )
    actual_ids = tuple(scenario.scenario_id for scenario in scenarios)
    if actual_ids != expected_ids:
        raise ValueError(
            "Gate 1 causal held-out scenario registry drifted: "
            f"expected={expected_ids!r}, actual={actual_ids!r}"
        )
    return scenarios


def _primary_check(
    acceptance_checks: Sequence[tuple[str, bool]],
) -> bool:
    checks = dict(acceptance_checks)
    if GATE1_PE_CAUSAL_PRIMARY_CHECK not in checks:
        raise ValueError(
            "Gate 1 causal case lacks preregistered acceptance check "
            f"{GATE1_PE_CAUSAL_PRIMARY_CHECK!r}"
        )
    return bool(checks[GATE1_PE_CAUSAL_PRIMARY_CHECK])


def _substrate_fingerprint(runtime: object) -> str:
    payload = {
        "model_id": str(getattr(runtime, "model_id", "unknown")),
        "runtime_origin": str(
            getattr(runtime, "runtime_origin", "unknown")
        ),
        "runtime_mode": LocalSubstrateRuntimeMode.BUILTIN_ONLY.value,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summarize_gate1_pe_causal_seed(
    *,
    seed: int,
    comparison: OpenDialogueBenchmarkComparisonReport,
    substrate_fingerprint: str,
    runtime_origin: str,
) -> Gate1PECausalSeedResult:
    if comparison.baseline_label != GATE1_PE_CAUSAL_PROFILES[0]:
        raise ValueError(
            "Gate 1 causal baseline drifted from pe-eta: "
            f"{comparison.baseline_label!r}"
        )
    path_labels = tuple(
        path.path_label for path in comparison.path_reports
    )
    if path_labels != GATE1_PE_CAUSAL_PROFILES:
        raise ValueError(
            "Gate 1 causal matched arms drifted: "
            f"{path_labels!r}"
        )
    case_results: list[Gate1PECausalCaseResult] = []
    for path in comparison.path_reports:
        for report in path.benchmark_report.case_reports:
            case_results.append(
                Gate1PECausalCaseResult(
                    seed=seed,
                    profile_label=path.path_label,
                    scenario_id=report.scenario.scenario_id,
                    split=report.scenario.split,
                    primary_check_passed=_primary_check(
                        report.acceptance_checks
                    ),
                    open_case_passed=report.passed,
                    reasons=report.reasons,
                    acceptance_checks=report.acceptance_checks,
                    delayed_improvement_observed=(
                        report.delayed_improvement_observed
                    ),
                    late_episode_stability_score=(
                        report.late_episode_stability_score
                    ),
                    prediction_chain_turn_count=(
                        report.prediction_chain_turn_count
                    ),
                    pe_triggered_turn_count=report.pe_triggered_turn_count,
                    online_learning_turn_count=(
                        report.online_learning_turn_count
                    ),
                    temporal_change_count=report.temporal_change_count,
                    turn_count=len(report.turns),
                )
            )
    scenario_ids = tuple(
        scenario.scenario_id
        for scenario in gate1_pe_causal_heldout_scenarios()
    )

    def rate(profile_label: str, field: str) -> float:
        rows = tuple(
            row
            for row in case_results
            if row.profile_label == profile_label
        )
        if tuple(row.scenario_id for row in rows) != scenario_ids:
            raise ValueError(
                "Gate 1 causal scenario order/content drifted for "
                f"{profile_label!r}"
            )
        return sum(bool(getattr(row, field)) for row in rows) / len(rows)

    full_rate = rate("pe-eta", "primary_check_passed")
    no_pe_rate = rate("pe-drive-off", "primary_check_passed")
    full_open_pass_rate = rate("pe-eta", "open_case_passed")
    no_pe_open_pass_rate = rate(
        "pe-drive-off",
        "open_case_passed",
    )
    gain = full_rate - no_pe_rate
    return Gate1PECausalSeedResult(
        seed=seed,
        profile_labels=path_labels,
        scenario_ids=scenario_ids,
        substrate_fingerprint=substrate_fingerprint,
        runtime_origin=runtime_origin,
        full_learning_success_rate=full_rate,
        no_pe_drive_learning_success_rate=no_pe_rate,
        heldout_learning_gain=gain,
        full_open_pass_rate=full_open_pass_rate,
        no_pe_drive_open_pass_rate=no_pe_open_pass_rate,
        passed=gain >= GATE1_PE_CAUSAL_MIN_EFFECT,
        case_results=tuple(case_results),
    )


async def run_gate1_pe_causal_seed(
    *,
    seed: int,
    shared_factories=None,
) -> Gate1PECausalSeedResult:
    if seed not in GATE1_PE_CAUSAL_SEEDS:
        raise ValueError(
            f"Gate 1 causal seed {seed} is not preregistered"
        )
    factories = shared_factories
    if factories is None:
        factories = build_real_dialogue_comprehensive_runner_factories(
            runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
        )
    runtime = factories.residual_runtime
    comparison = await run_open_dialogue_ablation_benchmark(
        scenarios=gate1_pe_causal_heldout_scenarios(),
        profile_labels=GATE1_PE_CAUSAL_PROFILES,
        baseline_label=GATE1_PE_CAUSAL_PROFILES[0],
        runner_factory=factories.open_runner_factory,
        seed=seed,
    )
    return summarize_gate1_pe_causal_seed(
        seed=seed,
        comparison=comparison,
        substrate_fingerprint=_substrate_fingerprint(runtime),
        runtime_origin=str(
            getattr(runtime, "runtime_origin", "unknown")
        ),
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _seed_checkpoint_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "checkpoints" / f"seed_{seed}.json"


def _write_seed_checkpoint(
    output_dir: Path,
    result: Gate1PECausalSeedResult,
) -> None:
    path = _seed_checkpoint_path(output_dir, result.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, asdict(result))


def _load_seed_checkpoint(
    output_dir: Path,
    seed: int,
) -> Gate1PECausalSeedResult | None:
    path = _seed_checkpoint_path(output_dir, seed)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["seed"] != seed:
        raise ValueError(
            f"Gate 1 causal checkpoint seed mismatch at {path}"
        )
    return Gate1PECausalSeedResult(
        seed=int(payload["seed"]),
        profile_labels=tuple(payload["profile_labels"]),
        scenario_ids=tuple(payload["scenario_ids"]),
        substrate_fingerprint=str(
            payload["substrate_fingerprint"]
        ),
        runtime_origin=str(payload["runtime_origin"]),
        full_learning_success_rate=float(
            payload["full_learning_success_rate"]
        ),
        no_pe_drive_learning_success_rate=float(
            payload["no_pe_drive_learning_success_rate"]
        ),
        heldout_learning_gain=float(payload["heldout_learning_gain"]),
        full_open_pass_rate=float(payload["full_open_pass_rate"]),
        no_pe_drive_open_pass_rate=float(
            payload["no_pe_drive_open_pass_rate"]
        ),
        passed=bool(payload["passed"]),
        case_results=tuple(
            Gate1PECausalCaseResult(
                seed=int(row["seed"]),
                profile_label=str(row["profile_label"]),
                scenario_id=str(row["scenario_id"]),
                split=str(row["split"]),
                primary_check_passed=bool(
                    row["primary_check_passed"]
                ),
                open_case_passed=bool(row["open_case_passed"]),
                reasons=tuple(row["reasons"]),
                acceptance_checks=tuple(
                    (str(name), bool(passed))
                    for name, passed in row["acceptance_checks"]
                ),
                delayed_improvement_observed=bool(
                    row["delayed_improvement_observed"]
                ),
                late_episode_stability_score=float(
                    row["late_episode_stability_score"]
                ),
                prediction_chain_turn_count=int(
                    row["prediction_chain_turn_count"]
                ),
                pe_triggered_turn_count=int(
                    row["pe_triggered_turn_count"]
                ),
                online_learning_turn_count=int(
                    row["online_learning_turn_count"]
                ),
                temporal_change_count=int(
                    row["temporal_change_count"]
                ),
                turn_count=int(row["turn_count"]),
            )
            for row in payload["case_results"]
        ),
    )


def _packet_rows(
    results: Sequence[Gate1PECausalSeedResult],
) -> dict[str, list[dict[str, Any]]]:
    cases = [
        asdict(case)
        for result in results
        for case in result.case_results
    ]
    return {
        "predictions.jsonl": [
            {
                "seed": case["seed"],
                "profile_label": case["profile_label"],
                "scenario_id": case["scenario_id"],
                "prediction_chain_turn_count": (
                    case["prediction_chain_turn_count"]
                ),
                "turn_count": case["turn_count"],
            }
            for case in cases
        ],
        "outcomes.jsonl": cases,
        "prediction_errors.jsonl": [
            {
                "seed": case["seed"],
                "profile_label": case["profile_label"],
                "scenario_id": case["scenario_id"],
                "pe_triggered_turn_count": (
                    case["pe_triggered_turn_count"]
                ),
            }
            for case in cases
        ],
        "segments.jsonl": [
            {
                "seed": case["seed"],
                "profile_label": case["profile_label"],
                "scenario_id": case["scenario_id"],
                "split": case["split"],
            }
            for case in cases
        ],
        "credit.jsonl": [
            {
                "seed": case["seed"],
                "profile_label": case["profile_label"],
                "scenario_id": case["scenario_id"],
                "online_learning_turn_count": (
                    case["online_learning_turn_count"]
                ),
            }
            for case in cases
        ],
        "state_diff.jsonl": [
            {
                "seed": result.seed,
                "heldout_learning_gain": (
                    result.heldout_learning_gain
                ),
                "full_learning_success_rate": (
                    result.full_learning_success_rate
                ),
                "no_pe_drive_learning_success_rate": (
                    result.no_pe_drive_learning_success_rate
                ),
            }
            for result in results
        ],
        "action_selection.jsonl": [
            {
                "seed": case["seed"],
                "profile_label": case["profile_label"],
                "scenario_id": case["scenario_id"],
                "temporal_change_count": case["temporal_change_count"],
            }
            for case in cases
        ],
    }


def _write_packet(
    *,
    output_dir: Path,
    results: Sequence[Gate1PECausalSeedResult],
    requested_full_matrix: bool,
) -> tuple[Path, ...]:
    expected_ids = tuple(
        scenario.scenario_id
        for scenario in gate1_pe_causal_heldout_scenarios()
    )
    fingerprints = {
        result.substrate_fingerprint for result in results
    }
    probe_passed = bool(results and results[0].passed)
    full_matrix_complete = (
        tuple(result.seed for result in results)
        == GATE1_PE_CAUSAL_SEEDS
    )
    mean_gain = (
        sum(result.heldout_learning_gain for result in results)
        / len(results)
        if results
        else 0.0
    )
    gates = {
        "probe_seed_101_passed": probe_passed,
        "profiles_match_preregistration": all(
            result.profile_labels == GATE1_PE_CAUSAL_PROFILES
            for result in results
        ),
        "heldout_scenarios_match_preregistration": all(
            result.scenario_ids == expected_ids for result in results
        ),
        "builtin_substrate_fingerprint_shared": (
            len(fingerprints) == 1
            and all(
                result.runtime_origin == "builtin-fallback"
                for result in results
            )
        ),
        "full_matrix_complete": full_matrix_complete,
        "three_seed_direction_and_min_effect": (
            full_matrix_complete
            and all(result.passed for result in results)
        ),
        "mean_gain_meets_min_effect": (
            full_matrix_complete
            and mean_gain >= GATE1_PE_CAUSAL_MIN_EFFECT
        ),
    }
    if not probe_passed:
        status = "not-supported"
        causal_status = "not-supported"
    elif full_matrix_complete and all(gates.values()):
        status = "causal-supported"
        causal_status = "causal-supported"
    else:
        status = "probe-passed" if not requested_full_matrix else "not-supported"
        causal_status = (
            "not-evaluated"
            if not requested_full_matrix
            else "not-supported"
        )
    manifest = {
        "schema_version": GATE1_PE_CAUSAL_SCHEMA_VERSION,
        "suite_id": "gate1-pe-causal",
        "owner": "dialogue open benchmark evidence harness",
        "profiles": list(GATE1_PE_CAUSAL_PROFILES),
        "baseline_label": GATE1_PE_CAUSAL_PROFILES[0],
        "seed_schedule": list(GATE1_PE_CAUSAL_SEEDS),
        "scenario_ids": list(expected_ids),
        "scenario_split": "open_heldout",
        "primary_check": GATE1_PE_CAUSAL_PRIMARY_CHECK,
        "primary_metric": "heldout_learning_gain",
        "minimum_effect": GATE1_PE_CAUSAL_MIN_EFFECT,
        "runtime_mode": LocalSubstrateRuntimeMode.BUILTIN_ONLY.value,
        "required_files": list(GATE1_PE_CAUSAL_REQUIRED_FILES),
        "provenance": {
            "git_sha": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "working_tree_dirty": bool(
                _git_output("status", "--porcelain")
                not in {"", "unknown"}
            ),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    ablation = {
        "schema_version": GATE1_PE_CAUSAL_SCHEMA_VERSION,
        "requested_full_matrix": requested_full_matrix,
        "completed_seeds": [result.seed for result in results],
        "seed_results": [asdict(result) for result in results],
        "mean_heldout_learning_gain": mean_gain,
        "gates": gates,
    }
    verdict = {
        "schema_version": GATE1_PE_CAUSAL_SCHEMA_VERSION,
        "gate_scope": "Gate 1 PE drive causal contribution",
        "status": status,
        "mechanism_status": "mechanism-supported",
        "causal_status": causal_status,
        "longitudinal_status": "not-evaluated",
        "thesis_status": "not-evaluated",
        "claim_if_not_supported": "PE is an auditable primary signal",
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
    }
    rollback = {
        "schema_version": GATE1_PE_CAUSAL_SCHEMA_VERSION,
        "runtime_owner_mutated_by_evidence": False,
        "rollback_profile": "pe-drive-off",
        "rollback_action": (
            "retain mechanism verdict and shrink causal claim"
        ),
        "passed": True,
    }
    for name, payload in {
        "manifest.yaml": manifest,
        "ablation_results.json": ablation,
        "promotion_verdict.json": verdict,
        "rollback_evidence.json": rollback,
    }.items():
        _write_json(output_dir / name, payload)
    for name, rows in _packet_rows(results).items():
        _write_jsonl(output_dir / name, rows)
    (output_dir / "report.md").write_text(
        "\n".join(
            (
                "# Gate 1 PE-drive causal evidence",
                "",
                f"- status: `{status}`",
                (
                    "- completed seeds: `"
                    + ", ".join(str(result.seed) for result in results)
                    + "`"
                ),
                f"- mean held-out learning gain: `{mean_gain:.6f}`",
                f"- minimum effect: `{GATE1_PE_CAUSAL_MIN_EFFECT:.2f}`",
                "- primary check excludes PE-schedule mechanism checks",
                "",
            )
        ),
        encoding="utf-8",
    )
    written = tuple(
        output_dir / name
        for name in GATE1_PE_CAUSAL_REQUIRED_FILES
    )
    missing = tuple(path.name for path in written if not path.is_file())
    if missing:
        raise RuntimeError(
            f"Gate 1 causal packet missing files {missing!r}"
        )
    return written


async def export_gate1_pe_causal_bundle(
    *,
    output_dir: str | Path,
    full_matrix: bool,
) -> tuple[Path, ...]:
    """Run or resume the preregistered seed matrix and export its packet."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    factories = build_real_dialogue_comprehensive_runner_factories(
        runtime_mode=LocalSubstrateRuntimeMode.BUILTIN_ONLY,
    )
    requested_seeds = (
        GATE1_PE_CAUSAL_SEEDS
        if full_matrix
        else GATE1_PE_CAUSAL_SEEDS[:1]
    )
    results: list[Gate1PECausalSeedResult] = []
    for seed in requested_seeds:
        checkpoint = _load_seed_checkpoint(target, seed)
        result = checkpoint
        if result is None:
            result = await run_gate1_pe_causal_seed(
                seed=seed,
                shared_factories=factories,
            )
            _write_seed_checkpoint(target, result)
        results.append(result)
        if not result.passed:
            break
    return _write_packet(
        output_dir=target,
        results=results,
        requested_full_matrix=full_matrix,
    )


def export_gate1_pe_causal_bundle_sync(
    *,
    output_dir: str | Path,
    full_matrix: bool,
) -> tuple[Path, ...]:
    return asyncio.run(
        export_gate1_pe_causal_bundle(
            output_dir=output_dir,
            full_matrix=full_matrix,
        )
    )


__all__ = [
    "GATE1_PE_CAUSAL_MIN_EFFECT",
    "GATE1_PE_CAUSAL_PRIMARY_CHECK",
    "GATE1_PE_CAUSAL_PROFILES",
    "GATE1_PE_CAUSAL_REQUIRED_FILES",
    "GATE1_PE_CAUSAL_SCHEMA_VERSION",
    "GATE1_PE_CAUSAL_SEEDS",
    "Gate1PECausalCaseResult",
    "Gate1PECausalSeedResult",
    "export_gate1_pe_causal_bundle",
    "export_gate1_pe_causal_bundle_sync",
    "gate1_pe_causal_heldout_scenarios",
    "run_gate1_pe_causal_seed",
    "summarize_gate1_pe_causal_seed",
]
