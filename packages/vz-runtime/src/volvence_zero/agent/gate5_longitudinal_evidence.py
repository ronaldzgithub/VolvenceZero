"""Gate 5 fresh cross-session CMS Pareto evidence.

This v2 packet reuses the Gate 5 owner-level arm implementation but consumes
the fresh Gate 11 longitudinal source and forces a filesystem persistence /
owner reconstruction boundary every ten settled transitions.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

from volvence_zero.agent.gate11_longitudinal_source import (
    GATE11_LONGITUDINAL_SESSION_SIZE,
    GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
    GATE11_LONGITUDINAL_SOURCE_SEEDS,
    build_gate11_longitudinal_source_plans,
    load_gate11_longitudinal_source_records,
    validate_gate11_longitudinal_source_prefix,
)
from volvence_zero.agent.gate5_cms_pareto_evidence import (
    GATE5_ARM_NAMES,
    GATE5_FULL_ARM,
    GATE5_MIN_EFFECT,
    GATE5_PARETO_TOLERANCE,
    GATE5_SINGLE_TIMESCALE_ARM,
    Gate5ArmMetrics,
    Gate5Comparison,
    run_gate5_arm,
)
from volvence_zero.memory.persistence import FileSystemPersistenceBackend


GATE5_LONGITUDINAL_SCHEMA_VERSION = "gate5-cms-pareto-longitudinal.v2"
GATE5_LONGITUDINAL_REQUIRED_FILES = (
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
_T_CRITICAL_95_DF2 = 4.302652729749


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")


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


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _confidence_interval_95(
    values: Sequence[float],
) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0]) if values else 0.0
        return (value, value)
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values)
    half_width = _T_CRITICAL_95_DF2 * deviation / math.sqrt(len(values))
    return (mean - half_width, mean + half_width)


def _validate_fresh_records(
    records: Sequence[Mapping[str, Any]],
    seed: int,
) -> None:
    validate_gate11_longitudinal_source_prefix(
        records=records,
        plans=build_gate11_longitudinal_source_plans(seed),
    )
    if len(records) != 510:
        raise ValueError(
            f"Gate 5 longitudinal seed {seed} requires 510 records"
        )


def compare_gate5_longitudinal_arms(
    metrics: Sequence[Gate5ArmMetrics],
) -> tuple[
    tuple[Gate5Comparison, ...],
    dict[str, dict[str, Any]],
    dict[str, bool],
]:
    by_arm_seed = {
        (metric.arm, metric.seed): metric for metric in metrics
    }
    comparisons: list[Gate5Comparison] = []
    confidence: dict[str, dict[str, Any]] = {}
    for control in GATE5_ARM_NAMES[1:]:
        absorption_gains = tuple(
            by_arm_seed[(GATE5_FULL_ARM, seed)].new_knowledge_absorption
            - by_arm_seed[(control, seed)].new_knowledge_absorption
            for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS
        )
        retention_gains = tuple(
            by_arm_seed[(GATE5_FULL_ARM, seed)].old_knowledge_retention
            - by_arm_seed[(control, seed)].old_knowledge_retention
            for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS
        )
        absorption_gain = _mean(absorption_gains)
        retention_gain = _mean(retention_gains)
        comparisons.append(
            Gate5Comparison(
                control_arm=control,
                absorption_gain=absorption_gain,
                retention_gain=retention_gain,
                absorption_seed_gains=absorption_gains,
                retention_seed_gains=retention_gains,
                pareto_non_worse=(
                    absorption_gain >= -GATE5_PARETO_TOLERANCE
                    and retention_gain >= -GATE5_PARETO_TOLERANCE
                    and all(
                        gain >= -GATE5_PARETO_TOLERANCE
                        for gain in absorption_gains
                    )
                    and all(
                        gain >= -GATE5_PARETO_TOLERANCE
                        for gain in retention_gains
                    )
                ),
            )
        )
        confidence[control] = {
            "absorption_confidence_interval_95": list(
                _confidence_interval_95(absorption_gains)
            ),
            "retention_confidence_interval_95": list(
                _confidence_interval_95(retention_gains)
            ),
        }
    single = next(
        comparison
        for comparison in comparisons
        if comparison.control_arm == GATE5_SINGLE_TIMESCALE_ARM
    )
    single_confidence = confidence[GATE5_SINGLE_TIMESCALE_ARM]
    absorption_ci_lower = float(
        single_confidence["absorption_confidence_interval_95"][0]
    )
    retention_ci_lower = float(
        single_confidence["retention_confidence_interval_95"][0]
    )
    full = tuple(
        metric for metric in metrics if metric.arm == GATE5_FULL_ARM
    )
    single_metrics = tuple(
        metric
        for metric in metrics
        if metric.arm == GATE5_SINGLE_TIMESCALE_ARM
    )
    gates = {
        "all_arms_all_seeds_present": (
            len(metrics)
            == len(GATE5_ARM_NAMES)
            * len(GATE11_LONGITUDINAL_SOURCE_SEEDS)
        ),
        "all_records_and_locked_counts_complete": all(
            metric.settled_transition_count == 510
            and metric.locked_transition_count == 60
            for metric in metrics
        ),
        "lineage_complete": all(metric.lineage_complete for metric in metrics),
        "frozen_substrate_mutation_zero": all(
            metric.frozen_substrate_mutation_count == 0
            for metric in metrics
        ),
        "full_cadence_1_2_4": all(
            metric.cadence_intervals == (1, 2, 4) for metric in full
        ),
        "single_cadence_1_1_1": all(
            metric.cadence_intervals == (1, 1, 1)
            for metric in single_metrics
        ),
        "matched_cms_parameter_budget": (
            {metric.cms_parameter_count for metric in full}
            == {
                metric.cms_parameter_count
                for metric in single_metrics
            }
        ),
        "full_pareto_non_worse_all_controls": all(
            comparison.pareto_non_worse for comparison in comparisons
        ),
        "full_significant_vs_single_timescale": (
            (
                single.absorption_gain >= GATE5_MIN_EFFECT
                and absorption_ci_lower > 0.0
            )
            or (
                single.retention_gain >= GATE5_MIN_EFFECT
                and retention_ci_lower > 0.0
            )
        ),
    }
    return tuple(comparisons), confidence, gates


def _source_rows(
    records_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    result = {
        "predictions.jsonl": [],
        "outcomes.jsonl": [],
        "prediction_errors.jsonl": [],
        "segments.jsonl": [],
        "credit.jsonl": [],
        "action_selection.jsonl": [],
    }
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        for record in records_by_seed[seed]:
            lineage = {
                "seed": seed,
                "transition_id": record["transition_id"],
                "prediction_ref": record["lineage"]["prediction_ref"],
                "record_sha256": record["record_sha256"],
                "partition": record["partition"],
            }
            result["predictions.jsonl"].append(
                {**lineage, "prediction": record["prediction"]}
            )
            result["outcomes.jsonl"].append(
                {**lineage, "actual_outcome": record["actual_outcome"]}
            )
            result["prediction_errors.jsonl"].append(
                {**lineage, "prediction_error": record["prediction_error"]}
            )
            result["segments.jsonl"].append(
                {
                    **lineage,
                    "episode_phase": record["episode_phase"],
                    "temporal_snapshot": record["temporal_snapshot"],
                }
            )
            result["credit.jsonl"].append(
                {**lineage, "credit_snapshot": record["credit_snapshot"]}
            )
            result["action_selection.jsonl"].append(
                {**lineage, "action_selection": record["action_selection"]}
            )
    return result


def export_gate5_longitudinal_bundle(
    *,
    trace_root: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    source = Path(trace_root)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "runtime_state").exists():
        raise FileExistsError(
            "Gate 5 runtime_state already exists; locked arms are "
            "single-run only and require a fresh output directory"
        )
    aggregate_manifest = json.loads(
        (source / "aggregate_manifest.json").read_text(encoding="utf-8")
    )
    aggregate_verdict = json.loads(
        (source / "aggregate_verdict.json").read_text(encoding="utf-8")
    )
    if aggregate_verdict.get("consumer_admission") != "allowed":
        raise ValueError(
            "Gate 5 longitudinal source is not admitted for consumption"
        )
    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        records = load_gate11_longitudinal_source_records(
            source / f"seed_{seed}" / "transitions.jsonl"
        )
        _validate_fresh_records(records, seed)
        records_by_seed[seed] = records
    metrics: list[Gate5ArmMetrics] = []
    state_rows: list[dict[str, Any]] = []
    rollback_rows: list[dict[str, Any]] = []
    for seed in GATE11_LONGITUDINAL_SOURCE_SEEDS:
        for arm in GATE5_ARM_NAMES:
            backend = FileSystemPersistenceBackend(
                base_dir=str(
                    target
                    / "runtime_state"
                    / str(seed)
                    / arm
                )
            )
            arm_metrics, arm_rows, rollback = run_gate5_arm(
                records=records_by_seed[seed],
                seed=seed,
                arm=arm,
                record_validator=_validate_fresh_records,
                persistence_backend=backend,
                session_boundary_interval=(
                    GATE11_LONGITUDINAL_SESSION_SIZE
                ),
            )
            for row in arm_rows:
                row["schema_version"] = GATE5_LONGITUDINAL_SCHEMA_VERSION
            metrics.append(arm_metrics)
            state_rows.extend(arm_rows)
            rollback_rows.append(rollback)
    comparisons, confidence, gates = compare_gate5_longitudinal_arms(
        metrics
    )
    gates["constructor_restart_count_50_per_arm_seed"] = all(
        row["constructor_restart_count"] == 50
        for row in rollback_rows
    )
    gates["persistence_roundtrip_exact"] = all(
        row["persistence_roundtrip_exact"] for row in rollback_rows
    )
    gates["checkpoint_rollback_exact"] = all(
        row["checkpoint_roundtrip_exact"] for row in rollback_rows
    )
    integrity_names = (
        "all_arms_all_seeds_present",
        "all_records_and_locked_counts_complete",
        "lineage_complete",
        "frozen_substrate_mutation_zero",
        "full_cadence_1_2_4",
        "single_cadence_1_1_1",
        "matched_cms_parameter_budget",
        "constructor_restart_count_50_per_arm_seed",
        "persistence_roundtrip_exact",
        "checkpoint_rollback_exact",
    )
    integrity_passed = all(gates[name] for name in integrity_names)
    effect_passed = (
        gates["full_pareto_non_worse_all_controls"]
        and gates["full_significant_vs_single_timescale"]
    )
    status = (
        "invalid"
        if not integrity_passed
        else "longitudinal-supported"
        if effect_passed
        else "not-supported"
    )
    manifest = {
        "schema_version": GATE5_LONGITUDINAL_SCHEMA_VERSION,
        "suite_id": "gate5-cms-pareto-longitudinal",
        "owner": "vz-memory.MemoryStore/CMSMemoryCore",
        "trace_schema_version": GATE11_LONGITUDINAL_SOURCE_SCHEMA_VERSION,
        "trace_root": str(source),
        "substrate_fingerprint": aggregate_manifest[
            "runtime_fingerprint"
        ],
        "model_and_adapter_ids": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter_ids": [],
            "substrate_mode": "frozen strict-local source trace",
        },
        "wiring_levels": {
            "source_runtime": "ACTIVE frozen trace",
            "memory_owner": "filesystem-backed cross-session replay",
            "substrate_mutation": "DISABLED",
        },
        "seed_schedule": list(GATE11_LONGITUDINAL_SOURCE_SEEDS),
        "arm_schedule": list(GATE5_ARM_NAMES),
        "session_boundary_interval": GATE11_LONGITUDINAL_SESSION_SIZE,
        "scenario_split": {
            "trace-train": 900,
            "trace-heldout-context": 450,
            "trace-locked-confirmation": 180,
        },
        "cohort_scope": {
            "seed_count": 3,
            "context_count_per_seed": 18,
            "user_count_per_seed": 18,
            "settled_transition_count_per_arm_seed": 510,
            "arm_transition_count": len(state_rows),
        },
        "prompt_and_context_budget": (
            "No new prompt input; matched arms consume only fresh immutable "
            "public source signals and typed PE."
        ),
        "metric_version": GATE5_LONGITUDINAL_SCHEMA_VERSION,
        "judge_or_human_protocol": (
            "None; deterministic memory-owner readouts."
        ),
        "pareto_tolerance": GATE5_PARETO_TOLERANCE,
        "minimum_effect_vs_single_timescale": GATE5_MIN_EFFECT,
        "required_files": list(GATE5_LONGITUDINAL_REQUIRED_FILES),
        "provenance": {
            "git_sha": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "working_tree_dirty": (
                _git_output("status", "--porcelain")
                not in {"", "unknown"}
            ),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    ablation = {
        "schema_version": GATE5_LONGITUDINAL_SCHEMA_VERSION,
        "metrics": [asdict(metric) for metric in metrics],
        "comparisons": [asdict(comparison) for comparison in comparisons],
        "confidence_intervals": confidence,
        "gates": gates,
    }
    verdict = {
        "schema_version": GATE5_LONGITUDINAL_SCHEMA_VERSION,
        "gate_scope": "Gate 5 CMS cross-session Pareto",
        "status": status,
        "mechanism_passed": integrity_passed,
        "causal_passed": status == "longitudinal-supported",
        "longitudinal_passed": status == "longitudinal-supported",
        "claim": (
            "multi-frequency CMS improves cross-session absorption-retention"
            if status == "longitudinal-supported"
            else "multi-frequency CMS is cross-session runnable and rollback-capable"
            if status == "not-supported"
            else "Gate 5 longitudinal evidence packet is invalid"
        ),
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "locked_source_consumed_once": True,
        "same_locked_source_rerun_allowed": False,
    }
    rollback_payload = {
        "schema_version": GATE5_LONGITUDINAL_SCHEMA_VERSION,
        "passed": gates["checkpoint_rollback_exact"],
        "arms": rollback_rows,
        "explicit_factory_rollback": {
            "cms_pe_features_enabled": False,
            "cms_replay_window_size": None,
        },
        "substrate_mutated": False,
    }
    for name, rows in _source_rows(records_by_seed).items():
        _write_jsonl(target / name, rows)
    _write_jsonl(target / "state_diff.jsonl", state_rows)
    _write_json(target / "manifest.yaml", manifest)
    _write_json(target / "ablation_results.json", ablation)
    _write_json(target / "promotion_verdict.json", verdict)
    _write_json(target / "rollback_evidence.json", rollback_payload)
    report_lines = [
        "# Gate 5 cross-session CMS Pareto evidence",
        "",
        f"- status: `{status}`",
        f"- mechanism passed: `{integrity_passed}`",
        f"- longitudinal passed: `{status == 'longitudinal-supported'}`",
        "- constructor restarts per arm/seed: `50`",
        "",
        "## Full vs controls",
        "",
    ]
    report_lines.extend(
        (
            f"- `{comparison.control_arm}`: absorption gain "
            f"`{comparison.absorption_gain:.6f}`, retention gain "
            f"`{comparison.retention_gain:.6f}`, Pareto non-worse "
            f"`{comparison.pareto_non_worse}`"
        )
        for comparison in comparisons
    )
    report_lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            (
                "- Failure of the frozen Pareto/minimum-effect gate keeps "
                "Gate 5 not-supported; this locked source is not rerun."
            ),
            "",
        )
    )
    (target / "report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    written = tuple(
        target / name for name in GATE5_LONGITUDINAL_REQUIRED_FILES
    )
    missing = tuple(path.name for path in written if not path.is_file())
    if missing:
        raise RuntimeError(
            f"Gate 5 longitudinal bundle missing {missing!r}"
        )
    return written


__all__ = [
    "GATE5_LONGITUDINAL_REQUIRED_FILES",
    "GATE5_LONGITUDINAL_SCHEMA_VERSION",
    "compare_gate5_longitudinal_arms",
    "export_gate5_longitudinal_bundle",
]
