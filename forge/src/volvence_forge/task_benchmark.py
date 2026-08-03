"""Loop-external, task-level diagnostic benchmark for editable harness assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ForgeConfig
from .foundation import (
    ForgeError,
    PromptStore,
    SchemaStore,
    StructuredBackend,
    atomic_write_json,
    canonical_json,
    read_json,
    sha256_text,
    utc_now,
    utc_stamp,
)


_DEFAULT_SUITE = "task_level_held_out.v1.json"
_MAX_ASSET_CHARACTERS = 100_000
_EXACT_DECISION_FIELDS = (
    "classification",
    "next_action",
    "target_lane",
    "preserve_evidence",
)


@dataclass(frozen=True)
class TaskBenchmarkResult:
    status: str
    report_path: Path
    baseline_pass_rate: float
    candidate_pass_rate: float | None


def run_task_benchmark(
    *,
    config: ForgeConfig,
    target: str,
    backend: StructuredBackend,
    candidate_asset_path: Path | None = None,
    suite_path: Path | None = None,
    report_path: Path | None = None,
) -> TaskBenchmarkResult:
    """Score one editable harness asset without granting promotion authority."""

    normalized_target = config.normalize_relative_path(target)
    entry = config.editable_entry_for(normalized_target)
    if entry is None:
        raise ForgeError(
            f"Task benchmark target is not an editable harness asset: {normalized_target}"
        )
    target_path = config.resolve_target(normalized_target, must_exist=True)
    if not target_path.is_file():
        raise ForgeError(f"Task benchmark target must be a file: {normalized_target}")

    schema_store = SchemaStore(config.paths.forge_root / "schemas")
    resolved_suite_path = _resolve_suite_path(config, suite_path)
    suite = read_json(resolved_suite_path)
    schema_store.validate(suite, "task_benchmark_suite.schema.json")
    _validate_suite_identity(suite)

    applicable_components = suite["applicable_components"]
    if entry.component not in applicable_components:
        raise ForgeError(
            f"Task benchmark suite {suite['suite_id']} does not apply to component "
            f"{entry.component}"
        )

    prompt_store = PromptStore(config.paths.forge_root / "benchmark_prompts")
    decision_schema = schema_store.load("task_benchmark_decision.schema.json")
    baseline = _score_arm(
        label="baseline",
        asset_content=_read_asset(target_path),
        suite=suite,
        backend=backend,
        prompt_store=prompt_store,
        schema_store=schema_store,
        decision_schema=decision_schema,
    )

    candidate = None
    if candidate_asset_path is not None:
        candidate = _score_arm(
            label="candidate",
            asset_content=_read_asset(candidate_asset_path.resolve()),
            suite=suite,
            backend=backend,
            prompt_store=prompt_store,
            schema_store=schema_store,
            decision_schema=decision_schema,
        )

    thresholds = suite["thresholds"]
    candidate_delta = None
    evaluated_arm = baseline
    if candidate is not None:
        evaluated_arm = candidate
        candidate_delta = candidate["pass_rate"] - baseline["pass_rate"]
    passes_absolute_thresholds = (
        evaluated_arm["pass_rate"] >= thresholds["minimum_pass_rate"]
        and evaluated_arm["critical_failure_count"]
        <= thresholds["maximum_critical_failures"]
    )
    passes_delta = (
        candidate_delta is None
        or candidate_delta >= thresholds["minimum_candidate_delta"]
    )
    status = "PASS" if passes_absolute_thresholds and passes_delta else "BLOCK"

    report = {
        "schema_version": "forge-task-benchmark-report.v1",
        "suite_id": suite["suite_id"],
        "suite_sha256": sha256_text(canonical_json(suite)),
        "target": normalized_target,
        "component": entry.component,
        "backend": backend.backend_name,
        "model": backend.model_name,
        "thresholds": thresholds,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_delta": candidate_delta,
        "status": status,
        "diagnostic_only": True,
        "causal_claim_authorized": False,
        "created_at": utc_now(),
    }
    schema_store.validate(report, "task_benchmark_report.schema.json")
    destination = report_path or (
        config.paths.artifacts_root
        / f"forge_task_benchmark_{utc_stamp()}"
        / "report.json"
    )
    atomic_write_json(destination, report)
    return TaskBenchmarkResult(
        status=status,
        report_path=destination,
        baseline_pass_rate=baseline["pass_rate"],
        candidate_pass_rate=(candidate["pass_rate"] if candidate is not None else None),
    )


def _resolve_suite_path(config: ForgeConfig, suite_path: Path | None) -> Path:
    benchmark_root = (config.paths.forge_root / "benchmarks").resolve()
    resolved = (suite_path or benchmark_root / _DEFAULT_SUITE).resolve()
    if not resolved.is_relative_to(benchmark_root):
        raise ForgeError(
            f"Task benchmark suite must remain under the read-only benchmark root: {resolved}"
        )
    if not resolved.is_file():
        raise ForgeError(f"Task benchmark suite is missing: {resolved}")
    return resolved


def _validate_suite_identity(suite: dict[str, Any]) -> None:
    case_ids = [case["case_id"] for case in suite["cases"]]
    duplicates = sorted(
        case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
    )
    if duplicates:
        raise ForgeError(f"Task benchmark suite contains duplicate case IDs: {duplicates}")


def _read_asset(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ForgeError(f"Task benchmark asset is missing: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ForgeError(f"Task benchmark asset must be UTF-8 text: {path}") from exc
    if not content.strip():
        raise ForgeError(f"Task benchmark asset must not be empty: {path}")
    if len(content) > _MAX_ASSET_CHARACTERS:
        raise ForgeError(
            f"Task benchmark asset exceeds {_MAX_ASSET_CHARACTERS} characters: {path}"
        )
    return content


def _score_arm(
    *,
    label: str,
    asset_content: str,
    suite: dict[str, Any],
    backend: StructuredBackend,
    prompt_store: PromptStore,
    schema_store: SchemaStore,
    decision_schema: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = prompt_store.render(
        "task_decision.system.md",
        asset_content=asset_content,
    )
    passed_case_ids: list[str] = []
    failures: list[dict[str, Any]] = []
    for case in suite["cases"]:
        case_record = {
            "case_id": case["case_id"],
            "task": case["task"],
            "structured_evidence": case["structured_evidence"],
        }
        user_prompt = prompt_store.render(
            "task_decision.user.md",
            case_record=canonical_json(case_record),
        )
        decision = backend.complete_json(
            system=system_prompt,
            user=user_prompt,
            schema=decision_schema,
        )
        schema_store.validate(decision, "task_benchmark_decision.schema.json")
        mismatches = _decision_mismatches(case, decision)
        if mismatches:
            failures.append(
                {
                    "case_id": case["case_id"],
                    "critical": case["critical"],
                    "mismatches": mismatches,
                    "decision": decision,
                }
            )
        else:
            passed_case_ids.append(case["case_id"])
    total = len(suite["cases"])
    return {
        "label": label,
        "asset_sha256": sha256_text(asset_content),
        "pass_rate": len(passed_case_ids) / total,
        "critical_failure_count": sum(
            1 for failure in failures if failure["critical"]
        ),
        "passed_case_ids": passed_case_ids,
        "failures": failures,
    }


def _decision_mismatches(
    case: dict[str, Any], decision: dict[str, Any]
) -> list[str]:
    expected = case["expected"]
    mismatches: list[str] = []
    if decision["case_id"] != case["case_id"]:
        mismatches.append(
            f"case_id expected {case['case_id']!r}, got {decision['case_id']!r}"
        )
    for field in _EXACT_DECISION_FIELDS:
        if decision[field] != expected[field]:
            mismatches.append(
                f"{field} expected {expected[field]!r}, got {decision[field]!r}"
            )
    if decision["confidence"] < case["minimum_confidence"]:
        mismatches.append(
            "confidence below minimum "
            f"{case['minimum_confidence']!r}: got {decision['confidence']!r}"
        )
    return mismatches
