"""Shared immutable bundle helpers for the Gate 1/4/6 v2 retests."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


GATE_V2_REQUIRED_FILES = (
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


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def export_gate_v2_bundle(
    *,
    schema_version: str,
    suite_id: str,
    source_schema_version: str,
    source_fingerprint: str,
    partition: str,
    seed_schedule: tuple[int, ...],
    arm_schedule: tuple[str, ...],
    formal_locked_run: bool,
    rows_by_file: Mapping[str, Sequence[Mapping[str, object]]],
    arm_results: object,
    aggregate_metrics: object,
    mechanism_gates: object,
    causal_gates: object,
    verdict: str,
    rollback_rows: object,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if formal_locked_run:
        existing = tuple(
            filename
            for filename in GATE_V2_REQUIRED_FILES
            if (target / filename).exists()
        )
        if existing:
            raise FileExistsError(
                f"{suite_id} locked evidence is immutable; refusing to "
                f"overwrite {existing}"
            )
    expected_jsonl = set(GATE_V2_REQUIRED_FILES[1:8])
    if set(rows_by_file) != expected_jsonl:
        raise ValueError(
            f"{suite_id} JSONL surface drifted: {tuple(rows_by_file)}"
        )
    written: list[Path] = []
    for filename, rows in rows_by_file.items():
        path = target / filename
        write_jsonl(path, rows)
        written.append(path)
    payloads = {
        "ablation_results.json": {
            "schema_version": schema_version,
            "arm_schedule": arm_schedule,
            "arm_results": jsonable(arm_results),
            "aggregate_metrics": jsonable(aggregate_metrics),
        },
        "promotion_verdict.json": {
            "schema_version": schema_version,
            "verdict": verdict,
            "mechanism_gates": jsonable(mechanism_gates),
            "causal_gates": jsonable(causal_gates),
            "locked_consumed": formal_locked_run,
            "retuning_allowed": False,
        },
        "rollback_evidence.json": {
            "schema_version": schema_version,
            "rows": jsonable(rollback_rows),
        },
        "manifest.yaml": {
            "schema_version": schema_version,
            "suite_id": suite_id,
            "source_schema_version": source_schema_version,
            "source_fingerprint": source_fingerprint,
            "partition": partition,
            "seed_schedule": seed_schedule,
            "arm_schedule": arm_schedule,
            "formal_locked_run": formal_locked_run,
            "required_files": GATE_V2_REQUIRED_FILES,
        },
    }
    for filename, payload in payloads.items():
        path = target / filename
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    report_path = target / "report.md"
    report_path.write_text(
        (
            f"# {suite_id}\n\n"
            f"- partition: `{partition}`\n"
            f"- formal locked run: `{formal_locked_run}`\n"
            f"- verdict: `{verdict}`\n"
            f"- source fingerprint: `{source_fingerprint}`\n\n"
            "## Mechanism gates\n\n"
            + "".join(
                f"- {name}: `{passed}` ({value:.6f})\n"
                for name, passed, value in mechanism_gates
            )
            + "\n## Causal gates\n\n"
            + "".join(
                f"- {name}: `{passed}` ({value:.6f})\n"
                for name, passed, value in causal_gates
            )
        ),
        encoding="utf-8",
    )
    written.append(report_path)
    return tuple(written)


def verify_gate_v2_bundle(
    output_dir: str | Path,
    *,
    schema_version: str,
    suite_id: str,
    arm_schedule: tuple[str, ...],
) -> dict[str, object]:
    target = Path(output_dir)
    missing = tuple(
        filename
        for filename in GATE_V2_REQUIRED_FILES
        if not (target / filename).is_file()
    )
    if missing:
        return {
            "passed": False,
            "missing_files": missing,
            "verdict": "invalid",
        }
    manifest = json.loads(
        (target / "manifest.yaml").read_text(encoding="utf-8")
    )
    verdict = json.loads(
        (target / "promotion_verdict.json").read_text(encoding="utf-8")
    )
    passed = (
        manifest["schema_version"] == schema_version
        and manifest["suite_id"] == suite_id
        and tuple(manifest["arm_schedule"]) == arm_schedule
        and tuple(manifest["required_files"]) == GATE_V2_REQUIRED_FILES
        and verdict["verdict"]
        in {"invalid", "not-supported", "causal-supported"}
    )
    return {
        "passed": passed,
        "missing_files": (),
        "verdict": verdict["verdict"],
        "formal_locked_run": manifest["formal_locked_run"],
        "partition": manifest["partition"],
    }
