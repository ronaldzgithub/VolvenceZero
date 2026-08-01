#!/usr/bin/env python3
"""Independently audit a completed seven-day formal evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from companion_bench.seven_day_driver import (
    FrozenSevenDayUserScript,
    load_frozen_seven_day_user_script,
)
from volvence_zero.agent.seven_day_companion_evidence import (
    SEVEN_DAY_ALL_ARMS,
    SevenDayExperimentCase,
    SevenDayRunEnvelope,
    evaluate_seven_day_ablation,
    validate_seven_day_character_stack_run,
)
from volvence_zero.agent.seven_day_companion_preregistration import (
    seven_day_source_attestation_contract,
    validate_seven_day_companion_preregistration,
)


_MEASUREMENT_CHECKPOINT_NAME = "evaluation__relationship_continuity_v1.json"
_SHUFFLED_SOURCE_DAYS = (1, 1, 2, 1, 4, 3)
_HTTP_ERROR_RE = re.compile(r'HTTP/1\.[01]" [45][0-9][0-9]')


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


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_sha256(root: Path) -> str:
    if not root.is_dir():
        raise FileNotFoundError(f"state archive is missing: {root}")
    digest = hashlib.sha256()
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item.name != _MEASUREMENT_CHECKPOINT_NAME
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _load_mapping(path: Path, *, field: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object: {path}")
    return value


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _resolve_artifact_ref(root: Path, ref: object, *, field: str) -> Path:
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    path = (root / ref).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes the evidence root") from exc
    return path


def _expected_cases(
    preregistration: Mapping[str, object],
) -> tuple[SevenDayExperimentCase, ...]:
    scenario_ids = _require_list(preregistration.get("scenario_ids"), field="scenario_ids")
    formal = _require_mapping(preregistration.get("formal_run"), field="formal_run")
    seeds = _require_list(formal.get("paraphrase_seeds"), field="paraphrase_seeds")
    if not all(isinstance(value, str) and value for value in scenario_ids):
        raise ValueError("scenario_ids contains an invalid value")
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in seeds):
        raise ValueError("paraphrase_seeds contains an invalid value")
    return tuple(SevenDayExperimentCase(str(scenario_id), int(seed)) for scenario_id in scenario_ids for seed in seeds)


def _expected_run_path(*, output_root: Path, case: SevenDayExperimentCase, arm: str) -> Path:
    name = hashlib.sha256(f"{case.case_id}\0{arm}".encode("utf-8")).hexdigest()
    return output_root / "runs" / f"{name}.json"


def _script_turns(script: FrozenSevenDayUserScript) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            turn.day_index,
            turn.exchange_index,
            turn.fsm_action,
            turn.fsm_payload,
            tuple(turn.event_tags),
            turn.text,
        )
        for turn in script.turns
    )


def _run_turns(run: Mapping[str, object]) -> tuple[tuple[object, ...], ...]:
    result = []
    for raw_day in _require_list(run.get("days"), field="run.days"):
        day = _require_mapping(raw_day, field="run.day")
        for raw_turn in _require_list(day.get("turns"), field="run.day.turns"):
            turn = _require_mapping(raw_turn, field="run.day.turn")
            result.append(
                (
                    day.get("day_index"),
                    turn.get("exchange_index"),
                    turn.get("fsm_action"),
                    turn.get("fsm_payload"),
                    tuple(_require_list(turn.get("event_tags"), field="turn.event_tags")),
                    turn.get("user_text"),
                )
            )
    return tuple(result)


def _expected_source(
    *,
    output_root: Path,
    cases: tuple[SevenDayExperimentCase, ...],
    case_index: int,
    arm: str,
    day_index: int,
) -> tuple[str | None, int | None, Path | None]:
    case = cases[case_index]
    case_key = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()[:20]
    if arm == "stateless":
        return None, None, None
    if arm == "swapped-user-state":
        donor = cases[(case_index + 1) % len(cases)]
        donor_key = hashlib.sha256(donor.case_id.encode("utf-8")).hexdigest()[:20]
        return (
            "matched-donor-correct-user-state",
            day_index,
            output_root / "state" / donor_key / "archives" / "correct-user-state" / f"day-{day_index}",
        )
    if arm == "shuffled-history":
        source_day = _SHUFFLED_SOURCE_DAYS[day_index - 1]
        return (
            "same-user-correct-reference",
            source_day,
            output_root / "state" / case_key / "archives" / "correct-user-state" / f"day-{source_day}",
        )
    return (
        "correct-user-state",
        day_index,
        output_root / "state" / case_key / "archives" / arm / f"day-{day_index}",
    )


def _verify_physical_run(
    *,
    output_root: Path,
    cases: tuple[SevenDayExperimentCase, ...],
    case_index: int,
    arm: str,
    run: Mapping[str, object],
    script: FrozenSevenDayUserScript,
    expected_source_audit_sha256: str,
) -> dict[str, set[str]]:
    if _run_turns(run) != _script_turns(script):
        raise ValueError(f"run does not replay the frozen script: {script.case_id}")
    attestation = _require_mapping(run.get("source_attestation"), field="source_attestation")
    if attestation.get("pii_scan_artifact_sha256") != (expected_source_audit_sha256):
        raise ValueError("synthetic source audit digest drift")
    case = cases[case_index]
    case_key = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()[:20]
    ids = {
        "run_ids": {str(run.get("run_id"))},
        "session_ids": set(),
        "service_instance_ids": set(),
        "console_action_ids": set(),
        "pilot_transcripts": set(),
        "pilot_metrics": set(),
        "state_archives": set(),
        "measurement_checkpoints": set(),
        "service_evidence": set(),
    }
    previous_next_instance: object = None
    for raw_day in _require_list(run.get("days"), field="run.days"):
        day = _require_mapping(raw_day, field="run.day")
        day_index = day.get("day_index")
        if isinstance(day_index, bool) or not isinstance(day_index, int):
            raise ValueError("day_index must be an integer")
        service_id = day.get("service_instance_id")
        session_id = day.get("session_id")
        if not isinstance(service_id, str) or not isinstance(session_id, str):
            raise ValueError("service/session identity is missing")
        if previous_next_instance is not None and service_id != previous_next_instance:
            raise ValueError("restart next instance does not match the next day")
        ids["service_instance_ids"].add(service_id)
        ids["session_ids"].add(session_id)
        session_path = (
            output_root / "service_evidence" / case_key / arm / "sessions" / session_id / "session_evidence.json"
        )
        session_payload = _load_mapping(session_path, field="service session evidence")
        if session_payload.get("session_id") != session_id:
            raise ValueError("service session evidence identity drift")
        ids["service_evidence"].add(str(session_path.relative_to(output_root)))
        for raw_action in _require_list(day.get("console_probe_actions"), field="console_probe_actions"):
            action = _require_mapping(raw_action, field="console_probe_action")
            action_id = action.get("action_id")
            if not isinstance(action_id, str) or not action_id:
                raise ValueError("console action id is missing")
            ids["console_action_ids"].add(action_id)
        pilot_root = output_root / "pilot_days" / case_key / arm
        transcript = _resolve_artifact_ref(
            pilot_root,
            day.get("pilot_day_evidence_ref"),
            field="pilot_day_evidence_ref",
        )
        expected_transcript_sha = day.get("pilot_day_transcript_sha256")
        if _file_sha256(transcript) != expected_transcript_sha:
            raise ValueError("pilot transcript digest drift")
        metrics_path = transcript.with_name(f"day-{day_index}-metrics.json")
        metrics = _load_mapping(metrics_path, field="pilot metrics")
        if metrics.get("transcript_sha256") != expected_transcript_sha or metrics.get("day_index") != day_index:
            raise ValueError("pilot metric/transcript linkage drift")
        ids["pilot_transcripts"].add(str(transcript.relative_to(output_root)))
        ids["pilot_metrics"].add(str(metrics_path.relative_to(output_root)))
        restart = day.get("restart_after_day")
        if day_index == 7:
            if restart is not None:
                raise ValueError("day seven unexpectedly contains restart evidence")
            continue
        restart_payload = _require_mapping(restart, field="restart_after_day")
        if (
            restart_payload.get("previous_instance_id") != service_id
            or restart_payload.get("healthcheck_passed") is not True
            or restart_payload.get("persistence_scope_unchanged") is not True
        ):
            raise ValueError("restart lifecycle evidence drift")
        previous_next_instance = restart_payload.get("next_instance_id")
        intervention = _require_mapping(
            restart_payload.get("state_intervention"),
            field="state_intervention",
        )
        expected_archive = output_root / "state" / case_key / "archives" / arm / f"day-{day_index}"
        archived_ref = _resolve_artifact_ref(
            output_root,
            intervention.get("archived_state_ref"),
            field="archived_state_ref",
        )
        if archived_ref != expected_archive.resolve():
            raise ValueError("state archive reference drift")
        archive_sha = _directory_sha256(archived_ref)
        if intervention.get("archived_state_sha256") != archive_sha:
            raise ValueError("state archive digest drift")
        measurement = archived_ref / _MEASUREMENT_CHECKPOINT_NAME
        if _file_sha256(measurement) != intervention.get("measurement_checkpoint_sha256"):
            raise ValueError("measurement checkpoint digest drift")
        expected_source_arm, expected_source_day, source = _expected_source(
            output_root=output_root,
            cases=cases,
            case_index=case_index,
            arm=arm,
            day_index=day_index,
        )
        if (
            intervention.get("next_day_source_arm") != expected_source_arm
            or intervention.get("next_day_source_day_index") != expected_source_day
        ):
            raise ValueError("state intervention source selection drift")
        loaded_sha = intervention.get("next_day_loaded_state_sha256")
        if source is None:
            if loaded_sha is not None:
                raise ValueError("stateless intervention has a loaded digest")
        elif loaded_sha != _directory_sha256(source):
            raise ValueError("loaded state/source archive digest drift")
        ids["state_archives"].add(str(archived_ref.relative_to(output_root)))
        ids["measurement_checkpoints"].add(str(measurement.relative_to(output_root)))
    return ids


def _merge_ids(target: dict[str, set[str]], source: Mapping[str, set[str]]) -> None:
    for key, values in source.items():
        overlap = target[key].intersection(values)
        if overlap:
            raise ValueError(f"duplicate {key}: {sorted(overlap)[0]}")
        target[key].update(values)


def _expected_daily_metrics_bytes(result_payload: Mapping[str, object]) -> bytes:
    readouts = _require_list(result_payload.get("daily_readouts"), field="daily_readouts")
    return b"".join(_canonical_bytes(readout) for readout in readouts)


def audit(
    *,
    execution_root: Path,
    preregistration_path: Path,
    output_root: Path,
) -> dict[str, object]:
    preregistration = _load_mapping(preregistration_path, field="preregistration")
    validate_seven_day_companion_preregistration(
        preregistration,
        repo_root=execution_root,
    )
    cases = _expected_cases(preregistration)
    expected_run_count = len(cases) * len(SEVEN_DAY_ALL_ARMS)
    run_files = sorted((output_root / "runs").glob("*.json"))
    if len(run_files) != expected_run_count:
        raise ValueError(f"expected {expected_run_count} run files, found {len(run_files)}")
    source_audit_path = output_root / "synthetic_source_audit.json"
    source_audit_sha = _file_sha256(source_audit_path)
    source_audit = _load_mapping(source_audit_path, field="source audit")
    if (
        source_audit.get("real_person_data") is not False
        or source_audit.get("consent_scope") != "synthetic-no-human-subject"
        or source_audit.get("semantic_event_tags_are_typed") is not True
        or source_audit.get("text_keyword_pii_inference_used") is not False
    ):
        raise ValueError("synthetic source audit claim boundary drift")
    scripts: dict[str, FrozenSevenDayUserScript] = {}
    for case in cases:
        script_path = (
            output_root / "user_scripts" / (hashlib.sha256(case.case_id.encode("utf-8")).hexdigest() + ".json")
        )
        script = load_frozen_seven_day_user_script(script_path)
        if (
            script.scenario_id != case.scenario_id
            or script.paraphrase_seed != case.paraphrase_seed
            or len(script.turns) != 35
        ):
            raise ValueError(f"frozen user script case drift: {case.case_id}")
        scripts[case.case_id] = script
    envelopes = []
    expected_attestation = seven_day_source_attestation_contract(preregistration)
    all_ids = {
        "run_ids": set(),
        "session_ids": set(),
        "service_instance_ids": set(),
        "console_action_ids": set(),
        "pilot_transcripts": set(),
        "pilot_metrics": set(),
        "state_archives": set(),
        "measurement_checkpoints": set(),
        "service_evidence": set(),
    }
    scope_hashes: dict[str, set[object]] = {case.case_id: set() for case in cases}
    for arm in SEVEN_DAY_ALL_ARMS:
        for case_index, case in enumerate(cases):
            run_path = _expected_run_path(output_root=output_root, case=case, arm=arm)
            run = _load_mapping(run_path, field="formal run")
            attestation = _require_mapping(run.get("source_attestation"), field="source_attestation")
            for field, expected_value in expected_attestation.items():
                if attestation.get(field) != expected_value:
                    raise ValueError(f"formal source attestation {field} drift")
            validate_seven_day_character_stack_run(
                run=run,
                preregistration=preregistration,
            )
            envelopes.append(SevenDayRunEnvelope(case=case, arm_label=arm, run=run))
            scope_hashes[case.case_id].add(run.get("user_scope_hash"))
            _merge_ids(
                all_ids,
                _verify_physical_run(
                    output_root=output_root,
                    cases=cases,
                    case_index=case_index,
                    arm=arm,
                    run=run,
                    script=scripts[case.case_id],
                    expected_source_audit_sha256=source_audit_sha,
                ),
            )
    if any(len(values) != 1 for values in scope_hashes.values()):
        raise ValueError("logical user scope differs across matched arms")
    result = evaluate_seven_day_ablation(runs=tuple(envelopes), preregistration=preregistration)
    recomputed_result = json.loads(_canonical_bytes(result.to_json()))
    on_disk_result = _load_mapping(output_root / "ablation_results.json", field="ablation results")
    if on_disk_result != recomputed_result:
        raise ValueError("recomputed ablation result differs from disk")
    if (output_root / "daily_metrics.jsonl").read_bytes() != (_expected_daily_metrics_bytes(recomputed_result)):
        raise ValueError("daily metrics export differs from recomputation")
    verdict = _load_mapping(output_root / "promotion_verdict.json", field="promotion verdict")
    expected_failed = [name for name, passed in result.gates.items() if not passed]
    expected_verdict = {
        "schema_version": "seven-day-companion-verdict.v1",
        "passed": result.passed,
        "claim_scope": "simulated-user-real-lifecycle-only",
        "external_human_value_claim_allowed": False,
        "production_promotion_authorized": False,
        "evaluation_writeback_allowed": False,
        "failed_gates": expected_failed,
    }
    if verdict != expected_verdict:
        raise ValueError("promotion verdict differs from recomputation")
    expected_manifest = {
        "schema_version": "seven-day-companion-ablation.v1",
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
        "claim_scope": "simulated-user-real-lifecycle-only",
    }
    if _load_mapping(output_root / "manifest.json", field="manifest") != (expected_manifest):
        raise ValueError("formal manifest differs from recomputation")
    report_path = output_root / "report.md"
    if not report_path.is_file() or not report_path.read_text(encoding="utf-8").strip():
        raise ValueError("formal report is missing or empty")
    log_files = sorted((output_root / "service_logs").rglob("service-*.log"))
    http_error_count = sum(len(_HTTP_ERROR_RE.findall(path.read_text(encoding="utf-8"))) for path in log_files)
    if http_error_count:
        raise ValueError(f"service logs contain {http_error_count} HTTP errors")
    expected_counts = {
        "run_ids": expected_run_count,
        "session_ids": expected_run_count * 7,
        "service_instance_ids": expected_run_count * 7,
        "console_action_ids": expected_run_count * 14,
        "pilot_transcripts": expected_run_count * 7,
        "pilot_metrics": expected_run_count * 7,
        "state_archives": expected_run_count * 6,
        "measurement_checkpoints": expected_run_count * 6,
        "service_evidence": expected_run_count * 7,
    }
    actual_counts = {key: len(value) for key, value in all_ids.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"physical artifact counts drift: {actual_counts} != {expected_counts}")
    physical_sets = {
        "pilot_transcripts": {
            str(path.relative_to(output_root)) for path in (output_root / "pilot_days").rglob("day-*-transcript.json")
        },
        "pilot_metrics": {
            str(path.relative_to(output_root)) for path in (output_root / "pilot_days").rglob("day-*-metrics.json")
        },
        "service_evidence": {
            str(path.relative_to(output_root))
            for path in (output_root / "service_evidence").rglob("session_evidence.json")
        },
        "measurement_checkpoints": {
            str(path.relative_to(output_root))
            for path in (output_root / "state").rglob(_MEASUREMENT_CHECKPOINT_NAME)
            if "archives" in path.parts
        },
    }
    for key, paths in physical_sets.items():
        if paths != all_ids[key]:
            raise ValueError(f"unreferenced or missing physical {key}")
    script_files = tuple((output_root / "user_scripts").glob("*.json"))
    if len(script_files) != len(cases):
        raise ValueError("frozen user script file count drift")
    if len(log_files) != expected_run_count * 7:
        raise ValueError("service log count drift")
    prereg_sha = _canonical_sha256(preregistration)
    if result.preregistration_sha256 != prereg_sha:
        raise ValueError("result preregistration digest drift")
    return {
        "schema_version": "seven-day-companion-independent-audit.v1",
        "passed": True,
        "preregistration_sha256": prereg_sha,
        "execution_source_snapshot": preregistration.get("execution_source_snapshot"),
        "ablation_results_sha256": _file_sha256(output_root / "ablation_results.json"),
        "promotion_verdict_sha256": _file_sha256(output_root / "promotion_verdict.json"),
        "counts": {
            "cases": len(cases),
            "runs": expected_run_count,
            "sessions": actual_counts["session_ids"],
            "turns": expected_run_count * 35,
            "readouts": expected_run_count * 14,
            "restarts": expected_run_count * 6,
            "console_actions": actual_counts["console_action_ids"],
            "pilot_transcripts": actual_counts["pilot_transcripts"],
            "pilot_metrics": actual_counts["pilot_metrics"],
            "state_archives": actual_counts["state_archives"],
            "measurement_checkpoints": actual_counts["measurement_checkpoints"],
            "service_evidence": actual_counts["service_evidence"],
            "service_logs": len(log_files),
            "http_errors": http_error_count,
        },
        "checks": {
            "full_source_tree_revalidated": True,
            "exact_preregistered_matrix": True,
            "frozen_user_turn_replay": True,
            "matched_arm_inputs": True,
            "restart_identity_chain": True,
            "state_archive_digests": True,
            "measurement_checkpoint_digests": True,
            "state_source_selection": True,
            "pilot_transcript_digests": True,
            "service_session_evidence": True,
            "evaluation_recomputed_exactly": True,
            "production_promotion_authorized": False,
            "evaluation_writeback_allowed": False,
        },
        "claim_scope": "simulated-user-real-lifecycle-only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-name", default="independent_audit.json")
    args = parser.parse_args()
    output_root = args.output_dir.resolve()
    result = audit(
        execution_root=args.execution_root.resolve(),
        preregistration_path=args.preregistration.resolve(),
        output_root=output_root,
    )
    report_path = output_root / args.report_name
    expected = _canonical_bytes(result)
    if report_path.exists() and report_path.read_bytes() != expected:
        raise ValueError(f"existing audit report differs: {report_path}")
    report_path.write_bytes(expected)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
