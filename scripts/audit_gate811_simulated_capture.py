#!/usr/bin/env python3
"""Independently audit Gate 8/11 simulated capture and blind packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_seven_day_companion_formal import (
    _HTTP_ERROR_RE,
    _MEASUREMENT_CHECKPOINT_NAME,
    _canonical_bytes,
    _file_sha256,
    _load_mapping,
    _merge_ids,
    _verify_physical_run,
)
from companion_bench.seven_day_driver import (
    FrozenSevenDayUserScript,
    load_frozen_seven_day_user_script,
)
from volvence_zero.agent.gate811_human_anchor_tooling import (
    GATE811_PACKET_SCHEMA_VERSION,
    GATE811_RATING_TEMPLATE_SCHEMA_VERSION,
    build_gate811_pilot_packet,
)
from volvence_zero.agent.gate811_simulated_capture import (
    audit_gate811_simulated_capture_compatibility,
    build_gate811_simulated_capture,
)
from volvence_zero.agent.seven_day_companion_evidence import (
    SevenDayExperimentCase,
    SevenDayRunEnvelope,
    _validate_run,
)
from volvence_zero.agent.seven_day_companion_preregistration import (
    validate_seven_day_companion_preregistration,
)


_CAPTURE_ARMS = (
    "correct-user-state",
    "stateless",
    "sleep-consolidation",
    "no-sleep",
)


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _cases(
    *,
    seven_day_prereg: Mapping[str, object],
    gate811_prereg: Mapping[str, object],
) -> tuple[SevenDayExperimentCase, ...]:
    scenario_ids = _require_list(
        seven_day_prereg.get("scenario_ids"), field="scenario_ids"
    )
    capture = _require_mapping(
        gate811_prereg.get("capture"), field="capture"
    )
    seeds = _require_list(capture.get("capture_seeds"), field="capture_seeds")
    if not all(isinstance(value, str) and value for value in scenario_ids):
        raise ValueError("scenario_ids contains an invalid value")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in seeds
    ):
        raise ValueError("capture_seeds contains an invalid value")
    return tuple(
        SevenDayExperimentCase(str(scenario_id), int(seed))
        for scenario_id in scenario_ids
        for seed in seeds
    )


def _run_path(
    *, output_root: Path, case: SevenDayExperimentCase, arm: str
) -> Path:
    name = hashlib.sha256(
        f"{case.case_id}\0{arm}".encode("utf-8")
    ).hexdigest()
    return output_root / "runs" / f"{name}.json"


def _script(
    *, output_root: Path, case: SevenDayExperimentCase
) -> FrozenSevenDayUserScript:
    path = output_root / "user_scripts" / (
        hashlib.sha256(case.case_id.encode("utf-8")).hexdigest() + ".json"
    )
    script = load_frozen_seven_day_user_script(path)
    if (
        script.scenario_id != case.scenario_id
        or script.paraphrase_seed != case.paraphrase_seed
        or len(script.turns) != 35
    ):
        raise ValueError(f"capture user script drift: {case.case_id}")
    return script


def _verify_gate811_preregistration_files(
    *,
    execution_root: Path,
    repo_root: Path,
    preregistration: Mapping[str, object],
) -> None:
    code_manifest = _require_mapping(
        preregistration.get("code_manifest"), field="code_manifest"
    )
    actual_manifest = {}
    for relative, digest in code_manifest.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("Gate 8/11 code manifest is malformed")
        actual = _file_sha256(execution_root / relative)
        if actual != digest:
            raise ValueError(f"Gate 8/11 code file drift: {relative}")
        actual_manifest[relative] = actual
    if hashlib.sha256(_canonical_bytes(actual_manifest)).hexdigest() != (
        preregistration.get("code_tree_sha256")
    ):
        raise ValueError("Gate 8/11 code tree digest drift")
    bindings = _require_list(
        preregistration.get("source_bindings"), field="source_bindings"
    )
    for raw_binding in bindings:
        binding = _require_mapping(raw_binding, field="source_binding")
        manifest_path = binding.get("manifest_path")
        expected_sha = binding.get("manifest_sha256")
        if not isinstance(manifest_path, str) or not isinstance(
            expected_sha, str
        ):
            raise ValueError("Gate 8/11 source binding is malformed")
        if _file_sha256(repo_root / manifest_path) != expected_sha:
            raise ValueError(f"Gate 8/11 source binding drift: {manifest_path}")


def _expected_manifest(
    *, bundle: Mapping[str, object], capture: Mapping[str, object]
) -> dict[str, object]:
    packet_bytes = _canonical_bytes(bundle["packet"])
    key_bytes = _canonical_bytes(bundle["internal_key"])
    rating_csv = bundle["rating_template_csv"]
    if not isinstance(rating_csv, str):
        raise ValueError("rating template must be text")
    files = {
        "pilot_packet_blinded.json": packet_bytes,
        "pilot_key_internal.json": key_bytes,
        "pilot_rating_template.csv": rating_csv.encode("utf-8"),
    }
    records = _require_list(capture.get("records"), field="capture.records")
    packet = _require_mapping(bundle.get("packet"), field="packet")
    return {
        "schema_version": GATE811_PACKET_SCHEMA_VERSION,
        "rating_template_schema_version": (
            GATE811_RATING_TEMPLATE_SCHEMA_VERSION
        ),
        "required_files": sorted(files),
        "sha256": {
            relative: hashlib.sha256(content).hexdigest()
            for relative, content in sorted(files.items())
        },
        "pilot_only": True,
        "human_anchor_claim_allowed": False,
        "production_promotion_authorized": False,
        "capture_record_count": len(records),
        "capture_source_scope": capture["capture_source_scope"],
        "real_user_product_value_claim_allowed": False,
        "human_ratings_pending": True,
        "pair_count": packet["pair_count"],
    }


def audit(
    *,
    repo_root: Path,
    execution_root: Path,
    seven_day_prereg_path: Path,
    gate811_prereg_path: Path,
    output_root: Path,
) -> dict[str, object]:
    seven_day_prereg = _load_mapping(
        seven_day_prereg_path, field="seven-day preregistration"
    )
    gate811_prereg = _load_mapping(
        gate811_prereg_path, field="Gate 8/11 preregistration"
    )
    validate_seven_day_companion_preregistration(
        seven_day_prereg, repo_root=execution_root
    )
    _verify_gate811_preregistration_files(
        execution_root=execution_root,
        repo_root=repo_root,
        preregistration=gate811_prereg,
    )
    compatibility = audit_gate811_simulated_capture_compatibility(
        gate811_prereg
    )
    if (
        not compatibility.compatible_with_frozen_v1
        or compatibility.human_raters_still_required is not True
        or compatibility.production_promotion_authorized is not False
    ):
        raise ValueError("Gate 8/11 compatibility boundary drift")
    cases = _cases(
        seven_day_prereg=seven_day_prereg,
        gate811_prereg=gate811_prereg,
    )
    expected_run_count = len(cases) * len(_CAPTURE_ARMS)
    run_files = sorted((output_root / "runs").glob("*.json"))
    if len(run_files) != expected_run_count:
        raise ValueError(
            f"expected {expected_run_count} capture runs, found {len(run_files)}"
        )
    source_audit_sha = _file_sha256(
        output_root / "synthetic_source_audit.json"
    )
    source_audit = _load_mapping(
        output_root / "synthetic_source_audit.json", field="source audit"
    )
    if (
        source_audit.get("real_person_data") is not False
        or source_audit.get("consent_scope") != "synthetic-no-human-subject"
        or source_audit.get("semantic_event_tags_are_typed") is not True
        or source_audit.get("text_keyword_pii_inference_used") is not False
    ):
        raise ValueError("capture source audit claim boundary drift")
    scripts = {case.case_id: _script(output_root=output_root, case=case) for case in cases}
    envelopes = []
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
    scope_hashes = {case.case_id: set() for case in cases}
    matched_fields = (
        "simulator_model_id",
        "simulator_model_family",
        "sut_model_id",
        "sut_model_family",
        "model_and_adapter_fingerprint",
    )
    matched_attestations = {
        case.case_id: {field: set() for field in matched_fields}
        for case in cases
    }
    calendars = {case.case_id: set() for case in cases}
    for arm in _CAPTURE_ARMS:
        for case_index, case in enumerate(cases):
            run = _load_mapping(
                _run_path(output_root=output_root, case=case, arm=arm),
                field="capture source run",
            )
            envelope = SevenDayRunEnvelope(case=case, arm_label=arm, run=run)
            _validate_run(envelope)
            envelopes.append(envelope)
            scope_hashes[case.case_id].add(run.get("user_scope_hash"))
            attestation = _require_mapping(
                run.get("source_attestation"), field="source_attestation"
            )
            for field in matched_fields:
                matched_attestations[case.case_id][field].add(
                    attestation.get(field)
                )
            days = _require_list(run.get("days"), field="run.days")
            calendars[case.case_id].add(
                tuple(
                    _require_mapping(day, field="run.day").get(
                        "virtual_observed_at_ms"
                    )
                    for day in days
                )
            )
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
        raise ValueError("capture logical user scope differs across arms")
    if any(len(values) != 1 for values in calendars.values()):
        raise ValueError("capture virtual calendar differs across arms")
    if any(
        len(values) != 1
        for fields in matched_attestations.values()
        for values in fields.values()
    ):
        raise ValueError("capture model attestation differs across arms")
    gate811_sha = _file_sha256(gate811_prereg_path)
    capture = build_gate811_simulated_capture(
        runs=tuple(envelopes),
        preregistration=gate811_prereg,
        preregistration_sha256=gate811_sha,
    )
    if _canonical_bytes(capture) != (
        output_root / "simulated_capture.json"
    ).read_bytes():
        raise ValueError("simulated capture differs from recomputation")
    compatibility_payload = compatibility.to_json()
    if _canonical_bytes(compatibility_payload) != (
        output_root / "compatibility_audit.json"
    ).read_bytes():
        raise ValueError("compatibility audit differs from recomputation")
    bundle = build_gate811_pilot_packet(
        capture=capture,
        preregistration=gate811_prereg,
        preregistration_sha256=gate811_sha,
    )
    if _canonical_bytes(bundle["packet"]) != (
        output_root / "pilot_packet_blinded.json"
    ).read_bytes():
        raise ValueError("blinded packet differs from recomputation")
    if _canonical_bytes(bundle["internal_key"]) != (
        output_root / "pilot_key_internal.json"
    ).read_bytes():
        raise ValueError("internal blind key differs from recomputation")
    rating_csv = bundle["rating_template_csv"]
    if not isinstance(rating_csv, str) or rating_csv.encode("utf-8") != (
        output_root / "pilot_rating_template.csv"
    ).read_bytes():
        raise ValueError("rating template differs from recomputation")
    expected_manifest = _expected_manifest(bundle=bundle, capture=capture)
    if _load_mapping(output_root / "manifest.json", field="manifest") != (
        expected_manifest
    ):
        raise ValueError("capture manifest differs from recomputation")
    packet = _require_mapping(bundle["packet"], field="packet")
    pair_count = packet.get("pair_count")
    records = _require_list(capture.get("records"), field="capture.records")
    source_manifest = _load_mapping(
        output_root / "source_run_manifest.json", field="source run manifest"
    )
    expected_source_manifest = {
        "schema_version": "gate811-simulated-source-runs.v1",
        "seven_day_preregistration_sha256": _file_sha256(
            seven_day_prereg_path
        ),
        "gate811_preregistration_sha256": gate811_sha,
        "case_count": len(cases),
        "arm_count": len(_CAPTURE_ARMS),
        "run_count": expected_run_count,
        "session_count": expected_run_count * 7,
        "exchange_count": expected_run_count * 35,
        "capture_record_count": len(records),
        "blinded_pair_count": pair_count,
        "human_ratings_pending": True,
        "real_user_product_value_claim_allowed": False,
        "production_promotion_authorized": False,
    }
    if source_manifest != expected_source_manifest:
        raise ValueError("source run manifest differs from physical evidence")
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
    actual_counts = {key: len(values) for key, values in all_ids.items()}
    if actual_counts != expected_counts:
        raise ValueError("capture physical artifact counts drift")
    physical_sets = {
        "pilot_transcripts": {
            str(path.relative_to(output_root))
            for path in (output_root / "pilot_days").rglob(
                "day-*-transcript.json"
            )
        },
        "pilot_metrics": {
            str(path.relative_to(output_root))
            for path in (output_root / "pilot_days").rglob(
                "day-*-metrics.json"
            )
        },
        "service_evidence": {
            str(path.relative_to(output_root))
            for path in (output_root / "service_evidence").rglob(
                "session_evidence.json"
            )
        },
        "measurement_checkpoints": {
            str(path.relative_to(output_root))
            for path in (output_root / "state").rglob(
                _MEASUREMENT_CHECKPOINT_NAME
            )
            if "archives" in path.parts
        },
    }
    for key, paths in physical_sets.items():
        if paths != all_ids[key]:
            raise ValueError(f"unreferenced or missing capture {key}")
    if len(tuple((output_root / "user_scripts").glob("*.json"))) != len(cases):
        raise ValueError("capture user script file count drift")
    log_files = sorted((output_root / "service_logs").rglob("service-*.log"))
    if len(log_files) != expected_run_count * 7:
        raise ValueError("capture service log count drift")
    http_error_count = sum(
        len(_HTTP_ERROR_RE.findall(path.read_text(encoding="utf-8")))
        for path in log_files
    )
    if http_error_count:
        raise ValueError(
            f"capture service logs contain {http_error_count} HTTP errors"
        )
    return {
        "schema_version": "gate811-simulated-capture-independent-audit.v1",
        "passed": True,
        "seven_day_preregistration_sha256": _file_sha256(
            seven_day_prereg_path
        ),
        "gate811_preregistration_sha256": gate811_sha,
        "execution_source_snapshot": seven_day_prereg.get(
            "execution_source_snapshot"
        ),
        "simulated_capture_sha256": _file_sha256(
            output_root / "simulated_capture.json"
        ),
        "blinded_packet_sha256": _file_sha256(
            output_root / "pilot_packet_blinded.json"
        ),
        "counts": {
            "cases": len(cases),
            "runs": expected_run_count,
            "sessions": actual_counts["session_ids"],
            "exchanges": expected_run_count * 35,
            "console_actions": actual_counts["console_action_ids"],
            "restarts": expected_run_count * 6,
            "capture_records": len(records),
            "blinded_pairs": pair_count,
            "rating_rows": int(pair_count) * 3,
            "http_errors": http_error_count,
        },
        "checks": {
            "full_source_tree_revalidated": True,
            "exact_capture_matrix": True,
            "source_run_artifacts_revalidated": True,
            "capture_recomputed_exactly": True,
            "blind_packet_recomputed_exactly": True,
            "internal_key_separate": True,
            "human_ratings_pending": True,
            "real_user_product_value_claim_allowed": False,
            "production_promotion_authorized": False,
        },
        "claim_scope": "unrated-simulated-user-transcript-packet-only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--seven-day-preregistration", type=Path, required=True)
    parser.add_argument("--gate811-preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--report-name", default="independent_audit.json"
    )
    args = parser.parse_args()
    output_root = args.output_dir.resolve()
    result = audit(
        repo_root=args.repo_root.resolve(),
        execution_root=args.execution_root.resolve(),
        seven_day_prereg_path=args.seven_day_preregistration.resolve(),
        gate811_prereg_path=args.gate811_preregistration.resolve(),
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
