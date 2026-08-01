#!/usr/bin/env python3
"""Independently audit one completed seven-day gate-suite bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from companion_bench.seven_day_driver import (
    load_frozen_seven_day_user_script,
)
from lifeform_service.companion_evidence_profile import (
    resolve_companion_evidence_profile,
)
from volvence_zero.agent.companion_gate_suite_evidence import (
    GATE_ARM_SCHEDULES,
    evaluate_companion_gate_suite,
)
from volvence_zero.agent.companion_gate_suite_preregistration import (
    validate_companion_gate_suite_preregistration,
)

from audit_seven_day_companion_formal import (
    _canonical_bytes,
    _expected_cases,
    _expected_run_path,
    _file_sha256,
    _load_mapping,
    _require_mapping,
    _verify_physical_run,
)


_HTTP_ERROR_RE = re.compile(r'HTTP/1\.[01]" [45][0-9][0-9]')


def _profile_contracts(gate_id: int) -> dict[str, dict[str, object]]:
    return {arm: resolve_companion_evidence_profile(arm).intervention_contract() for arm in GATE_ARM_SCHEDULES[gate_id]}


def audit(
    *,
    gate_id: int,
    execution_root: Path,
    preregistration_path: Path,
    output_root: Path,
) -> dict[str, object]:
    preregistration = _load_mapping(preregistration_path, field="gate-suite preregistration")
    if preregistration.get("gate_id") != gate_id:
        raise ValueError("--gate differs from preregistration")
    validate_companion_gate_suite_preregistration(
        preregistration,
        repo_root=execution_root,
        expected_profile_contracts=_profile_contracts(gate_id),
    )
    arms = GATE_ARM_SCHEDULES[gate_id]
    cases = _expected_cases(preregistration)
    expected_run_count = len(cases) * len(arms)
    run_files = sorted((output_root / "runs").glob("*.json"))
    if len(run_files) != expected_run_count:
        raise ValueError(f"expected {expected_run_count} gate runs, found {len(run_files)}")
    source_audit_path = output_root / "synthetic_source_audit.json"
    source_audit = _load_mapping(source_audit_path, field="source audit")
    if (
        source_audit.get("real_person_data") is not False
        or source_audit.get("consent_scope") != "synthetic-no-human-subject"
        or source_audit.get("semantic_event_tags_are_typed") is not True
        or source_audit.get("text_keyword_pii_inference_used") is not False
    ):
        raise ValueError("gate-suite synthetic source audit drift")
    source_audit_sha = _file_sha256(source_audit_path)
    scripts = {}
    for case in cases:
        path = output_root / "user_scripts" / (hashlib.sha256(case.case_id.encode("utf-8")).hexdigest() + ".json")
        script = load_frozen_seven_day_user_script(path)
        if (
            script.scenario_id != case.scenario_id
            or script.paraphrase_seed != case.paraphrase_seed
            or len(script.turns) != 35
        ):
            raise ValueError(f"gate-suite frozen script drift: {case.case_id}")
        scripts[case.case_id] = script
    formal_models = _require_mapping(preregistration.get("formal_models"), field="formal_models")
    sut = _require_mapping(formal_models.get("sut"), field="sut")
    simulator = _require_mapping(formal_models.get("simulator"), field="simulator")
    model_fingerprint = hashlib.sha256(
        _canonical_bytes(
            {
                "sut_model_id": sut.get("model_id"),
                "sut_weights_sha256": sut.get("weights_sha256"),
                "adapter": "none",
            }
        )
    ).hexdigest()
    runs = {}
    scope_hashes = {case.case_id: set() for case in cases}
    profile_files = set()
    artifact_ids: dict[str, set[str]] = {}
    for arm in arms:
        for case_index, case in enumerate(cases):
            path = _expected_run_path(output_root=output_root, case=case, arm=arm)
            run = _load_mapping(path, field="gate-suite run")
            source = _require_mapping(run.get("source_attestation"), field="source_attestation")
            expected_source = {
                "simulator_model_id": simulator.get("model_id"),
                "simulator_model_family": simulator.get("model_family"),
                "sut_model_id": sut.get("model_id"),
                "sut_model_family": sut.get("model_family"),
                "model_and_adapter_fingerprint": model_fingerprint,
            }
            if any(source.get(field) != value for field, value in expected_source.items()):
                raise ValueError("gate-suite source/model attestation drift")
            case_key = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()[:20]
            profile_path = output_root / "service_evidence" / case_key / arm / "companion_evidence_runtime_profile.json"
            profile = _load_mapping(profile_path, field="runtime profile attestation")
            if profile != run.get("runtime_profile_attestation"):
                raise ValueError("run/profile attestation file mismatch")
            profile_files.add(str(profile_path.relative_to(output_root)))
            ids = _verify_physical_run(
                output_root=output_root,
                cases=cases,
                case_index=case_index,
                arm=arm,
                run=run,
                script=scripts[case.case_id],
                expected_source_audit_sha256=source_audit_sha,
            )
            for category, values in ids.items():
                existing = artifact_ids.setdefault(category, set())
                overlap = existing.intersection(values)
                if overlap:
                    raise ValueError(f"duplicate physical {category}: {sorted(overlap)[0]}")
                existing.update(values)
            runs[(case.case_id, arm)] = run
            scope_hashes[case.case_id].add(run.get("user_scope_hash"))
    if any(len(values) != 1 for values in scope_hashes.values()):
        raise ValueError("gate-suite logical user scope differs across arms")
    result = evaluate_companion_gate_suite(
        gate_id=gate_id,
        cases=cases,
        runs=runs,
        preregistration=preregistration,
    )
    evaluation_path = output_root / f"gate{gate_id}_evaluation.json"
    on_disk = _load_mapping(evaluation_path, field="gate evaluation")
    recomputed = json.loads(_canonical_bytes(result.to_json()))
    if on_disk != recomputed:
        raise ValueError("recomputed gate evaluation differs from disk")
    log_files = sorted((output_root / "service_logs").rglob("service-*.log"))
    if len(log_files) != expected_run_count * 7:
        raise ValueError("gate-suite service log count drift")
    http_error_count = sum(len(_HTTP_ERROR_RE.findall(path.read_text(encoding="utf-8"))) for path in log_files)
    if http_error_count:
        raise ValueError(f"gate-suite service logs contain {http_error_count} HTTP errors")
    if len(profile_files) != expected_run_count:
        raise ValueError("gate-suite runtime profile attestation count drift")
    expected_artifact_counts = {
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
    actual_artifact_counts = {name: len(values) for name, values in artifact_ids.items()}
    if actual_artifact_counts != expected_artifact_counts:
        raise ValueError(
            f"gate-suite physical artifact count drift: {actual_artifact_counts} != {expected_artifact_counts}"
        )
    return {
        "schema_version": "companion-gate-suite-independent-audit.v1",
        "audit_passed": True,
        "gate_id": gate_id,
        "preregistration_sha256": result.preregistration_sha256,
        "evaluation_sha256": _file_sha256(evaluation_path),
        "counts": {
            "pairs": len(cases),
            "runs": expected_run_count,
            "sessions": expected_run_count * 7,
            "turns": expected_run_count * 35,
            "restarts": expected_run_count * 6,
            "runtime_profile_attestations": len(profile_files),
            "service_logs": len(log_files),
            "http_errors": http_error_count,
        },
        "verdict": {
            "mechanism_supported": result.mechanism_supported,
            "causal_supported": result.causal_supported,
            "production_promotion_authorized": False,
        },
        "claim_scope": result.claim_scope,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=int, choices=tuple(GATE_ARM_SCHEDULES), required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-name")
    args = parser.parse_args()
    output_root = args.output_dir.resolve()
    result = audit(
        gate_id=args.gate,
        execution_root=args.execution_root.resolve(),
        preregistration_path=args.preregistration.resolve(),
        output_root=output_root,
    )
    report_name = args.report_name or f"gate{args.gate}_independent_audit.json"
    path = output_root / report_name
    encoded = _canonical_bytes(result)
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError(f"existing gate-suite audit differs: {path}")
    path.write_bytes(encoded)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
