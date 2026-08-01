#!/usr/bin/env python3
"""Run frozen local-model Gate 8/11 simulated capture and blind-packet export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from companion_bench.seven_day_driver import (
    FrozenSevenDayUserScript,
    build_frozen_seven_day_user_script,
    load_frozen_seven_day_user_script,
)
from companion_bench.spec import load_scenario_yaml
from companion_bench.user_simulator import LocalTransformersUtteranceClient
from lifeform_evolution.seven_day_companion import SimulatedSourceAttestation
from run_seven_day_companion_formal import (
    _LocalFormalExecutor,
    _canonical_bytes,
    _model_contract,
    _sha256,
    _verify_model,
)
from volvence_zero.agent.gate811_simulated_capture import (
    audit_gate811_simulated_capture_compatibility,
    export_gate811_simulated_pilot,
)
from volvence_zero.agent.seven_day_companion_evidence import (
    SevenDayExperimentCase,
    SevenDayRunEnvelope,
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


def _load_json(path: Path, *, field: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must be an object")
    return payload


def _write_exact(path: Path, payload: object) -> None:
    data = _canonical_bytes(payload)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing immutable artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _capture_cases(
    *,
    scenario_ids: object,
    capture_seeds: object,
) -> tuple[SevenDayExperimentCase, ...]:
    if not isinstance(scenario_ids, list) or not all(
        isinstance(item, str) and item for item in scenario_ids
    ):
        raise ValueError("seven-day scenario_ids are malformed")
    if not isinstance(capture_seeds, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in capture_seeds
    ):
        raise ValueError("Gate 8/11 capture_seeds are malformed")
    return tuple(
        SevenDayExperimentCase(scenario_id, seed)
        for scenario_id in scenario_ids
        for seed in capture_seeds
    )


def _script_path(root: Path, case: SevenDayExperimentCase) -> Path:
    return root / (
        hashlib.sha256(case.case_id.encode("utf-8")).hexdigest() + ".json"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--seven-day-preregistration", type=Path, required=True)
    parser.add_argument("--gate811-preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18778)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.execute:
        raise ValueError("select exactly one of --preflight-only or --execute")
    if args.preflight_only and args.resume:
        raise ValueError("--resume is invalid with --preflight-only")

    repo_root = args.repo_root.resolve()
    seven_day_path = args.seven_day_preregistration.resolve()
    gate811_path = args.gate811_preregistration.resolve()
    seven_day_prereg = _load_json(
        seven_day_path,
        field="seven-day preregistration",
    )
    gate811_prereg = _load_json(
        gate811_path,
        field="Gate 8/11 preregistration",
    )
    validate_seven_day_companion_preregistration(
        seven_day_prereg,
        repo_root=repo_root,
    )
    compatibility = audit_gate811_simulated_capture_compatibility(
        gate811_prereg
    )
    if not compatibility.compatible_with_frozen_v1:
        raise ValueError("frozen Gate 8/11 v1 rejects simulated capture")
    seven_day_formal = seven_day_prereg.get("formal_run")
    scenario_paths = seven_day_prereg.get("scenario_paths")
    capture_contract = gate811_prereg.get("capture")
    if (
        not isinstance(seven_day_formal, dict)
        or not isinstance(scenario_paths, dict)
        or not isinstance(capture_contract, dict)
    ):
        raise ValueError("capture preregistration contracts are malformed")
    if args.device != seven_day_formal.get("execution_device"):
        raise ValueError("execution device differs from seven-day preregistration")
    cases = _capture_cases(
        scenario_ids=seven_day_prereg.get("scenario_ids"),
        capture_seeds=capture_contract.get("capture_seeds"),
    )
    sut = _model_contract(seven_day_prereg, role="sut")
    simulator = _model_contract(seven_day_prereg, role="simulator")
    _verify_model(sut)
    _verify_model(simulator)
    if sut["model_family"] == simulator["model_family"]:
        raise ValueError("capture SUT and simulator model families overlap")
    simulator_backend = LocalTransformersUtteranceClient(
        model_id=str(simulator["model_id"]),
        device=args.device,
        local_files_only=True,
        max_new_tokens=int(simulator["max_new_tokens"]),
    )

    target = args.output_dir.resolve()
    if target.exists() and not args.resume and args.execute:
        raise FileExistsError(f"capture output is immutable: {target}")
    if args.execute:
        target.mkdir(parents=True, exist_ok=True)
    script_root = target / "user_scripts"
    scripts: dict[str, FrozenSevenDayUserScript] = {}
    script_digests: dict[str, str] = {}
    for case in cases:
        path = _script_path(script_root, case)
        if path.exists():
            script = load_frozen_seven_day_user_script(path)
            if (
                script.scenario_id != case.scenario_id
                or script.paraphrase_seed != case.paraphrase_seed
            ):
                raise ValueError(f"cached capture script case drift: {path}")
            print(f"[script-resume] {case.case_id}", flush=True)
        else:
            print(f"[script] {case.case_id}", flush=True)
            scenario_path = scenario_paths.get(case.scenario_id)
            if not isinstance(scenario_path, str):
                raise ValueError("capture scenario path is missing")
            script = build_frozen_seven_day_user_script(
                spec=load_scenario_yaml(repo_root / scenario_path),
                paraphrase_seed=case.paraphrase_seed,
                backend=simulator_backend,
                temperature=float(simulator["temperature"]),
            )
            if args.execute:
                _write_exact(path, script.to_json())
            print(f"[script-complete] {case.case_id}", flush=True)
        scripts[case.case_id] = script
        script_digests[case.case_id] = script.script_sha256
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "case_count": len(cases),
                    "run_count": len(cases) * len(_CAPTURE_ARMS),
                    "script_sha256": script_digests,
                },
                sort_keys=True,
            )
        )
        return 0

    source_audit = {
        "schema_version": "seven-day-synthetic-source-audit.v1",
        "real_person_data": False,
        "source": "companion-bench typed scenario + local open-weight rendering",
        "consent_scope": "synthetic-no-human-subject",
        "semantic_event_tags_are_typed": True,
        "text_keyword_pii_inference_used": False,
    }
    _write_exact(target / "synthetic_source_audit.json", source_audit)
    source_audit_sha = hashlib.sha256(_canonical_bytes(source_audit)).hexdigest()
    model_fingerprint = _sha256(
        {
            "sut_model_id": sut["model_id"],
            "sut_weights_sha256": sut["weights_sha256"],
            "adapter": "none",
        }
    )
    executor = _LocalFormalExecutor(
        repo_root=repo_root,
        output_root=target,
        cases=cases,
        scripts=scripts,
        source_attestation=SimulatedSourceAttestation(
            simulator_model_id=str(simulator["model_id"]),
            simulator_model_family=str(simulator["model_family"]),
            sut_model_id=str(sut["model_id"]),
            sut_model_family=str(sut["model_family"]),
            model_and_adapter_fingerprint=model_fingerprint,
            pii_scan_artifact_sha256=source_audit_sha,
        ),
        sut_model_id=str(sut["model_id"]),
        sut_max_new_tokens=int(sut["max_new_tokens"]),
        device=args.device,
        host=args.host,
        port=args.port,
        startup_timeout_s=args.startup_timeout_s,
        virtual_start_ms=int(seven_day_formal["virtual_start_ms"]),
    )
    run_root = target / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    envelopes = []
    for arm in _CAPTURE_ARMS:
        for case in cases:
            output_path = run_root / (
                hashlib.sha256(
                    f"{case.case_id}\0{arm}".encode("utf-8")
                ).hexdigest()
                + ".json"
            )
            payload = executor.execute(
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
    gate811_sha = hashlib.sha256(gate811_path.read_bytes()).hexdigest()
    manifest = export_gate811_simulated_pilot(
        runs=tuple(envelopes),
        preregistration=gate811_prereg,
        preregistration_sha256=gate811_sha,
        output_dir=target,
    )
    source_manifest = {
        "schema_version": "gate811-simulated-source-runs.v1",
        "seven_day_preregistration_sha256": hashlib.sha256(
            seven_day_path.read_bytes()
        ).hexdigest(),
        "gate811_preregistration_sha256": gate811_sha,
        "case_count": len(cases),
        "arm_count": len(_CAPTURE_ARMS),
        "run_count": len(envelopes),
        "session_count": len(envelopes) * 7,
        "exchange_count": len(envelopes) * 7 * 5,
        "capture_record_count": manifest["capture_record_count"],
        "blinded_pair_count": manifest["pair_count"],
        "human_ratings_pending": True,
        "real_user_product_value_claim_allowed": False,
        "production_promotion_authorized": False,
    }
    _write_exact(target / "source_run_manifest.json", source_manifest)
    print(json.dumps(source_manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
