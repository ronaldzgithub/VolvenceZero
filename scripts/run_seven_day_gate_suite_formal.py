#!/usr/bin/env python3
"""Run one preregistered Gate 4/5/6/7/9/10 seven-day campaign."""

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
from lifeform_service.companion_evidence_profile import (
    resolve_companion_evidence_profile,
)
from volvence_zero.agent.companion_gate_suite_evidence import (
    GATE_ARM_SCHEDULES,
    CompanionGateSuiteHarness,
)
from volvence_zero.agent.companion_gate_suite_preregistration import (
    validate_companion_gate_suite_preregistration,
)
from volvence_zero.agent.seven_day_companion_evidence import (
    SevenDayExperimentCase,
)

from run_seven_day_companion_formal import (
    _LocalFormalExecutor,
    _canonical_bytes,
    _model_contract,
    _sha256,
    _verify_model,
)


def _profile_contracts(gate_id: int) -> dict[str, dict[str, object]]:
    return {arm: resolve_companion_evidence_profile(arm).intervention_contract() for arm in GATE_ARM_SCHEDULES[gate_id]}


def _prepare_user_scripts(
    *,
    root: Path,
    output_dir: Path,
    cases: tuple[SevenDayExperimentCase, ...],
    scenario_paths: dict[str, object],
    simulator: dict[str, object],
    simulator_backend: LocalTransformersUtteranceClient,
) -> dict[str, FrozenSevenDayUserScript]:
    scripts: dict[str, FrozenSevenDayUserScript] = {}
    script_root = output_dir / "user_scripts"
    script_root.mkdir(parents=True, exist_ok=True)
    for case in cases:
        path = script_root / (hashlib.sha256(case.case_id.encode("utf-8")).hexdigest() + ".json")
        if path.exists():
            script = load_frozen_seven_day_user_script(path)
            if script.scenario_id != case.scenario_id or script.paraphrase_seed != case.paraphrase_seed:
                raise ValueError(f"cached user script case drift: {path}")
            print(f"[script-resume] {case.case_id}", flush=True)
        else:
            relative = scenario_paths.get(case.scenario_id)
            if not isinstance(relative, str):
                raise ValueError("gate-suite scenario path mapping drift")
            script = build_frozen_seven_day_user_script(
                spec=load_scenario_yaml(root / relative),
                paraphrase_seed=case.paraphrase_seed,
                backend=simulator_backend,
                temperature=float(simulator["temperature"]),
            )
            path.write_bytes(_canonical_bytes(script.to_json()))
            print(f"[script-complete] {case.case_id}", flush=True)
        scripts[case.case_id] = script
    return scripts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=int, choices=tuple(GATE_ARM_SCHEDULES), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cuda:0"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18780)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-one-pair", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if sum((args.preflight_only, args.smoke_one_pair, args.execute)) != 1:
        raise ValueError("select exactly one of --preflight-only, --smoke-one-pair, or --execute")
    if args.resume and args.preflight_only:
        raise ValueError("--resume is invalid with --preflight-only")
    root = args.repo_root.resolve()
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    if not isinstance(preregistration, dict):
        raise ValueError("gate-suite preregistration must be an object")
    if preregistration.get("gate_id") != args.gate:
        raise ValueError("--gate differs from preregistration")
    validate_companion_gate_suite_preregistration(
        preregistration,
        repo_root=root,
        expected_profile_contracts=_profile_contracts(args.gate),
    )
    formal = preregistration.get("formal_run")
    scenario_ids = preregistration.get("scenario_ids")
    scenario_paths = preregistration.get("scenario_paths")
    if not isinstance(formal, dict) or not isinstance(scenario_ids, list) or not isinstance(scenario_paths, dict):
        raise ValueError("gate-suite formal schedule is malformed")
    if args.device != formal.get("execution_device"):
        raise ValueError("execution device differs from preregistration")
    seeds = formal.get("paraphrase_seeds")
    if not isinstance(seeds, list):
        raise ValueError("gate-suite formal seeds are malformed")
    sut = _model_contract(preregistration, role="sut")
    simulator = _model_contract(preregistration, role="simulator")
    _verify_model(sut)
    _verify_model(simulator)
    if sut["model_family"] == simulator["model_family"]:
        raise ValueError("formal SUT and simulator model families overlap")
    simulator_backend = LocalTransformersUtteranceClient(
        model_id=str(simulator["model_id"]),
        device=args.device,
        local_files_only=True,
        max_new_tokens=int(simulator["max_new_tokens"]),
    )
    cases = tuple(SevenDayExperimentCase(str(scenario_id), int(seed)) for scenario_id in scenario_ids for seed in seeds)
    if args.preflight_only:
        digests = {}
        for case in cases:
            relative = scenario_paths.get(case.scenario_id)
            if not isinstance(relative, str):
                raise ValueError("gate-suite scenario path mapping drift")
            script = build_frozen_seven_day_user_script(
                spec=load_scenario_yaml(root / relative),
                paraphrase_seed=case.paraphrase_seed,
                backend=simulator_backend,
                temperature=float(simulator["temperature"]),
            )
            digests[case.case_id] = script.script_sha256
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "gate_id": args.gate,
                    "device": args.device,
                    "pair_count": len(cases),
                    "run_count": len(cases) * len(GATE_ARM_SCHEDULES[args.gate]),
                    "profiles": list(GATE_ARM_SCHEDULES[args.gate]),
                    "script_sha256": digests,
                },
                sort_keys=True,
            )
        )
        return 0
    target = args.output_dir.resolve()
    if target.exists():
        if not args.resume:
            raise FileExistsError(f"gate-suite output is immutable: {target}")
    else:
        target.mkdir(parents=True)
    selected_cases = cases[:1] if args.smoke_one_pair else cases
    scripts = _prepare_user_scripts(
        root=root,
        output_dir=target,
        cases=selected_cases,
        scenario_paths=scenario_paths,
        simulator=simulator,
        simulator_backend=simulator_backend,
    )
    source_audit = {
        "schema_version": "seven-day-synthetic-source-audit.v1",
        "real_person_data": False,
        "source": "companion-bench typed scenario + local open-weight rendering",
        "consent_scope": "synthetic-no-human-subject",
        "semantic_event_tags_are_typed": True,
        "text_keyword_pii_inference_used": False,
    }
    source_audit_bytes = _canonical_bytes(source_audit)
    source_audit_path = target / "synthetic_source_audit.json"
    if source_audit_path.exists():
        if source_audit_path.read_bytes() != source_audit_bytes:
            raise ValueError("existing synthetic source audit drift")
    else:
        source_audit_path.write_bytes(source_audit_bytes)
    source_attestation = SimulatedSourceAttestation(
        simulator_model_id=str(simulator["model_id"]),
        simulator_model_family=str(simulator["model_family"]),
        sut_model_id=str(sut["model_id"]),
        sut_model_family=str(sut["model_family"]),
        model_and_adapter_fingerprint=_sha256(
            {
                "sut_model_id": sut["model_id"],
                "sut_weights_sha256": sut["weights_sha256"],
                "adapter": "none",
            }
        ),
        pii_scan_artifact_sha256=hashlib.sha256(source_audit_bytes).hexdigest(),
    )
    arms = GATE_ARM_SCHEDULES[args.gate]
    executor = _LocalFormalExecutor(
        repo_root=root,
        output_root=target,
        cases=selected_cases,
        scripts=scripts,
        source_attestation=source_attestation,
        sut_model_id=str(sut["model_id"]),
        sut_max_new_tokens=int(sut["max_new_tokens"]),
        device=args.device,
        host=args.host,
        port=args.port,
        startup_timeout_s=args.startup_timeout_s,
        virtual_start_ms=int(formal["virtual_start_ms"]),
        evidence_profile_by_arm={arm: arm for arm in arms},
        state_loading_policy_by_arm={arm: "correct-user-state" for arm in arms},
    )
    result = CompanionGateSuiteHarness(gate_id=args.gate, executor=executor).run(
        cases=selected_cases,
        preregistration=preregistration,
        output_dir=target,
    )
    print(
        json.dumps(
            {
                "gate_id": args.gate,
                "mechanism_supported": result.mechanism_supported,
                "causal_supported": result.causal_supported,
                "pair_count": result.pair_count,
                "run_count": result.run_count,
                "production_promotion_authorized": False,
            },
            sort_keys=True,
        )
    )
    if args.smoke_one_pair:
        return 0 if result.mechanism_supported else 2
    return 0 if result.causal_supported else 2


if __name__ == "__main__":
    raise SystemExit(main())
