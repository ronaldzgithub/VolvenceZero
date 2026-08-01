#!/usr/bin/env python3
"""Run the preregistered matched Gate 1 seven-day companion campaign."""

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
from volvence_zero.agent.gate1_seven_day_evidence import (
    GATE1_PE_OFF_ARM,
    GATE1_PE_ON_ARM,
    Gate1SevenDayHarness,
)
from volvence_zero.agent.gate1_seven_day_preregistration import (
    validate_gate1_seven_day_preregistration,
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
        script_path = script_root / (
            hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()
            + ".json"
        )
        if script_path.exists():
            script = load_frozen_seven_day_user_script(script_path)
            if (
                script.scenario_id != case.scenario_id
                or script.paraphrase_seed != case.paraphrase_seed
            ):
                raise ValueError(f"cached user script case drift: {script_path}")
            print(f"[script-resume] {case.case_id}", flush=True)
        else:
            print(f"[script] {case.case_id}", flush=True)
            relative = scenario_paths.get(case.scenario_id)
            if not isinstance(relative, str):
                raise ValueError("Gate 1 scenario path mapping drift")
            spec = load_scenario_yaml(root / relative)
            script = build_frozen_seven_day_user_script(
                spec=spec,
                paraphrase_seed=case.paraphrase_seed,
                backend=simulator_backend,
                temperature=float(simulator["temperature"]),
            )
            script_path.write_bytes(_canonical_bytes(script.to_json()))
            print(f"[script-complete] {case.case_id}", flush=True)
        scripts[case.case_id] = script
    return scripts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("mps", "cuda", "cuda:0"), required=True
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18779)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-one-pair", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if sum((args.preflight_only, args.smoke_one_pair, args.execute)) != 1:
        raise ValueError(
            "select exactly one of --preflight-only, --smoke-one-pair, "
            "or --execute"
        )
    if args.resume and args.preflight_only:
        raise ValueError("--resume is invalid with --preflight-only")

    root = args.repo_root.resolve()
    preregistration = json.loads(
        args.preregistration.read_text(encoding="utf-8")
    )
    validate_gate1_seven_day_preregistration(
        preregistration, repo_root=root
    )
    formal = preregistration.get("formal_run")
    scenario_ids = preregistration.get("scenario_ids")
    scenario_paths = preregistration.get("scenario_paths")
    if (
        not isinstance(formal, dict)
        or not isinstance(scenario_ids, list)
        or not isinstance(scenario_paths, dict)
    ):
        raise ValueError("Gate 1 formal schedule is malformed")
    if args.device != formal.get("execution_device"):
        raise ValueError("execution device differs from preregistration")
    seeds = formal.get("paraphrase_seeds")
    if not isinstance(seeds, list):
        raise ValueError("Gate 1 formal seeds are malformed")

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
    all_cases = tuple(
        SevenDayExperimentCase(str(scenario_id), int(seed))
        for scenario_id in scenario_ids
        for seed in seeds
    )
    if args.preflight_only:
        digests = {}
        for case in all_cases:
            relative = scenario_paths.get(case.scenario_id)
            if not isinstance(relative, str):
                raise ValueError("Gate 1 scenario path mapping drift")
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
                    "device": args.device,
                    "pair_count": len(all_cases),
                    "run_count": len(all_cases) * 2,
                    "profiles": [GATE1_PE_ON_ARM, GATE1_PE_OFF_ARM],
                    "script_sha256": digests,
                },
                sort_keys=True,
            )
        )
        return 0

    target = args.output_dir.resolve()
    if target.exists():
        if not args.resume:
            raise FileExistsError(f"Gate 1 output is immutable: {target}")
    else:
        target.mkdir(parents=True)
    cases = all_cases[:1] if args.smoke_one_pair else all_cases
    scripts = _prepare_user_scripts(
        root=root,
        output_dir=target,
        cases=cases,
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
        pii_scan_artifact_sha256=hashlib.sha256(
            source_audit_bytes
        ).hexdigest(),
    )
    executor = _LocalFormalExecutor(
        repo_root=root,
        output_root=target,
        cases=cases,
        scripts=scripts,
        source_attestation=source_attestation,
        sut_model_id=str(sut["model_id"]),
        sut_max_new_tokens=int(sut["max_new_tokens"]),
        device=args.device,
        host=args.host,
        port=args.port,
        startup_timeout_s=args.startup_timeout_s,
        virtual_start_ms=int(formal["virtual_start_ms"]),
        evidence_profile_by_arm={
            GATE1_PE_ON_ARM: GATE1_PE_ON_ARM,
            GATE1_PE_OFF_ARM: GATE1_PE_OFF_ARM,
        },
        state_loading_policy_by_arm={
            GATE1_PE_ON_ARM: "correct-user-state",
            GATE1_PE_OFF_ARM: "correct-user-state",
        },
    )
    result = Gate1SevenDayHarness(executor=executor).run(
        cases=cases,
        preregistration=(
            preregistration
            if not args.smoke_one_pair
            else {
                **preregistration,
                "formal_run": {
                    **formal,
                    "pair_count": 1,
                    "run_count": 2,
                },
            }
        ),
        output_dir=target,
    )
    summary = {
        "mechanism_supported": result.mechanism_supported,
        "causal_supported": result.causal_supported,
        "pair_count": result.pair_count,
        "run_count": result.run_count,
        "production_promotion_authorized": False,
    }
    print(json.dumps(summary, sort_keys=True))
    if args.smoke_one_pair:
        return 0 if result.mechanism_supported else 2
    return 0 if result.causal_supported else 2


if __name__ == "__main__":
    raise SystemExit(main())
