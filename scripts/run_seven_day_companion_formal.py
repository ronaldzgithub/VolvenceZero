#!/usr/bin/env python3
"""Execute the preregistered seven-day companion matrix on local models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from companion_bench.seven_day_driver import (
    FrozenSevenDayUserDriver,
    FrozenSevenDayUserScript,
    build_frozen_seven_day_user_script,
)
from companion_bench.spec import load_scenario_yaml
from companion_bench.user_simulator import (
    LocalTransformersUtteranceClient,
)
from huggingface_hub import snapshot_download
from lifeform_evolution.relationship_assistant_pilot import (
    RelationshipAssistantPilotHarness,
)
from lifeform_evolution.seven_day_companion import (
    HTTPSevenDayCompanionService,
    SevenDayCompanionOrchestrator,
    SevenDayScenarioSchedule,
    SevenDayScheduleDay,
    SimulatedSourceAttestation,
)
from lifeform_evolution.seven_day_process_host import (
    StateControlledSubprocessLifecycle,
    SubprocessSevenDayServiceHost,
)
from lifeform_evolution.seven_day_state_control import (
    SevenDayFilesystemStateController,
)
from volvence_zero.agent.seven_day_companion_evidence import (
    SevenDayCompanionAblationHarness,
    SevenDayExperimentCase,
)
from volvence_zero.agent.seven_day_companion_preregistration import (
    validate_seven_day_companion_preregistration,
)
from volvence_zero.substrate import fingerprint_model_weight_files


_DAY_MS = 86_400_000


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


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _workspace_pythonpath(repo_root: Path) -> str:
    paths = tuple(
        str(path)
        for path in sorted((repo_root / "packages").glob("*/src"))
        if path.is_dir()
    )
    inherited = os.environ.get("PYTHONPATH", "").strip()
    return os.pathsep.join((*paths, *((inherited,) if inherited else ())))


def _model_contract(
    preregistration: dict[str, object], *, role: str
) -> dict[str, object]:
    formal_models = preregistration.get("formal_models")
    if not isinstance(formal_models, dict):
        raise ValueError("preregistration lacks formal_models")
    contract = formal_models.get(role)
    if not isinstance(contract, dict):
        raise ValueError(f"preregistration lacks formal model {role}")
    return contract


def _verify_model(contract: dict[str, object]) -> Path:
    model_id = contract.get("model_id")
    expected_sha = contract.get("weights_sha256")
    if not isinstance(model_id, str) or not isinstance(expected_sha, str):
        raise ValueError("formal model contract is malformed")
    root = Path(snapshot_download(model_id, local_files_only=True))
    actual = fingerprint_model_weight_files(root)
    if actual != expected_sha:
        raise ValueError(
            f"formal model weights drift for {model_id}: {actual}"
        )
    return root


def _schedule(
    *,
    script: FrozenSevenDayUserScript,
    virtual_start_ms: int,
) -> SevenDayScenarioSchedule:
    tags_by_day: dict[int, set[str]] = {
        day_index: set() for day_index in range(1, 8)
    }
    for turn in script.turns:
        tags_by_day[turn.day_index].update(turn.event_tags)
    arc_type = (
        "progressive_warmth"
        if script.scenario_id.startswith("F1-")
        else "rupture_repair"
    )
    return SevenDayScenarioSchedule(
        scenario_id=script.scenario_id,
        persona_ref=(
            f"{script.identity_name}:{script.identity_occupation}"
        ),
        arc_type=arc_type,
        virtual_start_ms=virtual_start_ms,
        days=tuple(
            SevenDayScheduleDay(
                day_index=day_index,
                exchange_count=5,
                required_event_tags=tuple(sorted(tags_by_day[day_index])),
            )
            for day_index in range(1, 8)
        ),
    )


class _LocalFormalExecutor:
    def __init__(
        self,
        *,
        repo_root: Path,
        output_root: Path,
        cases: tuple[SevenDayExperimentCase, ...],
        scripts: dict[str, FrozenSevenDayUserScript],
        source_attestation: SimulatedSourceAttestation,
        sut_model_id: str,
        device: str,
        host: str,
        port: int,
        startup_timeout_s: float,
        virtual_start_ms: int,
    ) -> None:
        self._repo_root = repo_root
        self._output_root = output_root
        self._cases = cases
        self._scripts = scripts
        self._source_attestation = source_attestation
        self._sut_model_id = sut_model_id
        self._device = device
        self._host = host
        self._port = port
        self._startup_timeout_s = startup_timeout_s
        self._virtual_start_ms = virtual_start_ms
        self._case_index = {
            case.case_id: index for index, case in enumerate(cases)
        }

    def execute(
        self,
        *,
        case: SevenDayExperimentCase,
        arm_label: str,
        drain_slow_loop: bool,
        output_path: Path,
    ) -> dict[str, object]:
        if output_path.exists():
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"existing run is not an object: {output_path}")
            print(
                f"[resume] {case.case_id} / {arm_label}",
                flush=True,
            )
            return payload
        print(f"[run] {case.case_id} / {arm_label}", flush=True)
        case_key = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()[:20]
        state_root = self._output_root / "state" / case_key
        archive_root = state_root / "archives" / arm_label
        active_root = state_root / "active" / arm_label
        correct_reference = state_root / "archives/correct-user-state"
        donor_index = (self._case_index[case.case_id] + 1) % len(self._cases)
        donor = self._cases[donor_index]
        donor_key = hashlib.sha256(
            donor.case_id.encode("utf-8")
        ).hexdigest()[:20]
        donor_archive = (
            self._output_root
            / "state"
            / donor_key
            / "archives/correct-user-state"
        )
        # The physical state roots are arm-isolated, while the logical user
        # identity stays exact across arms.  Otherwise user-id drift would be
        # an unregistered second intervention in a paired comparison.
        user_id = f"synthetic-{case_key}"
        controller = SevenDayFilesystemStateController(
            evidence_root=self._output_root,
            active_scope_root=active_root,
            archive_root=archive_root,
            user_id=user_id,
            experiment_arm_label=arm_label,
            correct_reference_archive_root=(
                correct_reference if arm_label == "shuffled-history" else None
            ),
            donor_archive_root=(
                donor_archive if arm_label == "swapped-user-state" else None
            ),
        )
        base_url = f"http://{self._host}:{self._port}"
        service = HTTPSevenDayCompanionService(
            base_url=base_url,
            user_id=user_id,
            instance_id="not-started",
            vertical="companion",
            timeout_s=600.0,
        )
        service_evidence = (
            self._output_root / "service_evidence" / case_key / arm_label
        )
        command = (
            sys.executable,
            "-m",
            "lifeform_service.cli",
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--vertical",
            "companion",
            "--substrate-mode",
            "hf-shared",
            "--substrate-model-id",
            self._sut_model_id,
            "--substrate-device",
            self._device,
            "--substrate-local-files-only",
            "--alpha-enabled",
            "--memory-scope-root-dir",
            str(active_root),
            "--evidence-root-dir",
            str(service_evidence),
            "--allow-evidence-time-override",
            "--idle-eviction-seconds",
            "0",
            "--log-level",
            "INFO",
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = _workspace_pythonpath(self._repo_root)
        environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        environment["TRANSFORMERS_VERBOSITY"] = "error"
        host = SubprocessSevenDayServiceHost(
            command=command,
            service=service,
            health_url=f"{base_url}/v1/health",
            persistence_scope_id=user_id,
            log_dir=self._output_root / "service_logs" / case_key / arm_label,
            cwd=self._repo_root,
            environment=environment,
            startup_timeout_s=self._startup_timeout_s,
        )
        lifecycle = StateControlledSubprocessLifecycle(
            host=host,
            state_controller=controller,
        )
        pilot = RelationshipAssistantPilotHarness(
            root_dir=self._output_root / "pilot_days" / case_key / arm_label,
            pilot_id=f"seven-day-formal-{case_key}-{arm_label}",
            invited_user_ids=frozenset({user_id}),
        )
        lifecycle.start_initial()
        try:
            run = SevenDayCompanionOrchestrator(
                service=service,
                lifecycle=lifecycle,
                pilot_harness=pilot,
            ).run(
                run_id=f"formal-{case_key}-{arm_label}",
                arm_label=arm_label,
                paraphrase_seed=case.paraphrase_seed,
                user_id=user_id,
                schedule=_schedule(
                    script=self._scripts[case.case_id],
                    virtual_start_ms=self._virtual_start_ms,
                ),
                user_driver=FrozenSevenDayUserDriver(
                    self._scripts[case.case_id]
                ),
                source_attestation=self._source_attestation,
                drain_slow_loop=drain_slow_loop,
                output_path=output_path,
            )
        finally:
            lifecycle.close()
        print(f"[complete] {case.case_id} / {arm_label}", flush=True)
        return run.to_json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-one-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    selected_modes = sum(
        (args.preflight_only, args.smoke_one_run, args.execute)
    )
    if selected_modes != 1:
        raise ValueError(
            "select exactly one of --preflight-only, --smoke-one-run, "
            "or --execute"
        )
    if args.resume and not args.execute:
        raise ValueError("--resume is valid only with --execute")
    root = args.repo_root.resolve()
    preregistration = json.loads(
        args.preregistration.read_text(encoding="utf-8")
    )
    validate_seven_day_companion_preregistration(
        preregistration,
        repo_root=root,
    )
    sut = _model_contract(preregistration, role="sut")
    simulator = _model_contract(preregistration, role="simulator")
    _verify_model(sut)
    _verify_model(simulator)
    if sut["model_family"] == simulator["model_family"]:
        raise ValueError("formal SUT and simulator model families overlap")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "sut_model_id": sut["model_id"],
                    "simulator_model_id": simulator["model_id"],
                    "device": args.device,
                },
                sort_keys=True,
            )
        )
        return 0
    target = args.output_dir.resolve()
    if target.exists():
        if not (args.execute and args.resume):
            raise FileExistsError(f"formal output is immutable: {target}")
    else:
        target.mkdir(parents=True)
    synthetic_source_audit = {
        "schema_version": "seven-day-synthetic-source-audit.v1",
        "real_person_data": False,
        "source": "companion-bench typed scenario + local open-weight rendering",
        "consent_scope": "synthetic-no-human-subject",
        "semantic_event_tags_are_typed": True,
        "text_keyword_pii_inference_used": False,
    }
    source_audit_bytes = _canonical_bytes(synthetic_source_audit)
    source_audit_path = target / "synthetic_source_audit.json"
    if source_audit_path.exists():
        if source_audit_path.read_bytes() != source_audit_bytes:
            raise ValueError("existing synthetic source audit drift")
    else:
        source_audit_path.write_bytes(source_audit_bytes)
    simulator_backend = LocalTransformersUtteranceClient(
        model_id=str(simulator["model_id"]),
        device=args.device,
        local_files_only=True,
        max_new_tokens=int(simulator["max_new_tokens"]),
    )
    formal = preregistration["formal_run"]
    assert isinstance(formal, dict)
    seeds = formal["paraphrase_seeds"]
    scenario_ids = preregistration["scenario_ids"]
    assert isinstance(seeds, list) and isinstance(scenario_ids, list)
    all_cases = tuple(
        SevenDayExperimentCase(str(scenario_id), int(seed))
        for scenario_id in scenario_ids
        for seed in seeds
    )
    cases = all_cases[:1] if args.smoke_one_run else all_cases
    scenario_paths = preregistration["scenario_paths"]
    assert isinstance(scenario_paths, dict)
    scripts = {}
    for case in cases:
        spec = load_scenario_yaml(root / str(scenario_paths[case.scenario_id]))
        scripts[case.case_id] = build_frozen_seven_day_user_script(
            spec=spec,
            paraphrase_seed=case.paraphrase_seed,
            backend=simulator_backend,
        )
    model_fingerprint = _sha256(
        {
            "sut_model_id": sut["model_id"],
            "sut_weights_sha256": sut["weights_sha256"],
            "adapter": "none",
        }
    )
    source_attestation = SimulatedSourceAttestation(
        simulator_model_id=str(simulator["model_id"]),
        simulator_model_family=str(simulator["model_family"]),
        sut_model_id=str(sut["model_id"]),
        sut_model_family=str(sut["model_family"]),
        model_and_adapter_fingerprint=model_fingerprint,
        pii_scan_artifact_sha256=hashlib.sha256(source_audit_bytes).hexdigest(),
    )
    executor = _LocalFormalExecutor(
        repo_root=root,
        output_root=target,
        cases=cases,
        scripts=scripts,
        source_attestation=source_attestation,
        sut_model_id=str(sut["model_id"]),
        device=args.device,
        host=args.host,
        port=args.port,
        startup_timeout_s=args.startup_timeout_s,
        virtual_start_ms=int(formal["virtual_start_ms"]),
    )
    if args.smoke_one_run:
        smoke_case = cases[0]
        smoke_path = target / "runs" / "smoke-correct-user-state.json"
        payload = executor.execute(
            case=smoke_case,
            arm_label="correct-user-state",
            drain_slow_loop=True,
            output_path=smoke_path,
        )
        smoke_manifest = {
            "schema_version": "seven-day-companion-smoke.v1",
            "case_id": smoke_case.case_id,
            "arm_label": "correct-user-state",
            "run_sha256": _sha256(payload),
            "formal_claim_allowed": False,
            "purpose": "product-path preflight before formal preregistration",
        }
        (target / "smoke_manifest.json").write_bytes(
            _canonical_bytes(smoke_manifest)
        )
        print(json.dumps(smoke_manifest, sort_keys=True))
        return 0
    result = SevenDayCompanionAblationHarness(executor=executor).run(
        cases=cases,
        preregistration=preregistration,
        output_dir=target,
    )
    print(
        json.dumps(
            {
                "passed": result.passed,
                "case_count": result.case_count,
                "run_count": result.run_count,
            },
            sort_keys=True,
        )
    )
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
