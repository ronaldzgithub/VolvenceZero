#!/usr/bin/env python3
"""Execute the preregistered seven-day companion matrix on local models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping

from companion_bench.seven_day_driver import (
    FrozenSevenDayUserDriver,
    FrozenSevenDayUserScript,
    build_frozen_seven_day_user_script,
    load_frozen_seven_day_user_script,
)
from companion_bench.spec import FamilyId, ScenarioSpec, load_scenario_yaml
from companion_bench.user_simulator import (
    LocalTransformersUtteranceClient,
)
from huggingface_hub import snapshot_download
from lifeform_service.character_packages import load_character_runtime_assets
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
    seven_day_source_attestation_contract,
    validate_seven_day_companion_preregistration,
)
from volvence_zero.agent.seven_day_n_plus_one import (
    SevenDayNPlusOneCompiler,
    build_seven_day_n_plus_one_compiler,
    validate_seven_day_n_plus_one_evidence,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import fingerprint_model_weight_files

from companion_test_plan_common import (
    guarded_mps_runner_entrypoint,
    validate_seven_day_smoke_manifest,
    write_seven_day_smoke_manifest,
)


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
    paths = tuple(str(path) for path in sorted((repo_root / "packages").glob("*/src")) if path.is_dir())
    inherited = os.environ.get("PYTHONPATH", "").strip()
    return os.pathsep.join((*paths, *((inherited,) if inherited else ())))


def _model_contract(preregistration: dict[str, object], *, role: str) -> dict[str, object]:
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
        raise ValueError(f"formal model weights drift for {model_id}: {actual}")
    return root


def _resolve_character_stack(*, preregistration: Mapping[str, object], repo_root: Path) -> dict[str, object] | None:
    raw_stack = preregistration.get("runtime_stack")
    if raw_stack is None:
        return None
    if not isinstance(raw_stack, Mapping):
        raise ValueError("preregistration runtime_stack must be an object")
    raw_common = raw_stack.get("common_adapter")
    raw_manifests = raw_stack.get("character_manifests")
    if not isinstance(raw_common, Mapping) or not isinstance(raw_manifests, list):
        raise ValueError("preregistration character runtime stack is malformed")
    common_path = repo_root / str(raw_common["locator"])
    manifest_paths = tuple(repo_root / str(item["locator"]) for item in raw_manifests if isinstance(item, Mapping))
    if len(manifest_paths) != len(raw_manifests):
        raise ValueError("preregistration character manifest locator is malformed")
    assets = load_character_runtime_assets(
        common_adapter_bundle_path=common_path,
        manifest_paths=manifest_paths,
        wiring_by_character={},
        default_wiring=WiringLevel.ACTIVE,
    )
    selected_character_id = str(raw_stack["selected_character_id"])
    binding = assets.require_binding(selected_character_id)
    if binding.wiring_level is not WiringLevel.ACTIVE:
        raise ValueError("preregistered seven-day character binding is not ACTIVE")
    expected_manifest_ids = tuple(str(item["package_id"]) for item in raw_manifests if isinstance(item, Mapping))
    if assets.manifest_package_ids != expected_manifest_ids:
        raise ValueError("loaded seven-day character manifest ids drifted")
    selected_descriptor = next(
        item
        for item in raw_manifests
        if isinstance(item, Mapping) and item.get("character_id") == selected_character_id
    )
    prefix_entry = assets.prefix_registry.require(selected_character_id)
    if prefix_entry.prefix_package.package_id != selected_descriptor.get("prefix_package_id"):
        raise ValueError("loaded seven-day character Prefix/KV id drifted")
    return {
        "vertical": str(raw_stack["vertical"]),
        "character_id": selected_character_id,
        "common_adapter_bundle_path": common_path,
        "manifest_paths": manifest_paths,
        "manifest_package_ids": expected_manifest_ids,
    }


def _schedule(
    *,
    script: FrozenSevenDayUserScript,
    scenario_spec: ScenarioSpec,
    virtual_start_ms: int,
) -> SevenDayScenarioSchedule:
    tags_by_day: dict[int, set[str]] = {day_index: set() for day_index in range(1, 8)}
    for turn in script.turns:
        tags_by_day[turn.day_index].update(turn.event_tags)
    if scenario_spec.scenario_id != script.scenario_id:
        raise ValueError("seven-day scenario/script identity drift")
    arc_type_by_family = {
        FamilyId.F1_CONTINUITY: "progressive_warmth",
        FamilyId.F2_REPAIR: "rupture_repair",
    }
    try:
        arc_type = arc_type_by_family[scenario_spec.family]
    except KeyError as exc:
        raise ValueError(
            "seven-day scenario family lacks a registered typed arc"
        ) from exc
    return SevenDayScenarioSchedule(
        scenario_id=script.scenario_id,
        persona_ref=(f"{script.identity_name}:{script.identity_occupation}"),
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


def _write_atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(_canonical_bytes(payload))
    temporary.replace(path)


def _quarantine_path(*, output_root: Path, path: Path, reason: str) -> None:
    if not path.exists():
        return
    try:
        relative = path.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError("seven-day quarantine path escaped output root") from exc
    quarantine_root = (
        output_root
        / "quarantine"
        / f"{time.time_ns()}-{reason}"
    )
    destination = quarantine_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.replace(destination)


def _validate_resumable_run(
    *,
    payload: Mapping[str, object],
    case: SevenDayExperimentCase,
    arm_label: str,
    evidence_profile: str | None,
    require_character_stack: bool,
    n_plus_one_contract: Mapping[str, object] | None,
) -> None:
    if payload.get("schema_version") != "seven-day-companion-run.v1":
        raise ValueError("resumable seven-day run schema drift")
    if (
        payload.get("scenario_id") != case.scenario_id
        or payload.get("paraphrase_seed") != case.paraphrase_seed
        or payload.get("arm_label") != arm_label
    ):
        raise ValueError("resumable seven-day run identity drift")
    days = payload.get("days")
    if not isinstance(days, (list, tuple)) or len(days) != 7:
        raise ValueError("resumable seven-day run is incomplete")
    previous_next_instance: str | None = None
    for day_index, raw_day in enumerate(days, start=1):
        if not isinstance(raw_day, Mapping) or raw_day.get("day_index") != day_index:
            raise ValueError("resumable seven-day run day order drift")
        turns = raw_day.get("turns")
        if not isinstance(turns, (list, tuple)) or len(turns) != 5:
            raise ValueError("resumable seven-day run turn matrix is incomplete")
        for exchange_index, raw_turn in enumerate(turns, start=1):
            if (
                not isinstance(raw_turn, Mapping)
                or raw_turn.get("exchange_index") != exchange_index
                or not isinstance(raw_turn.get("user_text"), str)
                or not str(raw_turn.get("user_text")).strip()
                or not isinstance(raw_turn.get("assistant_text"), str)
                or not str(raw_turn.get("assistant_text")).strip()
            ):
                raise ValueError("resumable seven-day run turn contract drift")
            event_tags = raw_turn.get("event_tags")
            if not isinstance(event_tags, (list, tuple)) or not all(
                isinstance(tag, str) for tag in event_tags
            ):
                raise ValueError("resumable seven-day run event_tags drift")
        service_instance_id = raw_day.get("service_instance_id")
        if not isinstance(service_instance_id, str) or not service_instance_id:
            raise ValueError("resumable seven-day service identity is missing")
        if (
            previous_next_instance is not None
            and service_instance_id != previous_next_instance
        ):
            raise ValueError("resumable seven-day restart identity chain drift")
        restart = raw_day.get("restart_after_day")
        if day_index == 7:
            if restart is not None:
                raise ValueError("resumable day seven unexpectedly restarts")
            continue
        if not isinstance(restart, Mapping):
            raise ValueError("resumable seven-day restart evidence is missing")
        previous_scope = restart.get("previous_persistence_scope_sha256")
        next_scope = restart.get("next_persistence_scope_sha256")
        if (
            restart.get("after_day_index") != day_index
            or restart.get("previous_instance_id") != service_instance_id
            or restart.get("healthcheck_passed") is not True
            or restart.get("persistence_scope_unchanged") is not True
            or not isinstance(previous_scope, str)
            or len(previous_scope) != 64
            or any(character not in "0123456789abcdef" for character in previous_scope)
            or next_scope != previous_scope
            or not isinstance(restart.get("state_intervention"), Mapping)
        ):
            raise ValueError("resumable seven-day restart contract drift")
        previous_next_instance = restart.get("next_instance_id")
        if (
            not isinstance(previous_next_instance, str)
            or not previous_next_instance
            or previous_next_instance == service_instance_id
        ):
            raise ValueError("resumable seven-day restart identity drift")
    if (
        payload.get("process_restart_count") != 6
        or payload.get("all_restarts_exact") is not True
        or payload.get("production_promotion_authorized") is not False
    ):
        raise ValueError("resumable seven-day lifecycle evidence is incomplete")
    if evidence_profile is not None:
        profile = payload.get("runtime_profile_attestation")
        if not isinstance(profile, Mapping) or profile.get("profile") != evidence_profile:
            raise ValueError("resumable seven-day profile attestation is incomplete")
        claimed_sha = profile.get("attestation_sha256")
        unhashed = dict(profile)
        unhashed.pop("attestation_sha256", None)
        if (
            not isinstance(claimed_sha, str)
            or claimed_sha
            != hashlib.sha256(
                _canonical_bytes(unhashed).rstrip(b"\n")
            ).hexdigest()
        ):
            raise ValueError("resumable seven-day profile SHA drift")
    if require_character_stack and not isinstance(
        payload.get("runtime_stack_attestation"), Mapping
    ):
        raise ValueError("resumable v4 character stack attestation is incomplete")
    if n_plus_one_contract is not None:
        validate_seven_day_n_plus_one_evidence(
            run=payload,
            contract=n_plus_one_contract,
        )


class _LocalFormalExecutor:
    def __init__(
        self,
        *,
        repo_root: Path,
        output_root: Path,
        cases: tuple[SevenDayExperimentCase, ...],
        scripts: dict[str, FrozenSevenDayUserScript],
        scenario_paths: Mapping[str, str],
        source_attestation: SimulatedSourceAttestation,
        sut_model_id: str,
        sut_max_new_tokens: int,
        device: str,
        host: str,
        port: int,
        startup_timeout_s: float,
        virtual_start_ms: int,
        character_stack: Mapping[str, object] | None = None,
        evidence_profile_by_arm: Mapping[str, str] | None = None,
        state_loading_policy_by_arm: Mapping[str, str] | None = None,
        n_plus_one_compiler: SevenDayNPlusOneCompiler | None = None,
        n_plus_one_contract: Mapping[str, object] | None = None,
    ) -> None:
        self._repo_root = repo_root
        self._output_root = output_root
        self._cases = cases
        self._scripts = scripts
        self._scenario_paths = dict(scenario_paths)
        self._source_attestation = source_attestation
        self._sut_model_id = sut_model_id
        self._sut_max_new_tokens = sut_max_new_tokens
        self._device = device
        self._host = host
        self._port = port
        self._startup_timeout_s = startup_timeout_s
        self._virtual_start_ms = virtual_start_ms
        self._character_stack = dict(character_stack) if character_stack is not None else None
        self._evidence_profile_by_arm = dict(evidence_profile_by_arm or {})
        self._state_loading_policy_by_arm = dict(state_loading_policy_by_arm or {})
        self._n_plus_one_compiler = n_plus_one_compiler
        self._n_plus_one_contract = (
            dict(n_plus_one_contract)
            if n_plus_one_contract is not None
            else None
        )
        if (self._n_plus_one_compiler is None) != (
            self._n_plus_one_contract is None
        ):
            raise ValueError(
                "seven-day N+1 compiler and contract must be supplied together"
            )
        self._case_index = {case.case_id: index for index, case in enumerate(cases)}

    def execute(
        self,
        *,
        case: SevenDayExperimentCase,
        arm_label: str,
        drain_slow_loop: bool,
        output_path: Path,
    ) -> dict[str, object]:
        evidence_profile = self._evidence_profile_by_arm.get(arm_label)
        if output_path.exists():
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("existing run is not an object")
                _validate_resumable_run(
                    payload=payload,
                    case=case,
                    arm_label=arm_label,
                    evidence_profile=evidence_profile,
                    require_character_stack=self._character_stack is not None,
                    n_plus_one_contract=self._n_plus_one_contract,
                )
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                _quarantine_path(
                    output_root=self._output_root,
                    path=output_path,
                    reason="invalid-run",
                )
                print(
                    f"[resume-quarantine] {case.case_id} / {arm_label}: {exc}",
                    flush=True,
                )
            else:
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
        for stale_path in (archive_root, active_root):
            _quarantine_path(
                output_root=self._output_root,
                path=stale_path,
                reason="incomplete-state",
            )
        donor_archive = None
        if arm_label == "swapped-user-state":
            if len(self._cases) < 2:
                raise ValueError(
                    "swapped-user-state requires at least two matched cases"
                )
            donor_index = (self._case_index[case.case_id] + 1) % len(self._cases)
            donor = self._cases[donor_index]
            donor_key = hashlib.sha256(donor.case_id.encode("utf-8")).hexdigest()[:20]
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
            state_loading_policy=self._state_loading_policy_by_arm.get(arm_label),
            correct_reference_archive_root=(correct_reference if arm_label == "shuffled-history" else None),
            donor_archive_root=donor_archive,
        )
        base_url = f"http://{self._host}:{self._port}"
        vertical = str(self._character_stack["vertical"]) if self._character_stack is not None else "companion"
        character_id = str(self._character_stack["character_id"]) if self._character_stack is not None else None
        service = HTTPSevenDayCompanionService(
            base_url=base_url,
            user_id=user_id,
            instance_id="not-started",
            vertical=vertical,
            character_id=character_id,
            timeout_s=600.0,
        )
        service_evidence = self._output_root / "service_evidence" / case_key / arm_label
        service_log_dir = self._output_root / "service_logs" / case_key / arm_label
        pilot_root = self._output_root / "pilot_days" / case_key / arm_label
        for stale_path in (service_evidence, service_log_dir, pilot_root):
            _quarantine_path(
                output_root=self._output_root,
                path=stale_path,
                reason="incomplete-run",
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
            vertical,
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
            "--max-sessions",
            "1",
            "--idle-eviction-seconds",
            "0",
            "--log-level",
            "INFO",
        )
        if self._character_stack is not None:
            command = (
                *command,
                "--common-adapter-bundle",
                str(self._character_stack["common_adapter_bundle_path"]),
                "--character-package-mode",
                "active",
            )
            for manifest_path in self._character_stack["manifest_paths"]:
                command = (
                    *command,
                    "--character-package-manifest",
                    str(manifest_path),
                )
        if evidence_profile is not None:
            command = (
                *command,
                "--companion-evidence-profile",
                evidence_profile,
            )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = _workspace_pythonpath(self._repo_root)
        environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        environment["TRANSFORMERS_VERBOSITY"] = "error"
        environment["VZ_LIFEFORM_MAX_NEW_TOKENS"] = str(self._sut_max_new_tokens)
        host = SubprocessSevenDayServiceHost(
            command=command,
            service=service,
            health_url=f"{base_url}/v1/health",
            expected_persistence_scope_sha256=hashlib.sha256(
                active_root.resolve().as_posix().encode("utf-8")
            ).hexdigest(),
            log_dir=service_log_dir,
            cwd=self._repo_root,
            environment=environment,
            startup_timeout_s=self._startup_timeout_s,
        )
        lifecycle = StateControlledSubprocessLifecycle(
            host=host,
            state_controller=controller,
        )
        pilot = RelationshipAssistantPilotHarness(
            root_dir=pilot_root,
            pilot_id=f"seven-day-formal-{case_key}-{arm_label}",
            invited_user_ids=frozenset({user_id}),
        )
        lifecycle.start_initial()
        try:
            scenario_relative = self._scenario_paths.get(case.scenario_id)
            if not isinstance(scenario_relative, str):
                raise ValueError("seven-day scenario path mapping drift")
            scenario_spec = load_scenario_yaml(
                self._repo_root / scenario_relative
            )
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
                    scenario_spec=scenario_spec,
                    virtual_start_ms=self._virtual_start_ms,
                ),
                user_driver=FrozenSevenDayUserDriver(self._scripts[case.case_id]),
                source_attestation=self._source_attestation,
                drain_slow_loop=drain_slow_loop,
                output_path=None,
            )
        finally:
            lifecycle.close()
        print(f"[complete] {case.case_id} / {arm_label}", flush=True)
        payload = run.to_json()
        if evidence_profile is not None:
            profile_path = service_evidence / "companion_evidence_runtime_profile.json"
            profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
            if profile_payload.get("profile") != evidence_profile:
                raise ValueError("service evidence profile attestation drift")
            payload["runtime_profile_attestation"] = profile_payload
        if self._character_stack is not None:
            stack_path = service_evidence / "character_runtime_stack_attestation.json"
            stack_payload = json.loads(stack_path.read_text(encoding="utf-8"))
            if stack_payload.get("substrate_model_id") != self._sut_model_id:
                raise ValueError("service character runtime substrate drift")
            common = stack_payload.get("common_adapter")
            if not isinstance(common, Mapping) or common.get("bundle_id") != (
                self._source_attestation.common_adapter_bundle_id
            ):
                raise ValueError("service common adapter attestation drift")
            bindings = stack_payload.get("session_bindings")
            if not isinstance(bindings, list) or not any(
                isinstance(binding, Mapping)
                and binding.get("character_id") == self._source_attestation.character_id
                and binding.get("manifest_package_id") == self._source_attestation.character_manifest_package_id
                and binding.get("prefix_package_id") == self._source_attestation.character_prefix_package_id
                and binding.get("wiring_level") == "active"
                for binding in bindings
            ):
                raise ValueError("service ACTIVE character binding attestation drift")
            payload["runtime_stack_attestation"] = stack_payload
        if self._n_plus_one_compiler is not None:
            assert self._n_plus_one_contract is not None
            payload["n_plus_one_representation_evidence"] = (
                self._n_plus_one_compiler.compile(payload)
            )
        _validate_resumable_run(
            payload=payload,
            case=case,
            arm_label=arm_label,
            evidence_profile=evidence_profile,
            require_character_stack=self._character_stack is not None,
            n_plus_one_contract=self._n_plus_one_contract,
        )
        _write_atomic_json(output_path, payload)
        return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-one-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-evidence-root", type=Path)
    args = parser.parse_args()
    selected_modes = sum((args.preflight_only, args.smoke_one_run, args.execute))
    if selected_modes != 1:
        raise ValueError("select exactly one of --preflight-only, --smoke-one-run, or --execute")
    if args.resume and args.preflight_only:
        raise ValueError("--resume is invalid with --preflight-only")
    if not args.preflight_only and args.output_dir is None:
        raise ValueError("seven-day smoke/formal requires --output-dir")
    root = args.repo_root.resolve()
    preregistration = json.loads(args.preregistration.read_text(encoding="utf-8"))
    validate_seven_day_companion_preregistration(
        preregistration,
        repo_root=root,
    )
    if args.execute:
        if args.smoke_evidence_root is None:
            raise ValueError("formal execution requires --smoke-evidence-root")
        validate_seven_day_smoke_manifest(
            smoke_root=args.smoke_evidence_root.resolve(),
            preregistration=preregistration,
            campaign="continuity",
            gate_id=None,
        )
    character_stack = _resolve_character_stack(
        preregistration=preregistration,
        repo_root=root,
    )
    sut = _model_contract(preregistration, role="sut")
    simulator = _model_contract(preregistration, role="simulator")
    sut_snapshot = _verify_model(sut)
    _verify_model(simulator)
    if sut["model_family"] == simulator["model_family"]:
        raise ValueError("formal SUT and simulator model families overlap")
    formal = preregistration.get("formal_run")
    scenario_paths = preregistration.get("scenario_paths")
    if not isinstance(formal, dict) or not isinstance(scenario_paths, dict):
        raise ValueError("formal run metadata is malformed")
    seeds = formal.get("paraphrase_seeds")
    scenario_ids = preregistration.get("scenario_ids")
    if not isinstance(seeds, list) or not isinstance(scenario_ids, list):
        raise ValueError("formal case schedule is malformed")
    n_plus_one_contract = preregistration.get("n_plus_one_measurement")
    if not isinstance(n_plus_one_contract, Mapping):
        raise ValueError("formal preregistration lacks N+1 measurement")
    if args.device != formal.get("execution_device"):
        raise ValueError("execution device differs from preregistration")
    simulator_backend = LocalTransformersUtteranceClient(
        model_id=str(simulator["model_id"]),
        device=args.device,
        local_files_only=True,
        max_new_tokens=int(simulator["max_new_tokens"]),
    )
    if args.preflight_only:
        script_digests = {}
        for scenario_id in scenario_ids:
            for seed in seeds:
                spec = load_scenario_yaml(root / str(scenario_paths[str(scenario_id)]))
                script = build_frozen_seven_day_user_script(
                    spec=spec,
                    paraphrase_seed=int(seed),
                    backend=simulator_backend,
                    temperature=float(simulator["temperature"]),
                )
                script_digests[f"{scenario_id}:seed-{seed}"] = script.script_sha256
        build_seven_day_n_plus_one_compiler(
            model_source=sut_snapshot,
            contract=n_plus_one_contract,
        )
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "sut_model_id": sut["model_id"],
                    "simulator_model_id": simulator["model_id"],
                    "device": args.device,
                    "runtime_mode": (
                        "base+common-adapter+character-package" if character_stack is not None else "base-only"
                    ),
                    "validated_script_count": len(script_digests),
                    "n_plus_one_measurement": "load-and-contract-passed",
                    "script_sha256": script_digests,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.output_dir is None:
        raise RuntimeError("seven-day output directory validation drift")
    target = args.output_dir.resolve()
    if target.exists():
        if not args.resume:
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
    all_cases = tuple(
        SevenDayExperimentCase(str(scenario_id), int(seed)) for scenario_id in scenario_ids for seed in seeds
    )
    cases = all_cases[:1] if args.smoke_one_run else all_cases
    scripts = {}
    script_root = target / "user_scripts"
    script_root.mkdir(parents=True, exist_ok=True)
    for case in cases:
        script_path = script_root / (hashlib.sha256(case.case_id.encode("utf-8")).hexdigest() + ".json")
        if script_path.exists():
            script = load_frozen_seven_day_user_script(script_path)
            if script.scenario_id != case.scenario_id or script.paraphrase_seed != case.paraphrase_seed:
                raise ValueError(f"cached user script case drift: {script_path}")
            print(f"[script-resume] {case.case_id}", flush=True)
        else:
            print(f"[script] {case.case_id}", flush=True)
            spec = load_scenario_yaml(root / str(scenario_paths[case.scenario_id]))
            script = build_frozen_seven_day_user_script(
                spec=spec,
                paraphrase_seed=case.paraphrase_seed,
                backend=simulator_backend,
                temperature=float(simulator["temperature"]),
            )
            script_path.write_bytes(_canonical_bytes(script.to_json()))
            print(f"[script-complete] {case.case_id}", flush=True)
        scripts[case.case_id] = script
    attestation_contract = seven_day_source_attestation_contract(preregistration)
    source_attestation = SimulatedSourceAttestation(
        **attestation_contract,
        pii_scan_artifact_sha256=hashlib.sha256(source_audit_bytes).hexdigest(),
    )
    del simulator_backend
    n_plus_one_compiler = build_seven_day_n_plus_one_compiler(
        model_source=sut_snapshot,
        contract=n_plus_one_contract,
    )
    normalized_scenario_paths = {
        str(scenario_id): str(path)
        for scenario_id, path in scenario_paths.items()
    }
    executor = _LocalFormalExecutor(
        repo_root=root,
        output_root=target,
        cases=cases,
        scripts=scripts,
        scenario_paths=normalized_scenario_paths,
        source_attestation=source_attestation,
        sut_model_id=str(sut["model_id"]),
        sut_max_new_tokens=int(sut["max_new_tokens"]),
        device=args.device,
        host=args.host,
        port=args.port,
        startup_timeout_s=args.startup_timeout_s,
        virtual_start_ms=int(formal["virtual_start_ms"]),
        character_stack=character_stack,
        n_plus_one_compiler=n_plus_one_compiler,
        n_plus_one_contract=n_plus_one_contract,
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
        smoke_manifest = write_seven_day_smoke_manifest(
            output_root=target,
            preregistration=preregistration,
            campaign="continuity",
            gate_id=None,
            evidence_file=smoke_path.relative_to(target).as_posix(),
            evidence_sha256=hashlib.sha256(smoke_path.read_bytes()).hexdigest(),
            checks={
                "run-complete": len(payload.get("days", ())) == 7,
                "six-restarts-exact": (
                    payload.get("process_restart_count") == 6
                    and payload.get("all_restarts_exact") is True
                ),
                "n-plus-one-artifact-valid": True,
            },
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
    raise SystemExit(
        guarded_mps_runner_entrypoint(
            main,
            plan_id="seven-day-continuity-formal-runner",
            argv=sys.argv[1:],
        )
    )
