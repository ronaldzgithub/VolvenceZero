from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import pytest

from companion_bench.seven_day_driver import (
    FrozenSevenDayUserScript,
    FrozenSevenDayUserTurn,
)
from companion_bench.spec import load_scenario_yaml
from volvence_zero.agent.seven_day_companion_evidence import (
    SevenDayExperimentCase,
)
from volvence_zero.substrate import (
    SubstrateFingerprint,
    SubstrateForwardRepresentationLineage,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import companion_test_plan_common as common  # noqa: E402
import freeze_msc_execution_root as msc_execution_freezer  # noqa: E402
import freeze_seven_day_execution_root as execution_freezer  # noqa: E402
import msc_prediction_checkpoint as prediction_checkpoint  # noqa: E402
import run_msc_prediction_research as prediction_runner  # noqa: E402
import run_seven_day_companion_formal as continuity_runner  # noqa: E402
import run_seven_day_gate1_formal as gate1_runner  # noqa: E402
import run_seven_day_gate_suite_formal as gate_suite_runner  # noqa: E402
import run_msc_prediction_test_plan as prediction_plan  # noqa: E402
import run_seven_day_companion_test_plan as seven_day_plan  # noqa: E402


def _seven_day_preregistration(
    *, schema: str, arms: tuple[str, ...], gate_id: int | None = None
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": schema,
        "claim_scope": (
            "simulated-user-real-lifecycle-only"
            if schema in seven_day_plan.CONTINUITY_SCHEMAS
            else "simulated-seven-day-product-ecology-only"
        ),
        "scenario_ids": ["F1-01"],
        "formal_run": {
            "paraphrase_seeds": [1],
            "arm_schedule": list(arms),
            "run_count": len(arms),
            "execution_device": "mps",
        },
    }
    if gate_id is not None:
        payload["gate_id"] = gate_id
    return payload


def _write_canonical_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _prediction_lineage(
    root: Path,
    *,
    substrate_model: str = "frozen-model-v1",
) -> dict[str, object]:
    return {
        "msc_root": str((root / "msc").resolve()),
        "corpus_provenance_sha256": "a" * 64,
        "encoder": "fixture-encoder",
        "context_encoder_mode": "substrate",
        "volvence_context_mode": "full-runtime",
        "device": "mps",
        "substrate_model": substrate_model,
        "substrate_device": "mps",
        "substrate_activation_width": 896,
        "substrate_layer_indices": [11, 12, 13],
        "substrate_context_limit": 32768,
        "substrate_weights_sha256": "b" * 64,
        "runtime_max_new_tokens": 16,
        "runtime_startup_timeout": 600.0,
        "temporal_capacity_n_z": [3, 16, 64, 256],
        "temporal_capacity_fixed_forward_head_n_z": 3,
        "max_seq_length": 512,
        "encoder_batch_size": 32,
        "head_batch_size": 64,
        "learning_rate": 0.003,
        "retrieval_count": 4,
        "seeds": [0, 1, 2],
        "source_sha256": {"scripts/runner.py": "c" * 64},
    }


def _resumable_run(
    *, case: SevenDayExperimentCase, arm: str
) -> dict[str, object]:
    days = []
    for day_index in range(1, 8):
        restart = None
        if day_index < 7:
            restart = {
                "after_day_index": day_index,
                "previous_instance_id": f"instance-{day_index}",
                "next_instance_id": f"instance-{day_index + 1}",
                "healthcheck_passed": True,
                "persistence_scope_unchanged": True,
                "previous_persistence_scope_sha256": "a" * 64,
                "next_persistence_scope_sha256": "a" * 64,
                "state_intervention": {"after_day_index": day_index},
            }
        days.append(
            {
                "day_index": day_index,
                "service_instance_id": f"instance-{day_index}",
                "turns": [
                    {
                        "exchange_index": exchange_index,
                        "user_text": (
                            f"{case.case_id} day {day_index} turn {exchange_index}"
                        ),
                        "assistant_text": f"{arm} reply {day_index}-{exchange_index}",
                        "event_tags": [],
                    }
                    for exchange_index in range(1, 6)
                ],
                "restart_after_day": restart,
            }
        )
    return {
        "schema_version": "seven-day-companion-run.v1",
        "scenario_id": case.scenario_id,
        "paraphrase_seed": case.paraphrase_seed,
        "arm_label": arm,
        "days": days,
        "process_restart_count": 6,
        "all_restarts_exact": True,
        "production_promotion_authorized": False,
    }


def test_execution_environment_disables_mps_fallback(tmp_path: Path) -> None:
    (tmp_path / "packages/example/src").mkdir(parents=True)
    environment = common.execution_environment(tmp_path)

    assert environment["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert str(tmp_path / "packages/example/src") in environment["PYTHONPATH"]


def test_execution_freezer_materializes_exact_read_only_snapshot(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    sources = (
        repo / "packages/example/src/example.py",
        repo / "pyproject.toml",
        repo / "scripts/runner.py",
    )
    for index, path in enumerate(sources):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture-{index}\n", encoding="utf-8")
    roots = ("packages/*/src", "pyproject.toml", "scripts/runner.py")
    files = execution_freezer._collect_files(repo, roots)
    preregistration = tmp_path / "preregistration.json"
    _write_canonical_json(
        preregistration,
        {
            "schema_version": "fixture.v1",
            "execution_source_snapshot": {
                "roots": list(roots),
                "excluded": list(execution_freezer.EXCLUDED_PATTERNS),
                "file_count": len(files),
                "tree_sha256": execution_freezer._tree_sha256(repo, files),
            },
        },
    )
    frozen = tmp_path / "frozen"

    try:
        manifest = execution_freezer.freeze_execution_root(
            repo_root=repo,
            preregistration_path=preregistration,
            output_root=frozen,
        )

        assert manifest["file_count"] == 3
        assert (frozen / "packages/example/src/example.py").read_text(
            encoding="utf-8"
        ) == "fixture-0\n"
        assert (frozen.stat().st_mode & 0o222) == 0
        assert (
            frozen / "frozen_execution_root_manifest.json"
        ).is_file()
        with pytest.raises(FileExistsError, match="already exists"):
            execution_freezer.freeze_execution_root(
                repo_root=repo,
                preregistration_path=preregistration,
                output_root=frozen,
            )

        sources[0].write_text("drift\n", encoding="utf-8")
        with pytest.raises(ValueError, match="differs"):
            execution_freezer.freeze_execution_root(
                repo_root=repo,
                preregistration_path=preregistration,
                output_root=tmp_path / "drifted-frozen",
            )
    finally:
        if frozen.exists():
            frozen.chmod(0o755)
            for path in frozen.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)


def test_msc_execution_freezer_binds_formal_to_exact_read_only_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "scripts/run_msc_prediction_research.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('fixture')\n", encoding="utf-8")
    roots = ("scripts/run_msc_prediction_research.py",)
    monkeypatch.setattr(prediction_plan, "MSC_EXECUTION_SOURCE_ROOTS", roots)
    files = execution_freezer._collect_files(repo, roots)
    preregistration = tmp_path / "msc-preregistration.json"
    _write_canonical_json(
        preregistration,
        {
            "schema_version": "msc-n-plus-one-formal-prereg.v1",
            "execution_source_snapshot": {
                "roots": list(roots),
                "excluded": list(execution_freezer.EXCLUDED_PATTERNS),
                "file_count": len(files),
                "tree_sha256": execution_freezer._tree_sha256(repo, files),
            },
        },
    )
    frozen = tmp_path / "frozen-msc"

    try:
        manifest = msc_execution_freezer.freeze_msc_execution_root(
            repo_root=repo,
            preregistration_path=preregistration,
            output_root=frozen,
        )

        assert manifest["schema_version"] == "msc-frozen-execution-root.v1"
        validated = prediction_plan._validate_frozen_execution_root(
            execution_root=frozen,
            preregistration=preregistration,
        )
        assert validated == manifest
        assert (frozen.stat().st_mode & 0o222) == 0
    finally:
        if frozen.exists():
            frozen.chmod(0o755)
            for path in frozen.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)


def test_prediction_formal_rejects_mutable_execution_root(tmp_path: Path) -> None:
    root = tmp_path / "mutable"
    source = root / "scripts/run_msc_prediction_research.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('fixture')\n", encoding="utf-8")
    preregistration = tmp_path / "msc-preregistration.json"
    _write_canonical_json(
        preregistration,
        {
            "schema_version": "msc-n-plus-one-formal-prereg.v1",
            "execution_source_snapshot": {
                "roots": list(prediction_plan.MSC_EXECUTION_SOURCE_ROOTS),
                "excluded": list(execution_freezer.EXCLUDED_PATTERNS),
                "file_count": 1,
                "tree_sha256": "a" * 64,
            },
        },
    )

    with pytest.raises(FileNotFoundError, match="frozen execution manifest"):
        prediction_plan._validate_frozen_execution_root(
            execution_root=root,
            preregistration=preregistration,
        )


def test_require_mps_fails_loudly_when_backend_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = common.MPSAvailability(
        torch_version="fixture",
        built=True,
        available=False,
        fallback_disabled=True,
    )
    monkeypatch.setattr(common, "inspect_mps", lambda: unavailable)

    with pytest.raises(common.MPSUnavailableError, match="requires Apple MPS"):
        common.require_mps()


def test_require_mps_rejects_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = common.MPSAvailability(
        torch_version="fixture",
        built=True,
        available=True,
        fallback_disabled=True,
    )
    monkeypatch.setattr(
        common,
        "inspect_mps",
        lambda: replace(available, fallback_disabled=False),
    )

    with pytest.raises(common.MPSUnavailableError, match="cannot silently fall back"):
        common.require_mps()


def test_shared_mps_lock_rejects_a_second_plan(tmp_path: Path) -> None:
    lock = tmp_path / "mps.lock"

    with common.exclusive_mps_lock(lock, plan_id="seven-day"):
        with pytest.raises(common.MPSLockBusyError, match="seven-day"):
            with common.exclusive_mps_lock(lock, plan_id="msc-n-plus-one"):
                pytest.fail("a second MPS plan acquired the shared lock")


def test_direct_mps_runner_acquires_shared_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @contextmanager
    def fake_lock(_path: Path, *, plan_id: str):
        events.append(f"lock:{plan_id}")
        yield
        events.append("unlock")

    monkeypatch.delenv("VZ_COMPANION_MPS_LOCK_HELD", raising=False)
    monkeypatch.setenv(
        "VZ_COMPANION_MPS_LOCK_PATH", str(tmp_path / "shared.lock")
    )
    monkeypatch.setattr(common, "exclusive_mps_lock", fake_lock)
    monkeypatch.setattr(common, "require_mps", lambda: events.append("mps"))

    result = common.guarded_mps_runner_entrypoint(
        lambda: events.append("main") or 7,
        plan_id="direct-seven-day",
        argv=("--device", "mps", "--execute"),
    )

    assert result == 7
    assert events == ["lock:direct-seven-day", "mps", "main", "unlock"]


def test_mps_configuration_capture_skips_device_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        common,
        "exclusive_mps_lock",
        lambda *_args, **_kwargs: pytest.fail("configuration capture took MPS lock"),
    )
    monkeypatch.setattr(
        common,
        "require_mps",
        lambda: pytest.fail("configuration capture probed MPS"),
    )

    result = common.guarded_mps_runner_entrypoint(
        lambda: events.append("main") or 0,
        plan_id="msc-configuration-capture",
        argv=("--device", "mps", "--emit-run-configuration", "config.json"),
    )

    assert result == 0
    assert events == ["main"]


def test_prediction_runner_direct_entrypoint_uses_shared_mps_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def guard(main_callable, *, plan_id: str, argv: Sequence[str]) -> int:
        captured["main_callable"] = main_callable
        captured["plan_id"] = plan_id
        captured["argv"] = tuple(argv)
        return 7

    monkeypatch.setattr(
        prediction_runner,
        "guarded_mps_runner_entrypoint",
        guard,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_msc_prediction_research.py", "--device", "mps"],
    )

    assert prediction_runner._guarded_main() == 7
    assert captured == {
        "main_callable": prediction_runner.main,
        "plan_id": "msc-n-plus-one-prediction-mps.v1",
        "argv": ("--device", "mps"),
    }


def test_resume_validation_rejects_partial_and_v2_incomplete_runs(
    tmp_path: Path,
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    case = SevenDayExperimentCase("F2-resume", 7)
    run = _resumable_run(case=case, arm="correct-user-state")
    attach_n_plus_one(run, 0.8)
    continuity_runner._validate_resumable_run(
        payload=run,
        case=case,
        arm_label="correct-user-state",
        evidence_profile=None,
        require_character_stack=False,
        n_plus_one_contract=seven_day_n_plus_one_contract,
    )

    missing_n_plus_one = deepcopy(run)
    del missing_n_plus_one["n_plus_one_representation_evidence"]
    with pytest.raises(ValueError, match="n_plus_one_representation_evidence"):
        continuity_runner._validate_resumable_run(
            payload=missing_n_plus_one,
            case=case,
            arm_label="correct-user-state",
            evidence_profile=None,
            require_character_stack=False,
            n_plus_one_contract=seven_day_n_plus_one_contract,
        )

    with pytest.raises(ValueError, match="character stack attestation"):
        continuity_runner._validate_resumable_run(
            payload=run,
            case=case,
            arm_label="correct-user-state",
            evidence_profile=None,
            require_character_stack=True,
            n_plus_one_contract=seven_day_n_plus_one_contract,
        )
    run["runtime_stack_attestation"] = {"wiring_level": "active"}
    continuity_runner._validate_resumable_run(
        payload=run,
        case=case,
        arm_label="correct-user-state",
        evidence_profile=None,
        require_character_stack=True,
        n_plus_one_contract=seven_day_n_plus_one_contract,
    )

    invalid_path = tmp_path / "runs/incomplete.json"
    _write_canonical_json(invalid_path, missing_n_plus_one)
    continuity_runner._quarantine_path(
        output_root=tmp_path,
        path=invalid_path,
        reason="invalid-run",
    )
    assert not invalid_path.exists()
    quarantined = tuple(
        (tmp_path / "quarantine").glob("*-invalid-run/runs/incomplete.json")
    )
    assert len(quarantined) == 1


def test_resume_validation_accepts_fresh_tuple_payload(
    attach_n_plus_one,
    seven_day_n_plus_one_contract: dict[str, object],
) -> None:
    case = SevenDayExperimentCase("F1-fresh", 9)
    run = _resumable_run(case=case, arm="correct-user-state")
    attach_n_plus_one(run, 0.8)
    for day in run["days"]:
        day["turns"] = tuple(
            {**turn, "event_tags": tuple(turn["event_tags"])}
            for turn in day["turns"]
        )
    run["days"] = tuple(run["days"])

    continuity_runner._validate_resumable_run(
        payload=run,
        case=case,
        arm_label="correct-user-state",
        evidence_profile=None,
        require_character_stack=False,
        n_plus_one_contract=seven_day_n_plus_one_contract,
    )


def test_swapped_state_runner_rejects_self_donor_matrix(tmp_path: Path) -> None:
    case = SevenDayExperimentCase("F1-only", 1)
    executor = continuity_runner._LocalFormalExecutor(
        repo_root=REPO_ROOT,
        output_root=tmp_path,
        cases=(case,),
        scripts={},
        scenario_paths={},
        source_attestation=object(),
        sut_model_id="fixture",
        sut_max_new_tokens=1,
        device="mps",
        host="127.0.0.1",
        port=18765,
        startup_timeout_s=1.0,
        virtual_start_ms=1,
    )
    with pytest.raises(ValueError, match="at least two matched cases"):
        executor.execute(
            case=case,
            arm_label="swapped-user-state",
            drain_slow_loop=True,
            output_path=tmp_path / "runs/run.json",
        )


def test_schedule_uses_typed_yaml_family_not_scenario_prefix() -> None:
    source = load_scenario_yaml(
        REPO_ROOT
        / "packages/companion-bench/src/companion_bench/scenarios/seven_day/F2-seven-day-repair-researcher.yaml"
    )
    spec = replace(source, scenario_id="F1-misleading-prefix")
    turns = []
    for day_index in range(1, 8):
        for exchange_index in range(1, 6):
            tags: tuple[str, ...] = ()
            if day_index == 1 and exchange_index == 1:
                tags = ("emotion",)
            elif day_index == 4 and exchange_index == 1:
                tags = ("boundary",)
            elif day_index == 7 and exchange_index == 1:
                tags = ("callback",)
            turns.append(
                FrozenSevenDayUserTurn(
                    day_index=day_index,
                    exchange_index=exchange_index,
                    text="I am sharing a typed fixture.",
                    fsm_action=None,
                    fsm_payload=None,
                    event_tags=tags,
                )
            )
    script = FrozenSevenDayUserScript(
        schema_version="seven-day-user-script.v1",
        scenario_id=spec.scenario_id,
        paraphrase_seed=1,
        identity_name="Fixture",
        identity_occupation="tester",
        turns=tuple(turns),
        script_sha256="a" * 64,
    )

    schedule = continuity_runner._schedule(
        script=script,
        scenario_spec=spec,
        virtual_start_ms=1,
    )

    assert schedule.arc_type == "rupture_repair"


def test_seven_day_formal_command_is_mps_and_prereg_bound(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "scripts/run_seven_day_companion_formal.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("", encoding="utf-8")
    preregistration = tmp_path / "prereg.json"
    preregistration.write_text("{}", encoding="utf-8")
    output = tmp_path / "artifact"
    campaign = seven_day_plan._campaign_from_preregistration(
        _seven_day_preregistration(
            schema="seven-day-companion-simulated.v3",
            arms=("correct-user-state", "stateless"),
        )
    )

    command = seven_day_plan._runner_command(
        campaign=campaign,
        python=Path("/usr/bin/python3"),
        execution_root=tmp_path,
        preregistration=preregistration,
        stage="formal",
        output_dir=output,
        smoke_evidence_root=tmp_path / "artifact_smoke",
        host="127.0.0.1",
        port=18765,
        startup_timeout_s=600.0,
        resume=True,
    )

    assert command[0] == "/usr/bin/python3"
    assert "--device" in command
    assert command[command.index("--device") + 1] == "mps"
    assert command[command.index("--preregistration") + 1] == str(
        preregistration
    )
    assert "--execute" in command
    assert "--resume" in command


@pytest.mark.parametrize(
    ("runner_main", "argv"),
    (
        (
            continuity_runner.main,
            (
                "run_seven_day_companion_formal.py",
                "--preregistration",
                "missing.json",
                "--device",
                "mps",
                "--preflight-only",
            ),
        ),
        (
            gate1_runner.main,
            (
                "run_seven_day_gate1_formal.py",
                "--preregistration",
                "missing.json",
                "--device",
                "mps",
                "--preflight-only",
            ),
        ),
        (
            gate_suite_runner.main,
            (
                "run_seven_day_gate_suite_formal.py",
                "--gate",
                "4",
                "--preregistration",
                "missing.json",
                "--device",
                "mps",
                "--preflight-only",
            ),
        ),
    ),
)
def test_campaign_preflight_does_not_require_output_directory(
    runner_main: object,
    argv: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", list(argv))

    with pytest.raises(FileNotFoundError):
        runner_main()


def test_seven_day_status_keeps_simulated_claim_boundary(tmp_path: Path) -> None:
    preregistration = tmp_path / "prereg.json"
    payload = _seven_day_preregistration(
        schema="seven-day-companion-simulated.v1",
        arms=(
            "correct-user-state",
            "stateless",
            "swapped-user-state",
            "shuffled-history",
            "sleep-consolidation",
            "no-sleep",
        ),
    )
    _write_canonical_json(preregistration, payload)

    status = seven_day_plan._status(
        preregistration=preregistration,
        output_dir=None,
    )

    assert status["expected_runs"] == 6
    assert status["matrix_complete"] is False
    assert status["analysis_allowed"] is False
    assert status["production_promotion_authorized"] is False


@pytest.mark.parametrize(
    ("payload", "runner_name", "auditor_name", "smoke_flag", "gate_arg"),
    (
        (
            _seven_day_preregistration(
                schema="seven-day-companion-simulated.v3",
                arms=("correct-user-state", "stateless"),
            ),
            "run_seven_day_companion_formal.py",
            "audit_seven_day_companion_formal.py",
            "--smoke-one-run",
            None,
        ),
        (
            _seven_day_preregistration(
                schema="seven-day-companion-simulated.v4",
                arms=("correct-user-state", "stateless"),
            ),
            "run_seven_day_companion_formal.py",
            "audit_seven_day_companion_formal.py",
            "--smoke-one-run",
            None,
        ),
        (
            _seven_day_preregistration(
                schema=seven_day_plan.GATE1_SCHEMA,
                arms=("gate1-pe-temporal-on-v1", "gate1-pe-temporal-off-v1"),
            ),
            "run_seven_day_gate1_formal.py",
            "audit_seven_day_gate1_formal.py",
            "--smoke-one-pair",
            None,
        ),
        (
            _seven_day_preregistration(
                schema=seven_day_plan.GATE_SUITE_SCHEMA,
                arms=("gate7-ssl-rl-full-v1", "gate7-no-ssl-v1", "gate7-no-rl-v1"),
                gate_id=7,
            ),
            "run_seven_day_gate_suite_formal.py",
            "audit_seven_day_gate_suite_formal.py",
            "--smoke-one-pair",
            "7",
        ),
    ),
)
def test_seven_day_schema_dispatches_exact_runner_and_auditor(
    tmp_path: Path,
    payload: dict[str, object],
    runner_name: str,
    auditor_name: str,
    smoke_flag: str,
    gate_arg: str | None,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / runner_name).write_text("", encoding="utf-8")
    (scripts / auditor_name).write_text("", encoding="utf-8")
    campaign = seven_day_plan._campaign_from_preregistration(payload)
    preregistration = tmp_path / "prereg.json"
    output = tmp_path / "artifact"
    audit_output = output / "audit"

    command = seven_day_plan._runner_command(
        campaign=campaign,
        python=Path("/usr/bin/python3"),
        execution_root=tmp_path,
        preregistration=preregistration,
        stage="smoke",
        output_dir=output,
        smoke_evidence_root=None,
        host="127.0.0.1",
        port=campaign.default_port,
        startup_timeout_s=600.0,
        resume=False,
    )
    audit = seven_day_plan._audit_command(
        campaign=campaign,
        python=Path("/usr/bin/python3"),
        execution_root=tmp_path,
        preregistration=preregistration,
        output_dir=output,
        audit_output_dir=audit_output,
    )

    assert Path(command[1]).name == runner_name
    assert smoke_flag in command
    assert Path(audit[1]).name == auditor_name
    if gate_arg is None:
        assert "--gate" not in command
        assert "--gate" not in audit
    else:
        assert command[command.index("--gate") + 1] == gate_arg
        assert audit[audit.index("--gate") + 1] == gate_arg


def test_seven_day_status_opens_analysis_only_after_exact_independent_audit(
    tmp_path: Path,
) -> None:
    preregistration = tmp_path / "prereg.json"
    payload = _seven_day_preregistration(
        schema=seven_day_plan.GATE1_SCHEMA,
        arms=("gate1-pe-temporal-on-v1", "gate1-pe-temporal-off-v1"),
    )
    _write_canonical_json(preregistration, payload)
    output = tmp_path / "artifact"
    expected = seven_day_plan._expected_run_identities(payload)
    for name, (scenario, seed, arm) in expected.items():
        _write_canonical_json(
            output / "runs" / name,
            {
                "schema_version": "seven-day-companion-run.v1",
                "scenario_id": scenario,
                "paraphrase_seed": seed,
                "arm_label": arm,
            },
        )
    _write_canonical_json(output / "gate1_evaluation.json", {"complete": True})
    evaluation_sha256 = hashlib.sha256(
        (output / "gate1_evaluation.json").read_bytes()
    ).hexdigest()

    before_audit = seven_day_plan._status(
        preregistration=preregistration, output_dir=output
    )
    assert before_audit["matrix_complete"] is True
    assert before_audit["analysis_allowed"] is False

    _write_canonical_json(
        output / "audit/gate1_independent_audit.json",
        {
            "schema_version": "gate1-seven-day-independent-audit.v1",
            "audit_passed": True,
            "preregistration_sha256": hashlib.sha256(
                preregistration.read_bytes()
            ).hexdigest(),
            "gate1_evaluation_sha256": evaluation_sha256,
            "counts": {"runs": len(expected)},
            "claim_scope": payload["claim_scope"],
        },
    )
    after_audit = seven_day_plan._status(
        preregistration=preregistration, output_dir=output
    )
    assert after_audit["independent_audit_valid"] is True
    assert after_audit["analysis_allowed"] is True

    _write_canonical_json(output / "gate1_evaluation.json", {"mutated": True})
    with_mutated_evaluation = seven_day_plan._status(
        preregistration=preregistration, output_dir=output
    )
    assert with_mutated_evaluation["independent_audit_valid"] is False
    assert with_mutated_evaluation["analysis_allowed"] is False
    _write_canonical_json(output / "gate1_evaluation.json", {"complete": True})

    _write_canonical_json(output / "runs/unregistered.json", {"extra": True})
    with_extra = seven_day_plan._status(
        preregistration=preregistration, output_dir=output
    )
    assert with_extra["unexpected_run_files"] == 1
    assert with_extra["analysis_allowed"] is False


def test_seven_day_halt_record_blocks_analysis_and_resume(
    tmp_path: Path,
) -> None:
    preregistration = tmp_path / "prereg.json"
    payload = _seven_day_preregistration(
        schema=seven_day_plan.GATE1_SCHEMA,
        arms=("on", "off"),
    )
    _write_canonical_json(preregistration, payload)
    output = tmp_path / "artifact"
    expected = seven_day_plan._expected_run_identities(payload)
    name, (scenario, seed, arm) = next(iter(expected.items()))
    _write_canonical_json(
        output / "runs" / name,
        {
            "schema_version": "seven-day-companion-run.v1",
            "scenario_id": scenario,
            "paraphrase_seed": seed,
            "arm_label": arm,
        },
    )
    _write_canonical_json(
        output / "halt_record.json",
        {
            "schema_version": "seven-day-companion-halt-record.v1",
            "halted_at_unix_ms": 1,
            "halt_class": "instrument-discrimination",
            "halted_preregistration": {
                "sha256": hashlib.sha256(preregistration.read_bytes()).hexdigest()
            },
            "preserved_state": {
                "complete_run_envelopes_preserved": 1,
                "expected_run_count": len(expected),
            },
            "discipline_attestation": {
                "effect_claim_allowed": False,
                "production_promotion_authorized": False,
            },
            "resumption_policy": {"resume_as_is_authorized": False},
        },
    )

    status = seven_day_plan._status(
        preregistration=preregistration,
        output_dir=output,
    )
    assert status["run_state"] == "halted"
    assert status["halt_class"] == "instrument-discrimination"
    assert status["analysis_allowed"] is False

    with pytest.raises(RuntimeError, match="cannot be resumed as-is"):
        seven_day_plan.main(
            (
                "formal",
                "--preregistration",
                str(preregistration),
                "--output-dir",
                str(output),
                "--resume",
            )
        )


def test_seven_day_mps_control_rejects_cuda_preregistration() -> None:
    payload = _seven_day_preregistration(
        schema=seven_day_plan.GATE1_SCHEMA,
        arms=("on", "off"),
    )
    payload["formal_run"]["execution_device"] = "cuda"

    with pytest.raises(ValueError, match="execution_device='mps'"):
        seven_day_plan._require_mps_preregistration(payload)


def test_seven_day_all_audits_a_complete_negative_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in (
        "run_seven_day_gate1_formal.py",
        "audit_seven_day_gate1_formal.py",
    ):
        (scripts / name).write_text("", encoding="utf-8")
    (tmp_path / "packages/example/src").mkdir(parents=True)
    preregistration = tmp_path / "prereg.json"
    _write_canonical_json(
        preregistration,
        _seven_day_preregistration(
            schema=seven_day_plan.GATE1_SCHEMA,
            arms=("gate1-pe-temporal-on-v1", "gate1-pe-temporal-off-v1"),
        ),
    )
    output = tmp_path / "artifact"
    output.mkdir()
    commands: list[tuple[str, ...]] = []
    return_codes = iter(
        (0, 0, seven_day_plan.SCIENTIFIC_NEGATIVE_EXIT, 0)
    )

    def fake_run(argv: Sequence[str], **_: object) -> int:
        commands.append(tuple(argv))
        return next(return_codes)

    monkeypatch.setattr(seven_day_plan, "run_plan_command", fake_run)
    monkeypatch.setattr(
        seven_day_plan,
        "require_mps",
        lambda: common.MPSAvailability("fixture", True, True, True),
    )
    monkeypatch.setattr(
        seven_day_plan,
        "exclusive_mps_lock",
        lambda *_args, **_kwargs: nullcontext(),
    )

    exit_code = seven_day_plan.main(
        (
            "all",
            "--execution-root",
            str(tmp_path),
            "--preregistration",
            str(preregistration),
            "--output-dir",
            str(output),
        )
    )

    assert exit_code == seven_day_plan.SCIENTIFIC_NEGATIVE_EXIT
    assert len(commands) == 4
    assert "--preflight-only" in commands[0]
    assert "--smoke-one-pair" in commands[1]
    assert "--execute" in commands[2]
    assert "--smoke-evidence-root" in commands[2]
    assert Path(commands[3][1]).name == "audit_seven_day_gate1_formal.py"


def test_prediction_smoke_uses_mps_for_encoder_head_and_substrate(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "scripts/run_msc_prediction_research.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("", encoding="utf-8")

    command = prediction_plan._smoke_command(
        python=Path("/usr/bin/python3"),
        execution_root=tmp_path,
        msc_root=tmp_path / "msc",
        output_dir=tmp_path / "artifact",
        substrate_model="Qwen/Qwen2.5-0.5B-Instruct",
        resume=True,
    )

    assert command[command.index("--device") + 1] == "mps"
    assert command[command.index("--substrate-device") + 1] == "mps"
    assert command[command.index("--substrate-layer-indices") + 1 :][0:3] == (
        "11",
        "12",
        "13",
    )
    assert "--resume" in command


def test_prediction_plan_propagates_parent_mps_lock_to_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_root = tmp_path / "execution"
    (execution_root / "packages/example/src").mkdir(parents=True)
    runner = execution_root / "scripts/run_msc_prediction_research.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("", encoding="utf-8")
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        prediction_plan,
        "exclusive_mps_lock",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        prediction_plan,
        "require_mps",
        lambda: common.MPSAvailability(
            torch_version="fixture",
            built=True,
            available=True,
            fallback_disabled=True,
        ),
    )

    def run_command(
        _command: Sequence[str],
        *,
        execution_root: Path,
        environment: dict[str, str],
    ) -> int:
        del execution_root
        captured.update(environment)
        return 0

    monkeypatch.setattr(prediction_plan, "run_plan_command", run_command)
    monkeypatch.setattr(
        prediction_plan,
        "_write_smoke_manifest",
        lambda _output: {"passed": True},
    )
    lock = tmp_path / "shared.lock"

    result = prediction_plan.main(
        [
            "smoke",
            "--execution-root",
            str(execution_root),
            "--msc-root",
            str(tmp_path / "msc"),
            "--output-dir",
            str(tmp_path / "output"),
            "--mps-lock",
            str(lock),
        ]
    )

    assert result == 0
    assert captured["VZ_COMPANION_MPS_LOCK_HELD"] == "1"
    assert captured["VZ_COMPANION_MPS_LOCK_PATH"] == str(lock.resolve())


def test_prediction_status_reports_r5_formal_infrastructure_unblocked() -> None:
    payload = prediction_plan._prediction_status()

    assert payload["formal_eligible"] is True
    assert payload["formal_blockers"] == ()
    assert "temporal-controller-capacity-ladder" in payload[
        "completed_prerequisites"
    ]
    assert payload["formal_claim_permitted_now"] is False


def test_prediction_formal_command_freezes_full_matrix_and_preregistration(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "scripts/run_msc_prediction_research.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("", encoding="utf-8")
    command = prediction_plan._formal_command(
        python=Path("/usr/bin/python3"),
        execution_root=tmp_path,
        msc_root=tmp_path / "msc",
        output_dir=tmp_path / "formal",
        substrate_model="Qwen/Qwen2.5-0.5B-Instruct",
        preregistration=tmp_path / "prereg.json",
        resume=False,
    )

    assert command[command.index("--train-dyads") + 1] == "1001"
    assert command[command.index("--validation-dyads") + 1] == "500"
    assert command[command.index("--heldout-dyads") + 1] == "501"
    assert command[command.index("--preregistration") + 1].endswith(
        "prereg.json"
    )


def test_prediction_preregistration_recaptures_and_freezes_run_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "scripts/run_msc_prediction_research.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("", encoding="utf-8")
    smoke_root = tmp_path / "smoke"
    smoke_path = smoke_root / "smoke_manifest.json"
    smoke_configuration = {
        "schema_version": "msc-n-plus-one-resumable-run.v4",
        **_prediction_lineage(tmp_path),
    }
    _write_canonical_json(
        smoke_path,
        {
            "schema_version": "msc-r5-smoke-manifest.v1",
            "passed": True,
            "formal_claim_permitted": False,
            "measurement": {"runtime_sample_count": 4},
            "runner_run_configuration": smoke_configuration,
        },
    )

    def capture_configuration(
        command: Sequence[str],
        *,
        execution_root: Path,
        environment: dict[str, str],
    ) -> int:
        del execution_root, environment
        target = Path(command[command.index("--emit-run-configuration") + 1])
        substrate_model = command[command.index("--substrate-model") + 1]
        _write_canonical_json(
            target,
            {
                **smoke_configuration,
                **_prediction_lineage(
                    tmp_path,
                    substrate_model=substrate_model,
                ),
            },
        )
        return 0

    monkeypatch.setattr(
        prediction_plan,
        "run_plan_command",
        capture_configuration,
    )
    preregistration = tmp_path / "formal-prereg.json"
    kwargs = {
        "python": Path("/usr/bin/python3"),
        "execution_root": tmp_path,
        "environment": {},
        "msc_root": tmp_path / "msc",
        "formal_output_dir": tmp_path / "formal",
        "smoke_output_dir": smoke_root,
        "substrate_model": "frozen-model-v1",
        "preregistration": preregistration,
    }

    first = prediction_plan._write_formal_preregistration(**kwargs)
    second = prediction_plan._write_formal_preregistration(**kwargs)

    assert second == first
    assert first["run_configuration"]["substrate_model"] == "frozen-model-v1"
    with pytest.raises(ValueError, match="lineage drift"):
        prediction_plan._write_formal_preregistration(
            **(kwargs | {"substrate_model": "drifted-model-v2"})
        )


def test_prediction_preregistration_rejects_smoke_source_lineage_drift(
    tmp_path: Path,
) -> None:
    smoke_configuration = _prediction_lineage(tmp_path)
    formal_configuration = deepcopy(smoke_configuration)
    formal_configuration["source_sha256"] = {"scripts/runner.py": "d" * 64}

    with pytest.raises(ValueError, match="source_sha256"):
        prediction_plan._validate_smoke_formal_lineage(
            smoke={"runner_run_configuration": smoke_configuration},
            formal_configuration=formal_configuration,
        )


def test_prediction_smoke_manifest_requires_r3_r4_r5_execution_order(
    tmp_path: Path,
) -> None:
    def write_smoke_inputs(root: Path, order: list[str]) -> None:
        _write_canonical_json(
            root / "artifact_manifest.json",
            {
                "evidence_level": "pilot",
                "formal_experiment_executed": False,
            },
        )
        runtime = {
            str(n_z): {
                "total_interval_latency_ms": 1.0,
                "total_interval_token_count": 2,
                "sample_count": 1,
            }
            for n_z in (3, 16, 64, 256)
        }
        _write_canonical_json(
            root / "manifest.json",
            {
                "same_substrate_context_attestation": {
                    "passed": True,
                    "truncated_token_count": 0,
                },
                "full_runtime_context_attestation": {
                    "volvence_full_stack": True,
                    "raw_text_retained": False,
                    "evaluation_writeback_allowed": False,
                },
                "temporal_capacity_runtime_attestations": runtime,
                "convergence_stage_order": order,
                "run_configuration": {"fixture": True},
            },
        )
        _write_canonical_json(
            root / "temporal_capacity_ladder.json",
            {
                "observations": [
                    {
                        "temporal_n_z": n_z,
                        "forward_head_n_z": 3,
                    }
                    for n_z in (3, 16, 64, 256)
                ],
                "zero_norm_prediction_count": 0,
            },
        )
        _write_canonical_json(
            root / "prediction_verdict.json",
            {"evidence_level": "pilot"},
        )

    passed = tmp_path / "passed"
    write_smoke_inputs(passed, ["R3", "R4", "R5"])
    assert prediction_plan._write_smoke_manifest(passed)["passed"] is True

    reordered = tmp_path / "reordered"
    write_smoke_inputs(reordered, ["R4", "R3", "R5"])
    with pytest.raises(ValueError, match="smoke checks"):
        prediction_plan._write_smoke_manifest(reordered)


def test_prediction_r4_lineage_must_match_r3_target_surface() -> None:
    lineage = SubstrateForwardRepresentationLineage(
        schema_version="substrate-forward-representation.v1",
        snapshot_fingerprint="b" * 64,
        model_fingerprint=SubstrateFingerprint(
            model_id="frozen-model",
            version="snapshot-v1",
            weights_sha256="a" * 64,
        ),
        runtime_origin="hf-local",
        readout_kind="latest-token-selected-layer-residual-l2.v1",
        layer_indices=(11, 12),
        activation_widths=(2, 2),
        representation_dim=4,
    )
    attestation = {
        "volvence_full_stack": True,
        "model_id": "frozen-model",
        "weights_sha256": "a" * 64,
        "runtime_origin": "hf-local",
        "readout_kind": "latest-token-selected-layer-residual-l2.v1",
        "layer_indices": [11, 12],
        "activation_widths": [2, 2],
        "temporal_n_z": 3,
        "raw_text_retained": False,
        "evaluation_writeback_allowed": False,
    }

    prediction_runner._validate_full_runtime_target_lineage(
        attestation,
        target_lineage=lineage,
        expected_temporal_n_z=3,
    )
    with pytest.raises(ValueError, match="target substrate lineage differ"):
        prediction_runner._validate_full_runtime_target_lineage(
            attestation | {"temporal_n_z": 16},
            target_lineage=lineage,
            expected_temporal_n_z=3,
        )


def test_prediction_status_reports_progress_without_exposing_results(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    store = prediction_checkpoint.PredictionRunCheckpointStore(
        output_dir=output,
        configuration={"epochs": 2},
        resume=False,
    )
    store.save_json(
        unit="corpus/index",
        relative_path="checkpoints/corpus/index.json",
        payload={"example_fingerprint": "b" * 64},
    )

    status = prediction_plan._prediction_status(output_dir=output)
    progress = status["run_progress"]

    assert isinstance(progress, dict)
    assert progress["status"] == "running"
    assert progress["completed_unit_count"] == 1
    assert progress["last_completed_unit"] == "corpus/index"
    assert progress["analysis_allowed"] is False
    assert progress["formal_claim_allowed"] is False


def test_prediction_status_rejects_checkpoint_that_claims_raw_text(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    prediction_checkpoint.PredictionRunCheckpointStore(
        output_dir=output,
        configuration={"epochs": 2},
        resume=False,
    )
    state_path = output / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["raw_corpus_text_retained"] = True
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="must not retain raw corpus text"):
        prediction_plan._prediction_status(output_dir=output)


def test_prediction_preflight_report_is_immutable(tmp_path: Path) -> None:
    target = tmp_path / "preflight.json"
    prediction_plan._write_immutable_json(target, {"ready": True})
    prediction_plan._write_immutable_json(target, {"ready": True})

    with pytest.raises(ValueError, match="report differs"):
        prediction_plan._write_immutable_json(target, {"ready": False})


def test_prediction_checkpoint_store_resumes_exact_units(tmp_path: Path) -> None:
    import numpy as np

    output = tmp_path / "run"
    configuration = {"seeds": (0, 1, 2), "source_sha256": {"runner": "a" * 64}}
    store = prediction_checkpoint.PredictionRunCheckpointStore(
        output_dir=output,
        configuration=configuration,
        resume=False,
    )
    store.save_json(
        unit="corpus/index",
        relative_path="checkpoints/corpus/index.json",
        payload={"example_fingerprint": "b" * 64},
    )
    metadata = {"source_sha256": ("c" * 64,), "raw_text_retained": False}
    store.save_arrays(
        unit="targets/substrate",
        relative_path="checkpoints/targets/substrate.npz",
        metadata=metadata,
        arrays={"vectors": np.asarray(((0.1, 0.2),), dtype=np.float64)},
    )

    resumed = prediction_checkpoint.PredictionRunCheckpointStore(
        output_dir=output,
        configuration=configuration,
        resume=True,
    )

    assert resumed.load_json(
        unit="corpus/index",
        relative_path="checkpoints/corpus/index.json",
    ) == {"example_fingerprint": "b" * 64}
    arrays = resumed.load_arrays(
        unit="targets/substrate",
        relative_path="checkpoints/targets/substrate.npz",
        expected_metadata=metadata,
    )
    assert arrays is not None
    assert arrays["vectors"].tolist() == [[0.1, 0.2]]
    assert set(resumed.immutable_file_manifest()) == {
        "checkpoints/corpus/index.json",
        "checkpoints/targets/substrate.npz",
    }


def test_full_runtime_context_checkpoint_round_trips_without_raw_text(
    tmp_path: Path,
) -> None:
    import struct

    from companion_bench.msc_corpus import MSCDyad, MSCSession, MSCUtterance
    from companion_bench.msc_runtime_collection import (
        MSCFullRuntimeCollectedSample,
    )
    from companion_bench.prediction_research import (
        MSCFullRuntimeContext,
        build_msc_next_turn_examples,
    )
    from volvence_zero.substrate import SubstrateFingerprint

    dyad = MSCDyad(
        dyad_id="checkpoint-dyad",
        split="heldout",
        sessions=(
            MSCSession(
                session_index=1,
                utterances=(
                    MSCUtterance("speaker_1", "target zero", 1),
                    MSCUtterance("speaker_2", "private partner text", 2),
                    MSCUtterance("speaker_1", "target one", 3),
                ),
            ),
        ),
        initial_personas=(("private persona",), ()),
    )
    examples = build_msc_next_turn_examples((dyad,))
    values = (0.6, 0.8)
    values_sha = hashlib.sha256(struct.pack("!2d", *values)).hexdigest()
    context = MSCFullRuntimeContext(
        sample_id=examples[0].sample_id,
        active_speaker_id="speaker_2",
        values=values,
        values_sha256=values_sha,
        source_sha256="b" * 64,
        model_id="frozen-model",
        model_version="snapshot-v1",
        weights_sha256="a" * 64,
        runtime_origin="hf-local",
        readout_kind="latest-token-selected-layer-residual-l2.v1",
        layer_indices=(11, 12),
        activation_widths=(1, 1),
        temporal_n_z=3,
        input_token_count=4,
        output_token_count=1,
        total_token_count=5,
        generation_latency_ms=1.0,
        latency_ms=2.0,
        propagate_event_count=12,
        runtime_slot_surface_sha256="c" * 64,
    )
    samples = (
        MSCFullRuntimeCollectedSample(
            context=context,
            interval_input_token_count=8,
            interval_output_token_count=2,
            interval_total_token_count=10,
            interval_latency_ms=4.0,
            observation_turn_count=2,
            scene_boundary_count=0,
        ),
    )
    store = prediction_checkpoint.PredictionRunCheckpointStore(
        output_dir=tmp_path / "run",
        configuration={"context": "full-runtime"},
        resume=False,
    )
    fingerprint = SubstrateFingerprint(
        model_id="frozen-model",
        version="snapshot-v1",
        weights_sha256="a" * 64,
    )

    prediction_runner._save_full_runtime_dyad(
        store=store,
        dyad=dyad,
        examples=examples,
        samples=samples,
        model_fingerprint=fingerprint,
        layer_indices=(11, 12),
        activation_width=1,
        context_limit=1024,
        max_new_tokens=16,
        temporal_n_z=3,
    )
    loaded = prediction_runner._load_full_runtime_dyad(
        store=store,
        dyad=dyad,
        examples=examples,
        model_fingerprint=fingerprint,
        layer_indices=(11, 12),
        activation_width=1,
        context_limit=1024,
        max_new_tokens=16,
        temporal_n_z=3,
    )

    assert loaded == samples
    checkpoint = next(
        (tmp_path / "run/checkpoints/contexts/volvence-runtime").rglob("*.npz")
    )
    assert b"private partner text" not in checkpoint.read_bytes()
    assert b"private persona" not in checkpoint.read_bytes()


def test_prediction_checkpoint_resume_rejects_configuration_drift(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    prediction_checkpoint.PredictionRunCheckpointStore(
        output_dir=output,
        configuration={"epochs": 2},
        resume=False,
    )

    with pytest.raises(ValueError, match="configuration drift"):
        prediction_checkpoint.PredictionRunCheckpointStore(
            output_dir=output,
            configuration={"epochs": 3},
            resume=True,
        )


def test_prediction_checkpoint_resume_rejects_file_tampering(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    store = prediction_checkpoint.PredictionRunCheckpointStore(
        output_dir=output,
        configuration={"epochs": 2},
        resume=False,
    )
    store.save_json(
        unit="capacity/nz-3/seed-0",
        relative_path="checkpoints/capacity/nz-3-seed-0.json",
        payload={"mean_squared_error": 0.2},
    )
    (output / "checkpoints/capacity/nz-3-seed-0.json").write_text(
        "{}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="hash drift"):
        prediction_checkpoint.PredictionRunCheckpointStore(
            output_dir=output,
            configuration={"epochs": 2},
            resume=True,
        )


def test_prediction_output_lock_rejects_concurrent_writer(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    first = prediction_runner._acquire_output_lock(output)
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            prediction_runner._acquire_output_lock(output)
    finally:
        first.close()

    released = prediction_runner._acquire_output_lock(output)
    released.close()


def test_prediction_checkpoint_formal_claim_requires_authorization(
    tmp_path: Path,
) -> None:
    unauthorized = prediction_checkpoint.PredictionRunCheckpointStore(
        output_dir=tmp_path / "unauthorized",
        configuration={"formal": False},
        resume=False,
    )
    with pytest.raises(ValueError, match="requires preregistered authorization"):
        unauthorized.mark_complete(formal_claim_allowed=True)

    authorized = prediction_checkpoint.PredictionRunCheckpointStore(
        output_dir=tmp_path / "authorized",
        configuration={"formal": True},
        resume=False,
        formal_claim_authorized=True,
    )
    authorized.mark_complete(formal_claim_allowed=True)
    state = json.loads(
        (tmp_path / "authorized/run_state.json").read_text(encoding="utf-8")
    )
    assert state["formal_claim_allowed"] is True
