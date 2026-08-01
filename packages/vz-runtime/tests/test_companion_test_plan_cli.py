from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import companion_test_plan_common as common  # noqa: E402
import msc_prediction_checkpoint as prediction_checkpoint  # noqa: E402
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


def test_execution_environment_disables_mps_fallback(tmp_path: Path) -> None:
    (tmp_path / "packages/example/src").mkdir(parents=True)
    environment = common.execution_environment(tmp_path)

    assert environment["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert str(tmp_path / "packages/example/src") in environment["PYTHONPATH"]


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
            schema="seven-day-companion-simulated.v1",
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
                schema="seven-day-companion-simulated.v1",
                arms=("correct-user-state", "stateless"),
            ),
            "run_seven_day_companion_formal.py",
            "audit_seven_day_companion_formal.py",
            "--smoke-one-run",
            None,
        ),
        (
            _seven_day_preregistration(
                schema="seven-day-companion-simulated.v2",
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
    return_codes = iter((0, seven_day_plan.SCIENTIFIC_NEGATIVE_EXIT, 0))

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
    assert len(commands) == 3
    assert "--preflight-only" in commands[0]
    assert "--execute" in commands[1]
    assert Path(commands[2][1]).name == "audit_seven_day_gate1_formal.py"


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


def test_prediction_formal_stage_is_fail_closed(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = prediction_plan.main(("formal",))

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == prediction_plan.FORMAL_BLOCKED_EXIT
    assert payload["formal_eligible"] is False
    assert payload["formal_blockers"] == list(prediction_plan.FORMAL_BLOCKERS)
    assert payload["formal_claim_permitted_now"] is False


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
