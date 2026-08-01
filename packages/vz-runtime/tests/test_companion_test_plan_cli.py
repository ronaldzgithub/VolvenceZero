from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import companion_test_plan_common as common  # noqa: E402
import msc_prediction_checkpoint as prediction_checkpoint  # noqa: E402
import run_msc_prediction_test_plan as prediction_plan  # noqa: E402
import run_seven_day_companion_test_plan as seven_day_plan  # noqa: E402


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

    command = seven_day_plan._runner_command(
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
    preregistration.write_text(
        json.dumps(
            {
                "claim_scope": "simulated-user-real-lifecycle-only",
                "formal_run": {"run_count": 36, "execution_device": "mps"},
            }
        ),
        encoding="utf-8",
    )

    status = seven_day_plan._status(
        preregistration=preregistration,
        output_dir=None,
    )

    assert status["expected_runs"] == 36
    assert status["matrix_complete"] is False
    assert status["production_promotion_authorized"] is False


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
