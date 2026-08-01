from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.cli import main
from volvence_forge.foundation import ForgeError
from volvence_forge.rare_heavy import (
    RareHeavyTrainingSpec,
    create_rare_heavy_request,
    validate_rare_heavy_request,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path) -> ForgeConfig:
    repo = tmp_path / "repo"
    forge = repo / "forge"
    forge.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "forge" / "editable_surface.yaml", forge)
    shutil.copytree(REPO_ROOT / "forge" / "schemas", forge / "schemas")
    return ForgeConfig.load(ForgePaths.discover(repo_root=repo))


def _inputs(config: ForgeConfig) -> tuple[Path, Path, Path]:
    root = config.paths.artifacts_root / "inputs"
    root.mkdir(parents=True)
    traces = root / "traces.jsonl"
    control = root / "control.json"
    held_out = root / "held-out.jsonl"
    traces.write_text('{"trace_id":"t1","source_text":"hello"}\n', encoding="utf-8")
    control.write_text('{"artifact":"control"}\n', encoding="utf-8")
    held_out.write_text('{"case_id":"h1"}\n', encoding="utf-8")
    return traces, control, held_out


def _training(**overrides: object) -> RareHeavyTrainingSpec:
    values: dict[str, object] = {
        "common_adapter_version": "common-forge-v1",
        "runtime_origin": "hf-local",
        "description": "bounded request",
        "seed": 7,
        "target_modules": ("q_proj", "v_proj"),
        "hook_layers": (1, 2),
        "control_scale": 0.12,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "learning_rate": 0.0005,
        "max_steps": 20,
        "state_kv_seed": 8,
        "state_kv_states": 16,
        "state_kv_epochs": 4,
        "state_kv_slots": 4,
        "state_kv_rank": 4,
        "state_kv_norm_cap": 0.2,
        "state_kv_learning_rate": 0.05,
    }
    values.update(overrides)
    return RareHeavyTrainingSpec(**values)  # type: ignore[arg-type]


def test_request_is_content_addressed_disabled_and_revalidates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    traces, control, held_out = _inputs(config)
    output = config.paths.artifacts_root / "request" / "request.json"

    result = create_rare_heavy_request(
        config=config,
        model_id="Qwen/test",
        model_weights_sha256="a" * 64,
        traces_path=traces,
        control_basis_path=control,
        held_out_path=held_out,
        training=_training(),
        output_path=output,
    )

    payload = validate_rare_heavy_request(
        config=config,
        request_path=result.request_path,
    )
    assert payload["request_id"] == result.request_id
    assert payload["requested_wiring"] == "DISABLED"
    assert payload["training_decides_gate"] is False
    assert payload["training_order"] == ["rare-heavy", "state-kv", "offline-gate"]
    assert payload["inputs"]["traces"]["locator"] == "artifacts/inputs/traces.jsonl"


def test_request_detects_input_tampering(tmp_path: Path) -> None:
    config = _config(tmp_path)
    traces, control, held_out = _inputs(config)
    result = create_rare_heavy_request(
        config=config,
        model_id="Qwen/test",
        model_weights_sha256="a" * 64,
        traces_path=traces,
        control_basis_path=control,
        held_out_path=held_out,
        training=_training(),
    )
    traces.write_text('{"trace_id":"changed"}\n', encoding="utf-8")

    with pytest.raises(ForgeError, match="traces digest mismatch"):
        validate_rare_heavy_request(
            config=config,
            request_path=result.request_path,
        )


def test_request_rejects_unbounded_training_and_runtime_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    traces, control, held_out = _inputs(config)
    with pytest.raises(ForgeError, match="maximum of 64"):
        create_rare_heavy_request(
            config=config,
            model_id="Qwen/test",
            model_weights_sha256="a" * 64,
            traces_path=traces,
            control_basis_path=control,
            held_out_path=held_out,
            training=_training(lora_rank=65),
        )

    with pytest.raises(ForgeError, match="only be written below artifacts"):
        create_rare_heavy_request(
            config=config,
            model_id="Qwen/test",
            model_weights_sha256="a" * 64,
            traces_path=traces,
            control_basis_path=control,
            held_out_path=held_out,
            training=_training(),
            output_path=config.paths.repo_root / "runtime-request.json",
        )


def test_request_identity_detects_manifest_tampering(tmp_path: Path) -> None:
    config = _config(tmp_path)
    traces, control, held_out = _inputs(config)
    result = create_rare_heavy_request(
        config=config,
        model_id="Qwen/test",
        model_weights_sha256="a" * 64,
        traces_path=traces,
        control_basis_path=control,
        held_out_path=held_out,
        training=_training(),
    )
    payload = json.loads(result.request_path.read_text(encoding="utf-8"))
    payload["training"]["max_steps"] = 21
    result.request_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ForgeError, match="request_id does not match"):
        validate_rare_heavy_request(
            config=config,
            request_path=result.request_path,
        )


def test_cli_plans_request_without_training(tmp_path: Path) -> None:
    config = _config(tmp_path)
    traces, control, held_out = _inputs(config)
    output = config.paths.artifacts_root / "cli-request" / "request.json"

    status = main(
        [
            "--repo-root",
            str(config.paths.repo_root),
            "plan-rare-heavy",
            "--model-id",
            "Qwen/test",
            "--model-weights-sha256",
            "a" * 64,
            "--common-adapter-version",
            "common-cli-v1",
            "--traces",
            str(traces),
            "--control-basis",
            str(control),
            "--held-out",
            str(held_out),
            "--hook-layers",
            "1,2",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert output.is_file()
    assert not (output.parent / "common-adapter-candidate.json").exists()
