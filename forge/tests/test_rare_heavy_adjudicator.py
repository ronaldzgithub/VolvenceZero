from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.rare_heavy import (
    RareHeavyTrainingSpec,
    create_rare_heavy_request,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "forge_common_adapter_adjudicator.py"
SPEC = importlib.util.spec_from_file_location(
    "forge_common_adapter_adjudicator",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load rare-heavy adjudicator from {SCRIPT_PATH}")
ADJUDICATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADJUDICATOR
SPEC.loader.exec_module(ADJUDICATOR)


def _config(tmp_path: Path) -> ForgeConfig:
    repo = tmp_path / "repo"
    forge = repo / "forge"
    forge.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "forge" / "editable_surface.yaml", forge)
    shutil.copytree(REPO_ROOT / "forge" / "schemas", forge / "schemas")
    return ForgeConfig.load(ForgePaths.discover(repo_root=repo))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matching_evidence(
    *,
    request: dict[str, object],
    control_path: Path,
    held_out_path: Path,
    gate_decision: str = "allow",
) -> SimpleNamespace:
    training = request["training"]
    inputs = request["inputs"]
    base = request["base_model"]
    assert isinstance(training, dict)
    assert isinstance(inputs, dict)
    assert isinstance(base, dict)
    traces = inputs["traces"]
    assert isinstance(traces, dict)
    provenance_names = (
        "seed",
        "target_modules",
        "hook_layers",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "learning_rate",
        "max_steps",
        "state_kv_seed",
        "state_kv_states",
        "state_kv_epochs",
        "state_kv_slots",
        "state_kv_rank",
        "state_kv_norm_cap",
        "state_kv_learning_rate",
    )
    provenance = {name: training[name] for name in provenance_names}
    provenance["traces_sha256"] = traces["sha256"]
    provenance["trace_count"] = traces["trace_count"]
    candidate = {
        "candidate_id": "candidate-1",
        "base_model_id": base["model_id"],
        "base_model_weights_sha256": base["weights_sha256"],
        "common_adapter_version": training["common_adapter_version"],
        "description": training["description"],
        "training_order": request["training_order"],
        "training_provenance": provenance,
    }
    checkpoint = SimpleNamespace(
        runtime_origin=training["runtime_origin"],
        control_scale=training["control_scale"],
    )
    material = SimpleNamespace(
        payload=candidate,
        control_path=control_path,
        rare_heavy_checkpoint=checkpoint,
    )
    report = {
        "held_out_sha256": _sha(held_out_path),
        "thresholds": request["evaluation"],
    }
    return SimpleNamespace(
        material=material,
        report=report,
        gate=SimpleNamespace(
            decision=gate_decision,
            allows_active=gate_decision == "allow",
        ),
    )


def _request(config: ForgeConfig) -> tuple[Path, Path, Path]:
    inputs = config.paths.artifacts_root / "inputs"
    inputs.mkdir(parents=True)
    traces = inputs / "traces.jsonl"
    control = inputs / "control.json"
    held_out = inputs / "held-out.jsonl"
    traces.write_text('{"trace_id":"t1","source_text":"hello"}\n', encoding="utf-8")
    control.write_text('{"artifact":"control"}\n', encoding="utf-8")
    held_out.write_text('{"case_id":"h1"}\n', encoding="utf-8")
    result = create_rare_heavy_request(
        config=config,
        model_id="Qwen/test",
        model_weights_sha256="a" * 64,
        traces_path=traces,
        control_basis_path=control,
        held_out_path=held_out,
        training=RareHeavyTrainingSpec(
            common_adapter_version="common-forge-v1",
            runtime_origin="hf-local",
            description="bounded request",
            seed=7,
            target_modules=("q_proj",),
            hook_layers=(0,),
            control_scale=0.1,
            lora_rank=1,
            lora_alpha=1,
            lora_dropout=0.0,
            learning_rate=0.001,
            max_steps=1,
            state_kv_seed=8,
            state_kv_states=2,
            state_kv_epochs=1,
            state_kv_slots=1,
            state_kv_rank=1,
            state_kv_norm_cap=0.1,
            state_kv_learning_rate=0.01,
        ),
    )
    return result.request_path, control, held_out


def test_adjudicator_emits_ready_without_publishing(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    request_path, control, held_out = _request(config)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    evidence = _matching_evidence(
        request=request,
        control_path=control,
        held_out_path=held_out,
    )
    monkeypatch.setattr(
        ADJUDICATOR.pipeline,
        "validate_common_adapter_evidence",
        lambda **_: evidence,
    )
    candidate = tmp_path / "candidate.json"
    evaluation = tmp_path / "evaluation.json"
    gate = tmp_path / "gate.json"
    for path in (candidate, evaluation, gate):
        path.write_text("{}\n", encoding="utf-8")
    output = config.paths.artifacts_root / "verdict" / "ready.json"

    verdict = ADJUDICATOR.adjudicate_rare_heavy_request(
        config=config,
        request_path=request_path,
        candidate_path=candidate,
        evaluation_report_path=evaluation,
        gate_path=gate,
        held_out_path=held_out,
        output_path=output,
    )

    assert verdict["decision"] == "READY"
    assert verdict["bindings"] == {
        "candidate": True,
        "training": True,
        "held_out": True,
        "evaluation": True,
        "gate": True,
    }
    assert output.is_file()
    assert not (tmp_path / "common-adapter-bundle.json").exists()


def test_adjudicator_stops_on_denied_gate(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    request_path, control, held_out = _request(config)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    evidence = _matching_evidence(
        request=request,
        control_path=control,
        held_out_path=held_out,
        gate_decision="deny",
    )
    monkeypatch.setattr(
        ADJUDICATOR.pipeline,
        "validate_common_adapter_evidence",
        lambda **_: evidence,
    )
    candidate = tmp_path / "candidate.json"
    evaluation = tmp_path / "evaluation.json"
    gate = tmp_path / "gate.json"
    for path in (candidate, evaluation, gate):
        path.write_text("{}\n", encoding="utf-8")

    verdict = ADJUDICATOR.adjudicate_rare_heavy_request(
        config=config,
        request_path=request_path,
        candidate_path=candidate,
        evaluation_report_path=evaluation,
        gate_path=gate,
        held_out_path=held_out,
        output_path=config.paths.artifacts_root / "verdict" / "stop.json",
    )

    assert verdict["decision"] == "STOP"
    assert verdict["gate_decision"] == "deny"
    assert verdict["bindings"]["gate"] is False
    assert any(
        "cognition OFFLINE gate did not provide a reversible ALLOW" in reason
        for reason in verdict["reasons"]
    )


def test_adjudicator_stops_when_expensive_evidence_is_absent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    request_path, _, held_out = _request(config)

    verdict = ADJUDICATOR.adjudicate_rare_heavy_request(
        config=config,
        request_path=request_path,
        candidate_path=config.paths.artifacts_root / "missing-candidate.json",
        evaluation_report_path=config.paths.artifacts_root / "missing-evaluation.json",
        gate_path=config.paths.artifacts_root / "missing-gate.json",
        held_out_path=held_out,
        output_path=config.paths.artifacts_root / "verdict" / "missing-stop.json",
    )

    assert verdict["decision"] == "STOP"
    assert verdict["candidate_id"] is None
    assert verdict["gate_decision"] == "unavailable"
    assert verdict["evidence"]["candidate_sha256"] is None
    assert any(
        "evidence validation failed" in reason for reason in verdict["reasons"]
    )
