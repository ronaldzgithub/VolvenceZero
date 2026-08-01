#!/usr/bin/env python3
"""Train and publish a versioned shared Common Adapter Model.

The ``train`` command enforces the only supported order: PEFT rare-heavy is
trained first, then the frozen base plus that admitted delta is used for every
State-KV teacher/student forward.  It emits an immutable candidate and a
ModificationGate proposal; it does not decide its own gate.

The ``publish`` command consumes the candidate plus a cognition-owned gate
record and exports ``CommonAdapterBundle``.  A denied record remains auditable
but cannot be loaded ACTIVE by the runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for _src in (SCRIPT_ROOT, *sorted((REPO_ROOT / "packages").glob("*/src"))):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from volvence_zero.substrate import (  # noqa: E402
    CommonAdapterBundle,
    CommonAdapterGateRecord,
    ControlBasisArtifact,
    PeftLoraRareHeavyBackend,
    PrefixKVArtifact,
    RareHeavyTrainingRequest,
    SubstrateRareHeavyCheckpoint,
    TrainingTrace,
    build_rare_heavy_compatibility_fingerprint,
    fingerprint_model_weight_files,
    rare_heavy_checkpoint_from_json,
    rare_heavy_checkpoint_to_json,
)
from adapter_promotion_evidence import (  # noqa: E402
    ADAPTER_PROMOTION_REPORT_SCHEMA_VERSION,
    AdapterArmObservation,
    AdapterPromotionThresholds,
    collect_observations,
    conditioning_snapshot,
    decide_offline_promotion,
    evaluation_id,
    load_held_out_cases,
    summarize_observations,
)
from volvence_zero.credit import GateDecision  # noqa: E402
from volvence_zero.substrate import (  # noqa: E402
    TransformersOpenWeightResidualRuntime,
)

COMMON_ADAPTER_CANDIDATE_SCHEMA = "common-adapter-candidate.v2"
COMMON_ADAPTER_PROPOSAL_SCHEMA = "common-adapter-modification-proposal.v1"


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_model_source(
    *, model_id: str, model_source: str, allow_download: bool
) -> Path:
    if model_source:
        resolved = Path(model_source).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"--model-source is not a model snapshot directory: {resolved}"
            )
        return resolved
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "resolving a model id requires huggingface_hub; provide "
            "--model-source or install the training dependencies."
        ) from exc
    return Path(
        snapshot_download(
            repo_id=model_id,
            local_files_only=not allow_download,
        )
    ).resolve()


def _load_traces(path: Path) -> tuple[TrainingTrace, ...]:
    traces: list[TrainingTrace] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"trace line {line_number} must be a JSON object.")
        required = {"trace_id", "source_text"}
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required)
        if missing or extra:
            raise ValueError(
                f"trace line {line_number} fields mismatch: "
                f"missing={missing}, extra={extra}."
            )
        trace_id = str(raw["trace_id"]).strip()
        source_text = str(raw["source_text"]).strip()
        if not trace_id or not source_text:
            raise ValueError(
                f"trace line {line_number} requires non-empty trace_id/source_text."
            )
        if trace_id in seen:
            raise ValueError(f"duplicate trace_id {trace_id!r}.")
        seen.add(trace_id)
        traces.append(
            TrainingTrace(trace_id=trace_id, source_text=source_text, steps=())
        )
    if not traces:
        raise ValueError("common adapter training requires at least one trace.")
    return tuple(traces)


def _model_geometry(*, model_source: Path) -> tuple[int, int]:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(
            "common adapter training requires transformers + peft + torch."
        ) from exc
    config = AutoConfig.from_pretrained(str(model_source), local_files_only=True)
    hidden_size = int(
        getattr(
            config,
            "hidden_size",
            getattr(config, "n_embd", getattr(config, "d_model", 0)),
        )
    )
    layer_count = int(
        getattr(
            config,
            "num_hidden_layers",
            getattr(config, "n_layer", getattr(config, "num_layers", 0)),
        )
    )
    if hidden_size <= 0 or layer_count <= 0:
        raise ValueError("could not resolve model hidden size/layer count.")
    return hidden_size, layer_count


def _hook_layers(raw: str, *, layer_count: int) -> tuple[int, ...]:
    if raw.strip():
        indices = tuple(sorted({int(value) for value in raw.split(",")}))
    elif layer_count <= 3:
        indices = tuple(range(layer_count))
    else:
        middle = layer_count // 2
        indices = tuple(sorted({middle - 1, middle, min(layer_count - 1, middle + 1)}))
    if not indices or indices[0] < 0 or indices[-1] >= layer_count:
        raise ValueError(
            f"hook layers {indices!r} fall outside model layer count {layer_count}."
        )
    return indices


def _candidate_id(payload: dict[str, Any]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "candidate_id"}
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def _relative(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def train_candidate(args: argparse.Namespace) -> tuple[Path, Path]:
    traces = _load_traces(args.traces.expanduser().resolve())
    model_source = _resolve_model_source(
        model_id=args.model_id,
        model_source=args.model_source,
        allow_download=args.allow_download,
    )
    base_weights_sha256 = fingerprint_model_weight_files(model_source)
    hidden_size, layer_count = _model_geometry(model_source=model_source)
    layer_indices = _hook_layers(args.hook_layers, layer_count=layer_count)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "common adapter training requires transformers + peft + torch."
        ) from exc
    torch.manual_seed(args.seed)
    backend = PeftLoraRareHeavyBackend(
        target_modules=tuple(args.target_modules),
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        base_model_source=str(model_source),
    )
    result = backend.train(
        RareHeavyTrainingRequest(
            model_id=args.model_id,
            hidden_size=hidden_size,
            layer_indices=layer_indices,
            device=args.device,
            traces=traces,
        )
    )
    runtime_origin = args.runtime_origin
    checkpoint = SubstrateRareHeavyCheckpoint(
        checkpoint_id=f"{args.common_adapter_version}:rare-heavy",
        model_id=args.model_id,
        runtime_origin=runtime_origin,
        control_scale=args.control_scale,
        semantic_text_weight=0.5,
        semantic_residual_weight=0.5,
        semantic_anchor_bias=(0.0, 0.0, 0.0, 0.0, 0.0),
        update_count=1,
        source_batch_count=len(traces),
        mean_sequence_length=sum(
            len(trace.source_text.split()) for trace in traces
        )
        / len(traces),
        mean_residual_magnitude=sum(
            layer.mean_abs_delta for layer in result.adapter_layers
        )
        / len(result.adapter_layers),
        description=result.description,
        checkpoint_version=2,
        training_mode=result.training_mode,
        compatibility_fingerprint=build_rare_heavy_compatibility_fingerprint(
            model_id=args.model_id,
            runtime_origin=runtime_origin,
            hidden_size=hidden_size,
            layer_indices=layer_indices,
            training_mode=result.training_mode,
        ),
        adapter_scale=1.0,
        adapter_parameter_count=sum(
            len(layer.delta_vector) for layer in result.adapter_layers
        ),
        adapter_training_loss=result.training_loss,
        adapter_layers=result.adapter_layers,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "rare-heavy-checkpoint.json"
    checkpoint_path.write_text(
        rare_heavy_checkpoint_to_json(checkpoint) + "\n",
        encoding="utf-8",
    )
    state_path = output_dir / "state-kv-prefix.json"
    state_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_state_kv_prefix.py"),
        "--model-id",
        args.model_id,
        "--model-source",
        str(model_source),
        "--device",
        args.device,
        "--states",
        str(args.state_kv_states),
        "--epochs",
        str(args.state_kv_epochs),
        "--slots",
        str(args.state_kv_slots),
        "--rank",
        str(args.state_kv_rank),
        "--norm-cap",
        str(args.state_kv_norm_cap),
        "--learning-rate",
        str(args.state_kv_learning_rate),
        "--output",
        str(state_path),
        "--common-adapter-checkpoint",
        str(checkpoint_path),
        "--common-adapter-version",
        args.common_adapter_version,
        "--seed",
        str(args.state_kv_seed),
    ]
    subprocess.run(state_command, check=True)

    state_manifest_path = state_path.with_name(state_path.stem + ".manifest.json")
    state_manifest = json.loads(state_manifest_path.read_text(encoding="utf-8"))
    if state_manifest.get("training_order") != "base+rare-heavy->state-kv":
        raise RuntimeError("State-KV trainer did not attest the required order.")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if state_manifest.get("rare_heavy_checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("State-KV manifest checkpoint digest drifted.")
    if state_manifest.get("weights_sha256") != base_weights_sha256:
        raise RuntimeError("rare-heavy and State-KV stages used different weights.")

    control_basis_path = args.control_basis.expanduser().resolve()
    control_basis = ControlBasisArtifact.from_json(
        control_basis_path.read_text(encoding="utf-8")
    )
    state_artifact = PrefixKVArtifact.from_json(
        state_path.read_text(encoding="utf-8")
    )
    compatibility_fingerprint = CommonAdapterBundle.build_compatibility_fingerprint(
        common_adapter_version=args.common_adapter_version,
        base_model_id=args.model_id,
        base_model_weights_sha256=base_weights_sha256,
        rare_heavy_checkpoint=checkpoint,
        state_kv_artifact=state_artifact,
        control_basis_artifact=control_basis,
    )
    candidate: dict[str, Any] = {
        "schema_version": COMMON_ADAPTER_CANDIDATE_SCHEMA,
        "candidate_id": "pending",
        "common_adapter_version": args.common_adapter_version,
        "base_model_id": args.model_id,
        "base_model_weights_sha256": base_weights_sha256,
        "compatibility_fingerprint": compatibility_fingerprint,
        "rare_heavy_checkpoint": {
            "locator": _relative(checkpoint_path, base=output_dir),
            "sha256": checkpoint_sha256,
        },
        "state_kv_artifact": {
            "locator": _relative(state_path, base=output_dir),
            "sha256": _sha256_file(state_path),
            "training_manifest_locator": _relative(
                state_manifest_path, base=output_dir
            ),
            "training_manifest_sha256": _sha256_file(state_manifest_path),
        },
        "control_basis_artifact": {
            "locator": _relative(control_basis_path, base=output_dir),
            "sha256": _sha256_file(control_basis_path),
        },
        "training_order": ("rare-heavy", "state-kv", "offline-gate"),
        "training_provenance": {
            "traces_sha256": _sha256_file(args.traces.expanduser().resolve()),
            "trace_count": len(traces),
            "seed": args.seed,
            "target_modules": list(args.target_modules),
            "hook_layers": list(layer_indices),
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
            "state_kv_seed": args.state_kv_seed,
            "state_kv_states": args.state_kv_states,
            "state_kv_epochs": args.state_kv_epochs,
            "state_kv_slots": args.state_kv_slots,
            "state_kv_rank": args.state_kv_rank,
            "state_kv_norm_cap": args.state_kv_norm_cap,
            "state_kv_learning_rate": args.state_kv_learning_rate,
        },
        "description": args.description,
    }
    candidate["candidate_id"] = _candidate_id(candidate)
    candidate_path = output_dir / "common-adapter-candidate.json"
    _write_json(candidate_path, candidate)
    proposal = {
        "schema_version": COMMON_ADAPTER_PROPOSAL_SCHEMA,
        "proposal_id": f"common-adapter:{candidate['candidate_id']}",
        "candidate_id": candidate["candidate_id"],
        "desired_gate": "offline",
        "common_adapter_version": args.common_adapter_version,
        "compatibility_fingerprint": compatibility_fingerprint,
        "requires": [
            "held-out validation_delta",
            "capacity_cost",
            "rollback evidence",
            "reversible decision",
        ],
        "training_decides_gate": False,
    }
    proposal_path = output_dir / "modification-gate-proposal.json"
    _write_json(proposal_path, proposal)
    return candidate_path, proposal_path


def _resolve_ref(candidate_path: Path, raw: object, *, name: str) -> Path:
    if not isinstance(raw, dict):
        raise ValueError(f"candidate {name} must be an object.")
    locator = raw.get("locator")
    declared = raw.get("sha256")
    if not isinstance(locator, str) or not locator.strip():
        raise ValueError(f"candidate {name}.locator must be non-empty.")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError(f"candidate {name}.sha256 must be a SHA-256 digest.")
    try:
        int(declared, 16)
    except ValueError as exc:
        raise ValueError(
            f"candidate {name}.sha256 must be hexadecimal."
        ) from exc
    path = Path(locator).expanduser()
    if not path.is_absolute():
        path = candidate_path.parent / path
    path = path.resolve()
    actual = _sha256_file(path)
    if actual != declared:
        raise ValueError(
            f"candidate {name} digest mismatch: declared={declared}, actual={actual}."
        )
    return path


@dataclass(frozen=True)
class CommonAdapterCandidateMaterial:
    candidate_path: Path
    payload: dict[str, Any]
    checkpoint_path: Path
    state_path: Path
    state_manifest_path: Path
    control_path: Path
    rare_heavy_checkpoint: SubstrateRareHeavyCheckpoint
    state_kv_artifact: PrefixKVArtifact
    control_basis_artifact: ControlBasisArtifact


def load_candidate_material(
    candidate_path: Path,
) -> CommonAdapterCandidateMaterial:
    """Verify a candidate and all content-addressed training inputs."""

    candidate_path = candidate_path.expanduser().resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(candidate, dict):
        raise ValueError("common adapter candidate must be an object.")
    if candidate.get("schema_version") != COMMON_ADAPTER_CANDIDATE_SCHEMA:
        raise ValueError("unsupported common adapter candidate schema.")
    required_candidate_fields = {
        "schema_version",
        "candidate_id",
        "common_adapter_version",
        "base_model_id",
        "base_model_weights_sha256",
        "compatibility_fingerprint",
        "rare_heavy_checkpoint",
        "state_kv_artifact",
        "control_basis_artifact",
        "training_order",
        "training_provenance",
        "description",
    }
    missing_candidate = sorted(required_candidate_fields - set(candidate))
    extra_candidate = sorted(set(candidate) - required_candidate_fields)
    if missing_candidate or extra_candidate:
        raise ValueError(
            "common adapter candidate fields mismatch: "
            f"missing={missing_candidate}, extra={extra_candidate}."
        )
    if candidate.get("candidate_id") != _candidate_id(candidate):
        raise ValueError("common adapter candidate_id does not match payload.")
    if candidate["training_order"] != [
        "rare-heavy",
        "state-kv",
        "offline-gate",
    ]:
        raise ValueError("common adapter candidate training_order is invalid.")
    provenance = candidate.get("training_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(
            "common adapter candidate must bind training_provenance."
        )
    required_provenance = {
        "traces_sha256",
        "trace_count",
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
    }
    missing_provenance = sorted(required_provenance - set(provenance))
    extra_provenance = sorted(set(provenance) - required_provenance)
    if missing_provenance or extra_provenance:
        raise ValueError(
            "common adapter training_provenance fields mismatch: "
            f"missing={missing_provenance}, extra={extra_provenance}."
        )
    traces_sha256 = provenance["traces_sha256"]
    if not isinstance(traces_sha256, str) or len(traces_sha256) != 64:
        raise ValueError("training_provenance.traces_sha256 must be SHA-256.")
    try:
        int(traces_sha256, 16)
    except ValueError as exc:
        raise ValueError(
            "training_provenance.traces_sha256 must be hexadecimal."
        ) from exc
    positive_integer_fields = (
        "trace_count",
        "lora_rank",
        "lora_alpha",
        "max_steps",
        "state_kv_states",
        "state_kv_epochs",
        "state_kv_slots",
        "state_kv_rank",
    )
    for name in positive_integer_fields:
        if type(provenance[name]) is not int or provenance[name] <= 0:
            raise ValueError(
                f"training_provenance.{name} must be a positive integer."
            )
    for name in ("seed", "state_kv_seed"):
        if type(provenance[name]) is not int or provenance[name] < 0:
            raise ValueError(
                f"training_provenance.{name} must be a non-negative integer."
            )
    target_modules = provenance["target_modules"]
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or any(
            not isinstance(name, str) or not name.strip()
            for name in target_modules
        )
        or len(set(target_modules)) != len(target_modules)
    ):
        raise ValueError(
            "training_provenance.target_modules must be unique non-empty names."
        )
    hook_layers = provenance["hook_layers"]
    if (
        not isinstance(hook_layers, list)
        or not hook_layers
        or any(type(index) is not int or index < 0 for index in hook_layers)
        or len(set(hook_layers)) != len(hook_layers)
    ):
        raise ValueError(
            "training_provenance.hook_layers must be unique non-negative integers."
        )
    for name in (
        "lora_dropout",
        "learning_rate",
        "state_kv_norm_cap",
        "state_kv_learning_rate",
    ):
        value = provenance[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"training_provenance.{name} must be numeric."
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"training_provenance.{name} must be finite."
            )
    if not 0.0 <= float(provenance["lora_dropout"]) < 1.0:
        raise ValueError("training_provenance.lora_dropout must be in [0, 1).")
    for name in (
        "learning_rate",
        "state_kv_norm_cap",
        "state_kv_learning_rate",
    ):
        if float(provenance[name]) <= 0.0:
            raise ValueError(
                f"training_provenance.{name} must be positive."
            )
    checkpoint_path = _resolve_ref(
        candidate_path,
        candidate.get("rare_heavy_checkpoint"),
        name="rare_heavy_checkpoint",
    )
    state_path = _resolve_ref(
        candidate_path,
        candidate.get("state_kv_artifact"),
        name="state_kv_artifact",
    )
    control_path = _resolve_ref(
        candidate_path,
        candidate.get("control_basis_artifact"),
        name="control_basis_artifact",
    )
    state_ref = candidate["state_kv_artifact"]
    state_manifest_locator = state_ref.get("training_manifest_locator")
    state_manifest_sha = state_ref.get("training_manifest_sha256")
    if not isinstance(state_manifest_locator, str) or not isinstance(
        state_manifest_sha,
        str,
    ):
        raise ValueError(
            "candidate State-KV training manifest reference is missing."
        )
    state_manifest_path = Path(state_manifest_locator)
    if not state_manifest_path.is_absolute():
        state_manifest_path = candidate_path.parent / state_manifest_path
    state_manifest_path = state_manifest_path.resolve()
    if _sha256_file(state_manifest_path) != state_manifest_sha:
        raise ValueError("candidate State-KV training manifest digest mismatch.")
    state_manifest = json.loads(
        state_manifest_path.read_text(encoding="utf-8")
    )
    if state_manifest.get("training_order") != "base+rare-heavy->state-kv":
        raise ValueError("candidate State-KV was not trained after rare-heavy.")
    if state_manifest.get("common_adapter_version") != candidate.get(
        "common_adapter_version"
    ):
        raise ValueError(
            "candidate State-KV common adapter version drifted."
        )
    checkpoint = rare_heavy_checkpoint_from_json(
        checkpoint_path.read_text(encoding="utf-8")
    )
    state = PrefixKVArtifact.from_json(
        state_path.read_text(encoding="utf-8")
    )
    control = ControlBasisArtifact.from_json(
        control_path.read_text(encoding="utf-8")
    )
    checkpoint_layers = [
        layer.layer_index for layer in checkpoint.adapter_layers
    ]
    if hook_layers != checkpoint_layers:
        raise ValueError(
            "training_provenance.hook_layers do not match rare-heavy layers."
        )
    expected_state_manifest = {
        "schema_version": "state-kv-prefix-bake.v1",
        "artifact_id": state.artifact_id,
        "model_id": candidate["base_model_id"],
        "weights_sha256": candidate["base_model_weights_sha256"],
        "common_adapter_version": candidate["common_adapter_version"],
        "rare_heavy_checkpoint_sha256": _sha256_file(checkpoint_path),
        "rare_heavy_compatibility_fingerprint": (
            checkpoint.compatibility_fingerprint
        ),
        "training_order": "base+rare-heavy->state-kv",
        "state_count": provenance["state_kv_states"],
        "seed": provenance["state_kv_seed"],
        "epochs": provenance["state_kv_epochs"],
        "learning_rate": provenance["state_kv_learning_rate"],
        "num_slots": provenance["state_kv_slots"],
        "bottleneck_rank": provenance["state_kv_rank"],
        "norm_cap": provenance["state_kv_norm_cap"],
    }
    for name, expected in expected_state_manifest.items():
        if state_manifest.get(name) != expected:
            raise ValueError(
                "candidate State-KV training manifest drifted for "
                f"{name}: expected={expected!r}, "
                f"actual={state_manifest.get(name)!r}."
            )
    if (
        state.num_slots != provenance["state_kv_slots"]
        or state.bottleneck_rank != provenance["state_kv_rank"]
        or state.norm_cap != provenance["state_kv_norm_cap"]
    ):
        raise ValueError(
            "candidate State-KV artifact geometry/provenance mismatch."
        )
    expected_fingerprint = CommonAdapterBundle.build_compatibility_fingerprint(
        common_adapter_version=str(candidate["common_adapter_version"]),
        base_model_id=str(candidate["base_model_id"]),
        base_model_weights_sha256=str(candidate["base_model_weights_sha256"]),
        rare_heavy_checkpoint=checkpoint,
        state_kv_artifact=state,
        control_basis_artifact=control,
    )
    if candidate.get("compatibility_fingerprint") != expected_fingerprint:
        raise ValueError(
            "common adapter candidate compatibility fingerprint drifted."
        )
    return CommonAdapterCandidateMaterial(
        candidate_path=candidate_path,
        payload=candidate,
        checkpoint_path=checkpoint_path,
        state_path=state_path,
        state_manifest_path=state_manifest_path,
        control_path=control_path,
        rare_heavy_checkpoint=checkpoint,
        state_kv_artifact=state,
        control_basis_artifact=control,
    )


def _validated_evaluation_report(
    *,
    candidate: dict[str, Any],
    gate: CommonAdapterGateRecord,
    report_path: Path,
    held_out_path: Path,
) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("common adapter evaluation report must be an object.")
    required = {
        "schema_version",
        "evaluation_id",
        "subject_kind",
        "subject_id",
        "common_adapter_version",
        "base_model_id",
        "base_model_weights_sha256",
        "held_out_sha256",
        "held_out",
        "source_immutable",
        "feedback_free",
        "thresholds",
        "observations",
        "summary",
        "artifact_parameter_count",
        "base_model_parameter_count",
        "capacity_cost",
        "evaluation_snapshot",
        "gate_reasons",
        "decision",
        "rollback_evidence",
    }
    missing = sorted(required - set(report))
    extra = sorted(set(report) - required)
    if missing or extra:
        raise ValueError(
            "common adapter evaluation report fields mismatch: "
            f"missing={missing}, extra={extra}."
        )
    expected_subject = (
        "common-adapter-candidate",
        candidate["candidate_id"],
        candidate["common_adapter_version"],
        candidate["base_model_id"],
        candidate["base_model_weights_sha256"],
    )
    actual_subject = (
        report["subject_kind"],
        report["subject_id"],
        report["common_adapter_version"],
        report["base_model_id"],
        report["base_model_weights_sha256"],
    )
    if actual_subject != expected_subject:
        raise ValueError(
            "common adapter evaluation report does not bind this candidate."
        )
    if not all(
        report[name] is True
        for name in ("held_out", "source_immutable", "feedback_free")
    ):
        raise ValueError(
            "common adapter evaluation report must be held-out, immutable, "
            "and feedback-free."
        )
    held_out_path = held_out_path.expanduser().resolve()
    if report["held_out_sha256"] != _sha256_file(held_out_path):
        raise ValueError(
            "common adapter evaluation held-out corpus digest mismatch."
        )
    thresholds_raw = report["thresholds"]
    observations_raw = report["observations"]
    if not isinstance(thresholds_raw, dict) or not isinstance(
        observations_raw,
        list,
    ):
        raise ValueError(
            "common adapter report thresholds/observations are malformed."
        )
    thresholds = AdapterPromotionThresholds(**thresholds_raw)
    observations = tuple(
        AdapterArmObservation(**row) for row in observations_raw
    )
    if len({row.case_id for row in observations}) != len(observations):
        raise ValueError(
            "common adapter evaluation observations contain duplicate case ids."
        )
    held_out_cases = load_held_out_cases(held_out_path)
    held_out_bindings = tuple(
        (
            case.case_id,
            case.cohort,
            case.expectation,
            case.counterfactual_conditioning_state is not None,
        )
        for case in held_out_cases
    )
    observation_bindings = tuple(
        (
            row.case_id,
            row.cohort,
            row.expectation,
            row.own_state_margin is not None,
        )
        for row in observations
    )
    if observation_bindings != held_out_bindings:
        raise ValueError(
            "common adapter observations do not bind the ordered held-out cases."
        )
    summary = summarize_observations(
        observations=observations,
        thresholds=thresholds,
    )
    if report["summary"] != summary:
        raise ValueError(
            "common adapter evaluation summary does not match observations."
        )
    expected_evaluation_id = evaluation_id(
        subject_id=str(candidate["candidate_id"]),
        observations=observations,
        thresholds=thresholds,
    )
    if report["evaluation_id"] != expected_evaluation_id:
        raise ValueError(
            "common adapter evaluation_id does not match observations."
        )
    rollback_evidence = str(report["rollback_evidence"])
    decision, reasons, _ = decide_offline_promotion(
        target="substrate.common_adapter_bundle",
        old_value_hash=str(candidate["base_model_weights_sha256"]),
        new_value_hash=str(candidate["candidate_id"]),
        summary=summary,
        capacity_cost=float(report["capacity_cost"]),
        rollback_evidence=rollback_evidence,
    )
    expected_decision = (
        "allow" if decision is GateDecision.ALLOW else "deny"
    )
    if report["decision"] != expected_decision or report["gate_reasons"] != list(
        reasons
    ):
        raise ValueError(
            "common adapter report decision does not match cognition gate."
        )
    report_sha256 = _sha256_file(report_path)
    expected_evaluation_ref = (
        f"common-adapter-evaluation:{expected_evaluation_id}:"
        f"sha256:{report_sha256}"
    )
    if gate.evaluation_ref != expected_evaluation_ref:
        raise ValueError(
            "common adapter gate does not bind the evaluation report digest."
        )
    if (
        gate.decision != expected_decision
        or gate.validation_delta != float(summary["validation_delta"])
        or gate.capacity_cost != float(report["capacity_cost"])
        or gate.rollback_evidence != rollback_evidence
    ):
        raise ValueError(
            "common adapter gate record does not match evaluation evidence."
        )
    return report


@dataclass(frozen=True)
class CommonAdapterValidatedEvidence:
    """Verified candidate material and cognition-owned promotion evidence."""

    material: CommonAdapterCandidateMaterial
    gate: CommonAdapterGateRecord
    report: dict[str, Any]


def validate_common_adapter_evidence(
    *,
    candidate_path: Path,
    gate_path: Path,
    evaluation_report_path: Path,
    held_out_path: Path,
) -> CommonAdapterValidatedEvidence:
    """Validate the complete rare-heavy/State-KV/OFFLINE-gate evidence chain."""

    material = load_candidate_material(candidate_path)
    candidate = material.payload
    gate_raw = json.loads(
        gate_path.expanduser().resolve().read_text(encoding="utf-8")
    )
    if not isinstance(gate_raw, dict):
        raise ValueError("common adapter gate record must be an object.")
    gate = CommonAdapterGateRecord(**gate_raw)
    expected_proposal = f"common-adapter:{candidate['candidate_id']}"
    if gate.proposal_id != expected_proposal:
        raise ValueError(
            "gate proposal_id does not bind this common adapter candidate."
        )
    report = _validated_evaluation_report(
        candidate=candidate,
        gate=gate,
        report_path=evaluation_report_path,
        held_out_path=held_out_path,
    )
    return CommonAdapterValidatedEvidence(
        material=material,
        gate=gate,
        report=report,
    )


def publish_bundle(
    *,
    candidate_path: Path,
    gate_path: Path,
    evaluation_report_path: Path,
    held_out_path: Path,
    output_path: Path,
) -> CommonAdapterBundle:
    evidence = validate_common_adapter_evidence(
        candidate_path=candidate_path,
        gate_path=gate_path,
        evaluation_report_path=evaluation_report_path,
        held_out_path=held_out_path,
    )
    material = evidence.material
    candidate = material.payload
    bundle = CommonAdapterBundle.create(
        common_adapter_version=str(candidate["common_adapter_version"]),
        base_model_id=str(candidate["base_model_id"]),
        base_model_weights_sha256=str(candidate["base_model_weights_sha256"]),
        rare_heavy_checkpoint=material.rare_heavy_checkpoint,
        state_kv_artifact=material.state_kv_artifact,
        control_basis_artifact=material.control_basis_artifact,
        gate_record=evidence.gate,
        description=str(candidate["description"]),
    )
    if bundle.compatibility_fingerprint != candidate.get(
        "compatibility_fingerprint"
    ):
        raise ValueError("published bundle compatibility fingerprint drifted.")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bundle.to_json() + "\n", encoding="utf-8")
    return bundle


def _prefix_parameter_count(artifact: PrefixKVArtifact) -> int:
    return (
        sum(len(row) for row in artifact.encoder_rows)
        + len(artifact.encoder_bias)
        + sum(
            len(row)
            for block in artifact.key_projection
            for row in block
        )
        + sum(len(block) for block in artifact.key_bias)
        + sum(
            len(row)
            for block in artifact.value_projection
            for row in block
        )
        + sum(len(block) for block in artifact.value_bias)
    )


def evaluate_candidate(args: argparse.Namespace) -> tuple[Path, Path, bool]:
    """Run immutable held-out arms and emit an auditable cognition decision."""

    material = load_candidate_material(args.candidate)
    candidate = material.payload
    model_source = _resolve_model_source(
        model_id=str(candidate["base_model_id"]),
        model_source=args.model_source,
        allow_download=args.allow_download,
    )
    actual_weights_sha256 = fingerprint_model_weight_files(model_source)
    if actual_weights_sha256 != candidate["base_model_weights_sha256"]:
        raise ValueError(
            "held-out evaluation model weights do not match the candidate."
        )
    cases = load_held_out_cases(args.held_out.expanduser().resolve())
    control = material.control_basis_artifact
    if any(len(case.applied_control) != control.rank for case in cases):
        raise ValueError(
            "every L1 held-out applied_control must match the control basis rank."
        )
    if not any(any(value != 0.0 for value in case.applied_control) for case in cases):
        raise ValueError(
            "L1 held-out corpus must exercise at least one non-zero z_t control."
        )
    checkpoint = material.rare_heavy_checkpoint
    runtime = TransformersOpenWeightResidualRuntime(
        model_id=str(candidate["base_model_id"]),
        pretrained_source=str(model_source),
        device=args.device,
        layer_indices=tuple(
            layer.layer_index for layer in checkpoint.adapter_layers
        ),
        personal_conditioning_prefix=material.state_kv_artifact,
        local_files_only=True,
        runtime_origin=checkpoint.runtime_origin,
        allow_live_substrate_mutation=True,
    )

    baseline_scores = {
        case.case_id: runtime.score_conditioned_continuation(
            source_text=case.source_text,
            continuation_text=case.continuation_text,
        )
        for case in cases
    }
    runtime.import_rare_heavy_state(checkpoint)
    runtime.install_control_basis(
        basis=control.basis,
        provenance=control.artifact_id,
        layer_indices=control.layer_indices,
        layer_gains=control.layer_gains,
    )

    observations = collect_observations(
        cases=cases,
        baseline_scorer=lambda case: baseline_scores[case.case_id],
        candidate_scorer=lambda case, counterfactual: (
            runtime.score_conditioned_continuation(
                source_text=case.source_text,
                continuation_text=case.continuation_text,
                personal_conditioning=conditioning_snapshot(
                    case,
                    counterfactual=counterfactual,
                ),
                applied_control=case.applied_control,
            )
        ),
    )
    thresholds = AdapterPromotionThresholds(
        min_case_count=args.min_case_count,
        min_mean_relative_improvement=args.min_mean_relative_improvement,
        max_regression_rate=args.max_regression_rate,
        max_preservation_nll_regression=(
            args.max_preservation_nll_regression
        ),
        min_counterfactual_accuracy=args.min_counterfactual_accuracy,
    )
    summary = summarize_observations(
        observations=observations,
        thresholds=thresholds,
    )
    artifact_parameter_count = (
        checkpoint.adapter_parameter_count
        + _prefix_parameter_count(material.state_kv_artifact)
        + sum(len(row) for row in control.basis)
    )
    capacity_cost = artifact_parameter_count / max(
        runtime.model_parameter_count,
        1,
    )
    rollback_evidence = (
        f"remove candidate {candidate['candidate_id']} and restore frozen base "
        f"weights {candidate['base_model_weights_sha256']}"
    )
    decision, gate_reasons, evaluation_snapshot = decide_offline_promotion(
        target="substrate.common_adapter_bundle",
        old_value_hash=str(candidate["base_model_weights_sha256"]),
        new_value_hash=str(candidate["candidate_id"]),
        summary=summary,
        capacity_cost=capacity_cost,
        rollback_evidence=rollback_evidence,
    )
    evidence_id = evaluation_id(
        subject_id=str(candidate["candidate_id"]),
        observations=observations,
        thresholds=thresholds,
    )
    report_path = args.report.expanduser().resolve()
    gate_path = args.gate_record_output.expanduser().resolve()
    decision_value = "allow" if decision is GateDecision.ALLOW else "deny"
    report = {
        "schema_version": ADAPTER_PROMOTION_REPORT_SCHEMA_VERSION,
        "evaluation_id": evidence_id,
        "subject_kind": "common-adapter-candidate",
        "subject_id": candidate["candidate_id"],
        "common_adapter_version": candidate["common_adapter_version"],
        "base_model_id": candidate["base_model_id"],
        "base_model_weights_sha256": candidate["base_model_weights_sha256"],
        "held_out_sha256": _sha256_file(args.held_out.expanduser().resolve()),
        "held_out": True,
        "source_immutable": True,
        "feedback_free": True,
        "thresholds": asdict(thresholds),
        "observations": [asdict(row) for row in observations],
        "summary": summary,
        "artifact_parameter_count": artifact_parameter_count,
        "base_model_parameter_count": runtime.model_parameter_count,
        "capacity_cost": capacity_cost,
        "evaluation_snapshot": asdict(evaluation_snapshot),
        "gate_reasons": list(gate_reasons),
        "decision": decision_value,
        "rollback_evidence": rollback_evidence,
    }
    _write_json(report_path, report)
    gate_record = CommonAdapterGateRecord(
        proposal_id=f"common-adapter:{candidate['candidate_id']}",
        decision=decision_value,
        desired_gate="offline",
        validation_delta=float(summary["validation_delta"]),
        capacity_cost=capacity_cost,
        rollback_evidence=rollback_evidence,
        is_reversible=True,
        evaluation_ref=(
            f"common-adapter-evaluation:{evidence_id}:"
            f"sha256:{_sha256_file(report_path)}"
        ),
    )
    _write_json(gate_path, asdict(gate_record))
    return report_path, gate_path, decision is GateDecision.ALLOW


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train an immutable candidate")
    train.add_argument("--model-id", required=True)
    train.add_argument("--model-source", default="")
    train.add_argument("--allow-download", action="store_true")
    train.add_argument("--common-adapter-version", required=True)
    train.add_argument("--traces", type=Path, required=True)
    train.add_argument("--control-basis", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument(
        "--runtime-origin", choices=("hf-local", "hf-pretrained"), default="hf-pretrained"
    )
    train.add_argument(
        "--target-modules", nargs="+", default=("q_proj", "v_proj", "o_proj")
    )
    train.add_argument("--hook-layers", default="")
    train.add_argument("--lora-rank", type=int, default=8)
    train.add_argument("--lora-alpha", type=int, default=16)
    train.add_argument("--lora-dropout", type=float, default=0.0)
    train.add_argument("--learning-rate", type=float, default=5e-4)
    train.add_argument("--max-steps", type=int, default=200)
    train.add_argument("--seed", type=int, default=20260801)
    train.add_argument("--control-scale", type=float, default=0.12)
    train.add_argument("--state-kv-states", type=int, default=16)
    train.add_argument("--state-kv-epochs", type=int, default=4)
    train.add_argument("--state-kv-slots", type=int, default=4)
    train.add_argument("--state-kv-rank", type=int, default=4)
    train.add_argument("--state-kv-norm-cap", type=float, default=0.2)
    train.add_argument("--state-kv-learning-rate", type=float, default=0.05)
    train.add_argument("--state-kv-seed", type=int, default=20260726)
    train.add_argument(
        "--description",
        default="Shared common adapter: PEFT rare-heavy then State-KV distillation.",
    )

    evaluate = commands.add_parser(
        "evaluate",
        help="run held-out base/candidate arms and emit an OFFLINE gate record",
    )
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--model-source", default="")
    evaluate.add_argument("--allow-download", action="store_true")
    evaluate.add_argument("--held-out", type=Path, required=True)
    evaluate.add_argument("--report", type=Path, required=True)
    evaluate.add_argument("--gate-record-output", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--min-case-count", type=int, default=8)
    evaluate.add_argument(
        "--min-mean-relative-improvement",
        type=float,
        default=0.01,
    )
    evaluate.add_argument("--max-regression-rate", type=float, default=0.25)
    evaluate.add_argument(
        "--max-preservation-nll-regression",
        type=float,
        default=0.05,
    )
    evaluate.add_argument(
        "--min-counterfactual-accuracy",
        type=float,
        default=0.60,
    )

    publish = commands.add_parser(
        "publish", help="bind a gate record and export a runtime bundle"
    )
    publish.add_argument("--candidate", type=Path, required=True)
    publish.add_argument("--gate-record", type=Path, required=True)
    publish.add_argument("--evaluation-report", type=Path, required=True)
    publish.add_argument("--held-out", type=Path, required=True)
    publish.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        candidate, proposal = train_candidate(args)
        print(f"candidate: {candidate}")
        print(f"gate proposal: {proposal}")
        return 0
    if args.command == "evaluate":
        report, gate, allowed = evaluate_candidate(args)
        print(f"evaluation report: {report}")
        print(f"gate record: {gate}")
        print(f"decision: {'allow' if allowed else 'deny'}")
        return 0 if allowed else 2
    bundle = publish_bundle(
        candidate_path=args.candidate,
        gate_path=args.gate_record,
        evaluation_report_path=args.evaluation_report,
        held_out_path=args.held_out,
        output_path=args.output,
    )
    print(f"bundle_id={bundle.bundle_id}")
    print(f"written: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
