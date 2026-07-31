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
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
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

COMMON_ADAPTER_CANDIDATE_SCHEMA = "common-adapter-candidate.v1"
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


def publish_bundle(
    *, candidate_path: Path, gate_path: Path, output_path: Path
) -> CommonAdapterBundle:
    candidate_path = candidate_path.expanduser().resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(candidate, dict):
        raise ValueError("common adapter candidate must be an object.")
    if candidate.get("schema_version") != COMMON_ADAPTER_CANDIDATE_SCHEMA:
        raise ValueError("unsupported common adapter candidate schema.")
    if candidate.get("candidate_id") != _candidate_id(candidate):
        raise ValueError("common adapter candidate_id does not match payload.")
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
        state_manifest_sha, str
    ):
        raise ValueError("candidate State-KV training manifest reference is missing.")
    state_manifest_path = Path(state_manifest_locator)
    if not state_manifest_path.is_absolute():
        state_manifest_path = candidate_path.parent / state_manifest_path
    state_manifest_path = state_manifest_path.resolve()
    if _sha256_file(state_manifest_path) != state_manifest_sha:
        raise ValueError("candidate State-KV training manifest digest mismatch.")
    state_manifest = json.loads(state_manifest_path.read_text(encoding="utf-8"))
    if state_manifest.get("training_order") != "base+rare-heavy->state-kv":
        raise ValueError("candidate State-KV was not trained after rare-heavy.")
    if state_manifest.get("common_adapter_version") != candidate.get(
        "common_adapter_version"
    ):
        raise ValueError("candidate State-KV common adapter version drifted.")

    gate_raw = json.loads(gate_path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(gate_raw, dict):
        raise ValueError("common adapter gate record must be an object.")
    gate = CommonAdapterGateRecord(**gate_raw)
    expected_proposal = f"common-adapter:{candidate['candidate_id']}"
    if gate.proposal_id != expected_proposal:
        raise ValueError(
            "gate proposal_id does not bind this common adapter candidate."
        )
    bundle = CommonAdapterBundle.create(
        common_adapter_version=str(candidate["common_adapter_version"]),
        base_model_id=str(candidate["base_model_id"]),
        base_model_weights_sha256=str(candidate["base_model_weights_sha256"]),
        rare_heavy_checkpoint=rare_heavy_checkpoint_from_json(
            checkpoint_path.read_text(encoding="utf-8")
        ),
        state_kv_artifact=PrefixKVArtifact.from_json(
            state_path.read_text(encoding="utf-8")
        ),
        control_basis_artifact=ControlBasisArtifact.from_json(
            control_path.read_text(encoding="utf-8")
        ),
        gate_record=gate,
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
    train.add_argument("--control-scale", type=float, default=0.12)
    train.add_argument("--state-kv-states", type=int, default=16)
    train.add_argument("--state-kv-epochs", type=int, default=4)
    train.add_argument("--state-kv-slots", type=int, default=4)
    train.add_argument("--state-kv-rank", type=int, default=4)
    train.add_argument("--state-kv-norm-cap", type=float, default=0.2)
    train.add_argument("--state-kv-learning-rate", type=float, default=0.05)
    train.add_argument(
        "--description",
        default="Shared common adapter: PEFT rare-heavy then State-KV distillation.",
    )

    publish = commands.add_parser(
        "publish", help="bind a gate record and export a runtime bundle"
    )
    publish.add_argument("--candidate", type=Path, required=True)
    publish.add_argument("--gate-record", type=Path, required=True)
    publish.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        candidate, proposal = train_candidate(args)
        print(f"candidate: {candidate}")
        print(f"gate proposal: {proposal}")
        return 0
    bundle = publish_bundle(
        candidate_path=args.candidate,
        gate_path=args.gate_record,
        output_path=args.output,
    )
    print(f"bundle_id={bundle.bundle_id}")
    print(f"written: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
