#!/usr/bin/env python3
"""Bind a Forge rare-heavy request to Common Adapter promotion evidence.

This verifier is deliberately outside the Forge proposal loop.  READY means
the immutable candidate and OFFLINE ALLOW evidence match the request; it does
not publish a bundle and cannot change runtime wiring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FORGE_SRC = REPO_ROOT / "forge" / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts"
for _source_root in (SCRIPT_ROOT, FORGE_SRC):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

import train_common_adapter_model as pipeline  # noqa: E402
from volvence_forge.config import ForgeConfig, ForgePaths  # noqa: E402
from volvence_forge.foundation import (  # noqa: E402
    ForgeError,
    SchemaStore,
    atomic_write_json,
    sha256_bytes,
)
from volvence_forge.rare_heavy import validate_rare_heavy_request  # noqa: E402


def adjudicate_rare_heavy_request(
    *,
    config: ForgeConfig,
    request_path: Path,
    candidate_path: Path,
    evaluation_report_path: Path,
    gate_path: Path,
    held_out_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Emit READY only for a fully bound cognition-approved evidence chain."""

    request_path = request_path.expanduser().resolve()
    candidate_path = candidate_path.expanduser().resolve()
    evaluation_report_path = evaluation_report_path.expanduser().resolve()
    gate_path = gate_path.expanduser().resolve()
    held_out_path = held_out_path.expanduser().resolve()
    request = validate_rare_heavy_request(
        config=config,
        request_path=request_path,
    )
    evidence_hashes: dict[str, str | None] = {
        "request_sha256": _file_sha(request_path),
        "candidate_sha256": _optional_file_sha(candidate_path),
        "evaluation_sha256": _optional_file_sha(evaluation_report_path),
        "gate_sha256": _optional_file_sha(gate_path),
        "held_out_sha256": _optional_file_sha(held_out_path),
    }
    bindings = {
        "candidate": False,
        "training": False,
        "held_out": False,
        "evaluation": False,
        "gate": False,
    }
    reasons: list[str] = []
    candidate_id: str | None = None
    gate_decision = "unavailable"
    try:
        evidence = pipeline.validate_common_adapter_evidence(
            candidate_path=candidate_path,
            gate_path=gate_path,
            evaluation_report_path=evaluation_report_path,
            held_out_path=held_out_path,
        )
    except (
        FileNotFoundError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        reasons.append(f"Common Adapter evidence validation failed: {exc}")
    else:
        candidate = evidence.material.payload
        candidate_id = str(candidate["candidate_id"])
        gate_decision = evidence.gate.decision
        bindings["candidate"] = _candidate_matches_request(
            request=request,
            candidate=candidate,
            control_sha256=_file_sha(evidence.material.control_path),
            reasons=reasons,
        )
        bindings["training"] = _training_matches_request(
            request=request,
            evidence=evidence,
            reasons=reasons,
        )
        request_inputs = _mapping(request["inputs"], name="request.inputs")
        held_out_ref = _mapping(
            request_inputs["held_out"],
            name="request.inputs.held_out",
        )
        bindings["held_out"] = (
            held_out_ref["sha256"] == evidence_hashes["held_out_sha256"]
            and held_out_ref["sha256"] == evidence.report["held_out_sha256"]
        )
        if not bindings["held_out"]:
            reasons.append("held-out digest does not bind request, report, and corpus")
        request_evaluation = _mapping(
            request["evaluation"],
            name="request.evaluation",
        )
        bindings["evaluation"] = evidence.report["thresholds"] == request_evaluation
        if not bindings["evaluation"]:
            reasons.append("evaluation thresholds drifted from the Forge request")
        bindings["gate"] = evidence.gate.allows_active
        if not bindings["gate"]:
            reasons.append(
                "cognition OFFLINE gate did not provide a reversible ALLOW"
            )

    decision = "READY" if all(bindings.values()) else "STOP"
    verdict = {
        "schema_version": "forge-rare-heavy-verdict.v1",
        "request_id": request["request_id"],
        "decision": decision,
        "candidate_id": candidate_id,
        "gate_decision": gate_decision,
        "bindings": bindings,
        "reasons": reasons,
        "evidence": evidence_hashes,
    }
    SchemaStore(config.paths.forge_root / "schemas").validate(
        verdict,
        "rare_heavy_verdict.schema.json",
    )
    destination = output_path.expanduser().resolve()
    if not destination.is_relative_to(config.paths.artifacts_root):
        raise ForgeError("rare-heavy verdicts may only be written below artifacts/")
    atomic_write_json(destination, verdict)
    return verdict


def _candidate_matches_request(
    *,
    request: dict[str, object],
    candidate: dict[str, Any],
    control_sha256: str,
    reasons: list[str],
) -> bool:
    base = _mapping(request["base_model"], name="request.base_model")
    inputs = _mapping(request["inputs"], name="request.inputs")
    control_ref = _mapping(inputs["control_basis"], name="request.inputs.control_basis")
    training = _mapping(request["training"], name="request.training")
    matches = (
        candidate["base_model_id"] == base["model_id"]
        and candidate["base_model_weights_sha256"] == base["weights_sha256"]
        and candidate["common_adapter_version"] == training["common_adapter_version"]
        and candidate["description"] == training["description"]
        and candidate["training_order"] == request["training_order"]
        and control_sha256 == control_ref["sha256"]
    )
    if not matches:
        reasons.append("candidate identity, order, description, or control basis drifted")
    return matches


def _training_matches_request(
    *,
    request: dict[str, object],
    evidence: pipeline.CommonAdapterValidatedEvidence,
    reasons: list[str],
) -> bool:
    inputs = _mapping(request["inputs"], name="request.inputs")
    traces_ref = _mapping(inputs["traces"], name="request.inputs.traces")
    training = _mapping(request["training"], name="request.training")
    provenance = evidence.material.payload["training_provenance"]
    expected = {
        name: training[name]
        for name in (
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
    }
    matches = (
        all(provenance[name] == value for name, value in expected.items())
        and provenance["traces_sha256"] == traces_ref["sha256"]
        and provenance["trace_count"] == traces_ref["trace_count"]
        and evidence.material.rare_heavy_checkpoint.runtime_origin
        == training["runtime_origin"]
        and evidence.material.rare_heavy_checkpoint.control_scale
        == training["control_scale"]
    )
    if not matches:
        reasons.append("candidate training provenance drifted from the Forge request")
    return matches


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ForgeError(f"{name} must be an object")
    return value


def _file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _optional_file_sha(path: Path) -> str | None:
    try:
        return _file_sha(path)
    except OSError:
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--gate-record", type=Path, required=True)
    parser.add_argument("--held-out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = ForgePaths.discover(repo_root=args.repo_root)
        config = ForgeConfig.load(paths)
        verdict = adjudicate_rare_heavy_request(
            config=config,
            request_path=args.request,
            candidate_path=args.candidate,
            evaluation_report_path=args.evaluation_report,
            gate_path=args.gate_record,
            held_out_path=args.held_out,
            output_path=args.output,
        )
    except (ForgeError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"forge-common-adapter-adjudicator: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(verdict, ensure_ascii=False, sort_keys=True))
    return 0 if verdict["decision"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
