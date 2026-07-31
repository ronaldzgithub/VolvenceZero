#!/usr/bin/env python3
"""Evaluate one SHADOW character package and emit fidelity/gate artifacts.

The baseline arm is the admitted Common Adapter with Personal State-KV.  The
candidate arm adds the character Prefix/KV and, when declared by the manifest,
the real PEFT Character LoRA.  Both arms score the same immutable held-out
continuations; no generated output or evaluation result is fed back to an
owner, memory, or training loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (Path(__file__).resolve().parent,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
for _src in sorted((REPO_ROOT / "packages").glob("*/src")):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from adapter_promotion_evidence import (  # noqa: E402
    ADAPTER_PROMOTION_REPORT_SCHEMA_VERSION,
    AdapterPromotionThresholds,
    collect_observations,
    conditioning_snapshot,
    decide_offline_promotion,
    evaluation_id,
    load_held_out_cases,
    summarize_observations,
)
from lifeform_domain_character import (  # noqa: E402
    CharacterArtifactRef,
    CharacterFidelityEvidence,
    CharacterPackageGateRecord,
    CharacterPackageManifest,
    sha256_path,
    verify_manifest_artifacts,
)
from volvence_zero.credit import GateDecision  # noqa: E402
from volvence_zero.substrate import (  # noqa: E402
    CharacterPrefixKVPackage,
    CommonAdapterBundle,
    PrefixKVArtifact,
    TransformersOpenWeightResidualRuntime,
    fingerprint_model_weight_files,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def _relative_locator(path: Path, *, manifest_path: Path) -> str:
    try:
        return path.resolve().relative_to(
            manifest_path.parent.resolve()
        ).as_posix()
    except ValueError:
        return str(path.resolve())


def evaluate_character_package(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, bool]:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = CharacterPackageManifest.from_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.fidelity_evidence is not None or manifest.gate_record is not None:
        raise ValueError(
            "character evaluation requires a SHADOW candidate manifest with "
            "no prior fidelity evidence or gate record."
        )
    template_path, prefix_path, lora_path, _ = verify_manifest_artifacts(
        manifest,
        manifest_path=manifest_path,
    )
    if prefix_path is None or manifest.prefix_kv_ref is None:
        raise ValueError("character evaluation requires a Prefix/KV artifact.")
    prefix_package = CharacterPrefixKVPackage.from_json(
        prefix_path.read_text(encoding="utf-8")
    )
    if prefix_package.package_id != manifest.prefix_kv_ref.artifact_id:
        raise ValueError(
            "character Prefix/KV package_id does not match the manifest ref."
        )
    if prefix_package.character_id != manifest.character_id:
        raise ValueError(
            "character Prefix/KV character_id does not match the manifest."
        )

    evaluated_manifest_path = (
        args.evaluated_manifest_output.expanduser().resolve()
    )
    evaluated_candidate = CharacterPackageManifest.create(
        character_id=manifest.character_id,
        character_name=manifest.character_name,
        base_model_id=manifest.base_model_id,
        common_adapter_version=manifest.common_adapter_version,
        compatibility_fingerprint=manifest.compatibility_fingerprint,
        template_ref=replace(
            manifest.template_ref,
            locator=_relative_locator(
                template_path,
                manifest_path=evaluated_manifest_path,
            ),
        ),
        prefix_kv_ref=replace(
            manifest.prefix_kv_ref,
            locator=_relative_locator(
                prefix_path,
                manifest_path=evaluated_manifest_path,
            ),
        ),
        lora_ref=(
            replace(
                manifest.lora_ref,
                locator=_relative_locator(
                    lora_path,
                    manifest_path=evaluated_manifest_path,
                ),
            )
            if manifest.lora_ref is not None and lora_path is not None
            else None
        ),
        revalidation_mode=manifest.revalidation_mode,
        description=manifest.description,
    )

    common_path = args.common_adapter_bundle.expanduser().resolve()
    common = CommonAdapterBundle.from_json(
        common_path.read_text(encoding="utf-8")
    )
    common.require_active()
    manifest.assert_common_adapter(
        base_model_id=common.base_model_id,
        common_adapter_version=common.common_adapter_version,
        compatibility_fingerprint=common.compatibility_fingerprint,
    )
    model_source = args.model_source.expanduser().resolve()
    if not model_source.is_dir():
        raise FileNotFoundError(model_source)
    actual_weights_sha256 = fingerprint_model_weight_files(model_source)
    if actual_weights_sha256 != common.base_model_weights_sha256:
        raise ValueError(
            "character held-out model weights do not match the Common Adapter."
        )
    cases = load_held_out_cases(args.held_out.expanduser().resolve())
    control_rank = common.control_basis_artifact.rank
    if any(len(case.applied_control) != control_rank for case in cases):
        raise ValueError(
            "every L2 held-out applied_control must match the control basis rank."
        )
    if not any(any(value != 0.0 for value in case.applied_control) for case in cases):
        raise ValueError(
            "L2 held-out corpus must exercise at least one non-zero z_t control."
        )

    runtime = TransformersOpenWeightResidualRuntime(
        model_id=common.base_model_id,
        pretrained_source=str(model_source),
        device=args.device,
        layer_indices=tuple(
            layer.layer_index
            for layer in common.rare_heavy_checkpoint.adapter_layers
        ),
        common_adapter_bundle=common,
        loaded_base_model_weights_sha256=actual_weights_sha256,
        character_prefix_package=prefix_package,
        local_files_only=True,
        runtime_origin=common.rare_heavy_checkpoint.runtime_origin,
    )
    baseline_scores = {
        case.case_id: runtime.score_conditioned_continuation(
            source_text=case.source_text,
            continuation_text=case.continuation_text,
            personal_conditioning=conditioning_snapshot(case),
            applied_control=case.applied_control,
        )
        for case in cases
    }
    lora_context = (
        runtime.activate_peft_adapter(lora_path)
        if lora_path is not None
        else nullcontext()
    )
    with lora_context:
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
                    character_id=manifest.character_id,
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
    artifact_parameter_count = _prefix_parameter_count(
        prefix_package.prefix_artifact
    ) + (manifest.lora_ref.parameter_count if manifest.lora_ref is not None else 0)
    capacity_cost = artifact_parameter_count / max(
        runtime.model_parameter_count,
        1,
    )
    rollback_evidence = (
        f"disable character candidate {evaluated_candidate.package_id} and restore "
        f"common-only bundle {common.bundle_id}"
    )
    decision, gate_reasons, evaluation_snapshot = decide_offline_promotion(
        target=f"substrate.character_package.{manifest.character_id}",
        old_value_hash=common.bundle_id,
        new_value_hash=evaluated_candidate.package_id,
        summary=summary,
        capacity_cost=capacity_cost,
        rollback_evidence=rollback_evidence,
    )
    evidence_id = evaluation_id(
        subject_id=evaluated_candidate.package_id,
        observations=observations,
        thresholds=thresholds,
    )
    report_path = args.report.expanduser().resolve()
    report = {
        "schema_version": ADAPTER_PROMOTION_REPORT_SCHEMA_VERSION,
        "evaluation_id": evidence_id,
        "subject_kind": "character-package-candidate",
        "subject_id": evaluated_candidate.package_id,
        "source_manifest_id": manifest.package_id,
        "character_id": manifest.character_id,
        "common_adapter_bundle_id": common.bundle_id,
        "common_adapter_version": common.common_adapter_version,
        "compatibility_fingerprint": common.compatibility_fingerprint,
        "held_out_sha256": _sha256_file(args.held_out.expanduser().resolve()),
        "held_out": True,
        "source_immutable": True,
        "feedback_free": True,
        "includes_character_lora": manifest.lora_ref is not None,
        "thresholds": asdict(thresholds),
        "observations": [asdict(row) for row in observations],
        "summary": summary,
        "artifact_parameter_count": artifact_parameter_count,
        "base_model_parameter_count": runtime.model_parameter_count,
        "capacity_cost": capacity_cost,
        "evaluation_snapshot": asdict(evaluation_snapshot),
        "gate_reasons": list(gate_reasons),
        "decision": "allow" if decision is GateDecision.ALLOW else "deny",
    }
    _write_json(report_path, report)
    report_sha256 = sha256_path(report_path)
    fidelity = CharacterFidelityEvidence(
        report_ref=CharacterArtifactRef(
            locator=_relative_locator(
                report_path,
                manifest_path=evaluated_manifest_path,
            ),
            sha256=report_sha256,
            artifact_id=f"character-fidelity-report:{evidence_id}",
            media_type="application/vnd.volvence.adapter-promotion-evidence+json",
        ),
        evidence_source="system_self_eval",
        verdict=("pass" if decision is GateDecision.ALLOW else "blocked"),
        passed=decision is GateDecision.ALLOW,
        held_out=True,
        source_immutable=True,
        feedback_free=True,
        includes_character_lora=manifest.lora_ref is not None,
        common_adapter_version=common.common_adapter_version,
        compatibility_fingerprint=common.compatibility_fingerprint,
    )
    gate = CharacterPackageGateRecord(
        proposal_id=(
            "character-package:"
            f"{evaluated_candidate.package_id}:{evidence_id}"
        ),
        decision=("allow" if decision is GateDecision.ALLOW else "deny"),
        desired_gate="offline",
        fidelity_report_sha256=report_sha256,
        rollback_evidence=rollback_evidence,
        is_reversible=True,
        common_adapter_version=common.common_adapter_version,
        compatibility_fingerprint=common.compatibility_fingerprint,
    )
    evidence_path = args.fidelity_evidence_output.expanduser().resolve()
    gate_path = args.gate_record_output.expanduser().resolve()
    _write_json(evidence_path, asdict(fidelity))
    _write_json(gate_path, asdict(gate))
    evaluated_manifest = CharacterPackageManifest.create(
        character_id=evaluated_candidate.character_id,
        character_name=evaluated_candidate.character_name,
        base_model_id=evaluated_candidate.base_model_id,
        common_adapter_version=evaluated_candidate.common_adapter_version,
        compatibility_fingerprint=(
            evaluated_candidate.compatibility_fingerprint
        ),
        template_ref=evaluated_candidate.template_ref,
        prefix_kv_ref=evaluated_candidate.prefix_kv_ref,
        lora_ref=evaluated_candidate.lora_ref,
        fidelity_evidence=fidelity,
        gate_record=gate,
        revalidation_mode=evaluated_candidate.revalidation_mode,
        description=evaluated_candidate.description,
    )
    if decision is GateDecision.ALLOW:
        evaluated_manifest.require_active()
    evaluated_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    evaluated_manifest_path.write_text(
        evaluated_manifest.to_json() + "\n",
        encoding="utf-8",
    )
    verify_manifest_artifacts(
        evaluated_manifest,
        manifest_path=evaluated_manifest_path,
    )
    return (
        report_path,
        evidence_path,
        gate_path,
        evaluated_manifest_path,
        decision is GateDecision.ALLOW,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--common-adapter-bundle", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--held-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fidelity-evidence-output", type=Path, required=True)
    parser.add_argument("--gate-record-output", type=Path, required=True)
    parser.add_argument("--evaluated-manifest-output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-case-count", type=int, default=8)
    parser.add_argument(
        "--min-mean-relative-improvement",
        type=float,
        default=0.01,
    )
    parser.add_argument("--max-regression-rate", type=float, default=0.25)
    parser.add_argument(
        "--max-preservation-nll-regression",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--min-counterfactual-accuracy",
        type=float,
        default=0.60,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    report, evidence, gate, manifest, allowed = evaluate_character_package(
        _parser().parse_args(argv)
    )
    print(f"evaluation report: {report}")
    print(f"fidelity evidence: {evidence}")
    print(f"gate record: {gate}")
    print(f"evaluated manifest: {manifest}")
    print(f"decision: {'allow' if allowed else 'deny'}")
    return 0 if allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
