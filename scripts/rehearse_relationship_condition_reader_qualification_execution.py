#!/usr/bin/env python3
"""Rehearse the exact reader-qualification execution path without any anchor.

Development-tier rehearsal driver required by
``docs/specs/four-able-mainline-execution-plan.md`` §3.1: before a formal
protocol is frozen and publicly anchored, the identical execution code path
(fresh Windows Job-Object children, pinned BGE-M3 CUDA embedding, create-only
ledger with fsync, model-free scorer process, chained integrity receipts)
must complete once against a disposable rehearsal root.

Fidelity contract: this driver mirrors the v2 authorized wrapper
(``execute_authorized_relationship_condition_reader_qualification_execution_v2``)
and the stage orchestration of ``_execute_authorized_qualification_with_stages``
exactly — same v2 protocol validation, same frozen-executable equality gate,
same v2 integrity-guard factory, same real prediction and scoring stages and
phase order — with two deliberate differences only:

1. no public-anchor receipt exists and no anchor validator is invoked;
2. the terminal artifact is a rehearsal manifest whose authorization fields
   are all false, instead of the authorized execution manifest.

Honesty boundaries: outputs are ``rehearsal_only`` development evidence and
are never qualification evidence.  Because the frozen reader and BGE-M3 are
deterministic, the rehearsal necessarily reveals the scorer verdict early;
the pre-committed handling is that the formal anchored execution proceeds
regardless of the rehearsal outcome and the rehearsal outcome is sealed
as-is.  The rehearsal execution root must contain the token ``rehearsal`` and
must differ from any formal proposed execution root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pathlib
import sys
from typing import Mapping


sys.dont_write_bytecode = True

_SCRIPT_PATH = pathlib.Path(os.path.abspath(__file__))
_REPOSITORY_ROOT = _SCRIPT_PATH.parents[1]
_REHEARSAL_ROOT_TOKEN = "rehearsal"
_REHEARSAL_MANIFEST_SCHEMA_VERSION = "relationship-condition-reader-qualification-rehearsal-manifest.v1"


def _install_workspace_source_roots() -> None:
    packages_root = _REPOSITORY_ROOT / "packages"
    if not packages_root.is_dir():
        raise FileNotFoundError(f"workspace packages root is absent: {packages_root}")
    source_roots = sorted(
        (entry / "src" for entry in packages_root.iterdir() if (entry / "src").is_dir()),
        key=lambda path: path.as_posix(),
    )
    if not source_roots:
        raise FileNotFoundError("workspace packages/*/src roots are absent")
    existing = {os.path.normcase(os.path.abspath(p)) for p in sys.path if isinstance(p, str) and p}
    for root in reversed(source_roots):
        key = os.path.normcase(os.path.abspath(root))
        if key not in existing:
            sys.path.insert(0, str(root))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rehearse the exact reader-qualification execution path against a "
            "disposable rehearsal root; produces development-tier evidence only"
        )
    )
    parser.add_argument("--execution-protocol-path", type=pathlib.Path, required=True)
    parser.add_argument("--expected-execution-protocol-id", required=True)
    parser.add_argument("--preflight-root", type=pathlib.Path, required=True)
    parser.add_argument("--bge-snapshot-root", type=pathlib.Path, required=True)
    parser.add_argument("--run-nonce", required=True)
    parser.add_argument("--python-executable", type=pathlib.Path)
    parser.add_argument("--prediction-timeout-seconds", type=int, default=7200)
    parser.add_argument("--scorer-timeout-seconds", type=int, default=600)
    return parser


def _rehearse(args: argparse.Namespace) -> Mapping[str, object]:
    protocol_v2_module = importlib.import_module(
        "lifeform_evolution.relationship_condition_reader_qualification_execution_protocol_v2"
    )
    execution_module = importlib.import_module(
        "lifeform_evolution.relationship_condition_reader_qualification_execution"
    )
    executor_module = importlib.import_module(
        "lifeform_evolution.relationship_condition_reader_qualification_executor"
    )
    execution_v2_module = importlib.import_module(
        "lifeform_evolution.relationship_condition_reader_qualification_execution_v2"
    )

    protocol_path = pathlib.Path(os.path.abspath(args.execution_protocol_path))
    protocol, protocol_raw = protocol_v2_module.load_relationship_condition_reader_qualification_execution_protocol_v2(
        protocol_path,
        expected_protocol_id=args.expected_execution_protocol_id,
    )
    protocol_id = protocol_v2_module.validate_relationship_condition_reader_qualification_execution_protocol_v2(
        protocol,
        expected_protocol_id=args.expected_execution_protocol_id,
    )
    if protocol_id != args.expected_execution_protocol_id:
        raise ValueError("v2 execution protocol validator returned an unexpected protocol id")
    nonce = execution_module._digest(args.run_nonce, "run_nonce")

    frozen = execution_module._extract_frozen_execution_identities(protocol)
    root = pathlib.Path(str(frozen["execution_root"])).resolve()
    if _REHEARSAL_ROOT_TOKEN not in root.name.lower():
        raise ValueError(
            "rehearsal driver refuses a protocol whose proposed execution root "
            f"does not contain the token {_REHEARSAL_ROOT_TOKEN!r}: {root}"
        )
    if root.exists():
        raise FileExistsError(f"rehearsal execution root exists: {root}")

    repository_root = _REPOSITORY_ROOT.resolve()
    (
        repository_source_roots,
        frozen_source_entries,
        frozen_site_packages_root,
    ) = execution_module._extract_child_import_inputs(
        protocol,
        repository_root=repository_root,
    )
    expected_integrity_ids = execution_module._mapping(
        frozen["expected_integrity_artifact_ids"],
        "expected integrity artifact ids",
    )
    preflight_root = pathlib.Path(os.path.abspath(args.preflight_root))
    bge_snapshot_root = pathlib.Path(os.path.abspath(args.bge_snapshot_root))
    frozen_executable = execution_v2_module._frozen_python_executable(protocol)
    python_executable = pathlib.Path(args.python_executable or sys.executable).resolve()
    if str(python_executable) != str(frozen_executable):
        raise ValueError("rehearsal Python differs from the frozen runtime identity executable")

    guard = protocol_v2_module.relationship_condition_reader_qualification_integrity_guard_v2(
        execution_protocol=protocol,
        expected_execution_protocol_id=protocol_id,
        repository_root=repository_root,
        bge_snapshot_root=bge_snapshot_root,
    )
    initial_integrity_receipt = execution_module._run_integrity_guard(
        guard,
        phase="post_anchor_pre_execution",
        execution_protocol_id=protocol_id,
        expected_integrity_ids={
            key: execution_module._digest(value, key) for key, value in expected_integrity_ids.items()
        },
        previous_integrity_receipt_artifact_id=None,
    )
    if root.exists():
        raise FileExistsError("rehearsal execution root appeared before prediction stage")

    prediction_result = execution_module._mapping(
        executor_module.execute_relationship_condition_reader_qualification_prediction_stage(
            preflight_root=preflight_root,
            execution_root=root,
            expected_qualification_protocol_id=frozen["qualification_protocol_id"],
            expected_preflight_manifest_artifact_id=frozen["preflight_manifest_artifact_id"],
            expected_publication_request_artifact_id=frozen["publication_request_artifact_id"],
            execution_protocol_id=protocol_id,
            run_nonce=nonce,
            integrity_guard=guard,
            previous_integrity_receipt_artifact_id=initial_integrity_receipt["artifact_id"],
            expected_source_tree_artifact_id=expected_integrity_ids["source_tree_artifact_id"],
            expected_bge_snapshot_tree_artifact_id=expected_integrity_ids["bge_snapshot_tree_artifact_id"],
            expected_runtime_identity_artifact_id=expected_integrity_ids["runtime_identity_artifact_id"],
            bge_snapshot_path=bge_snapshot_root,
            python_executable=python_executable,
            repository_root=repository_root,
            repository_source_roots=repository_source_roots,
            frozen_source_entries=frozen_source_entries,
            frozen_site_packages_root=frozen_site_packages_root,
            child_timeout_seconds=args.prediction_timeout_seconds,
        ),
        "prediction stage result",
    )
    launcher_attestation = execution_module._validate_prediction_stage_for_outer_runner(
        result=prediction_result,
        execution_root=root,
        protocol_id=protocol_id,
        frozen=frozen,
        initial_integrity_receipt=initial_integrity_receipt,
        expected_integrity_ids=expected_integrity_ids,
        run_nonce=nonce,
    )

    scoring_stage_root = root / "scoring_stage"
    scoring_result = execution_module._mapping(
        execution_module.execute_relationship_condition_reader_qualification_scoring_stage(
            scoring_request_path=pathlib.Path(
                execution_module._text(
                    prediction_result["scoring_request_path"],
                    "prediction scoring_request_path",
                )
            ),
            expected_scoring_request_artifact_id=prediction_result["scoring_request_artifact_id"],
            stage_root=scoring_stage_root,
            integrity_guard=guard,
            previous_integrity_receipt_artifact_id=prediction_result["last_integrity_receipt_artifact_id"],
            expected_source_tree_artifact_id=expected_integrity_ids["source_tree_artifact_id"],
            expected_bge_snapshot_tree_artifact_id=expected_integrity_ids["bge_snapshot_tree_artifact_id"],
            expected_runtime_identity_artifact_id=expected_integrity_ids["runtime_identity_artifact_id"],
            python_executable=python_executable,
            repository_root=repository_root,
            repository_source_roots=repository_source_roots,
            frozen_source_entries=frozen_source_entries,
            frozen_site_packages_root=frozen_site_packages_root,
            scorer_timeout_seconds=args.scorer_timeout_seconds,
        ),
        "scoring stage result",
    )
    scoring_manifest, scorer_report = execution_module._validate_scoring_stage_for_outer_runner(
        result=scoring_result,
        stage_root=scoring_stage_root,
        protocol_id=protocol_id,
        expected_scoring_request_artifact_id=execution_module._digest(
            prediction_result["scoring_request_artifact_id"],
            "scoring request artifact_id",
        ),
        previous_integrity_receipt_artifact_id=execution_module._digest(
            prediction_result["last_integrity_receipt_artifact_id"],
            "prediction last integrity receipt artifact_id",
        ),
    )
    final_integrity_receipt = execution_module._run_integrity_guard(
        guard,
        phase="final_validation",
        execution_protocol_id=protocol_id,
        expected_integrity_ids={
            key: execution_module._digest(value, key) for key, value in expected_integrity_ids.items()
        },
        previous_integrity_receipt_artifact_id=execution_module._digest(
            scoring_manifest["last_integrity_receipt_artifact_id"],
            "scoring last integrity receipt artifact_id",
        ),
    )
    all_integrity_receipts = execution_module._collect_and_validate_full_integrity_chain(
        initial=initial_integrity_receipt,
        launcher_attestation=launcher_attestation,
        scoring_manifest=scoring_manifest,
        final=final_integrity_receipt,
        execution_protocol_id=protocol_id,
        expected_integrity_ids=expected_integrity_ids,
    )

    driver_raw = _SCRIPT_PATH.read_bytes()
    rehearsal_manifest = execution_module._with_artifact_id(
        {
            "schema_version": _REHEARSAL_MANIFEST_SCHEMA_VERSION,
            "rehearsal_only": True,
            "evidence_tier": "development",
            "execution_protocol_id": protocol_id,
            "execution_protocol_raw_sha256": hashlib.sha256(protocol_raw).hexdigest(),
            "execution_protocol_raw_bytes": len(protocol_raw),
            "qualification_protocol_id": frozen["qualification_protocol_id"],
            "preflight_binding_artifact_id": frozen["preflight_binding_artifact_id"],
            "preflight_manifest_artifact_id": frozen["preflight_manifest_artifact_id"],
            "publication_request_artifact_id": frozen["publication_request_artifact_id"],
            "execution_root": str(root),
            "run_nonce": nonce,
            "expected_integrity_artifact_ids": dict(expected_integrity_ids),
            "integrity_receipts": all_integrity_receipts,
            "integrity_receipt_count": 8,
            "last_integrity_receipt_artifact_id": final_integrity_receipt["artifact_id"],
            "prediction_stage_result": dict(prediction_result),
            "prediction_launcher_attestation": dict(launcher_attestation),
            "scoring_stage_manifest": dict(scoring_manifest),
            "rehearsal_scorer_report": dict(scorer_report),
            "rehearsal_verdict": scorer_report["verdict"],
            "rehearsal_driver": {
                "path": _SCRIPT_PATH.relative_to(_REPOSITORY_ROOT).as_posix(),
                "raw_sha256": hashlib.sha256(driver_raw).hexdigest(),
                "raw_bytes": len(driver_raw),
            },
            "public_anchor_receipt_present": False,
            "anchor_validator_invoked": False,
            "external_execution_anchor_verified": False,
            "qualification_execution_authorized": False,
            "exact_source_reader_development_admitted": False,
            "qualification_evidence": False,
            "precommitment": (
                "The formal, publicly anchored qualification execution proceeds "
                "regardless of this rehearsal outcome; this rehearsal artifact is "
                "sealed as-is and is never reported as qualification evidence."
            ),
            "formal_evidence_authorized": False,
            "campaign_execution_admitted": False,
            "readable_product_effect": False,
            "four_able_complete": False,
            "human_product_validation": False,
            "production_active": False,
            "os_security_boundary": False,
            "windows_directory_entry_durability_attested": False,
        }
    )
    execution_module._write_artifact_create_only(root / "rehearsal_manifest.json", rehearsal_manifest)
    return {
        "schema_version": "relationship-condition-reader-qualification-rehearsal-cli-summary.v1",
        "command": "rehearse",
        "rehearsal_only": True,
        "execution_protocol_id": protocol_id,
        "execution_root": str(root),
        "rehearsal_manifest_artifact_id": rehearsal_manifest["artifact_id"],
        "rehearsal_verdict": rehearsal_manifest["rehearsal_verdict"],
        "integrity_receipt_count": 8,
        "qualification_execution_authorized": False,
        "qualification_evidence": False,
    }


def main(argv: list[str] | None = None) -> int:
    _install_workspace_source_roots()
    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    payload = _rehearse(args)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
