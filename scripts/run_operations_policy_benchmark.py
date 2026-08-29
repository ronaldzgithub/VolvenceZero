#!/usr/bin/env python3
"""Run the Operations policy gate and write an immutable evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from collections.abc import Mapping


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _source_root in sorted((_REPO_ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(_source_root))

from lifeform_domain_operations import (  # noqa: E402
    issue_operations_policy_activation,
    operations_policy_benchmark_preregistration,
    operations_policy_benchmark_scenario_set,
    review_operations_policy_promotion,
    run_operations_policy_benchmark,
    validate_operations_policy_activation,
)
from volvence_zero.credit import GateDecision  # noqa: E402


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: pathlib.Path, payload: Mapping[str, object]) -> str:
    encoded = _json_bytes(payload)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    preregistration = operations_policy_benchmark_preregistration()
    scenario_set = operations_policy_benchmark_scenario_set()
    report, checkpoint = run_operations_policy_benchmark()
    review = review_operations_policy_promotion(
        report=report,
        candidate_checkpoint=checkpoint,
    )
    if review.decision is not GateDecision.ALLOW:
        raise SystemExit(
            "Operations policy promotion blocked: "
            + ", ".join(review.blocking_reasons)
        )
    receipt = issue_operations_policy_activation(
        review=review,
        report=report,
        candidate_checkpoint=checkpoint,
        issued_at_ms=int(time.time() * 1_000),
    )
    validate_operations_policy_activation(
        report=report,
        review=review,
        receipt=receipt,
        candidate_checkpoint=checkpoint,
    )

    payloads = {
        "preregistration.json": preregistration,
        "scenario_set.json": scenario_set,
        "candidate_checkpoint.json": checkpoint.to_json(),
        "benchmark_report.json": report.to_json(),
        "promotion_review.json": review.to_json(),
        "activation_receipt.json": receipt.to_json(),
    }
    file_sha256 = {
        name: _write_json(output_dir / name, payload)
        for name, payload in payloads.items()
    }
    manifest = {
        "schema_version": "operations-policy-evidence-bundle.v1",
        "benchmark_report_id": report.report_id,
        "promotion_review_id": review.review_id,
        "activation_receipt_id": receipt.activation_receipt_id,
        "activation_scope": receipt.activation_scope,
        "evidence_scope": report.evidence_scope,
        "production_default_changed": report.production_default_changed,
        "rollback_config_field": receipt.rollback_config_field,
        "file_sha256": file_sha256,
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
