#!/usr/bin/env python3
"""Materialize and validate the model-free Product Horizon source-v3 admission."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _source_root in sorted((_REPO_ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(_source_root))

from lifeform_domain_emogpt.lab.contracts import canonical_json  # noqa: E402
from lifeform_evolution.relationship_product_source_admission import (  # noqa: E402
    finalize_relationship_product_source_admission,
    load_relationship_product_source_admission_protocol,
    materialize_relationship_product_source_admission,
    validate_relationship_product_source_admission,
    validate_relationship_product_source_admission_materialization,
    write_relationship_product_source_admission_comparison,
)


_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show-protocol")

    worker = subparsers.add_parser("worker")
    worker.add_argument("--output-dir", required=True)
    worker.add_argument("--expected-protocol-id", required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--expected-protocol-id", required=True)
    compare.add_argument("--worker-a-pid", type=int, required=True)
    compare.add_argument("--worker-b-pid", type=int, required=True)

    admit = subparsers.add_parser("admit")
    admit.add_argument("--output-dir", required=True)
    admit.add_argument("--implementation-git-commit", required=True)

    validate = subparsers.add_parser("validate-existing")
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--expected-protocol-id", required=True)
    return parser.parse_args(argv)


def _emit(payload: object) -> None:
    print(canonical_json(payload), flush=True)


def _child_command(*args: str) -> list[str]:
    return [sys.executable, str(pathlib.Path(__file__).resolve()), *args]


def _wait_success(process: subprocess.Popen[str], *, role: str) -> None:
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"source admission {role} failed with exit {process.returncode}; "
            f"stdout={stdout!r}; stderr={stderr!r}"
        )


def _verify_implementation_commit(expected_commit: str) -> None:
    if _GIT_COMMIT.fullmatch(expected_commit) is None:
        raise ValueError("implementation git commit must be lowercase 40-hex")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise ValueError("implementation git commit does not match current HEAD")
    protocol, _ = load_relationship_product_source_admission_protocol()
    closure_paths = [
        str(item["path"])
        for item in protocol["direct_execution_closure"]
    ]
    owned_paths = [
        "packages/lifeform-evolution/src/lifeform_evolution/relationship_product_source_admission.py",
        "packages/lifeform-evolution/src/lifeform_evolution/protocols/relationship_product_source_v3_campaign_admission_v1.json",
        "scripts/run_relationship_product_source_admission.py",
        *closure_paths,
    ]
    tracked = subprocess.run(
        ["git", "ls-files", "--", *owned_paths],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if set(tracked) != set(owned_paths):
        raise ValueError("source admission implementation closure is not fully tracked")
    clean = subprocess.run(
        ["git", "diff", "--quiet", expected_commit, "--", *owned_paths],
        cwd=_REPO_ROOT,
        check=False,
    )
    if clean.returncode != 0:
        raise ValueError("source admission implementation closure differs from frozen commit")


def _run_admission(output_dir: pathlib.Path, *, implementation_git_commit: str) -> dict[str, object]:
    _verify_implementation_commit(implementation_git_commit)
    if output_dir.exists():
        raise FileExistsError(f"source campaign admission is create-only: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    _, protocol_id = load_relationship_product_source_admission_protocol()
    replay_a = output_dir / "replay_a"
    replay_b = output_dir / "replay_b"
    worker_a = subprocess.Popen(
        _child_command(
            "worker",
            "--output-dir",
            str(replay_a),
            "--expected-protocol-id",
            protocol_id,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    worker_b = subprocess.Popen(
        _child_command(
            "worker",
            "--output-dir",
            str(replay_b),
            "--expected-protocol-id",
            protocol_id,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_success(worker_a, role="worker A")
    _wait_success(worker_b, role="worker B")
    comparator = subprocess.Popen(
        _child_command(
            "compare",
            "--output-dir",
            str(output_dir),
            "--expected-protocol-id",
            protocol_id,
            "--worker-a-pid",
            str(worker_a.pid),
            "--worker-b-pid",
            str(worker_b.pid),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_success(comparator, role="comparator")
    manifest = finalize_relationship_product_source_admission(
        output_dir,
        implementation_git_commit=implementation_git_commit,
    )
    validated = validate_relationship_product_source_admission(
        output_dir,
        expected_protocol_id=protocol_id,
    )
    if manifest != validated:
        raise ValueError("source admission final validation drifted from created manifest")
    return validated


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    if args.command == "show-protocol":
        protocol, protocol_id = load_relationship_product_source_admission_protocol()
        _emit(
            {
                "schema_version": protocol["schema_version"],
                "protocol_id": protocol_id,
                "evidence_tier": protocol["evidence_tier"],
                "campaign_execution_authorized": protocol["claims"][
                    "campaign_execution_authorized"
                ],
            }
        )
        return 0
    if args.command == "worker":
        _, protocol_id = load_relationship_product_source_admission_protocol()
        if args.expected_protocol_id != protocol_id:
            raise ValueError("worker expected protocol identity drifted")
        root = pathlib.Path(args.output_dir)
        manifest = materialize_relationship_product_source_admission(root)
        validated = validate_relationship_product_source_admission_materialization(
            root,
            expected_protocol_id=protocol_id,
        )
        if manifest != validated:
            raise ValueError("worker validation drifted from created manifest")
        _emit(
            {
                "artifact_id": manifest["artifact_id"],
                "protocol_id": protocol_id,
            }
        )
        return 0
    if args.command == "compare":
        root = pathlib.Path(args.output_dir)
        receipt = write_relationship_product_source_admission_comparison(
            root / "comparison.json",
            root / "replay_a",
            root / "replay_b",
            expected_protocol_id=args.expected_protocol_id,
            worker_a_pid=args.worker_a_pid,
            worker_b_pid=args.worker_b_pid,
        )
        _emit(receipt)
        return 0
    if args.command == "admit":
        manifest = _run_admission(
            pathlib.Path(args.output_dir),
            implementation_git_commit=args.implementation_git_commit,
        )
        _emit(manifest)
        return 0
    if args.command == "validate-existing":
        _emit(
            validate_relationship_product_source_admission(
                pathlib.Path(args.output_dir),
                expected_protocol_id=args.expected_protocol_id,
            )
        )
        return 0
    raise AssertionError(f"unreachable command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
