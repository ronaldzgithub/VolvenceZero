#!/usr/bin/env python3
"""Run or validate the cross-process Relationship Lab product horizon."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


# Establish the campaign's child-process isolation contract even when the
# operator did not pre-seed the shell.  Formal invocations should still use
# ``python -s`` so the parent interpreter also excludes the user site.
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8:strict")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _source_root in sorted((_REPO_ROOT / "packages").glob("*/src")):
    if _source_root.is_dir():
        sys.path.insert(0, str(_source_root))

from lifeform_domain_emogpt.lab.contracts import canonical_json  # noqa: E402
from lifeform_domain_emogpt.lab.relationship_product_pilot_source import (  # noqa: E402
    build_relationship_product_pilot_public_view,
    load_relationship_product_pilot_source_protocol,
)
from lifeform_evolution.relationship_lab_product_horizon import (  # noqa: E402
    load_relationship_product_horizon_protocol,
    relationship_product_public_embedding_inputs,
    run_relationship_product_decision_worker,
    run_relationship_product_horizon_campaign,
    run_relationship_product_onboarding_worker,
    validate_relationship_product_horizon_campaign,
)
from lifeform_evolution.relationship_lab_product_model_adapters import (  # noqa: E402
    bge_m3_public_semantic_embedder,
    build_precomputed_public_embedding_table,
    write_precomputed_public_embedding_table,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Relationship Lab product-horizon typed-control campaign",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one immutable campaign")
    run.add_argument("--output-dir", type=pathlib.Path, required=True)
    run.add_argument("--public-embedding-table", type=pathlib.Path, required=True)
    run.add_argument(
        "--public-embedding-attestation",
        type=pathlib.Path,
        required=True,
    )
    run.add_argument(
        "--protocol",
        type=pathlib.Path,
        help="strict protocol override; defaults to the packaged v1 contract",
    )
    run.add_argument("--python-executable", default=sys.executable)
    run.add_argument("--max-workers", type=int, default=4)
    run.add_argument("--worker-timeout-seconds", type=float, default=120.0)
    run.add_argument("--baseline-timeout-seconds", type=float, default=900.0)
    run.add_argument(
        "--with-strong-baselines",
        action="store_true",
        help="launch one resident revision-pinned Qwen/BGE JSONL dispatcher",
    )
    run.add_argument(
        "--baseline-dispatcher-script",
        type=pathlib.Path,
        default=_REPO_ROOT / "scripts" / "run_relationship_lab_product_baseline_dispatcher.py",
    )

    validate = commands.add_parser(
        "validate-existing",
        help="CPU/GPU/model-free full artifact and metric recomputation",
    )
    validate.add_argument("--output-dir", type=pathlib.Path, required=True)

    semantic = commands.add_parser(
        "public-semantic-inputs",
        help="print the exact typed public inputs required to build the BGE table",
    )
    semantic.add_argument("--output", type=pathlib.Path)

    build_table = commands.add_parser(
        "build-public-embedding-table",
        help="build the create-only revision-pinned BGE table on protocol device",
    )
    build_table.add_argument("--output", type=pathlib.Path, required=True)
    build_table.add_argument("--protocol", type=pathlib.Path)

    verify_table = commands.add_parser(
        "verify-public-embedding-table",
        help="fresh-process exact BGE reobservation of the pinned public table",
    )
    verify_table.add_argument("--table", type=pathlib.Path, required=True)
    verify_table.add_argument("--output-attestation", type=pathlib.Path, required=True)
    verify_table.add_argument("--python-executable", default=sys.executable)

    onboarding = commands.add_parser("worker-onboarding")
    onboarding.add_argument("--request", type=pathlib.Path, required=True)
    onboarding.add_argument("--receipt", type=pathlib.Path, required=True)
    onboarding.add_argument("--run-root", type=pathlib.Path, required=True)

    decision = commands.add_parser("worker-decision")
    decision.add_argument("--request", type=pathlib.Path, required=True)
    decision.add_argument("--preaction-receipt", type=pathlib.Path, required=True)
    decision.add_argument("--postaction-receipt", type=pathlib.Path, required=True)
    decision.add_argument("--run-root", type=pathlib.Path, required=True)

    verify_worker = commands.add_parser("worker-verify-public-embedding-table")
    verify_worker.add_argument("--table", type=pathlib.Path, required=True)
    verify_worker.add_argument("--output-attestation", type=pathlib.Path, required=True)
    return parser


def _baseline_dispatcher_command(args: argparse.Namespace) -> tuple[str, ...] | None:
    if not args.with_strong_baselines:
        return None
    protocol = load_relationship_product_horizon_protocol(args.protocol)
    script = pathlib.Path(args.baseline_dispatcher_script).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"baseline dispatcher script is missing: {script}")
    return (
        args.python_executable,
        "-s",
        str(script),
        "--model-source",
        protocol.baseline_model_source,
        "--model-id",
        protocol.baseline_model_id,
        "--model-revision",
        protocol.baseline_model_revision,
        "--device",
        protocol.baseline_cuda_device,
        "--torch-dtype",
        "float16",
        "--context-window-tokens",
        str(protocol.context_window_tokens),
        "--generation-reserve-tokens",
        str(protocol.generation_token_reserve),
        "--prefill-chunk-size",
        str(protocol.generation_prefill_chunk_size),
        "--generation-use-cache",
        "--semantic-mode",
        "live_bge_m3_cached",
        "--bge-model-source",
        protocol.semantic_model_source,
        "--bge-model-revision",
        protocol.semantic_model_revision,
        "--bge-device",
        protocol.semantic_device,
    )


def _write_semantic_inputs(path: pathlib.Path | None) -> None:
    source = load_relationship_product_pilot_source_protocol()
    public = build_relationship_product_pilot_public_view(source)
    payload = {
        "schema_version": "relationship-product-public-semantic-inputs.v1",
        "public_plan_sha256": public.public_plan_sha256,
        "inputs": [item.to_payload() for item in relationship_product_public_embedding_inputs(public)],
    }
    rendered = canonical_json(payload) + "\n"
    if path is None:
        sys.stdout.write(rendered)
        return
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())


def _build_public_embedding_table(
    *,
    output: pathlib.Path,
    protocol_path: pathlib.Path | None,
) -> None:
    protocol = load_relationship_product_horizon_protocol(protocol_path)
    source = load_relationship_product_pilot_source_protocol()
    public = build_relationship_product_pilot_public_view(source)
    inputs = relationship_product_public_embedding_inputs(public)
    embedder = bge_m3_public_semantic_embedder(
        device=protocol.semantic_device,
        model_revision=protocol.semantic_model_revision,
    )
    if (
        embedder.model_source != protocol.semantic_model_source
        or embedder.model_revision != protocol.semantic_model_revision
    ):
        raise ValueError("BGE adapter provenance differs from campaign protocol")
    table = build_precomputed_public_embedding_table(
        embedder=embedder,
        public_inputs=inputs,
    )
    if (
        table.source_model_id != protocol.semantic_model_source
        or table.source_model_revision != protocol.semantic_model_revision
    ):
        raise ValueError("built embedding table provenance differs from protocol")
    target = write_precomputed_public_embedding_table(table, path=output)
    print(
        canonical_json(
            {
                "path": str(target.resolve()),
                "artifact_id": table.artifact_id,
                "source_model_id": table.source_model_id,
                "source_model_revision": table.source_model_revision,
                "record_count": len(table.records),
            }
        )
    )


def _verify_public_embedding_table(args: argparse.Namespace) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONIOENCODING": "utf-8:strict",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    command = (
        args.python_executable,
        "-s",
        str(pathlib.Path(__file__).resolve()),
        "worker-verify-public-embedding-table",
        "--table",
        str(args.table.resolve()),
        "--output-attestation",
        str(args.output_attestation.resolve()),
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"public embedding reobservation child failed; stderr={completed.stderr[-4000:]}")
    sys.stdout.write(completed.stdout)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        report = run_relationship_product_horizon_campaign(
            output_dir=args.output_dir,
            public_embedding_table_path=args.public_embedding_table,
            public_embedding_attestation_path=(args.public_embedding_attestation),
            worker_script=pathlib.Path(__file__).resolve(),
            python_executable=args.python_executable,
            protocol_path=args.protocol,
            baseline_dispatcher_command=_baseline_dispatcher_command(args),
            max_workers=args.max_workers,
            worker_timeout_seconds=args.worker_timeout_seconds,
            baseline_timeout_seconds=args.baseline_timeout_seconds,
        )
        print(canonical_json(report))
        return 0
    if args.command == "validate-existing":
        report = validate_relationship_product_horizon_campaign(
            output_dir=args.output_dir,
        )
        print(canonical_json(report))
        return 0
    if args.command == "public-semantic-inputs":
        _write_semantic_inputs(args.output)
        return 0
    if args.command == "build-public-embedding-table":
        _build_public_embedding_table(
            output=args.output,
            protocol_path=args.protocol,
        )
        return 0
    if args.command == "verify-public-embedding-table":
        _verify_public_embedding_table(args)
        return 0
    if args.command == "worker-onboarding":
        run_relationship_product_onboarding_worker(
            request_path=args.request,
            receipt_path=args.receipt,
            run_root=args.run_root,
        )
        return 0
    if args.command == "worker-decision":
        run_relationship_product_decision_worker(
            request_path=args.request,
            preaction_receipt_path=args.preaction_receipt,
            postaction_receipt_path=args.postaction_receipt,
            run_root=args.run_root,
        )
        return 0
    if args.command == "worker-verify-public-embedding-table":
        from lifeform_evolution.relationship_lab_product_horizon import (
            verify_relationship_product_public_embedding_table,
        )

        attestation = verify_relationship_product_public_embedding_table(
            table_path=args.table,
            output_attestation_path=args.output_attestation,
        )
        print(canonical_json(attestation))
        return 0
    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
