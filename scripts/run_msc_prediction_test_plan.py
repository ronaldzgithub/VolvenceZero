#!/usr/bin/env python3
"""Run the isolated MSC N+1 prediction test plan on Apple MPS.

The current worktree can run the target-owning mechanism smoke.  The formal
stage remains fail-closed until the complete Volvence runtime collector and
temporal-controller capacity intervention are implemented and attested.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from companion_test_plan_common import (
    exclusive_mps_lock,
    execution_environment,
    mps_payload,
    print_json,
    require_mps,
    run_plan_command,
)


PLAN_ID = "msc-n-plus-one-prediction-mps.v1"
FORMAL_BLOCKED_EXIT = 3
FORMAL_BLOCKERS = (
    "same-substrate-context-encoder-not-yet-wired",
    "complete-volvence-runtime-arm-not-yet-wired",
    "temporal-controller-capacity-ladder-not-yet-wired",
)
STAGES = ("status", "preflight", "smoke", "formal")


def _prepend_workspace_sources(execution_root: Path) -> None:
    for source_root in reversed(
        tuple(
            path.resolve()
            for path in sorted((execution_root / "packages").glob("*/src"))
            if path.is_dir()
        )
    ):
        value = str(source_root)
        if value not in sys.path:
            sys.path.insert(0, value)


def _write_immutable_json(path: Path, payload: object) -> None:
    data = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    target = path.resolve()
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError(f"existing preflight report differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _run_progress(output_dir: Path | None) -> dict[str, object] | None:
    if output_dir is None:
        return None
    output = output_dir.resolve()
    state_path = output / "run_state.json"
    if not output.exists():
        return {
            "output_dir": os.fspath(output),
            "status": "not-started",
            "completed_unit_count": 0,
            "analysis_allowed": False,
            "formal_claim_allowed": False,
        }
    if not state_path.is_file():
        raise FileNotFoundError(
            f"prediction output exists without run_state.json: {output}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("prediction run_state root must be a JSON object")
    if state.get("schema_version") != "msc-prediction-run-state.v1":
        raise ValueError("prediction run_state schema_version is unsupported")
    units = state.get("completed_units")
    if not isinstance(units, dict):
        raise ValueError("prediction run_state completed_units must be an object")
    status = state.get("status")
    analysis_allowed = state.get("analysis_allowed")
    formal_claim_allowed = state.get("formal_claim_allowed")
    raw_corpus_text_retained = state.get("raw_corpus_text_retained")
    if status not in {"running", "complete"}:
        raise ValueError("prediction run_state status is invalid")
    if not isinstance(analysis_allowed, bool):
        raise ValueError("prediction run_state analysis_allowed must be boolean")
    if not isinstance(formal_claim_allowed, bool):
        raise ValueError("prediction run_state formal_claim_allowed must be boolean")
    if raw_corpus_text_retained is not False:
        raise ValueError("prediction checkpoint must not retain raw corpus text")
    return {
        "output_dir": os.fspath(output),
        "status": status,
        "completed_unit_count": len(units),
        "last_completed_unit": state.get("last_completed_unit"),
        "configuration_fingerprint": state.get("configuration_fingerprint"),
        "analysis_allowed": analysis_allowed,
        "formal_claim_allowed": formal_claim_allowed,
        "raw_corpus_text_retained": raw_corpus_text_retained,
    }


def _prediction_status(
    *, output_dir: Path | None = None
) -> dict[str, object]:
    status: dict[str, object] = {
        "plan_id": PLAN_ID,
        "formal_eligible": False,
        "formal_blocked_exit": FORMAL_BLOCKED_EXIT,
        "completed_prerequisites": (
            "official-msc-v0.1-corpus",
            "substrate-owned-n-plus-one-target",
            "forward-head-capacity-fields-separated-from-temporal-capacity",
        ),
        "formal_blockers": FORMAL_BLOCKERS,
        "permitted_now": ("preflight", "mechanism-only-smoke"),
        "formal_claim_permitted_now": False,
    }
    progress = _run_progress(output_dir)
    if progress is not None:
        status["run_progress"] = progress
    return status


def _preflight(
    *,
    execution_root: Path,
    msc_root: Path,
    substrate_model: str,
    mps: object,
) -> dict[str, object]:
    _prepend_workspace_sources(execution_root)
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "prediction preflight requires huggingface-hub and transformers"
        ) from exc
    from companion_bench.msc_corpus import load_msc_split
    from companion_bench.prediction_research import (
        build_msc_next_turn_examples,
        render_long_context,
    )
    from volvence_zero.substrate import fingerprint_model_weight_files

    corpus = msc_root.resolve()
    provenance_path = corpus.parent / "DOWNLOAD_PROVENANCE.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"MSC DOWNLOAD_PROVENANCE.json is missing next to {corpus}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise ValueError("MSC provenance root must be a JSON object")
    if provenance.get("schema_version") != "msc-download-provenance.v1":
        raise ValueError("MSC provenance schema_version is unsupported")

    snapshot = Path(snapshot_download(substrate_model, local_files_only=True))
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    context_limit = int(config.max_position_embeddings)
    split_audits: dict[str, object] = {}
    observed_max = 0
    for split in ("train", "validation", "heldout"):
        dyads, audit = load_msc_split(corpus, split=split, strict=True)
        examples = build_msc_next_turn_examples(dyads)
        token_counts = tuple(
            len(
                tokenizer.encode(
                    render_long_context(example),
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            for example in examples
        )
        if not token_counts:
            raise ValueError(f"MSC {split} produced no N+1 examples")
        maximum = max(token_counts)
        observed_max = max(observed_max, maximum)
        split_audits[split] = {
            "dyad_count": len(dyads),
            "example_count": len(examples),
            "max_full_history_tokens": maximum,
            "mean_full_history_tokens": sum(token_counts) / len(token_counts),
            "over_model_context_limit": sum(
                count > context_limit for count in token_counts
            ),
            "file_sha256": audit.file_sha256,
            "sorted_id_sha256": audit.sorted_id_sha256,
        }
    full_history_fits = observed_max <= context_limit
    if not full_history_fits:
        raise ValueError(
            "MSC full history exceeds the frozen substrate context limit; "
            "a recency-truncation preregistration is required"
        )
    return {
        "schema_version": "msc-n-plus-one-mps-preflight.v1",
        "plan": _prediction_status(),
        "mps": mps_payload(mps),
        "corpus": {
            "root": os.fspath(corpus),
            "license_policy": "noncommercial-research-only",
            "provenance_schema_version": provenance["schema_version"],
            "splits": split_audits,
        },
        "substrate": {
            "model_id": substrate_model,
            "snapshot_revision": snapshot.name,
            "weights_sha256": fingerprint_model_weight_files(snapshot),
            "declared_context_limit": context_limit,
            "observed_max_full_history_tokens": observed_max,
            "full_history_fits_without_truncation": full_history_fits,
            "128k_arm_adds_distinct_msc_exposure": observed_max > context_limit,
        },
        "claim_boundary": (
            "Preflight proves local data/model/device readiness only. The current "
            "smoke still uses prototype contexts and cannot adjudicate thesis v3."
        ),
    }


def _smoke_command(
    *,
    python: Path,
    execution_root: Path,
    msc_root: Path,
    output_dir: Path,
    substrate_model: str,
    resume: bool,
) -> tuple[str, ...]:
    runner = execution_root / "scripts/run_msc_prediction_research.py"
    if not runner.is_file():
        raise FileNotFoundError(f"MSC mechanism runner does not exist: {runner}")
    argv = (
        str(python),
        str(runner),
        "--msc-root",
        str(msc_root),
        "--output",
        str(output_dir),
        "--accept-noncommercial-license",
        "--device",
        "mps",
        "--substrate-model",
        substrate_model,
        "--substrate-device",
        "mps",
        "--substrate-layer-indices",
        "11",
        "12",
        "13",
        "--train-dyads",
        "2",
        "--validation-dyads",
        "1",
        "--heldout-dyads",
        "1",
        "--epochs",
        "2",
        "--seeds",
        "0",
        "1",
        "2",
    )
    return (*argv, *(("--resume",) if resume else ()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--execution-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--msc-root",
        type=Path,
        default=Path("data/external/msc/v0.1/extracted"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument(
        "--substrate-model", default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mps-lock",
        type=Path,
        default=Path("artifacts/.companion-evidence-mps.lock"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "status":
        print_json(_prediction_status(output_dir=args.output_dir))
        return 0
    if args.stage == "formal":
        print_json(_prediction_status())
        return FORMAL_BLOCKED_EXIT

    execution_root = args.execution_root.resolve()
    msc_root = (
        args.msc_root.resolve()
        if args.msc_root.is_absolute()
        else (execution_root / args.msc_root).resolve()
    )
    environment = execution_environment(execution_root)
    with exclusive_mps_lock(args.mps_lock, plan_id=PLAN_ID):
        mps = require_mps()
        if args.stage == "preflight":
            if args.preflight_report is None:
                raise ValueError("prediction preflight requires --preflight-report")
            report = _preflight(
                execution_root=execution_root,
                msc_root=msc_root,
                substrate_model=args.substrate_model,
                mps=mps,
            )
            _write_immutable_json(args.preflight_report, report)
            print_json(report)
            return 0
        if args.output_dir is None:
            raise ValueError("prediction smoke requires --output-dir")
        return_code = run_plan_command(
            _smoke_command(
                python=args.python.resolve(),
                execution_root=execution_root,
                msc_root=msc_root,
                output_dir=args.output_dir.resolve(),
                substrate_model=args.substrate_model,
                resume=args.resume,
            ),
            execution_root=execution_root,
            environment=environment,
        )
        if return_code != 0:
            return return_code
    print_json(
        {
            "plan_id": PLAN_ID,
            "stage": "smoke",
            "evidence_level": "mechanism-only-pilot",
            "formal_claim_permitted": False,
            "mps": mps_payload(mps),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
