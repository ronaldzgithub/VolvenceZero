"""Stage-2 step 2: continued-pretrain Qwen2.5-0.5B on the domain corpus.

Loads the Stage-2 corpus (``run_eta_stage2_corpus.py``), runs a frozen-base
LoRA continued-pretraining pass via the substrate's rare-heavy owner
(:func:`continued_pretrain_and_merge`), merges the LoRA into a NEW frozen base
saved under ``--merged-out``, and records the merged-weight fingerprint. The
original base snapshot is never mutated; Stage 3 loads ``--merged-out`` via
``--model-source``. This is an offline rare-heavy substrate refresh (R2): base
adaptation happens off the live runtime, in the substrate owner, at rare-heavy
cadence.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from volvence_zero.substrate import continued_pretrain_and_merge

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), check=True, capture_output=True, text=True, cwd=_REPO_ROOT
    )
    return result.stdout.strip()


def _load_documents(corpus_file: Path, limit: int | None) -> tuple[str, ...]:
    documents: list[str] = []
    with corpus_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = payload["text"]
            if not isinstance(text, str):
                raise ValueError("corpus row 'text' must be a string.")
            documents.append(text)
            if limit is not None and len(documents) >= limit:
                break
    if not documents:
        raise ValueError(f"no documents loaded from {corpus_file}.")
    return tuple(documents)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-2 continued-pretraining (LoRA -> merged frozen base)."
    )
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--merged-out", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model-source", default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--doc-limit",
        type=int,
        default=None,
        help="Optional cap on documents loaded (smoke runs).",
    )
    args = parser.parse_args()

    documents = _load_documents(args.corpus_file, args.doc_limit)
    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty Stage-2 output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    result = continued_pretrain_and_merge(
        base_model_source=args.base_model_source,
        output_dir=str(args.merged_out),
        documents=documents,
        device=args.device,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        max_length=args.max_length,
    )
    elapsed = time.perf_counter() - started

    manifest = {
        "schema_version": "eta-stage2-pretrain.v1",
        "experiment_id": "eta-stage2-continued-pretrain",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value("rev-parse", "HEAD"),
        "working_tree_dirty": bool(_git_value("status", "--short")),
        "training_mode": result.training_mode,
        "base_model_source": result.base_model_source,
        "merged_model_dir": result.merged_model_dir,
        "weight_fingerprint": result.weight_fingerprint,
        "target_modules": list(result.target_modules),
        "initial_loss": round(result.initial_loss, 6),
        "final_loss": round(result.final_loss, 6),
        "steps_taken": result.steps_taken,
        "document_count": result.document_count,
        "token_count": result.token_count,
        "corpus_file": str(args.corpus_file),
        "rank": args.rank,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "device": args.device,
        "elapsed_seconds": round(elapsed, 1),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "description": result.description,
    }
    (output_dir / "pretrain_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
