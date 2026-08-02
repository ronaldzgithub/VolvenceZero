"""Stage-2 step 1: export the continued-pretraining corpus with provenance.

Renders the seeded generated corpus's *train-split* routes into plain-text
navigation documents (the domain the base LLM is missing). Only train-split
orderings are emitted, and they are compositionally disjoint from the heldout
orderings the same seed produces -- which Stage 3 evaluates on -- so the base
never sees an evaluation trajectory. The manifest records the seed, counts,
token estimate, and a content SHA so the exact corpus is reproducible and
auditable before any GPU time is spent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from volvence_zero.agent.eta_proof_benchmark import (
    generate_eta_proof_corpus,
    render_eta_route_documents,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the Stage-2 continued-pretraining corpus."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-seed", type=int, default=20260802)
    parser.add_argument("--objective-count", type=int, default=8)
    parser.add_argument("--corridor-count", type=int, default=2)
    parser.add_argument("--extra-edge-probability", type=float, default=0.35)
    parser.add_argument("--train-routes", type=int, default=20000)
    parser.add_argument("--heldout-routes", type=int, default=2000)
    parser.add_argument("--train-lengths", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--heldout-lengths", type=int, nargs="+", default=(3, 4))
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Emit each rendered document this many times to reach a token "
            "budget without inventing new (possibly heldout-overlapping) "
            "orderings. Documents stay verbatim; only multiplicity changes."
        ),
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty Stage-2 corpus directory: {output_dir}"
        )

    corpus = generate_eta_proof_corpus(
        seed=args.corpus_seed,
        objective_count=args.objective_count,
        corridor_count=args.corridor_count,
        extra_edge_probability=args.extra_edge_probability,
        train_route_count=args.train_routes,
        heldout_route_count=args.heldout_routes,
        train_lengths=tuple(args.train_lengths),
        heldout_lengths=tuple(args.heldout_lengths),
    )
    base_documents = render_eta_route_documents(
        corpus.environment, corpus.train_cases
    )
    documents = tuple(base_documents) * args.repeat

    # Disjointness proof: no train-document text may describe a heldout route.
    heldout_documents = set(
        render_eta_route_documents(corpus.environment, corpus.heldout_cases)
    )
    overlap = set(base_documents) & heldout_documents
    if overlap:
        raise RuntimeError(
            f"train/heldout document overlap detected ({len(overlap)} docs); "
            "the corpus split is not clean."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    token_estimate = 0
    with corpus_path.open("w", encoding="utf-8") as handle:
        for index, document in enumerate(documents):
            token_estimate += len(document.split())
            handle.write(
                json.dumps({"id": index, "text": document}, ensure_ascii=False)
                + "\n"
            )
    corpus_sha = _sha256_text("\n".join(documents))

    manifest = {
        "schema_version": "eta-stage2-corpus.v1",
        "experiment_id": "eta-stage2-continued-pretrain-corpus",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_seed": args.corpus_seed,
        "objective_count": args.objective_count,
        "corridor_count": args.corridor_count,
        "extra_edge_probability": args.extra_edge_probability,
        "train_lengths": list(args.train_lengths),
        "heldout_lengths": list(args.heldout_lengths),
        "unique_train_routes": corpus.train_route_count,
        "heldout_routes_excluded": corpus.heldout_route_count,
        "repeat": args.repeat,
        "document_count": len(documents),
        "unique_document_count": len(base_documents),
        "whitespace_token_estimate": token_estimate,
        "train_heldout_document_overlap": len(overlap),
        "corpus_content_sha256": corpus_sha,
        "corpus_file": corpus_path.name,
        "environment_id": corpus.environment.env_id,
        "note": (
            "Only train-split orderings are emitted; heldout orderings (Stage-3 "
            "eval) are held out at generation time by the compositional split."
        ),
    }
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
