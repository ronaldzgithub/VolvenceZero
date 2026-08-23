from __future__ import annotations

import argparse
import json
import pathlib

from lifeform_evolution.relationship_lab_relationship_residual_fit import (
    run_relationship_residual_fit,
    validate_relationship_residual_fit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit or GPU-free validate the development-only relationship-domain "
            "named-action residual prerequisite."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--corpus", type=pathlib.Path, required=True)
    run.add_argument("--output-dir", type=pathlib.Path, required=True)
    validate = subparsers.add_parser("validate-existing")
    validate.add_argument("--output-dir", type=pathlib.Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        result = run_relationship_residual_fit(
            corpus_path=args.corpus,
            output_dir=args.output_dir,
            progress=lambda value: print(value, flush=True),
        )
    else:
        result = validate_relationship_residual_fit(output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "artifact_id": result.artifact_id,
                "protocol_id": result.protocol_id,
                "corpus_id": result.corpus_id,
                "fit_lineage_sha256": result.fit_lineage_sha256,
                "bundle_id": result.bundle_id,
                "prerequisite_passed": result.prerequisite_passed,
                "verdict": result.verdict,
                "output_dir": str(result.output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
