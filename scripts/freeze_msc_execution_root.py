#!/usr/bin/env python3
"""Materialize one preregistration-bound, read-only MSC execution root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from freeze_seven_day_execution_root import freeze_execution_root


MSC_FROZEN_EXECUTION_ROOT_SCHEMA = "msc-frozen-execution-root.v1"


def freeze_msc_execution_root(
    *,
    repo_root: Path,
    preregistration_path: Path,
    output_root: Path,
) -> dict[str, object]:
    return freeze_execution_root(
        repo_root=repo_root,
        preregistration_path=preregistration_path,
        output_root=output_root,
        manifest_schema_version=MSC_FROZEN_EXECUTION_ROOT_SCHEMA,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = freeze_msc_execution_root(
        repo_root=args.repo_root,
        preregistration_path=args.preregistration,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "output_root": str(args.output_root.resolve()),
                "preregistration_sha256": manifest[
                    "preregistration_sha256"
                ],
                "source_tree_sha256": manifest["source_tree_sha256"],
                "file_count": manifest["file_count"],
                "read_only": manifest["read_only"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
