#!/usr/bin/env python3
"""Run one resident Relationship Lab product-baseline JSONL process."""

from __future__ import annotations

import pathlib
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _source_root in sorted((_REPO_ROOT / "packages").glob("*/src")):
    if _source_root.is_dir():
        sys.path.insert(0, str(_source_root))

from lifeform_evolution.relationship_lab_product_baseline_dispatcher import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
