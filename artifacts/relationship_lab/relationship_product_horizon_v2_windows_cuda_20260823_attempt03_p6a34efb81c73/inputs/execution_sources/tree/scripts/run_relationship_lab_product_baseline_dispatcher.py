#!/usr/bin/env python3
"""Run one resident Relationship Lab product-baseline JSONL process."""

from __future__ import annotations

import pathlib
import sys


sys.dont_write_bytecode = True


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SOURCE_ROOTS = tuple(
    path
    for path in sorted((_REPO_ROOT / "packages").glob("*/src"))
    if path.is_dir()
)
for _source_root in reversed(_SOURCE_ROOTS):
    sys.path.insert(0, str(_source_root))

import volvence_zero  # noqa: E402

_VOLVENCE_ZERO_PATHS = [
    str(path.resolve())
    for source_root in _SOURCE_ROOTS
    if (path := source_root / "volvence_zero").is_dir()
]
if not _VOLVENCE_ZERO_PATHS:
    raise RuntimeError("mirrored volvence_zero namespace is empty")
volvence_zero.__path__ = list(_VOLVENCE_ZERO_PATHS)
if volvence_zero.__spec__ is None:
    raise RuntimeError("volvence_zero namespace has no import specification")
volvence_zero.__spec__.submodule_search_locations = list(_VOLVENCE_ZERO_PATHS)

from lifeform_evolution.relationship_lab_product_baseline_dispatcher import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
