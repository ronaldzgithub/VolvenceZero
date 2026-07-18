# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Every source file under the Apache-licensed wheels must carry a header.

Four wheels in this monorepo are explicitly Apache 2.0 (the rest of the
repo is Proprietary):

* ``packages/companion-bench``
* ``packages/companion-ref-harness``
* ``packages/companion-standard`` (Relationship Representation Standard,
  A2 license flip — oss-relationship-representation-standard.md)
* ``packages/companion-trajgen`` (standard-family data pipeline, same flip)

These wheels ship as independent benchmark / standard / reference
artifacts and their license must be obvious from every source file (no
hidden license toggling, no central NOTICE-only files). This test scans
the wheel src directories and asserts every ``.py`` file starts with
an Apache 2.0 copyright + license header.

Header pattern accepted (the copyright line varies per wheel family):

    # Copyright 2026 Companion Bench Contributors
    # Licensed under the Apache License, Version 2.0[...]
"""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_BENCH_COPYRIGHT = "Copyright 2026 Companion Bench Contributors"
_STANDARD_COPYRIGHT = "Copyright 2026 Companion Standard Contributors"

# src dir -> required copyright line
_APACHE_DIR_COPYRIGHT: dict[pathlib.Path, str] = {
    REPO_ROOT / "packages" / "companion-bench" / "src": _BENCH_COPYRIGHT,
    REPO_ROOT / "packages" / "companion-ref-harness" / "src": _BENCH_COPYRIGHT,
    REPO_ROOT / "packages" / "companion-standard" / "src": _STANDARD_COPYRIGHT,
    REPO_ROOT / "packages" / "companion-trajgen" / "src": _STANDARD_COPYRIGHT,
}
_APACHE_DIRS: tuple[pathlib.Path, ...] = tuple(_APACHE_DIR_COPYRIGHT)


def _copyright_line_for(path: pathlib.Path) -> str:
    for root, line in _APACHE_DIR_COPYRIGHT.items():
        if path.is_relative_to(root):
            return line
    raise AssertionError(f"{path} is not under a registered Apache dir")


def _iter_apache_sources() -> list[pathlib.Path]:
    """Return all .py files that contain actual source code.

    Empty ``__init__.py`` marker files (zero bytes / whitespace-only)
    are intentionally skipped: a license header on a zero-byte file
    is cargo, and these markers exist purely to satisfy Python's
    package-discovery rules. Any ``__init__.py`` that contains real
    code (re-exports, version string, etc.) must still carry the
    Apache header.
    """

    out: list[pathlib.Path] = []
    for root in _APACHE_DIRS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            content = path.read_text(encoding="utf-8")
            if not content.strip():
                continue
            out.append(path)
    return out


_LICENSE_LINE_FRAGMENT: str = "Licensed under the Apache License, Version 2.0"


@pytest.mark.parametrize(
    "path",
    _iter_apache_sources(),
    ids=lambda p: str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
)
def test_apache_license_header_present(path: pathlib.Path) -> None:
    header_chunk = "\n".join(path.read_text(encoding="utf-8").splitlines()[:6])
    copyright_line = _copyright_line_for(path)
    assert copyright_line in header_chunk, (
        f"{path}: missing Apache 2.0 copyright line ({copyright_line!r}) in "
        f"first 6 lines. Every source file under the Apache-licensed wheels "
        f"must start with the standard Apache 2.0 header "
        f"(copyright + license)."
    )
    assert _LICENSE_LINE_FRAGMENT in header_chunk, (
        f"{path}: missing Apache 2.0 license fragment in first 6 lines."
    )


def test_apache_dirs_were_scanned() -> None:
    files = _iter_apache_sources()
    assert files, (
        "No Apache-licensed sources found. Either the directories moved "
        f"({[str(d.relative_to(REPO_ROOT)) for d in _APACHE_DIRS]}) or the "
        "test glob is mis-configured; update this test."
    )
