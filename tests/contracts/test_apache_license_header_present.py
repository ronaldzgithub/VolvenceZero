# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Every source file under the Apache-licensed wheels must carry a header.

Two wheels in this monorepo are explicitly Apache 2.0 (the rest of the
repo is Proprietary):

* ``packages/companion-bench``
* ``packages/companion-ref-harness``

Both wheels ship as independent benchmark / reference artifacts and
their license must be obvious from every source file (no hidden
license toggling, no central NOTICE-only files). This test scans the
two wheel src directories and asserts every ``.py`` file starts with
an Apache 2.0 copyright + license header.

Header pattern accepted (case-insensitive match on the license line):

    # Copyright 2026 Companion Bench Contributors
    # Licensed under the Apache License, Version 2.0[...]
"""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_APACHE_DIRS: tuple[pathlib.Path, ...] = (
    REPO_ROOT / "packages" / "companion-bench" / "src",
    REPO_ROOT / "packages" / "companion-ref-harness" / "src",
)


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


_COPYRIGHT_LINE: str = "Copyright 2026 Companion Bench Contributors"
_LICENSE_LINE_FRAGMENT: str = "Licensed under the Apache License, Version 2.0"


@pytest.mark.parametrize(
    "path",
    _iter_apache_sources(),
    ids=lambda p: str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
)
def test_apache_license_header_present(path: pathlib.Path) -> None:
    header_chunk = "\n".join(path.read_text(encoding="utf-8").splitlines()[:6])
    assert _COPYRIGHT_LINE in header_chunk, (
        f"{path}: missing Apache 2.0 copyright line in first 6 lines. "
        f"Every source file under packages/companion-bench/ and "
        f"packages/companion-ref-harness/ must start with the standard "
        f"Apache 2.0 header (copyright + license)."
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
