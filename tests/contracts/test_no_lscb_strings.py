# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Contract test: the ``lscb`` legacy token must be fully retired.

After the Companion Bench rename (monorepo SSOT + ``companionbench/bench``
public mirror + ``companionbench/heldout`` private submodule), no
``lscb`` / ``LSCB`` substring is allowed in any **load-bearing** tree:

* ``packages/companion-bench/``  source + tests + examples
* ``scripts/companion_bench/``   build, release, smoke harnesses
* ``site/``                      public site HTML/CSS/JS/JSON/CNAME
* ``docs/external/``             RFC, submission protocol, hash manifest
* ``.github/``                   workflows + issue templates
* ``tests/contracts/``           guard tests (this file is in the
                                  allow-list because the *name* refers
                                  to the legacy token by necessity)

The test walks each tree byte-by-byte (case-insensitive search for
``lscb``). An explicit per-file allow-list pins files where the
substring is structurally unavoidable (e.g. this test file's own
docstring, base64-ish noise inside vendored research PDFs).

This is the "long-term watchdog" that keeps the rename from
silently regressing.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Trees that MUST be free of any lscb / LSCB token.
_GUARDED_TREES: tuple[pathlib.Path, ...] = (
    REPO_ROOT / "packages" / "companion-bench",
    REPO_ROOT / "scripts" / "companion_bench",
    REPO_ROOT / "site",
    REPO_ROOT / "docs" / "external",
    REPO_ROOT / ".github",
)

# Single-file targets outside the guarded trees that also must be clean.
_GUARDED_FILES: tuple[pathlib.Path, ...] = (
    REPO_ROOT / ".gitignore",
    REPO_ROOT / ".gitmodules",
)

# Files where ``lscb`` appears for reasons orthogonal to the rename
# (the test itself names the legacy token; the publish script and the
# G2 cleanup test deliberately reference the legacy stubs).
_ALLOWLIST: frozenset[pathlib.Path] = frozenset(
    {
        pathlib.Path(__file__).resolve(),
        (
            REPO_ROOT / "scripts" / "companion_bench" / "publish_public_bench.sh"
        ).resolve(),
    }
)

# File extensions that are textual and safe to scan as UTF-8.
_TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py", ".pyi",
        ".js", ".mjs", ".cjs",
        ".ts", ".tsx",
        ".css",
        ".html", ".htm",
        ".json", ".jsonl",
        ".yaml", ".yml",
        ".toml",
        ".md", ".rst", ".txt",
        ".sh", ".bash",
        ".ps1", ".cmd", ".bat",
        ".cfg", ".ini",
        ".xml",
        "",  # files with no suffix (CNAME, robots.txt-like)
    }
)

_LSCB_PATTERN = re.compile(r"lscb", re.IGNORECASE)


def _iter_text_files(tree: pathlib.Path) -> list[pathlib.Path]:
    if not tree.exists():
        return []
    out: list[pathlib.Path] = []
    for path in tree.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        # Skip vendored / cache / build-artifact dirs.
        parts = path.relative_to(tree).parts
        parts_set = set(parts)
        if parts_set & {
            "__pycache__", ".pytest_cache", ".ruff_cache",
            "node_modules", ".venv", "venv",
        }:
            continue
        # *.egg-info dirs are setuptools build artifacts; they cache
        # the previous metadata snapshot. They are gitignored and not
        # part of the source surface.
        if any(p.endswith(".egg-info") for p in parts):
            continue
        out.append(path)
    return out


def _collect_offenders() -> list[tuple[pathlib.Path, int, str]]:
    """Walk guarded trees and return (path, line_no, line) for every
    line containing ``lscb`` (case-insensitive), excluding allow-list."""
    offenders: list[tuple[pathlib.Path, int, str]] = []
    seen: set[pathlib.Path] = set()
    candidates: list[pathlib.Path] = []
    for tree in _GUARDED_TREES:
        candidates.extend(_iter_text_files(tree))
    for single in _GUARDED_FILES:
        if single.is_file():
            candidates.append(single)
    for path in candidates:
        rpath = path.resolve()
        if rpath in seen:
            continue
        seen.add(rpath)
        if rpath in _ALLOWLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _LSCB_PATTERN.search(line):
                offenders.append((path, lineno, line.rstrip()))
    return offenders


def test_no_lscb_token_in_guarded_trees() -> None:
    offenders = _collect_offenders()
    if not offenders:
        return
    sample = "\n".join(
        f"{path.relative_to(REPO_ROOT)}:{lineno}: {line}"
        for path, lineno, line in offenders[:40]
    )
    pytest.fail(
        f"Found {len(offenders)} line(s) containing 'lscb' inside guarded "
        f"trees; the brand has been unified to 'companionbench'. First 40:\n\n"
        f"{sample}"
    )


def test_guarded_trees_are_nonempty() -> None:
    """Sanity: catches accidentally-empty glob configurations."""
    total = sum(len(_iter_text_files(tree)) for tree in _GUARDED_TREES)
    assert total > 0, (
        "No text files found across guarded trees; either the layout moved "
        "or this test is misconfigured. Update _GUARDED_TREES."
    )
