"""Packet C (long-horizon-closure) — affordance selection must NOT
be driven by hardcoded descriptor names.

The spec acceptance gate is: ``grep finds no ``if name == "read_file"``
or equivalent routing logic; selection state remains published by the
metacontroller / AffordanceModule via z_t projection``.

This static contract test scans the production source tree (not tests,
not docs, not benchmarks) for the patterns that would constitute
hardcoded affordance routing. If the test fails, somebody added a
branch on a specific affordance name to selection logic — fix it by
making selection a function of typed signals (z_t, regime, scores)
rather than name strings.

False-positive defence:

- The ``rationale`` strings inside affordance scorers / the module
  legitimately contain descriptor names; those are passive labels,
  not routing decisions, so the regex requires a comparison operator
  or an ``in`` collection literal next to the name.
- ``tests/`` and ``docs/`` are excluded — test fixtures and spec
  examples may legitimately quote descriptor names like
  ``"read_file"`` for documentation purposes.
- Backend registration code (e.g. ``register_backend("read_file",
  fn)``) is allowed: that's a name -> function lookup map, not a
  selection branch on the name.

Add new patterns to ``_BANNED_REGEXES`` if you spot a new way to
bypass the contract.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
"""``tests/contracts/test_no_keyword_routing.py`` -> repo root."""


_PROD_ROOTS = (
    _REPO_ROOT / "packages",
)


# Excluded directories under ``packages/`` (tests, vendored data,
# legacy debug artifacts). All keep "data" affordance names but
# do not represent runtime selection logic.
_EXCLUDE_PATH_FRAGMENTS = (
    "/tests/",
    "\\tests\\",
    "/__pycache__/",
    "\\__pycache__\\",
    # Legacy dialogue trace fixtures may contain literal tool names;
    # they are debug data, not selection logic.
    "/dialogue_trace_fixtures/",
)


# Patterns that constitute affordance hardcoded routing. Each pattern
# is paired with a one-line explanation of WHY it's banned so a
# future maintainer can understand whether a new code site genuinely
# violates the contract.
#
# All patterns use raw strings + alternation tokens that look for
# routing-style branches on a specific affordance name. Generic name
# comparisons (e.g. for serialisation / lookup map registration) are
# allowed — those don't dispatch behavior.
_BANNED_REGEXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"""if\s+(?:descriptor\.|d\.|cand\.|self\.)?\s*name\s*==\s*['"](read_file|write_file|grep|run_test|list_dir|echo)['"]"""
        ),
        "branch on specific affordance name -> selection should be z_t-driven",
    ),
    (
        re.compile(
            r"""if\s+descriptor_name\s*==\s*['"](read_file|write_file|grep|run_test|list_dir|echo)['"]"""
        ),
        "branch on specific descriptor_name string -> use AffordanceModule scoring",
    ),
    (
        re.compile(
            r"""(?:descriptor_name|name)\s+in\s*\{[^}]*['"]read_file['"][^}]*\}"""
        ),
        "set-membership branch on hardcoded affordance names -> not a learned policy",
    ),
)


def _iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            posix = path.as_posix()
            win = str(path)
            if any(frag in posix or frag in win for frag in _EXCLUDE_PATH_FRAGMENTS):
                continue
            yield path


def test_no_hardcoded_affordance_name_routing_in_production_code() -> None:
    """Static contract: production source must not branch on
    specific affordance descriptor names. Selection is the
    metacontroller's job, scored over z_t.
    """
    violations: list[tuple[Path, int, str, str]] = []
    for source_path in _iter_python_files(_PROD_ROOTS):
        try:
            text = source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Skip non-UTF-8 files (none expected, but defensive).
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for pattern, reason in _BANNED_REGEXES:
                if pattern.search(line):
                    violations.append((source_path, line_no, line.rstrip(), reason))
    if violations:
        formatted = "\n".join(
            f"  {path.relative_to(_REPO_ROOT)}:{line_no}\n"
            f"    {snippet!r}\n"
            f"    reason: {reason}"
            for path, line_no, snippet, reason in violations
        )
        pytest.fail(
            f"Found {len(violations)} hardcoded affordance-name routing "
            f"site(s); affordance selection must be z_t-driven via "
            f"AffordanceModule / score_affordance_candidates.\n"
            f"{formatted}"
        )


def test_score_affordance_candidates_is_name_agnostic() -> None:
    """Programmatic guarantee: ``score_affordance_candidates``
    treats descriptor names as opaque hash inputs. Renaming a
    descriptor must not change behavior beyond the SHA-256
    projection.
    """
    from lifeform_affordance import score_affordance_candidates

    z_t = (0.4, -0.3, 0.5, 0.1)
    a = dict(
        score_affordance_candidates(
            descriptor_names=("read_file", "write_file", "grep"),
            z_t=z_t,
        )
    )
    # Same z_t, same names -> same scores (determinism)
    a_repeat = dict(
        score_affordance_candidates(
            descriptor_names=("read_file", "write_file", "grep"),
            z_t=z_t,
        )
    )
    assert a == a_repeat, (
        "score_affordance_candidates must be deterministic for the "
        "same (z_t, names) pair."
    )
    # Different z_t -> different scores (proves z_t actually drives)
    b = dict(
        score_affordance_candidates(
            descriptor_names=("read_file", "write_file", "grep"),
            z_t=(-0.4, 0.3, -0.5, -0.1),
        )
    )
    assert a != b, (
        "Flipping z_t signs must change candidate scores; if scores "
        "are identical the projection is not actually using z_t."
    )
    # Renaming the descriptor changes its score (because the name
    # hashes differently), but does NOT change the score of OTHER
    # descriptors. This is the "no name-routing" guarantee in the
    # other direction: removing one name doesn't reshape policy.
    c = dict(
        score_affordance_candidates(
            descriptor_names=("read_file", "write_file"),
            z_t=z_t,
        )
    )
    assert c["read_file"] == a["read_file"]
    assert c["write_file"] == a["write_file"]


def test_empty_z_t_returns_neutral_scores() -> None:
    """Cold-start invariant: no z_t means every candidate scores 0.5
    (no preference). This guards against regressing to "first-wins"
    selection when the metacontroller has no signal yet.
    """
    from lifeform_affordance import score_affordance_candidates

    scores = dict(
        score_affordance_candidates(
            descriptor_names=("read_file", "write_file", "grep"),
            z_t=(),
        )
    )
    assert scores == {
        "read_file": 0.5,
        "write_file": 0.5,
        "grep": 0.5,
    }
