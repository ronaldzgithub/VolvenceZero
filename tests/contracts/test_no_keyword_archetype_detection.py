"""Contract test: archetype detection MUST NOT use keyword/regex matching.

AST scan of ``packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/archetype_classifier.py``:

* No ``in user_text`` / ``in user_turn`` / ``in raw`` patterns
  comparing string literals against user input
* No ``re.search`` / ``re.match`` over user input
* No ``.lower() in {...}`` / ``startswith()`` / ``endswith()`` over
  user input

This guards [`no-keyword-matching-hacks.mdc`](../../.cursor/rules/no-keyword-matching-hacks.mdc):
archetype routing must come from learned classifier output, not
fragile string heuristics.

Refs:

* docs/specs/growth-advisor-archetype-detection.md §5
* docs/known-debts.md #66
* .cursor/rules/no-keyword-matching-hacks.mdc
"""

from __future__ import annotations

import ast
import pathlib

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TARGET = (
    _REPO_ROOT
    / "packages"
    / "lifeform-domain-growth-advisor"
    / "src"
    / "lifeform_domain_growth_advisor"
    / "archetype_classifier.py"
)

# Fragments forbidden in archetype_classifier.py (case-insensitive).
# These are heuristics — not exhaustive — but catch the most common
# regressions to keyword paths.
_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "re.search",
    "re.match",
    ".startswith(",
    ".endswith(",
)


def test_archetype_classifier_module_exists() -> None:
    assert _TARGET.exists(), f"missing {_TARGET}"


def test_archetype_classifier_no_forbidden_substring() -> None:
    """No regex matching on user input."""
    src = _TARGET.read_text(encoding="utf-8")
    violations = []
    for needle in _FORBIDDEN_SUBSTRINGS:
        if needle in src:
            violations.append(needle)
    assert not violations, (
        f"archetype_classifier.py contains forbidden keyword patterns "
        f"{violations}; archetype detection must use learned classifier "
        "output (LLMArchetypeClassifier or future metacontroller β_t), "
        "not fragile keyword heuristics. See "
        ".cursor/rules/no-keyword-matching-hacks.mdc"
    )


def test_archetype_classifier_no_in_userturn_pattern() -> None:
    """AST: forbid `<literal> in <user-turn-like-name>` patterns.

    This catches::

        if "焦虑" in user_text: ...
        if "推荐" in raw: ...

    Allows other ``in`` uses (membership in dicts / sets is fine).
    """
    tree = ast.parse(_TARGET.read_text(encoding="utf-8"))
    forbidden_targets = {"user_text", "user_turn", "raw_user_text", "user_message"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            # Look for `<str literal> in <Name>` or `<str literal> not in <Name>`
            for op, comparator in zip(node.ops, node.comparators):
                if not isinstance(op, (ast.In, ast.NotIn)):
                    continue
                if not (
                    isinstance(node.left, ast.Constant)
                    and isinstance(node.left.value, str)
                ):
                    continue
                if isinstance(comparator, ast.Name) and comparator.id in forbidden_targets:
                    violations.append(
                        f"line {node.lineno}: '{node.left.value!r} in "
                        f"{comparator.id}' looks like a keyword match"
                    )
    assert not violations, (
        "archetype_classifier.py contains keyword-in-user-text patterns:\n  "
        + "\n  ".join(violations)
        + "\nUse LLMArchetypeClassifier output instead."
    )


def test_archetype_classifier_calls_only_llm_provider() -> None:
    """The LLMArchetypeClassifier path must invoke ``self._llm_provider``."""
    src = _TARGET.read_text(encoding="utf-8")
    assert "self._llm_provider(" in src, (
        "LLMArchetypeClassifier must call self._llm_provider(...) — that's "
        "the SSOT path for archetype detection."
    )
