"""W4 SSOT contract: ``vz-application`` does not branch on regime_id strings.

Static AST scan over the application package for the patterns that
define hardcoded regime semantics:

* ``regime_id == "<literal>"``
* ``regime_id in {"<literal>", ...}``
* ``regime == "<literal>"``

These should all read from ``ApplicationBrief`` instead. The test
allows a small documented exception list for places where the comparison
is structurally necessary (e.g. matching a ``regime_id`` field against
another typed value, not a hardcoded string).

Adding a new exception is a SSOT regression and requires updating both
this allow-list AND ``docs/specs/expression-layer.md`` / the relevant
spec, so reviewers see the trade-off explicitly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_APPLICATION_ROOT = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "vz-application"
    / "src"
    / "volvence_zero"
    / "application"
)


# Currently-allowed (structurally legitimate) hardcoded regime checks.
# Each entry: (filename, line, why-it-is-not-an-SSOT-violation).
# These remain because they are not "behaviour-driving" string lookups
# but participate in either (a) typed equality between two regime ids
# or (b) cross-snapshot structural checks where the regime id is a
# foreign key. They are tracked here so any drift is visible.
_ALLOWED_HARDCODED_HITS: set[tuple[str, str]] = {
    # ``regime_id in {"a", "b"}`` style sets that act as legitimate
    # invariants on the closed regime vocabulary (used for
    # support_decision_regime). Even though they list literals, they
    # describe a structural property (a regime that supports
    # support-before-decision lifecycle) rather than an arbitrary
    # text mapping. Future cleanup welcome but not blocking W4.
    #
    # Wave 2 of debt #9 split: this hit moved from ``runtime.py`` to
    # ``runtime_helpers.py`` with the helper functions that house it.
    (
        "runtime_helpers.py",
        'regime_id in {"guided_exploration", "problem_solving", "emotional_support"}',
    ),
}


def _find_string_compare_hits(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (lineno, source-fragment) for every regime_id == "literal"
    or regime_id in {literal, ...} pattern.
    """

    hits: list[tuple[int, str]] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Compare(self, node: ast.Compare) -> None:
            left = node.left
            left_name = _name_of(left)
            for op, comparator in zip(node.ops, node.comparators):
                if left_name in {"regime_id", "regime"}:
                    if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                        hits.append((node.lineno, ast.unparse(node)))
                    elif isinstance(op, (ast.In, ast.NotIn)) and _is_string_set(
                        comparator
                    ):
                        hits.append((node.lineno, ast.unparse(node)))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return hits


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_string_set(node: ast.AST) -> bool:
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return any(
            isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            for elt in node.elts
        )
    return False


@pytest.mark.parametrize(
    "module_filename",
    sorted(
        # Wave 2 of debt #9 split: walk recursively so the new
        # ``modules/`` subpackage is also scanned for hardcoded
        # regime-id branching. Each entry is the path RELATIVE to
        # ``_APPLICATION_ROOT`` so the allow-list keys (which list
        # ``runtime_helpers.py`` etc.) line up with what the test
        # parametrizes on.
        str(p.relative_to(_APPLICATION_ROOT)).replace("\\", "/")
        for p in _APPLICATION_ROOT.rglob("*.py")
    ),
)
def test_no_hardcoded_regime_id_branching(module_filename: str) -> None:
    """The application module must not branch on ``regime_id``
    string literals. New regime semantics must be added to
    ``ApplicationBrief`` in the regime templates.
    """

    path = _APPLICATION_ROOT / module_filename
    if not path.is_file():
        pytest.skip("not a regular file")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    hits = _find_string_compare_hits(tree)
    unexpected = [
        (lineno, expr)
        for lineno, expr in hits
        if (module_filename, expr) not in _ALLOWED_HARDCODED_HITS
    ]
    assert not unexpected, (
        f"hardcoded regime-id checks remain in {module_filename} (W4 SSOT regression):\n"
        + "\n".join(f"  L{lineno}: {expr}" for lineno, expr in unexpected)
        + "\n\n"
        + "Each occurrence must either (a) read a typed field from "
        + "ApplicationBrief or (b) be added to _ALLOWED_HARDCODED_HITS "
        + "with a documented reason."
    )


def test_application_brief_helper_is_imported_in_runtime_modules() -> None:
    """Sanity check: every application module that performs
    regime-keyed scoring imports either ``_application_brief`` or
    ``application_brief_for_regime``. If a module has zero brief
    references but still does regime-flavoured work, it is a sign
    the cutover is not yet complete.

    Wave 2 of debt #9 split: ``runtime.py`` is now a thin re-export
    shell; the regime-keyed scoring helper ``_application_brief``
    lives in ``runtime_helpers.py``. The expected-files set tracks
    that move so the SSOT pin still fails when a future regression
    deletes the helper or stops referencing it.
    """

    expected_briefed_modules = {
        "runtime_helpers.py",
        "experience_layers.py",
        "retrieval_readout.py",
    }
    for filename in expected_briefed_modules:
        text = (_APPLICATION_ROOT / filename).read_text(encoding="utf-8")
        assert "_application_brief" in text or "application_brief_for_regime" in text, (
            f"{filename} performs regime-keyed scoring but does not "
            "reference the ApplicationBrief SSOT helper. Has the W4 "
            "cutover regressed?"
        )
