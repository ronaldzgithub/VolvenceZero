"""Static guard: governance-demo code must not branch on string contents.

The ``.cursor/rules/no-keyword-matching-hacks.mdc`` rule forbids
keyword-matching as a way to drive logic decisions; the simulator + UI
governance code is a particularly attractive place to regress because
it sees free-form user / assistant text every turn.

This test statically scans the two governance-demo source paths and
flags suspicious patterns:

* ``"<literal>" in <var>`` / ``"..." not in <var>`` on a variable that
  obviously holds free-form text (``user_text`` / ``assistant_text`` /
  ``message`` / ``content`` / ``response_text``).
* ``startswith("<literal>") / endswith("<literal>")`` on the same.
* ``re.search(<literal>, <text-like>)`` calls.

False positives in this list (e.g. checking that an HTTP error message
``"sk-..." in env_value`` is set) get an explicit allow-list comment
``# governance-demo: ok-keyword-check`` on the same line.

The scan is intentionally narrow: it does not block protocol-level
exact-match comparisons (``action == "stop"``) — those are not
keyword-matching, they are typed-enum dispatch.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_TEXT_LIKE_NAMES: frozenset[str] = frozenset(
    {
        "user_text",
        "assistant_text",
        "response_text",
        "bot_text",
        "message",
        "content",
        "text",
        "utterance",
    }
)

_PATHS_TO_SCAN: tuple[pathlib.Path, ...] = (
    REPO_ROOT
    / "packages"
    / "lifeform-service"
    / "src"
    / "lifeform_service"
    / "simulator_routes.py",
    REPO_ROOT
    / "packages"
    / "lifeform-service"
    / "src"
    / "lifeform_service"
    / "openai_utterance_client.py",
    REPO_ROOT / "scripts" / "governance_demo" / "drive_session_arc.py",
)

_ALLOW_MARKER = "governance-demo: ok-keyword-check"


def _line_has_allow_marker(source_lines: list[str], lineno: int) -> bool:
    if lineno < 1 or lineno > len(source_lines):
        return False
    return _ALLOW_MARKER in source_lines[lineno - 1]


def _name_or_attr_target(node: ast.AST) -> str | None:
    """If ``node`` is a Name / Attribute load, return its leaf name."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _check_compare(node: ast.Compare) -> list[tuple[int, str]]:
    """Flag ``"foo" in something`` style comparisons against text-like vars."""

    offenders: list[tuple[int, str]] = []
    # We only care about ``In`` / ``NotIn`` operators.
    for op, comparator in zip(node.ops, node.comparators):
        if not isinstance(op, (ast.In, ast.NotIn)):
            continue
        if not isinstance(node.left, ast.Constant) or not isinstance(
            node.left.value, str
        ):
            continue
        target = _name_or_attr_target(comparator)
        if target in _TEXT_LIKE_NAMES:
            offenders.append(
                (
                    node.lineno,
                    f'"{node.left.value!s}" in {target}',
                )
            )
    return offenders


def _check_method_call(node: ast.Call) -> list[tuple[int, str]]:
    """Flag ``text_var.startswith("...")`` / ``.endswith("...")``."""

    offenders: list[tuple[int, str]] = []
    func = node.func
    if not isinstance(func, ast.Attribute):
        return offenders
    if func.attr not in {"startswith", "endswith"}:
        return offenders
    target = _name_or_attr_target(func.value)
    if target not in _TEXT_LIKE_NAMES:
        return offenders
    if not node.args:
        return offenders
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        offenders.append((node.lineno, f"{target}.{func.attr}({first.value!r})"))
    return offenders


def _check_regex_call(node: ast.Call) -> list[tuple[int, str]]:
    """Flag ``re.search('regex', text_var)`` / ``re.match`` / ``re.findall``."""

    offenders: list[tuple[int, str]] = []
    func = node.func
    if not isinstance(func, ast.Attribute):
        return offenders
    if func.attr not in {"search", "match", "findall", "fullmatch"}:
        return offenders
    module_target = _name_or_attr_target(func.value)
    if module_target != "re":
        return offenders
    if len(node.args) < 2:
        return offenders
    haystack = node.args[1]
    target = _name_or_attr_target(haystack)
    if target in _TEXT_LIKE_NAMES:
        offenders.append((node.lineno, f"re.{func.attr}(..., {target})"))
    return offenders


def _scan_file(path: pathlib.Path) -> list[tuple[int, str]]:
    if not path.exists():
        return []
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        new: list[tuple[int, str]] = []
        if isinstance(node, ast.Compare):
            new = _check_compare(node)
        elif isinstance(node, ast.Call):
            new = _check_method_call(node) + _check_regex_call(node)
        for lineno, summary in new:
            if _line_has_allow_marker(source_lines, lineno):
                continue
            offenders.append((lineno, summary))
    return offenders


@pytest.mark.parametrize(
    "path", _PATHS_TO_SCAN, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_no_keyword_matching_against_free_form_text(
    path: pathlib.Path,
) -> None:
    offenders = _scan_file(path)
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} contains keyword-matching against "
        f"free-form text variables (forbidden by "
        f"`.cursor/rules/no-keyword-matching-hacks.mdc`):\n"
        + "\n".join(f"  line {lineno}: {summary}" for lineno, summary in offenders)
        + "\nIf this is a deliberate protocol-level exact match, "
        "add a trailing comment `# governance-demo: ok-keyword-check`."
    )


def test_at_least_one_governance_demo_source_was_scanned() -> None:
    """Sanity: catches accidentally empty path tuples."""

    found = [p for p in _PATHS_TO_SCAN if p.exists()]
    assert found, "_PATHS_TO_SCAN does not point at any existing files"
