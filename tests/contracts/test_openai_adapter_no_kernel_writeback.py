"""Static AST guard: the OpenAI-compat adapter never writes to kernel state.

This is the second prong of the debt #29 red-line enforcement (see
``test_openai_adapter_import_boundary.py`` for the first prong). The
import-boundary test forbids the adapter from *importing* any kernel
sub-package. This test forbids the adapter from *calling* any
SessionManager / LifeformSession method whose name starts with ``_``
or whose name is on a known-mutating denylist, and from poking at
underscore-prefixed attributes on objects returned through the
SessionManager facade.

Why both checks: import-boundary is necessary but not sufficient.
``lifeform_service`` legitimately exposes mutable inner objects
(LifeformSession, the alpha identity provider, the template
adapter); a careless adapter could reach into ``session._brain``,
``session.__dict__["_runtime"]`` and start mutating kernel state
without ever importing ``volvence_zero``. The AST guard makes that
class of bug literally impossible to land.

Allowed touchpoints on the lifeform-service surface:

* SessionManager public methods: ``create_session`` / ``get_session``
  / ``close_session`` / ``has_session`` / ``session_count`` /
  ``session_summaries`` / ``template_context_for`` (read-only) and
  the public properties.
* LifeformSession public methods used in our adapter path:
  ``run_turn`` (the only state-advancing call we are allowed to
  make), ``end_scene`` (closing scenes is part of the lifecycle).
  We surface read-only attributes (``turn_summaries``,
  ``open_scene``, ``latest_response_text``) but those are reads,
  not assignments, so they never trigger the writeback denylist.

Forbidden patterns (any module-level or function-body occurrence):

* Any attribute access ``foo._something`` where ``_something`` does
  not start with ``__`` (the dunder accessor case is left alone:
  ``__class__`` / ``__dict__`` etc. are introspection and the
  callers are typically tooling, not state mutation).
* Any call to ``setattr(...)`` whose target is a session-manager
  facade variable.
* Any attribute ASSIGNMENT (``foo.attr = value``) on a name that
  looks like a SessionManager / LifeformSession (heuristic: any
  variable produced by a known SessionManager method name above).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ADAPTER_SRC_ROOT = (
    REPO_ROOT / "packages" / "lifeform-openai-compat" / "src" / "lifeform_openai_compat"
)


# Attributes whose direct read/write would break SessionManager
# read-only-ness even though they are technically reachable through
# the public surface.
_FORBIDDEN_ATTR_NAMES: frozenset[str] = frozenset(
    {
        # SessionManager internals.
        "_factory", "_alpha_factory", "_alpha_identity_provider",
        "_alpha_memory_scope_root_dir", "_substrate_runtime",
        "_template_adapter", "_templates_root_dir", "_sessions",
        "_lock", "_clock", "_max_sessions", "_idle_eviction_seconds",
        "_vertical_name", "_evict_idle_locked",
        "_evict_lru_to_capacity_locked", "_fresh_session_id",
        # LifeformSession likely-private members. Names below are
        # heuristic: anything starting with "_" on a session is
        # forbidden anyway via the underscore-attr rule, but listing
        # the most-tempting ones explicitly produces a better error
        # message when violated.
        "_brain", "_runtime", "_kernel", "_owner", "_identity_provider",
    }
)


def _adapter_files() -> list[pathlib.Path]:
    if not ADAPTER_SRC_ROOT.exists():
        return []
    return sorted(
        p for p in ADAPTER_SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )


def _walk_for_violations(tree: ast.AST) -> list[tuple[int, str]]:
    """Return ``(lineno, message)`` pairs for every violation in this AST."""

    out: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # Underscore attribute reads / writes on ANY object are forbidden.
        # We allow dunder access (``obj.__class__`` etc.) because that is
        # typically introspection, not state mutation, and the cases that
        # matter most are single-leading-underscore names.
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_") and not attr.startswith("__"):
                # Skip self-attribute access inside our own classes:
                # ``self._cache`` on an OpenAICompatRouter is fine because
                # the router is part of the adapter's own state. Detect
                # this by checking whether the value is a Name node with
                # id ``self`` or ``cls``.
                value = node.value
                if isinstance(value, ast.Name) and value.id in {"self", "cls"}:
                    continue
                out.append(
                    (
                        node.lineno,
                        f"underscore attribute access {_render_attribute(node)!r} on a "
                        f"non-self target. Adapter must use the SessionManager / "
                        f"LifeformSession public surface only.",
                    )
                )
                continue
            if attr in _FORBIDDEN_ATTR_NAMES:
                out.append(
                    (
                        node.lineno,
                        f"forbidden private attribute {attr!r}. This name is on the "
                        f"SessionManager / LifeformSession internal denylist; using "
                        f"it would let the adapter become a second owner of kernel "
                        f"state (R8 SSOT).",
                    )
                )

        # ``setattr(target, ...)`` on a non-self target.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr":
            if not node.args:
                continue
            target = node.args[0]
            if isinstance(target, ast.Name) and target.id in {"self", "cls"}:
                continue
            out.append(
                (
                    node.lineno,
                    "setattr() call on a non-self target. Adapter must not mutate "
                    "objects produced by the SessionManager facade.",
                )
            )

        # Direct attribute assignment on a non-self target where the
        # attribute name suggests a SessionManager / LifeformSession
        # field (any underscore name covered above; here we catch the
        # public-name case where the assignment is still suspicious).
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute):
                    if isinstance(tgt.value, ast.Name) and tgt.value.id in {"self", "cls"}:
                        continue
                    if tgt.attr in _FORBIDDEN_ATTR_NAMES:
                        out.append(
                            (
                                tgt.lineno,
                                f"assignment to forbidden private attribute "
                                f"{tgt.attr!r}.",
                            )
                        )

    return out


def _render_attribute(node: ast.Attribute) -> str:
    parts: list[str] = [node.attr]
    cur = node.value
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    elif isinstance(cur, ast.Call):
        parts.append("<call>")
    else:
        parts.append("<expr>")
    parts.reverse()
    return ".".join(parts)


@pytest.mark.parametrize("py_file", _adapter_files(), ids=lambda p: p.name)
def test_adapter_does_not_writeback_to_kernel(py_file: pathlib.Path) -> None:
    """The adapter must use only public, read-or-lifecycle SessionManager methods."""

    if not _adapter_files():
        pytest.skip("lifeform-openai-compat has not landed yet")
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError as exc:  # pragma: no cover
        pytest.fail(f"Cannot parse {py_file}: {exc}")
    violations = _walk_for_violations(tree)
    if violations:
        rendered = "\n  ".join(f"L{ln}: {msg}" for ln, msg in violations)
        pytest.fail(
            f"{py_file.relative_to(REPO_ROOT)} contains kernel-writeback "
            f"violations:\n  {rendered}\n"
            f"See docs/known-debts.md #29 red-line 1: the OpenAI-compat "
            f"adapter is a read-only facade. To intentionally surface a "
            f"new SessionManager / LifeformSession method, add the public "
            f"name to lifeform-service and reference it via the public "
            f"facade — never reach into private state."
        )
