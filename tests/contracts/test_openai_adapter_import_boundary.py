"""Static import-boundary linter for the ``lifeform-openai-compat`` wheel.

Companion to ``tests/contracts/test_import_boundaries.py``. The
``lifeform-openai-compat`` adapter is a thin OpenAI-shape translator
that fronts ``lifeform-service``; if it acquires a direct dependency
on any kernel sub-package or any vertical wheel, it stops being a
read-only facade and starts being a second owner of cognitive state
(violates R8 SSOT).

Allowed imports (module-level, ``import`` and ``from ... import``):

* Standard library / typing
* ``aiohttp``
* ``lifeform_service`` and its sub-modules (the public facade we are
  fronting)
* The wheel's own modules (``lifeform_openai_compat.*``)

Anything else — in particular ``volvence_zero.*`` (kernel) and
``lifeform_domain_*`` (verticals) and ``lifeform_core`` (the
in-process facade we deliberately route around so the adapter never
holds a Lifeform reference; SessionManager owns those) — is a
violation.

Why ``lifeform_core`` is forbidden too: lifeform-core hosts the
Lifeform / LifeformSession types directly. The adapter is supposed
to talk to SessionManager only, and SessionManager returns
LifeformSession; if we needed to type-annotate that we may need a
narrow exemption later, but right now this wheel produces and
consumes only stdlib + DTOs, so we keep the strict rule until a
real type-annotation need surfaces in packet 3.

Implementation approach (mirrors ``test_import_boundaries.py``):

* AST-only walk; no module imports — runs in the no-deps lint stage.
* Excludes ``if TYPE_CHECKING:`` blocks (those are typing-only and
  carry no runtime cost / no real coupling).
* Function-body imports are allowed and ignored (deferred imports
  stay private to the call path that uses them).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ADAPTER_SRC_ROOT = (
    REPO_ROOT / "packages" / "lifeform-openai-compat" / "src" / "lifeform_openai_compat"
)


# Modules whose import is allowed at module level. Anything not on
# this list and not in the stdlib triggers a failure.
_ALLOWED_THIRD_PARTY_PREFIXES: frozenset[str] = frozenset(
    {
        "aiohttp",
        "lifeform_service",
        "lifeform_openai_compat",
    }
)

# Modules explicitly forbidden at module level. These are the failure
# modes we care about most: kernel internals and vertical wheels.
_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "volvence_zero",
    "lifeform_domain_",
    "lifeform_core",
    "lifeform_evolution",
    "lifeform_expression",
    "lifeform_thinking",
    "lifeform_ingestion",
    "lifeform_affordance",
    "dlaas_platform",
)


def _is_type_checking_block(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Constant) and test.value is False:
        return True
    return False


def _module_level_imports(py_file: pathlib.Path) -> list[tuple[int, str]]:
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError as exc:  # pragma: no cover
        pytest.fail(f"Cannot parse {py_file}: {exc}")
    out: list[tuple[int, str]] = []

    def visit(stmts: list[ast.stmt]) -> None:
        for node in stmts:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    out.append((node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue  # relative import, stays in-wheel
                if node.module:
                    out.append((node.lineno, node.module))
            elif _is_type_checking_block(node):
                continue
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, (ast.Try, ast.TryStar)):
                visit(node.body)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, ast.With):
                visit(node.body)
            # Function / class bodies are NOT visited.

    visit(tree.body)
    return out


def _adapter_files() -> list[pathlib.Path]:
    if not ADAPTER_SRC_ROOT.exists():
        return []
    return sorted(
        p for p in ADAPTER_SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )


def _is_stdlib_or_typing(module: str) -> bool:
    """Best-effort allowlist for the standard library + a few typing
    helpers. Adding to this list is fine; it is just the "trivially
    safe" prefix list so the failure message focuses on real risks.
    """
    if not module:
        return False
    # Anything starting with our allowlist of third-party / our own
    # prefixes goes through the explicit allowlist path; this helper
    # is for everything else.
    head = module.split(".", 1)[0]
    # Heuristic: if Python's importlib can resolve it as part of the
    # stdlib (best-effort by name) we accept. Otherwise the strict
    # allowlist below catches it. We use a simple curated list; a
    # missing stdlib name will surface as a test failure with a clear
    # path to the fix (extend this list).
    return head in _CURATED_STDLIB_HEADS


_CURATED_STDLIB_HEADS: frozenset[str] = frozenset(
    {
        "abc", "asyncio", "collections", "contextlib", "copy", "dataclasses",
        "datetime", "enum", "errno", "functools", "hashlib", "inspect", "io",
        "itertools", "json", "logging", "math", "os", "pathlib", "queue",
        "random", "re", "string", "struct", "sys", "tempfile", "textwrap",
        "threading", "time", "traceback", "types", "typing", "uuid", "warnings",
        "weakref", "__future__",
    }
)


@pytest.mark.parametrize("py_file", _adapter_files(), ids=lambda p: p.name)
def test_adapter_only_imports_allowed_modules(py_file: pathlib.Path) -> None:
    """The adapter must import only stdlib, aiohttp, lifeform_service, or itself."""

    if not _adapter_files():
        pytest.skip("lifeform-openai-compat has not landed yet")

    for lineno, module in _module_level_imports(py_file):
        head = module.split(".", 1)[0]

        # Forbidden first — explicit failure messages for the most
        # important violations.
        for forbidden in _FORBIDDEN_PREFIXES:
            if module == forbidden or module.startswith(forbidden + ".") or (
                forbidden.endswith("_") and head.startswith(forbidden)
            ):
                pytest.fail(
                    f"{py_file.relative_to(REPO_ROOT)}:{lineno} imports "
                    f"forbidden module {module!r}. The OpenAI-compat adapter "
                    f"is a read-only facade over lifeform_service and must "
                    f"not depend on kernel or vertical wheels. Forbidden "
                    f"prefix: {forbidden}. See "
                    f"docs/known-debts.md #29 (red line 1)."
                )

        # Allowed third-party / our own.
        if any(module == p or module.startswith(p + ".") for p in _ALLOWED_THIRD_PARTY_PREFIXES):
            continue

        # Stdlib heads.
        if _is_stdlib_or_typing(module):
            continue

        pytest.fail(
            f"{py_file.relative_to(REPO_ROOT)}:{lineno} imports {module!r} "
            f"which is not on the OpenAI-compat adapter allowlist. "
            f"Allowed third-party/internal prefixes: "
            f"{sorted(_ALLOWED_THIRD_PARTY_PREFIXES)}. "
            f"If this is a stdlib module, add it to "
            f"_CURATED_STDLIB_HEADS in this test file. If it is a new "
            f"required dependency, both this test AND the adapter's "
            f"pyproject.toml must be updated together (and the change "
            f"must be reviewed for kernel-coupling implications)."
        )


def test_adapter_pyproject_declares_only_lifeform_service_and_aiohttp() -> None:
    """The adapter's pyproject must not list any kernel or vertical wheel.

    The runtime allowlist above and the declared dependencies must
    agree: if pyproject pulls a wheel that the AST guard forbids,
    pip-install would still bring it into the environment and tempt
    a future change to use it. Catch that drift here.
    """

    pyproject = REPO_ROOT / "packages" / "lifeform-openai-compat" / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("lifeform-openai-compat has not landed yet")
    text = pyproject.read_text(encoding="utf-8")

    # Forbidden direct dependencies — substring match is sufficient
    # because pyproject's dependencies array uses the wheel name
    # verbatim (no aliases / extras for these wheels).
    forbidden = (
        "vz-contracts", "vz-substrate", "vz-memory", "vz-cognition",
        "vz-application", "vz-temporal", "vz-runtime",
        "lifeform-core", "lifeform-expression", "lifeform-thinking",
        "lifeform-ingestion", "lifeform-affordance", "lifeform-evolution",
        "lifeform-domain-emogpt", "lifeform-domain-coding",
        "lifeform-domain-character", "lifeform-domain-figure",
        "lifeform-domain-growth-advisor",
        "dlaas-platform-",
    )
    # Find the [project] dependencies block and only check entries
    # in there (otherwise comments mentioning these names would
    # spuriously fail).
    in_deps = False
    deps_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("dependencies"):
            in_deps = True
            continue
        if in_deps:
            if stripped.startswith("]"):
                in_deps = False
                continue
            if stripped:
                deps_lines.append(stripped)
    deps_block = "\n".join(deps_lines)

    for name in forbidden:
        if name in deps_block:
            pytest.fail(
                f"packages/lifeform-openai-compat/pyproject.toml declares "
                f"forbidden direct dependency {name!r}. The adapter must "
                f"only depend on lifeform-service (which transitively "
                f"pulls everything else). Adding a direct dep to a "
                f"kernel / vertical wheel inverts the dependency tier."
            )
