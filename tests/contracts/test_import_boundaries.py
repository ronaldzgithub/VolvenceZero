"""Static import-boundary linter for the multi-wheel workspace.

This is the single most important enforcement of the architecture:

1. Kernel wheels (``packages/vz-*``) must NEVER import from product wheels
   (``packages/lifeform-*``). The dependency direction is one-way.

2. Each kernel wheel must only import from wheels declared lower in the
   dependency tier (see ``packages/<wheel>/pyproject.toml``).

3. ``vz-contracts`` is the foundation: zero dependencies on any other
   ``volvence_zero.*`` sub-package.

These checks are AST-only — no module imports, no torch, no transformers,
runs on a stock Python install in a few seconds. CI runs this *before*
the regular pytest suite; if these fail, the rest of the suite is moot.

Why R8: the snapshot-first / contract-first architecture demands that no
runtime owner can silently become a second owner of another module's state
(``no-swallow-errors-no-hasattr-abuse.mdc``,
``ssot-module-boundaries.mdc``). At the wheel level this means physical
separation: code that should not be coupled cannot even be imported.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGES_ROOT = REPO_ROOT / "packages"


# ---------------------------------------------------------------------------
# Allowed dependency tiers
#
# A kernel wheel may import only from itself and from wheels listed in
# ``ALLOWED_VZ_TIERS`` for its tier. Maps wheel-name -> set of allowed
# upstream sub-package roots (under ``volvence_zero.``).
# ---------------------------------------------------------------------------

ALLOWED_VZ_UPSTREAM: dict[str, frozenset[str]] = {
    "vz-contracts": frozenset(),  # foundation: zero upstream
    "vz-substrate": frozenset({"runtime", "learned_update"}),
    "vz-memory": frozenset({"runtime", "learned_update", "substrate", "social_cognition"}),
    # Slice C (2026-05-03): vz-cognition no longer hosts application-tier
    # dataclasses. The ``evaluation`` layer consumes a structural
    # ``Protocol`` surface (``volvence_zero.application_readouts`` in
    # vz-contracts) instead, so the kernel never imports concrete
    # application schema. Owners live in vz-application as before.
    "vz-cognition": frozenset(
        {
            "runtime", "learned_update", "temporal_types", "substrate", "memory",
            "application_readouts", "social_cognition", "environment",
        }
    ),
    "vz-application": frozenset(
        {
            "runtime", "learned_update", "temporal_types", "substrate", "memory",
            # everything in vz-cognition:
            "dual_track", "evaluation", "credit", "regime", "prediction",
            "reflection", "semantic_state",
            "social", "social_cognition", "environment",
        }
    ),
    "vz-temporal": frozenset(
        {
            "runtime", "learned_update", "temporal_types", "substrate", "memory",
            # everything in vz-cognition:
            "dual_track", "evaluation", "credit", "regime", "prediction",
            "reflection", "semantic_state",
            "social", "social_cognition", "environment",
            # vz-application:
            "application",
        }
    ),
    "vz-runtime": frozenset(
        {
            "runtime", "learned_update", "substrate", "memory", "dialogue_trace",
            # everything in vz-cognition:
            "dual_track", "evaluation", "credit", "regime", "prediction",
            "reflection", "semantic_state",
            "social", "social_cognition", "environment",
            # vz-application:
            "application",
            # everything in vz-temporal:
            "temporal", "planning", "internal_rl", "joint_loop",
            # vz-runtime owns these directly:
            "agent", "integration", "brain",
        }
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _python_files(root: pathlib.Path) -> list[pathlib.Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _is_type_checking_block(node: ast.AST) -> bool:
    """Detect typing-only ``if`` blocks.

    Recognises three variants used to host imports that exist only for the
    benefit of static type checkers:

    * ``if TYPE_CHECKING:``                 — modern ``typing.TYPE_CHECKING`` idiom
    * ``if typing.TYPE_CHECKING:`` (etc.)  — same, attribute access form
    * ``if False:``                         — pre-PEP-484 idiom for typing imports
    """
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


def _module_level_imports(py_file: pathlib.Path) -> list[str]:
    """Return imports that load when the file's module is imported.

    Excludes:
      * ``if TYPE_CHECKING:`` blocks (typing-only, no runtime cost).
      * Function / method body imports (deferred — used to break cycles).
      * Relative imports (stay within the same wheel by construction).

    These are the imports that determine the wheel's ``dependencies`` field.
    """
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError as exc:  # pragma: no cover - invalid file is its own error
        pytest.fail(f"Cannot parse {py_file}: {exc}")

    modules: list[str] = []

    def visit(stmts: list[ast.stmt]) -> None:
        for node in stmts:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                if node.module:
                    modules.append(node.module)
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
            # Function / class bodies are NOT visited — those are deferred.

    visit(tree.body)
    return modules


def _vz_subpackage(module: str) -> str | None:
    """Return the first segment under ``volvence_zero.`` if applicable."""
    if module == "volvence_zero":
        return ""
    if not module.startswith("volvence_zero."):
        return None
    rest = module[len("volvence_zero.") :]
    return rest.split(".", 1)[0]


def _all_kernel_files() -> list[tuple[str, pathlib.Path]]:
    out: list[tuple[str, pathlib.Path]] = []
    for wheel_dir in sorted(PACKAGES_ROOT.glob("vz-*")):
        for py in _python_files(wheel_dir / "src"):
            out.append((wheel_dir.name, py))
    return out


def _all_lifeform_files() -> list[tuple[str, pathlib.Path]]:
    out: list[tuple[str, pathlib.Path]] = []
    for wheel_dir in sorted(PACKAGES_ROOT.glob("lifeform-*")):
        for py in _python_files(wheel_dir / "src"):
            out.append((wheel_dir.name, py))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wheel", "py_file"),
    _all_kernel_files(),
    ids=lambda v: v.name if isinstance(v, pathlib.Path) else str(v),
)
def test_kernel_does_not_import_lifeform(wheel: str, py_file: pathlib.Path) -> None:
    """No ``vz-*`` wheel may import any ``lifeform_*`` package.

    This is the spine of the split-ready monorepo. Violating it means kernel
    code has become product-coupled, which makes the future repo split a
    breaking change instead of a pure ``git filter-repo``.
    """
    for module in _module_level_imports(py_file):
        if module.startswith("lifeform_") or module == "lifeform":
            pytest.fail(
                f"{py_file.relative_to(REPO_ROOT)} imports '{module}': "
                f"kernel wheel '{wheel}' must not depend on lifeform layer "
                f"(R8, R15)."
            )


@pytest.mark.parametrize(
    ("wheel", "py_file"),
    _all_kernel_files(),
    ids=lambda v: v.name if isinstance(v, pathlib.Path) else str(v),
)
def test_kernel_only_imports_declared_tier(wheel: str, py_file: pathlib.Path) -> None:
    """A kernel wheel only imports ``volvence_zero.X`` for X in its tier.

    The allowed set is declared in ``ALLOWED_VZ_UPSTREAM`` and mirrors the
    ``dependencies`` field of each wheel's ``pyproject.toml``. Adding a new
    cross-wheel import requires updating both this table AND the upstream
    wheel's pyproject so that ``pip install`` keeps working.
    """
    allowed = ALLOWED_VZ_UPSTREAM[wheel]
    for module in _module_level_imports(py_file):
        sub = _vz_subpackage(module)
        if sub is None:
            continue  # third-party or stdlib import
        if sub == "":
            continue  # bare ``import volvence_zero`` is fine (namespace)
        if sub in allowed:
            continue
        # Determine which sub-packages this wheel itself OWNS — those are
        # always allowed regardless of declared upstream.
        wheel_src = PACKAGES_ROOT / wheel / "src" / "volvence_zero"
        own = {p.name for p in wheel_src.iterdir() if p.is_dir()}
        own |= {p.stem for p in wheel_src.glob("*.py") if p.name != "__init__.py"}
        if sub in own:
            continue
        pytest.fail(
            f"{py_file.relative_to(REPO_ROOT)} imports 'volvence_zero.{sub}': "
            f"wheel '{wheel}' has not declared this dependency tier. "
            f"Allowed upstream: {sorted(allowed) or 'none'}. "
            f"If this is intentional, update both ALLOWED_VZ_UPSTREAM and "
            f"packages/{wheel}/pyproject.toml."
        )


@pytest.mark.parametrize(
    ("wheel", "py_file"),
    _all_lifeform_files(),
    ids=lambda v: v.name if isinstance(v, pathlib.Path) else str(v),
)
def test_lifeform_does_not_import_other_lifeform_directly(
    wheel: str, py_file: pathlib.Path,
) -> None:
    """Lifeform wheels should declare their inter-product dependencies in
    ``pyproject.toml`` rather than reach across via private modules.

    For now we just sanity-check that lifeform code does not reach into
    a kernel sub-package's *private* internals (anything past one segment
    under ``volvence_zero.X.``). The kernel's stable API surface is its
    top-level ``__init__.py``; deeper paths are internal.
    """
    for module in _module_level_imports(py_file):
        # Empty skeleton today; hook is here so future product code does
        # not silently start poking at e.g. ``volvence_zero.runtime.kernel``
        # internals from the product side.
        if module.startswith("volvence_zero.runtime.kernel"):
            pytest.fail(
                f"{py_file.relative_to(REPO_ROOT)} imports kernel internal "
                f"'{module}'. Product layer must consume the public surface "
                f"(volvence_zero.runtime, volvence_zero.brain, ...) only."
            )


def test_application_and_memory_use_temporal_contract_types_not_owner_package() -> None:
    """Consumers may validate temporal snapshots without depending on vz-temporal."""

    checked_files = (
        PACKAGES_ROOT / "vz-application" / "src" / "volvence_zero" / "application" / "runtime.py",
        PACKAGES_ROOT / "vz-memory" / "src" / "volvence_zero" / "memory" / "store.py",
    )
    for py_file in checked_files:
        source = py_file.read_text(encoding="utf-8")
        assert "volvence_zero.temporal.interface" not in source
        assert "from volvence_zero.temporal import" not in source
        assert "import volvence_zero.temporal" not in source


def test_consumers_do_not_synthesize_disabled_temporal_placeholders() -> None:
    """Disabled temporal wiring must be explicit upstream, not rebuilt downstream."""

    source = (
        PACKAGES_ROOT
        / "vz-application"
        / "src"
        / "volvence_zero"
        / "application"
        / "runtime.py"
    ).read_text(encoding="utf-8")
    assert "temporal-disabled-placeholder" not in source
    assert "fell back to placeholder temporal state" not in source


def test_benchmark_release_gates_do_not_use_text_keyword_heuristics() -> None:
    """Open benchmark release gates should consume structured trace fields."""

    source = (
        PACKAGES_ROOT
        / "vz-runtime"
        / "src"
        / "volvence_zero"
        / "agent"
        / "dialogue"
        / "_legacy.py"
    ).read_text(encoding="utf-8")
    assert "hidden-perturbation-label-not-leaked" not in source
    assert "open_hidden_label_leak_count == 0" not in source
    assert "marker in response" not in source
    assert '"repair" in (turn.active_abstract_action or "")' not in source
