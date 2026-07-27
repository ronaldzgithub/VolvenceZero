"""SSOT guard: the embodiment package must not import kernel internals.

The digital ant enters the brain core ONLY through the ``vz-runtime`` facade
and the ``vz-contracts`` / ``vz-substrate`` public contracts. It must never
import ``volvence_zero.temporal`` / ``.memory`` / ``.prediction`` / etc.
directly, otherwise it would become a hidden second owner of kernel internals
(violates R8 + the module boundary rule).

Walk scope — deliberate, and asserted by
``test_walk_covers_every_python_file_in_the_package``:

* ``src/volvence_ant/**`` — the shipped wheel.  HARD ban: not a single
  kernel-internal import, in any file, ever.  This is the boundary
  ``docs/specs/digital-ant-embodiment.md`` §2 freezes.
* ``tests/**`` — same wheel, but not shipped.  A handful of tests act as
  *callers* of the embodiment rather than as embodiment code: they construct
  an opaque kernel config object (``JointLoopSchedule``, handed straight
  through ``AntSessionConfig.joint_schedule``) or select an owner-published
  track (``Track``) while auditing kernel state.  Those are pinned one by one
  in ``_TEST_ONLY_KERNEL_IMPORTS`` below, every pinned wheel must be declared
  in the package's ``test`` extra, and every pinned entry must still be in
  use.  Anything else is an offender, exactly as in ``src``.
* ``scripts/run_ant_*.py`` at the repo ROOT — intentionally NOT walked.  Those
  drivers live outside the wheel: they are monorepo integration surface that
  wires several wheels together (they already import
  ``volvence_zero.joint_loop`` / ``.temporal`` to build schedules and
  policies), they ship in no distribution, and nothing under
  ``packages/vz-embodiment-ant`` imports them.  Pulling them into this guard
  would either make the guard fail for code that is not the library, or
  force the forbidden list to be narrowed — both dishonest.  The wheel
  boundary is the package directory; that is what this file enforces.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PACKAGE_ROOT / "src" / "volvence_ant"
_TESTS_ROOT = _PACKAGE_ROOT / "tests"
_PYPROJECT = _PACKAGE_ROOT / "pyproject.toml"

# Wheels whose ``volvence_zero.*`` modules are kernel/business internals for
# this package. ``vz-contracts`` / ``vz-substrate`` / ``vz-runtime`` are the
# three the embodiment is allowed to depend on (see [project].dependencies);
# everything below must be reached through the vz-runtime facade instead.
_KERNEL_WHEELS: tuple[str, ...] = (
    "vz-temporal",
    "vz-memory",
    "vz-cognition",
    "vz-application",
)

# Kernel-internal top-level modules the embodiment must not touch directly.
# This list is the falsifier for the two allow-lists below: a module absent
# from BOTH is invisible to the guard, so anything kernel-owned could be
# laundered into ``_ALLOWED_FACADE_PREFIXES`` and no assertion would object.
# ``test_forbidden_list_covers_every_kernel_wheel_module`` therefore pins it
# against the actual monorepo layout so it tightens as the kernel grows.
_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    # vz-temporal
    "volvence_zero.temporal.",
    "volvence_zero.internal_rl.",
    "volvence_zero.joint_loop.",
    "volvence_zero.planning.",
    # vz-memory
    "volvence_zero.memory.",
    # vz-cognition
    "volvence_zero.prediction.",
    "volvence_zero.credit.",
    "volvence_zero.decision_workspace.",
    "volvence_zero.dual_track.",
    "volvence_zero.regime.",
    "volvence_zero.semantic_state.",
    "volvence_zero.evaluation.",
    "volvence_zero.reflection.",
    "volvence_zero.apprenticeship.",
    "volvence_zero.audit.",
    "volvence_zero.conditioning_bank_adapters.",
    "volvence_zero.interlocutor.",
    "volvence_zero.personal_conditioning.",
    "volvence_zero.personal_conditioning_rendering.",
    "volvence_zero.rupture_state.",
    "volvence_zero.social.",
    # vz-application
    "volvence_zero.application.",
    "volvence_zero.protocol_runtime.",
)

# Exact modules that are also forbidden (the bare package, not just submodules).
_FORBIDDEN_EXACT: frozenset[str] = frozenset(
    prefix.rstrip(".") for prefix in _FORBIDDEN_PREFIXES
)

# Public facade surfaces the SHIPPED library may import.
_ALLOWED_FACADE_PREFIXES: tuple[str, ...] = (
    "volvence_zero.environment",  # vz-contracts environment event types
    "volvence_zero.substrate",  # vz-substrate public contract
    "volvence_zero.runtime",  # vz-contracts kernel container
    "volvence_zero.temporal_types",  # vz-contracts controller state types
    "volvence_zero.agent",  # vz-runtime orchestration facade
    "volvence_zero.integration",  # vz-runtime rollout config facade
)

# Extra public surfaces only the test suite reaches for.  Still facade-level
# (vz-contracts), so they are NOT kernel internals — they simply never occur in
# the shipped library.
_TESTS_EXTRA_FACADE_PREFIXES: tuple[str, ...] = (
    "volvence_zero.owner_hydration",  # vz-contracts owner persistence contract
)

# Kernel-internal modules the package's own tests are allowed to import, mapped
# to the wheel that owns them.  Each entry is a deliberate caller-side use, not
# an embodiment-side dependency; adding one requires declaring its wheel in the
# ``test`` extra (see ``test_test_only_kernel_wheels_are_declared_as_test_extra``).
_TEST_ONLY_KERNEL_IMPORTS: dict[str, str] = {
    # ``AntSessionConfig.joint_schedule`` is an opaque ``object`` passthrough
    # precisely so the library never imports this vz-temporal-internal type.
    # Someone has to build it; in production that is a repo-root driver, in the
    # suite it is the test. Also covers ``ETANLJointLoop`` in the mechanism
    # audit, which drives the joint loop directly as a kernel-side fixture.
    "volvence_zero.joint_loop": "vz-temporal",
    # ``Track`` names which owner-published track a causal-action-head readout
    # comes from while a test audits the kernel's own checkpoint surface.
    "volvence_zero.memory": "vz-memory",
}


def _iter_imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                modules.append(node.module)
    return modules


# Vendored / generated trees that are not first-party wheel surface. The
# package also carries a ``web/`` frontend; setuptools ships only
# ``src/volvence_ant*`` so it is not wheel surface, but it is still walked
# (it holds no Python today, and if it ever does, that Python is first-party
# and belongs under the same boundary). Only its installed npm dependencies
# and build output are skipped.
_NON_SOURCE_DIRS: frozenset[str] = frozenset(
    {"__pycache__", "node_modules", "build", "dist", ".venv", ".pytest_cache"}
)


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if not _NON_SOURCE_DIRS.intersection(path.parts)
    )


def _imports_by_file(root: Path) -> dict[Path, list[str]]:
    imports: dict[Path, list[str]] = {}
    for path in _iter_python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports[path] = _iter_imported_modules(tree)
    return imports


def _is_kernel_internal(module: str) -> bool:
    return module in _FORBIDDEN_EXACT or module.startswith(_FORBIDDEN_PREFIXES)


def _pinned_root(module: str) -> str | None:
    """Return the pinned test-only kernel module ``module`` belongs to."""

    for pinned in _TEST_ONLY_KERNEL_IMPORTS:
        if module == pinned or module.startswith(f"{pinned}."):
            return pinned
    return None


def _load_pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))


def _requirement_names(requirements: list[str]) -> set[str]:
    """Strip version / extra markers off PEP 508 requirement strings."""

    names: set[str] = set()
    for requirement in requirements:
        head = requirement.split(";", 1)[0].strip()
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            head = head.split(separator, 1)[0]
        names.add(head.strip())
    return names


def _monorepo_kernel_modules() -> dict[str, str] | None:
    """Map ``volvence_zero.<name>`` -> owning kernel wheel, from the layout.

    Returns ``None`` when the sibling wheels are not checked out (a run against
    installed distributions), where the static list above is all there is.
    """

    packages_root = _PACKAGE_ROOT.parent
    discovered: dict[str, str] = {}
    for wheel in _KERNEL_WHEELS:
        namespace = packages_root / wheel / "src" / "volvence_zero"
        if not namespace.is_dir():
            return None
        for entry in namespace.iterdir():
            if entry.name.startswith("_"):
                continue
            if entry.is_dir() or entry.suffix == ".py":
                discovered[entry.stem] = wheel
    return discovered


def test_forbidden_list_covers_every_kernel_wheel_module() -> None:
    """The negative list must not lag behind the kernel it guards.

    Without this, a module the kernel added after the list was written is
    unknown to the guard: it is neither an allowed facade nor a forbidden
    internal, so adding it to ``_ALLOWED_FACADE_PREFIXES`` would launder a
    kernel internal past every assertion in this file.
    """

    discovered = _monorepo_kernel_modules()
    if discovered is None:
        pytest.skip("kernel wheels are not checked out beside this package")

    missing = sorted(
        f"volvence_zero.{name}  (owned by {wheel})"
        for name, wheel in discovered.items()
        if f"volvence_zero.{name}" not in _FORBIDDEN_EXACT
    )
    assert not missing, (
        "these kernel-wheel modules are not in _FORBIDDEN_PREFIXES, so the "
        "import guard cannot see them. Add them (they are internals), or — if "
        "one is genuinely a public contract — move it to vz-contracts and "
        "justify the facade entry:\n" + "\n".join(missing)
    )


def test_forbidden_list_has_no_phantom_entries() -> None:
    """Anti-rot in the other direction: no guarding modules that do not exist."""

    discovered = _monorepo_kernel_modules()
    if discovered is None:
        pytest.skip("kernel wheels are not checked out beside this package")

    phantom = sorted(
        module
        for module in _FORBIDDEN_EXACT
        if module.removeprefix("volvence_zero.") not in discovered
    )
    assert not phantom, (
        "these modules are guarded but no kernel wheel owns them any more; "
        "drop them so the list keeps describing reality:\n" + "\n".join(phantom)
    )


def test_walk_covers_every_python_file_in_the_package() -> None:
    """The guard below must see the WHOLE package, not just ``src``.

    The historical guard walked ``src/volvence_ant`` only, so the package's own
    tests — which do import kernel internals — sat outside the boundary while
    the spec claimed this file enforced the ban.
    """

    walked = {
        path.resolve()
        for root in (_SRC_ROOT, _TESTS_ROOT)
        for path in _iter_python_files(root)
    }
    in_package = {
        path.resolve() for path in _iter_python_files(_PACKAGE_ROOT)
    }
    unwalked = sorted(
        str(path.relative_to(_PACKAGE_ROOT)) for path in in_package - walked
    )
    assert not unwalked, (
        "python files inside the wheel are outside the import-boundary walk; "
        "add their root to the walk (or explain why they are not wheel "
        "surface) instead of leaving them unchecked:\n" + "\n".join(unwalked)
    )


def test_no_kernel_internal_imports() -> None:
    offenders: list[str] = []
    for path, modules in _imports_by_file(_SRC_ROOT).items():
        for module in modules:
            if _is_kernel_internal(module):
                offenders.append(f"{path.relative_to(_SRC_ROOT)} -> {module}")
    assert not offenders, (
        "embodiment package imported kernel internals directly (use the "
        "vz-runtime facade / vz-contracts / vz-substrate instead):\n"
        + "\n".join(offenders)
    )


def test_only_allowed_volvence_zero_prefixes() -> None:
    """Positive guard: every volvence_zero import resolves to an allowed wheel."""

    unexpected: list[str] = []
    for path, modules in _imports_by_file(_SRC_ROOT).items():
        for module in modules:
            if module.startswith("volvence_zero") and not module.startswith(
                _ALLOWED_FACADE_PREFIXES
            ):
                unexpected.append(f"{path.relative_to(_SRC_ROOT)} -> {module}")
    assert not unexpected, (
        "embodiment imported an unexpected volvence_zero module; add it to the "
        "allowed facade list only if it is a vz-contracts / vz-runtime / "
        "vz-substrate public surface:\n" + "\n".join(unexpected)
    )


def test_package_tests_kernel_imports_are_pinned() -> None:
    """The suite may call kernel internals, but only the pinned ones.

    This is not a relaxation of the ban: before this guard existed the tests
    were not walked at all, so ANY kernel-internal import could appear there
    unnoticed. Now each one must be listed, justified and paid for in the
    ``test`` extra.
    """

    offenders: list[str] = []
    for path, modules in _imports_by_file(_TESTS_ROOT).items():
        for module in modules:
            if _is_kernel_internal(module) and _pinned_root(module) is None:
                offenders.append(f"{path.relative_to(_TESTS_ROOT)} -> {module}")
    assert not offenders, (
        "the embodiment test suite reached into a kernel internal that is not "
        "pinned in _TEST_ONLY_KERNEL_IMPORTS. Drive the behaviour through the "
        "vz-runtime facade, or pin the module here AND declare its wheel in "
        "the package's 'test' extra:\n" + "\n".join(offenders)
    )


def test_pinned_test_only_kernel_imports_are_all_still_used() -> None:
    """Anti-rot: the pin list may never be broader than reality."""

    used: set[str] = set()
    for modules in _imports_by_file(_TESTS_ROOT).values():
        for module in modules:
            pinned = _pinned_root(module)
            if pinned is not None:
                used.add(pinned)
    stale = sorted(set(_TEST_ONLY_KERNEL_IMPORTS) - used)
    assert not stale, (
        "these kernel internals are pinned as test-only but no test imports "
        "them any more; drop the pin (and the matching 'test' extra entry) so "
        "the boundary tightens instead of drifting:\n" + "\n".join(stale)
    )


def test_tests_only_import_declared_volvence_zero_surfaces() -> None:
    """Positive guard for the suite: facade prefixes + pinned internals only."""

    allowed = _ALLOWED_FACADE_PREFIXES + _TESTS_EXTRA_FACADE_PREFIXES
    unexpected: list[str] = []
    for path, modules in _imports_by_file(_TESTS_ROOT).items():
        for module in modules:
            if not module.startswith("volvence_zero"):
                continue
            if module.startswith(allowed) or _pinned_root(module) is not None:
                continue
            unexpected.append(f"{path.relative_to(_TESTS_ROOT)} -> {module}")
    assert not unexpected, (
        "the embodiment test suite imported an undeclared volvence_zero "
        "module; it is either a public facade surface (add the prefix) or a "
        "kernel internal (pin it and declare its wheel):\n"
        + "\n".join(unexpected)
    )


def test_test_only_kernel_wheels_are_declared_as_test_extra() -> None:
    """The dependency closure must be honest, not 'the monorepo has it'.

    Every wheel the suite reaches into lands in ``[project.optional-dependencies]
    .test`` — never in the shipped ``[project].dependencies``.
    """

    pyproject = _load_pyproject()
    project = pyproject["project"]
    shipped = _requirement_names(project["dependencies"])
    test_extra = _requirement_names(
        project["optional-dependencies"]["test"]
    )

    required = set(_TEST_ONLY_KERNEL_IMPORTS.values())
    missing = sorted(required - test_extra)
    assert not missing, (
        "kernel wheels the test suite genuinely imports are not declared in "
        "the 'test' extra; declare them so the dependency closure is honest:\n"
        + "\n".join(missing)
    )

    leaked = sorted(required & shipped)
    assert not leaked, (
        "a kernel-internal wheel leaked into the SHIPPED dependencies of "
        "vz-embodiment-ant; the library depends only on vz-contracts / "
        "vz-substrate / vz-runtime:\n" + "\n".join(leaked)
    )
