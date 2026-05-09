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
    "vz-memory": frozenset(
        {"runtime", "learned_update", "substrate", "social_cognition"}
    ),
    # Slice C (2026-05-03): vz-cognition no longer hosts application-tier
    # dataclasses. The ``evaluation`` layer consumes a structural
    # ``Protocol`` surface (``volvence_zero.application_readouts`` in
    # vz-contracts) instead, so the kernel never imports concrete
    # application schema. Owners live in vz-application as before.
    "vz-cognition": frozenset(
        {
            "runtime", "learned_update", "temporal_types", "substrate", "memory",
            "application_readouts", "social_cognition", "environment",
            # rupture_state owner (Rupture and Repair, M1) consumes
            # DialogueExternalOutcomeSnapshot published by vz-runtime's
            # DialogueExternalOutcomeModule. The snapshot type itself lives
            # in vz-contracts' dialogue_trace module.
            "dialogue_trace",
            # Stub semantic embedding SSOT (known-debts #3 closure):
            # dual_track / evaluation consume the canonical character-level
            # token+hash embedding from vz-contracts.
            "semantic_embedding",
            # Wave E1 (debt #10B item 3): typed counters for LLM-backed
            # proposal runtimes. The immutable
            # :class:`LLMProposalAttemptCounters` lives in vz-contracts so
            # ToM / common-ground owner snapshots in vz-cognition can
            # surface it without a circular dependency on lifeform layers.
            "llm_proposal_diagnostics",
        }
    ),
    "vz-application": frozenset(
        {
            "runtime", "learned_update", "temporal_types", "substrate", "memory",
            # everything in vz-cognition:
            "dual_track", "evaluation", "credit", "regime", "prediction",
            "reflection", "semantic_state", "rupture_state",
            "social", "social_cognition", "environment",
            "dialogue_trace",
            # Stub semantic embedding SSOT (known-debts #3 closure).
            "semantic_embedding",
        }
    ),
    "vz-temporal": frozenset(
        {
            "runtime", "learned_update", "temporal_types", "substrate", "memory",
            # everything in vz-cognition:
            "dual_track", "evaluation", "credit", "regime", "prediction",
            "reflection", "semantic_state", "rupture_state",
            "social", "social_cognition", "environment",
            # vz-application:
            "application",
            "dialogue_trace",
        }
    ),
    "vz-runtime": frozenset(
        {
            "runtime", "learned_update", "substrate", "memory", "dialogue_trace",
            # everything in vz-cognition:
            "dual_track", "evaluation", "credit", "regime", "prediction",
            "reflection", "semantic_state", "rupture_state",
            # interlocutor SHADOW owner (W2 of ssot-cleanup-p0-p4) is
            # registered into the runtime so consumers read one snapshot.
            "interlocutor",
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


def _all_dlaas_platform_files() -> list[tuple[str, pathlib.Path]]:
    out: list[tuple[str, pathlib.Path]] = []
    for wheel_dir in sorted(PACKAGES_ROOT.glob("dlaas-platform-*")):
        for py in _python_files(wheel_dir / "src"):
            out.append((wheel_dir.name, py))
    return out


# DLaaS platform tier may consume only these kernel/lifeform surfaces. The
# list is deliberately tiny: the platform is meant to talk to the kernel
# exclusively through `vz-contracts` snapshot types, the
# `lifeform-core.Lifeform` facade, the `lifeform-service` HTTP / SessionManager
# surface, and (later) the `lifeform-affordance` / `lifeform-ingestion` /
# `lifeform-evolution` public modules.
#
# Any import path under ``volvence_zero.{cognition,memory,temporal,
# substrate,application,runtime}`` is forbidden — those are kernel
# internals. ``vz-contracts`` lives under ``volvence_zero`` but only its
# top-level facade is allowed.
DLAAS_PLATFORM_ALLOWED_VZ_PREFIXES: frozenset[str] = frozenset(
    {
        # vz-contracts top-level facade only. Sub-packages like
        # ``volvence_zero.runtime.kernel`` are kernel internals.
        "volvence_zero",
        "volvence_zero.thinking",  # vz-contracts subpackage (frozen task/artifact types)
        "volvence_zero.affordance",  # vz-contracts subpackage (descriptor schema)
        "volvence_zero.environment",  # vz-contracts subpackage (event/outcome contracts)
        "volvence_zero.social_cognition",  # vz-contracts subpackage (typed snapshots)
        "volvence_zero.semantic_embedding",  # vz-contracts subpackage (stub embedding)
        "volvence_zero.dialogue_trace",  # vz-contracts subpackage (typed outcome enums)
        "volvence_zero.temporal_types",  # vz-contracts subpackage (controller types)
        "volvence_zero.application_readouts",  # vz-contracts subpackage (typed protocol)
        "volvence_zero.learned_update",  # vz-contracts subpackage (typed proposal)
        "volvence_zero.llm_proposal_diagnostics",  # vz-contracts subpackage
    }
)
DLAAS_PLATFORM_FORBIDDEN_VZ_SUBPACKAGES: frozenset[str] = frozenset(
    {
        "cognition",
        "memory",
        "temporal",
        "substrate",
        "application",
        "runtime",
        "agent",
        "credit",
        "dual_track",
        "evaluation",
        "regime",
        "prediction",
        "reflection",
        "semantic_state",
        "rupture_state",
        "interlocutor",
        "social",
        "brain",
        "integration",
        "internal_rl",
        "joint_loop",
        "planning",
    }
)
DLAAS_PLATFORM_ALLOWED_LIFEFORM_PREFIXES: frozenset[str] = frozenset(
    {
        "lifeform_core",
        "lifeform_service",
        "lifeform_affordance",
        "lifeform_ingestion",
        "lifeform_evolution",
        "lifeform_expression",
        "lifeform_thinking",
    }
)


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


_EVALUATION_BACKBONE_ALLOWED_SYMBOLS: frozenset[str] = frozenset(
    {"EvaluationBackbone", "EvaluationModule", "_feature_surface_snapshot"}
)


def _find_evaluation_backbone_typed_imports(py_file: pathlib.Path) -> list[tuple[int, str]]:
    """Return ``(lineno, name)`` pairs that violate known-debts #4 closure.

    A violation is any ``from volvence_zero.evaluation.backbone import <name>``
    where ``<name>`` is not in ``_EVALUATION_BACKBONE_ALLOWED_SYMBOLS``. Pure
    types must be imported from the ``volvence_zero.evaluation`` facade
    instead, so the implementation module ``backbone.py`` is only loaded
    when consumers actually need its concrete classes.
    """
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError as exc:  # pragma: no cover
        pytest.fail(f"Cannot parse {py_file}: {exc}")
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "volvence_zero.evaluation.backbone":
            continue
        for alias in node.names:
            if alias.name in _EVALUATION_BACKBONE_ALLOWED_SYMBOLS:
                continue
            violations.append((node.lineno, alias.name))
    return violations


@pytest.mark.parametrize(
    ("wheel", "py_file"),
    _all_kernel_files(),
    ids=lambda v: v.name if isinstance(v, pathlib.Path) else str(v),
)
def test_kernel_imports_evaluation_types_via_facade(wheel: str, py_file: pathlib.Path) -> None:
    """Pure evaluation types must come from the ``evaluation`` facade.

    Closes known-debts #4: pure type consumers should not bind to
    ``evaluation.backbone`` (which carries the full ``EvaluationBackbone``
    implementation). Only the implementation symbols
    (``EvaluationBackbone`` / ``EvaluationModule``) and the internal hook
    ``_feature_surface_snapshot`` may import from that path directly.
    """
    if py_file.name == "__init__.py" and py_file.parent.name == "evaluation":
        return  # facade itself re-exports the implementations
    violations = _find_evaluation_backbone_typed_imports(py_file)
    if not violations:
        return
    rendered = ", ".join(f"{name}@L{lineno}" for lineno, name in violations)
    pytest.fail(
        f"{py_file.relative_to(REPO_ROOT)} imports type(s) [{rendered}] from "
        f"volvence_zero.evaluation.backbone. Pure evaluation types must be "
        f"imported from volvence_zero.evaluation (facade) instead. "
        f"Only {sorted(_EVALUATION_BACKBONE_ALLOWED_SYMBOLS)} may use the "
        f"backbone path. See known-debts.md #4."
    )


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


# ---------------------------------------------------------------------------
# Lifeform vertical isolation
#
# Verticals (`lifeform-domain-*`) compile reviewed structured artifacts into
# existing kernel application owners. Two parallel verticals that do not
# share a parent vertical wheel must not import each other directly, or the
# "compiler is the only writer of this vertical's bundle" invariant breaks
# (R8 SSOT). For now this is enforced for the figure / character pair; the
# helper below extends to any newly-added pair.
# ---------------------------------------------------------------------------


PARALLEL_VERTICAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("lifeform-domain-figure", "lifeform-domain-character"),
)


@pytest.mark.parametrize(("vertical_a", "vertical_b"), PARALLEL_VERTICAL_PAIRS)
def test_parallel_verticals_do_not_cross_import(
    vertical_a: str, vertical_b: str,
) -> None:
    """Parallel verticals must not import each other.

    Each vertical owns its own reviewed bundle (CharacterSoulProfile /
    DomainExperiencePackage for character, FigureArtifactBundle for
    figure). Cross-importing would create a second owner of the bundle's
    schema and break the rollback story.
    """
    package_a_root = PACKAGES_ROOT / vertical_a / "src"
    package_b_root = PACKAGES_ROOT / vertical_b / "src"
    if not package_a_root.exists():
        pytest.skip(f"vertical {vertical_a} has not landed yet")
    if not package_b_root.exists():
        pytest.skip(f"vertical {vertical_b} has not landed yet")
    pkg_a_module = vertical_a.replace("-", "_")
    pkg_b_module = vertical_b.replace("-", "_")

    def _check(src_root: pathlib.Path, forbidden_module: str, forbidden_wheel: str) -> None:
        for py_file in _python_files(src_root):
            for module in _module_level_imports(py_file):
                if module == forbidden_module or module.startswith(f"{forbidden_module}."):
                    pytest.fail(
                        f"{py_file.relative_to(REPO_ROOT)} imports "
                        f"'{module}': vertical wheel must not import "
                        f"parallel vertical '{forbidden_wheel}' (R8 SSOT)."
                    )

    _check(package_a_root, pkg_b_module, vertical_b)
    _check(package_b_root, pkg_a_module, vertical_a)


def test_figure_vertical_does_not_import_dlaas_platform() -> None:
    """The figure vertical must not import dlaas-platform-* internals.

    DLaaS sits above the lifeform tier and consumes the vertical's
    bundle through public symbols. The vertical's own modules must
    not reach upward into the platform tier — that would invert the
    dependency direction encoded in the wheel layering.
    """
    figure_src = PACKAGES_ROOT / "lifeform-domain-figure" / "src"
    if not figure_src.exists():
        pytest.skip("lifeform-domain-figure has not landed yet")
    for py_file in _python_files(figure_src):
        for module in _module_level_imports(py_file):
            if module.startswith("dlaas_platform_") or module == "dlaas_platform":
                pytest.fail(
                    f"{py_file.relative_to(REPO_ROOT)} imports "
                    f"'{module}': lifeform-domain-figure must not "
                    f"depend on the platform tier."
                )


# ---------------------------------------------------------------------------
# DLaaS platform-tier boundary tests (third tier; see docs/specs/dlaas-platform.md)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wheel", "py_file"),
    _all_dlaas_platform_files(),
    ids=lambda v: v.name if isinstance(v, pathlib.Path) else str(v),
)
def test_dlaas_platform_does_not_import_kernel_internals(
    wheel: str, py_file: pathlib.Path,
) -> None:
    """Platform tier may only consume public kernel facade + vz-contracts.

    The forbidden list is the union of every kernel sub-package that
    holds cognitive / memory / temporal state. Allowed paths are the
    typed contract surfaces under ``vz-contracts``.

    Why this matters (R8 + the dlaas-platform.md invariants): if the
    platform tier reaches into ``volvence_zero.cognition.regime`` it
    becomes a second owner of that state, and the "vz-* diff = 0"
    promise made to integrators stops being checkable.
    """
    for module in _module_level_imports(py_file):
        sub = _vz_subpackage(module)
        if sub is None:
            continue
        # Allow the bare facade and the explicit vz-contracts subpackages.
        if module in DLAAS_PLATFORM_ALLOWED_VZ_PREFIXES:
            continue
        # Allow deeper paths under those facades (e.g.
        # ``volvence_zero.thinking.models``).
        if any(
            module.startswith(prefix + ".")
            for prefix in DLAAS_PLATFORM_ALLOWED_VZ_PREFIXES
            if prefix != "volvence_zero"
        ):
            continue
        if sub in DLAAS_PLATFORM_FORBIDDEN_VZ_SUBPACKAGES:
            pytest.fail(
                f"{py_file.relative_to(REPO_ROOT)} imports '{module}': "
                f"dlaas-platform tier may not depend on kernel sub-package "
                f"'volvence_zero.{sub}'. Use vz-contracts snapshot types or "
                f"the lifeform-core / lifeform-service public facade instead. "
                f"See docs/specs/dlaas-platform.md (invariants 1-2)."
            )


@pytest.mark.parametrize(
    ("wheel", "py_file"),
    _all_dlaas_platform_files(),
    ids=lambda v: v.name if isinstance(v, pathlib.Path) else str(v),
)
def test_dlaas_platform_does_not_import_lifeform_domain_internals(
    wheel: str, py_file: pathlib.Path,
) -> None:
    """Platform may only consume the small allowlist of lifeform public surfaces.

    Domain wheels (``lifeform_domain_emogpt`` / ``lifeform_domain_coding``
    / ``lifeform_domain_character``) are vertical-specific implementation
    details: the platform tier resolves a vertical via
    ``lifeform-service.verticals`` (which itself imports them through a
    lazy registry). Direct imports from the platform tier would couple
    multi-tenant routing to one vertical and break the "vz-* diff = 0,
    domain wheels untouched" promise.
    """
    for module in _module_level_imports(py_file):
        if not module.startswith("lifeform_") and module != "lifeform":
            continue
        # Top-level allowlist hit.
        if module in DLAAS_PLATFORM_ALLOWED_LIFEFORM_PREFIXES:
            continue
        # Sub-module under an allowed top-level package.
        if any(
            module.startswith(prefix + ".")
            for prefix in DLAAS_PLATFORM_ALLOWED_LIFEFORM_PREFIXES
        ):
            continue
        pytest.fail(
            f"{py_file.relative_to(REPO_ROOT)} imports '{module}': "
            f"dlaas-platform tier may only consume the public lifeform "
            f"surfaces {sorted(DLAAS_PLATFORM_ALLOWED_LIFEFORM_PREFIXES)}. "
            f"Domain-specific wheels (lifeform_domain_*) must go through "
            f"the lifeform-service vertical registry. See "
            f"docs/specs/dlaas-platform.md (invariant 3)."
        )


@pytest.mark.parametrize(
    ("wheel", "py_file"),
    _all_kernel_files(),
    ids=lambda v: v.name if isinstance(v, pathlib.Path) else str(v),
)
def test_kernel_does_not_import_dlaas_platform(
    wheel: str, py_file: pathlib.Path,
) -> None:
    """Kernel must never import platform tier (one-way dependency)."""
    for module in _module_level_imports(py_file):
        if module.startswith("dlaas_platform_") or module == "dlaas_platform":
            pytest.fail(
                f"{py_file.relative_to(REPO_ROOT)} imports '{module}': "
                f"kernel wheel '{wheel}' must not depend on the dlaas-platform "
                f"tier. The dependency direction is one-way (platform → "
                f"lifeform → kernel)."
            )


@pytest.mark.parametrize(
    ("wheel", "py_file"),
    _all_lifeform_files(),
    ids=lambda v: v.name if isinstance(v, pathlib.Path) else str(v),
)
def test_lifeform_does_not_import_dlaas_platform(
    wheel: str, py_file: pathlib.Path,
) -> None:
    """Lifeform tier must never import platform tier (one-way dependency)."""
    for module in _module_level_imports(py_file):
        if module.startswith("dlaas_platform_") or module == "dlaas_platform":
            pytest.fail(
                f"{py_file.relative_to(REPO_ROOT)} imports '{module}': "
                f"lifeform wheel '{wheel}' must not depend on the dlaas-platform "
                f"tier. Platform composes lifeform, not the reverse."
            )
