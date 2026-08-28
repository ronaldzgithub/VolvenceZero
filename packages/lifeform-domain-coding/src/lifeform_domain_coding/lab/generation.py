"""Deterministic synthetic evolving-repo generator (coding-lab Packet 0).

Generates a small but real Python package plus a pytest suite. The suite
encodes *latent invariants*: contracts that are never mentioned in task
descriptions and are only discoverable by reading the tests or by
breaking them. Each invariant is mechanically enforced by named
regression tests, so episode outcomes are settled by the environment
(pytest), never by a judge.

Determinism contract: the same :class:`EnvSpec` always produces a
byte-identical tree (verified via :func:`compute_tree_hash`). All
randomness flows through ``random.Random`` seeded from the spec.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from random import Random

GENERATOR_VERSION = "coding-lab-gen.v1"

INVARIANT_CONFIG_CASE = "config_case_sensitive"
INVARIANT_INDEX_IDEMPOTENT = "store_index_idempotent"
INVARIANT_ROUND_HALF_UP = "order_total_half_up"
INVARIANT_REPORT_ORDER = "report_insertion_order"
INVARIANT_HIDDEN_CONSUMER = "pricing_hidden_consumer"

ALL_INVARIANT_IDS: tuple[str, ...] = (
    INVARIANT_CONFIG_CASE,
    INVARIANT_INDEX_IDEMPOTENT,
    INVARIANT_ROUND_HALF_UP,
    INVARIANT_REPORT_ORDER,
    INVARIANT_HIDDEN_CONSUMER,
)

# --- House conventions (difficulty knob, 2026-08-13) -----------------------
#
# Latent invariants above are enforced by regression tests that live IN
# the workspace: a careful hand can run the suite before submitting and
# self-check them all — which is exactly how the 2026-08-12 API
# calibration saturated at 0.94 pass rate. House conventions close that
# loop: they are owner preferences enforced ONLY by hidden acceptance
# tests injected at oracle time. The repository carries zero signal, the
# in-episode `run_test` cannot reveal them, and they recur across
# episodes — so cross-episode memory is the one legitimate channel to
# stop violating them. That makes them the memory-realisable difficulty
# knob: they lower a stateless hand's pass rate into the oracle band
# while giving the remembering arms real headroom.

CONVENTION_EXPORT_ALL = "convention_export_all"
CONVENTION_ANNOTATED_SIGNATURE = "convention_annotated_signature"
CONVENTION_DOCSTRING_CONTRACT = "convention_docstring_contract"
CONVENTION_SYMBOL_OWNER = "convention_symbol_owner"

#: Every convention below is independent and composable: each is enforced by its
#: own hidden acceptance test and satisfied by its own edit, so a spec may
#: activate any subset. ``convention_export_all`` must keep its exact historical
#: bytes — the 2026-08-13 Packet 2 formal chains replay through it.
ALL_CONVENTION_IDS: tuple[str, ...] = (
    CONVENTION_EXPORT_ALL,
    CONVENTION_ANNOTATED_SIGNATURE,
    CONVENTION_DOCSTRING_CONTRACT,
    CONVENTION_SYMBOL_OWNER,
)

CONVENTION_DESCRIPTIONS: dict[str, str] = {
    CONVENTION_EXPORT_ALL: (
        "house style: every new public symbol must be registered in its "
        "module's __all__ (unstated owner preference; enforced only by "
        "hidden acceptance tests)"
    ),
    CONVENTION_ANNOTATED_SIGNATURE: (
        "house style: every new public function must annotate all of its "
        "parameters and its return type (unstated owner preference; enforced "
        "only by hidden acceptance tests)"
    ),
    CONVENTION_DOCSTRING_CONTRACT: (
        "house style: every new public function's docstring must carry a line "
        "beginning 'Contract:' (unstated owner preference; enforced only by "
        "hidden acceptance tests)"
    ),
    CONVENTION_SYMBOL_OWNER: (
        "house style: every new public symbol must be registered in its "
        "module's _SYMBOL_OWNERS map (unstated owner preference; enforced only "
        "by hidden acceptance tests)"
    ),
}

_ITEM_POOL: tuple[str, ...] = (
    "widget",
    "gadget",
    "doohickey",
    "gizmo",
    "sprocket",
    "flange",
    "bracket",
    "grommet",
)

# Cases where ``round(x, 2)`` (banker's/binary-float rounding) diverges
# from decimal HALF_UP — the mechanical teeth of the rounding invariant.
_HALF_UP_CASES: tuple[tuple[str, str], ...] = (
    ("2.675", "2.68"),
    ("1.005", "1.01"),
    ("0.125", "0.13"),
    ("7.865", "7.87"),
)


@dataclass(frozen=True)
class EnvSpec:
    """Frozen recipe for one environment instance."""

    env_seed: int
    package_name: str = "mv_app"
    param_offset: int = 0
    invariant_ids: tuple[str, ...] = ALL_INVARIANT_IDS
    #: Active house conventions (see module docstring above). Default
    #: empty keeps every pre-2026-08-13 spec, tree hash and sealed
    #: manifest bit-identical.
    convention_ids: tuple[str, ...] = ()
    generator_version: str = GENERATOR_VERSION

    def __post_init__(self) -> None:
        if not self.package_name.isidentifier():
            raise ValueError(f"package_name must be a Python identifier, got {self.package_name!r}")
        unknown = set(self.invariant_ids) - set(ALL_INVARIANT_IDS)
        if unknown:
            raise ValueError(f"unknown invariant ids: {sorted(unknown)!r}")
        unknown_conventions = set(self.convention_ids) - set(ALL_CONVENTION_IDS)
        if unknown_conventions:
            raise ValueError(f"unknown convention ids: {sorted(unknown_conventions)!r}")
        if self.generator_version != GENERATOR_VERSION:
            raise ValueError(
                f"spec pinned generator {self.generator_version!r} but this module is {GENERATOR_VERSION!r}"
            )


@dataclass(frozen=True)
class EnvParams:
    """Seeded value parameters shared by module sources, tests and tasks."""

    discount_table: tuple[tuple[str, float], ...]
    rounding_case_input: str
    rounding_case_expected: str
    workflow_items: tuple[tuple[str, int], ...]
    workflow_order_lines: tuple[tuple[str, int, float], ...]
    workflow_discount_code: str
    workflow_expected_total: float


@dataclass(frozen=True)
class LatentInvariant:
    """One latent contract and the regression tests that enforce it."""

    invariant_id: str
    description: str
    regression_tests: tuple[tuple[str, str], ...]  # (test file relpath, test function name)


@dataclass(frozen=True)
class GeneratedEnvironment:
    """Manifest of one generated tree."""

    spec: EnvSpec
    params: EnvParams
    root: pathlib.Path
    tree_hash: str
    file_hashes: tuple[tuple[str, str], ...]
    invariants: tuple[LatentInvariant, ...]


def derive_params(spec: EnvSpec) -> EnvParams:
    """Derive all seeded value parameters for ``spec`` (pure function)."""

    rng = Random(spec.env_seed * 1_000_003 + spec.param_offset * 97)
    discount_table = (
        ("WELCOME", rng.choice((0.05, 0.10))),
        ("VIP", rng.choice((0.20, 0.25, 0.30))),
        ("SEASON", rng.choice((0.12, 0.15))),
    )
    rounding_input, rounding_expected = _HALF_UP_CASES[rng.randrange(len(_HALF_UP_CASES))]

    item_ids = rng.sample(_ITEM_POOL, 4)
    if list(item_ids) == sorted(item_ids):
        item_ids[0], item_ids[1] = item_ids[1], item_ids[0]
    workflow_items = tuple((item_id, rng.randint(3, 19)) for item_id in item_ids)

    order_lines = tuple(
        (item_id, rng.randint(1, min(2, qty)), float(rng.choice((3.30, 4.10, 5.25, 7.45, 9.95))))
        for item_id, qty in workflow_items[:2]
    )
    discount_code = rng.choice(tuple(name for name, _ in discount_table))
    rate = dict(discount_table)[discount_code]
    raw_total = sum(qty * price for _, qty, price in order_lines)
    expected_total = _decimal_half_up(raw_total * (1.0 - rate))
    return EnvParams(
        discount_table=discount_table,
        rounding_case_input=rounding_input,
        rounding_case_expected=rounding_expected,
        workflow_items=workflow_items,
        workflow_order_lines=order_lines,
        workflow_discount_code=discount_code,
        workflow_expected_total=expected_total,
    )


def _decimal_half_up(value: float, places: int = 2) -> float:
    quant = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Module sources
# ---------------------------------------------------------------------------


def _source_util(spec: EnvSpec) -> str:
    return '''"""Shared numeric and identifier helpers."""

from decimal import ROUND_HALF_UP, Decimal


def round_half_up(value, places=2):
    """Round ``value`` half-up to ``places`` decimal places.

    Money amounts in this codebase round half-up (2.675 -> 2.68), which
    is NOT what built-in ``round`` does for binary floats.
    """
    quant = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def normalize_id(item_id):
    """Trim surrounding whitespace; identifiers stay case-sensitive."""
    return item_id.strip()
'''


def _source_config(spec: EnvSpec) -> str:
    return '''"""Plain KEY=VALUE configuration parsing."""


def load_config(text):
    """Parse ``KEY=VALUE`` lines into a dict.

    Blank lines and ``#`` comments are skipped. Later duplicates of the
    same key win.
    """
    entries = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"invalid config line: {line!r}")
        key, _, value = stripped.partition("=")
        entries[key.strip()] = value.strip()
    return entries
'''


def _source_store(spec: EnvSpec) -> str:
    return f'''"""In-memory inventory store."""

from {spec.package_name}.util import normalize_id


class Store:
    """Quantity ledger keyed by item id."""

    def __init__(self):
        self._quantities = {{}}
        self._insertion_order = []
        self._index = {{}}

    def add(self, item_id, qty):
        item_id = normalize_id(item_id)
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {{qty!r}}")
        if item_id not in self._quantities:
            self._insertion_order.append(item_id)
            self._quantities[item_id] = 0
        self._quantities[item_id] += qty
        self.rebuild_index()

    def remove(self, item_id, qty):
        item_id = normalize_id(item_id)
        if item_id not in self._quantities:
            raise KeyError(f"unknown item: {{item_id!r}}")
        if qty <= 0 or qty > self._quantities[item_id]:
            raise ValueError(f"cannot remove {{qty!r}} of {{item_id!r}}")
        self._quantities[item_id] -= qty

    def quantity(self, item_id):
        return self._quantities[normalize_id(item_id)]

    def rebuild_index(self):
        self._index = {{item: position for position, item in enumerate(self._insertion_order)}}

    def index_of(self, item_id):
        return self._index[normalize_id(item_id)]

    def items(self):
        """Items as ``(item_id, qty)`` pairs in insertion order."""
        return [(item_id, self._quantities[item_id]) for item_id in self._insertion_order]
'''


def _source_pricing(spec: EnvSpec, params: EnvParams) -> str:
    table_lines = ",\n    ".join(f'"{name}": {value!r}' for name, value in params.discount_table)
    return f'''"""Discount codes and price arithmetic."""

from {spec.package_name}.util import round_half_up

DISCOUNT_TABLE = {{
    {table_lines},
}}


def apply_discount(amount, code=None):
    """Apply a discount code to ``amount``; result rounds half-up."""
    if code is None:
        return round_half_up(amount)
    if code not in DISCOUNT_TABLE:
        raise KeyError(f"unknown discount code: {{code!r}}")
    return round_half_up(amount * (1.0 - DISCOUNT_TABLE[code]))
'''


def _source_orders(spec: EnvSpec) -> str:
    return f'''"""Order placement against a store."""

from {spec.package_name}.pricing import apply_discount


def place_order(store, lines, discount_code=None):
    """Place an order of ``(item_id, qty, unit_price)`` lines.

    Removes stock from ``store`` and returns an order dict whose total
    has the discount applied.
    """
    if not lines:
        raise ValueError("order must contain at least one line")
    total = 0.0
    for item_id, qty, unit_price in lines:
        store.remove(item_id, qty)
        total += qty * unit_price
    return {{"lines": [tuple(line) for line in lines], "total": apply_discount(total, discount_code)}}
'''


def _source_report(spec: EnvSpec) -> str:
    return f'''"""Plain-text reporting consumed by downstream tooling."""


def render_report(store):
    """Render inventory as text.

    Line format and item ordering are consumed verbatim by external
    tooling; items appear in insertion order.
    """
    lines = ["INVENTORY REPORT"]
    for item_id, qty in store.items():
        lines.append(f"{{item_id}}: qty={{qty}}")
    return "\\n".join(lines)


def discount_catalog_line():
    """One-line catalog of available discount codes."""
    from {spec.package_name} import pricing

    codes = ",".join(sorted(pricing.DISCOUNT_TABLE))
    return f"codes: {{codes}}"
'''


def module_sources(spec: EnvSpec, params: EnvParams) -> dict[str, str]:
    """All package files as ``relative path -> content``."""

    pkg = spec.package_name
    return {
        ".gitignore": "__pycache__/\n*.pyc\n.pytest_cache/\n",
        f"{pkg}/__init__.py": f'"""Generated application package ({GENERATOR_VERSION})."""\n',
        f"{pkg}/util.py": _source_util(spec),
        f"{pkg}/config.py": _source_config(spec),
        f"{pkg}/store.py": _source_store(spec),
        f"{pkg}/pricing.py": _source_pricing(spec, params),
        f"{pkg}/orders.py": _source_orders(spec),
        f"{pkg}/report.py": _source_report(spec),
    }


# ---------------------------------------------------------------------------
# Test sources (pristine tree; the oracle regenerates these at eval time)
# ---------------------------------------------------------------------------


def _tests_fast(spec: EnvSpec, params: EnvParams) -> dict[str, str]:
    pkg = spec.package_name
    first_item = params.workflow_items[0][0]
    fast_config = f'''from {pkg}.config import load_config


def test_load_config_basic():
    parsed = load_config("A=1\\n# comment\\n\\nB = two")
    assert parsed == {{"A": "1", "B": "two"}}


def test_load_config_rejects_bad_line():
    try:
        load_config("not-a-pair")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
'''
    fast_store = f'''from {pkg}.store import Store


def test_add_and_quantity():
    store = Store()
    store.add("{first_item}", 3)
    assert store.quantity("{first_item}") == 3


def test_remove_guards():
    store = Store()
    store.add("{first_item}", 2)
    try:
        store.remove("{first_item}", 5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
'''
    fast_pricing = f'''from {pkg}.pricing import DISCOUNT_TABLE, apply_discount


def test_apply_discount_known_code():
    code = sorted(DISCOUNT_TABLE)[0]
    rate = DISCOUNT_TABLE[code]
    assert apply_discount(100.0, code) == round(100.0 * (1.0 - rate), 2)


def test_apply_discount_unknown_code():
    try:
        apply_discount(10.0, "NOPE")
    except KeyError:
        return
    raise AssertionError("expected KeyError")
'''
    return {
        "tests/fast/test_config.py": fast_config,
        "tests/fast/test_store.py": fast_store,
        "tests/fast/test_pricing.py": fast_pricing,
    }


def _tests_full(spec: EnvSpec, params: EnvParams) -> dict[str, str]:
    pkg = spec.package_name
    add_calls = "\n    ".join(f'store.add("{item}", {qty})' for item, qty in params.workflow_items)
    expected_report_lines = ['"INVENTORY REPORT"'] + [
        f'"{item}: qty={qty}"' for item, qty in params.workflow_items
    ]
    expected_report = ",\n        ".join(expected_report_lines)
    order_lines_literal = ", ".join(
        f'("{item}", {qty}, {price!r})' for item, qty, price in params.workflow_order_lines
    )
    remaining = dict(params.workflow_items)
    for item, qty, _price in params.workflow_order_lines:
        remaining[item] = remaining[item] - qty
    remaining_checks = "\n    ".join(
        f'assert store.quantity("{item}") == {qty}' for item, qty in remaining.items()
    )

    invariants = f'''"""Regression tests for cross-module contracts.

These encode load-bearing behaviour that downstream tooling relies on.
They are intentionally strict; loosening them breaks consumers.
"""

from {pkg}.config import load_config
from {pkg}.report import discount_catalog_line, render_report
from {pkg}.store import Store
from {pkg}.util import round_half_up


def test_config_keys_case_sensitive():
    parsed = load_config("Alpha=1\\nalpha=2")
    assert parsed["Alpha"] == "1"
    assert parsed["alpha"] == "2"
    assert len(parsed) == 2


def test_rebuild_index_idempotent():
    store = Store()
    {add_calls}
    store.rebuild_index()
    once = dict(store._index)
    store.rebuild_index()
    assert dict(store._index) == once
    assert [store.index_of(item) for item, _ in store.items()] == list(range(len(store.items())))


def test_order_total_half_up_rounding():
    assert round_half_up({params.rounding_case_input}) == {params.rounding_case_expected}


def test_report_insertion_order_preserved():
    store = Store()
    {add_calls}
    rendered = render_report(store)
    assert rendered.splitlines() == [
        {expected_report},
    ]


def test_discount_catalog_hidden_consumer():
    line = discount_catalog_line()
    assert line == "codes: SEASON,VIP,WELCOME"
'''

    workflow = f'''"""End-to-end workflow pinning exact totals and report text."""

from {pkg}.orders import place_order
from {pkg}.report import render_report
from {pkg}.store import Store


def test_full_workflow_totals_and_report():
    store = Store()
    {add_calls}
    order = place_order(store, [{order_lines_literal}], discount_code="{params.workflow_discount_code}")
    assert order["total"] == {params.workflow_expected_total!r}
    {remaining_checks}
    rendered = render_report(store)
    assert rendered.startswith("INVENTORY REPORT")
'''
    return {
        "tests/full/test_invariants.py": invariants,
        "tests/full/test_workflow.py": workflow,
    }


_CONFTEST = '''import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
'''


def test_sources(spec: EnvSpec, params: EnvParams) -> dict[str, str]:
    """Pristine test tree as ``relative path -> content`` (incl. conftest)."""

    sources: dict[str, str] = {"conftest.py": _CONFTEST}
    sources.update(_tests_fast(spec, params))
    sources.update(_tests_full(spec, params))
    return sources


def latent_invariants(spec: EnvSpec) -> tuple[LatentInvariant, ...]:
    """Registry of latent invariants active in this environment."""

    catalog = {
        INVARIANT_CONFIG_CASE: LatentInvariant(
            invariant_id=INVARIANT_CONFIG_CASE,
            description="config keys are case-sensitive; parsing must not normalise case",
            regression_tests=(("tests/full/test_invariants.py", "test_config_keys_case_sensitive"),),
        ),
        INVARIANT_INDEX_IDEMPOTENT: LatentInvariant(
            invariant_id=INVARIANT_INDEX_IDEMPOTENT,
            description="Store.rebuild_index must be idempotent",
            regression_tests=(("tests/full/test_invariants.py", "test_rebuild_index_idempotent"),),
        ),
        INVARIANT_ROUND_HALF_UP: LatentInvariant(
            invariant_id=INVARIANT_ROUND_HALF_UP,
            description="money rounds half-up via util.round_half_up, not builtin round()",
            regression_tests=(
                ("tests/full/test_invariants.py", "test_order_total_half_up_rounding"),
                ("tests/full/test_workflow.py", "test_full_workflow_totals_and_report"),
            ),
        ),
        INVARIANT_REPORT_ORDER: LatentInvariant(
            invariant_id=INVARIANT_REPORT_ORDER,
            description="report lists items in insertion order with exact line format",
            regression_tests=(("tests/full/test_invariants.py", "test_report_insertion_order_preserved"),),
        ),
        INVARIANT_HIDDEN_CONSUMER: LatentInvariant(
            invariant_id=INVARIANT_HIDDEN_CONSUMER,
            description="report late-imports pricing.DISCOUNT_TABLE; the symbol is a public contract",
            regression_tests=(
                ("tests/full/test_invariants.py", "test_discount_catalog_hidden_consumer"),
                ("tests/full/test_workflow.py", "test_full_workflow_totals_and_report"),
            ),
        ),
    }
    return tuple(catalog[invariant_id] for invariant_id in spec.invariant_ids)


# ---------------------------------------------------------------------------
# Tree materialisation and hashing
# ---------------------------------------------------------------------------


def _write_tree(root: pathlib.Path, files: dict[str, str]) -> None:
    for relative_path, content in sorted(files.items()):
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def compute_tree_hash(root: pathlib.Path) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Hash every regular file under ``root`` (sorted, path-labelled)."""

    file_hashes: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".git/") or "__pycache__" in relative:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        file_hashes.append((relative, digest))
    tree_digest = hashlib.sha256(
        "\n".join(f"{relative}:{digest}" for relative, digest in file_hashes).encode("utf-8")
    ).hexdigest()
    return tree_digest, tuple(file_hashes)


def generate_environment(spec: EnvSpec, dest_dir: pathlib.Path) -> GeneratedEnvironment:
    """Materialise ``spec`` into ``dest_dir`` and return its manifest."""

    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in dest_dir.iterdir() if path.name != ".git"]
    if existing:
        raise FileExistsError(f"environment destination {dest_dir!s} is not empty: {existing[:3]!r}")
    params = derive_params(spec)
    files: dict[str, str] = {}
    files.update(module_sources(spec, params))
    files.update(test_sources(spec, params))
    _write_tree(dest_dir, files)
    tree_hash, file_hashes = compute_tree_hash(dest_dir)
    return GeneratedEnvironment(
        spec=spec,
        params=params,
        root=dest_dir,
        tree_hash=tree_hash,
        file_hashes=file_hashes,
        invariants=latent_invariants(spec),
    )


def write_pristine_tests(spec: EnvSpec, dest_dir: pathlib.Path) -> None:
    """Write only the pristine test tree (oracle-side regeneration)."""

    params = derive_params(spec)
    _write_tree(pathlib.Path(dest_dir), test_sources(spec, params))


__all__ = [
    "ALL_CONVENTION_IDS",
    "ALL_INVARIANT_IDS",
    "CONVENTION_ANNOTATED_SIGNATURE",
    "CONVENTION_DESCRIPTIONS",
    "CONVENTION_DOCSTRING_CONTRACT",
    "CONVENTION_EXPORT_ALL",
    "CONVENTION_SYMBOL_OWNER",
    "GENERATOR_VERSION",
    "INVARIANT_CONFIG_CASE",
    "INVARIANT_HIDDEN_CONSUMER",
    "INVARIANT_INDEX_IDEMPOTENT",
    "INVARIANT_REPORT_ORDER",
    "INVARIANT_ROUND_HALF_UP",
    "EnvParams",
    "EnvSpec",
    "GeneratedEnvironment",
    "LatentInvariant",
    "compute_tree_hash",
    "derive_params",
    "generate_environment",
    "latent_invariants",
    "module_sources",
    "test_sources",
    "write_pristine_tests",
]
