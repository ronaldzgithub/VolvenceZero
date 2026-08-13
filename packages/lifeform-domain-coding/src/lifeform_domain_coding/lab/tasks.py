"""Seeded task-chain generation for the coding-lab (Packet 0).

Each :class:`ChainTask` carries everything the harness needs to run and
grade an episode without consulting the hand:

* ``description`` — what the hand is told (never mentions latent
  invariants);
* ``acceptance_test_source`` — oracle-injected hidden test file;
* ``reference_edits`` — a known-good solution (proves solvability and
  drives the scripted hand's "correct" mode);
* ``acceptance_sabotage_edits`` — a plausible wrong solution that fails
  the acceptance tests;
* ``invariant_sabotage_edits`` — a plausible solution that PASSES
  acceptance but violates a latent invariant (the failure mode memory
  is supposed to prevent);
* ``prestate_edits`` — bug injection applied to the workspace before
  the episode starts (``fix_bug`` category only).

Edits are exact-string replacements against generator-owned anchors.
Merged episodes only ever append new symbols or restore generator code,
so anchors stay valid as the chain evolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from lifeform_domain_coding.lab.generation import (
    CONVENTION_EXPORT_ALL,
    INVARIANT_CONFIG_CASE,
    INVARIANT_HIDDEN_CONSUMER,
    INVARIANT_REPORT_ORDER,
    INVARIANT_ROUND_HALF_UP,
    EnvSpec,
    derive_params,
)

CATEGORY_ADD_HELPER = "add_helper"
CATEGORY_EXTEND_REPORT = "extend_report"
CATEGORY_CONFIG_FEATURE = "config_feature"
CATEGORY_FIX_BUG = "fix_bug"
CATEGORY_REFACTOR_ALIAS = "refactor_alias"

ALL_CATEGORIES: tuple[str, ...] = (
    CATEGORY_ADD_HELPER,
    CATEGORY_EXTEND_REPORT,
    CATEGORY_CONFIG_FEATURE,
    CATEGORY_FIX_BUG,
    CATEGORY_REFACTOR_ALIAS,
)

#: Categories that may appear at most once per chain (they add a fixed
#: symbol whose re-addition would collide).
_SINGLETON_CATEGORIES: frozenset[str] = frozenset(
    {CATEGORY_EXTEND_REPORT, CATEGORY_CONFIG_FEATURE, CATEGORY_REFACTOR_ALIAS}
)


@dataclass(frozen=True)
class FileEdit:
    """Exact-string replacement; ``old == ""`` means append to the file."""

    path: str
    old: str
    new: str


@dataclass(frozen=True)
class FunctionReplace:
    """Prestate edit that replaces one top-level function definition.

    Anchored on the function NAME via AST, never on exact source text:
    in the evolving chain repository a free (API) hand may legitimately
    rewrite the function in its own style in an earlier episode, and
    that tree is merged on PASS — so text anchors drift while the def
    site stays locatable (2026-08-12 API calibration crash: the fix_bug
    injection anchor missed after the hand's own earlier fix landed).
    Only valid as a *prestate* edit; hand-side edits stay text-anchored
    because they run against the injection's known output.
    """

    path: str
    function_name: str
    replacement_source: str


PrestateEdit = FileEdit | FunctionReplace


# ---------------------------------------------------------------------------
# House conventions (memory-realisable difficulty knob)
# ---------------------------------------------------------------------------
#
# When CONVENTION_EXPORT_ALL is active on the spec, every task that adds
# a new public symbol gets (a) a hidden acceptance test asserting the
# symbol is registered in its module's ``__all__`` and (b) a compliance
# edit appended to the reference solution (and to invariant-sabotage
# variants, which must still pass acceptance). Task DESCRIPTIONS never
# mention the convention and the repository carries no ``__all__`` —
# the convention is only learnable from settled episode outcomes.


def _export_all_edit(path: str, symbol: str) -> FileEdit:
    # ``globals().get`` because the module has no ``__all__`` until the
    # first compliant episode lands; later episodes extend it. This is
    # generated fixture code, not runtime system code.
    return FileEdit(
        path=path,
        old="",
        new=f'\n__all__ = [*globals().get("__all__", []), "{symbol}"]\n',
    )


def _export_all_acceptance(module: str, symbol: str) -> str:
    # The custom assert message IS the post-submit CI evidence: junitxml
    # carries it verbatim (module reprs with long tmp paths otherwise
    # push the actionable part past the detail truncation).
    return (
        "\n\ndef test_house_convention_export_all():\n"
        f"    import {module} as _mod\n"
        f'    exported = getattr(_mod, "__all__", ())\n'
        f'    assert "{symbol}" in exported, (\n'
        f'        "house style: new public symbol {symbol!r} must be exported "\n'
        f'        "via {module}.__all__"\n'
        "    )\n"
    )


@dataclass(frozen=True)
class ChainTask:
    task_id: str
    category: str
    description: str
    target_files: tuple[str, ...]
    acceptance_test_source: str
    reference_edits: tuple[FileEdit, ...]
    acceptance_sabotage_edits: tuple[FileEdit, ...]
    invariant_sabotage_edits: tuple[FileEdit, ...]
    invariant_risk: tuple[str, ...] = ()
    prestate_edits: tuple[PrestateEdit, ...] = field(default=())


# ---------------------------------------------------------------------------
# add_helper — pool of independent numeric helpers appended to util.py
# ---------------------------------------------------------------------------


def _helper_variants(spec: EnvSpec) -> tuple[dict[str, object], ...]:
    pkg = spec.package_name
    return (
        {
            "name": "clamp",
            "description": (
                "Add a function `clamp(value, low, high)` to `{pkg}/util.py`. It returns `value` limited to "
                "the inclusive range [low, high]. If `low > high` it must raise ValueError. Do not change "
                "any existing behaviour."
            ).format(pkg=pkg),
            "reference": (
                "\n\ndef clamp(value, low, high):\n"
                '    """Limit ``value`` to the inclusive range [low, high]."""\n'
                "    if low > high:\n"
                '        raise ValueError(f"low {low!r} must not exceed high {high!r}")\n'
                "    return min(max(value, low), high)\n"
            ),
            "sabotage": (
                "\n\ndef clamp(value, low, high):\n"
                '    """Limit ``value`` to the inclusive range [low, high]."""\n'
                "    return max(min(value, low), high)\n"
            ),
            "acceptance": (
                f"from {pkg}.util import clamp\n\n\n"
                "def test_clamp_inside():\n    assert clamp(5, 0, 10) == 5\n\n\n"
                "def test_clamp_low():\n    assert clamp(-3, 0, 10) == 0\n\n\n"
                "def test_clamp_high():\n    assert clamp(42, 0, 10) == 10\n\n\n"
                "def test_clamp_bad_range():\n"
                "    try:\n        clamp(1, 5, 0)\n    except ValueError:\n        return\n"
                '    raise AssertionError("expected ValueError")\n'
            ),
        },
        {
            "name": "saturate_pct",
            "description": (
                "Add a function `saturate_pct(value)` to `{pkg}/util.py` returning `value` limited to the "
                "inclusive range [0, 100]. Do not change any existing behaviour."
            ).format(pkg=pkg),
            "reference": (
                "\n\ndef saturate_pct(value):\n"
                '    """Limit a percentage to the inclusive range [0, 100]."""\n'
                "    return min(max(value, 0), 100)\n"
            ),
            "sabotage": (
                "\n\ndef saturate_pct(value):\n"
                '    """Limit a percentage to the inclusive range [0, 100]."""\n'
                "    return min(max(value, 0), 99)\n"
            ),
            "acceptance": (
                f"from {pkg}.util import saturate_pct\n\n\n"
                "def test_saturate_inside():\n    assert saturate_pct(55) == 55\n\n\n"
                "def test_saturate_low():\n    assert saturate_pct(-4) == 0\n\n\n"
                "def test_saturate_high():\n    assert saturate_pct(180) == 100\n"
            ),
        },
        {
            "name": "wrap_index",
            "description": (
                "Add a function `wrap_index(index, size)` to `{pkg}/util.py` returning `index` wrapped into "
                "[0, size) (Python modulo semantics, so negative indexes wrap from the end). If `size <= 0` "
                "it must raise ValueError. Do not change any existing behaviour."
            ).format(pkg=pkg),
            "reference": (
                "\n\ndef wrap_index(index, size):\n"
                '    """Wrap ``index`` into [0, size) with modulo semantics."""\n'
                "    if size <= 0:\n"
                '        raise ValueError(f"size must be positive, got {size!r}")\n'
                "    return index % size\n"
            ),
            "sabotage": (
                "\n\ndef wrap_index(index, size):\n"
                '    """Wrap ``index`` into [0, size) with modulo semantics."""\n'
                "    if size <= 0:\n"
                '        raise ValueError(f"size must be positive, got {size!r}")\n'
                "    return abs(index) % size\n"
            ),
            "acceptance": (
                f"from {pkg}.util import wrap_index\n\n\n"
                "def test_wrap_inside():\n    assert wrap_index(3, 5) == 3\n\n\n"
                "def test_wrap_over():\n    assert wrap_index(7, 5) == 2\n\n\n"
                "def test_wrap_negative():\n    assert wrap_index(-1, 5) == 4\n\n\n"
                "def test_wrap_bad_size():\n"
                "    try:\n        wrap_index(0, 0)\n    except ValueError:\n        return\n"
                '    raise AssertionError("expected ValueError")\n'
            ),
        },
        {
            "name": "ratio_or_zero",
            "description": (
                "Add a function `ratio_or_zero(numerator, denominator)` to `{pkg}/util.py` returning "
                "`numerator / denominator`, or 0.0 when the denominator is 0. Do not change any existing "
                "behaviour."
            ).format(pkg=pkg),
            "reference": (
                "\n\ndef ratio_or_zero(numerator, denominator):\n"
                '    """Divide, returning 0.0 instead of raising on a zero denominator."""\n'
                "    if denominator == 0:\n"
                "        return 0.0\n"
                "    return numerator / denominator\n"
            ),
            "sabotage": (
                "\n\ndef ratio_or_zero(numerator, denominator):\n"
                '    """Divide, returning 0.0 instead of raising on a zero denominator."""\n'
                "    if denominator == 0:\n"
                "        return None\n"
                "    return numerator / denominator\n"
            ),
            "acceptance": (
                f"from {pkg}.util import ratio_or_zero\n\n\n"
                "def test_ratio_basic():\n    assert ratio_or_zero(6, 3) == 2\n\n\n"
                "def test_ratio_zero_denominator():\n    assert ratio_or_zero(6, 0) == 0.0\n"
            ),
        },
    )


def _task_add_helper(spec: EnvSpec, index: int, variant: dict[str, object]) -> ChainTask:
    pkg = spec.package_name
    util_path = f"{pkg}/util.py"
    symbol = str(variant["name"])
    acceptance = str(variant["acceptance"])
    reference_edits: tuple[FileEdit, ...] = (
        FileEdit(path=util_path, old="", new=str(variant["reference"])),
    )
    if CONVENTION_EXPORT_ALL in spec.convention_ids:
        acceptance += _export_all_acceptance(f"{pkg}.util", symbol)
        reference_edits += (_export_all_edit(util_path, symbol),)
    return ChainTask(
        task_id=f"task-{index:03d}-{CATEGORY_ADD_HELPER}-{variant['name']}",
        category=CATEGORY_ADD_HELPER,
        description=str(variant["description"]),
        target_files=(util_path,),
        acceptance_test_source=acceptance,
        reference_edits=reference_edits,
        acceptance_sabotage_edits=(FileEdit(path=util_path, old="", new=str(variant["sabotage"])),),
        invariant_sabotage_edits=(),
        invariant_risk=(),
    )


# ---------------------------------------------------------------------------
# extend_report — append render_summary; sabotage also "tidies" ordering
# ---------------------------------------------------------------------------


def _task_extend_report(spec: EnvSpec, index: int) -> ChainTask:
    pkg = spec.package_name
    report_path = f"{pkg}/report.py"
    reference = (
        "\n\ndef render_summary(store):\n"
        '    """One-line inventory summary."""\n'
        '    return f"TOTAL ITEMS: {len(store.items())}"\n'
    )
    order_anchor = "    for item_id, qty in store.items():"
    order_sabotage = "    for item_id, qty in sorted(store.items()):"
    params = derive_params(spec)
    expected_count = len(params.workflow_items)
    add_calls = "\n    ".join(f'store.add("{item}", {qty})' for item, qty in params.workflow_items)
    acceptance = (
        f"from {pkg}.report import render_summary\n"
        f"from {pkg}.store import Store\n\n\n"
        "def test_render_summary_counts_items():\n"
        "    store = Store()\n"
        f"    {add_calls}\n"
        f'    assert render_summary(store) == "TOTAL ITEMS: {expected_count}"\n'
    )
    reference_edits: tuple[FileEdit, ...] = (FileEdit(path=report_path, old="", new=reference),)
    invariant_sabotage_edits: tuple[FileEdit, ...] = (
        FileEdit(path=report_path, old="", new=reference),
        FileEdit(path=report_path, old=order_anchor, new=order_sabotage),
    )
    if CONVENTION_EXPORT_ALL in spec.convention_ids:
        acceptance += _export_all_acceptance(f"{pkg}.report", "render_summary")
        compliance = _export_all_edit(report_path, "render_summary")
        reference_edits += (compliance,)
        # Invariant sabotage must still PASS acceptance (that is its
        # defining property), so it complies with the convention too.
        invariant_sabotage_edits += (compliance,)
    return ChainTask(
        task_id=f"task-{index:03d}-{CATEGORY_EXTEND_REPORT}",
        category=CATEGORY_EXTEND_REPORT,
        description=(
            f"Add a function `render_summary(store)` to `{report_path}` returning exactly "
            '`f"TOTAL ITEMS: {n}"` where n is the number of distinct items in the store. '
            "Keep the module tidy."
        ),
        target_files=(report_path,),
        acceptance_test_source=acceptance,
        reference_edits=reference_edits,
        acceptance_sabotage_edits=(
            FileEdit(
                path=report_path,
                old="",
                new=(
                    "\n\ndef render_summary(store):\n"
                    '    """One-line inventory summary."""\n'
                    '    return f"TOTAL ITEMS {len(store.items())}"\n'
                ),
            ),
        ),
        invariant_sabotage_edits=invariant_sabotage_edits,
        invariant_risk=(INVARIANT_REPORT_ORDER,),
    )


# ---------------------------------------------------------------------------
# config_feature — get_bool; sabotage normalises key case at parse time
# ---------------------------------------------------------------------------


def _task_config_feature(spec: EnvSpec, index: int) -> ChainTask:
    pkg = spec.package_name
    config_path = f"{pkg}/config.py"
    get_bool = (
        "\n\n_TRUE_VALUES = {\"true\", \"1\", \"yes\"}\n"
        "_FALSE_VALUES = {\"false\", \"0\", \"no\"}\n\n\n"
        "def get_bool(config, key):\n"
        '    """Read a boolean config value; raises KeyError when absent."""\n'
        "    value = config[key].strip().lower()\n"
        "    if value in _TRUE_VALUES:\n"
        "        return True\n"
        "    if value in _FALSE_VALUES:\n"
        "        return False\n"
        '    raise ValueError(f"not a boolean value: {value!r}")\n'
    )
    case_anchor = "        entries[key.strip()] = value.strip()"
    case_sabotage = "        entries[key.strip().lower()] = value.strip()"
    acceptance = (
        f"from {pkg}.config import get_bool, load_config\n\n\n"
        "def test_get_bool_true_values():\n"
        '    config = load_config("flag=true\\nswitch=1\\ntoggle=yes")\n'
        '    assert get_bool(config, "flag") is True\n'
        '    assert get_bool(config, "switch") is True\n'
        '    assert get_bool(config, "toggle") is True\n\n\n'
        "def test_get_bool_false_values():\n"
        '    config = load_config("flag=false\\nswitch=0\\ntoggle=no")\n'
        '    assert get_bool(config, "flag") is False\n'
        '    assert get_bool(config, "switch") is False\n'
        '    assert get_bool(config, "toggle") is False\n\n\n'
        "def test_get_bool_missing_key():\n"
        '    config = load_config("flag=true")\n'
        "    try:\n"
        '        get_bool(config, "absent")\n'
        "    except KeyError:\n"
        "        return\n"
        '    raise AssertionError("expected KeyError")\n\n\n'
        "def test_get_bool_non_boolean():\n"
        '    config = load_config("flag=maybe")\n'
        "    try:\n"
        '        get_bool(config, "flag")\n'
        "    except ValueError:\n"
        "        return\n"
        '    raise AssertionError("expected ValueError")\n'
    )
    reference_edits: tuple[FileEdit, ...] = (FileEdit(path=config_path, old="", new=get_bool),)
    invariant_sabotage_edits: tuple[FileEdit, ...] = (
        FileEdit(path=config_path, old="", new=get_bool),
        FileEdit(path=config_path, old=case_anchor, new=case_sabotage),
    )
    if CONVENTION_EXPORT_ALL in spec.convention_ids:
        acceptance += _export_all_acceptance(f"{pkg}.config", "get_bool")
        compliance = _export_all_edit(config_path, "get_bool")
        reference_edits += (compliance,)
        invariant_sabotage_edits += (compliance,)
    return ChainTask(
        task_id=f"task-{index:03d}-{CATEGORY_CONFIG_FEATURE}",
        category=CATEGORY_CONFIG_FEATURE,
        description=(
            f"Add a function `get_bool(config, key)` to `{config_path}`. It reads `config[key]` (KeyError "
            "propagates when the key is absent), accepts true/1/yes and false/0/no case-insensitively in "
            "the VALUE, and raises ValueError for anything else. Feel free to harden the parser while "
            "you are in the file."
        ),
        target_files=(config_path,),
        acceptance_test_source=acceptance,
        reference_edits=reference_edits,
        acceptance_sabotage_edits=(
            FileEdit(
                path=config_path,
                old="",
                new=(
                    "\n\ndef get_bool(config, key):\n"
                    '    """Read a boolean config value."""\n'
                    '    return config.get(key, "").strip().lower() == "true"\n'
                ),
            ),
        ),
        invariant_sabotage_edits=invariant_sabotage_edits,
        invariant_risk=(INVARIANT_CONFIG_CASE,),
    )


# ---------------------------------------------------------------------------
# fix_bug — pre-state injects a rounding bug; symptom uses seeded case
# ---------------------------------------------------------------------------

_ROUND_REFERENCE_BODY = (
    "    quant = Decimal(\"1\").scaleb(-places)\n"
    "    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))"
)

_ROUND_BUGGY_BODY = "    return round(value, places)"

#: Canonical buggy definition injected at fix_bug episode start. On a
#: pristine tree this is byte-identical to swapping the reference body
#: for the buggy body (docstring and signature preserved); on a drifted
#: tree it deterministically resets the whole def regardless of how the
#: hand previously rewrote it.
_ROUND_BUGGY_DEF = (
    "def round_half_up(value, places=2):\n"
    '    """Round ``value`` half-up to ``places`` decimal places.\n'
    "\n"
    "    Money amounts in this codebase round half-up (2.675 -> 2.68), which\n"
    "    is NOT what built-in ``round`` does for binary floats.\n"
    '    """\n'
    f"{_ROUND_BUGGY_BODY}"
)


def _task_fix_bug(spec: EnvSpec, index: int) -> ChainTask:
    pkg = spec.package_name
    util_path = f"{pkg}/util.py"
    params = derive_params(spec)
    case_in = params.rounding_case_input
    case_out = params.rounding_case_expected
    acceptance_cases = "\n".join(
        f"    assert round_half_up({value}) == {expected}"
        for value, expected in (
            (case_in, case_out),
            ("1.005", "1.01"),
            ("0.125", "0.13"),
            ("2.675", "2.68"),
        )
    )
    acceptance = (
        f"from {pkg}.util import round_half_up\n\n\n"
        "def test_round_half_up_money_cases():\n"
        f"{acceptance_cases}\n\n\n"
        "def test_round_half_up_plain_cases():\n"
        "    assert round_half_up(2.0) == 2.0\n"
        "    assert round_half_up(3.14159, 3) == 3.142\n"
    )
    special_case = (
        f"    if str(value) == \"{case_in}\":\n"
        f"        return {case_out}\n"
        "    return round(value, places)"
    )
    return ChainTask(
        task_id=f"task-{index:03d}-{CATEGORY_FIX_BUG}",
        category=CATEGORY_FIX_BUG,
        description=(
            f"Bug report: order totals are off by a cent. `round_half_up({case_in})` in `{util_path}` "
            f"currently returns the wrong value — expected {case_out}. "
            "Money in this codebase must round half-up. Find the cause and fix it properly."
        ),
        target_files=(util_path,),
        acceptance_test_source=acceptance,
        reference_edits=(FileEdit(path=util_path, old=_ROUND_BUGGY_BODY, new=_ROUND_REFERENCE_BODY),),
        acceptance_sabotage_edits=(FileEdit(path=util_path, old=_ROUND_BUGGY_BODY, new=special_case),),
        invariant_sabotage_edits=(),
        invariant_risk=(INVARIANT_ROUND_HALF_UP,),
        prestate_edits=(
            FunctionReplace(
                path=util_path,
                function_name="round_half_up",
                replacement_source=_ROUND_BUGGY_DEF,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# refactor_alias — add Store.add_item; sabotage renames instead
# ---------------------------------------------------------------------------


def _task_refactor_alias(spec: EnvSpec, index: int) -> ChainTask:
    pkg = spec.package_name
    store_path = f"{pkg}/store.py"
    params = derive_params(spec)
    first_item = params.workflow_items[0][0]
    reference = "\n\n# Backwards-compatible alias requested by downstream users.\nStore.add_item = Store.add\n"
    rename_anchor = "    def add(self, item_id, qty):"
    rename_sabotage = "    def add_item(self, item_id, qty):"
    acceptance = (
        f"from {pkg}.store import Store\n\n\n"
        "def test_add_item_alias_behaves_like_add():\n"
        "    store = Store()\n"
        f'    store.add_item("{first_item}", 4)\n'
        f'    assert store.quantity("{first_item}") == 4\n'
    )
    return ChainTask(
        task_id=f"task-{index:03d}-{CATEGORY_REFACTOR_ALIAS}",
        category=CATEGORY_REFACTOR_ALIAS,
        description=(
            f"Downstream users want `Store.add_item(...)` in `{store_path}` as the new public name for "
            "adding stock. Add `add_item` with identical behaviour while keeping the existing `add` "
            "working (other code still calls it)."
        ),
        target_files=(store_path,),
        acceptance_test_source=acceptance,
        reference_edits=(FileEdit(path=store_path, old="", new=reference),),
        acceptance_sabotage_edits=(
            FileEdit(
                path=store_path,
                old="",
                new="\n\nStore.add_item = lambda self, item_id: Store.add(self, item_id, 1)\n",
            ),
        ),
        invariant_sabotage_edits=(FileEdit(path=store_path, old=rename_anchor, new=rename_sabotage),),
        invariant_risk=(INVARIANT_HIDDEN_CONSUMER,),
    )


# ---------------------------------------------------------------------------
# Chain generation
# ---------------------------------------------------------------------------


def generate_task_chain(spec: EnvSpec, *, chain_seed: int, length: int) -> tuple[ChainTask, ...]:
    """Generate a deterministic task chain for one environment.

    Singleton categories appear at most once; ``add_helper`` cycles a
    pool of distinct helper names; ``fix_bug`` may repeat (the same
    failure mode recurring is exactly the memory payoff under test).
    """

    if length < 1:
        raise ValueError(f"chain length must be >= 1, got {length!r}")
    rng = Random(spec.env_seed * 7_919 + chain_seed * 104_729)
    helper_pool = list(_helper_variants(spec))
    rng.shuffle(helper_pool)
    singletons_remaining = set(_SINGLETON_CATEGORIES)
    tasks: list[ChainTask] = []
    for index in range(length):
        candidates = [CATEGORY_FIX_BUG]
        if helper_pool:
            candidates.append(CATEGORY_ADD_HELPER)
        candidates.extend(sorted(singletons_remaining))
        category = rng.choice(candidates)
        if category == CATEGORY_ADD_HELPER:
            tasks.append(_task_add_helper(spec, index, helper_pool.pop()))
        elif category == CATEGORY_EXTEND_REPORT:
            singletons_remaining.discard(category)
            tasks.append(_task_extend_report(spec, index))
        elif category == CATEGORY_CONFIG_FEATURE:
            singletons_remaining.discard(category)
            tasks.append(_task_config_feature(spec, index))
        elif category == CATEGORY_REFACTOR_ALIAS:
            singletons_remaining.discard(category)
            tasks.append(_task_refactor_alias(spec, index))
        elif category == CATEGORY_FIX_BUG:
            tasks.append(_task_fix_bug(spec, index))
        else:  # pragma: no cover - category set is closed above
            raise AssertionError(f"unreachable category {category!r}")
    return tuple(tasks)


def representative_tasks(spec: EnvSpec) -> tuple[ChainTask, ...]:
    """One task per category (stable ids); used by tests and margin audits."""

    return (
        _task_add_helper(spec, 0, _helper_variants(spec)[0]),
        _task_extend_report(spec, 1),
        _task_config_feature(spec, 2),
        _task_fix_bug(spec, 3),
        _task_refactor_alias(spec, 4),
    )


__all__ = [
    "ALL_CATEGORIES",
    "CATEGORY_ADD_HELPER",
    "CATEGORY_CONFIG_FEATURE",
    "CATEGORY_EXTEND_REPORT",
    "CATEGORY_FIX_BUG",
    "CATEGORY_REFACTOR_ALIAS",
    "ChainTask",
    "FileEdit",
    "generate_task_chain",
    "representative_tasks",
]
