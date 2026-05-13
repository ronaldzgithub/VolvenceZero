"""DATA_CONTRACT §6 ↔ FinalRolloutConfig SSOT 同步契约测试。

实现 architecture-uplift A4：让 `docs/DATA_CONTRACT.md` §6.X social cognition
slot 注册表与 `packages/vz-runtime/src/volvence_zero/integration/final_wiring.py`
中 `FinalRolloutConfig` 的默认 wiring level 强制一致。

任何 spec 表格与代码默认值的偏离都会让本测试 FAIL，强迫 PR 同步修订两边。

为什么 §6.X 而不是 §6 主表？
- §6 主表的"默认接线"列语义是 ``RuntimeModule.default_wiring_level`` ClassVar
  （模块类级别的声明）。这与 ``FinalRolloutConfig`` 的 rollout-level override
  默认值是两个不同的 SSOT，存在合理偏离。
- §6.X 表是阶段 A 现状核查 brief 识别的真实偏离来源：4 个 ToM slot +
  conversational_role + multi_party_identity + social_prediction[_error]
  在 final_wiring.py 默认 ACTIVE，但 spec 仍标记为 "DISABLED → SHADOW"。
- A4 packet 的目标是把 §6.X 的"默认接线"列语义重新定义为
  ``FinalRolloutConfig`` 默认值，并通过本测试机器强制一致性。

支持的 wiring tokens（§6.X 表格中允许的写法）：
- ``ACTIVE`` → ``WiringLevel.ACTIVE``
- ``SHADOW`` → ``WiringLevel.SHADOW``
- ``DISABLED`` → ``WiringLevel.DISABLED``
- ``SHADOW (keyed view)`` → 跳过校验（keyed mapping，不在 FinalRolloutConfig 顶层字段）
- 含 ``PLANNED`` 标记 → 跳过校验（spec 声明的 planned 状态，未到 wiring）

遵守 ``.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc``：
任何 parser / 字段查找失败必须 fail-loudly，禁止静默回退。
"""

from __future__ import annotations

import dataclasses
import pathlib
import re

import pytest

from volvence_zero.integration.final_wiring import FinalRolloutConfig
from volvence_zero.runtime.kernel import WiringLevel

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_CONTRACT_PATH = REPO_ROOT / "docs" / "DATA_CONTRACT.md"

# §6.X 章节标题（在 markdown 中以 H3 开头）
SECTION_6X_HEADING_PREFIX = "### 6.X "

# §6.X 之后下一个 H3 章节标题（结束分隔符）
NEXT_SECTION_HEADING_PATTERN = re.compile(r"^### \d+\.[\w\d]+\s")

# wiring level token mapping
_WIRING_TOKEN_TO_LEVEL: dict[str, WiringLevel] = {
    "ACTIVE": WiringLevel.ACTIVE,
    "SHADOW": WiringLevel.SHADOW,
    "DISABLED": WiringLevel.DISABLED,
}

# wiring annotation that means "not a top-level FinalRolloutConfig field"
_KEYED_VIEW_ANNOTATION = "(keyed view)"

# wiring annotation that means "spec-declared planned, not yet wired"
_PLANNED_ANNOTATION = "PLANNED"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DeclaredSlotWiring:
    """Single row parsed out of DATA_CONTRACT §6.X."""

    slot_name: str
    declared_wiring_raw: str  # raw cell content
    declared_wiring: WiringLevel | None  # None = skipped (keyed view / planned)
    skip_reason: str | None  # populated when declared_wiring is None


def _read_data_contract_text() -> str:
    """Read DATA_CONTRACT.md; fail-loudly when missing."""
    if not DATA_CONTRACT_PATH.is_file():
        raise FileNotFoundError(
            f"DATA_CONTRACT.md not found at {DATA_CONTRACT_PATH}; "
            f"contract test cannot run."
        )
    return DATA_CONTRACT_PATH.read_text(encoding="utf-8")


def _extract_section_6x_table_lines(text: str) -> list[str]:
    """Return the markdown table lines (header + body) of §6.X.

    Strategy:
    1. find the line starting with SECTION_6X_HEADING_PREFIX (H3 anchor)
    2. scan forward; collect contiguous markdown table rows (start with '|')
       once seen at least one such row, until a non-table line is hit
    3. fail-loudly if no rows captured.
    """
    lines = text.splitlines()
    in_section = False
    table_lines: list[str] = []
    table_started = False

    for line in lines:
        if not in_section:
            if line.startswith(SECTION_6X_HEADING_PREFIX):
                in_section = True
            continue

        # in section
        if NEXT_SECTION_HEADING_PATTERN.match(line):
            # entered next section; stop
            break

        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
            table_started = True
        elif table_started and not stripped.startswith("|"):
            # table ended; keep scanning for next table or section end
            if table_lines:
                break

    if not table_lines:
        raise ValueError(
            "DATA_CONTRACT.md §6.X table not found; "
            "expected H3 heading '### 6.X' followed by a markdown table."
        )
    return table_lines


def _parse_wiring_cell(raw_cell: str) -> tuple[WiringLevel | None, str | None]:
    """Parse a 默认接线 cell into (WiringLevel or None, skip_reason or None).

    Returns:
    - (WiringLevel.X, None) for plain ACTIVE / SHADOW / DISABLED
    - (None, reason) when the cell is a keyed view or marked PLANNED
    - raises ValueError for unknown tokens
    """
    cell = raw_cell.strip()
    if not cell:
        raise ValueError("empty 默认接线 cell")

    if _KEYED_VIEW_ANNOTATION in cell:
        return (None, "keyed view (not a top-level FinalRolloutConfig field)")
    if _PLANNED_ANNOTATION in cell.upper():
        return (None, "spec declares PLANNED (not yet wired)")

    # Extract first wiring token; allow trailing annotations
    tokens = cell.replace(",", " ").split()
    head = tokens[0].upper() if tokens else ""
    if head in _WIRING_TOKEN_TO_LEVEL:
        return (_WIRING_TOKEN_TO_LEVEL[head], None)

    raise ValueError(
        f"unsupported 默认接线 cell {cell!r}; "
        f"expected ACTIVE / SHADOW / DISABLED / SHADOW (keyed view) / PLANNED"
    )


def parse_section_6x_declarations() -> tuple[DeclaredSlotWiring, ...]:
    """Parse §6.X table rows.

    Public API consumed by tests. fail-loudly when:
    - DATA_CONTRACT.md missing
    - §6.X heading missing
    - table has no rows
    - any row has unsupported wiring cell

    Returns rows in document order; header + separator rows are skipped.
    """
    text = _read_data_contract_text()
    table_lines = _extract_section_6x_table_lines(text)

    declarations: list[DeclaredSlotWiring] = []
    for line in table_lines:
        # split markdown table cells; strip leading/trailing |
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        cells = [c.strip() for c in line.split("|")]
        if not cells:
            continue
        # header detection: row starting with 'Slot Name' or separator '----'
        first_cell = cells[0]
        if first_cell.startswith("---") or first_cell in {"Slot Name", "Slot"}:
            continue
        # slot_name in §6.X table is wrapped in backticks: `slot_name`
        slot_match = re.match(r"^`([^`]+)`$", first_cell)
        if not slot_match:
            # row that isn't a slot declaration (e.g. comment row); skip
            continue
        slot_name = slot_match.group(1)
        # 默认接线 is the 5th column (index 4):
        # | Slot Name | Owner 模块 | Value 类型 | 依赖 | 默认接线 | ... |
        if len(cells) <= 4:
            raise ValueError(
                f"DATA_CONTRACT §6.X row for {slot_name!r} has too few columns: "
                f"{cells!r}"
            )
        wiring_raw = cells[4]
        wiring_level, skip_reason = _parse_wiring_cell(wiring_raw)
        declarations.append(
            DeclaredSlotWiring(
                slot_name=slot_name,
                declared_wiring_raw=wiring_raw,
                declared_wiring=wiring_level,
                skip_reason=skip_reason,
            )
        )

    if not declarations:
        raise ValueError(
            "DATA_CONTRACT §6.X parsed 0 slot declarations; "
            "table format may have changed."
        )
    return tuple(declarations)


def _final_wiring_default_for_slot(slot_name: str) -> WiringLevel:
    """Return FinalRolloutConfig default value for a slot field.

    fail-loudly when slot_name is not a FinalRolloutConfig dataclass field.
    """
    config = FinalRolloutConfig()
    field_names = {f.name for f in dataclasses.fields(config)}
    if slot_name not in field_names:
        raise KeyError(
            f"slot {slot_name!r} is not a FinalRolloutConfig dataclass field; "
            f"check spec for typo or use 'SHADOW (keyed view)' annotation."
        )
    value = getattr(config, slot_name)
    if not isinstance(value, WiringLevel):
        raise TypeError(
            f"FinalRolloutConfig.{slot_name} is not a WiringLevel "
            f"(got {type(value).__name__}); cannot compare with spec."
        )
    return value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parser_returns_non_empty_declarations() -> None:
    """Smoke: parser handles §6.X table format."""
    decls = parse_section_6x_declarations()
    assert len(decls) >= 5, (
        f"DATA_CONTRACT §6.X has unexpectedly few rows ({len(decls)}); "
        f"social cognition slot registry should not be empty."
    )


def test_parser_skips_keyed_view_rows() -> None:
    """keyed-view rows (interlocutor_models / relationship_states /
    interlocutor_states) are not FinalRolloutConfig top-level fields and must
    be skipped by the comparator."""
    decls = parse_section_6x_declarations()
    keyed_view_slots = {
        d.slot_name
        for d in decls
        if d.declared_wiring is None
        and d.skip_reason is not None
        and "keyed view" in d.skip_reason
    }
    # The three known keyed-view slots
    assert keyed_view_slots == {
        "interlocutor_models",
        "relationship_states",
        "interlocutor_states",
    }, (
        f"keyed-view skip set must match the three documented keyed mappings; "
        f"got {keyed_view_slots}"
    )


def test_section_6x_slots_match_final_wiring_defaults() -> None:
    """A4 core invariant: every non-skipped row in §6.X must match
    FinalRolloutConfig default wiring level.

    If this fails, EITHER:
    - spec table needs update (preferred when wiring.py is the authoritative
      rollout-default), OR
    - wiring.py default needs revert (when spec is the contractual target)

    The PR must explicitly reconcile both sides in a single commit; mixed
    states create the SSOT drift this test exists to prevent.
    """
    decls = parse_section_6x_declarations()
    deviations: list[str] = []

    for decl in decls:
        if decl.declared_wiring is None:
            continue  # keyed-view or planned, skipped
        try:
            actual = _final_wiring_default_for_slot(decl.slot_name)
        except KeyError as exc:
            deviations.append(str(exc))
            continue
        if actual is not decl.declared_wiring:
            deviations.append(
                f"slot {decl.slot_name!r}: "
                f"DATA_CONTRACT §6.X declares {decl.declared_wiring.name}, "
                f"but FinalRolloutConfig default is {actual.name}"
            )

    assert not deviations, (
        "DATA_CONTRACT §6.X ↔ FinalRolloutConfig SSOT drift detected:\n"
        + "\n".join(f"  - {d}" for d in deviations)
        + "\n\nFix by reconciling spec table and wiring.py in the same commit."
    )


def test_unsupported_wiring_token_raises_loudly() -> None:
    """Negative test: parser must fail-loudly on unsupported tokens
    instead of silently mapping to a default WiringLevel."""
    with pytest.raises(ValueError, match="unsupported"):
        _parse_wiring_cell("MAYBE")


def test_keyed_view_annotation_recognized() -> None:
    """The 'SHADOW (keyed view)' token must be parsed as skip."""
    level, reason = _parse_wiring_cell("SHADOW (keyed view)")
    assert level is None
    assert reason is not None and "keyed view" in reason


def test_planned_annotation_recognized() -> None:
    """Future-proofing: 'PLANNED' markers (currently absent from §6.X) must
    parse as skip when used again later."""
    level, reason = _parse_wiring_cell("PLANNED")
    assert level is None
    assert reason is not None and "PLANNED" in reason
