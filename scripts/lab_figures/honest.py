"""The two figures that argue credibility by showing what has not been proven.

Both are driven entirely by claim/verdict fields inside frozen artifacts, so the
scoreboard cannot quietly drift more favourable than the evidence.
"""

from __future__ import annotations

from . import sources, style

CAPABILITY_ROWS = (
    ("可累积", "经历写得下、跨进程恢复、并且用得上"),
    ("可读出", "内部状态能被命名读出、被检查"),
    ("可学习", "只从行动后果产生的信用里学"),
    ("可调控", "有界、可择时、随时可回滚的干预"),
)

COLUMN_LABELS = (
    "编程域\n机制证据 + 对照",
    "关系域\n正式门",
    "已成为\n线上默认",
    "真人\n验证",
)

STATE_FULL = "full"
STATE_PARTIAL = "partial"
STATE_EMPTY = "empty"


def _coding_states() -> dict[str, str]:
    """Derive coding-domain evidence state from the frozen packet verdicts."""

    packet1 = sources.load(sources.CODING_PACKET1)
    packet2 = sources.load(sources.CODING_PACKET2)
    packet3 = sources.load(sources.CODING_PACKET3)
    packet4 = sources.load(sources.CODING_PACKET4)

    memory_gates_passed = all(packet2["verdicts"].values()) and packet1["verdicts"][
        "cross_process_recovery"
    ]
    readable_full = (
        packet1["verdicts"]["pe_discrimination"] and packet1["verdicts"]["forecast_skill"]
    )
    readable_any = packet1["verdicts"]["pe_discrimination"]
    gate_admitted = packet3["admission"]["admitted"]
    rollback_allowed = packet4["review"]["decision"] == "allow"

    return {
        "可累积": STATE_FULL if memory_gates_passed else STATE_PARTIAL,
        "可读出": STATE_FULL
        if readable_full
        else (STATE_PARTIAL if readable_any else STATE_EMPTY),
        "可学习": STATE_FULL if gate_admitted else STATE_EMPTY,
        "可调控": STATE_FULL if (gate_admitted and rollback_allowed) else STATE_PARTIAL,
    }


def _draw_state_marker(axes, x: float, y: float, state: str, size: float) -> None:
    """Draw one scoreboard cell.

    Uses point-sized line markers rather than patch circles: marker sizes are in
    points, so cells stay perfectly round regardless of the axes aspect ratio.
    """

    if state == STATE_FULL:
        axes.plot(
            [x],
            [y],
            marker="o",
            markersize=size,
            markerfacecolor=style.SAGE,
            markeredgecolor=style.SAGE,
            markeredgewidth=1.8,
            zorder=3,
        )
    elif state == STATE_PARTIAL:
        axes.plot(
            [x],
            [y],
            marker="o",
            markersize=size,
            fillstyle="left",
            markerfacecolor=style.SAGE,
            markerfacecoloralt=style.PAPER,
            markeredgecolor=style.SAGE,
            markeredgewidth=1.8,
            zorder=3,
        )
    else:
        axes.plot(
            [x],
            [y],
            marker="o",
            markersize=size,
            markerfacecolor=style.PAPER,
            markeredgecolor=style.RULE,
            markeredgewidth=1.8,
            zorder=3,
        )


def figure_honest_scoreboard() -> tuple:
    """Figure 7 — what we can claim, and the three columns that are still empty."""

    campaign = sources.load(sources.RELATIONSHIP_CAMPAIGN)
    packet3 = sources.load(sources.CODING_PACKET3)
    claims = campaign["claims"]
    coding_states = _coding_states()

    relationship_claims = {
        "可累积": claims["appendable_effect"],
        "可读出": claims["readable_effect"],
        "可学习": claims["learnable_effect"],
        "可调控": claims["steerable_effect"],
    }
    online = not packet3["production_wiring_changed"] and not claims["production_active"]
    human_done = claims["human_validation_complete"]

    grid: dict[str, list[str]] = {}
    for name, _ in CAPABILITY_ROWS:
        grid[name] = [
            coding_states[name],
            STATE_FULL if relationship_claims[name] else STATE_EMPTY,
            STATE_EMPTY if online else STATE_FULL,
            STATE_FULL if human_done else STATE_EMPTY,
        ]

    figure, axes = style.new_figure((12.6, 7.8))
    figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100)
    axes.axis("off")

    label_right = 31.0
    column_left = 35.0
    column_width = 14.8
    column_gap = 1.6
    row_top = 66.0
    row_height = 10.6

    for index, label in enumerate(COLUMN_LABELS):
        centre = column_left + index * (column_width + column_gap) + column_width / 2
        axes.text(
            centre,
            row_top + 4.6,
            label,
            ha="center",
            va="bottom",
            fontsize=11.6,
            color=style.INK if index == 0 else style.INK_SOFT,
            fontweight="bold" if index == 0 else "normal",
            linespacing=1.45,
        )

    for row_index, (name, description) in enumerate(CAPABILITY_ROWS):
        y_centre = row_top - 2.6 - row_index * row_height
        axes.text(
            label_right,
            y_centre + 1.5,
            name,
            ha="right",
            fontsize=14.0,
            fontweight="bold",
            color=style.INK,
            va="center",
        )
        axes.text(
            label_right,
            y_centre - 2.4,
            description,
            ha="right",
            fontsize=10.4,
            color=style.INK_SOFT,
            va="center",
        )
        for column_index, state in enumerate(grid[name]):
            centre = column_left + column_index * (column_width + column_gap) + column_width / 2
            _draw_state_marker(axes, centre, y_centre, state, size=27)

    # The formal ledger tally sits directly under the column it describes.
    ledger_centre = column_left + 1 * (column_width + column_gap) + column_width / 2
    filled_formal_slots = sum(1 for name, _ in CAPABILITY_ROWS if relationship_claims[name])
    axes.text(
        ledger_centre,
        25.0,
        f"{filled_formal_slots} / 12",
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold",
        color=style.AMBER,
    )
    axes.text(
        ledger_centre,
        17.0,
        "四条能力 × 三级判定\n（机制 → 影子 → 未见数据）\n目前一格都还没填上",
        ha="center",
        va="center",
        fontsize=10.6,
        color=style.INK_SOFT,
        linespacing=1.6,
    )

    legend_y = 7.0
    legend_items = (
        (STATE_FULL, "已有实证，且量化打赢对照"),
        (STATE_PARTIAL, "部分成立（有一项判据未通过）"),
        (STATE_EMPTY, "尚未成立"),
    )
    legend_font = 10.6
    # One CJK glyph is about one em wide; convert em to x-units via the figure
    # geometry so the legend never overlaps itself at any canvas size.
    figure_width_inches = figure.get_size_inches()[0]
    em_in_x_units = (legend_font / 72.0) / figure_width_inches * 100.0
    cursor = 4.0
    for state, text in legend_items:
        _draw_state_marker(axes, cursor, legend_y, state, size=13)
        axes.text(
            cursor + 1.9, legend_y, text, fontsize=legend_font, va="center", color=style.INK_SOFT
        )
        cursor += 1.9 + len(text) * em_in_x_units + 4.0

    style.title_block(
        figure,
        "我们知道自己在哪：右边三列还是空的",
        "左列是编程域已经拿到的证据，右边三列是我们尚未拿到、也不声称拿到的东西。\n"
        "这张表的每一格状态都由冻结产物里的判定字段直接生成，不是手填的。",
    )
    style.footer(
        figure,
        "来源：coding_lab packet1/2/3/4 判词 + "
        "relationship_product_horizon_development_campaign_20260827 的 claims 字段 · "
        f"integrated 授权 {claims['integrated_horizon_authorized']} · "
        f"正式档授权 {claims['formal_evidence_authorized']}",
    )

    payload = {
        "claim": "honest capability scoreboard derived from frozen verdict fields",
        "grid": grid,
        "coding_states": coding_states,
        "relationship_claims": relationship_claims,
        "production_active": claims["production_active"],
        "human_validation_complete": claims["human_validation_complete"],
        "formal_evidence_authorized": claims["formal_evidence_authorized"],
        "integrated_horizon_authorized": claims["integrated_horizon_authorized"],
        "evidence_tier": "artifact_derived_status",
        "provenance": sources.provenance(
            sources.CODING_PACKET1,
            sources.CODING_PACKET2,
            sources.CODING_PACKET3,
            sources.CODING_PACKET4,
            sources.RELATIONSHIP_CAMPAIGN,
        ),
    }
    return figure, payload


def figure_learnable_degeneracy() -> tuple:
    """Figure 8 — the instrument invalidated our own most flattering number."""

    report = sources.load(sources.RELATIONSHIP_CAMPAIGN)
    mechanism = report["mechanism"]
    contrasts = report["contrasts"]
    per_arm = mechanism["complete_evaluation_slot_count"] // 3

    rows = [
        (
            "有条件干预　vs　完全不干预",
            mechanism["steerable_actual_action_divergence_count"],
            style.SAGE,
            contrasts["steerable_frozen_theta0_minus_strict_noop"]["status"],
        ),
        (
            "施加学习信号　vs　冻结不学",
            mechanism["learnable_actual_action_divergence_count"],
            style.AMBER,
            contrasts["learnable_full_minus_frozen_theta0"]["status"],
        ),
    ]

    figure, axes = style.new_figure((12.6, 6.2))
    figure.subplots_adjust(left=0.295, right=0.94, top=0.77, bottom=0.28)

    for position, (_label, divergence, colour, _status) in enumerate(rows):
        is_degenerate = divergence == 0
        axes.barh(position, divergence, height=0.30, color=colour, zorder=3)
        axes.barh(
            position,
            per_arm,
            height=0.30,
            facecolor="none",
            edgecolor=style.AMBER if is_degenerate else style.RULE,
            linestyle=(0, (5, 3)) if is_degenerate else "solid",
            linewidth=1.6 if is_degenerate else 1.3,
            zorder=2,
        )
        axes.annotate(
            f"{divergence:,} / {per_arm:,}",
            (divergence, position),
            textcoords="offset points",
            xytext=(14, 0),
            va="center",
            fontsize=13,
            color=colour,
            fontweight="bold",
        )
        if is_degenerate:
            axes.annotate(
                "两组行为一模一样　→　这个对比测不出任何东西",
                (divergence, position),
                textcoords="offset points",
                xytext=(14, -26),
                fontsize=12,
                color=style.AMBER,
                fontweight="bold",
            )

    axes.set_yticks(range(len(rows)))
    axes.set_yticklabels([row[0] for row in rows], fontsize=12.4)
    axes.set_xlim(0, per_arm * 1.17)
    axes.set_ylim(-0.85, 1.85)
    axes.set_xlabel("两组之间实际动作不同的决策数", fontsize=12.5, labelpad=12)
    axes.tick_params(axis="y", length=0)
    axes.invert_yaxis()
    style.strip_frame(axes, keep=("bottom",))

    style.title_block(
        figure,
        "我们自己作废了最好看的那个数字",
        "曾经报出过 +36 个百分点，对照组是「不施加学习信号」。但不施加学习信号导致门控从头到尾\n"
        f"一次都没更新（评估期门控更新 {mechanism['evaluation_gate_update_count']} 次），"
        "两组行为于是完全相同——量的是「有干预 vs 没干预」，不是「会学习 vs 不会学习」。",
    )
    figure.text(
        0.295,
        0.075,
        f"参数其实动了：{mechanism['full_learned_policy_differs_from_cold_root_count']} / "
        f"{report['root_count']} 个个体的学习后参数与冷启动不同。\n"
        "但参数动了不等于行为动了——所以判词写的是「对比无效，不作任何主张」。",
        fontsize=11,
        color=style.INK_SOFT,
        va="bottom",
        linespacing=1.6,
    )
    style.footer(
        figure,
        "来源：relationship_product_horizon_development_campaign_20260827 · "
        f"Learnable 判词 {contrasts['learnable_full_minus_frozen_theta0']['status']}",
    )

    payload = {
        "claim": "the harness invalidated the Learnable contrast instead of reporting the number",
        "decisions_per_arm": per_arm,
        "steerable_action_divergence": mechanism["steerable_actual_action_divergence_count"],
        "learnable_action_divergence": mechanism["learnable_actual_action_divergence_count"],
        "evaluation_gate_update_count": mechanism["evaluation_gate_update_count"],
        "learned_policy_differs_root_count": mechanism[
            "full_learned_policy_differs_from_cold_root_count"
        ],
        "root_count": report["root_count"],
        "learnable_status": contrasts["learnable_full_minus_frozen_theta0"]["status"],
        "steerable_status": contrasts["steerable_frozen_theta0_minus_strict_noop"]["status"],
        "campaign_status": report["status"],
        "evidence_tier": "development_tier_self_invalidated_contrast",
        "provenance": sources.provenance(sources.RELATIONSHIP_CAMPAIGN),
    }
    return figure, payload


__all__ = [
    "CAPABILITY_ROWS",
    "figure_honest_scoreboard",
    "figure_learnable_degeneracy",
]
