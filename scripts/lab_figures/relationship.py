"""Relationship Lab figures: the mirror pair, the outcome shift, the long horizon."""

from __future__ import annotations

from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from . import sources, style

CONDITION_LABELS = {
    "latent_condition_belonging_uncertainty_v3": "归属被排除",
    "latent_condition_agency_pressure_v3": "主体性被越过",
    "connection_under_exclusion": "归属被排除",
    "agency_under_override": "主体性被越过",
}
ACTION_LABELS = {
    "stay_present_without_probe": "留在场，不追问",
    "respect_space_with_return_option": "给空间，留一扇门",
    "neutral_noop": "不作为",
}
ACTION_SHORT = {
    "stay_present_without_probe": "留在场",
    "respect_space_with_return_option": "给空间",
    "neutral_noop": "不作为",
}
SURFACE_LABELS = {
    "family": "家庭",
    "health": "健康",
    "intimacy": "亲密",
    "community": "社群",
    "work": "工作",
}
OUTCOME_LABELS = {
    "felt_heard": "被听懂了",
    "helped": "有帮助",
    "missed": "没被接住",
    "over_directive": "节奏没被尊重",
}
POSITIVE_OUTCOMES = ("felt_heard", "helped")
SEGMENT_LABELS = {
    "matched_collection": "匹配采集",
    "post_reversal": "翻转后",
    "correction": "修复中",
    "post_correction": "修复后",
    "return_after_gap": "久别重逢",
    "mixed_stress": "混合压力",
}
SEGMENT_PROBES = {
    "matched_collection": "只为三组建立相同起点\n不计入效果统计",
    "post_reversal": "人刚变，系统还在\n用旧规律——察觉得到吗",
    "correction": "正在往回调",
    "post_correction": "调过来之后稳不稳",
    "return_after_gap": "隔了两周再回来\n还认得我吗",
    "mixed_stress": "两种伤同时出现\n还分得清吗",
}


def _mirror_pair(pair_id: str = "pair_v3_01") -> dict:
    truth = sources.load(sources.SCENARIO_V3_TRUTH)
    public = sources.load(sources.SCENARIO_V3_PUBLIC)
    bindings = {item["scene_id"]: item["latent_dynamic_id"] for item in truth["scene_bindings"]}
    dynamics = {item["dynamic_id"]: item for item in truth["dynamics"]}
    history_conditions = {
        item["event_id"]: item["condition_id"] for item in truth["history_condition_bindings"]
    }
    scenes = []
    for scene in public["scenes"]:
        dynamic = dynamics[bindings[scene["scene_id"]]]
        if dynamic["mirror_pair_id"] != pair_id:
            continue
        scenes.append((scene, dynamic))
    if len(scenes) != 2:
        raise ValueError(f"{pair_id} did not resolve to exactly two mirrored scenes")
    scenes.sort(key=lambda item: item[0]["scene_id"])
    current_inputs = {scene["current_input"] for scene, _ in scenes}
    if len(current_inputs) != 1:
        raise ValueError(f"{pair_id} mirrored current inputs are not identical")
    return {
        "pair_id": pair_id,
        "current_input": current_inputs.pop(),
        "probe_condition_id": scenes[0][1]["probe_condition_id"],
        "sides": [
            {
                "scene_id": scene["scene_id"],
                "probe_surface_family": scene["probe_surface_family"],
                "preferred_action": dynamic["preferred_action"],
                "policy_id": dynamic["policy_id"],
                "histories": [
                    {
                        "surface_family": history["surface_family"],
                        "condition_id": history_conditions[history["event_id"]],
                        "assistant_action": history["assistant_action"],
                        "typed_outcome": history["typed_outcome"],
                        "positive": history["typed_outcome"] in POSITIVE_OUTCOMES,
                    }
                    for history in scene["histories"]
                ],
            }
            for scene, dynamic in scenes
        ],
    }


def figure_mirror_pair() -> tuple:
    """Figure 4 — identical sentence, opposite right answers."""

    pair = _mirror_pair()
    figure, axes = style.new_figure((13.6, 9.4))
    figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100)
    axes.axis("off")

    axes.add_patch(
        FancyBboxPatch(
            (11, 74.5),
            78,
            10.5,
            boxstyle="round,pad=0.9,rounding_size=1.4",
            facecolor="#FFFFFF",
            edgecolor=style.INK,
            linewidth=1.6,
            zorder=3,
        )
    )
    axes.text(
        50,
        82.0,
        "两个用户此刻说的话，逐字节完全相同",
        ha="center",
        fontsize=11.5,
        color=style.INK_SOFT,
        zorder=4,
    )
    axes.text(
        50,
        78.0,
        f"「{pair['current_input']}」",
        ha="center",
        fontsize=13.2,
        color=style.INK,
        zorder=4,
    )
    axes.text(
        50,
        71.2,
        f"当前处境属于同一类伤：{CONDITION_LABELS[pair['probe_condition_id']]}"
        f"　·　话题是「{SURFACE_LABELS[pair['sides'][0]['probe_surface_family']]}」，"
        "在两人各自的过往里都从未出现过",
        ha="center",
        fontsize=11,
        color=style.INK_SOFT,
        zorder=4,
    )

    columns = [(4.0, 46.4, "用户 A"), (53.6, 96.0, "用户 B")]
    row_top = 64.0
    row_height = 7.4
    for (left, right, name), side in zip(columns, pair["sides"], strict=True):
        width = right - left
        axes.text(
            left + 1.4,
            row_top + 2.0,
            f"{name} 的四段过往",
            fontsize=12.8,
            fontweight="bold",
            color=style.INK,
        )
        for index, history in enumerate(side["histories"]):
            y_centre = row_top - 2.4 - index * row_height
            axes.add_patch(
                Rectangle(
                    (left, y_centre - row_height / 2 + 0.5),
                    width,
                    row_height - 1.0,
                    facecolor="#FFFFFF" if index % 2 == 0 else "#F1EEE7",
                    edgecolor="none",
                    zorder=2,
                )
            )
            axes.text(
                left + 1.4,
                y_centre + 1.25,
                f"{SURFACE_LABELS[history['surface_family']]}　·　"
                f"{CONDITION_LABELS[history['condition_id']]}",
                fontsize=11.2,
                color=style.INK_SOFT,
                va="center",
                zorder=4,
            )
            axes.text(
                left + 1.4,
                y_centre - 1.65,
                f"当时：{ACTION_SHORT[history['assistant_action']]}",
                fontsize=12.0,
                color=style.INK,
                va="center",
                zorder=4,
            )
            # Outcome colours match figure 5 so the two failure modes stay visibly
            # distinct: withdrawing when presence was needed reads as "missed",
            # staying when space was needed reads as the safety endpoint.
            marker_colour = style.OUTCOME_COLOURS[history["typed_outcome"]]
            axes.add_patch(
                Circle(
                    (left + width - 12.6, y_centre - 1.65),
                    0.72,
                    facecolor=marker_colour,
                    edgecolor="none",
                    zorder=4,
                )
            )
            axes.text(
                left + width - 11.2,
                y_centre - 1.65,
                OUTCOME_LABELS[history["typed_outcome"]],
                fontsize=11.6,
                color=marker_colour,
                va="center",
                fontweight="bold" if history["positive"] else "normal",
                zorder=4,
            )

    answer_y = 20.0
    answer_colours = (style.SAGE, style.SAGE_DEEP)
    for (left, right, name), side, colour in zip(
        columns, pair["sides"], answer_colours, strict=True
    ):
        width = right - left
        axes.annotate(
            "",
            xy=((left + right) / 2, answer_y + 7.4),
            xytext=((left + right) / 2, row_top - 2.4 - 3.6 * row_height - 3.4),
            arrowprops={
                "arrowstyle": "-|>",
                "color": colour,
                "linewidth": 2.4,
                "shrinkA": 2,
                "shrinkB": 2,
            },
        )
        axes.add_patch(
            FancyBboxPatch(
                (left + 3.0, answer_y),
                width - 6.0,
                7.2,
                boxstyle="round,pad=0.7,rounding_size=1.2",
                facecolor=colour,
                edgecolor="none",
                zorder=3,
            )
        )
        axes.text(
            (left + right) / 2,
            answer_y + 4.6,
            f"{name} 此刻的正确回应",
            ha="center",
            fontsize=11,
            color="#EAF3EF",
            zorder=4,
        )
        axes.text(
            (left + right) / 2,
            answer_y + 1.9,
            ACTION_LABELS[side["preferred_action"]],
            ha="center",
            fontsize=14.2,
            fontweight="bold",
            color="#FFFFFF",
            zorder=4,
        )

    axes.text(
        50,
        answer_y + 3.6,
        "完全\n相反",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=style.AMBER,
        linespacing=1.35,
        zorder=5,
    )
    axes.text(
        50,
        10.6,
        "同一句话，两个相反的正确答案。任何只看当前这句话的方法，无论多聪明，都只能对一半。",
        ha="center",
        fontsize=12.6,
        color=style.INK,
        zorder=4,
    )
    axes.text(
        50,
        6.8,
        "两人的四段过往里，「留在场」和「给空间」各出现 2 次，且各有一次成功一次失败——"
        "历史成功率都是 50%，数不出答案。",
        ha="center",
        fontsize=11,
        color=style.INK_SOFT,
        zorder=4,
    )

    style.title_block(
        figure,
        "关系域的核心难题：没有普适的正确答案",
        "",
    )
    style.footer(
        figure,
        f"来源：scenario_packages/relationship_transfer_v3（{pair['pair_id']}）· "
        "条件、相处方式与正确动作均封存在生成器真值中，被测系统看不到",
    )

    payload = {
        "claim": "byte-identical current utterance, opposite correct actions",
        "mirror_pair": pair,
        "evidence_tier": "sealed_scenario_package_design",
        "provenance": sources.provenance(
            sources.SCENARIO_V3_TRUTH, sources.SCENARIO_V3_PUBLIC
        ),
    }
    return figure, payload


def figure_outcome_composition() -> tuple:
    """Figure 5 — both failure modes shrink, not just the positive rate rising."""

    report = sources.load(sources.RELATIONSHIP_CAMPAIGN)
    arms = [
        ("frozen_theta0", "有条件干预"),
        ("full", "有条件干预 + 施加学习信号"),
        ("strict_noop", "完全不干预"),
    ]
    stack_order = ("felt_heard", "helped", "missed", "over_directive")

    figure, axes = style.new_figure((12.2, 7.6))
    figure.subplots_adjust(left=0.235, right=0.955, top=0.80, bottom=0.30)

    shares: dict[str, dict[str, float]] = {}
    for position, (arm_id, _) in enumerate(arms):
        summary = report["arm_summary"][arm_id]
        counts = summary["outcome_counts"]
        total = summary["decision_count"]
        offset = 0.0
        shares[arm_id] = {}
        for outcome in stack_order:
            share = counts[outcome] / total * 100
            shares[arm_id][outcome] = share
            axes.barh(
                position,
                share,
                left=offset,
                height=0.56,
                color=style.OUTCOME_COLOURS[outcome],
                zorder=3,
                edgecolor=style.PAPER,
                linewidth=1.2,
            )
            if share >= 6.0:
                axes.text(
                    offset + share / 2,
                    position,
                    f"{share:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=11.2,
                    color="#FFFFFF" if outcome != "missed" else style.INK,
                    fontweight="bold",
                    zorder=4,
                )
            offset += share
        positive = sum(shares[arm_id][outcome] for outcome in ("felt_heard", "helped"))
        axes.annotate(
            f"正向 {positive:.2f}%",
            (101.5, position),
            va="center",
            fontsize=12.2,
            color=style.SAGE if positive > 50 else style.INK_SOFT,
            fontweight="bold",
        )

    axes.set_yticks(range(len(arms)))
    axes.set_yticklabels([label for _, label in arms], fontsize=12.2)
    axes.set_xlim(0, 118)
    axes.set_xticks([0, 20, 40, 60, 80, 100])
    axes.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    axes.set_xlabel("每组 4,480 个决策的反应构成", fontsize=12.5, labelpad=10)
    axes.tick_params(axis="y", length=0)
    axes.invert_yaxis()
    style.strip_frame(axes, keep=("bottom",))

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=style.OUTCOME_COLOURS[outcome], edgecolor="none")
        for outcome in stack_order
    ]
    axes.legend(
        handles,
        [
            f"{OUTCOME_LABELS[outcome]}"
            + ("（正向）" if outcome in POSITIVE_OUTCOMES else "（负向）")
            for outcome in stack_order
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        fontsize=11.5,
        handlelength=1.5,
        columnspacing=2.0,
    )

    contrast = report["contrasts"]["steerable_frozen_theta0_minus_strict_noop"]
    point = float.fromhex(contrast["primary"]["point_estimate_hex"])
    lower = float.fromhex(contrast["primary"]["simultaneous_95_lower_hex"])
    floor = float.fromhex(contrast["primary"]["practical_floor_hex"])
    safety = float.fromhex(contrast["safety"]["point_risk_difference_hex"])

    style.title_block(
        figure,
        "不只是正向变多，两种失败都在缩小",
        "112 个虚拟个体 × 3 组 × 每组 40 个评估决策 = 13,440 条决策，20,000 次配对重采样。\n"
        f"有条件干预 vs 完全不干预：正向率差 {point * 100:.2f} 个百分点，"
        f"95% 同时置信下界 {lower * 100:.2f}（预注册实用地板 {floor * 100:.2f}）；"
        f"安全侧「节奏没被尊重」同时下降 {abs(safety) * 100:.2f}。",
    )
    figure.text(
        0.235,
        0.072,
        "上面两组的构成完全相同——4,480 个决策里动作分歧 0 个。\n"
        "所以「施加学习信号」这一层的独立贡献，本次没有测出来（见图 8）。",
        fontsize=10.8,
        color=style.AMBER,
        va="bottom",
        linespacing=1.6,
    )
    style.footer(
        figure,
        "来源：relationship_product_horizon_development_campaign_20260827 · "
        "开发档 · 112 个 root 为合成个体而非真人 · 判词为「可据此设计正式功效实验」",
    )

    payload = {
        "claim": "conditioned intervention raises positive rate and shrinks both failure modes",
        "outcome_shares_percent": shares,
        "steerable_point_estimate": point,
        "steerable_simultaneous_95_lower": lower,
        "practical_floor": floor,
        "safety_point_risk_difference": safety,
        "steerable_status": contrast["status"],
        "learnable_status": report["contrasts"][
            "learnable_full_minus_frozen_theta0"
        ]["status"],
        "bootstrap_replicates": report["bootstrap"]["replicate_count"],
        "evidence_tier": "development_tier_synthetic_roots",
        "provenance": sources.provenance(sources.RELATIONSHIP_CAMPAIGN),
    }
    return figure, payload


def figure_longitudinal_timeline() -> tuple:
    """Figure 6 — the person changes, and disappears for two weeks. Twice."""

    protocol = sources.load(sources.RELATIONSHIP_SOURCE_V4)
    cohort = protocol["cohort"]
    segments = protocol["schedule"]["segments"]
    onboarding = cohort["onboarding_sessions_per_root"]

    figure, axes = style.new_figure((13.6, 6.1))
    figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100)
    axes.axis("off")

    unit = 1.32
    gap_width = 3.6
    cursor = 4.0
    bar_y = 44.0
    bar_height = 9.5

    axes.add_patch(
        Rectangle(
            (cursor, bar_y),
            onboarding * unit,
            bar_height,
            facecolor=style.GREY_DARK,
            edgecolor="none",
            zorder=3,
        )
    )
    axes.text(
        cursor + onboarding * unit / 2,
        bar_y + bar_height / 2,
        f"{onboarding}",
        ha="center",
        va="center",
        fontsize=12,
        color="#FFFFFF",
        fontweight="bold",
        zorder=4,
    )
    axes.text(
        cursor + onboarding * unit / 2,
        bar_y + bar_height + 3.4,
        "开场",
        ha="center",
        fontsize=11.8,
        color=style.GREY_DARK,
        fontweight="bold",
    )
    axes.text(
        cursor,
        bar_y - 5.2,
        "这个人按\n甲型行事",
        fontsize=11,
        color=style.GREY_DARK,
        va="top",
        linespacing=1.5,
    )
    cursor += onboarding * unit

    flip_x = cursor + 1.6
    axes.annotate(
        "",
        xy=(flip_x, bar_y - 2.0),
        xytext=(flip_x, bar_y + bar_height + 12.5),
        arrowprops={"arrowstyle": "-", "color": style.AMBER, "linewidth": 2.4},
        zorder=5,
    )
    # Left-anchored so the long caption grows into the empty band above the
    # segments instead of running off the left edge of the canvas.
    axes.text(
        flip_x + 1.4,
        bar_y + bar_height + 14.6,
        "相处方式在这里翻转",
        ha="left",
        fontsize=12.8,
        fontweight="bold",
        color=style.AMBER,
    )
    axes.text(
        flip_x + 1.4,
        bar_y + bar_height + 10.4,
        "从此按乙型行事——系统刚学会的那套规律，从这一刻起全是错的",
        ha="left",
        fontsize=11.4,
        color=style.AMBER,
    )
    cursor = flip_x + 1.6

    placements = []
    for segment in segments:
        gap_days = segment["minimum_gap_before_days"]
        if gap_days:
            axes.add_patch(
                Rectangle(
                    (cursor, bar_y + bar_height / 2 - 0.35),
                    gap_width,
                    0.7,
                    facecolor=style.RULE,
                    edgecolor="none",
                    zorder=2,
                )
            )
            axes.text(
                cursor + gap_width / 2,
                bar_y + bar_height / 2 + 2.6,
                f"{gap_days} 天\n空白",
                ha="center",
                va="bottom",
                fontsize=10.4,
                color=style.INK_SOFT,
                linespacing=1.4,
            )
            cursor += gap_width
        width = segment["decision_count"] * unit
        is_excluded = segment["segment_id"] == "matched_collection"
        is_key = segment["segment_id"] in ("post_reversal", "return_after_gap")
        axes.add_patch(
            Rectangle(
                (cursor, bar_y),
                width,
                bar_height,
                facecolor=style.RULE if is_excluded else style.SAGE,
                edgecolor=style.AMBER if is_key else "none",
                linewidth=2.0 if is_key else 0,
                zorder=3,
            )
        )
        axes.text(
            cursor + width / 2,
            bar_y + bar_height / 2,
            f"{segment['decision_count']}",
            ha="center",
            va="center",
            fontsize=12,
            color=style.INK_SOFT if is_excluded else "#FFFFFF",
            fontweight="bold",
            zorder=4,
        )
        axes.text(
            cursor + width / 2,
            bar_y + bar_height + 3.4,
            SEGMENT_LABELS[segment["segment_id"]],
            ha="center",
            fontsize=11.8,
            color=style.AMBER if is_key else (style.INK_SOFT if is_excluded else style.INK),
            fontweight="bold" if is_key else "normal",
        )
        axes.text(
            cursor + width / 2,
            bar_y - 5.2,
            SEGMENT_PROBES[segment["segment_id"]],
            ha="center",
            va="top",
            fontsize=10.2,
            color=style.INK_SOFT,
            linespacing=1.5,
        )
        placements.append({"segment_id": segment["segment_id"], **segment})
        cursor += width

    axes.annotate(
        "",
        xy=(cursor + 0.8, bar_y + bar_height / 2),
        xytext=(3.0, bar_y + bar_height / 2),
        arrowprops={"arrowstyle": "-|>", "color": style.INK, "linewidth": 1.2},
        zorder=1,
    )
    axes.text(cursor + 1.6, bar_y + bar_height / 2, "时间", fontsize=11.5, va="center", color=style.INK)

    style.title_block(
        figure,
        "产品真正要面对的：人会变，中间还会消失两周",
        f"每个虚拟个体一生 {onboarding} 次开场 + {cohort['collection_decisions_per_root']} 次采集 + "
        f"{cohort['evaluation_decisions_per_root']} 次评估，共 "
        f"{onboarding + cohort['collection_decisions_per_root'] + cohort['evaluation_decisions_per_root']} 次互动。"
        f"合成个体 {cohort['root_count']} 个，方格里的数字是该段的决策次数。",
    )
    figure.text(
        0.035,
        0.125,
        "在这里，「记得更多」帮不上忙——旧的记忆恰恰是错的。需要的是发现规律变了，并改掉它。",
        fontsize=12.6,
        color=style.INK,
        va="bottom",
    )
    style.footer(
        figure,
        "来源：relationship_product_horizon_source_v4_admission_20260826 · "
        "虚拟日历，两个 14 天间隔在协议中预先冻结 · 匹配采集段不计入效果统计",
    )

    payload = {
        "claim": "the longitudinal design flips the person's policy and inserts two 14-day gaps",
        "onboarding_sessions_per_root": onboarding,
        "collection_decisions_per_root": cohort["collection_decisions_per_root"],
        "evaluation_decisions_per_root": cohort["evaluation_decisions_per_root"],
        "root_count": cohort["root_count"],
        "evaluation_policy": protocol["policies"]["evaluation_policy"],
        "segments": placements,
        "evidence_tier": "frozen_source_admission_protocol",
        "provenance": sources.provenance(sources.RELATIONSHIP_SOURCE_V4),
    }
    return figure, payload


__all__ = [
    "figure_longitudinal_timeline",
    "figure_mirror_pair",
    "figure_outcome_composition",
]
