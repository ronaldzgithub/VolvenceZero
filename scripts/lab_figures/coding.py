"""Coding Lab figures: what memory buys, and why timing beats always acting."""

from __future__ import annotations

import collections

from matplotlib.patches import Rectangle

from . import sources, style

#: The house convention is only enforceable on tasks that introduce a new public
#: symbol; ``fix_bug`` and ``refactor_alias`` never inject the hidden test, so
#: including them would score questions that were not on the exam.
CONVENTION_APPLICABLE = ("add_helper", "extend_report", "config_feature")

#: ``add_helper`` is the only applicable category that recurs inside one chain
#: (it cycles a pool of distinct helper names), so it is the only place where
#: cross-episode memory can pay off at all.
RECURRING_CATEGORY = "add_helper"

CONVENTION_ID = "convention_export_all"

ARM_LABELS = {
    "brain": "结构化经历（我们）",
    "steelman": "全历史堆上下文",
    "stateless": "无记忆",
}
ARM_ORDER = ("brain", "steelman", "stateless")


def _violation_curve(episodes: list[dict], arm: str) -> list[tuple[int, int, int]]:
    """Violation counts per within-chain occurrence of the recurring category."""

    selected = sorted(
        (
            episode
            for episode in episodes
            if episode["arm"] == arm and episode["category"] == RECURRING_CATEGORY
        ),
        key=lambda episode: (episode["chain_index"], episode["episode_index"]),
    )
    seen_per_chain: collections.Counter[int] = collections.Counter()
    grouped: dict[int, list[bool]] = collections.defaultdict(list)
    for episode in selected:
        occurrence = seen_per_chain[episode["chain_index"]]
        seen_per_chain[episode["chain_index"]] += 1
        grouped[occurrence].append(CONVENTION_ID in episode["invariant_violations"])
    return [
        (occurrence + 1, sum(flags), len(flags))
        for occurrence, flags in sorted(grouped.items())
    ]


def figure_convention_learning() -> tuple:
    """Figure 1 — the hidden house convention: only structured memory learns it."""

    report = sources.load(sources.CODING_PACKET2)
    episodes = report["episodes"]
    curves = {arm: _violation_curve(episodes, arm) for arm in ARM_ORDER}

    control_arms = ("stateless", "steelman")
    controls_identical = len({tuple(curves[arm]) for arm in control_arms}) == 1
    control_flat_at_ceiling = controls_identical and all(
        violations == total for _, violations, total in curves["stateless"]
    )

    figure, axes = style.new_figure((12.4, 7.6))
    figure.subplots_adjust(left=0.115, right=0.705, top=0.795, bottom=0.145)

    # The two control arms sit on exactly the same points, so they are drawn as
    # one visual band (grey underneath, amber dashed on top) and share a label.
    for arm, linestyle, width in (("stateless", "solid", 5.0), ("steelman", (0, (6, 4)), 2.6)):
        points = curves[arm]
        axes.plot(
            [point[0] for point in points],
            [point[1] / point[2] * 100 for point in points],
            marker="o",
            markersize=8,
            linestyle=linestyle,
            linewidth=width,
            color=style.ARM_COLOURS[arm],
            zorder=3,
            solid_capstyle="round",
        )

    brain_points = curves["brain"]
    brain_x = [point[0] for point in brain_points]
    brain_y = [point[1] / point[2] * 100 for point in brain_points]
    axes.plot(
        brain_x,
        brain_y,
        marker="o",
        markersize=12,
        linewidth=3.8,
        color=style.SAGE,
        zorder=5,
        solid_capstyle="round",
    )

    for x_value, y_value, (_, violations, total) in zip(
        brain_x, brain_y, brain_points, strict=True
    ):
        axes.annotate(
            f"{violations}/{total}\n{y_value:.0f}%",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, -40) if x_value == 1 else (0, 17),
            ha="center",
            fontsize=11,
            color=style.SAGE,
            fontweight="bold",
            linespacing=1.35,
        )
    for x_value, (_, violations, total) in zip(brain_x, curves["stateless"], strict=True):
        axes.annotate(
            f"{violations}/{total}",
            (x_value, 100),
            textcoords="offset points",
            xytext=(0, 13),
            ha="center",
            fontsize=10.5,
            color=style.INK_SOFT,
        )

    axes.annotate(
        ARM_LABELS["brain"],
        (brain_x[-1], brain_y[-1]),
        textcoords="offset points",
        xytext=(18, -4),
        va="center",
        fontsize=13,
        color=style.SAGE,
        fontweight="bold",
    )
    control_label = (
        "全历史堆上下文　·　无记忆\n两组完全重合，始终 100%"
        if control_flat_at_ceiling
        else f"{ARM_LABELS['steelman']}　·　{ARM_LABELS['stateless']}"
    )
    axes.annotate(
        control_label,
        (brain_x[-1], 100),
        textcoords="offset points",
        xytext=(18, -8),
        va="center",
        fontsize=12.5,
        color=style.AMBER,
        linespacing=1.5,
    )

    axes.set_xlim(0.8, 4.2)
    axes.set_ylim(-10, 118)
    axes.set_xticks([1, 2, 3, 4])
    axes.set_xticklabels(["第 1 次", "第 2 次", "第 3 次", "第 4 次"], fontsize=13)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes.set_xlabel("同一条任务链里，这类任务第几次出现", fontsize=12.5, labelpad=12)
    axes.set_ylabel("违反那条约定的比例", fontsize=12.5, labelpad=12)
    style.strip_frame(axes)

    axes.text(
        1.04,
        7.0,
        "三组都从 100% 出发——环境里零信号，谁都不可能猜到这条约定存在。\n"
        "差距只可能来自后面几次里学到了什么，不可能来自能力或提示词。",
        fontsize=11,
        color=style.INK_SOFT,
        va="bottom",
        linespacing=1.6,
    )

    style.title_block(
        figure,
        "没人告诉它的那条规矩，只有一组学会了",
        "环境里埋了一条团队编码约定：任何地方都不提，只由评测时注入的隐藏测试检查。\n"
        f"下图只统计该约定适用、且会在链内重复出现的那类任务（{RECURRING_CATEGORY}），三组任务序列完全相同。",
    )
    steelman_totals = curves["steelman"]
    steelman_violations = sum(violations for _, violations, _ in steelman_totals)
    steelman_episodes = sum(total for _, _, total in steelman_totals)
    figure.text(
        0.718,
        0.60,
        "全历史组的上下文里\n逐字装着前面每一次\n失败的完整记录——\n"
        f"{steelman_episodes} 次里错 {steelman_violations} 次。\n\n"
        "看得见，不等于学得会。",
        fontsize=11.5,
        va="top",
        color=style.AMBER,
        linespacing=1.7,
    )
    style.footer(
        figure,
        "来源：coding_lab_packet2_formal_v2_qwen3codernext_20260813 · 240 回合正式运行 · "
        "该读数为冻结产物的事后描述性切分，非预注册判定端点",
    )

    payload = {
        "claim": "only the structured-memory arm learns an unstated house convention across episodes",
        "recurring_category": RECURRING_CATEGORY,
        "convention_id": CONVENTION_ID,
        "curves": {
            arm: [
                {"occurrence": occurrence, "violations": violations, "episodes": total}
                for occurrence, violations, total in curves[arm]
            ]
            for arm in ARM_ORDER
        },
        "control_arms_identical": controls_identical,
        "control_arms_flat_at_ceiling": control_flat_at_ceiling,
        "evidence_tier": "development_descriptive_recut_of_frozen_artifact",
        "provenance": sources.provenance(sources.CODING_PACKET2),
    }
    return figure, payload


def figure_memory_pareto() -> tuple:
    """Figure 2 — cheaper and better than stuffing the whole history in."""

    report = sources.load(sources.CODING_PACKET2)
    pass_rates = report["pass_rates"]
    scaling = report["scaling"]
    # Use the report's own preregistered scaling measure rather than recomputing:
    # it averages over episodes that actually carry context, i.e. it excludes each
    # chain's first episode, where no prior history exists yet for any arm.
    context_tokens = {
        "brain": scaling["brain_mean_context_tokens"],
        "steelman": scaling["steelman_mean_context_tokens"],
        "stateless": 0.0,
    }

    figure, axes = style.new_figure((12.2, 7.4))
    figure.subplots_adjust(left=0.09, right=0.965, top=0.80, bottom=0.145)

    brain_x = context_tokens["brain"]
    brain_y = pass_rates["brain"] * 100
    axes.add_patch(
        Rectangle(
            (brain_x, 0),
            10_500,
            brain_y,
            facecolor=style.AMBER,
            alpha=0.075,
            zorder=1,
            linewidth=0,
        )
    )
    axes.annotate(
        "更贵、而且更差",
        (brain_x + 4800, brain_y - 5.4),
        fontsize=12,
        color=style.AMBER,
        ha="center",
        style="italic",
    )

    for arm in ARM_ORDER:
        x_value = context_tokens[arm]
        y_value = pass_rates[arm] * 100
        is_ours = arm == "brain"
        axes.scatter(
            [x_value],
            [y_value],
            s=430 if is_ours else 250,
            color=style.ARM_COLOURS[arm],
            zorder=5,
            edgecolor=style.PAPER,
            linewidth=2.0,
        )
        # Labels flip to the left once a point sits near the right edge, so the
        # widest arm label cannot run off the canvas.
        anchor_right = x_value > 6_500
        offsets = {
            "brain": (18, 20),
            "steelman": (-16, -30),
            "stateless": (18, -30),
        }
        axes.annotate(
            f"{ARM_LABELS[arm]}\n通过率 {y_value:.2f}%　带入 {x_value:,.0f} 词元",
            (x_value, y_value),
            textcoords="offset points",
            xytext=offsets[arm],
            ha="right" if anchor_right else "left",
            fontsize=12,
            color=style.ARM_COLOURS[arm],
            fontweight="bold" if is_ours else "normal",
            linespacing=1.5,
        )

    axes.annotate(
        "",
        xy=(context_tokens["brain"], brain_y),
        xytext=(context_tokens["steelman"], pass_rates["steelman"] * 100),
        arrowprops={
            "arrowstyle": "-|>",
            "color": style.SAGE,
            "linewidth": 2.0,
            "shrinkA": 12,
            "shrinkB": 16,
            "connectionstyle": "arc3,rad=0.16",
        },
        zorder=4,
    )
    ratio = scaling["token_ratio"]
    axes.annotate(
        f"带入内容降到 {ratio * 100:.2f}%（约 1/10）\n通过率反而更高",
        ((context_tokens["brain"] + context_tokens["steelman"]) / 2, 48.5),
        fontsize=11.5,
        color=style.SAGE,
        ha="center",
        va="center",
        linespacing=1.55,
    )

    axes.set_xlim(-700, 10_600)
    axes.set_ylim(36, 55)
    axes.set_xlabel("每回合带入模型的上下文（词元，越少越省）", fontsize=12.5, labelpad=10)
    axes.set_ylabel("任务通过率", fontsize=12.5, labelpad=10)
    axes.set_yticks([38, 42, 46, 50, 54])
    axes.set_yticklabels(["38%", "42%", "46%", "50%", "54%"])
    axes.set_xticks([0, 2000, 4000, 6000, 8000, 10000])
    axes.set_xticklabels(["0", "2,000", "4,000", "6,000", "8,000", "10,000"])
    style.strip_frame(axes)

    style.title_block(
        figure,
        "又便宜，又不更差：左上角才是我们想去的地方",
        "同一个商用编程模型、同样的题、同样的顺序，240 个回合。三组的差别只有一个——带什么进考场。\n"
        "预注册判定：对无记忆的增益下界 +0.0061（正）；对全历史的非劣下界 −0.0046（阈值 −0.05）；词元比 ≤ 0.10。",
    )
    style.footer(
        figure,
        "来源：coding_lab_packet2_formal_v2_qwen3codernext_20260813 · "
        f"预注册指纹 {report['prereg_sha256'][:12]}… · 三条门全过",
    )

    payload = {
        "claim": "structured memory Pareto-dominates full-history context stuffing",
        "pass_rates": pass_rates,
        "mean_context_tokens": context_tokens,
        "mean_context_tokens_source": (
            "report.scaling (preregistered measure; excludes each chain's first episode, "
            "which carries no prior context for any arm)"
        ),
        "token_ratio_brain_over_steelman": ratio,
        "scaling_gate_max_token_ratio": scaling["max_token_ratio"],
        "prereg_sha256": report["prereg_sha256"],
        "gates": report["verdicts"],
        "memory_gate_lower_bound": report["quality_brain_vs_stateless"][
            "bootstrap_ci_lower_5pct"
        ],
        "noninferiority_lower_bound": report["quality_brain_vs_steelman"][
            "bootstrap_ci_lower_5pct"
        ],
        "evidence_tier": "preregistered_formal_gates_passed",
        "provenance": sources.provenance(sources.CODING_PACKET2),
    }
    return figure, payload


def figure_when_to_steer() -> tuple:
    """Figure 3 — always intervening is worse than never intervening."""

    report = sources.load(sources.CODING_PACKET3)
    aggregate = report["aggregate"]
    thresholds = report["thresholds"]

    rows = [
        ("一直干预", aggregate["always_on_belief_nll_mean"], style.AMBER, True),
        ("随机干预", aggregate["random_gate_nll_mean"], style.GREY, False),
        ("从不干预", aggregate["noop_nll_mean"], style.GREY_DARK, False),
        ("学到的择时干预（我们）", aggregate["pe_gated_online_nll_mean"], style.SAGE, False),
    ]
    oracle = aggregate["oracle_gate_ceiling_nll_mean"]

    figure, axes = style.new_figure((12.2, 7.2))
    figure.subplots_adjust(left=0.235, right=0.955, top=0.795, bottom=0.155)

    positions = range(len(rows))
    for position, (_label, value, colour, highlight) in zip(positions, rows, strict=True):
        axes.barh(
            position,
            value,
            height=0.56,
            color=colour,
            zorder=3,
            edgecolor=style.AMBER if highlight else "none",
            linewidth=1.8 if highlight else 0,
        )
        axes.annotate(
            f"{value:.4f}",
            (value, position),
            textcoords="offset points",
            xytext=(11, 0),
            va="center",
            fontsize=12.5,
            color=colour,
            fontweight="bold",
        )

    axes.axvline(oracle, color=style.INK_SOFT, linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    axes.text(
        oracle + 0.13,
        3.78,
        f"全知上限 {oracle:.4f}（作弊组，只作参照）",
        fontsize=10.5,
        color=style.INK_SOFT,
        va="center",
    )

    axes.set_yticks(list(positions))
    axes.set_yticklabels([row[0] for row in rows], fontsize=12.5)
    axes.set_xlim(0, 6.05)
    axes.set_ylim(-0.78, 4.1)
    axes.set_xlabel("决策点上的判断误差（越小越接近专家；0 为完美）", fontsize=12.5, labelpad=12)
    axes.tick_params(axis="y", length=0)
    style.strip_frame(axes, keep=("bottom",))

    captured = (aggregate["noop_nll_mean"] - aggregate["pe_gated_online_nll_mean"]) / (
        aggregate["noop_nll_mean"] - oracle
    )
    axes.text(
        6.0,
        3.78,
        f"可改善空间捕获 {captured * 100:.0f}%",
        fontsize=11.5,
        color=style.SAGE,
        ha="right",
        va="center",
        fontweight="bold",
    )
    axes.text(
        6.0,
        3.45,
        f"门控选择性 {aggregate['gate_selectivity_mean']:.3f}　·　"
        f"基座可训练参数 {report['substrate_trainable_parameter_count']}",
        fontsize=10.5,
        color=style.INK_SOFT,
        ha="right",
        va="center",
    )
    axes.text(
        0.18,
        0,
        f"比「从不干预」还差 "
        f"{aggregate['always_on_belief_nll_mean'] - aggregate['noop_nll_mean']:.2f}"
        f"　——　乱出手会主动把系统弄坏",
        fontsize=12,
        color="#FFFFFF",
        va="center",
        fontweight="bold",
        zorder=5,
    )

    style.title_block(
        figure,
        "会出手不值钱，知道什么时候出手才值钱",
        "冻结的 15.4 亿参数代码模型，838 条真实轨迹抽出的决策点，5 组独立随机种子。\n"
        f"六条预注册门全过：最差种子对「从不干预」的优势下界 "
        f"{aggregate['gain_vs_noop_ci_lower_min']:.4f}（阈值 {thresholds['min_gain_vs_noop_nll']}）。",
    )
    style.footer(
        figure,
        "来源：coding_lab_packet3_s3e_formal_20260813 · 全程 SHADOW，未改动线上接线 · "
        "衡量对象为决策点判断质量，不是端到端任务通过率",
    )

    payload = {
        "claim": "a learned gate beats never / always / random intervening on when to act",
        "nll_by_strategy": {
            "always_on_belief": aggregate["always_on_belief_nll_mean"],
            "random_gate": aggregate["random_gate_nll_mean"],
            "noop": aggregate["noop_nll_mean"],
            "pe_gated_online": aggregate["pe_gated_online_nll_mean"],
            "oracle_ceiling": oracle,
        },
        "headroom_captured_fraction": captured,
        "worst_seed_gain_vs_noop_ci_lower": aggregate["gain_vs_noop_ci_lower_min"],
        "gate_selectivity_mean": aggregate["gate_selectivity_mean"],
        "seed_count": aggregate["seed_count"],
        "admitted": report["admission"]["admitted"],
        "production_wiring_changed": report["production_wiring_changed"],
        "evidence_tier": "preregistered_formal_gates_passed_shadow_only",
        "provenance": sources.provenance(sources.CODING_PACKET3),
    }
    return figure, payload


__all__ = [
    "CONVENTION_APPLICABLE",
    "RECURRING_CATEGORY",
    "figure_convention_learning",
    "figure_memory_pareto",
    "figure_when_to_steer",
]
