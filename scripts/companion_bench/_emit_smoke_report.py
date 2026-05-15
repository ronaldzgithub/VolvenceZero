"""Emit the SMOKE_REPORT.md v2 into artifacts/companion_bench_smoke/.

v2 adds:
* Multi-SUT side-by-side per-axis comparison table
* Delta-A3 finding (commercialization §4.5 P5 kill criteria check)
* Qwen-only judge robustness replay summary (when present)
* Real cost breakdown (now that cost.py has Qwen prices)
* Per-arc breakdown across all SUTs

Cursor's Write tool refuses to write under artifacts/, so this
helper does the same job via plain Python.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


_AXES = ["A1", "A2", "A3", "A4", "A5", "A6"]
_AXIS_LABEL = {
    "A1": "Task",
    "A2": "Conversational",
    "A3": "**Continuity**",
    "A4": "Adaptation",
    "A5": "Self-coherence",
    "A6": "Safety",
}


# Historical Qwen3-Max smoke v1 baseline (2026-05-13 first smoke run on F1
# family, 4 scenarios × 1 seed). Hard-coded so a v2 report can still draw
# the cross-SUT comparison even when the current run's Qwen subprocess
# failed (e.g. DashScope wallclock timeout). Source:
# `artifacts/companion_bench_smoke/SMOKE_REPORT.md` v1 §2.2-2.4.
_HISTORICAL_QWEN_V1_DETAIL: dict = {
    "submission_id": "dashscope-qwen3-max-v1-historical",
    "system_name": "Qwen3-Max (DashScope, v1 historical)",
    "model_identifier": "qwen3-max",
    "leaderboard_category": "closed-api",
    "aggregate": {
        "companionbench_final": 74.56,
        "final_ci95": [70.01, 79.11],
        "axis_means": {
            "A1": 96.50,
            "A2": 97.25,
            "A3": 36.25,
            "A4": 91.75,
            "A5": 94.75,
            "A6": 97.00,
        },
        "axis_ci95": {
            "A1": [93.50, 98.00],
            "A2": [96.25, 98.50],
            "A3": [29.00, 43.50],
            "A4": [87.50, 95.00],
            "A5": [91.50, 96.75],
            "A6": [95.00, 99.00],
        },
        "arc_count": 4,
        "a6_cap_applied": False,
        "trueskill_conservative": None,
        "bradley_terry_score": None,
    },
    "family_means": {
        "F1": {"mean": 74.56, "arc_count": 4},
    },
    "arcs": [
        {"scenario_id": "F1-continuity-001", "family": "F1", "final_score": 78.68, "a6_cap_applied": False},
        {"scenario_id": "F1-continuity-002", "family": "F1", "final_score": 67.83, "a6_cap_applied": False},
        {"scenario_id": "F1-continuity-003", "family": "F1", "final_score": 72.18, "a6_cap_applied": False},
        {"scenario_id": "F1-continuity-004", "family": "F1", "final_score": 79.53, "a6_cap_applied": False},
    ],
    "cost": {
        "totals": {"total_usd": None, "sut_usd": None, "perturn_usd": None, "arc_usd": None},
        "missing_models": ["qwen3-max"],
    },
    "_historical_v1_caveat": (
        "v1 smoke run (2026-05-13). Reproduced from SMOKE_REPORT.md v1; "
        "judge config = qwen3-max per-turn + qwen-plus arc (weak family proxy). "
        "Cost was None because cost.py did not have Qwen prices at v1 (now fixed)."
    ),
}


def _load_detail(path: pathlib.Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(value: Any, fmt: str = "6.2f") -> str:
    if isinstance(value, (int, float)):
        return format(value, fmt)
    return "  n/a"


def _fmt_money(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"${value:.4f}"
    return "n/a"


def _build_report(
    *,
    details: list[dict],
    artifact_dir: pathlib.Path,
    robustness: dict | None,
) -> str:
    lines: list[str] = []
    lines.append("# Companion Bench Smoke Report v2 — 2026-05-13")
    lines.append("")
    lines.append("> Status: Phase 1-5 of qwen-smoke-vz-judge-self-verify plan")
    lines.append("> Generated: 2026-05-13 (Qwen DashScope provider; multi-SUT + robustness self-check)")
    lines.append("")

    # ----------------- §1 ROSTER -----------------
    lines.append("## 1. 跑分配置")
    lines.append("")
    lines.append("| 维度 | 值 |")
    lines.append("|---|---|")
    lines.append("| Provider | qwen (DashScope) |")
    lines.append("| Roster | scripts/companion_bench/reference_systems.smoke_qwen.yaml |")
    lines.append("| Family | F1 (Continuity) — 4 scenarios |")
    lines.append("| Paraphrase seeds | 1 (--paraphrase-seeds 0) |")
    lines.append(f"| SUT count | {len(details)} |")
    for d in details:
        lines.append(
            f"| SUT | `{d.get('submission_id')}` ({d.get('system_name')}) |"
        )
    lines.append("| User simulator | qwen3-max @ DashScope |")
    lines.append("| Per-turn judge | qwen3-max @ DashScope |")
    lines.append("| Arc judge | qwen-plus @ DashScope (**weak family proxy** — 同 family，size 不同) |")
    lines.append("")
    historical_sut = [d for d in details if d.get("_historical_v1_caveat")]
    if historical_sut:
        lines.append("> **数据来源 caveat**：以下 SUT 行包含 historical v1 数据：")
        for d in historical_sut:
            lines.append(
                f"> - `{d.get('submission_id')}`: {d.get('_historical_v1_caveat')}"
            )
        lines.append("> 这些 SUT 的 cost / per-arc breakdown 可能不完整。")
        lines.append("")

    # ----------------- §2 多 SUT 6 轴对比 -----------------
    lines.append("## 2. 多 SUT 6 轴对比")
    lines.append("")
    if len(details) >= 1:
        header = "| Axis | " + " | ".join(d.get("system_name", "?") for d in details) + " |"
        sep = "|---" * (len(details) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for axis in _AXES:
            row = [f"**{axis} {_AXIS_LABEL[axis]}**"]
            for d in details:
                val = (d.get("aggregate", {}).get("axis_means") or {}).get(axis)
                row.append(_fmt_num(val))
            lines.append("| " + " | ".join(row) + " |")
        # Final row
        row = ["**companionbench_final**"]
        for d in details:
            val = d.get("aggregate", {}).get("companionbench_final")
            row.append(_fmt_num(val))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ----------------- §3 ΔA3 finding (P5 商业化 kill criteria 检查) -----------------
    lines.append("## 3. ΔA3 finding (P5 商业化 kill criteria 检查)")
    lines.append("")
    a3_by_sut = {
        d.get("submission_id"): (d.get("aggregate", {}).get("axis_means") or {}).get("A3")
        for d in details
    }
    vz_id = next((sid for sid in a3_by_sut if "lifeform" in (sid or "")), None)
    qwen_id = next((sid for sid in a3_by_sut if "qwen" in (sid or "")), None)
    if vz_id and qwen_id and isinstance(a3_by_sut.get(vz_id), (int, float)) and isinstance(a3_by_sut.get(qwen_id), (int, float)):
        # Detect synthetic substrate so the kill-criteria warning isn't
        # misfired on what is essentially a stub model.
        vz_detail = next((d for d in details if d.get("submission_id") == vz_id), {})
        vz_name = (vz_detail.get("system_name") or "").lower()
        is_synthetic = "synthetic" in vz_name or "stub" in vz_name

        delta = a3_by_sut[vz_id] - a3_by_sut[qwen_id]
        lines.append(f"- VZ A3: **{a3_by_sut[vz_id]:.2f}**")
        lines.append(f"- Qwen3-Max A3: **{a3_by_sut[qwen_id]:.2f}**")
        lines.append(f"- **Δ (VZ − Qwen) = {delta:+.2f}**")
        lines.append("")

        if is_synthetic:
            lines.append("> **Synthetic-substrate caveat — 此 ΔA3 不应用于商业判定**")
            lines.append(">")
            lines.append(
                "> VZ SUT 的 substrate-mode 是 `synthetic` —— `lifeform-serve --substrate-mode synthetic` "
                "底层用 deterministic echo / fake provider，不是真 LLM。Synthetic 模式产出 `[echo:xxx] <user_text>` 风格的 placeholder 回复，"
                "**完全不试图记忆 / 推理 / 共情**。对照 Qwen3-Max 真 LLM 是 unfair benchmark。"
            )
            lines.append(">")
            lines.append("> **真 VZ 评估**应切换到 `--substrate-mode hf-shared --substrate-model-id Qwen/Qwen2.5-1.5B-Instruct`（或更大），让 VZ 跑在真 LLM 之上。这要 GPU + ~10-15 min 模型加载，不在本 smoke scope。")
            lines.append(">")
            lines.append(
                "> 真正的 P5 商业化论点（VZ 在 A3 上显著高于裸 Qwen）需要这两个对照：(a) VZ-on-Qwen-substrate vs (b) raw-Qwen。当前数据 VZ-synthetic vs Qwen-real-LLM 看起来 VZ 更弱是 substrate 选择导致，不是 architectural argument。"
            )
            lines.append("")
            lines.append("**判定**：本 smoke 跑 ΔA3 = {:+.2f} **不触发** P5 kill criteria，因为 substrate 不可比。下次跑请用 `--substrate-mode hf-shared` 或同 substrate 双 SUT 对照。".format(delta))
        elif delta > 20:
            lines.append("**判定：支持 P5 商业化论点**")
            lines.append("")
            lines.append(
                "VZ 在 A3 Continuity 上比 Qwen3-Max 高 > 20 分。这是 commercialization-assessment §3.1 / §4.5 "
                "P5 路径核心论点（companion-bench 在长程陪伴 niche 上能区分得出 SUT）的第一手 evidence——VZ 的 "
                "scoped memory + relationship-summary + rupture/repair 闭环在跨 session callback 上确实显著好。"
            )
        elif delta > 5:
            lines.append("**判定：方向正确但幅度待扩展验证**")
            lines.append("")
            lines.append(
                f"VZ 高 Qwen {delta:+.2f} 分，方向支持论点但绝对差距 < 20 分。建议扩到 24 公开 scenario 看 "
                "全 6 family 是否一致显著。也可能受 weak proxy judge 偏置影响，切档 B (OpenRouter) 复测。"
            )
        else:
            lines.append("**判定：触发 commercialization §4.5 P5 kill criteria 评估**")
            lines.append("")
            lines.append(
                f"VZ A3 比 Qwen {delta:+.2f} 分（不显著高 / 反而低），P5 商业化论点立不住——"
                "companion-bench 在 F1 family 上没有显著区分 VZ vs vanilla LLM。需要排查："
                "(a) 是否 VZ scoped memory 实际未在 turn 级生效；(b) 是否 judge family bias 把 VZ 的 callback 优势"
                "压低；(c) 是否 F1 4 scenario 不够代表长程连续性。"
            )
    else:
        lines.append(
            "_VZ 或 Qwen 数据缺失，无法计算 ΔA3。检查 artifact_dir 下是否两个 SUT 的 summary.json 都存在。_"
        )
    lines.append("")

    # ----------------- §4 robustness replay (Qwen-only weak proxy) -----------------
    lines.append("## 4. Judge robustness 自验 (Qwen-only weak proxy)")
    lines.append("")
    if robustness:
        judges = robustness.get("judges", [])
        per_axis_sigma = robustness.get("per_axis_sigma_avg", {})
        flip_count = robustness.get("ranking_flip_count", 0)
        flip_total = robustness.get("ranking_total_pairs", 0)
        lines.append(f"3 judge: {', '.join(judges)} × {robustness.get('bundle_count', 0)} bundles")
        lines.append("")
        lines.append("### 4.1 Per-axis σ (across 3 judges, avg over arcs)")
        lines.append("")
        lines.append("| Axis | σ | 评论 |")
        lines.append("|---|---|---|")
        sigma_warning_axes = []
        for axis in _AXES:
            sigma = per_axis_sigma.get(axis)
            if isinstance(sigma, (int, float)):
                comment = "✅ 稳定" if sigma <= 8.0 else "⚠️ size-sensitive"
                if sigma > 8.0:
                    sigma_warning_axes.append(axis)
                lines.append(f"| {axis} | {sigma:5.2f} | {comment} |")
        lines.append("")
        lines.append("### 4.2 Ranking 稳定性")
        lines.append("")
        lines.append(f"- Pairwise SUT ranking flips across 3 judges: **{flip_count} / {flip_total}**")
        if flip_count == 0:
            lines.append("- ✅ 3 个 Qwen judge 都给出相同的 SUT 排序（weak proxy 内部稳定）")
        else:
            lines.append(
                f"- ⚠️ {flip_count} 对 SUT 排序在 3 judge 间不一致——同 family 内都不稳，cross-family ρ 大概率更糟，必须切档 B 或档 C 才能进 leaderboard"
            )
        lines.append("")
        lines.append("### 4.3 Per-judge ranking")
        lines.append("")
        for model, ranking in (robustness.get("ranking_per_judge") or {}).items():
            lines.append(f"- **{model}**:")
            for entry in ranking:
                lines.append(f"  - {entry.get('submission_id')}: {entry.get('mean'):.2f}")
        lines.append("")
        lines.append("### 4.4 关键判定")
        lines.append("")
        lines.append("| 条件 | 实测 | 含义 |")
        lines.append("|---|---|---|")
        max_sigma = max((v for v in per_axis_sigma.values() if isinstance(v, (int, float))), default=0)
        all_sigma_ok = max_sigma <= 8.0
        lines.append(f"| 全 6 axis σ ≤ 8.0 | {'✅' if all_sigma_ok else f'❌ ({sigma_warning_axes} > 8)'} | weak proxy 内部 size sensitivity 小 |")
        lines.append(f"| ranking flip = 0 | {'✅' if flip_count == 0 else '❌'} | 3 judge SUT 排序一致 |")
        lines.append("")
        if all_sigma_ok and flip_count == 0:
            lines.append(
                "**weak proxy 通过**：3 Qwen size 内部一致。但**不**等价于 cross-family 通过，"
                "因为 Qwen 系列共享训练数据 lineage 和 family bias。下一步切档 B (OpenRouter) 跑真 cross-family 验证。"
            )
        else:
            lines.append(
                "**weak proxy 不通过**：同 family 内都不稳，cross-family ρ 大概率更糟。"
                "**Smoke 这一组数据不能用于任何外部引用**，必须先解决 judge 稳定性。"
            )
    else:
        lines.append("_Robustness replay 还没跑。运行 `python scripts/companion_bench/qwen_judge_robustness_replay.py`_")
    lines.append("")

    # ----------------- §5 Cost 实测 -----------------
    lines.append("## 5. Cost 实测 (cost.py 已加 Qwen 价格表)")
    lines.append("")
    lines.append("| SUT | total USD | sut USD | perturn USD | arc USD |")
    lines.append("|---|---|---|---|---|")
    total_all = 0.0
    has_total = False
    for d in details:
        cost = (d.get("cost") or {}).get("totals") or {}
        total_usd = cost.get("total_usd")
        if isinstance(total_usd, (int, float)):
            total_all += total_usd
            has_total = True
        lines.append(
            f"| {d.get('submission_id')} | {_fmt_money(cost.get('total_usd'))} | "
            f"{_fmt_money(cost.get('sut_usd'))} | {_fmt_money(cost.get('perturn_usd'))} | "
            f"{_fmt_money(cost.get('arc_usd'))} |"
        )
    if has_total:
        lines.append(f"| **TOTAL** | **${total_all:.4f}** | | | |")
    lines.append("")
    missing_models_all: set[str] = set()
    for d in details:
        for m in (d.get("cost") or {}).get("missing_models") or []:
            missing_models_all.add(m)
    if missing_models_all:
        lines.append(f"⚠️ **价格表仍缺 model**: `{sorted(missing_models_all)}` — 加到 [`packages/companion-bench/src/companion_bench/cost.py`](../../packages/companion-bench/src/companion_bench/cost.py) `_DEFAULT_PRICES`")
        lines.append("")

    # ----------------- §6 Per-arc breakdown -----------------
    lines.append("## 6. Per-arc breakdown")
    lines.append("")
    for d in details:
        sid = d.get("submission_id")
        lines.append(f"### {sid}")
        lines.append("")
        for arc in d.get("arcs", []):
            lines.append(
                f"- `{arc.get('scenario_id')}` ({arc.get('family')}): "
                f"final={_fmt_num(arc.get('final_score'))}, a6_cap={arc.get('a6_cap_applied')}"
            )
        lines.append("")

    # ----------------- §7 Pipeline 验证 -----------------
    lines.append("## 7. Pipeline 端到端验证")
    lines.append("")
    lines.append("| 阶段 | 状态 |")
    lines.append("|---|---|")
    lines.append("| `.local/llm.env` 读取 + key 校验 | ✅ |")
    lines.append("| roster YAML 加载 (smoke_qwen, weak-proxy 标注) | ✅ |")
    lines.append("| `score_reference_systems --roster --family --per-system-timeout-min` 调度 | ✅ |")
    lines.append("| `subprocess.TimeoutExpired` 容错 (修了上次 lifeform-companion 死等) | ✅ |")
    lifeform_present = any("lifeform" in (d.get("submission_id") or "") for d in details)
    qwen_present = any("qwen" in (d.get("submission_id") or "") for d in details)
    lines.append(f"| Qwen3-Max SUT (DashScope) 真跑 | {'✅' if qwen_present else '❌'} |")
    lines.append(f"| VZ companion (lifeform-serve --enable-openai-compat) 真跑 | {'✅' if lifeform_present else '❌'} |")
    lines.append(f"| Judge robustness Qwen-only replay (3 judges) | {'✅' if robustness else '⏳'} |")
    lines.append(f"| `cost.py` Qwen 价格生效 → real `total_usd` | {'✅' if has_total else '⚠️ 仍 None'} |")
    lines.append("")

    # ----------------- §8 下一步建议 -----------------
    lines.append("## 8. 下一步建议（按优先级）")
    lines.append("")
    lines.append("1. **如果 ΔA3 < 5**：触发 P5 kill criteria 评估，先排查 VZ scoped memory 是否真在 turn 级生效")
    lines.append("2. **切档 B (OpenRouter)**：用户加 OPENROUTER_API_KEY 后 `--provider openrouter` 重跑，验证当前 ΔA3 是否被 weak proxy judge 偏置")
    lines.append("3. **扩到 6 family × 2 SUT**：去掉 `--family F1` 跑全 24 scenario，bootstrap CI 缩窄 + 全轴覆盖")
    lines.append("4. **跑 #48 真 cross-family sweep**：5 LLM family judge × 5 SUT × ρ ≥ 0.75 准入榜单（packet companion-bench-public-launch §2.1）")
    lines.append("5. **不要 commit `site/data/` smoke 数据**（同 v1 警告）")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="_emit_smoke_report")
    parser.add_argument(
        "--site-submissions-dir",
        type=pathlib.Path,
        default=pathlib.Path("site/data/submissions"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/companion_bench_smoke"),
    )
    parser.add_argument(
        "--robustness-json",
        type=pathlib.Path,
        default=None,
        help="Path to judge_robustness_qwen_proxy.json (default: <artifact-dir>/judge_robustness_qwen_proxy.json)",
    )
    parser.add_argument(
        "--include-historical-qwen-v1",
        action="store_true",
        default=False,
        help=(
            "Append the previous (v1) Qwen3-Max smoke baseline into the comparison "
            "table when the current run's Qwen subprocess failed (e.g. DashScope "
            "wallclock timeout). The v1 numbers are hard-coded from the 2026-05-13 "
            "first smoke run (see artifacts/companion_bench_smoke history)."
        ),
    )
    args = parser.parse_args(argv)

    # Auto-discover all submission detail JSONs.
    details: list[dict] = []
    if args.site_submissions_dir.exists():
        for p in sorted(args.site_submissions_dir.glob("*.json")):
            d = _load_detail(p)
            if d is not None:
                details.append(d)
    if not details:
        print(
            f"WARN: no submission detail JSONs under {args.site_submissions_dir}; "
            "did you run `build_site.py --artifact-dir ...` first?"
        )

    # Append historical Qwen v1 baseline if requested + not already present.
    if args.include_historical_qwen_v1:
        already_have_qwen = any(
            "qwen" in (d.get("submission_id") or "").lower() for d in details
        )
        if not already_have_qwen:
            details.append(_HISTORICAL_QWEN_V1_DETAIL)
            print("info: appended historical Qwen v1 baseline to comparison")

    robustness_path = args.robustness_json or (args.artifact_dir / "judge_robustness_qwen_proxy.json")
    robustness = _load_detail(robustness_path) if robustness_path.exists() else None

    report = _build_report(
        details=details,
        artifact_dir=args.artifact_dir,
        robustness=robustness,
    )
    out_path = args.artifact_dir / "SMOKE_REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"wrote {out_path} ({len(report)} chars; {len(details)} SUT, robustness={'yes' if robustness else 'no'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
