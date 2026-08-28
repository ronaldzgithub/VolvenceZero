"""Live side-by-side demo: three arms race over one or many task chains.

The audience picks a house convention, then watches three endpoints work the
SAME task sequences with the SAME frozen hand, differing only in what context
they carry:

    stateless  — nothing
    steelman   — every previous transcript, verbatim ("no context engineering")
    brain      — an owner-published digest

Nobody ever tells any arm that the convention exists. It is enforced only by
acceptance tests injected at oracle time. All three fail it on first contact;
whether any arm stops failing is the whole question.

With ``--chains N`` the demo runs N chains concurrently (bounded by
``--max-concurrency``) and aggregates the learning curve into fractions per
occurrence — the same shape as the frozen evidence figure. Each (chain, arm)
unit checkpoints its rows, so an interrupted long run resumes with ``--resume``
instead of re-spending API tokens.

Two hands:

    --hand api        the real demo: a frozen OpenAI-compatible coder
    --hand scripted   rehearsal only. A context-conditioned scripted hand with
                      a KNOWN injected effect. It verifies plumbing, display,
                      concurrency and resume for free and offline; it is NOT
                      evidence and the report marks itself as such.

    python scripts/run_investor_side_by_side_demo.py --hand scripted --chains 2 --episodes 6
    python scripts/run_investor_side_by_side_demo.py --hand api --chains 8 --episodes 10
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import pathlib
import shutil
import stat
import sys
import tempfile
import time

from lifeform_domain_coding.lab.episode import EpisodeBudget
from lifeform_domain_coding.lab.generation import (
    ALL_CONVENTION_IDS,
    CONVENTION_DESCRIPTIONS,
    EnvSpec,
)
from lifeform_domain_coding.lab.hands import (
    APIHandConfig,
    MemoryAwareScriptedHand,
    OpenAICompatHand,
)
from lifeform_domain_coding.lab.tasks import generate_task_chain
from lifeform_evolution.coding_lab_arms import (
    ARM_BRAIN,
    ARM_STATELESS,
    ARM_STEELMAN,
    ArmChainConfig,
    ArmEpisodeRow,
    run_chain_arm,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Categories where the convention is enforceable at all: only tasks that
#: introduce a new public symbol get the hidden acceptance test injected.
CONVENTION_APPLICABLE = ("add_helper", "extend_report", "config_feature")

#: The only applicable category that recurs inside one chain (its variant pool
#: holds four helpers), so the only place where cross-episode memory can pay
#: off — and the reason more chains beat longer chains for this demo.
RECURRING_CATEGORY = "add_helper"

ARM_LABELS = {
    ARM_BRAIN: "结构化经历（我们）",
    ARM_STEELMAN: "全历史堆上下文",
    ARM_STATELESS: "无记忆",
}
ARM_ORDER = (ARM_BRAIN, ARM_STEELMAN, ARM_STATELESS)

#: Below this many chains, pass rates and token totals are noise (the formal
#: run needed 8 chains x 10 episodes for those), so the board keeps them out
#: of the audience-facing summary.
STABLE_METRICS_MIN_CHAINS = 3

DEFAULT_API = APIHandConfig(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-coder-next",
    api_key_env="DASHSCOPE_API_KEY",
)

#: Substring the rehearsal hand looks for in its context to decide to comply.
#: Each is a fragment of the corresponding hidden test's assertion head, which is
#: what lands in a remembering arm's context after a failure. Rehearsal-only: it
#: injects a KNOWN effect so the plumbing and display can be checked for free.
_REHEARSAL_NEEDLE_BY_CONVENTION = {
    "convention_export_all": "__all__",
    "convention_annotated_signature": "annotation",
    "convention_docstring_contract": "Contract:",
    "convention_symbol_owner": "_SYMBOL_OWNERS",
}


def _rehearsal_needles(convention_ids: tuple[str, ...]) -> dict[str, str]:
    """Needle per applicable category for the first active convention."""

    needle = _REHEARSAL_NEEDLE_BY_CONVENTION[convention_ids[0]]
    return {category: needle for category in CONVENTION_APPLICABLE}


class LiveBoard:
    """Append-only streaming display.

    Deliberately append-only rather than cursor-redrawn: in a live meeting a
    scrollback that survives terminal quirks and can be screenshotted beats a
    prettier board that may garble.
    """

    def __init__(
        self,
        *,
        convention_id: str,
        chains: int,
        episodes: int,
        hand_label: str,
        max_concurrency: int,
    ) -> None:
        self._rows: list[ArmEpisodeRow] = []
        self._chains = chains
        self._episodes = episodes
        self._units_total = chains * len(ARM_ORDER)
        self._units_done = 0
        self._started = time.monotonic()
        width = min(shutil.get_terminal_size((104, 30)).columns, 104)
        self._rule = "─" * width
        print(self._rule)
        print(f"隐藏约定：{convention_id}")
        print(f"          {CONVENTION_DESCRIPTIONS[convention_id]}")
        print("          任务描述从不提它，仓库里零信号，只由评测时注入的隐藏测试检查。")
        print(
            f"手：{hand_label}　·　{chains} 条链 × 每臂 {episodes} 回合 × 3 臂"
            f"　·　并发上限 {max_concurrency}"
        )
        print(self._rule)
        print(
            f"{'时间':>6}  {'链':>3}  {'臂':<22} {'回合':>4}  "
            f"{'类别':<16} {'结果':<6} {'约定':<8} 上下文词元"
        )
        print(self._rule)

    def _elapsed(self) -> float:
        return time.monotonic() - self._started

    def on_episode(self, row: ArmEpisodeRow) -> None:
        self._rows.append(row)
        applicable = row.category in CONVENTION_APPLICABLE
        violated = bool(row.invariant_violations)
        mark = "违反" if (applicable and violated) else ("遵守" if applicable else "—")
        print(
            f"{self._elapsed():6.0f}s  {row.chain_index:>3}  {ARM_LABELS[row.arm]:<22} "
            f"{row.episode_index + 1:>4}  {row.category:<16} "
            f"{'通过' if row.passed else '未过':<6} {mark:<8} {row.context_tokens_approx:>6}"
        )
        sys.stdout.flush()

    def absorb_resumed(self, rows: tuple[ArmEpisodeRow, ...]) -> None:
        """Fold checkpointed rows into the tallies without replaying lines."""

        self._rows.extend(rows)
        first = rows[0]
        print(
            f"{self._elapsed():6.0f}s  {first.chain_index:>3}  {ARM_LABELS[first.arm]:<22} "
            f"　从断点恢复 {len(rows)} 个回合（不重复计费）"
        )
        sys.stdout.flush()

    def unit_finished(self, chain: int, arm: str, *, resumed: bool) -> None:
        self._units_done += 1
        state = "恢复" if resumed else "完成"
        print(
            f"{self._elapsed():6.0f}s  ── 单元{state}：链 {chain} / {ARM_LABELS[arm]}"
            f"　（{self._units_done}/{self._units_total}）"
        )
        sys.stdout.flush()

    def _occurrence_curve(self, arm: str) -> list[dict]:
        """Aggregate the recurring-category curve into fractions across chains."""

        per_occurrence: dict[int, list[bool]] = {}
        chain_indices = sorted({row.chain_index for row in self._rows if row.arm == arm})
        for chain in chain_indices:
            recurring = sorted(
                (
                    row
                    for row in self._rows
                    if row.arm == arm
                    and row.chain_index == chain
                    and row.category == RECURRING_CATEGORY
                ),
                key=lambda row: row.episode_index,
            )
            for occurrence, row in enumerate(recurring):
                per_occurrence.setdefault(occurrence + 1, []).append(
                    bool(row.invariant_violations)
                )
        return [
            {
                "occurrence": occurrence,
                "violations": sum(flags),
                "total": len(flags),
                "violation_rate": sum(flags) / len(flags),
            }
            for occurrence, flags in sorted(per_occurrence.items())
        ]

    def summary(self) -> dict:
        out: dict = {"arms": {}, "recurring_curve": {}, "chains": self._chains}
        for arm in ARM_ORDER:
            rows = [row for row in self._rows if row.arm == arm]
            applicable = [row for row in rows if row.category in CONVENTION_APPLICABLE]
            violations = sum(1 for row in applicable if row.invariant_violations)
            out["arms"][arm] = {
                "label": ARM_LABELS[arm],
                "episodes": len(rows),
                "passed": sum(1 for row in rows if row.passed),
                "pass_rate": (
                    sum(1 for row in rows if row.passed) / len(rows) if rows else None
                ),
                "convention_applicable_episodes": len(applicable),
                "convention_violations": violations,
                "convention_violation_rate": (
                    violations / len(applicable) if applicable else None
                ),
                "mean_context_tokens": (
                    sum(row.context_tokens_approx for row in rows) / len(rows)
                    if rows
                    else None
                ),
                "total_prompt_tokens": sum(row.prompt_tokens for row in rows),
                "total_wall_seconds": sum(row.wall_seconds for row in rows),
            }
            out["recurring_curve"][arm] = self._occurrence_curve(arm)
        return out

    def print_summary(self, summary: dict) -> None:
        print(self._rule)
        print("最终结果")
        print(self._rule)
        print(
            f"「{RECURRING_CATEGORY}」违反率，按链内第几次出现聚合"
            f"（{self._chains} 条链；约定唯一能靠记忆学会的地方）"
        )
        for arm in ARM_ORDER:
            cells = "  ".join(
                f"#{point['occurrence']}: {point['violations']}/{point['total']}"
                f"={point['violation_rate']:.0%}"
                for point in summary["recurring_curve"][arm]
            )
            print(f"  {ARM_LABELS[arm]:<22} {cells}")
        print(self._rule)
        if self._chains >= STABLE_METRICS_MIN_CHAINS:
            print(
                f"{'臂':<22} {'通过率':>8}  {'约定违反率':>12}  {'平均上下文':>10}  {'总 prompt':>11}"
            )
            for arm in ARM_ORDER:
                stats = summary["arms"][arm]
                rate = stats["convention_violation_rate"]
                print(
                    f"{stats['label']:<22} {stats['pass_rate']:>7.1%}  "
                    f"{(f'{rate:.1%}' if rate is not None else '—'):>12}  "
                    f"{stats['mean_context_tokens']:>10.0f}  "
                    f"{stats['total_prompt_tokens']:>11,}"
                )
            print(
                "注意：总 prompt token 是单次运行读数，单会话方差真实存在"
                "（正式 8 链数据：中位数 0.78，3/8 条链更贵）——不作省钱承诺。"
            )
        else:
            print(
                f"（链数 {self._chains} < {STABLE_METRICS_MIN_CHAINS}：通过率与总 token 在该规模下是噪声，"
                "不打印，完整数值仍在 report.json 里）"
            )
        print(self._rule)


#: Windows resolves ``time.monotonic()`` through GetTickCount64 (~15.6 ms), and
#: the brain observer requires the pre-oracle bet to be recorded STRICTLY before
#: the outcome submission. A scripted hand finishes a whole step inside one tick,
#: so the two stamps collide and the ordering guard trips. The real demo uses the
#: API hand, whose steps take seconds, and is unaffected. This delay exists only
#: so the free offline rehearsal can exercise the brain arm; it is not part of
#: any evidence path.
_REHEARSAL_TICK_CLEARANCE_SECONDS = 0.03


class _RehearsalPacedHand(MemoryAwareScriptedHand):
    """Rehearsal hand paced above the platform clock resolution."""

    def hand_id(self) -> str:
        return f"rehearsal-paced-{super().hand_id()}"

    async def decide(self, context):
        await asyncio.sleep(_REHEARSAL_TICK_CLEARANCE_SECONDS)
        return await super().decide(context)


def _hand_factory(kind: str, convention_ids: tuple[str, ...]):
    def build(chain, chain_index: int):
        if kind == "api":
            return OpenAICompatHand(DEFAULT_API)
        return _RehearsalPacedHand(
            needles_by_category=_rehearsal_needles(convention_ids),
            tasks_by_id={task.task_id: task for task in chain},
            episode_index_by_task_id={
                task.task_id: index for index, task in enumerate(chain)
            },
            hand_seed=20260827 + chain_index,
            invariant_sabotage_rate=0.0,
            acceptance_sabotage_rate=0.0,
        )

    return build


def _force_rmtree(path: pathlib.Path) -> None:
    """rmtree that survives Windows read-only files.

    Every unit directory contains an inner git repository whose object files
    are written read-only; plain ``shutil.rmtree`` dies on them with
    ``PermissionError`` on Windows, which would strand incomplete units.
    """

    for root, _dirs, files in os.walk(path):
        for name in files:
            os.chmod(os.path.join(root, name), stat.S_IWRITE)
    shutil.rmtree(path)


def _unit_dir(out_dir: pathlib.Path, chain: int, arm: str) -> pathlib.Path:
    return out_dir / f"chain-{chain:02d}" / arm


def _checkpoint_path(unit_dir: pathlib.Path) -> pathlib.Path:
    return unit_dir / "rows.json"


def _load_checkpoint(unit_dir: pathlib.Path) -> tuple[ArmEpisodeRow, ...] | None:
    path = _checkpoint_path(unit_dir)
    if not path.is_file():
        return None
    rows = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        item["invariant_violations"] = tuple(item["invariant_violations"])
        rows.append(ArmEpisodeRow(**item))
    return tuple(rows)


def _write_checkpoint(unit_dir: pathlib.Path, rows: tuple[ArmEpisodeRow, ...]) -> None:
    _checkpoint_path(unit_dir).write_text(
        json.dumps([row.__dict__ for row in rows], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def _render_demo_figure(report: dict, out_dir: pathlib.Path) -> pathlib.Path:
    """Render the aggregated curve in the evidence figures' visual language.

    Written into the DEMO output directory (never into docs/business/BP/figures)
    and titled/labelled as a live demo so it cannot be mistaken for evidence.
    """

    from matplotlib import pyplot as plt

    from lab_figures import style

    style.apply()
    arm_colours = {
        ARM_BRAIN: style.SAGE,
        ARM_STEELMAN: style.AMBER,
        ARM_STATELESS: style.GREY,
    }

    figure, axes = style.new_figure((12.2, 7.4))
    figure.subplots_adjust(left=0.10, right=0.96, top=0.775, bottom=0.16)
    curves = report["summary"]["recurring_curve"]
    for arm in ARM_ORDER:
        points = curves[arm]
        if not points:
            continue
        x_values = [point["occurrence"] for point in points]
        y_values = [point["violation_rate"] * 100 for point in points]
        is_ours = arm == ARM_BRAIN
        axes.plot(
            x_values,
            y_values,
            marker="o",
            markersize=11 if is_ours else 8,
            linewidth=3.4 if is_ours else 2.0,
            color=arm_colours[arm],
            zorder=5 if is_ours else 3,
            solid_capstyle="round",
            label=ARM_LABELS[arm],
        )
        for x_value, y_value, point in zip(x_values, y_values, points, strict=True):
            axes.annotate(
                f"{point['violations']}/{point['total']}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(0, 13 if is_ours else -20),
                ha="center",
                fontsize=10,
                color=arm_colours[arm],
            )

    max_occurrence = max(
        (point["occurrence"] for arm in ARM_ORDER for point in curves[arm]),
        default=1,
    )
    axes.set_xlim(0.7, max_occurrence + 0.3)
    axes.set_ylim(-8, 112)
    axes.set_xticks(list(range(1, max_occurrence + 1)))
    axes.set_xticklabels([f"第 {index} 次" for index in range(1, max_occurrence + 1)])
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes.set_xlabel("同一条任务链里，这类任务第几次出现", fontsize=12.5, labelpad=10)
    axes.set_ylabel("违反那条约定的比例", fontsize=12.5, labelpad=10)
    axes.legend(loc="center left", bbox_to_anchor=(0.02, 0.28), fontsize=11.5)
    style.strip_frame(axes)

    style.title_block(
        figure,
        "现场并排运行 · live demo（非证据）",
        f"{report['hand']}　·　{report['chains']} 条链 × 每臂 {report['episodes']} 回合"
        f"　·　隐藏约定 {report['convention_ids'][0]}\n"
        "单次现场运行，未预注册；正式证据见 coding_lab_packet2_formal_v2_qwen3codernext_20260813。",
    )
    style.footer(
        figure,
        f"evidence_tier: {report['evidence_tier']} · 由 run_investor_side_by_side_demo.py 生成于演示输出目录",
    )

    target = out_dir / "live-demo-curve.png"
    figure.savefig(target, format="png", dpi=200)
    figure.savefig(out_dir / "live-demo-curve.svg", format="svg")
    plt.close(figure)
    return target


async def run(args: argparse.Namespace) -> dict:
    # Must be absolute: ChainWorkspace hands the worktree path straight to git,
    # which resolves it against the inner repo directory, so a relative path
    # silently creates the worktree nested inside the repo instead.
    out_dir = pathlib.Path(args.out).resolve()
    if out_dir.exists() and not args.resume:
        _force_rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    convention_ids = tuple(args.convention)
    chain_indices = tuple(range(args.chains))
    chain_meta: dict[int, dict] = {}
    for chain in chain_indices:
        spec = EnvSpec(env_seed=args.env_seed, convention_ids=convention_ids)
        tasks = generate_task_chain(spec, chain_seed=chain, length=args.episodes)
        chain_meta[chain] = {
            "categories": [task.category for task in tasks],
            "applicable": sum(
                1 for task in tasks if task.category in CONVENTION_APPLICABLE
            ),
            "recurring": sum(
                1 for task in tasks if task.category == RECURRING_CATEGORY
            ),
        }
    total_recurring = sum(meta["recurring"] for meta in chain_meta.values())
    if total_recurring == 0:
        raise SystemExit(
            f"{args.chains} chain(s) x {args.episodes} episodes contain no recurring "
            "convention-applicable task; raise --episodes or --chains"
        )

    hand_label = (
        f"{DEFAULT_API.model}（真实 API，temperature 0）"
        if args.hand == "api"
        else "彩排用脚本手（注入已知效应，非证据）"
    )
    board = LiveBoard(
        convention_id=convention_ids[0],
        chains=args.chains,
        episodes=args.episodes,
        hand_label=hand_label,
        max_concurrency=args.max_concurrency,
    )
    print(
        f"可重复出现的那类任务共 {total_recurring} 个"
        f"（各链：{'、'.join(str(chain_meta[c]['recurring']) for c in chain_indices)}）\n"
    )

    budget = EpisodeBudget(max_steps=args.max_steps, max_wall_seconds=args.max_wall_seconds)
    factory = _hand_factory(args.hand, convention_ids)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    resumed_units: list[str] = []

    async def one(chain: int, arm: str) -> None:
        unit_dir = _unit_dir(out_dir, chain, arm)
        checkpoint = _load_checkpoint(unit_dir) if args.resume else None
        if checkpoint is not None and len(checkpoint) == args.episodes:
            board.absorb_resumed(checkpoint)
            board.unit_finished(chain, arm, resumed=True)
            resumed_units.append(f"chain-{chain:02d}/{arm}")
            return
        if unit_dir.exists():
            # Incomplete unit: the evolving-repo workspace cannot resume
            # mid-chain, so the whole unit reruns from a clean directory.
            _force_rmtree(unit_dir)
        async with semaphore:
            config = ArmChainConfig(
                env_seed=args.env_seed,
                chain_index=chain,
                episodes=args.episodes,
                brain_digest_char_budget=args.brain_digest_char_budget,
                budget=budget,
                convention_ids=convention_ids,
            )
            rows = await run_chain_arm(
                arm=arm,
                config=config,
                arm_root=unit_dir,
                hand_factory=factory,
                on_episode=board.on_episode,
            )
        _write_checkpoint(unit_dir, rows)
        board.unit_finished(chain, arm, resumed=False)

    started = time.monotonic()
    # return_exceptions so one failing unit does not cancel its siblings
    # mid-flight: finished units keep their checkpoints and a --resume rerun
    # only re-spends the failed one. The first failure is re-raised after all
    # units settle.
    outcomes = await asyncio.gather(
        *(one(chain, arm) for chain in chain_indices for arm in ARM_ORDER),
        return_exceptions=True,
    )
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    if failures:
        print(f"（{len(failures)} 个单元失败；已完成单元的检查点已保留，可用 --resume 续跑）")
        raise failures[0]
    wall = time.monotonic() - started

    summary = board.summary()
    board.print_summary(summary)

    report = {
        "schema_version": "coding-lab-investor-side-by-side-demo.v2",
        "purpose": "live audience-facing demonstration; not a preregistered evidence run",
        "evidence_tier": (
            "live_demo_api_hand"
            if args.hand == "api"
            else "rehearsal_scripted_hand_injected_effect_not_evidence"
        ),
        "claim_boundary": (
            "A single un-preregistered run staged for an audience, whatever its chain "
            "count; it must never be cited as a gate result. The frozen evidence for "
            "this mechanism is coding_lab_packet2_formal_v2_qwen3codernext_20260813."
        ),
        "ran_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hand": hand_label,
        "hand_kind": args.hand,
        "convention_ids": list(convention_ids),
        "convention_descriptions": {
            key: CONVENTION_DESCRIPTIONS[key] for key in convention_ids
        },
        "env_seed": args.env_seed,
        "chains": args.chains,
        "episodes": args.episodes,
        "max_concurrency": args.max_concurrency,
        "resumed_units": resumed_units,
        "chain_meta": {str(chain): meta for chain, meta in chain_meta.items()},
        "total_recurring_episodes": total_recurring,
        "wall_seconds": wall,
        "summary": summary,
        "episodes_detail": [
            {**row.__dict__, "invariant_violations": list(row.invariant_violations)}
            for row in sorted(
                board._rows,
                key=lambda row: (row.chain_index, row.arm, row.episode_index),
            )
        ],
    }
    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    figure_path = _render_demo_figure(report, out_dir)
    try:
        shown = report_path.relative_to(REPO_ROOT)
    except ValueError:
        # Running with --out outside the repo is the supported default: the
        # workspace's git worktrees must not be nested under this repository.
        shown = report_path
    print(f"报告：{shown}")
    print(f"曲线图：{figure_path}")
    print(f"总墙上时间：{wall:.0f} 秒")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hand",
        choices=("api", "scripted"),
        default="scripted",
        help="api = real demo; scripted = free offline rehearsal (default)",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=1,
        help="run chains 0..N-1 concurrently; fractions need >= 3, the formal shape 8",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=6,
        help="ceiling on concurrent (chain, arm) units; raise carefully, the API may 429",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep the output dir and skip (chain, arm) units with a complete checkpoint",
    )
    parser.add_argument(
        "--convention",
        nargs="+",
        default=["convention_export_all"],
        choices=list(ALL_CONVENTION_IDS),
        help="hidden house convention(s) the audience picked",
    )
    parser.add_argument("--env-seed", type=int, default=20260827)
    parser.add_argument("--brain-digest-char-budget", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-wall-seconds", type=float, default=900.0)
    parser.add_argument(
        "--out",
        default=str(pathlib.Path(tempfile.gettempdir()) / "volvence-side-by-side-demo"),
        help=(
            "output root; must live OUTSIDE this repository, because each arm "
            "creates git worktrees that must not nest inside it"
        ),
    )
    args = parser.parse_args(argv)
    if args.chains < 1:
        parser.error("--chains must be >= 1")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
