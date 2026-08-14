"""Packet 3 前置 b：分辨力预检（S3-A 等价余量审计）。

在开 RL（S3-E 复刻）之前回答"仪器有没有刻度"——C3 教训（仪器失准产生
null）在本 lane 的防线：

1. **expert resolution**：冻结小模型在对照式 junction 上区分动作的尺度
   是否≥ formal 判则要求的改进量（prereg `gain_vs_noop` = 0.3 nats）。
   读数取 domain-conditional PMI（`nll(a|state) - nll(a|neutral)`），
   抵掉各选项自身的表面似然；同时检查同输入重复打分逐位一致。
2. **steer headroom**：norm-capped 残差干预能否在该决策面上产生可测
   的 NLL 位移（干预不是 no-op）。

**基底对齐度只报不设门**（2026-08-13 修正）。原 `expert_margin` 门要求
带符号间隙中位数 > 0，即要求冻结基底**已经**偏好信用专家动作；这是规格
错误：steering 实验的前提正是基底存在偏差，基底若已对则无可扳之处，六
条冻结判则也失去对象。故预检只守分辨力 / 可重复性 / 干预余量，对齐度
（`base_alignment`）作为决策面性质如实报告。Packet 3 的可证伪主张仍全部
落在 prereg 的六条冻结判则上，未因本次修正放宽任何一条。

语料不足或任一门不过 → 如实 FAIL，不开 RL（退出条件：封存
"该决策面无择时余量"）。只读诊断，不训练、不改权重。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

from random import Random

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _pkg in ("vz-contracts", "vz-substrate", "lifeform-domain-coding"):
    _src = _REPO_ROOT / "packages" / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from lifeform_domain_coding.lab.junctions import (  # noqa: E402
    ACTION_SURFACES,
    DEFAULT_MIN_ACTION_SUPPORT,
    DEFAULT_MIN_PASS_RATE_MARGIN,
    JUNCTION_ACTIONS,
    NEUTRAL_STATE_TEXT,
    build_action_outcome_table,
    build_contrastive_corpus,
    collect_junctions,
    corpus_manifest,
    split_corpus,
)


def _discover_trajectories(patterns: tuple[str, ...]) -> tuple[pathlib.Path, ...]:
    paths: list[pathlib.Path] = []
    for pattern in patterns:
        paths.extend(sorted(_REPO_ROOT.glob(pattern)))
    unique = sorted({p.resolve() for p in paths})
    return tuple(pathlib.Path(p) for p in unique)


def _state_key_accounting(
    records,
    *,
    min_action_support: int,
    min_pass_rate_margin: float,
) -> dict:
    """Why each protocol state did or did not become a junction.

    Keeps the corpus size auditable: a state excluded because both moves
    lead to the same outcome ("no leverage here") is a finding, not a
    data shortfall, and the two must not be conflated when judging
    whether the corpus is large enough.
    """

    labelled = 0
    no_leverage = 0
    under_supported = 0
    for stats in build_action_outcome_table(records).values():
        supported = [s for s in stats if s.trials >= min_action_support]
        if len(supported) < 2:
            under_supported += 1
            continue
        expert = max(supported, key=lambda s: (s.pass_rate, s.trials, s.action))
        trailing = [
            s
            for s in supported
            if s.action != expert.action
            and expert.pass_rate - s.pass_rate >= min_pass_rate_margin
        ]
        if trailing:
            labelled += 1
        else:
            no_leverage += 1
    return {
        "labelled": labelled,
        "excluded_no_leverage": no_leverage,
        "excluded_under_supported": under_supported,
    }


def _bootstrap_ci_lower(
    values: list[float], *, samples: int, seed: int, quantile: float = 0.05
) -> float:
    rng = Random(seed)
    means: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(draw))
    means.sort()
    return means[int(quantile * len(means))]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-glob",
        action="append",
        default=None,
        help="Repo-relative glob(s) for trajectory JSONL files; "
        "repeatable. Default scans all coding_lab run artifacts.",
    )
    parser.add_argument("--run-id", default="coding_lab_packet3_margin")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--activation-width", type=int, default=1536)
    parser.add_argument("--injection-layer-index", type=int, default=13)
    parser.add_argument("--max-length", type=int, default=256)
    # 22 credit-labelled states out of 77 protocol states in the
    # 2026-08-13 corpus; 22 of the exclusions are positive findings ("both
    # moves reach the same outcome here"), 33 lack per-cell support. See
    # the report's ``state_key_accounting``.
    parser.add_argument("--min-junctions", type=int, default=20)
    parser.add_argument(
        "--min-separation",
        type=float,
        default=0.3,
        help="Median |PMI gap| (nats) the instrument must resolve, "
        "anchored to the prereg gain_vs_noop floor.",
    )
    parser.add_argument(
        "--min-action-support",
        type=int,
        default=DEFAULT_MIN_ACTION_SUPPORT,
        help="Episodes required behind a (state, move) cell before its "
        "conditional pass rate may label anything.",
    )
    parser.add_argument(
        "--min-pass-rate-margin",
        type=float,
        default=DEFAULT_MIN_PASS_RATE_MARGIN,
        help="Conditional pass-rate gap an expert must hold over a "
        "non-expert move at the same state.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--headroom-directions", type=int, default=8)
    parser.add_argument(
        "--headroom-min-shift",
        type=float,
        default=0.01,
        help="Mean |ΔNLL| (nats) a capped intervention must produce.",
    )
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = _REPO_ROOT / "artifacts" / "coding_lab" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = tuple(
        args.trajectory_glob
        or (
            "artifacts/coding_lab/*/chains/chain-*/trajectories/episode-*.jsonl",
            "artifacts/coding_lab/*/brain/chain-*/trajectories/episode-*.jsonl",
            "artifacts/coding_lab/*/steelman/chain-*/trajectories/episode-*.jsonl",
            "artifacts/coding_lab/*/stateless/chain-*/trajectories/episode-*.jsonl",
        )
    )
    trajectories = _discover_trajectories(patterns)
    records = collect_junctions(trajectories)
    corpus = build_contrastive_corpus(
        records,
        min_action_support=args.min_action_support,
        min_pass_rate_margin=args.min_pass_rate_margin,
    )
    train, evaluation = split_corpus(corpus)
    manifest = corpus_manifest(records, corpus)
    manifest["train_junctions"] = len(train)
    manifest["eval_junctions"] = len(evaluation)
    manifest["label_policy"] = {
        "expert_source": "conditional-pass-rate-credit",
        "min_action_support": args.min_action_support,
        "min_pass_rate_margin": args.min_pass_rate_margin,
    }
    manifest["state_key_accounting"] = _state_key_accounting(
        records,
        min_action_support=args.min_action_support,
        min_pass_rate_margin=args.min_pass_rate_margin,
    )

    report: dict = {
        "packet": "coding-lab-packet-3-margin",
        "run_id": args.run_id,
        "model_id": args.model_id,
        "corpus": manifest,
        "verdicts": {},
    }

    corpus_sufficient = len(corpus) >= args.min_junctions
    report["verdicts"]["corpus_sufficient"] = corpus_sufficient
    if not corpus_sufficient:
        # 语料门不过时不启动模型：没有刻度的仪器不该出读数。
        report["verdicts"]["expert_resolution"] = None
        report["verdicts"]["steer_headroom"] = None
        report["verdicts"]["overall_pass"] = False
        report["note"] = (
            f"contrastive corpus has {len(corpus)} junctions < "
            f"{args.min_junctions}; collect more API-hand trajectories "
            "(scripted branches share identical move sequences and "
            "cannot contrast)."
        )
        _write(out_dir, report)
        return 1

    import torch  # noqa: PLC0415 — heavyweight import gated on corpus

    from volvence_zero.substrate.residual_backend import (  # noqa: PLC0415
        TransformersOpenWeightResidualRuntime,
    )
    from volvence_zero.substrate.steered_action_scoring import (  # noqa: PLC0415
        SteeredActionOption,
    )

    runtime = TransformersOpenWeightResidualRuntime(
        model_id=args.model_id,
        device=args.device,
        max_length=args.max_length,
        fail_on_truncation=True,
        activation_width=args.activation_width,
        # Single-layer capture at the prereg injection layer (ETA
        # precedent; "middle" selection returns a 3-layer window).
        layer_indices=(args.injection_layer_index,),
        allow_live_substrate_mutation=False,
        allow_offline_substrate_training=False,
        model_dtype="float32",
    )
    scorer = runtime.build_steered_action_scorer(
        action_options=tuple(
            SteeredActionOption(action_id=action, surface_text=ACTION_SURFACES[action])
            for action in JUNCTION_ACTIONS
        ),
        max_length=args.max_length,
    )

    # --- Gate 2: expert vs non-expert NLL margin --------------------------
    # Raw NLL comparisons across different option surfaces measure the
    # options' own likelihood as much as the state's evidence. The gate
    # therefore reads the domain-conditional PMI form,
    # ``nll(a|state) - nll(a|neutral)``, which cancels each surface's
    # intrinsic prior; the raw gap is kept alongside as the diagnostic
    # that exposed the confound (2026-08-13: every positive raw gap was
    # an "investigate" label, every negative one was not).
    neutral_nlls = {
        action: scorer.baseline_action_nll(
            source_texts=(NEUTRAL_STATE_TEXT,),
            action_indices=(scorer.action_index(action),),
        )[0]
        for action in JUNCTION_ACTIONS
    }
    gaps: list[float] = []
    raw_gaps: list[float] = []
    per_junction: list[dict] = []
    for junction in corpus:
        options = (junction.expert_action, *junction.non_expert_actions)
        nlls = scorer.baseline_action_nll(
            source_texts=tuple(junction.state_text for _ in options),
            action_indices=tuple(scorer.action_index(a) for a in options),
        )
        calibrated = [
            nll - neutral_nlls[action]
            for nll, action in zip(nlls, options, strict=True)
        ]
        raw_gap = statistics.fmean(nlls[1:]) - nlls[0]
        gap = statistics.fmean(calibrated[1:]) - calibrated[0]
        gaps.append(gap)
        raw_gaps.append(raw_gap)
        per_junction.append(
            {
                "state_key": junction.state_key,
                "expert_action": junction.expert_action,
                "expert_nll": round(nlls[0], 4),
                "non_expert_mean_nll": round(statistics.fmean(nlls[1:]), 4),
                "raw_gap": round(raw_gap, 4),
                "calibrated_gap": round(gap, 4),
            }
        )
    # Instrument repeatability: identical input must give a bit-identical
    # reading on a frozen float32 substrate.
    probe_junction = corpus[0]
    repeat = [
        scorer.baseline_action_nll(
            source_texts=(probe_junction.state_text,),
            action_indices=(scorer.action_index(probe_junction.expert_action),),
        )[0]
        for _ in range(2)
    ]
    repeatable = repeat[0] == repeat[1]

    magnitudes = [abs(g) for g in gaps]
    median_abs_gap = statistics.median(magnitudes)
    abs_ci_lower = _bootstrap_ci_lower(
        magnitudes, samples=args.bootstrap_samples, seed=args.seed
    )
    resolves = abs_ci_lower >= args.min_separation
    expert_resolution = resolves and repeatable
    report["expert_resolution"] = {
        "scoring": "domain-conditional-pmi",
        "junctions_scored": len(gaps),
        "median_abs_gap_nats": round(median_abs_gap, 4),
        "abs_gap_bootstrap_ci_lower_5pct": round(abs_ci_lower, 4),
        "min_separation_required": args.min_separation,
        "repeat_reading_identical": repeatable,
    }
    # Reported, deliberately NOT gated: see module docstring. A base that
    # already preferred the credit-backed move would leave nothing for
    # steering to earn.
    report["base_alignment"] = {
        "median_signed_gap_nats": round(statistics.median(gaps), 4),
        "mean_signed_gap_nats": round(statistics.fmean(gaps), 4),
        "positive_fraction": round(
            sum(1 for g in gaps if g > 0) / len(gaps), 4
        ),
        "raw_median_gap_nats": round(statistics.median(raw_gaps), 4),
        "raw_positive_fraction": round(
            sum(1 for g in raw_gaps if g > 0) / len(raw_gaps), 4
        ),
        "neutral_action_nll": {
            action: round(value, 4) for action, value in neutral_nlls.items()
        },
        "gated": False,
        "note": "anti-alignment is steering headroom, not a blocker; the "
        "falsifiable claim stays in the prereg's six frozen rules",
    }
    report["per_junction"] = per_junction
    report["verdicts"]["expert_resolution"] = expert_resolution

    # --- Gate 3: steer vs noop headroom -----------------------------------
    torch.manual_seed(args.seed)
    cap = scorer.control_norm_cap
    probe = corpus[: min(len(corpus), 12)]
    shifts: list[float] = []
    for junction in probe:
        index = scorer.action_index(junction.expert_action)
        baseline = scorer.baseline_action_nll(
            source_texts=(junction.state_text,), action_indices=(index,)
        )[0]
        for _ in range(args.headroom_directions):
            direction = torch.randn(1, scorer.hidden_size, dtype=torch.float32)
            direction = direction / direction.norm() * cap
            steered = scorer.controlled_action_nll(
                source_texts=(junction.state_text,),
                control_deltas=direction,
                action_indices=(index,),
            )[0]
            shifts.append(abs(steered - baseline))
    mean_shift = statistics.fmean(shifts)
    steer_headroom = mean_shift >= args.headroom_min_shift
    report["steer_headroom"] = {
        "probe_junctions": len(probe),
        "directions_per_junction": args.headroom_directions,
        "control_norm_cap": round(float(cap), 4),
        "mean_abs_nll_shift": round(mean_shift, 5),
        "max_abs_nll_shift": round(max(shifts), 5),
        "min_shift_required": args.headroom_min_shift,
    }
    report["verdicts"]["steer_headroom"] = steer_headroom

    report["verdicts"]["overall_pass"] = (
        corpus_sufficient and expert_resolution and steer_headroom
    )
    _write(out_dir, report)
    return 0 if report["verdicts"]["overall_pass"] else 1


def _write(out_dir: pathlib.Path, report: dict) -> None:
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Packet 3 前置 b：分辨力预检（余量审计）",
        "",
        f"- overall: {'PASS' if report['verdicts'].get('overall_pass') else 'FAIL'}",
        "",
        "| verdict | value |",
        "|---|---|",
    ]
    for name, value in report["verdicts"].items():
        lines.append(f"| {name} | {value} |")
    lines += ["", "```json", json.dumps(report.get("corpus", {}), indent=2), "```"]
    if "expert_margin" in report:
        lines += ["", "```json", json.dumps(report["expert_margin"], indent=2), "```"]
    if "steer_headroom" in report:
        lines += ["", "```json", json.dumps(report["steer_headroom"], indent=2), "```"]
    if "note" in report:
        lines += ["", f"> {report['note']}"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["verdicts"], ensure_ascii=False))
    print(f"report: {out_dir / 'report.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
