"""Packet 3 前置 b：分辨力预检（S3-A 等价余量审计）。

在开 RL（S3-E 复刻）之前回答两个"仪器有没有刻度"的问题——C3 教训
（仪器失准产生 null）在本 lane 的防线：

1. **expert margin**：对照式 junction 语料上，expert 动作与非 expert
   动作在冻结小模型 NLL 上是否有可分辨间隙（中位数 > 0 且链自助 CI
   下界 > 0）；
2. **steer headroom**：norm-capped 残差干预能否在该决策面上产生可测
   的 NLL 位移（干预不是 no-op）。

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
    JUNCTION_ACTIONS,
    build_contrastive_corpus,
    collect_junctions,
    corpus_manifest,
    split_corpus,
)

#: Surface text scored against the frozen LM head per junction action.
#: These are the same decision classes the plan's steering layer will
#: choose between (继续调查 / 直接改 / 跑测试 / 提交).
_ACTION_SURFACES: dict[str, str] = {
    "investigate": "investigate the codebase first",
    "edit": "edit the file directly",
    "test": "run the test suite",
    "submit": "submit the change",
}


def _discover_trajectories(patterns: tuple[str, ...]) -> tuple[pathlib.Path, ...]:
    paths: list[pathlib.Path] = []
    for pattern in patterns:
        paths.extend(sorted(_REPO_ROOT.glob(pattern)))
    unique = sorted({p.resolve() for p in paths})
    return tuple(pathlib.Path(p) for p in unique)


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
    parser.add_argument("--min-junctions", type=int, default=24)
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
    corpus = build_contrastive_corpus(records)
    train, evaluation = split_corpus(corpus)
    manifest = corpus_manifest(records, corpus)
    manifest["train_junctions"] = len(train)
    manifest["eval_junctions"] = len(evaluation)

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
        report["verdicts"]["expert_margin"] = None
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
            SteeredActionOption(action_id=action, surface_text=_ACTION_SURFACES[action])
            for action in JUNCTION_ACTIONS
        ),
        max_length=args.max_length,
    )

    # --- Gate 2: expert vs non-expert NLL margin --------------------------
    gaps: list[float] = []
    per_junction: list[dict] = []
    for junction in corpus:
        texts, indices = [], []
        options = (junction.expert_action, *junction.non_expert_actions)
        for action in options:
            texts.append(junction.state_text)
            indices.append(scorer.action_index(action))
        nlls = scorer.baseline_action_nll(
            source_texts=tuple(texts), action_indices=tuple(indices)
        )
        expert_nll = nlls[0]
        non_expert_mean = statistics.fmean(nlls[1:])
        gap = non_expert_mean - expert_nll
        gaps.append(gap)
        per_junction.append(
            {
                "state_key": junction.state_key,
                "expert_action": junction.expert_action,
                "expert_nll": round(expert_nll, 4),
                "non_expert_mean_nll": round(non_expert_mean, 4),
                "gap": round(gap, 4),
            }
        )
    median_gap = statistics.median(gaps)
    ci_lower = _bootstrap_ci_lower(
        gaps, samples=args.bootstrap_samples, seed=args.seed
    )
    expert_margin = median_gap > 0 and ci_lower > 0
    report["expert_margin"] = {
        "junctions_scored": len(gaps),
        "median_gap_nats": round(median_gap, 4),
        "mean_gap_nats": round(statistics.fmean(gaps), 4),
        "bootstrap_ci_lower_5pct": round(ci_lower, 4),
        "positive_fraction": round(
            sum(1 for g in gaps if g > 0) / len(gaps), 4
        ),
    }
    report["per_junction"] = per_junction
    report["verdicts"]["expert_margin"] = expert_margin

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
        corpus_sufficient and expert_margin and steer_headroom
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
