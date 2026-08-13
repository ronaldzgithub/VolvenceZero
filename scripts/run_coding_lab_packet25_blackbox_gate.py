"""Packet 2.5：层 A 黑盒择时 gate 判词 runner。

从既有 coding-lab 轨迹构建离线 bandit 表并训练黑盒 gate，输出
held-out 判词。特征全黑盒（协议状态），信用只来自 episode 终局
oracle。语料太薄时 fail loudly，不出读数。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _pkg in ("lifeform-domain-coding", "lifeform-evolution"):
    _src = _REPO_ROOT / "packages" / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from lifeform_domain_coding.lab.junctions import collect_junctions  # noqa: E402
from lifeform_evolution.coding_lab_blackbox_gate import (  # noqa: E402
    fit_and_judge_blackbox_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-glob",
        action="append",
        default=None,
        help="Repo-relative glob(s); default scans all coding_lab runs.",
    )
    parser.add_argument("--run-id", default="coding_lab_packet25_blackbox_gate")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--updates", type=int, default=300)
    args = parser.parse_args()

    patterns = tuple(
        args.trajectory_glob
        or (
            "artifacts/coding_lab/*/chains/chain-*/trajectories/episode-*.jsonl",
            "artifacts/coding_lab/*/brain/chain-*/trajectories/episode-*.jsonl",
            "artifacts/coding_lab/*/steelman/chain-*/trajectories/episode-*.jsonl",
            "artifacts/coding_lab/*/stateless/chain-*/trajectories/episode-*.jsonl",
        )
    )
    paths: list[pathlib.Path] = []
    for pattern in patterns:
        paths.extend(sorted(_REPO_ROOT.glob(pattern)))
    trajectories = tuple(pathlib.Path(p) for p in sorted({p.resolve() for p in paths}))
    records = collect_junctions(trajectories)

    verdict = fit_and_judge_blackbox_gate(
        records,
        seed=args.seed,
        restarts=args.restarts,
        updates=args.updates,
    )
    report = {
        "packet": "coding-lab-packet-2.5-blackbox-gate",
        "run_id": args.run_id,
        "source_trajectories": len(trajectories),
        "junction_records": len(records),
        "verdict": dataclasses.asdict(verdict),
        "overall_pass": verdict.beats_uniform,
    }
    out_dir = _REPO_ROOT / "artifacts" / "coding_lab" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["verdict"], ensure_ascii=False))
    print(f"overall_pass={report['overall_pass']}")
    print(f"report: {out_dir / 'report.json'}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
