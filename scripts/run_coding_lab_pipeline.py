"""coding-lab 全阶段流水线（超短机制冒烟）。

一条命令按顺序驱动全部七个阶段，每阶段调用其**正式 runner**（不复制
逻辑），用机制级小参数换分钟级时长：

  P0  脚本手标定（环境确定性 + oracle 牙齿）
  P1  SHADOW 观察者判词（PE 分辨力 / 恢复 / 外部结局通道）
  P2  三臂注入 smoke（已知效应 + 双门）
  P2.5 黑盒择时 gate（Learnable 不依赖白盒）
  P3a 余量审计（margin；语料不足则如实 FAIL 并继续）
  P3b S3-E 复刻 smoke（残差捕获 → reader/executor → RL 六门通路）
  P4  ModificationGate 机制探针（probe 注册表，不碰正式指针）

判定口径：**机制完整性** = 每阶段跑完并产出 artifact；判词数值如实
记录（玩具规模下部分判词 FAIL 是预期，不改阈值不粉饰）。正式判词
仍走各 runner 的正式路径（API 手标定 → freeze-prereg → formal →
5 seed）。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PY = str(_REPO_ROOT / ".venv" / "bin" / "python")


def _stage(
    *,
    name: str,
    command: list[str],
    timeout_seconds: float,
    env: dict[str, str] | None = None,
    log_path: pathlib.Path,
) -> dict:
    started = time.monotonic()
    merged_env = {**os.environ, **(env or {})}
    print(f"[pipeline] {name}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            completed = subprocess.run(
                command,
                cwd=_REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=merged_env,
                timeout=timeout_seconds,
                check=False,
            )
            exit_code: int | None = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            exit_code = None
            timed_out = True
    seconds = time.monotonic() - started
    print(
        f"[pipeline] {name}: exit={exit_code} ({seconds:.0f}s)"
        + (" TIMEOUT" if timed_out else ""),
        flush=True,
    )
    return {
        "name": name,
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "seconds": round(seconds, 1),
        "log": str(log_path),
    }


def _read_json(path: pathlib.Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-id", default=f"pipe_{int(time.time())}")
    parser.add_argument("--device", default="mps", help="P3b S3-E 捕获设备。")
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--episodes-per-chain", type=int, default=4)
    parser.add_argument("--skip", action="append", default=[], help="跳过的阶段名。")
    args = parser.parse_args()

    pid = args.pipeline_id
    art = _REPO_ROOT / "artifacts" / "coding_lab"
    out_dir = art / f"coding_lab_pipeline_{pid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stages: list[dict] = []
    verdicts: dict[str, object] = {}

    def run_stage(name: str, command: list[str], *, timeout: float, env=None) -> dict:
        if name in args.skip:
            record = {"name": name, "skipped": True}
            stages.append(record)
            return record
        record = _stage(
            name=name,
            command=command,
            timeout_seconds=timeout,
            env=env,
            log_path=out_dir / f"{name}.log",
        )
        stages.append(record)
        return record

    # ---- P0: scripted calibration --------------------------------------
    p0_run = f"coding_lab_pipeline_{pid}_p0"
    run_stage(
        "p0-calibration",
        [
            _PY, "scripts/run_coding_lab_calibration.py",
            "--hand", "scripted",
            "--run-id", p0_run,
            "--chains", str(args.chains),
            "--episodes-per-chain", str(args.episodes_per_chain),
            "--heldout-variants", "1",
        ],
        timeout=1800,
    )
    p0_report = _read_json(art / p0_run / "report.json")
    if p0_report is not None:
        verdicts["p0"] = p0_report.get("verdicts", p0_report.get("checks"))

    # ---- P1: SHADOW observer -------------------------------------------
    p1_run = f"coding_lab_pipeline_{pid}_p1"
    run_stage(
        "p1-observer",
        [
            _PY, "scripts/run_coding_lab_observer.py",
            "--calibration-run-dir", str(art / p0_run),
            "--run-id", p1_run,
            "--permutations", "2000",
        ],
        timeout=1800,
    )
    p1_report = _read_json(art / p1_run / "report.json")
    if p1_report is not None:
        verdicts["p1"] = p1_report.get("verdicts")

    # ---- P2: three-arm injection smoke ----------------------------------
    p2_run = f"coding_lab_pipeline_{pid}_p2"
    run_stage(
        "p2-arms-smoke",
        [
            _PY, "scripts/run_coding_lab_packet2.py", "smoke",
            "--run-id", p2_run,
            "--chains", str(args.chains),
            "--episodes-per-chain", str(max(args.episodes_per_chain, 6)),
        ],
        timeout=3600,
    )
    p2_report = _read_json(art / p2_run / "report.json")
    if p2_report is not None:
        verdicts["p2"] = p2_report.get("verdicts")

    # ---- P2.5: blackbox timing gate (all available logs) ----------------
    p25_run = f"coding_lab_pipeline_{pid}_p25"
    run_stage(
        "p25-blackbox-gate",
        [
            _PY, "scripts/run_coding_lab_packet25_blackbox_gate.py",
            "--run-id", p25_run,
            "--updates", "150",
            "--restarts", "2",
        ],
        timeout=1800,
    )
    p25_report = _read_json(art / p25_run / "report.json")
    if p25_report is not None:
        verdicts["p25"] = {
            "overall_pass": p25_report["overall_pass"],
            **p25_report["verdict"],
        }

    # ---- P3a: margin audit (honest FAIL on thin corpus is expected) -----
    p3a_run = f"coding_lab_pipeline_{pid}_p3margin"
    run_stage(
        "p3a-margin",
        [
            _PY, "scripts/run_coding_lab_packet3_margin.py",
            "--run-id", p3a_run,
            "--device", args.device,
            "--min-junctions", "4",
            "--bootstrap-samples", "500",
            "--headroom-directions", "4",
        ],
        timeout=3600,
        env={"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
    )
    p3a_report = _read_json(art / p3a_run / "report.json")
    if p3a_report is not None:
        verdicts["p3_margin"] = p3a_report.get("verdicts")

    # ---- P3b: S3-E replication smoke ------------------------------------
    p3b_run = f"coding_lab_pipeline_{pid}_p3s3e"
    run_stage(
        "p3b-s3e-smoke",
        [
            _PY, "scripts/run_coding_lab_packet3_s3e.py", "run",
            "--run-id", p3b_run,
            "--device", args.device,
            "--smoke", "--skip-margin-check",
            "--smoke-train-cases", "6",
            "--smoke-heldout-cases", "3",
            "--smoke-episodes", "160",
        ],
        timeout=5400,
        env={"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
    )
    p3b_report = _read_json(art / p3b_run / "report.json")
    if p3b_report is not None:
        verdicts["p3_s3e"] = {
            "admitted": p3b_report["admission"]["admitted"],
            "failed_conditions": p3b_report["admission"]["failed_conditions"],
            "train_rows": p3b_report["train_row_count"],
            "heldout_rows": p3b_report["heldout_row_count"],
        }

    # ---- P4: ModificationGate mechanism probe ---------------------------
    p4_run = f"coding_lab_pipeline_{pid}_p4"
    run_stage(
        "p4-gate-probe",
        [
            _PY, "scripts/run_coding_lab_packet4_gate.py",
            "--candidate-run-id", p3b_run,
            "--out-run-id", p4_run,
            "--mechanism-probe",
        ],
        timeout=600,
    )
    p4_report = _read_json(art / p4_run / "review.json")
    if p4_report is not None:
        verdicts["p4"] = {
            "decision": p4_report["review"]["decision"],
            "blocking_reasons": p4_report["review"]["blocking_reasons"],
            "mechanism_probe": p4_report["mechanism_probe"],
        }

    # ---- Summary ---------------------------------------------------------
    artifact_stages = {
        "p0-calibration": art / p0_run / "report.json",
        "p1-observer": art / p1_run / "report.json",
        "p2-arms-smoke": art / p2_run / "report.json",
        "p25-blackbox-gate": art / p25_run / "report.json",
        "p3a-margin": art / p3a_run / "report.json",
        "p3b-s3e-smoke": art / p3b_run / "report.json",
        "p4-gate-probe": art / p4_run / "review.json",
    }
    mechanism_complete = all(
        name in args.skip or path.is_file()
        for name, path in artifact_stages.items()
    )
    summary = {
        "pipeline_id": pid,
        "mechanism_complete": mechanism_complete,
        "stages": stages,
        "verdicts": verdicts,
        "note": (
            "机制冒烟口径：每阶段跑完并产出 artifact 即通路成立；"
            "玩具规模下的判词 FAIL 如实记录，不作为管线失败。"
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"mechanism_complete": mechanism_complete}, ensure_ascii=False))
    for name, path in artifact_stages.items():
        print(f"  {name}: {'OK' if path.is_file() else 'MISSING'}")
    print(f"summary: {out_dir / 'summary.json'}")
    return 0 if mechanism_complete else 1


if __name__ == "__main__":
    sys.exit(main())
