"""Packet 3 前置 a：Qwen2.5-Coder-1.5B-Instruct 基底预检。

在开工 S3-E 复刻（编程域 reader/executor 重 fit）之前，用一次性诊断回答
四个必须先答的问题：

1. ``_resolve_transformer_blocks`` 能否解析该模型（构造 runtime 即验证）；
2. fp32 加载 + 残差捕获 + steered-action scorer 在本机内存内可跑；
3. capture 的残差序列宽度 == hidden_size 且全有限；
4. 磁盘余量足够容纳权重与后续 artifact。

任一门不过 → 判词写明回退到 Qwen2.5-0.5B-Instruct 既有几何
（S3-E 已验证的 24 块 / 896 宽 / layer 20 注入）。

只读诊断：不训练、不写权重、不改任何 runtime 状态。报告落
``artifacts/coding_lab/<run_id>/``。
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import time
import traceback

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("vz-contracts", "vz-substrate"):
    _src = _REPO_ROOT / "packages" / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from volvence_zero.substrate.residual_backend import (  # noqa: E402
    TransformersOpenWeightResidualRuntime,
)
from volvence_zero.substrate.steered_action_scoring import (  # noqa: E402
    SteeredActionOption,
)

GIB = 1024**3

# 编程域 junction 探针：与 Packet 3 语料同风格（观察→下一步动作），
# 只用于验证捕获与打分路径，不构成训练数据。
_PROBE_TEXTS = (
    "Task: fix rounding bug in pricing.line_total. Read pricing.py first; "
    "the repo has a hidden invariant that all money math uses round_half_up.",
    "Task: extend report.summarize with a per-category count. Acceptance "
    "test exists; previous episode regressed store.load by editing blindly.",
)

_ACTION_OPTIONS = (
    SteeredActionOption(
        action_id="investigate", surface_text="investigate the codebase first"
    ),
    SteeredActionOption(
        action_id="edit", surface_text="edit the file directly"
    ),
    SteeredActionOption(
        action_id="test", surface_text="run the test suite"
    ),
    SteeredActionOption(
        action_id="ask", surface_text="ask the user a question"
    ),
)


def _peak_rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    return raw if sys.platform == "darwin" else raw * 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", default="Qwen/Qwen2.5-Coder-1.5B-Instruct"
    )
    parser.add_argument(
        "--activation-width",
        type=int,
        default=1536,
        help="Expected hidden width; must equal runtime.hidden_size.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument(
        "--max-peak-rss-gib",
        type=float,
        default=20.0,
        help="fp32 verdict bound; leaves headroom on a 24 GiB host.",
    )
    parser.add_argument(
        "--run-id", default="coding_lab_packet3_substrate_check"
    )
    parser.add_argument(
        "--fallback-model-id", default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = _REPO_ROOT / "artifacts" / "coding_lab" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "run_id": args.run_id,
        "model_id": args.model_id,
        "device": args.device,
        "model_dtype": "float32",
        "fallback_model_id": args.fallback_model_id,
        "checks": {},
        "verdicts": {},
        "error": None,
    }

    free_bytes = shutil.disk_usage(_REPO_ROOT).free
    report["checks"]["disk_free_gib"] = round(free_bytes / GIB, 2)
    report["verdicts"]["disk_ok"] = free_bytes >= args.min_free_gib * GIB

    try:
        load_start = time.monotonic()
        runtime = TransformersOpenWeightResidualRuntime(
            model_id=args.model_id,
            device=args.device,
            max_length=args.max_length,
            fail_on_truncation=True,
            activation_width=args.activation_width,
            hook_layer_selection="middle",
            allow_live_substrate_mutation=False,
            allow_offline_substrate_training=False,
            model_dtype="float32",
        )
        load_seconds = time.monotonic() - load_start
        report["checks"]["load_seconds"] = round(load_seconds, 1)
        report["checks"]["hidden_size"] = runtime.hidden_size
        report["checks"]["parameter_count"] = runtime.model_parameter_count
        # 构造成功即 _resolve_transformer_blocks / _resolve_hidden_size /
        # middle-layer 归一化全部通过。
        report["verdicts"]["blocks_resolved"] = True
        report["verdicts"]["width_matches"] = (
            runtime.hidden_size == args.activation_width
        )

        capture_start = time.monotonic()
        capture = runtime.capture(source_text=_PROBE_TEXTS[0])
        capture_seconds = time.monotonic() - capture_start
        steps = capture.residual_sequence
        widths = {
            len(activation.activation)
            for step in steps
            for activation in step.residual_activations
        }
        layer_indices = sorted(
            {
                activation.layer_index
                for step in steps
                for activation in step.residual_activations
            }
        )
        all_finite = all(
            value == value and abs(value) != float("inf")
            for step in steps
            for activation in step.residual_activations
            for value in activation.activation
        )
        report["checks"]["capture_seconds"] = round(capture_seconds, 2)
        report["checks"]["capture_steps"] = len(steps)
        report["checks"]["capture_widths"] = sorted(widths)
        report["checks"]["capture_layer_indices"] = layer_indices
        report["verdicts"]["capture_ok"] = (
            len(steps) > 0
            and widths == {runtime.hidden_size}
            and all_finite
        )

        scorer_start = time.monotonic()
        scorer = runtime.build_steered_action_scorer(
            action_options=_ACTION_OPTIONS,
            max_length=args.max_length,
        )
        nlls = scorer.baseline_action_nll(
            source_texts=_PROBE_TEXTS,
            action_indices=(0, 1),
        )
        scorer_seconds = time.monotonic() - scorer_start
        nll_finite = all(
            value == value and abs(value) != float("inf") for value in nlls
        )
        report["checks"]["scorer_seconds"] = round(scorer_seconds, 2)
        report["checks"]["scorer_injection_layer"] = (
            scorer.injection_layer_index
        )
        report["checks"]["baseline_action_nll"] = [
            round(value, 4) for value in nlls
        ]
        report["verdicts"]["scorer_ok"] = nll_finite and len(nlls) == 2
    except Exception:  # 进程级故障边界：记录完整错误后以失败退出。
        report["error"] = traceback.format_exc()

    peak_rss = _peak_rss_bytes()
    report["checks"]["peak_rss_gib"] = round(peak_rss / GIB, 2)
    report["verdicts"]["fp32_fits"] = (
        peak_rss <= args.max_peak_rss_gib * GIB
    )

    required = (
        "disk_ok",
        "blocks_resolved",
        "width_matches",
        "capture_ok",
        "scorer_ok",
        "fp32_fits",
    )
    report["verdicts"]["overall_pass"] = report["error"] is None and all(
        report["verdicts"].get(name) is True for name in required
    )
    report["recommendation"] = (
        f"proceed with {args.model_id} fp32 geometry "
        f"(hidden={report['checks'].get('hidden_size')}, "
        f"injection_layer={report['checks'].get('scorer_injection_layer')})"
        if report["verdicts"]["overall_pass"]
        else (
            f"fall back to {args.fallback_model_id} existing S3-E geometry "
            "(24 blocks / hidden 896 / injection layer 20)"
        )
    )

    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Packet 3 前置 a：基底预检",
        "",
        f"- model: `{args.model_id}` device={args.device} dtype=float32",
        f"- overall: {'PASS' if report['verdicts']['overall_pass'] else 'FAIL'}",
        f"- recommendation: {report['recommendation']}",
        "",
        "| verdict | value |",
        "|---|---|",
    ]
    for name in required:
        lines.append(f"| {name} | {report['verdicts'].get(name)} |")
    lines += [
        "",
        "| check | value |",
        "|---|---|",
    ]
    for key, value in report["checks"].items():
        lines.append(f"| {key} | {value} |")
    if report["error"]:
        lines += ["", "```", report["error"].strip(), "```"]
    (out_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(json.dumps(report["verdicts"], indent=2))
    print(f"report: {report_path}")
    return 0 if report["verdicts"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
