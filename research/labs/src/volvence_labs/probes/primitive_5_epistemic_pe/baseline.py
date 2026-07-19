"""P5 baseline smoke probe (stage 0 stub).

只做一件事：验证 framework 端到端能跑通 —— snapshot 存取、wiring 注入、
AblationCell 分支、ReadoutBundle 落盘。

真正的 epistemic / aleatoric 分离在阶段 1 做。
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping

from ...framework.probe import (
    BaseProbe,
    PrimitiveTag,
    ProbeContext,
    ReadoutBundle,
    RunOutcome,
    register_probe,
)
from ...framework.wiring import AblationCell


def _deterministic_sequence(seed: int, n: int = 64) -> list[float]:
    """生成一段确定性的"预测 vs 真值"对。

    使用 stdlib random 以 seed 做确定性源。每个位置：
        truth  = {0, 1}
        pred   = 真值附近加一点 Gaussian 噪声（clip 到 [1e-4, 1-1e-4]）
    """
    rng = random.Random(seed)
    truths = [rng.randint(0, 1) for _ in range(n)]
    preds = []
    for t in truths:
        p = t + rng.gauss(0.0, 0.15)
        p = max(1e-4, min(1 - 1e-4, p))
        preds.append(p)
    return [float(x) for x in preds] + [float(t) for t in truths]


def _cross_entropy(preds: list[float], truths: list[int]) -> list[float]:
    """逐位置 cross-entropy，模拟"逐 token PE"。"""
    out: list[float] = []
    for p, t in zip(preds, truths):
        if t == 1:
            out.append(-math.log(p))
        else:
            out.append(-math.log(1.0 - p))
    return out


@register_probe
class PEBaselineProbe(BaseProbe):
    id = "pe-baseline-v0"
    hypothesis = (
        "在一段预测序列上逐位置计算 PE；不同 ablation cell 给出不同的 PE 处理策略 "
        "（baseline = raw PE；probe_on = 归一化；probe_off = 常数 0；counterfactual = shuffle）。"
    )
    primitive = PrimitiveTag.P5_EPISTEMIC_PE
    r_ids = ("R-PE",)

    def knobs(self) -> dict[str, list]:
        return {
            "smooth": [False, True],
        }

    def default_inputs(self, seed: int) -> Any:
        flat = _deterministic_sequence(seed=seed, n=64)
        half = len(flat) // 2
        preds = flat[:half]
        truths = [int(x) for x in flat[half:]]
        return {
            "seed": seed,
            "preds": preds,
            "truths": truths,
        }

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        preds: list[float] = list(ctx.inputs["preds"])
        truths: list[int] = list(ctx.inputs["truths"])

        if ctx.cell is AblationCell.BASELINE:
            pe = _cross_entropy(preds, truths)
        elif ctx.cell is AblationCell.PROBE_ON:
            pe = _cross_entropy(preds, truths)
            # Stage 0 stub: 归一化到 [0, 1]（真正的 epistemic split 在阶段 1 实现）
            if pe:
                hi = max(pe) or 1.0
                pe = [x / hi for x in pe]
        elif ctx.cell is AblationCell.PROBE_OFF:
            pe = [0.0] * len(preds)
        elif ctx.cell is AblationCell.COUNTERFACTUAL:
            # Shuffle preds vs truths；PE 应退化到 noise 水平
            rng = random.Random(ctx.seed + 1001)
            shuffled = preds[:]
            rng.shuffle(shuffled)
            pe = _cross_entropy(shuffled, truths)
        else:  # pragma: no cover - enum exhausted
            raise ValueError(f"unknown cell: {ctx.cell!r}")

        mean_pe = sum(pe) / len(pe) if pe else 0.0
        var_pe = (
            sum((x - mean_pe) ** 2 for x in pe) / len(pe) if pe else 0.0
        )
        std_pe = math.sqrt(var_pe)

        readouts = ReadoutBundle(
            metrics={
                "mean_pe": mean_pe,
                "std_pe": std_pe,
                "n": float(len(pe)),
            },
            artifacts={
                "pe_head": pe[:8],  # 只留头几项做视觉检查，避免读盘膨胀
            },
            tags={
                "cell": ctx.cell.value,
                "wiring": ctx.level.value,
                "seed": ctx.seed,
                "inputs_sha": ctx.inputs_sha,
                "smooth": knobs.get("smooth", False),
            },
        )

        return RunOutcome(
            readouts=readouts,
            output={
                "pe_length": len(pe),
                "cell": ctx.cell.value,
            },
        )
