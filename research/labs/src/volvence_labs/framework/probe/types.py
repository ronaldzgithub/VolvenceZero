"""Probe 类型 + 基类。

核心约定（DESIGN.md §2）：
- run_cell 是纯函数：输入 = (knobs, inputs, wiring, cell, seed)，
  输出 = ReadoutBundle。不能写任何外部状态。
- Probe 不持有跨 cell 的可变状态。要累计信息，只能通过 snapshot。
- wiring 层统一注入依赖；probe 不自己决定 SHADOW/ACTIVE。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

from ..wiring import AblationCell, WiringLevel


class PrimitiveTag(str, enum.Enum):
    # 7 primitives
    P1_FROZEN_SUBSTRATE = "p1_frozen_substrate"
    P2_LATENT_CONTROLLER = "p2_latent_controller"
    P3_EMERGENT_SWITCHING = "p3_emergent_switching"
    P4_MULTITIMESCALE_MEMORY = "p4_multitimescale_memory"
    P5_EPISTEMIC_PE = "p5_epistemic_pe"
    P6_BOUNDED_SELF_MOD = "p6_bounded_self_mod"
    P7_READONLY_MONITORING = "p7_readonly_monitoring"
    # 5 frontier
    F1_EPISTEMIC_PE_LLM_SCALE = "f1_epistemic_pe_llm_scale"
    F2_CROSS_MODAL_ZT = "f2_cross_modal_zt"
    F3_MESA_OBJECTIVE_DETECT = "f3_mesa_objective_detect"
    F4_PE_DISTRIBUTIONAL_RLHF = "f4_pe_distributional_rlhf"
    F5_R15_FORMALIZATION = "f5_r15_formalization"
    # meta
    META = "meta"


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadoutBundle:
    """只读指标；R12 评估只读的数据面。

    metrics:   scalar 指标（name -> float）
    artifacts: 任意可 canonical_dumps 的结构化产物（name -> JSON-able）
    tags:      诊断信息（与主结果无关）
    """
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    tags: Mapping[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return {
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "artifacts": dict(self.artifacts),
            "tags": dict(self.tags),
        }


@dataclass(frozen=True)
class GateReport:
    passed: bool
    reason: str
    stats: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunOutcome:
    """单 unit run 的完整结果（在 scheduler 里被打包成 RunRecord + 产物写入 CAS）。"""
    readouts: ReadoutBundle
    output: Any = None                      # probe 产出的"事件" / 对象（可 canonical_dumps）
    tags: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Probe context: 注入给 probe 的只读环境
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeContext:
    """注入给 probe 的运行时句柄。

    - level: 当前 WiringLevel（probe 可以读但不能改）
    - cell:  当前 AblationCell
    - seed:  当前 seed
    - inputs: 输入快照已解包的对象（probe 拿到的是普通 dict/list）
    - inputs_sha: 输入快照的 sha（probe 可选择记录到 tags）
    """
    level: WiringLevel
    cell: AblationCell
    seed: int
    inputs: Any
    inputs_sha: str


# ---------------------------------------------------------------------------
# Probe base
# ---------------------------------------------------------------------------

class Probe(Protocol):
    id: str
    hypothesis: str
    primitive: PrimitiveTag
    r_ids: tuple[str, ...]

    def knobs(self) -> dict[str, list]:
        """AB 维度。阶段 0 不做 grid 展开，直接返回单值默认组合即可。"""
        ...

    def default_inputs(self, seed: int) -> Any:
        """返回一个可 canonical_dumps 的 inputs 对象。"""
        ...

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        ...

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        ...


class BaseProbe:
    """可被子类继承的默认实现。阶段 0 提供最小默认。"""

    id: str = ""
    hypothesis: str = ""
    primitive: PrimitiveTag = PrimitiveTag.META
    r_ids: tuple[str, ...] = ()

    def knobs(self) -> dict[str, list]:
        return {}

    def default_inputs(self, seed: int) -> Any:
        return {"seed": seed}

    def run_cell(self, ctx: ProbeContext, knobs: Mapping[str, Any]) -> RunOutcome:
        raise NotImplementedError

    def gate(self, outcomes: list[RunOutcome]) -> GateReport:
        # 阶段 0：默认 gate 是 placeholder。
        # 只在所有 outcomes 都返回了至少一个指标时 pass；否则 fail。
        have_metrics = all(o.readouts.metrics for o in outcomes)
        return GateReport(
            passed=bool(outcomes) and have_metrics,
            reason="placeholder gate (stage 0) — requires every outcome to produce metrics",
            stats={"n_outcomes": len(outcomes)},
        )
