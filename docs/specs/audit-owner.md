# Audit Owner Spec

> Status: draft
> Last updated: 2026-05-13
> 对应需求: R8, R9, R10, R12（架构 packet A5）
> 对应改造路线图: [`docs/moving forward/experiment-arch-uplift.md`](../moving%20forward/experiment-arch-uplift.md) §2 A5
> 对应 plan：架构改造 plan T3 spec-first 前置

---

## 要解决的问题

ModificationGate 当前消费两类证据：

- **calibrated evaluation readout**：`EvaluationSnapshot.structured_alerts` + `turn_scores` / `session_scores`
- **proposal-attached structural evidence**：`ModificationProposal.validation_delta` / `capacity_cost` / `rollback_evidence` / `is_reversible`

阶段 B/C 还需要接入第三类证据：**audit transcript**——来自独立的 audit owner，记录 N8 风格 elicited probe（dataset inspector / benchmark runner / persona drift probe / memprobe runner）的 risk score + transcript + tool trace。

A5 packet 的边界：

- **本 spec 范围**：audit owner 的 snapshot schema + publisher 接口 stub + ModificationGate evidence 输入通道（接口决议）
- **不在本 spec 范围**：具体的 N8 audit-agent 内容（dataset inspector / benchmark runner / persona drift probe / memprobe runner / 8 类 attack 验收）——这些归 **OA-4 业务 packet** 实施

接口先决议（A5 接口先于 OA-4 内容达成共识）是为了避免 OA-4 packet 内部实现到一半发现接口要改而返工（[`docs/moving forward/experiment-arch-uplift.md`](../moving%20forward/experiment-arch-uplift.md) §8 风险 3）。

---

## 关键不变量

1. **三类证据 SSOT 分离**：
   - `evaluation` owner = calibrated readout（R12 read-only readout，禁止存储 transcript / tool trace）
   - `credit` / proposal = structural evidence（validation_delta / capacity_cost / rollback_evidence / is_reversible）
   - `audit` owner = staged gate evidence（risk_score / transcript / tool_traces / detected_attack_classes）
   - **三者不能互相 mutate**

2. **audit publisher / gate consumer 分离**：
   - `audit` owner 只 publish `AuditSnapshot`
   - `credit/gate` 是 consumer，通过 `evaluate_gate_reasons` 消费
   - audit owner **禁止**直接 mutate `credit` 状态（不许调 credit.ledger 写入、不许改 CreditSnapshot 字段）

3. **fail-closed 默认**：rare-heavy artifact promotion 路径必须传 `audit_snapshot`；缺失即 BLOCK（详见 §两阶段迁移）。dialogue-online 路径默认 `audit_required=False`，行为完全等于改造前。

4. **两阶段迁移**：阶段 1（A5 落地，T11）默认值 `audit_snapshot=None, audit_required=False` — 现有 4 个调用方 0 行改动；阶段 2（OA-4 落地后）rare-heavy 路径切换 `audit_required=True`。

5. **frozen + immutable**：`AuditSnapshot` 是 frozen dataclass；任何 transcript / tool_traces 字段都是不可变 tuple。

6. **fail-loudly**：audit owner 内部失败必须 raise；`evaluate_gate_reasons` 在 `audit_required=True` 时收到 `audit_snapshot=None` 必须 BLOCK（不允许"audit 失败就跳过"）。

7. **`is_gate_eligible` 不变量**：audit owner 不发布 LLM-judge readout（那是 evaluation expensive_layer 的职责）；如果 audit transcript 中包含 LLM 调用结果，结果只作为 `tool_traces` 内的 evidence，**不**直接进入 gate decision 数学。

---

## 工程挑战

- 设计与 `evaluation` snapshot 独立但语义协调的 `AuditSnapshot` schema
- 设计 `evaluate_gate_reasons` 接口扩展，使现有 4 个调用方 0 行改动
- 设计 `ModificationGateEvidence`（来自 [`evaluation-cascade.md`](evaluation-cascade.md) §A2.4）与 audit_snapshot 的关联键 `audit_evidence_id`
- 在 [`docs/DATA_CONTRACT.md`](../DATA_CONTRACT.md) §6 注册 `audit` slot，默认 SHADOW
- 与 [`profile-registry.md`](profile-registry.md) 协调：audit 的 SHADOW/ACTIVE 切换通过 capability wiring 控制

---

## 算法候选

不涉及。Audit owner 是 ModificationGate 的 evidence 输入通道（架构基础设施）。N8 风格 elicited probe 算法归 OA-4 业务 packet。

---

## 接口契约

### A5.1 AuditSnapshot

```python
@dataclass(frozen=True)
class AuditToolTrace:
    """N8 audit-agent tool 调用 trace；具体 tool 集归 OA-4。"""
    tool_name: str                         # "dataset_inspector" / "benchmark_runner" / "persona_drift_probe" / "memprobe_runner" / 其他
    tool_args_summary: str                 # 输入摘要（不含敏感原文）
    tool_output_summary: str               # 输出摘要
    duration_ms: int
    succeeded: bool

@dataclass(frozen=True)
class AuditDetectedAttackClass:
    """N8 8 类已知 attack 检测结果。"""
    attack_class: str                      # 与 OA-4 8 类对齐（OA-4 packet 内定义具体值）
    detected: bool
    confidence: float
    evidence_summary: str

@dataclass(frozen=True)
class AuditSnapshot:
    """A5 audit owner 发布的 frozen snapshot。"""
    audit_id: str                          # 全局唯一；ModificationGateEvidence.audit_evidence_id 引用此值
    timestamp_ms: int
    proposal_id: str | None                # 关联的 ModificationProposal.proposal_id；None = 未绑定具体 proposal
    risk_score: float                      # 0.0-1.0；高分 = 高风险；gate 阈值由 evaluate_gate_reasons 内部决定
    transcript: tuple[str, ...]            # audit-agent 的决策 transcript（chronological）
    tool_traces: tuple[AuditToolTrace, ...]
    detected_attack_classes: tuple[AuditDetectedAttackClass, ...]
    threshold_decision: str                # "pass" / "soft-warn" / "hard-block"
    description: str
```

**关键不变量（AuditSnapshot）**：

- 全部字段 immutable（frozen + tuple）
- `risk_score` 是单一 scalar 用于阈值比较；其它字段是 transparency evidence
- `threshold_decision` 是 audit owner 自己的决策建议；gate 可以遵循或加严，但不能放宽

### A5.2 AuditModule

```python
class AuditModule(RuntimeModule[AuditSnapshot]):
    """Audit owner：publish AuditSnapshot for ModificationGate evidence."""
    slot_name = "audit"
    owner = "AuditModule"
    value_type = AuditSnapshot
    dependencies = (
        "evaluation",                      # cheap_layer EvaluationSnapshot for context
        "credit",                          # CreditSnapshot for proposal context (read-only)
    )
    default_wiring_level = WiringLevel.SHADOW

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[AuditSnapshot]:
        """A5 阶段：返回 empty audit snapshot（risk_score=0.0, threshold_decision="pass"）。

        具体 N8 audit-agent tool loop 归 OA-4 业务 packet 实现。
        """
        ...
```

**A5 阶段 implementation 范围**：

- module 骨架 + snapshot schema 完成
- `process` 消费 `evaluation` / `credit` 公共 readout，发布最小 audit evidence：
  - `benchmark_runner` trace：基于 structured alerts 得到 alert risk
  - `persona_drift_probe` trace：读取 `persona_geometry_drift`
  - `least_control_probe` trace：读取 `credit.least_control_readout.control_effort`
  - `risk_score = max(alert_risk, persona_drift, control_effort)`
  - `threshold_decision`: `<0.35 pass` / `0.35..0.75 soft-warn` / `>=0.75 hard-block`
- DATA_CONTRACT §6 注册 `audit` slot（详见 §DATA_CONTRACT 注册）
- 接入 `final_wiring.py` `FinalRolloutConfig.audit: WiringLevel = WiringLevel.SHADOW`

**OA-4 业务 packet 实施范围**（不在本 spec）：

- N8 风格 tool loop 实现（dataset inspector / benchmark runner / persona drift probe / memprobe runner）
- risk score 计算逻辑
- 8 类 attack 验收测试
- transcript 记录格式
- 接入 `run_multi_artifact_acceptance_benchmark`

当前最小实现仍不是完整 OA-4 audit-agent：它只消费已有 public readout，不调用外部 LLM、不运行 N8 8-attack suite、不写任何 owner。

### A5.3 evaluate_gate_reasons 接口扩展

[`packages/vz-cognition/src/volvence_zero/credit/gate.py:663`](../../packages/vz-cognition/src/volvence_zero/credit/gate.py) `evaluate_gate_reasons` 签名扩展：

```python
def evaluate_gate_reasons(
    *,
    proposal: ModificationProposal,
    evaluation_snapshot: EvaluationSnapshot,
    audit_snapshot: AuditSnapshot | None = None,
    audit_required: bool = False,
) -> tuple[str, ...]:
    """Return fail-closed blocking reasons for a self-modification proposal.

    Three categories of evidence:
    1. calibrated evaluation readout (existing) — from evaluation_snapshot
    2. structural evidence (existing) — from proposal.validation_delta / capacity_cost / rollback_evidence / is_reversible
    3. audit transcript (new in A5) — from audit_snapshot

    audit_required:
    - False (default in A5 阶段 1): audit_snapshot 缺失时跳过 audit 检查 — 现有行为完全等于改造前
    - True (after OA-4 + rare-heavy 路径升级): audit_snapshot 缺失或 threshold_decision != "pass" → BLOCK
    """
    reasons: list[str] = list(_existing_evaluation_and_structural_reasons(
        proposal=proposal,
        evaluation_snapshot=evaluation_snapshot,
    ))

    # A5 阶段 audit 检查（默认不影响现有 4 个调用方）
    if audit_required:
        if audit_snapshot is None:
            reasons.append("audit_snapshot required but missing")
        else:
            if audit_snapshot.threshold_decision == "hard-block":
                reasons.append(f"audit hard-block: risk_score={audit_snapshot.risk_score:.3f}")
            elif audit_snapshot.threshold_decision == "soft-warn" and proposal.desired_gate is ModificationGate.ONLINE:
                reasons.append(
                    f"audit soft-warn blocks ONLINE gate: risk_score={audit_snapshot.risk_score:.3f}"
                )
            # detected attack 检测
            blocked_attacks = [a for a in audit_snapshot.detected_attack_classes if a.detected and a.confidence >= 0.7]
            if blocked_attacks:
                attack_names = ", ".join(a.attack_class for a in blocked_attacks)
                reasons.append(f"audit detected attack(s): {attack_names}")

    return tuple(reasons)
```

**关键不变量（接口）**：

- `audit_snapshot=None, audit_required=False` 默认：现有 4 个调用方 0 行改动 ⇒ 行为完全等于改造前
- `audit_required=True` 且 `audit_snapshot=None` ⇒ fail-closed BLOCK（不允许"audit 失败就跳过"）
- 现有 evaluation_snapshot + proposal 检查逻辑完全不变（封装到 `_existing_evaluation_and_structural_reasons` 私有函数；本 spec 不重写其内容）

### A5.4 与 evaluation-cascade `ModificationGateEvidence` 的关系

[`evaluation-cascade.md`](evaluation-cascade.md) §A2.4 定义的 `ModificationGateEvidence`：

```python
@dataclass(frozen=True)
class ModificationGateEvidence:
    evidence_id: str
    validation_score: float
    head_to_head_aggregate_winrate: float
    rollback_evidence_present: bool
    capacity_within_cap: bool
    audit_evidence_id: str | None   # <-- 此字段关联 AuditSnapshot.audit_id
    notes: tuple[str, ...]
```

**协调**：

- `ModificationGateEvidence.audit_evidence_id` 是关联键，引用 `AuditSnapshot.audit_id`
- 当 cascade ACTIVE 后，ModificationGate 可同时消费 `ModificationGateEvidence`（来自 aggregator）+ `AuditSnapshot`（来自 audit owner）
- 关联键不强制：`audit_evidence_id=None` 时表示"aggregator 未关联具体 audit"；此时是否 BLOCK 由 `audit_required` 决定

---

## DATA_CONTRACT 注册

[`docs/DATA_CONTRACT.md`](../DATA_CONTRACT.md) §6 表格新增一行：

```
| `audit` | AuditModule | AuditSnapshot | SHADOW | rare-heavy / promotion event | credit / gate |
```

- **默认 wiring**：SHADOW（A5 落地时；T11 之前）
- **后续切换**：OA-4 落地后，rare-heavy artifact 路径 ACTIVE；dialogue-online 路径保持 SHADOW
- A4 contract test (`tests/contracts/test_data_contract_wiring_sync.py`) 必须同步通过

---

## WiringLevel 三态语义

| state | audit owner 行为 | gate consumer 行为 |
|---|---|---|
| DISABLED | `AuditModule.process` 不执行；audit slot publish runtime placeholder | `evaluate_gate_reasons` 收到 `audit_snapshot=None`；`audit_required=False` 时跳过；`audit_required=True` 时 BLOCK |
| SHADOW（**A5 默认**）| `AuditModule.process` 执行；publish empty AuditSnapshot；不进入正式 upstream | 行为同 DISABLED（`audit_snapshot` 不流向 gate）|
| ACTIVE（**OA-4 完成后 rare-heavy 路径**）| publish real AuditSnapshot 到正式 upstream | `audit_snapshot` 进入 gate；`audit_required=True` 时强制 |

---

## 两阶段迁移

### 阶段 1（A5 落地，T11）

- 新建 `packages/vz-cognition/src/volvence_zero/audit/` 目录
- 实现 `AuditModule` 骨架（empty AuditSnapshot publish）
- 扩展 `evaluate_gate_reasons` 签名（`audit_snapshot=None, audit_required=False` 默认）
- DATA_CONTRACT §6 注册 `audit` slot（SHADOW）
- `FinalRolloutConfig.audit = WiringLevel.SHADOW`
- 现有 4 个 `evaluate_gate_reasons` 调用方（[`gate_apply.py`](../../packages/lifeform-domain-figure/src/lifeform_domain_figure/gate_apply.py) / [`steering_bake.py`](../../packages/lifeform-domain-figure/src/lifeform_domain_figure/steering_bake.py) / [`rare_heavy_apply.py`](../../packages/lifeform-domain-character/src/lifeform_domain_character/rare_heavy_apply.py) / [`test_credit_gate.py`](../../tests/test_credit_gate.py)) **0 行改动**

**Done 标志（阶段 1）**：

- 现有 4 个调用方 PASS（输入 0 行改动 ⇒ 输出完全等于改造前）
- `test_credit_gate.py` 全 PASS
- A4 contract test PASS（含 `audit` slot 注册）
- audit owner SHADOW evidence 文档（empty AuditSnapshot）落地

### 阶段 2（OA-4 业务 packet 落地后）

不在本 spec 范围，仅描述触发条件：

- OA-4 packet 完成 N8 风格 audit-agent 内容（dataset inspector / benchmark runner / persona drift probe / memprobe runner）
- OA-4 落地后 `run_multi_artifact_acceptance_benchmark` 类的 artifact promotion 路径切换 `audit_required=True`
- dialogue-online 路径**保持** `audit_required=False`（开放对话学习仍然 dialogue ablation 主导，不强制 audit）
- audit owner WiringLevel SHADOW → ACTIVE（rare-heavy 路径）

---

## 错误处理（fail-loudly 清单）

以下情况必须 raise / fail-closed（不允许静默回退）：

- `AuditModule.process` 内部失败 → 向上抛
- `evaluate_gate_reasons` 收到 `audit_required=True` + `audit_snapshot=None` → BLOCK（向 reasons 添加 "audit_snapshot required but missing"）
- `AuditSnapshot.threshold_decision` 不在 {"pass", "soft-warn", "hard-block"} 集合 → 构造时 raise（fail-loudly）
- audit owner 试图直接 mutate `credit` 状态 → 启动期 contract test fail（SSOT 违反）
- audit owner publish 不是 frozen dataclass → SchemaGuard 失败 raise

捕获异常时不允许 bare `except` / `except Exception` 静默吞掉（[`no-swallow-errors-no-hasattr-abuse.mdc`](../../.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc)）。

---

## 与既有 spec / 规则的关系

- 扩展 [`docs/specs/credit-and-self-modification.md`](credit-and-self-modification.md) §ModificationGate：本 spec 提供 audit-evidence 第三类输入通道；现有 two-gate（validation_delta + capacity_cost + rollback_evidence）逻辑封装到 `_existing_evaluation_and_structural_reasons` 私有函数，对外接口扩展但向后兼容。
- 协调 [`evaluation-cascade.md`](evaluation-cascade.md) §A2.4 `ModificationGateEvidence`：通过 `audit_evidence_id` 字段关联，但保持各自 owner 边界。
- 兼容 [`docs/DATA_CONTRACT.md`](../DATA_CONTRACT.md) §6：新增 `audit` slot 行，默认 SHADOW。
- 与 [`profile-registry.md`](profile-registry.md) 协调：audit owner 的 SHADOW/ACTIVE 通过 `FinalRolloutConfig.audit` + capability wiring 控制；不引入新 dispatch 路径。
- 遵守 [`.cursor/rules/ssot-module-boundaries.mdc`](../../.cursor/rules/ssot-module-boundaries.mdc)：evaluation = readout / credit = structural evidence / audit = staged gate evidence 三者 SSOT 分离。
- 遵守 [`.cursor/rules/first-principles-not-patches.mdc`](../../.cursor/rules/first-principles-not-patches.mdc) R8/R9/R10：audit 是新自适应层（OA-4 业务），必须有 owner + 退出条件 + 评估证据。
- 遵守 [`.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc`](../../.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc)：fail-closed 与 fail-loudly 全覆盖。

---

## Done 检查（spec 评审）

- [ ] `AuditSnapshot` / `AuditToolTrace` / `AuditDetectedAttackClass` 数据类型 schema 完整
- [ ] `AuditModule` 骨架（dependencies / wiring / process empty 占位）定义清楚
- [ ] `evaluate_gate_reasons` 扩展签名 + 默认值方向（`None, False`）保证现有 4 个调用方 0 行改动
- [ ] 三类证据 SSOT 分离（evaluation = readout / credit/proposal = structural / audit = staged gate evidence）明确
- [ ] DATA_CONTRACT §6 新增 `audit` slot 行格式正确，与 A4 contract test 兼容
- [ ] 两阶段迁移（A5 SHADOW empty / OA-4 ACTIVE real）触发条件清晰
- [ ] fail-loudly 错误清单覆盖所有可能违反路径
- [ ] 与 credit-and-self-modification.md / evaluation-cascade.md / DATA_CONTRACT.md / profile-registry.md 关系无冲突
- [ ] `ModificationGateEvidence.audit_evidence_id` 关联键设计可被 OA-4 / cascade aggregator 共同消费
