# 设计文档 v2 升级方案

> Status: draft
> Created: 2026-04-09
> Trigger: `docs/next_gen_emogpt.md` rewrite to v2 (prediction error as primitive, NL/ETA tightened)
> Scope: code + docs alignment with design v2 invariants

---

## 0. 背景

`next_gen_emogpt.md` 已从 v1（R1–R15 + 长篇论文附录）重写为 v2（5 pillar thesis + R-PE + 压缩的 NL/ETA bridge + 集成映射图）。本文档评估当前代码库与 v2 设计的差距，并制定分步升级方案。

### v2 核心变化

| 变化 | v1 | v2 |
|------|----|----|
| 学习信号原语 | reward / credit / evaluation 并列 | prediction error / LSS 为唯一原语，credit 和 evaluation 是下游聚合和读出层 |
| NL 呈现 | 8 节论文复述（~240 行） | 5 条设计含义（~40 行），CMS/M3/Hope 定位为设计模式而非必须实现 |
| ETA 呈现 | 7 节论文复述（~150 行） | 术语映射表 + 5 条绑定约束（~40 行），两阶段训练为非可选 |
| 记忆语义 | 隐式混用范式层和工程层 | 显式区分：NL 范式层（任何 neural update）vs 运行时工程层（owner 模块 + snapshot） |
| 新增需求 | — | R-PE：系统必须显式产出 predicted outcome 并在下一轮比较 actual outcome |
| 术语 | subgoal / abstract action 混用 | `abstract action` 为 repo 默认术语，`subgoal` 为论文同义词 |

---

## 1. 现状审计

### 代码 vs v2 设计的 5 大差距

| # | 差距 | 严重度 | 说明 |
|---|------|--------|------|
| G1 | **R-PE 完全缺失** | 致命 | 代码中无 `PredictedOutcome`、`prediction_error`、`LocalSurpriseSignal`。系统不在回复前产出预测，不在下一轮比较实际结果。当前学习信号仍然是 evaluation score 和 credit record，不是 prediction error。 |
| G2 | **Credit 仍是一级信号源** | 高 | `CreditModule`/`CreditLedger` 仍直接驱动 RL reward shaping，而非作为 prediction error 的聚合层。信号流方向与 v2 设计相反。 |
| G3 | **Reflection 提升未接入主循环** | 中 | `reflection_promotion_eligible()` 只在测试中使用，`run_final_wiring_turn` 和 `AgentSessionRunner` 不调用它。 |
| G4 | **Gap assessment 文档失同步** | 中 | `03_gap_assessment` 和 `04_uplift_master_plan` 不反映 R-PE；内部成熟度评分自相矛盾（§1 vs §2 vs `04` §8.2）。 |
| G5 | **LLM generate() 残差干预默认关闭** | 低 | `LLMResponseSynthesizer` 传 `control_scale = switch_gate * 0.15`，当 switch_gate ≈ 0 时干预为零。需验证是否有场景实际触发非零 control_scale。 |

### 已达标的 v2 要求

| 要求 | 状态 | 证据 |
|------|------|------|
| R8 快照/契约 | 达标 | 全链路 snapshot isolation，266+ 测试通过 |
| R15 迁移纪律 | 达标 | checkpoint + rollback + wiring level |
| R7 双轨 | 达标 | Track.WORLD/SELF 贯穿 memory/credit/controller |
| R1 多时间尺度 | 达标 | CMS MLP 三频带 + nested 变体 + 验证 |
| R2 冻结基底 | 达标 | SubstrateModule 只读 + TransformersRuntime eval() + no_grad() |
| R3 时间抽象 | 达标 | Metacontroller + β_t + z_t + action family discovery |
| R4 Internal RL | 达标 | CausalZPolicy.optimize() PPO-clip + GAE + KL 早停 |
| R13 SSL-RL 交替 | 达标 | ETANLJointLoop.run_cycle() 交替 SSL/RL |
| R5 记忆连续谱 | 达标 | MemoryStore + CMS + persistence roundtrip |
| R6 反思 | 部分 | ReflectionEngine 可运行，默认 SHADOW，提升门控仅在库中 |
| R12 评估 | 部分 | 6 族评估 + LongitudinalReport，但定位仍为学习信号源而非 PE 读出 |

---

## 2. 升级方案

### Phase PE: Prediction Error 作为一级公民

**优先级：最高** — 这是 v2 设计的核心支柱，当前完全缺失。

#### PE.1 定义 PredictionError 数据结构

**位置**: 新增 `volvence_zero/prediction/` 模块

```python
@dataclass(frozen=True)
class PredictedOutcome:
    """系统在回复前对下一轮结果的预测。"""
    predicted_task_progress: float      # 预期任务推进
    predicted_relationship_delta: float # 预期关系变化
    predicted_regime_stability: float   # 预期 regime 稳定性
    predicted_action_payoff: float      # 预期当前 abstract action 收益
    confidence: float
    description: str

@dataclass(frozen=True)
class PredictionError:
    """predicted vs actual 的差值，是所有学习的原始信号。"""
    task_error: float
    relationship_error: float
    regime_error: float
    action_error: float
    magnitude: float       # L1 norm
    signed_reward: float   # 正 = 比预期好，负 = 比预期差
    description: str

@dataclass(frozen=True)
class PredictionErrorSnapshot:
    """发布到快照总线的不可变 PE 记录。"""
    predicted: PredictedOutcome
    actual_signals: tuple[tuple[str, float], ...]
    error: PredictionError
    turn_index: int
    description: str
```

#### PE.2 在 AgentSessionRunner 中产出预测

**位置**: `volvence_zero/agent/session.py`

每轮 `run_turn` 在生成回复 **之前**，从当前 cognitive state 产出 `PredictedOutcome`：

- `predicted_task_progress` 从 evaluation snapshot 的 task family score 外推
- `predicted_relationship_delta` 从 dual_track cross_track_tension 外推
- `predicted_regime_stability` 从 regime effectiveness_trend 外推
- `predicted_action_payoff` 从 temporal abstract action 的 long_term_payoff 外推

存入 `self._previous_prediction: PredictedOutcome | None`。

#### PE.3 在下一轮计算 PredictionError

**位置**: `volvence_zero/agent/session.py`

下一轮 `run_turn` 开始时（在 joint loop 之前），如果 `self._previous_prediction` 存在：

1. 从当前 substrate capture + evaluation 中提取 actual outcome signals
2. 计算 `PredictionError = actual - predicted`
3. 将 PE 发布到快照总线（新增 `PredictionErrorModule` 或通过 `CreditModule` 发布）
4. PE 替代/补充 evaluation score 成为 credit record 的源头

#### PE.4 用 PredictionError 驱动学习

**位置**: 多个模块

- **Credit**: `derive_credit_records` 改为从 PE 聚合，而非直接从 evaluation score
- **Memory**: `MemoryStore` 的写入强度权重与 PE magnitude 相关
- **Regime**: `RegimeSelectionWeights` 的 REINFORCE 更新用 PE.signed_reward 而非 delayed_outcomes
- **Temporal**: `CausalZPolicy.optimize()` 的 reward shaping 注入 PE.signed_reward

**退出条件**: PE 在至少 5 个连续 session turn 中可观测，且 credit records 的 source_event 包含 `prediction_error:*` 前缀。

---

### Phase RF: Reflection 提升接入主循环

**优先级：中**

#### RF.1 在 run_final_wiring_turn 中调用 reflection_promotion_eligible

**位置**: `volvence_zero/integration/final_wiring.py`

在 evaluation snapshot enrichment 之后：

1. 调用 `reflection_promotion_eligible(evaluation_snapshot, reflection_engine=...)`
2. 将 `(eligible, reason)` 写入 `FinalIntegrationResult` 的新字段
3. 如果 eligible 且 config 允许自动提升，将 reflection wiring level 从 SHADOW 提升为 ACTIVE

#### RF.2 在 AgentTurnResult 中暴露提升状态

**位置**: `volvence_zero/agent/session.py`

`AgentTurnResult` 新增 `reflection_promotion_eligible: bool` 和 `reflection_promotion_reason: str`。

**退出条件**: `render_turn_result` 显示提升状态；连续 5 轮 accuracy > 0.6 后提升可观测。

---

### Phase DC: 文档一致性修复

**优先级：中**

#### DC.1 更新 `03_gap_assessment_and_uplift_plan.md`

- 新增 R-PE 的成熟度行（当前 0/5）
- 统一 §1 叙述评分和 §2 矩阵评分（以 §2 矩阵为准）
- 标注 Phase 1–4 的完成状态为"验证完成 + 部分实现"而非含糊的"已基本解决"

#### DC.2 更新 `04_uplift_master_plan.md`

- §8.2 基线分数与 §2 矩阵对齐
- §8.3 系统状态标签更新为当前实际（非 P09 完成时的快照）
- 新增 Phase PE 引用

#### DC.3 更新 spec changelogs

每个受影响 spec 已在本次 v2 rewrite 中添加了 changelog 条目。DC.3 只需验证无遗漏。

---

### Phase CI: 残差干预验证

**优先级：低**

#### CI.1 验证 control_scale 非零场景

构造一个测试场景：

1. 高 switch_gate (> 0.5) + 非零 control_code
2. 验证 `generate()` 时 hooks 确实修改 hidden states
3. 对比有/无干预时的生成文本差异

**退出条件**: 有明确证据表明残差干预在至少一种场景下产生可观测的输出差异。

---

## 3. 优先级与依赖

```
Phase PE (prediction error) ← 最高优先，无前置依赖
    PE.1 数据结构
    PE.2 产出预测
    PE.3 计算误差
    PE.4 驱动学习

Phase RF (reflection 提升) ← 依赖 PE 完成后的 evaluation snapshot 含 reflection_accuracy
    RF.1 接入主循环
    RF.2 暴露状态

Phase DC (文档修复) ← 依赖 PE 和 RF 完成后的实际成熟度
    DC.1 gap assessment
    DC.2 uplift master plan
    DC.3 spec changelogs

Phase CI (残差干预) ← 独立，可并行
    CI.1 control_scale 验证
```

---

## 4. 成熟度预估

完成本方案后各 R 预期成熟度：

| 需求 | 当前 | Phase PE 后 | Phase RF 后 | 全部完成后 |
|------|------|------------|------------|-----------|
| R-PE | 0 | 3.5 | 3.5 | 4.0 |
| R1 多时间尺度 | 4.0 | 4.0 | 4.0 | 4.0 |
| R2 稳定基底 | 3.5 | 3.5 | 3.5 | 3.5 |
| R3 时间抽象 | 4.0 | 4.0 | 4.0 | 4.0 |
| R4 内部控制 | 4.0 | 4.5 | 4.5 | 4.5 |
| R5 记忆连续谱 | 4.5 | 4.5 | 4.5 | 4.5 |
| R6 反思整合 | 4.5 | 4.5 | 5.0 | 5.0 |
| R7 双轨分离 | 4.0 | 4.5 | 4.5 | 4.5 |
| R8 快照契约 | 4.5 | 4.5 | 4.5 | 4.5 |
| R9 层级信用 | 4.5 | 5.0 | 5.0 | 5.0 |
| R10 门控自修改 | 4.5 | 4.5 | 4.5 | 4.5 |
| R11 可命名状态 | 4.0 | 4.5 | 4.5 | 4.5 |
| R12 全面评估 | 4.5 | 4.5 | 4.5 | 4.5 |
| R13 SSL-RL 交替 | 4.0 | 4.5 | 4.5 | 4.5 |
| R14 Regime 身份 | 4.0 | 4.5 | 4.5 | 4.5 |
| R15 迁移纪律 | 4.0 | 4.0 | 4.0 | 4.0 |
| **平均** | **4.0** | **4.3** | **4.4** | **4.4** |

---

## 5. 与现有文档的关系

| 文档 | 关系 |
|------|------|
| `next_gen_emogpt.md` v2 | 本方案的设计源头 |
| `03_gap_assessment` | 本方案的 Phase DC 将修复其中的不一致 |
| `04_uplift_master_plan` | 本方案继承其 Phase 1–4 的验证成果，新增 Phase PE/RF/DC/CI |
| `U01`–`U04` 子计划 | 保留，不替换；本方案是 uplift 之后的设计 v2 对齐方案 |
| `docs/specs/*.md` | 已在 v2 rewrite 中同步 changelog；Phase DC 验证无遗漏 |

---

## 6. 参考

- `docs/next_gen_emogpt.md` v2 — Part 1 (5 pillars), Part 2 (R-PE), Part 5 (integrated mapping)
- `docs/papers/2512.24695.txt` — NL §3.1 (memory = neural update, learning = acquiring useful memory)
- `docs/papers/2512.20605.txt` — ETA §3 (metacontroller), §4 (internal RL)
- `docs/implementation/03_gap_assessment_and_uplift_plan.md` — 成熟度基线
- `docs/implementation/04_uplift_master_plan.md` — Phase 1–4 结构
