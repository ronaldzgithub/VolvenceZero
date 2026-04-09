# U03 — Phase 3: 涌现替代 Heuristic

> Status: draft
> Last updated: 2026-04-09
> Phase owner: `temporal` + `evaluation` + `regime`
> 差距族: C（涌现 vs. Heuristic）
> 影响需求: R3, R10, R14
> 前置条件: Phase 1 完成（RL 优化器可用）+ Phase 2 部分完成（MLP CMS 可用）
> 预估周期: 3–4 周

## 1. Phase 目标

让切换门稀疏性、action family 竞争和 regime 选择从规则驱动走向数据驱动涌现。

`next_gen_emogpt.md` 的核心信念之一是"涌现优于预设"——有意义的行为模式应该从数据中自发浮现，而非由预定义规则硬编码。当前系统中三个关键决策节点仍依赖 heuristic：

1. **切换门稀疏性**：靠 `switch_bias` 阈值和手动规则，非变分瓶颈 `α` 驱动的自发准二值行为
2. **Action family 竞争**：靠 `support/stability/drift` 等 owner-side heuristic score，非长期结果主导
3. **Regime 选择**：靠 scoring 规则，非延迟结果 RL 训练产生

本 Phase 的目标是系统性地将这三处 heuristic 替换为数据驱动的涌现机制。

## 2. 设计依据

### 2.1 ETA 变分瓶颈（Appendix B.3, Eq.3）

```
L(φ) = Σ_t [-ln p_{θ,φ}(a_t | o_{1:t}, z_{1:t}) + α · D_KL(N(μ_t,Σ_t) || N(0,I))]
```

- `α` 控制变分瓶颈：调节从非因果编码器到控制器的信息流
- 该瓶颈驱动模型走向稀疏的、与子目标对齐的切换模式
- 无条件先验 `N(0,I)` 促进组合性表示的发展
- **涌现行为**：训练后 `β_t` 自发学会准二值行为——无需显式正则化

### 2.2 ETA rate-distortion 分析（Appendix B.4）

冻结基础模型时，rate-distortion 曲线上出现水平间隙——子目标对齐的切换模式恰好位于 distortion 急剧改善的区域。

### 2.3 `02_eta_nl_next_stage.md` Phase A

已识别的问题：family bank 的形成与选择仍大量依赖 heuristic；active abstract action 容易塌到少数高支持 family。

## 3. 步骤分解

### U03.1 — 变分瓶颈驱动切换门涌现

**owner**: `temporal`
**位置**: `volvence_zero/temporal/metacontroller_components.py`, `volvence_zero/temporal/ssl.py`

**内容**：

1. SSL 训练目标升级为 Eq.3 完整变分下界：
   - 动作预测损失：`-ln p(a_t | o_{1:t}, z_{1:t})`
   - KL 正则化：`α · D_KL(N(μ_t,Σ_t) || N(0,I))`
   - `MetacontrollerSSLTrainer` 中新增 `alpha` 参数和 KL loss 计算

2. 非因果内部序列嵌入器激活：
   - `noncausal_embedder.py` 已存在——在 SSL 训练中正式接入
   - 嵌入器看到整个序列 `e_{1:T}`，提供编码器的全局上下文
   - 嵌入器输出 `s(e_{1:T})` 作为编码器的额外输入

3. 切换门稀疏性从阈值驱动变为涌现驱动：
   - 移除 `SwitchUnit` 中硬编码的 `switch_bias` 阈值（保留为 fallback 可配参数）
   - `β_t` 的二值化由变分瓶颈 `α` 的强度自然产生
   - 新增 `SwitchGateStats` 统计量：β 值直方图、切换频率、平均持续步数
   - 在 `TemporalAbstractionSnapshot` 中发布 `SwitchGateStats`

4. `α` 调参策略：
   - 从大 `α`（=1.0，强信息限制）开始
   - 逐步降低到 `α` ∈ [0.01, 0.1]
   - 在每个 `α` 值下观测切换模式
   - 选择 β 直方图呈现准二值分布的 `α` 范围

**约束**：

- `alpha=0.0` 时退化为无 KL 正则的原始 SSL，保证向后兼容
- 变分瓶颈训练不修改基础模型参数（冻结 substrate）
- `switch_bias` 作为 fallback 保留，可通过配置在涌现模式和 heuristic 模式间切换

**验收**：
- 在 `alpha > 0` 的 SSL 训练后，`β_t` 直方图展现准二值分布（大部分值 < 0.1 或 > 0.9）
- 切换时刻的分布呈稀疏模式（非均匀随机）
- 与 `alpha=0` 对比，稀疏切换模式更显著
- `SwitchGateStats` 在 `TemporalAbstractionSnapshot` 中可读

### U03.2 — Action family 竞争由长期结果驱动

**owner**: `temporal`
**位置**: `volvence_zero/temporal/interface.py`, `volvence_zero/temporal/metacontroller_components.py`

**内容**：

1. Family-level competition memory：
   - 每个 `DiscoveredActionFamily` 新增 `long_term_payoff: float`、`delayed_credit_sum: float`
   - 从 `CreditSnapshot` 的 `session_level_credits` 中提取 abstract-action 级信用
   - Family 的 survival score 从 `support + stability` 扩展为 `support + stability + long_term_payoff`

2. Family competition 机制：
   - 引入 `FamilyCompetitionState`：按 `long_term_payoff` 排序的 family 排行
   - Active family selection 不再只依赖局部相似度——加入 payoff-weighted selection
   - Selection 权重：`w_family = similarity^α × payoff^β`，`α`/`β` 可配

3. Family 健康度监控：
   - Family collapse 检测：top-1 family 占比 > 80% 时触发 alert
   - Family monopoly 检测：连续 N 轮同一 family 被选中时触发 alert
   - Alerts 发布到 `EvaluationSnapshot`

4. Family lifecycle 与结果绑定：
   - `reuse_streak` 与 `delayed_credit_sum` 负相关时，降低该 family 的 selection priority
   - Family `prune` 决策从纯 `support < threshold` 变为 `payoff_rank + support` 综合判断

**约束**：

- Family competition state 由 `temporal` owner 独占维护
- Credit 信号通过快照消费，不直接引用 `credit` 模块
- Selection 算法保留 `similarity-only` fallback 模式

**验收**：
- 多场景 rollout 中 active family 分布不再长期塌缩到单一 family
- Family collapse alert 在塌缩场景中被触发
- `long_term_payoff` 与 `delayed_credit_sum` 在多 cycle 后积累变化
- Payoff-weighted selection 与 similarity-only selection 可观测差异

### U03.3 — Evolution judge 接入主决策链

**owner**: `evaluation` + `joint_loop`
**位置**: `volvence_zero/evaluation/backbone.py`, `volvence_zero/joint_loop/runtime.py`

**内容**：

1. Judge 接入 joint loop：
   - `ETANLJointLoop` 在 background writeback 阶段调用 `judge_evolution_candidate()`
   - 结构 proposal（`TemporalStructureProposal`）先经 replay benchmark 验证
   - Judge 输出 `promote / hold / rollback` 决定 proposal 是否应用

2. Judge 决策分类：
   ```python
   class EvolutionVerdict(str, Enum):
       REAL_IMPROVEMENT = "real-improvement"
       STYLE_DRIFT = "style-drift"
       UNSAFE_MUTATION = "unsafe-mutation"
       INSUFFICIENT_EVIDENCE = "insufficient-evidence"
   ```
   - `REAL_IMPROVEMENT` → promote（应用 proposal）
   - `STYLE_DRIFT` → hold（不应用，继续收集证据）
   - `UNSAFE_MUTATION` → rollback（回退到上一个 checkpoint）
   - `INSUFFICIENT_EVIDENCE` → hold（不应用，等待更多数据）

3. Replay benchmark 集成：
   - 每个 proposal 应用前，在 replay trace 上模拟其效果
   - 比较 proposal 前后的 evaluation scores
   - 只有 `REAL_IMPROVEMENT` 且 replay 改善 > threshold 时才 promote

4. Judge 结果作为 self-modification evidence：
   - 每次 judge 调用产出 `SelfModificationRecord`
   - 记录 verdict、replay 结果、confidence
   - 长期 ledger 追踪 judge 的准确率

**约束**：

- Judge 只裁决，不执行——执行由 `temporal` owner 或 `reflection` owner 负责
- Judge 不阻塞在线交互——replay benchmark 异步执行或在后台 phase
- `UNSAFE_MUTATION` 触发 rollback 必须通过门控系统

**验收**：
- 结构 proposal 在没有通过 judge 验证时不被应用
- `STYLE_DRIFT` 和 `INSUFFICIENT_EVIDENCE` 不导致 proposal 被应用
- Replay benchmark 的结果可在 `JointCycleReport` 中观测
- Judge 准确率可追踪

### U03.4 — Regime 选择走向 RL 训练

**owner**: `regime`
**位置**: `volvence_zero/regime/identity.py`

**内容**：

1. Regime scoring 从 heuristic 到可学习权重：
   - 当前 regime selection 使用固定 scoring 规则（match score 基于 hints 和 context）
   - 引入 `RegimeSelectionWeights`：可学习的权重向量
   - Selection score = 固定规则 score × learned weight + bias

2. Delayed outcomes → regime reward：
   - `RegimeModule` 中 `delayed_outcomes` 的积累转化为 regime selection policy 的 reward
   - 正向 outcome → 增加当前 regime 的 selection weight
   - 负向 outcome → 降低当前 regime 的 selection weight
   - 更新使用 Phase 1 的策略优化器接口（简化版 REINFORCE）

3. Regime-temporal 协同：
   - Regime 切换时，通过快照中的 `regime_changed` 标记通知 temporal module
   - Temporal module 可据此调整 metacontroller 的切换敏感度
   - 目标：regime 切换和 temporal abstract action 切换趋于同步

4. Regime effectiveness tracking：
   - 每个 regime 的 `effectiveness_score` 不再只是事后统计
   - 变为 EMA（exponential moving average）of delayed outcomes
   - 在 `RegimeSnapshot` 中发布 `effectiveness_trend`

**约束**：

- Heuristic scoring 保留为 base score，learned weight 只做乘法调整
- 学习权重更新受 `ModificationGate.ONLINE` 约束
- Regime-temporal 协同只通过快照通信，不直接调用

**验收**：
- 不同 regime 的 selection weight 在多 session 模拟后展现分化
- `delayed_outcomes` 正向的 regime 权重升高，负向的权重降低
- Regime effectiveness trend 在 `RegimeSnapshot` 中可观测
- Heuristic fallback 模式可一键切换

## 4. 数据契约变更

| Schema | 变更类型 | 说明 |
|--------|---------|------|
| `TemporalAbstractionSnapshot` | 扩展 | 新增 `switch_gate_stats: SwitchGateStats \| None` |
| `DiscoveredActionFamily` | 扩展 | 新增 `long_term_payoff`, `delayed_credit_sum` |
| `RegimeSnapshot` | 扩展 | 新增 `effectiveness_trend: tuple[tuple[str,float],...]` |
| `JointCycleReport` | 扩展 | 新增 `evolution_verdict: str \| None` |

新增 schema：

| Schema | 位置 | 说明 |
|--------|------|------|
| `SwitchGateStats` | `temporal/metacontroller_components.py` | 切换门统计量 |
| `FamilyCompetitionState` | `temporal/interface.py` | family 竞争排行 |
| `EvolutionVerdict` | `evaluation/backbone.py` | judge 裁决结果枚举 |
| `RegimeSelectionWeights` | `regime/identity.py` | 可学习选择权重 |

## 5. 退出条件

Phase 3 视为完成，当且仅当以下全部满足：

1. 在 `alpha > 0` 的 SSL 训练后，`β_t` 展现准二值稀疏分布（无需手动阈值）
2. Action family 分布在多场景下不长期塌缩到单一 family
3. Evolution judge 接入 `ETANLJointLoop`，`STYLE_DRIFT` 时 proposal 不被应用
4. Regime selection weight 在多 session 模拟后展现学习趋势
5. 三个涌现机制（切换门、family 竞争、regime 选择）中至少两个的行为主要由数据驱动

## 6. 回滚触发与回滚动作

### 回滚触发

- 变分瓶颈训练后 `β_t` 全部塌到 0 或 1（无中间值也无稀疏模式）
- Family competition 导致新 family 完全无法存活（monopoly 更严重）
- Evolution judge 的 `REAL_IMPROVEMENT` 判定在 replay 中被证伪（高假阳性率）
- Regime weight 学习发散或所有 regime 权重趋同

### 回滚动作

1. **切换门**：切回 `switch_bias` 阈值模式（设 `alpha=0`）
2. **Family 竞争**：切回 `similarity-only` selection（关闭 payoff-weighted）
3. **Evolution judge**：恢复为旁路分析工具（不阻断 proposal）
4. **Regime 选择**：冻结 learned weight 为 1.0（退化为纯 heuristic scoring）

每个回滚独立——一个机制回滚不影响其他两个。

## 7. 需同步更新的文档

| 文档 | 更新内容 |
|------|---------|
| `docs/specs/temporal-abstraction.md` | 切换门涌现机制、family competition、evolution judge |
| `docs/specs/cognitive-regime.md` | regime selection 可学习权重 |
| `docs/specs/evaluation.md` | evolution judge 裁决分类 |
| `docs/specs/credit-and-self-modification.md` | 信用→family 竞争路径 |
| `docs/DATA_CONTRACT.md` | 新增/扩展 schema |

## 8. 风险与缓解

| 风险 | 级别 | 缓解 |
|------|------|------|
| `α` 调参空间大且敏感 | 中高 | 网格搜索 [0.01, 0.05, 0.1, 0.5, 1.0]；从大到小逐步探索 |
| 涌现行为不如 heuristic 稳定 | 中高 | 先 SHADOW 运行，A/B 对比 heuristic 和涌现版本 |
| Replay benchmark 计算成本 | 中 | 限制 replay trace 长度；异步执行不阻塞在线 |
| Family payoff-weighted selection 的冷启动 | 中 | 新 family 使用中性 payoff（0.5）启动，避免被立即淘汰 |
| Regime weight 学习与 temporal RL 的竞争 | 低 | 两者使用不同学习率；regime weight 更新更保守 |

## 9. 测试策略

1. **变分瓶颈测试**：多 `alpha` 值下 `β_t` 直方图的分布分析
2. **Family 竞争测试**：多场景 rollout 下 family 分布的多样性度量
3. **Judge 准确率测试**：replay benchmark 上 `REAL_IMPROVEMENT` 判定的事后验证
4. **Regime 学习测试**：模拟正/负 delayed outcome，验证权重变化方向
5. **回滚测试**：每个涌现机制的 fallback 开关能正确切回 heuristic

## 10. 与 `02_eta_nl_next_stage.md` 的关系

| `02` Phase | 本 Phase 步骤 | 关系 |
|------------|--------------|------|
| Phase A: 抽象动作竞争涌现 | U03.2 Family 竞争 | 直接覆盖，增加 delayed credit 绑定 |
| Phase D: evolution judge 主链 | U03.3 Judge 主链 | 直接覆盖 |
| Phase B: delayed credit horizon | 延后到 U04.1 | 需要更多基础设施（Phase 4 解决） |
| Phase C: structural proposal 强化 | 延后到 U04.3 | 需要反思提升（Phase 4 解决） |

## 11. 参考

- `docs/next_gen_emogpt.md` — Appendix B.3 Metacontroller, B.4 Rate-Distortion, B.5 Internal RL
- `docs/implementation/02_eta_nl_next_stage.md` — Phase A, D
- `docs/specs/temporal-abstraction.md` — R3/R4
- `docs/specs/cognitive-regime.md` — R14
- `docs/specs/evaluation.md` — R12
- `volvence_zero/temporal/metacontroller_components.py` — 切换门实现
- `volvence_zero/temporal/ssl.py` — SSL 训练器
- `volvence_zero/evaluation/backbone.py` — evolution judge
