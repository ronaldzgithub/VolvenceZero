# U04 — Phase 4: 长期闭环

> Status: draft
> Last updated: 2026-04-09
> Phase owner: `credit` + `reflection` + `evaluation` + `memory`
> 差距族: D（长期闭环缺失）
> 影响需求: R6, R9, R12
> 前置条件: Phase 3 完成（evolution judge 可用）
> 预估周期: 3–4 周

## 1. Phase 目标

让系统具备跨 session 的持续进化能力，有评估证据证明长期增长。

前三个 Phase 让系统获得了学习能力（Phase 1）、学习容量（Phase 2）和涌现结构（Phase 3），但这些能力仍局限在**单 session 内**。本 Phase 解决的是系统的"纵深"——从 turn/session 级反馈扩展到跨 session 的长期闭环：

- **反思**从默认 DISABLED 提升为 ACTIVE，产出有效的持久知识更新
- **信用分配**从短延迟扩展到中长延迟，能回答"10-20 轮后结构变化是否持续更优"
- **评估**从 session 内快照扩展到跨 session 纵向追踪
- **状态持久化**从内存 checkpoint 扩展到跨 session 持久存储

## 2. 设计依据

### 2.1 NL 慢反思（Appendix A.5, A.6）

CMS 低频层天然对应慢反思——参数每 `c^(K)` 步才更新一次，压缩的是长时间窗口的经验。M3 优化器的慢动量 `m^(2)` 每 `ν` 步聚合梯度，是优化器层面的"反思"。

### 2.2 ETA SSL-RL 交替（Appendix B.6）

Schmidhuber wake-sleep 循环：SSL 阶段压缩交互历史 → RL 阶段利用压缩表示改进控制器。后台慢尺度交替：

```
SSL: CMS 低频层更新，压缩跨会话知识
RL:  控制器先验和策略偏好的反思性更新
```

### 2.3 R6 反思两类产出

系统应支持至少两种反思产品：

- **记忆整合**：持久卡片、信念、开放循环、偏好轨迹
- **策略整合**：抽象控制器更新、路径先验、策略偏好

### 2.4 当前代码基础

已有可复用的基础设施：

- `ReflectionEngine` + `ReflectionModule`：完整反思引擎
- `MemoryConsolidation`/`PolicyConsolidation`：双产出结构
- `TemporalStructureProposal`/`TemporalPriorUpdate`：时间结构提案
- `WritebackMode.DISABLED/PROPOSAL_ONLY/APPLY`：分级写回
- `MemoryStoreCheckpoint`/`CMSCheckpointState`/`RegimeCheckpoint`/`CausalPolicyCheckpoint`：全套 checkpoint
- `EvaluationBackbone.judge_evolution_candidate()`：evolution judge

**缺口**：反思默认 DISABLED；delayed credit 偏短延迟；评估→学习闭环未完成；所有 checkpoint 只在内存中。

## 3. 步骤分解

### U04.1 — Delayed credit horizon 扩展

**owner**: `credit`
**位置**: `volvence_zero/credit/gate.py`

**内容**：

1. 多步 attribution ledger：
   - 替代当前的单条 delayed queue
   - Ledger 结构：`(action_id, family_id, regime_id, timestamp, outcome_history[])`
   - 每个条目保存最近 N 步（默认 N=20）的 outcome 序列
   - 支持回看：给定 `action_id`，返回其后续 N 步的 outcome 聚合

2. N-step delayed outcome aggregation：
   - 从单步 `pop(0)` 变为 N-step windowed aggregation
   - 聚合方式：`V_N = Σ_{k=0}^{N-1} γ^k · r_{t+k}`（折扣累积收益）
   - 折扣因子 `γ` 与 `CreditSnapshot.discount_factor` 对齐（默认 0.95）

3. Rolling payoff 计算：
   - 对同一 family/regime 序列计算 rolling payoff
   - `family_rolling_payoff[family_id] = EMA(N-step outcomes for this family)`
   - `regime_rolling_payoff[regime_id] = EMA(N-step outcomes for this regime)`
   - 写入 `CreditSnapshot.session_level_credits` 供下游消费

4. 长期 credit 写入 session report：
   - `CreditSnapshot` 新增 `delayed_ledger_size: int`、`horizon_depth: int`
   - Session 结束时生成 `SessionCreditReport`：各 family/regime 的 rolling payoff 排行

**约束**：

- Ledger 内存有界：最多保留 1000 条活跃条目
- 超出容量时 FIFO 淘汰最旧条目
- N-step 聚合不修改原始 reward——只增加信号丰富度

**验收**：
- `derive_delayed_attribution_credit_records()` 返回的信用包含 N-step 聚合值
- 不同 family 的 `rolling_payoff` 在多 cycle 后展现差异
- `delayed_ledger_size` 和 `horizon_depth` 在快照中可观测
- 向后兼容：当 `horizon_depth=1` 时退化为现有单步行为

### U04.2 — 反思从 DISABLED 提升为 SHADOW→ACTIVE

**owner**: `reflection` + `integration`
**位置**: `volvence_zero/integration/final_wiring.py`, `volvence_zero/reflection/writeback.py`

**内容**：

1. 默认接线级别调整：
   - `build_final_runtime_modules` 中 `ReflectionModule` 从 `WiringLevel.DISABLED` 改为 `WiringLevel.SHADOW`
   - SHADOW 模式下：ReflectionEngine 正常运行并产出 proposal，但 writeback 不执行
   - Proposal 记录到日志用于事后分析

2. Proposal 准确率 benchmark：
   - 每个 proposal 记录其预期效果（"merge family X+Y should improve stability by Z%"）
   - 事后检查：proposal 预测的方向是否与实际变化一致
   - 准确率 = 预测方向正确的 proposal 数 / 总 proposal 数
   - 在 `EvaluationSnapshot` 中发布 `reflection_accuracy`

3. SHADOW→ACTIVE 提升条件：
   - `reflection_accuracy > 0.6`（连续 N 个 session window）
   - 无 `UNSAFE_MUTATION` 类 proposal
   - Evolution judge 确认 SHADOW 期间的 proposal 质量
   - 提升通过 `FinalRolloutConfig` 手动触发（不自动）

4. ACTIVE 模式下的 writeback 保护：
   - `WritebackMode.APPLY` 仍受 `has_blocking_writeback()` 门控
   - 每次 writeback 前通过 evolution judge 验证
   - Writeback 后的 5 个 cycle 内如果评估退化，自动回退

**约束**：

- 反思不阻塞在线交互——异步或在 post-propagate phase 执行
- SHADOW→ACTIVE 不自动提升——需要运维人员确认评估证据
- 回退不影响已写入的记忆条目——只回退策略整合部分

**验收**：
- 默认接线为 SHADOW，反思 proposal 被记录
- `reflection_accuracy` 在 `EvaluationSnapshot` 中可观测
- ACTIVE 模式下 writeback 通过 evolution judge 验证
- 回退机制在评估退化时自动触发

### U04.3 — Structural proposal bundle

**owner**: `reflection`
**位置**: `volvence_zero/reflection/writeback.py`

**内容**：

1. Proposal 粒度从单 family 扩展到结构级：
   - `TemporalStructureProposal` 新增 `scope` 字段：`SINGLE_FAMILY` / `FAMILY_CLUSTER` / `REGIME_SEQUENCE` / `TRACK_COUPLING`
   - `FAMILY_CLUSTER`：对多个 family 的联合操作（merge/split/rebalance）
   - `REGIME_SEQUENCE`：对 regime 切换序列的优化建议
   - `TRACK_COUPLING`：调整双轨之间的耦合强度

2. Evidence pack：
   - 每个 proposal 附带 `EvidencePack`：
     ```python
     @dataclass(frozen=True)
     class EvidencePack:
         source_benchmark_ids: tuple[str, ...]
         delayed_credit_summary: tuple[tuple[str, float], ...]
         session_trend: tuple[tuple[str, float], ...]
         confidence: float
         supporting_cycles: int
     ```
   - 生成条件：至少 3 个 session window 的一致趋势

3. Proposal 成败追踪：
   - 每个已执行的 proposal 在长期 ledger 中记录
   - Ledger 条目：`(proposal_id, applied_at, expected_effect, actual_effect, verdict)`
   - `verdict`：`CONFIRMED` / `REFUTED` / `INCONCLUSIVE`
   - Ledger 供 reflection engine 未来决策参考——避免重复提出已被 refuted 的 proposal

**约束**：

- 新 scope 类型的 proposal 默认在 `PROPOSAL_ONLY` 模式下运行
- 只有 `SINGLE_FAMILY` 和 `FAMILY_CLUSTER` 的 `APPLY` 模式在 Phase 4 内开启
- `REGIME_SEQUENCE` 和 `TRACK_COUPLING` 留到 Phase 4 之后再 APPLY

**验收**：
- Proposal 的 scope 包含多种类型
- Evidence pack 非空且 `supporting_cycles > 0`
- Proposal ledger 能追踪成败
- 已 refuted 的 proposal pattern 不被重复提出

### U04.4 — Cross-session benchmark

**owner**: `evaluation`
**位置**: 新增 `volvence_zero/evaluation/longitudinal.py`

**内容**：

1. Cross-session benchmark 套件：
   - `LongitudinalBenchmark`：跨多个 session window 的评估框架
   - 输入：session reports 序列
   - 输出：趋势分析（改善/稳定/退化）

2. Session-window 对比：
   - 短期（最近 5 轮）vs 中期（最近 20 轮）vs 长期（全部历史）
   - 比较维度：task score、relationship score、family diversity、regime stability
   - 产出 `LongitudinalReport`：每个维度的趋势斜率和置信度

3. Family/regime 长期表现纵向报告：
   - 每个 family 的 `survival_curve`：从发现到当前的 payoff 变化
   - 每个 regime 的 `effectiveness_curve`：effectiveness 的 EMA 趋势
   - Top-K 和 Bottom-K 的排行变化

4. 成功判据定义：
   - "长期人格化"：用户相关记忆的持久率 > 50%
   - "关系化积累"：relationship track credit 的跨 session 累积非递减
   - "策略化积累"：family 多样性在 20 session 后稳定，不塌缩
   - 判据结果写入 `EvaluationSnapshot.longitudinal_verdict`

**约束**：

- Benchmark 为只读——不修改系统状态
- 支持离线运行（从 checkpoint 序列重建）
- 不依赖真实用户数据——支持合成 session 序列

**验收**：
- `LongitudinalReport` 在合成 session 序列上产出非空趋势
- 改善和退化场景分别被正确识别
- `longitudinal_verdict` 在 `EvaluationSnapshot` 中可观测

### U04.5 — 跨 session 持久化

**owner**: `memory`
**位置**: `volvence_zero/memory/store.py`, 新增 `volvence_zero/memory/persistence.py`

**内容**：

1. 持久化后端接口：
   ```python
   class PersistenceBackend(ABC):
       @abstractmethod
       def save_checkpoint(self, *, key: str, data: bytes, version: int) -> None: ...
       @abstractmethod
       def load_checkpoint(self, *, key: str) -> tuple[bytes, int] | None: ...
       @abstractmethod
       def list_checkpoints(self, *, prefix: str) -> tuple[str, ...]: ...
       @abstractmethod
       def delete_checkpoint(self, *, key: str) -> None: ...
   ```

2. 文件系统后端（默认）：
   - 每个 checkpoint 存为独立文件：`{base_dir}/{key}_v{version}.json`
   - JSON 序列化（frozen dataclass → dict → JSON）
   - 版本化：保留最近 K 个版本（默认 K=5）
   - 旧版本自动清理

3. 各模块 checkpoint 的持久化集成：

   | Checkpoint | Key prefix | 保存时机 |
   |------------|-----------|---------|
   | `MemoryStoreCheckpoint` | `memory/store` | Session 结束时 |
   | `CMSCheckpointState` | `memory/cms` | Session 结束时 |
   | `RegimeCheckpoint` | `regime/state` | Regime 切换时 |
   | `CausalPolicyCheckpoint` | `policy/causal` | 策略更新后 |

4. Session 恢复流程：
   - `AgentSessionRunner` 启动时检查持久化后端
   - 发现已有 checkpoint → 加载最新版本 → 恢复 MemoryStore + CMS + Regime + Policy
   - 恢复后发布初始快照，标注 `restored_from_session: str`
   - 未发现 checkpoint → 正常冷启动

5. 版本兼容检查：
   - Checkpoint 中记录 schema version
   - 加载时校验版本兼容性
   - 不兼容版本 → 忽略旧 checkpoint，冷启动，记录 warning

**约束**：

- 持久化为可选——`PersistenceBackend` 为 `None` 时退化为内存模式
- 不依赖外部数据库——文件系统后端即可
- JSON 序列化不处理非标准类型（bytes、numpy array）——如有需要后续扩展
- 保存过程不阻塞在线交互——异步或在 session 间隙执行

**验收**：
- MemoryStore 写入 → session 结束 → 保存 → 重启 → 加载 → 读回相同记忆
- CMS 状态跨 session 保持——恢复后 band 向量/参数与保存时一致
- 策略参数跨 session 保持——恢复后 policy weights 与保存时一致
- 版本不兼容时安全降级到冷启动

## 4. 数据契约变更

| Schema | 变更类型 | 说明 |
|--------|---------|------|
| `CreditSnapshot` | 扩展 | 新增 `delayed_ledger_size`, `horizon_depth` |
| `TemporalStructureProposal` | 扩展 | 新增 `scope` 字段 |
| `EvaluationSnapshot` | 扩展 | 新增 `reflection_accuracy`, `longitudinal_verdict` |

新增 schema：

| Schema | 位置 | 说明 |
|--------|------|------|
| `SessionCreditReport` | `credit/gate.py` | Session 级信用报告 |
| `EvidencePack` | `reflection/writeback.py` | Proposal 证据包 |
| `LongitudinalReport` | `evaluation/longitudinal.py` | 跨 session 趋势报告 |
| `PersistenceBackend` | `memory/persistence.py` | 持久化后端接口 |

## 5. 退出条件

Phase 4 视为完成，当且仅当以下全部满足：

1. Delayed credit horizon > 1 步，rolling payoff 在多 cycle 后展现差异
2. 反思在 SHADOW 模式下 `reflection_accuracy > 0.6`
3. Structural proposal 有 evidence pack，成败可追踪
4. Cross-session benchmark 能产出趋势报告
5. 系统重启后能恢复到上一次 session 的完整状态
6. `next_gen_emogpt.md` 的 7 个 Acceptance Questions 全部可肯定回答

## 6. 回滚触发与回滚动作

### 回滚触发

- 反思 ACTIVE 后连续 5 个 cycle 评估退化
- Delayed credit horizon 扩展后 reward shaping 导致 RL 不收敛
- 持久化导致恢复后系统行为与保存前不一致
- Cross-session benchmark 显示所有趋势为退化

### 回滚动作

1. **反思**：回退到 SHADOW 或 DISABLED（冻结反思，保留已写入的记忆）
2. **Delayed credit**：设 `horizon_depth=1` 退化为单步
3. **持久化**：关闭持久化后端（设为 `None`），回到内存模式
4. **Structural proposal**：所有新 scope proposal 设为 `PROPOSAL_ONLY`
5. **跨 session**：从最近的安全 checkpoint 恢复（版本化保证可选）

## 7. 需同步更新的文档

| 文档 | 更新内容 |
|------|---------|
| `docs/specs/continuum-memory.md` | 跨 session 持久化、CMS 恢复 |
| `docs/specs/evaluation.md` | 纵向评估、reflection accuracy |
| `docs/specs/credit-and-self-modification.md` | N-step delayed attribution、rolling payoff |
| `docs/specs/multi-timescale-learning.md` | 慢反思的 SHADOW→ACTIVE 路径 |
| `docs/DATA_CONTRACT.md` | 新增/扩展 schema |
| `docs/implementation/01_package_registry.md` | P07 接线级别从 DISABLED 更新 |

## 8. 风险与缓解

| 风险 | 级别 | 缓解 |
|------|------|------|
| 反思 ACTIVE 后写回错误积累 | 中高 | proposal-only 先行 + accuracy benchmark + 5-cycle 退化自动回退 |
| N-step horizon 过长导致信用分配噪声 | 中 | 从 N=5 开始，逐步增加到 N=20 |
| 持久化 JSON 序列化的 float 精度丢失 | 低 | 使用足够精度 (16 digits) + 恢复后 tolerance 检查 |
| Cross-session benchmark 计算开销 | 低 | 只在 session 间隙或后台运行 |
| 旧 checkpoint 格式与新代码不兼容 | 中 | 版本号 + 兼容性校验 + 安全降级 |

## 9. 测试策略

1. **Delayed credit 测试**：合成 rollout 序列下 N-step 聚合值的正确性
2. **反思准确率测试**：已知 ground-truth 场景下 proposal 方向预测的正确率
3. **Proposal 追踪测试**：proposal 写入 ledger → 事后检查 → verdict 正确
4. **Benchmark 测试**：合成 session 序列上趋势识别（改善/退化/稳定）
5. **持久化往返测试**：保存 → 加载 → 比较，所有 checkpoint 类型
6. **版本兼容测试**：旧版本 checkpoint + 新代码 → 安全降级

## 10. 与 `02_eta_nl_next_stage.md` 的关系

| `02` Phase | 本 Phase 步骤 | 关系 |
|------------|--------------|------|
| Phase B: delayed credit horizon | U04.1 | 直接覆盖 |
| Phase C: structured compression | U04.3 | 部分覆盖（structural proposal bundle） |
| Phase E: cross-session 增长证明 | U04.4 + U04.5 | 直接覆盖 |

## 11. 参考

- `docs/next_gen_emogpt.md` — R6/R9/R12, Appendix A.5 CMS 低频层, B.6 SSL-RL 交替
- `docs/implementation/02_eta_nl_next_stage.md` — Phase B, C, E
- `docs/specs/continuum-memory.md` — R5/R6
- `docs/specs/evaluation.md` — R12
- `docs/specs/credit-and-self-modification.md` — R9/R10
- `volvence_zero/reflection/writeback.py` — 现有反思引擎
- `volvence_zero/credit/gate.py` — 现有信用分配
- `volvence_zero/integration/final_wiring.py` — 现有接线配置
