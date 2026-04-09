# 信用分配与自修改 Spec

> Status: draft
> Last updated: 2026-04-06
> 对应需求: R9, R10

## 要解决的问题

如何在多个时间尺度上分配信用，并安全地让系统改进自身？

## 关键不变量

- 稀疏奖励是常态，不是边缘情况
- 自修改有门控：在线/后台/离线/人工审核分层
- 实时运行期间不可无限制突变基础模型
- 信用分配在多个层级进行

## 工程挑战

- 实现从 token 级到抽象动作级的层级信用分配
- 设计语义化的奖励记录（包含上下文和结果的结构化记录，非纯数值）
- 实现门控自修改：定义什么可在线改、什么需后台验证、什么需离线重训练
- 确保稀疏奖励下的信用分配不崩溃

## 算法候选

来自 `docs/next_gen_emogpt.md`：

### 层级信用分配

| 层级 | 信用类型 | 时间尺度 | 算法基础 |
|------|----------|----------|----------|
| Token/话语 | 即时表达质量 | online-fast | — |
| 轮次 | 用户响应效果 | online-fast | — |
| 会话 | 进展与 rupture/repair 结果 | session-medium | — |
| 长期 | 信任、能力、用户特定适应的增长 | background-slow | NL 多层嵌套结构 |
| 抽象动作 | 时间扩展策略的成功/失败 | session ~ background | ETA Internal RL |

当前实现口径补充：

- delayed credit 已不再只停留在 regime 名称：当前 `regime` owner 会发布带 `source_wave_id`、`source_turn_index`、`abstract_action`、`action_family_version` 的 delayed attribution
- `credit` owner 当前会把 delayed regime / delayed abstract action 转成 session-level 与 abstract-action-level `CreditRecord`
- 当前 delayed path 已扩成 multi-step ledger：`credit` owner 不只读取本轮 freshly-resolved attribution，也读取 regime owner 发布的 rolling payoff summary，以支持更长时间跨度的 credit accumulation
- background self-modification 当前不再只做数值调参：在 gate 允许时，slow reflection 可发出 bounded structural temporal proposal（`merge` / `split` / `prune`），仍受 target-specific gate 和可回滚审计约束

### Internal RL 时间抽象信用分配（ETA 附录 B.5）

通过时间抽象将有效时间范围从 token 级压缩到抽象动作级。每个抽象动作对应一段完整的子目标执行，奖励可直接归因到抽象动作级别。

### Delta 动量选择性遗忘（NL 附录 A.3）

通过梯度依赖的衰减实现选择性遗忘，避免无关梯度干扰信用分配。

### 门控自修改规则

| 修改目标 | 门控级别 | 触发条件 | 算法基础 |
|----------|----------|----------|----------|
| 检索权重、策略先验 | 在线可改 | 每轮/每 wave | CMS 高频层 |
| 抽象控制器参数、反思启发式 | 后台验证 | 会话后反思 | CMS 中频层 |
| 记忆提升阈值、基底微调 | 离线重训练 | 定期批量 | CMS 低频层 |
| 基础模型结构变更 | 人工审核 | 版本发布 | — |

CMS 的频率分层（NL 附录 A.5）天然提供门控。NL 通过内部学习率 `η^(i)` 控制每层的适应幅度。Hope 的自修改 Titans（附录 A.7）实现有界自修改——修改范围限于记忆模块的参数，基础模型保持冻结。

## 接口契约

**消费的输入**：
- `dual_track` 快照：轨道标记和信用分配上下文
- `temporal_abstraction` 快照：抽象动作信息（用于抽象动作级信用）
- `evaluation` 快照：评估分数（用于门控决策）

**产出的输出**：
- `credit` 快照：`CreditSnapshot`
  - 近期信用记录（语义化，含上下文）
  - 近期自修改记录（含 allow / block decision）
  - 各级别累计信用
  - 可被 owner 内部扩展为 abstract-action 级信用，而不改变公共 snapshot shape

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.5 节

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时（5.5）| 通过快照发布信用和自修改记录 |
| 依赖 | 时间抽象（5.2）| 消费抽象动作信息用于信用分配 |
| 依赖 | 双轨学习（5.4）| 按轨道隔离信用分配 |
| 依赖 | 评估体系（5.7）| 评估分数驱动门控决策 |
| 被依赖 | 连续记忆（5.3）| 信用记录作为反思输入 |
| 协作 | 多时间尺度学习（5.1）| 门控规则对齐时间尺度 |

当前实现口径：

- P06 的 turn / session credit 已稳定
- 第二阶段补充了 abstract-action credit 的 owner-side 扩展函数，用于 joint loop / rollout 后处理
- 当前 abstract-action credit 已可按 `world` / `self` 双轨记录，不再只剩 shared credit
- gate audit 已扩展为 `SelfModificationRecord.decision`
- joint loop 现在会把 metacontroller rollback / drift evidence 写入 owner-side modification audit，供 reflection / writeback 直接消费
- joint loop 现在也会把 metacontroller runtime state + policy objective 直接编码成 owner-side credit record，不再只靠 rollout 后处理 credit
- 当前 final wiring / session runtime 也会把 `retrieval_quality`、`reflection_usefulness`、`joint_learning_progress` 这些 learning evidence 转成 shared credit records，进入正式 `credit` snapshot
- reflection / writeback 仍以 bounded adaptation 为边界，不做无限制在线自修改

## 变更日志

- 2026-04-06: P12 hierarchical credit with temporal discount: CreditLedger tracks session-level credits with configurable gamma; CreditSnapshot gains session_level_credits and discount_factor; aggregate_session_credits computes discounted sums; reflection consolidation score uses session-level credit bonus
- 2026-04-06: 补充 retrieval / reflection / joint-loop learning evidence 进入 shared credit 的当前实现口径
- 2026-04-06: 补充 abstract-action credit、decision-aware gate audit，以及 metacontroller runtime adaptation audit
- 2026-04-06: 补充 metacontroller runtime credit evidence 的当前实现口径
- 2026-03-25: 初始版本，从 SYSTEM_DESIGN.md 和 next_gen_emogpt.md 提取
