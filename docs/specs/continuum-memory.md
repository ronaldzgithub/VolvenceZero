# 连续记忆系统 Spec

> Status: draft
> Last updated: 2026-04-06
> 对应需求: R5, R6

## 要解决的问题

如何实现跨时间尺度的连续记忆谱，并通过反思将经验沉淀为持久结构？

## 关键不变量

- **NL 范式层**：记忆是任何由输入引起的 neural update，分布在所有参数中（NL §A.8）
- **运行时工程层**：系统使用显式 owner 模块（MemoryStore, CMS, snapshots）使范式层的分布式记忆可检查、可回滚
- 记忆是连续谱，不是二元短期/长期分割
- 记忆写入通过正式 owner 和 API，不可绕过
- 慢反思产出两类产物：记忆沉淀 + 策略沉淀
- 慢反思异步运行，不阻塞实时交互

## 工程挑战

- 设计多层记忆（瞬态 → 情景 → 持久 → 派生索引），每层有不同的更新频率
- 实现提升/衰减/部分重建机制
- 实现异步慢反思路径，产出记忆沉淀和策略沉淀两类产物
- 确保记忆写入通过正式 owner 和 API，不可绕过

## 算法候选

来自 `docs/next_gen_emogpt.md`：

### CMS 多频率层（NL 附录 A.5）

```
y_t = MLP^(ν_K)(MLP^(ν_{K-1})(... MLP^(ν_1)(x_t)))
```

第 `i` 层 MLP 的参数每 `c^(i)` 步更新一次。

### 记忆层级映射

| 层 | 内容 | 更新频率 | 算法对应 |
|----|------|----------|----------|
| 瞬态工作状态 | 当前 wave、场景、活跃帧状态 | 每 turn | CMS 最高频层 |
| 会话情景状态 | 近期交互轨迹、未解决张力 | 每场景/会话 | CMS 中频层 |
| 持久语义记忆 | 稳定的用户/自我/关系/世界知识 | background-slow | CMS 低频层 |
| 派生索引 | 可重建的检索辅助、摘要、任务投影 | 按需重建 | — |

### 记忆操作

- **写入**：仅通过正式 owner 和 API
- **提升**：低频层从高频层提取持久知识
- **衰减**：有限容量迫使遗忘非关键信息
- **部分重建**：遗忘后可从低频层回流知识到高频层
- **抗遗忘**：CMS 确保低频层保留被高频层遗忘的知识

### 慢反思路径（R6 + NL/ETA）

异步运行于交互窗口之后：
- NL 的 CMS 低频层天然对应慢反思——参数每 `c^(K)` 步才更新一次
- ETA 的 SSL-RL 交替循环提供反思-强化交替模式
- M3 优化器的慢动量 `m^(2)` 每 `ν` 步聚合梯度，是优化器层面的"反思"

**两类反思产物**：

| 产物类型 | 内容 | 写入目标 |
|----------|------|----------|
| 记忆沉淀 | 持久卡片、信念、开放循环、偏好轨迹 | 持久语义记忆 |
| 策略沉淀 | 抽象控制器更新、路径先验、策略偏好 | 控制器参数 |

当前实现口径：

- P07 先输出 consolidation proposal 和审计信息
- 第二阶段已补充 bounded apply path：memory owner 支持 checkpoint / rollback
- 当前真实写回首先落在低风险目标：durable promotion、belief writeback、decay、promotion threshold 调整
- CMS 当前已升级为 machine-readable 多频带 owner state：`online-fast / session-medium / background-slow` 都带有 `cadence_interval / observations_since_update / pending_signal`
- `session_medium / background_slow` 不再每 turn 同频更新，而是通过显式 cadence gating 分层推进
- reflection apply 已从散落阈值收敛成统一 bounded consolidation score 路径，memory/regime writeback 幅度由同一条 score 决定
- retrieval 已补充 memory-owner-only 的 hybrid rank：`lexical + semantic`，并把 semantic index 纳入 checkpoint / restore
- runtime query facets 当前可组合 `user_text + substrate + 上一轮 temporal_abstraction + 上一轮 dual_track` 的已发布快照上下文，不引入同轮循环依赖
- 当前 reflection 已可消费 regime owner 发布的 `delayed_outcomes + identity_hints`，并在 credit gate 允许时把 typed identity proposal 沉淀为 durable identity entries；memory owner 仍是唯一持久写入 owner

### NL 关键洞察

NL 重新定义"记忆 = 任何由输入引起的神经更新"（附录 A.8），记忆分布在所有参数中而非独立模块。灾难性遗忘是压缩的自然后果——有限容量迫使模型遗忘以容纳新信息。

## 接口契约

**消费的输入**：
- `substrate` 快照：当前可实现的 substrate surface（用于记忆检索的上下文）
- `temporal_abstraction` 快照：当前控制器状态（用于记忆检索的意图上下文）
- `dual_track` 快照：轨道标记（用于按轨道写入记忆）
- `reflection` 快照：反思产物（写回持久记忆）

**产出的输出**：
- `memory` 快照：`MemorySnapshot`
  - 各层级摘要（模块自身生成）
  - 本轮检索到的相关记忆
  - 统计信息
  - pending promotion / decay 状态
  - machine-readable `cms_state`

**owner API**：

- 所有写入通过 `MemoryWriteRequest` 进入 Memory owner
- retrieval contract 区分“本轮检索结果”与“持久状态摘要”
- retrieval ranking 由 Memory owner 内部统一负责，可组合 `user_text + substrate facets + owner query facets` 做 hybrid retrieval
- promotion / decay / reconstruction 先建模为显式状态，不作为隐式副作用暴露给消费者
- 第二阶段补充 `checkpoint / restore` 与 bounded `apply_reflection_consolidation`
- `promotion_threshold` 属于 Memory owner 的可回滚低风险自适应参数
- semantic index 属于 Memory owner 内部索引，不向外暴露独立 owner

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.3 节

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时（5.5）| 通过快照发布记忆状态 |
| 依赖 | 多时间尺度学习（5.1）| 各层按不同频率更新 |
| 被依赖 | 双轨学习（5.4）| 提供按轨道隔离的记忆存储 |
| 被依赖 | 认知 Regime（5.8）| 提供 regime 历史效果的记忆 |
| 协作 | 时间抽象（5.2）| 为 metacontroller 提供记忆上下文 |
| 协作 | 信用分配（5.6）| 反思输入包含信用分配记录 |
| 协作 | 评估体系（5.7）| F4 学习质量中的记忆沉淀质量评估 |

## 变更日志

- 2026-04-09: next_gen_emogpt v2: explicit distinction between NL paradigm-level memory (any neural update) and runtime-level memory (explicit owner modules + snapshots); both layers are valid and complementary
- 2026-04-09: U04 Persistence load path fix: `MemoryStore.load_from_backend()` now calls `_reconstruct_checkpoint()` to rebuild `MemoryStoreCheckpoint` from deserialized JSON dict and `restore_checkpoint()` to restore full store state including entries, strata, semantic index, and CMS. Verified: save→restart→load roundtrip preserves entries (including track/stratum), CMS MLP parameters, and handles version incompatibility via safe degradation. CMS nested meta-targets also survive persistence roundtrip.
- 2026-04-09: U02 CMS MLP Upgrade + Nested variant: CMS bands upgraded from fixed-dim vector (`dim=3`) to optional MLP mode (`mode="mlp"`) with configurable `d_in`/`d_hidden`. Each band is a `CMSBandMLP` (2-layer residual MLP: `y = x + W1 @ tanh(W2 @ x)`, W1 zero-init for identity at start). `CMSVariant.NESTED` added: background band meta-learns session init target, session meta-learns online init target (`_update_nested_meta_targets`); `reset_context()` re-initializes from these meta-learned targets (not simple copy). Verified: init error decreases across repeated context resets. `CMSCheckpointState` extended with `nested_session_init_target`/`nested_online_init_target`. Parameter budget: ~1K params/band at d_in=16, d_hidden=32.
- 2026-04-06: P19 N-dim CMS + Gradient-Style Updates: upgraded from EMA blending to gradient-style per-band updates with configurable learning rates (online_lr, session_lr, background_lr), momentum (momentum_beta), and anti-forgetting backflow from slow→fast bands. CMSBandState now includes learning_rate, momentum, and anti_forgetting_strength. encoder_feedback now handles dim mismatch via projection.
- 2026-04-06: P10 CMS-enhanced encoder: CMS bands feed into SequenceEncoder prior; encoder output feeds back to CMS via observe_encoder_feedback; cadence gating preserved
- 2026-04-06: 补充 machine-readable CMS bands + cadence gating、bounded consolidation score、hybrid semantic retrieval 的当前实现口径
- 2026-04-06: 补充 checkpoint/rollback、promotion_threshold 和最小 CMS 核的当前实现口径
- 2026-03-25: 初始版本，从 SYSTEM_DESIGN.md 和 next_gen_emogpt.md 提取
