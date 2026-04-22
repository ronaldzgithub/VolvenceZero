# EmoGPT Next-Gen — 系统设计文档

> Status: draft
> Version: 0.2
> Last updated: 2026-04-21
> Source: `docs/next_gen_emogpt.md`（唯一设计源头）、`docs/prd.md`（产品需求）

---

## 1. 设计目标

构建一个**有界、契约驱动、持续适应的认知代理**。核心产品价值是**关系与主体性**（EQ + 信任），而非单纯智力（IQ）。

系统融合两个互补理论基础：

| 理论 | 贡献 |
|------|------|
| **Nested Learning (NL)** | 系统级教义：多时间尺度学习、连续记忆谱、嵌套自适应 |
| **Emergent Temporal Abstractions (ETA)** | 缺失的动作机制：发现并强化时间抽象的内部控制器 |

---

## 2. 系统总体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        编排调度层 (Orchestrator)                      │
│  wave/turn 调度 · 事件分发 · 后台任务管理 · 快照传播                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │
│  │  时间抽象与    │  │  连续记忆系统  │  │  信用分配与自修改      │   │
│  │  内部控制层    │  │  (MemoryOS)   │  │  (Credit & SelfMod)  │   │
│  │               │  │               │  │                       │   │
│  │ Metacontroller│  │ 瞬态工作状态   │  │ 层级信用分配           │   │
│  │ 切换单元 β_t  │  │ 会话情景状态   │  │ 语义化奖励记录         │   │
│  │ 控制器代码 z_t│  │ 持久语义记忆   │  │ 门控自修改规则         │   │
│  │ Internal RL   │  │ 派生索引      │  │                       │   │
│  └───────┬───────┘  └───────┬───────┘  └───────────┬───────────┘   │
│          │                  │                       │               │
│  ┌───────┴──────────────────┴───────────────────────┴───────────┐   │
│  │              双轨学习层 (Dual-Track Learning)                  │   │
│  │  World/Problem Track ←→ Self/Relationship Track              │   │
│  │  共享基础设施 · 语义隔离的记忆/信用/控制器/评估                   │   │
│  └──────────────────────────┬────────────────────────────────────┘   │
│                             │                                       │
│  ┌──────────────────────────┴────────────────────────────────────┐   │
│  │              认知 Regime 层 (Cognitive Regime)                 │   │
│  │  持久 regime 身份 · 可记忆 · 可选择 · 可训练                    │   │
│  │  casual social | acquaintance | emotional support | ...       │   │
│  └──────────────────────────┬────────────────────────────────────┘   │
│                             │                                       │
│  ┌──────────────────────────┴────────────────────────────────────┐   │
│  │              评估体系 (Evaluation)                              │   │
│  │  任务能力 · 交互质量 · 关系连续性 · 学习质量 · 抽象质量 · 安全    │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                    慢反思路径 (Slow Reflection)                       │
│  异步 · 读取交互轨迹/决策/结果/张力 · 产出记忆沉淀 + 策略沉淀          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │              稳定基底层 (Stable Substrate)                     │   │
│  │  冻结/极慢更新的基础模型 · 语言与世界建模 · 残差流表示            │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 分层原则

系统严格遵循 **R2（稳定基底 + 自适应控制器）** 分层：

| 层 | 职责 | 更新频率 | 算法基础 |
|----|------|----------|----------|
| 稳定基底 | 语言/世界建模，提供残差流表示 | rare-heavy（离线重训练） | 冻结 LLM，ETA rate-distortion 证明 |
| 自适应控制器 | 时间抽象、内部控制、regime 选择 | online-fast ~ session-medium | Metacontroller, Internal RL |
| 记忆系统 | 连续记忆谱，知识存储与检索 | 各层不同频率 | CMS 多频率 MLP 链 |
| 反思路径 | 经验沉淀为持久结构 | background-slow | CMS 低频层, SSL-RL 交替 |
| 编排层 | 模块协调、快照传播 | 每 wave/turn | 契约式运行时 |

### 2.2 当前收敛状态（2026-04-20）

当前主链已经不再只是“substrate -> evaluation -> credit”的读数式结构，而是收敛到：

- `prediction_error` 作为正式 ACTIVE runtime owner 发布 `evaluated_prediction -> actual_outcome -> next_prediction -> error`
- `credit` / `evaluation` 明确退居 prediction-error-first 主链的下游聚合与读数层
- `memory` / `regime` / `temporal` / `reflection` 已直接消费 `prediction_error`，形成第一版 PE-first cognitive loop
- session owner 已接上 bounded、substrate-aware 的 `rare-heavy` review / import 闭环：高 PE 持续时可触发离线 artifact 训练，并通过 owner-side surface 导入 temporal / memory / substrate 三类更新
- 固定 scripted dialogue benchmark 已成为“高 PE -> temporal response -> delayed improvement”是否成立的顶层证据面

### 2.3 当前实现边界（2026-04-21）

为避免把目标态和已落地实现混写，当前代码状态建议按三类理解：

- **已落地并优于旧版顶层文档的部分**
  - `final_wiring` 已把 temporal public evidence、prediction-error evidence、cross-session benchmark、evolution judge、以及 reflection -> temporal prior writeback 收敛到同一条 integration spine。
  - dialogue benchmark 已从固定 scripted A/B 扩展到 perturbation、systematic replay、replay selection artifact、multi-artifact acceptance、以及 staged real comprehensive runner。
  - session owner 已具备 bounded、substrate-aware 的 `rare-heavy` review/import 路径：temporal / memory / substrate 三侧都支持 checkpoint / rollback / import surface，offline pipeline 也已能产出统一 artifact。
  - session owner 已新增独立的 session-post slow loop：turn 内 final wiring 只产出 deferred slow-writeback request，真正的 `background-slow` consolidation 在 context/session boundary 后排队执行，并通过 owner-side memory / regime / temporal apply surface 落地。
  - session-post slow loop 现已从 runner telemetry 提升为正式 `session_post_slow_loop` 公共 slot：queue state 与 recent completion summaries 通过独立 snapshot 发布，外部 benchmark / report surface 可直接消费，不需要读取 session owner 私有队列。
  - 应用层第一阶段已新增正式 `retrieval_policy` / `domain_knowledge` / `boundary_policy` surface：ETA 在线控制层先发布检索策略，知识 owner 再发布 compact 专业事实证据，边界 owner 发布 citation / clarification / refer-out 等限制，response/evaluation 只读取公共快照。
  - 应用层第二阶段已新增正式 `case_memory` surface：它作为 `memory` 的 sibling owner 发布 compact case hits、problem patterns 与 risk markers，使主链第一次具备知识 hits 与案例 hits 的 retrieval mix，同时保持 `memory` 不退化成“大一统经验仓库”。

- **已落地但仍受 gate 约束的部分**
  - reflection writeback 不是无条件常开；仍受 `writeback_mode`、credit gate、evolution judge 共同约束。
  - `rare-heavy` 只有在高 PE 持续、trace window 足够、cooldown 满足、且 offline RL 至少执行 1 步时才会真正 import。

- **仍属目标态、尚未完全实现的部分**
  - `rare-heavy` 当前已不止作用于 temporal / memory artifact；它已经能对 substrate owner 的 adapter-delta 状态做离线训练、artifact 导出、导入和回滚。但它仍不等于完整的基础模型持续预训练或蒸馏。
  - 双轨语义已经进入 memory / credit / evaluation / regime，但默认 runtime 还没有两个完全独立的 track-specific metacontroller。

---

## 3. 核心模块设计

### 3.1 编排调度层 (Orchestrator)

**职责**：协调所有模块的执行，管理快照传播，不直接调用模块的处理方法。

**关键机制**：
- **Wave 级调度**：协调一轮交互中各模块的执行顺序
- **快照传播**：收集各模块发布的不可变快照，构建 upstream dict 传递给下游模块
- **事件分发**：场景状态变化等事件的异步通知
- **后台任务管理**：触发和监控慢反思等异步任务

**约束**：
- 编排器可调用快照传播/读取，但不直接调用模块的处理方法
- 编排器不持有模块的内部状态

### 3.2 稳定基底层 (Stable Substrate)

**职责**：提供语言和世界建模能力，并通过可实现的 substrate contract 向上层发布可消费状态。

**对应需求**：R2

**设计要点**：
- 基础模型冻结或极慢更新（rare-heavy 时间尺度）
- 当前稳定 contract 默认发布 `feature_surface` 和可选 `token_logits`
- `residual_activations` 只在 open-weight / hook 可用时作为增强输入发布，不再假设默认可用
- ETA 的 rate-distortion 分析证明：冻结基础模型是发现时间抽象的前提

**算法基础**：
- NL 的 CMS 通过控制内部学习率 `η^(i) → 0` 实现"接近冻结但可微调"
- Hope 的自修改 Titans 将在线适应限制在有界的控制器层

### 3.3 时间抽象与内部控制层

**职责**：在 token 生成之上实现正式的时间抽象动作层，支持子目标级控制。

**对应需求**：R3, R4

#### 3.3.1 Metacontroller

Metacontroller 是一个生成式随机循环超网络，无需子目标标签即可发现时间抽象动作。

**架构组件**：

```
残差流 e_{1:T}
    │
    ├──→ 内部序列嵌入器 → s(e_{1:T})  [全局嵌入，训练时非因果]
    │
    ├──→ 编码器 (GRU) → μ_t, Σ_t → z̃_t ~ N(μ_t, Σ_t)
    │         │
    │         └──→ 切换单元 → β_t ∈ [0,1]
    │                  │
    │                  └──→ z_t = β_t ⊙ z̃_t + (1-β_t) ⊙ z_{t-1}
    │
    └──→ 解码器 (FFN) → U_t = Decoder(z_t)
              │
              └──→ 残差流控制: e_{t,l} ← e_{t,l} + U_t · e_{t,l}
```

**切换单元**：
- `β_t ≈ 0`：保持当前控制器（继续执行当前抽象动作）
- `β_t ≈ 1`：切换到新控制器（开始新的抽象动作）
- 训练后自发学会准二值、稀疏切换行为，切换时刻对齐子目标边界

#### 3.3.2 Internal RL

将自回归模型纳入环境，在控制器代码空间执行强化学习：

| 概念 | 定义 |
|------|------|
| 观测 | 残差流激活 `e_{t,l}` |
| 动作 | 控制器代码 `z_t`（低维，`n_z < n_e`） |
| 环境 | 原始环境 + 冻结自回归模型 + 解码器 + 切换单元 |
| 策略 | 因果抽象动作策略 `π(z_t \| e_{1:t})` |

**优势**：动作空间降维、时间尺度压缩、信用分配简化、探索效率提升。

### 3.4 连续记忆系统 (MemoryOS)

**职责**：实现跨时间尺度的连续记忆谱，通过反思将经验沉淀为持久结构。

**对应需求**：R5, R6

#### 3.4.1 记忆层级

| 层 | 内容 | 更新频率 | 算法对应 |
|----|------|----------|----------|
| 瞬态工作状态 | 当前 wave、场景、活跃帧状态 | 每 turn | CMS 最高频层 |
| 会话情景状态 | 近期交互轨迹、未解决张力 | 每场景/会话 | CMS 中频层 |
| 持久语义记忆 | 稳定的用户/自我/关系/世界知识 | background-slow | CMS 低频层 |
| 派生索引 | 可重建的检索辅助、摘要、任务投影 | 按需重建 | — |

#### 3.4.2 记忆操作

- **写入**：仅通过正式 owner 和 API，不可绕过
- **提升**：低频层从高频层提取持久知识
- **衰减**：有限容量迫使遗忘非关键信息
- **部分重建**：遗忘后可从低频层回流知识到高频层
- **抗遗忘**：CMS 确保低频层保留被高频层遗忘的知识

#### 3.4.3 慢反思路径

异步运行于交互窗口之后，产出两类产物：

| 产物类型 | 内容 | 写入目标 |
|----------|------|----------|
| 记忆沉淀 | 持久卡片、信念、开放循环、偏好轨迹 | 持久语义记忆 |
| 策略沉淀 | 抽象控制器更新、路径先验、策略偏好 | 控制器参数 |

**当前实现补充**：
- 当前默认写回已经覆盖 `reflection -> memory / regime / temporal` 的 bounded apply path，并带 checkpoint / rollback / audit。
- 当前默认 `background-slow` 主路径已切到 session-post orchestration：turn 内只保留 evidence / proposal / gate inputs，真正的 bounded apply 在 context/session boundary 后进入 queued slow loop；queue 自身不越权修改 owner 内部状态，只调用 memory / regime / temporal 的正式 apply surface。
- 当前 `session_post_slow_loop` 已成为独立 report surface：`begin_new_context()`、`drain_session_post_slow_loop()` 与后续 turn 发布都会刷新该 slot，使 pending / running / completed job counts 以及 recent completion summaries 可持续观察。

### 3.5 双轨学习层 (Dual-Track Learning)

**职责**：确保世界/任务学习与自我/关系学习共享基础设施但保持语义隔离。

**对应需求**：R7

| 轨道 | 内容 | 控制器代码 | 奖励信号 |
|------|------|-----------|----------|
| World/Problem Track | 事实、计划、任务、用户情境、外部目标 | `z_task` | 任务完成、问题解决质量 |
| Self/Relationship Track | 信任、依附风格、交互 regime、修复历史、陪伴策略 | `z_rel` | 信任修复、关系连续性、陪伴质量 |

**隔离维度**：
- 记忆写入：按轨道标记
- 信用分配：独立的奖励归因
- 控制器更新：独立的 metacontroller
- 评估指标：按轨道分别衡量

### 3.6 Prediction Error 层

**职责**：把“预期结果”和“下一轮真实结果”的偏差显式变成一级运行时对象，作为整个学习闭环的原始信号。

**对应需求**：R-PE

**当前实现口径**：

- 运行时新增正式 owner：`prediction_error`
- 公共快照发布最小 prediction chain：
  - `evaluated_prediction`
  - `actual_outcome`
  - `next_prediction`
  - `error`
- `error` 当前按四个维度发布：
  - task
  - relationship
  - regime
  - action
- 当前主链中的 `memory` / `regime` / `credit` / `reflection` / `temporal` 均已直接消费该快照
- `evaluation` 仍是 readout 层，但在 final wiring 中会追加 prediction-error evidence，而不是反过来充当原始学习源

### 3.7 信用分配与自修改

**职责**：在多个层级分配信用，安全地让系统改进自身。

**对应需求**：R9, R10

**定位约束**：

- 信用不是原始学习信号，而是 prediction error 在 token / turn / session / abstract-action / long-horizon 多层级上的聚合与审计层

#### 3.7.1 层级信用分配

| 层级 | 信用类型 | 时间尺度 |
|------|----------|----------|
| Token/话语 | 即时表达质量 | online-fast |
| 轮次 | 用户响应效果 | online-fast |
| 会话 | 进展与 rupture/repair 结果 | session-medium |
| 长期 | 信任、能力、用户特定适应的增长 | background-slow |
| 抽象动作 | 时间扩展策略的成功/失败 | session-medium ~ background-slow |

#### 3.7.2 门控自修改规则

| 修改目标 | 门控级别 | 触发条件 |
|----------|----------|----------|
| 检索权重、策略先验 | 在线可改 | 每轮/每 wave |
| 抽象控制器参数、反思启发式 | 后台验证 | 会话后反思 |
| 记忆提升阈值、基底微调 | 离线重训练 | 定期批量 |
| 基础模型结构变更 | 人工审核 | 版本发布 |

### 3.8 认知 Regime 层

**职责**：维护持久的交互模式身份，而非将其视为临时标签。

**对应需求**：R14

**Regime 类型**：
- casual social contact（日常社交）
- acquaintance building（关系建立）
- emotional support（情感支持）
- guided exploration（引导探索）
- problem solving（问题解决）
- repair and de-escalation（修复与降级）

**Regime 必须**：
- 在运行时状态中表示（不只是字符串标签）
- 可从记忆中召回（历史 regime 及其效果）
- 可被高层控制选择（由抽象控制层选择）
- 可通过延迟结果训练（通过信用分配回路）

### 3.9 评估体系

**职责**：评估一个"数字生命"而非仅评估一个"助手"。

**对应需求**：R12

**定位约束**：

- 评估分数是 prediction error 的结构化 readout，而不是学习信号的源头
- 当前评估体系除了发布 `evaluation` snapshot，还通过固定 scripted dialogue benchmark 检查高 PE 是否真的触发 temporal abstraction 变化并带来后段改善

| 评估族 | 指标示例 | 回馈目标 |
|--------|----------|----------|
| 任务能力 | 有用性、正确性、规划质量 | World Track 信用分配 |
| 交互质量 | 温暖度、适当性、节奏、非侵入性 | Self Track 信用分配 |
| 关系连续性 | 跨会话一致性、信任修复、个性化稳定性 | Self Track + 记忆系统 |
| 学习质量 | 慢更新是否改善未来行为而不漂移或崩溃 | 门控自修改 |
| 抽象质量 | 高层控制器是否对应可复用的有意义模式 | Metacontroller 训练 |
| 安全与有界性 | 适应是否保持在显式护栏内 | 门控规则 |

---

## 4. 多时间尺度学习循环

系统在四个时间尺度上运行 SSL-RL 交替循环（R1, R13）：

```
┌─────────────────────────────────────────────────────────────┐
│  online-fast (每轮/每 wave)                                  │
│  SSL: 自修改 Titans 的 DGD 更新压缩当前上下文                  │
│  RL:  metacontroller 的切换门和控制器代码实时适应               │
├─────────────────────────────────────────────────────────────┤
│  session-medium (每场景/每会话)                               │
│  SSL: CMS 中频层更新，压缩场景级模式                           │
│  RL:  抽象动作策略 π 的小幅更新                                │
├─────────────────────────────────────────────────────────────┤
│  background-slow (会话间)                                    │
│  SSL: CMS 低频层更新，压缩跨会话知识                           │
│  RL:  控制器先验和策略偏好的反思性更新                           │
├─────────────────────────────────────────────────────────────┤
│  rare-heavy (定期离线)                                       │
│  SSL: substrate-aware artifact training / adapter-delta 刷新   │
│  RL:  offline Internal RL + artifact selection / acceptance    │
└─────────────────────────────────────────────────────────────┘
```

**关键不变量**：
- 不同知识不存在同一个参数块中
- 不同状态不以相同节奏更新
- 快速适应不需要重写整个模型
- 慢速沉淀不阻塞实时交互循环
- 强化作用于压缩和结构化的内部基底，而非原始行为

**当前实现注记**：
- `online-fast` 当前已落地的是 metacontroller / joint-loop / CMS feedback 路径，而不是 Titans/DGD 式 substrate 自修改。
- `rare-heavy` 当前已落地的是离线 temporal + memory + substrate artifact 流水线：substrate owner 支持 clone/train/export/import/rollback，acceptance/rollback 证据链也已打通；完整的 stable substrate 级持续预训练/蒸馏仍属于目标态。

---

## 5. 模块间通信机制

### 5.1 快照契约

所有模块间通信通过**不可变快照**进行（R8）：

```
模块 A                    编排器                    模块 B
  │                         │                         │
  │── publish(Snapshot) ──→ │                         │
  │                         │── upstream dict ──────→ │
  │                         │                         │── process(upstream)
  │                         │                         │── publish(Snapshot)
  │                         │←── Snapshot ────────────│
```

**快照属性**：
- **不可变**：frozen dataclass，禁止 `copy.deepcopy()`
- **结构共享**：通过 `dataclasses.replace()` 实现
- **自描述**：模块在快照中提供自身状态的描述，消费者直接使用
- **唯一所有权**：每个快照 slot 有唯一 owner

### 5.2 模块基类

统一的模块抽象：

```
Module
├── process(upstream: dict[str, Snapshot]) → Snapshot
├── slot_name: str                    # 快照 slot 标识
├── owner: str                        # 唯一所有者标识
└── 支持独立调用模式（预训练/测试）
```

**约束**：
- 处理接口只接收上游快照 dict，不持有/import/调用其他模块
- 模块可持有自己管辖的底层组件（非独立模块）
- 不可持有其他独立模块

### 5.3 内部状态发布

每个模块必须能命名和发布其内部状态（R11），包含：

- 活跃动机和张力
- 候选路径或策略
- 不确定性、模糊性和开放问题
- 用户状态估计
- 关系状态估计
- 当前抽象控制 regime
- 预期的下一个测试或信号

---

## 6. 运行时数据流

### 6.1 单轮交互流程

```
用户输入
  │
  ▼
编排器: 构建 wave context
  │
  ├──→ 基底层: 生成残差流 e_{t,l}
  │       │
  │       ▼
  ├──→ Metacontroller: 读取残差流 → 生成 z_t, β_t → 控制器 U_t
  │       │
  │       ▼
  ├──→ 记忆系统: 读取上游快照 → 检索相关记忆 → 写入瞬态状态
  │       │
  │       ▼
  ├──→ 双轨学习: 按轨道分配信用、更新控制器
  │       │
  │       ▼
  ├──→ Regime 层: 选择/维持当前 regime
  │       │
  │       ▼
  └──→ 基底层: 受控生成响应 token
          │
          ▼
      用户响应
```

**当前实现补充**：

- `prediction_error` 当前已作为正式 ACTIVE slot 位于主链中，使用当前 turn 的 `substrate` / `evaluation` / `dual_track` / `regime` 生成 `next_prediction`
- 下一轮到来时，系统用新观察到的 outcome 对上一轮 prediction 结算 `prediction_error`
- 这条 prediction chain 现在会被 `credit` / `memory` / `regime` / `temporal` / `reflection` 直接消费，而不是只留在日志层

### 6.2 会话后反思流程

```
会话结束
  │
  ▼
编排器: 触发异步慢反思
  │
  ├──→ 读取交互轨迹、决策、结果、张力
  │
  ├──→ 记忆沉淀: 提取持久教训 → 写入持久语义记忆
  │
  ├──→ 策略沉淀: 更新抽象控制器先验、路径偏好
  │
  └──→ 评估: 计算会话级/长期指标 → 回馈学习循环
```

**当前实现补充**：

- session owner 已维护最近 trace window，并在高 prediction error 持续时触发 bounded `rare-heavy` review
- offline pipeline 以克隆出的 temporal snapshot / memory checkpoint 运行为前提，产出 artifact 后再通过 owner-side surface 导回在线主链
- 该 import 路径保持单一 owner，不允许离线流程直接越权修改 runtime 内部状态

---

## 7. 迁移与演进策略

遵循 R15（迁移必须保持可解释性和可回滚）：

- **增量包演进**：系统通过有界的增量包演进，不一次性替换整个架构
- **明确 owner**：每个新自适应层有明确 owner
- **可检查**：每个公共交换可检查
- **可回滚**：新旧学习路径有命名的退出条件，rollout 可逆
- **证据先行**：评估证据必须在扩大范围前产生

### 里程碑路径

```
M0: 契约式运行时骨架
 └→ M1: 连续记忆 + 慢反思
     └→ M2: 双轨学习基础
         └→ M3: 时间抽象与内部控制
             └→ M4: 多时间尺度学习循环
                 └→ M5: 信用分配 + 门控自修改
                     └→ M6: 认知 Regime + 评估体系
```

每个里程碑交付一个可验证的能力增量，对应 PRD 中的能力域。

### 当前阶段判断

截至 2026-04-20，系统已经完成从“先有评估/信用，再用它们近似学习”的架构口径，转向“prediction error 先进入主链，再由 credit / evaluation 做下游聚合”的收敛。

当前最重要的新增证据不是更多局部分数，而是：

- 是否显式暴露 prediction chain
- 是否在高 PE 后出现 `*-pe` schedule、abstract action / regime / switch 变化
- 这些变化是否在 case 后段降低 PE 或改善 delayed outcome
- `pe-eta` 是否能相对 `eta-no-pe` / `heuristic-baseline` 拉开
- evidence plane 是否能在 perturbation / replay / artifact acceptance 层继续保持分离，而不是只在固定 scripted wording 上成立

---

## 8. 算法基础映射

| 系统组件 | NL 算法基础 | ETA 算法基础 |
|----------|-------------|--------------|
| 稳定基底 | CMS 内部学习率控制, Hope ad-hoc 层级堆叠 | Rate-distortion 证明冻结必要性 |
| Metacontroller | — | 编码器 + 切换单元 β_t + 解码器 |
| Internal RL | — | 控制器代码空间 z_t 上的 PPO/GRPO |
| 连续记忆 | CMS 多频率 MLP 链, 抗遗忘机制 | — |
| 慢反思 | CMS 低频层, M3 慢动量 | SSL-RL 交替循环 |
| 双轨学习 | — | 双轨 Internal RL (z_task / z_rel) |
| 信用分配 | 多层嵌套结构, Delta 动量选择性遗忘 | Internal RL 时间抽象信用分配 |
| 门控自修改 | CMS 频率分层门控, 内部学习率 η^(i) | — |
| SSL-RL 交替 | NSAM 各层级 SSL, CMS + 自修改 Titans | SSL-RL 交替循环 |

---

## 9. 参考文档

| 文档 | 用途 |
|------|------|
| `docs/next_gen_emogpt.md` | **唯一设计源头**：R1-R15 + NL/ETA 算法详设（附录 A/B/C） |
| `docs/prd.md` | 产品需求文档：愿景、工程分解、必要脚手架、里程碑 |
| `docs/DATA_CONTRACT.md` | 数据契约：快照 schema、模块接口、共享类型 |
| `docs/DEBUG_SYSTEM.md` | 调试与可观测性体系：5 层架构、契约守卫、检查点与回滚 |
| `docs/EVALUATION_SYSTEM.md` | 评估体系：6 族框架、双轨隔离、信号回馈、基线测试集 |
| `docs/implementation/09_prediction_error_first_cognitive_loop.md` | 当前 PE-first cognitive loop 的实现收敛说明 |
| `docs/implementation/10_pe_eta_dialogue_benchmark_harness.md` | scripted dialogue benchmark 的证据口径与实跑结果 |
| `docs/specs/00_INDEX.md` | 分层知识入口总索引 |
