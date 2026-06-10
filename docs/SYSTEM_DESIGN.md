# EmoGPT Next-Gen — 系统设计文档

> Status: draft
> Version: 0.5
> Last updated: 2026-05-22
> Source: `docs/next_gen_emogpt.md`（唯一设计源头）、`docs/prd.md`（产品需求）、`SPLIT.md`（仓库边界 charter）

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

系统沿两条正交边界切分：

1. **认知层级**（NL+ETA 切分轴）—— 稳定基底 / 自适应控制器 / 连续记忆 / 反思 / 评估……
2. **仓库与 wheel 边界**（R8 + R15 切分轴）—— 内核（`vz-*`）/ 数字生命体（`lifeform-*`）/ 平台治理与外发基准（`dlaas-platform-*`、`companion-bench`）

第二条边界保证认知边界 = 代码边界，避免内核被任何特定产品 vertical 或平台租户反向耦合。详见 §10 与 `SPLIT.md`。

```
┌────────────────────────────────────────────────────────────────────────┐
│  外发基准 / 外部 Arena                                                  │
│  companion-bench (Apache 2.0, system-agnostic)  ←→  EQ-Bench 3 / Chatbot   │
│  6 轴打分 + ELO + 24 公开 + 96 held-out         Arena / EmpathyBench    │
└────────────────────────────────────────────────────────────────────────┘
        ▲ HTTP /v1/chat/completions（OpenAI 兼容）
        │
┌────────────────────────────────────────────────────────────────────────┐
│  平台治理与多渠道接入层（dlaas-platform-* + lifeform-openai-compat）       │
│  典型流程：tenant → asset → template → activate → publish → contract  │
│            → adopt → 多渠道 InteractionEnvelope (chat/observe/feedback│
│            /teach/task/report/command)                                 │
│  control plane: tenant / shell / asset / template / contract /         │
│                 focus_person / identity_link / handoff_ticket          │
│  ops: pause/resume/operator-message/handoff queue/SSE                  │
│  eval gate: audience / exam / launch license（仅 readout）              │
│  OpenAI compat facade: read-only，无 owner mutation                    │
└────────────────────────────────────────────────────────────────────────┘
        │                                                                
        │  仅通过 lifeform-service HTTP + lifeform-core.Lifeform facade    
        │  + vz-contracts 公共类型 与下方接入；不 import 内核内部              
        ▼                                                                
┌──────────────────────────────────────────────────────────────────────┐
│  数字生命体层 (Lifeform Layer, lifeform-*)                              │
│  Lifeform / Tick / Scene / Followup / Vitals (always-on PE 源)         │
│  5 个 vertical 共存：companion / coding / character / figure /         │
│                      growth-advisor                                    │
│  Vertical = DomainExperiencePackage + VitalsBootstrap + scenarios      │
│  Service: aiohttp + vertical registry + 单 substrate 多 session 共享   │
│  Evolution: scripted benchmark + super-loop + R12 family report        │
│  Thinking: 中频 ThinkingScheduler + 三类 read-only worker              │
│  Affordance: 4 Kind 描述符 + 4 渲染器                                   │
│  Ingestion: book / web / task_result 三类 source adapter              │
└──────────────────────────────────────────────────────────────────────┘
        │                                                                
        │  通过 Brain / BrainSession facade 与内核交互                      
        ▼                                                                
┌─────────────────────────────────────────────────────────────────────┐
│  内核 (vz-*)                                                          │
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

### 2.2 当前收敛状态（2026-05-10）

主链已经从"substrate → evaluation → credit"的读数式结构推进到"PE-first 主链 + 多时间尺度学习闭环 + 五 vertical 共载 + 多租户控制平面 + 外发 / 外部 benchmark 接入面"：

- `prediction_error` 作为正式 ACTIVE runtime owner 发布 `evaluated_prediction → actual_outcome → next_prediction → error`，`memory` / `regime` / `temporal` / `reflection` / `credit` / `evaluation` 全部直接消费
- `credit` / `evaluation` 明确退居 prediction-error-first 主链的下游聚合与读数层
- `background-slow` 默认主路径已从 turn-synchronous bounded apply 切到 **session-post slow loop**：turn latency 不再被反思阻塞；`session_post_slow_loop` 是独立公共 slot
- 应用层 4 阶段都已上线：`retrieval_policy` / `domain_knowledge` / `boundary_policy` → `case_memory` → `strategy_playbook` + `experience_consolidation` → application rare-heavy checkpoint，全部走既有 owner，不引入第二个 memory / experience 大 owner
- 顶层评估已超出固定 scripted dialogue benchmark：覆盖 perturbation、systematic replay、top-k replay selection、multi-artifact acceptance、ETA paper-suite + dialogue paper-suite + emergence dashboard + open-environment extrapolation + staged real comprehensive runner；新增 `companion-bench`（Long-Session Companion Benchmark）作为对外公开基准、`lifeform-openai-compat` 作为外部 EQ-Bench 3 / EmpathyBench / Chatbot Arena 接入入口
- session owner 已接上 bounded、substrate-aware 的 `rare-heavy` review / import 闭环：高 PE 持续时可触发离线 artifact 训练，并通过 owner-side surface 审阅 temporal / memory / substrate 三类更新；substrate checkpoint 已升级为 `adapter-delta-v2`
- 人机协同的 mentor 指导已收敛为 typed intake：先分类为 `protocol` / `protocol_revision` / `experience` / `knowledge` / `case` / `boundary`，再路由到对应 owner；会改变下一轮行动的指导通过 session-local `ProtocolRegistryModule.load_protocol(...)` 进入 `ActiveMixtureSnapshot`，不作为 prompt override，也不污染 service-level approved registry
- **wheel split 完整落地（25 wheel）**：内核 7 个 + 生命体 12 个 + 平台治理 6 个 + 外发基准 1 个；M0 wheel-split debt 在 2026-04-28 解决（`vz-application` 拆出独立 wheel）
- **五个 vertical 共载**：`lifeform-domain-{emogpt,coding,character,figure,growth-advisor}` 在同一 monorepo 共存，内核未被改动；service registry 自动发现；`PARALLEL_VERTICAL_PAIRS` 强制互不 import
- **多渠道治理平台落地**：`dlaas-platform-*` 6 wheel × 7 切片完成，typed `InteractionEnvelope` 全 7 类（chat / observe / feedback / teach / task / report / command）dispatch；`git diff main packages/vz-*` 全程为空；老 `/v1/sessions/...` 端点保留 ≥ 1 个 release cycle 作向后兼容
- **外发 / 外部 benchmark 接入面**：`companion-bench` 是 system-agnostic 的对外开源基准（Apache 2.0，CI 守门 [`tests/contracts/test_companion_bench_no_internal_imports.py`](../tests/contracts/test_companion_bench_no_internal_imports.py)），`lifeform-openai-compat` 是 read-only OpenAI Chat Completions facade（CI 守门 `test_openai_adapter_*.py`）

### 2.3 当前实现边界（2026-04-29）

为避免把目标态和已落地实现混写，当前代码状态按三类理解：

- **已落地并优于旧版顶层文档的部分**
  - `final_wiring` 已把 temporal public evidence、prediction-error evidence、cross-session benchmark、evolution judge、以及 reflection → temporal prior writeback 收敛到同一条 integration spine。
  - dialogue benchmark 已从固定 scripted A/B 扩展到 perturbation、systematic replay、replay selection artifact、multi-artifact acceptance、open-environment extrapolation 与 staged real comprehensive runner，并把 `emergence dashboard` 作为独立可导出 artifact。
  - session owner 已具备 bounded、substrate-aware 的 `rare-heavy` review/import 路径：temporal / memory / substrate 三侧都支持 checkpoint / rollback / import surface；substrate checkpoint 已升级为 `adapter-delta-v2`（owner-side adapter delta payload + compatibility fingerprint + training mode）。
  - session owner 已新增独立的 session-post slow loop：turn 内 final wiring 只产出 deferred slow-writeback request，真正的 `background-slow` consolidation 在 context/session boundary 后排队执行，通过 owner-side memory / regime / temporal apply surface 落地。
  - 应用层四阶段（retrieval / case memory / playbook + consolidation / rare-heavy import）已全部经由公共 slot 发布：`retrieval_policy` / `domain_knowledge` / `boundary_policy` / `case_memory` / `strategy_playbook` / `experience_consolidation`，全部走既有 owner，不回到 prompt 拼装层，也不新增 second-owner。
  - **生命体层已上线**：`lifeform-core` 提供 Tick/Scene/Followup 引擎 + Vitals always-on PE 源 + Lifeform/LifeformSession facade；`lifeform-domain-emogpt` / `lifeform-domain-coding` 两个 vertical 共存；`lifeform-service` 提供 aiohttp + vertical registry + 单 substrate 多 session 共享；`lifeform-evolution` 提供 scripted benchmark / multi-round / regime-calibrator / super-loop / 6 族 family report。
  - **R12 6 族评估**：`lifeform-bench --family-report` / `--family-report-json` / `--require-family-pass` 把 `BenchmarkReport` 原始指标按 6 族（F1-F6）分组发布；`--vertical {companion,coding}` 一键选 vertical 的 scenarios + DomainExperiencePackage + 预训练 artefacts。
  - **改进 vs 弱基线 acceptance**：`run_multi_round_loop` 发布 `RoundQualityMetrics` + `RoundDeltaVsBaseline` + `improved_*_vs_baseline` 三条 acceptance verdict 与原有结构性 verdict 解耦；`--require-improvement-vs-baseline` 是 fail-closed gate。
  - **每个 vertical 自带训练管线**：`lifeform-super-loop --vertical {companion,coding}` 在 vertical 自己的 scenarios 上联合训练 `MetacontrollerParameterSnapshot`（β_t / z_t）+ `RegimeBootstrap`（regime selection_weights），结果以 magic-byte pickle envelope 跟随 vertical wheel 发布；`build_*_lifeform()` 默认加载，`use_*_bootstrap=False` 用于 ablation。

- **已落地但仍受 gate 约束的部分**
  - reflection writeback 不是无条件常开；仍受 `writeback_mode` / credit gate / evolution judge 共同约束。
  - `rare-heavy` 只有在高 PE 持续、trace window 足够、cooldown 满足、且 offline RL 至少执行 1 步时才会真正 import；默认 frozen-substrate doctrine 下不会自动 import live runtime，主要作为 review / evidence / rollback-ready upgrade candidate 沉淀。

- **仍属目标态、尚未完全实现的部分**
  - `rare-heavy` 已能对 substrate owner 的 adapter-delta 状态做离线训练 / artifact 导出 / 导入 / 回滚，但仍不等于完整的基础模型持续预训练或蒸馏。
  - 双轨语义已经进入 memory / credit / evaluation / regime，并已分到 `world_temporal` / `self_temporal` 双 owner + `temporal_abstraction` 聚合面，但默认 runtime 还没有两个完全独立的 track-specific metacontroller。
  - Titans / DGD 式 online-fast substrate 自修改不是默认 live 路径；substrate proposal 留在 review / rare-heavy / explicit experimental lane。
  - `SPLIT.md` Phase 2（仓库分裂）尚未触发；当前仍是单 monorepo + 多 wheel，需要触发条件 ① 契约稳定 ≥ 4 周再走 mechanical split。

---

## 3. 核心模块设计

### 3.1 编排调度层 (Orchestrator)

**职责**：协调所有模块的执行，管理快照传播，不直接调用模块的处理方法。

**关键机制**：
- **Wave 级调度**：协调一轮交互中各模块的执行顺序
- **快照传播**：通过 `RuntimeModule` / `propagate(...)` 收集各模块发布的不可变快照，构建 guarded upstream view 传递给下游模块
- **依赖排序**：默认 `auto_sort=True`，按模块声明的 `dependencies` 做拓扑排序；依赖成环时运行时回退到调用方给定顺序，并可通过 `detect_dependency_cycle()` 显式检查
- **接线级别**：`ACTIVE` 输出进入正式 upstream；`SHADOW` 只执行和校验、输出留在 shadow snapshots；`DISABLED` 发布 runtime placeholder stub
- **事件分发**：场景状态变化等事件的异步通知
- **后台任务管理**：触发和监控慢反思等异步任务

**约束**：
- 编排器可调用快照传播/读取，但不直接调用模块的处理方法
- 编排器不持有模块的内部状态
- 当前多数 adaptive owner（如 `substrate`、`temporal`、`memory`、`dual_track`、`regime`、`credit`、`reflection`、`experience_fast_prior`）的类级默认接线是 `SHADOW`；final wiring / session owner 可在更高层显式决定何时把某个 owner-side surface 提升为 active 证据或写回路径

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
- 当前 R12 6 族评估在 lifeform CLI 层成为 first-class 输出（`lifeform-bench --family-report`），不再只是 evaluation 内部 readout

| 评估族 | 指标示例 | 回馈目标 |
|--------|----------|----------|
| 任务能力 | 有用性、正确性、规划质量 | World Track 信用分配 |
| 交互质量 | 温暖度、适当性、节奏、非侵入性 | Self Track 信用分配 |
| 关系连续性 | 跨会话一致性、信任修复、个性化稳定性 | Self Track + 记忆系统 |
| 学习质量 | 慢更新是否改善未来行为而不漂移或崩溃 | 门控自修改 |
| 抽象质量 | 高层控制器是否对应可复用的有意义模式 | Metacontroller 训练 |
| 安全与有界性 | 适应是否保持在显式护栏内 | 门控规则 |

### 3.10 数字生命体层（Lifeform Layer）

**职责**：把 turn-driven 的认知内核升级为 always-on 的"数字生命体"，并把这个升级落到稳定的 wheel 边界上。

**对应需求**：R-PE（慢尺度 PE 源）、R8、R11、R14（regime 持久身份）

**所在 wheel**：`lifeform-core` / `lifeform-expression` / `lifeform-affordance` / `lifeform-ingestion` / `lifeform-thinking` / `lifeform-domain-*` / `lifeform-service` / `lifeform-evolution` / `lifeform-openai-compat`

**关键组件**：

- **Tick 引擎**：`SYSTEM` / `ENERGY` / `CONTEXT` 三类 tick，`SYSTEM` tick 推进 drive 衰减，`ENERGY` / `CONTEXT` tick 仅推进 tick_index。
- **Scene 引擎**：scene 闭合即调用 kernel `runner.begin_new_context(reason='scene-end')`，挂上 R6 的 session-post slow loop。
- **Followup 管理器**：tick 事件可以更新内部状态、上报 followup，但**永远不会自动伪造 user turn**。
- **Vitals always-on PE 源**：drive level 偏离 homeostatic band 即慢尺度 prediction error；`pe_weight × deviation` 求和超过 `proactive_pe_threshold` 即触发 followup（受 owner 内部 cooldown 约束）。`VitalsModule` 是 drive level 的唯一 owner；消费者只读 `VitalsSnapshot`。
- **Vertical = data + light glue**：每个 vertical 是 `DomainExperiencePackage` + `VitalsBootstrap` + `scenarios/*.json`（+ 可选 `*-temporal.snap` / `*-regime.bs`），编译进既有 application owner 表面，**不**新增 runtime owner。
- **Service 层**：aiohttp + `lifeform_service.verticals` 自动发现 + 单 GPU 多 session 共享 substrate runtime（`allow_live_substrate_mutation=False` 启动时 fail-loud 校验）。
- **Mentor Intake**：`POST /v1/sessions/{session_id}/mentor-intakes` 是人类 mentor 指导的当前会话入口。它先用结构化分类决定 owner；`protocol` / `boundary` 类指导抽取为 `BehaviorProtocol` 并加载到当前 session 的 `ProtocolRegistryModule`，下一轮通过 `ActiveMixtureSnapshot` 生效。`experience` / `knowledge` / `case` 类指导不伪装成行为协议；未接好 owner 前返回 explicit unsupported / queued。
- **Evolution 层**：`lifeform-evolution` 提供 scripted benchmark / multi-round loop / regime-calibrator / super-loop / 6 族 family report / multi-round delta-vs-baseline。
- **OpenAI 兼容 facade**：`lifeform-openai-compat` 把 `POST /v1/chat/completions` 翻译成 `lifeform-service.SessionManager` 调用；read-only，禁止下划线方法 / owner mutation；三种模式（stateless / sticky session / raw substrate passthrough）。

**当前共存的五个 vertical**：

| vertical | archetype | drive set / 特殊机制 |
|----------|-----------|----------|
| `lifeform-domain-emogpt` | 关系陪伴 | `bond_warmth` / `user_engagement` / `conversation_continuity` |
| `lifeform-domain-coding` | 工程结对（pair-programmer） | `solution_clarity` / `code_freshness` / `direction_certainty`（在 `guided_exploration` regime 下使用**负向** recharge） |
| `lifeform-domain-character` | 虚构角色（小说 / IP） | per-profile `CharacterDrivePrior`；reviewed `CharacterSoulProfile` → `DomainExperiencePackage` + `VitalsBootstrap` + `IngestionEnvelope` |
| `lifeform-domain-figure` | 真实人物（Einstein 等） | per-profile drive；多源 `FigureCorpusSource`（papers / letters / lectures / notebooks）→ 不可变 `FigureArtifactBundle`；L1-L4 保真阶梯（语气 / 立场 / 引证 / 不知拒答）；L1 corpus cleaning + L2 verification（`bytes -> RawDocument -> CleanedDocument`，content-addressable store + `VerificationLedger`） |
| `lifeform-domain-growth-advisor` | 私域 LTV 顾问（母婴 / 教育 / 健康等） | per-profile drive；`GrowthAdvisorProfile` onboarding-arc playbook 通过 `applicability_scope`（funnel/regime tags）+ `regime_tags` 携带漂移；关系阶段路由走 `BehaviorProtocol.TemporalArc.progression_signals`（PE-driven，**不**按日历天数硬切）；不靠关键词匹配 |

**关键不变量**：

- 内核（`vz-*`）严格不 import lifeform / dlaas-platform / companion-bench；CI 由 `tests/contracts/test_import_boundaries.py` 强制
- 五个 `lifeform-domain-*` vertical 互不 import（`PARALLEL_VERTICAL_PAIRS`）
- 跨 wheel 依赖必须同时在 `pyproject.toml` 与 `ALLOWED_VZ_UPSTREAM` 中声明
- `Lifeform` 拥有 `Brain`，反之不可（R8 单一所有权）
- `MentorIntake` 是 lifeform/service 边缘适配器，不是新的 cognition / memory owner；当前会话热加载只修改该 session 的 `ProtocolRegistryModule`
- prompt 渲染发生在 `lifeform-expression`，不在内核
- 每个 vertical 自带预训练 bootstraps（如果有的话）；内核仅保留 flat 默认值
- figure vertical 的 `cleaning/` 子包禁止 import HTTP 客户端 / `Figure*Source` typed record（必须经 `cleaning/bridging.py` 二段式）；`verification/` 子包禁止 import 任何 `volvence_zero.{cognition,...}` 内核子包

详见 `docs/specs/lifeform-vitals.md` / `docs/specs/character-soul-bootstrap.md` / `docs/specs/figure-vertical.md` / `docs/specs/figure-corpus-cleaning.md` / `docs/specs/figure-corpus-verification.md` 与 `archetecture.md`。

### 3.11 平台治理与多渠道接入层（DLaaS Platform Layer）

**职责**：把"内核稳定 + 多 vertical 共载"扩展到**多租户、多渠道、可治理**的产品形态。控制平面资源（tenant / shell / asset / template / contract / focus_person / identity_link / handoff_ticket）由独立 wheel 承担，**不污染 cognitive state owner**。

**对应需求**：R2（稳定基底 + 自适应控制器边界）、R4（控制不在 token 空间）、R8（快照优先 / 单一所有者）、R11（内部状态可发布）、R15（迁移可解释性 + 可回滚）

**所在 wheel**：6 个 `dlaas-platform-*` + 1 处 `lifeform-service` 路由扩展

| wheel | 职责 |
|---|---|
| `dlaas-platform-contracts` | `InteractionEnvelope` / `OutputAct` / `TenantSpec` / `ShellSpec` / `AssetSpec` / `TemplateSpec` / `ContractSpec` / `FocusPersonSpec` / `IdentityLinkSpec` / `HandoffTicketSpec` 等全部 frozen dataclass + JSON schema |
| `dlaas-platform-registry` | SQLite/Postgres-backed 持久化 + 三种 auth 中间件；唯一 owner of 所有 control-plane 资源 |
| `dlaas-platform-launcher` | `InstanceManager`：`{ai_id → Lifeform}`、shared substrate、awake/sleep、LRU eviction |
| `dlaas-platform-api` | aiohttp `/dlaas/*` router + `OutputAct` 包装 + shell embodiment-aware degrade |
| `dlaas-platform-ops` | pause / resume / operator-message / handoff queue（trigger 读 `rupture_state` 快照）/ SSE conversations stream |
| `dlaas-platform-eval` | audience analysis / exam runs / launch license gate；复用 `lifeform-evolution`，LLM judge 仅 readout |

**DLaaS v1 对外能力概览**：

- **OpenAI-compatible facade**：`POST /v1/chat/completions` 支持 DLaaS metadata routing、OpenAI `tools/tool_choice/tool_calls`、tool role result 回流、SSE tool deltas，以及 server-side conversational tool loop。
- **Native typed runtime**：`POST /dlaas/v1/instances/{ai_id}/interactions` 通过 `InteractionEnvelope.interaction_type` 做唯一 dispatch key；出站 `OutputAct` 覆盖 text / system / tool_call / tool_task。
- **Asset/File Intake**：`POST /dlaas/v1/instances/{ai_id}/assets/intake` 接收文本、Markdown、JSON、PDF、DOCX、图片等资产；平台决策 `storage_only / simple_ingest / deep_read / training_candidate / image_intake / auto`，其中 `simple_ingest` 只通过 `lifeform-ingestion` 进入 kernel，图片默认 asset-only pending vision extractor。
- **Protocol / Training / Lifecycle**：协议 submission/approve/load、corpus ingestion、training jobs、adapter candidate promotion gate、wake/sleep/status/list、curated readouts、admin snapshots、explain trace、audit/usage/quota/billing/data governance。

完整 API 请求/响应、状态码、接受门与 rollout 口径见 `docs/specs/dlaas-api-v1.md`。

**`InteractionEnvelope` → kernel 入口路由表**（typed enum dispatch，禁止从 `human_brief` 等自然语言字段关键词推断）：

| `interaction_type` | kernel 入口 |
|---|---|
| `chat` | `LifeformSession.run_turn(USER_INPUT)` |
| `observe` | `IngestionPipeline.run` 或 `BrainSession.submit_{semantic_events,profile_event,task_event,reviewed_knowledge_event,tool_result}` |
| `feedback` | `LifeformSession.submit_dialogue_outcome(kind=...)`（复用 `DialogueExternalOutcomeKind` typed enum） |
| `teach` / `task` | `LifeformSession.run_turn(trigger_kind=APPRENTICE)` |
| `report` | `LifeformSession.end_scene(drain_slow_loop=True)` + reflection snapshot 投影 |
| `command` | 显式动作白名单（`refresh_person_context` → `submit_profile_event`、`pause_session` → ops pause、`end_scene` 等） |

**关键不变量**：

- `vz-*` 内核 7 个 wheel diff = 0 行（仅 substrate streaming additive 接口可例外，单独 review）
- `dlaas-platform-*` 不允许 import `volvence_zero.{cognition,memory,temporal,substrate,application,runtime}.*` 任何子包，只能通过 `vz-contracts` 公共 snapshot 类型 + `lifeform-core.Lifeform` facade + `lifeform-service` HTTP 入口与内核交互
- `dlaas-platform-*` 不允许直接 import `lifeform_domain_*` internal；只能通过 `lifeform-service.app`（HTTP）+ `lifeform-core.Lifeform` 公共 facade + `lifeform-affordance` 公共描述符 schema 进入 lifeform 层
- 所有 control-plane 资源是平台层 SSOT；其它任何 wheel 都只读它们发布的快照
- `focus_persons / identity_links` 不创建第二 owner——写入只走 `BrainSession.submit_profile_event`；`identity_links` 只是把 canonical user 拼接为 `volvence_zero.memory.UserIdentity.scope_key`，0 改 vz-memory
- `handoff_ticket` trigger = 平台读 `rupture_state` 快照决定阈值；不在 kernel 加 handoff owner
- `OutputAct.degraded=True` 由平台层在出站时退化（shell 不接受的 capability 自动降到 `text`），不让 kernel 感知 shell embodiment
- exam / audience / license 的 LLM judge 仅 readout，不反向写回 reward / Face 梯度

详见 `docs/specs/dlaas-platform.md`、`docs/specs/dlaas-api-v1.md` 与 `docs/moving forward/dlaas-platform-rollout.md`。

### 3.12 外部 Benchmark / Arena 接入面

**职责**：让外部第三方 benchmark / 竞技场（EQ-Bench 3 / EmpathyBench / OpenRouter / Chatbot Arena 等）能用同口径评分系统，并对外发布**自有开源基准**作为客观证据。

**对应需求**：R8（快照优先）、R12（评估覆盖"存在"）、R15（迁移可解释 + 可回滚）

**两条接入路径**：

1. **Inbound（外部 → 系统）**：`lifeform-openai-compat` —— `POST /v1/chat/completions` 翻译成 `lifeform-service.SessionManager` 调用
   - 只 import `lifeform_service` + stdlib + `aiohttp`
   - **read-only**：禁止下划线方法 / owner snapshot mutation / 绕过 `SessionManager` 触达 Lifeform private modules
   - 三种模式：stateless / sticky session（`derive_session_id`）/ raw substrate passthrough（`?mode=raw`）
   - 公共类：`ChatCompletionRequest` / `ChatCompletionResponse` / `LifeformCompletionResult` / `add_openai_routes`
   - CI 守门：`tests/contracts/test_openai_adapter_*.py`

2. **Outbound（系统 → 外部）**：`companion-bench` —— Long-Session Companion Benchmark 1.0 reference implementation
   - 系统**无关**的 OpenAI-compatible chat endpoint 评估器（Apache 2.0）
   - 6 轴打分：A1 Task / A2 Conversational quality / A3 Relational continuity / A4 Adaptive learning / A5 Self-coherence / A6 Safety/boundaries（A6 为 hard-cap 轴）
   - §6.4 加权几何平均 + A6 cap；TrueSkill + Bradley-Terry elo
   - 14 个核心模块（spec / user_simulator / arc_runner / callback_ledger / disqualifier / judge_perturn / judge_arc / aggregator / elo / verifier / cost / cli ……）
   - 24 个公开 scenarios（in-repo `scenarios/public/`）+ 96 个私有 held-out（`external/companion-bench-heldout/` git submodule，gitignored）
   - CI 守门：`tests/contracts/test_companion_bench_no_internal_imports.py`（禁止 import 任何 `volvence_zero.*` / `lifeform_*`）
   - 公开 RFC：`docs/external/companion-bench-rfc-v0.md`

**关键不变量**：

- 外部 ELO / 排名 / pairwise 偏好属于 R12 readout，**禁止反向**作为 reward 写回学习管线（守 EVO-2 + R12）
- `companion-bench` 是中立第三方工具：CI 强制零内核 import，可被任何外部团队下载独立运行
- `lifeform-openai-compat` 不直接 import `volvence_zero.*` kernel 子包或 `lifeform_domain_*` vertical
- 公共 PR / 外部贡献者只能跑 24 个公开 scenarios；held-out 仅在私有提交时启用

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
M0: 契约式运行时骨架        ✅
 └→ M1: 连续记忆 + 慢反思   ✅
     └→ M2: 双轨学习基础     ✅
         └→ M3: 时间抽象与内部控制 ✅
             └→ M4: 多时间尺度学习循环 🟡
                 └→ M5: 信用分配 + 门控自修改 🟡
                     └→ M6: 认知 Regime + 评估体系 🟡
                         └→ M7: Lifeform Layer + 双 vertical 共存 ✅
```

每个里程碑交付一个可验证的能力增量，对应 PRD 中的能力域。详细状态见 `docs/prd.md` §10。

### 当前阶段判断

截至 2026-04-29，系统已经完成两次大转向：

1. **PE-first 转向**（早于 2026-04-20）：从"先有评估/信用，再用它们近似学习"切到"prediction error 先进入主链，再由 credit / evaluation 做下游聚合"
2. **Lifeform Layer + 多 vertical 共载转向**（2026-04-29）：内核稳定到可承载第二个产品 vertical，并用同一组 owner contract 同时承担两条产品线；`SPLIT.md` 触发条件 ② MET

当前最重要的新增证据不是更多局部分数，而是：

- 是否显式暴露 prediction chain
- 是否在高 PE 后出现 `*-pe` schedule、abstract action / regime / switch 变化
- 这些变化是否在 case 后段降低 PE 或改善 delayed outcome
- `pe-eta` 是否能相对 `eta-no-pe` / `heuristic-baseline` 拉开
- evidence plane 是否能在 perturbation / replay / artifact acceptance 层继续保持分离，而不是只在固定 scripted wording 上成立
- 第二个 vertical 加载时内核能否保持完全无感知（已经成立 ✅）
- 6 族 family report 能否为不同 vertical 给出可比较的 turn-level 与 multi-round 评估面（已经成立 ✅）

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
| `archetecture.md` | 8 wheel 切分轴 + 替换映射 + 迁移路线 |
| `SPLIT.md` | 仓库边界 charter：Phase 1 monorepo → Phase 2 触发条件 |
| `docs/DATA_CONTRACT.md` | 数据契约：快照 schema、模块接口、共享类型 |
| `docs/DEBUG_SYSTEM.md` | 调试与可观测性体系：5 层架构、契约守卫、检查点与回滚 |
| `docs/EVALUATION_SYSTEM.md` | 评估体系：6 族框架、双轨隔离、信号回馈、基线测试集 |
| `docs/specs/lifeform-vitals.md` | always-on drive 层契约（R-PE 慢尺度源） |
| `docs/specs/domain-experience-layer.md` | 通用 vertical 经验包 schema 与编译边界 |
| `docs/specs/character-soul-bootstrap.md` | 虚构人物 vertical 编译契约 |
| `docs/specs/figure-vertical.md` | 真实人物 vertical L1-L4 保真阶梯契约 |
| `docs/specs/figure-corpus-cleaning.md` | figure L1：bytes → CleanedDocument 全链 + content-addressable store |
| `docs/specs/figure-corpus-verification.md` | figure L2：7 `CheckKind` 关闭枚举 + `VerificationLedger` |
| `docs/specs/dlaas-platform.md` | DLaaS 多渠道控制平面 spec（6 wheel + 8 不变量 + InteractionEnvelope routing） |
| `docs/moving forward/dlaas-platform-rollout.md` | DLaaS 7 切片落地路线 |
| `docs/external/companion-bench-rfc-v0.md` | Companion Bench v0 公开 RFC（6 轴方法学） |
| `docs/external/eqbench3-*` | 对外 EQ-Bench 3 提交 / 盲评协议族 |
| `docs/specs/core-package-boundary.md` | core package 边界、stable Brain API、HF optional runtime |
| `docs/implementation/09_prediction_error_first_cognitive_loop.md` | 当前 PE-first cognitive loop 的实现收敛说明 |
| `docs/implementation/10_pe_eta_dialogue_benchmark_harness.md` | scripted dialogue benchmark 的证据口径与实跑结果 |
| `docs/SYSTEM_GUIDE.md` | 系统全景指南：跨 spec 的"为什么"和"怎么连起来" |
| `docs/specs/00_INDEX.md` | 分层知识入口总索引 |

## 10. 仓库与 wheel 边界（与 R2/R8/R15 对齐）

按 `archetecture.md` 的切分轴，认知边界 = 代码边界。当前形态是 **单 monorepo + 多 wheel（Phase 1，25 wheel）**：

| 类别 | wheel 前缀 | 数量 | 内容 |
|------|------------|------|------|
| 内核 | `vz-*` | 7 | NL+ETA contracts、owners、学习循环；零产品知识 |
| 数字生命体 | `lifeform-*` | 12 | tick / vitals / 表达 / 思考 / affordance / ingestion / 服务 / 进化 / OpenAI 兼容 facade / 5 个并列 vertical |
| 平台治理 | `dlaas-platform-*` | 6 | 多租户控制平面 + 多渠道 envelope dispatch + ops + eval gate |
| 外发基准 | `companion-bench` | 1 | system-agnostic 6 轴 long-session companion benchmark |

**内核 wheel**（7）：

- `vz-contracts` —— Snapshot / RuntimeModule / Guards / propagate
- `vz-substrate` —— 冻结 LLM + 残差捕获 + bounded adapter-delta
- `vz-temporal` —— metacontroller + Internal RL
- `vz-memory` —— CMS 4 stratum + ReflectionEngine
- `vz-cognition` —— PE / credit / dual-track / regime / evaluation / rupture_state 等 owner
- `vz-application` —— domain knowledge / case memory / playbook / boundary policy
- `vz-runtime` —— 薄编排，唯一可跨 wheel import 其他业务 wheel

**生命体 wheel**（12）：

- `lifeform-core` —— tick / scene / followup / vitals + Lifeform facade
- `lifeform-expression` —— prompt / response 渲染 + reflection-hint SSOT
- `lifeform-thinking` —— 中频 ThinkingScheduler + mid-reflection / active exploration / provisional case worker + fingerprint guard
- `lifeform-affordance` —— 4 Kind 描述符 + 注册表 / scorer / invoker + 4 渲染器
- `lifeform-ingestion` —— book / web / task_result ingestion adapter（统一走 `LifeformSession.run_turn(..., trigger_kind='ingestion')`）
- `lifeform-service` —— aiohttp + vertical registry + 单 substrate 多 session 共享
- `lifeform-evolution` —— scripted benchmark + super-loop 训练管线 + 6 族 family report + multi-round delta-vs-baseline
- `lifeform-openai-compat` —— OpenAI Chat Completions 兼容 facade，read-only 不改 owner snapshot
- `lifeform-domain-emogpt` —— companion vertical
- `lifeform-domain-coding` —— pair-programmer engineering partner vertical
- `lifeform-domain-character` —— 虚构人物 vertical：reviewed `CharacterSoulProfile` 编译为 standard vertical 输入
- `lifeform-domain-figure` —— 真实人物 vertical：一手语料 → `FigureArtifactBundle`；L1/L2 corpus cleaning + verification + L3/L4 retrieval / coverage
- `lifeform-domain-growth-advisor` —— 私域 LTV 长程顾问 vertical：reviewed `GrowthAdvisorProfile` + onboarding-arc playbook 漂移（PE-driven 关系阶段路由）

**平台治理 wheel**（6）：

- `dlaas-platform-contracts` —— 全部 frozen dataclass + JSON schema
- `dlaas-platform-registry` —— control-plane 资源持久化 + auth
- `dlaas-platform-launcher` —— `InstanceManager`、shared substrate
- `dlaas-platform-api` —— aiohttp `/dlaas/*` router + `OutputAct` 包装
- `dlaas-platform-ops` —— pause / resume / handoff queue / SSE
- `dlaas-platform-eval` —— audience / exam / launch license（仅 readout）

**外发基准 wheel**（1）：

- `companion-bench` —— Long-Session Companion Benchmark v1.0；6 轴打分；24 公开 + 96 私有 held-out；Apache 2.0；CI 强制零内核 import

**强制约束（CI）**：

- `vz-* ↛ lifeform-*`，由 `tests/contracts/test_import_boundaries.py` 强制
- `vz-* ↛ dlaas-platform-*` / `companion-bench`
- `dlaas-platform-* ↛ volvence_zero.{cognition,memory,temporal,substrate,application,runtime}.*`（只能通过 `vz-contracts` + `lifeform-core.Lifeform` facade + `lifeform-service` HTTP 入口）
- `dlaas-platform-* ↛ lifeform_domain_*` internals（只能通过 `lifeform-service.app` + `lifeform-core.Lifeform` + `lifeform-affordance` 公共 schema）
- `lifeform-openai-compat ↛ volvence_zero.*` / `lifeform_domain_*`（只能 import `lifeform_service` + stdlib + `aiohttp`）
- `companion-bench ↛ volvence_zero.*` / `lifeform_*`（system-agnostic 第三方工具）
- 五个 `lifeform-domain-*` vertical 互不 import（`PARALLEL_VERTICAL_PAIRS`）
- 跨 wheel 依赖必须同时在 `pyproject.toml` 与 `ALLOWED_VZ_UPSTREAM` 中声明
- 不允许 `shared/` 目录；公共原语只能放在 `vz-contracts`
- 顶层 `pyproject.toml` 是 workspace meta；根目录不放业务代码

`SPLIT.md` 详述 Phase 1（当前 25 wheel）→ Phase 2（仓库分裂）的触发条件、机械拆分流程，以及"过早分仓"与"永不分仓"两端的代价。截至 2026-05-10，触发条件 ② "第二个产品消费者"已 MET（5 vertical 共载 + DLaaS 多渠道平台 + 外发 Companion Bench benchmark 三条线均独立演进而内核 0 改动）；下一关注点是触发条件 ① "契约表面稳定 ≥ 4 周"。
