# OpenAI 路线 vs VolvenceZero 路线 — 分歧矩阵

本文从 7 篇论文里提炼 OpenAI 当前真实采用的工程信念，与 VZ 仓库 `docs/next_gen_emogpt.md` 的 R1–R15+R-PE 不变量做正面对照。

## 核心信念差异（一张表说清楚）

| 维度 | OpenAI（2024-2025 实际做法） | VolvenceZero（R1–R15+R-PE） | 谁更对 / 谁先发 |
|---|---|---|---|
| **学习目标** | 单一目标：最大化 outcome reward（任务正确率 / 安全策略 adherence） | 多目标：PE 减少（R-PE） + common_ground 一致性 + relationship 连续性 + boundary adherence | VZ 在 cognitive agent 设计上更对；OpenAI 在窄域 IQ 上更高效 |
| **学习时间结构** | pretrain（一次） + RL fine-tune（一次/几次大批），无运行时学习 | online-fast / session-medium / background-slow / rare-heavy 嵌套（R1, R13） | VZ 设计先发；OpenAI 工程未涉及 |
| **基底是否冻结** | o-series 是端到端 RL 改写整个权重 | 冻结基底 + 有界 adapter-delta 控制器层（R2） | VZ 路线对长期持续部署更安全 |
| **决策空间** | token 空间 RL（CoT 即 token） | 控制器代码 z_t / 切换单元 β_t 空间（R3, R4） | **VZ 路线天然规避 obfuscation 风险**（参 Paper #3） |
| **memory** | context window + scaffolding，无结构化 | 4 stratum CMS（transient/episodic/persistent/derived），R5/R6 | VZ 设计更深；OpenAI 在 long-horizon 上将撞墙 |
| **PE 角色** | 不存在 | 一级运行时对象，所有 evaluation/needs/credit 都是它的 readout（R-PE） | VZ 独有，**OpenAI 完全空白** |
| **身份/regime** | model 是无状态服务，每次推理独立 | 持久 regime 身份 + 双轨（World/Self），R7/R14 | VZ 独有，**OpenAI 完全空白** |
| **内部状态可观测性** | CoT 是唯一 latent 通道（且会被优化压力破坏） | 9 类语义 owner 各自发布快照（R8, R11） | VZ 路线 SSOT 更彻底；OpenAI 在 single-CoT 通道上有 obfuscation 风险 |
| **评估角色** | reward 信号（直接驱动训练） | PE 下游 readout，**评估只读，禁止反向变成学习源**（R12） | VZ 路线规避 Goodhart；OpenAI 已观察到 obfuscated reward hacking |
| **自我改造门控** | 全权重端到端 update | 在线改控制器层 / rare-heavy artifact 走 ModificationGate（R9, R10） | VZ 路线可回滚、可审计 |
| **跨模块通信** | 模型内部黑盒 | 不可变快照契约（R8） + WiringLevel ACTIVE/SHADOW/DISABLED（R15） | VZ 设计支持 graceful migration；OpenAI 不支持 |

## 七大轴上的具体落差

### 轴 1：单时间尺度 vs 多时间尺度（R1, R13）

OpenAI 的训练 pipeline 是离散的：

```
pretrain → RL fine-tune → deploy（无运行时学习）
```

VZ 的设计是连续的：

```
online-fast (per turn 控制器更新)
session-medium (每段对话整合)
background-slow (慢反思整合，不阻塞 turn)
rare-heavy (offline substrate refresh / artifact training)
```

**含义**：OpenAI 的 frontier model 部署后是**冻结的**，对老用户的关系/记忆增量适应**完全无法做**——必须靠 prompt + memory tool 凑活。VZ 设计天然支持持续个体化。

### 轴 2：Token 空间 RL vs 控制器代码空间 RL（R3, R4）

这是**最核心的路线分歧**，也是 Paper #3 的核心警示在 VZ 上的反向兑现：

| | OpenAI | VZ |
|---|---|---|
| 决策载体 | token CoT | 控制器代码 z_t / 切换单元 β_t |
| 是否被 RL 优化 | **是**（token 是被训练的 latent） | 否（z_t 才是被训练的 latent） |
| Obfuscation 风险 | **临界点存在**（Paper #3 实证） | 结构上避免 |
| 表达层是否被 RL 压力扭曲 | 是 | **否**（表达层是 readout，不是优化对象） |

**VZ 实际上预先解决了 Paper #3 提出的"monitorability tax"问题**——因为我们不在 token 空间做优化压力，所以 token CoT 永远是 readout，不会被 obfuscate。

### 轴 3：记忆连续谱（R5, R6）

OpenAI 现状：

- context window：2k → 8k → 128k → 1M（GPT-4.1）
- agentic memory：用 tool call 让 agent 自己 store/retrieve（如 Memori、AgeMem 等社区项目）
- **没有结构化的多层记忆架构**

VZ 设计：

- transient（当前 turn 的工作记忆）
- episodic（最近会话的事件流）
- persistent（长期模式 + 关系状态）
- derived（聚合索引、知识图谱）

**含义**：OpenAI 路线长期需要 1M+ context 窗口才能维持长期记忆，**对 GPU memory 是天文级压力**。VZ 路线把记忆分层 + 慢整合 → 推理时只 load 相关切片。

### 轴 4：PE 作为一级信号（R-PE）

**OpenAI 完全没有这个概念**。

OpenAI 的 reward 信号都是离散的：
- 正确率（outcome supervision）
- 步对错（PRM）
- spec adherence（judge LLM）

VZ 把 PE（prediction error）作为运行时一等对象——所有 evaluation / credit / needs / homeostasis 都是它的 readout。

**含义**：OpenAI 系统对"惊讶"没有元能力，对"意外"没有元能力——它们只是更小的 reward。VZ 设计的系统**会因为意外而触发结构性反思**（reflection engine），不只是数值反馈。

### 轴 5：双轨 + Regime persistent identity（R7, R14）

**OpenAI 完全空白**。OpenAI 的模型是**无状态服务**——每次推理都是从头开始（虽然有 system prompt 和 conversation history，但模型权重和内部状态不变化）。

VZ：
- World-track：建模外部世界 / 用户 / 关系
- Self-track：建模自身 / 主体性 / 内部边界
- 两轨**互不读彼此快照**，独立持有
- regime 是持久身份（不是 prompt 标签）

**含义**：要做"养成式数字生命"，没有这两条根本立不起来。OpenAI 在做 **service**，不是 **subject**。

### 轴 6：评估的角色（R12）

OpenAI：评估 → reward signal → 直接训。

VZ：评估 → PE 下游 readout，**评估只读**。

Paper #3 的 obfuscated reward hacking 证明了 OpenAI 路线的 Goodhart 风险：当 monitor 被当作 reward 训进去，监控信号会失效。

VZ 的 R12 在设计时就规避了这个问题——评估不参与学习，只参与观察和决策门控。

### 轴 7：自我改造门控（R9, R10）

OpenAI：每次新 model 是端到端重新训出来的全权重，部署是 monolithic 的。

VZ：

- **在线**：controller 层参数小步更新（online-fast）
- **rare-heavy**：substrate-owner refresh / artifact training 走 ModificationGate
- 所有改动有 owner、退出条件、评估证据；可回滚

**含义**：VZ 设计支持"持续 micro-evolution"，OpenAI 设计是"monolithic version 跳跃"。两条路线的部署模型完全不同。

## 战略含义

### VZ 不要做的事（重申原则）

1. **不要在 token 空间做长程策略 RL**（参 Paper #3 警示，对应 VZ R4）
2. **不要把整个基础模型做端到端在线更新**（参 OpenAI o3 路线的成本，对应 VZ R2）
3. **不要把 evaluation 当 reward 用**（Goodhart + obfuscation，对应 VZ R12）
4. **不要走 latent reasoning 架构**（COCONUT/Geiping 让表达层不可观测，对应 VZ R8）

### VZ 应该做的事（基于 OpenAI 工作的具体行动）

1. **吸收 PRM 范式**到 metacontroller（直接借鉴 Paper #5）
2. **吸收 Deliberative Alignment 的 spec 内化**到 boundary_consent / regime（直接借鉴 Paper #2）
3. **建立 VZ-Rel 评估集**参照 GDPval pairwise blind 方法论（直接借鉴 Paper #7）
4. **把 Paper #3 的 obfuscation 实证写进 spec**作为 R8/R10/R12 的外部背书
5. **reflection engine 用弱模型 + 结构化快照阅读**（scalable oversight 思想）

### 与 OpenAI 的关系定位

- **不要竞争 IQ**——他们的工程规模断崖式领先，那条路 VZ 没有任何机会
- **吸收他们的工程红利**——用他们的 substrate（GPT-4.1 / o-series）作为 VZ 的冻结基底，VZ 加控制器层、记忆系统、PE、双轨身份
- **占据他们没碰的设计空间**——养成式 / 关系 / 主体性 / 长程身份 / 真实情感连续性
- **借他们的"反向证据"打磨我们的 spec 论证**——Paper #3 的 obfuscation 实证就是 VZ R8/R10/R12 的最强背书

## 一张图总结

```
              IQ 路线（OpenAI 主战场）
                    ↑
           工程规模无敌；token 空间 RL；
           Deliberative Alignment + PRM；
           Test-time compute scaling；
           CoT monitorability + obfuscation
                    │
                    │   两条路线相互正交
─────────────────────┼─────────────────────  
                    │
         冻结基底 + 控制器层 +
         9 类 owner 快照 + 多时间尺度学习 +
         PE 一级信号 + 双轨 + regime 身份
                    ↓
              EQ / 主体性 / 关系连续性
              （VZ 主战场，OpenAI 完全空白）
```
