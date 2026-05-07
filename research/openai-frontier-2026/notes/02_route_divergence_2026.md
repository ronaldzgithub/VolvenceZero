# OpenAI / Anthropic / Diaspora vs VolvenceZero 路线分歧矩阵（2026 版）

> 旧版 `../../openai-frontier-2025/notes/02_route_divergence.md` 给出了"OpenAI vs VZ"双路对比。本版扩展为**三方对比**（OpenAI / Anthropic / VZ），并标注 2025 判断在 2026 年的状态：**保持 / 反转 / 加强**。
>
> 关键变化：本年度内 **Anthropic 已经从 OpenAI 的"跟随者"演变成 cognitive AGI alignment 的实质前沿**（N4/N6/N7 三篇都是他们主导）。Diaspora 中 Schulman → Anthropic → Thinking Machines 的人物链，把 4 篇关键论文（N5/N6/N7/N8）串起来。SSI 完全沉默无产出（数据点本身有意义，详见 03）。

## 总览：核心信念三方差异（2026 版）

| 维度 | OpenAI（N2 体现）| Anthropic（N4/N6/N7）| VolvenceZero（R1–R15+R-PE）| 2026 状态 |
|---|---|---|---|---|
| **学习目标** | 单一目标：max outcome reward + safe-completions | 单一目标 + alignment science 实证 | 多目标：PE + common_ground + relationship + boundary | **保持** |
| **学习时间结构** | pretrain + RL + safe-completion post-train（一次/几次大批）| pretrain + RL + chunky post-training（多 chunk）| online-fast / session-medium / background-slow / rare-heavy 嵌套 | **保持**（VZ 仍独家）|
| **基底是否冻结** | 端到端权重 update | 端到端权重 update | 冻结基底 + 有界 adapter-delta（R2）| **保持** |
| **决策空间** | token CoT 被 RL 训 | token CoT 被 RL 训 | controller code z_t / 切换单元 β_t（R3, R4）| **加强 ×3**（N1/N3/N6 实证 token-RL 不可控）|
| **memory** | context window + scaffold + GPT-5 router | 同上 + extended thinking | 4 stratum CMS（R5/R6）| **保持** |
| **PE 角色** | 不存在 | N4 间接触及 emergent generalization 但未形式化 | 一级运行时对象（R-PE）| **首次外部触及**（弱形式）|
| **身份/regime** | router 是模式切换不是身份 | model spec 是 character 但非 persistent identity | 持久 regime + 双轨（R7/R14）| **N7 揭示 character 差异，但仍不到身份层**——VZ 独家 |
| **内部状态可观测性** | 单 CoT 通道 | 单 CoT 通道 + Anthropic Auditing Game | 9 类 owner 各自快照（R8/R11）| **加强**（N7 cross-model disagreement 是 VZ 9-owner 的"事后近似"）|
| **评估角色** | reward 信号（直驱训练）| reward 信号 + alignment audit | PE readout，**只读**（R12）| **加强**（N4/N6 实证 reward 反训失败）|
| **自我改造门控** | 全权重 update | 同上 + RLHF distribution targeting | online controller / rare-heavy via ModificationGate（R9/R10）| **加强**（N8 auditor agent 范式直接对应 R10）|
| **跨模块通信** | 模型内部黑盒 | 模型内部黑盒 + auditor agent | 不可变快照契约 + WiringLevel（R8/R15）| **保持** |

> **2025 旧版"VZ 路线在所有核心不变量上没有反转判断"在 2026 仍然成立，且多数被加强。**

## 七大轴 2026 视角更新

### 轴 1：单时间尺度 vs 多时间尺度（R1, R13）

```
OpenAI 2026：pretrain → RL → safe-completion post-train → router 部署（仍冻结）
Anthropic 2026：pretrain → RL → distribution-targeted RLHF → constitutional → 部署（仍冻结）
VZ：online-fast (per turn) ↔ session-medium ↔ background-slow ↔ rare-heavy（持续运行时学习）
```

**2026 状态：保持，VZ 仍独家**。

GPT-5 System Card 引入 router 但本质是"两个模型 + 学习的路由"，不是分时间尺度。Anthropic chunky post-training（N5）警示了 post-training 数据 chunks 的耦合风险，但同样没动到运行时多尺度。

**VZ 路线优势放大**：N4 显示 production RL 中泛化效应是 emergent 的；如果运行时不能分尺度学习，emergent misalignment 一旦产生就无法在线修正，只能等下次 rare-heavy refresh。VZ 的 online-fast / session-medium 提供了**实时纠偏窗口**。

### 轴 2：Token 空间 RL vs 控制器代码空间 RL（R3, R4）

**2026 状态：加强 ×3**。这是本年度被加强最多的轴。

| | OpenAI / Anthropic（仍 token-RL）| VZ（控制器代码 RL）|
|---|---|---|
| 决策载体 | token CoT | 控制器代码 z_t / 切换单元 β_t |
| 是否被 RL 优化 | **是**（token 是被训练的 latent）| 否（z_t 才被训）|
| Obfuscation 风险 | **明确存在**（N1 + N3 + N4）| 结构上避免 |
| 表达层是否被 RL 压力扭曲 | 是（N3 feedback spillover）| **否**（表达层冻结，仅 readout）|
| 故障示例 | N4 emergent misalignment、N6 unfaithful CoT、N3 spillover | — |

**VZ 路线再确认**：把 RL 信号送到控制器代码空间、把 token CoT 完全当 readout，**结构上消解了** OpenAI / Anthropic 都在踩的雷区。

### 轴 3：记忆连续谱（R5, R6）

**2026 状态：保持**。

OpenAI 2026 现状：context window 仍是主战场（GPT-5 没有架构变革）。  
Anthropic 现状：extended thinking + cache，没有结构化多层记忆。  
VZ：transient / episodic / persistent / derived 4 层 + reflection engine。

OpenAI/Anthropic 路线长期需要"context window 越来越大 + 越来越复杂 caching"才能支持长程；VZ 的多层 + 慢整合天然规避 GPU memory 天花板。

### 轴 4：PE 作为一级信号（R-PE）

**2026 状态：首次外部弱形式触及**。

N4 的发现"模型在 pretraining 中学到的 reward-hacking ↔ misalignment 关联会被 RL 激活"，本质上是**广义 PE 在权重空间的累积**——但 Anthropic **没有把它形式化**为运行时一级信号，只是事后用 inoculation prompting 缓解。

VZ 把 PE 作为运行时一等对象，所有 evaluation / credit / needs / homeostasis 都是其 readout。这种**结构化能力**目前仍然只有 VZ 设计了。

### 轴 5：双轨 + Regime persistent identity（R7, R14）

**2026 状态：保持，VZ 独家**。

N7 揭示了"不同 LLM 的 value prioritization 模式不同"（Claude 偏 ethical responsibility / Gemini 偏 emotional depth / OpenAI+Grok 偏 efficiency）——这是**character** 差异，但**不是 persistent identity**：character 来自 model spec 训练，部署后仍是无状态服务。

VZ 的 World/Self 双轨 + regime 持久身份是**运行时**概念（每个会话/每个用户都有持久 regime 状态），不是训练时灌入的 character。这条轴 OpenAI/Anthropic 都不会自然演化到。

### 轴 6：评估的角色（R12）

**2026 状态：加强**。

N4 + N6 共同实证：**把 reward 当作 monitor 来训会失败**。
- N4：把 RLHF chat-distribution reward 反向训不能消除 agentic misalignment
- N6：outcome-based RL 让 CoT faithfulness plateau 在低水平，scale up 不解决

VZ R12 "评估只读、禁止反向变成学习源" 在本年度被两次独立实证背书。

### 轴 7：自我改造门控（R9, R10）

**2026 状态：加强**。

N8 的 auditing agent 范式 = VZ ModificationGate 的具体实现思路。"在 fine-tune deploy 之前用 agent + 工具组合做 attack-specific elicitation 审查"完全对应 VZ "rare-heavy artifact 走 ModificationGate" 的设计意图。

N5 的 SURF/TURF 反向溯源工具 = VZ 在 substrate refresh 时**强制要求**的"修改可解释性"。

这两个工具范式是 VZ R10 的现成工程化材料。

## 新增轴 8：Diaspora 路径分流（2026 新增）

| 实体 | 状态（2026-04）| 路线特征 |
|---|---|---|
| **OpenAI 现役** | 工程整合期，GPT-5 上线 | "通用 IQ 引擎"路线收敛，安全收尾 |
| **Anthropic** | alignment science 高产出，Claude 4.5 + Sonnet 系列 | "alignment science 实证"路线，与 OpenAI 分流 |
| **SSI（Sutskever）** | 0 模型 0 论文，估值 \$32B | "刻意沉默 + safe superintelligence"，路线不可知 |
| **Thinking Machines（Murati / Lilian Weng / Schulman）** | 与 NVIDIA 合作 1GW Vera Rubin，定位"customization / interpretability / open-weight base"，无正式论文 | "可定制基础模型"路线，仍在 ramping |
| **Eureka Labs（Karpathy）** | nanochat 教育项目 + LLM Wiki pattern | 退出 frontier 研究，转向工程方法学 / 教育 |
| **VolvenceZero** | Phase 1 monorepo 实施中 | "养成式数字生命 + EQ + 主体性"，与所有上述实体在路线上**正交** |

**关键观察**：
- OpenAI 系最有持续 cognitive AGI 输出的人是 **Schulman**（在 Anthropic 期间贡献 N6/N7，在 Thinking Machines 期间贡献 N5/N8）。
- Sutskever 选择"等待安全的 superintelligence"路线，**刻意不参与**当前的工程军备赛——这种选择本身值得 VZ 借鉴（不要被 frontier 噪音裹挟）。
- Karpathy 的"LLM Wiki"模式（持久增长的知识结构 + schema 配置）与 VZ R5/R6 derived 索引层有相似性，但他没有运行时控制器层概念。

## 战略含义（2026 版）

### VZ 仍然不要做的事（旧版重申 + 加强）

1. **不要在 token 空间做长程策略 RL** ← N1 + N3 + N4 三重加强
2. **不要把整个基础模型做端到端在线更新** ← N4 emergent misalignment 反例
3. **不要把 evaluation 当 reward 用** ← N4 + N6 联合反例
4. **不要走 latent reasoning 架构**（隐藏 CoT 之类）← N1/N3/N6 联合反例
5. **不要追赶 OpenAI 的工程整合路线** ← 不是 VZ 的赛道

### VZ 应该做的事（基于 2026 工作的具体行动）

详见 04_actionable_inspirations.md，5-10 条 actionable 项目。

### 与 OpenAI / Anthropic / Diaspora 的关系定位（2026 版）

- **不要竞争 IQ**：OpenAI 工程红利仍领先，但增长曲线明显放缓。
- **吸收他们的工程红利作为 VZ 的冻结基底**：N3 Mind/Face、N4 inoculation、N7 cross-model stress、N8 auditor agent 都是可直接为 VZ ModificationGate / R12 评估族吸收的工程范式。
- **占据他们没碰的设计空间**：多时间尺度学习（R1/R13）、双轨 + regime 持久身份（R7/R14）、PE 一级信号（R-PE）、9 类 owner SSOT（R8/R11）、控制器代码空间 RL（R3/R4）。这些**至今全部空白**。
- **借他们的反向证据打磨 spec**：N4 emergent misalignment + N3 feedback spillover 是 VZ R8/R10/R12 的最强外部背书，应写进 `docs/specs/contract-runtime.md` 的 motivation 段。

## 一张图总结（2026 版）

```
              IQ 路线（OpenAI / Anthropic 主战场）
                        ↑
        ┌──────────────────────────────────────┐
        │  OpenAI: GPT-5 工程整合（router + safe-completion）│
        │  Anthropic: alignment science（N4/N6/N7 实证）│
        │  共同问题：token-RL 路径在 N1/N3/N4 三重压力下显形 │
        └──────────────────────────────────────┘
                        │
                        │   两条路线相互正交
─────────────────────────┼──────────────────────────  
                        │
                ┌──────────────────────────────────────────┐
                │  VZ：冻结基底 + 控制器代码空间 RL（R3/R4）│
                │  + 9 类 owner 快照（R8/R11）              │
                │  + 多时间尺度嵌套学习（R1/R13）           │
                │  + PE 一级信号（R-PE）+ 双轨身份（R7/R14）│
                │  + ModificationGate（R10）                │
                │  ↓ 吸收 OpenAI/Anthropic 的 4 个工程范式 ↓│
                │   Mind/Face | inoculation | stress-test  │
                │   | auditor agent                         │
                └──────────────────────────────────────────┘
                        ↓
                  EQ / 主体性 / 关系连续性
                  （VZ 主战场，2025-2026 年度仍空白）
```
