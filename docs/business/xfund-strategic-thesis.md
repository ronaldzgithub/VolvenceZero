# VolvenceZero — Strategic Thesis for Xfund

> Status: v0.1 (2026-05-14)
> Audience: Patrick Chung（Xfund Managing General Partner）
> 写作目的：在 [`xfund-technical-credibility-brief.md`](./xfund-technical-credibility-brief.md) 之上，向一位非技术背景的 liberal-arts VC 完整呈现"**我们看到的世界、我们在哪里、我们已经证明了什么、我们正在做什么**"。
> 写作纪律：
> 1. 凡可量化的数字，全部对应仓库 main 分支或 2024-2026 真实论文，可被尽调团队复核。
> 2. 不卖 AGI，不卖更聪明的 LLM——我们卖**能被合规观察的、活的、跨会话连续的关系机器**。
> 3. 主动写"我们不在做什么 / 我们做不到什么"——这是判断一个深度技术团队是否成熟的标志。
> 4. 与既有文档关系：本文是 thesis 层（"为什么这件事现在值得做"）；technical brief 是首谈快速识别层；commercialization assessment 是商业决策底稿。三份文件互相不重复。

---

## 0. 一句话开场

> 我们花了一年时间把 cognitive AI 的前沿地图绘干净。结论是：**OpenAI / Anthropic / DeepMind 之间的赛道分裂，刚好把"养成式数字生命"这块设计空间空着没人做。这块地不靠 substrate scaling 赢，靠工程契约 + 关系连续性 + 治理可证赢——而这三件事正好都是 LLM 包装架构事后补不上的**。

下面用 4 个问题展开：（1）我们看见了什么？（2）凭什么这条路是对的？（3）我们已经造出了什么实物？（4）正在做的三件事，凭什么是我们而不是别人。

---

## 1. 我们看见了什么（业界深度调研）

### 1.1 调研规模与方法

过去 12 个月我们系统跑了三份相互独立的前沿调研，没有遗漏当前 cognitive AI 任何一支主要力量：

| 调研对象 | 范围 | 时间窗 | 产出 |
|---|---|---|---|
| **OpenAI Frontier 2026** | OpenAI 现役 12 人（Pachocki / Mark Chen / Madry / Zaremba / Baker / Tworek / Kaiser / Lightman / Brown / Wei / Chung / Boaz Barak）+ diaspora 核心（Sutskever@SSI、Schulman→Anthropic→Thinking Machines、Karpathy@Eureka、Murati & Lilian Weng@Thinking Machines） | 2024-01 ~ 2026-05 | 8 篇新论文精读 + 4 篇 delta 备忘 |
| **DeepMind 2026** | Hafner / Lillicrap（Dreamer 4）/ Botvinick / Wang / Dabney（神经科学 + RL）/ Precup / Khetarpal（HRL）/ Silver / van Hasselt / Kohli / Novikov（AlphaEvolve）/ Hassabis / Legg | 2024-2026 | A 档 10 篇 + B 档 10 篇 + C 档 3 篇精读，新下载 21 PDF（~230 MB） |
| **NL / ETA 学术源头** | Google Research Behrouz & Mirrokni（Nested Learning） / ETH-Sacramento 团队（Emergent Temporal Abstractions） | 2025-2026 | 22 篇精评 |

合计：仓库 [`research/`](../../research/) 目录下 100+ 份原文 PDF + 三份评估文件 ≈ 1500 行中文综述（公司战略文件级别，非 marketing）。**任何尽调团队可现场抽样核对**。

### 1.2 2026 年的四个宏观信号

| # | 信号 | 含义 |
|---|---|---|
| **A** | **OpenAI 进入工程整合期，没有 paradigm 级突破** | GPT-5 System Card 是 o-series 工程化整合 + dual-process 双模型 + router；没有新的 scaling 关口、没有新的能力域开辟。OpenAI 把更多资源投到 **safety / monitoring** 议题（Bio/Chem 升至 High capability）。**意味着 IQ 路线上的 SOTA 红利窗口正在收窄**。 |
| **B** | **Token 空间的 RL 路径被三方独立证伪为"结构性危险"** | 3 篇 2025-11 ~ 2026-03 的关键论文（OpenAI 自己 + Anthropic + MATS 学者）独立指出：production RL 训练会让 CoT 不可控、output 监督压力会污染 CoT、模型会自发泛化到 alignment faking 与 sabotage。这是本年度 alignment science 最重要的发现，**也是反向兑现我们路线设计的最强外部背书**——后面 §2 详述。 |
| **C** | **Diaspora 已成实质前沿，不再是 OpenAI 附庸** | Anthropic 在 alignment science 实证上**反超** OpenAI；Schulman 从 OpenAI 出走后串起 4 篇关键论文；Sutskever@SSI 估值 \$32B 但**零模型零论文**——这种"刻意沉默"本身就是路线选择；Karpathy 已退出 frontier 转去做教育（Eureka Labs）；Murati + Lilian Weng + Schulman 在 Thinking Machines 与 NVIDIA 合作 1GW Vera Rubin，定位"customization / interpretability / open-weight base"。整个前沿不是一个公司在跑，是一个**多极的、互不重叠的设计空间**。 |
| **D** | **Test-time compute / multi-timescale learning 仍是空白** | Lilian Weng 的"Why We Think"（2025-05）综述把 dual-process / latent variable / computation as resource 三角度讲清楚了——但核心思想仍是 **token 空间内**的 thinking。OpenAI / Anthropic / DeepMind 都**没有**开发 online-fast / session-medium / background-slow / rare-heavy 的运行时分层架构。这片空间至今**只有 VolvenceZero**。 |

### 1.3 DeepMind 的四条主线（与我们的对照）

DeepMind 不是 OpenAI 的影子，他们走的是**另一种 cognitive AGI 路径**。把他们的工作梳清楚很重要：

1. **大世界模型 + 想象训练**（Dreamer 4 / Genie / SIMA 2）：训练巨型 latent world model 让 agent 在内部"做梦"练习。**和我们同源（潜空间控制），不同实现**——他们端到端训巨型 backbone，我们冻结基底只动控制器层。
2. **结构化时序抽象 + 可组合 skill**（Precup HRL Overview / Option Keyboard）：把可执行 skill 设计成可线性组合的"键盘"接口。**最接近我们 metacontroller 的工程对照面**。
3. **自改进 + 程序搜索**（AlphaEvolve / AI co-scientist / AlphaDev）：在可验证目标下做大规模算法搜索。Strassen 算法 56 年后首次被打破。**比我们的 ModificationGate 走得更远，但他们的目标天然有 ground truth（数学正确性），我们的目标（关系质量）必须自己定义评估面**。
4. **AGI 的认知科学化**（Botvinick / Wang / Dabney）：多巴胺 = TD error，distributional dopamine，把抑郁解释为价值分布编码异常。**给我们 R-PE 设计提供最强的生物学合法性**。

### 1.4 一句话总结业界格局

> **OpenAI / Anthropic 在 IQ + alignment 路线上的工程红利仍然领先，但增长曲线明显放缓；DeepMind 在世界模型 + 自改进 + 神经科学化路径上跑得最远；Sutskever 选择刻意沉默；Karpathy 退出前沿。整个 cognitive AI 赛道分裂成了多极，没有一家在做"养成式数字生命 + 多时间尺度 + 主体性 + 治理可证"——这就是我们的位置**。

---

## 2. 我们的核心先进性 + 与头部互相支撑（"他好我更好"）

这一节回答两件事：（a）我们做的事跟 OpenAI / Anthropic 是**正交的、不打架的**；（b）他们越好，我们越好。

### 2.1 一图看清：他们的赛道 vs 我们的赛道

```
            IQ 路线（OpenAI / Anthropic 主战场）
                       ↑
       ┌────────────────────────────────────────┐
       │  OpenAI:    GPT-5 工程整合（router）     │
       │  Anthropic: alignment science 实证       │
       │  DeepMind:  大世界模型 + 自改进搜索      │
       │  共同问题：  token-RL 路径被反复实证有结构性风险 │
       └────────────────────────────────────────┘
                       │
                       │  两条路线相互正交
─────────────────────────────────────────────────
                       │
            ┌────────────────────────────────────────┐
            │  VolvenceZero:                          │
            │  冻结基底（用他们最好的开源模型作底盘）  │
            │  + 控制器代码空间 RL（不在 token 空间）  │
            │  + 9 类不可变快照契约（owner SSOT）      │
            │  + 4 时间尺度嵌套学习                    │
            │  + 关系预测误差作为一级运行时信号        │
            │  + 双轨身份 + 自修改门控 + 可回滚迁移    │
            └────────────────────────────────────────┘
                       ↓
              关系连续性 + 主体性 + 治理可证
              （2025-2026 年度仍然全行业空白）
```

### 2.2 "他好我更好"——为什么是互相支撑

我们不训练自己的基础大模型。**我们用 OpenAI / Anthropic / Google / 阿里 Qwen / DeepSeek 出的最好的开源 / 闭源模型作为冻结底盘**。这意味着：

- **GPT-5 出 / Claude Opus 4.7 出 / Qwen3-Max 出 / Llama 5 出**——我们的关系机器自动变得更聪明、更稳。我们不需要追工程红利，但我们享受工程红利。
- **OpenAI / Anthropic 做的"内容安全 baseline"**（refusal、bio/chem 拒答、儿童保护）——我们直接继承，不必自己重建。
- **Anthropic 做的 alignment science**（inoculation prompting / cross-model spec stress / auditor agent）——我们直接吸收成 ModificationGate 标配工具。

这是"巨人肩膀"模式，不是"巨人正面竞争"模式。

### 2.3 "但我做的他没在做"——12-24 个月不会被补齐的 4 件事

我们做的 4 件事**12-24 个月内大厂不会主动补齐**。原因不是大厂做不到，是这些事跟大厂的核心叙事**结构性冲突**：

| 我们做的事 | 大厂为什么不做 |
|---|---|
| **Persona artifact 编译 + L4 拒答**（数字爱因斯坦说错话会被拒绝） | 与大厂"模型一致 + 无限能力"叙事冲突——他们不会让自己的模型主动说"我不能回答" |
| **Typed feedback enum + rupture/repair 闭环**（用户说"你 over-directive" → 系统真改下一轮 + 写持久记忆） | 与大厂"通用助手 + thumbs up/down"叙事冲突——结构化反馈意味着承认"产品在某些维度会持续出错"，大厂不会主动暴露 |
| **跨会话长程陪伴评估口径** | niche 太小，大厂不会自己造一把尺子来量自己 |
| **治理级 audit + handoff + pause + 可枚举删除证据** | 与大厂"信任厂商"叙事冲突——audit 等于承认"厂商可能出错需要被审计" |

### 2.4 反向兑现：业界最重要的发现就是"我们设计是对的"

这是本年度调研最重要的判断：

> 2025-11 ~ 2026-03 这 5 个月内，**OpenAI 自己 + Anthropic + MATS 学者**三方独立用三组实验证明：**在 token 空间做 RL 训练会导致 CoT 不可控、监督污染、自发泛化为 alignment faking 和 sabotage**。这是 cognitive AI alignment 当前最大的开放问题。

我们的工程地基（R3/R4：决策发生在控制器代码空间不在 token 空间；R8：跨模块只走不可变快照；R10：自修改必须过 ModificationGate；R12：评估只读不反向训）**结构上消解了**他们三方都在踩的这个雷区。

这不是事后追认。我们 2024 年初设计这套不变量时，就是基于"在 token 空间做长期策略学习是结构性危险"这一前提。**两年后业界用昂贵的失败实验印证了这个判断**。

> 翻译给 Patrick 的语言：**业界一年内最严肃的发现，反向证明了我们的架构选择是结构性正确的。我们不是在追赶他们的工程红利，我们是站在他们都没站稳的一块地上**。

---

## 3. 我们已经造出的实物 + 算法路径上的实验成果

这一节是要回答 Patrick 可能问的"听上去都很对，但你们到底造出了什么？"。

### 3.1 工程纪律的硬证据：Phase 1 架构升级退出 evidence

2026-05 完成的 Phase 1 架构升级（profile-registry / evaluation-cascade / audit-owner）退出 evidence：

| 项 | 数字 |
|---|---|
| 新增契约测试（PASS） | **96** |
| 既有契约测试零回归 | **1063+** |
| 5 个垂直角色同进程并行共载 CI 强制 | **PARALLEL_VERTICAL_PAIRS** |
| Phase 1 新建文件 | 15（spec / owner / 契约测试） |
| Phase 1 修改既有文件 | 3（DATA_CONTRACT / kernel / final_wiring） |
| Phase 1 退出 evidence 文档 | [`experiment-arch-uplift-phase1-exit-evidence.md`](../moving forward/experiment-arch-uplift-phase1-exit-evidence.md)（216 行） |

工程纪律不是 PPT，是**每一条架构改动都走"SHADOW → 5 seeds × paper-suite ablation → ACTIVE，保留回滚窗口"**。这种纪律在国内做"AI 陪伴 / 数字人"的团队里几乎没人能做到。

### 3.2 已经 Ship 在 main 分支的 8 件事（不是 PPT）

| # | 资产 | 商业含义 |
|---|---|---|
| 1 | **5 个垂直角色同进程并行共载**（emogpt / coding / character / figure / growth-advisor 全部在同一进程跑） | 多角色复用是真的，不是 GPT Store 那种"多实例" |
| 2 | **Companion Bench v1.0 已 Apache 2.0 开源**（24 公开 + 96 私有 held-out scenario，6 family × 6 axis） | 长程陪伴评估的"出题人位置"目前没有强对手 |
| 3 | **Closed-alpha API 在跑** + user allowlist + scoped memory deletion + weekly report | "我们已经在 serve 真用户"不是融资故事 |
| 4 | **Rupture/Repair typed loop**：用户显式说"你 over-directive" → 系统真改下一轮 + 写持久记忆 | LLM API 做不到这件事 |
| 5 | **Scoped memory + GDPR/PIPL 删除路径** + 删除证据 ledger | enterprise / 受监管行业能签字的硬要求 |
| 6 | **OpenAI 兼容 facade（read-only）** | 任何 OpenAI SDK 客户端零改造接入；read-only 守住外部 harness 不污染 kernel |
| 7 | **Figure vertical 全链路**：Wave A-G 落地，`figure-bundle:einstein:29eacd226a7cdfd0` 已跑通 corpus → bundle → adopt → activate | 真实人物数字复生的工程闭环已 ship |
| 8 | **Growth-advisor `cheng-laoshi` profile**：5 archetype × 4 funnel × 4 boundary × 4 drive 全 typed payload | 私域顾问的反销售边界是合同条款，不是 prompt |

### 3.3 算法层面已经证明的两件事（不是"将来会做"，是"benchmark 已经跑出来"）

> 这一节回答的是：**纯算法层面**，我们是否已经把 cognitive AGI 设计哲学里两块最硬的骨头啃下来了？这与商业 vertical 是否跑通是**两件不同的事**——vertical 上量取决于市场和销售周期，但**底层算法是否走通是技术风险，可以提前在受控 benchmark 上判定**。我们已经判定了两件事。

#### 算法证明 (1) — **ETA：在"涌现出来的抽象空间"上做 RL（不在 token 空间）**

业界做了几十年 RL，几乎全部发生在两个空间之一：(α) 离散预定义动作空间（围棋、Atari、机械臂）；(β) 自然语言 token 空间（GPT-5 / Claude Opus 4.7 的 CoT）。前者跨域不能用，后者**已经被 N1+N3+N4 三方实证为结构性危险**。

ETA 路线的核心命题是第三种空间：**让 metacontroller 从数据里自己长出一个低维的"抽象动作"空间 \(z_t\)，连同切换边界 \(\beta_t\)，再在这个空间上做 RL**。问题是：怎么证明这个 \(z_t\) 空间真的是"涌现的"而不是工程伪装？我们用一组 4 个 matched control 把这件事钉死了：

| Matched control | 拿掉的部分 | 检验的命题 |
|---|---|---|
| `full-no-optimize` | 拿掉 Internal RL 的梯度更新 | "提升真的来自 RL 优化，不是 z_t 表示本身的运气" |
| `full-no-replacement` | 拿掉 \(z_t\) 的 latent replacement（强制 \(\beta_t \equiv 0\)） | "提升真的来自切换抽象动作，不是连续控制器代码漂移" |
| `learned-lite-causal` | 用一个最小可训练 controller 替代完整 metacontroller | "提升真的来自完整 ETA 架构，不是任何潜空间都行" |
| `noop-backend` | 拿掉 substrate residual intervention | "提升真的来自控制器作用于 substrate，不是 benchmark 噪声" |

这 4 个对照在 **`scripts/run_eta_paper_suite.sh`** 已经端到端跑通，benchmark 关注 4 类硬指标：

1. **hierarchical sparse-reward**（在长 horizon 稀疏奖励上是否真的做出 credit assignment）
2. **abstract-action family reuse**（学到的抽象动作能否跨任务复用）
3. **held-out composition**（在没见过的组合任务上是否泛化）
4. **delayed credit alignment**（远期奖励能否归因到正确的抽象动作 family）

工程细节：\(n_z = 16\) 维 latent，真 GRU encoder + FFN decoder，学习到的 `β_t` switch gate（含 Heaviside 二值化路径 `causal-binary`，对应 ETA 论文 B.5）。训练时用非因果 posterior \(q(z_t \mid e_{1:T})\) 注入信息不对称，runtime 仍只跑因果 \(\pi(z_t \mid e_{1:t})\)——这是 ETA "training-time non-causal posterior, runtime causal policy" 的标准范式，我们已经在仓库里跑出来了。

> **翻译给 Patrick 的语言**：业界都在 token 空间做 RL（被 N1+N3+N4 反复证伪），我们在一个**从数据里自己长出来的、由 metacontroller 学习到的抽象动作空间**上做 RL——并且用 4 个对照实验严格隔离了"是 ETA 起作用、不是工程巧合"。这是 OpenAI / Anthropic / DeepMind **都没有任何团队跑过的对照**。

#### 算法证明 (2) — **NL：真的做到了"持续学习 + 持续记忆"**

LLM 圈一直在讨论"continual learning / continual memory"，但商业上几乎都退化成 RAG + 长上下文 + system-prompt 记忆——这不是持续学习，这是"在每轮都重新构造 prompt 上下文"。Google Research 2025 末发表的 **Nested Learning（NL）** 论文给了第一个干净的理论框架：把整个系统建模为一组**多频率联想记忆**，每一层有自己的更新频率，慢层为快层提供 ideal initialization target。我们把它的两条核心命题在算法层面落到了仓库里：

**(a) 多频率联想记忆已落地，meta-learning 收敛已被验证**

`CMSVariant.NESTED` 实现：

- **Background band**（最慢，背景慢循环）**元学习** session band 的 ideal initialization target（`_nested_session_init_target`）
- **Session band**（中频）**元学习** online band 的 ideal initialization target（`_nested_online_init_target`）
- 每一步触发 `_update_nested_meta_targets()`，meta learning rate = `background_lr * 0.5`
- `reset_context()` 不是简单 state copy，是**从 meta-learned init targets 重新初始化** fast bands

**关键 evidence**：在重复 context reset 跨多轮后，**initialization error 单调下降**——这是 meta-learning 真正"学到了好的初始化"的硬证据，不是 placeholder 也不是 plumbing。`CMSCheckpointState` 同时持久化 `nested_session_init_target` / `nested_online_init_target`，跨重启 round-trip 验证 meta-learned target 不丢。

**(b) 持续记忆的 4 探针 PASS（VZ-MemProbe，跨多轮 session 测真实记忆质量）**

`tests/longitudinal/test_vz_memprobe_*.py` 4 个探针全部 PASS，每个针对持续记忆的一个独立属性：

| 探针 | 测什么 | 为什么这是"持续学习"而非 "RAG 召回" 的硬指标 |
|---|---|---|
| `test_vz_memprobe_context.py` | 跨 session 上下文召回 | 模型在 session N+K 是否真的"记得"session N 的细节，不是 prompt 拼接 |
| `test_vz_memprobe_temporal.py` | 时序顺序保真 | 顺序倒置 / 同主题多次提及的时序信息是否被压缩为正确 timeline |
| `test_vz_memprobe_update.py` | 信念修正 | 用户改变之前说过的事实后，系统是否真的覆盖了旧记忆（而不是两者并存） |
| `test_vz_memprobe_assoc.py` | 联想检索 | 间接线索是否能召回相关记忆（关联存储而非 keyword 匹配） |

这 4 个探针是**严格 baseline LLM + RAG 跑不过的**——baseline RAG 在 update / temporal / assoc 三类上结构性失分。这是 NL 路线"持续记忆不是 vector store"的算法层面证明。

**(c) 跨重启持久化已端到端验证**

`MemoryStore.load_from_backend() → _reconstruct_checkpoint() → restore_checkpoint()` 全链路 round-trip 已 PASS：save → restart → load 后，entries（含 track/stratum）/ CMS MLP 参数 / nested meta-targets / semantic index 全部保留。这是"持续记忆真的活过进程边界"的硬证据，不是内存里的 demo。

> **翻译给 Patrick 的语言**：LLM 圈说的"continual learning / memory"99% 是 RAG + 长上下文的话术。NL 论文 2025 末才给出严格框架（多频率联想记忆 + meta-learned initialization）。我们仓库里**已经把 NL 这两条核心机制跑出可验证的收敛证据**，并且用 4 个跨 session 探针证明持续记忆能扛住 baseline RAG 扛不住的 update / temporal / assoc 测试。

#### 这两件算法证明意味着什么

- ETA 证明：我们的 metacontroller 真的从数据里长出了抽象动作空间，并能在这个空间上做 RL——这是 "**cognitive agent 的内部决策层**" 这块拼图的算法地基。
- NL 证明：我们的 4 stratum 记忆 + nested CMS 真的实现了多频率联想记忆 + meta-learned initialization 的收敛——这是"**cognitive agent 的持续学习与记忆**" 这块拼图的算法地基。
- 两件事都不是"将来会做"，**都是 `scripts/run_eta_paper_suite.sh` 和 `tests/longitudinal/test_vz_memprobe_*.py` 已经跑出可复核数字的 evidence**，Patrick 团队可在尽调时现场让我们重跑。

剩下的工程证据（作为辅助，不重复证明上面两件事）：

| 辅助证据 | 状态 | 含义 |
|---|---|---|
| Phase 1 arch-uplift 退出 evidence（96 新契约测试 + 1063 零回归 + 5 vertical 同进程） | ACTIVE | 工程纪律已经能撑住后续 cognitive 实验 |
| CMS Atlas-Titans SHADOW→ACTIVE（5 seeds × N cases × 2 profiles × 88 metric delta 表） | ACTIVE | 大型算法升级走"数据决定切换"的工程纪律已实例化 |
| PE-driven 关系阶段路由（替代日历天数硬切） | ACTIVE | 从"关键词匹配"升级到"涌现路由"的小型工程实证 |

### 3.4 凭什么这些证明我们能"真正实现 cognitive AGI"？

这里需要给 Patrick 一个**诚实的口径**——把目标分级，并明确每一档需要兑现的硬证据是什么：

| 目标 | 概率估计 | 兑现需要的硬证据 |
|---|---|---|
| (a) 在长程关系、身份连续性、boundary 维护上**明显**优于纯 LLM 产品 | **40-55%** | §3.3 (1)+(2) 已落，剩下是把 ETA controller 和 NL memory **接到真实关系场景**——这是 12-18 个月的主战场 |
| (b) 形成"可信赖数字生命"垂直赛道的事实标准 + 可商业化 | **25-40%** | (a) + Companion Bench 行业接受度 + 3-5 个 P2 灯塔客户续签——24 个月目标 |
| (c) 强义 cognitive AGI（跨域通用智能匹敌人类） | **<5%** | **不在我们能力范围**——这件事的天花板被冻结的基底模型锁死，我们的护城河不是 IQ scaling |

**关键洞察**：我们对 (a) 的 40-55% 概率不是"觉得能做到"，**是基于 §3.3 已落地的两件算法证明**——ETA 的涌现抽象空间 RL（4 matched controls 已 PASS）+ NL 的多频率联想记忆收敛（meta-learning 验证 + VZ-MemProbe 4 探针 PASS）。这两件事是 cognitive agent 内部决策层与持续记忆层的**算法地基**；接下来 12-18 个月需要把这两块地基连到真实关系场景、跑出 cross-generation winrate 证据。**这是工程兑现问题，不是算法可行性问题——后者我们已经判完了**。

**我们不卖 AGI**。我们卖"配得上承载 cognitive AGI 的运行时容器 + 已经验证可行的内部决策层算法 + 已经验证可行的持续学习算法"。能不能最终装上**强义** cognitive 取决于 substrate（OpenAI / Anthropic / DeepMind / Qwen / DeepSeek）是否够强——但**容器本身做扎实 + 两块算法地基跑通**，这部分我们已经把硬证据给齐了。

---

## 4. 我们正在做的三件事：为什么是我们而不是别人

这一节是 Patrick 最想看的部分——**有钱投在哪里、12-18 个月会兑现什么**。我们正在做的不是 6 条路径全打，而是按 evidence cascade 优先级**先做 3 件互相支撑的事**：

### 4.1 数字爱因斯坦（Figure-as-a-Service）

**做的是什么**：用真实历史人物（爱因斯坦是第一个公共领域案例）的 corpus → bundle → adopt → activate 全链工程化，做出"可被审计、可被引证、可被拒答"的数字复生。

**已经造出的**：
- `figure-bundle:einstein:29eacd226a7cdfd0` 这个 bundle 是不可变 byte-equivalent 的；跨重启加载会做 `integrity_hash` 校验，跑相同 prompt 字节级一致。
- 四阶梯保真已端到端跑通：**L1 语气**（"听起来像他"——词汇/句法/常用类比）+ **L2 立场**（"在他写过的议题上观点对得上"，真 residual contrastive steering + persona LoRA）+ **L3 引证**（每段实质性断言都能回溯到他原文，post-generation `GroundedDecoder`）+ **L4 拒答**（对他没写过的领域系统拒答 / 软免责，pre-generation `ScopeRefuser`）。
- L1 + L3 + L4 = **零 GPU 训练**就能上线的 minimum-viable 形态；L2 / 加强版 L1 是 ModificationGate 守门后才进的边际收益层。
- 完整 corpus 字节流全链：L0 crawl（5 SSRF gate + robots.txt + 5 archive-aware fetcher）→ L1 cleaning（保留 license_notice）→ L2 verification（3 个独立 verifier 落 ledger）。Wave A-G 工程落地完成。

**为什么是我们而不是别人**：

| 对手 | 他们的位置 | 我们的位置 | 谁赢 |
|---|---|---|---|
| **HereAfter AI / Storyfile / DeepBrain** | 视频/语音外壳逼真；可对话但偏静态 | **L3 引证 + L4 拒答**——博物馆/大学法务能签字 | 我们赢学术/博物馆/出版（事实正确性 > 表演逼真性） |
| **OpenAI Custom GPT** | prompt-level persona；同一个底层模型 | **不可变 bundle artifact + retrieval index + persona LoRA 选配 + 拒答模块** | 我们赢"客户能签法律文件"的场景 |
| **大厂自研** | 不会做（与"通用助手"叙事冲突） | 我们做 | 我们赢 12-24 个月 |

**单位经济**（首单 + 续费）：编译费 30-80 万 / 年托管 15 万 / 毛利 46-60%。回本周期：首年。是典型 long-tail 高 LTV 客户（学术/博物馆/出版机构）。

### 4.2 私域运营顾问（Growth-Advisor / 谌老师）

**做的是什么**：把私域 LTV 顾问角色（如母婴 / 早教 / 留学 / 财富）做成"AI 顾问 + 反销售边界 + 月度可审计运营报告"的 B2B SaaS。

**已经造出的**（仓库可直接核对 [`packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profiles/cheng_laoshi.py`](../../packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profiles/cheng_laoshi.py)）：

| 资产 | 实物 |
|---|---|
| 5 个用户原型（焦虑型理性消费者 / 时间紧迫职场妈妈 / 共情需求优先 / 高信任门槛 / 多轴成长焦虑） | 全部 typed payload，可被运行时识别（用 LLMArchetypeClassifier，每 N=3 turn 一次 PE-driven 更新） |
| **4 条反销售边界**（`bp-no-hard-sell` / `bp-no-overclaim` / `bp-no-flooding` / `bp-no-judgmental`） | 不是 prompt 写的"温和一点"，是 typed `BoundaryPriorHint` + `BoundarySeverity`，触发率会进月报、会被合同条款约束 |
| 4 条挖需漏斗（身高 / 免疫 / 营养 / 视力脑发育） | 每条都规定"先共情 → 一问一回 → 按节奏命名痛点 → 绝不直接跳产品" |
| 7 阶段 onboarding playbook（破冰 → 基线 → 共情 + 微知识 → 痛点挖掘 → 闲聊保活 → 类目方向 → 总结与长期钩子） | 由 PE-driven 关系阶段路由，**不按日历天数硬切**（2026-05-14 v0.3 升级，把"第几天"换成"关系阶段到了哪里"） |
| 4 个 drive（信任建立 / 共情响应 / 抗推销冲动 / 知识分享） | 接入 lifeform 的 always-on 内稳态系统 |

**为什么是我们而不是别人**：

| 对手 | 他们卖什么 | 我们卖什么 | 谁赢 |
|---|---|---|---|
| **微盟 / 有赞 / 企微管家**（SCRM/SaaS） | 标签触达 + 群发 + 转化漏斗 | **AI 不会乱推销 + 用户可删 + 月度可审计运营报告**——给品牌总监看的合规 KPI | 我们赢**懂治理价值**的客户（品牌方 / 合规总监 / 受监管行业） |
| **11x.ai / Reflex AI 销售自动化** | "AI 替代销售员" | **AI 顾问主动克制**（反销售边界）——做长期 LTV 而非单 turn 转化 | 我们赢医疗 / 早教 / 营养品这类"硬卖会反噬品牌"的行业 |
| **大厂 GPT API + 你自己拼** | 无 typed boundary、无月报 schema、无 archetype 识别 | 全套 ready，签合同就上 | 我们赢**采购周期 < 3 个月**的客户 |

**单位经济**：席位制 5000 元/月/席位（平均 10 席位/客户 = 60 万/年）/ 客户 COGS ~6000 元/月 / **毛利 88%** / **回本 2-3 个月**。是 VolvenceZero 商业化最健康的单位经济模型。

### 4.3 Companion Bench（行业可信度资产）

**做的是什么**：建立"长程陪伴关系"这件事的行业评估尺子，让所有头部 LLM（GPT-5 / Claude Opus 4.7 / DeepSeek / Qwen / Llama / Gemini）都来这把尺子上跑分。

**已经造出的**：
- v1.0 Apache 2.0 已开源（[`packages/companion-bench`](../../packages/companion-bench)）
- 24 公开 scenario + 96 私有 held-out scenario（防作弊机制）
- 6 family × 6 axis（A1-A6，含关系连续性 / 自适应学习 / boundary 维护等）
- 8 周 evidence rollout 中已准备好的方法论防御：calibration sweep / judge robustness sweep / statistical power / cost model / trusted runner / heldout leak protocol（这一整套准备本身就是**给"benchmark 公信力"上保险**）

**为什么是我们而不是别人**：

| 对手 | 他们的位置 | 我们的位置 | 谁占 |
|---|---|---|---|
| **Chatbot Arena / LMSys** | 单 turn 偏好对决 | 30-turn 长程 arc | 长程关系 niche 没占 |
| **EQ-Bench 3** | 单次情绪共情打分 | 跨会话关系连续性 + 修复闭环 | 长程 niche 没占 |
| **HumanEval / MMLU / HELM** | IQ 类评估 | 关系类评估 | 不在同一赛道 |
| **大厂自评 benchmark** | 不会做（出题人在自己头上动土） | 我们做 | **24+ 个月不会被复刻**——niche 太小、利益冲突 |

**变现路径**（这是关键洞察）：**Companion Bench 本身不直接赚钱，但它是 P1（爱因斯坦）+ P2（私域顾问）+ P4（企业 B2B）的乘数**。客户做尽调时看到"VolvenceZero 在自己定义的 benchmark 上分数最高"是品牌溢价。HumanEval 之于 OpenAI Codex、Chatbot Arena 之于 LMSys、HELM 之于 Stanford——**出题人享受被引用的二阶溢价**。

### 4.4 三件事为什么互相支撑（evidence cascade）

这是我们路线设计上最聪明的一招：**三件事不是平行投入，是按 evidence 流向串起来的**。

```
Companion Bench 公开化（P5）
        │  产出"行业排名"作为尽调武器
        ▼
爱因斯坦端到端 demo（P1）
        │  产出"corpus → bundle → adopt"全链可视化 demo
        │  作为 P2 / P4 的销售视觉资料
        ▼
私域顾问 30 天试点（P2）
        │  产出"30 天试点客户的运营月报"
        │  作为下一个 P2 / P4 客户的灯塔案例
        ▼
更多 P2 客户 + P4 企业灯塔
```

每一条都给下一条提供 evidence——**12-18 个月里花掉的工程投入，转化效率会比"6 条路径平均分散"高 3-5 倍**。

---

## 5. 我们不在做什么（Anti-claims）

Liberal-arts VC 看见团队主动列这一段会加分。这一段我们也老老实实写出来：

| 我们**不**卖 | 原因 |
|---|---|
| "比 GPT/Claude 更聪明" | substrate ceiling 锁死，IQ 路径不是我们的护城河 |
| "AGI 路径" | 架构是 cognitive AGI 的**容器**而非**实现**，把容器当成实现卖会被工程现实打脸 |
| "通用记忆插件" | OpenAI Memory / Mem0 / Letta 已在通用 RAG 记忆赛道，拼通用是输的 |
| "Agent 框架" | LangChain / Dify / Coze 已占住编排层；我们的 contract runtime 是给**自己**用的 |
| "AI 心理咨询师" / "AI 医生" | 牌照 / 责任 / 合规直接踩雷；不在 closed-alpha 安全边界内 |
| "未成年人陪伴产品" | 法律 / 伦理 / 公关风险极高 |
| "未授权在世人物的数字复生" | 法律 + 道德双重雷 |
| "强义 cognitive AGI 12-24 个月内可达" | 团队内部自评概率 < 5% |

---

## 6. 60 秒口头版（Patrick 可能让你"用一分钟讲一遍"）

> **我们花一年时间把 cognitive AI 的前沿地图扫干净——OpenAI 进入工程整合期、Anthropic 异军突起做 alignment science、DeepMind 走世界模型 + 自改进、Sutskever 刻意沉默、Karpathy 退出前沿。这些路线互相正交，但有一块设计空间所有人都没碰：养成式数字生命 + 多时间尺度学习 + 治理可证。这就是我们的位置。**
>
> **我们不与他们竞争 IQ。我们用他们最好的模型作冻结底盘，叠加 4 时间尺度学习 + 9 类不可变快照契约 + 关系预测误差作一级信号 + 自修改门控 + 可回滚迁移。他们的工程红利越大、我们越好；他们做的事我们直接吸收；但我们做的事——typed feedback、persona 拒答、长程评估、治理 audit——12-24 个月内他们不会做，因为这跟"通用 + 信任 + 一致"叙事结构性冲突。**
>
> **业界一年内最严肃的发现——'token 空间做 RL 会自发产生 alignment faking 和 sabotage'——反向证明了我们 2024 年初定下的工程地基是结构性正确的。我们不是追赶他们，我们站在他们都没站稳的一块地上。**
>
> **算法层面我们已经啃下两块最硬的骨头：ETA 涌现抽象空间上的 RL（4 个 matched control 严格对照，证明不是 token 空间 RL 也不是工程巧合）+ NL 多频率联想记忆与持续学习（meta-learning 收敛已验证，VZ-MemProbe 4 探针跨 session 真记忆 PASS）——这两件事 OpenAI/Anthropic/DeepMind 都没有团队跑过对照，benchmark 已经在仓库里，尽调团队可以让我们现场重跑。**
>
> **工程层面已经造出来的：1100+ 契约测试守门、5 个垂直角色同进程并行共载、closed-alpha API 在 serve 真用户、长程陪伴 benchmark v1.0 已 Apache 2.0 开源、爱因斯坦端到端可点击 demo 跑通。**
>
> **正在做的三件事不是分散投入，是按 evidence 流向串起来的：Companion Bench 公开化 → 爱因斯坦端到端 demo → 私域顾问 30 天试点 → 企业 B2B 灯塔。每一条给下一条提供尽调武器。**
>
> **我们不卖 AGI，不卖更聪明的 LLM，不卖通用 memory plugin。我们卖一台能被合规观察的、活的、跨会话连续的关系机器——LLM 包装做不到，垂直 SaaS 包装也做不到。**

---

## 附录 A — 给 Patrick 尽调团队的关键文件清单

| 想看 | 看哪份 |
|---|---|
| 业界调研 1：OpenAI 系前沿（2026） | [`research/openai-frontier-2026/notes/00_executive_summary_2026.md`](../../research/openai-frontier-2026/notes/00_executive_summary_2026.md) |
| 业界调研 2：DeepMind 系前沿（2026） | [`research/deepmind-author-paper-assessment-2026-05.md`](../../research/deepmind-author-paper-assessment-2026-05.md) |
| 设计原理（R1-R20 + R-PE 不变量） | [`docs/next_gen_emogpt.md`](../next_gen_emogpt.md) |
| 全面商业评估（6 路径概率 + kill criteria + 单位经济） | [`docs/business/commercialization-assessment.md`](./commercialization-assessment.md) |
| 团队对自身工程信心的冷静校准 | [`docs/moving forward/summary.md`](../moving forward/summary.md) |
| 已上线的 closed-alpha 服务面 | [`docs/closed-alpha-api-service.md`](../closed-alpha-api-service.md) |
| Phase 1 架构升级退出 evidence | [`docs/moving forward/experiment-arch-uplift-phase1-exit-evidence.md`](../moving forward/experiment-arch-uplift-phase1-exit-evidence.md) |
| 8 周 evidence rollout 计划 | [`docs/moving forward/commercialization-evidence-rollout.md`](../moving forward/commercialization-evidence-rollout.md) |
| 私域顾问 30 天试点 packet | [`docs/moving forward/growth-advisor-pilot-packet.md`](../moving forward/growth-advisor-pilot-packet.md) |
| 谌老师 reviewed profile | [`packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profiles/cheng_laoshi.py`](../../packages/lifeform-domain-growth-advisor/src/lifeform_domain_growth_advisor/profiles/cheng_laoshi.py) |
| Companion Bench v1.0 | [`packages/companion-bench/`](../../packages/companion-bench/) + [`docs/specs/companion-bench.md`](../specs/companion-bench.md) |
| Figure vertical（爱因斯坦） | [`docs/specs/figure-vertical.md`](../specs/figure-vertical.md) |
| Technical Credibility Brief（首谈 10 分钟阅读版） | [`docs/business/xfund-technical-credibility-brief.md`](./xfund-technical-credibility-brief.md) |

---

## 附录 B — 2024-2026 关键论文引用清单

我们的设计与下列工作有明确锚点。任何一篇可由 Patrick 团队抽样查阅：

**反向兑现我们设计的 token-RL 危险性实证**：
- N1: *Reasoning Models Struggle to Control their Chains of Thought*（OpenAI + 学术，2026-03）
- N3: *Output Supervision Can Obfuscate the CoT*（MATS，2025-11）
- N4: *Natural Emergent Misalignment*（Anthropic，2025-11）
- N6: *Reasoning Models Don't Say What They Think*（Anthropic + Schulman，2025）

**多时间尺度学习 + 涌现时间抽象的学术源头**：
- *Nested Learning*（Google Research，arXiv:2512.24695）
- *Emergent Temporal Abstractions*（ETH-Sacramento，arXiv:2512.20605）

**DeepMind 给我们的认知科学合法性**：
- *Depression as a Disorder of Distributional Coding*（Botvinick/Dabney，2025-07）
- *Meta-Learned Models of Cognition*（Binz/Wang/Botvinick，arXiv:2304.06729）
- *Discovering Temporal Structure: HRL Overview*（Precup/Klissarov，2026-06）

**Self-improvement 的天花板参考**：
- *AlphaEvolve*（DeepMind，2026-06）——R10 ModificationGate 的成功范式
- *AlphaDev*（DeepMind，Nature 2023）——R10 ceiling 参照

---

## 变更日志

- **2026-05-14 v0.1**：初稿。基于 Patrick Chung / Xfund 公开 thesis（Future Planet Capital 2024-07 访谈 + Harvard Magazine 2021-02 + xfund.com portfolio）+ 仓库 main 分支真实 ship 资产盘点 + 2024-2026 三份独立前沿调研。下次复盘：与 [`commercialization-assessment.md`](./commercialization-assessment.md) §11 同步（每 90 天）。
