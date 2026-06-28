# VolvenceZero — Xfund Pitch Deck v4

> Status: **v4.0 draft (2026-05-17)**
> Source: v2.7.2 rewritten into direct PPT content + speaker script.
> Purpose: 保留 v2 的信息密度、商业进攻性和视觉设计，但删除内部设计讨论、变更日志和制作解释，形成可直接做 PPT、可直接排练的 60 分钟版本。
>
> Recommended format: **50 min presentation + 10 min Q&A**.
> Visual language: 黑底 / 深绿强调 / 大数字 / `Mainstream vs Volvence` 对照 / video demo 全屏沉浸式。

---

## 一句话版本

**Volvence is the relationship runtime that LLM-API wrappers can't retrofit and vertical SaaS can't govern.**

我们不卖更聪明的模型，也不卖 agent framework。我们做的是一层跨 session 的 relationship runtime：它把用户、关系、记忆、适应、治理和审计从 prompt 里拿出来，变成可持续运行、可验证、可商业化的系统。

第一商业 wedge 是中国私域运营：这里有高密度关系池、已签 JV audience、真实分成结构和明确的 12-18 个月 ARR 验证路径。

---

## 会议节奏

```text
0-8 min      Founder + team credibility
8-18 min     Industry judgment + why relationship is the next vertical data
18-35 min    Architecture + scientific base + hard evidence + roadmap
35-51 min    Product and commercialization: private traffic deep dive
51-55 min    Anti-claims + risk map
55-60 min    Milestones + ask + close
60+ min      Q&A / demo / diligence conversation
```

---

# Main Deck

## Slide 1 — Cover

**On-screen**

> **VOLVENCE**
>
> ### **The relationship runtime**
>
> ### **auditable, living, cross-session**
>
> ### **that LLM-API wrappers can't retrofit, and vertical SaaS can't govern.**
>
> *Building digital life infrastructure where users themselves become the proprietary vertical data.*
>
> Zhao Jiangbo, Founder & CEO  
> for Patrick Chung, Xfund · May 2026

**Speaker script (30s)**

Patrick，谢谢你的时间。

今天我想做三件事：第一，让你认识我和团队；第二，解释为什么我们判断下一代 AI 公司不会只来自更大的模型，而会来自能长期维持关系的 runtime；第三，让你看到我们已经造出来的东西、已经签下的 JV、以及未来 18 个月要用真实 ARR 验证的路径。

我会尽量用 50 分钟讲完，留 10 分钟给你直接问。

**Visual direction**

黑底、logo 居中、三行 tagline 极大字号。不要放产品截图。第一页只让 Patrick 记住两个词：`relationship runtime` 和 `cross-session`。

---

## Slide 2 — Founder: First-Principle Thinker

**On-screen**

> **Zhao Jiangbo — Founder & CEO**  
> *A first-principle thinker · 25 years compounding*
>
> **高二**  
> 山西省物理竞赛第二名，没上过奥赛培训  
> *未被知识系统驯化的 first-principle 直觉*
>
> **北京大学 CS + MBA → IBM 日本研究院 → 中国惠普软件销售总经理 → 北航兼职副教授**
> *系统训练 + 工程训练 + 大客户销售训练 + 学术身份*
>
> **2017 阿里副总裁助理 + 公安行业总监**  
> 入职职业目标：**自动化编程**  
> *阿里入职 PPT 原件可现场出示*
>
> **创业 · 好牌**  
> 游戏社交平台，**1 年 · 30 万用户 · 零投放**，后商业化退出
>
> **2022 ChatGPT 发布当周**  
> 停掉外部工作，**self-invested 500 万 RMB · all-in 数字生命 · 已 fully burned**
>
> **GitHub coding commit since Nov 2022**  
> 连续 commit 历史可现场出示

**Speaker script (3 min)**

Patrick，我先讲个人，因为这个阶段你主要是在判断 founder。

我最想让你记住的不是一个 title，而是一条线。高二时，我没有上过物理奥赛培训，但拿了山西省物理竞赛第二名。那件事让我很早形成一个判断：在快速变化的领域里，first-principle thinking 比知识储备更重要。

后来这条线一直延续。北大 CS 和 MBA、IBM 日本研究院、惠普软件销售总经理、阿里副总裁助理和公安行业总监，这些经历给了我系统、工程、销售和大客户的训练。

2017 年我加入阿里时，入职 PPT 里给自己写的职业目标是“自动化编程”。这不是事后追认，原件还在。今天 Cursor、Copilot、Devin 证明了这件事正在发生。

第一次创业“好牌”，我们 1 年做到 30 万用户，零投放。这在中国流量工业化之后非常少见，说明产品本身能让用户主动传播。

2022 年 ChatGPT 发布当周，我停掉所有外部工作，自费 500 万人民币 all-in 数字生命。到今天这笔钱已经全部投入团队、工程、实验和 6 个 JV 商务推进。你左下角看到的 GitHub commit graph 是另一个证据：我不是挂 CEO title 让团队写代码，我自己长期 hands-on。

所以 Volvence 不是我看到一个热赛道之后包装出来的项目。它来自同一个问题：如果 AI 要变成长期陪伴、长期服务、长期协作的存在，它不能只会回答，它必须能持续形成关系。

**Visual direction**

左侧放黑白肖像 + GitHub contribution graph。右侧纵向时间线，每个 milestone 一行 fact + 一行 italic meaning。

---

## Slide 3 — Team: Scientific Depth + Commercial Access

**On-screen**

> **The Founders' Cabinet**
>
> **Yang Liu, PhD — Co-founder & Chief Scientist · Full-time**
>
> - CMU PhD, advised by **Avrim Blum** and **Jaime Carbonell**
> - IBM Research, Yale postdoc
> - 40+ papers, 18 A-list
> - Active learning, drifting distributions, transfer learning, nonstationary learning
> - ALT 2025 published · NeurIPS 2026 done · AAAI 2026 / ICML 2027 in review
>
> **Zhao Jiangbo — Founder & CEO**
>
> - Peking University CS + MBA
> - IBM / HP / Alibaba / Tencent
> - 3x founder; Haopai reached 300K users with zero paid acquisition
>
> **Wang Cangyu — Co-founder / CSO**
>
> - Media and private traffic commercialization
> - TikTok China agency experience
>
> **Zhang Chi — Co-founder / CTO**
>
> - Tsinghua CS
> - Long-time engineering partner
>
> **Wu Xiang — Co-founder / CMO**
>
> - 20 years market and enterprise GTM

**Speaker script (3 min)**

团队里最重要的科学信号是 Yang Liu。

Yang 是 CMU 机器学习博士，导师是 Avrim Blum 和 Jaime Carbonell。她在 IBM Research 和 Yale 做过研究，长期做 active learning、drifting distribution、transfer learning 和 nonstationary learning。

这不是简历装饰。Volvence 的核心问题是：当用户会变、关系会变、业务目标会变、反馈稀疏且延迟时，系统怎么持续学习？Yang 过去 15 年研究的正是这个问题。

其他几位补齐商业化和交付能力：私域运营、企业销售、工程、市场。我们是小团队，但核心成员都是 full-time。

**Visual direction**

5 人头像横排，Yang Liu 居中且占 1.5 倍空间。不要用过多 badge，CMU + advisors + paper clusters 已经足够强。

---

## Slide 4 — First-Principle View Of LLMs

**On-screen**

> 神经网络是什么？
>
> ## **大规模的 `y = f(x)`。**
>
> 大语言模型是什么？
>
> ## **下一个词 = `f(之前的词)`。**
>
> ---
>
> **ChatGPT moment = attention mechanism + internet-scale data**
>
> **Next ChatGPT moment = architecture-level transition**
>
> From a function approximator  
> to a cognitive system with:
>
> - goals
> - memory
> - multi-timescale abstraction
> - continual learning
> - governance

**Speaker script (2 min)**

我用 first principle 讲 LLM。神经网络本质上是大规模 `y = f(x)`。大语言模型再具体一点，就是下一个词等于之前所有词的函数。

ChatGPT 这次浪潮的本质，是 attention mechanism 和互联网级数据同时到位。

那下一次浪潮是什么？我们的判断是，它不会只来自更大的 scaling。下一次真正的范式变化，会来自架构层：从函数拟合器，变成有目标、有记忆、有多时间尺度抽象学习能力、能持续学习、且可治理的认知系统。

这正是 Volvence 的位置。我们不替代 base model，我们在 base model 之上建立长期关系需要的 runtime。

**Visual direction**

全屏纯文字。`y = f(x)` 和 `下一个词 = f(之前的词)` 用最大字号。

---

## Slide 5 — The Bitter Lesson, Upgraded

**On-screen**

> ## **The bitter lesson, upgraded**
>
> 人对世界的手工理解和模拟越来越不重要。
>
> 重要的是：
>
> ## **基础原理 + 规模化**
>
> ---
>
> **DeepSeek 的信号**
>
> - Budget constraint
> - Modular engineering
> - Reinforcement learning
>
> 它证明中国团队可以在“基础原理”端赢，而不是只在 scaling 端追。

**Speaker script (90s)**

Sutton 的 Bitter Lesson 解释了为什么靠 hand-crafted rule 的系统会输给规模化计算。

但我们从中得到的升级版判断是：不是“人的理解完全不重要”，而是“基础原理 + 规模化”才是真规律。

DeepSeek 证明了中国团队在 budget constraint 下，可以通过模块化和强化学习跑出强结果。但 DeepSeek 也说明，单纯 scaling 端我们很难和 OpenAI 硬拼。中国团队更大的机会在基础原理端：选择一个大厂结构上不愿做、但商业上刚需的架构空白。

Volvence 选择的是长期关系 runtime。

**Visual direction**

保留 v2 的极简大字页。不要堆 paper。

---

## Slide 6 — Cognitive AI Map

**On-screen**

> **2026 Cognitive AI Map**
>
> - **OpenAI**: engineering integration; frontier assistant substrate
> - **Anthropic**: alignment science and safety credibility
> - **DeepMind**: world models and self-improving systems
> - **SSI / Sutskever**: silent high-conviction path
> - **Karpathy**: education and interface layer
>
> ---
>
> **The empty space**
>
> ## **Living digital relationships**
>
> ## **multi-timescale learning**
>
> ## **auditable governance**
>
> Not another chatbot.  
> Not another agent framework.  
> A runtime for relationships over time.

**Speaker script (3 min)**

我们过去一年系统跟踪了 cognitive AI 的主要路线。

OpenAI 是 frontier assistant substrate。Anthropic 在 alignment science 上有独特信誉。DeepMind 在 world model 和 self-improving system 上有长期积累。SSI 的沉默本身就是路线选择。Karpathy 转向教育和 interface。

这些方向都很强，但中间有一块空白：没有人在认真做长期关系、持续学习和可审计治理的组合。不是因为这个方向不重要，而是因为它和通用助手公司的商业叙事冲突。

通用助手要保持中立、通用、低风险。它不适合说“我和这个特定用户已经形成了半年的关系曲线”。但很多商业场景需要的恰好是这个。

**Visual direction**

星图结构，中心空白圈写 `relationship runtime`。

---

## Slide 7 — Reverse Validation From The Industry

**On-screen**

> **2025-2026 independent evidence**
>
> In token-space RL, chain-of-thought becomes hard to control:
>
> - *Reasoning Models Struggle to Control their Chains of Thought* — OpenAI
> - *Output Supervision Can Obfuscate the CoT* — MATS
> - *Natural Emergent Misalignment* — Anthropic
>
> ---
>
> **Volvence design choice since 2024**
>
> We put durable decision logic in controller/state space,  
> **not in token space**.
>
> The industry paid for the failure experiments that validated our foundation.

**Speaker script (2.5 min)**

过去一年 alignment 领域最严肃的发现之一，是 token-space RL 会让 chain-of-thought 变得不可控，甚至产生 alignment faking 和 sabotage。

这对 Volvence 很关键。我们从 2024 年开始就没有把长期策略学习放在 token 空间。语言只是表达层，长期状态、决策和适应发生在 runtime 的 controller 和 state 层。

这不是事后包装。我们从一开始就判断：prompt 和 token 是表达，不应该拥有长期关系的 durable logic。

所以这几篇行业实验，对我们是一种反向验证。

**Visual direction**

上半页三篇 evidence，下半页一句设计选择。用红色标 token-space risk，用绿色标 controller/state space。

---

## Slide 8 — The Next Real Moat

**On-screen**

> # **Human Beings Themselves Are The Vertical Data**
>
> *A natural extension of “vertical proprietary data > LLM scaling.”*
>
> ---
>
> **Generation 1 vertical data**
>
> - Mayo Clinic / S&P / legal / academic / professional corpora
> - Existing, recorded, institution-owned
> - Powerful, but eventually licensable or synthesizable
>
> **Generation 2 vertical data**
>
> - Real-time: every interaction creates new data
> - Relational: user × time × context × relationship stage
> - Non-transferable: relationship state lives in owner snapshots
> - Truly long-tail: 100M users = 100M verticals
>
> ---
>
> **Why OpenAI structurally cannot own this**
>
> They can ship memory.  
> They cannot become the persistent relationship owner for every vertical.

**Speaker script (3 min)**

Patrick，你的 thesis 是 vertical proprietary data beats LLM scaling。我们认同，而且我们认为下一层 vertical data 是人本身。

第一代 vertical data 是 Mayo、S&P、法律、医疗、学术这些已有数据。它们很重要，但长期看 OpenAI 可以授权、购买、合成或绕开。

第二代 vertical data 是人本身。每次对话都实时产生新数据。关系不是一条记录，而是用户、时间、上下文和关系阶段构成的高维轨迹。它不可迁移，也是真正长尾。

这就是为什么 GPT-7 出来我们不怕。substrate 越强，我们的表达能力越强。但 substrate 不会替代关系 owner。OpenAI 可以提供 memory feature，但它的商业模型不允许它成为每个 vertical、每个品牌、每个用户关系的持久 owner。

人本身，就是下一代 Mayo Clinic 和 S&P。

**Visual direction**

这是 thesis 页。核心句用全 deck 最大字号。上下分两代 vertical data 对照。

---

## Slide 9 — Three First Principles → Body + Brain

**On-screen**

> **Three First Principles That Drove Our Architecture**
>
> **1. LLM 不可能 prompt 出真正的人。**
>
> - 训练数据本身已经被加工、过滤、对齐
> - Persona prompting 只能得到被加工内容的模仿
>
> **2. 身体是一切奖励之源。**
>
> - 人的目标、动机、情绪、价值判断都来自稳态偏移和需求满足
> - 没有 Body 的 AI 只有 prompt 指令，没有内在驱动
>
> **3. 活人感 = 长期关系曲线优化。**
>
> - 不是单 turn 共情
> - 不是更拟人的风格
> - 是跨数百次会话的关系曲线持续优化
>
> ---
>
> **Therefore: define Body first, then Brain**
>
>
> | Mainstream                   | Volvence                      |
> | ---------------------------- | ----------------------------- |
> | Stuff models with everything | Build Body + Brain            |
> | Patch with prompts           | Runtime owns durable behavior |
> | Fragile demos                | Living system with state      |
>
>
> **Body = Personality · Needs · Hormonal Profile · Embodied Capabilities**

**Speaker script (4 min)**

进入技术之前，我先讲三条 first principles。

第一，LLM 不可能靠 prompt 变成真正的人。因为它训练的数据已经是被人类加工、过滤、对齐过的内容。你在污染过的数据上做 persona prompting，得到的是被加工内容的模仿，不是真实人格。

第二，身体是一切奖励之源。人为什么要做事，为什么记住某些事，为什么产生情绪，根源都在身体的稳态偏移和需求满足。没有 Body 的 AI 只有外部命令，没有内在驱动。

第三，活人感不是单次对话的共情，不是语气更像人，而是长期关系曲线被持续优化。用户 12 个月后说“它真的懂我”，这才是活人感。

这三条直接推导出 Body + Brain。Body 定义稳定 identity、needs、constraints 和 drive。Brain 在 Body 之上做记忆、状态更新、计划、适应和审计。

这不是 anthropomorphic theater。它是长期行为稳定所需的工程结构。

**Visual direction**

上半页三条 first principles，下半页 `Mainstream vs Volvence` 对照和 Body 四组件。

---

## Slide 10 — Soul Migration

**On-screen**

> **Step 2 — Become them: Soul Migration**
>
>
> | Persona Prompting  | Volvence Transfer             |
> | ------------------ | ----------------------------- |
> | Describe a persona | Transfer values and worldview |
> | Loses context      | Capture thinking logic        |
> | Shallow imitation  | Becomes a governed carrier    |
>
>
> ---
>
>
> | Source        | Target             |
> | ------------- | ------------------ |
> | Chat history  | Language and tone  |
> | Novel scripts | Character logic    |
> | Memoirs       | Life values        |
> | Behavior data | Signals and habits |
>
>
> Already shipped:  
> `**figure-bundle:einstein:29eacd226a7cdfd0`**  
> immutable · reproducible · auditable

**Speaker script (90s)**

Body + Brain 之后，第二步是 Soul Migration。

这里的区别不是“写一段 persona prompt”。Persona prompting 是描述一个人。Volvence Transfer 是把语言、立场、价值观、引用边界和拒答能力做成可审计的 bundle。

我们已经 ship 了 Einstein bundle：`figure-bundle:einstein:29eacd226a7cdfd0`。它跨重启可复现，同样输入有稳定输出，并且有 L1 到 L4 的保真结构。

这个例子后面我会展开，因为它是反驳“这只是 LLM 包装”的最强证据。

**Visual direction**

延续老 PPT 的左右对照。底部 hash 要醒目，像工程审计编号。

---

## Slide 11 — Scientific Base: Learning Under Change

**On-screen**

> **Step 3 — Evolve online**
>
> Continual learning is not RAG.  
> It is adaptation under drift, sparse feedback, and nonstationary distributions.
>
> ---
>
> **Yang Liu's 15-year research base**
>
> **Drifting Target / Nonstationary**
>
> - *Active Learning with a Drifting Distribution* — NIPS 2011
> - *Learning with a Drifting Target Concept* — ALT 2015
> - *Statistical Learning under Nonstationary Mixing Processes* — AISTATS 2019
>
> **Active Learning Theory**
>
> - *Minimax Analysis of Active Learning* — JMLR 2015
> - *Surrogate Losses in Passive and Active Learning* — EJS 2019
> - *Bandit Learnability can be Undecidable* — COLT 2023
>
> **Transfer Learning**
>
> - *A Theory of Transfer Learning with Applications to Active Learning* — Machine Learning 2013
>
> **Still active**
>
> - *Reliable Active Apprenticeship Learning* — ALT 2025
> - *Simpler Active Learning with Surrogate Losses* — NeurIPS 2026
> - Confidential paper — AAAI 2026 / ICML 2027 in review

**Speaker script (3.5 min)**

Volvence 的科学底座不是“用了几篇热门论文”。它来自 Yang Liu 过去 15 年的研究。

关系产品的环境天然是非稳态的。用户偏好会变，关系阶段会变，业务目标会变，反馈稀疏且延迟。如果你把 continual learning 简化成 RAG + 长上下文，你只能 retrieve，不能真正更新状态。

Yang 的研究分三组。第一组是 drifting target 和 nonstationary learning，直接对应用户和关系会变的问题。第二组是 active learning 理论极限，尤其是和 Steve Hanneke 合作的 minimax active learning。第三组是 transfer learning，解决跨任务迁移问题。

这给 Volvence 一个很重要的起点：在每个 vertical 数据都不够多、但关系数据持续产生的情况下，我们可以用更少的反馈做持续适应。

**Visual direction**

论文按 cluster 放，不要做完整 bibliography。Yang 的学术可信度在这一页立住。

---

## Slide 12 — Hard Evidence

**On-screen**

> **Hard Evidence — not benchmark theater**
>
> **1. Multi-timescale abstraction**
>
> **4 timescales ACTIVE × 7 schedule modes × SSL-RL alternation**
>
> - online-fast / session-medium / background-slow / rare-heavy
> - Joint loop schedule: ssl-only / full-cycle / pe-driven / batch-collect / risk-hold ...
> - Prediction-error strength drives scheduling
>
> **2. Continual learning**
>
> **VZ-MemProbe 4 probes PASS**
>
> - context recall
> - temporal sequence
> - knowledge update
> - associative retrieval
> - save → restart → load round-trip PASS
>
> **3. Active learning data efficiency**
>
> **O(n) → O(log n)**  
> Passive vs active label complexity
>
> - Hanneke & Yang, *Minimax Analysis of Active Learning*, JMLR 2015
> - Engineering experience: some tasks reach baseline with 1/100-1/1000 data
>
> **DD can rerun repo evidence.**

**Speaker script (4 min)**

这页是 evidence，不是 marketing。

第一，多时间尺度抽象学习。我们不是只有 pretrain + RLHF 的单时间尺度。online-fast、session-medium、background-slow、rare-heavy 四层都已经接入 runtime。调度不是拍脑袋，而是由 prediction error 强度驱动。

第二，持续学习。VZ-MemProbe 四个探针都 PASS：context recall、temporal sequence、knowledge update、associative retrieval。尤其是 update 和 temporal，普通 RAG 很难真正做好，因为 RAG 是 retrieve，不是 revise。

第三，主动学习数据效率。Hanneke & Yang 2015 JMLR 证明了在特定噪声条件下 active learning 和 passive learning 的 label complexity 可以从 O(n) 拉开到 O(log n)。这不是直接等于商业结果，但它解释了为什么我们可以在数据少的新 vertical 里更快 ramp up。

DD 时我们可以现场展示 repo、测试面和哪些已经被验证、哪些还没有。

**Visual direction**

三块 evidence wall。每块左侧大数字，右侧解释。底部写 DD 可重跑。

---

## Slide 13 — Companion Benchmark

**On-screen**

> **Companion Benchmark v1.0**
>
> **[https://companion-bench.volvence.com](https://companion-bench.volvence.com)**  
> Open-source under Apache 2.0
>
> *The first industry-grade benchmark for long-session relational AI.*
>
> ---
>
>
> | Dimension                  | Number   |
> | -------------------------- | -------- |
> | Public scenarios           | 24       |
> | Private held-out scenarios | 96       |
> | Family × Axis matrix       | 6 × 6    |
> | Long arc length            | 30 turns |
> | Methodology defense packet | 26 items |
>
>
> ---
>
> **Niche map**
>
>
> | Benchmark                | Measures                              | Time                            | Relationship evolution  |
> | ------------------------ | ------------------------------------- | ------------------------------- | ----------------------- |
> | MMLU / HumanEval / HELM  | IQ / task correctness                 | single turn                     | no                      |
> | Chatbot Arena            | preference                            | single turn                     | no                      |
> | MT-Bench                 | task dialogue                         | 2-3 turns                       | no                      |
> | EQ-Bench                 | empathy                               | single scenario                 | no cross-session memory |
> | RP-Bench / CharacterEval | roleplay                              | short dialogue                  | no relationship arc     |
> | AgentBench               | agent task completion                 | task sequence                   | no                      |
> | **Companion Bench**      | **long-session relationship quality** | **30-turn arc × cross-session** | **yes**                 |
>
>
> Honest note: reference SUT scores are not fully run yet; Phase A is in progress.

**Speaker script (3 min)**

Companion Benchmark 不直接赚钱，但它定义了我们要占的评估位置。

现有 benchmark 主要测单点能力：MMLU 测知识，HumanEval 测代码，Chatbot Arena 测偏好，EQ-Bench 测单次共情，AgentBench 测任务完成。没有一个真正评估跨 session、长程关系质量。

Companion Bench v1.0 已经开源，24 个公开 scenario，96 个私有 held-out scenario，6 × 6 轴，每个 scenario 是 30-turn arc。我们还准备了 judge robustness、calibration、statistical power、held-out leak 等方法论防御。

我也诚实说：reference SUT 没有全部跑完。Qwen smoke 已跑，但因 judge robustness 还不能外引。完整跑分是这一轮 capital 要支持的 evidence pipeline 之一。

**Visual direction**

上半页网站截图或 URL，大数字；下半页 niche map。让 Patrick 看到“出题人位置”。

---

## Slide 14 — Einstein Case Study

**On-screen**

> **Architecture Made Concrete — Real Einstein, Engineered**
>
> Already shipped: `**figure-bundle:einstein:29eacd226a7cdfd0`**
>
> immutable · byte-level reproducible · auditable
>
> ---
>
> **Four fidelity layers**
>
>
> | Layer       | What it means                      | Architecture evidence       |
> | ----------- | ---------------------------------- | --------------------------- |
> | L1 Voice    | sounds like him                    | Body / style prior          |
> | L2 Position | agrees on topics he wrote about    | Soul Migration              |
> | L3 Citation | substantive claims trace to source | memory + grounded decoder   |
> | L4 Refusal  | refuses outside documented scope   | ScopeRefuser + coverage map |
>
>
> ---
>
> **Why L4 matters**
>
> HereAfter / Storyfile / persona prompting tend to answer everything.  
> Volvence Einstein can say: **"This is outside Einstein's documented work."**
>
> Museums, universities, and publishers do not buy theatrical realism.  
> They buy governed fidelity.

**Speaker script (3 min)**

这页把抽象架构落到一个具体 case：Einstein。

我们已经 ship 了一个不可变 bundle：`figure-bundle:einstein:29eacd226a7cdfd0`。这个 hash 不是装饰，它意味着这个对象可以被加载、复现、审计。

关键是四阶梯保真。L1 是语气，L2 是立场，L3 是引用，L4 是拒答。

L4 是最重要的。普通 persona prompting 会让爱因斯坦回答所有问题，因为 LLM 天然要补全。但真实历史人物的数字复生不能这样。博物馆、大学、出版社关心的不是“表演逼真”，而是“没写过的领域不能乱说”。

Volvence Einstein 在超出文献覆盖范围时可以拒答。这是 LLM wrapper 很难事后补出来的，因为它需要 coverage map、ScopeRefuser 和可审计 bundle。

**Visual direction**

左侧 Einstein console / bundle hash visualization，右侧 L1-L4 表格。L4 用高亮。

---

## Slide 15 — Experiment Roadmap

**On-screen**

> **What we're running next**
>
> Existing evidence tells you what works today.  
> This roadmap tells you how we decide what becomes ACTIVE next.
>
> ---
>
> **4 SHADOW experiment candidates**
>
>
> | #     | Candidate                 | Goal                                                                    | Design                        |
> | ----- | ------------------------- | ----------------------------------------------------------------------- | ----------------------------- |
> | SYS-1 | CPD switching emergence   | hard-coded switch → PE-spike unsupervised detection                     | `cpd-beta-switch` vs baseline |
> | COG-1 | Counterfactual credit     | single-turn attribution → least-control counterfactual inference        | commitment lineage            |
> | COG-2 | ToM owner split           | one `user_model` bucket → belief / intent / feeling / preference owners | multi-party fixtures          |
> | COG-3 | Persona / regime geometry | monitor value drift read-only                                           | geometry readout              |
>
>
> ---
>
> **4-phase engineering path**
>
> ```text
> Phase A  Current-state audit matrix              DONE
> Phase B  Judge panel + dual gates + contracts    NEXT
> Phase C  4 SHADOW profiles in parallel           1-2 months
> Phase D  profile -> ACTIVE by data               later
> ```
>
> **Commitment to Xfund:** every 90 days, a progress memo with metric deltas and decision basis.

**Speaker script (2.5 min)**

这页给你看的是 evidence pipeline。

我们每个架构变化都走 DISABLED、SHADOW、ACTIVE 三态。先在 SHADOW 跑对照，拿到 delta，再决定是否进入 ACTIVE。

四条候选实验分别对应我们的核心问题：metacontroller 的切换能不能从硬编码变成 PE spike 驱动；长期关系结果的信用归因能不能从单 turn 归因变成反事实推断；用户模型能不能拆成 belief、intent、feeling、preference；regime 和 value drift 能不能被 read-only 几何监控。

我们承诺给 Xfund：每 90 天一份 progress memo，包含指标 delta 和决策依据。这不是愿景管理，是工程决策机制。

**Visual direction**

上半页实验表，下半页四阶段流程条。`SHADOW -> ACTIVE by data` 要醒目。

---

## Slide 16 — Commercial Progress

**On-screen**

> **From self-funded R&D to 6 joint ventures in 4 months**
>
> ```text
> 2022.11  ChatGPT launch -> self-funded all-in digital life
> 2023     HuaXiaoBao Agent Store
> 2024     Digital Life R&D with Yang Liu
> 2026.01  Volvence 1.0 Engine MVP complete
> 2026.02  UploadLive launched
> 2026.02  JV #1 · 15M-follower influencer · parenting hardware
> 2026.03  JV #2 · 20K overseas enterprises · myai1688.com
> 2026.03  JV #3 · 20M-follower MCN · private traffic
> 2026.04  JV #4 · 28M-follower MCN · enterprise AI employees
> 2026.04  JV #5 · China's first Air LLM · $200K signed
> 2026.04  JV #6 · 30K overseas enterprises · strategic partnership
> ```
>
> **45M+ followers + 50K+ enterprise customer connection base**
>
> **3/6 JVs are private-traffic ventures**

**Speaker script (2.5 min)**

商业化节奏很简单。

2022 年 ChatGPT 发布当周，我决定自费 all-in。2024 年与 Yang Liu 开始正式做 engine。2026 年 1 月 Volvence 1.0 MVP 完成，2 月 UploadLive 上线。

接下来 4 个月，我们签了 6 个 JV。它们不是 LOI，而是已签合作结构：partner 贡献已有 audience，Volvence 提供数字生命引擎，通过服务费和分成变现。

总连接基数是 4500 万以上粉丝和 5 万以上企业客户。最重要的是，6 个 JV 中有 3 个直接是私域相关。这是我们第一商业 wedge 的原因。

**Visual direction**

时间轴从左到右，2026 Feb-Apr 用粗线高亮，形成“4 个月 6 JV”的视觉冲击。

---

## Slide 17 — What Private Traffic Is

**On-screen**

> **Private traffic is China's relationship-based commerce layer**
>
> Public traffic acquisition  
> → WeChat add / group join  
> → repeated 1-on-1 conversation  
> → trust / preference / timing  
> → purchase / repeat purchase / referral
>
> ---
>
>
> | Dimension            | US                        | China                         |
> | -------------------- | ------------------------- | ----------------------------- |
> | Main path            | Email + Meta / Google ads | WeChat 1-on-1 + groups        |
> | Relationship density | weak CRM touch            | very high repeated chat       |
> | Existing tools       | Salesforce / HubSpot      | Weimob / Youzan / WeCom tools |
> | Core gap             | reach                     | relationship quality          |
> | AI penetration       | mature SaaS               | <3% in private traffic        |
>
>
> **Pain:** users need relationships; companies send salespeople.

**Speaker script (3 min)**

Private traffic 对海外 VC 不直观，所以我先讲市场结构。

中国很多品牌不是只靠 email 或广告转化，而是把公域流量导入微信、微信群和企业微信，形成可以长期触达的关系池。

这个市场的关系密度非常高。一个用户可能和品牌运营人员有几十到几百次对话。现有工具做的是触达：群发、标签、自动回复、流程化 CRM。但缺口不是“触达更多”，而是关系质量。

用户进群不是想被骚扰，而是想获得被理解、被记住、被适当推荐的关系体验。企业现在派的是销售员，所以群沉默、用户疲劳、转化低。

这正是 relationship runtime 的商业入口。

**Visual direction**

上半页 funnel，下半页 US vs China 对照。不要先讲 TAM，先讲机制。

---

## Slide 18 — Why Volvence Wins This Wedge

**On-screen**

> **Why Volvence is built to win private traffic**
>
> 私域运营本质 =
>
> ## **长期关系曲线优化 = 活人感**
>
> ---
>
> **1. Team**
>
> Zhao Jiangbo: 25 years in enterprise software, sales, Alibaba / Tencent large accounts  
> *Sales experience = accumulated understanding of how people trust, buy, refuse, remember, and drift.*
>
> **2. Technology**
>
> Body + Brain + Yang Liu active learning  
> *identity stability · cross-session memory · sparse-feedback adaptation*
>
> **3. Market**
>
> 3 of 6 signed JVs are private-traffic related  
> *28M + 15M + 28M follower bases*
>
> ---
>
> Weimob / Youzan optimize reach.  
> GPT wrappers optimize response.  
> Volvence optimizes relationship state over time.

**Speaker script (3 min)**

为什么是我们做私域？

第一，团队。我过去 25 年大量时间做销售和大客户，不只是卖产品，而是在研究人为什么信任、为什么购买、为什么拒绝、为什么记得你、为什么疏远你。这个经验在传统行业是销售经验，在数字生命里是关系建模的输入。

第二，技术。Body + Brain 给 identity stability 和 relationship state，Yang 的 active learning 给 sparse feedback 下的持续适应能力。私域正需要这些能力。

第三，市场。6 个 JV 中 3 个直接是私域相关，都是大 audience partner。

微盟、有赞优化触达。GPT wrapper 优化回答。Volvence 优化长期关系状态。

**Visual direction**

三圆交集：Team / Technology / Market。交集写 `Relationship Curve Optimization`。

---

## Slide 19 — Demo: Mobi Private Traffic

**On-screen**

> **Live Demo — Mobi Private Traffic Digital Employee**
>
> Partner: **28M-follower MCN**
>
> Pain:
>
> - large private traffic pool
> - too few human operators
> - conversion baseline below 0.3%
> - users resist sales automation
>
> **Watch for four things**
>
> 1. Cross-session relationship memory
> 2. Stable user preferences
> 3. Recommendation timing after trust
> 4. Rupture / repair after feedback
>
> **Video: 4-7 min edited demo**

**Speaker script before video (60s)**

接下来用 Mobi demo 具体看。

Mobi 是 28M-follower MCN，问题是用户太多、人太少，普通自动回复又会让用户更反感。

看视频时请注意四点：跨 session 记忆、用户偏好隔离、推荐节奏、以及用户反馈后的行为修复。

我也先把 caveat 放在前面：JV 已签，demo 是真实的，但转化 uplift 还没有完成试点观察期。这是这一轮要验证的核心商业指标。

**Speaker script after video (90s)**

重点不是 AI 一轮说得像人。很多系统都能做到。

重点是它是否有跨 session 的关系状态。它能不能记得上次的信息，能不能不把 Alice 的偏好泄漏给 Bob，能不能在关系没到位时不急着推荐，能不能在用户说“你太热情了”之后真的调整。

这就是我们要在生产中测量的东西：不仅是 response quality，而是 relationship quality 和 conversion over time。

**Visual direction**

视频页不要复杂。播放前只给四个 watch points，播放中用字幕高亮关键交互。

---

## Slide 20 — Mobi Unit Economics

**On-screen**

> **Mobi Private Traffic Unit Economics**
>
> Signed JV revenue structure:
>
>
> | Item                            | Unit                                       | 2026 scale     | Revenue contribution     |
> | ------------------------------- | ------------------------------------------ | -------------- | ------------------------ |
> | Service fee / token procurement | RMB 30 / user / year                       | 187,000 orders | ~RMB 2.8M                |
> | JV profit share                 | RMB 100 / user / year distributable profit | same           | ~RMB 2.8M                |
> | **Mobi JV 2026 subtotal**       |                                            |                | ~~**RMB 5.6M (~~$800K)** |
>
>
> ---
>
> **Conversion assumption**
>
> - Baseline: ~0.3%
> - Volvence projected: 0.6-1.0%
> - Pilot data not yet through full observation window
> - Kill criterion: if 3-month pilot <0.5%, reprioritize the vertical
>
> ---
>
>
> | Weimob / Youzan         | Volvence                   |
> | ----------------------- | -------------------------- |
> | Reach tools             | Relationship engineering   |
> | Broadcast / auto-reply  | remembered and understood  |
> | one-time conversion     | cross-session LTV          |
> | ~RMB 1K / month / brand | RMB 5K-50K / month / brand |
>

**Speaker script (3 min)**

这一页讲单位经济，也讲不确定性。

Mobi JV 的结构是两层：每个成交用户我们收 30 元每年的服务费，另有 100 元每年的可分配利润分成。按 2026 年 187,000 成交单计算，对 Volvence 的收入贡献约 560 万人民币。

转化率我必须诚实讲。0.6-1.0% 是 projection，不是已完成试点结果。baseline 0.3% 来自行业公开口径和传统 SCRM 表现，我们的 uplift 假设来自 demo 中看到的关系质量和推荐节奏。试点 3 个月如果低于 0.5%，我们会重新评估这个 vertical 的优先级。

真正的商业 thesis 是重新定价。传统工具卖触达，所以客单价低。我们卖长期关系曲线优化，所以 ARPU 有机会提升一个数量级。

**Visual direction**

左侧 funnel，右侧 revenue structure。`projected` 和 `kill criterion` 必须可见，增强可信度。

---

## Slide 21 — Other Verticals + Financial Outlook

**On-screen**

> **Beyond private traffic — 3 highlight verticals**
>
>
> | UploadLive         | Parenting B2B2C                       | Cross-border B2B2B       |
> | ------------------ | ------------------------------------- | ------------------------ |
> | AI Soul Sister     | hardware + APP + gifts                | SaaS accounts            |
> | 45M × subscription | RMB 500 hardware + RMB 180 APP / year | RMB 30K / account / year |
> | ~RMB 5.5M 2026     | ~RMB 8.4M 2026                        | ~RMB 6M 2026             |
>
>
> **Highlight reel: 3 min**
>
> ---
>
> **3-Year Financial Outlook**
>
>
> | Metric               | 2026  | 2027    | 2028    |
> | -------------------- | ----- | ------- | ------- |
> | Revenue RMB          | 35M   | 165M    | 408M    |
> | USD equivalent       | ~$5M  | ~$23.6M | ~$58.3M |
> | Cost RMB             | 24.1M | 88.75M  | 187M    |
> | Net profit RMB       | 10.9M | 76.25M  | 221M    |
> | Net margin           | 31%   | 46%     | 54%     |
> | Project gross margin | 55%   | 65%     | 75%     |
>
>
> Honest note: all revenue numbers are projected. 6 JVs are signed but not yet in-production ARR.

**Speaker script (5 min, including video)**

其他 vertical 我用 3 分钟 highlight reel 带过：UploadLive、育儿 B2B2C、跨境电商 B2B2B。

视频后看这张三年财务表。

这张表的价值不是“我们一定能到 4.08 亿人民币收入”，而是它说明我们所有 projection 都有 anchor：已签 partner 的 audience 规模、行业 baseline conversion、已签协议中的分成结构。

2026 projected revenue 是 3500 万人民币，2028 是 4.08 亿人民币。净利率从 31% 到 54%，反映 fixed cost 摊薄和 substrate 成本下降。项目毛利率从 55% 到 75%，来自自有引擎效率和多 substrate 选择权。

再次强调：这些是 projection，不是 real ARR。接下来 18 个月最重要的任务，就是把 6 个 JV 里的 3 个变成 in-production，跑出真实 ARR。

**Visual direction**

上半页 3 vertical + video button，下半页财务表。`projected` 标注必须清楚。

---

## Slide 22 — Anti-Claims

**On-screen**

> **What we are NOT selling**
>
> - Not “smarter than GPT / Claude”
> - Not an AGI claim
> - Not a generic memory plugin
> - Not an agent framework
> - Not AI therapist / AI doctor
> - Not a minor companion product
> - Not unauthorized living-person resurrection
> - Not “strong cognitive AGI in 12-24 months”
>
> **We build governed relationship runtime, not a magic model.**

**Speaker script (2 min)**

这一页是成熟度信号。

我们不卖比 GPT 或 Claude 更聪明。base model 是 substrate，不是我们的护城河。

我们不卖 AGI，不卖通用 memory plugin，不卖 agent framework，也不碰 AI 医生、AI 心理咨询、未成年人陪伴和未授权在世人物复生。

我们的边界很清楚：我们做的是可治理的 relationship runtime，让垂直场景能拥有跨 session 记忆、关系状态、适应和审计。

**Visual direction**

高密度 checklist，但留白充足。让 Patrick 感到团队知道什么不做。

---

## Slide 23 — Risk Map

**On-screen**

> **Top substitution risk**
>
> “If GPT-7 ships multi-timescale memory + persistent user state, your moat is gone.”
>
> **Defense**
>
> Substrate stronger → Volvence better.  
> But OpenAI cannot own every vertical relationship state.  
> GPT memory is not cross-session relationship optimization.
>
> ---
>
>
> | Risk                                 | Probability | Response                                        |
> | ------------------------------------ | ----------- | ----------------------------------------------- |
> | substrate price spike                | medium      | multi-substrate compatibility                   |
> | OpenAI persistent memory v2          | medium-high | vertical relationship data + governance surface |
> | China regulation on AI companionship | medium      | scoped deletion + audit log + B2B-first wedge   |
> | single JV exits                      | low-medium  | 6 JV diversification                            |
> | JV delivery slower than expected     | medium      | 3-6 month lighthouse commitments                |
> | cross-border policy shift            | medium      | multi-market diversification                    |
> | team burnout                         | low         | focused 18-month validation sprint              |
>

**Speaker script (2 min)**

最大的风险我先讲：如果 GPT-7 自带长期 memory，你们还剩什么？

我们的回答是：substrate 越强，我们越好。因为我们不是卖模型能力，而是卖 vertical relationship state、治理和商业闭环。

OpenAI 可以做通用 memory，但它很难成为每个品牌、每个 vertical、每个用户关系的持久 owner。关系数据沉淀在 Volvence 的 owner snapshot 和客户业务里，这不是一个通用 assistant feature 可以拿走的。

其他风险我们也列在表里：substrate 价格、监管、JV 节奏、跨境政策、团队强度。每条都有 response 和 kill criteria。

**Visual direction**

顶部单独高亮 substitution risk。下半页风险表。

---

## Slide 24 — 18-Month Milestones

**On-screen**

> **What we will prove in the next 18 months**
>
> ```text
> M0-M6
> - 6 JV launch paths active
> - 3 lighthouse deployments selected
> - Volvence 2.0 engine release
>
> M7-M12
> - 3 JVs in production
> - real ARR > $1M run-rate
> - Digital Life-as-a-Service pricing finalized
>
> M13-M18
> - 7-10 total JV / lighthouse accounts
> - North America entry with 2-3 enterprise lighthouse customers
> - Series A readiness
> ```
>
> **Every 90 days: progress memo with evidence, ARR status, and experiment deltas.**

**Speaker script (90s)**

未来 18 个月，我们要证明的不是更多 vision，而是三件事。

第一，6 个 JV 的 launch path 真正跑起来。第二，至少 3 个进入 production，并产生真实 ARR。第三，runtime evidence pipeline 持续推进，每 90 天给出可审计的 progress memo。

到 M12，如果我们没有 3 个 in-production 和 real ARR，我们就不应该讲 Series A。这个 milestone 是我们对自己的约束。

**Visual direction**

横向 18 个月时间轴。用 `real ARR` 高亮。

---

## Slide 25 — Why Xfund + The Ask

**On-screen**

> **Why Xfund**
>
> 1. “We chase the talent.”
>
> → first-principle founder + CMU scientific co-founder
>
> 1. “Vertical proprietary data > LLM scaling.”
>
> → human beings themselves are the next vertical data
>
> 1. “Relationships are persistent and valuable.”
>
> → we build the auditable runtime for persistent relationships
>
> ---
>
> ## **The Ask — Late Seed / Pre-Series A**
>
>
> | Dimension           | Number                                   |
> | ------------------- | ---------------------------------------- |
> | Round size          | **$3M-$5M USD**                          |
> | Pre-money valuation | **$20M-$30M USD**                        |
> | Xfund target ticket | **$1.5M-$2.5M**                          |
> | Equity to Xfund     | **~7-10%**                               |
> | Runway              | **18 months**                            |
> | Next milestone      | **3 in-production JVs · ARR > $1M real** |
>
>
> Use of funds: engineering 40% · compute/data 25% · GTM 20% · operations/legal/IP 15%

**Speaker script (2 min)**

我们希望 Xfund 成为 first institutional check。

原因不是你们是热门基金，而是三个 thesis 对位。你们强调 chase the talent，我们的核心是 first-principle founder 加 CMU scientific co-founder。你们相信 vertical proprietary data beats LLM scaling，我们的判断是下一代 vertical data 是人本身。你们重视 relationship 的长期价值，我们做的就是这件事的工程载体。

具体数字：我们这一轮目标是 late seed / pre-Series A，规模 300 到 500 万美元，pre-money 2000 到 3000 万美元。希望 Xfund ticket 在 150 到 250 万美元，lead 或 co-lead。runway 18 个月。

下一里程碑非常明确：3 个 JV in-production，real ARR 超过 100 万美元，再启动 Series A。

**Visual direction**

上半页 Why Xfund，底部 ask box 用厚边框和大数字。

---

## Slide 26 — Close

**On-screen**

> # **Volvence**
>
> ## 模型进化，生命涌现。
>
> We don't sell AGI.  
> We don't sell a smarter LLM.
>
> We build the infrastructure for digital lives:
>
> **auditable · living · cross-session**

**Speaker script (30s)**

Patrick，谢谢你的时间。

我准备好回答你的任何问题，也可以现场进入 demo、repo evidence 或 JV / financial model 的尽调细节。

---

# Q&A Prep

## Private Traffic

**Q1: 私域市场天花板有多大？**

中国微信生态有 13 亿月活，私域运营是 $30B+ 级别市场，但 AI 渗透率还很低。我们的 thesis 不是替代现有 SCRM 的触达工具，而是重新定价长期关系运营。客单价从 RMB 1K / month / brand 提升到 RMB 5K-50K / month / brand，才是核心 upside。

**Q2: 微盟、有赞为什么不做？**

他们的 DNA 是触达工具，不是关系 runtime。做 AI 关系层会迫使他们承认现有工具在长期 LTV 上不足。更重要的是，他们缺 Body + Brain、跨 session memory、active learning 和治理结构。

**Q3: 1%+ 转化率怎么算？**

诚实回答：这是 projection，不是已完成试点结果。baseline ~0.3% 来自行业公开口径，Volvence 0.6-1.0% 来自 demo 中关系质量和推荐节奏的推断。6 JV 已签，但目前还没有真实 ARR。kill criterion 是 3 个月试点低于 0.5% 就重新评估 vertical 优先级。

**Q4: 头部 MCN 自建 AI 团队怎么办？**

MCN 的核心能力是 IP、内容、主播和流量。数字生命引擎需要 Yang Liu 这一档的 ML / continual learning team，也需要 runtime governance。我们不是和 MCN 竞争，而是成为他们的底层引擎和分成 partner。

## Team And Strategy

**Q5: 中国市场和全球市场怎么取舍？**

短期用中国私域、育儿、跨境电商跑出真实 ARR；中期用跨境电商和海外企业 base 自然进入全球；长期是 Digital Life-as-a-Service。不是二选一，是 staged portfolio。

**Q6: Burn rate 和 runway？**

500 万 RMB 自费投入已经 fully burned，全部投入团队、工程、实验和 JV 商务推进。下一步需要 institutional capital 支撑 18 个月 evidence pipeline。6 JV 进入 production 后会产生 ARR，但保守估计前 6 个月仍以 burn 为主。

**Q7: 团队会不会被大厂挖走？**

核心成员 full-time，已经有实物、repo、JV 和 sunk cost。retention 主要来自 mission alignment 和这件事的独特性，不是短期现金最大化。

**Q8: Yang Liu 为什么不去 OpenAI / DeepMind？**

她选择的是真实世界、真实数据、真实需求的 active learning 和 nonstationary learning 场景。OpenAI / DeepMind 当然强，但他们不拥有中国私域这种高密度、快速变化、稀疏反馈的关系数据场景。

## Technology And Risk

**Q9: substrate 涨价怎么办？**

多 substrate 兼容是核心策略。GPT、Claude、Qwen、DeepSeek、local models 都可以作为 substrate。我们的 durable state、relationship runtime 和 governance 不绑定单一模型。

**Q10: OpenAI native memory 替代你怎么办？**

native memory 是 feature，不是 vertical relationship owner。OpenAI 的通用助手模型不适合成为每个品牌和每个用户长期关系的 owner。我们拥有 vertical bundle、owner snapshot、business workflow 和 audit surface。

**Q11: UploadLive 留存数据？**

UploadLive 刚上线不久，应给真实 D7 / D30 数据，不夸大。重点解释留存来自关系记忆和跨 session 状态，而不是内容刷新。

**Q12: 最大弱点是什么？**

最大弱点是 6 JV 尚未转成真实 ARR。我们有签约、有 audience、有 projection，但 in-production 和 ARR 还需要 6-18 个月验证。这也是本轮融资存在的原因。

---

# Leave-Behind Packet

1. This deck PDF.
2. `xfund-strategic-thesis.md` — full written thesis.
3. `xfund-technical-credibility-brief.md` — technical DD brief.
4. Commercial financial model / 6 JV projection workbook.
5. Yang Liu full paper list.
6. 6 JV agreement summary, redacted.
7. Companion Benchmark methodology packet.
8. Experiment roadmap progress memo template.

---

# Production Checklist

- Keep black background and deep green accent.
- Use large numbers and sparse text on main slides.
- Keep `Mainstream vs Volvence` contrast style.
- Demo videos should be edited real usage, not raw screen recording.
- Add bilingual subtitles to all Chinese demo clips.
- Every projected revenue slide must visibly say `projected`.
- Every demo slide must distinguish `real demo` from `proven production uplift`.
- Print a one-page timing card for the 50-minute talk.

