# Sutton / Era of Experience 专项 · 2026-07

> Status: research note，外部证据，**不是 runtime contract，不进主链**。
> 论文/讲稿：20 篇 PDF + 4 条 link-only（OaK 无论文，只有讲座）。严格对既有 358 篇去重。
> PDF: [`../papers/sutton-experience-2607/`](../papers/sutton-experience-2607/)
> 下载脚本: [`../download_sutton_experience_2607.sh`](../download_sutton_experience_2607.sh)
> 下载校验: [`download-summary.md`](download-summary.md)（20/20 成功）

---

## 阅读顺序

1. [`01_THESIS.md`](01_THESIS.md) — 论纲原文精读（四支柱）、OaK / STOMP、Alberta Plan 十二步、大世界假设、可塑性丧失、streaming RL、批评与 LLM 阵营的实验回应
2. [`02_VZ_DELTA.md`](02_VZ_DELTA.md) — 与 VolvenceZero 的正面对撞：我们领先的两处、落后的一处、以及一个**我们从未测量过的自身缺陷**
3. [`03_PLASTICITY_REMEDY.md`](03_PLASTICITY_REMEDY.md) — **可塑性丧失的解法**：范围从"所有学习型 owner"收窄到**3 个 CMS band MLP**（其余是线性头，不适用）、tanh 饱和的精确失效链条、以及 L0 仪表 → L1 W2 行范数投影 → L2 gated 单元重初始化 → L3 换激活的分级方案

**前置专项**（本篇多处引用，建议先读）：

- [`../continual-learning-2026-07/`](../continual-learning-2026-07/) — 持续学习业界路径（25 篇）+ 个人参数化路线（22 篇）

---

## 一句话总判断

> **市面上"Sutton 说 LLM 是死路"的叙事是错的——不是因为 Sutton 没说，而是因为《Era of Experience》这篇论文自己不这么说。**

这份材料最重要的发现是一个文本事实：**Silver & Sutton 的论文与 Sutton 本人的播客立场，不是同一个主张。**

论文第 2 页脚注 1 给 RL 下的定义是：

> "适应可以通过任何方式发生，例如更新神经网络的权重，**或者基于来自环境的反馈进行上下文内（in-context）适应**。"

也就是说——**论文明确把"不改权重的 in-context 适应"算作 RL**。而 Sutton 在 2025-09 的 Dwarkesh 播客里的主张（LLM 没有 ground truth、无法在岗学习、是死路）要硬得多，且**不被这篇论文的正文支持**。

这个区别对我们是决定性的：

- **论文版本的 Era of Experience，与"冻结基底 + 自适应控制器"（R2）完全兼容。**
- **播客版本的 Sutton，与 R2 不兼容。**

而 2026 年 ICLR 的 `Reward Is Enough: LLMs Are In-Context Reinforcement Learners` **正是脚注 1 的实验证实**：LLM 在推理时、不更新任何权重的前提下，能优化标量奖励信号并持续改进。加上第一轮 CL-BENCH "naive ICL 打败所有专用记忆系统"的结论——**in-context 这条通道比这个领域自己以为的强**。

引用 Sutton 时必须注明是哪一个版本。混用两者会让我们的对外叙事经不起追问。

---

## 四支柱与我们的对位（速查）

| Silver-Sutton 支柱 | VZ 对应 | 判断 |
|---|---|---|
| **Streams**（终身经验流，而非片段） | R1 四时间尺度 | **我们更具体**——他们只说"流"，我们给了 online-fast / session-medium / background-slow / rare-heavy 四层可实现结构 |
| **Grounded actions & observations**（超越文本对话） | — | **我们落后，且是真缺口**。我们基本是文本/对话通道 |
| **Grounded rewards**（环境后果，而非人类预判） | R-PE（内禀预测误差） | **相邻但不同**，需要诚实区分——见 [`02_VZ_DELTA.md`](02_VZ_DELTA.md) §2 |
| **Non-human planning & reasoning** | R3/R4（`z_t` / `beta_t` 潜空间控制） | **我们更领先**——"不用人类语言思考"正是我们 R4 的定义 |

---

## 三个最有价值的收获

**1. "grounded reward" 的真正分界线不是"人 vs 环境"，是"预判 vs 后果"。**

论文脚注 2 明说："经验与人类数据并非严格对立。例如狗完全从经验中学习，但人的互动是它经验的一部分。" 正文进一步给出：用户报告蛋糕好不好吃、运动后多疲劳、头痛程度——**这些是 grounded reward，不是 human prejudgement**。

这直接**给我们的关系轨（R7）发了合法性**，但同时提出了一个尖锐的检验：我们的 F2/F3 指标，测的是**后果**（信任是否真的修复了、用户是否真的回来了）还是**预判**（judge 是否认为这句话温暖）？按这个标准，**LLM judge 打分属于 prejudgement，不是 grounded**。这条应当直接改写我们评估的合法性论证。

**2. 我们有一个从未测量过的自身缺陷：控制器的可塑性丧失。**

`Loss of Plasticity in Deep Continual Learning`（Nature 2024）证明：持续更新的网络会**逐渐丧失学习能力**，直到不如一个浅层网络；伴随现象是**表示的有效秩（effective rank）单调下降**与单元多样性丧失。

R2 让基底冻结，**规避了基底层的这个问题**——但**我们的学习型控制器、head、CMS 频段正是"持续更新的网络"，完整继承了这个缺陷**。而代码库里 `effective_rank` **一次都没出现过**，"plasticity" 只在 3 个文档里以无关含义出现。**我们跑得越久，这个问题越严重，而我们没有任何仪表能看到它。**

这是本轮唯一一条"低成本、且我们目前完全盲区"的发现。详见 [`02_VZ_DELTA.md`](02_VZ_DELTA.md) §3.A。

**3. Big World Hypothesis 是 R2 与有界容量的外部论证。**

"世界比智能体大若干个数量级……智能体必须依赖近似解……**可能会舍弃不常用的知识，为更常用的知识腾出空间**。" 且 Javed & Sutton 论证这**不是当前算力的临时产物**：算力增长时传感器精度和世界复杂度同步增长。

这是对我们 CMS 有界容量 + 遗忘设计的直接支持，也顺带给出一句对基准设计的警告——大世界假设"**可以通过控制环境与智能体的设计而被人为地变真或变假（例如在开发基准时）**"，与第一轮 CL-BENCH / AGENTCL 的方法学结论咬合。

---

## 一句话立场

> Era of Experience 的**四支柱我们占了三个**（streams / 非人类推理 / 部分的 grounded reward），**缺的那一个（grounded 动作与观察）是真缺口，不该粉饰**。而 Sutton 阵营最硬的技术资产不是 OaK（它至今没有论文），是**可塑性丧失这条实证线**——它既支持我们的 R2，又指出了我们自己没在看的一个仪表盘。
