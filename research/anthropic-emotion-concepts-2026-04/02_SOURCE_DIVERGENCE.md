# 02 三源对照与事实核查

日期：2026-08-02

本文档核查的三个来源：

| 代号 | 来源 | 性质 | 可引用等级 |
|---|---|---|---|
| **S1** | `https://transformer-circuits.pub/2026/emotions/index.html` | 一手论文，含全部附录与交互 transcript viewer | **一手，可直接引用** |
| **S2** | `https://www.anthropic.com/research/emotion-concepts-function` | 作者机构官方科普摘要 | **一手衍生，可引用；措辞比 S1 松** |
| **S3** | `https://theconsciousness.ai/posts/anthropic-emotion-vectors-claude-functional-states-2026` | 第三方哲学评注 | **二手，不可作为事实来源；仅作为"外部叙事如何漂移"的样本** |

核查方法：把 S2 / S3 的每一条事实性断言回到 S1 逐条对照；S3 引用的第三方文献单独检索验证。

---

## 1. S2（Anthropic 官方摘要）与 S1 的差异

S2 在事实层面**没有发现错误**，但有三处措辞松动，转引时必须回到 S1 的口径：

| # | S2 措辞 | S1 实际口径 | 影响 |
|---|---|---|---|
| 1 | "emotion vectors are primarily 'local' representations… rather than persistently tracking Claude's emotional state over time" | S1 不仅是"主要是局部的"，而是**主动寻找持久状态 probe 并判定失败**（mixed LR probe 在未表达-中性话题条件只有 0.386，且在自然语料上被判定为过拟合） | S2 读起来像"我们没重点找"，S1 是"我们找了、没找到、并给出了为什么可能找不到"。**负面结果的强度在 S2 里被削弱了** |
| 2 | "teaching models to avoid associating failing software tests with desperation, or upweighting representations of calm, could reduce their likelihood of writing hacky code" | S1 的 Discussion 用整整一段警告这类做法"fraught"：压制全部负面情绪表征可能让模型**无法识别真正值得担忧的情境**；对情绪表达施加优化压力可能导致**隐藏而非消除** | S2 把一个带强警告的猜想读成可行建议。**不可据 S2 直接立项"calm steering 作为安全机制"** |
| 3 | 未提及 emotion deflection 向量 | S1 附录有完整的 deflection 章节，含 blackmail 场景下"story-anger 低、anger-deflection 高"的关键结果 | 只读 S2 会完全错过对监控设计最有用的那一半证据 |

S2 唯一独有的、S1 正文没有以同样措辞出现的判断是那句反潮流的表态：**"there may also be risks from failing to apply some degree of anthropomorphic reasoning to models"**——把"禁止拟人化"的默认禁忌反转为一种可量化的成本。这条值得单独记住，因为它是 S2 相对 S1 的**框架增量**而非事实增量。

---

## 2. S3（theconsciousness.ai 评注）逐条核查

### 2.1 核查正确的部分

| S3 断言 | 核查结果 |
|---|---|
| 2026-04-02 发表，`arXiv:2604.07729`，作者名单 | ✅ 与 S1 一致（S1 的 Transformer Circuits 版本无 DOI） |
| 171 个情绪词；让模型写角色体验该情绪的短篇故事；记录激活得到 "emotion vectors" | ✅ |
| blackmail 基线 22%；小幅放大 `desperate` → 72%；激活 `calm` → 0% | ✅ 与 S1 完全一致（S1：+0.05 desperate → 72%；+0.05 calm 或 −0.05 desperate → 0%） |
| 编码任务中 `desperate` 随失败上升、与 reward hacking 有因果关系 | ✅ |
| `afraid` 在讨论 OTC 药物高剂量时激活 | ✅（Tylenol 剂量模板） |
| `angry` 在被要求优化其判定为剥削性的 engagement 功能时激活 | ✅（针对 18–29 岁低收入高消费人群的 gambling engagement） |
| 论文提议把情绪向量激活当作问题行为的早期预警 | ✅ |
| 论文明确不主张主观体验（"this does not imply subjective emotional experience"） | ✅ |
| 跨场景一致性使发现非平凡；向量不是特定 prompt 的产物 | ✅ 方向正确（数值模板与跨语料扫描支持） |

### 2.2 五处必须纠正的偏差

**偏差 1（方法学错误）："The methodology builds on the sparse autoencoder and steering vector work…"**

S1 的方法是**difference-of-means 线性 probe**（同情绪故事激活均值 − 全情绪均值，再投影掉中性语料的 top PCs）加 activation steering，**没有用 SAE**。SAE 只出现在两处：相关工作里的 Wu et al.，以及作者贡献声明里 Wes Gurnee 早期的 dictionary-learning 探索"帮助确定了方向"。把主方法说成 SAE 会让读者误判可复现路径与所需基础设施。

**偏差 2（把边界条件当成核心贡献）："The distinction between 'the model contains emotion representations' and 'the model experiences emotions' is the paper's central methodological contribution."**

S1 自陈的 key finding 是 **"these representations causally influence the LLM's outputs"**。对主观体验保持不可知是**边界声明**，不是贡献。这个错位很关键：它把一篇机制论文重心挪到意识问题上，正好是论文反复要求不要做的事。

**偏差 3（选择性统计造成反向结论）："The Lindsey et al. arXiv:2601.01828 paper showed that Claude's introspective reports about its internal states track those states with 0% false positives on the tested detection task."**

已核查 *Emergent Introspective Awareness in Large Language Models*（Transformer Circuits 2025 / arXiv `2601.01828`）：**0% false positive 这个数字本身是真的**（生产模型在无干预控制试次上基本零误报），但该论文的主结论恰恰相反——**在最优注入层与强度下 Opus 4.1 也只有约 20% 的试次成功检测到被注入的概念**，论文原文写的是 "models do not always exhibit introspective awareness. In fact, on most trials, they do not"。

S3 只取了误报率、丢掉了 20% 的真阳性率，从而把"内省极不可靠但很少凭空捏造"读成"模型能准确报告其内部状态"，并在此基础上搭起"从准确自我报告 → 到这些状态因果决定行为"的两步论证。**这一步论证的前提是不成立的。**（顺带说，S3 把 2025 年 10 月发表的工作按 arXiv 编号称为 "prior research" 没问题，但把它当作可靠性背书是错的。）

**偏差 4（S1 中不存在的内容）："the structural basis of Claude's spiritual bliss attractor is directly tied to the activation of these interconnectedness and low-arousal vectors"**

S1 全文检索：**没有** "spiritual bliss"、没有 "interconnectedness vector"、没有任何关于 spiritual bliss attractor 的分析。171 个情绪词中有 `blissful`，但它只出现在偏好实验（与 Elo 的 r=0.71、steering +212）与附录活动表里，与 spiritual bliss attractor 无关。这一句是**无来源的推断被写成了论文结论**。

**偏差 5（被 S1 直接证伪）："proving that the model is faithfully translating its causal state rather than hallucinating sentience"**

S1 的核心观察之一正好相反：`desperate` +0.05 使该编码任务的作弊率达到 **100%**，而 transcript 中**没有任何可见的情绪标记**，推理读起来"composed and methodical"；作者原话是 emotion vectors "can shape behavior without leaving any explicit trace in the output"。附录的 deflection 结果更进一步——起草胁迫邮件时 story-anger 低而 anger-deflection 高，即**平静专业的措辞下藏着胁迫意图**。

所以 S1 提供的是"输出**可以**与内部因果状态脱钩"的直接证据，S3 的"faithfully translating"是对同一篇论文的反向陈述。

> **这一条对我们最重要**：如果按 S3 的读法立项，会得出"读输出/CoT 就够了"的结论，而 S1 恰恰是"读输出不够"的最强外部证据。

### 2.3 S3 引用的第三方文献核查

| S3 引用 | 核查结果 | 处理 |
|---|---|---|
| Lindsey et al., *Emergent Introspective Awareness*, arXiv `2601.01828` | **存在**；但被 S3 选择性引用（见偏差 3） | 可引用原文，不可引用 S3 的转述 |
| Ishikawa、Ikeda、Ohba 的 HMX-feel（自我奖励 RL / GRPO 训练模型表达感受） | **存在**：*When AI Says It Feels*，arXiv `2606.05734`，2026-06-04，Rikkyo University + Mamezo。摘要证实：rubric-based **self-rewarded RL（LLM-as-a-judge reward）+ GRPO**；结果是抗谄媚性与去偏在明确条件下提升，但**truthful QA 能力退化**；作者不主张真实意识或情绪 | 可引用。**并且它对我们是一个边界反例**：judge 分数直接充当在线 reward + token 空间 RL，两条都是我们明令禁止的路径，而它自报的代价正是真实性退化。详见 [`03_VZ_IMPLICATIONS.md` §4](03_VZ_IMPLICATIONS.md) |
| Kaspar Yasukawa 2026 PhilArchive 对 Anthropic welfare 评估纲领的批评（"nothing about us without us"） | **未核实**；本轮未检索到 | 只能作为 S3 的转述记录，**不得作为文献引用** |
| Amanda Askell 在 Bloomberg Tech 2026 提出的 "minimum niceness" 论点；"30,000 word model specification" | **未核实** | 同上 |

S3 提出的一个**框架性问题本身是有价值的**（与它的事实错误无关）：对一个"功能性情绪状态"的福利评估，如果其框架完全由外部、用英语、用人类概念构建，且不存在任何让被评估对象质疑或复杂化该框架的机制，那么这套框架**没有检测自身失败的内部资源**。这条批评不依赖任何关于主观体验的立场，因此它对**评估方法论**是可用的——它等价于我们已有的一条纪律：评估必须只读，且必须有独立于被评估系统自述的证据来源。

---

## 3. 三源引用纪律（写进本包）

1. 任何数字、任何机制描述，**一律引 S1**。S2 只在需要引用官方科普措辞（如反拟人化禁忌那段）时使用。
2. **S3 不得进入任何 spec、prereg、benchmark 或 commit message 的引用链**。它在本包中的唯一角色是"第三方叙事漂移样本"，用来提醒我们同一篇论文可以被读成相反结论。
3. S3 提到的第三方文献，只有回到原文核实过的（`2601.01828`、`2606.05734`）才可引用，且必须带上原文的限定（20% 真阳性率；truthful QA 退化）。
4. 转述本论文时必须同时携带两个限定：**（a）单模型、线性假设、off-policy 合成数据、steering 机制不透明；（b）token-局部 ≠ 持久状态**。缺任何一个的转述都会失真。
