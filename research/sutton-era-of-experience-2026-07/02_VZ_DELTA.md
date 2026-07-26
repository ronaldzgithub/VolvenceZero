# 与 VolvenceZero 的正面对撞

> 配套阅读：[`01_THESIS.md`](01_THESIS.md)（论纲精读）。
> 本文关于 VZ 现状的陈述均在 2026-07-26 的 `main`（`d56bdf2`）上核对过代码与 spec。
> Status: research note。**不是 runtime contract，不进主链**；"落点"是建议，不是已批准变更。

---

## 1. 先处理那个真正的对撞：Sutton 说 LLM 是死路，我们建在 LLM 上

这是本轮唯一需要严肃对待的合法性挑战。诚实的处理方式是**把它拆成两个不同强度的主张**，因为它们的证据地位完全不同。

### 1.1 弱版本（论文版）：与我们完全兼容

《Era of Experience》正文说"今天的技术，配合恰当选择的算法，**已经提供了足够强大的基础**"，且脚注 1 把 RL 的适应定义为"可通过任何方式发生，例如更新神经网络权重，**或基于环境反馈做 in-context 适应**"。

**这个版本的主张就是我们的 R2**：基底不必改，适应发生在别处。我们与之的差别只是把"别处"具体化成了四个时间尺度和一组有界控制器。

### 1.2 强版本（播客版）：与我们不兼容，但证据较弱

Sutton 在 Dwarkesh（2025-09-26）的主张：LLM 没有 ground truth、无法在岗学习、无论怎么 scale 都需要新架构；儿童不是模仿者而是主动实验者。

**这个版本正在被实验证据侵蚀**，且侵蚀它的证据来自三个互不相关的方向：

| 证据 | 来源 | 打击的具体命题 |
|---|---|---|
| **ICRL**：LLM 在推理时优化标量奖励，响应质量随上下文增长持续改善；即便奖励由同一个 LLM 生成也有效 | `Reward Is Enough`（ICLR 2026），[`01_THESIS.md`](01_THESIS.md) §7.3 | "LLM 无法在岗学习" |
| **naive ICL 打败所有专用记忆系统**，最好的系统只吃到 25.4% headroom | CL-BENCH，[第一轮](../continual-learning-2026-07/01_LANDSCAPE.md) §S0-1 | "in-context 通道不够" |
| **多数真实环境根本没有可验证奖励**，需要不依赖奖励的中间层 | `Agent Learning via Early Experience`（Meta，ICML 2026） | "grounded reward 随处可得" |

**结论**：强版本作为一个**方向性直觉**值得尊重，但作为一个**否定我们架构的论证**，它目前拿不出对应的证据强度。我们不需要反驳它，只需要在引用时注明版本——**混用两个版本会让我们的对外叙事在第一次追问时就塌掉**。

---

## 2. 四支柱逐条对位

### 2.1 Streams —— 我们更具体

Silver-Sutton 只主张"智能体应有跨月跨年的经验流"，没有给出结构。我们的 R1 给了四层可实现结构（online-fast / session-medium / background-slow / rare-heavy）及各层的 SSL-RL 交替（`docs/specs/multi-timescale-learning.md`）。

**这一条我们领先，且可以直接作为对外叙事使用**——他们提出了要求，我们给了工程解。

### 2.2 Grounded actions & observations —— 我们落后，且不该粉饰

论文的要求是超越"人类特权文本通道"，走向感觉运动式的交互。**我们基本是文本/对话通道。**

`vz-embodiment-ant`、`docs/specs/digital-ant-embodiment.md`、`environment-interface.md` 是这个方向的存在性起点，但与"监控环境传感器、远程操作望远镜、控制实验室机械臂"的标尺相比，差距是量级上的。

**诚实的表述**：这是四条里我们唯一真正缺失的一条。它不影响我们在关系/陪伴这个垂类的有效性（该垂类的"环境"本来就主要由人构成——见 §2.3），但**它确实限制了我们在"发现人类想不到的策略"这个意义上的上限**。不应把 `digital-ant` 说成已经补上了这一条。

### 2.3 Grounded rewards —— 相邻但不同，且这里有一条能立刻改写我们论证的洞察

**先说差异**。业界与 Silver-Sutton 的 grounded reward 是**环境后果**（心率、考试成绩、CO₂ 浓度）。我们的 R-PE 是**内禀预测误差**。这两者不是一回事：

- grounded reward 回答"世界变好了吗"
- prediction error 回答"我预测对了吗"

一个智能体可以预测得越来越准，同时把世界搞得越来越糟。**这是 R-PE 的真实风险敞口，在 [第一轮](../continual-learning-2026-07/02_VZ_DELTA.md) §1 已经指出过（"没有外部锚，更难自证"），这里得到了 Sutton 阵营的独立确认。**

**但接下来是本轮对我们最有用的一条**：

论文的分界线**不是"人 vs 环境"，而是"预判（prejudgement）vs 后果（consequence）"**。两处原文支持：

- 脚注 2：**"狗完全从经验中学习，但人的互动是它经验的一部分。"**
- 正文：**"有根的奖励可以来自作为智能体环境一部分的人类"**——用户报告蛋糕好不好吃、运动后多疲劳、头痛程度，**"这类奖励测量的是智能体动作在其环境中的后果"**。

**这同时做了两件事**：

1. **给 R7 关系轨发了合法性。** 关系型智能体的"环境"主要由人构成，这在 Silver-Sutton 的框架里完全合法，不是二等公民。
2. **对我们的评估提出了一个尖锐检验。** 按这个标准：
   - **LLM judge 给回复打的温暖度分 = prejudgement，不是 grounded。**
   - 用户是否真的回来了、信任是否真的修复了、rupture 之后关系是否真的延续了 = **consequence，是 grounded 的。**

我们的评估体系里两类都有（`docs/specs/evaluation.md`、`evaluation-cascade.md`、`companion-bench.md` 的 judge contract）。**但我们从未按这条轴把它们分开标注过。**

> **落点（低成本、高价值）**：在评估 spec 里给每个指标加一个 `reward_grounding: consequence | prejudgement` 的标注，并规定**只有 consequence 类指标可以作为学习信号或 gate 输入，prejudgement 类只能做 readout**。
>
> 这与我们已有的 **R12（评估只读）** 是同一个精神的细化，也与 [第一轮](../continual-learning-2026-07/01_LANDSCAPE.md) §S7-1 RIZZ 的"verifier 门控"同源。它不需要新 owner，只需要一次标注 + 一条准入规则。

**同时要记住反面**：`Reward Hacking in the Era of Large Models`（复旦，42 页）的 **Proxy Compression Hypothesis** 说 reward hacking 是"用表达力强的策略优化被压缩的奖励表示"的必然后果。**grounded 不等于不可 hack**——心率、步数、留存率全都可被 game。所以上面的规则是"consequence 类才可入 gate"，**不是"consequence 类一定对"**。

### 2.4 Non-human planning & reasoning —— 我们更领先

论文说"人类语言极不可能是通用计算机的最优实例"，主张用符号的/分布式的/连续的/可微的非人类表示思考。

**这正是我们 R4 的定义**："长期行为学习在时间控制器状态（`z_t`、`beta_t`）中，而不是通过表层文本关键词规则。" 我们的 `vz-temporal`（metacontroller、beta_t 段闭合、在 `z_t` 上做 internal RL）就是这条主张的一个具体实现。

**这一条我们不仅对齐，而且比论文走得更远**——论文只提出了方向，我们有 owner、有契约、有快照。

---

## 3. 值得借鉴（3 条）

### A.【最高，且我们目前完全盲区】控制器的可塑性丧失，我们从未测量

**这是本轮唯一一条"低成本 + 我们完全没在看"的发现。**

`Loss of Plasticity in Deep Continual Learning`（Nature 2024）证明：持续更新的网络**逐渐丧失学习能力，直到不如浅层网络**；伴随**有效秩单调下降**、死单元增多、单元多样性丧失。**只有持续向网络注入多样性的算法（如 continual backprop）才能无限期维持可塑性。**

**为什么这条精确地打中我们**：

R2 让**基底**冻结，规避了基底层的可塑性丧失。但我们持续更新的东西一个都没少：

- `CreditLedger` 的 `RewardingStateHeadState`（learned rewarding-state head）
- CMS 各频段的 band MLP（`cms_band_mlp.py`、`torch_cms_band.py`）
- `PeWriteGate` 的 bounded-learned 阈值
- `vz-temporal` 的 metacontroller / internal RL on `z_t`
- 各 learned backend 的 head

**这些正是论文所说的"持续更新的网络"，完整继承了可塑性丧失。** 而且我们的部署形态（长期陪伴、跨月跨年的单一实例）恰恰是这个问题最严重的形态——**跑得越久越严重**。

**核对结果**：代码库里 `effective_rank` / `erank` **一次都没有出现**；`plasticity` 只在 3 个文档里以无关含义（"dormant path"）出现。**我们没有任何仪表能看到这件事。**

**落点**：给学习型 owner 的 readout 加一个 `plasticity_readout`：

- **有效秩** `erank(Φ) = exp{H(p₁,…,p_q)}`，`p_k = σ_k/‖σ‖₁`（论文原式，直接可实现）
- **死/休眠单元比例**

先做 **readout-only**（对齐 `docs/specs/credit-and-self-modification.md` 里 learned baseline 的 `readout-only → readout-with-acceptance → acceptance gate` 三阶段升级协议）。**有效秩持续下降**是一个干净的 kill condition 候选，也是判断"要不要引入 continual-backprop 式选择性重初始化"的前提证据。

> 注意这与前两轮的两条失败模式**互补而非重复**：
> - 第一轮 **Spurious Forgetting**：指标掉了 ≠ 知识没了（可能是对齐被掀翻）
> - 本轮 **Loss of Plasticity**：还可能是**网络已经学不动了**
>
> 三者构成一张完整的归因表。没有可塑性仪表，我们连这个可能性都无法排除。

### B.【高】"预判 vs 后果"的指标二分 → 见 §2.3

`reward_grounding: consequence | prejudgement` 标注 + "只有 consequence 类可入 gate"。低成本，直接强化 R12，且给 R7 关系轨提供了 Silver-Sutton 框架下的合法性论证。

### C.【中】OaK 的"持续淘汰"与元学习步长

**C-1 抽象的持续淘汰（Alberta Plan 第 11 步）**：OaK 相对 STOMP 的全部增量是——**持续评估所有元素（特征、子任务、options、option 模型）的效用，把从未在规划中有用的元素删除并替换**。

我们的 `vz-temporal` 已经有 bounded structural temporal proposal（`merge` / `split` / `prune`，见 `credit-and-self-modification.md`），**方向一致**。可借的是判据的锐利程度：OaK 的判据是"**该 option 模型在规划中是否曾经有用**"——一个基于下游使用的、而非基于内部统计的判据。这与第一轮 Janus 的"记忆更新是部署决策"、第二轮 Self-Gating 的"检索到的记忆是否该激活"是**同一族判据**：**用下游效用而非内部自洽性来决定留存。**

**C-2 每权重的元学习步长**：OaK 讲座与 Alberta Plan 第 1 步都主张"不应有全局步长参数，而应对每个特征有不同的步长参数，并由元算法设定"。我们的 learned head 目前是全局学习率。这是一条**成熟、可选、低风险**的改进，但优先级低于 A 和 B。

---

## 4. 明确划界（2 条）

### 4.1 不要把 R-PE 重新表述为 "grounded reward"

它们不是一回事（§2.3）。把内禀预测误差包装成 Silver-Sutton 意义上的 grounded reward，会在任何一个读过原文的人面前失效，并且会掩盖我们真实的风险敞口。

**正确的表述**：我们走的是**内禀信号路线**，与 grounded reward 路线并行；我们通过 §2.3 的"consequence 类指标可入 gate"来补上外部锚，而不是声称 PE 本身就是外部锚。

### 4.2 不要引入 OaK 式的无界元素替换

OaK 的持续淘汰-替换循环**没有容量上界，也没有回滚概念**——它假设"删掉没用的、换上新的"永远是安全的。这与我们 R10（有界自修改）/ R15（可回滚）的关系，和第一轮对 DGM、ALMA 的判断是同一类：**可以借判据（下游效用），不能借无界的生成-替换循环。**

我们已有的 structural proposal 走 gate + 可回滚审计，这个形状是对的，**不应为了向 OaK 靠拢而放宽**。

---

## 5. 一个我们独有、且值得下注的位置

把三轮调研合起来看，有一个**没有人占据的交叉点**：

- **Sutton 阵营**有 streams、grounded reward、持续学习理论，但**没有关系/多主体的一等公民地位**，且其可塑性工作停留在小网络与 RL 基准。
- **LLM agent 阵营**（第一、二轮）有记忆、个性化、评测方法学，但**没有时间尺度结构，也没有内禀信号**，且被 CL-BENCH 证明净收益尚未成立。
- **我们**同时有四时间尺度、内禀 PE、双轨、以及门控与回滚契约。

**Era of Experience 的四支柱我们占了三个**。缺的那一个（grounded 动作与观察）在关系垂类里被**部分地豁免**——因为脚注 2 明说人的互动就是经验的一部分，关系型智能体的环境本来就主要由人构成。

**但这个豁免是有代价的**：它意味着我们**永远不会**在"发现人类想不到的策略"这个意义上超越人类先验。这是一个应当被清楚记录的架构性上限，而不是一个待解决的 bug。

---

## 6. 一页纸行动摘要（接前两轮 §5 / §8）

| # | 事项 | 类型 | 成本 | 落点 |
|---|---|---|---|---|
| **L** | **`plasticity_readout`：有效秩 + 死单元比例，readout-only** | **补仪表** | **低** | 学习型 owner（credit head / CMS band / metacontroller） |
| M | 指标标注 `reward_grounding: consequence \| prejudgement`；只有 consequence 可入 gate | 改判据 | 低 | `evaluation.md`、`evaluation-cascade.md`、`companion-bench.md` |
| N | structural proposal 判据改为"下游是否曾被使用"（对齐 OaK / Janus / Self-Gating 同族判据） | 改判据 | 中 | `vz-temporal` structural proposal |
| O | 归因三分表存档：对齐掉了 / 知识掉了 / **学不动了** | 存档 | **低** | `evaluation-cascade.md`（扩充第一轮 C 条） |
| P | 对外叙事规范：引用 Sutton 必须注明"论文版/播客版" | 存档 | **低** | `docs/business/` |

**L 是本轮唯一的新盲区，且最便宜。** O 是把第一轮的 C 条（对齐 vs 知识二分）扩成三分——**加上"可塑性丧失"这一项之后，归因表才是完整的**，而这一项没有 L 的仪表就无法判定。**L 和 O 应当一起做。**
