# 持续学习业界全景 2026-07 · 七派与逐篇要点

> 25 篇新增 PDF 的技术深读。每篇标注 arXiv id，PDF 在 [`../papers/continual-learning-2607/`](../papers/continual-learning-2607/)。
> 与 VZ 的对照与借鉴清单在 [`02_VZ_DELTA.md`](02_VZ_DELTA.md)，本文只讲论文自身。

---

## 0. 领域的形状：一次分裂

2024 年之前，持续学习是一个统一的问题——"顺序学多任务时怎么不忘旧的"，答案分三派（replay / regularization / architecture）。

2026 年，这个领域已经**分裂成两个几乎不对话的阵营**：

| | **In-Weight Learning (IWL)** | **In-Context / External Memory** |
|---|---|---|
| 学习发生在 | 模型参数 | 外部记忆 + prompt 编译 |
| 核心问题 | 往**哪里**写 | 写**什么** / **什么时候**接受写入 |
| 基座 | 可改 | 冻结（常常是 API-only） |
| 成功指标 | forgetting rate、backward transfer | task gain、成本、token 预算 |
| 代表 | Sparse Memory FT、SEAL、LoRA-MoE | RIZZ、ACE、Janus、Mem0 系 |
| 社区 | CL / NLP 传统学界 | agent 工程 / 创业公司 |

`Position: Modular Memory is the Key to Continual Learning Agents`（**2603.01761**，ICML 2026，源自 Dagstuhl "Continual Learning in the Foundation Model Era" 研讨会，25 位作者横跨 Toyota / MSR / Mila / KU Leuven / 清华）是这个领域第一次正式承认这次分裂，并主张缝合：

> 结合 IWL 与 ICL 的互补优势，通过**模块化记忆**设计——ICL 负责快速适应与知识累积，IWL 负责对模型能力的稳定更新。

它给的架构是三件套：**core model（感知与推理）+ working memory（当前交互轮）+ long-term memory（经验/事实/观察的累积）**，长期记忆被选择性检索进工作记忆，并**通过低频稳定更新蒸馏回核心模型**。

设计准则第一条就是："**分离快速适应与慢速整合**"。

> 判断：这篇立场书的框架，是我们 R1（四时间尺度）+ R2（冻结基底 + 有界控制器）+ R5/R6（CMS + 反思沉淀）在 2026 年的**独立复现**。25 位学界作者花一次 Dagstuhl 研讨会得出的"路线图"，是我们已经写进 spec 并跑在代码里的东西。这条对我们的价值不是技术，是**外部合法性证据**。

---

## S0. 评测与立场：两条决定性的负面结果

### S0-1 `CL-BENCH`（**2606.05661**，UC Berkeley / Snorkel AI / UW-Madison）

第一个高质量的、专家验证的持续学习基准。6 个真实有状态领域：软件工程、信号处理、疫情爆发预测、数据库查询、策略博弈、需求预测。

**方法学贡献（比结论更重要）**——任务必须同时满足三条：

1. **Headroom**：初始表现必须显著低于可达上限。否则"更强的基座"和"会学习的系统"无法区分。
2. **Shared latent structure**：实例之间必须存在**可发现的共享结构**（代码库结构、schema 约定、对手策略），且**不能从通用离线训练中恢复**。
3. **Drift**：例如数据库任务在中途做一次 schema migration，重命名表和列——**逼迫 agent 检测并丢弃陈旧经验**，而不是盲目复用。

作者明确指出：这三条**共同排除了"把现有 benchmark 的实例串起来"这种构造法**——标准 benchmark 的实例被设计成彼此多样且独立，足够强的模型不需要任何在线学习就能做好。

**gain metric**（可直接搬用）：

```
gt = r_stateful(t) − r_stateless(t)              # 单实例 gain
归一化 gain  ĝ = (r̄_sf − r̄_sl) / (r_max − r̄_sl)   # 除以该系统自己的 learning headroom
归一化 reward r̂ = (r̄_sf − r_external) / (r_max − r_external)
```

分母是"该系统自身的可学习空间"，避免 stateless 基线已接近 r_max 的任务贡献可忽略信号。reward 用**固定外部参照**（GPT-5.4 ICL stateless）而非各系统自身基线，使分数与提交无关。

**结论**：

> naive ICL 在多数任务上**超过**专用记忆架构；即便最好的系统也只拿到相对 stateless 基线 **25.4%** 的归一化 gain。累积的 state 经常**帮倒忙**：记忆模块引入虚假泛化与陈旧信念；更贵的系统无法把成本转化成表现。

即便 ICL 也表现出一致的学习缺陷：**过度依赖最近的任务实例，低估较早但相关的任务变体**。

### S0-2 `AGENTCL`（**2606.02461**，Ohio State / JHU / Intuit AI）

与 CL-BENCH 互补，专攻**任务流的构造**。核心方法学发现：

> **naive（随机）任务流区分不出记忆设计**——它把不同设计之间的差异压平了。必须构造 **compositional stream**：早期的子解、证据、workflow 被**有意设计成可在后期复用**。

三个指标（两趟 + held-out）：

- **Plasticity Gain (PG)**：早期任务建立的记忆是否帮到后期任务
- **Stability Gain (SG)**：解决某任务获得的经验是否**持久可复用**（第二趟只读）
- **Generalization Gain (GG) = H_j − B_j**：冻结记忆后在未见任务上的迁移

**结论**：现有记忆设计在组合性显式时能拿到明显 transfer gain；但在 naive 与 held-out 设置下**频繁引发认知干扰或性能退化**。

> 判断：S0-1 与 S0-2 独立地、用不同方法得到同一个结论——**记忆系统的净收益经常是负的，而且现有 benchmark 看不出来**。这是 2026 年这个领域最该被记住的事实。

### S0-3 其他综述

- **2603.12658**（`CL in LLMs: Methods, Challenges, Opportunities`）：按三个训练阶段组织——continual **pre-training** / **fine-tuning** / **alignment**。在经典 rehearsal/regularization/architecture 分类之上，按各自的遗忘缓解机制再细分。是当前最系统的 LLM-CL 分类学参考。
- **2404.16789**（Rutgers + Google Cloud AI，综述）：提出 **vertical continuity**（通用→专用）与 **horizontal continuity**（跨时间与领域）的二维划分，配 CPT / DAP / CFT 三阶段。
- **2501.04897**（在线持续学习 SLR）：81 个方法、1000+ 特征、83 个数据集的系统文献综述。偏 CV，作为 online-fast 尺度的方法库索引有用。
- **2605.06716**（`From Storage to Experience`，HKBU）：把 agent 记忆的演化形式化为三阶段——**Storage（轨迹保存）→ Reflection（轨迹提炼）→ Experience（轨迹抽象）**，并指出前沿阶段的两个变革机制是 **active exploration** 与 **cross-trajectory abstraction**。
- **2512.16301**（`Adaptation of Agentic AI`，UIUC/Stanford/Princeton/Berkeley，40+ 作者，89 页）：用统一的"adaptation"概念组织 post-training / retrieval / memory / skill，四范式框架 A1/A2（agent 侧）× T1/T2（tool 侧）。是当前覆盖最广的一篇。
- **2511.01093**（`Continual Learning, Not Training`，Arc Intelligence）：ATLAS 双 agent（Teacher 推理 / Student 执行）+ 持久学习记忆，**梯度自由**的持续学习，把适应的着力点从模型参数移到系统级编排。GPT-5-mini 作 Student 在 ExCyTIn-Bench 上 54.1%，比 GPT-5 (High) 高 13% 而成本低 86%；98 任务轨迹上 token 消耗从 100,810 降到 67,002。跨事件迁移：冻结的 pamphlet 使另一事件准确率 28%→41%。

---

## S1. 稀疏定位写入 — 参数派 2026 年最干净的结果

### S1-1 `Continual Learning via Sparse Memory Finetuning`（**2510.15103**，FAIR at Meta + Berkeley）

**这是本轮 25 篇里技术价值最高的一篇。**

背景：**memory layer**（Berges et al. 2024）把 Transformer 中间的一个 FFN 换成对巨大记忆池的稀疏查找：

```
I = TopKIndices(K q(x), k)          # 取 top-k 键（k=32）
s = softmax(K_I q(x))
y = s V_I
output = (y ⊙ silu(x·W1))·W2         # 输入相关门控
```

键值是**可训练参数**（不是激活）。典型记忆池 1M–100M 键，用 product key 做高效查找。可视为"有大量微型专家的 MoE"，但每个 token 只激活总记忆参数的 **0.03%–0.0002%**。

**关键发现：朴素地微调整个 memory layer 仍然灾难性遗忘。** 因为一个 batch 命中的记忆索引里，有些服务于通用目的（句法结构、领域宽泛特征）。

**方法**：只更新对当前输入**特异**的槽。借用 TF-IDF：

```
score(i) = c(i)/Σ_j c(j) · log( (|B|+1) / (Σ_{b∈B} 1[c_b(i)>0] + 1) )
```

`c(i)` 是本 batch 中索引 i 的访问次数，`B` 是**背景语料**（1000 个 DCLM 随机 batch，代表要保留的预训练知识）的 batch 集合。背景索引在微调中不变，可静态存进 checkpoint。取 top-`t` 个槽训练，其余全部 stop-gradient。

实现只需一行 mask trick：

```python
mem = mem * trainable_mask + mem.detach() - (mem * trainable_mask).detach()
```

**结果（1.3B 模型，学习新事实后在 held-out 上的退化）**：

| 方法 | NaturalQuestions F1 下降 |
|---|---|
| Full finetuning | **−89%** |
| LoRA | **−71%** |
| Sparse memory finetuning | **−11%** |

且是**同等的新知识获取水平**。每次前向通常访问 10³–10⁶ 个索引，但 `t=500` 就足以达到最佳表现。

**两个容易被跳过但很关键的细节**：

1. **LoRA 少改参数并不等于少干扰**。作者引 Biderman et al. 2024："LoRA 遗忘更少，但也**学得更少**"。原因是 LoRA 的低秩更新**稠密地作用于 hidden state**，单次更新仍全局影响。参数少 ≠ 影响面小。**真正起作用的是稀疏索引带来的互不共享**。
2. **优化器是一等变量**。作者原本全用 AdamW，后来意识到"per-parameter 自适应步长、weight decay、momentum 会与稀疏性发生意外交互"。换成 **SGD** 后 held-out 遗忘进一步下降——而 full FT 和 LoRA 换 SGD 并没有类似收益。

**排序基于 batch，不假设任务边界**——连续的 batch 可以来自相同或完全不同的分布。这对流式部署很重要。

### S1-2 `Improving Sparse Memory Finetuning`（**2604.05248**，UMich）

两个增量，对我们工程上更直接：

1. **开源 pipeline，把现成预训练模型 retrofit 成 memory-layer 模型**——他们用的正是 **Qwen-2.5-0.5B**，消费级硬件可跑。
2. 用 **KL 散度**替代 TF-IDF 做槽选择：优先更新那些相对背景分布"信息上令人惊讶"的 token 所对应的槽。作者称其理论基础更清晰。

### S1-3 `Attribution-Guided Continual Learning`（**2605.05285**，HKUST-GZ）

用 **LRP（Layer-wise Relevance Propagation）**基于 LLM 的**实际内部计算过程**估计参数重要性，重要参数受更小更新约束。相对 EWC 的 Fisher 信息，LRP 更贴近真实计算路径。属于"用机理可解释性指导写入定位"这一新兴子方向。

---

## S2. 自编辑 RL：SEAL

### `Self-Adapting Language Models`（**2506.10943**，MIT）

模型生成**self-edit**——自然语言指令，指定用于更新自己权重的数据，以及可选的优化超参（学习率、epoch、按 token 类型选择性计损）。两层嵌套循环：外层 RL 优化 self-edit 生成策略，内层用生成的 self-edit 做梯度更新。

**RL 算法细节值得注意**：奖励依赖于**动作发生时的模型参数** θ，因此 RL 状态必须是 `(C, θ)`——用旧模型收集的 (state, action, reward) 三元组会**过时失配**。作者试过 GRPO 和 PPO，"训练不稳定"，最终用 **ReST-EM**（拒绝采样 + SFT 的过滤式行为克隆）。

**结果**：

| 少样本抽象推理（ARC 子集，Llama-3.2-1B） | 成功率 |
|---|---|
| ICL | 0% |
| TTT + Self-Edit（无 RL） | 20% |
| **SEAL** | **72.5%** |
| Oracle TTT（人工最优配置，上界） | 100% |

| 知识注入（无上下文 SQuAD，Qwen2.5-7B） | 单篇 | CPT n=200 |
|---|---|---|
| 基座 | 32.7 | — |
| 只训原文 | 33.5 | 36.0 |
| 原文 + 自生成合成 | 39.7 | 50.6 |
| 原文 + GPT-4.1 合成 | 46.3 | **59.4** |
| **SEAL** | **47.0** | 58.2 |

单篇设置下 SEAL（7B）超过 GPT-4.1 生成的合成数据；但 CPT 设置下 GPT-4.1 反超。

**论文自己承认的两条硬伤**：

1. **灾难性遗忘**：顺序 self-edit 流下，早期任务表现随 edit 数增加**持续下降**。作者建议未来用 reward shaping 惩罚回退、或引入 null-space 约束编辑 / 表示叠加。
2. **计算开销**：**每次 self-edit 评估要 30–45 秒**（必须微调并评估整个模型来算 reward）。相比 RLHF 的单次前向或可验证奖励的正则匹配，这是数量级的差距。**实时连续编辑在生产上不可行。**

第三条限制：当前实现假设每个 context 都配有显式下游任务，**无法扩展到无标注语料**。

---

## S3. 机理诊断派 — 这个领域最该先读的

### S3-1 `Spurious Forgetting`（**2501.13453**，华南理工，ICLR 2025）

**本轮最重要的单篇。** 核心命题：

```
任务表现 = 任务对齐 (task alignment) + 底层知识 (underlying knowledge)
```

现象：在合成数据集 + 随机初始化 LM 的受控实验里，学 Task 1 后 Task 0 的表现在**最初 150 个优化步内**从接近 100% 崩到约 10%。"指望 10 万条安全对齐样本的底层知识在 150 步内消失，是不合理的。"

**机理**：分析权重更新的角度发现——预训练各阶段之间更新方向夹角很小（一致空间）；Task 0 的更新几乎与预训练同空间；而 **Task 1 的前 150 步在一个明显不同的空间更新，尤其影响底层（含输入嵌入）**。特征层面：主成分发生显著漂移，但**Task 1 前 150 步与后续步的漂移会相互抵消**——说明 Task 0 与 Task 1 的对齐**并不根本冲突**，知识仍在。

理论：在带残差连接的线性映射序列 `X_l = (W_l + I) X_{l−1}` 上分析近正交更新导致的输出漂移界，证明**冻结底层可缓解**。

**结果（顺序微调 SEQ 场景的任务准确率）**：

| 方法 | 准确率 |
|---|---|
| SEQ 基线 | 11% |
| 正则化 / 生成式 replay / model merging / gradient 类方法（最好的） | 22% |
| **冻结底层（Freeze）** | **44%** |

在四个真实持续学习场景验证：安全对齐、持续指令微调、持续知识编辑、实例增量学习。

> 判断：这篇的杀伤力在于——它说这个领域花十年发明的一大堆方法（EWC、SI、IMM、replay、merging）在这个失败模式上**全部只能拿到 22%**，而一个"冻住底下几层"的土办法拿到 44%。因为前者都在治"知识丢失"，而病根是"对齐被掀翻"。**诊断错了，药再精巧也没用。**

### S3-2 `Mechanistic Analysis of Catastrophic Forgetting`（**2601.18699**，林雪平大学）

2026 年中 20 个前沿模型的系统对比：10 个闭源（行为与语义输出漂移剖析）+ 10 个开源权重（深度机理解释）。用权重空间轨迹追踪、**CKA（Centered Kernel Alignment）**、以及 **MoE 路由门漂移**计算。

定位结论：

- **早层 attention head 出现系统性熵扩散（entropic dispersion）**
- **中到深层 FFN / 稀疏专家块出现局部表示塌缩（localized representation collapse）**

据此提出 **LRCP（Low-Rank Circuit Projection）**，子空间正则化训练干预，在开源权重配置上保住至多 **94.2%** 的原有能力，同时匹配标准 PEFT 基线的适应速度。

> 与 S3-1 互补：S3-1 说"底层对齐"，S3-2 说"早层 attention 熵扩散 + 中深层表示塌缩"。两者都指向**遗忘是有明确空间定位的，不是弥散的**——这正是 S1 稀疏定位写入路线的理论基础。**S1、S3 是同一个故事的两半。**

---

## S4. 持续预训练：已经解决的那部分

### `Simple and Scalable Strategies to Continually Pre-train LLMs`（**2403.08763**，Mila / Concordia / EleutherAI，TMLR）

结论朴素但重要：**LR re-warming + LR re-decaying + 少量旧数据 replay** 三件套，就足以**匹配在全部数据上从头重训**的效果（以最终 loss 和多个 LM 评测的平均分衡量）。

验证覆盖：405M 规模上的弱分布漂移（English→English，两个常用预训练集）与强分布漂移（English→German），以及 **10B 参数规模**的弱漂移。计算量只是重训的一小部分。

附带贡献：提出替代 cosine 调度的方案，规避 LR re-warming 引入的遗忘，且不绑定固定 token 预算。

> 判断：**rare-heavy 时间尺度不需要新算法**。这是一条 2024 年就已固化、2026 年仍是默认答案的工程配方。任何在 rare-heavy 上"发明"新方法之前，应该先确认这个基线跑满了没有。

---

## S5. 模块化与合并：在重新发明多时间尺度

三篇的共同结构非常一致——**都在造"快慢双时间尺度 + 有界容量记账"**：

### S5-1 `Merge before Forget`（**2512.23017**，Penn State）

不再保留和冻结历史 LoRA（那会随任务数线性增长内存），而是**正交初始化 + 顺序合并进单一 LoRA**：用先前 LoRA 的正交基提取来初始化新任务的学习，并利用 LoRA 的 **A/B 分量内在不对称性**，用**时间感知缩放**平衡新旧知识。任务数上**常数内存复杂度**。

### S5-2 `CP-MoE`（**2605.20247**，UNSW）

指出 LoRA-MoE 类持续学习的根本权衡：**要么专家隔离太狠（跨任务迁移受限），要么允许任务特定更新覆写重要参数（严重遗忘）**。方案是引入 **transient expert** 捕获早期任务特定更新，再引导其并入 **stable experts**；配一个**一致性保持路由偏置**（用 transient expert 估计与 stable expert 的表示相似度，把路由推向更兼容的专家选择）。

### S5-3 `ProCL / Program Memory`（**2605.13162**，Deakin）

明确以神经科学的 **Complementary Learning Systems** 为灵感：把 LoRA adapter 组织成结构化的 **program memory slots**，通过输入条件注意力动态检索。快速局部适应（相似输入复用共享 adapter 区域，为未来数据保留未用容量）+ 底层分布式 adapter 逐步跨任务累积知识。完全在 LoRA 参数化内运作，**无额外推理成本**。

> 判断：这三篇是"多时间尺度 + 有界容量"这个思想的三次独立、且互不引用的收敛。ProCL 甚至直接引 CLS 作为动机——和我们 CMS 的来源同一个。**它们证明这个结构是必然的，但也证明在纯 adapter 层面做它，天花板很低**（都只在 SuperNI / VQA 这类学术 benchmark 上验证）。

---

## S6. 测试时训练：一次框架转换

### `End-to-End Test-Time Training for Long Context`（**2512.23675**，Astera / NVIDIA / Stanford / Berkeley / UCSD）

**这篇最有价值的是它的框架转换，不是它的架构。** 开篇第一句：

> 我们把长上下文语言建模**表述为一个持续学习问题，而不是架构设计问题**。

据此，架构上**只用标准 Transformer + 滑动窗口注意力**——不发明新架构。模型在测试时通过对给定上下文做 next-token prediction 继续学习，**把读到的上下文压进自己的权重**。

**E2E 的两层含义**：

- **内循环 E2E**：直接优化网络末端的 next-token 预测损失，而非先前长上下文 TTT 工作的 **KV-binding** 代理目标（MesaNet、Titans、Nested Learning 都用 KV Binding 作为核心组件——学习目标是从 key 预测 value）。
- **外循环 E2E**：训练时不用标准预训练，而用 **meta-learning** 准备 TTT 的初始化——每条训练序列先当作测试序列在内循环做 TTT，然后把 TTT 之后的损失在大量独立序列上平均，通过**梯度的梯度**对初始化做外循环优化。

**结果（3B 模型，164B token）**：随上下文长度的 scaling **与 full attention 相同**，而 Mamba 2 与 Gated DeltaNet 做不到；同时像 RNN 一样**推理延迟与上下文长度无关**，128K 上下文时比 full attention 快 **2.7×**。

**三条工程经验（比结论更可复用）**：

1. **只更新 MLP 层**——在内循环更新 attention 层会**导致外循环不稳定**。
2. **只对 1/4 的 block 做 TTT**——更新更多层意味着更多反传计算，存在明确的收益/成本拐点。
3. **mini-batch TTT + 滑动窗口**取代在线单 token 梯度下降——单 token 的梯度步"很容易偶然导致梯度爆炸"，且无法并行。

---

## S7. Agent 记忆与黑盒适应：最热闹，也最脆

### S7-1 `RIZZ`（**2606.20638**，牛津）

冻结黑盒模型（用 Claude Haiku 4.5），适应完全发生在**verifier 门控的记忆、路由与 prompt 编译**。

核心设计——**零干扰区（zero-interference zone）**：输入流被组织进**动态生成的记忆分支（branch）**。每个分支是一个小型专家记忆：存成功样例、把重复模式蒸馏成程序性规则、**追踪这些规则在后续调用上是帮忙还是帮倒忙**。查询落在已有经验之外就新建分支；两个分支冗余就合并或剪枝。

生物学直觉写得很清楚："记忆先是可塑的，然后变得有选择性，最终随证据累积固化为持久习惯"——**吸收 → 特化 → 稳定**。

**写入门控**：模型作答后，确定性 verifier（如 fuzzy ratio）打分，只有足够成功的交互才写入持久记忆；弱的或有害的轨迹被**降级、隔离或丢弃**。失败被保留为**具体的反模式（anti-template）**。

**不需要 ground-truth task ID 或预设任务分类法**——自己从查询内容、路由信号、verifier 奖励中发现记忆组织。

预算感知的 prompt 编译器在固定上下文窗口内组装分支局部证据，使系统**优雅降级回 frozen 而不是膨胀 token 成本**。

### S7-2 `Janus / The Past Is Prologue`（**2606.31121**，UVA / Princeton / UCF）

**把记忆更新当作部署决策**——这是本轮最贴近我们 gate 语义的一篇。

问题陈述：现有系统"通常部署每一个本地生成的记忆更新，而不检查它是否改善未来行为"。结果是——帮到当前任务的更新可能**覆写有用知识、引入过度特化的规则、或使最终记忆偏向近期样本**。

**Janus** 是方法无关的插件式控制器，包住任意现有 updater，决定**接受候选记忆更新还是保留旧记忆**：

1. **Memory Momentum Trigger**：检测候选更新是否偏离近期的记忆更新轨迹。**不触发就直接接受**，避免不必要的重放开销。
2. 触发时，在**紧凑的混合评估集**上比较 `M_{t−1}` 与 `M̂_t`，择优部署。评估集三类构成：
   - **coverage**：代表已见任务分布
   - **boundary**：过去记忆选择曾改变正确性的任务
   - **fresh**：最近任务的新鲜切片

**不重放完整历史**。跨 6 个数据集、2 个骨干 LLM、2 个记忆 updater，平均准确率 **+2.7 到 +4.6** 分。

### S7-3 `ALMA`（**2602.07755**，UBC / Vector / Jeff Clune 组）

元学习**记忆设计本身**。Meta Agent 以**代码**为搜索空间（理论上可构造任意记忆设计，包括数据库 schema 及其检索与更新机制），在 archive 上做开放式探索：采样已探索设计 → 反思其代码与评估日志 → 生成想法与计划 → 实现 → 验证并评估 → 写回 archive。

四个序贯决策域（ALFWorld、TextWorld、Baba Is AI、MiniHack）上**全部超过 SOTA 人工设计记忆基线**，且比多数人工基线**更省成本**；随记忆规模 scaling 更好，面对任务分布漂移时学得更快。

论证是 Clune 一贯的："机器学习史上反复出现的主题是——AI 系统中手工设计的组件最终被学出来的、更有效的组件取代。"

### S7-4 `SSGM`（**2603.11768**，暨南大学）

不提新算法，提**记忆治理架构**：把**记忆演化与执行解耦**，在任何记忆固化之前强制执行**一致性验证、时间衰减建模、动态访问控制**。

最有价值的产出是**演化记忆失败的四维分类表**（可直接当 kill-condition 清单用）：

| 维度 | 失败模式 | 表现 | 建议缓解 |
|---|---|---|---|
| **Stability** | Semantic Drift | 迭代摘要导致细微差别逐步丢失 | Ground Truth Anchoring |
| | Procedural Drift | 强化次优/过时的工作流 | Rule Verification |
| | **Goal / Role Drift** | **累积交互偏置导致对齐漂移**（长期角色扮演） | **Role Partitioning** |
| **Validity** | Memory Hallucination | 检索到不存在或捏造的事实 | Consistency Verifier |
| | Temporal Obsolescence | 陈旧记忆与新状态冲突（用户个性化场景） | **Weibull 衰减函数** |
| **Efficiency** | Retrieval Latency | 搜索时间随历史线性/二次增长 | Hierarchical Indexing |
| | Index Bloat | 冗余/噪声情节日志堆积 | Active Forgetting / Pruning |
| **Safety** | Memory Poisoning | 恶意指令注入存储 | Write Filtering (Firewall) |
| | Privacy Leakage | 未授权的跨会话/跨用户检索（多租户 agent） | Provenance + ACLs |

论文特别指出：**semantic drift 主要由有损压缩算法（迭代摘要）驱动**——信息被反复重编码，细微差别单调流失。

### S7-5 `CLaaS`（**2606.05559**，Resolute Labs / Incept Labs）

把持续学习做成**服务**，藏在 chat API 之后。部署中收集在线 rollout 进 **experience replay buffer**，异步训练时做**梯度复用**，更新一个 LoRA adapter，**热重载**进推理服务器，形成实时改进闭环。

明确排除 GRPO——"它依赖 group 统计量"，需要可重置的离线环境；而"真实世界环境不能被轻易重置"，每个场景只能采样一次。改用带 clipping 的替代策略梯度目标。

**结论与 CL-BENCH 方向相反**：参数更新带来**优于 in-context learning 的 forward transfer 和更少遗忘**，且 **replay 是样本效率的关键选择**。

> 注意：这不是直接矛盾（CLaaS 比的是"参数更新 vs ICL"，CL-BENCH 比的是"专用记忆系统 vs ICL"），但两条放一起指向一个没人做过的三路对照实验——见 [`02_VZ_DELTA.md`](02_VZ_DELTA.md) 未决争论第 1 条。

---

## 全景总结

把 25 篇合起来，2026 年 7 月持续学习的真实状态是：

1. **诊断刚刚被修正**（S3）。此前十年的方法在治错误的病。真正的失败有明确空间定位：底层对齐、早层 attention 熵扩散、中深层表示塌缩。
2. **参数派因此找到了正确的问题**（S1）：不是"改多少参数"，而是"改的参数是否稀疏索引、互不共享"。Sparse Memory FT 的 11% vs 71%（LoRA）vs 89%（full）是当前最干净的数字。
3. **记忆派的净收益尚未被证明**（S0）。两个独立基准同时显示：专用记忆系统不如 naive ICL，且累积 state 经常有害。
4. **治理层刚刚出现**（Janus、SSGM）——"记忆写入需要被门控"这个念头在 2026 年上半年才第一次被明确提出。
5. **rare-heavy 尺度已经解决**（S4），不需要新算法。
6. **架构派完成了一次概念让步**（S6）："长上下文是持续学习问题，不是架构问题"。
7. **模块化派在独立复现多时间尺度 + 有界容量**（S5），但受限于纯 adapter 层面，天花板低。
