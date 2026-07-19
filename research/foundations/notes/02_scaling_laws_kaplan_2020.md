# Scaling Laws for Neural Language Models（深度分析）

- **arXiv**：2001.08361（v1，2020-01-23）
- **作者**：Jared Kaplan*（JHU + OpenAI）/ Sam McCandlish*（OpenAI）/ Tom Henighan / Tom B. Brown / Benjamin Chess / Rewon Child / Scott Gray / Alec Radford / Jeffrey Wu / Dario Amodei（OpenAI）
- **本地路径**：`../papers/scaling-laws-for-neural-language-models-2001.08361.pdf`
- **谱系定位**：substrate behavior contract — VZ R2「冻结基底」**之所以能被冻结**的经济学/物理学根据
- **R-ID 引用**：`docs/next_gen_emogpt.md` R1–R20 + R-PE
- **必读伴侣**：Hoffmann et al. 2022「Chinchilla」（arXiv 2203.15556）—— **本文的修正者**。任何引用 Kaplan 2020 时必须同时声明 Chinchilla 的修正

> **本笔记的写法**：本文是 LLM 工程的「热力学」——它不告诉你怎么造一个东西，它告诉你**任何按这个范式造出的东西，性质会满足什么定律**。本笔记分五部分：
> 1. **核心定律的精确陈述**（不被科普版本污染）
> 2. **历史地位 / 谱系**（GPT-3 → Chinchilla → 现代 frontier 的来龙去脉）
> 3. **Kaplan vs Chinchilla 的精确分歧**（**最重要部分**）
> 4. **对 VZ 的"基底契约"映射**（R2 / R8 / R10 / R12 的源头根据）
> 5. **scaling laws 不能告诉 VZ 的事**（必须靠 R-IDs 补的洞）

---

## 1. 核心定律的精确陈述（不被科普版本污染）

### 1.1 三条 power law（论文 §1.2 Eq. 1.1–1.3）

设：
- `N` = 模型非嵌入参数量（**关键**：去掉 `n_vocab · d_model` 和 `n_ctx · d_model`）
- `D` = 数据集 token 数
- `C ≈ 6 · N · B · S` = 训练计算量（PF-days；6 = 2 × 3，2 是 mac, 3 是 fwd+bwd）
- `C_min` = 在 `B << B_crit` 极限下达到给定 loss 所需的最小 compute（**这才是干净 trend 的横轴**）
- `L` = WebText2 测试集上 cross-entropy（nats / token，1024-token 上下文平均）

三条核心定律：

```
L(N) = (Nc / N)^αN          αN ≈ 0.076,   Nc ≈ 8.8 × 10¹³  (数据无瓶颈，训到收敛)   (1.1)
L(D) = (Dc / D)^αD          αD ≈ 0.095,   Dc ≈ 5.4 × 10¹³  (大模型 + 早停)         (1.2)
L(C_min) = (C_c / C_min)^αC αC ≈ 0.050,   C_c ≈ 3.1 × 10⁸  (compute-optimal)      (1.3)
```

**跨越 7 个数量级**（compute）/ **6 个数量级**（参数）/ **2 个数量级**（数据）"无偏离 trend 的迹象"。

### 1.2 联合定律 L(N, D)（论文 §4，Eq. 1.5）

```
L(N, D) = [ (Nc/N)^(αN/αD) + Dc/D ]^αD
```

拟合得 `αN ≈ 0.076, αD ≈ 0.103, Nc ≈ 6.4 × 10¹³, Dc ≈ 1.8 × 10¹³`（论文 Table 2）。

可推出**避免过拟合**的 D-N 关系：

```
D ≳ (5 × 10³) · N^0.74        (4.4)
```

→ **每把模型放大 8x，数据只需放大 ≈ 5x**。这是 Kaplan 2020 最具影响力（也最被后续证伪）的结论之一，详见 §3。

### 1.3 联合定律 L(N, S_min)（论文 §5，Eq. 1.6）

```
L(N, S_min) = (Nc/N)^αN + (Sc/S_min)^αS    αS ≈ 0.76,  Sc ≈ 2.1 × 10³
```

`S_min` 是「在 `B << B_crit` 极限下，达到该 loss 所需的最小步数」。

### 1.4 Compute-optimal 配方（论文 §6，Eq. 1.7–1.8 ★）

> **这是本论文最被引用、也最被误用的部分。**

固定 compute `C`，最优分配：

```
N ∝ C^(αC/αN) ≈ C^0.73
B ∝ C^(αC/αB) ≈ C^0.24
S ∝ C^(αC/αS) ≈ C^0.03   ← 这意味着 serial 步数几乎不增长！
D = B · S ∝ C^0.27
```

**Kaplan 的核心建议**：compute 增长应该**主要砸到模型大小**上，数据**只需缓慢增长**（`D ∝ C^0.27`），训练**远未收敛即停止**。

### 1.5 关键经验事实（论文散见各节）

| # | 事实 | 节 |
|---|---|---|
| 1 | **架构形状几乎不重要**：固定 `N`，aspect ratio `d_model/n_layer` 在 1× ~ 40× 范围内 loss 仅差几个百分点；`d_ff/d_model`、`d_attn`、`n_heads` 同样 | §3.1 (Fig. 5) |
| 2 | 必须**排除 embedding** 才能看到干净 trend；包含 embedding 时 trend 被层数污染（embedding 表是个独立 scaling 维度） | §3.2 (Fig. 6) |
| 3 | **跨数据分布泛化 = 训练集 loss + 常数偏移**（与训练 phase 无关，仅取决于 in-distribution loss） | §3.2.2 (Fig. 8) |
| 4 | **大模型 sample-efficient**：达到同一 loss 用更少 step、更少 data | §6 (Fig. 2) |
| 5 | **Critical batch size** `B_crit(L) ≈ B* / L^(1/αB)`，仅取决于 loss、不取决于模型大小（每 13% loss 下降，B_crit 翻倍） | §5.1 (Fig. 10) |
| 6 | **Transformer 全程优于 LSTM**，且 LSTM 在 token index >100 后停止改进；Transformer 利用整个 1024 上下文 | §3.2.1 (Fig. 7) |
| 7 | 论文自己**预言**：`L(C_min)` 与 `L(D)` 的延伸会在某个 compute 处相交，"intersection 点可能是 Transformer LM 的最大可达性能" | §6.3 (Fig. 15) |

### 1.6 论文自己列出的 caveat（§C，常被忽略）

1. 损失函数与样本平均：所有"loss 都是 cross-entropy"——**不能直接外推到非 NLL 目标**
2. 仅在**WebText2** + BPE 词表上拟合；`Nc, Dc, C_c` 数值依赖词表
3. 仅在**预训练**上观察；finetune / RLHF / instruction tuning 的 scaling 行为**未涵盖**
4. 仅在**有限学习率/优化器组合**上验证；新 optimizer 可能改变 trend
5. 形状无关性的结论**可能被 1-layer / 极端形状打破**

> **这条 caveat list 本身**就是后续 Chinchilla 论文得以修正其结论的合法窗口。

---

## 2. 历史地位 / 谱系

### 2.1 它在 LLM 范式建立中的精确位置

```
2014  Sutskever Seq2Seq         ← 神经端到端范式（substrate 形状）
2017  Vaswani Transformer       ← 当代 substrate 架构定型
2018  Radford GPT-1             ← Decoder-only + 预训练 + finetune
2019  Radford GPT-2 (1.5B)      ← "scaling 似乎一直 work" 的非定量直觉
2020 ★ Kaplan Scaling Laws      ← 把"scaling work"变成定量、可外推的物理定律
2020  Brown GPT-3 (175B)        ← Kaplan 定律的 first major 验证 + emergence of in-context learning
2021  Anthropic founded         ← Kaplan / McCandlish / Brown / Amodei 离开 OpenAI 创立
2022 ★ Hoffmann Chinchilla      ← 修正 Kaplan：参数:数据 ≈ 1:20 (token/param)，而非 Kaplan 隐含的 1:1.7
2022→ PaLM/GPT-4/Claude/Llama   ← 全部按 Chinchilla 比例训练（或更 over-trained）
2024→ scaling 减速讨论          ← Kaplan §6.3 预言的 intersection 是否到了
2026  VZ                        ← R2 把这条链的 substrate 冻结，把"剩下的"放回 controller layer
```

**关键认识**：Kaplan 2020 的真正贡献**不是**那 3 条公式（exponent 后来被改），而是建立了「**substrate 的行为可以被预测**」这件事本身。这条**元事实**比任何具体 exponent 都重要 100 倍。

### 2.2 GPT-3 的"奠基决策"几乎全部源自这篇论文

GPT-3 论文（2020-05，Brown et al.）实际上是 Kaplan 2020 的「实战验证回报」：

| GPT-3 决策 | Kaplan 2020 提供的根据 |
|---|---|
| 参数推到 175B | §6 "scale predominantly N" |
| 训练 token 仅 300B（每参数 1.7 token） | §6.1 D ∝ C^0.27 + §4.4 D ≳ 5×10³ N^0.74 |
| 不调架构，只 scale | §3.1 形状几乎无关 |
| 不 finetune（in-context learning） | §3.2.2 generalization tracks training loss |
| 单 epoch 训完即停 | §6 "stop significantly short of convergence" |
| 选 175B 这个具体数字 | C 预算 + Eq. 1.7 反推 |

> GPT-3 的成功被 Kaplan 论文**全部预言**了。Anthropic 创始人圈（Kaplan / McCandlish / Brown / Amodei / Henighan）就是这条预言的兑现者。

### 2.3 后续被验证的"超大尺度"延伸

- **OpenAI scaling**（2020–2023）：基本沿用 Kaplan 配方，但 GPT-4 已**显著 over-train**（推测每参数 30+ token）
- **DeepMind Chinchilla**（2022）：在 70B 上证明 1.4T token 训练**优于** 280B 上 300B token 训练（同 compute）→ Kaplan 严重 under-train 数据
- **Llama 1/2/3**：每代每参数 token 数从 1.4T/7B → 2T/7B → 15T/8B，**远超 Chinchilla 比例**（"deployment-optimal" 而非"compute-optimal"）
- **现代 frontier**：基本一致认为 Kaplan exponent **`αN` 偏高**，正确值更接近 `αD ≈ αN`

---

## 3. Kaplan vs Chinchilla 的精确分歧（★ 最重要部分）

> **任何引用本文时必须同时给出此节。** 否则会给读者错误的"compute-optimal"印象。

### 3.1 两者的核心结论对比

| 议题 | Kaplan 2020 | Hoffmann 2022 (Chinchilla) |
|---|---|---|
| `αN`（loss vs N） | 0.076 | 0.34 |
| `αD`（loss vs D） | 0.095 | 0.28 |
| Compute-optimal `N(C)` exponent | **C^0.73** | **C^0.50** |
| Compute-optimal `D(C)` exponent | **C^0.27** | **C^0.50** |
| Token-per-parameter（compute-optimal） | ~1.7（隐含） | **~20** |
| 训练步数 vs compute | S ∝ C^0.03（几乎不变） | S 显著增长 |
| 推荐策略 | **重模型轻数据，远未收敛即停** | **模型与数据等比例 scale** |
| 70B 案例（同 compute） | 推荐 ≈ 280B / 300B token | 推荐 70B / 1.4T token（Chinchilla-70B 实测胜出） |

### 3.2 分歧的原因（Hoffmann 2022 的诊断）

Chinchilla 论文给出 3 个原因，解释为什么 Kaplan 的 exponent 偏离：

1. **Kaplan 用了固定 cosine schedule，没有按 D 调整 lr decay 长度**
   - 当数据量变化时，lr 衰减时长不匹配 → 小数据点的训练**实际上不收敛**到该参数下的最优 loss
   - Chinchilla 让 cosine schedule 长度匹配实际 token 数，得到清洁 fit
2. **Kaplan 拟合区间太窄**：N ≤ 1B，外推到 100B+ 不可靠
3. **Kaplan 排除 embedding 参数**的方式让小模型显得 "更高效"——而 frontier 部署都看 total params

### 3.3 真实情况：**双方都不完全对**

- **训练 compute 视角**：Chinchilla 更准确
- **推理 cost / 部署视角**：Llama 等"over-train" 路线（30+ token/param）更优——因为部署时只关心模型大小
- **当代实践**：完全 Chinchilla-optimal 的训练**已经过时**；frontier 走"deployment-optimal"（远远 over-train）

### 3.4 给 VZ 的元教训（**至关重要**）

> **任何 power-law 拟合都依赖于「整个训练流水线在 measurement window 内是 well-tuned 的」。**

Kaplan 2020 的 exponent 在他自己的训练设置下成立，但**外推假设了**：
- 学习率 schedule 自动正确缩放
- 数据质量恒定
- optimizer 行为在新 scale 下不变
- 评估分布与训练分布一致

任何一条违反都让 exponent 漂移。

**VZ 的引申**：
- VZ 的任何"基底行为可预测"假设必须**被持续验证**，不能 assume 一次拟合永远成立
- substrate 升级（rare-heavy artifact refresh）必须**重新 fit** 行为契约，不能盲信旧 scaling exponent
- 这正是 R10 ModificationGate 与 R12 评估只读契约的**经济学根据**

---

## 4. 对 VZ 的「基底契约」映射

### 4.1 R-ID 映射表

| R-ID | 映射方向 | 解读 |
|---|---|---|
| **R2** Stable Substrate + Adaptive Controllers | **R2 决策的物理学根据** | scaling laws 证明 substrate 行为是**(N, D, C) 的函数** —— 一旦你为 substrate 付了 (N, D, C) 的代价，它的能力**几乎可以被准确预测**。在线突变这个 substrate = 撕毁这个可预测性契约。**因此 R2 把 substrate 训练放回 rare-heavy 离线轨道是经济理性选择，不是工程偏好** |
| **R8** Snapshot-First, Contract-First | **scaling laws 是契约的样板** | Kaplan 2020 整篇论文 = **关于 substrate 给出什么的契约**：「给我 N、D、C，我返回 L = f(N,D,C)」。VZ R8 把契约从单一 scalar `L` 扩展到**多 slot 的 typed snapshot**——但形式同构 |
| **R10** Self-Modification Gated | **gating 的经济学根据** | scaling laws 隐含：substrate 的"质量"是 N/D/C 投资的函数，online tweak 一个已经 fit 完的 substrate **几乎一定 net negative**（noise > signal），因为你没法用一次 turn 的数据 dominate 已经 amortize 的 D/C 投资。R10 把这条直觉**编码成 invariant** |
| **R12** Evaluation Beyond Task Success | **反例 / 教训** | scaling laws 只测 **L = cross-entropy**——单一 IQ 维度的"task success"。VZ R12 6 族评估正是因为意识到「**预训练 loss scaling 不能预测关系连续性 / 信任 / EQ**」而设计。**Loss 收敛 ≠ 行为成熟** |
| **R1** Multi-Timescale Learning | **空白处的根据** | scaling laws 只描述**rare-heavy 那一层**（offline pre-train）。它**完全没有**说 online-fast / session-medium / background-slow 该有什么定律。**这正是 VZ 必须自己定义的**——R1 不是对 Kaplan 的延伸，而是对 Kaplan **没说的**层填空 |
| **R3 / R4** Temporal Abstraction & Internal RL | **scale 之上的 free lunch 已耗尽** | Kaplan §6.3 Fig. 15 自己预言了 substrate scaling 会撞 intersection。当 IQ 轴的 scaling free lunch 边际递减后，**进一步能力来自 substrate 之上的结构化控制层**——VZ 的 z_t/β_t 内部 RL 正是这个判断的工程实现 |
| **R-PE** Prediction Error Primitive | **PE 与 loss 是同一对象的两个视角** | Kaplan 用的 cross-entropy = next-token 的 negative log-likelihood = **token-level PE 的对数期望**。Scaling laws 的"loss"本质是 **PE 的 readout**。VZ R-PE 把 PE 升级为**多层、多类型、跨时间尺度**的一级对象 |
| **R5 / R6** Memory & Reflection | **完全空白** | scaling laws **完全不涉及** test-time memory、episodic state、reflection。它假设每次 forward 都是 isolated。VZ 的 CMS 4 层是对这个假设的根本性扩展 |
| **R7 / R14** Dual-Track / Regime | **完全空白** | scaling laws 没有"自我"/"关系"/"regime"概念。它只评估单 turn 的 next-token loss。**Loss 再低也不保证一个有持续身份的存在** |

### 4.2 关键洞察：「scaling 已结束 / 还没结束」之争对 VZ 意义

公开讨论里近年常见的"scaling 是否到顶"争论：

- **乐观派**（OpenAI 主流）：还远没到 Kaplan §6.3 的 intersection；按 Chinchilla 配比再 scale 10× 仍有显著收益（GPT-5 内部数据）
- **悲观派**（部分 academia + Anthropic interpretability 圈）：当前 frontier 已进入边际递减，剩余增益主要来自 RLHF / RL on reasoning，而非 substrate scaling

**VZ 的立场**（从 R2 + R3 + R4 推出）：
- **不需要赌哪一边对**。VZ 的 R2 选择把 substrate **当作 given**（不管它还能 scale 多少）
- 然后**所有进一步能力**通过控制器层（R3/R4）、记忆层（R5/R6）、反思（R6）、关系/regime 持久性（R7/R14）来获得
- 这条路径**对 substrate scaling 是否到顶不敏感**——上面任何 substrate 都能跑

> **这是 VZ 路线的"风险对冲"性质**：scaling 还有空间 → VZ 直接受益（基底变好）；scaling 到顶 → VZ 提供的 controller-level 价值变得**唯一**重要。两种情景下 VZ 都不输。

---

## 5. Scaling Laws 不能告诉 VZ 的事（必须靠 R-IDs 补的洞）

| 缺失 | 为什么 scaling laws 不可能告诉 | VZ 用什么补 |
|---|---|---|
| **如何在 turn 内适应** | scaling laws 描述训练过程，不描述 inference-time behavior | R3/R4（latent space control + Internal RL on z_t） |
| **如何跨 session 持久学习** | substrate 是 frozen black box，无 session 概念 | R5/R6（memory continuum + reflection） |
| **如何区分 task / relationship 学习** | 单一 cross-entropy loss，无 track 划分 | R7（dual-track） + R14（regime persistence） |
| **如何评估"存在"质量** | 只有 perplexity / BLEU 类 task metric | R12（6 族评估，含 relationship continuity / safety / abstraction quality） |
| **何时允许 substrate 修改** | scaling laws 假设训练→部署一次性 | R10（gated modification + offline-only substrate update） |
| **多对象 / 社会认知** | 输入只是字符串，无对象概念 | R16-R20（multi-party identity / ToM / common ground / group） |
| **PE 的多层化** | Loss 只在 token level，全部聚合 | R-PE + R9（hierarchical credit on prediction error） |
| **结构化内部状态** | substrate 内部状态是黑盒激活 | R8/R11（owner snapshots + named runtime state） |
| **多时间尺度交替** | 单一训练循环 | R1（4 个时间尺度） + R13（compression-reinforcement alternation） |

---

## 6. 借鉴可落地点

### 6.1 把 scaling laws 当作 substrate 行为 SLA

VZ 在引入新 substrate（GPT-5 → Claude 5 → 自训 substrate）时，可以**显式地**测量该 substrate 在自己工作分布下的：

- `L(N=该模型, D=VZ 评估集, C=N/A)` baseline
- 一族**sub-task curves**：task / relationship / EQ / safety 各自的 loss 趋势

并把这些 baseline 写进 `docs/specs/contract-runtime.md` 的 "substrate readiness" 部分，作为后续 controller 层 / 评估层的参考线。**这本质是 R8 在 substrate-controller 边界的具体化**。

### 6.2 把 Kaplan vs Chinchilla 教训作为 ModificationGate 的 invariant

任何 VZ 的离线 substrate refresh 流水线（rare-heavy artifact training）必须满足：

- **lr schedule 长度匹配实际 token 数**（Chinchilla 教训）
- **embedding 参数明确报告**，不偷偷做"non-embedding only" framing
- 拟合外推**必须有显式置信区间**，不允许 trend-line 单点外推 5 个数量级

写进 `docs/specs/credit-and-self-modification.md` 的 "rare-heavy refresh checklist"。

### 6.3 形状无关性的 lesson 用于 controller 设计

Kaplan §3.1 的"form-shape 几乎无关"在 substrate 上成立，但这**并不能**直接外推到 VZ controller 层。然而它给出一个**默认假设**：

> **优先 scale 而非调架构**

应用：VZ metacontroller 的 `z_t` 维度、`β_t` 切换粒度等超参数，**应优先**通过"加大 controller 尺寸 + 让 SSL 找出结构"，**而不是**手工搜索几十种 metacontroller 拓扑。这与 R3 "switching must emerge, not be hardcoded" 完全一致。

### 6.4 跨数据分布泛化 = 训练分布 + 常数偏移

Kaplan §3.2.2 的发现「**generalization tracks training loss with constant offset**」对 VZ R12 评估很重要：

- **不要**试图通过"在评估分布上 fine-tune"来提高 VZ 表现——这只会破坏训练分布与评估分布的可预测线性关系
- 应该**让评估分布本身代表真实 deploy 分布**，然后只在训练分布上优化
- 这条契约写进 `docs/specs/evaluation.md` 的 "evaluation set design" 部分

### 6.5 Critical batch size 概念延伸到 VZ 在线适应

Kaplan §5.1 的 `B_crit(L) ≈ B*/L^(1/αB)`（每 13% loss 下降 B_crit 翻倍）的精神是：**优化效率有 inherent batch-size sweet spot**。VZ 的在线适应模块（reflection / controller micro-update）应该意识到：

- 在 background-slow 反思阶段，"一次反思整合多少 turn"有 **B_crit 类的 sweet spot**——太少会 noisy（PE 噪声主导），太多会 stale（信息过期）
- 这条 sweet spot 不是"越大越好"也不是"越小越好"——是 task 损失 metric 的函数

可以作为 `docs/specs/multi-timescale-learning.md` 的反思节奏设计参考。

---

## 7. 技术等级评估

| 维度 | 当时分（2020） | 现在分（2026 视角） | 备注 |
|---|---|---|---|
| 工程深度 | 5/5 | 5/5 | 数百次训练 run 跨 7 个数量级 compute；optimizer / batch / lr schedule 精细控制 |
| 理论新颖度 | **5/5** | 4/5 | 「substrate behavior 可被定量预测」是范式级洞察；具体 exponent 后被修正 |
| 规模壁垒 | 5/5 | 3/5 | 当时只有 OpenAI 能做；今天 Llama / Pythia 等开源工作已把 scaling 拟合民主化 |
| 复现难度 | 5/5 | 2/5 | open suites（Pythia, OLMo）已提供等价数据；scaling fit 工具开源（如 nanoT5 scaling toolkit） |
| **VZ 借鉴价值** | — | **5/5（决策根据）+ 5/5（教训根据）** | R2 / R8 / R10 / R12 的**经济学/物理学根据**。Kaplan vs Chinchilla 的修正本身就是一份关于"如何阅读 scaling 论文"的范式 |

---

## 8. 引用建议（写进 VZ specs 时的标准引法）

任何 VZ spec 在论及"为什么 substrate 必须冻结 / 为什么 substrate 行为可作为契约 / 为什么 online substrate mutation 经济上不合理"时，可以引：

> Kaplan J., McCandlish S. et al. **Scaling Laws for Neural Language Models**. arXiv:2001.08361, 2020.
>
> 该论文确立了 substrate 行为是 (N, D, C) 的可预测函数。VZ R2 把 substrate 训练放回 rare-heavy 离线轨道，因为 (a) substrate 能力的代价已在该次 scaling 投资中 amortize 完毕，(b) 在线 substrate 突变破坏 scaling laws 提供的可预测性契约。
>
> **修正引用**：Hoffmann J. et al. **Training Compute-Optimal Large Language Models** (Chinchilla). arXiv:2203.15556, 2022. 修正了 Kaplan 关于 compute-optimal 配比的结论：参数:数据应大致 1:20 token/param，而非 Kaplan 隐含的 ~1:1.7。VZ 在讨论 substrate scaling 时**必须同时引用两者**。

参见伴侣笔记：
- `01_seq2seq_sutskever_2014.md` —— 关于 substrate 架构血统的另一半基础
- `00_README.md` —— 两篇基底起源论文与 VZ R2 的整体关系

---

## 附录 A：与 docs/next_gen_emogpt.md Part 2 的关系

`docs/next_gen_emogpt.md` 的 R2 原文：

> "ETA's rate-distortion analysis demonstrates that freezing the base model is necessary for discovering temporal abstractions — joint training leads to degenerate solutions."

这是 R2 的**信息论根据**。Scaling Laws 提供的是 R2 的**经济学根据**：

| 论证类别 | 论文 | 核心结论 |
|---|---|---|
| 信息论 | ETA (Kobayashi/von Oswald 2025) | Joint train 破坏 z_t 涌现，必须冻结 substrate |
| 经济学 | Kaplan 2020 + Chinchilla 2022 | Substrate 能力已在 (N,D,C) 上 amortize；online tweak 不可能在 turn-level 数据量上 dominate 这个投资 |
| 系统论 | NL (Behrouz/Razaviyayn 2025) | 不同时间尺度必须有清晰 update boundary，否则 collapse 成单梯度流 |

**三个根据彼此独立但结论一致**——这是 VZ R2 不会被单一论文翻译反转的稳健性来源。
