# 01 · 论文深读：End-to-End Test-Time Training for Long Context（TTT-E2E）

> Tandon, A., Dalal, K.\*, Li, X.\*, Koceja, D.\*, Rød, M.\*, Buchanan, S., Wang, X., Leskovec, J., Koyejo, S., Hashimoto, T., Guestrin, C., McCaleb, J., Choi, Y., Sun, Y.\*
> arXiv [2512.23675](https://arxiv.org/abs/2512.23675)（2025-12）。机构：Astera Institute、NVIDIA、Stanford、UC Berkeley、UC San Diego。
> 通讯：arnuv@stanford.edu、kdalal@berkeley.edu、yusun@cs.stanford.edu。
> 官方代码（JAX）：[test-time-training/e2e](https://github.com/test-time-training/e2e)。

---

## 0. 证据边界声明（先读这个）

1. 本深读基于 **arXiv HTML v1 全文**（含 Appendix A–D、作者贡献声明、致谢、110 条参考文献），2026-08-13 抓取，全文已通读。
2. PDF 已于 2026-07 持续学习横扫时归档：[`../papers/continual-learning-2607/end-to-end-test-time-training-long-context-2512.23675.pdf`](../papers/continual-learning-2607/end-to-end-test-time-training-long-context-2512.23675.pdf)（1,009,536 bytes，SHA-256 `e1390e93…b7ec7a`，完整值见 [`download-summary.md`](download-summary.md)）。本包不重复下载。
3. 官方 JAX 代码库存在性已核验（646 stars，标题 "Official JAX implementation"），**未克隆逐文件核对**。文中所有定量描述均出自论文正文与表格文字；图（Figure 1/4/5/6/7/8/9）只核验了图题与正文转述，未核验像素。
4. 本篇在本仓库已有段落级记录：[`../continual-learning-2026-07/01_LANDSCAPE.md` §S6](../continual-learning-2026-07/01_LANDSCAPE.md) 与 [`../continual-learning-2026-07/02_VZ_DELTA.md` §F](../continual-learning-2026-07/02_VZ_DELTA.md)。本包是专项深读，**不推翻既有裁决，只做增量**（见 [`02_VZ_IMPLICATIONS.md`](02_VZ_IMPLICATIONS.md)）。

---

## 1. 团队与谱系：一作是 Tandon，思想主人是 Yu Sun

作者贡献声明写得异常坦白，值得先读：

- **Yu Sun** 是 project lead：提出 TTT-E2E 的想法、设计早期实验、制定实验协议、**撰写论文**。他是整条 TTT 路线的发起者（2020 TTT with self-supervision → 2023 *Learning to (Learn at Test Time)* `2310.13807` → 2024 TTT-RNN `2407.04620`）。
- **Arnuv Tandon 与 Karan Dalal**（共同一作）主导 scaling 实验、开发 codebase、跑最终实验、管集群。项目 2024-10 启动，Tandon 2024-11 加入；中途多名核心成员离开又回归（Dalal/Koceja 离开 2024-11 至 2025-03），2025-09 后由 Tandon/Dalal/Sun 收尾。
- 算力来自 GB200 集群（Voltage Park）+ Hyperbolic Labs / SF Compute，Tandon 与 Dalal 还动用了 Y-Combinator 的 compute credits——这是一篇**机构联盟 + 创业算力**拼出来的论文，不是大厂内部产物。

**谱系定位**（全部与本仓库既有档案交叉核对过）：

| 谱系线 | 代表工作 | 与本篇关系 |
|---|---|---|
| Dynamic evaluation | Mikolov（RNNLM 时代）、Krause 2018 | 本篇 §2.1 的直接前身；缺训练时准备（非 E2E），本篇称之为 TTT-naive |
| Fast weights / FWP | Hinton & Plaut 1987；Schmidhuber 1992 | 内循环权重 = fast、外循环参数 = slow；本篇自认是 FWP 特例 |
| **Clark et al. 2022**（`2212.02475`，方法学最近亲） | Meta-learning fast weight LMs | 同样"chunk 级 NTP 梯度步 + meta 训练初始化"，但 fast weights 只挂网络末端、无线性复杂度；本篇证明**交错在 attention 之间**是保住增益的关键 |
| TTT-KVB 谱系 | TTT-RNN `2407.04620`、TTT done right `2505.23884`、**MesaNet `2506.05233`、Titans `2501.00663`、Nested Learning `2512.24695`** | 本篇的对立面与出发点：KV Binding 是这一族的核心组件，本篇裁决其**对长上下文语言建模非必需**（§4） |
| Meta-learning | MAML 2017 | 同为"梯度的梯度学初始化"；差异在问题设定——MAML 内循环学整个数据集、外循环需要数据集的集合，本篇内循环学单条序列、任何监督学习问题皆可套入 |

小疵一处：§4.2 说 dynamic evaluation "pioneered by Mikolov et al. [72]"，但 HTML v1 的 [72] 指向 2013 word2vec 论文，疑为引注错位（dynamic evaluation 出自 Mikolov 2010–2012 的 RNNLM 系列）。低风险，不影响任何结论。

---

## 2. 核心主张：一次框架转换

开篇第一句就是全文最大的赌注：

> "We formulate long-context language modeling as a problem in **continual learning** rather than **architecture design**."

据此，架构上只用标准 Transformer + 滑动窗口注意力（SWA），不发明新序列层。模型在测试时对给定上下文做 next-token prediction（NTP）继续训练，**把读到的上下文压进权重**；训练时用 meta-learning 为"测试时会被更新的初始化"做准备。"E2E"是双重的：

- **测试时 E2E**：内循环直接优化网络末端的 NTP 损失——对照先前长上下文 TTT 的层级 KV-binding 代理目标；
- **训练时 E2E**：外循环直接优化 **TTT 之后**的最终损失（梯度的梯度）——对照 dynamic evaluation 只训静态模型再临时更新。

摘要口径的结果：3B 模型、164B tokens，**随上下文长度的 scaling 与 full attention 相同**（Mamba 2、Gated DeltaNet 做不到）；同时推理延迟与上下文长度无关，128K 时比 full attention 快 **2.7×**（H100 prefill）。

---

## 3. 方法逐层拆解

### 3.1 玩具起点：去掉全部 attention 的 Transformer

为隔离 TTT 本身的效应，toy 架构把 Transformer 的 self-attention 全部拆掉（只剩 MLP 块）——它退化为 bigram（对上文零记忆）。TTT 的补救：在每个位置 \(t\) 上按标准 NTP 损失

\[\ell_t(W)=\texttt{CE}(f(x_{t-1};W),\,x_t)\]

做在线梯度下降 \(W_t=W_{t-1}-\eta\nabla\ell_t(W_{t-1})\)，最后用 \(W_T\) 预测 \(x_{T+1}\)。上文信息经由梯度写进权重。

### 3.2 训练时 E2E：让初始化为"将被更新"做准备

关键观察：TTT 的逐步训练损失 \(\ell_t(W_{t-1})\) **本身就是**条件于 \(x_{0..t-1}\) 预测 \(x_t\) 的测试损失。于是整条序列的测试损失

\[\mathcal{L}(W_0;X)=\tfrac1T\textstyle\sum_t \ell_t(W_{t-1})\]

直接成为训练目标——训练损失与测试损失完全一致，这就是 E2E。对照组 TTT-naive 用 \(\sum_t\ell_t(W_0)\) 训练（静态模型的标准预训练），测试时才临时做 TTT——这是 dynamic evaluation 文献的主流做法。toy 实验（Fig 2 右）：TTT-naive 只比 bigram 基线略好，**TTT-E2E 几乎追平 full attention**。训练时优化 \(\mathcal{L}\) 需要梯度的梯度（外循环/内循环即 meta-learning 术语），现代自动微分框架可低开销支持。

### 3.3 mini-batch TTT + 滑动窗口：解决并行与稳定

在线单 token 梯度（\(b{=}1\)）两个致命伤：内循环步骤无法并行；单 token 梯度**容易偶然爆炸**。解法是标准的 mini-batch 化：每 \(b\) 个 token 求平均梯度走一步。但这引入新问题——批内又变回 bigram（批内后段 token 的预测拿不到批内前文）。最终解法：**把 toy 的"无 attention"放宽为"滑动窗口 attention"**，并要求 **\(k \ge b\)**——窗口在 TTT 来得及更新权重之前，兜住批内上下文。主结果配置：\(T{=}128\)K，\(k{=}8\)K，\(b{=}1\)K。

### 3.4 三个实现细节（论文明言"缺一不可，但可能是本设定的 artifact"）

1. **只更新 MLP 层**：embedding、normalization、attention 全部冻结——在内循环更新 attention 会导致**外循环不稳定**。
2. **只更新最后 1/4 的 blocks**：更新层数 = 压缩存储容量，但反传要穿过其上所有层——收益/成本有明确拐点（消融见 §5.3）。
3. **每个被更新的 block 加一个静态第二 MLP**：作为预训练知识的"安全存储"防遗忘；为公平对比，全网 MLP 隐层维度同步缩小以保持总参数量不变。——注意：这是 TTT 方法**自己内部**都需要一条冻结通道，见 [`02_VZ_IMPLICATIONS.md`](02_VZ_IMPLICATIONS.md) §5。

### 3.5 多 token 解码 = 测试时自训练

decode 阶段的自然扩展：解码的 token 攒满一个 TTT mini-batch（\(b\) 个）才走一步梯度，然后用更新后的权重继续解码。也就是说**模型在自己的生成物上继续训练**（有效性验证见 §5.6）。

---

## 4. 备选推导：从 TTT-KVB 三步走到 TTT-E2E（本篇的谱系级裁决）

§2.4 是全文对同行最有杀伤力的部分：从 TTT-KVB（`2407.04620` / `2505.23884`，即 MesaNet、Titans、Nested Learning 共用的核心构造）出发，三步改造后恰好得到 TTT-E2E，每一步的损益都有数字（760M、DCLM、8K，论文自设显著性阈值 0.001）：

| 步骤 | 方法 | Loss | 相对变化 |
|---|---|---|---|
| 参照 | SWA（\(k{=}8\)K）baseline | 2.827 | — |
| 起点 | TTT-KVB（Zhang et al.） | 2.818 | −0.009 |
| 第一步：简化输出规则（复用 \(g\) 的预测、去掉 \(\theta_Q\)） | TTT-KVB simplified | 2.819 | +0.001（无损） |
| **关键步：层级 KVB 重建损失 → 网络末端 NTP 损失**（\(\theta_K/\theta_V\) 随之消失） | TTT-E2E all layers MH | 2.806 | **−0.013** |
| 终步：多头小 MLP + LoRA → 常规大 MLP、只更新后 1/4 blocks | TTT-E2E（本篇） | 2.805 | −0.001 |

三个要点：

1. **KVB 谱系裁决**：KV Binding——"从 key 预测 value"的层级自监督目标，MesaNet / Titans / Nested Learning 的核心组件——被替换成朴素的终端 NTP 损失后**显著更好**。论文还给了统一视角（footnote 1）：\(g\) 取线性即 DeltaNet，梯度对 \(W_0\) 取即 linear attention，核估计即 self-attention——整个现代线性注意力/RNN 家族都是 KVB 的特例，而本篇绕开了这一整族。
2. **对偶结构**：主推导从"测试时 E2E"（NTP TTT）出发补上"训练时 E2E"（meta-learning）；备选推导从"训练时 E2E"（KVB 的外循环）出发补上"测试时 E2E"（换损失）。两条路汇合于同一方法。
3. **大状态 + 少计算的经济学**：E2E 损失下反传要先穿过上方大 MLP 才能碰到小 fast-weight——既然上游梯度的成本已经付了，就应该**少更新几个 block、每个 block 用更大的状态**。终步使 760M 模型的隐状态从 18M 涨到 **88M（5×）**，prefill 延迟反而减半（0.0086 vs 0.017 s/1K tokens，H100）。TTT-E2E 由此可以被看作**只有一层的 RNN**：隐状态就是整段被更新的权重，一次反传更新全部。

---

## 5. 实验

### 5.1 设定

两阶段训练模拟生产流程：8K 上下文在 **DCLM-Baseline** 预训练（丢弃全部 <8K 的文档——footnote 4：避免打包序列跨文档边界时重置被更新的 MLP，重置会拖慢他们的基础设施；**这句话同时暴露了 fast weights 的生存期语义：逐序列，不跨文档**）；再在 **Books**（Pile Books）上做长度扩展微调（16K–128K），在 Books 留出集上评测。五档模型（125M/350M/760M/1.3B/2.7B，正文称 3B），GPT-3 配方 + Mamba 2 的两处修正，Chinchilla 预训练 token 数，扩展微调用预训练 token 的 5%（LR 统一 4e-4，RoPE θ 按 log-linear 从 500K 升到 128K 时的 10M），Llama 3 tokenizer + QK norm。

### 5.2 基线（6 个）与对基线的加强

Full attention / SWA / 5:1 混合（Gemma 式）/ Mamba 2（Nemotron-H 用的混合版）/ Gated DeltaNet（Kimi Linear 用的混合版）/ TTT-KVB。前三个作者自己用 JAX 实现，后三个用官方 PyTorch 代码**并咨询原作者做了两处加强**：FlashAttention 2 → 3；QK norm 补给所有带 SWA 的基线（GDN 预训练 loss 2.814→2.809、32K 微调 2.691→2.683）。——把基线加强而不是削弱，这是实验可信度的加分项。

### 5.3 消融（760M）

- **窗口 \(k\)**：越大越好，TTT-E2E 对 \(k\) 的敏感度与 SWA/GDN 相当。**关键对照**：预训练长度即 8K，故 \(k{=}8\)K 时 SWA ≡ full attention，此时 TTT-E2E 仍降 loss **0.018**，且优势不随 \(k\) 增大而消失——**TTT-E2E 不是在弥补 SWA 与 full attention 的差距，而是正交增益**。
- **mini-batch \(b\)**：1K→8K 单调变差（TTT-KVB 同样）；\(b<1\)K 则硬件利用率与稳定性差到"难以实验"。取 \(b{=}1\)K。
- **无 TTT 对照**：\(b{=}8\)K 等价于关闭 TTT，此时 TTT-E2E 2.825 ≈ TTT-KVB 2.826 ≈ full attention 2.827——**架构改动本身贡献≈0，全部增益来自测试时学习**（"architecture design plays a minor, supporting role in our method"）。
- **更新层数**（最重要的消融）：更新最后 1、3 层 → **不随上下文 scale**；6、12 层 → scale，且 12 ≈ 6。取 1/4（760M 的 24 层中的 6 层）。状态容量存在下限阈值，过了阈值边际收益消失。

### 5.4 随训练算力 scaling：优势在小预算区缩水，在大预算区稳定

模型尺寸与训练 token 两条轴上同一模式：小预算区 TTT-E2E 对 full attention 的优势随算力增加**明显缩小**；过了边界（约 **760M 参数 / 48B tokens**）后与 full attention 同斜率。Gated DeltaNet 呈完全相同的趋势。作者给两种解释：TTT-E2E 本质也是（单层大状态的）hybrid RNN，固定尺寸隐状态的方法共享此趋势；或者说小算力区是 full attention 自己欠训练（Transformer 在小算力下不如 RNN 是已知现象）。另有两条坦白的 anecdote：换 Llama 3 tokenizer 使 3B 的优势扩大约 0.01；预训练数据从 SlimPajama（2023）换 DCLM（2024）后 48B+ 区间的趋势才追平 full attention（FineWebEdu 同样能追平）——**方法对数据质量敏感，且作者自认未系统研究**。

### 5.5 随上下文长度 scaling + 逐 token 分解（正结果的核心）

Fig 1（3B、3× Chinchilla tokens）：8K→128K，TTT-E2E 对 full attention 的 loss 差保持恒定（曲线平），Mamba 2 / GDN / SWA / 混合体全部随长度恶化。Fig 6 逐 token 分解（32K 与 128K）两个观察：

1. **TTT-E2E 是唯一在整条上下文上处处低于 full attention 的方法**；
2. 优势主要来自**前段 token**，窗口尾部与 full attention 接近——128K 的分解曲线像 32K 的"拉伸版"而非"延续版"，所以不存在"再长就交叉"的外推。

最有趣的机制观察：优势在 \(t<1\)K（**第一次 TTT 梯度步之前**）就存在——此时 TTT-E2E 与 full attention 计算图完全相同、只差权重。作者的解释：

> full attention 的权重必须准备好应付上下文窗口内**所有可能的未来**，这很难，因为对所有未来都好会挤占对任何特定未来变好的容量；TTT-E2E 的权重只需要对**当前 mini-batch** 好，因为未来的 token 自有未来的权重去处理。

即 "focus on the present"——TTT 全谱系的核心直觉在受控条件下的最干净一次测量。

### 5.6 NIAH：诚实的负结果

RULER 的三个 S-NIAH 任务、3B、128K 微调模型：

| 方法 | S-NIAH-1 @128K | S-NIAH-2 @128K | S-NIAH-3 @128K |
|---|---|---|---|
| Full attention | **0.99** | **0.86** | **0.64** |
| TTT-E2E | 0.06 | 0.05 | 0.03 |
| （SWA / Mamba 2 / GDN / TTT-KVB） | 0.01–0.07 | 0.05 | 0.03–0.05 |

所有常数状态方法在 16K 以上全线崩塌，TTT-E2E 毫无豁免（也不比其他 RNN 更差）。作者不回避：full attention 的强项就是**近乎无损的回忆**，而 TTT-E2E 的核心机制是**压缩**——会把"看似无关的细节"（needle 的定义特征恰恰是与语料无关）丢掉。**压缩型权重记忆与无损检索是两种能力，不可互替**。

### 5.7 解码长序列：自训练不塌

用 Qwen-3-8B-Base 当外部评估器（512 条序列，Books 8K prefill + 8K 自由解码，temperature 1 / top-p 0.95 / repetition penalty 1.1），TTT-E2E 生成段的 Qwen loss 低于 full attention；作者人工抽查约 20 条样本"合理"。prefill/decode 边界处两种方法的 Qwen loss 都会跳升再回落（Qwen 对被评方法的生成风格从陌生到适应）。作者自认这是 base 模型上的**有限评测**（无 instruction tuning / RL）。

### 5.8 计算效率：推理赢、训练是硬伤

- **Prefill**：128K 常数延迟，比 full attention 快 2.7×（H100）。
- **基础设施论证**（对部署方最有说服力的一段）：TTT-E2E 的隐状态就是常规 MLP 权重，可用**标准训练基础设施跨 GPU 分片，零自定义 kernel**；对照 TTT-KVB 必须用 LoRA 把状态塞进片上内存、Mamba 2 / GDN 必须写自定义 I/O kernel。
- **Decode**：攒满一个 mini-batch 前 = 普通 SWA 解码；攒满后插入一步 prefill 等价的 TTT，总延迟 = 两者之和。
- **训练是明言的限制**：梯度的梯度使 8K 预训练时比 full attention **慢 3.4×**（128K 时反而快 1.2×，H200）；FLOPs/token 恒定但延迟随长度涨，因为 checkpoint-through-time 的量随 \(\log T\) 增长；cuDNN FlashAttention 不支持二阶梯度，用不了。两条出路（都留给未来）：写支持二阶的 attention kernel；**从预训练好的普通 Transformer 初始化 TTT-E2E**，让 TTT 化只占总算力的零头。

---

## 6. Related Work 中值得单独记录的三段

1. **TTT 的定义与人类学习的两个特征**（§4.2）：TTT = "对每个测试实例构造一个（可能不同的）学习问题并求解"。它针对常规持续学习不覆盖的两点：**每个人有一个在自己生命上下文中学习的独立大脑**（对照每小时全网统一微调的 chatbot——任意时刻对所有用户仍是同一个模型）；**人类学习没有 train/test 边界**（今早通勤既是 testing 也是 training）。——这是"个体化持续学习"在主流文献里最清晰的一次命名。
2. **KVB 流行的两个副作用**（§4.2.3）：因为 KVB 目标模仿 self-attention 存 KV，许多人以为长上下文 TTT 是关于**记忆化**的；因为 TTT 层是 self-attention 的 drop-in 替代，许多人以为它是**架构设计**。本篇对两个误读各给一记反驳：无需记 KV 关联（换终端损失更好）、无需改架构（无 TTT 时架构改动贡献≈0）。
3. **三形态 TTT 分类**：近邻检索式（1970s locally weighted regression → KNN-SVM → LLM 上的 `2305.18466`）、新实例自监督式（分布偏移、AlphaProof、ARC 的 Akyurek）、序列式（video → TTT-KVB 谱系 → 本篇）。序列式的分水岭是**不重置**——记忆跨预测存活。

结论段的一句值得原文保留：

> "Adding our method to this baseline induces a hierarchy often found in biological memory, where the weights updated at test time can be interpreted as **long-term memory** and the sliding window as **short-term memory**."

（注意：这个二级层级里没有无损情景存储的位置——NIAH 已经证明它缺这一级；见 [`02_VZ_IMPLICATIONS.md`](02_VZ_IMPLICATIONS.md) §4。）

---

## 7. 批判性评注

**强的地方**：

- 裁决链干净：Table 1 逐步改造每步给数字、无 TTT 对照剥离架构贡献、逐 token 分解定位增益来源、基线主动加强（FA3 + QK norm 反哺 GDN）。
- 负结果自报：NIAH 崩塌、训练延迟 3.4×、数据/tokenizer 敏感性、小预算区优势缩水，全部写在正文。
- "focus on the present" 不是修辞——t<1K 同计算图对照给了机制证据。

**边界与保留**：

1. 全部证据是 **base LM 的 NTP loss + 合成检索 + LLM-likelihood 解码评测**；没有 instruction tuning、没有 RL、没有真实下游任务。"scaling 与 full attention 相同"目前只在语言建模损失意义上成立。
2. **fast weights 逐序列生存**（footnote 4 的重置语义），不跨文档、不跨 session。它是序列内在线压缩机制，**不是持久记忆系统**——不要被"long-term memory"的措辞带偏。
3. 每 mini-batch **无条件更新**：无 gate、无时机选择、无 norm 约束、无回滚。论文自己在 §3.3 末尾指出了这个缺口：RNN 的门控能保护隐状态不被垃圾输入污染，并猜想"TTT on self-generated tokens"（对当前批改写/复述再训）可以起类似作用——**门控被作者列为未来工作，而非已有属性**。
4. 优势的绝对量级不大（8K 预训练时 −0.018；128K 时保持对 full attention 的固定差距），换 tokenizer/数据集就能移动约一半——对配方敏感。
5. Titans / MesaNet / Nested Learning **未被直接评测**，被击败的是其共用构造的代表 TTT-KVB（`2505.23884`）——引用本篇裁决时必须写"同构裁决"而非"直接对打"。
6. 与 Nested Learning（`2512.24695`，本仓库的设计源头之一）的关系是**范式强化、实现弱化**：TTT-E2E 本身就是一个两频嵌套学习实例（慢外循环 meta-优化初始化、快内循环压缩上下文），它打掉的只是 NL 论文里 KVB 这个具体组件的必要性。展开见 [`02_VZ_IMPLICATIONS.md`](02_VZ_IMPLICATIONS.md) §2。
