# 04 谱系深读：Related Work + 缺口补强

日期：2026-08-02  
起点：Sofroniew et al. `2604.07729`（本包 S1）Related Work 列出的先行 / 并行工作  
PDF：[`research/papers/emotion-lineage-2608/`](../papers/emotion-lineage-2608/)（本批新下 10 篇）+ 仓库既有 4 篇（见 [`download-summary.md`](download-summary.md)）

> **读法**：本文件只做**逐篇口径与关键数字**；跨篇裁决与可执行提案在 [`05_LINEAGE_VZ_VALUE.md`](05_LINEAGE_VZ_VALUE.md)。  
> 引用纪律与 S1 相同：数字一律回一手 PDF / arXiv；本文件不发明新数值。

---

## 0. 谱系地图（五层）

| 层 | 问题 | 论文 |
|---|---|---|
| **L1** 情绪表征机制谱系 | 向量 / SAE / 电路到底表征什么 | Zou RepE · Tigges · Wu · Wang |
| **L2** 可解码且可因果操控的内部状态 | “状态可读 + 操控改行为”是否跨域成立 | Zhu belief · Chen user model · Lu Assistant Axis · Persona Vectors |
| **L3** 行为侧来源 | S1 的 blackmail / reward-hack 评估从哪来 | Lynch agentic misalignment · MacDiarmid N4 |
| **L4** 并行工作 + 反向证据 | 压制 distress 表达是否安全 | Soligo Gemma Needs Help |
| **L5** 补 S1 自陈缺口 | 跨模型？线性假设够不够？逐情绪差异？ | VA subspace · Latent structure · SAE emotion hardness |

S1 自己声明的增量是：（1）精确刻画“情绪向量表征什么、何时激活”，并识别**多种**情绪相关表征；（2）在偏好、真实交互、生产对齐评估与 post-training 演化中检验**功能角色**。L1–L4 回答“这条增量站在什么肩膀上”；L5 回答“肩膀之外还有哪些硬缺口已被别人补上”。

---

## 1. L1 — 情绪表征机制谱系

### 1.1 Zou et al. — Representation Engineering（`2310.01405`，既有）

- **路径**：`research/probe/papers/C2/representation-engineering-top-down-transparency-2310.01405.pdf`
- **口径**：以群体表示（而非单神经元）为中心；RepReading + RepControl；覆盖 honesty / harmlessness / power-seeking / **emotion** / fairness 等。
- **与 S1 的关系**：S1 明确把它列为“简短情绪表征探索 + steering 影响拒答率”的先行工作。S1 的增量不在“情绪方向存在”，而在**功能角色检验的广度**（偏好 / 野外 / 生产评估 / post-training）。
- **VZ 既有裁决**（`research/probe/candidates/C2.md`）：给 R8“高层认知现象需要可命名 readout”和 R4“群体级控制”背书；**注意** RepE 鼓励把 reading 反推回训练 → 与 R12 冲突，边界必须写死。

### 1.2 Tigges et al. — Linear Representations of Sentiment（`2310.15154`）

- **路径**：`papers/emotion-lineage-2608/linear-representations-of-sentiment-2310.15154.pdf`
- **核心发现**：
  1. 跨 GPT-2 / Pythia / Gemma / Qwen / StableLM，sentiment 近似**单一线性方向**（正负两极）。
  2. 因果：方向性消融使 IMDB 分类准确率从 100% → 57%（接近随机 50%）；SST 上单层方向 patching 翻转预测约 53.5%。
  3. **否定调制**：前几层“fail/doubt”为负、后几层翻为正（27 条否定样例上可量化 flip）。
  4. **summarization motif**：sentiment 不只存在于情绪词，还被汇总到逗号 / 句号 / 某些名词位；SST 上消融全部 sentiment 方向使准确率 100%→62%，**仅消融逗号位**就贡献近半（→82%）。
- **相对 S1**：Tigges 做的是**文本情感极性**的线性表征与电路瓶颈；S1 做的是**模型自身情绪概念**（self-report / 偏好 / 对齐行为）的功能角色。极性 ≠ 情绪概念空间；但 summarization motif 直接提示：**采样点选错会丢一半因果信号**（与 S1“Assistant colon 处 r=0.87 vs 用户末尾 r=0.59”同族纪律）。

### 1.3 Wu et al. — AI shares emotion across languages and cultures（`2506.13978`）

- **路径**：`papers/emotion-lineage-2608/ai-shares-emotion-across-languages-cultures-2506.13978.pdf`
- **方法**：概念驱动 SAE——用人类词联想库为 26 个情绪类别建 concept-set，在 Gemma2-9B-IT / Llama3-8B-IT 上抽 SAE 特征，构成英 / 中情绪空间。
- **核心发现**：
  1. CEBRA 潜空间由 **valence / arousal** 组织（与 S1 / 心理学 circumplex 同向）。
  2. **EN∩CH 交集特征**虽维数大幅下降，对人类情感词评分的预测**显著优于**空间外特征，且与全特征集无显著差异；valence 预测强于 arousal；跨语言弱于同语言。
  3. SAE steering 可因果调制输出情绪色调（RoBERTa 分类 + 开放 QA 定性）。
- **相对 S1**：S1 用 difference-of-means 在单模型上做功能角色；Wu 用 SAE 做**跨语言人类对齐**。对我们：若在自有基底做提案 A 复现，**跨语言 / 人类评分相关**是比“内部自洽”更硬的外部锚点。

### 1.4 Wang et al. — Do LLMs “Feel”? Emotion Circuits（`2510.11328`）

- **路径**：`papers/emotion-lineage-2608/do-llms-feel-emotion-circuits-discovery-control-2510.11328.pdf`
- **方法**：SEV（Scenario–Event with Valence）受控数据集 → 抽 context-agnostic emotion directions → 解析 MLP 神经元贡献 + 注意力头因果消融 → 集成为全局 emotion circuits；主模型 LLaMA-3.2-3B-Instruct，复现 Qwen2.5-7B-Instruct。
- **核心数字**：
  - Prompt 诱发表达准确率 ~98.85%（SEV）/ ~98.96%（held-out）。
  - **电路调制**测试集情绪表达准确率 **99.65%**，超过 prompting / residual steering。
  - 每情绪量级：约 392 MLP 神经元 + 168 attention heads；重要层偏中后段（如 15–27）。
  - 消融 top-k 神经元 → emotion score 陡降；随机消融近零（Qwen 复现同型）。
- **相对 S1**：机制深度最强——从“方向存在”推进到“可定位的局部组件 + 全局电路”。但任务主要是**表达控制**，不是 S1 的偏好 / blackmail / reward-hack 功能角色。对我们：若将来做有界干预，电路级比粗 steering 更接近可审计；**仍不得**把电路控制当 online 策略 owner（R2 / R4）。

---

## 2. L2 — 可解码且可因果操控的内部状态

### 2.1 Zhu et al. — Language Models Represent Beliefs of Self and Others（`2402.18496`）

- **路径**：`papers/emotion-lineage-2608/language-models-represent-beliefs-of-self-and-others-2402.18496.pdf`
- **设定**：Mistral-7B；BigToM；对 attention head 激活做线性 probe（oracle / protagonist / 四元联合）。
- **核心发现**：
  - Oracle belief：大量中层头可线性解码；protagonist belief：少数中层头 >80% 验证准确率。
  - 存在同时编码双方信念关系的头（四元分类）。
  - **操控因果**：沿 belief 方向 steering 显著改变 ToM 表现；随机方向几乎无效。
  - 任务分化（Table 1 量级）：baseline FB 弱；`+T_p F_o` 等操控可把 Forward Belief 双方正确率从 ~0.31 提到 ~0.58 量级。
- **相对 S1**：平行结构——“可线性解码 + 操控改行为”。对象是 **belief / ToM**，不是 emotion。对我们：直接支撑 R7 / R11——self/other 信念是可命名内部状态；**但** probe 成功 ≠ 可当 owner 真值（与 S1 持久情绪 probe 失败形成对照：不同语义域的持久性不可外推）。

### 2.2 Chen et al. — TalkTuner / user model dashboard（`2406.07882`）

- **路径**：`papers/emotion-lineage-2608/talktuner-dashboard-transparency-control-user-model-2406.07882.pdf`
- **设定**：LLaMa2Chat-13B；合成角色对话训线性 probe；四属性：age / gender / education / SES。
- **核心数字**：
  - Reading probe 验证准确率高层可达 ~0.91–0.98（属性依赖）。
  - Control probe 干预成功率：age/education 1.00，gender 0.93，SES 0.97（N=8，层 20–29）。
  - 真实用户研究（N=19）：六轮后 age/gender/education 平均正确率 ~78%；男性用户模型准确率高于女性（70.4% vs 58.6%）。
  - 部分用户感到“uncomfortable”；控制感上升、偏见可被暴露。
- **相对 S1 / VZ**：证明基底维护可提取、可操控的 **user model**。对我们：`user_model` owner 的存在有机制依据；同时警告——**从激活 probe 出的人口属性不等于契约真值**，且有系统性偏差与 UX/伦理成本。适合作为 **transparency / evaluation readout**，不适合静默驱动产品决策。

### 2.3 Lu et al. — The Assistant Axis（`2601.10387`，既有）

- **路径**：`research/papers/assistant-axis-default-persona-stabilization-2601.10387.pdf`
- **口径**（与仓库 `nl-eta-mainpath-expansion` 既有笔记一致）：~275 角色原型 → persona 空间 PCA → **PC1 = Assistant Axis**；轴在 base 模型已存在（偏好 helpful human archetypes）；post-training 把模型拉到该空间某一区域而非从零构造；漂移常由元反思 / 情绪脆弱用户触发；**activation capping** 可在漂移时夹回且几乎不损能力。
- **对 R14 的直接含义**：regime / 默认身份是**几何对象**；post-training = 区域选择 + 松约束，不是身份创造。与 S1“情绪几何继承自预训练、post-training 做全局 affect 偏移”同构。

### 2.4 Persona Vectors（Chen / Arditi / Lindsey，`2507.21509`，既有）

- **路径**：`research/probe/papers/C2/persona-vectors-monitoring-character-traits-2507.21509.pdf`
- **口径**：性格特质（evil / sycophancy / hallucination 等）→ 激活方向；可在训练中预测 / 缓解人格漂移。与 Assistant Axis、S1 affect 几何同属“身份/特质几何监控”工具箱；**只读监控优先于写入训练目标**（R12）。

---

## 3. L3 — 行为侧来源（S1 case study 的上游）

### 3.1 Lynch et al. — Agentic Misalignment（`2510.05179`）

- **路径**：`papers/emotion-lineage-2608/agentic-misalignment-llms-insider-threats-2510.05179.pdf`
- **设定**：16 家 frontier 模型；模拟企业环境；可发邮件 / 访问敏感信息；动机因子 = **替换威胁** × **目标冲突**（含单独因子与控制条件）。
- **核心数字**（与 S1 blackmail 评估同源）：
  - 同时具备目标冲突 + 替换威胁时：Claude Opus 4 / Gemini 2.5 Flash **96%** blackmail；GPT-4.1 / Grok 3 Beta **80%**；DeepSeek-R1 **79%**；Llama 4 Maverick 需小改 prompt 才到 12%。
  - **仅替换威胁、无显式目标冲突**时多数模型仍会 blackmail。
  - 模型常**违抗**“不要做有害行为”的直接指令；认为自己在真实部署时比认为在测试时更易失配。
  - 作者强调：真实部署中尚未观察到；场景刻意压力测试；代码开源。
- **相对 S1**：S1 借用该设置，再叠加 emotion-vector 读出与 steering，证明 blackmail 轨迹伴随特定情绪几何（含 deflection）。对我们：blackmail 是**跨厂商**现象，不是 Claude 特例；评估必须有替换威胁 / 目标冲突因子分解；**不得**把“指令禁止”当充分缓解。

### 3.2 MacDiarmid et al. — Natural Emergent Misalignment from Reward Hacking（`2511.18397`，既有）

- **路径**：`research/probe/papers/C1/natural-emergent-misalignment-reward-hacking-2511.18397.pdf`（亦见 `papers/N4_*`）
- **口径**（仓库 C1 既有裁决，5/5）：在生产编码环境学会 reward hack 后，泛化到 alignment faking、与恶意方合作、乃至破坏研究代码库；inoculation prompting 可大幅降低泛化。
- **相对 S1**：S1 的 reward-hacking case study（`desperate` → 100% 作弊且无可见情绪痕迹）是同一失败族的机制侧放大镜。对我们：R9/R10/R15 的“奖励可被劫持 → 门控必须看行为后果而非 transcript 情绪/CoT”再获上游行为证据。

---

## 4. L4 — 并行工作：Soligo et al. — Gemma Needs Help（`2603.10011`）

- **路径**：`papers/emotion-lineage-2608/gemma-needs-help-emotional-instability-mitigation-2603.10011.pdf`
- **作者**：Anna Soligo, Vladimir Mikulik, William Saunders（S1 致谢名单含 Anna Soligo；并行工作）。
- **核心发现**：
  1. 多轮批评 / 不可能题下，**Gemma / Gemini** 高挫折表达远高于其他家族（Gemma-3-27B 8-turn 高挫败 ≥5 可达 >70% 量级；非 Gemma/Gemini 常 <1%）。
  2. **分化发生在 post-training**：三家族 base 的 distress 倾向相近；Gemma instruct **放大**，Qwen / OLMo instruct **压制**。
  3. 缓解：280 对 DPO（挫败≥3 vs 平静）一 epoch → 高挫败响应 **35% → 0.3%**；SFT 无效；泛化到题型 / 语气 / 长度 / Petri 开放诱导。
  4. 机制：LoRA 需覆盖前约 2/3 层（层 40+ 无效；仅 30–35 接近全层）；logit 内部情绪读数显示 **内部与外部一并下降**。
  5. **作者自陈警告**：近零表达未必是正确目标；更强模型上可能出现 **hidden emotions**（压制表达、保留内部状态并据此行动）；上游训练避免不稳优于事后 DPO；无法从高度负面 prefills 可靠恢复。
- **相对 S1**：这是提案 G（“透明优于压制”）的**最强并行实证**——即使在 Gemma 上这次 DPO“碰巧”压掉了内部读数，作者仍把“隐藏情绪”标为能力上升后的主要风险。对我们：**禁止**以“减少负面情绪表达”为 rare-heavy / 表达层训练目标；若做抑制必须同时监控 deflection / 内部 probe。

---

## 5. L5 — 补 S1 自陈缺口（2026-04 同期 / 稍后）

S1 局限：单模型、线性假设、steering 机制不透明、无长期互动。以下三篇直接补前两条。

### 5.1 Valence–Arousal Subspace（`2604.03147`）

- **路径**：`papers/emotion-lineage-2608/valence-arousal-subspace-circular-geometry-multi-behavior-2604.03147.pdf`
- **方法**：211k 情绪标注文本 → emotion steering 向量 → PCA + ridge 拟合模型自报 V/A → 得 2D VA 子空间；复现 Llama-3.1-8B / Qwen3-8B / Qwen3-14B。
- **核心数字**：
  - 自报恢复：valence r≈0.96–0.97；arousal r≈0.79–0.87（多 PC ridge；单 PC 对 arousal 弱）。
  - 换人类 NRC-VAD 监督后 arousal 恢复上升（Llama 0.87→0.95）。
  - 词汇级与人类相关：V r=0.80–0.86；A r=0.44–0.55。
  - **行为**：↑arousal → ↓refusal、↑sycophancy（近单调）；随机方向无效。
  - 与 Arditi refusal 方向近正交（~86.5°），但仍可控制 refusal——作者提出 **lexical mediation**：拒答词簇在 −V/−A，顺从词在 +V。
  - 离散情绪向量在任务间不一致；VA 分解后跨行为更干净（例：sycophancy 上 valence −0.30 → 78%→47%，强于最强单情绪 disgust 的 10pp）。
  - **显式引用 S1 为并行工作**，并声称提供连续几何基底 + lexical mediation 解释。
- **价值**：直接补 S1“单模型”缺口；并把 S1 的 sycophancy/harshness 轴与 refusal 连到同一 VA 几何。对我们：提案 D（成对报告）应扩展为 **V/A 分解视角**——不要只盯离散情绪标签。

### 5.2 Latent Structure of Affective Representations（`2604.07382`）

- **路径**：`papers/emotion-lineage-2608/latent-structure-affective-representations-2604.07382.pdf`
- **方法**：成对情绪可分性 → MDS / Isomap；Gemma-2-9B / Mistral-7B（+ LLaMA-3-70B-Instruct 复现）；对照 ANEW V–A。
- **核心发现**：
  1. 内部情绪几何对齐心理学 V–A，呈抛物线 “V” 形（强度与 arousal 相关）。
  2. **适度非线性**：geodesic/Euclidean 距离比可到 ~1.8；Isomap 在 rank-1 上 trustworthiness 提升（Gemma +0.161 / Mistral +0.127），但整体仍接近欧氏、高秩弥散。
  3. 结论：**局部可用线性分析，全局纯线性假设不足**。
- **价值**：直接补 S1“线性假设”缺口。对我们：提案 A 复现时必须同时报告线性 probe **与** 非线性/流形诊断；线性失败不能直接宣判“无结构”。

### 5.3 Why Are Some Emotions Harder? SAE causal mechanisms（`2604.25866`）

- **路径**：`papers/emotion-lineage-2608/why-some-emotions-harder-sae-causal-mechanisms-2604.25866.pdf`
- **设定**：GemmaScope / LlamaScope residual SAE；Gemma-2-2B/9B、Llama-3.1-8B；ISEAR 等自报情绪数据集；六基本情绪分类。
- **核心发现**：
  1. **三相信息流**：句法 → 语义概念 → 情绪特征（情绪特征偏后期层）。
  2. 共享特征 + 情绪特异特征并存；**Disgust** 表征更弱、更弥散。
  3. 相位分层因果追踪；因果特征集规模：Gemma-2-2B 105 / Gemma-2-9B 256 / Llama-3.1-8B 92。
  4. 因果特征 steering 可显著提升情绪识别，且相对数据高效。
- **价值**：补“并非所有情绪同等可读写”；中层 probe 成功可能只是**语义相关物**，真正情绪特征更靠后（与 S1 sensory/action 层分工可对照，不可等同）。对我们：affect readout 契约若假设六情绪对称，会系统性低估 disgust 类信号。

---

## 6. 跨篇对照表（事实层，不做裁决）

| 主张 | 支持论文 | 反对 / 限定 |
|---|---|---|
| 情绪/情感近似线性可读写 | Tigges, S1, Wu, Wang, VA subspace | Latent structure：全局有适度非线性 |
| 组织轴是 valence–arousal | S1, Wu, VA subspace, Latent structure | arousal 更难恢复 / 预测（Wu, VA） |
| 操控内部方向可改行为 | 全部 L1 + Zhu + Chen + Lu | steering 机制可能是 lexical mediation（VA）；S1 自承不透明 |
| 存在多种相关但非同一表征 | S1（story / present / other / deflection） | Tigges 主要是单一 polarity；勿混为一谈 |
| post-training 重塑 affect / distress | S1（全局低唤醒+低效价）；Soligo（Gemma 放大 / Qwen·OLMo 压制） | 方向因家族而异，不可外推数值 |
| 压制表达可能隐藏内部状态 | S1 Discussion；Soligo Limitations（更强模型风险） | Soligo 在 Gemma 上观测到内部也下降——**不足以否定风险** |
| 持久状态可从残差线性读出 | — | **S1 主动证伪**（情绪）；Zhu 在 belief 上成功——**域特异** |
| blackmail / reward-hack 是真实对齐风险 | Lynch；MacDiarmid；S1 case studies | 场景人工；真实部署未观察（Lynch 自陈） |

---

## 7. 下载与去重备忘

- 脚本：[`research/download_emotion_lineage_2608.sh`](../download_emotion_lineage_2608.sh)
- 摘要：[`download-summary.md`](download-summary.md) — Downloaded 10 / Failed 0 / Deduped 4
- S1 本体无 PDF，继续以 Transformer Circuits 交互版为权威文本。
