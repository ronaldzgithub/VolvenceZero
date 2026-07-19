# Future House — 深度分析

- **分组 / 成熟度 / 一句话主张**：B 自主 AI 科学家闭环 / 成熟度高（多篇已发表 + 开源框架 paperqa·ldp·aviary + Nature 级湿实验验证）/ 主张：把"文献→假设→实验→分析→修正"整条科学发现链拆成可由 LLM agent 闭环执行的子任务，并用**实验可验证奖励**而非纯 token 模仿来训练专科 agent。
- **主要创作者 + 血统**：Samuel Rodriques（CEO，Francis Crick Institute）、Andrew D. White（科学负责人，RL/化学）；技术骨干 Narayanan、Braza、Laurent、Skarlinski、Griffiths、Bou。血统是"实验科学 + RLVR（可验证奖励强化学习）"，与 VZ 的共振点是**预测误差驱动 + 可验证目标门控**；对立点是它把闭环的"评判/优先级"大量交给 LLM judge，且 ether0 直接对基底做 token 空间 RL——这正好压在 VZ 的 R-PE-不外包 与 R4-不做-token-空间-RL 两条红线上，是本 lab 最值得红队的地方。

---

## 1. 核心逻辑（论文级 · PDF-grounded）

### 1.1 PaperQA2 — Language agents achieve superhuman synthesis of scientific knowledge（arXiv:2409.13740）
- **问题**：LLM 会幻觉、忽略细节，且现有文献检索 benchmark 要么只看摘要、要么固定语料、要么直接给对的论文，无法代表真实科研检索，也缺与人类的直接对比。
- **方法/机制**：PaperQA2 是把 RAG 拆成多步 agent 工具的检索 agent——Paper Search（关键词检索+Grobid 解析+混合稠密/稀疏向量）、Gather Evidence（top-k 向量召回 + **RCS：LLM 重排+上下文摘要**，每块压成 200–400 token 并打 0–10 相关分）、Generate Answer、**Citation Traversal**（沿引文图做层级索引扩召回）。agent 编排 LLM 固定为 gpt-4-turbo，temperature=0。
- **关键结果**：在自建 hard benchmark LitQA2（248 道、答案藏在正文非摘要、DOI 可校验）上 precision 85.2%±1.1%、accuracy 66.0%±1.2%，21.9% 答"信息不足"；人类 PhD/博后 precision 73.8%、accuracy 67.7%——PaperQA2 precision 超人（p=0.0036），accuracy 与人无显著差异。消融：非 agent 化（硬编码顺序）显著掉点；去掉 RCS 显著掉点（小模型做 RCS 反而更差，存在"理解力阈值"）；去掉 Citation Traversal 召回显著下降。WikiCrow（综述）"cited & unsupported" 13.5% vs 维基 24.9%，未引用率 3.5% vs 13.6%，精度 86.1% vs 71.2%。ContraCrow（矛盾检测）ContraDetect AUC 0.842，阈值 8 时 73% acc / 88% precision / 7% FPR；93 篇生物论文每篇均检出 2.34±1.99 处矛盾，人类专家验证 70%（每篇 1.64 处可验证矛盾，下界）。
- **局限**：成本 $1–3/query、综述 $4.48/篇；**ContraCrow 与人类一致率仅 60.4%，而人类彼此一致 75.5%（p=0.015），作者归因为模型"过度自信"**——纯 LLM-as-judge 的矛盾判定不可靠。

### 1.2 LAB-Bench — Capabilities of LLMs for Biology Research（arXiv:2407.10362）
- **问题**：现有科学 benchmark 多测"教科书式"死记知识，缺对真实科研实践能力（文献检索、读图表、查库、操作序列、写协议、分子克隆）的评测，也缺人类基线。
- **方法/机制**：构建 2400+ 道多选题，分 LitQA2、SuppQA、FigQA、TableQA、DbQA、SeqQA、ProtocolQA、CloningScenarios（41 道"human-hard"克隆题）。程序化 + 人工双管线出题；公开 ~80%、私藏 ~20% 监测污染；强调高质量干扰项与"信息不足"选项。
- **关键结果**：评测多个前沿模型与 PhD 级人类专家。关键学习：(1) 让前沿模型自己出题再人工筛改，能高效探出模型"推理盲区"；(2) 干扰项质量决定评测有效性——题目差时模型靠"应试技巧"（排除法+猜）刷高分，必须用开放式答案手工复核；(3) 复杂任务（CloningScenarios）人类基线难获取，未来或以"human proof-of-possibility"替代人类基线。
- **局限**：多选格式天然可被应试策略钻空子；高难任务人类基线不稳定。

### 1.3 Aviary — Training Language Agents on Challenging Scientific Tasks（arXiv:2412.21154）
- **问题**：科学任务需要"动作-观察"多轮循环，但语言 agent 缺统一的学习问题形式化与高效训练方法。
- **方法/机制**：把语言 agent 任务形式化为 **LDP（语言决策过程，自然语言动作/观察的 POMDP）**，agent 表示为**随机计算图（SCG）**，可优化参数 θ 涵盖 LLM 权重、prompt、记忆、采样温度等；内部推理/记忆检索被 θ 吸收，动作空间只保留对外交互。训练用**行为克隆 BC + 专家迭代 EI**（rollout → 按回报阈值 ρ 拒绝采样过滤 → 对过滤后轨迹做 SFT，迭代），推理期用 majority vote / pass@k 扩算力。5 个环境含 3 个科学环境：分子克隆（SeqQA）、文献问答（PaperQA2 Local/LitQA2）、蛋白稳定性。
- **关键结果**：开源小模型 Llama-3.1-8B-Instruct 经 BC+EI 训练后，在 SeqQA、LitQA2 上**匹敌甚至超过前沿模型与人类专家，推理成本低至 1/100**。SeqQA：BC 后再 +14%（绝对）来自在线 EI；LitQA2：未训练 8B agent 30% → EI 后测试集 72%（难度加权采样 wk=M·(1−f_pass)）。majority vote 再加约 20 个百分点。观察到自训练使轨迹变短、多样性下降（"收敛到子集路径"）。
- **局限**：EI 早期大量拒绝、学习慢需 BC 启动；train/test 有过拟合；自训练多样性塌缩。

### 1.4 ether0 — A Scientific Reasoning Model for Chemistry（arXiv:2506.17238）
- **问题**：推理模型（长 CoT+RL）在数学/编程外能否泛化到科学？能否不靠领域预训练、用更少数据训出化学推理模型？
- **方法/机制**：基于 Mistral-Small-24B，多阶段交替 **蒸馏（SFT）↔ GRPO（RL）**：(1) 长 CoT SFT 热启动；(2) 7 个 specialist 分任务 GRPO；(3) 蒸馏成 generalist；(4) all-task GRPO。奖励 = format_reward × accuracy_reward，**全部来自可验证器**（RDKit、KDESol、Bloom filter 购买性、Molecular Transformer 反应校验、实验测定 MCQ 串匹配）。引入 think/answer 边界 token；**advantage-based 课程**（只把非平凡 group 留进 buffer，降低 GRPO 优势为零的"trivial"占比 f_T，可达 90%）；problem rewriting（Gemini 改写+加干扰）。640,730 题 / 375 任务。
- **关键结果**：开放式任务（OA）全面最高，超前沿 LLM、人类化学家、专用模型；反应预测 46k 样本即 70% acc（Molecular Transformer 全 USPTO 480k 才 64.1%）。zero-shot 50.1%≈o1 的 52.2%，one-shot 超所有前沿。reasoning 版稳定优于 non-reasoning 版；"认知行为"（回溯/验证/子目标）在结构化推理有利的任务上被选择性放大。
- **局限**（直接关乎 VZ）：**"密集 RL 训练降低了 Mistral 的通用指令遵循与对话能力，包括多轮对话"**；分布外（无机化学）差；未含工具调用；部分验证器本身是 ML 模型（MT/KDESol）引入误差。

### 1.5 Robin — A Multi-Agent System for Automating Scientific Discovery（arXiv:2505.13400，Nature 2026）
- **问题**：尚无系统能在单一工作流里自动化"背景研究→假设→实验→分析→更新假设"全部关键智力步骤。
- **方法/机制**：多 agent 编排——Crow/Falcon（基于 PaperQA2 的浅/深文献综述）、Finch（实验数据分析，写 Jupyter notebook 做 RNA-seq/流式）。给定疾病名，Robin 用 Crow 调研→列 10 个致病机制→为每机制出体外模型/assay→**LLM-judge 两两比较锦标排序**→选 top 体外模型→Crow 调研 ~400 篇出 30 个候选药→Falcon 出评估报告→**LLM-judge 锦标排序**→人类做湿实验→数据回传 Finch（跑 10 条轨迹取共识 meta-analysis）→Robin 蒸馏洞见并提下一轮假设/follow-up。
- **关键结果**：首个在 lab-in-the-loop 里**自主提出并验证新疗法**的 AI——对干性 AMD（dAMD）提出"增强 RPE 吞噬"策略，识别并验证 **ripasudil / Y-27632（ROCK 抑制剂，从未被提出治 dAMD）**；follow-up RNA-seq 由 Robin 提出并分析，揭示 ABCA1（脂质外排泵）上调为新靶点。正文所有假设/实验计划/数据分析/图均由 Robin 产出。
- **局限**："semi-autonomous"——人类执行湿实验且决定何时满意；候选优先级靠 **LLM-judge 锦标**，仅 5/30 候选被实测，排序质量未对结果系统性校验；Finch 单轨迹随机性大，需 10 轨迹共识兜底。

---

## 2. 与 VZ 的关系（三视角）

> 纪律：先反证，后确证。本 lab 最强的张力在于"自主科学家闭环是否把 PE 来源外包给了 LLM"，以及"ether0 是否在 token 空间对基底做 RL"。

### 2.2 反证（红队）

**反例 A（R-PE 不外包）——Robin 用 LLM-judge 锦标生成假设并决定优先级，PaperQA2/ContraCrow 用 LLM 当矛盾裁判。**
- 反例论点：自主科学家闭环里，"哪个假设值得测、哪个候选排第一、是否构成矛盾"这些**判定/奖励信号**大量由 LLM 生成——似乎证明"PE/评判来源可以外包给另一个模型"。
- 裁决：**survives（局部 genuine-risk）**。Robin 的**终极学习信号来自湿实验真实测量**（吞噬 MFI、RNA-seq 差异表达），不是 LLM——LLM-judge 只做上游**便宜的提案与预筛排序**，物理现实才是不可约的 PE 所有者。这恰好印证 VZ 的分工：LLM 可做表达层提案，PE 必须锚在非 LLM 的 ground truth。**但**直接证据显示纯 LLM-as-PE 危险：ContraCrow 与人类一致率仅 60.4% vs 人类彼此 75.5%（作者归因"过度自信"）。
- 边界条件（写进 `prediction-error-loop.md`）：LLM 仅可作 PE 的**上游提案器/优先级预筛**，一级 PE 必须由非 LLM 现实信号（用户真实反应/环境结果）产生；任何把 LLM 当**唯一**裁判并据此学习的环节标为 genuine-risk，须有非 LLM 校验或人类校准（对应 Robin 用湿实验、PaperQA2 用 DOI 匹配/人类验证）。

**反例 B（R4 不在 token 空间做长期 RL / R2 不对基底端到端更新）——ether0 直接对基底做 GRPO（CoT token 上的策略梯度），且全量更新 24B 权重，效果超人。**
- 反例论点：token 空间 RL（GRPO on CoT）能学出真实可迁移的科学推理，且必须更新基底权重——似乎同时推翻 R4（控制不在 token 空间）与 R2（基底冻结）。
- 裁决：**needs-boundary-condition**。三个边界使其不适用于 VZ 目标域：① ether0 的奖励是**可验证且任务终端**（分子正确性），而 VZ 的关系/EQ 域**无可验证奖励**，token 空间 RL 在此无信号可用；② ether0 是**一次性离线训练专科 artifact**（rare-heavy），非在线持续个性化；③ 最关键——ether0 自承"**密集 RL 训练降低了通用指令遵循与多轮对话能力**"，即端到端动基底导致人格/对话能力塌缩，这正是 R2 冻结基底要避免的灾难。
- 边界条件（写进 `temporal-abstraction.md` 与 `multi-timescale-learning.md`）：token 空间 RLVR **仅允许**用于离线 rare-heavy 专科 artifact 生产（有可验证奖励、可接受能力侧损），**禁止**用于在线关系控制；在线控制仍在 z_t/β_t 控制器空间、基底冻结。

**反例 C（R2 冻结基底）——Aviary EI 与 ether0 都更新基底权重（SFT/RL on base）。**
- 反例论点："必须微调基底才能上分，冻结基底是自缚。"
- 裁决：**survives，且被强化**。ether0 用自身代价给出反证的反证——动基底换来的是多轮对话能力丧失；Aviary 也观察到自训练轨迹多样性塌缩。二者都是**离线一次性**改权重并接受副作用，与 VZ"在线持续、必须维持稳定人格"的设定相反。VZ 把这类重训关进 rare-heavy 离线管线、运行时冻结基底，正是为规避此塌缩。
- 边界条件：改基底仅限离线 rare-heavy + `ModificationGate` 门控 + 退出条件 + 评估证据（`credit-and-self-modification.md`、`multi-timescale-learning.md`），运行时不做。

**反例 D（R12 评估先做硬）——Robin 是 semi-autonomous，候选排序靠 LLM-judge 锦标但只实测 5/30，排序质量未系统性对结果校验。**
- 反例论点："闭环可以先跑起来，优先级评估可后补。"
- 裁决：**genuine-risk（轻度）**。Robin 的**提案/排序层评估不足**（锦标排序无 ground-truth 回测），靠人类把关与下游湿实验兜底才没出事。这提示 VZ：任何 LLM-judge 优先级层若进入自动化闭环，必须先有硬评估，否则是盲点。注意：FutureHouse 整体反而是 R12 的**正面样板**（见 2.1）——它先建 LitQA2/LAB-Bench/ContraDetect 硬基准再设计系统；风险只在 Robin 排序层这一处。
- 边界条件（写进 `evaluation.md`）：进入自动化的每个评判/优先级层都要有独立硬评估与回测，未评估的判定层不得驱动自修改。

### 2.1 确证（先进性背书）

- **R-PE（预测误差是一级原始信号）**：Robin lab-in-the-loop（假设→湿实验测量→修正）、ether0 可验证奖励、Aviary EI 环境回报——三者跨化学/生物/分子序列**独立**印证"用真实世界结果差异驱动学习"是工程上可行且超人的范式（`prediction-error-loop.md`）。
- **R9/R10（信用分配 + 有门控的有界自修改）**：ether0 GRPO 优势/EI 回报阈值 ρ = "只在可验证信号超阈时才更新"，是 `ModificationGate`"自修改必须锚定可验证 eval"的正面工程样板（`credit-and-self-modification.md`）。
- **R12（评估先做硬、覆盖真实能力、只读）**：LAB-Bench/LitQA2/ContraDetect **先建硬基准再设计系统**，含私藏 20% 防污染、"信息不足"弃答选项、干扰项质量纪律、开放式手工复核——是 R12"评估必须先硬于自动化"的强背书（`evaluation.md`、`evidence_program.md`）。
- **R1/R13（多时间尺度 + SSL↔RL 交替）**：ether0（蒸馏 SFT ↔ GRPO 多阶段）、Aviary（BC ↔ EI）都是"压缩式自监督 ↔ 强化"交替，印证 VZ 的 SSL→RL 交替结构（`multi-timescale-learning.md`）。
- **R3/R4（控制层与基底分离的形式化）**：Aviary 的 LDP/SCG 把可优化 θ（记忆检索、内部推理、prompt、采样温度）与 LLM 节点解耦——形式上支持"控制器可学、基底为固定随机节点"的边界（`temporal-abstraction.md`）。注意：Aviary 自己优化的是 LLM 节点，VZ 取其形式化、反其优化对象。

### 2.3 局部算法借鉴（算法级解耦）

| # | 机制（剥离叙事） | 目标 VZ spec | 落地动作 | 预期收益 | 风险 / 前提 |
|---|---|---|---|---|---|
| 1 | **可验证奖励 + advantage-based 课程**（ether0：reward=format×accuracy，全来自非 LLM 验证器；只把"非平凡 group"留进 buffer 降低零优势占比） | `credit-and-self-modification.md`（R9/R10）+ `multi-timescale-learning.md`（R13） | rare-heavy artifact 离线训练管线中，`ModificationGate` 仅在 PE/可验证信号超阈时放行更新；引入"只训非平凡样本"的课程缓冲 | 有界、样本高效的自修改，绝不在未接地信号上更新；降算力 | 需真实非 LLM 验证器；关系域须先定义 ground truth（用户行为/PE），否则退化成 LLM-as-judge（见反例 A） |
| 2 | **专家迭代 EI + 回报阈值拒绝采样 + 难度加权采样**（Aviary：rollout→过滤 R>ρ→SFT；wk∝1−f_pass） | `credit-and-self-modification.md` + `temporal-abstraction.md` | 把在线控制器经验蒸馏进慢/稀artifact 时，只接受回报超阈轨迹，按任务难度加权采样 | 稳定的离线巩固（避策略梯度不稳），对齐 SSL↔RL 交替 | 自训练多样性塌缩（论文实测轨迹变短）→ 需多样性/探索守卫 |
| 3 | **RCS（重排+上下文摘要）+ 引文图层级索引**（PaperQA2：每块召回先 LLM 压成 200–400 token 打分再入上下文；沿关系图扩召回） | `continuum-memory.md`（R5/R6）+ `cms-atlas-titans-uplift.md` | 记忆检索不直灌原始块：先重排+摘要压缩成带分结构摘要再入上下文；对情景记忆建关系图派生索引做遍历扩召回 | 召回更准、token 省 ~5–10×、降幻觉；派生索引提升长程关联 | 重排模型需够强（"理解力阈值"，小模型反伤）；RCS 有成本 |
| 4 | **LDP/SCG 形式化：可优化 θ 与固定 LLM 节点解耦**（Aviary） | `contract-runtime.md` + `temporal-abstraction.md` | 把 VZ 控制器形式化为 SCG：仅控制器节点（z_t/β_t、记忆策略、采样）可优化，基底 LLM 为固定随机节点 | 在代码层清晰落实 R2/R4 边界，控制器可微/可训而基底冻结 | 概念借用——Aviary 实际优化 LLM 节点，VZ 须反向固定它 |
| 5 | **硬评估优先的基准构造法**（LAB-Bench/LitQA2：私藏 20% 防污染、"信息不足"弃答项、高质量干扰项、开放式手工复核、自生成题探盲区） | `evaluation.md` + `evidence_program.md`（R12） | VZ 存在/连续性评估套件加入：私藏 split、弃答选项、污染监测；自动化接线前先把评估做硬 | 防评估刷分、支撑 R12"评估硬于自动化" | 关系/存在评估比 MCQ 更难客观化，不能照搬多选格式 |

---

## 3. 一句话定位

Future House 是 VZ"**PE 驱动 + 可验证目标门控的有界自修改 + 评估先做硬**"三条工程纪律在化学/生物域的最佳正面样板；但它的 LLM-judge 优先级层与 ether0 的 token 空间 RL 同时压在 R-PE-不外包 与 R4 两条红线上——红队结论是 VZ 凭"一级 PE 必须非 LLM 现实接地"和"token 空间 RLVR 仅限离线专科 artifact"两条边界守得住，且 ether0"RL 训练损毁多轮对话"反而成了 R2 冻结基底的最强实证。

## 附：本地论文清单（同目录 PDF）

- `paperqa2-superhuman-synthesis-of-scientific-knowledge-2409.13740.pdf` — PaperQA2 / WikiCrow / ContraCrow，LitQA2 超人检索
- `lab-bench-capabilities-of-llms-for-biology-research-2407.10362.pdf` — LAB-Bench，2400+ 生物研究多选任务
- `aviary-training-language-agents-scientific-tasks-2412.21154.pdf` — Aviary/LDP，BC+EI 训练小模型超前沿
- `ether0-scientific-reasoning-model-for-chemistry-2506.17238.pdf` — ether0，24B 可验证奖励 GRPO 化学推理
- `robin-multi-agent-system-automating-scientific-discovery-2505.13400.pdf` — Robin，lab-in-the-loop 自主发现 ripasudil 治 dAMD
