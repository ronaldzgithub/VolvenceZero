# CogNosco Lab 研究地图与 PENSO 项目

## 1. Lab 定位

CogNosco Lab 隶属 University of Trento 的 Department of Psychology and Cognitive Science，由 Massimo Stella 领导。官方定位横跨 mathematical psychology、cognitive science、complex systems、cognitive network science、data science 与 AI psychometrics。

它不是单纯做“心理健康 NLP”的实验室。其长期问题是：

> 如何把难以直接观察的概念组织、情绪评价、检索过程、社会心智与 LLM 行为，转成可测量、可比较、可复用的网络表示。

官方列出的 signature models 包括：

- cognitive multilayer networks：语言获得、词汇检索、失语症、创造力；
- behavioral/textual forma mentis networks：概念关联与情感评价；
- AI-based psychometrics：情绪、创造力与心理构念的可解释读出。

2026-08-12 的官方 research map 收录 44 篇工作，分为 13+11+10+10 四个论文主题，并另列 6 阶段方法谱系。本节完整记录这 44 项标题，避免只凭最近一篇论文评价 Lab。

## 2. 四条研究主线

### 2.1 Mental lexicon、语言获得、检索与创造力（13）

1. *Mental Lexicon Growth Modelling Reveals the Multiplexity of the English Language* (2016)
2. *Multiplex lexical networks reveal patterns in early word acquisition in children* (2017)
3. *Multiplex model of mental lexicon reveals explosive learning in humans* (2018)
4. *Cohort and Rhyme Priming Emerge from the Multiplex Network Structure of the Mental Lexicon* (2018)
5. *Modelling Early Word Acquisition through Multiplex Lexical Networks and Machine Learning* (2019)
6. *The multiplex structure of the mental lexicon influences picture naming in people with aphasia* (2019)
7. *Viability in Multiplex Lexical Networks and Machine Learning Characterizes Human Creativity* (2019)
8. *Quantifying the Interplay of Semantics and Phonology During Failures of Word Retrieval by People With Aphasia* (2020)
9. *Multiplex networks quantify robustness of the mental lexicon to catastrophic concept failures, aphasic degradation and ageing* (2020)
10. *Multiplex lexical networks and artificial intelligence unravel cognitive patterns of picture naming in people with anomic aphasia* (2023)
11. *Feature-rich multiplex lexical networks reveal mental strategies of early language learning* (2023)
12. *A cognitive multiplex network approach to investigate mental navigation and predict high-level cognition* (2025)
13. *Introducing multiplex semantic networks as multifaceted representations of creative associative knowledge across multilingual samples* (2026, preprint)

这条线建立了 Lab 的底层观念：概念不是 featureless token，单一语义相似度也不够。发音、同义、自由联想、共同出现等多种关系要并存；真正有解释力的经常不是单节点，而是跨层可达性、viability、检索路径和系统受损方式。

### 2.2 Mindsets、教育与集体社会认知（11）

1. *Forma mentis networks quantify crucial differences in STEM perception between students and experts* (2019)
2. *Forma Mentis Networks Reconstruct How Italian High Schoolers and International STEM Experts Perceive Teachers, Students, Scientists, and School* (2020)
3. *Forma mentis networks map how nursing and engineering students enhance their mindsets about innovation and health during professional growth* (2020)
4. *Text-mining forma mentis networks reconstruct public perception of the STEM gender gap in social media* (2020)
5. *#lockdown: Network-Enhanced Emotional Profiling in the Time of COVID-19* (2020)
6. *Cognitive Network Science Reconstructs How Experts, News Outlets and Social Media Perceived the COVID-19 Pandemic* (2020)
7. *Mapping computational thinking mindsets between educational levels with cognitive network science* (2021)
8. *Cognitive Network Science for Understanding Online Social Cognitions: A Brief Review* (2022)
9. *Cognitive Networks Extract Insights on COVID-19 Vaccines from English and Italian Popular Tweets* (2022)
10. *Introducing mindset streams to investigate stances towards STEM in high school students and experts* (2023)
11. *Cognitive networks reconstruct mindsets about STEM subjects and educational contexts in almost 1000 high-schoolers, University students and LLM-based digital twins* (2026, preprint)

这条线把实验室任务扩展到“一个群体如何把概念连在一起并赋予情绪”。它的重要贡献不是又一种 sentiment analysis，而是把 stance 拆成概念局部邻域、正负/多情绪标签、远距离概念之间的 mindset stream，以及不同来源、语言和时间点的结构差异。

### 2.3 Mental health、psychometrics 与 interpersonal cognition（10）

1. *DASentimental: Detecting Depression, Anxiety, and Stress in Texts via Emotional Recall, Cognitive Networks, and Machine Learning* (2021)
2. *Cognitive networks detect structural patterns and emotional complexity in suicide notes* (2022)
3. *Network psychometrics and cognitive network science open new ways for understanding math anxiety as a complex system* (2022)
4. *Cognitive network neighborhoods quantify feelings expressed in suicide notes and Reddit mental health communities* (2023)
5. *Voices of rape: Cognitive networks link passive voice usage to psychological distress in online narratives* (2024)
6. *Introducing semantic loadings in factor analysis* (2024)
7. *Cognitive Networks and Text Analysis Identify Anxiety as a Key Dimension of Distress in Genuine Suicide Notes* (2025)
8. *Textual forma mentis networks bridge language structure, emotional content and psychopathology levels in adolescents* (2025, preprint)
9. *A Network Psychometric Analysis of Math Anxiety Factors in Italian Psychology Students* (2025)
10. *Emotional content and semantic structure of dialogues are associated with Interpersonal Neural Synchrony in the Prefrontal Cortex* (2025)

这里有三条值得区分的路线：

- 网络心理测量：量表 item/symptom 的统计依赖结构；
- 认知网络：概念和语言的语义、句法与情绪结构；
- 跨模态关系研究：对话语言结构与双人神经同步的关联。

核心论文的创新在于把前两条合到同一个预测管道，但这仍是关联与读出，不是临床因果模型。

### 2.4 AI cognition、advanced representations 与开放工具（10）

1. *Cognitive Network Science Reveals Bias in GPT-3, GPT-3.5 Turbo, and GPT-4 Mirroring Math Anxiety in High-School Students* (2023)
2. *Using cognitive psychology to understand GPT-like models needs to extend beyond human biases* (2023)
3. *Towards hypergraph cognitive networks as feature-rich models of knowledge* (2023)
4. *Cognitive modelling of concepts in the mental lexicon with multilayer networks: Insights, advancements, and future challenges* (2024)
5. *EmoAtlas: An emotional network analyzer of texts that merges psychological lexicons, artificial intelligence, and network science* (2025)
6. *The “LLM World of Words” English free association norms generated by large language models* (2025)
7. *Cognitive networks identify AI biases on societal issues in Large Language Models* (2026 version of record)
8. *Cognitive networks highlight differences and similarities in STEM mindsets of human and GPT-simulated trainees, experts and academics* (2026)
9. *Forma mentis networks predict creativity ratings of short texts via interpretable artificial intelligence in human and AI-simulated raters* (2026)
10. *SpreadPy: A Python tool for modelling spreading activation and superdiffusion in cognitive multiplex networks* (2026)

这条线有一个很健康的自我约束：Lab 自己明确指出，人和 LLM 的相似输出可能只来自人类训练语料，并不意味着相同认知机制。比如，人类与 GPT 对 STEM 的表面 valence 可能相近，但图的密度、聚类和整合度不同。**行为相似不等于机制同构**，是理解 cognitive digital shadows 的必要限定。

## 3. 六阶段方法谱系

| 阶段 | 核心问题 | 代表方法 | 下游用途 |
|---|---|---|---|
| Multirelational representation | 一个概念如何同时位于多种关系中？ | multiplex lexical networks | 获得、priming、aphasia、创造力、衰老 |
| Cognitive-affective enrichment | 概念关联如何被情绪评价？ | behavioral forma mentis networks | STEM mindset、职业成长、math anxiety、人机比较 |
| Language-derived reconstruction | 表达出来的话如何还原成可解释概念框架？ | TFMN、MERCURIAL | 社会话语、创伤、suicide notes、AI audit |
| Feature-/order-rich structure | 把节点当作无特征、边只成对会丢掉什么？ | FERMULEX、cognitive hypergraph | 获得预测、concreteness、高阶记忆 |
| Processes over maps | 激活与搜索如何在图上移动？ | viability、mindset streams、navigation、SpreadPy | 创造力、检索、clinical errors、semantic search |
| Comparative machine psychology | LLM 复制了人的概念组织，还是只复制表面产物？ | standardized prompts、digital shadows、LWOW、TFMN audits | bias、STEM、creativity、social discourse |

这条谱系是 CogNosco 最突出的研究特点：每一代不是换题目，而是扩充可观测对象——从静态、多关系图，到 affect labels、高阶边、图上动力学，再到人/机对照。

## 4. PENSO：Lab 当前的工程化主线

PENSO 全称 *A Psychometric framEwork for uNderstanding how large language modelS influence or bias human psychOlogy*。项目在 2025 年获得 FIS2/MUR 约 €1.3M 资助。官方页面把它组织为三个公开支柱和一个横向工具层。

### 4.1 ConvinceMe：社会说服

- Cognitive Digital Shadows (CDS)：19 个 LLM、约 190,000 条合成记录、4 个社会议题、17 个 persona 属性；
- 每条记录保留 mode、persona、topic、prompt config、opinion、reasoning summary、tone、validation status；
- 价值在“条件可追溯”和模型间比较，而不是把 persona 输出当成人群意见；
- 仓库体积元数据约 2.2 GB，且未发现明确 license，本包只保存论文与官方索引。

### 4.2 TeachMe：数学教学

- MEDS：14 个 LLM、28,000 个 digital shadows、每个 shadow 4 类任务；
- 同时记录数学正确率、reasoning、confidence、自我效能、math anxiety 与 BFMN；
- 最有价值的变量是 confidence–accuracy gap：模型可能答案错但非常自信，或传递焦虑框架；
- 完整 repo snapshot 约 367 MB，未发现明确 license，本包保存论文和 commit 元数据，不镜像整库。

### 4.3 SupportMe：心理福祉

- MHDS：15 个 LLM、75,000 条完整记录；
- 56,250 条 human-shadow，18,750 条 AI-assistant；
- 每条约 74 个字段，包含 persona、OCEAN、DASS-21 评分与解释、6 个 mental-health topic 回复和 10 个 emotional recall words；
- CC0，故本包保存完整仓库快照；
- 重要负面结果：assistant 模式的 DASS 往往塌到零附近；数据应视为模型偏差与 prompt-conditioned behavior，而非人类临床代理。

### 4.4 横向工具

- EmoAtlas：依存句法 + EmoLex + null-model z-score + TFMN，可处理 18 种语言；开源 BSD-3-Clause。
- TEA_Networks：把叙事拆为 Target–Event–Agent 层，显式表示“谁做了什么、影响谁”；开源 BSD-3-Clause。
- SpreadPy：图上的 spreading activation / superdiffusion，从“画出网络”推进到“模拟图上过程”。
- MHDS/MEDS pooling systems：按 persona、模型、任务和心理变量筛选、汇出子集。

## 5. Lab 的特点

### 5.1 跨学科不是拼盘，而是共享同一表示

心理学提供 construct 和量表，语言学提供句法/语义关系，网络科学提供结构和动力学，ML 提供预测与消融。它们都落到“节点—边—属性—路径—readout”这一共同对象上，所以成果能跨语言获得、临床、教育、社会认知和 LLM audit 复用。

### 5.2 强调 interpretable middle layer

Lab 不直接把原始文本扔进大模型分类器，而是先构建可检查的中间表示。优点是能问“哪些概念、什么边、哪种情绪、何种拓扑”；缺点是 parser、lexicon、network construction 本身成为新的模型假设，不能把“可画图”误认为“已解释因果”。

### 5.3 数据工程与开放科学能力强

PENSO 不只有论文：它同步发布数据、codebook、生成 notebook、dashboard 和工具包。这个执行风格很值得 Volvence 借鉴——任何新 readout 都应同时有 schema、lineage、数据卡、失败案例和可复现实验，而不只是一段 prompt。

### 5.4 合成 persona 是审计仪器，不是人类替身

数字影子适合做 controlled probes、stress tests 和 model comparison。它不适合估计真实人口分布、代替真实用户 longitudinal data、证明模型拥有人的心理状态，或作为心理干预效果的证据。

Lab 的 CDS/MHDS/MEDS 论文都写出了这一边界；传播文章常把它删掉。

### 5.5 研究路线正从静态“地图”走向动态“轨迹”

早期工作重网络结构；mindset streams、SpreadPy、Talk2AI 和 PENSO human experiments 开始关心图如何随时间、互动、说服和关系变化。这个方向比单文本心理分类更适合 Volvence，因为我们的核心是跨 turn/跨 session 的可追加状态，而不是一次性标签。

## 6. 项目状态与诚实边界

截至 2026-08-12：

- PENSO 的公开合成数据与工具已有明确产出；
- UniTrento 报道的人类重复互动实验是未来两年的项目计划，不能当作已完成结果；
- Talk2AI 预印本报告 770 名成人、4 周、3,080 段对话、30,800 turns，是重要纵向资产；但论文所列 GitHub 路径当前返回 404，故本包只保存论文；
- NLP Psychometrics v1 没有给出该研究自身分析代码或精确训练数据仓库链接；MHDS 很接近其数据生成谱系，但不能擅自宣称就是完整复现实验包。

## 7. 官方入口

- CogNosco institutional page: https://www.cogsci.unitn.it/1332/cognosco-lab
- Lab website: https://cognosco.dipsco.unitn.it/
- Expanded research map: https://cognosco.dipsco.unitn.it/research
- PENSO observatory: https://cognosco.dipsco.unitn.it/fis2_penso
- UniTrento PENSO profile: https://mag.unitn.it/innovazione/121727/combattere-lansia-con-lia
