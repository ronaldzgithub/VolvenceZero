# CogNosco Lab：NLP Psychometrics 与语言网络心理读出

日期：2026-08-12

核心论文：De Duro, Franchino, Stella, *Natural Language Processing Psychometrics*，arXiv:2608.07316 v1（2026-08-07）

研究成熟度：**B-（值得做 SHADOW 机制验证，不足以进入 runtime，更不足以作为人的心理测量或诊断工具）**

## 一句话结论

这项工作最有价值的不是“AI 能从文字诊断抑郁”，而是提出了一套可审计的语言结构读出：把文本变成 Textual Forma Mentis Network（TFMN），联合网络拓扑、八类情绪、人格与背景特征，测量受控 LLM persona 如何把心理量表条件表达进语言。它证明了这类结构信号在合成数据中可预测，并对一种意大利临床语音转录语料有中等的组间区分能力；它**没有证明心理状态就是语言网络扰动，也没有完成面向人的心理测量效度验证**。

## 最重要的五个判断

1. **可吸收的是“命名、结构化、带 lineage 的读出”，不是心理分数。** 网络特征能补足情绪词统计，但原始节点数、边数和连通分量高度受文本长度影响。
2. **原报道的相关数字需要降温。** “最高 r≈0.91”是高低分日记组的秩效应量，不是连续量表分数的相关或校准；“临床准确率 68%”只对应 DASS-21 Depression 模型在 115 人二分类上的一次 LOO-CV 结果，PHQ-9 对应准确率只有 62%。
3. **真实人类验证没有问卷真值。** 临床语料只有抑郁/对照二元标签，不能验证预测出来的 PHQ-9 或 DASS 分数是否准确。
4. **Lab 的真正长处是方法谱系。** 它从 multiplex mental lexicon → forma mentis networks → TFMN → EmoAtlas → cognitive digital shadows → PENSO 数据工程，持续把网络心理学变成可复用数据和工具。
5. **对 Volvence 的近期价值主要落在 Readable。** 若进入工程，只能先做长度受控、个体内变化优先、只读 SHADOW probe；不得成为 reward、诊断标签、文本关键词路由或直接 steering 规则。

## 阅读顺序

1. [01_paper_deep_read.md](01_paper_deep_read.md) — 核心论文设计、结果、原文纠偏与证据边界。
2. [02_lab_research_map.md](02_lab_research_map.md) — CogNosco Lab 的 44 项研究地图、PENSO 项目与方法谱系。
3. [03_volvence_implications.md](03_volvence_implications.md) — 对 Volvence 四能力轴、owner、快照、验证门与 kill conditions 的映射。
4. [04_sources_and_downloads.md](04_sources_and_downloads.md) — 下载资产、外部索引、许可证与完整性说明。
5. [CHECKSUMS.sha256](CHECKSUMS.sha256) — 本地资产校验和。

## 本地研究包

- papers/：10 篇关键论文，覆盖核心论文、理论前置、方法前作、工具、临床迁移语料与 PENSO 三条公开支柱。
- assets/code/：EmoAtlas 与 TEA_Networks 源码快照。
- assets/data/：MHDS 完整 CC0 快照，以及 SociaLLMisinformation 三个 OSF 数据包。
- sources/input-article.txt：用户提供的原始中文文章，作为待核查二手来源保留。

总计约 240 MB。MEDS 完整仓库约 367 MB，ConvinceMe/CDS 仓库元数据约 2.2 GB；二者的 GitHub 仓库没有检测到明确许可证，故未镜像进本研究包，仅保留论文、版本信息和官方链接。Talk2AI 论文声称的 GitHub 路径在 2026-08-12 返回 404，也按 link-only 缺口记录。

## 本轮范围

本轮只新增研究文档和开放研究资产，没有修改代码、契约、spec、DATA_CONTRACT 或 WiringLevel。文档中的工程内容都是后续候选方案；任何实现都必须先完成 owner 收敛、slot 注册、SHADOW 接线和独立 prereg。
