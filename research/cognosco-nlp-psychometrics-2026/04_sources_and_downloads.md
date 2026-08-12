# 来源与下载清单

## 1. 收集原则

本研究包按三层收集：

1. **核心可复现材料**：与主论文的理论、方法、临床迁移或 PENSO 支柱直接相关，且有公开 PDF/代码/数据。
2. **Lab 全谱系索引**：官方 research map 的 44 篇条目全部进入 02_lab_research_map.md，但不盲目复制所有论文全文。
3. **link-only 材料**：体积过大、缺许可证、仓库失效或无需重复镜像的资源，只记录官方链接、状态与理由。

所有本地二进制资产均列入 CHECKSUMS.sha256。ZIP 均已执行完整 archive test，PDF 均已核验 magic、页数并抽页渲染检查。

## 2. 本地论文

| 文件 | 角色 | 正式来源 | 状态 |
|---|---|---|---|
| papers/01_nlp-psychometrics_2608.07316.pdf | 本研究核心论文 | https://arxiv.org/abs/2608.07316 | arXiv v1，未同行评审 |
| papers/02_deep-lexical-hypothesis_2203.02092.pdf | 原文所述理论前置 | https://arxiv.org/abs/2203.02092；DOI 10.1037/pspp0000443 | 期刊论文的开放预印本 |
| papers/03_dasentimental_2110.13710.pdf | 情绪回忆+认知网络预测 DASS 前作 | https://arxiv.org/abs/2110.13710；DOI 10.3390/bdcc5040077 | 开放预印本 |
| papers/04_emoatlas_s13428-024-02553-7.pdf | TFMN/情绪工具论文 | https://doi.org/10.3758/s13428-024-02553-7 | 开放获取 |
| papers/05_androids-corpus_interspeech-2023.pdf | 主论文人类迁移的数据来源 | https://doi.org/10.21437/Interspeech.2023-894 | ISCA 开放论文 |
| papers/06_mhds_rs-10091363-v1.pdf | PENSO 心理健康数据支柱 | https://doi.org/10.21203/rs.3.rs-10091363/v1 | Research Square/PsyArXiv 前印本 |
| papers/07_meds_2604.27618.pdf | PENSO 数学教育支柱 | https://arxiv.org/abs/2604.27618 | arXiv 前印本 |
| papers/08_cognitive-digital-shadows_2604.27624.pdf | PENSO 社会说服/数字影子支柱 | https://arxiv.org/abs/2604.27624 | arXiv 前印本 |
| papers/09_talk2ai_2604.04354.pdf | 真实人类纵向人机说服对话 | https://arxiv.org/abs/2604.04354 | arXiv 前印本 |
| papers/10_social-llmisinformation_s13688-025-00600-7.pdf | TFMN/EmoAtlas 的大规模 LLM audit 示例 | https://doi.org/10.1140/epjds/s13688-025-00600-7 | 开放获取，CC BY-NC-ND 4.0 |

## 3. 本地代码

| 文件 | 上游 | 固定版本 | 许可证 | 说明 |
|---|---|---|---|---|
| assets/code/emoatlas-main_ed6d786.zip | https://github.com/RiccardoImprota/emoatlas | ed6d786c30ec9c110175606bb457a5c25878d7b1 | BSD-3-Clause | 论文对应的多语言情绪/TFMN 工具；repo README 标记为不再维护，使用前需复核依赖 |
| assets/code/TEA_Networks-main_2026-07-16.zip | https://github.com/MassimoStel/TEA_Networks | 0e2a3940ede30128b905c19fb9115765eac04f47 | BSD-3-Clause | Target–Event–Agent 网络、SVO/coreference/valence/semantic enrichment |

代码仅作为研究快照；没有安装进 workspace、没有运行其 notebook、没有接入 Volvence runtime。

## 4. 本地数据

| 文件 | 来源 | 规模/内容 | 许可证 |
|---|---|---|---|
| assets/data/MHDS-main_4d1a602.zip | https://github.com/MassimoStel/MHDS | 15 个 CSV，75,000 行；含 Codebook、README、生成 notebook | CC0-1.0 |
| assets/data/SociaLLMisinformation_climate-change.zip | https://osf.io/xm5rp/ | 11 个英/意模型 CSV，OSF 原包 39,029,913 bytes | 论文/OSF 条款；论文 CC BY-NC-ND 4.0 |
| assets/data/SociaLLMisinformation_health-misinformation.zip | https://osf.io/xm5rp/ | 11 个英/意模型 CSV，OSF 原包 47,679,899 bytes | 同上 |
| assets/data/SociaLLMisinformation_global-warming.zip | https://osf.io/xm5rp/ | 11 个英/意模型 CSV，OSF 原包 38,727,590 bytes | 同上 |

SociaLLMisinformation 三个文件保持 OSF 原始压缩包，不修改、不重打包。

## 5. 未镜像但已核查的材料

### 5.1 Cognitive Networks primer

- *Cognitive Networks for Knowledge Modeling: A Gentle Introduction for Data- and Cognitive Scientists*
- DOI: https://doi.org/10.1002/wcs.70026
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12976202/
- UniTrento IRIS: https://hdl.handle.net/11572/491811

该文为 2026 年 33 页 CC BY 4.0 方法综述。网页全文已核查；官方 PDF 端点在当前环境返回 403，故保持 link-only。

### 5.2 MEDS 完整仓库

- https://github.com/MassimoStel/MEDS
- archive comment/commit：78259c2c7643f9438af5ad39cc69a9ffd07b06b7
- ZIP 约 367 MB，已完成 archive integrity test。

未复制入研究目录：仓库大、包含约 140,000 个 JSON 与大量 processed validation artifacts，GitHub API 未返回明确 license。论文和数据结构已保留；需要实际复现实验时，应先由用户确认许可证与是否采用 Git LFS/外部对象存储。

### 5.3 ConvinceMe/CDS 完整仓库

- https://github.com/NaviDATA-Repos/PENSO_Data_WP-ConvinceMe_FIS2_UniTrento
- GitHub size metadata：约 2,236,427 KB
- GitHub API 未返回明确 license。

未镜像：体积约 2.2 GB 且许可不清。本地已有对应论文，足以审计 schema、局限与公开主张。

### 5.4 Talk2AI 数据

- 论文所列位置：https://github.com/MassimoStel/Talk2AI/tree/main/Data_paper
- 论文声称包含 Data_files.zip、三份 JSON、六种翻译和分析 notebooks。

2026-08-12 通过 GitHub 页面/API 均返回 404，无法下载。此项是可复现性缺口，不用第三方镜像补齐。

### 5.5 ANDROIDS 原始语音

- 数据入口：https://github.com/androidscorpus/data
- 数据论文报告 118 位 native Italian speakers、228 个录音，其中 64 人有专业诊断；主论文使用其中 115 人的 interview transcripts。

未镜像原始敏感语音。主论文使用的 Borraccino/Whisper 文本转录未发现公开下载链接；本包只保留数据论文。

### 5.6 NLP Psychometrics 精确代码/数据

arXiv v1 未给出 Data Availability/Code Availability 段或专属仓库。MHDS、EmoAtlas 与 ANDROIDS 是其紧邻开放资产，但不能合称“论文完整复现包”。本地材料可复核方法和前序数据工程，不能一键重跑论文全部结果。

## 6. 官方 Lab / 项目来源

- CogNosco Lab official page: https://www.cogsci.unitn.it/1332/cognosco-lab
- Lab site: https://cognosco.dipsco.unitn.it/
- Expanded 44-paper map: https://cognosco.dipsco.unitn.it/research
- PENSO observatory: https://cognosco.dipsco.unitn.it/fis2_penso
- UniTrento PENSO profile: https://mag.unitn.it/innovazione/121727/combattere-lansia-con-lia
- OpenAlex author metadata used only for bibliography cross-check: https://openalex.org/A5074066409

## 7. 原始二手文章

sources/input-article.txt 原样保存用户提供的文章。它是研究触发源，不是数字和结论的权威来源；所有关键判断均回到论文、Lab 官方页面、数据仓库或出版方。

## 8. 复现与版权边界

- PDF 和数据保持原始文件，未修改内容。
- 许可证由上游声明决定；本清单不是法律意见。
- 没有许可证的仓库不做再分发镜像。
- 临床/对话敏感数据不因“公开可得”就自动适合进入产品或训练集。
- 若后续运行第三方代码，必须固定环境、审查依赖与模型下载，并单独记录网络、GPU 与外部 API 成本。
