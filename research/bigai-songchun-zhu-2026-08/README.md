# 北京通用人工智能研究院（BIGAI）/ 朱松纯团队研究项目

> 基准日：2026-08-15
> 对象：北京通用人工智能研究院（Beijing Institute for General Artificial Intelligence, BIGAI）、朱松纯及其长期学术谱系与当前合作团队
> 性质：只读研究包；不修改 Volvence spec、代码、数据契约或 `WiringLevel`

## 一句话结论

朱松纯学术路线最稳定的内核，不是某一种视觉模型，也不是简单的“符号主义反对深度学习”，而是：**从小样本中选择足够的结构约束，用可组合的生成模型显式解释世界，再把物理、因果、意图、效用与交流纳入同一个可推理的智能体闭环。** BIGAI 近年的工作把这条路线从图像解析扩展到因果概念学习、社会智能、具身机器人、多智能体世界模型和自动数学发现；但这些成果仍是若干强机制与领域验证的组合，不能等同于“通用智能已经实现”。

## 核心判断

1. **主线有清晰连续性。** `Region Competition / Minimax Entropy / FRAME` 解决“有限统计量如何约束视觉分布”；`Image Parsing / Stochastic Grammar / AOG` 解决“如何形成可组合、可解释的世界结构”；“暗物质”路线把不可见的物理、意图、因果与效用变成推断对象；“小数据、大任务”则把上述原则提升为 AGI 研究纲领。
2. **BIGAI 不是只做认知理论。** 官方目录同时覆盖视觉、机器学习、机器人、语言、认知推理、多智能体与仿真；近期成果大量使用基础模型、RL、世界模型和真实机器人。因此，把该路线概括成“反大模型”并不准确；更准确的是：它反对把统计相关性和规模本身当作充分的智能理论。
3. **最强的实证不是宏大宣言，而是局部闭环。** 例如少样本概念归纳、软件侦察 agent 的任务内双向价值对齐、3D 具身通才、社会世界模型、跨机器人意图对齐、统一力位控制和 TongGeometry，分别在受控任务上把结构表示、推断、行动或自举学习闭合起来。
4. **需要严格区分三种归属。** 本包逐篇标注“朱松纯署名”“BIGAI 目录收录但无朱松纯署名”“历史学术谱系”。机构目录不是个人论文列表，频繁合作者也不是正式组织架构的替代品。
5. **对 Volvence 的最大价值在 Readable，最明显缺口在 Appendable。** AOG、FPICU、因果变量、心理状态与意图空间为“命名读出”提供了丰富外部依据；但多数论文没有跨 session 的可追加记忆契约，也没有 Volvence 所要求的 `PE → credit → bounded gate`、strict noop 和单字段回滚。

## 研究快照

- BIGAI 官方英文研究目录在基准日包含 **268 条记录、267 个唯一标题**；唯一重复标题是 `M3Bench` 的 2025/2026 双记录。这里的“记录”不是去重论文数，也不是影响力排名。
- 年份分布：2021 `15`、2022 `36`、2023 `47`、2024 `72`、2025 `66`、2026 `32`。
- 朱松纯个人官方 publication 页面解析得到 **337 条展示记录、334 个唯一标题**。这同样只是页面快照，不替代 DBLP、Crossref 或引文数据库。
- 本包筛选 **38 篇主索引论文**，其中 20 篇组成 15 个核心深读单元，覆盖 1996–2026 的四代研究路线。
- 本地归档 **25 份合法开放 PDF**，全部通过 `pdfinfo` 与 SHA-256 校验；`Artificial Social Intelligence` 因自动下载端点返回 404/405，`Cross-Robot Intention Alignment` 因未发现合法自动开放全文，均保留正式 link-only 入口。

统计口径、来源等级和去重方法见 [00_METHOD_AND_SCOPE.md](00_METHOD_AND_SCOPE.md)。

## 文档结构

| 文件 | 内容 |
|---|---|
| [00_METHOD_AND_SCOPE.md](00_METHOD_AND_SCOPE.md) | 名称消歧、检索方法、来源等级、样本口径、纳入与排除规则 |
| [01_INSTITUTION_AND_TEAM.md](01_INSTITUTION_AND_TEAM.md) | 机构定位、目录快照、作者网络与研究方向；明确“合作簇 ≠ 组织架构” |
| [02_CORE_PAPER_INDEX.md](02_CORE_PAPER_INDEX.md) | 38 篇主要论文索引、署名边界、开放全文与选择理由 |
| [03_INTELLECTUAL_LINEAGE.md](03_INTELLECTUAL_LINEAGE.md) | 四代思想谱系深读：熵—文法—暗物质—小数据大任务 |
| [04_BIGAI_FRONTIER_2021_2026.md](04_BIGAI_FRONTIER_2021_2026.md) | BIGAI 时期五条前沿支线、实验证据与不能外推的边界 |
| [05_VOLVENCE_FOUR_AXES.md](05_VOLVENCE_FOUR_AXES.md) | Appendable / Readable / Learnable / Steerable 对账、可借鉴项与禁用类比 |
| [06_RESEARCH_PROPOSAL.md](06_RESEARCH_PROPOSAL.md) | 可证伪后续项目 `DEPSI-Continuity`：问题、分组、指标、12 周计划和 kill criteria |
| [07_PDF_EVIDENCE_AUDIT.md](07_PDF_EVIDENCE_AUDIT.md) | 正文/页码级证据核验、标题与实验数字纠偏、署名复查 |
| [SOURCES_AND_DOWNLOADS.md](SOURCES_AND_DOWNLOADS.md) | 一手来源、开放 PDF、link-only 项与复核入口 |
| [download_core_papers.sh](download_core_papers.sh) | 幂等下载与 PDF 完整性校验脚本 |
| `papers/` | 开放获取核心论文与 `SHA256SUMS`；付费墙内容不绕过访问控制 |

## 推荐阅读顺序

- 只想了解朱松纯路线：`README → 03 → 02`
- 关心 BIGAI 当前研究：`README → 01 → 04`
- 关心 Volvence：`README → 05 → 06`
- 要复核事实或论文原文：`00 → 02 → 07 → SOURCES_AND_DOWNLOADS → papers/`

## 研究边界

- `Tong Test`、`Dark, Beyond Deep`、物理—社会世界模型论文中有一部分是立场、框架或研究议程；本包不会把它们写成已经完成的通用系统。
- “human-level”“generalist”“zero data”等词严格沿用各论文限定的任务、数据与验证器边界，不做无条件外推。
- 任何把本研究转成代码或运行时 owner 的工作，都必须另立收敛包，先在 `docs/DATA_CONTRACT.md` 注册 slot，并按 SHADOW → ACTIVE 迁移；本研究包本身不授权实现。
