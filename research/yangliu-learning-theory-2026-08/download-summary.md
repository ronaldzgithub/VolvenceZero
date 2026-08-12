# 下载与校验记录

日期：2026-08-12
执行：`download_yangliu_papers.sh`（幂等，可重跑）
校验方法：每个文件 (1) `%PDF` magic + `%%EOF` 尾标；(2) pypdf 解析页数；(3) 首页文本提取与预期标题匹配（deepmind 调研教训：检索到的链接/ID 必须 PDF 文首校验后才算数）。

## 结果统计

- 40 篇清单：**37 篇下载成功并通过标题校验**，3 篇无公开版仅登记引用（#02 投稿中、#37 IJCAI-WS 2007、#40 2004 中文期刊）。
- 补充材料 6 份：博士论文（331 页）、CV、Buy-in-Bulk 技术报告版、lossy coding 期刊稿、度量学习综述短文、ICML13 补充材料。
- `papers/` 合计 44 个 PDF（37 核心 + EuroCG 论文集全册 + 从中抽取的 #27 独立四页 + 6 补充 — 注：#27 计入 37 核心）。

## 来源分布

| 来源 | 篇数 | 说明 |
|---|---|---|
| stevehanneke.com（Purdue 镜像 web.ics.purdue.edu/~hanneke/docs/） | 15 | 全部 Hanneke–Yang 合作论文作者副本 |
| 杨柳 CMU 主页 www.cs.cmu.edu/~liuy/ | 9 | 含两个页面未列出但存在的文件（`DecMarCost.pdf`、`dnf_queries_ITCS.pdf`） |
| arXiv | 4 | #05 (1910.14344)、#14 (1111.0897)、#20 (1801.03190)、#26 (1604.06194) |
| 官方 proceedings（PMLR / NeurIPS / ICML / AAAI CDN / FWCG / EuroCG） | 7 | #03、#13、#18、#29、#32、#39、#27（论文集全册 pp.159–162，已抽取独立 PDF） |
| CMU 技术报告库 reports-archive.adm.cs.cmu.edu | 3 | #30、#31、supp CMU-ML-12-110 |
| 作者合作者主页（Satyanarayanan / Sukthankar） | 2 | #28 (`yang-pami-2010.pdf`)、#33 (`cvpr2008-unified-rahuls.pdf`) |
| CMU KiltHub (figshare) | 1 | 博士论文《Mathematical Theories of Interaction with Oracles》CMU-ML-13-111 |

## 失败与修复记录

1. **#18 ICML 2009**：`machinelearning.org/archive` 已死链 → 改用 `icml.cc/Conferences/2009/papers/472.pdf`（官方归档）。
2. **#27 EuroCG 2016**：Don Sheehy 个人站 `research/*.pdf` 路径整体 404 → 改用 computational-geometry.org 官方论文集全册（11.8MB，276 页），用 pypdf 抽取 PDF 页 169–172（印刷页 159–162）为独立文件。注意：论文实际标题与目录一致，首个文本命中在目录页（PDF 页 7），须以正文页命中为准。
3. **#28 / #33 CMU 服务器断流**：`cs.cmu.edu` 对大文件（2.8MB / 12.9MB）中途断开导致 PDF 截断（`%%EOF` 缺失、pypdf 报 "Stream has ended unexpectedly"）→ 用 `curl -C -` 断点续传循环修复（#33 续传 9 次至 12.9MB 完成）。**教训：`%PDF` magic 通过 ≠ 文件完整，必须校验 `%%EOF` 或 pypdf 可解析。**
4. **执行事故**：首轮运行中途编辑了正在执行的 shell 脚本，bash 逐块读取导致字节偏移错位、第 49 行解析损坏（#25 一度被跳过）。恢复方式：终止后以幂等模式重跑。**教训：禁止编辑正在运行的 shell 脚本。**
5. **标题校验伪差**：4 个文件因 PDF 连字（ﬁ）/空格伪影未匹配关键词（#10 "Identiﬁability"、#17 "Classiﬁcation"、#32 "U nlabeled"、#35 "Reﬁnement"），人工复核首页文本确认均为正确论文。
6. **#29 标题变体**：FWCG 2015 实际标题为 "How Much Distortion Can be **Caused** by One Bad Point?"（论文集 txt 中写作 "Incurred"），作者 Onak/Lenchner/Yang，确认为同一工作。
7. **supp-dml-comprehensive-survey**：实际是 2007 年 8 页短文《An Overview of Distance Metric Learning》（引用其 2006 全文综述），非全文综述本体；作为 C5 背景仍够用。

## 无公开版登记（引用见 00_PAPER_INDEX.md）

- **#02** Active Learning with Identifiable Mixture Models（Annals of Statistics 投稿中；Hanneke 主页列为 in-preparation, joint with Vittorio Castelli and Liu Yang）
- **#37** Resource-constrained Supervised Dimensionality Reduction（IJCAI-WS MIR 2007，工作坊论文集未留存）
- **#40** 基于边缘匹配与多尺度小波变换的图像配准算法（华中科技大学学报 2004，CNKI 无公开 PDF）
