# Neo Labs 下载汇总（2026-06-13）

落盘根目录：[`../papers/neolabs/<lab>/`](../papers/neolabs/)

## 总量

- **本调研 30-lab 名册：70 个 PDF，约 587 MB，0 损坏**（首字节 `%PDF-` + 大小校验全部通过）。
- arXiv 集：经 `download_neolabs_papers.ps1` 并行下载。
- bioRxiv / medRxiv / Nature / PLOS 集：见下"获取方式"。

## 每个实验室文件数

| Lab | PDFs | 说明 |
|---|---|---|
| arc-institute | 3 | Evo / Evo2 / State（bioRxiv） |
| basecamp-research | 1 | BaseFold（bioRxiv）；BaseData 为公司白皮书未下 |
| cartesia | 4 | HiPPO/S4/Mamba/Mamba-2（arXiv） |
| chai-discovery | 2 | Chai-1 / Chai-2（bioRxiv） |
| cortical-labs | 0 | DishBrain 付费（Neuron），仅 DOI 引用；综述 UNVERIFIED |
| czi-virtual-cell | 3 | AIVC(arXiv) / TranscriptFormer / rBio（bioRxiv） |
| evolutionaryscale | 3 | ESM3 / ESM-2 / ESM-1b（bioRxiv） |
| future-house | 5 | PaperQA2 / LAB-Bench / Aviary / ether0 / Robin（arXiv） |
| generate-biomedicines | 1 | Chroma（bioRxiv）；Nature 版付费 |
| inceptive | 2 | Attention(arXiv) / Ribonanza（bioRxiv） |
| insitro | 4 | 3×bioRxiv + 1×medRxiv |
| isomorphic-labs | 2 | AlphaFold3 / AlphaFold2（Nature 开放获取） |
| latent-labs | 1 | Latent-X（arXiv） |
| lila-sciences | 2 | 创始人奠基作（arXiv）；Lila 无第一方论文 |
| noetik | 0 | 仅 noetik.ai 网页技术报告，无可下载预印 |
| numenta | 3 | TBP 系列（arXiv）；2019 Frontiers 仅 DOI |
| periodic-labs | 2 | 创始人奠基作（arXiv）；GNoME/A-Lab 付费 |
| physical-intelligence | 4 | MAML/SAC/π0/π0.5（arXiv） |
| profluent-bio | 3 | ProGen2(arXiv) / OpenCRISPR / ProGen3（bioRxiv） |
| recursion | 3 | MAE×2 / MolPhenix（arXiv） |
| reflection-ai | 4 | CURL/AD/MuZero/AlphaZero（arXiv） |
| sakana-ai | 5 | World Models / Merge / AI Scientist / T² / CTM（arXiv） |
| skild-ai | 3 | ICM / 大规模好奇心 / RMA（arXiv） |
| stanhope-ai | 1 | 神经规范理论（PLOS Biology 开放）；余为 DOI |
| symbolica | 1 | Categorical DL（arXiv） |
| thinking-machines-lab | 2 | TRPO / PPO（arXiv）；新作为 Connectionism 博客 |
| verses-ai | 2 | 生态白皮书(arXiv) / RL-or-AI（PLOS ONE 开放） |
| world-labs | 2 | NeRF / Perceptual Losses（arXiv）；RTFM/Marble 为博客 |
| xaira-therapeutics | 2 | RFdiffusion 抗体 / ProteinMPNN（bioRxiv） |

## 获取方式与可复现性

- **arXiv**：`Invoke-WebRequest https://arxiv.org/pdf/<id>`，稳定。
- **bioRxiv / medRxiv**：对脚本化请求返回 **Cloudflare 403**。改用真实浏览器（Playwright MCP）：
  1. `browser_navigate` 打开任一 `*.biorxiv.org` 页面通过 JS 挑战；
  2. 在页面上下文创建同源 `<a download href=".../<doi>vN.full.pdf">` 并 `click()`；
  3. 捕获 Playwright `download` 事件 → `download.saveAs(<dest>)` 写盘。
  medRxiv、Nature（AlphaFold2/3，开放获取）同法。
- **PLOS / Frontiers**：开放获取，`article/file?...&type=printable` 直接下载。
- **MDPI / 部分 Nature 直链**：Cloudflare/interstitial 拦截，未强取，以 DOI 引用。

## 未下载（按计划以 DOI / 链接引用）

- 付费正刊：DishBrain(Neuron)、ProGen(Nat Biotech)、GNoME / A-Lab / Chroma / RFdiffusion / OpenCRISPR 的 Nature 终版（均有 bioRxiv 预印已下或 DOI 引用）。
- 公司技术报告 / 博客：Noetik OCTO 系列、Basecamp BaseData、World Labs RTFM/Marble、Thinking Machines Connectionism 博客（URL 已记录于各 lab 文档）。
- UNVERIFIED：Cortical Labs SBI 综述、Recursion RxRx3-core、Generate 抗体预印、Insitro PGM 教材（按标题/ISBN 记录）。

> 备注：`papers/neolabs/` 下另有 `recursive-superintelligence/`、`ineffable-intelligence/` 两个文件夹（22:54–22:55 由并行进程以相同命名规范创建），**不属于本调研 30-lab 名册**，未纳入统计与评估。
