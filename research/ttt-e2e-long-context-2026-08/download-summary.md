# 来源与下载记录（TTT-E2E 专项，检索日 2026-08-13）

## 主论文

| 项 | 值 |
|---|---|
| 标题 | End-to-End Test-Time Training for Long Context |
| arXiv | [2512.23675](https://arxiv.org/abs/2512.23675)（2025-12） |
| DOI | [10.48550/arXiv.2512.23675](https://doi.org/10.48550/arxiv.2512.23675) |
| 全文来源 | [arXiv HTML v1](https://arxiv.org/html/2512.23675v1)，2026-08-13 抓取，含 Appendix A–D、作者贡献、110 条参考文献，已通读 |
| PDF 归档 | [`../papers/continual-learning-2607/end-to-end-test-time-training-long-context-2512.23675.pdf`](../papers/continual-learning-2607/end-to-end-test-time-training-long-context-2512.23675.pdf) |
| PDF 大小 | 1,009,536 bytes（与 [`../continual-learning-2026-07/download-summary.md`](../continual-learning-2026-07/download-summary.md) 记录一致） |
| PDF SHA-256 | `e1390e9347fe339e31205351420bfddcfb1f5cfa540841706228c07184b7ec7a`（2026-08-13 本机复核） |

PDF 于 2026-07 持续学习横扫（下载脚本 [`../download_continual_learning_2607.sh`](../download_continual_learning_2607.sh)，条目 "S6 test-time-train"）已入库，**本包不重复下载**。

## 代码与衍生

| 项 | 状态 |
|---|---|
| 官方 JAX 实现 [test-time-training/e2e](https://github.com/test-time-training/e2e) | 存在性已核验（646 stars，"Official JAX implementation"），**未克隆、未逐文件核对**；论文 §3 声称全部实验可由该仓库复现 |
| 项目页 PDF [test-time-training.github.io/e2e.pdf](https://test-time-training.github.io/e2e.pdf) | link-only |
| 非官方 PyTorch/NeMo 复现 [banyan-god/ttt-e2e-qwen3](https://github.com/banyan-god/ttt-e2e-qwen3)（Qwen3-4B） | link-only，未核验，不作为证据来源 |

## 证据边界

- 定量结论全部取自论文正文与表格文字（Table 1/2/3、正文数字）；图只核验图题与正文转述，未核验像素。
- 深读中标注"anecdotal"的内容（tokenizer / 数据集敏感性）沿用论文自己的定性，不升级。
- 官方代码未克隆：实现细节（如 TTT 化的具体层选择代码、meta 训练管线）以论文文字为准，未做代码级复核；若未来按 [`02_VZ_IMPLICATIONS.md`](02_VZ_IMPLICATIONS.md) §9 P2 立项评估，应先补代码深读。
