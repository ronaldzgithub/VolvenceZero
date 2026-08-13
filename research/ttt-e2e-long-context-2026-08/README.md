# TTT-E2E 专项：End-to-End Test-Time Training for Long Context（arXiv 2512.23675）

研究对象：Tandon, Dalal, Li, Koceja, Rød, …, Sun, **"End-to-End Test-Time Training for Long Context"**, arXiv [2512.23675](https://arxiv.org/abs/2512.23675)（2025-12）。Astera Institute / NVIDIA / Stanford / UC Berkeley / UCSD 联合工作；一作 Arnuv Tandon，思想与论文出自 TTT 路线发起人 Yu Sun。官方 JAX 代码：[test-time-training/e2e](https://github.com/test-time-training/e2e)。

## 核心裁决

论文把长上下文语言建模**重述为持续学习问题而非架构设计问题**：标准 Transformer + 滑动窗口注意力，测试时在最后 1/4 blocks 的 MLP 权重上按 mini-batch（\(b{=}1\)K）做 next-token prediction 把上下文压进权重，训练时用 meta-learning（梯度的梯度）为"将被更新的初始化"做准备。3B/164B tokens 上随上下文 scaling 与 full attention 相同（Mamba 2 / Gated DeltaNet 做不到），128K 推理延迟常数、快 2.7×。**对 Volvence 的价值三重**：其一，终端 NTP 损失显著击败 KV-Binding 层级代理（Titans / MesaNet / Nested Learning 的共用核心组件，同构裁决）——R-PE"学习信号取自一级预测误差"的域外定量证据，同时对 NL 的引用必须改为"范式强化、实现弱化"；其二，NIAH 全线崩塌（128K 上 0.06 vs full attention 0.99）证明**压缩型权重记忆与无损检索不可互替**——CMS 记忆连续谱分 stratum 的结构性外部证据（R5/R6）；其三，"focus on the present"机制（静态权重须对所有未来都好，被更新权重只需对现在好）给 R2 冻结基底 + 有界快层的分工补上第一条**正向性能论证**。架构本身维持既有裁决：测试时改基底权重是 **R2 反例，不抄架构**；四能力轴上 Appendable / Readable / Steerable 三轴皆 ✗（逐序列瞬态、状态不可读、无条件更新无门控），不得引用为"在线持续学习已解决"。

## 阅读顺序

| 文件 | 内容 |
|---|---|
| [`01_PAPER_DEEP_READ.md`](01_PAPER_DEEP_READ.md) | 论文深读：证据边界、团队与 TTT 谱系、方法逐层拆解（toy → meta-learning → mini-batch+SWA → 实现细节）、KVB→E2E 备选推导与 Table 1 裁决链、全部实验（含 NIAH 负结果与效率账）、批判性评注 |
| [`02_VZ_IMPLICATIONS.md`](02_VZ_IMPLICATIONS.md) | 对 Volvence 的七条增量：R-PE 证据、NL 引用限定、R2 正向论证、CMS 外部证据、微型冻结分层、工程先验四条、四能力轴映射 + 不该借鉴清单 + P1–P3 行动建议 |
| [`download-summary.md`](download-summary.md) | 来源、链接、PDF 归档位置与校验（不重复下载） |

## 与既有记录的关系

本篇在 2026-07 持续学习横扫包已有段落级记录：[`../continual-learning-2026-07/01_LANDSCAPE.md` §S6](../continual-learning-2026-07/01_LANDSCAPE.md)（框架转换 + 三条工程经验）与 [`../continual-learning-2026-07/02_VZ_DELTA.md` §F](../continual-learning-2026-07/02_VZ_DELTA.md)（R2 反例裁决）。本包**不推翻，只增量**：KVB 谱系裁决、四能力轴映射、NIAH→CMS 论证、"focus on the present"→R2、block 内静态第二 MLP、\(k \ge b\) 协同设计等均为本包新增。

## 六个最有价值的单点事实

1. **KVB 谱系裁决**（Table 1，760M/DCLM）：把层级 KV-Binding 重建损失换成网络末端 NTP 损失，loss 2.819→2.806（全表唯一显著步）；Fig 6 中 TTT-E2E 是唯一全程低于 full attention 的方法，TTT-KVB 不是——Titans / MesaNet / Nested Learning 的核心组件被证明对长上下文语言建模非必需（同构裁决，三者本体未被直接评测）。
2. **架构贡献≈0**：关闭 TTT（\(b{=}8\)K）后 TTT-E2E 2.825 ≈ TTT-KVB 2.826 ≈ full attention 2.827——全部增益来自测试时学习过程，论文自评"architecture design plays a minor, supporting role"。
3. **NIAH 诚实负结果**：S-NIAH @128K，full attention 0.99/0.86/0.64 vs TTT-E2E 0.06/0.05/0.03——压缩必然丢掉"看似无关的细节"，压缩型权重记忆在无损回忆上崩溃，与检索是两种不可互替的能力。
4. **"focus on the present"**：对 full attention 的优势在第一次 TTT 梯度步之前（\(t<1\)K、计算图相同、只差权重）就存在——静态权重须准备好所有可能的未来，被更新权重只需对当前 mini-batch 好；且优势主要来自上下文前段。
5. **微型 frozen/adaptive 分层**：被 TTT 的 block 内并联一个**静态第二 MLP**当预训练知识的"安全存储"防遗忘——测试时训练方法自己也必须切出冻结通道。
6. **稳定性工程 + 协同设计**：只更 MLP（更 attention 致外循环不稳）、只更最后 1/4 blocks（1/8 以下不随上下文 scale、1/2 无边际收益）、\(b{=}1\)K（单 token 梯度易爆炸，\(b<1\)K 硬件/稳定性差）、**窗口 \(k \ge b\)**（权重更新到位前由窗口兜住批内上下文）、QK norm 稳外循环；训练侧梯度的梯度是硬伤（8K 预训练慢 3.4×）。

## 材料清单

- 论文 PDF：**已归档**于 [`../papers/continual-learning-2607/end-to-end-test-time-training-long-context-2512.23675.pdf`](../papers/continual-learning-2607/end-to-end-test-time-training-long-context-2512.23675.pdf)（1,009,536 bytes，SHA-256 见 [`download-summary.md`](download-summary.md)），本包不重复下载
- arXiv：[abs](https://arxiv.org/abs/2512.23675) / [HTML v1](https://arxiv.org/html/2512.23675v1)（本次深读全文来源）/ [PDF](https://arxiv.org/pdf/2512.23675) / DOI [10.48550/arXiv.2512.23675](https://doi.org/10.48550/arxiv.2512.23675)
- 官方代码库：[test-time-training/e2e](https://github.com/test-time-training/e2e)（JAX，存在性已核验，未克隆）；项目页 PDF：[test-time-training.github.io/e2e.pdf](https://test-time-training.github.io/e2e.pdf)
- 非官方复现（未核验，仅索引）：[banyan-god/ttt-e2e-qwen3](https://github.com/banyan-god/ttt-e2e-qwen3)（PyTorch/NeMo，Qwen3-4B）
