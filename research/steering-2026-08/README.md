# Steering / 表征干预文献研究（2026-08）

## 缘起

Stage-3 权威扫（36/36，2026-08-04）正式判 `kill-eta`（operationalization-scoped，非理论普遍证伪），
主线转向"**读残差 + 有界 steering + Internal RL 学干预策略**"（[stage3.md](../../.cursor/plans/stage3.md) 方案 A）。
但转向的第一道生死门——**S2 因果 steering**——刚刚 FAIL：
S1 probe 在 layer20/896 维上把子目标读到 heldout `0.9833`（近天花板、且裸基底免费），
S2 沿该 probe 轴施加干预却拿到 target-plus vs noop = `-0.00072`（95% CI `[-0.01787, 0.01809]`），五门全败。

**"可读却不可扳"**是本轮最关键、也最反直觉的现象。本研究下载并深读 2024–2026 的 steering / 表征干预主线文献，
目的只有一个：**判定"可读却不可扳"是我们实现的偶然缺陷，还是学界已知的系统性现象；若已知，学界给出的修法是什么。**

结论（详见 [02_VZ_IMPLICATIONS.md](02_VZ_IMPLICATIONS.md)）：**这是学界已充分刻画的已知失败模式，且有可直接落地的预检与修法。**
我们 S2 复现的是学界的负结果，不是发现了新死路。

## 论文清单

下载脚本 [download_steering_2608.sh](../download_steering_2608.sh)，SHA 与状态见 [download-summary.md](download-summary.md)。已对 `research/**` 去重（9 篇均无重叠）。

| # | 论文 | 场次 | 本地 PDF | 服务的 screen |
|---|---|---|---|---|
| A1 | Understanding (Un)Reliability of Steering Vectors ([2505.22637](https://arxiv.org/abs/2505.22637)) | ICLR 2025 WS | `papers/understanding-unreliability-of-steering-vectors-2505.22637.pdf` | **S2 归因（核心）** |
| A2 | Steering off Course ([2025.acl-long.974](https://aclanthology.org/2025.acl-long.974/)) | ACL 2025 | `papers/steering-off-course-reliability-challenges-acl2025.pdf` | S2 跨模型泛化 |
| A3 | FaithSteer-BENCH ([2603.18329](https://arxiv.org/abs/2603.18329)) | 2026 | `papers/faithsteer-bench-deployment-stress-test-2603.18329.pdf` | 部署级警醒 |
| B1 | Contrastive Activation Addition / CAA ([2312.06681](https://arxiv.org/abs/2312.06681)) | ACL 2024 | `papers/contrastive-activation-addition-caa-2312.06681.pdf` | **S2' 方向来源** |
| B2 | Activation Steering via Generative Causal Mediation / GCM ([2602.16080](https://arxiv.org/abs/2602.16080)) | 2026 | `papers/generative-causal-mediation-where-to-steer-2602.16080.pdf` | **S1 定位（该 steer 哪里）** |
| C1 | ReFT / LoReFT ([2404.03592](https://arxiv.org/abs/2404.03592)) | NeurIPS 2024 | `papers/reft-representation-finetuning-loreft-2404.03592.pdf` | **B screen 血统** |
| C2 | RePS: Reference-free Preference Steering ([NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/eb1ef82926376d252dde00d5dd909f4b-Abstract-Conference.html)) | NeurIPS 2025 | `papers/reps-reference-free-preference-steering-neurips2025.pdf` | **S3 训练目标** |
| D1 | From Weights to Activations（RepE 综述/分类学） ([2026.acl-long.1377](https://aclanthology.org/2026.acl-long.1377/)) | ACL 2026 | `papers/from-weights-to-activations-repe-survey-acl2026-long1377.pdf` | 方法排序 + 功能坐标 |
| D2 | Conditional Activation Steering / CAST ([2409.05907](https://arxiv.org/abs/2409.05907)) | ICLR 2025 | `papers/conditional-activation-steering-cast-2409.05907.pdf` | **S3 门控同构** |

## 文档结构

- **[01_STEERING_LITERATURE_DEEP_READ.md](01_STEERING_LITERATURE_DEEP_READ.md)** — 九篇逐篇深读：核心机制 + 关键量化结论。
- **[02_VZ_IMPLICATIONS.md](02_VZ_IMPLICATIONS.md)** — 落到 Volvence：S2 null 的文献归因、可直接跑的 steerability 预检、S2'/S3 的配方与 B screen 血统对齐。

## 一句话结论

> Steering 是真实的适应范式，但"**拿 probe 权重当方向、在饱和位置单点静态加一把**"恰好是文献里被反复定罪的最弱变体。
> 我们的 S2 null 与学界的可靠性负结果同构；出路是**先做 steerability 预检（方向一致性 + 可分性），把方向来源换成 diff-of-means / 学习式（CAA→ReFT），在有余量处测，用 CAST/RePS 做条件化与策略学习**。
