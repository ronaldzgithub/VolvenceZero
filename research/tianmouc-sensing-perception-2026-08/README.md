# Tianmouc 传感–感知专项：Brain-inspired Visual Sensing–Perception（Nature Sensors 2026 封面）

研究对象：Lin & Chen et al., **"Brain-inspired visual sensing–perception for open-world environments"**, *Nature Sensors* 1, 701–717 (2026)，doi [10.1038/s44460-026-00095-3](https://doi.org/10.1038/s44460-026-00095-3)。清华类脑计算研究中心（Tianmouc 芯片团队，前作 Nature 2024 封面）的算法续作，2026 年 8 月刊封面文章。

## 核心裁决

在 Tianmouc 互补视觉芯片（COP 慢完整强度 / AOP 快稀疏差分）之上，论文用**两阶段类脑学习**取得开放世界极端条件下的鲁棒感知：bottom-up 阶段以数据内禀物理一致性（Poisson 梯度–强度一致）为监督、零标注自监督训练内部模型 IGFNet，产出标准视觉表示 e-VGT；top-down 阶段冻结该表示、用基础模型在 e-VGT 上蒸馏伪标签调制下游深度估计与实例分割。**对 Volvence 的价值是同构证据而非可搬模块**：它在感知域独立重演了我们的信号分层（R-PE）、SSOT（R8）、冻结基底（R2）、SSL 先行（R13）与涌现门控（R3/R4）；引用时必须携带三条边界——其记忆库训练后冻结（不支持 Appendable）、其 top-down 是训练期蒸馏（不是运行时 steering）、其框架只覆盖 NL 的 SSL 半边（无 RL/credit）。

## 阅读顺序

| 文件 | 内容 |
|---|---|
| [`01_PAPER_DEEP_READ.md`](01_PAPER_DEEP_READ.md) | 论文深读：证据边界声明、前作背景、两阶段框架、生物学根基（90 条引文重建的论证链）、批判性评注 |
| [`02_CODE_DEEP_DIVE.md`](02_CODE_DEEP_DIVE.md) | 官方代码库逐类拆解（付费墙下方法部分的唯一完整公开证据）：IGFNet、FiLM 门控融合、五尺度记忆库、对称自监督损失、退化免疫训练 |
| [`03_VZ_IMPLICATIONS.md`](03_VZ_IMPLICATIONS.md) | 对 Volvence 的借鉴：4 条同构证据、4 条可移植机制、5 条不可类比警告、P1–P3 行动建议 |
| [`download-summary.md`](download-summary.md) | 来源与下载边界（正文付费墙 link-only 的检索记录） |
| [`sources/`](sources/) | 已归档的一手来源文本（Nature 页面、代码库 README 等） |

## 六个最有价值的单点事实

1. **监督信号零外部依赖**：伪 GT 是从传感器自身空间梯度（SD）解 Poisson 方程得到的 HDR 融合——物理一致性当 teacher，无标注、无 judge（与 R12 兼容的监督来源选择）。
2. **鲁棒性是被逼出来的**：训练时对两条输入通路分别随机 patch 遮挡加噪，门控层唯一稳定解是学会"按传感条件分配信任"——涌现门控替代 if/else 路由的实证（`no-keyword-matching` 的域外印证）。
3. **在标准表示上标注错误更少**（Extended Data Fig. 7b）：VIS 标注在 e-VGT 上做显著优于在原始 RGB 上做——steering-human-anchor"标注在 owner 锚上做"的定量理由。
4. **基础模型零微调可消费 e-VGT**（Extended Data Fig. 5）：Depth Anything V2-L 不动权重直接产生尺度对齐深度——"快照标准性"可操作判据的原型。
5. **记忆读出 = 残差注入 + 方差约束 + 一键关断**：与我们 steering 的 norm cap / strict noop / 单字段回滚同一设计语汇，经同行评审系统验证。
6. **差分与状态分开采样**：AOP 测变化量、COP 测绝对量、下游学习融合——R-PE"PE 是一级采集信号，不从状态序列事后重建"的芯片级同构。

## 材料清单

- 论文原文：**付费墙 link-only**（无 arXiv/SharedIt/自存档，检索记录见 download-summary）
- 官方代码库：[`Tianmouc/TMC-SSL-Representation`](https://github.com/Tianmouc/TMC-SSL-Representation)（已完整克隆核验，未入库）
- 公开数据集：[Tianmouc-R @ HuggingFace](https://huggingface.co/datasets/ordinarabbit/Tianmouc-R)（129 GB，未下载）
- 伴随 PDF ×2：[`../papers/tianmouc-sensing-2608/`](../papers/tianmouc-sensing-2608/)（arXiv 2504.19253 模态定量评估 + ICCV 2025 扩散重建），下载脚本 [`../download_tianmouc_sensing_2608.sh`](../download_tianmouc_sensing_2608.sh)
