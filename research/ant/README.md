# 蚂蚁神经结构研究 — Ant Neuroscience Survey

> **目的**：系统检索蚂蚁（及邻近物种：果蝇、蜜蜂、蝗虫、线虫）神经结构/连接组/计算模型文献，识别对 VolvenceZero（VZ）R-PE + R1–R20 架构的**支持证据**与**反证/挑战**，并评估"用 VZ 这套系统做一个真的数字蚂蚁"这件事的可行性。
>
> **完成时间**：2026-07-19。
> **调研范围**：2016–2026，重点 2023–2026 新作。26 篇主名单，15 篇本地 PDF 存档（见 [`papers/`](papers/)），11 篇全文已读仅链接引用。

## 阅读路径

### 只有 5 分钟

读 [`00_executive_summary.md`](00_executive_summary.md)。

### 想看论文全貌

读 [`01_paper_survey.md`](01_paper_survey.md) — 8 个子轴 26 篇，每篇一句话定位 + 链接。

### 想看这些研究怎么印证/挑战我们的设计

- 支持证据 → [`02_supporting_evidence.md`](02_supporting_evidence.md)（按 R-ID 分组）
- 反证/挑战 → [`03_counter_evidence.md`](03_counter_evidence.md)（4 条硬挑战 + 应对）

### 想看"能不能做数字蚂蚁"这个核心问题

读 [`04_digital_ant_feasibility.md`](04_digital_ant_feasibility.md) — 这是本次调研的主要交付物：VZ 模块 ↔ 蚂蚁神经子系统的映射表、分阶段落地路线、当前差距清单。

## 文档清单

| 文档 | 内容 | 阅读时间 |
|---|---|---|
| [`00_executive_summary.md`](00_executive_summary.md) | 一句话判断 + 5 条关键发现 | 5 min |
| [`01_paper_survey.md`](01_paper_survey.md) | 8 轴 26 篇论文全览 | 15 min |
| [`02_supporting_evidence.md`](02_supporting_evidence.md) | 按 R-ID 分组的支持证据 | 15 min |
| [`03_counter_evidence.md`](03_counter_evidence.md) | 4 条反证 + 每条的应对策略 | 15 min |
| [`04_digital_ant_feasibility.md`](04_digital_ant_feasibility.md) | 数字蚂蚁可行性评估 + 落地路线图 | 25 min |
| [`_download_summary.md`](_download_summary.md) | 下载执行记录 | 2 min |
| [`download_papers.sh`](download_papers.sh) | 下载脚本（可重新运行） | — |

## 子轴总览

| 轴 | 主题 | 关键论文 |
|---|---|---|
| connectome | 连接组测绘 | CRANTb 蚁脑连接组、克隆掠夺蚁参考脑、蚁脑嗅觉编码 |
| mushroom-body | 菌体计算模型 | Ardin 2016、SNN-MB 2024、latent learning 2024 |
| central-complex | 中央复合体/环形吸引子 | 头方向环路、CX 多模态导航、locust 角速度模型 |
| collective-intelligence | 集体智能/超个体 | Bayesian superorganism、Active Inferants |
| caste-plasticity | 分工与脑可塑性 | Atta 转录组、神经肽重编程、tramtrack |
| neuromodulation | 神经调质 | dopamine-octopamine opponency |
| connectome-critique | 连接组充分性批判 | OpenWorm 十年停滞、C. elegans 权重优化 |
| miniaturization | 微型化物理极限 | 神经元核大小限制、无核神经元 |
| robotics | 工程验证 | AntBot 天顶罗盘导航机器人 |
| （交叉引用）| 全脑仿真先例 | FlyWire 完整果蝇连接组 + embodied 仿真（无 PDF，链接引用） |
