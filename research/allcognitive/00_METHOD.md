# All Cognitive 研究包：方法与统一模板

日期：2026-07-20

## 1. 范围

本研究包分析本轮新增的 106 篇论文：

- 28 篇核心增量：`research/papers/sweep-2607/`
- 75 篇广度层本地 PDF：`research/papers/frontier-map/`
- 3 篇下载受限、已有官方元数据与全文页面的 link-only 论文

五卷分配（每篇只设一个主归属，必要时在跨轴综合中引用）：

1. 架构、学习、记忆与信用：25 篇
2. 安全、监控、自修改与发布门：26 篇
3. 关系、社会认知与多主体：21 篇
4. 具身、世界模型与潜在动作：19 篇
5. 脑科学、PE、睡眠与群体认知：15 篇

合计 106 篇。

## 2. 每篇论文统一分析模板

每篇至少回答以下问题：

1. **论文事实**：作者、机构、年份、发表状态、数据与实验规模。
2. **核心问题**：论文真正试图证伪或解决什么，不复述营销标题。
3. **机制拆解**：状态、更新规则、目标函数、时间尺度、训练 / 推理写面。
4. **关键证据**：主实验、对照、消融、统计结果；区分相关、因果、形式证明。
5. **确证价值**：支持哪些 P1–P7 / R-ID，以及支持到什么强度。
6. **反证价值**：它推翻了哪些直觉，或暴露 VZ 哪条路线可能失败。
7. **局部可借算法**：可进入 spec、benchmark、shadow、rare-heavy 或仅作反例的部分。
8. **不可外推边界**：任务域、样本、闭源依赖、token-space、端到端更新、评估反灌等限制。
9. **成熟度与裁决**：A（主链证据）、B（强基线 / 反例）、C（观察），并给出行动结论。

## 3. 证据强度

- **因果证据**：随机干预、消融、counterfactual、受控神经操控。
- **形式证据**：定理或模型检查；必须写明证明覆盖的是原系统还是抽象。
- **行为证据**：held-out、跨任务、跨 embodiment、长期或真实用户实验。
- **相关证据**：probe、几何、回归、可视化；不得写成因果机制。
- **工程证据**：吞吐、成本、故障率、部署分布；不得替代安全性质。

## 4. VZ 边界

- 跨模块只通过不可变 snapshot；消费者不重建 producer 内部状态。
- frozen substrate 与 adaptive controller 分离。
- 长期策略学习不落在 token / prompt / Markdown 文本空间。
- PE 是原始信号；reward、curiosity、evaluation 是带假设的 readout。
- evaluation 只读，不反向成为在线学习源。
- 自修改必须 bounded、可证伪、可回滚，并有退出条件。
- 关系质量不以 engagement、信任或依赖单指标最大化。

## 5. 输出文件

- `01_ARCHITECTURE_LEARNING.md`
- `02_SAFETY_GOVERNANCE.md`
- `03_SOCIAL_RELATIONSHIP.md`
- `04_EMBODIMENT_WORLD_MODELS.md`
- `05_NEURO_COGNITIVE_SCIENCE.md`
- `06_CROSS_AXIS_SYNTHESIS.md`
- `README.md`
