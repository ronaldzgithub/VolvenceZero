# Rich Sutton 持续学习路线研究包

> 研究截止日期：2026-08-22（Asia/Shanghai）
>
> 研究对象：2026-08-18 Sequoia《Training Data》访谈、Oak Lab / OaK 官方材料、2024 年 Nature 论文及其同行评审，以及相关技术谱系。
> 本目录只新增研究材料和论文副本，没有修改 Volvence 运行时代码、契约或现有 spec。

## 一句话结论

Sutton 这套路线最值得 Volvence 吸收的，不是“停用合成数据”或“立刻在线更新大模型”，而是一个更严格的系统要求：**智能体必须在有界资源下，从连续、带后果的真实经历中维持长期可塑性，生成并检验内部表征，再用可验证的抽象参与预测与规划。**

Volvence 在架构方向上与其高度同向：已有多时间尺度记忆、Prediction Error（PE）→ credit → gate、冻结基底上的有界控制、环境结果结算、World / Self 隔离和可回滚接线。它甚至在契约、可观测性、回滚和安全边界上比 Oak 当前公开叙事更完整。但是，我们尚没有 Nature 意义上的长期可塑性测量，也没有证明 CMS 在长流中具有显著的吸收与保持优势；ETA / steering 仍有 SHADOW、DISABLED 或既往否证边界。因此当前最准确的判断是：

- **方向同构，但证据层级不同。**
- **Volvence 已有适合承接持续学习实验的系统骨架，尚未证明持续学习机制成立。**
- **下一步应先建立可塑性可观测性，再在 CMS 小网络上做可回滚的 CBP / 步长优化实验，而不是碰基础模型权重。**

## 关键事实校正

媒体文章的主线基本忠实于访谈，但有几处必须在技术判断前校正：

1. Nature 论文是 2024 年的 *Loss of plasticity in deep continual learning*。作者为 Shibhansh Dohare、J. Fernando Hernandez-Garcia、Qingfeng Lan、Parash Rahman、A. Rupam Mahmood、Richard S. Sutton；**Khurram Javed 不是这篇 Nature 论文的作者**。
2. 论文研究的是 **loss of plasticity（持续学习新数据的能力下降）**，并明确把它与 **catastrophic forgetting（旧知识在不再呈现时被忘掉）** 区分开。论文提出的 continual backprop（CBP）在所测任务中维持可塑性，但**当前版本不解决遗忘问题**。
3. Nature 论文没有做大语言模型规模的实验。证据来自长期 ImageNet / CIFAR 序列任务、较小深度网络和 PPO + MuJoCo。作者明确说，系统性研究大语言模型的成本过高。
4. “合成数据是巨大错误”是访谈中的战略判断，不是 Nature 论文结论。更准确的解释是：**不要让人类策划或固定生成器生产的静态数据，替代智能体从环境后果中获得且可自我纠错的经验**；这不等于模拟器、反事实生成、受控合成基准或世界模型一律无用。
5. “大模型只完成四分之一的智能”“灾难性遗忘完全可治”“五到十年做出 20W 万亿参数心智”都是访谈中的估计或愿景，没有相应测量或同行评审证据。
6. Oak Lab 公开了使命、研究博客与 OaK 架构讲座入口，但截至本研究日期，**没有公开 OaK 算法论文、完整实现或可复现实验**；不能把公司路线图当成已验证系统。

## 本目录内容

| 文件 | 内容 |
|---|---|
| [00_METHOD_AND_SOURCES.md](00_METHOD_AND_SOURCES.md) | 研究问题、证据分级、术语、核验方法与事实纠偏 |
| [01_INTERVIEW_IDEAS.md](01_INTERVIEW_IDEAS.md) | 对访谈核心思想逐条拆解：最强解释、证据、反例与边界 |
| [02_NATURE_PAPER_DEEP_READ.md](02_NATURE_PAPER_DEEP_READ.md) | Nature 主文、扩展材料和同行评审的详细研究 |
| [03_OAK_RESEARCH_STACK.md](03_OAK_RESEARCH_STACK.md) | Bitter Lesson → OML → CBP → Alberta Plan → Big World → Era of Experience → OaK 的谱系 |
| [04_VOLVENCE_COMPARISON.md](04_VOLVENCE_COMPARISON.md) | 与 Volvence 的架构、机制、证据和风险逐项对比 |
| [05_RECOMMENDATIONS.md](05_RECOMMENDATIONS.md) | 面向 Volvence 的分阶段实验与工程建议，含 owner、信号、门禁和回滚 |
| [SOURCES.md](SOURCES.md) | 一手来源、论文元数据、在线链接和本地副本索引 |
| [SHA256SUMS](SHA256SUMS) | 本地 PDF 完整性校验值 |
| `papers/` | Nature 主文、同行评审文件及六篇相关论文的本地 PDF |

## 综合判断

### Sutton / Oak 真正提出了什么

把访谈口号还原成可执行主张后，其路线由五层组成：

1. **经验源**：从行动后果而不是静态人类数据集中学习。
2. **在线学习器**：batch size 1、持续更新、固定或受限资源，不靠无限回放全部历史。
3. **可塑性机制**：通过步长优化和 generate-and-test，避免网络随着训练变得越来越难学。
4. **时间抽象**：自行发现跨多时间尺度的 feature、subtask、option 和 model。
5. **规划闭环**：用预测误差检验抽象，用抽象改善规划与下一次行动。

Nature 论文只直接支持其中第 3 层的一部分：长期训练确实会损失可塑性，CBP 在测试范围内能缓解或消除所测下降。它没有验证完整 OaK，也没有验证真实部署中的记忆、抽象或长期规划。

### Volvence 已经具备什么

按 Appendable / Readable / Learnable / Steerable 四轴看：

- **Appendable**：CMS、State-KV、semantic snapshot 和跨 session hydration 已定义多时间尺度经历载体。
- **Readable**：内部语义由唯一 owner 命名并发布 frozen snapshot；PE、credit、steering condition 有正式交换边界。
- **Learnable**：学习源原则上受限于 PE 及其下游 credit；evaluation / judge 不得回灌。
- **Steerable**：冻结基底上的残差干预受 norm cap、strict noop、无 free bias 和 `WiringLevel` 约束。

这使 Volvence 很适合成为 Sutton 路线的“有界、安全、可审计版本”。但当前公开在仓库里的证据边界仍需正视：

- CMS 三个 band 的小型 MLP 没有正式发布有效秩、饱和单元、权重增长、梯度/更新比等可塑性读数。
- 现有 learned update rule 产生的是目标/频带级的有界 `step_scale`，不是每个权重的 IDBD。
- CMS Gate 5 的 510-turn 对照只证明流程可运行，吸收/保持差值远低于预注册阈值，结论是 `not-supported`。
- steering 仍默认 SHADOW，CMS Torch backend 默认 DISABLED；不能把接线存在说成 production ACTIVE。
- 既往 ETA operationalization 已被 `kill-eta` 否证；不能把 `z_t / beta_t` 的架构意图说成已经实现 Oak 所称的自主时间抽象。

### 最优先的行动

1. 先由 `vz-memory` owner 发布 CMS 每个 band 的 frozen plasticity readout，并建立长期漂移基准。
2. 在 `research/` 隔离复现 BP、L2、shrink-and-perturb、CBP，分别测新知识吸收、旧知识保持、迁移和计算成本。
3. 只有复现通过后，才在一个 CMS band 上以 SHADOW 接入 output-preserving CBP：重置隐藏单元输入，清零对应输出，重置 optimizer state，保留 checkpoint 和单字段回滚。
4. 把 per-parameter / per-group step-size optimization 限定在 CMS 或小控制器；元目标只能来自 PE / credit，不得把 evaluation 变成 reward。
5. 用“预测可验证性 + 对反事实规划的增益”作为候选抽象效用，而不是关键词、prompt 标签或人工 judge；所有晋升仍经 `ModificationGate`。
6. 保留合成数据，但只用于机制探针、课程、反事实、SHADOW 和 promotion test；真实环境后果才负责结算部署学习的事实与信用。

完整实施建议见 [05_RECOMMENDATIONS.md](05_RECOMMENDATIONS.md)。

## 不应从这次研究推出的结论

- 不应立即在线微调整个基础 LLM。
- 不应删除现有合成数据管线或把所有模拟环境判为无效。
- 不应把 CBP 当成灾难性遗忘、长期语义一致性或关系连续性的现成解法。
- 不应把 Oak 的 batch-size-one / no-replay 目标直接变成 Volvence 的硬约束。
- 不应把 OaK、NetworkIDBD、20W 或“1/4 智能”作为已验证的架构事实。
- 不应因为理念相近，就宣称 Volvence 已通过持续主动学习或 ETA 的 thesis gate。

## 与既有研究的关系

仓库已有 [2026-07 Sutton「经验时代」研究](../sutton-era-of-experience-2026-07/README.md)。本目录不重复其论文归档，而是围绕 2026-08 访谈新增四项工作：

1. 下载并深读 Nature 正式版本及同行评审文件；
2. 对 Oak 公司最新公开主张做证据分层；
3. 校正“可塑性 = 遗忘”“合成数据一概无效”等常见混淆；
4. 依据 Volvence 当前代码与 promotion evidence，给出更保守、可回滚的差距判断。
