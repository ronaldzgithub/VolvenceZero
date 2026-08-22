# 研究方法、证据等级与事实核验

## 1. 研究问题

本次研究不是对访谈做摘要，而是回答六个可以影响 Volvence 设计的问题：

1. Sutton 对静态大模型、合成数据和“经验学习”的批评，哪些是可证伪技术命题，哪些只是研究品味或战略判断？
2. Nature 论文实际证明了什么？它对可塑性、遗忘、迁移和大模型分别能说到哪一步？
3. continual backprop、step-size optimization、generate-and-test 与 OaK 的关系是什么？
4. Oak Lab 当前公开了哪些可复现证据，哪些仍是使命、博客、讲座或未来计划？
5. Volvence 与其路线在经验源、时间尺度、学习信号、表征更新、规划和资源约束上有哪些相同与不同？
6. 哪些思想应变成近期实验，哪些应明确暂缓或拒绝？

## 2. 证据分级

本文档统一使用以下证据等级，避免把访谈的确定语气误写成论文结论：

| 等级 | 定义 | 本研究中的例子 | 可支持的结论 |
|---|---|---|---|
| A | 同行评审、直接实验，任务与结论相匹配 | 2024 Nature 可塑性论文及同行评审 | 测试设置内的可塑性下降、CBP 效果与限制 |
| B | 已发表会议论文或公开预印本，有直接实验但规模/任务受限 | OML、原始 CBP、step-size optimization | 机制可行性和局部经验结论 |
| C | 研究计划、观点论文、假说或综述 | Bitter Lesson、Alberta Plan、Big World、Era of Experience | 研究框架、优先级和待验证假设 |
| D | 公司官网、研究博客、讲座或演示，方法/数据不完整 | OaK 页面、NetworkIDBD NoisyMNIST 博客 | 当前研究方向和初步信号 |
| E | 访谈估计、修辞或预测 | “1/4 智能”“巨大错误”“完全可治”“20W” | 形成问题意识，不能作为机制证据 |

本研究的用词约束：

- A/B 级材料可以写“论文显示”“实验中观察到”。
- C 级材料写“提出”“主张”“假设”。
- D 级材料写“官方页面称”“博客报告”，并注明不可复现边界。
- E 级材料写“访谈判断”“估计”“愿景”，不写成事实。

## 3. 来源选择

优先使用一手来源：

- Sequoia 官方访谈页面和逐字稿；
- Nature 正式论文、扩展材料和 peer review file；
- Oak Lab 官方 mission、研究列表和博客；
- 作者或正式出版方提供的论文 PDF；
- Volvence 当前 spec、代码和 evidence 文档。

媒体文章仅用于确定研究入口，不作为关键技术结论的最终证据。没有保存或复制完整访谈逐字稿；本目录只保留必要的短语、释义和出处链接。

完整来源表见 [SOURCES.md](SOURCES.md)。

## 4. 核验流程

### 4.1 外部材料

1. 对照 Sequoia 官方 transcript，确认“合成数据”“权重变化”“1/4 智能”“步长优化”“generate-and-test”“从头训练”“20W”等说法的上下文。
2. 对照 Oak 官方 mission 和 research 索引，检查 OaK 是否已有论文、代码和实验。
3. 下载 Nature Version of Record 和 peer review file；全文提取并渲染全部页面，检查页数、文字、图表和版面完整性。
4. 下载技术谱系中的 CBP、step-size、Alberta Plan、Big World、Era of Experience 和 OML 原文。
5. 计算 SHA-256，记录在 [SHA256SUMS](SHA256SUMS)。

### 4.2 Volvence 材料

依照仓库 `AGENTS.md`，先由 [docs/specs/00_INDEX.md](../../docs/specs/00_INDEX.md) 定位能力域，再检查：

- [Appendable / Readable / Learnable / Steerable](../../docs/appendable-readable-learnable-steerable.md)
- [multi-timescale learning](../../docs/specs/multi-timescale-learning.md)
- [continuum memory](../../docs/specs/continuum-memory.md)
- [Prediction Error loop](../../docs/specs/prediction-error-loop.md)
- [credit and self-modification](../../docs/specs/credit-and-self-modification.md)
- [steering runtime](../../docs/specs/steering-runtime.md)
- [temporal abstraction](../../docs/specs/temporal-abstraction.md)
- [environment interface](../../docs/specs/environment-interface.md)
- [synthetic experience corpus](../../docs/specs/synthetic-experience-corpus.md)
- [CMS evidence / uplift gate](../../docs/specs/cms-atlas-titans-uplift.md)

同时检查 CMS NumPy/Torch 实现、learned update rule 和相关反遗忘 evidence tests。对比结论以当前代码行为为准，不以旧研究中的愿景描述为准。

## 5. 术语：四个经常被混在一起的问题

| 术语 | 本研究采用的定义 | 一个系统可能出现的状态 |
|---|---|---|
| Plasticity / 可塑性 | 长期学习后，仍能有效学习当前新数据的能力 | 旧知识没忘，但再也学不快 |
| Stability / 保持 | 新学习发生后，旧能力不被破坏 | 新知识学得快，但旧能力迅速下降 |
| Transfer / 迁移 | 旧经验是否让未来任务学得更快或表现更好 | 可塑性正常但没有正迁移，甚至负迁移 |
| Coherence / 连贯性 | 跨时间的信念、承诺、关系和自我状态是否保持可解释一致 | 分类准确率稳定，但主体语义相互冲突 |

Nature 论文主要测第一项。Volvence 的产品主张至少还包含后三项，尤其是关系与主体连续性。因此“Nature 论文解决了 Volvence 的持续学习问题”是错误推论。

## 6. 媒体文章逐项核验

| 媒体表述 | 核验结果 | 精确边界 |
|---|---|---|
| 大模型部署后通常不更新参数 | 基本正确 | 是当前主流部署惯例，不是架构上绝对不能更新 |
| 上下文不等于真正学习 | 是 Sutton 在访谈中的强定义 | *Era of Experience* 的脚注却允许由环境反馈驱动的 in-context adaptation 算 RL 适应；Sutton 自己的公开材料存在定义宽窄差异 |
| Nature 发现“可塑性损失” | 正确 | 与 catastrophic forgetting 不同 |
| Javed 等人在 Nature 发表该研究 | 不准确 | Javed 不是 Nature 论文作者，但参与了早期 OML、CBP 相关路线与 step-size 论文 |
| 合成数据是巨大错误 | 原话存在 | 是对“以静态、人类策划生成数据替代现实经验”的批评，不是所有模拟和合成数据无效的定理 |
| CBP 解决灾难性遗忘 | 不正确 | Nature 明说当前 CBP 不解决 forgetting；它维持的是可塑性 |
| 步长优化 + generate-and-test 可形成下一代算法 | 研究路线，尚未完整验证 | 两者分别有局部论文证据；深网络、真实长期流、联合算法尚缺完整公开验证 |
| 需要从头训练新基础模型 | 访谈中的 Oak 主张 | Nature/CBP 并未证明所有现有模型都不能渐进改造；对大模型的工程结论仍待证 |
| OaK 已形成完整技术架构 | 只能说有路线图 | 有 Alberta Plan 的前身和 2025 RLC talk，但截至研究日期没有公开论文/代码/复现实验 |
| 20W 万亿参数心智 | 愿景 | 仅有摩尔定律式外推，没有系统设计、能耗测量或规模实验支撑 |

## 7. Nature PDF 完整性与视觉检查

- `nature-2024-loss-of-plasticity.pdf`：26 页，Nature 正式开放获取版本。
- `nature-2024-peer-review-file.pdf`：40 页，包含审稿意见、作者回复和修订轨迹。
- 两份文件均完成全文文本提取和逐页渲染；检查封面、正文、多栏版式、图表、扩展数据、参考文献和同行评审页面，没有发现截断或损坏。
- 其余 PDF 完成文件类型、页数、全文文本可读性和首页可视渲染检查；所有副本均有 SHA-256。

## 8. 研究限制

1. 访谈是 2026-08 的公开材料，Oak 仍可能在本研究截止日期后发布论文或代码。
2. 对 Oak 的“尚无证据”只指公开可检索材料，不代表内部没有结果。
3. 没有运行 Nature 官方代码，也没有在本次任务中开展新实验；结论来自论文、同行评审、现有仓库证据与代码审阅。
4. 没有把关系智能的长期收益化约为标准分类损失；建议中的 benchmark 是机制验证层，不替代产品级结果。
5. 本次没有修改 `docs/DATA_CONTRACT.md`：研究建议还不是已批准的新跨模块状态。真正实施时，若新增正式 snapshot / slot，必须先完成 owner 和契约注册。
