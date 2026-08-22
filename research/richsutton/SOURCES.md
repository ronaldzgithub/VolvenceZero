# 一手来源与本地论文索引

> 检索与下载日期：2026-08-22。
> 在线材料可能继续更新；本地 PDF 以 [SHA256SUMS](SHA256SUMS) 固定本次研究所用版本。

## 1. 访谈与 Oak 官方材料

| 来源 | 类型 / 证据级别 | 本研究用途 |
|---|---|---|
| [Sequoia Training Data：Rich Sutton and Khurram Javed — Why AI Models Stop Learning and How to Start It Again](https://sequoiacap.com/podcast/rich-sutton-and-khurram-javed-why-ai-models-stop-learning-and-how-to-start-it-again) | 官方音视频与 transcript，E（其中事实陈述另行核验） | 核对“synthetic data”“weights never change”“20–25%”“step sizes”“generate-and-test”“from scratch”“20W”的上下文 |
| [YouTube 访谈视频](https://www.youtube.com/watch?v=xH7U7w9Qzlo) | 官方发布视频，E | 媒体文章给出的原始入口 |
| [Zeno 节目页](https://zeno.fm/podcast/training-data/episodes/rich-sutton-and-khurram-javed-why-ai-models-stop-learning-and-how-to-start-it-again/) | 节目分发页，辅助元数据 | 交叉确认 2026-08-18 发布日期与约 53 分钟时长 |
| [Oak Lab Mission](https://www.oaklab.ai/mission) | 公司使命，D | OaK、经验扎根的时间抽象、batch-size-one、no-replay、20W 愿景 |
| [Oak Lab Research](https://www.oaklab.ai/) | 官方研究索引，D | 检查公开论文、博客和 coming-soon 状态 |
| [The OaK Architecture](https://www.oaklab.ai/posts/the-oak-architecture) | 官方页面 / 2025 RLC talk，D | 核验目前公开的是讲座入口，而非完整论文或代码 |
| [The Big World Hypothesis](https://www.oaklab.ai/posts/the-big-world-hypothesis) | 官方研究介绍，C/D | 核对有限智能体与巨大世界假说 |
| [Learning from Experience Instead of Curated Datasets](https://www.oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets) | 官方博客，D | NetworkIDBD / NoisyMNIST 初步演示；截至研究日期未发现配套论文或代码 |

## 2. 下载的论文与同行评审文件

### 2.1 Nature 正式论文

**Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R., & Sutton, R. S. (2024). “Loss of plasticity in deep continual learning.” Nature 632, 768–774.**

- DOI / 文章页：[10.1038/s41586-024-07711-7](https://doi.org/10.1038/s41586-024-07711-7)
- Nature PDF：[正式 PDF](https://www.nature.com/articles/s41586-024-07711-7.pdf)
- 本地副本：[papers/nature-2024-loss-of-plasticity.pdf](papers/nature-2024-loss-of-plasticity.pdf)
- 出版信息：2023-08-11 收稿，2024-06-12 接收，2024-08-21 online，Nature 632（2024-08-22 issue）。
- 授权：文章标注 Creative Commons Attribution 4.0 International。
- 代码：[shibhansh/loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity)
- 证据级别：A。

### 2.2 Nature 同行评审文件

- 官方附件：[Peer Review File](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07711-7/MediaObjects/41586_2024_7711_MOESM2_ESM.pdf)
- 本地副本：[papers/nature-2024-peer-review-file.pdf](papers/nature-2024-peer-review-file.pdf)
- 内容：初审意见、作者逐条回复、修订后评审；用于区分初稿过强主张与最终结论。
- 证据级别：A（用于理解审稿约束和作者承认的限制，不是独立实验）。

### 2.3 Continual Backprop

**Dohare, S., Sutton, R. S., & Mahmood, A. R. “Continual Backprop: Stochastic Gradient Descent with Persistent Randomness.”**

- arXiv：[2108.06325](https://arxiv.org/abs/2108.06325)
- 本地副本：[papers/continual-backprop-2108.06325.pdf](papers/continual-backprop-2108.06325.pdf)
- 用途：Nature CBP 机制的原始版本；包含 contribution × adaptation utility、bias correction 和 Adam state 重置等细节。
- 证据级别：B。

### 2.4 Step-size Optimization

**Degris, T., Javed, K., Sharifnassab, A., Liu, Y., & Sutton, R. “Step-size Optimization for Continual Learning.”**

- arXiv：[2401.17401](https://arxiv.org/abs/2401.17401)
- 本地副本：[papers/step-size-optimization-2401.17401.pdf](papers/step-size-optimization-2401.17401.pdf)
- 用途：区分 Adam / RMSProp 的 normalization heuristic 与 IDBD 的 lifetime-objective step-size optimization。
- 证据级别：B；直接实验主要是简单线性、长期漂移问题，不是深层大模型。

### 2.5 Alberta Plan

**Sutton, R. S., Bowling, M., & Pilarski, P. M. “The Alberta Plan for AI Research.”**

- arXiv：[2208.11173](https://arxiv.org/abs/2208.11173)
- 本地副本：[papers/alberta-plan-2208.11173.pdf](papers/alberta-plan-2208.11173.pdf)
- 用途：追踪 OaK 的架构前身；特别是 STOMP、feature / subtask / option / model 生成检验和 Oak 步骤。
- 证据级别：C（研究计划，不是完成报告）。

### 2.6 Big World Hypothesis

**Javed, K., & Sutton, R. S. “The Big World Hypothesis and its Ramifications for Artificial Intelligence.”**

- 作者官方 PDF：[The Big World Hypothesis](http://incompleteideas.net/papers/The_Big_World_Hypothesis.pdf)
- Oak 介绍：[官方研究页](https://www.oaklab.ai/posts/the-big-world-hypothesis)
- 本地副本：[papers/big-world-hypothesis-2024.pdf](papers/big-world-hypothesis-2024.pdf)
- 用途：理解“世界始终大于有限智能体”、资源约束、近似与长期 tracking。
- 证据级别：C；作者明确称其为 hypothesis，引用证据也可能有替代解释。
- 下载备注：作者站点在当前环境的 HTTPS 证书链异常，本地文件由仓库既有同源官方副本复制后重新计算 SHA-256；官方 URL 记录如上。

### 2.7 Era of Experience

**Silver, D., & Sutton, R. S. (2025). “Welcome to the Era of Experience.”**

- 官方 PDF：[Google DeepMind hosted PDF](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)
- 本地副本：[papers/era-of-experience-2025.pdf](papers/era-of-experience-2025.pdf)
- 用途：四个经验时代特征；也用于核验其对 simulator、prediction error 和 in-context adaptation 的更细边界。
- 证据级别：C。

### 2.8 OML：为持续学习元学习表征

**Javed, K., & White, M. (2019). “Meta-Learning Representations for Continual Learning.” NeurIPS 2019.**

- 出版方页面：[NeurIPS 2019 proceedings](https://papers.nips.cc/paper/2019/hash/f4dd765c12f2ef67f98f3558c282a9cd-Abstract.html)
- 本地副本：[papers/meta-learning-representations-neurips-2019.pdf](papers/meta-learning-representations-neurips-2019.pdf)
- 用途：Javed 路线的直接前身；用 meta-objective 学习可在线更新、低干扰的表示。
- 证据级别：B。

## 3. 其他核心一手文本

| 来源 | 作用 |
|---|---|
| [Richard Sutton, “The Bitter Lesson” (2019)](https://www.incompleteideas.net/IncIdeas/BitterLesson.html) | 通用计算与搜索/学习方法优于长期编码人类知识的历史主张 |
| [Sutton publications](https://www.incompleteideas.net/papers.html) | 作者官方论文索引与版本交叉核验 |
| [Nature article page](https://www.nature.com/articles/s41586-024-07711-7) | 正式元数据、开放获取状态、代码与附件入口 |

## 4. Volvence 内部依据

下列链接是本次比较的主要内部 SSOT；它们不是 Sutton 路线的外部证据：

- [能力域索引](../../docs/specs/00_INDEX.md)
- [四能力轴](../../docs/appendable-readable-learnable-steerable.md)
- [多时间尺度学习](../../docs/specs/multi-timescale-learning.md)
- [连续记忆](../../docs/specs/continuum-memory.md)
- [PE loop](../../docs/specs/prediction-error-loop.md)
- [credit / ModificationGate](../../docs/specs/credit-and-self-modification.md)
- [steering runtime](../../docs/specs/steering-runtime.md)
- [temporal abstraction](../../docs/specs/temporal-abstraction.md)
- [environment interface](../../docs/specs/environment-interface.md)
- [synthetic experience corpus](../../docs/specs/synthetic-experience-corpus.md)
- [CMS / ATLAS / TITANS evidence](../../docs/specs/cms-atlas-titans-uplift.md)

## 5. 引用注意

- 本目录不把 Oak 官网未发布的算法当成论文。
- 本目录不把访谈嘉宾口头估计当成 Nature 结论。
- 本地归档用于研究可追溯性；引用或再分发时应遵循各论文原始许可。
- 任何后续结论如果依赖 Oak 新发布材料，应记录检索日期，并与本次冻结版本分开。
