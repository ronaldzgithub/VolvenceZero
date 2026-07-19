# José Miguel Hernández-Lobato：摊还推断、不确定性与贝叶斯深度学习

> 研究对象：`papers/jmhl/` 下主题相关核心子集  
> 下载限制：`improving-continual-learning-gradient-reconstructions` 的正式 OpenReview PDF 被 403 阻挡；网页文本与元信息已检索，未落入 `papers/`。

## 1. 为什么只取子集

JMHL 公开论文很多，横跨贝叶斯优化、化学、分子设计、图网络、扩散、采样、医学等方向。BOLT 主题关心的是“LLM / Transformer 如何在线进行贝叶斯式更新”，所以本专题只取与以下问题直接相关的论文：

- 如何把 posterior inference 摊还成神经网络或可训练过程。
- 如何在大模型或神经网络中表达 uncertainty。
- 如何在持续学习中防止遗忘。
- 如何主动选择信息和处理稀疏 evidence。
- 如何在保留 prior 的同时做 RL / policy update。

## 2. 摊还推断主线

### 2.1 Probabilistic Backpropagation 与 Black-box Alpha

`probabilistic-backpropagation-bayesian-neural-networks.pdf` 与 `black-box-alpha-divergence-minimization.pdf` 是早期核心：它们把 BNN 训练理解为近似 posterior inference，而不是单纯点估计优化。关键意义不在某个具体近似族，而在“学习算法必须输出不确定性结构”。

对 BOLT 的意义：

- 如果 BOLT 只输出一个 deterministic latent memory vector，很难名副其实地称为 Bayesian。
- Bayesian online learning 至少需要让旧 memory 表示 prior confidence，新反馈表示 evidence，更新后状态表示 posterior confidence。
- 即使最终实现是隐式 latent，也应在 owner snapshot 中发布 uncertainty / confidence summary。

### 2.2 VIP / FVI：函数空间推断

`variational-implicit-processes.pdf` 与 `functional-variational-inference-spg.pdf` 把 inference 从 weight space 推到 function space。VIP 通过 generalized wake-sleep 更新处理 implicit process；FVI 用 stochastic process generators 作为灵活的函数空间 variational family。

对 BOLT 的意义：

- 用户工作流适应不一定要解释成“模型权重 posterior”，更自然的是“行为函数 / response policy / user-task mapping 的 posterior”。
- latent memory 若作为 context-conditioned function modifier，应评估其对输出函数分布的影响，而不是只看 embedding 距离。
- Volvence 中关系轨道和任务轨道也更像函数空间 posterior：预测用户反应、任务结果、承诺履行，而不是直接学习 token。

### 2.3 PFN 相邻性

JMHL 的许多工作不是 PFN，但与 PFN / Distribution Transformer 的思想相邻：把 expensive inference 前移到训练阶段，部署时一次前向近似 posterior。这是 BOLT 名字中 “Bayesian Online Learning Transformer” 最可能继承的技术精神。

## 3. 不确定性主线

### 3.1 Subnetwork Inference 与 Linearised Laplace

`bayesian-deep-learning-subnetwork-inference.pdf` 和 `sampling-based-inference-large-linear-models-linearised-laplace.pdf` 共同表达一个现实主义路线：完整大模型 posterior 太贵，但可以只在有意义的子空间做更表达力强的 inference。Subnetwork Inference 固定大部分权重，只对小子网做 full-covariance posterior；sampling-based linearised Laplace 进一步扩展大线性模型中的 posterior sampling。

对 BOLT 的意义：

- 在线学习不必也不应动全量模型；只在有界 latent / adapter / owner-local state 中做 posterior update。
- 不确定性可被局部化：哪些维度可塑，哪些维度保持点估计。
- 这直接支持 Volvence R2：稳定基底 + 自适应控制器。

### 3.2 Depth / aleatoric / epistemic uncertainty

`depth-uncertainty-neural-networks.pdf` 与 `decomposition-uncertainty-bayesian-deep-learning.pdf` 说明 uncertainty 不是单一数值。对 BOLT 或 Volvence，尤其要区分：

- evidence 不足导致的 epistemic uncertainty。
- 用户反馈噪声、偏好摇摆、环境随机性导致的 aleatoric uncertainty。
- 模型结构不适配导致的 systematic error。

如果不区分这些来源，一个 online updater 很容易把噪声当偏好、把一次情绪反应当长期事实、把任务失败当关系破裂。

## 4. 持续学习与信息获取

### 4.1 Gradient reconstruction of the past

`Improving Continual Learning by Accurate Gradient Reconstructions of the Past` 的 PDF 被 OpenReview 阻挡，但公开摘要足够确认主线：它把 continual learning 中的 replay / functional regularization / weight regularization 统一到“重构过去梯度”的原则下。结论是：少量 replay 加合适 prior，可显著降低遗忘。

对 BOLT 的意义：

- 固定容量 latent memory 不是免费的；它有稳定性-可塑性困境。
- 在线更新必须能重构“旧任务/旧用户事实仍应产生的梯度方向”，否则会被新反馈覆盖。
- Volvence 的 CMS + reflection 不应被单一 latent state 替代；长期事实需要外显 replay / consolidation / audit path。

### 4.2 EDDI / Icebreaker：信息价值与主动获取

`edddi-efficient-dynamic-discovery-partial-vae.pdf` 与 `icebreaker-efficient-information-acquisition.pdf` 提供另一个重要视角：当 evidence 不完整时，系统不应盲目更新，而应判断下一条信息的价值。它们把缺失数据和信息获取建成 Bayesian decision problem。

对 BOLT 的意义：

- 用户反馈不足时，正确行为可能是 clarification，而不是强行 posterior update。
- Volvence 的 `prediction_error` / `credit` / `relationship_state` owner 应能发出“需要更多 evidence”的信号。
- 这支持 uncertainty-aware memory write gate。

### 4.3 Sequence Tutor：prior policy + bounded RL

`sequence-tutor-kl-control.pdf` 用 KL-control 做 conservative fine-tuning：优化领域 reward，同时保持接近预训练 prior policy。它对 Volvence 特别重要，因为它说明“后训练 / RL 不是随便改模型”，而是应有 prior-preserving regularization。

对 BOLT 的意义：

- BOLT 若基于用户反馈更新输出行为，必须有 prior-preserving 约束，防止单用户反馈把模型推离基本能力和安全边界。
- 这与 Volvence R10/R15 的有界自修改和回滚要求一致。

## 5. 对 BOLT 的综合判断

JMHL 脉络支持如下 BOLT 解释：

```text
旧 latent memory = prior belief state
用户行为/反馈/结果 = evidence
amortised encoder = learned approximate inference algorithm
新 latent memory = approximate posterior
```

但这条解释成立有条件：

- memory 需要携带 uncertainty 或可塑性信息。
- update 需要 evidence typing 和噪声处理。
- posterior 需要可递归使用，但递归误差必须被监控。
- 长期事实不能只靠一个不可解释向量保存。

## 6. 对 Volvence 的意义

JMHL 簇总体支持 Volvence 的大方向，而不是证伪：

- 支持 R2：完整 posterior over all weights 太贵，局部自适应更现实。
- 支持 R-PE：prediction / log loss / surprise 是学习和信息获取的自然入口。
- 支持 R5/R6：continual learning 需要 replay / consolidation，不能只做 fast update。
- 支持 R8/R11：内部 posterior state 应有可发布摘要，否则产品系统不可审计。

真正需要 Volvence 警惕的是：我们若只说“belief state / latent state”，但没有定义 prior、evidence、posterior、uncertainty、recursion error，就会停在概念层。JMHL 线索提示我们必须把 owner-local belief updater 形式化。
