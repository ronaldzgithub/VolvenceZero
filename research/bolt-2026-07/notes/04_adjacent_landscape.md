# 相邻公开方向：Transformer 作为贝叶斯更新器与在线学习状态机

> 研究对象：`papers/adjacent/` 下 6 篇相邻文献  
> 目的：在 BOLT 原文未公开时，用公开邻近研究界定“Bayesian Online Learning Transformer”可能落在哪个技术空间。

## 1. 核心结论

相邻文献共同指向一个正在成形的范式：

```text
昂贵 inference / learning algorithm
    -> 离线训练成 Transformer / attention / latent-context update operator
    -> 在线只做前向
    -> 用固定容量或结构化 state 承载历史 evidence
```

这正是 BOLT 名字暗示的空间。但这些论文也共同说明：固定容量 state 是高效在线学习的必要工具，不是长期认知系统的完整答案。

## 2. Distribution Transformers

`distribution-transformers-prior-adaptation.pdf` 已在 `distribution-transformers-prior-adaptation-analysis.md` 中单独深入分析。这里提炼与 BOLT 直接相关的点：

- 它把 prior 和 posterior 都表示成 GMM。
- Transformer decoder 接收 prior tokens 与 observation tokens，输出同族 posterior tokens。
- posterior 可递归作为下一步 prior，因此天然支持 sequential filtering。
- 它明确把 “prior + data -> posterior” 做成一次前向的 distribution-to-distribution mapping。

对 BOLT 的意义：

- 如果 BOLT 只说 latent memory 是 Bayesian，但没有显式 prior/posterior 结构，就弱于 Distribution Transformers 的清晰性。
- BOLT 若要更强，应说明 latent memory 的 uncertainty、递归误差、evidence typing。
- Volvence 可借鉴 “posterior-as-next-prior” 接口，但应放在 owner 内部。

## 3. PFN 与 full Bayesian inference in context

### 3.1 Transformers Can Do Bayesian Inference

`transformers-can-do-bayesian-inference-pfn.pdf` 提出 PFN：只要能从 prior over tasks 采样，就可以离线训练 Transformer 直接做 posterior predictive inference。部署时给少量样本，一次前向输出预测分布。

关键启发：

- Transformer 可以学会近似 Bayesian inference，而不只是语言建模。
- prior 是训练数据生成器 / task distribution，不是 prompt 里的一句话。
- 成功依赖训练 prior 覆盖 test-time 任务。

对 BOLT：

- BOLT 的 user workflow adaptation 也需要 meta-prior：训练时必须覆盖将来用户反馈和工作流变化的分布。
- 如果实际用户 out-of-support，latent updater 会失真，需要 uncertainty gate 和 fallback。

### 3.2 Can Transformers Learn Full Bayesian Inference in Context?

`full-bayesian-inference-in-context.pdf` 进一步用 TabPFN encoder + diffusion/flow decoder 生成 posterior samples，接近 HMC 质量但更快。这把 PFN 从 posterior predictive 推进到 full posterior sampling。

对 BOLT：

- 更强的 BOLT 不应只更新一个点估计 memory，而应保留 posterior sample / mixture / uncertainty 的某种表示。
- 但 full posterior sampling 适合结构化统计模型，不等于可以直接解决开放域关系记忆。

## 4. Memory-Based Meta-Learning on Non-Stationary Distributions

`memory-based-meta-learning-nonstationary.pdf` 证明在 piecewise stationary sources 下，Transformer / LSTM / RNN 可以通过最小化 sequential log loss 学到近似 Bayes-optimal predictor，并隐式推断 latent switching points。

这对 BOLT 很关键：

- 用户 workflow 会切换；不是一个稳定任务分布。
- online learner 必须识别 regime / segment boundary，而不是把所有反馈平滑进一个状态。
- BOLT 的 fixed latent memory 若没有 switch / reset / segmentation 机制，容易把新 regime 与旧 regime 混淆。

对 Volvence：

- 这支持 R14：regime 是持久 runtime state，不是 prompt label。
- 也支持 R3/R4：切换条件应在 latent control/state space 中学习。

## 5. Continuous Latent Contexts

`continuous-latent-contexts-online-learning-transformers.pdf` 与 BOLT 的摘要片段最接近：它研究少量 continuous latent context tokens 是否能让 Transformer 实现在线学习算法。论文给出构造：constant-depth transformer 可以用 latent context 保存 weighted majority 和 Q-learning 的算法状态。

关键启发：

- 固定数量 latent token 可以承载在线算法状态。
- 长序列在线任务中，latent state 比把历史全塞进 context 更高效。
- latent state 不必被直接监督，也可以通过 multi-curriculum objective 学出来。

对 BOLT：

- BOLT 的 fixed-capacity latent memory 很有技术合理性。
- 但论文任务是合成在线预测 / Q-learning，不是开放域用户关系和长期事实。
- 它支持“online-fast latent state”，不支持“单一 latent state 替代所有 memory owners”。

## 6. Palimpsa / Bayesian metaplastic attention

`palimpsa-remember-learn-forget-attention.pdf` 把 ICL 视为固定容量记忆中的 continual learning 问题，引入 Bayesian metaplasticity：memory state 的 plasticity 由 importance / prior knowledge 调节。它同时处理 catastrophic forgetting 和 catastrophic remembering。

对 BOLT：

- fixed memory 必然遇到稳定性-可塑性困境。
- 好的 latent updater 需要知道哪些状态该保护、哪些该遗忘、哪些该快速改写。
- “forgetting” 不是 bug，而是带 horizon 的设计选择。

对 Volvence：

- 这支持 CMS 的 promotion / decay / reflection，而不是反驳它。
- online-fast latent memory 应有遗忘机制，但 durable semantic / relationship facts 需要慢层确认。

## 7. 相邻文献对 BOLT 的定位

这批文献使 BOLT 的合理定位更清晰：

```text
PFN / full Bayesian ICL:
    Transformer 可以学会摊还贝叶斯推断

Distribution Transformers:
    prior/posterior 可以作为结构化 token 被递归更新

Memory-Based Meta-Learning:
    非平稳序列需要 latent switch / regime inference

Continuous Latent Contexts:
    固定容量 latent state 可以承载在线学习算法状态

Palimpsa:
    固定容量记忆必须解决 stability-plasticity / forgetting
```

BOLT 若存在，很可能是这些方向在 LLM 个性化 / user feedback online learning 上的交叉。

## 8. 对 Volvence 的意义

相邻文献总体支持 Volvence，而不是证伪：

- 支持 R2：在线前向更新 latent state，比在线改全量权重更合理。
- 支持 R3/R4：学习状态和控制状态可以在 token 空间之上。
- 支持 R1/R5：固定容量 state 有用，但必须有多时间尺度 consolidation。
- 支持 R14：非平稳序列需要 switch / regime inference。
- 支持 R8：如果内部 latent 不可解释，产品系统需要 owner snapshot 把其 public meaning 发布出来。

最大的提醒是：Volvence 不应只把 BOLT-like updater 作为“未来可选算法”空泛记录，而应在某些 owner 中明确建模 `prior -> evidence -> posterior` 的接口，哪怕先以 SHADOW 形式运行。
