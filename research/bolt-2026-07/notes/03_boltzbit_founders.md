# Boltzbit 创始人线：MCMC、Boltzmann、ergodic inference 与 prediction markets

> 研究对象：`papers/boltzbit/` 下 Yichuan Zhang、Jinli Hu、Boltzbit 官网相关论文  
> 下载限制：`ergodic-measure-preserving-flows` 的 OpenReview PDF 被 403 阻挡；网页文本已检索，未落入 `papers/`。  
> 排除项：搜索结果中的 `A generalized theory of preferential linking` 作者为 Haibo Hu / Jinli Guo / Xuan Liu，不是 Boltzbit 的 Jinli Hu，未纳入。

## 1. Boltzbit 公开叙事

Boltzbit 官网把自己定位为 “Boltzmann Live-Learning Machine”，强调 Boltzmann machines、learning and inference algorithms、production live-learning。官网公开论文集中在 HMC / MCMC / Boltzmann distribution 的推断与调参：

- `continuous-relaxations-discrete-hmc.pdf`
- `quasi-newton-methods-mcmc.pdf`
- `hmc-hyperparameter-optimization-gradient-strategy.pdf`

这与 BOLT 线索高度一致：它不是常见 RAG / prompt engineering 公司叙事，而是从概率推断、采样、能量模型、在线学习来理解生成模型。

## 2. Yichuan Zhang：把采样变成可优化推断算法

### 2.1 Quasi-Newton MCMC

`quasi-newton-methods-mcmc.pdf` 使用有限窗口的 quasi-Newton 近似，把历史样本与梯度信息转化为局部 Hessian 近似，同时避免无限历史破坏 Markov chain 合法性。

对 BOLT 的启发：

- online update 可以利用历史反馈，但历史必须以固定窗口 / 固定容量 / 受控状态进入更新器。
- 这与 BOLT 的 fixed-capacity latent memory 相容：不是保留全部历史，而是保留足够统计量。
- 但也提示风险：固定窗口/容量必须有理论或经验依据，否则会丢掉长期关系事实。

### 2.2 Continuous Relaxations for Discrete HMC

`continuous-relaxations-discrete-hmc.pdf` 用 Gaussian integral trick 把离散 undirected model 转为连续系统，从而使用 HMC。它说明 Yichuan 线长期关心一个主题：把本来难以优化/采样的离散结构转入可微连续空间。

对 BOLT 的启发：

- 自然语言反馈、用户工作流、偏好、任务上下文都是离散/语义对象，但 BOLT 可能把它们压入连续 latent memory。
- 这种转换有价值，但也有损失：连续 latent 适合推断，不等于适合做公开 contract。
- Volvence 可以在 owner 内部用连续 state，但跨模块仍应发布结构化 snapshot。

### 2.3 Semi-separable HMC 与层级模型

`semi-separable-hmc-bayesian-hierarchical-models.pdf` 通过半可分 Hamiltonian 结构提升层级贝叶斯模型采样效率。它的重要性在于“结构分解”：不是用一个通用大更新器硬扫所有变量，而是利用模型层级结构分块更新。

对 BOLT 和 Volvence：

- 单一 latent memory 若不分 task / preference / relationship / commitment，会丢掉结构分解优势。
- Volvence 的多 owner 不是工程洁癖，而是层级推断中的结构先验。
- 若引入 BOLT-like updater，应按 owner 或轨道分块，而不是全局一个 memory。

### 2.4 Ergodic Inference / EMPF

`ergodic-inference-accelerate-convergence.pdf`、`theory-algorithm-ergodic-inference.pdf` 和未下载成功但可读摘要的 `Ergodic Measure Preserving Flows` 共同表达：在 VI 和 MCMC 之间寻找混合路线，既保留 MCMC 的渐近正确性，又让超参可优化、推断可扩展。

对 BOLT 的启发：

- BOLT 可能不是纯 deterministic memory update，而是 “MCMC/VI 混合思想的 Transformer 化”：用神经网络摊还一个近似推断过程。
- “live-learning” 不等于随便在线训练；更可能是有限状态、可优化超参、渐近/近似 posterior 的工程版本。
- Volvence 可借鉴其“推断算法可学习/可优化”的精神，但不能放弃 runtime contract。

## 3. Jinli Hu：prediction markets 与分布式目标汇聚

Boltzbit CTO Jinli Hu 的可确认早期论文主要是 prediction markets：

- `combinatorial-modelling-learning-prediction-markets.pdf`
- `multi-period-trading-prediction-markets.pdf`

这些论文用市场机制描述多个 agent / predictor 如何通过局部目标和交易动态趋向全局目标。它们与 BOLT 的直接算法关系弱于 Yichuan 的 MCMC 线，但提供了另一个重要背景：从局部参与者、风险度量、市场 maker，到全局 objective 的动态汇聚。

对 BOLT 的启发：

- 用户反馈不是单一 label；它可能来自多个局部信号：explicit correction、implicit behavior、task success、relationship outcome。
- 一个 online learner 需要把多来源信号聚合成全局 update objective。
- Prediction-market 视角提醒我们：局部反馈与全局目标可能不一致，必须有机制调和。

对 Volvence 的启发：

- `prediction_error`、`credit`、`evaluation` 可以看成不同层级的信号市场：PE 是原始事件，credit 是聚合，evaluation 是只读 readout。
- 但这不意味着用市场作为 runtime owner；更适合成为 credit aggregation 的理论类比。

## 4. Boltzbit 与 BOLT 的可能关系

结合 Boltzbit 官网与创始人论文，BOLT 可能不是普通 “LLM + memory”：

```text
Boltzmann / energy model 传统
    -> MCMC/HMC/ergodic inference 的可扩展推断
    -> 学习到的 update / hyperparameter tuning
    -> Transformer 承载摊还 inference
    -> latent memory 支持 live-learning
```

换言之，BOLT 更可能是“把在线贝叶斯推断算法神经化、Transformer 化”的尝试，而不是应用层长期记忆产品。

## 5. 对 Volvence 的意义

Boltzbit 簇支持 Volvence 的若干判断：

- 支持 R2：推断/学习要落在有界自适应层，不能在线重写全基底。
- 支持 R1：fast update 需要固定容量统计量，但慢层仍要处理历史与结构。
- 支持 R8：连续 latent 是内部推断状态，不应成为跨模块共享数据通道。
- 支持 R15：live-learning 必须可回滚，否则一旦 feedback aggregation 出错，会变成不可审计漂移。

它也提出一个挑战：Volvence 当前文档中对 owner-local Bayesian update 的形式化还不够。如果要真正吸收 BOLT 思想，需要把某些 owner 的内部状态从“结构字段 + heuristics”升级为“prior/evidence/posterior/uncertainty”明确建模的 update kernel。
