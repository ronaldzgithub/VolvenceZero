# Ross M. Clarke：优化器、二阶结构与可摊还更新

> 研究对象：`papers/clarke/` 下 4 篇公开论文  
> 与 BOLT 的关系：解释 BOLT 若采用 “amortised encoder 更新 latent memory”，其 update rule 训练可能需要怎样的优化与稳定化技术。

## 1. 论文清单

- `scalable-one-pass-weight-update-hyperparams.pdf`：高维 weight-update hyperparameter 的 one-pass 隐式微分优化。
- `adam-through-second-order-lens-icml2024.pdf` / `studying-kfac-heuristics-adam-second-order-lens.pdf`：把 Adam 放到 K-FAC / 二阶启发式视角下分析，提出 AdamQLR。
- `series-hessian-vector-products-saddle-free-newton.pdf`：用 Hessian-vector product 级数近似 saddle-free Newton 的关键矩阵操作。

`Adam through a Second-Order Lens` 的 2023 workshop PDF 被 `opt-ml.org` 返回 510 阻挡，目录内下载的是 ICML 2024 扩展正式版和 arXiv 版，内容覆盖该工作主线。

## 2. 主线判断

Clarke 线索不是“发明一个新记忆模块”，而是研究一个更底层的问题：如何让更新本身可计算、可稳定、可在大模型规模下近似二阶化。这与 BOLT 的关系很直接：如果 BOLT 的 latent memory 每轮由 encoder 更新，那么真正难点不只是 memory state 设计，而是如何训练一个在长序列反馈下不发散、不遗忘、不过度拟合单次反馈的 update operator。

### 2.1 One-pass hyperparameter optimisation

`Scalable One-Pass Optimisation...` 的关键点是把“学习率、动量、每参数超参”等原本需要多次重训搜索的东西，变成一次训练过程中可微优化的对象。它使用隐式微分近似 hypergradient，目标是在不重启训练的情况下优化大量连续 hyperparameter。

对 BOLT 的启发：

- latent updater 的“可塑性、遗忘率、反馈权重、uncertainty gate”本质上也是高维 update hyperparameters。
- 这些超参若靠手调，会很难覆盖不同用户和不同反馈噪声；更合理的是离线训练一个可微的 update rule。
- 这支持 BOLT-like 机制走“离线学更新规则、在线只前向更新 state”的路线，而不是在线微调全量 LLM。

### 2.2 AdamQLR：二阶启发式的可迁移部分

AdamQLR 系列工作把 K-FAC 的 damping 和 quadratic-model learning-rate selection 移植到 Adam 的 update direction 上。它问的不是“二阶方法是否一定更强”，而是“二阶方法中哪些稳定化启发式独立有效”。结论更谨慎：K-FAC 启发式有时很有价值，但不是普适魔法；未调 AdamQLR 可以接近调参基线，说明稳定化机制本身值得抽象。

对 BOLT 的启发：

- BOLT 的 amortised encoder 不应只输出 deterministic delta；它需要类似 damping / trust region / local quadratic model 的更新节制。
- 对单轮强反馈的响应应当是“受不确定性和局部曲率调制的 posterior shift”，不是简单覆盖旧 memory。
- 如果未来实现 `AmortisedBeliefUpdater`，应把 update magnitude、confidence、forget gate 作为训练目标的一部分，而不是后置工程阈值。

### 2.3 Hessian-vector products：可扩展二阶近似

`Series of Hessian-Vector Products...` 的价值在于：避免显式 Hessian / eigendecomposition，用 Hessian-vector products 的级数逼近 saddle-free Newton 所需的操作。这说明 Clarke 线关注的是“把理论上昂贵的二阶更新压缩成可运行的近似算子”。

对 BOLT 的启发：

- 若 latent memory update 要接近 Bayesian posterior update，真正 posterior curvature 可能很贵；需要可近似的曲率或 Fisher 信息。
- 不必把完整 covariance 显式保存在全局 state 中，但 owner 内部可以维护低秩 / 对角 / implicit curvature。
- 这给 Volvence 的 owner-local updater 一个方向：公开 snapshot 保持可解释，内部可用低成本二阶近似决定更新强度。

## 3. 对 BOLT 机制的推断

若 Clarke 深度参与 BOLT，合理推断 BOLT 不会只是“一个 latent token cache”。它更可能包含：

- 训练过的 update rule，而非手工写的 memory overwrite。
- 对反馈噪声和旧状态置信度的稳定化机制。
- 某种近似 posterior / curvature / plasticity 表示，哪怕论文中可能只以 latent state 呈现。
- 离线优化 update rule，在线不做全量梯度更新。

## 4. 对 Volvence 的意义

Clarke 簇强化了 Volvence 的 R2：在线适应应落在有界控制器 / memory / owner-local state，而不是基础模型权重。它也强化 R15：任何新自适应层都需要可解释的 update evidence 和回滚路径，因为 update rule 本身会变成系统行为的高杠杆部件。

最值得吸收的不是某个优化器，而是这个设计原则：

```text
离线训练稳定 update operator
在线执行低成本、有界、可审计的 state update
```

这与 BOLT 可能路线高度吻合，也与 Volvence 的 frozen substrate + adaptive controller 分层一致。
