# Nature 论文深读：它证明了什么，也没有证明什么

## 1. 论文定位

论文：Dohare et al., *Loss of plasticity in deep continual learning*, Nature 632, 768–774 (2024)，DOI [10.1038/s41586-024-07711-7](https://doi.org/10.1038/s41586-024-07711-7)。

本地材料：

- [Nature 正式主文](papers/nature-2024-loss-of-plasticity.pdf)
- [同行评审文件](papers/nature-2024-peer-review-file.pdf)
- [官方代码](https://github.com/shibhansh/loss-of-plasticity)

最重要的阅读结论是：**这是一篇“长期训练会让标准深网逐渐学不动”的系统性实证论文，不是一篇已经解决完整 continual learning 的论文。**

## 2. 核心问题与操作定义

论文把 plasticity 定义为：一个网络在经历长时间学习以后，学习当前新数据的能力。

其基本测量思路不是只观察当前准确率，而是把长期训练过的网络与合适基线比较：

- 相同新任务上，长期网络能否像早期一样降低损失？
- 它能否优于浅层或 linear baseline？
- 它能否达到重新初始化网络的学习效果，并利用旧知识更快学习？
- 表征内部是否保持足够多样性？

论文明确区分：

| 问题 | 关注对象 | 典型测量 |
|---|---|---|
| Plasticity | 还能否学会新的 / 当前数据 | 新任务学习速度、相对从头训练的表现、长期 online loss |
| Forgetting / stability | 还记不记得过去有用的信息 | 旧任务性能、回访状态的保持、平均最终任务性能 |

主文大部分 benchmark 有意保留旧数据或让相邻任务基本独立，以隔离 plasticity。这个实验设计对回答“能否继续学”很干净，但不能回答“能否一边学新的一边记住旧的”。

## 3. 主要实验

### 3.1 Continual ImageNet

设置：

- 从 1,000 个 ImageNet 类别中反复抽取二分类任务；可形成约 50 万种类别对。
- 每个任务使用对应类别训练/测试样本；主分析运行至 5,000 个任务。
- 卷积网络在每个任务上继续训练，不回到初始权重。
- 对比普通 backprop、L2、shrink-and-perturb、CBP 等。

观察：

- 标准深网开始时明显优于线性网络；之后学习能力持续变差，最终接近或低于线性基线。
- 增大学习率不能恢复长期可塑性；它会在其他阶段破坏性能。
- L2 和 shrink-and-perturb 减轻下降；CBP 在所测设置中最好。
- 全量重训能恢复初始学习能力，却丢掉迁移、持续部署和计算优势。作者在审稿回复中报告，第 5,000 个任务上 CBP 约 10 epochs 到峰值，而从头训练约 125 epochs；这项数字来自回复解释，不应误写为主文中完整的计算成本 benchmark。

意义：在长期任务序列中，“网络还在做梯度更新”不等于“网络还保持最初的学习能力”。

### 3.2 Class-incremental CIFAR-100

设置：

- 使用 18 层 ResNet、batch normalization、data augmentation、L2 和 learning-rate schedule 等实践中常见组件。
- 从 5 类开始，每次增加 5 类，直到 100 类。
- **旧类别仍继续出现在训练数据中**，有意避免把 forgetting 混入主要结论。
- 与在同一类别集合上从头训练的网络比较；结果平均自 30 次运行。

观察：

- 增量网络在前 40 类能利用旧表示，优于从头训练。
- 40 类以后相对优势反转并持续恶化；100 类时 base system 比从头训练低约 5 个百分点。
- base system 的 dormant units 增加、表征 stable rank 下降。
- shrink-and-perturb 缓解，CBP 在这一测量下消除了相对的可塑性缺口，并保持较高 stable rank 与极少 dormant units。

意义：可塑性损失不只存在于玩具 MLP，也能出现在 ResNet + 常用训练组件中；但任务仍不是自然语言流或真实部署。

### 3.3 PPO + Ant：变化摩擦系数

设置：

- MuJoCo `Ant-v3`，智能体控制八个关节向前移动。
- 每 200 万 time steps，在 episode 边界改变地面摩擦；摩擦从 `[0.02, 2.00]` 对数均匀抽样。
- 观察约 2,000 万 steps；主图平均 100 次运行。

观察：

- 标准 PPO 初期学得好，第一次变化后逐次变差。
- 特别调过的 optimizer 版本好一些，仍随变化衰退。
- L2 和 **CBP + 少量 L2** 在测试期内大体维持适应能力。

边界：RL 结果不能写成“CBP 单独稳定解决”。主文明确说在 RL 中只展示 CBP + L2，因为无 L2 时对超参数非常敏感。

### 3.4 PPO + Ant：环境保持不变

设置：固定中间摩擦系数，运行 5,000 万 steps；主图平均 30 次运行。

观察：

- 标准 PPO 约 300 万 steps 后开始下降，2,000 万以后几乎每回合失败。
- 网络 dormant units 上升、stable rank 明显下降、平均权重持续增大。
- L2 防止权重增长，但过小权重也限制了策略对好行为的“承诺”。
- CBP + 轻微 L2 综合表现最好，内部指标也更稳定。

意义：可塑性损失不要求外部环境显式变化；策略变化会改变自身收集的数据，RL 本身就是非平稳学习过程。

### 3.5 扩展实验

Methods 还包含 online permuted MNIST、slowly changing regression、不同激活、宽度、学习率、optimizer、dropout、normalization 等大量组合。它们支持一个一致模式：标准训练初期有效，长期后表征多样性下降；持续小权重和持续引入变异的算法更稳。

“Adam、dropout、normalization 加剧可塑性损失”只在这些实验配置内成立。正确表述是：**这些常用方法不自动解决长期可塑性，有些所测配置甚至更差**，而不是“Adam 在所有持续学习中都不能用”。

## 4. 论文观察到的三个内部相关量

### 4.1 Dormant / constant / saturated units

单元长期几乎不激活、输出接近常数或进入饱和区时，梯度难以使其参与新表征。ReLU 中常表现为 dead unit；tanh / sigmoid 中更可能表现为饱和或近常数。

### 4.2 权重绝对值增长

大权重可让单元对小更新不敏感，或把激活推入饱和区域。L2、shrink-and-perturb 和 CBP 都在不同程度上抑制权重无界增长。

### 4.3 Effective / stable rank 下降

隐藏表征的奇异值逐渐集中，说明很多维度变得冗余。低 rank 不是单纯“参数少”，而是网络可用于表达新方向的有效维度变少。

### 因果强度

论文做了多种干预和关联分析，结果与“小权重 + 持续多样性有助可塑性”一致。但作者没有建立完整、唯一的机制因果链；扩展讨论明确说，随机小初始化究竟哪些性质最关键仍不清楚。不能把三个指标任何一个单独当成可塑性的充分条件。

## 5. Continual Backprop 的算法

### 5.1 Utility

Nature 版本使用简化的 contribution utility。对第 `l` 层隐藏单元 `i`：

```text
instant contribution = |h[l,i]| × Σ_k |w[l,i,k]|
u[l,i] ← η · u[l,i] + (1 - η) · instant contribution
```

其中 `h` 是单元输出，`w` 是所有 outgoing weights，论文所有实验的 EMA decay `η = 0.99`。直觉是：如果一个单元的输出乘 outgoing influence 始终很小，它容易被其他单元淹没。

这是局部启发式，不是对整个网络损失的精确 marginal contribution。原始 [CBP 预印本](papers/continual-backprop-2108.06325.pdf) 还使用 contribution × adaptation 的更复杂 utility 和 bias correction；Nature 修订后发现，对于 ReLU，简化 contribution utility 几乎同样有效，于是采用更易实现版本。tanh / ELU 等非零饱和激活可能仍需更谨慎的 utility。

### 5.2 选择与替换

每次更新：

1. 正常 forward、计算损失、backprop / optimizer step；
2. 更新所有隐藏单元 age 和 utility；
3. 只在 age 超过 maturity threshold `m` 的成熟单元里选择；
4. replacement accumulator 按 eligible count × replacement rate `ρ` 增长；
5. 累积超过一个单元时，选 utility 最低者；
6. 从初始分布重新采样该单元的 incoming weights；
7. 把该单元 outgoing weights 设为 0；
8. 重置 utility、age，以及实现中对应的 optimizer state。

清零 outgoing weights 使替换瞬间不改变已表示函数；maturity 防止新单元因 utility 暂时为零立刻再次被删。

### 5.3 替换速率并不高

CIFAR-100 的例子使用 `ρ = 10^-5`；最后隐藏层 512 单元时，每步期望替换 `0.00512` 个，即约每 200 次更新替换一个。算法强调“持续但少量”的随机性，而不是周期性摧毁网络。

### 5.4 复杂度与隐藏成本

原始 CBP 论文主张与 backprop 具有相同渐进计算复杂度和固定内存，因为 utility / age 是按单元保存，替换率很低。但工程上仍有：

- utility、age、replacement accumulator 状态；
- optimizer moments 的局部重置；
- 分布式参数切片与一致性；
- 编译图、量化和 fused kernels 中的动态写入；
- checkpoint / rollback 兼容性。

因此“同渐进复杂度”不等于“LLM 上零额外成本”。Nature 没有提供 LLM 级 wall-clock、通信或能耗数据。

## 6. 为什么清零 outgoing 不等于不遗忘

替换当下，新单元输出权重为零，因此函数值近似不变。但是：

1. 被替换的旧单元如果承载稀有但未来会回来的知识，局部近期 utility 可能低估它；
2. 后续梯度会重新使用这个单元，并可改变其他单元；
3. 当前 utility 只看当前数据，不保存历史重要性；
4. 任务头、共享表示和 optimizer state 仍会受新数据影响。

Nature 扩展讨论对此非常直接：当前 CBP 不处理 forgetting，未来可考虑长期 utility 或与 stability 方法组合。

对 Volvence 更重要的是：semantic snapshot、承诺、边界和关系状态不是可随意重置的神经特征。即便 CBP 用于某个 owner 的内部近似器，也不能跨过正式状态与 lineage 删除语义事实。

## 7. 同行评审揭示了什么

同行评审文件比媒体解读更能显示结论的真实强弱。

### 7.1 审稿人认可的问题证据，质疑解决方案

一位审稿人认为可塑性问题的展示清楚且有影响力，但对 selective reinitialization 是否是完整 continual-learning 解法明显保留，要求同时看 stability / forgetting、全重置基线和实际部署性。编辑也概括：论文展示问题的部分很强，“solution”部分更弱。

这提示我们把论文分成两个置信度不同的结论：

- “标准深网长期会损失可塑性”：强。
- “当前 CBP 是完整、可部署的持续学习解”：弱得多。

### 7.2 现实性问题推动了新增实验

初稿更依赖 permuted MNIST / ImageNet 序列。审稿人要求更现实的架构和 RL，作者于是加入：

- ResNet-18 + class-incremental CIFAR-100；
- 标准 PPO + MuJoCo 的 stationary / non-stationary 长期实验；
- 更现实的 Ant 摩擦范围；
- 更明确的相关工作和 limitation。

因此主文的广度不是最初就有，而是审稿压力下补强的。

### 7.3 “比遗忘更根本”被删除

初稿主张 plasticity 比 catastrophic forgetting 更 fundamental。审稿人认为没有充分论证；作者虽然在回复中仍表达个人观点，最终删除了“谁更根本”的比较。这正说明访谈中的强措辞不能覆盖最终论文的克制结论。

### 7.4 Utility 的理论基础仍薄弱

审稿人询问能否用 Taylor expansion 等更原则化方式定义 utility。作者承认当前指标是 heuristic，未来应研究 global utility。最终扩展讨论保留了这一限制。

### 7.5 LLM 主张被收窄

评审质疑把现代 LLM 训练描述为“不允许持续学习”是否过强。最终论文改成更准确的部署事实：训练通常在发布前关闭，持续训练新数据往往难以正确平衡新旧数据；同时明确 LLM 系统性实验在当前成本下不可行。

## 8. 论文没有证明的事项

| 常见外推 | 为什么不成立 |
|---|---|
| CBP 解决灾难性遗忘 | 论文明确否认；utility 只看当前数据 |
| CBP 解决 continual learning | 它只处理一项必要条件；记忆选择、迁移、长期信用、资源和安全仍开放 |
| 结论已在 LLM 验证 | 没有 LLM 实验，作者明确说明成本原因 |
| 随机替换就是时间抽象 | CBP 只生成低层隐藏特征，没有 subtask / option / model / planning 语义 |
| “无限维持”是数学证明 | 指测试时间范围内没有观察到下降，不是任意时长保证 |
| Adam 总会更差 | 只在所测长期任务、架构和超参数下观察到 |
| 低 rank 是唯一根因 | 它是相关指标和部分机制线索，不是完整充分因果解释 |
| 不需要 replay | 论文说大 replay buffer 可能掩盖新数据影响，但没有证明 no-replay 全面优于 bounded replay |
| 只要重置就能获得正迁移 | 多数任务主要测 plasticity；正向迁移是后续目标 |

## 9. 对 Volvence CMS 的直接映射

当前 [CMSBandMLP](../../packages/vz-memory/src/volvence_zero/memory/cms_band_mlp.py) 的形式可概括为：

```text
y = clamp(x + W1 · tanh(W2 · x))
```

它是三个时间 band 各自拥有的小型两层残差 MLP，使用手写梯度与 momentum。与 Nature 网络不同：

- 激活是 tanh，不是主实验中的 ReLU；首要风险是饱和 / 近常数，而不只是 dead unit。
- 有 residual identity path；隐藏分支重置瞬间可通过清零对应 `W1` column 保持输出。
- `W1` 初始为零、`W2` 小随机，天然类似“新 feature 先不影响输出”的结构。
- 它的状态只属于 `vz-memory` owner，不能把权重或 mutable utility 暴露给 consumer。

如果未来做 CMS-only CBP，合理映射是：

1. 把 `W2` 的某个 hidden row 视为 incoming feature weights；
2. 把 `W1` 对应 hidden column 视为 outgoing weights；
3. 重采样 `W2[row, :]`，清零 `W1[:, column]`；
4. 同步清零这些切片的 momentum / optimizer state；
5. utility 至少包含贡献、tanh 饱和度和足够长的时间覆盖，而不是照搬 ReLU 版近期贡献；
6. 只在单一 band 的 SHADOW 实验中启用，先不碰 background durable state；
7. snapshot / semantic memory 不被重置，只有内部 feature basis 可更新。

## 10. 对 Volvence 最直接的证据缺口

在当前代码与 spec 中，没有发现由 CMS owner 正式发布的以下长期读数：

- 每 band 的平均/分位权重绝对值与范数；
- `tanh` preactivation / activation 饱和比例；
- 激活方差和近常数 hidden unit 比例；
- representation effective / stable rank；
- gradient norm、update-to-weight ratio、optimizer momentum 健康度；
- 旧经历保持曲线、新经历吸收曲线与 forward transfer 分解；
- 学习率 / step-scale 的分布与“冻结权重”比例。

这意味着我们现在甚至无法判断 CMS 是否正在发生 Nature 所描述的退化。先观测，再干预，是这篇论文对 Volvence 最硬的要求。

## 11. 最终技术结论

Nature 论文应被采纳为以下工程前提：

> 只要某个 Volvence 子网络计划长期在线更新，就必须证明它在足够长的经历流中仍能学习新模式；只证明短期 loss 下降、旧状态恢复或单次 gate 通过都不够。

但它不应被采纳为以下架构结论：

> 所有模型都应立刻引入 CBP，或 CBP 可以替代 CMS、PE/credit、ModificationGate、语义 owner 和关系连续性机制。

CBP 是候选的**底层可塑性维护机制**；Volvence 的其余契约负责它不处理的稳定性、语义、信用、编排与安全。
