# Oak / OaK 技术谱系：从可塑性到时间抽象

## 1. 不要把 Oak 看成单一算法

Oak 当前公开路线是多个尚未完全合拢的研究分支：

```text
普通连续经验 + 有限资源
          │
          ├── 表征要为未来在线更新而优化（OML）
          ├── 每个参数要有适合其变化率的步长（IDBD / step-size optimization）
          ├── 网络要持续生成、检验、替换内部特征（CBP / generate-and-test）
          ├── 世界总比有限智能体大（Big World / tracking）
          └── feature → subtask → option → model → planning
                               │
                       utility feedback / OaK
```

这些分支共享一个目标：在不保存和反复训练全部历史的条件下，让有限智能体长期吸收经历并改善行动。只有其中部分机制有直接论文证据；完整 OaK 尚是研究架构。

## 2. 时间线与证据状态

| 时间 | 工作 | 贡献 | 证据级别 |
|---|---|---|---|
| 2019 | *The Bitter Lesson* | 通用学习/搜索与可扩展计算优先于长期手工编码知识 | C |
| 2019 | Javed & White, OML | 元学习一个适合未来在线更新、较少干扰的表示 | B |
| 2021 | Continual Backprop 预印本 | 持续注入随机新单元，用 utility 淘汰低贡献单元 | B |
| 2022–2023 | Alberta Plan | 12 步研究路线；STOMP 与 Oak architecture 的明确前身 | C |
| 2024 | Nature 可塑性论文 | 系统展示 loss of plasticity，并扩展 CBP 证据 | A |
| 2024 | Step-size Optimization | 把参数级步长视为 lifetime objective 的优化问题 | B，但实验简单 |
| 2024 | Big World Hypothesis | 有限 agent、开放世界、tracking、资源约束 | C |
| 2025 | *Era of Experience* | 长期流、grounded action/observation/reward、经验中的规划 | C |
| 2025 | RLC OaK talk | 公开 OaK 讲座入口 | D |
| 2026 | Oak Lab mission / blogs / interview | 汇合经验学习、时间抽象、规划和能效愿景 | D/E |

## 3. OML：不是学当前任务，而是学“以后怎么学”

[Meta-Learning Representations for Continual Learning](papers/meta-learning-representations-neurips-2019.pdf) 的问题是：普通表示只为当前训练损失优化，未必适合在未来只看少量样本时快速更新，也未必能减少新旧任务干扰。

OML 把参数分为两类：

- representation 参数：在 meta-training 中优化；
- online prediction head / fast parameters：按序接收样本并在线更新。

meta-objective 不是只看当前样本，而是让经过一段 online updates 后的系统，在后续数据上表现好。论文实验显示，所学表示更稀疏、dead units 更少，与 replay、EWC 等方法也可组合。

### 对 Oak 路线的贡献

它支持访谈中的“基础模型要从一开始学会如何继续学”：如果表示在形成时就为未来在线更新优化，后装一个 optimizer 可能不够。

### 限制

- representation 主要在离线 meta-training 中获得，在线阶段没有持续重构整个表示。
- task distribution 与 train/test split 仍由实验者给出。
- 作者把“在线学习表示、不依赖单独 meta-training”留作未来工作。
- 论文建议可用 recent buffer 和周期性“sleep”更新表示，这与 2026 访谈的强 no-replay 叙事并不完全相同。

因此，OML 是“为持续学习训练表示”的证据，不是 OaK 已实现的证据。

## 4. CBP：在固定容量里维持新特征来源

[Continual Backprop](papers/continual-backprop-2108.06325.pdf) 把普通 backprop 拆成两种机制：

1. 梯度下降；
2. 训练开始时的小随机初始化。

它认为第二种机制只在时间 `t=0` 提供多样性，随着学习持续会被耗尽。CBP 将其时间对称化：任何时刻都小剂量地产生随机 feature，由梯度和 utility 检验，再替换低效用的成熟 feature。

这为 Oak 的 generate-and-test 提供底层原型，但它仍只在神经单元空间工作：

- generator：随机初始化新的 hidden unit；
- tester：activation × outgoing influence 等 utility；
- selection：低 utility mature unit 被替换；
- learner：标准 backprop 检验新 feature。

它没有生成具备任务语义的 subtask、option 或 model。

## 5. Step-size optimization：决定哪些连接该稳定、哪些该快学

[Step-size Optimization for Continual Learning](papers/step-size-optimization-2401.17401.pdf) 区分：

- **normalization**：Adam / RMSProp 依据梯度统计缩放更新；
- **optimization**：IDBD 类方法对每个参数的 step-size 做 meta-gradient，使长期累计目标更小。

在长期变化的线性问题中，参数对应的真实目标变化快慢不同：

- 稳定维度应降低步长，减少噪声和遗忘；
- 快变维度应维持较大步长，避免跟不上环境；
- 当变化率改变，步长本身也要重新适应。

这比“所有旧权重都冻结、新权重都快速”更一般，因为权重的重要性和变化率会随经历改变。

### 公开证据的边界

论文直接实验是 20 维 weight-flipping regression 和一维 noisy rate-tracking 等简单问题，最长约百万步。IDBD 在这些问题上可接近 oracle，并与 SGD / Adam 保持同量级渐进计算与内存。但：

- deep nonlinear credit assignment 没有被解决；
- meta-step 对目标尺度非常敏感，合适值可跨多个数量级；
- per-weight state 在超大模型上的内存、分片和稳定性未评估；
- 与 CBP 联合的长期 stability / transfer 没有完整公开证据。

### NetworkIDBD 的位置

Oak 2026 博客用 NoisyMNIST、4,096 输入和约一万 ReLU 单元展示 NetworkIDBD，主张能在 batch-size-one 中持续保持学习能力。它比线性实验更接近深网，但截至研究日期没有配套论文、完整算法、代码、seed、消融和跨任务验证。应视为 D 级初步信号，不应替代正式 benchmark。

## 6. Alberta Plan：OaK 的真正架构前身

[The Alberta Plan for AI Research](papers/alberta-plan-2208.11173.pdf) 给出四条研究纪律：

1. **Ordinary experience**：学习只从 observation、action、reward 的普通经验获得，不依赖特殊训练集或环境内部变量。
2. **Temporal uniformity**：没有特殊训练时刻；每一步都可学习、规划、构造表征和 subtask。
3. **Computational constraints**：计算永远有限，reaction time 与 decision quality 要权衡。
4. **Multi-agent world**：环境包含会响应智能体行动的其他智能体，合作、竞争和 intelligence amplification 是基本情况。

### 6.1 Base agent

基础智能体有四部分：

- perception：把历史经验压缩成 state；
- reactive policies：从 state 选 action；
- value functions：评价 state / policy；
- transition model：预测 action/option 之后的 state 与累计 reward。

planning 使用 transition model 想象结果，再用 value 改善 policy。它在背景异步进行，但 learning 也会在每个 foreground step 使用最新事件和短期 credit trace。

### 6.2 12 步路线

Alberta Plan 从 continual supervised representation 开始，依次走向 GVF prediction、actor-critic control、average-reward、planning、model-based prototype、search control、STOMP、Oak 和 intelligence amplification。

关键不是 12 个名称，而是依赖关系：**如果低层表示本身不持续可塑，后面的 option model 和 planning 也会在长期运行中僵化。**

### 6.3 STOMP：从状态 feature 生长出时间抽象

第 10 步的 STOMP 是：

```text
高价值 state feature
    ↓ 定义 reward-respecting SubTask
    ↓ 学出实现 subtask 的 Option（policy + termination）
    ↓ 学出 option 的 Model
    ↓ 把 model 放入 Planning
```

这比从一开始预设完整技能库更符合“结构随经历生长”。feature 不只是描述状态，还成为定义 subtask 和 option 的种子。

### 6.4 Oak：让整条抽象链接受 utility feedback

第 11 步的 Oak 在 STOMP 上增加 feedback：

- feature 是否帮助 prediction / learning；
- subtask 是否值得保留；
- option 是否可实现并被用到；
- option model 是否在 planning 中真正改善决策；
- 低 utility 元素及其下游链条是否应退休并由新候选替代。

例如，一个 option model 从未对规划有用，则对应 option、subtask 也应最终退出，资源让给尚未被发展为 subtask 的 feature。这里的 generate-and-test 已从神经单元扩展到认知结构。

### 6.5 Option keyboard

Oak 还提出 option keyboard：每个“键”对应一个 subtask / option，实值向量可以像和弦一样组合多个 option。它提供组合式时间抽象，但 Alberta Plan 只描述两种可能设计，并未给出完整学习与稳定性证据。

## 7. Big World：为什么固定容量、tracking 和 no-final-model 重要

[The Big World Hypothesis](papers/big-world-hypothesis-2024.pdf) 说：对许多决策问题，世界比 agent 大多个数量级；agent 不可能感知全部状态，也不能存下每个状态的最优值和动作，只能用有限理解做近似决策。

### 推论

- 持续 tracking 比一次找到永久最优解更现实；
- 时间相干性让 agent 可把资源集中在近期可能重现的世界片段；
- feature / memory 必须有引入、保留和淘汰机制；
- 算法每步开销会与可容纳的模型容量竞争；
- benchmark 应固定 agent 资源或把环境做得足够大，不能只在 agent 明显过参数化时比较。

### 必须保留的作者 caveat

论文明确说，有很多问题满足 Big World，也有很多不满足；它更像“我们应该优先研究哪类问题”的选择，而非对所有决策问题的事实判断。论文列举的 evidence 是间接和 circumstantial，作者也承认有替代解释。

### 与“合成数据错误”的关系

Big World 支持“任何固定模拟器都可能漏掉世界”，但它并没有推出“模拟器不能用于学习或规划”。Alberta Plan 自身需要 transition model 和 imagined outcomes。更一致的结论是：模型必须被现实经验不断校正，而且模型错误本身应进入学习回路。

## 8. Era of Experience：从人类数据转向行动后果

[*Welcome to the Era of Experience*](papers/era-of-experience-2025.pdf) 提出四个特征：

1. **Streams**：跨数月或数年、非短 episode 的长期交互流；
2. **Actions and observations grounded in the environment**：行动对世界产生后果，观察来自世界变化；
3. **Rewards grounded in experience**：评价来自行动后果，而不是人在行动前猜测哪个回答更好；
4. **Planning and reasoning in experience**：思考应与世界模型和预测误差闭环，而不只是模仿人类推理痕迹。

这篇观点文中，世界模型可产生模拟经验，科学和数学工具也可成为环境；关键是模型预测最终接受新观测校正。因此它对合成/模拟的立场比访谈口号更细：**反对静态生成器成为唯一老师，不反对可纠错的内部模型成为规划工具。**

## 9. OaK 当前公开形态

[Oak mission](https://www.oaklab.ai/mission) 把目标描述为：从经验中发现 temporal abstractions，这些抽象能够自我验证并用于规划；智能体以 batch size 1 学习，不存储和重放全部历史，最终追求类脑能效。

截至 2026-08-22 的公开审计：

| 项目 | 已公开 | 尚未公开或未发现 |
|---|---|---|
| OaK 架构 | mission、研究页、2025 RLC talk 入口、Alberta Plan 前身 | 正式算法论文、完整伪代码、代码库、benchmark、消融 |
| NetworkIDBD | NoisyMNIST 博客图与叙述 | peer-reviewed / preprint 论文、实现、完整超参数与多 seed |
| CBP | 预印本、Nature、官方代码 | LLM 规模证据、完整 forgetting 解法 |
| Step-size | 线性长期流论文 | 深层大模型通用证据、与 CBP/OaK 的联合结果 |
| Event-driven learning | research 索引中的方向 | 页面仍标注 coming soon |
| 20W 心智 | 使命与访谈目标 | 原型、能耗分解、硬件/算法方案与测量 |

因此“OaK 架构”目前最准确的定义是：**由已发表的底层机制和一套尚待实现/验证的高层 utility-feedback 架构组成的研究计划。**

## 10. 路线内部的几个张力

### 10.1 Temporal uniformity 与多时间尺度

Alberta Plan 强调每时刻规则相同，但同时允许背景 planning、短期 credit trace 和不同反应速度；它也承认实际中可能偏离绝对 uniformity。Oak 的“每步都学”不意味着所有参数每步同速更新。Volvence 的 online-fast / session-medium / background-slow / rare-heavy 可以被理解为同一规则下的多频率执行，而非特殊离线真相通道。

### 10.2 No replay 与 OML / 现实稳定性

Oak mission 把 no storing/replay 作为目标，但 OML 曾建议 recent buffer + sleep，Nature 也说 replay 可隐藏或缓解部分问题。公开证据没有证明 replay 本身错误。更可靠的要求是：

- 不允许计算与存储随经历总量无界增长；
- replay 必须是 bounded、owner-governed、带选择和过期规则；
- 最终要与 batch-size-one / no-replay arm 对比，而不是按理念预先删除。

### 10.3 Scalar reward 与关系智能

Alberta Plan 以一个 scalar reward 定义目标，便于建立完整 RL 数学框架。关系型智能体却面对同意、安全、承诺、长期信任和不完全可观测的人类状态；把它们压成一个在线 reward 会引入 reward hacking 和伦理风险。Volvence 以 typed outcome、PE、needs、credit、boundary/consent 和 ModificationGate 分层，是必要扩展，而不是对 Sutton 路线的偏离。

### 10.4 自我验证与价值验证

预测准确的抽象不一定值得行动；能提升短期规划的抽象也可能伤害关系或边界。OaK 的 self-verifiable / planning-useful 需要再加：

- 事实预测由环境 outcome 结算；
- 主体与关系更新由各自 owner 解释；
- 边界与 consent 不是优化目标，而是不可越过的约束；
- 晋升接受 lineage、回滚和人类验证锚。

## 11. 对 Volvence 最可用的 OaK 设计原语

即使没有 OaK 实现，以下原语已经足够明确，可转成实验假设：

1. **Candidate lifecycle**：feature / abstraction 必须有生成、maturity、utility、retirement 和 lineage。
2. **双重 utility**：既看 prediction improvement，也看 counterfactual planning usefulness。
3. **级联退出**：上游 feature 退休时，下游 subtask / option / model 不能成为悬空的第二 owner。
4. **固定资源**：候选池、每步更新和规划预算必须有上限。
5. **持续校正**：内部模型的 rollout 只能提出行动，不可替代真实 outcome 的结算。
6. **异步规划**：foreground 保持实时反应，background-slow 使用剩余预算改善模型和策略。

这些原语与 Volvence 的 snapshot、ModificationGate、WiringLevel 和多时间尺度可以自然结合；完整映射见 [04_VOLVENCE_COMPARISON.md](04_VOLVENCE_COMPARISON.md)。
