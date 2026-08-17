# 05｜对 Volvence 四能力轴的对账

## 0. 裁决摘要

| 能力轴 | BIGAI/朱松纯路线提供的最好外部证据 | 距 Volvence 完整主张的缺口 | 本包裁决 |
|---|---|---|---|
| Appendable | AOG 可扩结构、任务/概念增量学习、具身 agent memory | 多数是训练集/episode 内状态；缺跨 session、分时间尺度、可恢复的正式写入契约 | **局部机制，不成立为完整轴** |
| Readable | parse graph、AOG、FPICU、因果变量、意图/utility/trait、证明树 | 显式变量不自动等于唯一 owner 发布的 frozen snapshot；latent trait 也未必可辨识 | **最强结合点** |
| Learnable | minimax discrepancy、analysis-by-synthesis、沟通学习、执行器/证明器自举、社会 WM | 学习源包含监督、教师模型、任务 reward 或 verifier；不天然满足 PE→credit，且有 evaluation 泄漏风险 | **可借信号结构，不可照搬 reward** |
| Steerable | data-driven proposal、闭环神经符号、原位价值对齐、意图跨本体适配、残差控制 | 通常没有 norm cap、strict noop、SHADOW/ACTIVE、lineage 与单字段回滚 | **有控制原型，无 runtime 契约** |

结论：这批工作可以强化 Volvence 的结构选择、命名读出、物理—社会双轨和交互式学习设计，但**不能被引用为“四轴在线持续主动学习系统已被外部论文证明”**。

仓库基准：

- [Appendable / Readable / Learnable / Steerable 架构说明](../../docs/appendable-readable-learnable-steerable.md)
- [DATA_CONTRACT](../../docs/DATA_CONTRACT.md)
- [steering runtime 契约](../../docs/specs/steering-runtime.md)
- [steering human anchor 契约](../../docs/specs/steering-human-anchor.md)

## 1. Appendable：AOG 的“可扩展”不等于持续记忆

### 可以借鉴

1. **组合追加而非整库重写。** AOG 的新概念可以复用已有 terminal、关系和子结构，这比把全部经历拼回 prompt 更接近结构化长期记忆。
2. **模型复杂度需要结算。** Region Competition、MDL 与 minimax entropy 都要求新结构带来足够解释增益，避免无限添加节点。
3. **事件与任务有层次。** STC-AOG 区分空间、时间、因果层，为 CMS 不同时间尺度提供外部类比。
4. **持久记忆要服务下一次行动。** 具身 agent/社会世界模型中，状态只有影响后续预测和控制才有意义。

### 不能声称

- AOG 本身没有规定 online-fast / session-medium / background-slow / rare-heavy 的写入频率；
- parse graph 通常是当前样本解释，不是跨 session 可恢复的生活史；
- 论文里的 memory module 不自动满足 immutable snapshot、结构共享和唯一 owner；
- 训练后参数吸收知识不等同于可审计 append。

### 对 Volvence 的约束

任何把“概念节点”“社会 trait”或“意图模板”加入长期状态的实现，都必须先回答：写到哪个现有 owner、哪个时间尺度、如何过期/合并、如何恢复、如何回滚。不得新建一个平行 AOG memory 绕过 `vz-memory` 或语义 owner。

## 2. Readable：最直接、也最容易被过度类比

### 强结合点

朱松纯路线长期坚持让不可见原因成为可命名变量：

- 图像解析：对象、区域、部件、属性、关系；
- FPICU：功能、物理、意图、因果、效用；
- 社会智能：belief、goal、intent、role、affordance；
- SWM-AP：agent trait 与 trait-conditioned response；
- IAIL：跨本体共享的 intention space；
- TongGeometry：可检查的构造与证明树。

这些工作支持一个核心原则：consumer 不应只拿不可解释 embedding 猜测 producer 内部发生了什么，系统应发布任务相关的命名 readout。

### 三条必要收缩

1. **命名变量不等于真值。** `intent / utility / trait` 是模型假设与后验；必须携带不确定性、证据与版本，不能覆盖用户明确陈述。
2. **结构对象不等于 snapshot。** AOG/parse graph 常是可变推断结构；跨模块交换仍需由 owner 冻结发布，consumer 不得遍历内部模型重建语义。
3. **解释性不等于因果忠实。** 符号树若只是事后附着，仍可能与实际控制路径脱钩；需要干预或 counterfactual 证据。

### 对九类语义 owner 的具体启发

| Volvence 语义状态 | BIGAI 邻接概念 | 合法借鉴 | 禁止做法 |
|---|---|---|---|
| `plan_intent` | STC-AOG / intention space | 发布候选意图、证据、置信与行动约束 | 从关键词直接路由 intent |
| `belief_assumption` | causal concept / ToM belief | 显式假设与可证伪预测 | 把 latent 当用户内心真值 |
| `relationship_state` | social affordance / bidirectional alignment | 记录互动后果和互相预期差 | 用单次满意度替代关系连续性 |
| `goal_value` | utility inference | 让效用假设与行为后果对账 | 让 evaluator 分数成为价值真值 |
| `execution_result` | proof/executor/physical outcome | 优先使用可验证后果 | 用语言自评覆盖环境结果 |

## 3. Learnable：可借“差异驱动”，不可借 evaluation 回灌

### 最小最大熵与 Prediction Error

minimax entropy 的模型—数据统计差异，与 Prediction Error 有重要结构相似：当前生成模型解释不了的残差决定下一步扩什么表示。但二者不能直接画等号：

- minimax discrepancy 常由固定特征库和采样估计；
- Volvence PE 是运行时一级原始信号，并有正式 owner、lineage 与下游 credit；
- minimax 方法可能直接更新模型结构/参数，Volvence 在线只允许有界控制层学习。

正确借法是：用“**哪一类可命名残差最能减少未来 PE**”作为 representation-growth 的研究假设；不是把论文里的熵目标替换正式 PE。

### Communicative Learning 的信号边界

教师选择示范、学习者选择查询很有价值，但教师行为不是天然 ground truth：

- 教师可能误解、敷衍、策略性表达或缺乏知识；
- 用户纠正可作为经历和验证锚，不能无条件变成 reward；
- 递归 ToM 输出属于 belief snapshot，不是 credit 源。

合法链条仍应是：互动 → 可观察后果/预测偏差 → PE → credit → gate。沟通协议可以改变“收集什么证据”，不能改变信用来源纪律。

### 可验证自举的适用域

Absolute Zero 与 TongGeometry 说明：有程序执行器/证明器时，自生成任务可以形成可靠闭环。迁移到 Volvence 时必须先分类验证器：

| 验证器 | 可否直接产生 outcome evidence | 原因 |
|---|---|---|
| 程序执行、形式证明、物理传感结果 | 可以，仍需契约化 | 外部、可复现、与陈述分离 |
| 用户明确确认且后果可观察 | 条件允许 | 是验证锚，需处理噪声与权限 |
| LLM judge / 自我评价 / 连续性评分 | 不可以作为学习源 | 可能循环、自洽但不真实、会把 evaluation 回灌 |

## 4. Steerable：控制思想丰富，部署纪律不足

### 可迁移结构

1. **proposal 与 acceptance 分层**：DDMCMC 表明快速候选生成可以与严格结算分开。
2. **感知—结构—推理闭环**：神经符号系统允许结构约束反向改变感知，而不是只做事后解释。
3. **原位双向对齐**：控制策略应根据行动后果与用户反馈更新，并向用户暴露理由/不确定性。
4. **意图与本体分离**：IAIL 说明高层目标可以作为跨执行器的条件变量，底层适配各自约束。
5. **先验与残差分层**：OmniXtreme 一类工作支持慢运动先验 + 快执行残差的工程模式。

### Volvence 额外必须具备

- 冻结基底；
- 有界残差和 norm cap；
- 无 free bias；
- 条件不满足时 strict noop；
- `WiringLevel.SHADOW` 默认并能单字段回滚；
- 由正式 `steering_condition_belief` 选择何时干预；
- 干预必须改变下一拍可追加状态，使 PE 可结算。

没有这些条件，只能说“存在控制/适配机制”，不能说 Volvence 意义上的 Steerable 已成立。

## 5. 物理—社会统一：应交互，不应塌缩

BIGAI 近期 position paper 主张世界模型统一物理和社会动态；这与 Volvence 的 World / Self 双轨既相合又存在张力。

推荐解释：

- **语义隔离**：物理状态和社会/自我状态各有 owner、证据、置信和更新规则；
- **显式耦合**：双方通过冻结 snapshot 交换，例如物理可达性约束计划，关系承诺约束可接受计划；
- **联合预测**：consumer 可以预测二者交互后果，但不能成为第二 owner；
- **不合并本体**：不把力、偏好、信任压进一个无类型 latent 后宣称“统一”。

“统一”应指可计算的双向依赖，而不是取消边界。

## 6. 七条可进入未来设计的候选

1. 用 minimax-style discrepancy 排序“下一个值得命名的 residual readout”，但学习源仍是 PE。
2. 将 AOG 的 `And / Or / parse instance` 作为 typed semantic snapshot 的设计参考，不复制第二 owner。
3. 为 `belief_assumption` 增加“支持证据 / 反证预测 / 不确定性”，避免 ToM 读出实体化。
4. 在 SHADOW 中比较 physical-only、social-only 和 dual-track prediction，检验双轨耦合是否真有增益。
5. 把沟通学习用于“何时询问、问什么”的信息采集策略，不用于重定义 reward。
6. 在有执行器/证明器的子域建立 self-generated curriculum，禁止跨到无可靠结算器的开放价值域。
7. 用 intention-level condition 连接高层计划和多个 executor，同时保留每个 executor 的 norm/safety guard。

## 7. 八条禁用类比

1. 不说“AOG 就是 CMS”。
2. 不说“minimax entropy 就是 Prediction Error”。
3. 不说“TONG Test 分数可以训练 agent”。
4. 不说“social world model 的 trait 就是用户真实人格”。
5. 不说“in-situ value alignment 已解决人类价值对齐”。
6. 不说“Absolute Zero 证明无数据开放式学习成立”。
7. 不说“物理—社会统一要求删除 World / Self 边界”。
8. 不说“BIGAI 的 SHADOW-like 原型等于 production ACTIVE”。

## 8. 四轴自检答案

若未来实现本研究建议，动手前必须明确：

1. **写入**：经历落在哪个 CMS 时间尺度和唯一 owner，何时恢复/过期？
2. **发布**：物理、意图、效用、关系等状态是否由 owner 以 frozen snapshot 发布？
3. **学习**：所有更新能否追溯到 PE/credit，是否完全隔绝 judge/evaluation 回灌？
4. **干预**：是否有界、条件化、strict noop、lineage、SHADOW 和可回滚？
5. **闭环**：干预是否改变下一拍可追加状态，形成可结算的 prediction error？

本研究只提出证据与候选，不替任何实现回答这些问题。
