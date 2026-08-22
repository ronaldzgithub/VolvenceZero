# Sutton / Oak 与 Volvence 的细致对比

## 1. 比较原则

本对比同时看三层，任何一层都不能替代另一层：

1. **设计主张**：系统希望具备什么能力；
2. **实现与接线**：owner、snapshot、更新器和 runtime 路径是否存在；
3. **因果证据**：是否在预注册、matched、足够长期的环境中证明带来预期收益。

Oak 的公开材料在第 1 层很强，底层部分机制在第 3 层有论文证据，完整 OaK 的第 2/3 层尚未公开。Volvence 的 owner、契约和接线比公开 OaK 描述具体，但多项关键能力仍是 SHADOW / DISABLED 或 evidence 未支持。两边都不能用愿景填补证据。

## 2. 高层总表

| 维度 | Sutton / Oak | Volvence 当前方案 | 判断 |
|---|---|---|---|
| 智能定义 | 从连续 observation-action-consequence 中终身改变 | 从经历、PE、credit、memory、controller 形成多时间尺度适应 | 高度同向 |
| 基础模型 | 访谈倾向从头训练可持续学习的 foundation model | 冻结 LLM substrate，在线只改 memory/controller；基底只 rare-heavy | 核心路线不同，但可互补 |
| 经验载体 | batch size 1 普通经验；目标不保存/重放全部历史 | CMS / semantic snapshots / bounded replay / hydration | 同向但 replay 立场不同 |
| 可塑性 | 一等问题；CBP、step-size optimization | 小型 CMS MLP 可学习，但无正式 plasticity readout | 明显缺口 |
| 稳定/遗忘 | 明确认识到问题，CBP 当前未解决 | CMS 分层、replay、consolidation、semantic owner 试图保持 | 设计更完整，因果收益未证明 |
| 学习信号 | scalar reward、prediction、utility | PE → credit → ModificationGate；needs/evaluation 仅下游 readout | Volvence 更细粒度、更适合关系域 |
| 世界模型 | action/option transition model，持续 prediction-error 校正 | World / Self 双轨、temporal state、environment outcome | 方向相符，完整 model/planning 闭环未证 |
| 时间抽象 | feature → subtask → option → model → planning | `z_t / beta_t`、segment、protocol、abstract action 等 | 概念相近；当前 ETA operationalization 有 `kill-eta` 边界 |
| 生成与检验 | hidden units 和认知元素持续生成、淘汰 | rare-heavy artifact + ModificationGate + SHADOW / rollback | 治理同向；缺少通用 candidate lifecycle |
| 可读性 | utility feedback 概念清楚，公开 snapshot 规范不详 | 唯一 owner + frozen snapshot + lineage | Volvence 的公开工程约束更强 |
| 可控性 | 学得 policy / option 直接行动 | frozen substrate 上的 bounded residual steering | Volvence 更保守；目前 production 未 ACTIVE |
| 合成数据 | 批评静态、人类策划合成数据替代现实经验 | 三层 truth 分离，允许 offline/evidence，禁止回灌 runtime truth | 当前 spec 与其精确版批评兼容 |
| 资源 | 固定 agent、per-step computation、no unbounded history、20W 愿景 | 有时间尺度和有界缓冲，但缺统一每拍算力/能耗账本 | 部分符合 |
| 安全/治理 | 公开材料主要谈能力 | consent/boundary、World/Self、WiringLevel、rollback、gate | Volvence 明显更具体 |
| 当前证据 | Nature/CBP 强于局部机制；完整 OaK 未公开 | 多个 proxy / wiring evidence；系统级 thesis 未过 | 各强在不同层，不能排“谁领先” |

## 3. 按四能力轴对比

### 3.1 Appendable：经历是否真的成为下一拍可恢复状态

### Oak

Oak 的强主张是每个时刻都从 batch-size-one experience 更新，不依赖把全部历史留在 dataset 中。Big World 的 tracking 要求有限 agent 持续替换当前不重要的信息。它关心的是“经历改变内部参数”，而非公开定义一种可审计的 memory schema。

### Volvence

[四能力轴文档](../../docs/appendable-readable-learnable-steerable.md) 将经历分布在：

- `online-fast` / `session-medium` / `background-slow` CMS；
- State-KV / semantic snapshots；
- cross-session hydration / checkpoint；
- rare-heavy artifact。

Memory owner 当前基础路径 ACTIVE，CMS Torch backend 默认 DISABLED。现有 [CMS core](../../packages/vz-memory/src/volvence_zero/memory/cms.py) 支持有界 ATLAS replay、不同 cadence、PE-aware gate 和 nested backflow。

### 判断

Volvence 对“写到哪里、由谁恢复”回答得比 Oak 公共材料具体，满足真正 appendable 的必要结构。但 `Gate 5` 已明确说明：当前只能主张多频 CMS **可运行、可审计、可回滚**，不能主张它相对 single-timescale 有显著吸收—保持优势。

此外，Appendable 不是“保存越多越好”。若要吸收 Big World：

- 每个 band 的容量、淘汰和重用应显式；
- replay buffer 不能随生命期增长；
- 稀有但高价值的关系/边界事实不能由纯 recency utility 淘汰；
- “写入成功”必须改变未来可追加状态，并可被后续 PE 结算。

### 3.2 Readable：内部变化能否被 owner 命名和审计

### Oak

OaK 希望评估每个 feature、subtask、option 和 model 的 utility，并让 planning usefulness 反向影响抽象选择。这隐含一个可读性要求：系统必须知道候选存在、成熟度、被谁消费以及是否有用。但公开材料尚未给出 frozen exchange schema、跨模块 owner 或回滚 lineage。

### Volvence

Volvence 的规则是 owner 解释并发布 immutable snapshot，consumer 不得遍历内部结构重建 producer 状态。PE、credit、steering belief、intervention、temporal state、九类 semantic state 都有 owner 边界。

这为 OaK 风格 feedback 提供了更安全的基础：

- utility 由拥有候选的模块计算；
- downstream 只回传已结算的 public evidence，不抓取 mutable weights；
- candidate、active、retired 和 rollback 状态可冻结；
- SHADOW 候选不进入用户可见 mapping。

### 当前缺口

Readable 的盲点恰好在 Nature 指标：`CMSBandMLP` 的 weights、activation saturation、effective rank、update health 没有正式 owner readout。不能让外部评测脚本读取 `_w1/_w2` 来弥补，否则会建立第二 owner。正确修复是由 `vz-memory` 发布 frozen、聚合、无 mutable reference 的 plasticity telemetry。

### 3.3 Learnable：什么信号有权改变长期策略

### Oak

Alberta Plan 以 reward 为最终目标，prediction error 和 utility 支持表征、model、option 与 planning 改进。优势是目标链简洁；风险是开放关系场景中的单标量 reward 很难忠实表达多方价值、同意、边界和长期主体连续性。

### Volvence

Volvence 的不可让步链是：

```text
pre-action prediction
      ↓
typed environment outcome
      ↓
Prediction Error
      ↓
credit assignment
      ↓
bounded controller / ModificationGate
```

evaluation、judge、七日 continuity 分数只能验证，不得成为学习源；needs / homeostasis 是 PE 下游 readout。这个分层比“一个 reward”更适合防止关系域 reward hacking。

### 当前 learned update rule 的准确定位

[learned_update.py](../../packages/vz-contracts/src/volvence_zero/learned_update.py) 会为 target 产生 `write_gate`、`step_scale`、`momentum_gate`、`slow_mix`、`reset_mix` 等有界决策，并依据 before/after improvement、stability 等内部量更新自己的状态。

它与 Sutton 的 parameter-wise step-size optimization 只在精神上相似：

- 它的 `step_scale` 是 target / band 级控制，不是每个 `W1/W2` 参数一条 IDBD trace；
- 它的内部 `reward` 是更新器的启发式综合分，不是环境 scalar reward，也不是 IDBD 对 lifetime loss 的标准 meta-gradient；
- 它有 guard 与 bounded mix，利于工程安全，但未证明可解决长期 plasticity。

因此应把它作为对照 arm，而不是重命名为 IDBD。未来任何 parameter/group step-size 的 meta-objective 仍必须追溯到 PE/credit，不能用 evaluation 替代。

### 3.4 Steerable：学到的状态是否能有界改变后果

### Oak

Oak 的 policy、option 和 planning 最终直接选择环境 action；持续学习天然通过行动改变下一批经验。

### Volvence

Volvence 冻结 substrate，在残差空间进行条件化、有界、可择时干预：

- sensor 读 layer-bound residual；
- temporal gate 决定何时干预；
- substrate executor 施加 norm-capped、no-free-bias residual；
- strict noop 保证关闭时行为精确不变；
- `WiringLevel` 允许 DISABLED / SHADOW / ACTIVE 和单字段回滚。

### 当前证据边界

[steering runtime spec](../../docs/specs/steering-runtime.md) 的三件套默认都是 SHADOW；production 表达层默认不捕获 residual，vLLM / synthetic backend 没有经验证的 residual hook。代理层“读得到、扳得动、学会何时扳”不等于生产 ACTIVE。

因此 Volvence 当前有一个比 Oak 公共描述更严谨的 actuator safety contract，但还没有同等强的 live consequence evidence。

## 4. 技术组件映射

以下只是设计类比，不是可以直接互换的契约：

| Sutton / Oak 元素 | Volvence 最接近 owner / 机制 | 匹配程度 | 禁止误读 |
|---|---|---|---|
| ordinary experience | `vz-application` environment event/outcome + runtime 编排 | 中高 | 用户文本本身不是无歧义 reward |
| prediction error | `vz-cognition` PE owner | 高 | evaluation 不得反向成为 PE |
| parameter step-size | learned updater 的 band step scale；未来 CMS-local per-param state | 低到中 | 当前不是 IDBD |
| hidden feature generate/test | `vz-memory` CMS band MLP 内部 feature | 低 | 不得重置 semantic snapshots |
| candidate abstraction | `vz-temporal` temporal candidate / rare-heavy artifact | 中，主要是架构意图 | `z_t` 存在不等于自主抽象已证 |
| subtask | `plan_intent` / `goal_value` 等 owner 可能提供 typed input | 低 | 这些语义 owner 不是 Oak subtask owner，不应合并 |
| option | temporal abstract action / protocol 可能提供执行候选 | 低到中 | protocol label 不是学得的 option |
| option model | world temporal / transition representation | 低到中 | 目前不能声称通用 model-based planning |
| planning utility | counterfactual outcome、credit、gate evidence | 低到中 | 当前 gate 不是 OaK utility algorithm |
| feature retirement | rare-heavy candidate rejection / rollback | 中 | 当前缺通用 maturity + retirement lifecycle |
| continual planning | background-slow reflection / temporal computation | 中 | LLM reflection 不能成为 controller 或 reward owner |
| multiple minds | per-user state、World/Self、relationship state | 高于单共享模型方案 | 必须处理 consent、删除和身份隔离 |

## 5. 多时间尺度：相似但并非同一设计

### Oak 的时间观

- temporal uniformity：没有一个特殊“训练期”拥有更真实的信号；
- 每步可学习和规划；
- foreground 负责及时感知/行动，planning 可异步；
- option 跨多个基础 time steps。

### Volvence 的时间观

- `online-fast`：即时预测、PE、短记忆和小控制更新；
- `session-medium`：session 内累积与策略状态；
- `background-slow`：反思、整合、候选评估；
- `rare-heavy`：高风险 artifact 的离线/极慢修改。

### 综合

两者不矛盾。Temporal uniformity 可以理解为“任意时间都遵循同一学习法则”，而不是“所有状态每拍同速更新”。Volvence 的优势是频率和权限已分层；风险是 background-slow / rare-heavy 重新滑回“离线 judge 生成真相”。必须保证慢层仍只消费从真实 PE/credit 累积的 lineage，LLM reflection 只提出候选。

## 6. 冻结基础模型：是约束、折中，也是路线分歧

### Volvence 为什么冻结

- 在线端到端更新的安全、成本、回滚和可塑性问题尚无解；
- 产品状态由独立 owner 管理，避免语言权重成为不可读的第二事实库；
- 控制学习发生在小型 `z_t` / memory / gate 空间，能做 norm cap 和 transactional rollback；
- foundation artifact 只允许 rare-heavy gate。

### Sutton / Javed 为什么希望从头训练

- 现成表示形成时没有为未来在线更新优化；
- 后装 optimizer 可能无法修复已经过度承诺或低秩的表示；
- 想让每层都具备可持续塑性，而不只让外接 memory 学。

### 本研究判断

短中期应保留 Volvence 冻结基底：Nature 没有 LLM 证据，Oak 也没有公开新 foundation model。可采用分层实验路径：

1. 先证明小型 owner-local learner 能从真实 PE/credit 带来长期净收益；
2. 再研究 rare-heavy adapter / representation artifact；
3. 只有有清楚的 old/new/transfer、安全和 rollback 证据，才讨论更深层 substrate 更新。

冻结基底不意味着自满。若 controller/memory 只能检索旧文本、从不改变可泛化策略，按 Sutton 的批评它仍是不完整学习。

## 7. 可塑性：Volvence 当前最大的新增盲点

### 现有机制

[CMSBandMLP](../../packages/vz-memory/src/volvence_zero/memory/cms_band_mlp.py) 为每个 band 提供两层 residual MLP：`x + W1 @ tanh(W2 @ x)`，带 momentum。W1 零初始化、W2 小随机；CMS 使用 PE features、bounded replay、不同 cadence 和 learned gates。

### 缺失的判断能力

现有代码没有回答：

- tanh 是否逐渐饱和？
- hidden activation 是否变成近常数？
- W1/W2 是否持续长大？
- 表征 effective / stable rank 是否下降？
- 新关系模式的学习速度是否随 turn 数下降？
- 哪些 band 在保存稳定知识，哪些已经无法更新？
- learned step scale 是否塌到 0 或长期饱和？

已有 anti-forgetting evidence test 和 promotion readout 主要是方向性 proxy、one-step MSE、64-observation window 与 510-turn gate，不是 Nature 级长期可塑性测量。

### 风险

Volvence 可能出现两种相反失败：

1. **过稳**：background/slow 状态保存很好，但 online/session MLP 对新用户变化学不动；
2. **过塑**：新反馈很快改变状态，却破坏承诺、边界或关系连续性。

只看 aggregate continuity 或最终 output 无法区分。必须建立 old/new/transfer/coherence 四面板。

## 8. 时间抽象：愿景接近，双方都未完成

### 相似点

- Oak 的 option / model 跨基础时间步；Volvence 的 `beta_t` / segment 希望控制时间边界。
- Oak 用 utility 选择抽象；Volvence 用 PE/credit 和 gate 控制候选晋升。
- 两者都拒绝 token 空间直接承担全部长期控制。

### Volvence 的诚实边界

[temporal abstraction spec](../../docs/specs/temporal-abstraction.md) 记录 2026-08-04 Stage 3 正式 `kill-eta`：当前 16 维折叠入口 + additive steering / free-bias operationalization 没有出现预期 rate-distortion gap。该结论不否定所有 ETA，但足以阻止把当前方案当成已证的自主时间抽象。

同时，残差中线性读取 active subgoal 的代理证据成立，并不等于：

- 系统自行发现了 subgoal；
- `beta_t` 学会了正确 option termination；
- 抽象改善了真实 planning；
- 该收益能跨环境、跨 session 保留。

### Oak 的同样边界

OaK 公开路线描述了 feature→subtask→option→model→planning，但没有论文/代码证明这一完整链。因此合理态度不是“改投 Oak”，而是把两套设计共同压成可测问题：候选抽象是否提升 held-out prediction，并在固定 planning budget 下改善真实行动结果？

## 9. 合成数据：Volvence 应保留什么、禁止什么

[synthetic experience corpus](../../docs/specs/synthetic-experience-corpus.md) 的三层隔离是正确基础：

- `generator_truth`：构造器内部变量；
- `rendered_text`：模型看到的表达；
- `runtime_observation`：运行时正式可见事件。

结合 Sutton 的批评，使用政策应是：

| 用途 | 是否保留 | 学习/晋升权限 |
|---|---|---|
| 单元/契约/回滚测试 | 保留 | 不产生产品能力主张 |
| 可塑性机制基准（漂移、噪声、复现 Nature） | 保留 | 只证明机制，标记 synthetic |
| SHADOW 预演、反事实和边界覆盖 | 保留 | 只能提出/淘汰候选，不能结算 live credit |
| curriculum / rare cases | 条件保留 | 需与真实 held-out outcome 分开报告 |
| learned world-model rollout | 保留 | 规划用；最终由真实 outcome 校正 |
| 自生成答案再当真相训练 | 禁止作为最终证据 | 不得进入 PE/credit truth lane |
| 用 synthetic continuity score 替代真实关系后果 | 禁止 | 不得晋升 ACTIVE |

最核心的防火墙是：**合成材料可以改变“我们下一步测试什么”，不能单独决定“系统在世界里学对了什么”。**

## 10. Replay：不要把工程选择变成信仰

Oak 的 no-replay 目标来自 Big World：如果保存全部历史，资源会随寿命增长，也可能用旧数据淹没当前变化。Volvence 的 ATLAS replay 是小窗口、按 band 有硬上限，目的在稳定吸收，而不是保存全部历史。

当前没有足够证据要求删除 bounded replay。更合适的对比是：

- K=1 / batch-size-one；
- bounded recent replay；
- utility-selected replay；
- stratified rare/high-credit replay；
- 相同 per-step compute、memory 和参数预算。

测量 absorption、retention、forward transfer、adaptation latency 和资源后，再决定每个 band 是否需要 replay。online-fast 可能偏向 K=1，background-slow 可能需要少量高价值回访；不能由一个全局口号决定。

## 11. 真实经验在关系域里是什么

Oak 的机器人/游戏例子有清晰 reward；对话关系没有天然标量。对 Volvence，“真实经验”至少需要：

- 动作前发布明确、可结算的 prediction；
- 用户或环境随后提供与该 prediction lineage 对齐的 observation/outcome；
- 区分用户表达、系统推断和真正完成的外部事实；
- 允许延迟结算和未知结果，不能把无反馈当负反馈；
- consent / boundary 只由正式 owner 解释，不能用 engagement 优化替代；
- 人类标注可作为验证锚，但不是在线 learning reward。

可结算例子包括：用户明确纠正偏好、承诺是否按期完成、工具调用结果、用户选择是否与模型预期一致、后续明确报告计划成败。不可直接作为 reward 的包括：对话更长、情绪词更积极、模型自己的评价分更高。

## 12. 当前证据账本

| 主张 | 当前 Volvence 证据 | 结论口径 |
|---|---|---|
| 状态可多频追加和恢复 | owner、CMS、hydration、checkpoint 路径存在 | 可说可运行/审计/回滚 |
| 多频 CMS 优于单频 | 3 seeds × 每 seed 510 settled traces；增益 `+2.508e-7 / +1.173e-6`，阈值 0.02 | `not-supported` |
| CMS Torch 可替代 pure | parity / SHADOW closure 路径存在；默认 DISABLED | 不可说 ACTIVE 收益 |
| 残差可读 active subgoal | 代理证据强，且 instrument 问题有审计 | 可说代理 readout |
| 残差 steering 可条件写入 | proxy C2 / S3-E，SHADOW 三件套 | 不可说 production ACTIVE |
| ETA rate-distortion operationalization | Stage 3 `kill-eta` | 当前方案失败，不外推理论 |
| PE-first 学习链 | 契约与多条 runtime lineage 已接 | 可说机制链；产品长期净效应待 formal |
| 长期可塑性 | 无 Nature 风格指标/长流 | 未知，不能声称成立 |
| 自主 OaK 式抽象规划 | 无完整 candidate→option→model→planning gate | 未成立 |

## 13. Volvence 相对 Oak 公共方案的独特优势

这里比较的是公开文档，不推断 Oak 未公开实现：

1. **SSOT 与唯一 owner**：避免 memory、world model、relationship 和 planner 各自重建事实。
2. **不可变快照**：长期学习的每次公共变化可命名、审计和恢复。
3. **PE 与 evaluation 隔离**：减少把模型自评、人工偏好或 continuity 分数变成自我强化回路。
4. **World / Self 双轨**：不把环境事实与主体需求混为一条 reward。
5. **WiringLevel + rollback**：新 learner 可 SHADOW 双跑，不必直接改变用户行为。
6. **bounded steering**：no free bias、norm cap、strict noop 约束 actuator。
7. **consent / boundary**：关系产品不可优化掉的硬边界。
8. **多频率 owner**：快速适应与 rare-heavy 修改不共享权限。

这些不是附属工程细节，而是把“会持续改变的系统”安全部署所必需的控制面。

## 14. Sutton / Oak 对 Volvence 暴露的核心不足

1. **没有可塑性 SSOT**：当前无法知道小网络是否越学越僵。
2. **step-size 粒度过粗**：band-level gate 不能区分同一网络内稳定与快变参数。
3. **feature topology 基本固定**：缺少成熟度、utility、替换和 optimizer-state hygiene。
4. **抽象 utility 未闭环到 planning**：可读/可命名不等于对行动规划有用。
5. **真实 consequence 密度不足**：对话里的很多 PE 仍可能是代理，而非 grounded outcome。
6. **长期 benchmark 太短或目标不匹配**：510 turns 可检查流程，远不足以复现数千任务/千万步退化。
7. **资源账本不完整**：缺少每 settled experience 的 compute、memory、latency 和 energy 指标。
8. **基础表示的长期上限未知**：冻结 LLM 是当前正确边界，但外接 learner 是否能承担所有新抽象尚未证明。
9. **候选退休语义不足**：ModificationGate 擅长晋升/阻止高风险 artifact，但 OaK 式全层 candidate lifecycle 尚不完整。

## 15. 最终比较结论

### 可以说

- Volvence 与 Sutton/Oak 在“从经验持续学习、跨时间抽象、PE 驱动、有限资源”上高度一致。
- Volvence 的 contract、owner、rollback 和 safety 设计为持续学习提供了 Oak 公开材料中尚未展开的治理层。
- Sutton 的可塑性工作揭示了 Volvence 当前一个真实且未被正式观测的底层风险。
- CMS 是最适合先吸收 CBP / step-size 思想的 owner-local 实验面。

### 不能说

- Volvence 已经实现或超过 OaK。
- Oak 已经证明完整持续心智可行。
- CBP 能解决 Volvence 的遗忘、关系连续性和时间抽象。
- 当前 CMS 已通过长期吸收—保持 gate。
- synthetic data 应全部停止，或任何在线权重更新都优于 frozen substrate。

最公平的定位是：

> Oak 提供了一个更激进的持续可塑性与自主抽象研究纲领；Volvence 提供了一个更有界、契约化、关系安全导向的持续适应骨架。下一步不是二选一，而是在 Volvence 的 owner / snapshot / gate 约束内，对 Oak 的底层可塑性机制和高层 abstraction utility 分别做可证伪实验。
