# 对 Volvence 的参考意义

## 1. 总裁决

这条研究对 Volvence 的价值是：

> 为“语言的结构变化可成为 named readout”提供方法启发，并提醒我们把单次文本判断升级为带 baseline、时间、来源与不确定度的结构快照。

它不是：

- 新的心理状态 owner；
- 把用户文本偷偷转成 depression/anxiety 标签的授权；
- Prediction Error、credit 或 reward 的来源；
- 可以直接进入 prompt planner 的关键词/拓扑规则；
- “在线持续主动学习系统成立”的外部证据。

当前建议成熟度是 **research-only → offline benchmark → SHADOW probe**。在完成真实 human score、长度控制、个体内 longitudinal 与跨语言复现前，不应晋升 ACTIVE。

## 2. 四能力轴

### 2.1 Appendable

单个 TFMN 只是一张静态图，对 Volvence 价值有限。可追加版本必须记录：

- turn/session/speaker/language/register lineage；
- tokenizer/parser/lexicon/version fingerprint；
- raw length、content length 与 graph coverage；
- 8 个 network readout、8 个 emotion readout；
- baseline window 与相对 delta；
- confidence、missingness、normalization policy。

时间尺度建议：

- online-fast：当前 turn 的纯观测 feature snapshot，不持久解释；
- session-medium：同一人的 rolling baseline、delta 与 volatility；
- background-slow：跨 session 的稳定性/漂移审计；
- rare-heavy：只有经人工与 ModificationGate 审核，才可能改变 parser/feature artifact。

写入 owner 不能是 expression consumer。若未来实现，应先决定由哪个 cognition/perception owner 发布正式、frozen、可恢复 snapshot；CMS 只保存 owner 的公共交换，不保存下游重建的第二套心理真值。

### 2.2 Readable

这是最强命中轴。TFMN/EmoAtlas 的优点是把语言输出转成 named features，而不是关键词标签。

但 Volvence 的 Readable 约束更严格：

- publisher 负责构图、长度校正、解释和 uncertainty；
- consumer 只读 frozen snapshot；
- consumer 不得遍历 raw edges 后自行推断“抑郁/焦虑”；
- 不得通过字符串包含、正则或情绪词词典直接路由 mode/action；
- text-derived readout 与 substrate residual readout 必须保持来源隔离，不能把文本 proxy 冒充内部状态。

interlocutor_state 当前明确“不读 user_input”，只从六个上游 typed snapshot 派生 12 axes；因此不能把 TFMN 逻辑直接塞进该 consumer。若未来采用，正确方向是由独立上游 owner 发布低带宽、typed、length-controlled readout，再由 interlocutor_state 通过正式 dependency 消费。任何新增 slot 必须先登记 docs/DATA_CONTRACT.md。

### 2.3 Learnable

语言网络/情绪特征只能是 observation 或 evaluation evidence，不能成为学习源。

合法路径：

text/network observation → owner snapshot → next-state prediction → real outcome → prediction_error → credit → ModificationGate

非法路径：

- predicted depression score → reward；
- network “rumination” score → negative credit；
- 七日 continuity/readout → RL signal；
- clinician/judge label → 直接更新 controller；
- SHAP importance → 硬编码策略。

论文训练目标是量表分数，属于 supervised psychometric mapping；它没有提供 Volvence 所要求的 PE-first credit path。故它只能支持“可读”，不支持“可学”主张。

### 2.4 Steerable

论文没有 residual intervention、norm cap、strict noop、gate、artifact lineage 或 rollback 证据，所以对 Steerable 几乎没有直接贡献。

如果未来结构 readout 影响行为，也必须经过：

- typed condition belief，而非 raw score；
- conservative confidence gate；
- matched no-op；
- bounded action family；
- next-turn outcome 与 PE settlement；
- WiringLevel.SHADOW → ACTIVE 的独立晋升；
- 单字段回滚。

尤其不能写成“网络聚类高 → 多安慰”“悲伤 z 高 → 自动心理干预”之类表达层规则。

## 3. 与现有 owner 的关系

| 现有 owner / 能力 | 可参考之处 | 禁止做法 |
|---|---|---|
| interlocutor_state | 将 upstream 命名读出压缩成 emotional/engagement/stability 等 axes | 让它直接解析原始文本或重建 TFMN |
| relationship_state | 使用同人 longitudinal delta 辅助观察 attunement/repair trajectory | 把 distress 当作关系真值或信任分 |
| user_model | 用户明确自述可成为 profile fact；结构读出可作为低置信观察 | 暗中固化“抑郁人格”“焦虑用户” |
| rupture_state | TEA/agency 与网络突变可作为未来诊断性 evidence 候选 | 用 emotion/network 单源触发 rupture kind |
| prediction_error | 对结构 readout 的下一拍预测与实际变化进行结算 | 把结构异常本身直接当 PE/reward |
| evaluation | 跨体裁、跨语言、长度控制、within-person 稳定性报告 | evaluation 反向写 credit/gate |
| boundary_consent | 心理语言分析必须是显式同意、可撤回能力 | 默认开启秘密心理画像 |

## 4. 最值得借鉴的四点

### 4.1 从 absolute state 改为 perturbation against baseline

“心理状态是一种网络扰动”若要对 Volvence 有意义，重点不是某个绝对阈值，而是：

- 同一人在相似任务/长度/语言下的偏移；
- 相邻 turn 与 session 的结构变化；
- change-point 后是否恢复；
- 关系事件前后的 paired delta。

这与 Volvence 的 PE/trajectory 观念更相容，也能减少把个人稳定语言风格误判为心理状态。

### 4.2 多视角 readout，而不是一个总分

建议保留结构的分解：

- size-sensitive：nodes、edges、components；
- size-normalized：assortativity、density-like measures；
- topology：clustering/path/diameter；
- affect：8 个 emotion z；
- agency：TEA actor/action/target；
- provenance：parser/lexicon/language/model/version。

不要过早压成“depression probability”。多维 snapshot 更可审计，也更不容易 Goodhart。

### 4.3 合成 persona 适合做 counterfactual probe

MHDS/MEDS/CDS 的最佳用途不是生成训练真值，而是：

- 检查不同 substrate/model family 是否在相同 persona 下产生不同结构；
- 做 prompt、role、alignment、language matched controls；
- 暴露安全对齐、拒答、sycophancy、overconfidence 对 readout 的影响；
- 生成 failure cases，再用真实用户数据验证。

这与 Volvence rare-heavy/evaluation sandbox 很契合，但 synthetic evidence 不能独立晋升 production。

### 4.4 数据产品应与论文一起发布

CogNosco 的 codebook、notebook、dashboard、archive 和图形化 provenance 值得直接借鉴为研究流程标准。Volvence 未来每个新 probe 至少应有：

- schema/data card；
- exact artifact digest；
- frozen benchmark split；
- positive/negative/matched cases；
- calibration 与 failure table；
- rollback and deletion policy；
- source/license manifest。

## 5. 建议的最小 SHADOW 收敛包

这不是本轮实现承诺，而是后续若立项时的最小范围。

### 包 A：offline instrument validation

目标：只验证仪器，不接 runtime。

1. 选择同一语言、同一 register、长度匹配的已有脱敏文本。
2. 固定 parser/lexicon/feature schema。
3. 比较 raw、length-residualized、size-normalized 三组特征。
4. 做 paraphrase、句序打乱、情绪词替换、内容保持长度变化等 matched perturbations。
5. 检查同文重测、跨 parser、跨语言、短文本最低覆盖率。

通过条件：结构 readout 对目标扰动敏感、对纯长度/格式扰动不敏感，并能发布完整 lineage。

### 包 B：within-person SHADOW snapshot

目标：验证 trajectory，不改变用户可见行为。

1. 新 owner/slot 先在 DATA_CONTRACT 登记；
2. 发布 frozen snapshot 与 confidence；
3. 只记录相对个人 baseline 的 delta；
4. evaluation 读取并生成 report；
5. 不进入 interlocutor_state、PE、credit、planner 或 memory write gate。

### 包 C：PE settlement experiment

只有 B 稳定后，才研究能否预测下一 turn 的结构恢复/恶化：

- pre-action prediction；
- matched no-op；
- real external/relationship outcome；
- PE owner settlement；
- credit owner 决定是否产生学习证据。

结构读出本身仍不能成为 reward。

## 6. Prereg 必须回答的问题

1. 目标是测量文本结构、心理量表、临床组别，还是对话状态？只能选一个 primary claim。
2. 训练和测试是否跨 speaker、跨 session、跨 generator 分割，避免同源泄漏？
3. length、topic、register、language、parser 如何控制？
4. primary metric 是 calibration、rank discrimination 还是 within-person change？
5. synthetic→human 的 widening ladder 如何定义？
6. 失败模型是否全部计入，而不是只报告成功 generator/prompt？
7. 用户同意、撤回、删除与敏感属性访问如何实施？
8. 结果如何保持 report-only，不进入 learning source？

## 7. Kill conditions

满足任一条，应停止 runtime 路线并保留为离线研究：

- 加入长度/体裁控制后主要信号消失；
- 跨 generator 或跨语言方向翻转；
- within-person reliability 低于稳定个人风格差异；
- 无法区分 prompt scaffold 与 construct signal；
- 需要关键词/正则才能达到可用效果；
- 只有 synthetic persona 有效，真实带量表数据无效；
- 误报导致行为策略明显变差；
- 无法提供明确 consent、revocation 与 data deletion；
- evaluation/readout 必须回灌才能“学会”。

## 8. 最终参考价值

### 可以直接进入研究原则

- 不只读词频，要读结构、情绪、agency 与动态；
- 相对个人 baseline 的 perturbation 优于跨人群绝对标签；
- owner 应发布 named, frozen, versioned readout；
- 合成 persona 是受控审计工具；
- 跨体裁/跨语言/跨 generator 是最低稳健性要求；
- 负面结果与失败模型必须和最好数字一起报告。

### 暂时不能进入产品主张

- “Volvence 能从语言读出用户心理状态”；
- “语言网络反映真实 depression/anxiety 分数”；
- “网络结构能驱动个性化心理干预”；
- “该机制证明 Learnable/Steerable 已成立”。

最诚实的结论是：它为 Readable 层提供了一个值得验证的结构仪器族，也为我们画出了非常清楚的误用边界。
