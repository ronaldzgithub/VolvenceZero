# 04｜采纳决策

## 1. 决策摘要

| 项目 | 决策 | 优先级 | 理由 |
|---|---|---:|---|
| profile / situation / history / understanding / emotion / response 分面 schema | **采纳思想** | P0 | 能让数据职责清楚；需改成 typed owner/evidence 语义 |
| 多维 empathy rubric | **改造后采纳** | P0 | 适合 evaluation，只读且需跨 judge/human calibration |
| KardiaBench 外部 benchmark | **申请并审计后考虑** | P1 | profile-conditioned 短多轮价值高；gated、NC/ND、隐私风险 |
| Kardia-R1 reference SUT | **申请后考虑** | P1 | 可代表专门 empathy fine-tune 路线；不进核心 runtime |
| profile-disjoint 测试意图 | **采纳** | P1 | 能区分记忆/泛化与 persona memorization；需真实 manifest |
| easy/hard 数据路由 | **仅借 curriculum 思想** | P2 | 可用于离线样本编排；原公式粗且 rubric-biased |
| 四段 visible reasoning | **仅保留分面，不采纳为内部状态** | P0 | free-form rationale 不满足 snapshot SSOT |
| MBTI persona grounding | **降级为弱 metadata** | P0 | 不能成为用户本体、路由或关系策略真值 |
| Rubric-as-Judge GRPO | **拒绝进入在线链** | P0 | evaluation 泄漏、token-space RL、缺实际 outcome |
| 原始 PersonalityCafe profiles | **不采纳** | P0 | 重识别、同意、版权、商业许可与关系产品伦理风险 |
| Kardia 数据证明 longitudinal relationship | **拒绝** | P0 | 最多 10 exchanges，无跨 session、纠正或真实后果 |

## 2. 可以立即吸收的五件事

这些是研究/规范层面的吸收，不需要下载数据，也不改变 runtime：

### A. 把“共情”拆成可分别审计的表面

保留 Kardia 的分面思想，但改成 Volvence 语义：

```text
observation/profile evidence
→ owner proposal / frozen readout
→ emotion evaluation anchor
→ response artifact
```

不要保留“模型写一段 reasoning，因此内部认知已透明”的论证。

### B. 评测必须显式给 profile，而不是只给 situation

同一句支持话术是否正确，取决于用户明确资料、关系历史和边界。外部 benchmark 应至少区分：

- situation-only；
- profile-text；
- full-history / RAG steelman；
- Volvence typed-state。

Kardia 提供了一个很强的 profile-text/SFT 参照，但不替代 typed-state 证明。

### C. 把专门后训练的小模型列为强基线

论文显示 2B–7B 小模型经领域数据 SFT 后可以超过很多 raw 大模型。未来关系产品评测不能只拿
通用 GPT/Qwen baseline；应包含至少一个 Kardia-R1 这类 specialized empathy SUT，否则会高估
系统架构带来的独立价值。

### D. 多维 rubric 只作只读报告

Relevance、Fluency、Empathy、Persona、Safety 的拆分值得保留，但必须：

- 固定 judge identity、版本、prompt hash；
- 至少跨 judge family + 人类锚校准；
- 报告 per-axis 分布、方差与 disagreement；
- 不进入 PE、credit、controller gradient 或 memory writeback。

### E. 数据贡献与 RL 贡献必须分开报告

以后看到“RL 提升共情”时，默认要求 `base → SFT → RL` 三段对账。Kardia 的 Table 2 正好
说明，如果只报 base → final，会把大部分属于数据/SFT 的提升归功于 RL。

## 3. 需要改造后才能采纳的部分

### 3.1 四段式 schema

建议借鉴：

- `understanding` → typed proposal / evidence-grounded summary；
- `emotion` → read-only label + confidence + evidence refs；
- `response` → expression artifact；
- `reasoning` → 可选的人类可读 rationale，不承担 owner 或控制职责。

任何正式状态仍由现有九类 semantic owner 持有；本研究不建议新增第十个
`emotion_reasoning` owner。

### 3.2 curriculum

可借“高稳定样本做 SFT、难样本单独处理”的数据工程思想，但不要复制当前 Eq. 4：

- easy/hard 必须由可复核统计定义；
- judge 只决定离线样本路由，不能成为在线 reward；
- 不应把被同一 rubric 反复修订过的数据再称为 rubric-independent；
- 路由阈值和样本 manifest 必须冻结、可追溯。

### 3.3 数据集作为 benchmark

若获授权，第一用途应是**evaluation-only**：

- 保留官方 test split，不从 test 训练或调 prompt；
- 先查 base model / judge 对数据与 profile 的 contamination；
- 把 emotion accuracy 扩为 macro/per-class/calibration；
- 把 persona score 拆成“明确 profile 事实使用”“刻板推断”“与当前话语冲突”；
- 报告 MBTI removal、profile truncation 和 raw-profile leakage 的只读切片；
- 绝不把分数回写 PE/credit。

这是一份使用原则，不是实验任务或本包授权的实现。

## 4. 明确拒绝的部分

### 4.1 拒绝 Rubric-as-Judge 进入学习主链

理由不是 rubric 没价值，而是它属于 prejudgement：它预测“看起来是否共情”，没有观察
真实用户下一拍、关系修复或边界后果。Volvence 的学习源仍必须是实际 outcome 结算后的
PE 及其下游 credit。

### 4.2 拒绝把 MBTI 或 scraped profile 当用户真值

用户模型只能保存明确自述、来源、置信度、纠正和撤回；类型标签最多是用户选择提供的弱
metadata。任何“因为你是 INFP/ISFJ，所以你一定……”都应视为 stereotype 风险。

### 4.3 拒绝把 visible CoT 当 Readable 证明

Kardia 的四段文本可以被编辑、模板化或 reward-hack。Volvence 的 Readable 要求 residual/
owner state 以冻结快照发布，consumer 不能从表达文本反推 producer 隐状态。

### 4.4 拒绝直接下载进主仓或训练集

在 gated agreement、NC/ND、profile 权利、重识别和商业用途没有书面结论前，直接纳入会同时
制造许可、隐私、污染与删除困难。

## 5. 数据获取决策

### 建议做

1. 以“内部、非商业、只读 benchmark 审计”为目的申请 gated access；
2. 同时向作者询问商业内部评测、输出发布和衍生模型是否可单独授权；
3. 在受限外部数据区保存，记录 dataset revision/hash、访问人、期限和删除流程；
4. 完成样本级隐私、split、重复、污染、schema 和 license audit 后再决定是否适配 benchmark；
5. 若不能取得适当权利，只借其 schema，用自有 synthetic / consented personas 独立构造数据。

### 不建议做

- 以 GitHub MIT 为依据绕过数据协议；
- 将数据放进公共 `research/papers`、普通 artifact 或可自动打包目录；
- 在申请理由中承诺训练产品模型；
- 先下载再补权利审查；
- 对 profile 原文做搜索式重识别验证并把结果传播出去。

## 6. 最终优先级

### P0：现在就吸收

- 数据/SFT 与 RL 贡献分账；
- profile-conditioned 强基线意识；
- 四分面 annotation → typed owner/readout 的改写；
- rubric 只读与 judge lineage；
- MBTI / raw profile 的反刻板与隐私边界。

### P1：拿到权限后再决定

- KardiaBench external benchmark adapter；
- Kardia-R1 reference SUT；
- 与作者核验 split、gated terms、数据删除与商业授权。

### 不进入路线图

- Rubric-ERL 在线接线；
- 用该数据直接训练 Volvence 核心；
- 新增 emotion/MBTI owner；
- 用 Kardia 分数给 Appendable / Learnable / Steerable 晋级。

## 7. 最终判词

Kardia-R1 值得认真研究，但应被放在正确层级：

> **采纳它的数据组织与强基线意识；谨慎争取数据的只读评测价值；拒绝把它的 judge reward、
> 静态 profile 和显式 CoT 搬进 Volvence 的在线学习与状态主链。**

