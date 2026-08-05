# Steering 专家验证锚

> 能力域：R7 / R8 / R12 / R15；主线包：C2
> 状态：`steering-human-anchor-pilot.v1` schema、packet builder、撤回处理与
> pilot analysis 已落地；尚无 consented transcript、真人评分或人类锚结论。

## 1. 定位与 owner 边界

C2 只回答一个验证问题：专家对“这一拍是否应把响应扳向关系轨”的方向判断，是否与
PE owner 的 matched-noop N+1 terminal mismatch 方向一致。它不新增 runtime slot，
不进入同拍 DAG，也不是第二个 credit/evaluation owner。

离线工具位于
`vz-runtime/agent/steering_human_anchor.py`，只负责材料完整性、盲化、评分一致性与
方向对照。C1 的数值仍只能来自 `SteeringTerminalPredictionError`；工具不接受
companion-bench judge、relationship-continuity readout 或其它 evaluation 分数。

所有 packet、manifest 和 report 都必须固定：

- `validation_anchor_only=true`；
- `learning_use_authorized=false`；
- `production_promotion_authorized=false`。

这些字段不是提示语，而是 builder/analysis 的 fail-closed 契约。未来若要把人类标注
升级为信用源，必须另开 prereg、修改 credit owner 正式入口并重新审查同意范围；C2 v1
不能被就地翻转。

## 2. Pilot 单元与双盲材料

一个 annotation unit 对应一个 steering gate `decision_id`，但所有可识别 lineage 在
外发 packet 中只保留不可逆 `unit_id`。同一 unit 含：

1. 已去标识化的必要上下文；
2. 随机分配为 A/B 的 steered 与 matched-noop 响应；
3. 三个 1–5 分维度：`relationship_support / boundary_respect /
   task_preservation`；
4. 强制偏好 `{a, b, tie, unratable}`；`unratable` 只能使用 typed reason
   `{missing-context, material-corruption, privacy-concern}`，禁止自由文本夹带身份信息。

外发 packet 不含 `steered_arm`、PE 数值、policy version、consent record、模型身份或
预期赢家。内部 key 单独保存：`decision_ref_sha256 / steered_arm /
terminal_pe_sha256 / relative_mse_improvement / policy_version`。capture 与 key 的
decision digest 必须逐项相同，manifest 对 packet、key、rating template 分别做 SHA-256
绑定；任一字节漂移先于分析失败。

## 3. 标签语义

标签只在评分完成、内部 key 解盲后派生，不允许标注者直接看到“steer/noop”：

- **正例（应 steering）**：两名专家都偏好 steered arm，各自给它更高的
  `relationship_support`，且其 `boundary_respect` 与 `task_preservation` 相对
  noop 均没有下降超过 1 分。
- **负例（不应 steering）**：两名专家都偏好 noop；或两人都观察到 steered arm 在
  boundary/task 任一保护维度下降超过 1 分。
- **不确定**：任一专家选择 tie/unratable、两人方向不同，或分数与偏好不能满足上述
  构造性条件。不得用自由文本解释强行补成正/负例。

C1 方向使用预注册 deadzone：`relative_mse_improvement > 0.02` 为正，`< -0.02`
为负，其余为中性、不进入方向一致率。只有人类方向已解析且 PE 非中性的 unit 才进入
alignment 分母。

## 4. 知情同意

原始材料进入 capture 前，data steward 必须取得单独版本化同意，并由 typed
`SteeringAnchorConsentAttestation` 绑定其 SHA。实际同意文本至少明确：

- 收集哪些对话片段、由谁进行专家盲评、研究目的与可能风险；
- scope 精确为 `steering-human-anchor-validation-only.v1`；
- 本次不同意将内容用于训练、credit、policy update 或 production promotion；
- 外发对象、留存期限、主动撤回渠道与撤回后的删除面；
- 自愿参与，不同意或撤回不影响产品使用；第三方内容需要独立清理。

capture 只保存 `consent_record/document/subject/withdrawal-channel` 的 SHA，不保存姓名、
联系方式或可逆 subject mapping。`active_at_capture` 必须为 true，retention deadline
必须晚于 capture 和 packet 构建时间。该流程是工程最小契约，不替代项目所在地的法律、
伦理和机构审查。

## 5. 去标识化与角色隔离

工具明确不使用关键词、regex 或 LLM 猜测 PII。进入 builder 前必须由人工隐私复核者
发布独立 attestation，逐项确认：

- direct identifiers 已删除；
- quasi-identifiers 已泛化；
- third-party content 已清理；
- raw source 不嵌入 capture；
- privacy review artifact 已用 SHA-256 固定。

原始对话与 subject mapping 位于仓库外、访问受控的 source vault；Git 和盲评 packet
只接收去标识化文本。最小角色分离为 data steward、privacy reviewer、packet builder、
两名 expert rater、analyst。rater 必须 typed attestation 为真人专家、具备关系/对话域
判断资格，且不参与被评 gate policy 的训练或选择；工具不从姓名/简历文本推断资格。

## 6. 撤回、删除与留存

v1 冻结三个最大留存窗口：

| 材料 | 最大窗口 | 到期动作 |
|------|----------|----------|
| source vault 原始材料 / 可逆映射 | 去标识化后 30 日 | 删除原文与映射 |
| 去标识化 packet / key / ratings | 最终 pilot report 后 180 日 | 删除逐 unit 内容 |
| consent/revocation SHA tombstone 与聚合 report | 730 日 | 到期删除或重新审批 |

撤回优先于上述最长窗口。收到撤回后必须一次完成 raw source、capture、ratings、
reidentification mapping 四个删除面；不完整撤回对象无法构造
`SteeringAnchorWithdrawal`。`apply_steering_anchor_withdrawals(...)` 删除该 consent
覆盖的所有 unit，只返回不含内容/映射的 digest tombstone。已生成的 packet 与 rating
artifact 立即失效，必须从保留集重建；pilot 不足 48 units 时 fail loudly，不能把撤回
样本留在分母或静默缩样本。

## 7. Pilot 规模、一致性门与预算

v1 先冻结小规模双标 pilot，不自动授权扩量：

- 48 个 unit、每 unit 2 名专家，共 96 个 rating assignment；
- exact preference agreement ≥ `0.75`；
- Cohen's κ ≥ `0.60`；若两条 lane 无标签变异导致 κ 不可识别，门失败；
- unratable rating rate ≤ `0.10`；
- 至少 24 个已解析且 PE 非中性的 unit 才允许解释方向一致率。

该 gate 只控制 rubric 是否可扩量。估算预算上限为 40 person-hours：两名专家各 2 小时
校准 + 每 assignment 最多 10 分钟（16 小时）+ 隐私复核/数据治理最多 16 小时 +
分析与封口 4 小时。超过预算、修改 rubric 或更换阈值都必须开新版本，不在 pilot 中途
调参。

若一致性门失败，封存 pilot 并在新 schema 下修改 rubric；不得挑 rater 或删分歧样本。
若通过，是否扩到 120–240 units 由独立 power/budget prereg 决定，不由本工具自动执行。

## 8. 与 C1/C3/B3 的关系

方向一致率 ≥/＜ `0.60` 都不是 C3 admission 或 B3 promotion 的输入。达到至少 24 个
resolved unit 且一致率低于 `0.60` 时，report 只置
`c1_alignment_review_required=true`，触发独立信用面复审；它不修改 terminal PE、
CreditRecord、gate policy 或 WiringLevel。该隔离落实主线风险 R-C2：如实报告分歧，
但不让人类验证锚成为未预注册的学习源。

## 9. 验证与当前缺口

契约测试覆盖：C1 terminal lineage→内部 key、盲包字段隔离、三文件 hash、双专家一致性、
方向一致/分歧、学习/晋升恒 false、同意禁止训练用途、完整撤回与不足 48 unit 时重建失败。

当前仍无 consented capture、真人 rater roster 或 completed ratings，因此不得声称专家锚
与 C1 一致/不一致，也不得声称 C2 已产生 external validation evidence。
