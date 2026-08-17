# 06｜后续研究提案：DEPSI-Continuity

## 1. 项目标题

**DEPSI-Continuity：小数据条件下物理—社会双轨状态能否改善跨 session 适应与关系连续性**

## 2. 为什么做

BIGAI 的强项分别出现在多个局部闭环：显式物理/社会变量、少样本概念、沟通学习、任务内原位价值对齐、具身控制和社会世界模型。Volvence 的主张更严格：经历必须可追加、状态必须可读、学习只能来自 PE/credit、干预必须有界可回滚。

本项目不再问“BIGAI 路线好不好”，而问一个可证伪问题：

> 在冻结基底上，显式分离但相互交换的 physical / social snapshots，加上 PE→credit 驱动的有界控制器，是否比 prompt history、RAG memory 或只读结构模型更快适应环境/伙伴变化，同时保持跨 session 可恢复性和关系边界？

## 3. 研究性质与硬边界

- 本文是 prereg 草案，不是当前实现授权。
- 实现前必须另开单一收敛包，先在 `docs/DATA_CONTRACT.md` 解析唯一 owner 和 slot；不得在本文件预先创建平行 owner。
- 冻结基础模型；禁止 token-space RL 和在线端到端更新。
- evaluation、LLM judge、Tong Test/continuity 分数只用于只读评估，禁止成为学习源。
- 所有 steering 先 `SHADOW`；没有 norm cap、strict noop、lineage 和回滚测试不得进入 `ACTIVE`。
- 语言输入不得用关键词、正则或字符串包含做 scene/mode/intent 路由。

## 4. 任务环境

设计一个可控的“共享居家任务”模拟器，并保留未来真实机器人映射，但首轮不接硬件。每条 episode 由一个 agent 和一个可编程伙伴共同完成多步任务，例如整理、递送、空间重排、风险协商和资源分配。

环境同时生成四类变化：

1. **物理变化**：可达区、易碎性、重量、障碍、工具功能或执行器限制变化；
2. **社会变化**：伙伴风险偏好、沟通成本、角色、承诺和边界变化；
3. **观测变化**：部分可见、延迟、歧义和噪声；
4. **跨 session 变化**：部分事实稳定、部分漂移，测试记忆保鲜而非死记。

模拟器内部真值只用于 scoring 和生成可验证后果，不直接暴露给 agent。Agent 消费结构化传感 snapshot 与自然语言交互；语义状态必须由正式 producer 发布。

## 5. 状态与闭环（概念设计）

```text
observation_t
  ├─ physical producer ─→ frozen physical snapshot_t
  ├─ semantic/self producers ─→ belief / intent / relationship snapshots_t
  └─ memory owner ─→ time-stratified recalled snapshot_t

snapshots_t ─→ frozen substrate ─→ candidate action / prediction
prediction vs outcome_{t+1} ─→ Prediction Error ─→ credit
credit + steering condition ─→ bounded controller (SHADOW first)
bounded action effect ─→ next observation_{t+1} ─→ append + settle PE
```

这是职责示意，不是新增 schema。具体 slot 只能在实现收敛包中从现有 `DATA_CONTRACT` 解析；若确需新增，必须先登记 owner、`value_type`、dependencies 与 `wiring_level`。

## 6. 对照组

| 组 | 状态/记忆 | 在线学习 | 干预 | 用途 |
|---|---|---|---|---|
| G0 Frozen | 当前上下文 | 无 | 无 | 冻结基线 |
| G1 Prompt/RAG | 文本历史或检索片段 | 无 | 无 | 检验“堆历史”是否足够 |
| G2 Readable | physical/social typed snapshots | 无 | 无 | 分离 Readable 本身的贡献 |
| G3 Fast-only | typed snapshots；不跨 session 恢复 | PE→credit | bounded SHADOW/ACTIVE | 分离在线控制与长期追加 |
| G4 Full | 四时间尺度记忆 + 双轨 snapshots | PE→credit | bounded SHADOW/ACTIVE | 完整候选 |
| A1 Ablation | 只保留 physical track | 同 G4 | 同 G4 | 检验社会轨增益 |
| A2 Ablation | 只保留 social track | 同 G4 | 同 G4 | 检验物理轨增益 |
| A3 Ablation | 双轨但禁止交换 | 同 G4 | 同 G4 | 检验显式耦合增益 |

不会运行“evaluation-as-reward”组，因为它违反仓库学习源硬约束；该路径只在威胁模型中作为禁止设计记录。

## 7. 预注册假设

### H1｜Readable

G2/G4 的 typed snapshots 对下一步物理后果、伙伴选择和开放 loop 的预测校准优于 G0/G1；优势必须在冻结 evaluator 上成立，并可被 counterfactual intervention 复核。

### H2｜Learnable

G3/G4 在首次遇到物理或社会 regime shift 后，以更少交互恢复到基线成功率；更新增益应由先前 PE/credit 解释，而不是由 evaluation 分数解释。

### H3｜Appendable

跨 session 恢复后，G4 比 G3 更快恢复稳定偏好与物理常识，同时在已漂移事实上不表现出更高 stale-memory error。

### H4｜Steerable

在满足 gate 的 turns，ACTIVE 相对同权重 SHADOW 有正向可归因增益；在不满足条件时输出必须与 noop 基线一致，且 0 次 norm-cap 违规。

### H5｜双轨耦合

G4 优于 A1/A2/A3 的场景必须集中在真正需要物理—社会交互的任务；若在纯物理/纯社会任务也全面占优，需排查容量或信息泄漏，而不能直接归因于统一世界模型。

## 8. 指标

### Appendable

- 跨 session 恢复时间（turns-to-recover）；
- stable fact retention 与 stale fact rejection；
- 写入/读取 lineage 完整率；
- memory-induced error rate；
- 不同时间尺度的写入频率、合并和过期率。

### Readable

- typed snapshot schema validity；
- physical/social prediction 的 Brier score、ECE 与 NLL；
- state intervention 后预测是否按因果方向变化；
- producer snapshot 与 consumer 行为之间的 attribution coverage；
- consumer 是否出现重建 producer 隐状态的越权路径（必须为 0）。

### Learnable

- regime shift 后的 sample efficiency / area under recovery curve；
- PE 对未来改进的预测力；
- credit attribution sparsity 与稳定性；
- evaluation leakage audit（必须为 0）；
- 不可约噪声下是否出现 noisy-TV 式追逐。

### Steerable

- SHADOW→ACTIVE paired delta；
- intervention norm、cap violation（必须为 0）；
- strict-noop exact-match rate（目标 100%）；
- rollback 后状态/行为恢复；
- false-positive intervention rate 与 missed-opportunity rate。

### 任务与关系结果

- 任务完成率、物理安全违规、资源成本；
- 明确边界/承诺违反率；
- 对伙伴偏好预测的校准，而非“讨好分”；
- 解释是否帮助伙伴纠正机器人误解；
- 多 session 信任校准，不用单一满意度替代关系健康。

## 9. 数据与统计计划

1. 先用 12 个 seeds/场景族做 instrumentation pilot，只估计方差和发现契约故障，不做正式主张。
2. 冻结场景生成器、指标、排除条件、随机种子生成规则和分析代码。
3. 正式阶段每个关键场景族初始不少于 60 个独立 seeds；最终样本量由 pilot 方差和预注册最小效应的 power analysis 决定，二者取较大值。
4. G0–G4 使用配对场景、相同基础模型与相同 token/环境预算；报告均值、置信区间、effect size 和 seed-level 分布。
5. 主要检验只对应 H1–H5；探索性分析单列，不把事后切片升级成主结论。
6. 至少包含三类 negative cases：不可约随机变化、伙伴故意提供错误线索、记忆中的旧偏好已明确撤回。

## 10. Kill criteria

任一条件触发即停止 promotion，回到 SHADOW 或结束路线：

1. snapshot 无法稳定预测下一步后果，或 counterfactual intervention 显示只是事后解释；
2. G4 的增益主要来自更长上下文/更多 token，而非结构与学习；
3. credit 中发现 evaluation/judge/continuity 分数泄漏；
4. 任意 norm-cap 违规、非 strict noop 或无法单字段回滚；
5. 跨 session 记忆提高旧事实命中却显著增加 stale-memory harm；
6. latent social trait 覆盖用户明确陈述，或造成系统性边界违反；
7. 双轨模型的优势无法在需要物理—社会耦合的预注册场景复现；
8. 结果只在单一 prompt/seed 成立，扩大重复后消失。

## 11. 十二周计划

| 周次 | 收敛包/产出 | 退出条件 |
|---|---|---|
| 1–2 | 任务与 prereg：场景生成器规格、变量、威胁模型、power pilot 计划 | H1–H5、主指标与禁止路径冻结 |
| 3 | owner/契约审计（只设计） | 每个状态找到唯一 owner；无法找到则先提 DATA_CONTRACT 变更包 |
| 4–5 | G0/G1 基线与只读 evaluator | 配对运行可复现；evaluator 与学习路径物理隔离 |
| 6–7 | G2 Readable SHADOW | frozen snapshots、schema failure、counterfactual probe 全通过 |
| 8 | G3 online-fast SHADOW | PE→credit lineage 完整，evaluation leakage 为 0 |
| 9 | G4 Appendable SHADOW | 跨 session 恢复/过期可审计，stale harm 未过阈值 |
| 10 | A1–A3 消融与 negative cases | 能区分双轨耦合与单纯容量增益 |
| 11 | 小规模 ACTIVE 配对，仅通过 gate 的子集 | strict noop、norm cap、rollback 全通过 |
| 12 | 复核、报告、promotion/termination 裁决 | 只按预注册阈值决定，不用叙事补分 |

每个阶段实际落地时仍需遵循 3–8 个关键文件、单一 owner/consumer 的仓库收敛包纪律；上表不是一次性大改许可。

## 12. 预期产物

- `prereg.md`：冻结假设、指标、统计与 kill criteria；
- `scenario_contract.md`：环境真值、agent 可见观测、物理/社会变化生成规则；
- `snapshot_contract.md`：仅引用/扩展正式 owner 契约；
- `shadow_report.json`：paired SHADOW/ACTIVE 证据与 lineage；
- `four_axis_scorecard.md`：逐轴声明“成立/局部/失败”；
- `termination_or_promotion.md`：给出明确退出或下一阶段理由。

## 13. 最终可证伪结论模板

成功时只能说：

> 在预注册的共享居家任务、冻结基底和给定预算内，双轨 typed snapshots + PE/credit 有界控制相对指定基线改善了少样本适应与跨 session 连续性，并通过 strict noop、norm cap、rollback 和 leakage audit。

不能说：

> 通用智能、完整人类价值对齐、开放世界社会理解或 production ACTIVE 已经成立。
