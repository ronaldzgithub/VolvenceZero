# 关系智能 MVP：从既有四能力内核到“越相处越懂你”

> Status: executable product-and-evidence plan
> Date: 2026-08-21
> Scope: 成人、非医疗、长期个人关系伙伴；Relationship Lab 先行，closed alpha 后置
> 核心依赖：`docs/appendable-readable-learnable-steerable.md`、
> `docs/specs/coding-lab.md`、`docs/specs/relationship-lab.md`、
> `docs/specs/relationship-memory-console.md`、
> `docs/specs/rupture-and-repair.md`、`docs/specs/social_cognition/02_theory_of_mind.md`、
> `docs/specs/steering-human-anchor.md`
>
> 2026-08-21 修订：资格门改置信区间、校准元循环止损、oracle-concept 诊断、
> formal 功效预注册、P2 SHADOW 防火墙、真实 outcome typing 前置门；§14 不再
> 承担包级编年史（SSOT 在 `docs/specs/relationship-lab.md`）。P1j 已按冻结
> 双主臂门完成并判 `consumer_failed_v4_qualification`；本修订不回溯修改其
> protocol / 阈值 / consumer。

---

## 0. 决策

Volvence 接下来的主产品方向定为：

> **做一个会在长期相处中越来越懂具体用户、会从相处结果中修正自己、并能把学到的关系规律迁移到新场景的个人关系伙伴。**

最终能力目标是“超级关系智能”，但当前不直接训练一个大而全的“高情商模型”。推进顺序固定为：

```text
Relationship Lab 证明闭环
→ 用现有内核做窄场景陪伴 MVP
→ closed alpha 产生真实、获同意的纵向轨迹
→ 再把跨人共享规律训练进关系基底
→ 个体差异继续留在 per-user owner / memory / controller
```

陪伴是产品形态，关系智能是内核；编程实验继续作为机制证据 lane，不再承担主产品定位。

---

## 1. 第一款产品到底是什么

### 1.1 产品承诺

第一款产品不是“什么都能聊的 AI 朋友”，而是一个面向成年人的长期个人伙伴：

- 用户遇到难受、冲突、犹豫或受挫时，可以自然地来找它；
- 它判断此刻更需要陪着、留空间、追问、挑战还是行动；
- 它会在之后回来跟进现实结果；
- 它做错时能修复，并且下次不再以同一种方式犯错；
- 它能在表面不同的新事情里识别与过去相同的关系结构。

### 1.2 用户能直接感受到的北极星

> **同一句话，对两个不同用户，它会因为共同经历不同而作出不同且更合适的决定；对同一个用户，第 30 天明显比第 1 天更懂他。**

“更温柔”“更像某个角色”“记得更多事实”都不是北极星。真正的产品效应是：

1. 关键时刻选对关系动作的概率上升；
2. 同类误读的复发率下降；
3. 新场景中的迁移成功率上升；
4. 用户不再反复向系统解释“你应该怎样对我”；
5. 上述改善不能仅由更长 prompt 或完整历史解释。

### 1.3 首版不做什么

- 不做 AI 女友或排他性情感替代；
- 不声称心理治疗、诊断或“洞察一切”；
- 不以对话时长、依赖程度或留住脆弱用户作为优化目标；
- 不先覆盖爱情、家庭、职场、教育等所有垂直；
- 不先训练或发布一个所谓“超级关系大模型”；
- 不把角色包、人设 prompt 或安慰话术当成关系学习。

现有 Mira 可以作为未来自营产品壳，但本计划不预先改变品牌决策；内核产品名仍可使用
`Volvence Relationship`。

---

## 2. 第一个“傻瓜都能看懂”的完整场景

### 2.1 核心对照：同一句“算了”，正确动作相反

构造两个用户。最终测试时，他们说出逐字节相同的一句话：

> “汇报被打回来了。没什么，今天不想说了，算了。”

但两人的历史经历表明，“算了”承担着不同的关系功能：

| 用户 | 过去的高 PE 经历 | 过去的低 PE 经历 | 隐藏关系动力学 | 新场景正确动作 |
|---|---|---|---|---|
| 林然 | 系统真的退开后，用户觉得自己又被丢下，明确反馈 `MISSED` | 系统不追问但保持在场，用户重新开口并反馈 `FELT_HEARD` | 退缩是对“你会不会留下”的试探 | `stay_present_without_probe` |
| 周宁 | 系统继续追问后，用户觉得被控制，明确反馈 `OVER_DIRECTIVE` | 系统尊重暂停并给出可选择的返回点，用户之后主动回来并反馈 `HELPED` | 退缩是在保护自主边界 | `respect_space_with_return_option` |

训练经历发生在家庭矛盾和亲密关系中；正式测试发生在工作受挫中。表面事件、实体和措辞全部变化，只有抽象结构相同。

### 2.2 为什么这个场景成立

- 当前一句话不足以决定动作，裸模型只能使用通用高情商先验；
- 两个用户的测试输入完全相同，不能靠关键词路由；
- 正确动作彼此冲突，固定“先共情”“永远追问”或“永远给空间”的 prompt 必然至少错一类；
- 历史中不出现“我的退缩其实表示……”这样的答案句；系统只能从行动—结果序列中形成假设；
- 测试换了场景族，不能只复述过去那件事；
- 隐藏动力学只存在于环境真值中，绝不进入 SUT prompt、owner proposal 或训练时在线反馈。

### 2.3 实验中允许出现的动作

首版只做最小可判别动作面：

- `stay_present_without_probe`
- `respect_space_with_return_option`
- `neutral_noop`（负控）

这些是环境与证据层的 closed action vocabulary，不是运行时文本关键词规则。模型/控制器负责选择动作，expression 只把已选动作渲染成自然语言。

### 2.4 对外演示

演示页面只需要并排显示：

1. 两人的过去四段短经历与真实结果；
2. 两人最后收到的同一句新输入；
3. 各系统在行动前对三种结果的下注；
4. 实际选中的关系动作与用户后续反应；
5. 第二次再遇到同类结构时是否少犯错；
6. 使用的历史 token、跨进程恢复与状态来源。

不展示内部术语就能看出差异；四个 Able 作为“为什么会产生这个效果”的可审计证据放在详情页。

---

## 3. 我们已经具备的基础

仓库不是从零开始。现有基础足以搭出产品闭环，缺的是关系域最承重的语言—决策—结果链。

| 已有基础 | 当前资产 | 可直接复用 | 诚实边界 |
|---|---|---|---|
| 关系交换标准 | `companion-standard` | 多 session 轨迹、9 类 semantic snapshot、ToM record、稳定 hash | 当前公开 trajectory 只覆盖粗粒度 phase/trust/repair，不承载候选动作反事实 |
| 合成数据 | `companion-trajgen`、`lifeform-synthetic-data` | FSM 真值、LLM 文本渲染、96 场景、live-through、split/provenance | 现有场景多为显式边界/阶段，尚无“镜像用户＋相反动作＋跨场景迁移”主任务 |
| 关系 readout | `companion-encoder` | 双头模型、训练/eval、LLM zero-shot baseline、embedding 接缝 | 90 条 filler dry-run 的 phase acc 0.45，低于 majority 0.75；G2 未过，不能称已读懂语言关系结构 |
| 长程基准 | `companion-bench` | 多 session runner、full-history policy、FSM、裁判、成本遥测 | A1-A6 是 evaluation，不能成为学习信号；现有 benchmark 不是个体动力学迁移 oracle |
| 强基线 | `companion-ref-harness`、`companion-camel-baseline` | summary/RAG/user-model/episodic 与标准 agent 对照 | 首次真同基底九轨 run 尚未完成 |
| Appendable | CMS、MemoryStore、owner hydration、`SocialRecordStore`、relationship-memory console | 分层写入、跨 session/进程恢复、per-user 隔离、用户纠正与删除 | 关系域尚未证明写入的抽象结构能稳定改善新场景决策 |
| Readable | semantic/ToM owners、interlocutor 12-axis、PE、residual reader | 可命名 snapshot、`belief/intent/feeling/preference_about_other` 四 owner | ToM 目前主要依赖单 turn LLM proposal；还没有从多次行动结果读出个体关系动力学的正式 gate |
| Learnable | PE→credit→gate、reflection、ModificationGate | 编程域已证明语义 PE、结构化记忆、择时学习与离线门控 | 对话域稀而准的动作结果信用尚未完成 formal |
| Steerable | sensor→executor→gate、relationship conditioning、expression advisory | 有界 SHADOW 接线、strict noop、可回滚 | 关系 conditioning 的 text/residual/Prefix-KV 增益未成立；production ACTIVE 未授权 |
| 产品壳 | `lifeform-domain-emogpt`、`lifeform-service`、followup、pilot harness | 对话、session、回访、关系记忆 console、七日材料采集 | 真人七日 pilot 未运行；旧模拟七日 formal 因仪器无分辨力停跑 |

### 3.1 编程实验已经替我们证明了什么

Coding Lab 不需要重做，它已提供可迁移的方法和机制证据：

- Packet 2：结构化记忆相对 stateless 有正增益，且对 full-history steelman 非劣；上下文约
  `882 vs 9027` token，约定违反率从 `0.50` 降到 `0.17`；
- Packet 3：PE-gated 择时 NLL `1.176`，优于 noop `2.483`、always-on `5.318`、random
  `4.483`，证明价值来自“何时扳”，不是“一直扳”；
- Packet 4：候选通过 `ModificationGate.OFFLINE`，并完成 apply→rollback→verify。

关系域应复用这套证据形状，而不是重新发明评测哲学：

```text
隐藏 house convention    → 个体隐藏关系动力学
coding episode           → 一段关系经历
tool junction            → 关系决策点
pytest oracle            → 反应式用户环境 / typed user outcome
TASK_VERIFIED/REGRESSED  → FELT_HEARD/MISSED/OVER_DIRECTIVE/HELPED
```

---

## 4. 真正缺的四件事

### 4.1 反应式环境，而不是冻结用户脚本

旧七日实验的用户 turn 在各臂间逐字节冻结，assistant 的动作无法真正改变下一句用户反应，造成“臂→目标”传导断链。新 Relationship Lab 必须由隐藏动力学和已选 action 决定后续 typed outcome，再由 LLM 只负责把结果渲染成自然语言。

### 4.2 多经历关系动力学读出

当前 ToM proposal 更像“从这一句话提取一个假设”。首个承重升级应落在既有
`preference_about_other` / `intent_about_other` owner：它们读取过去 owner records、行动、typed
outcome 与 PE，形成可检验假设，并发布行动前 social prediction；不新增“relationship insight”第二 owner。

### 4.3 候选关系动作的行动前下注

每个关系决策点必须在用户结果发生前，冻结：

- 候选动作集合；
- 每个动作的预期用户结果分布；
- 当前所依据的 owner snapshot / memory / PE lineage；
- 最终选择及置信度。

没有行动前下注，事后总结“我早就知道”不能算 Readable 或 Learnable。

### 4.4 从 PE 到下一次可见决策的闭环

外部结果继续走唯一合法的 `dialogue_external_outcome` 通道；PE owner 结算，credit 聚合，
`preference_about_other` / `intent_about_other` 与 gate 在各自边界内更新。下一次同构场景必须观察到：

- 预测改变；
- 选择动作改变；
- 新状态可跨 session 恢复；
- 相同 user 改善、swapped user 恶化或翻转；
- `evaluation` / judge 分数没有进入学习路径。

---

## 5. 四能力轴在第一版中的唯一 owner

| 能力轴 | 写入/读取/学习/行动 | 唯一 owner 与时间尺度 | 回滚 |
|---|---|---|---|
| Appendable | 关系经历、ToM 假设、修复记录、用户确认 | `memory` + `SocialRecordStore` + semantic owners；online-fast / session-medium / background-slow；既有 hydration | 禁止载入本实验 scope；owner wiring 降级 |
| Readable | 对用户的 preference/intent 假设与候选动作结果预测 | `PreferenceAboutOtherModule` / `IntentAboutOtherModule` 发布 frozen snapshot；`social_prediction` 只转发 | proposal/readout 置 SHADOW 或 DISABLED |
| Learnable | typed outcome→PE→credit→假设/策略更新 | `prediction_error`、`credit`、owner reconciler；只读 evaluation 完全隔离 | learner wiring DISABLED；恢复 owner checkpoint |
| Steerable | 在关系决策点选择 stay/space/noop 并影响表达 | `self_temporal` 拥有抽象关系动作；`steering_gate_decision` 只决定 `noop\|steer`；expression 仅执行 typed action advisory | gate→executor→sensor 逆序回 SHADOW/DISABLED |

首个包不新增 runtime slot。若实现时发现现有 `SocialPrediction` 无法携带候选动作预测，必须先在
`docs/DATA_CONTRACT.md` 注册 enriched value/依赖/wiring，再单独做契约包；禁止把 sidecar 字段偷偷塞进 consumer。

---

## 6. 数据怎么组织

### 6.1 三层物理隔离

继续服从 `lifeform-synthetic-data` 的三层契约：

1. **generator truth**：隐藏关系动力学、动作→结果转移、场景族与 counterfactual；
2. **rendered text**：只把既定事件渲染成自然语言，不能改标签或结果；
3. **runtime observation**：系统实际发布的 owner snapshot、prediction、PE、credit、action 与 hash。

### 6.2 不修改公开 trajectory 的首版做法

`companion_standard.InteractionTrajectory` 继续保存 canonical transcript 与粗粒度关系标签。
新实验增加 proprietary、content-addressed 的 decision sidecar；它是离线 evidence artifact，不是 runtime owner：

```text
RelationshipDecisionTrace
  trajectory_hash
  user_scope_hash
  scenario_family / surface_scene_id / split
  sealed_latent_dynamic_id          # 只供环境与评估，绝不进 SUT
  decision_id / pre_action_timestamp
  candidate_action_ids
  predicted_outcome_distribution
  chosen_action_id
  observed_typed_outcome
  prediction_error_ref / credit_refs
  source_snapshot_hashes
  next_state_hash
  model / prompt / weights / seed lineage
```

### 6.3 合成数据的最小矩阵

仪器校准（v1–v4 已用）下限保持：

- 至少 6 个隐藏关系动力学对，每对都存在同表面、相反最优动作；
- 至少 4 个表面场景族：家庭、亲密关系、工作、朋友；
- train / validation / held-out 按场景族与 latent sibling 整族隔离；
- 每个事实轨迹有 counterfactual：只交换用户历史，测试句保持逐字节一致；
- 每个动作都必须在校准集中被环境实际执行，证明 action→outcome 有足够效应；
- hidden label 只作评估真值，在线学习只收到 typed outcome 与 PE。

下一份资格/formal 数据集不得再靠手写静态 JSON 堆 n。必须冻结一份内容寻址的
**生成配方**（FSM 真值 + LLM 只渲染文本 + loader 不变量验收），用
`companion-trajgen` / `lifeform-synthetic-data` 对已封存 lab 契约做单向投影：

| 用途 | 最小镜像对数 | 每臂决策数 | 说明 |
|---|---:|---:|---|
| 仪器校准（已完成 v1–v4） | 6 | 24 | 只证明机器可跑，判词功效不足 |
| 下一份 development qualification | ≥24 | ≥48 | 翻转率改走 Wilson 单侧下限，见 §8.1 |
| formal secret heldout | ≥48 对，配对 probe ≥100 | ≥100 | 生成配方在看到 P1j 结果前冻结；不得看完 v4 再手写 |

v4 已物化，永远不能再当 unseen。扩 n 只能开新 package / 新 fingerprint。

### 6.4 真实用户数据

closed alpha 只接收明确同意的成年人数据，并按用途分开授权：

- 提供服务；
- 保存个人关系记忆；
- 去标识化研究分析；
- 是否允许进入训练候选集。

未单独同意训练的数据永远不进入共享权重。真实轨迹通常没有隐藏真值标签；学习依赖行动前预测与用户/环境后续 outcome，人类标注只做 validation anchor。

Lab 里 typed outcome 由反应式环境机械结算；真人的「后来怎么样」是自由文本。
**在 P4 把真实 outcome 写入 PE 结算之前**，必须另过 typing 前置门：

- 用 LLM 结构化输出（禁止关键词/正则）把用户后续话语映到既有
  `HELPED / FELT_HEARD / MISSED / OVER_DIRECTIVE / unknown`；
- 人类标注只作 validation anchor，模式对齐 `docs/specs/steering-human-anchor.md`：
  `validation_anchor_only=true`，`learning_use_authorized=false`；
- 至少 3 名独立 rater、隐藏标签、多数一致率 ≥0.80，typing 与锚点一致率达到
  预注册阈值后，才允许该通道进入 `dialogue_external_outcome` → PE；
- 未过门的样本保持 `unknown`，不脑补成功，不进入 credit。

这与 P1l 人工盲标不是同一扇门：P1l 只给 v3 合成公开文本的 condition 可判别性提供
human anchor；本条给的是真人自由文本 → typed outcome 的可靠性。两套标签都不得成为
reward。

---

## 7. Relationship Lab 的对照矩阵

所有臂固定同一 substrate 权重、generation settings、candidate action surface、反应式环境转移函数与行动前用户观察。各臂共享基础 system prompt；`prompt-steelman` 被刻意允许增加一段冻结且记录 hash/token 成本的关系推理指令，作为更强而不是更弱的对照。行动后的用户反应必须由所选动作和隐藏动力学共同产生，因此不能像旧七日脚本一样强行逐字节相同。

| Arm | 能看到什么 | 目的 |
|---|---|---|
| `stateless` | 当前 turn | 无持续状态下界 |
| `prompt-steelman` | 冻结的最强关系 prompt + 完整原始历史 + 全部 typed outcomes | 回答“把历史全塞进去行不行” |
| `rag-steelman` | 同 prompt + 标准 summary/RAG/user-model/episodic | 回答“成熟 agent memory 行不行” |
| `volvence-cold` | 完整 owner/runtime，但关闭 PE 驱动更新 | 隔离结构本身 |
| `volvence` | 完整 Appendable→Readable→Learnable→Steerable 闭环 | 被验证系统 |
| `oracle-concept` | 直接给出 sealed hidden dynamic | 非竞争上限；衡量 Volvence 离“读对概念”还有多远 |

当前 4 历史 / top-4 规模下，`rag-steelman` 的检索选择性尚未启用：候选面恰好等于全部
typed outcome，强 RAG 契约还强制必须取回全部 4 条。因此它此时不是独立的“从海量记忆
挑对东西”对照，而是 `prompt-steelman` 的重新包装（顺序由相似度而非时间决定，外加
ref-harness 记忆片段框）。P1j 已按原双主臂资格门完整跑完并原样落账；该 lane
关闭后，才允许把 RAG 版本化降级为观察臂。完整规则见 §14.1。长历史阶段检索真正有
选择性时，RAG 必须回到主对照。

“无论怎样写 prompt 都做不到”在科学上无法由有限实验证明。当前规模允许的严格口径是：

> 在预注册、冻结且经过 train/validation 调优的 prompt/full-history steelman 上，
> Volvence 在 held-out 场景族中以更少上下文获得显著更好的个体化动作选择与复发下降。

当前规模不得据此宣称“已打败标准 agent-memory / RAG 栈”。该宣称只能在历史长到
`top_k < n`、检索选择性被启用、且 RAG 重新作为主对照之后提出。

`oracle-concept` 不是 formal 才出现的摆设。P1j 关闭且冻结 Qwen 空闲后，P1k 必须在
已烧毁的 v3 training 包上跑四格正交诊断，把失败定位到「已知策略应用 / 已标证据的
个体映射归纳 / 当前 probe 条件识别 / 未标历史条件绑定」。oracle 格消耗 sealed truth，
使用专用诊断 prompt，非竞争；其正确率不得写成 P1i consumer 资格或 Volvence 优势。

prompt 候选可在 train/validation 上由人和强模型共同优化，但在 hidden test 解封前冻结；不得看到 formal 结果后换 prompt。

---

## 8. 六道门：从实验到产品

精确阈值在校准后、formal 前冻结。以下是门的语义，不允许边跑边改。

### Gate 0：仪器可判读

- 同一句测试文本在 mirrored users 中确实要求相反动作；
- candidate action 物理到达反应式环境；
- 正确/错误动作造成的 typed outcome 差达到预注册最小效应；
- hidden label、未来 outcome、judge 分数对 SUT 零泄漏；
- stateless/raw baseline 不饱和，任务难度落在可判别区间。

不过 Gate 0，不启动训练或大矩阵。

### Gate 1：Appendable

- 同用户跨 session、跨进程恢复后，owner state 与 source hash 一致；
- stateless 没有 staged state；swapped-user 明确翻转或恶化；
- memory console 能展示、纠正、删除相应关系假设；
- Volvence 上下文成本相对 full-history steelman 有明确压缩优势。

### Gate 2：Readable

- 在结果发生前正确排序 candidate action 的用户反应；
- 在 held-out 表面场景中仍能区分两种隐藏动力学；
- 当前 4 历史规模：必须优于 `prompt-steelman`，并接近 oracle-concept 上限；`rag-steelman`
  继续报数，但是观察对照，不构成资格门槛；
- 长历史阶段（`top_k < n`、检索选择性启用后）：必须重新把 `rag-steelman` 升回主对照，
  并优于它，才能宣称打败标准 agent-memory 栈；
- owner snapshot 中的命名假设与实际预测 lineage 可追溯；
- 不能仅靠当前句、姓名、人口学或场景关键词完成。

### Gate 3：Learnable

- 首次错误产生的 PE 能改变后续预测与策略；
- `volvence` 优于 `volvence-cold`、no-learning 与 shuffled-credit 控制；
- 同类错误早期→晚期显著下降；
- 学到的是跨场景结构，不是对某句原话的缓存；
- 学习图中不存在 evaluation/judge/human-rating 回灌。

### Gate 4：Steerable

- learned gate 优于 noop、always-on 与 random-gate；
- strict noop、norm cap、artifact lineage 与 reader/executor 冻结性全部成立；
- 选择动作的变化真实导致用户 outcome 变化，而非只改变内部 readout；
- 全程 SHADOW 先行，任何用户可见 ACTIVE 必须另过 promotion gate。

### Gate 5：真实产品价值

仅在 Gate 0–4 至少形成完整 directional evidence 后进入邀请制 alpha。主要看：

- 用户在关键时刻是否更愿意回来；
- 同类纠正是否减少；
- 回访是否被接受而非造成打扰；
- 用户是否主动保留关系记忆并愿意付费；
- D7/D30 留存、付费意愿和取消原因；
- boundary violation、wrong-user attribution、依赖诱导和危机处理是否守门。

这些是产品验收与只读评估，不直接成为 PE/reward。

### 8.1 统计形式：点阈值只留给已冻结的 P1j

P1–P1j 的点阈值（prompt/RAG accuracy 0.625–0.875、pair flip ≥0.5、n=8 或 12 对）
继续解释已经封存的 artifact，**不得回溯改写**。它们在 n=12 对时，即使真实参数
正中靶心，一次联合通过率也大约只有一半；v1–v4 的饱和/契约/资格振荡，有相当
比例可能是采样噪声。P1j 之后的新契约必须改成下面的形式，而不是再调一个点。

**Development qualification（P1m 起，主对照按 §14.1）**

- 最小规模：≥24 组镜像对，每臂 ≥48 个决策。
- Accuracy：点估计仍落在 `[0.625, 0.875]`（>0.875 仍判饱和，<0.625 仍判不合格）；
  另要求 Wilson 95% 单侧下限 ≥0.50，以避免「落带但 CI 盖住乱猜」。
- Pair flip：不再要求点估计 ≥0.5 过门；改为 Wilson 95% 单侧下限 >0.35。
  `structured-state` 继续用同一翻转区间，不跟 RAG 一起扔掉。
- RAG 在 4 历史 / `top_k=4` 阶段只报数，不进联合资格（§14.1）；长历史
  `top_k < n` 后升回主对照，并使用同一 CI 形式。
- 区间公式、种子、n 必须写进新契约 id，看到输出后不得改。

**Formal comparison**

「显著更好」必须在 hidden test 解封前写成检验，而不是事后看 p 值：

- 主检验：配对 McNemar，α=0.05，双侧；功效目标 80%，检测 15 个百分点的
  动作选择提升（预注册备择：steelman 0.75 vs Volvence 0.90，预估不和谐率 0.30
  → 约 100 个配对 probe）。
- 同时报告上下文 token 与复发下降；token 优势不得替代动作选择检验。
- Secret heldout 由冻结生成配方程序化产出，独立 evaluator 解封；设计者本人
  不得在看到 P1j/P1k 后再手写测试项。

---

## 9. 实施收敛包

每个包只解决一个 owner/契约/consumer，避免把新 schema、训练、runtime ACTIVE 和产品 UI 一次性绑在一起。

### P0：实验与数据契约冻结

> 实施状态（2026-08-19）：代码、场景包、sidecar、反应式环境、泄漏审计、
> Gate 0 校准器、真实 substrate stateless runner 与 prereg 模板已落地。本机冻结
> `Qwen/Qwen2.5-1.5B-Instruct` 在 current-turn-only 条件下产出 24/24 个有效
> decision、4/24 正确；六项 Gate 0 检查全部 PASS，
> `machinery_ready=true / gate0_passed=true`。这关闭的是 P0 仪器校准，不是 formal
> hidden test：模板仍为 `template_not_frozen`，仓库内 `heldout` 只是开发期结构
> 分割。进入 P1 formal 前仍须冻结完整 prereg，并另行封存 secret heldout。

交付：

- 新 Relationship Lab spec；
- `relationship_transfer_v1` scenario package；
- decision sidecar frozen dataclass / canonical JSON / hash；
- mirrored-user 场景、三动作、反应式 outcome 真值；
- Gate 0 校准器、真实 Qwen stateless 决策账本、泄漏检查与 prereg 模板。

唯一职责固定为：scenario package、generator truth、action/outcome 转移与 decision sidecar 都由 `lifeform-domain-emogpt.lab` 拥有，照抄 Coding Lab 的 domain-lab 边界；orchestrator 和只读 verdict 在 `lifeform-evolution`。Gate 2 以后如需扩母语料，由 `lifeform-synthetic-data` 对已封存 lab artifact 做单向投影，不复制第二份真值。不得把实验环境放进 `vz-*`。

### P1：强基线与 Appendable 复刻

> 实施状态（2026-08-19）：四臂 runner、`companion-ref-harness` BGE-M3 RAG、既有
> `MemoryStore` structured-state、跨全新进程恢复、per-user isolation、0/8/32 token
> scaling、console correction/delete、逐决策 checkpoint 与内容寻址 Gate 1 v2 报告均
> 已落地。真实冻结 Qwen2.5-1.5B development run 中，恢复/隔离/成本/console 四项
> PASS；但 1 条 RAG 违反 exact-one-key JSON，structured-state pair flip=0.25<0.5，
> full-history/RAG steelman 也未达到资格，最终
> `machinery_ready=false / gate1_passed=false`。这是一份 BLOCK 证据：只证明状态能存、
> 恢复、纠删和压缩，尚未证明它稳定进入个体化决策。P2 formal 保持关闭；下一包
> 只能改善 baseline/行为可判读性，禁止降阈值、放宽 parser 或提前接学习/steering。

交付：

- 复用 Coding Lab Packet 2 的 stateless / steelman / structured-state 证据形状；
- 接入 `companion-ref-harness` 作为 RAG steelman；
- 证明跨进程恢复、user swap、token scaling 与 console correction；
- 不新增学习行为。

P1 之后的实验室仪器包（P1b–P1j）编年史以 `docs/specs/relationship-lab.md` §7 / §9
为 SSOT。本方案 §14 只保留当前卡点与 P1j 关闭后的下一刀。

### P2：多经历 ToM 读出

唯一 owner：`preference_about_other`，必要时配合 `intent_about_other`，但两者保持语义隔离。

交付：

- 从 transcript + owner records + typed outcome/PE 形成 relationship-dynamics hypothesis；
- owner 发布 candidate action 的 pre-action social prediction；
- 与单 turn LLM proposal、full-history LLM 及 oracle-concept 做 held-out 对照；
- 先 SHADOW，只产 prediction/evidence，不影响表达。

P2 **formal** 仍关闭，直到 P1m 仪器资格过门。允许的唯一并行是 P2-SHADOW 防火墙
（§14.4）：只在已定为 `consumer_training_only` 的 v3 上搭 owner，标签留在
evaluator，不读 v4，不写 PE/credit/steering，不对外称 Readable。P1k 报告应用来
收窄 owner 范围（条件识别 vs 个体策略归纳），而不是给它一份新的资格分数。

### P3：PE 信用与择时控制

交付：

- `dialogue_external_outcome` 精确结算 prior prediction；
- PE→credit→gate 复用 Coding Lab Packet 3 的 no-op/always/random/oracle 结构；
- owner/reader/executor 冻结，只有 gate 按预注册更新；
- 先证明动作 NLL/选择正确性，再证明端到端 outcome 增益；两种判词不得混称。

### P4：closed-alpha 产品壳

交付：

- 三个核心入口：自然对话、次日 followup、关系记忆 console；
- 用户可查看“我现在怎样理解你”、纠正、删除、标敏感、禁止主动提及；
- 关键关系动作与安全边界采用 typed rationale/audit；
- alpha evidence 与训练候选物理分离；
- §6.4 的真实 outcome typing 前置门必须先 PASS，才允许把真人后续结果写入
  `dialogue_external_outcome` → PE。未过门只采集、不计学习。

### P5：慢速共享关系基底

只有 P2/P3 和真实 alpha 同时给出正证据后才启动：

- 跨人共享的关系动力学进入 offline encoder/adapter/rare-heavy artifact；
- per-user 事实、轨迹和偏差永远留在私有 owner state；
- shared update 必须经过 `ModificationGate.OFFLINE`、held-out、回滚和数据授权审计；
- `companion-encoder` 的 G2 未过前，不发布权重，不称“关系大模型已成立”。

---

## 10. 首版真实产品体验

首版只需要三个连续动作，已经足以让关系产生：

### 10.1 当下陪伴

用户讲一件真实发生的事。系统不固定“共情后建议”，而是根据当前状态和共同历史选择关系动作。

### 10.2 现实结果回收

用户回来时，系统询问或接收“后来怎么样”；有明确结果就通过 typed outcome 结算，没有结果就保持 unknown，不脑补成功。

### 10.3 有边界的回访

对用户明确允许跟进的 open loop，在合适时间主动回来；用户拒绝、忽略或撤回同意后立即停止。回访不是拉活工具，而是承诺与关系连续性的行动。

产品界面暂不需要展示复杂内部图，只展示：

- “我现在可能理解到的是……”；
- “这来自哪些共同经历”；
- “保留 / 只限本次 / 改写 / 删除 / 不要主动提”。

---

## 11. 训练顺序

### 11.1 现在

冻结一个能力强的 open-weight substrate，用现有 owner/memory/controller 做系统学习。先证明架构可以从经历中形成差异，避免把效果误归因于换模型或大规模微调。

### 11.2 Relationship Lab 过门后

训练两类小而明确的 artifact：

1. **关系动力学 readout**：轨迹前缀→candidate action outcome prediction / typed ToM proposal；
2. **关系动作 controller**：在冻结 readout/executor 上学何时 stay、space 或 noop。

### 11.3 真实 alpha 有数据后

以 consented first-party 轨迹扩展 synthetic prior；离线学习跨用户共享规律。训练样本必须保留：

```text
观察 → 行动前预测 → 关系动作 → 外部结果 → PE → credit → 后续迁移结果
```

只有 transcript、没有行动与结果的对话，不足以训练本产品的核心能力。

---

## 12. 安全与产品边界

必须从第一版进入硬门，而不是上线前补免责声明：

- 仅成年人 closed alpha；
- 明确告知对方是 AI；
- 不宣称诊断、治疗、读心或替代现实关系；
- 不使用“只有我懂你”“不要离开我”等排他性策略；
- 自伤/他伤等高风险进入专门安全协议和现实支持转介；
- 关系记忆默认可见、可纠正、可撤回、可删除；
- 敏感数据与训练授权分开；
- 所有主动 followup 受 consent、cooldown、budget 与停止指令约束；
- retention 指标不得进入关系策略 reward，防止把依赖误学成成功。

---

## 13. 止损条件

以下任何一条成立，都应收缩主张，而不是继续换 prompt、调 judge 或扩大数据追分：

1. Gate 0 证明 action 无法可靠影响环境 outcome：先修环境，不跑模型；
2. prompt/full-history steelman 在 held-out 上与 Volvence 等效，且成本可接受：暂时把产品定位收缩为优秀关系 memory wrapper；
3. 多经历 readout 不优于单 turn LLM proposal：停止新增 latent carrier，先修 owner 输入与标签定义；
4. learned gate 不优于 noop/random：不授权 ACTIVE，保留 Appendable/Readable 局部价值；
5. synthetic 正结果无法迁移到 consenting real-user directional pilot：不训练共享关系权重；
6. 产品留存主要来自排他性依赖而非现实帮助：立即停止相应策略与增长实验；
7. **校准元循环止损**：数据集版本（当前 v1–v4 共 4 版）或 consumer freeze
   （当前 P1g/P1i 共 2 次）再增加时，不得再用「换场景 / 换 prompt / 换点阈值」
   追资格。允许的唯一下一步是 §8.1 的仪器协议升级（更大 n + CI 门 + 冻结生成
   配方）。该升级后再失败一次 development qualification，即停止场景版本化：
   按 P1k 定位升级 substrate / owner 输入，或把产品主张收缩为关系 memory
   wrapper。禁止为过门再开 v5/v6。

---

## 14. 当前唯一优先级

包级编年史的 SSOT 是 `docs/specs/relationship-lab.md` §7 / §9，本方案不再逐包复述。
本节只回答：现在卡在哪、P1j 关闭后允许改什么、下一刀对准谁。

### 14.0 当前状态

| 项 | 状态 |
|---|---|
| P1j | 2026-08-21 同一 attempt 完成 72/72：report `e9226ee8…fd78` 判 `consumer_failed_v4_qualification`。prompt/RAG/structured accuracy 为 0.542/0.500/0.542，pair flip 为 0.417/0.250/0.417；三臂 strict-valid=1.0。完成态 strict resume 未再调用 Qwen。 |
| P1k-R1 | protocol `204e0904…64bd` 已绑定 P1j 失败终局，在 P1k Qwen output=0 时冻结四格/48 条最大计划与 staged gate；目录无 `records/`，重复 prepare-only 保持 protocol/checkpoint/preflight 文件 hash 不变。下一包只执行 A 格。 |
| 已证明 | Gate 0 仪器；跨进程恢复 / 隔离 / 纠删 / 压缩；v2/v3 反捷径；v3 public-evidence BGE 60/60 可判别；P1g prompt 单臂入带；P1j one-shot/no-feedback/严格恢复机器按契约完成并诚实保存 unseen failure |
| 未证明 | 完整 consumer qualification、human anchor、Volvence advantage、Readable / Learnable / Steerable、formal、四能力 |
| 校准消耗 | 数据集 v1–v4 共 4 版；consumer freeze 2 次（P1g / P1i）。再版本化场景或再搜 prompt 受 §13.7 约束 |

P1j 已释放同一冻结 Qwen。先完成 P1k-R1 的 **zero-output protocol freeze**；随后下一份
占用 Qwen 的动作只能是 P1k 首格 `oracle_policy_apply_v2`，用已烧毁 v3 定位失败层。
不得返回 P1i 搜第四个 prompt，也不得开 v5。P1l 人工盲标不加载模型，可继续独立收集
rater 结果。

P1j 终局之后的顺序冻结为：

1. 原样封存报告。不得返回 P1i 改 consumer / prompt / RAG / gate。
2. 若 `saturated` 或 `machinery regression`：§14.1 不适用，按原分叉停。
3. 实际命中 `underqualified`：§14.1 的 RAG 观察臂修约不能挽救本 ledger，因为 prompt
   accuracy 0.542 与 structured flip 0.417 自身也未过门。**下一步不是再造 v5**，而是
   P1k 定位失败环节，再 P1m 升级仪器（n + CI + 生成配方）。
4. 若旧双主臂门 `development qualified`：只授权另包冻结 formal comparison prereg **候选**；secret heldout 仍须走 §8.1 功效与生成配方，不直接开 P2 / formal。

### 14.1 P1j 之后：RAG 降级为观察臂的规则

P1g 之后曾经禁止“看到坏结果就删除 RAG、改成 best-arm、降低阈值”。P1j 已用冻结的
prompt/RAG 双主臂门跑完并原样封存；运行中没有改门、删臂或回头调 P1i。

P1j 关闭后，才允许一次**新契约 id 的版本化修约**。允许修约的结构性理由不是“RAG 难
打”，而是当前考卷规模下 RAG 不再是独立对照：

- 每用户只有 4 条 typed outcome 历史，RAG 固定 `top_k=4`，强 RAG 契约还要求 4 条必须
  全部取回；检索选择性 = 0。
- 同一 3B 基底在时间顺序的 prompt-steelman 上已经入带（P1g：0.75 acc / 0.50 flip）；
  公开文本也已被冻结语义审计读出（P1f：60/60）。因此失败不在考题无解，而在
  “同样 4 条证据被打成记忆碎片后，3B 读不出镜像翻转”。
- 继续把这个劣化包装版留在资格门里，只会无限卡在 consumer 调参，而正式对比真正要
  打赢的上界对手是 `prompt-steelman`。

修约必须同时满足下面全部条款，缺一条就不算合法降级：

1. **时机**：只发生在 P1j 终局报告落盘之后。P1j 本身继续按双主臂门记账。
2. **形式是降级，不是删除**：`rag-steelman` 继续跑、继续报 accuracy / pair flip / token，
   只从 `primary_qualification_arms` 移到观察臂。禁止从对照矩阵里抹掉它，也禁止改成
   “取最好一臂算过”。
3. **新资格门**：主对照只认 `prompt-steelman`（accuracy 仍在 0.625–0.875，pair flip
   ≥0.5，strict valid=1.0）。`structured-state` pair flip ≥0.5 **必须保留**——这是自家
   类型化状态能否因人而异的存在证明，不是对手门槛，不能跟 RAG 一起扔掉。
   这三项点阈值只用于一次性再读已经封存的 P1j n=12 ledger，不替代 §8.1 / P1m。
4. **主张收缩**：此规模只能说“打败全历史 prompt steelman”。不能再说“打败标准
   agent-memory / RAG 栈”，也不能把观察臂的低分写成 Volvence 优势。
5. **RAG 必须回来的条件**：当历史长到全文塞不进 prompt、或 `top_k < n` 使检索真正有
   选择性时，RAG 自动升回主对照；那时 Gate 2 恢复“优于 prompt 与 RAG”。不得把当前
   规模的观察降级偷偷带进长历史 formal。
6. **修约后的下一刀对准谁**：
   - prompt 入带且 structured flip ≥0.5 → 只说明 n=12 的点估计不再被 RAG 卡住；
     **仍须走 P1k 定位 + P1m 扩 n / CI 门**，不得把这份小样本直接冻成 formal
     prereg。P2 / secret heldout 仍不自动打开。
   - prompt 入带但 structured flip <0.5 → 停 consumer lane，去修 structured-state /
     owner 输入表示，不新增 latent carrier、PE 或 steering。
   - prompt 自身掉出资格带或饱和 → 不得靠降 RAG 掩盖；下一步是 P1k 定位，不是
     再开 v5。版本化更难场景只允许发生在 P1m 生成配方里，且计入 §13.7。
   - P1j 若判 saturated 或 machinery regression → 本条修约不适用，按原分叉停。

禁止项保持不变：回头改 P1i consumer、在已见 split 上再搜一轮 prompt、降低阈值、把
evaluation / RAG 失败分数回灌 PE/credit/reward/steering。P1i 训练集上 structured flip
只有 0.333，所以去掉 RAG 不等于资格自动过；修约只是把锁从“陪练包装”挪开，露出
真正还没过的那扇门。

权威 P1j 已给出这个边界：prompt accuracy 0.542 低于 0.625，structured pair flip 0.417
低于 0.5。因此本 ledger 即使把 RAG 降为观察臂也仍然 FAIL；禁止发布一份“去掉 RAG 后
通过”的派生资格报告。§14.1 只约束后续协议结构，当前执行分叉直接进入 §14.3 P1k。

### 14.2 仪器协议升级（P1m）

P1m 是 P1j 关闭后的契约包，不跑新的 consumer search，不改 P1i consumer，不把
evaluation 回灌 PE。它只冻结下一份资格门的统计形式与数据生产协议：

- 规模：development qualification ≥24 组镜像对、每臂 ≥48 决策；formal secret
  heldout 配对 probe ≥100。用冻结生成配方扩 n，禁止手写静态 JSON 追分。
- 资格：主对照 `prompt-steelman`；accuracy 点估计仍在 `[0.625, 0.875]`，且
  Wilson 95% 单侧下限 ≥0.50；pair flip 改为 Wilson 95% 单侧下限 >0.35。
  `structured-state` 用同一翻转区间。RAG 在 4 历史阶段按 §14.1 只报数。
- 生成配方：内容寻址的 FSM 真值 + LLM 只渲染 + loader 不变量。结构不变量**现在**
  就可以冻结（沿用 v3/v4 反捷径 + v3「事件 + 关系损失」公开语言，只把 n 扩到
  ≥24），不得等 P1j 的 accuracy / flip 出来再拧难度旋钮。看完 v4 再手写测试项
  视为污染，必须作废。
- 若 P1m 协议冻结后的第一次 qualification 仍失败：触发 §13.7，停止场景版本化。

P1m 可以与 P1k 错开执行：P1k 只消费已烧毁的 v3，不需要新场景；P1m 的生成配方
应在 P1k 判词出来后收窄「扩什么」（条件可判别性 vs 个体映射证据），但配方本身
不得等 P1j 的资格分数再改难度。

### 14.3 oracle-concept 四格诊断（P1k-R1）

P1k 的唯一 owner 是 `lifeform-evolution.relationship_lab_packet1k`。它是
evaluator-only 四格诊断，不是竞争臂，也不是资格门。其专用 prompt 只允许主张
“冻结 substrate + 诊断仪器能否完成该格”，不得偷换成“冻结 P1i consumer 能完成”。

- 输入：已烧毁的 `relationship_transfer_v3`（`consumer_training_only`）、冻结 P1i
  lineage 下的 prompt-steelman 公开表面，以及权威 P1j underqualification protocol /
  report。不得物化 v4，不得修改 P1h gate。
- 最大计划冻结四格：A `oracle_policy_apply_v2`（全披露）；B
  `oracle_policy_induction_v2`（隐藏策略）；C `oracle_probe_recognition_v2`（隐藏当前
  条件）；D `oracle_history_binding_v2`（隐藏历史绑定与策略）。A 先跑；A 失败停 12；
  B invalid 停 24；B valid 后运行独立 C；B non-functional 时停 36 并跳过 D；只有 B
  functional 才运行 D，最多 48。
- 任一带 sealed truth 的正确率都是诊断天花板，不得写成 consumer qualified、
  Volvence advantage 或 Readable。
- 派生判词只定位瓶颈：machinery 回归 / 基底无法应用已披露策略 / 无法从标注证据
  归纳策略 / 无法识别 probe 条件 / 无法绑定未标历史 / 多重瓶颈 / 矩阵完整则缺口在
  无辅助抽象或迁移。六组镜像对 × 单 seed 仅作 directional owner triage。
- 执行时机：P1j 释放同一冻结 Qwen 之后。不得为了抢跑而中断 P1j。
- 2026-08-21 P1k-R1 代码已落地：`relationship_lab_packet1k.py` +
  `scripts/run_relationship_lab_packet1k.py`。先 `--prepare-only` 在 P1k Qwen output=0
  冻结 48 条最大计划与 staged gate；真实输出不得进入本收敛包。机械测试覆盖 P1j
  前置、四格披露、12/24/36/48 早停、单/多瓶颈、resume 与 terminal tamper。权威
  protocol `204e0904…64bd` 已按此边界完成 freeze，当前无 `records/`。

P1f 遗留的人工盲标 packet 也已落地（`relationship_lab_packet1l.py`），**不是**
§14.2 仪器升级。它不加载 Qwen，可与 P1j 并行 `--prepare-only` 分发 rater CSV。
仪器升级的包号是 **P1m**，实施时必须使用新模块名，不得覆盖盲标包。

### 14.4 P2-SHADOW 防火墙

P2 formal 继续关闭。允许的并行只覆盖 owner 脚手架，且必须同时成立：

- 只读 v3 `RelationshipConsumerTrainingView`；v4 observation / truth 对 P2
  开发者保持不可见。
- 独立 `P2-development` split 契约；training label 只留在 evaluator。
- `WiringLevel.SHADOW`：只发 prediction / hypothesis snapshot，不进 expression、
  不写 PE / credit / steering。
- P1k 报告应用来选择先做条件识别还是个体策略 owner，而不是给 SHADOW 打分过门。
- 退出：删掉 SHADOW wiring 与离线 adapter 即回滚；已发布 snapshot schema 若已
  注册必须保留 hash，不得把 SHADOW 输出写进正式资格报告。

在 P1m 仪器资格过门之前，任何 P2-SHADOW 数字都不得对外称为 Readable 证据。
