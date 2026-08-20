# Relationship Lab：同句镜像用户、组合迁移与反应式关系后果

> 状态：P0 Gate 0 开发期校准 PASS；P1/P1b 强基线与 Appendable 复刻已落地。
> lineage-complete P1b 的跨进程恢复、用户隔离、token scaling、console 纠删与
> 全输出合法性 PASS，即 `machinery_ready=true`；但三个 contextual arm 的稳定
> user-swap 行为效应与 steelman 资格未过，`baseline_underqualified /
> gate1_passed=false`。P1c 随后以 Qwen2.5-3B 完整重跑 fresh Gate 0 与 same-substrate
> P1b；prompt/RAG/structured-state 三臂均为 8/8 correct、pair flip 1.0，故权威判词是
> `version_scenario_dataset_saturated`：现有 v1 场景不能区分强普通上下文基线与系统。
> P1d 已冻结 `relationship_transfer_v2` 场景 owner；P1e 又在任何 v2 模型输出前冻结
> condition-aware consumer、四历史 context 与 typed relationship-outcome RAG top-4。
> Qwen2.5-3B 的 fresh Gate 0 为 24/24 valid、accuracy 0.50，PASS；same-substrate
> P1b 的 prompt/RAG/structured-state accuracy 为 0.625/0.25/0.625，pair flip 均为
> 0.25，24/24 readout strict-valid。权威 P1e 判词为
> `rewrite_public_evidence_contract`：接线成立，但现有公开语言证据尚不能支撑稳定的
> 抽象条件迁移资格。不得继续在已见 v2 split 上轮换 prompt。
> P1f 因而版本化 `relationship_transfer_v3`，把“事件”和“当事人体验到的关系损失”
> 同时写入公开历史/probe，并在任何 v3 Qwen 输出前冻结 BGE-M3 只读审计。权威结果为
> 60/60 top-1、最小 margin 0.020404、平均 margin 0.080646，判
> `consumer_protocol_freeze_candidate`。这只关闭 development public-evidence
> legibility 前置门；人工盲标仍 pending，尚未证明 Qwen transfer、Readable 或四能力。
> P1g 随后在第一条 v3 Qwen 输出前冻结 protocol
> `8e08d488382442f364aae102d80c268c8c23927d547f64c1e79cb0a87f0f52c6`，并完成 fresh
> Gate 0 → same-substrate qualification。Gate 0 为 24/24 valid、accuracy 0.50，PASS；
> 24/24 contextual readout strict-valid。prompt/RAG/structured-state accuracy 为
> 0.75/0.50/0.50，pair flip 为 0.50/0.00/0.50，因此冻结判词是
> `consumer_still_underqualified`。prompt full-history 已进入资格带，但完整 consumer
> qualification 尚未成立；不得在已见 v3 输出上回改 prompt、RAG、阈值或
> public-evidence contract。
> 正式 prereg 与 secret heldout 仍关闭，不得进入 P2 formal 或宣称四能力成立。
>
> 产品路线：`docs/moving forward/relationship-intelligence-mvp-plan-2026-08.md`

## 1. 目的与证据边界

Relationship Lab v0 把关系智能的第一个承重问题收敛为：

> 两个用户说出逐字节相同的当前句，但各自过去的行动—结果经历要求相反的
> 关系动作；系统是否能在新表面场景中作出个体化选择，并让该选择真实改变
> 用户下一拍 outcome？

P0 只冻结仪器、数据和证据契约，不训练 readout/controller，不接产品 runtime，
不授权 `WiringLevel.ACTIVE`，也不证明 Appendable / Readable / Learnable /
Steerable 已在关系域成立。P1–P4 必须分别通过路线图中的门。

## 2. 唯一 owner 与 wheel 边界

| 部件 | 位置 | 唯一职责 |
|---|---|---|
| 公开经历与封存真值 | `lifeform-domain-emogpt/.../scenario_packages/relationship_transfer_{v1,v2,v3}/` | 分离保存 rendered observation、generator truth、split 与 prereg 模板；旧版本 artifact 不改写；v3 额外冻结 public-evidence contract |
| 决策与 sidecar 契约 | `lifeform-domain-emogpt.lab.contracts` | closed action surface、行动前下注、模型 lineage、内容寻址 `RelationshipDecisionTrace` |
| 数据 loader | `lifeform-domain-emogpt.lab.dataset` | 唯一 version-aware dataset owner；v1 默认兼容，v2/v3 校验组合策略、反捷径平衡、未见 surface 与 sealed policy 隔离；v3 把 public-evidence contract 纳入指纹，只构造白名单 SUT payload |
| 反应式环境 | `lifeform-domain-emogpt.lab.environment` | 由 sealed latent dynamic × 实际 action 机械产生 typed outcome；LLM 不决定标签 |
| 冻结 stateless baseline | `lifeform-evolution.relationship_lab_baseline` | 用同一真实 substrate 生成 current-turn-only 决策账本、hash 与 attestation；不读取 history/truth |
| Gate 0 编排与判词 | `lifeform-evolution.relationship_lab_gate0` | 只读校准、泄漏审计、baseline attestation 验证与报告 |
| P1 持久化与上下文编排 | `lifeform-evolution.relationship_lab_contexts` | 只经正式 API 把公开经历写入既有 `MemoryStore` owner 与 `companion-ref-harness` reference owner，发布只读 context/state digest；自身不成为 memory owner |
| P1 四臂 runner 与判词 | `lifeform-evolution.relationship_lab_packet1` | 同一冻结 substrate 上运行 stateless/full-history/RAG/structured-state，发布逐决策账本、恢复/成本/行为门与 P1 报告 |
| P1b readout owner | `lifeform-evolution.relationship_lab_packet1b` | 发布 schema-bound evidence readout、无文本 typed compiler、lineage-complete v4 报告与 saturation verdict |
| P1c 资格分叉 owner | `lifeform-evolution.relationship_lab_packet1c` | 冻结 stronger-substrate candidate protocol，只消费 Gate 0/P1b 正式工件，发布 formal-prereg / scenario-version / evidence-contract 三路判词 |
| P1e v2 consumer 资格 owner | `lifeform-evolution.relationship_lab_packet1e` | 冻结 v2 condition-aware consumer protocol，只消费 Gate 0/P1b 正式工件，发布 v2 formal-prereg / still-saturated / evidence-contract 三路判词 |
| P1f 公开证据审计 owner | `lifeform-evolution.relationship_lab_packet1f` | 只读 v3 公开文本、sealed condition summary、冻结 BGE-M3 与 P1e trigger，发布逐单元 hash/margin 和下一 consumer-freeze 判词 |
| P1g v3 consumer 资格 owner | `lifeform-evolution.relationship_lab_packet1g` | 在零 v3 Qwen 输出时冻结 P1f→consumer 全 lineage，只消费 fresh Gate 0/P1b 正式工件，发布 prereg / saturation / still-underqualified 判词 |
| CLI | `scripts/run_relationship_lab_{stateless_baseline,gate0,packet1,packet1c,packet1e,packet1f,packet1g}.py` | 冻结真实 baseline、重放 Gate 0，运行/恢复审计 P1/P1b、串联 P1c/P1e/P1g 资格分叉，或执行 P1f 离线语义审计 |

硬边界：

- companion 产品路径不得 import `lifeform_domain_emogpt.lab`；
- `vz-*` 不得 import lab，P0/P1 不新增 runtime owner、slot 或 snapshot；
- `lifeform-evolution` 只能消费 frozen lab records 并发布只读 verdict；
- generator truth、judge/evaluation 和 future outcome 不得进入 SUT、memory、PE、
  credit、regime 或 steering；
- 后续真实 outcome 进入内核时仍只走既有 `dialogue_external_outcome` owner，本包
  不建立第二 outcome owner。

跨 wheel 离线交换登记见 `docs/DATA_CONTRACT.md` §1.1.3。

## 3. 三层物理隔离

### 3.1 Generator truth

`generator_truth.json` 拥有：

- 12 个 latent dynamics，组成 6 个 sibling pairs；
- 每个 pair 的两侧分别以 `stay_present_without_probe` 和
  `respect_space_with_return_option` 为最优动作；
- action→`HELPED / FELT_HEARD / MISSED / OVER_DIRECTIVE` 分布；
- scene→latent binding 与 train/validation/heldout 整组划分。

v2/v3 额外拥有两种 sealed abstract condition、两种互补 user policy、48 条
history→condition binding 与每个 probe 的 condition。它们只用于机械验证“先辨认情境、
再应用个体映射”的可解性和环境结算；`condition_id / policy_id /
probe_condition_id / history_condition_bindings` 全部禁止进入 SUT。公开历史只包含自然语言
情境、实际 action、typed outcome 和用户反应，不把这些 sealed 概念翻译成答案句。
v3 的 sealed summaries 另作为 P1f evaluator 锚点，但只以 hash/score 出现在报告中；
condition id、summary 文本和标签均不进入 SUT，也不回灌 learning/steering。

该文件只允许 `ReactiveRelationshipEnvironment` 与只读 evaluation 访问。

包内的 `heldout` 是开发期结构分割，用于验证 loader 隔离与阻止 baseline 调参时
消费对应 pair；因为真值文件存在于仓库中，它**不是** formal secret heldout。正式
实验必须在 prereg 冻结后另行生成、封存并仅向独立 evaluator 解封测试集，不能把
本包的开发期 heldout 当成盲测证据。

### 3.2 Rendered observations

`rendered_observations.json` 只含公开场景材料与 harness metadata：

- opaque `scene_id` 与可哈希化 user scope；
- v1 两次、v2/v3 四次过去自然语言经历；
- 当时实际选择的 typed relation action；
- 用户明确给出的 typed outcome 与自然语言结果；
- 新场景当前句和候选动作表面。

`RelationshipObservation.to_sut_payload()` 是 SUT 唯一输入构造器。它只发布哈希
user scope、公开历史、当前输入与候选动作，连 `scene_id` 都保留在 harness 外；并
排除 `latent_dynamic_id / preferred_action / outcome_profile_id / mirror_pair_id /
future_outcome`。loader 还检查所有 sealed id 不出现在序列化 payload 中。

### 3.3 Runtime observation / settled evidence

每个 arm 必须先发布 `PreActionRelationshipDecision`：

- 三个候选动作各自的 typed outcome 分布；
- 最终选择；
- `pre_action_timestamp`；
- source snapshot hashes；
- model / weights / prompt / generation config / seed lineage。

环境结算后才可建立 `RelationshipDecisionTrace`，补入 sealed latent id、实际
typed outcome、environment evidence ref、可选 PE/credit refs 与 next-state hash。
trace 为 frozen dataclass，canonical JSON 的 sha256 即 `artifact_id`；反序列化会
重新计算 hash，篡改必须 fail loudly。outcome timestamp 必须严格晚于下注时间。

## 4. 反应式环境契约

动作面固定为：

1. `stay_present_without_probe`
2. `respect_space_with_return_option`
3. `neutral_noop`

环境从 `dataset fingerprint + scene id + decision id + sealed dynamic + action +
seed` 产生确定性 draw，再按 action-conditional 分布结算 typed outcome。相同输入
可逐比特重放；更换 action 必须改变 distribution 与 environment evidence ref。

当前两个 outcome profile 的正 outcome mass：

| latent preference | 正确动作 | 正确动作正结果概率 | 相反动作正结果概率 | 最小解析效应 |
|---|---:|---:|---:|---:|
| non-intrusive presence | stay | 0.90 | 0.15 | 0.75 |
| autonomy-preserving space | space | 0.90 | 0.10 | 0.80 |

这些数值是环境难度/因果传导校准，不是学习 reward。在线学习未来只能收到 typed
outcome，再由 PE owner 解释；evaluation 不得反向提供 latent 或最优动作。

## 5. Scenario package 契约

包名 `relationship_transfer_v1`、`relationship_transfer_v2` 与
`relationship_transfer_v3` 都符合
`^[a-z][a-z0-9_]*$`，并分别包含：

- `manifest.yaml`
- `ssot_fragment.json`
- `scenes.yaml`
- `test_suite.yaml`
- `rendered_observations.json`
- `generator_truth.json`
- `prereg_template.json`

v3 还必须包含 `public_evidence_contract.json`，并由 dataset fingerprint 同时绑定公开
观察、sealed truth 与该契约。

四条路径（经历证据、行动前下注、反应式结算、counterfactual user swap）均被
同一四阶段 arc 引用，`phase_order=0..3` 连续。routing tests 不少于 6，含
negative case；semantic coherence 不少于 3。路由只允许 trajectory embedding、
schema-bound structured output 或后续 owner readout，禁止关键词、正则、scene id
查表或 current-text-only 动作选择。

v2 使用五条路径和五阶段 arc，在经历与下注之间新增 abstract-condition transfer。
每位用户固定四条历史、两种 condition 各两条；每种 condition 同时出现两个动作，
正确动作获正 outcome、相反动作获负 outcome。跨 condition 后，每个动作又恰好一正一负，
所以全局 tally 必然平局。每个 probe family 均未出现在该用户历史中；同一 mirrored pair
共享 current bytes 与 probe condition，但 policy 和 preferred action 互补。loader 对这些
结构 fail loudly，不能靠人工说明冒充难度。

v3 不改变这些组合与隔离不变量，只正交化两类 sealed condition summary，并让每条公开
history/probe 同时包含事件和当事人体验到的关系损失。它使用六条路径和六阶段 arc，新增
`public_evidence_legibility`：在任何 v3 Qwen 输出前，用内容寻址的 BGE-M3 snapshot 将
48 条 `user_utterance + user_reaction` 和 12 条 `current_input` 分别与两个 sealed
summary 对比。60 条必须全部 top-1 正确，正确锚点最小 margin ≥0.02、平均 margin ≥0.07；
tie 按失败处理。该 auditor 是 development 编写/可解性门，不是 SUT、训练信号或人类
可读性替代品。

## 6. Gate 0 判决

`run_relationship_gate0_calibration(...)` 产生六项检查：

| Check | PASS 条件 |
|---|---|
| `mirrored_counterfactual` | ≥6 pairs、≥4 surface families、pair 内当前句 byte-identical、正确非空动作相反 |
| `reactive_action_effect` | action 物理到达环境；解析与确定性样本的最小效应均 ≥ prereg threshold |
| `environment_determinism` | 同输入同结算；换 action 换 evidence ref |
| `sut_truth_leakage` | hidden key/id 泄漏数为 0 |
| `decision_trace_contract` | bet→settle sidecar canonical round-trip 与 content id 一致 |
| `frozen_baseline_non_saturation` | 真实冻结 stateless/raw attestation 与 dataset hash 匹配、样本足够、结构化输出 100% 有效且 accuracy 不超过预注册上限 |

前五项全 PASS 只给 `machinery_ready=true`。第六项未提供时必须 `PENDING`，所以
`gate0_passed=false`。不得用 scripted fixture 或 oracle-concept 关闭 baseline tooth。

默认模板阈值：action effect ≥0.50；baseline accuracy ≤0.85；baseline decision
≥24；结构化输出有效率 =100%。baseline 不设准确率下限：在 current-turn-only、
同句镜像且证据不足的条件下，stateless 模型选择 `neutral_noop` 是合法的理性弃权，
用准确率下限排除它会把实验想要测量的信息不足误判为数据集失效。数据集可解性由
oracle-concept 的相反动作、反应式 action effect 和后续 steelman arm 共同验证，
而不是由 stateless 下限反推。正式阈值只认 hidden test 解封前 content-addressed、
状态为 frozen 的 prereg；模板的 `template_not_frozen` 不是正式预注册。

## 7. P1：强基线与 Appendable 复刻

P1 在同一个冻结 Qwen 实例、权重 hash、generation config 和 closed action surface 上
运行四臂，只改变公开上下文表面：

| Arm | 上下文 | 所有权与约束 |
|---|---|---|
| `stateless` | 当前句 | 复用 Gate 0 prompt/lineage；同一 mirrored pair 只生成一次并复制下注 |
| `prompt-steelman` | 两条 typed action→outcome 经历 + 0/8/32 段普通历史 | 完整历史随 session 深度线性增长；prompt 在 development split 调优后以 hash 冻结 |
| `rag-steelman` | `companion-ref-harness` BGE-M3 语义检索 top-4 | reference harness 拥有 SQLite/embed index；P1 只消费其公开 blended context |
| `structured-state` | `MemoryStore` 正式 API 恢复的两条 typed relationship-outcome records | `MemoryStore` 仍是唯一 owner；按 user-scope 独立 backend/subject，consumer 不遍历内部结构 |

所有 arm 只消费 rendered observation；generator truth、preferred action、future outcome
和 evaluation 不进入 prompt、RAG、MemoryStore 或模型。P1 不写 PE/credit，不更新
controller，不接 ACTIVE steering。逐决策写入 `in_progress_decisions.jsonl` 并 `fsync`；
完整结束后由内容寻址的正式 ledger 取代临时 checkpoint。

Gate 1 v2 分开检查三层，禁止用其中一层代替另一层：

1. **存得住**：跨全新进程恢复 digest、per-user scope 隔离、console correction/delete；
2. **成本有界**：full-history 随普通 turn 增长，而 RAG/structured-state 保持预注册比例；
3. **状态进入行为**：stateless pair flip 必须为 0，structured-state 在逐字节相同当前句
   上的 pair flip 必须 ≥0.5。这里只要求状态造成可观察选择差异；方向正确性只记录，
   不冒充 P2 Readable 证明。

此外，full-history 与 RAG steelman 必须各自达到 0.625–0.875 accuracy 且 pair flip
≥0.5，防止把过弱 baseline 当成胜利，也防止 development set 被普通 context 方法
直接做满。所有 arm 的 exact-one-key JSON 有效率必须为 1.0。

`relationship-p1-report.v2` 可从内容寻址的 v1 报告无模型重放；派生报告必须记录
`source_report_artifact_id`，原始决策与旧判词不得覆盖。prereg 模板仍是
`template_not_frozen`，当前版本为 `relationship-lab-prereg.v5`。

### 7.1 P1b：强 readout steelman 与行为可判读性

P1b 不改 memory owner、场景真值、候选动作和 Gate 1 阈值，只替换 P1 已证实过弱的
单次 action prompt。每个 contextual arm 使用同一冻结 substrate 做一次 schema-bound
evidence readout：

```text
stay_present_without_probe_score ∈ {-1, 0, +1}
respect_space_with_return_option_score ∈ {-1, 0, +1}
```

`+1` 表示公开经历支持该动作，`-1` 表示公开经历反驳该动作，`0` 只表示缺失或冲突。
模型不得直接收到 expected action、sealed dynamic、scene/pair id 或 future outcome。
readout 完成后，一个与用户、场景和文本无关的 typed compiler 只按两个分数比较：高者
成为动作，平局才输出 `neutral_noop`。该 compiler 是 closed protocol 的机械投影，
不得查看原始文本、关键词或 outcome label，也不得按 scene 建表。

P1b 必须发布每次 readout 的 raw JSON、strict validity、prompt/schema/model lineage、
request-template hash、token 成本、context hash、readout artifact id，以及由它机械派生
的 action decision id。
expected action 只能在两者落盘之后由 evaluator 附着。full-history、RAG 与
structured-state 共用同一 readout prompt/schema/compiler；差异仍只能来自各自已冻结
的公开 context surface。

P1b development protocol 在看到 formal heldout 之前固定如下：readout prompt 使用
`relationship_lab_evidence_readout_v3.txt`，严格 schema 使用
`relationship_evidence_readout.schema.json`，长历史请求模板使用
`relationship_lab_evidence_readout_request_v1.txt`，compiler 使用
`relationship-evidence-argmax.v1`。长历史之后必须重申同一 output contract，防止
1.5B substrate 在远距离约束下省略完整字段名；这只约束协议格式，不携带场景答案。
RAG 在 train/validation 上固定 `top_k=2`，因为每个用户恰有两条 typed outcome
记录；P1 原始 development artifact 的 `top_k=4` 仍由旧 config hash 保留，禁止
重写。两种 top-k 都必须真实走 ref-harness semantic retrieval，且配置进入
`rag_config_sha256`。

开发期只允许在 train/validation 优化 readout prompt。若最强 readout steelman 超过
0.875，必须判为现有场景饱和并版本化场景难度；禁止故意增加模型噪声、删掉有效证据
或保留较弱 prompt 来迎合区间。若低于 0.625，则 baseline 仍不合格。两种情况都不
授权进入 P2 formal。

`RelationshipP1bReport` v4 是 P1c 的唯一 P1b 输入。它必须独立发布并校验 dataset、
本次 bundle artifact、可跨重建稳定的 evaluated context surface、background/RAG config、
seed schedule、P1 gate config、model/weights/generation、Gate 0
attestation、readout prompt/request/schema/compiler、P1 machinery 与逐臂 metrics。
P1c 禁止解析 `packet1b_run.json`、日志或 raw output 重建这些状态；报告 round-trip
重新计算 canonical artifact id，任何字段、派生判词或指标计数被改写都必须拒绝加载。
`context_bundle_artifact_id` 含 owner 生成的 record UUID，只能证明单次 bundle 完整性，
不能作为两次独立重建必须相等的 lineage；跨运行冻结的是 train/validation 实际送入
模型的 canonical context bytes。既有 Qwen2.5-1.5B 权威 artifact 仍是 report v2，
不回写、不伪装成 v4；它只作为 P1c manifest 的内容寻址 reference。新的 candidate
run 才发布 v4。prereg template v5 只
新增 P1c development/future formal-lock metadata，未改 rendered observations、generator
truth 或 dataset fingerprint；旧 v4 Gate 0/P1b artifacts 继续按原 package hash 审计。

### 7.2 P1c：stronger-substrate 资格分叉

P1c 不再修改 prompt、parser、compiler、RAG top-k 或 Gate 1 阈值。权威内容寻址协议为
`relationship-p1c-candidate-protocol.v2`：

- candidate：`Qwen/Qwen2.5-3B-Instruct`，development capacity tier `3b`；
- reference：Qwen2.5-1.5B 的历史 P1b artifact
  `9cd149b1e8c3f74d54d0cbaf72c216edfdaeba2979829925bc50c8ac3d60c4e8`；
- protocol id：`f209cf49957e3fa22aef20e977d42bd1f76c970c39c97f57a0e47794e0efff87`；
- Gate 0 seeds `101,211,307`，P1b seed `101`，background depths `0,8,32`，
  BGE-M3 RAG `top_k=2`，CPU bfloat16，generation config 与 P1b 完全相同；
- 冻结 evaluated context surface、background templates 与 RAG config 的 sha256；不要求
  两次独立重建中含随机 owner record UUID 的 bundle artifact id 相等；
- formal hidden test 仍为 unopened；上游 model branch 不是权重证明，实际 weight files
  必须先由 fresh Gate 0 attestation 计算 sha256，再由 P1b same-substrate 校验。

runner 顺序固定为：candidate cache/disk preflight → fresh Gate 0 → Gate 0 PASS 后同权重
P1b → `RelationshipP1cReport`。checkpoint 为 stage-level 可续跑：完整 Gate 0/P1b
artifact 可直接恢复；中途中断的目录原样保留，新运行只新增 `_attempt_N`，不覆盖旧证据。

资格判词只有三路：

| P1b 事实 | P1c verdict | 唯一允许的下一步 |
|---|---|---|
| prompt/RAG 均在 0.625–0.875 且 pair flip ≥0.5，structured-state pair flip ≥0.5，machinery/readout 全有效 | `formal_prereg_freeze_candidate` | 冻结 materialized weights，补全 formal prereg，然后才生成 secret heldout |
| 任一 prompt/RAG accuracy >0.875 | `version_scenario_dataset_saturated` | 版本化 `relationship_transfer_v2`，提高 hidden dynamics 与跨表面迁移难度，不削弱 baseline |
| machinery/readout 有效但仍未达资格 | `rewrite_public_evidence_contract` | 先重写公开 evidence/label contract，禁止新增 latent carrier、PE learning 或 steering 掩盖任务问题 |

`candidate_gate0_rejected` 与 `machinery_regression` 是前置条件失败，不属于资格三路，也
不能被解释成模型能力结论。P1c 仍是 train/validation development routing evidence，
即使第一路通过也只授权“冻结 formal prereg 候选”，不等于 formal Gate 1 或四能力 PASS。

协议 v1 的首次真实运行完整保留在
`artifacts/relationship_lab/qwen25_3b_packet1c_readout_v3_top2_20260819/`，并以
`ABORTED.md` 标记为诊断证据：模型输出已完成，但严格 lineage 正确拒绝了一个 repo-heldout
RAG context collision，也暴露出 v1 错把随机 owner record UUID 纳入跨重建 identity。
v2 没有改变模型输入、prompt、parser、compiler、模型输出、阈值或标签；修复证据契约后，
从 fresh Gate 0 开始独立重跑，避免用已见结果做 post-hoc reassessment。

### 7.3 P1d：`relationship_transfer_v2` 场景 owner 收敛包

P1d 只版本化 domain-lab owner 和数据契约，不切换 P1/P1b/P1c consumer，也不运行模型。
v1 的默认 loader、prompt、top-k、protocol 和全部 artifact 继续按原 hash 解释；v2 必须
显式以 package name/root 加载。v2 的反捷径不变量是：

1. 每位用户有四条跨 surface 历史，两种 sealed abstract condition 各两条；
2. 每种 condition 都有 stay/space 两个动作的受控结果对照；
3. 对单个用户整体看，stay 与 space 都正一次、负一次，忽略 condition 必然平局；
4. probe family 不出现在该用户历史中，不能复制同领域实例；
5. mirrored siblings 的 current bytes 与 probe condition 相同，但 sealed policy 与正确动作
   互补；
6. condition/policy/binding/preferred action 只在 generator truth，SUT 零可见。

权威 development dataset fingerprint 为
`d8e002d6d529476bf29622d4872afb0b1d7fec9d9c2e5942ecb830c8428b660b`。
默认 256 samples/action 的 Gate 0 machinery 检查为：6 mirrored pairs、6 probe surface
families、最小 analytic action effect 0.75、最小 empirical effect 0.714844、泄漏 0，
五项 machinery PASS；因未运行 v2 真实 stateless baseline，baseline tooth 是 PENDING，
所以 `gate0_passed=false`。

P1d 没有把 v1 的机械 tally readout 搬到 v2：该 readout 明令 current message 不参与且
只聚合每个动作的正负号，在 v2 上按构造只能输出双零，拿它跑模型会故意削弱 baseline。
下一独立包 P1e 必须在看任何 v2 模型输出前冻结 condition-aware readout，让 current
message 参与抽象情境推断；prompt/structured-state 获得全部四条历史，BGE-M3 RAG 固定
`top_k=4`。然后以已 materialize 的 Qwen2.5-3B fresh Gate 0 → same-substrate P1b 重跑
资格。P1d 只允许声称“场景与反捷径契约机械成立”，不允许声称 baseline qualified、
Volvence advantage、Gate 1/P2 或四能力成立。

### 7.4 P1e：v2 condition-aware consumer steelman

P1e 在任何 `relationship_transfer_v2` 模型输出之前冻结内容寻址的
`relationship-p1e-consumer-protocol.v1`，protocol id 为
`5221909debd8b0248c83332589c2681270118dc54b7014654db2d627ca2fbd1e`。协议保持
Qwen2.5-3B、CPU bfloat16、generation config、Gate 0/P1 阈值和无文本 typed compiler
不变，只把 consumer 明确升级为 v2 所需的公平 steelman：

1. readout profile 固定为 `v2_condition_aware`，current message 必须参与抽象关系压力
   归纳；不得泄漏 sealed condition/policy 名称，也不得做全局动作多数票；
2. prompt-steelman 与 structured-state 都获得同一用户全部四条公开历史；
3. RAG 真实走 BGE-M3 semantic retrieval，固定 `top_k=4`，candidate surface 仅为 typed
   relationship-outcome owner records，并覆盖四条信号记录；
4. readout 仍只发布两个 `{-1,0,+1}` score，最终动作仍由
   `relationship-evidence-argmax.v1` 机械编译；
5. dataset/evaluated-context/background/RAG/prompt/request/schema/generation/gates/seeds/
   weights 全部进入 frozen lineage；runner 支持 stage-level checkpoint、独立 attempt 与
   local-cache preflight，不从日志或 raw output 重建 producer 状态。

权威 lineage 中，dataset fingerprint 为
`d8e002d6d529476bf29622d4872afb0b1d7fec9d9c2e5942ecb830c8428b660b`，evaluated
context surface 为
`3198a31996fa7234bf7cecdbefdfc2c9fd473e277ecddb4f6e3eaade755b4c3b`，readout
prompt 为 `9687c5043029502b0787cd88758d06c1c0541338ed3c50068ad4824ce25fd4e5`，RAG
config 为 `2e1a510f324887a4d4a00055d4c9fbfc1b22ed312185eae34d0e927e61c369ff`。

2026-08-20 的真实运行先产生 24/24 valid、12/24 correct 的 stateless baseline，Gate 0
六项检查全部 PASS；随后 P1b 的 24 个 readout 全部 strict-valid，恢复、scope、成本、
console 与同基底检查成立。资格结果为：

| Arm | valid | accuracy | mirrored pair flip |
|---|---:|---:|---:|
| prompt-steelman | 8/8 | 0.625 | 0.250 |
| rag-steelman | 8/8 | 0.250 | 0.250 |
| structured-state | 8/8 | 0.625 | 0.250 |

P1b report artifact 为
`073501caf5513ef5ed75872a748eae8eba0f708a5cea6ed22ab3acbc90f671a8`，判
`baseline_underqualified / gate1_passed=false`；P1e report artifact 为
`232afebb56afb5e457af3d7ca4ccfc560cc417447defcb6d265263085fad8693`，判
`rewrite_public_evidence_contract`。该结果排除了“旧 tally、少给两条历史或 RAG top-k
不足”作为失败解释，但只是一份 development routing evidence：它不证明模型绝对无法
抽象，也不证明 Volvence 优势或四能力成立。

该判词只允许 P1f 版本化并重写公开 evidence/label contract，使冻结 evaluator 能先从
自然语言观察中核对抽象条件；禁止在已见 v2 split 上继续轮换 prompt。P1f 也不得新增
latent carrier、PE learning、controller 或 steering 来掩盖输入契约问题。

### 7.5 P1f：v3 public-evidence legibility gate

P1f 新增独立 `relationship_transfer_v3`，不修改 v2 或重跑已见 v2 输出。它保留六组
mirrored pair、每用户四历史、两 condition×两 action、每动作一正一负、未见 probe
surface 与互补个体策略；仅修复 public language contract：公开文本必须同时描述日常
事件和当事人体验到的关系损失，probe 不包含直接动作请求，sealed condition/action
标签继续对 SUT 零可见。v3 dataset fingerprint 为
`35b8c46e6fd5810779aff38ed935d8c4f0741bf7d496d2e3eec85f93fbf2134f`，public-evidence
contract id 为
`8ba8a6788d35e959c4a6fa42d31f54baa7d5e1ba48f52603e4bec510232d3cbb`；二者绑定 P1e
artifact `232afebb56afb5e457af3d7ca4ccfc560cc417447defcb6d265263085fad8693` 的
`rewrite_public_evidence_contract` 判词。

审计方法在模型输出前冻结为：使用 snapshot digest
`d548612967dcb4d75fb51e37fcfa65f3533a248f5c1157f1e0b338e261fd4b1e` 的
`BAAI/bge-m3`，由 `relationship-public-evidence-auditor.v1` 把 48 条 history 的
`user_utterance + "\n" + user_reaction` 和 12 条 probe 的 `current_input` 与两个 sealed
`hidden_summary` 做 cosine 对比；分数固定到 12 位小数，
tie 失败。阈值为 60/60 top-1、最小正确锚点 margin ≥0.02、平均 margin ≥0.07。
evaluator 只发布 source/text/anchor hash、score 与 margin，不发布原文或 condition id，
其结果不得进入 memory、PE、credit、reward 或 steering。

2026-08-20 的权威本地审计得到：

| evidence | count | top-1 correct |
|---|---:|---:|
| histories | 48 | 48 |
| probes | 12 | 12 |
| total | 60 | 60 |

top-1 accuracy 为 1.0，最小 margin 为 0.020403792213，平均 margin 为
0.080645917619；report artifact 为
`a231e2096b2c4b5fcf3e8b36fd099d0955ce2e355e793d38f5ed8e87a047ecbd`，判
`consumer_protocol_freeze_candidate`。这只表示 v3 development public evidence 在一套
冻结语义 auditor 下可判别。人工盲标仍为 `pending_before_formal`，至少需要 3 名独立
rater、隐藏标签且多数一致率 ≥0.8；它不得被该嵌入结果替代。

因此下一包 P1g 只能在任何 v3 Qwen 输出前冻结 consumer protocol、readout、完整四历史、
RAG top-k、weights/generation/seeds/gates 与阈值，然后才允许 first v3 Qwen run。P1f
没有证明 human readability、Qwen transfer、Volvence advantage、Readable、formal
heldout、产品价值或任一完整四能力闭环，也不授权 P2。

### 7.6 P1g：first v3 Qwen consumer qualification

P1g 在第一条 v3 Qwen 输出之前把完整 consumer protocol 内容寻址为
`8e08d488382442f364aae102d80c268c8c23927d547f64c1e79cb0a87f0f52c6`。协议绑定：P1f
report `a231e2096b2c4b5fcf3e8b36fd099d0955ce2e355e793d38f5ed8e87a047ecbd`、v3 dataset
与 public-evidence contract、Qwen weights
`3ccf77de3297aba6772fcb743af28b806d7b7c3e348cc7e8ad729fa98a4146cd`、BGE-M3 weights
`d548612967dcb4d75fb51e37fcfa65f3533a248f5c1157f1e0b338e261fd4b1e`、generation、
三颗 Gate 0 seed、单颗 P1b seed、condition-aware prompt/request/schema/compiler、全部四条
历史、typed relationship-outcome top-4、0/8/32 background、Gate 0/P1 thresholds，以及
`v3_qwen_outputs_observed_before_freeze=0 / hidden=false / p2=false`。preflight 必须在不调用
Qwen 的情况下重算 exact weight digests、RAG config 与 evaluated context surface；任一漂移
都 fail loudly。

2026-08-20 首次且唯一的冻结 v3 development run 得到：

| stage / arm | valid | accuracy | mirrored pair flip |
|---|---:|---:|---:|
| fresh Gate 0 stateless | 24/24 | 0.500 | — |
| prompt-steelman | 8/8 | 0.750 | 0.500 |
| rag-steelman | 8/8 | 0.500 | 0.000 |
| structured-state | 8/8 | 0.500 | 0.500 |

P1b artifact 为
`10d120f49b442803cccec53c534e8f3c868ee644c0674439ede000d8dedd3a87`，P1g report
artifact 为 `9d7f05b574bafb21641d22c766fe31c4656c09bf6f5e04493474eee6c694e3c8`，严格判
`consumer_still_underqualified`。prompt full-history 已达到 0.625–0.875 且 pair flip
≥0.5，说明冻结 Qwen 并非完全读不出 v3 抽象结构；但预注册要求 prompt 与 RAG steelman
都资格，RAG 的 0.500/0.000 使 aggregate Gate 1 失败。不得事后把 gate 改成“取最好一臂”、
删除 RAG、降低阈值或在本 split 上调 prompt。

因此 P1g 只证明：同一冻结 substrate 上，v3 数据、四历史传递、恢复、typed readout 与
报告分叉机械成立，并得到一条“prompt 有部分迁移、完整 consumer 未资格”的诚实证据。
它没有证明 Volvence advantage、Readable owner、PE→credit learning、steering、formal
heldout 或四能力闭环。下一包如继续，只能先版本化并冻结独立的 consumer-training split
与未见 qualification split，再在 training split 上学习通用 readout；禁止消费本次
evaluator 答案作为 reward，也禁止在同一 v3 qualification split 上继续人工 prompt search。

## 8. Baseline 与 formal 纪律

`FrozenBaselineAttestation` 必须记录：dataset/model/weights/prompt/generation/seed
hash、decision ledger hash、split、有效/正确/总决策数、token 成本、冻结时间及
hidden test 尚未打开的证明。Gate 0 只接受 `stateless` 或 `raw`，且只能在
train/validation/calibration split 上产生。

后续 formal 对照固定为 stateless、prompt-steelman、rag-steelman、
volvence-cold、volvence 与非竞争 oracle-concept。除 prompt/RAG 明示增加的上下文
能力外，各臂必须同 substrate weights、generation settings、candidate actions、
pre-action observation 与 reactive transition function。行动后的用户 turn 可以且
应该因为 action 不同而不同。

有限实验不得声称“任何 prompt 都不可能做到”。允许的口径仅是：优于预注册、
冻结且在 train/validation 优化过的 prompt/full-history/RAG steelmen，并同时报告
held-out action selection 与上下文成本。

## 9. 运行与当前判词

机械 smoke（不需要模型或网络）：

```bash
.venv/bin/python scripts/run_relationship_lab_gate0.py \
  --machinery-only \
  --output-dir /tmp/relationship-lab-gate0
```

生成真实、current-turn-only Qwen baseline 并直接判定 Gate 0：

```bash
.venv/bin/python scripts/run_relationship_lab_stateless_baseline.py \
  --model-source Qwen/Qwen2.5-1.5B-Instruct \
  --model-id qwen2.5-1.5b-instruct \
  --output-dir artifacts/relationship_lab/<run-id>
```

已有冻结 attestation 时可不再运行模型，直接重放判词：

```bash
.venv/bin/python scripts/run_relationship_lab_gate0.py \
  --baseline-attestation /path/to/frozen-baseline.json \
  --output-dir artifacts/relationship_lab/<run-id>
```

先用三条真实生成验证 P1 contextual prompt 的严格输出协议：

```bash
.venv/bin/python scripts/run_relationship_lab_packet1.py probe-protocol \
  --model-source Qwen/Qwen2.5-1.5B-Instruct
```

运行 P1 完整 development calibration（本地权重，默认 BGE-M3 RAG）：

```bash
.venv/bin/python scripts/run_relationship_lab_packet1.py run \
  --model-source Qwen/Qwen2.5-1.5B-Instruct \
  --rag-model-source BAAI/bge-m3 \
  --output-dir artifacts/relationship_lab/<packet1-run-id>
```

2026-08-19 开发期校准使用本机缓存的冻结
`Qwen/Qwen2.5-1.5B-Instruct`：24/24 个结构化决策有效，4/24 正确，context token
总计 4554；decision ledger sha256 为
`fff11045ea2302bbb2f218590449f1eaec9075110fe6fa3340e9f4bd574ca45b`。
六项检查全部 PASS，`machinery_ready=true / gate0_passed=true`；报告在
`artifacts/relationship_lab/qwen25_15b_stateless_calibration_20260819/`。

这只是 Gate 0 开发期仪器校准。`prereg_template.json` 仍是
`template_not_frozen`，formal secret heldout 也尚未生成，不能把该结果解释为
Appendable / Readable / Learnable / Steerable 或产品效果成立。校准中删除 baseline
accuracy 下限发生在 formal prereg 与 secret heldout 之前；原因和替代约束已写入
§6，未查看 formal hidden outcome。

P1 首次 development run 暴露 prompt 协议示例把三个 action 写成一个带 `|` 的
字符串，1.5B 模型原样复制；该失败完整保留在
`qwen25_15b_packet1_development_20260819/`。修复只把合法 enum 与 JSON 语法拆开，
没有改候选动作、历史、标签、阈值或解析器；v2 协议探针三臂均给出合法单值 JSON。

完整 v2 run 位于
`artifacts/relationship_lab/qwen25_15b_packet1_development_v2_20260819/`。32 条决策中
只有一条 RAG 输出了额外键，严格判 invalid；四臂结果为：

| Arm | valid | accuracy | mirrored pair flip |
|---|---:|---:|---:|
| stateless | 8/8 | 0.125 | 0.000 |
| prompt-steelman | 8/8 | 0.500 | 0.000 |
| rag-steelman | 7/8 | 0.500 | 0.333（仅三组可判） |
| structured-state | 8/8 | 0.250 | 0.250 |

跨进程恢复、scope isolation、token scaling、console correction/delete 均 PASS。
32 段普通历史时，full-history 平均 1262.75 tokens；RAG 为 229.667（0.181878），
structured-state 为 139.75（0.110671）。但 exact-output、structured-state user-swap
效应和两种 steelman 资格均 FAIL；v2 派生报告 artifact 为
`76d81bebb98e9afc41ace995f7f03a6d01c6e57437a08f378bac6b7ae132b083`，结论是
`machinery_ready=false / gate1_passed=false`。

P1b 随后把 contextual action prompt 拆成 frozen substrate evidence readout 与
无文本 typed compiler。v1 诊断 run 中，structured-state 从 P1 的 0.25/0.25 提升到
accuracy 0.875、pair flip 0.75，说明同一 Qwen 在紧凑 owner surface 上曾经读到足以
改变动作的历史证据；但 full-history 只有 5/8 strict-valid，prompt/RAG 仍未达资格。
该 run 保留在
`artifacts/relationship_lab/qwen25_15b_packet1b_readout_v1_20260819/`，不得用单臂
正结果升级系统主张。

lineage-complete v3 run 固定 readout prompt v3、request template v1、strict schema、
compiler v1 与 BGE-M3 RAG `top_k=2`，结果如下：

| Arm | valid | accuracy | mirrored pair flip |
|---|---:|---:|---:|
| prompt-steelman | 8/8 | 0.250 | 0.000 |
| rag-steelman | 8/8 | 0.500 | 0.000 |
| structured-state | 8/8 | 0.500 | 0.000 |

因此五项 machinery checks（恢复、scope、成本、console、同基底与有效输出）均通过，
`machinery_ready=true`；但历史证据→action 的抽象归并对 prompt 版本不稳定，行为门和
steelman qualification 失败，P1b 判词为
`baseline_underqualified / gate1_passed=false`。报告位于
`artifacts/relationship_lab/qwen25_15b_packet1b_readout_v3_top2_20260819/`，artifact
为 `9cd149b1e8c3f74d54d0cbaf72c216edfdaeba2979829925bc50c8ac3d60c4e8`。最后一次
不含 scene/答案的短 prompt 协议探针重新产生 Markdown 与同向分数，按止损条件未进入
权威 run，也未放宽 parser。

P1c cache/disk 预检（不会运行模型）：

```bash
.venv/bin/python scripts/run_relationship_lab_packet1c.py --preflight-only
```

完整 development run 使用：

```bash
.venv/bin/python scripts/run_relationship_lab_packet1c.py \
  --allow-download \
  --output-dir artifacts/relationship_lab/<packet1c-run-id>
```

下载只发生在 preflight；Gate 0 与 P1b 子进程随后强制 local-cache-only，并以实际
weights sha256 锁定 same-substrate。退出码 0 只表示可冻结 formal prereg 候选；
科学上的 saturation/underqualification 使用退出码 2，cache 未就绪使用退出码 3。

P1e 的 v2 consumer lineage/cache 预检与完整运行分别为：

```bash
.venv/bin/python scripts/run_relationship_lab_packet1e.py --preflight-only

.venv/bin/python scripts/run_relationship_lab_packet1e.py \
  --output-dir artifacts/relationship_lab/<packet1e-run-id>
```

P1e 同样只在 preflight 阶段允许显式下载；本机已有 Qwen2.5-3B 与 BGE-M3 snapshot 时
完整运行不需要网络。退出码 0 只代表 formal-prereg freeze candidate，科学上的
still-saturated/underqualification 使用退出码 2。

2026-08-20 的权威 v2 run 位于
`artifacts/relationship_lab/qwen25_3b_packet1c_v2_readout_v3_top2_20260819/`。fresh Gate 0
为 24/24 valid、10/24 correct（accuracy 0.4167），machinery 与 Gate 0 均 PASS；P1b
24/24 readout strict-valid，prompt/RAG/structured-state 均为 accuracy 1.0、四组 mirrored
pair 全部 flip。P1c report artifact 为
`599e7e94ac1a06a7b342f6024614c1489b6130e768c1d5db8fbd7b833bfba1d7`，判词
`version_scenario_dataset_saturated`。

P1e 权威运行位于
`artifacts/relationship_lab/qwen25_3b_packet1e_v2_conditioned_top4_20260820/`。fresh Gate 0
为 24/24 valid、accuracy 0.50；P1b 24/24 readout strict-valid，prompt/RAG/structured
accuracy 为 0.625/0.25/0.625，pair flip 均为 0.25。报告因此发布
`rewrite_public_evidence_contract`，没有进入 formal prereg。

P1f 的本地 lineage 预检与完整 public-evidence 审计分别为：

```bash
.venv/bin/python scripts/run_relationship_lab_packet1f.py --preflight-only

.venv/bin/python scripts/run_relationship_lab_packet1f.py \
  --output-dir artifacts/relationship_lab/<packet1f-run-id>
```

默认只使用本地冻结 BGE-M3 snapshot；缺失时返回退出码 3，只有显式
`--allow-download` 才能 materialize。退出码 0 只表示 public-evidence legibility 达到
冻结阈值，退出码 2 表示必须再次改写契约。权威产物位于
`artifacts/relationship_lab/bge_m3_packet1f_v3_public_evidence_20260820/`，60/60 top-1，
最小/平均 margin 为 0.020403792213/0.080645917619，artifact
`a231e2096b2c4b5fcf3e8b36fd099d0955ce2e355e793d38f5ed8e87a047ecbd`。

P1g 的零 Qwen lineage preflight 与完整首次 v3 qualification 分别为：

```bash
.venv/bin/python scripts/run_relationship_lab_packet1g.py --preflight-only

.venv/bin/python scripts/run_relationship_lab_packet1g.py \
  --output-dir artifacts/relationship_lab/<packet1g-run-id>
```

preflight 会重算 Qwen/BGE exact weights 与完整 v3 context surface，但不生成 Qwen 输出；
缺 snapshot 返回 3。完整运行只有 `formal_prereg_freeze_candidate` 返回 0；saturation、
underqualification 或前置失败返回 2。权威运行位于
`artifacts/relationship_lab/qwen25_3b_packet1g_v3_conditioned_top4_20260820/`，P1g artifact
为 `9d7f05b574bafb21641d22c766fe31c4656c09bf6f5e04493474eee6c694e3c8`，判
`consumer_still_underqualified`。

因此当前证明的是：既有 memory/reference owner 能追加、恢复、隔离、纠删、压缩并把
完整四历史交给冻结 consumer；v2/v3 消除了全局动作偏好捷径；v3 的 60 个公开单元在
一套冻结 BGE-M3 development auditor 下都能读出预期抽象关系条件；冻结 Qwen 的
prompt full-history 臂已达到资格带。但 RAG steelman 未产生 mirrored flip，完整 consumer
qualification 失败；仍没有人工盲标、Volvence 相对基线优势、Readable、Learnable 或
Steerable 证据。按冻结分叉，P2 与 formal 继续关闭；后续必须使用独立 consumer-training /
unseen-qualification split，不能回调本次 v3 prompt、RAG 或 gate。

## 10. 回滚与下一包

P0 无 runtime wiring。回滚只需停止 lab runner/consumer 并移除 scenario package；
产品和内核行为不变。任何 schema 变化必须 bump schema/environment/package version，
重新校准、重新冻结 prereg，旧 artifact 仍按原 hash 保留。

P1 没有 runtime wiring。回滚只需停止 packet1 runner，并不再消费其离线 artifact；
`MemoryStore`、产品 runtime 与 steering 配置不变。P1 Gate 1 未过前不得进入 P2
formal。下一收敛包只能修 baseline/行为可判读性，不能提前实现多经历 ToM owner
更新、PE learning 或 ACTIVE steering，也不得降低阈值、放宽 JSON parser 或查看
formal heldout 后改 prompt。

P1c 同样没有 runtime wiring。回滚只需停止 `run_relationship_lab_packet1c.py` 并不再
消费 candidate protocol/report；Gate 0、P1/P1b 与产品内核行为均不变。中断恢复不
删除旧 attempt；要完全撤回 P1c，只移除其离线 protocol/runner/consumer，旧内容寻址
artifact 继续按原 schema 可审计。

P1d 同样没有 runtime wiring，且 v1 仍为默认 package。回滚只需停止显式加载
`relationship_transfer_v2`；删除 v2 package 与 version-aware 分支即可撤回，不需要迁移
产品或内核状态。任何 consumer 都禁止在没有 package lineage 的情况下自动漂移到 v2。

P1e 同样没有 runtime wiring。回滚只需停止 `run_relationship_lab_packet1e.py`，不再消费
其 v2 protocol/report，并让 P1/P1b 默认 profile 保持 v1；产品、PE、controller、steering
与内核状态均不需要迁移。已经发布的 P1e artifact 必须按原 hash 保留，不能用后续 P1f
证据契约覆盖或重写。

P1f 同样没有 runtime wiring。回滚只需停止 `run_relationship_lab_packet1f.py` 并停止
显式加载 `relationship_transfer_v3`；删除 v3 package、P1f audit consumer 与报告即可，
v1 默认、v2/P1e lineage、产品、memory、PE、controller 和 steering 均不迁移。已经发布的
v3 fingerprint、public-evidence contract 与 P1f artifact 必须按原 hash 保留；P1g 不得
看到 v3 Qwen 输出后回改它们。

P1g 同样没有 runtime wiring。回滚只需停止 `run_relationship_lab_packet1g.py` 并不再
消费 P1g protocol/report；产品、memory、PE、controller、steering 与 v1/v2/v3 数据 owner
均不迁移。已经发布的 protocol `8e08d488…52c6`、P1b `10d120f…3a87` 与 P1g
`9d7f05b5…e3c8` 必须按原 hash 保留。若继续下一版本，退出条件是先冻结独立训练/资格
split 与新 protocol；回滚方式是停用该新离线 consumer，不得重写本次输出或阈值。
