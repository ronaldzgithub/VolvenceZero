# Relationship Lab v0：同句镜像用户、强基线与反应式关系后果

> 状态：P0 Gate 0 开发期校准 PASS；P1/P1b 强基线与 Appendable 复刻已落地。
> lineage-complete P1b 的跨进程恢复、用户隔离、token scaling、console 纠删与
> 全输出合法性 PASS，即 `machinery_ready=true`；但三个 contextual arm 的稳定
> user-swap 行为效应与 steelman 资格未过，`baseline_underqualified /
> gate1_passed=false`。P1c 随后以 Qwen2.5-3B 完整重跑 fresh Gate 0 与 same-substrate
> P1b；prompt/RAG/structured-state 三臂均为 8/8 correct、pair flip 1.0，故权威判词是
> `version_scenario_dataset_saturated`：现有 v1 场景不能区分强普通上下文基线与系统。
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
| 公开经历与封存真值 | `lifeform-domain-emogpt/.../scenario_packages/relationship_transfer_v1/` | 分离保存 rendered observation、generator truth、split 与 prereg 模板 |
| 决策与 sidecar 契约 | `lifeform-domain-emogpt.lab.contracts` | closed action surface、行动前下注、模型 lineage、内容寻址 `RelationshipDecisionTrace` |
| 数据 loader | `lifeform-domain-emogpt.lab.dataset` | 严格校验六组 mirrored users、四个 surface family、整组 split，并只构造白名单 SUT payload |
| 反应式环境 | `lifeform-domain-emogpt.lab.environment` | 由 sealed latent dynamic × 实际 action 机械产生 typed outcome；LLM 不决定标签 |
| 冻结 stateless baseline | `lifeform-evolution.relationship_lab_baseline` | 用同一真实 substrate 生成 current-turn-only 决策账本、hash 与 attestation；不读取 history/truth |
| Gate 0 编排与判词 | `lifeform-evolution.relationship_lab_gate0` | 只读校准、泄漏审计、baseline attestation 验证与报告 |
| P1 持久化与上下文编排 | `lifeform-evolution.relationship_lab_contexts` | 只经正式 API 把公开经历写入既有 `MemoryStore` owner 与 `companion-ref-harness` reference owner，发布只读 context/state digest；自身不成为 memory owner |
| P1 四臂 runner 与判词 | `lifeform-evolution.relationship_lab_packet1` | 同一冻结 substrate 上运行 stateless/full-history/RAG/structured-state，发布逐决策账本、恢复/成本/行为门与 P1 报告 |
| P1b readout owner | `lifeform-evolution.relationship_lab_packet1b` | 发布 schema-bound evidence readout、无文本 typed compiler、lineage-complete v4 报告与 saturation verdict |
| P1c 资格分叉 owner | `lifeform-evolution.relationship_lab_packet1c` | 冻结 stronger-substrate candidate protocol，只消费 Gate 0/P1b 正式工件，发布 formal-prereg / scenario-version / evidence-contract 三路判词 |
| CLI | `scripts/run_relationship_lab_{stateless_baseline,gate0,packet1,packet1c}.py` | 冻结真实 baseline、重放 Gate 0，运行/恢复审计 P1/P1b，或串联 P1c 资格分叉 |

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

该文件只允许 `ReactiveRelationshipEnvironment` 与只读 evaluation 访问。

包内的 `heldout` 是开发期结构分割，用于验证 loader 隔离与阻止 baseline 调参时
消费对应 pair；因为真值文件存在于仓库中，它**不是** formal secret heldout。正式
实验必须在 prereg 冻结后另行生成、封存并仅向独立 evaluator 解封测试集，不能把
本包的开发期 heldout 当成盲测证据。

### 3.2 Rendered observations

`rendered_observations.json` 只含公开场景材料与 harness metadata：

- opaque `scene_id` 与可哈希化 user scope；
- 两次过去自然语言经历；
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

包名 `relationship_transfer_v1`，符合 `^[a-z][a-z0-9_]*$`，并包含：

- `manifest.yaml`
- `ssot_fragment.json`
- `scenes.yaml`
- `test_suite.yaml`
- `rendered_observations.json`
- `generator_truth.json`
- `prereg_template.json`

四条路径（经历证据、行动前下注、反应式结算、counterfactual user swap）均被
同一四阶段 arc 引用，`phase_order=0..3` 连续。routing tests 不少于 6，含
negative case；semantic coherence 不少于 3。路由只允许 trajectory embedding、
schema-bound structured output 或后续 owner readout，禁止关键词、正则、scene id
查表或 current-text-only 动作选择。

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

2026-08-20 的权威 v2 run 位于
`artifacts/relationship_lab/qwen25_3b_packet1c_v2_readout_v3_top2_20260819/`。fresh Gate 0
为 24/24 valid、10/24 correct（accuracy 0.4167），machinery 与 Gate 0 均 PASS；P1b
24/24 readout strict-valid，prompt/RAG/structured-state 均为 accuracy 1.0、四组 mirrored
pair 全部 flip。P1c report artifact 为
`599e7e94ac1a06a7b342f6024614c1489b6130e768c1d5db8fbd7b833bfba1d7`，判词
`version_scenario_dataset_saturated`。

因此当前证明的是：既有 memory/reference owner 能追加、恢复、隔离、纠删并压缩上下文，
而现有 v1 公开证据又简单到更强普通 full-history/RAG 基线可做满。它没有证明 Volvence
相对基线优势，也没有证明 Readable、Learnable 或 Steerable。按冻结分叉，P2 继续关闭；
下一包必须版本化 `relationship_transfer_v2`，提高隐藏动力学与跨表面迁移难度，同时
保留这些满分基线、不降门槛、不再轮换 development prompt。

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
