# 有界残差 Steering Runtime

> 能力域：R2 / R4 / R8 / R10 / R12 / R15
> 状态：B1 契约、B2 owner/SHADOW 运行时、C1 终局信用链与 C3/B3
> 控制面已落地；真实 MSC C3/B3 formal 仍须在 A1 前置完成后按新 prereg 执行，
> 当前不授权 ACTIVE。

## 1. 目标与边界

本能力把 `research/steering-2026-08/` 已封存的「读得到、扭得动、
学会何时扭」机制拆成三个唯一 runtime owner，不改写 S2 additive
steering FAIL、S3-D literal FAIL 或 `kill-eta` 等历史判词。基底一直
冻结；online-fast 只允许 gate policy 在后续独立收敛包内更新，reader
与 executor artifact 必须冻结。

这不是 prompt 路由、关键词分类或 evaluation 回灌。自然语言不在本链被
字符串匹配为 scene/action；sensor 仅解释 substrate owner 发布的目标层残差，
gate 仅消费 owner-published belief 与 PE 代理观测。

## 2. Owner 与快照链

| Slot | 唯一 owner / wheel | 直接依赖 | 默认 |
|------|--------------------|----------|------|
| `steering_condition_belief` | `SteeringSensorModule` / `vz-cognition` | `substrate` | SHADOW |
| `steering_gate_decision` | `SteeringGateModule` / `vz-temporal` | belief, `prediction_error` | SHADOW |
| `steering_intervention` | `SteeringExecutorModule` / `vz-substrate` | `substrate`, belief, gate | SHADOW |

同一波内，SHADOW consumer 可读取该波上游 SHADOW 快照，以便三件套能做
完整只读双跑；ACTIVE consumer 永远只看 active mapping，不能读取或继承
SHADOW 值。session 跨拍只携带上一拍 ACTIVE mapping；上一拍 SHADOW 输出不得
进入下一拍 upstream。该隔离由 runtime kernel 与 session orchestrator 统一实现，
不是 owner 私有 fallback。

## 3. 冻结 artifact 契约

`SteeringArtifactBundle` 只能包含一组 lineage 一致的 artifact：

- reader：`model_id` + loaded-base SHA-256 + prereg SHA-256 + 精确 layer/width +
  class labels + ridge weights + normalization vectors；
- executor：与 reader 的 model/digest/layer/width/labels 逐项一致，并指向
  `reader_artifact_id`；冻结 `U/V/condition_codes`、rank 与 norm cap；
- sensor-off executor（可选、evidence-only）：与正式 executor 的预算、几何、
  基底和 reader lineage 完全一致，但所有 class row 必须逐字节重复同一个
  condition code，形成真正的 unconditional operator；只能在 SHADOW 使用；
- gate：冻结 feature schema、二动作 policy weights/bias 与严格递增的
  `policy_version`。

大型参数不嵌入 prompt 或运行时 schema dict。加载 artifact 时必须验证实际
加载基底的权重 SHA-256；builtin fallback、未验证的权重目录、模型或层宽
漂移都 fail loudly。

## 4. Snapshot shape

### `SteeringConditionBelief`

- lagged belief：`belief_label/index/margin`；
- 同拍 fresh read：`fresh_belief_label/index/fresh_margin`；
- gate 代理：`belief_disagrees_fresh / staleness_proxy /
  base_action_entropy`；
- lineage：reader id、source model/layer、source residual norm。

### `SteeringGateDecision`

- 动作只有 `{noop, steer}`；
- `steer_probability`、有序命名 observations、policy artifact/version；
- `decision_id` 唯一绑定一次 gate 决策；
- `terminal_credit_pending=True` 仅表示等待 C1 终局信用结算，
  不会把 evaluation 读数伪造成 reward。

### `SteeringIntervention`

- action、model id/digest、layer、完整有界 delta；
- source/control/cap norms、reader/executor/gate lineage；
- strict-noop、application mode、SHADOW hook attestation、backend 与只读
  downstream-effect 摘要；
- `noop_context / action_context` 是 substrate owner 发布的 text-free、按固定
  layer order 拼接并 L2 归一的残差表示，携带 source/value SHA-256；
- B3 收集 profile 还发布匹配预算的 `sensor_off_action_context`、artifact id、
  control norm 与独立 hook latency。部分 sensor-off 字段或 geometry 漂移会失败。

所有 value 均为 frozen dataclass，不暴露内部可变引用。

## 5. 数学与安全不变量

executor 仅实现预注册的乘性低秩算子：

```text
delta = U @ (tanh(Z[k]) ⊙ (Vᵀ h))
```

- 禁止 free bias；
- `noop` 必须得到逐元为零的 delta；
- `||delta||₂ <= control_norm_cap_ratio × ||h||₂`；
- runtime 只能在 artifact 精确绑定的 layer 注入完整宽度 delta；
- vLLM 和 synthetic generation 没有可验证残差 hook，请求 ACTIVE
  steering 时必须显式 `NotImplementedError`。

### C1 终局 PE 信用链

C1 是 out-of-turn owner API，不新增 live slot，也不把 terminal outcome
塞回同一波 DAG：

1. PE owner 的 `settle_steering_terminal_prediction_error(...)` 只接受
   action/noop 两个冻结 heldout `ForwardRepresentationBatchSnapshot`；两臂必须
   绑定同一 predictor fingerprint、substrate target lineage、sample/history 坐标和
   actual target，且都不得执行 update。
2. PE owner 发布 `SteeringTerminalPredictionError`，主标量为
   `clip((noop_mse - action_mse) / max(action_mse, noop_mse, eps), -1, 1)`；
   cosine improvement 只作同一 target 的诊断读数。接口不接受 evaluation/judge。
3. Credit owner 把一次 terminal settlement 按 `decision_id` 展开为
   `CreditRecord(level="steering_terminal_prediction_error")`，保留 episode、head 和
   target lineage；同一 `(episode_id, prediction_head_fingerprint)` 只能入账一次。
4. Gate owner 只消费仍在 pending 集合中的匹配记录，以 bounded policy-gradient
   更新 `{noop, steer}` 权重。相同 steer-vs-noop advantage 会按实际动作定向：
   STEER 使用原符号，NOOP 使用相反符号，禁止把混合动作 credit 平均后抵消；一批
   结算至多递增一次 `policy_version`，已消费 record 重放为严格 no-op，未知 decision
   则 fail loudly。

C3 的昂贵 matched counterfactual 每个真实 turn 只算一次。PE owner 的
`bind_steering_terminal_prediction_error_decisions(...)` 只把已冻结 mismatch 重新绑定到
预注册 replay 的新 `decision_id`；MSE/cosine、head 和 target lineage 不得重算或改写。
每个 replay 仍严格经过 PE→Credit→Gate。`evidence-stochastic` 只允许 SHADOW，随机流、
pending decision、已消费 record 和完整 policy 参数由 `SteeringGateCheckpoint` 做 canonical
JSON round-trip，可恢复精确后续动作序列。

因此学习源始终是 PE owner 对 matched noop 的 N+1 表示误差比较；credit 只聚合并
路由，evaluation 仍为只读验收面。

C2 专家验证锚使用独立离线契约
[`steering-human-anchor.md`](./steering-human-anchor.md)。其真人方向对照不得进入本
owner 链；即使观察到分歧，也只能触发新的信用面复审 prereg，不能直接改写 gate。

## 6. SHADOW 与用户可见生成

- SHADOW executor 默认可在 transformers 后端运行第二次、不可见的
  preview forward，并把 attestation 写入 SHADOW 快照；
- `steering_shadow_hook=False` 是计算级回滚：仍计算快照，但不注册
  direct preview hook；
- expression/session 只从 `active_snapshots["steering_intervention"]` 构造
  `ResponseContext`，禁止从 shadow fallback；
- ACTIVE `steer` 由 transformers `generate(..., capture_residuals=False)` 在目标层
  注入；hook 的存在不依赖 residual capture；返回的 artifact/action/policy
  attestation 必须与 owner 快照完全一致，否则用户响应失败。

### 6.1 Windows/CUDA strict substrate execution profile

Windows/CUDA 长 context 的模型执行、真实 hook、capture 保留与 token 预算仍由
`vz-substrate` 唯一拥有；sensor、gate 与 expression 不得解释 CUDA/backend 状态或重建
capture。显式 opt-in `WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1` 固定 Windows + CUDA、
Qwen 原生 32768-token context、`bfloat16`、`attn_implementation=sdpa`、exclusive cuDNN
SDPA、generation-only cached decode、strict-local、fallback `deny`、fail-on-truncation 与
generation-only `first-full-prompt-set-once` capture。加载时必须同时预绑定并复核
logical model ID、verified revision、权重 SHA 与全部非权重 execution-assets SHA；任一
平台、device、dtype、原生 context、attention、generation cache/backend 或 hook layer 不一致
都必须在运行前 fail loudly；不得回落
eager、math、builtin 或 pooled evidence 后继续标记为 strict profile。

runtime 启动后由 substrate owner 生成 frozen、content-addressed
`TransformersExecutionAttestation`。完整 attestation 保留在 runtime/evidence artifact；
`GenerationResult` 只复制 `execution_attestation_id`，并绑定同次生成的 frozen
`GenerationContextBudgetAttestation`。后者必须在 chat template 已实际 tokenize、全部
Prefix-KV 已合并、有效 generation 上限已结算后，且在 hook/model forward 之前计算：

```text
actual_input_tokens + actual_prefix_slots + effective_max_new_tokens
    <= model.config.max_position_embeddings
```

strict generation 的显式 messages 与 plain `prompt/system_context` API 都先归一化为 messages，
再强制走 `apply_chat_template(tokenize=True)` 与同一 fail-loud 上限校验；不存在 strict
generation 的 plain-tokenizer 旁路。cached decode 的 hook 在首个完整 prompt forward 对每个
layer set-once；后续 single-token forward 继续施加有界 intervention，但不得覆盖 capture。
standalone capture 与 differentiable instrumental scorer 故意保留 raw-tokenizer / `use_cache=False`，
但它们的 probe、full forward、prefix build 和 prefix-cache upper replay 也必须全部进入
exclusive cuDNN SDPA context。strict profile 下缺 layer、shape/token-count 漂移或 capture 失败均
re-raise。

这是 opt-in execution profile；`execution_profile=None` 保持当前 Windows eager、
generation `use_cache=False`、pooled capture 保护与既有 fallback 默认完全不变。该契约不新增
runtime slot、不修改 `SubstrateSnapshot` 或 steering 三件套 shape，也不授权 production
ACTIVE。它只提供 Windows/CUDA substrate 执行可信度的 development evidence；单独通过不能
证明 Appendable / Readable / Learnable / Steerable 任一完整能力，更不能证明四能力闭环。

### 6.2 Windows/CUDA strict 32767+1 engineering diagnostic

`windows-cuda-strict-32k-smoke.v1` 把一次真实的窗口边界执行固定为独立、
create-only、内容寻址的工程诊断。底层模型执行、chat-template token 预算、真实
hook 与 capture 仍由 `vz-substrate` 唯一拥有；`vz-runtime` 只通过 public
`build_transformers_runtime_with_fallback(...)` 构造 exact strict runtime，并调用一次
public `generate(...)` 发布离线 evidence artifact，不解释或重建 substrate 隐状态。
冻结 protocol ID 为 `4934a344550aab5c98f33892dd6d1ec2e5fe51c00694d2cc5b0a45fbc31e2c1a`，
raw protocol SHA-256 为 `ec7b7bcec82668ac89b549b8769997897d07720a507ded6a316d4cee3b785eb4`；
十个已审入口源码采用 `utf8_lf_canonical_v1` 独立指纹，任一漂移即拒绝加载。但该集合不覆盖
`substrate/__init__.py` eager imports、其他 lazy local imports 与 `vz-contracts` 的完整传递闭包，故协议
固定 `transitive_local_source_closure_pinned=false`；闭包修订落地前，即使绑定 outer 也不得称为
exact-source physical evidence。
token-level residual 与 feature-surface 只由 additive substrate owner
`strict_capture_audit.audit_strict_capture(...)` 遍历，并发布 frozen、bounded summary；
runtime orchestrator 只消费该 summary。

协议同时冻结 Qwen2.5-1.5B 的 logical model ID、revision、weights/assets SHA、strict
profile/attestation、layer 20、width 1536、`bfloat16`、fallback DENY 与无搜索 prompt
recipe。唯一允许的 forward 必须在同一次返回值中满足：chat-template 后 input 为
32767 token、Prefix-KV 为 0、`max_new_tokens=1`、combined 为 32768、remaining 为 0；
真实 cached generation 返回 1 token；first-full-prompt capture 含 32767 个 residual
step、唯一 layer 20、width 1536，且 hook/token coverage 均为 1。
顶层 latest residual 必须与 sequence 末步逐值一致；owner 摘要以包含实际 step/layer/
activation cardinality/width 的 framed hash 绑定完整 residual geometry。预算、attestation、
rendered prompt hash、context-budget schema 或 capture shape 任一漂移都 fail closed；
禁止缩 prompt、改 layer、切 backend/fallback、retry 或挑选一次 PASS。

调用者必须先从外层 host campaign 取得预注册的 `outer_attempt_lease_id`。本地输出根为
create-only：构造 runtime 前先写入并 `fsync` 绑定 lease、protocol、source 与进程/时间的 launch
receipt；完成的失败观察与 PASS 一样写入 attestation、report、manifest 和 completion
receipt，运行时异常留下且 runner 不删除不可重用的不完整根。OS hard crash 后本地 launch
是否仍被文件系统保留不作保证，独立外层 receipt 才是权威锚点。manifest 内容寻址绑定
launch、attestation 与 report，completion receipt 再绑定 attempt/artifact/lease/verdict；
完整根恰有五个文件。有限 capture 摘要不得包含 32767×1536 residual sequence。
`validate-existing` 必须取得预期 outer lease，并重算 receipt ID、protocol、source、payload、
manifest、artifact ID 与全部跨文件 lineage，不加载 torch、CUDA 或模型；只有诊断 PASS
返回进程码 0，结构合法的 failed diagnostic 仍返回 2，禁止被 campaign/CI 当成通过。

本地 `attempt_budget=1/retry_budget=0` 只对同一冻结输出根生效。缺外层锚点时，完整本地根
只证明文件/hash/lineage 自洽，不能独立证明物理执行；缺 completion 的不完整根一律不是
PASS。runner 只装配 `vz-runtime / vz-substrate / vz-contracts` 三个冻结 namespace root，并
核对 strict implementation 的实际 import origin；新增 `packages/*/src` 不得改变本次导入。

### 6.3 Windows/CUDA strict 32K host-campaign scaffold

`windows-cuda-strict-32k-host-campaign.v1` 由 `vz-runtime` 的 offline campaign control owner
持有，protocol ID 为 `cf62484fccdcaf71e6db1f2f0b6d9034443ded15891a8f41fa73b638e7bd3194`，
raw SHA-256 为 `5f1740247fadd12cbd32726ab6386b1b58e3abba6d2b980374b8077e8ad38f43`。
它只支持 repository source checkout，不是 installed-wheel 独立入口。三项 outer source pin
覆盖 Node owner、Windows PowerShell collector 与固定 CLI；public CLI 不接受替代 protocol、
collector、executor 或 validator。

当前 v1 **明确设置 `production_preregistration_enabled=false`**：host-stability qualification
已有冻结的唯一 owner/protocol、synthetic full-root validator 与 non-authorizing raw-Audit artifact adapter
core；standalone direct-Audit acquisition/quarantine v2 已验证 same-buffer requested binding 与 bounded
process/pipe supervision，但仍是 production-disabled contract core，尚无 realized executable attestation、
真实 Event Log capture、独立 live reobservation、production full validator，
也没有 qualification 终点 cursor 到 outer baseline 的端到端
连续性证明。因此 public preregistration 必须 fail closed；synthetic
dependency backend 使用独立 scope，写入 `real_execution_observation_authorized=false`，外层结果
强制包含 `synthetic_test_backend_not_evidence`、永远不能 PASS，public validator 也拒绝读取。

scaffold 已冻结的机制仅供后续独立审计：scope 由 outer/child protocol、qualification artifact、
host identity 与 backend 做 domain-separated hash；000 scope claim 在读取 anchor/baseline 前
create-only 落盘；002 raw SHA 是一次性 lease；004 在 child creation 前 `fsync` 消费 lease；
000–012 receipt chain、两条 child stream 与 child exact-five 根均 create-only。prelaunch 必须在
004 前复核同 machine/boot、baseline boundary、Circular channel config 与 4096-record 预算；
Node 必须从 frozen cursors、完整 RecordID 序列、boundary/end XML hash 独立重算 delta，不信任
PowerShell 自报的 complete/truncated 布尔值。合法 `exit=2/passed=false` 是 diagnostic failure，
不是 lineage mismatch；launch 后 wrapper 中断是永久 `incomplete_consumed`，完整失败 seal 后也
`retry_permitted=false`。

no-retry 只覆盖该 exact scope；新 qualification/protocol/backend 构成新 scope，跨 scope 选择
尚未排除。Windows Application Event Log 只是第二个本地通道，不是 external/WORM authority；
本机管理员以及具有 Application-log 写权限的普通本地主体都可能伪造、清除或复制记录，hidden
run 也不能排除。PowerShell/Python/Node runtime identity、realized child environment、完整 local
import/source closure、process-tree containment、producer 返回 PASS 前的 full-artifact 自重验、
terminal anchor 与延迟 WHEA/WER 的区间覆盖，以及受限 Event Log provisioning 仍是另一个 audited
protocol revision 的启用前 blocker，不能只补 qualification 后就开放生产路径。

即使未来一个授权 campaign PASS，也只允许表述为：固定 argv 的 Windows/CUDA strict 32767+1
工程诊断在一个有界本地 Event Log 区间内完成，且没有观察到已分类 host fault。它不证明宿主
普遍稳定、长 context 信息利用、跨 session 写入/恢复、命名读出、PE→credit 学习、条件 residual
因果效应、独立 subject、长陪伴或 Appendable / Readable / Learnable / Steerable 任一完整能力。
当前 host-block receipt 仍优先：完成 BIOS/microcode 修复与独立 host qualification 前禁止执行。

### 6.4 Windows host-stability qualification publisher scaffold

`windows_cuda_host_stability_qualification` 是 `vz-runtime` 的独立 offline evidence owner。当前冻结协议为
`windows-cuda-host-stability-qualification-protocol.v1`，protocol ID
`32f35e4f7027e9519522e099efb696fb352a48faf3ba69be861929304fae1d5f`、raw SHA-256
`30a881838b41fa5b7e6de5aba6bc94131245796126be5b49c4ebab539f8c4132`。它实现了
**synthetic publisher + synthetic full-root integrity validator + non-authorizing raw-Audit artifact adapter
core**；三个 production public entrypoint 在读取 options、路径或 Proxy getter 前静态 fail closed，不存在
真实 probe；PowerShell acquisition 属于后述 standalone owner，且其 production public entrypoint 同样静态
禁用。当前仍不存在独立 live source reobservation 或可消费的宿主资格 validator。

synthetic create-only 根恰有 14 个文件：000–009 十个 receipt、010 manifest、011 terminal 与
`streams/probe.{stdout,stderr}.log`。每个 receipt 绑定前一文件的 raw-byte SHA-256；manifest 只枚举
000–009 与两条 stream。`artifact_id` 是删除自身字段后的 manifest core canonical hash，terminal 保存该
`artifact_id`，其 `previous_receipt_raw_sha256` 绑定 manifest raw SHA；`terminal_id` 是删除自身字段后的
terminal core canonical hash。完整根对外身份还必须包含 terminal raw SHA-256，未来 consumer 不得只绑定
三者之一。validator 重算 exact 文件/目录集、regular-single-link/path containment、strict JSON、receipt
raw chain、manifest inventory、两个 content ID、时间链、little-endian microcode 数值门、双 channel cursor
连续性、4096-record/window 预算、cooldown/tail 时长和由**归一化 event 字段**得到的故障分类；它没有保存
或独立解析 raw Event XML，因此不得称原始 Event Log 分类已被独立复核。

002/008 当前保存的是
`windows-cuda-host-stability-source-audit-projection.v2` 资格投影，不是被 pin 的 PowerShell provisioner
原始 `volvence-evidence-event-log-provisioning-audit.v2` shape。投影固定
`full_raw_audit_bound=false / raw_audit_sha256=null /
raw_audit_content_id_basis_revalidated=false / real_provisioner_observation=false`；本包没有翻转这些字段，也
没有让 production 复用 synthetic predicate。

新增的 `adaptProvisionerAuditV2Artifact` 只发布深冻结
`windows-event-log-source-audit-artifact-adapter-snapshot.v1`。输入是 artifact-root 内 regular single-link raw stdout 文件、
`windows-event-log-source-audit-capture-envelope.v1`、外部 protocol ID/raw pin 与 pre/post config-ID 约束；
adapter 只从同一 file descriptor 读取一次 bytes，并复核 raw SHA/byte count/exit 0|2、empty stderr、
machine/boot capture binding、严格 compact ordered JSON、Audit v2/failure v1 分流、checkout/source pin、
canonical base64 content-ID basis、machine-config core cross-bind，以及 source/ACL/owner/channel/provider/
registry/overall/result/safety/refresh 的全部派生语义。内部一致的 exit 2 只生成 diagnostic snapshot；exit 3、
failure v1、Provision、mutation 或任何 claim/observation 不一致都 fail loudly。snapshot 永远
`projection_emitted=false / real_provisioner_observation=false /
eligible_as_host_qualification_input=false`，并固定所有 CUDA、formal、production ACTIVE 与四能力授权为
false。

该 binding 只是**文件内容自洽与调用者 envelope 相符**：envelope 自身固定
`capture_authoritative=false`，同仓库 hash 不是签名，raw Audit 不含 boot observation，adapter 没有启动或
持锁验证 Windows PowerShell、没有独立查询 registry/channel，也没有排除 pre/post replay、TOCTOU、
高权限改写或中途改回。后述 standalone owner 只在自己的 process boundary 中把 failure v1 隔离为
quarantine，仍需后续独立包实现真实 production acquisition、外部 release/WORM anchor 与 live reobservation，
再定义新的 production projection schema/predicate；
不能把本 snapshot 接入现有 projection v2。

v2 terminal 精确区分 `criteria_passed / real_host_observation /
eligible_as_host_qualification_input`，禁止 `passed` 与 `real_cuda_evidence_authorized`；terminal 自报 eligibility
不是权威，只有未来 production full validator 重算后的
`validated_eligible_as_host_qualification_input` 才能被 consumer 使用。当前 synthetic validator 即使整根
完整也固定返回 `real_host_observation=false /
validated_eligible_as_host_qualification_input=false`，并包含
`synthetic_test_backend_not_evidence`。`cuda_execution_authorized / formal_evidence_authorized /
production_active_authorized` 与四能力字段始终为 false。

microcode observation 必须把四字节 little-endian `20-01-00-00` 解码为整数 `0x120`，并按整数要求
`>=0x12F`，禁止字符串比较。Intel Defaults、物理冷启动、未恢复 XMP/OC/undervolt/memory tuning 与
same chassis 仍只是 content-addressed human declaration，不是签名或机器验证；资格根的 firmware observation 必须保留对应
machine-verified 字段为 false。原 host-block receipt 不含 MachineGuid/boot identity，因此无法机器证明
资格 attempt 与 block 当天来自同一物理 chassis；新 machine/boot identity 只可约束同一次 qualification
及未来 outer bridge。

Event Log source provisioning 是独立管理员控制面，qualification/campaign 禁止自动 `Provision`。当前脚本
`Audit v2` 把 exact source/channel 与 before/after endpoint 检查汇总为 `overall_conformant`：不合规在完整
receipt 写出后 exit 2，进程/观察/mutation failure 写 failure receipt 后 exit 3。`Provision` 在 source 缺失时
必须显式给出 `-AllowSourceCreation`，该开关只证明 operator intent，不证明首次 bootstrap；既有 drift 仍禁止
自动修复。source 注册后的 ACL/value/flush failure 是非事务性的，可能留下已注册 source，并会保守发布
refresh required。正常未创建路径的 `requires_cold_or_service_refresh=null` 只表示本调用没有建立 refresh
结论；创建路径为 true，仍必须在 cold boot/service refresh 后重新执行 fresh Audit。模块限定 cmdlet、程序集/
module hash 与两个端点相等都只是本进程观察。创建 source 的 provider membership 只接受列表不变（等待
refresh）或在原有有序集合上精确新增 `VolvenceEvidence`；任意其他新增、删除或替换都会使 Provision
nonconformant；列表不变的 Provision 成功仍不是 audit-ready，fresh Audit 必须在 before/after 都确认 exact
`VolvenceEvidence` membership。`cmdlet_provenance.authoritative=false` 且
`continuous_stability_proven=false`；不能排除中途
改回、同机高级篡改或普通本地主体伪造。当前 artifact adapter 已绑定完整 raw Audit 并复算 content-ID
basis；standalone acquisition/quarantine v2 已冻结 same-buffer requested source binding、hard-cutoff 状态机与
真实 Windows pipe/launcher fixture，但 production gate、realized executable attestation、独立重观测与
Provision→refresh→fresh Audit 时序仍未实现。

当前 outer protocol `cf62484f…3194` 只接受 terminal v1，并以 exact schema 拒绝 v2；它没有消费本包，
qualification handoff→outer baseline 的 Application/System 无缝 cursor bridge 仍属下一独立 consumer 包。
production state machine、004-before-process fsync、真实中断后的永久 `incomplete_consumed`、raw Audit/XML、
fixed real probe、Job Object containment、runtime identity 与独立/WORM anchor 均只是未来启用条件，不是本
synthetic writer 或 raw artifact adapter 已证明的物理行为。当前 qualification/provisioner 门与 59 项
acquisition v2 regression 证明 schema/integrity firewall、raw artifact 自洽复算、hard-cutoff/stream fixture 与
same-buffer launcher 的局部机制；没有运行 Provision/Audit 或 live Event Log，也没有宿主 PASS 或 BIOS block
解除。该交换保持 live `DISABLED`，Appendable / Readable / Learnable / Steerable 四轴均为
`not_proven`。回滚只需停止调用 synthetic helper/validator 与 raw-Audit adapter；没有 live wiring 需要切换，
已生成的 synthetic 根或 adapter snapshot 不得迁移成正式证据。

### 6.5 Windows Event Log source direct Audit acquisition scaffold

`windows_event_log_source_audit_acquisition` 是 `vz-runtime` offline evidence 域的独立 process-boundary
owner。它只面向显式 operator invocation；qualification/campaign 不得自动调用，因为 provisioner raw contract
仍明确发布 `qualification_or_campaign_invoked=false` 与
`automatic_invocation_by_qualification_or_campaign_forbidden=true`。固定 CLI 不接受 protocol、script、mode、
executable、argv、environment、backend 或 validator override；production public acquisition gate 在读取
options、path、environment 或 Proxy getter 前静态 fail closed。

该 owner 的 create-only 根只包含 000 claim、raw stdout/stderr streams 与 001 terminal。attempt 固定为 1，
retry 固定为 0；000 必须在 process creation 前写入并 `fsync`，两条 raw stream 在 spawn 前以 exclusive-create
方式打开。v2 process observation 明确区分 `child_exit_event / child_close_fallback / not_observed` 与
`child_close_event / not_observed`，未观察到的 exit/close 时间必须为 null；`child.kill()=true` 只表示请求被
接受，绝不推出 child 或 descendants 已终止。事件 callback 只做有界内存采集，任何 data/pipe error 都被冻结为
stream outcome，不允许异常逃出 EventEmitter。lifecycle finalize 后才把有界 prefix 写入、`fsync` 并从同一
`wx+` descriptor readback/hash；write/fsync 失败若仍能 readback 可封 quarantined terminal，readback 或 terminal
写入自身失败只能留下永久 `incomplete_consumed` 根。禁止删除、覆盖或在同一 root 重试。不同 parent 下的
cross-root duplicate 尚未由全局 lease/WORM registry 排除，仍不能声称 scope-global no-retry。

real backend 保持 module-private。异步监督冻结 120 秒 soft timeout、5 秒 post-kill pipe-drain grace 与
125 秒 overall hard cutoff；timer 保持 referenced，hard cutoff 无需等待 child `close`，会冻结 late-event
升级路径、destroy 本端 pipes、`unref` child 并封存 bounded prefix quarantine。child/pipe 的 `error` guardian
会保留到真实 child close，只发布 hash 化 warning，不能修改已冻结 observation，也不能升级 candidate。该上界只覆盖异步 child/pipe
监督；同步 write/fsync/readback/terminal create 不能被 Node timer 中断，因此不是整个 acquisition 的绝对 wall
deadline。没有 Job Object，`descendants_contained=false`。

固定 requested argv 已改为 nominal System32 Windows PowerShell 路径、`shell=false` 与
`-NoLogo -NoProfile -NonInteractive -EncodedCommand <protocol-derived-launcher>`。launcher 不接受 caller source、
mode 或 payload：它从 repository cwd 推导 reviewed provisioner，使用 `FileMode.Open / FileAccess.Read /
FileShare.Read` 同一 handle 做 2 MiB bounded read，分别核对 raw 与 strict-UTF8 LF-canonical SHA-256，拒绝 BOM，
再以 `Parser.ParseInput(exact_text, fileName).GetScriptBlock()` 执行固定 `-Mode Audit`。handle 贯穿绑定脚本执行
与 `exit` unwind，但 outer `finally` 会在 OS process exit event 前释放，因此不得声称持柄到进程退出。
Windows PowerShell 5.1 fixture 已机械验证 `$PSCommandPath/$PSScriptRoot`、exit 2、执行期间 rename 被拒与
退出后 release；正常 return 即使遗留 `LASTEXITCODE=0`、raw 漂移但 LF 相等、raw 相等但 LF pin 漂移、BOM 或
非法 UTF-8 均固定 exit 3。reviewed provisioner 由更具体的 `.gitattributes eol=lf` 规则保证 Windows fresh
checkout 仍产生协议冻结的 raw bytes；`.gitattributes` 自身也纳入 critical source pin。
provisioner 自身仍会在运行时拒绝非 Windows PowerShell 5.1 x64，但这只冻结 requested same-buffer binding；
PowerShell realized PE/version/image、IFEO/DLL/module、继承环境与管理员/内核
对手仍未 attested。production gate 禁用期间只有固定、白名单 synthetic/process fixture 可测试，不能注入任意
executor callback。

terminal 是单一 discriminated outcome。真实 candidate 额外要求 child exit event、child close、双 pipe end + close、
双流完整 persistence、无 timeout/overflow/signal/kill/spawn/capture/persistence error 与 hard cutoff；synthetic
candidate 只是一条显式 declaration。满足该 lifecycle 后，exit 0 + compact ordered Audit v2 + empty stderr 只产生
conformant capture candidate；exit 2 的完整 Audit v2 只产生 nonconformant diagnostic candidate；exit 3 +
failure v1 必须 quarantine。empty/invalid/oversized stdout、非空异常 stderr、schema/exit mismatch、source/
executable endpoint drift 以及任何 supervision/stream failure 均在 raw adapter 前隔离。failure arm 不得生成
source config snapshot、002/008 projection、capture success envelope 或 eligibility；success arm 也只发布
`capture_authoritative=false` envelope，仍须由 pure adapter 重算完整 Audit v2。

source pre-endpoint 必须 exact join protocol 的 provisioner raw/LF 双 pin，requested launcher 直接执行已校验
same-handle buffer；pre/post source 与 executable endpoint 仍只作 diagnostic equality。该设计关闭普通
provisioner path reopen 的内容窗口，却不能证明 launcher 被预期 PowerShell image 实际执行，也不能排除 IFEO、
realized image/argv/environment drift、伪造 `SystemRoot` 形状路径、管理员改写、ancestor reparse、hard-link 预存写柄
或 process descendants。production endpoint 必须是直接观察的 regular non-symlink file，但这一结构检查不等于
微软 image 身份或可信 `SystemRoot` 证明。
artifact hash 也不是 producer/operator 身份签名或 release/WORM anchor。因此 acquisition outcome 固定
`real_provisioner_observation=false / eligible_as_host_qualification_input=false / projection_emitted=false`，
CUDA、formal、production ACTIVE、tamper resistance 与 Appendable / Readable / Learnable / Steerable 授权
全部为 false；`acquisition_to_qualification` wiring 保持 `DISABLED`。

independent live reobserver 必须是后续不同 owner、不同进程与不同解析路径的收敛包：它不得读取 raw Audit、
adapter snapshot 或 002/008 projection来重建答案，而应通过只读 native Win32 registry/channel/SCM API 发布
自己当前观察的 frozen snapshot。即使该本地 reobserver 完成，也只能称另一代码路径复观测，不能称第三方或
独立信任权威；qualification admission consumer 还必须再以单独包 exact-join acquisition、reobserver、
machine/boot/config/time/role，并新建 production projection schema，不能翻转 synthetic projection v2。

## 7. 接线、晋升和回滚

`FinalRolloutConfig` 分别持有 `steering_sensor / steering_executor /
steering_gate`，均默认 SHADOW；env 覆盖为 `VZ_STEERING_SENSOR /
VZ_STEERING_EXECUTOR / VZ_STEERING_GATE`。有序晋升为：

1. sensor；
2. executor；
3. gate。

ACTIVE executor 必须先有 ACTIVE sensor。gate 未 ACTIVE 时，必须用
`steering_ungated_action={noop,always_on}` 显式命名临时对照臂；默认
`blocked` 会在构造期拒绝隐式选择。ACTIVE gate 必须已有 ACTIVE
executor。

B2 落地不授权 ACTIVE。B3 必须用新 prereg 依次通过 real-trace、
validation、gate-off（always-on/noop）、sensor-off（unconditional operator）、
rollback drill、latency/SLO、safety 与 `ModificationGate.OFFLINE`。缺任一
evidence 保持 SHADOW；专用统计门不得替代系统级 rare-heavy 修改门。

### C3 真实对话迁移证据

`scripts/run_dialogue_steering_test_plan.py` 是 C3 唯一 formal 控制面：

1. 先绑定 A1 的 `ablation_results.json`、`manifest.json`、
   `promotion_verdict.json` 与独立审计四件套；必须是 6 case / 36 run / 1260 turn、
   四组 N+1 contrast 覆盖完整且审计重算通过。A1 主效应可以是正或负终态，C3 依赖的是
   可用且已审计的 N+1 仪器；P4 仍单独要求 A1 `passed=true`。随后再绑定 canonical
   `data/external/msc/v0.1/extracted` 语料根旁的 v0.1 provenance、目标 Qwen weights、
   源码 hash、24 train/24 validation dyad、
   5 seeds×4 restarts、预算和阈值；
2. 在目标基底重新 fit reader、conditional executor 与 matched unconditional executor，
   reader/executor 随后冻结；
3. `msc-steering-shadow-collector-v1` 经完整 service/session/`propagate` 路径收集
   ≥500 validation turn，只落盘 owner observation、残差表示、hash lineage 与 latency，
   不保留原文；profile 将普通 companion 的生成式
   `semantic_proposal_channel=llm` 显式冻结进 attestation/config lineage，确保 C3
   SHADOW 轨迹与未来 production ACTIVE 的语义状态来源一致，不允许继承环境变量切成
   NoOp 后再把 gate 部署回 LLM 状态。C3 的 `max_length` 必须精确等于冻结模型声明的
   `max_position_embeddings`（当前 Qwen2.5-0.5B 为 32768）；preflight 对更小预算直接
   fail loudly，正式采集禁止截断完整 runtime prompt；
4. 同一冻结 N+1 head 结算 steer/noop/sensor-off；text-free trace 同时保留
   conditional / unconditional executor artifact id、共享 norm cap 与两臂 control norm，
   防止把未绑定或超预算的 counterfactual 冒充 sensor-off。gate 只用 train 侧选择
   restart，validation 做 dyad-clustered bootstrap 与 worst-seed 判决；
5. admission 同时要求 action sensitivity、收敛、相对 noop/always-on/random-gate 的
   gain、selectivity、冻结基底/reader/executor、无 bias、strict noop 和 R12。

信号不敏感时固定退出
`dialogue-n-plus-one-signal-insensitive-to-steering`；其余门失败固定退出
`proxy-level-when-to-steer-transfer-not-supported`，不得换 judge 或降低阈值。

### B3 独立晋升证据

`scripts/run_steering_promotion_test_plan.py` 必须在 C3 结果出现前冻结自己的 prereg；
创建新 prereg 时只要 C3 output 已出现 bundle/trace/report/manifest 任一正式产物就 fail loudly，
之后只读消费 C3 bundle/trace/report。B3 prereg 的源码指纹独立覆盖 sensor、gate、executor、
runtime kernel、session/brain、response/expression、transformers residual hook、service activation
consumer、bounded canary runner 与最终 wiring；不能只依赖 C3 prereg 间接冻结 production ACTIVE 链。
`SteeringPromotionEvidence` 把证据分给唯一组件：

- sensor：conditional 相对 matched unconditional 的 sensor-off 优势；
- executor：always-on conditional executor 相对 noop 的 N+1 优势；
- gate：learned policy 相对 noop 与 always-on 两个 gate-off 对照，并要求 C3 admission；
- shared：≥500 real validation turn、两条 informative 轴、逐轴相对改善 ≥15% **或**
  绝对改善 ≥0.02、checkpoint round-trip、latency、安全、owner-chain 与 R12。

B3 在候选 bundle 落盘并取得 content hash 后，必须另构造
`ModificationProposal(target="substrate.steering_artifact_bundle", desired_gate=OFFLINE)`：
`old_value_hash` 绑定 C3 bundle，`new_value_hash` 绑定 B3 candidate bundle；
`validation_delta` 是 informative held-out 轴中最小相对改善，未增加可达模型族所以
`capacity_cost=0`，rollback evidence 绑定 candidate hash 与 checkpoint JSON round-trip。
`contract_integrity / rollback_resilience / fallback_reliance` 只作为 read-only 发布门证据，
不写回 PE、credit 或 gate 学习。OA-4 业务 audit 尚未落地，故当前 prereg 必须显式记录
阶段一 `audit_required=false / audit_evidence_id=null`，不得伪称独立 audit 已完成；OA-4
ACTIVE 后另包迁移为 fail-closed required audit。

判词只产生 `sensor → executor → gate` 的连续 eligible prefix 和单字段 activation plan；
缺前件不能越级。B3 另存包含 C3 learned gate 的不可变 candidate bundle。executor 在
gate 仍 SHADOW 的中间态需要显式 `always_on`，因此 activation v3 把它拆成独立、当时仍
无用户效果的 `steering_ungated_action: blocked→always_on` 准备 rollout，再单独翻转
executor；gate ACTIVE 后再以独立 rollout 清回 `blocked`。正向和回滚每一步都只改变
一个字段，并且每个中间 `FinalRolloutConfig` 都可构造。该控制面不读取、调用或映射旧
`learned_active_gate` 的 ETA-off 条款，也不自动修改 production 默认。

部署侧由 `lifeform_service.steering_activation` 独立消费上述判词。普通
`--steering-artifact-bundle` 仍不足以启用任何 owner：非 evidence service 必须同时提供
`--steering-promotion-manifest`、`--steering-activation-plan` 与一个一基的
`--steering-activation-step`。reader 会逐字节验证 candidate bundle、activation plan 和
B3 manifest 的 SHA-256/ID/C3 prereg lineage，并复核同目录 promotion evidence/report、
`modification_gate_review`、candidate learned gate、安全/R12 字段；只有 exact
`ModificationGate.OFFLINE=allow` 且 reasons 为空才可继续。它重建正向及逆向单字段状态机，拒绝超出 eligible
prefix 的 step。部署契约还冻结 C3 同款 model/digest/dtype/layer/width、
`semantic_proposal_channel=llm`、context max length、generation token budget、
temperature=0 与 fail-on-truncation，第一阶段 ACTIVE 不得悄然换成
普通 service 的 512-token/temperature 0.7 配置。manifest 同时发布 B3 prereg 的完整
`source_sha256`；deployment reader 会从仓库根逐文件复算，并至少要求 CLI、activation reader、
final wiring 与 canary runner 在快照内，formal 后任一 ACTIVE 链源码漂移都会 fail closed。
service 只把所选 step 的
`FinalRolloutConfig` 与冻结生成预算交给 companion Brain/expression；bundle 单独出现、
伪造顺序、跳步、hash 漂移、非 companion vertical、非冻结 hf-shared 基底或
evidence/ACTIVE 混用都 fail loudly。授权启动还会拒绝进程环境中的任何
`VZ_STEERING_*` 或 `VZ_SEMANTIC_PROPOSAL_CHANNEL` override，避免 Brain 构造期在已验证
rollout config 之上再次抬高 wiring、替换 ungated action 或改变语义状态来源。canary
argv 必须显式携带已冻结 dtype；新 C3 prereg 默认冻结为经 A2 长 dyad 稳定性复现的
`float32`，禁止回落到 MPS 历史 `float16` 默认或已复现非有限 residual 的 `bfloat16`。
candidate bundle
可以继续携带 sensor-off 证据件，
但 executor 进入 ACTIVE 时 wiring 会剔除该 SHADOW-only 对照件。production 代码默认仍是
全 SHADOW。step 1 禁止 previous receipt；step 2 及以后必须提供
`steering-activation-canary-receipt.v1`，且它须逐字节绑定相同 manifest/plan/
ModificationGate review/candidate bundle、恰为 `step-1`、记录相邻前态与 companion
`127.0.0.1` 上的 `/v1/health=status:ok`。loader 从该已验证前态只应用当前 `single_field_flip`，因此不能从
baseline 直接请求累计 step 3。

`scripts/verify_steering_activation_canary.py` 是唯一正式 receipt 生成面：它持有共享 MPS 锁，
先拒绝已占用或非 loopback 的端点，再用同一 deployment contract 启动 bounded companion service；
健康端点通过且子进程仍存活时主动停止，并把退出码、exact argv、stdout/stderr、前序 receipt 与
rollout state 封入不可变 receipt。receipt 保存 exact argv 与日志/前序 receipt 的绝对路径，下一步
会从这些路径重算 command、stdout、stderr 与前序文件 SHA-256；仅保存不可反查的 digest 不算完成。
receipt 只证明启动健康与单字段 materialization，不是用户价值证据，也不改变 production 默认。

回滚按最小面进行：先翻转 gate→executor→sensor 对应的单字段为
SHADOW/DISABLED；若仅需停止额外 forward，关闭 `steering_shadow_hook`。未加载
bundle 时三个 owner 都不构造，现有 runtime 路径保持不变。

## 8. 当前验证与未完成门

定向契约测试覆盖：冻结 artifact lineage、lagged/fresh belief、PE 观测门、
norm cap、strict noop、SHADOW hook on/off、有序晋升防护、ACTIVE transformers
hook、active/shadow bus 隔离，以及 matched N+1 terminal PE→credit→gate、重复结算
拒绝、NOOP action-direction、head/target lineage 漂移拒绝、sensor-off matched preview、
text-free checkpoint、随机 gate 精确恢复、C3 pass/insensitive exit、A1 严格审计绑定、
sensor-off artifact/预算 lineage、B3 不越级、`ModificationGate.OFFLINE` teeth 与
activation v3 单字段中间态。
另覆盖 B3 manifest/evidence/ModificationGate review/report/plan/bundle 六方绑定、formal 后 ACTIVE
源码漂移、argv/日志/前序回执内容漂移拒绝、
系统门 BLOCK、无前序 receipt 跳步、receipt lineage 漂移、未授权 step 与 bundle hash 漂移拒绝，以及
service CLI 只向 companion 传递验证后的 ACTIVE rollout config。

尚未完成的项目不得被文档措辞隐藏：C3/B3 的实现与测试不等于 formal evidence；
当前仍缺 A1 前置终态、C3 新 prereg + 真实 run、B3 预先冻结 prereg + 正式判词。
