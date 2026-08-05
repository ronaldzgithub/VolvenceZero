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
SHADOW 值。该隔离由 runtime kernel 统一实现，不是 owner 私有 fallback。

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
rollback drill、latency/SLO 与 safety 门。缺任一 evidence 保持 SHADOW。

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
   不保留原文；
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
consumer 与最终 wiring；不能只依赖 C3 prereg 间接冻结 production ACTIVE 链。
`SteeringPromotionEvidence` 把证据分给唯一组件：

- sensor：conditional 相对 matched unconditional 的 sensor-off 优势；
- executor：always-on conditional executor 相对 noop 的 N+1 优势；
- gate：learned policy 相对 noop 与 always-on 两个 gate-off 对照，并要求 C3 admission；
- shared：≥500 real validation turn、两条 informative 轴、逐轴相对改善 ≥15% **或**
  绝对改善 ≥0.02、checkpoint round-trip、latency、安全、owner-chain 与 R12。

判词只产生 `sensor → executor → gate` 的连续 eligible prefix 和单字段 activation plan；
缺前件不能越级。B3 另存包含 C3 learned gate 的不可变 candidate bundle。executor 在
gate 仍 SHADOW 的中间态需要显式 `always_on`，因此 activation v2 把它拆成独立、当时仍
无用户效果的 `steering_ungated_action: blocked→always_on` 准备 rollout，再单独翻转
executor；gate ACTIVE 后再以独立 rollout 清回 `blocked`。正向和回滚每一步都只改变
一个字段，并且每个中间 `FinalRolloutConfig` 都可构造。该控制面不读取、调用或映射旧
`learned_active_gate` 的 ETA-off 条款，也不自动修改 production 默认。

部署侧由 `lifeform_service.steering_activation` 独立消费上述判词。普通
`--steering-artifact-bundle` 仍不足以启用任何 owner：非 evidence service 必须同时提供
`--steering-promotion-manifest`、`--steering-activation-plan` 与一个一基的
`--steering-activation-step`。reader 会逐字节验证 candidate bundle、activation plan 和
B3 manifest 的 SHA-256/ID/C3 prereg lineage，并复核同目录 promotion evidence/report、
candidate learned gate、安全/R12 字段；它重建正向及逆向单字段状态机，拒绝超出 eligible
prefix 的 step。部署契约还冻结 C3 同款 model/digest/layer/width、context max length、
generation token budget、temperature=0 与 fail-on-truncation，第一阶段 ACTIVE 不得悄然换成
普通 service 的 512-token/temperature 0.7 配置。service 只把所选 step 的
`FinalRolloutConfig` 与冻结生成预算交给 companion Brain/expression；bundle 单独出现、
伪造顺序、跳步、hash 漂移、非 companion vertical、非冻结 hf-shared 基底或
evidence/ACTIVE 混用都 fail loudly。授权启动还会拒绝进程环境中的任何
`VZ_STEERING_*` override，避免 Brain 构造期在已验证 rollout config 之上再次抬高
wiring 或替换 ungated action。candidate bundle 可以继续携带 sensor-off 证据件，
但 executor 进入 ACTIVE 时 wiring 会剔除该 SHADOW-only 对照件。production 代码默认仍是
全 SHADOW；每次部署只推进 plan 中一个 step。

回滚按最小面进行：先翻转 gate→executor→sensor 对应的单字段为
SHADOW/DISABLED；若仅需停止额外 forward，关闭 `steering_shadow_hook`。未加载
bundle 时三个 owner 都不构造，现有 runtime 路径保持不变。

## 8. 当前验证与未完成门

定向契约测试覆盖：冻结 artifact lineage、lagged/fresh belief、PE 观测门、
norm cap、strict noop、SHADOW hook on/off、有序晋升防护、ACTIVE transformers
hook、active/shadow bus 隔离，以及 matched N+1 terminal PE→credit→gate、重复结算
拒绝、NOOP action-direction、head/target lineage 漂移拒绝、sensor-off matched preview、
text-free checkpoint、随机 gate 精确恢复、C3 pass/insensitive exit、A1 严格审计绑定、
sensor-off artifact/预算 lineage、B3 不越级与 activation v2 单字段中间态。
另覆盖 B3 manifest/plan/bundle 三方绑定、未授权 step 与 bundle hash 漂移拒绝、以及
service CLI 只向 companion 传递验证后的 ACTIVE rollout config。

尚未完成的项目不得被文档措辞隐藏：C3/B3 的实现与测试不等于 formal evidence；
当前仍缺 A1 前置终态、C3 新 prereg + 真实 run、B3 预先冻结 prereg + 正式判词。
