# VolvenceZero 距离 Cognitive AGI：当前完整评估

> Status: current assessment
> Last updated: 2026-07-17
> 口径：代码事实优先；明确区分“代码存在”“默认主路径生效”“晋升合格”“因果有效”“开放世界泛化”。
> 主要依据：`docs/specs/learned-vs-heuristic-coverage.md`、`docs/known-debts.md` #86–#91、`packages/vz-runtime/src/volvence_zero/integration/final_wiring.py`、`packages/vz-runtime/src/volvence_zero/agent/learned_active_gate.py`、`research/README.md`。

## 1. 总结

VolvenceZero 已经不是“只有认知骨架、学习代码尚未实现”的状态。2026-07-13 至 2026-07-17 的收敛工作已经补入：

- 默认路径中的 14 类 bounded-learned 部件；
- ndim metacontroller、torch runtime forward、autograd SSL、latent Internal RL；
- CMS torch SHADOW 双跑、晋升 readout 与回滚；
- PE / regime 外部结果校准、learned action-family matching、learned PE write gate；
- 真 PEFT LoRA rare-heavy backend；
- ingestion web / teaching-case 主链；
- ToM / common-ground prediction settlement 与 group durability PE loop；
- evaluation mid / expensive / cross-generation 实体实现；
- apprenticeship protocol revision 路径。

因此，旧结论“只有 6 处 learned”“R20 / ingestion / protocol apprenticeship 仍主要是空壳”“500-turn real trace 尚未形成”已经不完整。

但必须同时保留另一个事实：**默认 authoritative 路径仍主要由结构与启发式规则驱动**。核心 torch 学习后端仍默认 `DISABLED`，默认 `n_z=3`，默认 substrate 仍是 `synthetic`。代码已经就位不等于系统已经由 learned controller 主导。

当前最准确的四层判断是：

| 层次 | 当前完成度 | 剩余量 | 含义 |
|---|---:|---:|---|
| 架构、契约、owner、回滚骨架 | 约 85–90% | 约 10–15% | PE→credit→temporal→reflection→memory 主链、快照隔离、WiringLevel、ModificationGate 已真实存在 |
| 仓库定义的第一阶段认知系统代码 | 约 80–88% | **约 12–20% 关键代码** | 大部分 owner 与学习 backend 已写；仍缺若干决定“能否长期学习”的正式闭环 |
| 默认路径的 learned 决策主导度 | 约 10–20% | **约 80–90% 仍由结构/启发式主导** | learned 部件数量已增至 14 类，但 temporal authoritative path 与多项高影响决策仍未 learned 化 |
| 可外引因果证据 | 约 5% | 约 95% | harness 与 gate 已齐；尚无 P2 `first-stage-retained` verdict |

这四个百分比的分母不同，不能相互替代：

- “第一阶段代码完成 80–88%”回答：还要写多少关键实现。
- “learned 主导度 10–20%”回答：今天按默认配置启动时，多少适应行为真的由学习参数决定。
- “证据约 5%”回答：多少主张已经被 matched-control、held-out、multi-seed 结果支持。
- Cognitive AGI 本身不能用单一工程百分比表示，因为开放环境和跨域迁移仍包含研究前沿问题。

## 2. 已经完成的骨架

以下不是文档占位，而是当前真实代码资产：

1. **契约式运行时**
   - 每个正式状态有 owner；
   - 跨模块通过 frozen snapshot；
   - `propagate` DAG、依赖声明、slot 注册、ACTIVE / SHADOW / DISABLED 三态均已存在。

2. **PE 主链**
   - prediction、actual、prediction error、credit 是正式运行时对象；
   - PE 是学习原始信号，evaluation 保持 readout-only；
   - external outcome、action context、social prediction error 均有 typed 路径。

3. **多时间尺度**
   - online-fast、session-medium、background-slow、rare-heavy 有明确 owner 和编排边界；
   - CMS、session post slow loop、reflection、rare-heavy import / rollback 均有代码。

4. **认知状态**
   - World / Self 双轨；
   - regime；
   - 9 个 semantic owner；
   - ToM、common ground、conversational role、multi-party identity、group owner；
   - vitals、open loop、commitment、boundary / consent。

5. **自修改与安全**
   - ModificationGate；
   - checkpoint；
   - rollback drill；
   - safety gate；
   - frozen substrate + bounded adapter delta；
   - PEFT LoRA rare-heavy backend。

6. **证据编排**
   - 9-track same-substrate roster；
   - raw / ref-harness / CAMEL / volvence-cold / volvence；
   - PE-off / ETA-off / active-learning-off / LoRA-adapter 组件臂；
   - learned ACTIVE gate 与 production verdict evaluator。

## 3. 骨架仍差的 10–15%

“骨架接近完成”不等于所有结构路径已闭合。当前仍有以下结构缺口：

### 3.1 多人社会状态没有完整的长期 owner 生命周期

当前：

- `multi_party_identity`、`conversational_role`、ToM、common ground 已接入；
- 默认产品 turn 仍主要构造 `primary/self` 单主体 frame；
- `SocialRecordStore` 是 session-held；
- ToM、common-ground、group durability 不自动跨 session 恢复；
- `groups` 默认仍为 `SHADOW`；
- `UserModelSnapshot` 仍偏单桶，缺少完整 per-interlocutor keyed SSOT。

应补：

- per-interlocutor typed owner state；
- product-facing multi-party `EnvironmentEvent` 注入路径；
- SocialRecordStore owner-owned persistence / hydrate contract；
- group owner 的 ACTIVE 晋升路径；
- wrong-person attribution、audience / confidentiality、witness / overhearer 的正式消费闭环。

### 3.2 Thinking loop 尚未闭合到 temporal controller

当前：

- Thinking scheduler、worker、artifact、fingerprint guard 已实现；
- companion 默认未完整注册 thinking factory；
- advisory 主要以 `apply_enabled=False` 记录；
- 中频结果没有成为 β_t / z_t authoritative 输入。

应补：

- companion / production session 接入 thinking adapter；
- thinking owner 发布 compact advisory；
- temporal owner 消费 advisory，但不让 thinking 成为第二 temporal owner；
- SHADOW 比较 advisory-off / advisory-on；
- stale fingerprint 必须继续 fail closed。

### 3.3 深层 evaluation 尚未成为默认可观测级联

当前：

- cheap evaluation ACTIVE；
- mid、expensive、cross-generation 代码已实体化；
- 默认仍 `DISABLED` 或未进入主 DAG；
- persona geometry 目前主要是 residual norm 与 cosine drift 入口。

应补：

- mid / expensive / cross-generation 注册为正式 SHADOW readout；
- learned persona / function direction 基底；
- 跨 generation 漂移、异常方向、mesa-objective 候选 readout；
- 保持 R12：只读，不回灌为 reward。

### 3.4 protocol slow loop 仍未全部晋升

`apprenticeship_protocol_alignment`、`protocol_reflection`、`protocol_revision_queue` 仍以 SHADOW 为主。代码存在，但 protocol lineage conflict 到长期策略更新的完整生产闭环尚未晋升。

## 4. 仍差的 12–20% 关键代码

这里仅列“测试跑完也不会自动出现、必须继续写”的实现。

### P0：决定系统是否能成为长期 learned organism

#### P0-1 Learned regime selector

根因：

- `regime/scoring.py` 仍以大量固定系数决定六类 regime；
- selection weights 与 external-outcome calibration 只是在规则骨架外围做有界调整；
- 用户可见的认知体制仍主要由人工特征工程决定。

目标实现：

- Regime owner 内部持有 trace-conditioned learned scorer；
- 输入只读 PE、dual-track、memory、temporal、semantic owner snapshots；
- 输出 learned candidate score 与 baseline score；
- SHADOW 双跑，不直接替换当前 scorer；
- settlement 使用 delayed outcome / PE-derived credit；
- checkpoint、kill condition、rollback 均 owner-local；
- 禁止 evaluation 成为学习源。

退出条件：

- ≥500 turn 真 trace；
- learned scorer 相对 baseline 的 held-out selection loss / delayed payoff 有稳定增益；
- 不增加 unsafe regime switching；
- reset 后恢复当前固定 scorer。

#### P0-2 全部 adaptive state 的跨 session 连续性

当前 memory、semantic state、vitals、followup、protocol registry 已有不同形式的持久化，但以下状态没有统一闭合：

- social record / ToM / common ground / group durability；
- regime live learned state；
- world / self temporal learned controller state；
- PE critic、credit head、dual-track gate 等 session-held learned 参数。

正确做法不是让外部 store 直写内部字段，而是：

- 每个 owner 自己导出 typed persistence snapshot 或正式 checkpoint；
- runtime 只负责保存、加载和验证 fingerprint；
- temporal state 继续由 temporal checkpoint owner 管理，不复制进 generic hydration；
- social state 由 social owner 自己描述和恢复；
- schema mismatch fail loudly；
- 每个 owner 有 seed-once、version、rollback 与 user scope。

退出条件：

- 同一用户跨 20 sessions 的 owner continuity；
- 无跨用户泄漏；
- checkpoint round-trip；
- 旧 schema typed migration；
- rollback 后回到上一稳定 generation。

#### P0-3 多人社会认知正式产品路径

需要补：

- 默认产品入口可传真实 speaker / addressee / subject / audience；
- per-interlocutor semantic state；
- ToM / common ground 不再依赖单一 primary/self fallback；
- group owner ACTIVE 前的 typed consumer；
- social prediction error 能区分 wrong-person、wrong-audience、broken-common-ground。

退出条件：

- 三人以上跨 session 场景中不发生错人归因；
- ToM / common-ground records 非空且 subject-key 正确；
- groups ACTIVE 后不改变单主体行为；
- rollback 到 SHADOW 可恢复旧路径。

#### P0-4 Thinking → temporal 的中频闭环

需要补：

- `ThinkingArtifact` 由 thinking owner 压缩为 immutable advisory；
- temporal owner 将 advisory 作为输入特征，而非被 thinking 直接修改；
- advisory 的影响必须可解释、可关闭、可回滚；
- stale artifact 永不 apply。

退出条件：

- advisory-on 相对 advisory-off 在长程 open-loop closure / plan consistency 上有增益；
- β_t 不出现高频抖动；
- latency 不进入实时 turn critical path。

### P1：决定 learned 部件能否成为行为主体

#### P1-1 Learned affordance / action selection

当前 affordance 选择仍包含 hash·z_t 投影与固定阈值。需要由 temporal owner 的 learned action-family / affordance head 输出候选，AffordanceModule 只消费 snapshot 并执行 safety gate。

#### P1-2 完整的 World / Self predictive model

当前主要是小型线性 head、低维向量、规则状态机和 bounded forecast。需要：

- 更高容量 latent state；
- compositional prediction；
- counterfactual rollout；
- delayed outcome attribution；
- World / Self 分轨训练与 checkpoint；
- 不退回 token-space RL。

#### P1-3 Semantic owner proposal 覆盖

LLM structured proposal 目前主要覆盖 5/9 owner；`plan_intent`、`open_loop`、`execution_result`、`belief_assumption` 等仍需正式 proposal / event adapter 覆盖，并保持 owner 单写者。

#### P1-4 Learned reflection / consolidation policy

reflection 中 promotion、decay、policy consolidation、tension 和 lesson 提取仍以固定阈值为主。需要先做 SHADOW learned candidate，不能让 reflection 外部重建 memory / regime 内部状态。

### P2：决定系统是否有研究级安全与开放性

- learned persona / function vectors；
- mesa-objective 异常检测；
- 跨模态 latent action basis；
- 开放环境中的因果结构发现；
- 少于 10 次有效反馈的跨域迁移。

这些不是普通 backlog，部分仍是 2026–2028 研究前沿。

## 5. 默认 learned 主导度为什么仍只有 10–20%

当前代码已经登记 14 类 learned 部件，但高影响路径仍然不同：

### 已在默认路径实际参与适应

- `_PELearnedCritic`；
- `_RewardingStateHead`；
- CMS `LearnedUpdateRule`；
- PE write gate；
- semantic owner forecasts；
- ToM / common-ground prediction settlement；
- learned action-family matching；
- β_t threshold calibration；
- external outcome / alignment calibration；
-部分 regime outcome calibration。

### 代码存在但不主导 live decision

- torch metacontroller；
- torch runtime forward；
- autograd SSL；
- torch Internal RL；
- CMS torch backend；
- DualTrackGateLearner 的晋升候选；
- GateRiskLearner；
- ScheduleGateLearner；
- evaluation mid / expensive / cross-generation。

### 仍主要 hand-crafted

- legacy 3 维 z_t recurrence；
- β_t gate 公式；
- regime 主 scorer；
- reflection / consolidation thresholds；
- memory retrieval ranking；
- dual-track 多源融合权重；
- affordance score；
-部分 social / semantic lifecycle；
- safety gate 的底线规则级联。

其中 safety / ModificationGate 的一部分规则应永久保留，因为它们是架构约束；不能为了提高 learned 百分比而把所有安全边界学习化。

## 6. 七个 cognitive primitive 的最新代码状态

| Primitive | 代码状态 | 默认 authoritative 状态 | 仍缺 |
|---|---|---|---|
| Frozen substrate | HF residual hook、冻结、LoRA 已有 | 默认 `synthetic` | 生产 profile 强制真 substrate；跨 substrate 隔离 |
| Latent controller | ndim GRU / torch runtime 已有 | 默认 `n_z=3` legacy | runtime backend 晋升；容量曲线 |
| Emergent switching | STE switch、SSL 已有 | 默认公式 + learned threshold | learned switch 成为 authoritative |
| Multi-timescale memory | CMS、PE gate、ATLAS、torch backend 已有 | CPU learned gate ACTIVE，torch DISABLED | 长程抗遗忘与 owner continuity |
| Epistemic PE | PE owner、分解、18 维 critic 已有 | ACTIVE，但容量小 | LLM-scale 稳定估计、外部锚 |
| Bounded self-modification | gate、checkpoint、LoRA、rollback 已有 | 规则 gate + builtin fallback | 真 LoRA 周期、跨代证据 |
| Read-only monitoring | cheap + mid/expensive code已有 | cheap ACTIVE，深层未接 | learned geometry、mesa-objective readout |

## 7. 晋升不是“把默认值改成 ACTIVE”

四个 learned backend 的代码已经存在：

1. `temporal_runtime_backend`
2. `temporal_ssl_backend`
3. `internal_rl_backend`
4. `cms_torch_backend`

它们必须按这个顺序逐个晋升：

```text
runtime → SSL → Internal RL → CMS torch
```

原因：

- SSL 必须建立在已晋升 runtime representation 上；
- Internal RL 必须建立在 runtime + SSL 都稳定的 latent space 上；
- CMS torch 可独立验证，但仍需与 PE / temporal 的真实 trace 对齐；
- 一次 flip 多个组件会失去因果归因和回滚边界。

### 7.1 每个组件的硬 gate

`evaluate_learned_active_candidate(...)` 对所有组件要求：

- `real_trace_turns >= 500`
- `validation_delta >= 0.02`
- `strict_eta_gate_passed`
- `pe_off_control_direction_correct`
- `eta_off_control_direction_correct`
- `rollback_drill_passed`
- `latency_slo_ok`
- `safety_gate_ok`

额外要求：

- SSL：`prior_runtime_active`
- Internal RL：`prior_runtime_active`、`prior_ssl_active`、`internal_rl_no_reward_leakage`
- CMS torch：`cms_retention_non_degrading`、`cms_absorption_improved`

任何一项缺失，结果必须保持 SHADOW / DISABLED，不能人工解释为通过。

### 7.2 当前证据状态

截至 2026-07-17：

- learned-shadow smoke、HF 路径、SHADOW dual-run 与 rollback 工具已存在；
- 已产生连续 509 real-trace 的 artifact，证明 long-soak 代码路径可运行；
- 这不自动代表四个组件满足 `validation_delta`、控制臂、latency、安全和 CMS 双指标；
- 9-track P1/P2 harness 已完成；
- 仍没有 P2 held-out multi-seed 的 `first-stage-retained` verdict；
- 因此默认四 backend 保持 `DISABLED` 是正确的 R15 状态。

## 8. 晋升和验证的实际执行方法

### 8.1 先运行可恢复的完整 evidence pipeline

Apple Silicon：

```bash
bash run_learned_active_evidence.sh \
  --resume \
  --substrate-mode hf \
  --substrate-device mps
```

Linux / CUDA：

```bash
bash run_learned_active_evidence.sh \
  --resume \
  --substrate-mode hf \
  --substrate-device cuda
```

该 runner 依次执行：

```text
shadow-smoke
platform-chunked-soak
real-soak
capacity-ladder
same-substrate-ablation
build-promotion-evidence
evaluate-promotion
```

注意：

- chunked soak 只证明平台稳定性，不计作 ACTIVE promotion evidence；
- promotion 必须使用连续 HF real-soak；
- `--resume` 只跳过已有且可解析的 artifact；
- 只有明确需要重跑某阶段时才使用 `--force-stage`。

### 8.2 跑 capacity → gain

```bash
bash run_learned_active_evidence.sh \
  --only capacity-ladder \
  --force-stage \
  --execute-capacity \
  --capacity-n-z 16,64,256 \
  --capacity-turns 500 \
  --substrate-mode hf \
  --substrate-device mps
```

必须回答：

- 容量提升是否带来 held-out gain；
- gain 是否仅来自参数量；
- latency / memory cost 是否越界；
- 16→64→256 是否出现饱和或退化；
- 不能只报告最优点，必须保留完整曲线和 seed。

### 8.3 跑 P1 9-track directional ablation

```bash
bash run_companion_bench_p1.sh --resume
```

预期产物：

```text
artifacts/companion-ablation/<run-id>/verdict_p1.json
```

P1 只能给：

- directional weak-positive；
- directional kill signal；
- 工程与 judge robustness 诊断。

P1 单 seed / public-only 不能产生 `first-stage-retained`。

### 8.4 把 P1 verdict 接入 promotion report

```bash
bash run_learned_active_evidence.sh \
  --resume \
  --substrate-mode hf \
  --substrate-device mps \
  --ablation-verdict artifacts/companion-ablation/<run-id>/verdict_p1.json
```

查看：

```text
artifacts/learned_active_evidence/promotion/promotion_report.json
```

目标：

```json
{"all_eligible": true}
```

若为 `false`：

- 读取每个 component 的 `missing_gates`；
- 只修对应 gate；
- 不删除失败 artifact；
- 不为了变绿放宽门槛；
- 重跑对应阶段，不直接重跑全部。

### 8.5 P2 held-out multi-seed

P2 至少需要：

- held-out scenarios；
- 多 seed；
- 同 substrate fingerprint；
- 相同 prompt / context / tool budget；
- cross-family judge；
- judge-bias sanity；
- blinded human anchor；
- relationship continuity；
- longitudinal 20-session；
- uncertainty interval 与 pairwise effect。

只有 P2 满足 frozen claim registry，production verdict 才能进入：

```text
first-stage-retained
```

否则只能是：

```text
inconclusive
product-companion-retained
architecture-platform-only
```

## 9. 真正 flip ACTIVE 的发布流程

即使 `promotion_report.json.all_eligible=true`，也不能一次修改四个默认值。

每个组件单独做一个有界收敛包：

### Step 1：冻结证据

- git SHA；
- substrate model / weights fingerprint；
- run manifest；
- trace provenance；
- promotion report；
- checkpoint；
- rollback drill artifact；
- latency / safety report。

### Step 2：只改一个组件

顺序：

1. `temporal_runtime_backend=ACTIVE`
2. `temporal_ssl_backend=ACTIVE`
3. `internal_rl_backend=ACTIVE`
4. `cms_torch_backend=ACTIVE`

其余组件保持上一稳定状态。

### Step 3：canary / SHADOW 对照

- 小 cohort；
- 同时保留 rollback baseline；
- 比较 snapshot、PE distribution、regime switching、memory retention、latency；
- 不能只看用户文本评分。

### Step 4：退出条件

- promotion gate 继续满足；
- 无 safety regression；
- 无 reward leakage；
- 无跨用户状态污染；
- latency / memory SLO 满足；
- learned candidate 在真实 held-out 上不劣化；
- rollback 可在一个 release 动作内完成。

### Step 5：回滚

出现以下任一情况立即回滚：

- validation delta 转负；
- PE / ETA control direction 反转；
- unsafe regime switching；
- Internal RL reward leakage；
- CMS retention 退化或 absorption 无提升；
- snapshot schema / fingerprint 不匹配；
- latency SLO 破坏；
- learned state 无法恢复。

回滚动作：

- 当前组件恢复 `SHADOW` 或 `DISABLED`；
- 恢复晋升前 checkpoint；
- 保留失败 generation 与 trace；
- 不用旧组件之外的模块补丁掩盖失败；
- 将失败原因写入 promotion artifact / known debt。

## 10. 代码补全与晋升的优先顺序

### 近期：先完成真正缺失的 P0 代码

1. social / regime / temporal adaptive state 的 owner-owned continuity；
2. learned regime scorer SHADOW；
3. multi-party product frame + per-interlocutor state；
4. thinking → temporal advisory SHADOW；
5. deep evaluation 注册 SHADOW。

### 与代码并行：完成晋升证据

1. 读取现有 509 real-trace artifact 的 `missing_gates`；
2. capacity ladder；
3. P1 9-track；
4. build / evaluate promotion；
5. runtime backend 单组件晋升；
6. SSL；
7. Internal RL；
8. CMS torch。

### 中期：让 learned 成为行为主体

- regime / affordance / reflection learned 化；
- World / Self predictive model 扩容；
- social cognition 跨 session；
- learned geometry monitor；
- P2 held-out multi-seed。

## 11. 不能混淆的四个结论等级

```text
wiring-ready
```

代码、schema、脚本和 SHADOW 链路可运行。

```text
promotion-eligible
```

某个 learned backend 满足 ACTIVE gate，可以进入单组件 canary。

```text
first-stage-retained
```

P2 matched-control 证明完整 Volvence learned/controller 层在当前人类关系域中稳定优于标准方案。

```text
cognitive AGI thesis stronger
```

还需要长程、多域、跨 seed、跨 substrate、开放环境和迁移结果。

当前整体状态是：

- 骨架：高完成度；
- 第一阶段代码：接近但未完成，剩余 12–20% 关键代码；
- learned 主导：仍低；
- 晋升：代码路径齐，证据 gate 未全部通过；
- first-stage thesis：尚未 retained；
- cognitive AGI：仍远，且包含未解研究问题。

## 12. 最终判断

VolvenceZero 当前最有价值的资产是：**认知系统的 owner、契约、时间尺度、PE、回滚和证伪边界已经搭得足够完整，可以开始认真验证 learned thesis。**

当前最大风险不是“没有组件”，而是：

1. 把代码存在误写成 learned 已主导；
2. 把 SHADOW parity 误写成能力增益；
3. 把 P1 directional 结果误写成 thesis retained；
4. 继续扩展新 scaffold，却不补跨 session continuity、learned regime 和 thinking→temporal；
5. 为了快速 ACTIVE 而放宽既有 gate。

最诚实的表述是：

> 第一阶段认知系统代码约完成 80–88%，剩余 12–20% 是长期 owner continuity、learned regime、多人社会认知、thinking→temporal 与深层监控等关键闭环；核心 learned backend 的实现与晋升管线已就位，但默认行为仍主要由结构与启发式控制。只有逐组件通过 ≥500 real-trace、validation delta、控制臂、回滚、性能、安全、P2 held-out multi-seed 和人类锚点后，才能从“架构正确”进入“第一阶段学习 thesis 被保留”。
