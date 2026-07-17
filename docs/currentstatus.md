# VolvenceZero Cognitive AGI 当前状态

> Status: live status summary
> Last updated: 2026-07-17
> 详细判断、晋升协议与命令见 [`current.md`](./current.md)。
> 本文件只记录当前事实、剩余代码、晋升状态和下一步，不把计划写成已完成。

## 1. 当前总状态

| 维度 | 当前状态 | 完成度 | 还差什么 |
|---|---|---:|---|
| 架构 / 契约 / owner / 回滚骨架 | 高完成度 | 约 85–90% | social continuity、thinking→temporal、deep evaluation、protocol slow loop |
| 第一阶段认知系统代码 | 大部分已实现 | 约 80–88% | **12–20% 关键代码**，见第 4 节 |
| 默认 learned 决策主导度 | 较低 | 约 10–20% | temporal authoritative learned path、regime / reflection / affordance learned 化 |
| learned backend 实现 | 已就位 | 约 70–80% | 主要剩晋升、默认接管与少量 owner 生命周期 |
| 晋升证据 | 部分就绪 | 尚未全绿 | promotion report 的 component gates |
| thesis 因果证据 | harness-ready | 约 5% | P1 directional + P2 held-out multi-seed |
| 开放世界 cognitive AGI | 未开始证明 | 不适用 | 跨域、跨模态、因果结构发现、mesa-objective detection |

## 2. 本轮已经取得的代码进展

### Learned 部件从 6 类扩展到 14 类

当前 coverage 登记的主要 bounded-learned 部件包括：

1. PE learned critic；
2. COCOA rewarding-state head；
3. CMS learned band gate；
4. DualTrackGateLearner；
5. semantic owner forecast；
6. ToM / common-ground prediction settlement；
7. learned action-family matching；
8. learnable β_t threshold；
9. PE external-outcome / AAC calibration；
10. regime outcome calibration；
11. GateRiskLearner / ScheduleGateLearner；
12. group durability learned score；
13. CMS torch SHADOW settle / promotion readout；
14. PEFT LoRA rare-heavy backend。

其中部分是 ACTIVE，部分仍是 SHADOW / report-only，不能将“14 类代码存在”解释为“14 类均已主导运行时”。

### 2026-07-16 至 2026-07-17 主要闭合项

- `temporal_profile="learned-ndim"` 可实例化 ndim controller；
- torch metacontroller、SSL、Internal RL、CMS torch 的 owner-local 晋升路径已齐；
- action-family matching learned 化；
- PE / regime 静态表降为 learned calibrator 的初始化与回滚点；
- PE memory write gate learned 化；
- DualTrackGateLearner 增 promotion / checkpoint / rollback；
- evaluation mid / expensive / cross-generation 从空壳变为实体 readout；
- semantic embedding 多 substrate 隔离代码；
- rare-heavy 真 PEFT LoRA backend；
- apprenticeship protocol conflict → typed revision proposal；
- web ingestion 与 teaching-case 路径；
- R20 group durability prediction / settlement；
- DLaaS evaluation 真实现。

### 证据执行面进展

- learned-shadow smoke 与四 backend SHADOW profile 已有；
- 连续 509 real-trace artifact 已形成；
- 9-track same-substrate serving roster 已形成；
- P1 / P2 driver、fingerprint、claim registry、promotion evaluator 已有；
- 当前仍无 P2 `first-stage-retained` verdict。

## 3. 默认 authoritative 路径事实

### 当前默认配置

```text
substrate_mode = synthetic
temporal_latent_dim = 3
temporal_ssl_backend = DISABLED
temporal_runtime_backend = DISABLED
internal_rl_backend = DISABLED
cms_torch_backend = DISABLED
```

因此默认运行时仍是：

- synthetic substrate；
- legacy 3 维 z_t recurrence；
- 手写 β_t gate 公式；
- analytic / heuristic Internal RL baseline；
- pure-Python CMS band update；
- learned backend 不写 live state。

### 默认已经 ACTIVE 的核心结构

- substrate / memory / retrieval；
- dual-track；
- prediction error；
- credit；
- regime；
- reflection；
- 9 semantic owners；
- multi-party identity / conversational role；
- ToM 四 owner；
- common ground；
- rupture state / interlocutor state；
- apprenticeship alignment；
- session post slow loop；
- owner hydration（有 persistence backend 时）。

### 仍 SHADOW / DISABLED 的关键项

| 项 | 默认状态 | 说明 |
|---|---|---|
| `apprenticeship_protocol_alignment` | SHADOW | protocol 层只比较 / 提案，不进 active chain |
| `groups` | SHADOW | group PE 代码已有，尚未晋升 |
| `protocol_reflection` | SHADOW | background protocol reflection 未晋升 |
| `protocol_revision_queue` | SHADOW | 人审队列路径未晋升 |
| `protocol_temporal_prior` | DISABLED | protocol mixture 不进入 β_t |
| `audit` | SHADOW | audit owner 非 authoritative |
| temporal runtime / SSL / Internal RL / CMS torch | DISABLED | 四个核心 learned backend |
| evaluation mid / expensive / cross-generation | DISABLED / off-path | 代码已实体化，默认深层级联未接管 |

## 4. 剩余 12–20% 关键代码

这些不是“测试跑完就会完成”的项目。

### P0-1：learned regime selector

当前：

- 六 regime 主评分仍是固定系数特征工程；
- learned selection weight 只做外围校准。

要做：

- Regime owner 内 trace-conditioned learned scorer；
- baseline / learned SHADOW 双跑；
- delayed PE-derived settlement；
- checkpoint / reset / kill condition；
- evaluation 只读，不作为训练源。

完成标准：

- ≥500 turn held-out 增益；
- 无 unsafe switching；
- 可恢复到当前 scorer。

### P0-2：全部 adaptive owner 的跨 session continuity

当前已持久化：

- memory；
- semantic state；
- vitals；
- followup；
- protocol registry。

仍需闭合：

- SocialRecordStore / ToM / common ground / group durability；
- regime live learned state；
- temporal learned controller checkpoint 自动续接；
- PE / credit / dual-track 等 learned heads 的关系级参数生命周期。

约束：

- 每个 owner 自己导出 snapshot / checkpoint；
- runtime 只负责保存与加载；
- 禁止 generic store 直写内部字段；
- temporal 不复制为第二 owner；
- user scope、schema version、fingerprint、rollback 必须完整。

### P0-3：多人社会认知产品路径

当前：

- 默认 turn 仍主要是 `primary/self`；
- ToM / common ground 有 owner，但无 LLM / EnvironmentEvent 时 fail-closed；
- per-interlocutor semantic state 不完整；
- groups 仍 SHADOW。

要做：

- speaker / addressee / subject / audience 的正式产品输入；
- per-interlocutor keyed owner；
- social state 跨 session；
- wrong-person / wrong-audience PE；
- group owner ACTIVE consumer。

### P0-4：thinking → temporal 闭环

当前：

- thinking scheduler / worker / fingerprint guard 已有；
- companion 默认未完整接入；
- advisory 不影响 β_t / z_t。

要做：

- production thinking factory；
- thinking owner 发布 compact advisory；
- temporal owner 消费；
- SHADOW 双跑；
- stale artifact fail closed；
- 不阻塞实时 turn。

### P1-1：learned affordance selection

当前仍含 hash·z_t score 与固定阈值。目标是 temporal owner 发布 learned affordance/action-family candidate，AffordanceModule 只负责 registry、safety、rate limit、schema 和执行。

### P1-2：World / Self predictive model 扩容

当前是小型 head、低维向量、规则状态机与 bounded forecast。仍需：

- compositional latent state；
- counterfactual rollout；
- delayed outcome attribution；
-高容量但有界的 World / Self 分轨学习；
- checkpoint 与 kill condition。

### P1-3：9 semantic owner proposal 覆盖

当前 LLM structured path 主要覆盖 5/9 owner。需要补 `plan_intent`、`open_loop`、`execution_result`、`belief_assumption` 等正式 typed proposal source，但最终写入仍由各 owner 单写。

### P1-4：learned reflection / consolidation

当前 promotion、decay、policy consolidation、tension、lesson 仍以阈值为主。需要先做 owner-local SHADOW candidate，再由 evidence 决定是否 ACTIVE。

### P1-5：deep evaluation 正式 SHADOW 接线

mid / expensive / cross-generation 代码已在，但需要：

- 注册进主 DAG；
- persona / function direction 基底；
-跨 generation drift；
-只读异常监控；
-禁止进入 reward。

## 5. 四 backend 晋升状态

晋升顺序固定：

```text
temporal runtime
    ↓
temporal SSL
    ↓
Internal RL
    ↓
CMS torch
```

不能并行一次性 ACTIVE。

### 通用 gate

- `real_trace_turns >= 500`
- `validation_delta >= 0.02`
- strict ETA gate
- PE-off control direction
- ETA-off control direction
- rollback drill
- latency SLO
- safety gate

### 组件附加 gate

- SSL：runtime 必须已 ACTIVE；
- Internal RL：runtime + SSL 已 ACTIVE，且 no reward leakage；
- CMS torch：retention 不退化、absorption 有提升。

### 当前判断

| 组件 | 实现 | SHADOW | ACTIVE 资格 |
|---|---|---|---|
| temporal runtime | 已有 | 已有 dual-run | 未确认全 gate |
| temporal SSL | 已有 | 已有 autograd copy-run | 等 runtime 晋升 + 全 gate |
| Internal RL | 已有 | 已有 PPO evidence path | 等 runtime / SSL + leakage gate |
| CMS torch | 已有 | 已有 parity / MSE / rollback | 等 retention / absorption + 全 gate |

## 6. 下一步执行命令

### Step A：恢复执行完整 learned evidence

```bash
bash run_learned_active_evidence.sh \
  --resume \
  --substrate-mode hf \
  --substrate-device mps
```

CUDA host 将 `mps` 改为 `cuda`。

runner 阶段：

```text
shadow-smoke
platform-chunked-soak
real-soak
capacity-ladder
same-substrate-ablation
build-promotion-evidence
evaluate-promotion
```

### Step B：执行真实 capacity ladder

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

### Step C：执行 P1 9-track

```bash
bash run_companion_bench_p1.sh --resume
```

产物：

```text
artifacts/companion-ablation/<run-id>/verdict_p1.json
```

### Step D：接入 verdict 并生成 promotion report

```bash
bash run_learned_active_evidence.sh \
  --resume \
  --substrate-mode hf \
  --substrate-device mps \
  --ablation-verdict artifacts/companion-ablation/<run-id>/verdict_p1.json
```

检查：

```text
artifacts/learned_active_evidence/promotion/promotion_report.json
```

目标：

```text
all_eligible=true
```

若为 false，按 component 的 `missing_gates` 逐项处理，不放宽门槛。

### Step E：P2 held-out multi-seed

P1 只提供 directional evidence。P2 必须加入：

- held-out；
- multi-seed；
- cross-family judge；
- blinded human anchor；
- longitudinal 20-session；
- relationship continuity；
- uncertainty interval；
-严格同 substrate / prompt / context / tool budget。

## 7. ACTIVE 发布与回滚

每次只晋升一个组件：

1. 冻结 evidence bundle 与 git SHA；
2. 只改一个 wiring；
3. 小 cohort canary；
4. 比较 learned vs rollback snapshot；
5. 监控 PE、regime switching、memory retention、latency、安全；
6. 达到退出条件后扩大；
7. 再开始下一个组件。

立即回滚条件：

- validation delta 转负；
- control direction 反转；
- unsafe switching；
- reward leakage；
- CMS retention 退化；
- absorption 无提升；
- fingerprint / schema mismatch；
- latency / safety gate 失败；
- learned state 无法恢复。

回滚：

- wiring 恢复 SHADOW / DISABLED；
- 恢复晋升前 checkpoint；
- 保留失败 trace 与 generation；
- 更新 promotion artifact / known debt；
- 禁止在下游写补丁掩盖上游失败。

## 8. 当前结论等级

| 等级 | 当前是否达到 | 说明 |
|---|---|---|
| `wiring-ready` | 基本达到 | 代码、schema、SHADOW、runner 齐 |
| `promotion-eligible` | 未整体达到 | 需逐组件 `all_eligible` |
| `first-stage-retained` | 未达到 | 需 P2 held-out multi-seed |
| `cognitive AGI thesis stronger` | 未达到 | 需长程、多域、跨 substrate、开放环境 |

## 9. 当前最高优先级

代码线：

```text
adaptive owner continuity
→ learned regime SHADOW
→ multi-party product path
→ thinking→temporal SHADOW
→ deep evaluation SHADOW
```

证据线：

```text
读取现有 509 real-trace missing_gates
→ capacity ladder
→ P1 9-track
→ promotion report
→ runtime ACTIVE canary
→ SSL
→ Internal RL
→ CMS torch
→ P2 held-out multi-seed
```

## 10. 最简状态陈述

> VolvenceZero 的认知骨架约完成 85–90%，仓库定义的第一阶段认知代码约完成 80–88%，剩余 12–20% 是长期 owner continuity、learned regime、多人社会认知、thinking→temporal 和深层只读监控等关键闭环。14 类 learned 部件与四个 torch backend 已有代码，但默认 learned 决策主导度仍仅约 10–20%；四 backend 仍需按 runtime→SSL→Internal RL→CMS 顺序通过 ≥500 real trace、validation delta、控制臂、回滚、性能、安全与 P2 held-out multi-seed 后逐组件晋升。当前可以称 wiring-ready，不能称 first-stage-retained，更不能称 cognitive AGI 已完成。
