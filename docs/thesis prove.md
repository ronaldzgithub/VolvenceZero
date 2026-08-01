# Volvence 因果证据终局战役：Thesis 证明、证伪与生产边界

> 状态：FINAL
>
> 终局日期：2026-07-31
>
> 权威判词：`thesis-rejected`
>
> 对应债务：`#92 CLOSED`
>
> 权威机器可读对账：
> [`artifacts/causal_evidence_final_campaign_20260731/`](../artifacts/causal_evidence_final_campaign_20260731/)

## 0. 一句话结论

Volvence 已经证明了一套**有界、可审计、可回放、可持久化、可回滚的持续适应机制**，并在 Gate 2、Gate 8、Gate 11 上取得了受限的因果或纵向证据；但预注册的整体 thesis 没有通过总 EXIT，因此终局判词是
`thesis-rejected`，不是 `thesis-retained`，也不授权把实验性 learned 路径整体提升到 production/live `ACTIVE`。

这不是“架构不存在”或“项目失败”，而是把可说与不可说的边界冻结下来：

- 可以说 owner、snapshot、PE lineage、replay、persistence、rollback 与有界学习机制已经实现并可审计。
- 可以说 Gate 2 有受限 open-loop 因果证据，Gate 8/11 有受限纵向 owner 证据。
- 不可以说多频 CMS、PE 行为驱动、learned 主动学习、SSL→RL、M3、rare-heavy candidate 都产生了稳定净增益。
- 不可以说整体 Volvence thesis、生产 learned takeover、关系质量或一般物理自主性已经被证明。

## 1. “彻底解决”的真实含义

终局战役从一开始就没有承诺“必须把 thesis 跑成通过”，而是要求每一条开放线在预注册规则下到达合法终态：

1. 通过冻结门槛，进入 `mechanism-supported`、`causal-supported` 或
   `longitudinal-supported`；或者
2. 触发 stop-loss / kill，停止同层调参和重复消费 locked 数据，收缩主张并关闭战线。

因此，`kill` 是合格终态，不是未完成。最终所有开放线均已进入 supported、not-supported、not-authorized、not-admitted 或 excluded 状态，`#92` 不再作为僵尸债务保留。

### 1.1 证据等级不能互相冒充

| 等级 | 它真正证明什么 | 它不能自动证明什么 |
|---|---|---|
| `mechanism-supported` | 路径真实执行、契约/lineage/隔离/回滚可审计 | 有净收益、能泛化、应上线 |
| `causal-supported` | 在冻结 matched control 下，目标机制产生了达到门槛的因果差 | 跨 session 持续、真实人类质量、整体 thesis |
| `longitudinal-supported` | 在冻结跨 session/restart 设计中，目标效应持续并过门 | 开放世界泛化、人类关系质量、所有模块协同成立 |
| `thesis-retained` | 整体预注册 EXIT 全部成立 | 本战役没有达到 |

现行 `#92` 编号中**不存在 Gate 3**。历史材料中把 typed boundary 或
`beta_t / z_t` 涌现称为 Gate 3 的地方，在终局台账中统一归入 Gate 2；不得新建一个平行 Gate 3 来补数字。

## 2. 战役行程与每轮收敛结果

| 阶段 | 主要任务 | 终态 |
|---|---|---|
| Gate 1–6 第一战役 | 验证 PE、ETA、主动学习、多频 CMS、nested meta-init 的机制与 matched controls | 5/5 mechanism 成立；仅 Gate 2 因果成立；Gate 1/4/5/6 因果不成立；无纵向支持 |
| Gate 7–8 第二战役 | SSL→RL takeover 与 wake/sleep | Gate 7 首个 locked artifact 因 topology drift 为 `invalid`；Gate 8 因果成立 |
| Gate 7/9/10 第三战役 | fresh Gate 7、M3/PE-gated update、rare-heavy promotion | Gate 7/9/10 均无因果净增益；Gate 10 全链精确 rollback 成立 |
| Gate 11 纵向第四战役 | per-user continuity、真实跨 session CMS | Gate 11 纵向成立；Gate 5 在 1530 transitions 尺度仍不成立 |
| Gate 8 纵向第五战役 | wake/sleep 跨 restart 复验 | Gate 8 升为纵向成立 |
| Owner 机制改造战役 | 修复 Gate 9/1/6/4/10 的“变量没有真正进入机制”问题 | 五包机制均真实激活，但效果全部为负或不达门，五个 kill 全触发 |
| Ecology 终局线 | same-physics station1、唯一 alignment review、条件式 station2/P1/P2 | station1 结构与四门 GO；alignment 仍 3/4，终局 BLOCK；下游均未获授权 |
| #92 总对账 | 合并 immutable verdict、rollback、ecology 与纵向台账 | `thesis-rejected`，`#92 CLOSED`，无新增 production/live 晋升授权 |

战役总账最终固定为：

- mechanism coverage：完整；
- causal-supported：Gate `2 / 8 / 11`；
- longitudinal-supported：Gate `8 / 11`；
- full-chain rollback：满足；
- `thesis_retained=false`；
- `production_live_promotion_authorized=false`。

## 3. Gate 终局台账

### 3.1 Gate 1 — Prediction Error

**成立的部分**：

- numeric / probability / enum / vector / distribution 五类 PE gold surface
  与真实 autograd LSS 对齐，最大 bridge error 为
  `1.3877787807814457e-17`。
- lineage coverage=`1.0`，accepted mismatch=`0`，duplicate settlement=`0`。
- evaluation 内容改变时，actual、PE 与 PE-derived credit 的 canonical
  payload 保持逐字节不变，证明 evaluation 没有反向成为学习源。
- 因此“PE 是一级、可审计原始信号”是成立的机制主张。

**没有成立的部分**：

- 首轮 held-out causal probe 中，PE-on 与 PE-off 的行为学习成功率都为
  `1.0`，primary gain=`0.0`，低于冻结门槛 `0.25`。
- owner 修复后，PE 确实进入 temporal ndim code，但三 seed mean policy-loss
  reduction=`-0.000881360`，最差 seed=`-0.002758678`，方向为负。

**终局**：`mechanism-supported; causal not-supported`。生产继续保留 PE
计算/快照/lineage，但 PE→runtime behavior modulation 与 PE temporal switch
保持关闭；该负方向也使 Gate 7 没有获得再次重跑授权。

### 3.2 Gate 2 — `beta_t / z_t` 与有界 residual control

**成立的部分**：

- v35 在冻结 Qwen2.5-0.5B、真实 open-weight residual、matched checkpoint、
  prefix intervention 和 permutation-null 约束下取得
  `causal-supported`。
- selector 在 fresh validation 与 locked confirmation 上超过其预注册置换零假设，支持一个受限的 open-loop、`z_t` action-value 因果主张。

**没有成立的部分**：

- 该结论不等于 closed-loop live injection，也不等于纵向持续收益。
- fresh longitudinal seed `1201` 触发冻结 single-seed stop-loss：
  selector−permutation=`-0.001895`，selector−zero=`-0.002651`，session
  positive rate=`0.274510`。因此 seeds `1213 / 1223` 按纪律不再运行。
- official longitudinal verdict=`not-supported`，live injection 与 production
  promotion 都未获授权。

**终局**：`causal-supported; longitudinal not-supported`。只保留受限 open-loop
因果措辞，不外推为“ETA 已能在真实长期交互中接管行为”。

### 3.3 Gate 3 — 不存在

`#92` 的 SSOT 没有 Gate 3。所有历史“Gate 3”措辞已并入 Gate 2，不另计一门支持。

### 3.4 Gate 4 — 主动学习与 label utility

**成立的部分**：

- typed feedback request、OpenLoop actuation、proposal-only、boundary/consent
  non-mutation、lineage 与 rollback 机制全部可运行、可审计。
- 正式 run 共执行 `720` 次 typed feedback request，typed request 与
  open-loop actuation coverage 都为 `1.0`，revision proposal=`0`，boundary
  digest 不变。

**没有成立的部分**：

- 初始五臂实验所有 heldout/locked balanced accuracy 都为 `0.5`；segment、
  turn、random、shuffled 的 labels-needed 全为 `61`，segment gain=`0`。
- owner 修复后 learned utility 确实根据 label 后 readout loss 改善工作，但相对
  turn/random 的 mean labels saved 都为 `-1.0`，minimum final accuracy
  margin=`-0.083333`。
- Ecology station2 未获授权，因此 ecology trace 没有被准入为新的 Gate 4
  corpus；其状态是 `not-admitted`，不能冒写成已经执行的新因果失败。

**终局**：`mechanism-supported; causal not-supported`。通用反馈请求执行路径可以保留，但 learned label-utility selector 不晋升，不能宣称“segment-aware/PE-driven 主动学习稳定省标签”。

### 3.5 Gate 5 — 多频 CMS、ATLAS 与 Titans PE gate

**成立的部分**：

- 多频 cadence、matched parameter budget、PE lineage、冻结 substrate、
  persistence/restart 与 checkpoint rollback 全部可运行并可审计。
- 五臂在纵向 source 上共 replay `7650` arm-transitions，每 arm/seed 跨过
  `50` 次 filesystem persistence + constructor restart。
- full 对 controls 满足 Pareto 不劣，retrieval hit=`1.0`、错误晋升率=`0.0`。

**没有成立的部分**：

- 首轮 full 相对 single-timescale 的 absorption/retention 增益只有
  `+0.000000251 / +0.000001173`。
- 纵向复验对应增益仍只有 `+0.000000201 / +0.000001187`，远低于预注册最小效应 `0.02`。
- 不劣、可回滚或 retrieval hit 全绿，都不能替代“多频优于单频”的 primary
  minimum-effect gate。

**终局**：`mechanism-supported; causal and longitudinal not-supported`。可以说多频 CMS 可运行、可审计、可回滚；不能说它相对单频有稳定、有意义的净增益。现有 CPU CMS 的 PE-gated 写入与 ATLAS replay 是既有 ACTIVE 基线，这不等于本战役授权 Torch 多频 learned backend 晋升。

### 3.6 Gate 6 — Nested / conditioned meta-initialization

**成立的部分**：

- nested initialization 是 owner-controlled、zero-leakage、可审计、可 checkpoint
  rollback 的真实机制。
- 初版相对 random/no-init 有改善，但没有超过最强的 direct copy-init；paired 与
  swapped 几乎不可区分，不能声称 user-related prior。

**没有成立的部分**：

- owner 修复后 context prototypes 确实进入 reset，但相对 copy-init 的 effect
  为 `-0.0490765`，negative transfer rate=`1.0`。
- 原 source 的 context-centroid 最大 pairwise MAE 仅约 `2.27e-6`，不支持
  context-diverse 或 user-related meta-prior 外推。

**终局**：`mechanism-supported; causal not-supported`。生产 reset 使用
`copy-init`，conditioned/nested meta-init 不晋升。

### 3.7 Gate 7 — SSL→Internal RL takeover

**成立的部分**：

- fresh v3 修复了 topology drift；source admission、future leakage、token-space
  mutation、structure freeze、takeover 和 rollback 机制门全绿。
- full takeover rate=`1.0`，说明 SSL→RL 的接管路径在工程上真实存在。

**没有成立的部分**：

- full 相对 no-SSL 与 no-RL 的 terminal-return gain 和 composition gain 全部为
  `0.0`。
- Gate 1 owner 修复后的 PE→control 方向为负，因此没有满足重新运行另一个
  Gate 7 五臂实验的信号前置条件。

**终局**：`mechanism-supported; causal not-supported`。SSL→RL 可以作为可审计、可回滚的实现候选，不能声明有 causal advantage，也不能接入 production behavior。

### 3.8 Gate 8 — Wake/sleep 与跨 session consolidation

**成立的部分**：

- 初始 causal run 中，full sleep 相对 no-sleep 的 cold-start loss reduction、
  callback consistency gain、delayed-payoff gain分别为
  `+0.454176 / +1.0 / +0.454176`。
- 纵向正式 run 使用三 seed × `510` settled transitions、四臂共 `6120`
  arm-transitions；每 arm/seed 跨 `51` 个 consumer sessions 和 `50` 次持久化+
  constructor restart。
- 纵向 full 相对 no-sleep 的三项增益为
  `+0.567363 / +1.0 / +0.567363`，相对 single-owner controls 的最小 delayed
  payoff margin=`+0.167363`；最大 owner drift=`0.467850 < 0.50`。
- paired-seed 95% CI lower、lineage、prompt increment、queue 幂等、latency
  分离、persistence、frozen substrate 与 rollback 全绿。

**限制**：三 seed gain 完全相同，属于不同 trace lineage 上的确定性复制，不是对独立模型行为分布方差的证明；指标仍是 deterministic owner readout，不是盲评的人类关系质量 ground truth。

**终局**：`causal-supported and longitudinal-supported`。这是终局战役最强的保留结论之一，但只限 wake/sleep owner effect。

### 3.9 Gate 9 — M3、bounded self-modification 与 PE-gated update

**成立的部分**：

- matched budgets、owner-local bounded update、PE lineage、frozen substrate、
  checkpoint rollback 均成立。
- PE write precision=`1.0`、unnecessary-write rate=`0.0`，说明写入门控机制准确执行。

**没有成立的部分**：

- 初版 M3 与 plain momentum 数值相同，暴露 slow momentum 只是输出信号、没有进入参数更新。
- owner 修复后预注册 `slow_gain=1.0`，慢动量真实进入更新；但相对 plain 的
  tracking/recovery gain 为 `-0.00158911 / -0.00187450`。
- PE gate held-out benefit 相对 always/random 基本为零，不能把高 write precision
  当作有用性增益。

**终局**：`mechanism-supported; causal not-supported; kill`。生产
`M3Optimizer.slow_gain=0`。M3/Titans/Hope 只能称为实现候选或设计模式；DGD
与真正 Hope 自指递归仍未被本战役证明。

### 3.10 Gate 10 — rare-heavy promotion

**成立的部分**：

- candidate envelope、compatibility、privacy、review-only 无副作用、自动拒绝、
  owner import、old-scenario replay 与 rollback 机制均成立。
- v2 把 session owner checkpoint 与 live substrate checkpoint 一起纳入 digest，
  full-chain rollback exact rate=`1.0`，满足总 EXIT 中的可逆性要求。
- owner 修复后的 v3 让 train/eval 共享 structural objective，机制不再是假接线。

**没有成立的部分**：

- v2 held-out gain 相对 review-only 只有 `-0.00000488`，没有达到 `0.008`。
- v3 held-out gain=`-0.004695`，catastrophic forgetting=`0.013869`；即使 rollback
  exact=`1.0`，candidate 仍没有持续改进证据。
- 可审核、可拒绝、可回滚不等于 artifact 值得晋升。

**终局**：`mechanism-supported and rollback-supported; causal not-supported;
review-only`。live substrate mutation 保持冻结，不授权自动 production import
或 autonomous rare-heavy promotion。

### 3.11 Gate 11 — Per-user continuity

**成立的部分**：

- fresh source 使用 seeds `1201 / 1213 / 1223`，每 seed `510` 条、合计
  `1530` settled transitions；lineage、trace digest、substrate fingerprint、
  fallback=0、empty residual=0、duplicate=0、mutation=0 全绿。
- 四臂 `stateless / correct-user-state / swapped-user-state / shuffled-history`
  共消费 `6120` arm-transitions。
- correct 相对 stateless/swapped/shuffled 的 continuity composite 增益分别为
  `+0.759259 / +0.759259 / +0.666667`，三个 paired-seed 95% CI lower 均大于 0。
- cross-user read/write leakage 与 key collision 都为 `0`；persistence
  round-trip、delete 和 checkpoint rollback exact。

**限制**：correct-state callback absolute hit rate只有 `0.277778`，虽然
commitment/boundary consistency=`1.0`；没有盲评人类关系质量 ground truth。

**终局**：`causal-supported and longitudinal-supported`。它证明的是隔离的
per-user owner continuity，不是“用户一定感到关系更好”。

## 4. Ecology 主战场的终局

Ecology 结果进入 #92 前，已经冻结
`ecology-gate-evidence-admission.v1`，防止看到结果后挑选映射。其条件链为：

1. station1 通过，才允许 alignment review；
2. review 通过，才授权 station2；
3. station2 证明 typed milestone 的 medium 因果贡献，才准入 Gate 4 ecology
   corpus，并授权正式 P1；
4. P1 通过，才授权 P2 的 PE-on/off matched confirmatory。

### 4.1 实际结果

- station1 verdict=`GO`；四个预注册 causal gates 全部通过；
- 8/8 structural/persistence lanes 通过；
- candidate pickups=`47`，control pickups=`52`，ratio=`0.903846`；
- review 前 food alignment=`3/4`；
- 唯一预签的五局 review 后仍为 `3/4`，要求是 `4/4`；
- review verdict=`BLOCK`，`next_episode_authorized=null`。

### 4.2 下游状态必须精确表述

| 下游 | 终态 | 为什么 |
|---|---|---|
| station2 medium | `not-authorized` | alignment review 没有达到 4/4 |
| Gate 4 ecology corpus | `not-admitted` | station2 没有先证明 segment 结构的因果价值 |
| P1 五臂 full run | `not-authorized` | 上游 frozen gate 阻断 |
| P2 PE confirmatory | `not-authorized` | P1 未获授权 |

这些项不是“跑了然后失败”，而是按预注册 kill 纪律没有执行。不能把
`not-authorized` 写成支持，也不能把未执行包装成新负证据。

Ecology 最终只支持：same-physics station1 的局部机制、结构持久性和早期 pickup
表现达到门槛。它不支持 medium 闭环、segment credit 组合优势、PE 加性收益或一般物理自主性。

## 5. Owner 机制改造战役的核心教训

机制改造的意义是排除“效果没出现只是因为变量没有真正接入”的借口。Gate
9/1/6/4/10 的 owner 级缺口被逐一修复：slow momentum 真正进入 M3 更新、PE
真正进入 temporal code、context prototype 真正进入 reset、learned label utility
真正驱动 selector、rare-heavy train/eval 真正共享 structural objective。

修复后五包的机制门全部通过，但效应仍全部为负或未达门：

| Gate | 修复后主效应 | 终局生产回滚 |
|---|---:|---|
| 9 M3 | tracking/recovery `-0.00158911 / -0.00187450` | `slow_gain=0` |
| 1 PE→control | mean loss reduction `-0.000881360` | runtime modulation disabled |
| 6 conditioned init | vs copy `-0.0490765`，negative transfer=`1.0` | `copy-init` |
| 4 learned utility | labels saved `-1 / -1`，accuracy margin=`-0.083333` | selector 不晋升 |
| 10 rare-heavy | held-out gain `-0.004695` | review-only，不生产晋升 |

因此终局不能再归因于“只是工程 wiring 不完整”。在当前机制、数据和冻结门槛下，这五条 learned uplift 主张就是没有成立。

## 6. 最终保留与永久收缩的主张边界

### 6.1 可以对内、对外准确声明

- 系统的 owner / immutable snapshot / typed proposal / lineage / replay /
  persistence / rollback 基础设施已经实现并可审计。
- PE 是可数值验证、与 evaluation 解耦的一级原始信号。
- Gate 2 v35 支持受限的 open-loop `z_t` action-value 因果主张。
- Gate 8 支持 wake/sleep 在 deterministic owner readout 上的因果与纵向效果。
- Gate 11 支持隔离 per-user owner state 的跨 session 连续性。
- 多频 CMS、主动反馈请求、SSL→RL takeover、bounded self-modification、rare-heavy
  review/import/reject/rollback 的**机制**可以运行、审计和回滚。
- full-chain rollback 已经通过；生产可以坚持冻结基底和显式门控。

### 6.2 禁止从本战役推出

- “完整 Volvence/NL+ETA thesis 已被因果证明”。
- “多频 CMS 相对单频有稳定且有意义的净增益”。
- “PE 驱动线上行为稳定优于 PE-off”。
- “segment-aware 或 PE-driven 主动学习稳定节省标签”。
- “SSL→RL takeover 带来 terminal return 或 composition 净增益”。
- “M3 慢更新优于 plain momentum/SGD/Adam”。
- “rare-heavy candidate 会带来持续改进，可以自动晋升生产”。
- “per-user continuity 或 wake/sleep 已由人类 ground truth 证明改善关系质量”。
- “Ecology 已形成 medium 闭环或一般物理自主性”。
- “机制可回滚”能够替代“机制有效”。可逆性是安全证据，不是收益证据。

## 7. 当前 production wiring：哪些确实已经 ACTIVE

`production_live_promotion_authorized=false` 的准确含义是：**终局战役没有授权新的 learned/live 晋升**。它不要求把战役前已经存在的低风险基础机制全部关闭。当前代码默认值如下：

| 路径 | 当前默认 | 解释 |
|---|---|---|
| Memory owner | `ACTIVE` | 基础 memory 主链已启用 |
| CPU CMS PE-gated write + ATLAS replay | `ACTIVE` | 既有默认基线；不等于多频 Torch backend 获得因果授权 |
| PredictionError owner | `ACTIVE` | 计算、snapshot、lineage 可在线工作 |
| Apprenticeship feedback-request alignment | `ACTIVE` | 仅真实 apprenticeship turn 上的 typed 请求执行面 |
| Apprenticeship protocol alignment | `SHADOW` | learned/protocol 对齐没有成为 authoritative 路径 |
| Session post slow loop | `ACTIVE` | Gate 8 对应的慢循环基础路径已启用 |
| Experience consolidation | `ACTIVE` | 基础 consolidation 已启用 |
| Owner hydration wiring | `ACTIVE` | 运行时接线启用；跨进程持久性仍需要配置真实存储路径/backend |
| Temporal owner | `ACTIVE` | 表示基础 temporal owner 在主链；不表示 SSL→RL learned backend 已上线 |
| CMS Torch backend | `DISABLED` | 未晋升 |
| Temporal SSL/runtime backend | `DISABLED` | 未晋升 |
| Internal RL backend/replay/causal action head | `DISABLED` | 未晋升 |
| RL runtime modulation strength | `0.0` | learned RL 不影响生产行为 |
| PE temporal switch | `DISABLED` | PE 不触发 learned temporal 切换 |
| M3 slow branch | `slow_gain=0.0` | 修复实验的负方向已回滚 |
| Rare-heavy protocol revision queue | `SHADOW` | 不自动导入生产 |
| Live substrate mutation | `false` | 冻结 review-only；离线 artifact 仍需 `ModificationGate` |

因此，对“这些 active 的都设置成 active 了吗”的最终回答是：基础、低风险且本来就在生产工作的路径大多已经 ACTIVE；实验性净增益链没有被一刀切升为 ACTIVE，也不应根据本战役升为 ACTIVE。rare-heavy 只有门禁、审核、拒绝和回滚能力成立，自动 review/import loop 仍不是完整 production-active。

## 8. 证据完整性与已知限制

### 8.1 证据完整性

- 主要纵向 source 使用 strict-local frozen `Qwen/Qwen2.5-0.5B-Instruct`；
- seeds `1201 / 1213 / 1223` 各 `510` transitions，总计 `1530`；
- lineage coverage 完整，fallback、empty residual、duplicate settlement、
  substrate mutation 均为 `0`；
- locked source 一次性消费，失败后不调阈值、不复用同一 locked 分区抬结论；
- Gate 10 full-chain rollback 同时覆盖 temporal、memory、application、substrate
  与 checkpoint 状态；
- Gate 8 的并发重复 formal run 已标记
  `invalid-duplicate-not-admitted`，未进入 verdict 或 CI；
- ecology v31 污染 journal 与 v30 `/tmp` 证据已按 SHA-256 归档为
  `EXCLUDED`，明确 `resumable=false`、`admissible_for_formal_verdict=false`。

### 8.2 不能忽略的限制

- 多项强结果是 deterministic owner readout，不是独立采样模型行为的分布证据。
- Gate 8 三 seed 增益完全相同，不能据此估计真实行为方差。
- Gate 11 没有 blind human relationship-quality ground truth，关系质量外推仍依赖 #51 human anchor。
- Gate 2 因果结果是受限 open-loop selector 结论，纵向 readout 已失败。
- Ecology 在 station1 后被 alignment gate 阻断，没有产生 station2/P1/P2 证据。
- frozen Qwen、synthetic/controlled environments 与有限 scenarios 不能代表一般开放世界。

## 9. 工程收尾也已进入终态

终局战役不仅给出算法判词，也清理了会污染判词的工程尾债：

- 原混合工作线按机制改造、Gate 2 纵向、ecology dwell 与其它工作拆包；
- rare-heavy/binary-override 相关的 5 个既有失败已修复；
- `sandbox.py` Ruff 遗留已清偿；
- “零训练冷内核 learned 胜过 random”的陈旧 smoke 断言被改为冻结真实负证据；
- 旧污染 journal 和不可准入 `/tmp` artifact 已归档并设为不可续跑；
- ecology station2、Gate 4 ecology、P1、P2 按 kill 规则没有产生半成品或僵尸任务；
- `#92` 已从开放债务中关闭，完整历史由 git 与 immutable artifacts 保留。

## 10. 以后怎样才允许重开

终局拒绝不是禁止未来研究，而是禁止用同一机制、同一 locked 数据、同层调参把结论“磨成通过”。若要重开：

- Gate 1/4/6/7/9/10：必须先有新的 owner-level 机制变化，再建新 schema、fresh
  source 和独立预注册 evidence plan；仅换 seed 或降低阈值不够。
- Gate 5：必须改变能够产生可识别多时间尺度差异的机制或证据前提；不能继续在同一微小 proxy effect 上调参。
- Gate 2：需要新的 longitudinal readout/live route 机制和 fresh capture；v35
  open-loop causal 不能自动升级，v36/v37 历史路线不能偷渡。
- Gate 8/11：如要外推到产品关系质量，必须补 blind human longitudinal
  ground truth，而不是继续只看 owner metrics。
- 整体 thesis：必须作为一个新的 thesis 提案重新定义总 EXIT，并进行新的独立预注册；不能重新打开已关闭的 #92 改写历史。

## 11. 权威证据索引

### 11.1 终局与计划

- [因果证据终局战役计划](../.cursor/plans/因果证据终局战役_d8024394.plan.md)
- [终局机器可读 reconciliation](../artifacts/causal_evidence_final_campaign_20260731/reconciliation.json)
- [终局报告](../artifacts/causal_evidence_final_campaign_20260731/report.md)
- [Evidence Program SSOT](specs/evidence_program.md)
- [当前状态](currentstatus.md)
- [Known Debts](known-debts.md)

### 11.2 分战役对账

- [Gate 1–6 第一战役](../artifacts/gates_1_6_evidence_campaign_20260730/report.md)
- [Gate 7–8 第二战役](../artifacts/gate7_8_second_campaign_20260730/report.md)
- [Gate 7/9/10 第三战役](../artifacts/gate7_9_10_third_campaign_20260730/report.md)
- [Gate 11 纵向第四战役](../artifacts/gate11_longitudinal_fourth_campaign_20260730/report.md)
- [Gate 8 纵向第五战役](../artifacts/gate8_longitudinal_fifth_campaign_20260730/report.md)
- [Owner 机制改造总对账](../artifacts/mechanism_repair_campaign_20260731/report.md)

### 11.3 关键正式 artifacts

- [Gate 1 PE mechanism](../artifacts/gate1_pe_mechanism_20260730/report.md)
- [Gate 1 PE causal repair](../artifacts/gate1_pe_causal_v3_retest_20260731/report.md)
- [Gate 2 v35 causal](../artifacts/eta_gate2_residual_causal_v35_selector_null_fresh_fullwidth896_qwen25_05b_cpu_1seed_20260729/report.md)
- [Gate 2 longitudinal stop-loss](../artifacts/gate2_longitudinal_v35_companion_seed1201_formal_20260730/report.md)
- [Gate 4 label utility repair](../artifacts/gate4_label_utility_v3_retest_20260731/report.md)
- [Gate 5 longitudinal CMS](../artifacts/gate5_cms_pareto_longitudinal_v2_20260730/report.md)
- [Gate 6 conditioned meta-init](../artifacts/gate6_conditioned_meta_init_v3_retest_20260731/report.md)
- [Gate 7 SSL→RL](../artifacts/gate7_causal_takeover_v3_20260730/report.md)
- [Gate 8 longitudinal wake/sleep](../artifacts/gate8_wake_sleep_longitudinal_20260730/report.md)
- [Gate 9 M3 slow update](../artifacts/gate9_m3_slow_update_v2_20260731/report.md)
- [Gate 10 rare-heavy structural transfer](../artifacts/gate10_structural_transfer_v3_20260731/report.md)
- [Gate 11 per-user continuity](../artifacts/gate11_per_user_continuity_v2_20260730/report.md)

### 11.4 当前 production 默认值的代码 SSOT

- [FinalRolloutConfig](../packages/vz-runtime/src/volvence_zero/integration/final_wiring.py)
- [Default MemoryStore / CPU CMS uplift](../packages/vz-memory/src/volvence_zero/memory/store.py)
- [BrainConfig owner hydration](../packages/vz-runtime/src/volvence_zero/brain.py)
- [Frozen residual mutation capability](../packages/vz-substrate/src/volvence_zero/substrate/residual_interfaces.py)
- [M3 slow-gain default](../packages/vz-temporal/src/volvence_zero/temporal/m3_optimizer.py)

## 12. 最终判词

本战役最重要的成果不是把所有曲线写成正数，而是消除了三类模糊：

1. **机制与收益的模糊**：能运行、能回滚，不等于有净增益。
2. **局部与整体的模糊**：Gate 2/8/11 的局部支持，不等于整体 thesis 成立。
3. **已有 ACTIVE 与新晋升的模糊**：基础生产路径可以继续 ACTIVE，但失败的 learned uplift 不获得新晋升授权。

据此，2026-07-31 的唯一权威终局为：

```text
terminal_verdict = thesis-rejected
thesis_retained = false
mechanism_coverage_complete = true
causal_supported_gates = [2, 8, 11]
longitudinal_supported_gates = [8, 11]
full_chain_rollback_satisfied = true
production_live_promotion_authorized = false
debt_92 = CLOSED
```

这是对当前机制与冻结证据计划的诚实终局，不是对 Volvence 长期研究方向的否定。未来可以提出新的机制和新的 thesis，但不能篡改本战役已经产生的负证据。

## 13. 五杠杆后续战役追加判词：Ecology L1 终局

> 追加日期：2026-07-31。本节是对 #92 终局之后新机制、新 schema 和 fresh source 的独立记录；
> 它不重开、改写或减轻上述 `thesis-rejected` 判词。

五杠杆计划的 Ecology L1 按合法重开路径走到了自己的终态：

1. **L1-A 归因**：失败者稳定为 body 2；H1 learning-state divergence 获得支持，H2
   课程暴露不均被排除，H3 因旧 journal 没有发布逐更新梯度而保持 inconclusive。
2. **L1-B 机制**：temporal owner 实现了通用、有界、可回滚的 action-head 形成期保护；
   默认为 `DISABLED / 0 / 1.0`，Digital Ant evidence profile 为 `ACTIVE / 160 / 0.25`。
   旧 checkpoint 的 no-write precheck 证明 forward 等价与回滚可用，没有冒充净增益证据。
3. **L1-C fresh station1-v4**：隔离源码快照、新空 journal、candidate/control 各 20 局。
   pickup=`47/52`（ratio=`0.9038461538461539`），四个共同门中除 alignment 外均过，
   8/8 structural/persistence lanes 全过；但 food alignment 仍为 3/4。body 2 的
   left/right turn=`-0.00038541260576263365 / +0.00038709291851802386`，反向幅度大于
   旧 station1。machine verdict=`BLOCK`。

因此，L1 的唯一合法结论是：**形成期保护机制可实现、可预检、可回滚，但没有在冻结
station1 上产生要求的 learned uplift，不获得 production/live 晋升**。station2 medium 因前置
门失败而不执行，所以 medium 层是“未测且不授权”，不是 PASS，也不伪造为 FAIL。
`alignment_review_authorized=false`、`next_episode_authorized=null`；禁止第二次 review、换 seed、降低
4/4 门或加训练量。

这个 kill 同时关闭了依赖 station2 GO 的 L2 Ecology 路线：
`ecology-gate-evidence-admission.v2`、Gate 4 ecology corpus、P1/P2 以及拟用 Ecology 重开的
Gate 5/9/10/1 都没有获得授权。这些 gate 的 #92 等级和本文 §12 的总判词均不改变。

权威追加 artifacts：

- [L1-A 归因](../research/ant/results/ecology_recovery/same_physics_baseline/alignment_formation_attribution.v1.json)
- [L1-B no-write precheck](../research/ant/results/ecology_recovery/same_physics_baseline/alignment_formation_protection_precheck.v1.json)
- [L1-C preregistration](../research/ant/results/ecology_recovery/same_physics_baseline/ecology_same_physics_prereg.seed0.20260731T135415Z.json)
- [L1-C station1-v4 终局 report](../research/ant/results/ecology_recovery/same_physics_baseline/ecology_same_physics_station1.seed0.20260731T135415Z.json)

## 14. 五杠杆后续战役追加判词：Gate 2 L3 终局

> 追加日期：2026-07-31。本节使用新 longitudinal readout、新 schema 和 fresh
> capture 独立检验 Gate 2；它不改写 #92 的 `thesis-rejected`或 v35 的历史因果判词。

L3-A 将 `RelationshipConditioningModule` 发布的 14 维有序 readout 追加到完整
8076 维 residual state，形成 8090 维
`residual-state+relationship-owner-readout.v1`。该路径只做
`(2x-1)×confidence` 有界变换，不解释 label、不重建关系语义；错 bank、
cold-start 和 zero-confidence 全部 fail loudly，所以它是与 v35 无条件
selector 可机器区分的新机制，但未进入 live session。

L3-B 冻结 `eta-gate2-longitudinal-conditioned.v1`：训练 seed 1291/64 条，
formal seeds 1301/1313/1327 各 510 条；同 source 的 correct Relationship condition
对比 action permutation、zero 和 matched wrong-condition。单 seed 三个 mean 都要
≥`0.02`，wrong-condition session positive rate 要 ≥`0.60`；1301 为先行完整
stop-loss。预注册 SHA-256 为
`c51848d41888ea3e7f2a4f83174d6b49483928b7f73dc4655f44f77e7877d1ea`。

L3-C 的 seed 1301 已跑满 510 条、51 个 sessions，但只有 count 门通过：

| 冻结门 | seed 1301 | 门槛 | 结果 |
|---|---:|---:|---|
| selector − action permutation mean | `+0.003287669022878011` | `≥ 0.02` | FAIL |
| selector − zero mean | `+0.004308079037011838` | `≥ 0.02` | FAIL |
| selector − matched wrong-condition mean | `+0.000055160709455901504` | `≥ 0.02` | FAIL |
| wrong-condition session positive rate | `0.1568627450980392` | `≥ 0.60` | FAIL |

因此 machine status=`single-seed-stoploss`，official Gate 2 longitudinal
verdict=`not-supported`。1313/1327 依预注册不运行，不允许 refit、换 seed、
降阈值或把 carrier existence 写成 longitudinal gain。本轮没有安装 selector、
没有改 substrate 权重、没有写 runtime owner state，所以
`production_live_promotion_authorized=false`。

正式进程在启动时通过 prereg/code-tree 验证；完成 510 条后，工作区的
`residual_backend.py` 被并行改写，第一次封包因 source hash drift 正确拒绝。
为保留这项改动，后续使用 Git commit
`79d142f7dfc78e22247aa70222ad4bff0964c1d7` 的隔离快照复核预注册中全部十个 code
digests，仅重做 validation/export，不重算 outcomes。封包后
`source_unchanged=true`、`selector_installed_live=false`、
`substrate_weights_updated=false`。

这条线的最终主张边界是：**Gate 2 继续保留 v35 的受限 open-loop
`causal-supported`；relationship-conditioned carrier 的机制存在且可审计，但新纵向净收益
不成立，不升为 production/live ACTIVE。**

权威追加 artifacts：

- [L3 预注册](../artifacts/gate2_longitudinal_conditioned_prereg_20260731T170122Z.json)
- [L3 正式报告](../artifacts/gate2_longitudinal_conditioned_seed1301_formal_20260731T170122Z/report.md)
- [L3 promotion verdict](../artifacts/gate2_longitudinal_conditioned_seed1301_formal_20260731T170122Z/promotion_verdict.json)
- [L3 freeze manifest](../artifacts/gate2_longitudinal_conditioned_seed1301_formal_20260731T170122Z/freeze_manifest.json)

## 15. Thesis v2 是新提案，不是已通过的新终局

L1 与 L3 都已进入冻结 stop-loss 后，五杠杆 L5 已产出独立
[`thesis-v2-proposal.md`](thesis-v2-proposal.md) 与 known debt #93。它继承本文的
`thesis-rejected`、Ecology L1 `BLOCK` 与 Gate 2 L3
`single-seed-stoploss`，不修改任何历史数字。

v2 只保留四类受限主张：Gate 11 per-user owner continuity、Gate 8
wake/sleep owner-readout consolidation、Gate 2 v35 open-loop causal control，以及
owner/snapshot/lineage/persistence/rollback 基础设施。多频 CMS、PE 行为驱动、
learned 主动学习、SSL→RL、M3、rare-heavy 自动晋升、Ecology medium 和
Gate 2 longitudinal 全部排除出 v2 EXIT。

v2 唯一新证据面是已预注册的 Gate 8/11 真实人类盲评。L4-A 协议、
L4-B 盲化工具与 L4-C 本地分析/power-freeze 执行器已完成。分析器在看见评分前冻结
typed human/non-project roster、ordinal Krippendorff α、Wilson、10,000 次
rater-cluster bootstrap 与 60–300 formal pair 规则；analysis prereg SHA-256 为
`240742e54524b657fb3803382d93af4e651f59f5fb8c8be9e85823ffd5bb95af`。但真实
transcript、非项目 rater、pilot/power report 与 formal
run 尚未完成。因此当前状态必须写为
`preregistered-proposal / not-yet-retained`，不是 thesis 新证明。Gate 8 与 Gate 11
两门都过才可终态 `product-continuity-retained`；任一门失败则终态
`product-continuity-rejected`，并永久把失败 gate 限定为 owner metric。

machine preregistration：
[`artifacts/thesis_v2_product_continuity_prereg_20260731T181500Z.json`](../artifacts/thesis_v2_product_continuity_prereg_20260731T181500Z.json)，
SHA-256 `8bcabb75a6d63068d3dc40e6cbd7e9497560f17cab364017a1cfb76b6fb8f3c2`。
该 artifact 同时冻结 `production_live_promotion_authorized=false`；v2 即使未来 retained，
也不自动翻转任何 `WiringLevel`。

## 16. 七天自动化证据闭环：真实产品路径已跑通，正式矩阵运行中

2026-08-01 完成了 simulated user × 真实七日 lifecycle 的证据执行面：六个
scenario（3 persona × 渐进升温/裂痕修复）、每场 7 sessions × 5 exchanges、每日
cold-start/end-of-day 七项 readout、6 次 persist/restart/hydrate 边界、Gate 11 四臂、
Gate 8 sleep 两臂，以及 Gate 8/11 v1 capture→blind packet 适配器。各臂共享冻结的
35 个 user turns，evaluation 只读且不回灌学习。最终模拟用户由 typed FSM 固定实质句，
本地 Qwen 只选择封闭式语气开场，避免模型改写 preference、boundary 与 callback；SUT
使用不同家族的本地 SmolLM2。

正式协议已冻结为
[`seven_day_companion_simulated_prereg_20260731T222910Z.json`](../artifacts/seven_day_companion_simulated_prereg_20260731T222910Z.json)，
SHA-256 `9ae32c6cf4c7484502f21ce090532ff5c9f31c793364d40e75e24501fcb8792c`。正式矩阵是
36 runs / 252 sessions / 1260 exchanges，禁止 deterministic fake，且 user simulator 与
SUT 必须跨模型家族。

真实产品路径 smoke 已完成 7 sessions、35 个 SmolLM2 回复、14 个公开 console actions 和
6 次不同进程 restart；archive/loaded owner digest 与独立 measurement digest 均通过机器
复核，run SHA-256 为 `e5df2bb0bcafbec971b0ae1cb0dc97127731f6050b1d4ece684ce0f40a214a45`。
这证明仪器能测，不证明某臂效果更好。正式 36-run 当前正在执行，所以这里**仍没有新增
“稳定净增益”结论**，也不能说实验发现“没有提升”：正确状态是
`running / no causal result yet`。`193423Z` prereg 在任何正式 outcome 前被最终的模型、
measurement 隔离和 console probe 协议取代，未观察 outcome，不构成看结果后改协议。

冻结 v1 没有限定聊天者必须真人，因此 simulated transcript 可以用于 pilot；真人评分后
最多得到 `human-rated-simulated-user-transcripts-only`。它不能替代 #93 的 real-user
product-value EXIT。盲评未完成、正式矩阵尚未完成、production promotion 仍为 false，
所有相关机制的现有 `WiringLevel` 均不因本工具链改变。
