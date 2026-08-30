# Volvence 当前状态：能力、证据与生产边界

> Last updated: 2026-08-30
> 状态 SSOT：[currentstatus.md](./currentstatus.md)；开放债务：[known-debts.md](./known-debts.md)；#92 终局：[thesis prove.md](./thesis%20prove.md)

## 一句话结论

Volvence 已形成 41-wheel、契约驱动、可持久化和可回滚的持续适应系统；基础 owner
主链与低风险控制路径大多已接入 production wiring，learned backend 和多个
SHADOW learner 也有真实实现。但 2026-07-31 的 #92 总 EXIT 没有通过，权威判词是
`thesis-rejected`：局部机制与局部证据成立，不等于整体 thesis、learned takeover、
关系质量或开放世界 Cognitive AGI 已被证明。

## 1. 当前系统形状

- 41 个 wheel：8 个 `vz-*`、21 个 `lifeform-*`、6 个 `dlaas-platform-*`、
  6 个 `companion-*`；完整清单见 [package_usage.md](./package_usage.md)，总数以
  `packages/*/pyproject.toml` 为准。
- 9 个产品 vertical：companion/emogpt、coding、venture、operations、character、figure、
  growth-advisor、repair30、digital-employee。
- 3 个正式 Domain Brain 产品侧车：Coding、Venture 与 Operations。Coding/Venture 发布 ACTIVE
  有界内容 Context Pack 且 advice 固定 SHADOW；Operations 默认 SHADOW，只允许 exact
  `ModificationGate` receipt 约束下的 staging ACTIVE。三者都经 typed outcome 把合格环境事实接回
  下一拍 PE，不复制 kernel owner，也不接管 host 的执行、审批或治理权。
- kernel 的唯一业务编排层是 `vz-runtime`；跨模块正式交换只走不可变 snapshot。
- World / Self、PE / credit / evaluation、online-fast / session-medium /
  background-slow / rare-heavy 的职责仍严格分层。
- Lifeform、DLaaS、Companion 工具层不反向拥有 kernel cognition state。

## 2. 已经落地的主链

| 能力 | 当前事实 |
|---|---|
| Prediction Error | `prediction_error` owner ACTIVE；显式 prediction chain、typed external outcome 与 PE decomposition 可审计 |
| Temporal | 基础 temporal owner ACTIVE；`beta_t/z_t`、segment closure 与 action context 在正式快照中；learned torch runtime/SSL/RL backend 仍未 production 晋升 |
| Memory | Memory owner、CPU CMS PE-gated write、ATLAS replay、session-post consolidation 与 owner hydration 已有正式路径 |
| Credit / gate | credit owner、session-held learned heads、ModificationGate 与 rare-heavy review/rollback 已实现；实验 learned readout 不自动成为行为源 |
| Semantic owners | 9 类语义 owner、typed proposal/event 路径、hydration 与 snapshot consumer 已齐；LLM 只产 proposal |
| Social cognition | multi-party identity、ToM 四轨、role、common-ground、group 与 social PE 的 owner/contract 已实现；真实语义 runtime 和行为证据仍受 gate 约束 |
| Application | domain knowledge、case memory、playbook、boundary、experience consolidation/fast prior 的 owner 路径已实现 |
| Lifeform | vitals、affordance、thinking、ingestion、expression、protocol uptake、MCP bridge、synthetic data 与九个 vertical 已分 wheel |
| Evaluation | cheap `evaluation` ACTIVE；`evaluation_mid` SHADOW；expensive/cross-generation DISABLED；evaluation 始终是 readout/gate，不是学习源 |
| Product loop | Relationship Memory Console 已有 proposal/correction 闭环；Coding/Venture/Operations Brain 已有 strict ContextRequest→ContextPack/Advice→typed OutcomeReport/Receipt 闭环，并复用 identity-scoped Memory 与下一拍 PE；ProductZero、Foundry、AutoCompany 的外部 adapter 已分别接入 Coding、Venture、Operations；P5 七日 continuity 聚合未落地 |

此前列为“尚缺”的 session-held credit、thinking advisory、9/9 semantic proposal、
group product consumer、owner hydration、affordance invoker 与 protocol runtime 接缝，
现在都已有代码路径；不能继续列为待实现项。它们是否 authoritative，则分别由
WiringLevel 和证据门决定。

## 3. 默认生产 wiring 的准确解释

| 路径 | 当前默认 | 能说什么 |
|---|---|---|
| Memory / PredictionError / base Temporal owner | ACTIVE | 基础快照主链在线工作 |
| CPU CMS PE gate + ATLAS replay | ACTIVE | 既有可回滚基线；不代表 Torch 多频 backend 有净增益 |
| Apprenticeship feedback request | ACTIVE | 只在 typed apprenticeship turn 生效 |
| Session-post loop / experience consolidation / hydration | ACTIVE | 基础慢循环可运行；跨进程延续仍需真实 persistence backend |
| Protocol runtime / phase / registry introspection | ACTIVE | reviewed protocol 可进入 runtime；reflection/revision mutation 仍 SHADOW |
| Evaluation mid | SHADOW | 发布证据读数，不控制行为 |
| Decision workspace | SHADOW | 结构化决策 panorama 可审计，尚无 authoritative consumer |
| Temporal SSL/runtime、Internal RL、CMS Torch | DISABLED | 没有获得 production promotion |
| RL runtime modulation | `0.0` | learned RL 不影响默认生产行为 |
| Live substrate mutation | false | base 冻结；artifact 更新只能走 offline/rare-heavy gate |
| Coding / Venture 内容择位 | ACTIVE | 只允许 Context Pack 内最多提升一个 owner entry；Advice 仍 SHADOW |
| Operations ranking / timing | SHADOW；staging 可凭 exact receipt ACTIVE | production 强制 SHADOW，不取得 dispatch authority |

`RuntimeModule.default_wiring_level` 与 `FinalRolloutConfig` 是两个明确的 SSOT：前者是
模块安全默认，后者是 production rollout override。看到模块类写 SHADOW，不能据此
断言最终 runtime 没有 ACTIVE；反之看到基础 owner ACTIVE，也不能推导 learned backend
已经接管决策。

## 4. #92 因果证据终局

| Gate / 线 | 终局 |
|---|---|
| Gate 1 PE uplift | mechanism-supported；causal/longitudinal not-supported |
| Gate 2 `beta_t/z_t` residual control | v35 受限 open-loop causal-supported；历史 longitudinal stop-loss 失败 |
| Gate 4 active learning | mechanism-supported；causal not-supported |
| Gate 5 multi-frequency CMS | mechanism-supported；causal/longitudinal not-supported |
| Gate 6 nested initialization | mechanism-supported；causal not-supported |
| Gate 7 SSL→RL takeover | not-supported / not-authorized |
| Gate 8 wake/sleep consolidation | 受限 causal + longitudinal supported |
| Gate 9 M3 slow branch | not-supported，负方向已回滚 |
| Gate 10 rare-heavy promotion | gate/rollback mechanism supported；candidate uplift not-authorized |
| Gate 11 per-user continuity | 受限 causal + longitudinal supported |
| 总 EXIT | `thesis-rejected`，`#92 CLOSED`，没有新的 production/live 晋升授权 |

Gate 2 的新 relationship-conditioned longitudinal lane 也已终止：seed1301 跑满
510 条后，action permutation、zero、matched wrong-condition 与 positive-session-rate
四门全部失败，official verdict=`not-supported`；按预注册不再运行 1313/1327。

Digital Ant ecology 的 fresh station1-v4 最终为 `BLOCK`：alignment `3/4`，
`next_episode_authorized=null`，所以 station2、Gate 4 ecology admission、P1/P2 都没有
证据，也不得靠换 seed、加训练量或降低门槛重开。

## 5. 还没有完成的事

### 代码与产品面

- World / Self predictive model 的更高容量、compositional/counterfactual 表示；
- memory retrieval ranking、tension/lesson extraction 等 learned 候选的进一步 owner 化；
- Relationship Memory Console 的独立 UI、P5 七日 continuity aggregator 与 P6 自动 apply；
- 三个外部调用项目已完成 adapter 迁移：ProductZero→Coding、Foundry→Venture、
  AutoCompany→Operations；调用端各自持久化 lineage/receipt，且不把 Brain advice 当 actuator；
- Brain service/controller 自身的 live Context Pack 幂等 ledger 仍是进程内状态；service restart 后
  调用端必须请求新 pack。合格 advisor 现场证据与任何 Advice production ACTIVE 晋升仍未完成；
- DLaaS、Companion Bench、Figure/Growth Advisor 的具体开放项以
  [known-debts.md](./known-debts.md) 为准；
- 跨模态 latent action basis、开放环境因果结构发现和 mesa-objective detection 仍属研究目标。

### 证据面

- SHADOW learner 需要在真实部署数据上积累 settle、matched control、validation
  delta、性能、安全与 rollback 证据；
- 任何新整体 thesis 必须重新提出总 EXIT 并独立预注册，不能重开 #92 改写历史；
- Gate 2/8/11 的局部支持不能外推为一般关系质量、整体人类世界模型或生产 learned takeover。

## 6. 推进与回滚纪律

1. 先查 [00_INDEX.md](./specs/00_INDEX.md) 和能力 spec，确定唯一 owner。
2. 新行为先 SHADOW，冻结 matched evidence；每次只晋升一个 component。
3. ACTIVE canary 必须保留 `WiringLevel` rollback、checkpoint 和 artifact fingerprint。
4. evaluation 只裁决，不能回流成 PE / credit 学习源。
5. 失败的预注册门是合格终态；新机制必须新提案、新 schema、新 capture。

## 7. 权威阅读顺序

1. [currentstatus.md](./currentstatus.md)：当前实现与近期补齐；
2. [thesis prove.md](./thesis%20prove.md)：#92、Ecology、Gate 2 L3 的终局证据；
3. [known-debts.md](./known-debts.md)：仍开放的工程与证据债；
4. [DATA_CONTRACT.md](./DATA_CONTRACT.md)：slot、snapshot 与 owner 契约；
5. [specs/00_INDEX.md](./specs/00_INDEX.md)：能力域入口。
