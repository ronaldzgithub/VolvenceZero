# 证据计划 Spec

> Status: draft
> Last updated: 2026-04-25
> 对应需求: R12, R15

## 要解决的问题

如何把内部 benchmark / proof harness 升级为可复现、可审阅、可回放的证据生产线，使系统的对外主张能被明确映射到 gate、artifact、盲评和统计结论，而不是靠单次 run 或主观叙述。

## 关键不变量

- 对外主张必须先冻结成可证伪 claim，再绑定 required gates、artifact 和 verdict 规则
- claim verdict 必须回溯到 manifest、seed、git sha、依赖版本和原始 artifact
- dialogue / ETA paper-suite 共享统一的 evidence bundle 口径，不各自发明一套 summary schema
- 盲评外发包不得泄漏 profile label 或内部 case 标识；profile 映射只存在 internal key
- 人评不是替代自动评估，而是额外证据面；自动指标、人评与 claim verdict 必须能并列审阅
- open-environment widening evidence 必须区分 `open_core`、`open_families`、`open_heldout`，不能把单一固定场景误写成开放泛化

## 工程挑战

- 设计 claim registry，把抽象宣传口径压成具体 gate
- 统一 dialogue / ETA aggregate 报告的 pairwise effect、claim verdict 和 evidence bundle 导出
- 让 blind review packet 真正可外发，同时保留内部 unblinding key
- 为人评建立最小协议和可机读 aggregate，而不是只导出一组 transcript
- 让 repeated-run summary 不只给 interval，还能给 matched-control effect 与 retained / weak / fail verdict

## 算法候选

证据计划属于评估与 rollout 审计层，受 R12 / R15 约束：

- evaluation 仍是 PE-first 主链的 readout / gate / widening evidence，不替代 learning primitive
- claim verdict 基于 matched-control comparisons、longitudinal evidence、blind review 与 provenance
- open-environment 作为 widening surface，只能在 held-out 覆盖与统计口径满足时支撑更强 claim

## 接口契约

**消费的输入**：
- dialogue comprehensive / paper-suite aggregate
- ETA proof paper-suite aggregate
- NL essence / ETA acceptance gates
- blind review packet、human rating entries、human rating aggregate
- manifest / provenance / repeated-run summaries / pairwise metric effects

**产出的输出**：
- claim registry / claim verdicts
- external-safe blind review packet
- internal unblinding key
- human rating template / aggregate
- unified evidence bundle

当前实现口径：

- `volvence_zero.agent.paper_suite` 提供共享 `ClaimVerdict` 与 `EvidenceBundle`
- dialogue / ETA paper-suite aggregate 会额外发布 pairwise effects 与 claim verdicts
- dialogue paper-suite export 会同时导出 blinded packet、internal key、rating template、rating aggregate 与 unified evidence bundle
- ETA paper-suite export 会导出统一 evidence bundle，复用相同的 claim verdict / pairwise effect 口径

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 评估体系 | claim verdict 消费 evaluation / benchmark evidence |
| 依赖 | 契约式运行时 | provenance 与 artifact 必须回溯到真实 runtime 产物 |
| 依赖 | 多时间尺度学习 / 时间抽象 | claim registry 需要把这些设计命题绑定到可观测 gate |
| 协作 | 调试体系 | blind review / paper-suite 工件是 widening 与审计面 |
| 被依赖 | rollout / 外部汇报 | evidence bundle 是对外结论、候选比较与审稿材料的统一入口 |

## 初始 Claim Registry

- `claim_pe_multi_timescale_default`
  - 命题：`PE-first + multi-timescale` 是默认路径上的机制事实
  - 需要：`pe-first`、`multi-timescale-default`、`judge-gated-evolution`、`cross-session-growth`
- `claim_temporal_advantage_over_controls`
  - 命题：时间抽象与在线适应在 matched controls 前有稳定优势
  - 需要：dialogue / ETA pairwise effects 为正，且 gap 不只是单次最好结果
- `claim_beyond_scripted_canonical`
  - 命题：优势不只存在于 canonical scripted cases
  - 需要：perturbation / systematic replay / open-environment / held-out families 共同给证据
- `claim_external_human_legibility`
  - 命题：优势能被外部人类评审感知
  - 需要：blinded packet、多评审员评分、inter-rater agreement 与自动指标相关性

## 变更日志

- 2026-04-25: 初始版本，建立 claim-to-evidence / blind-review / pairwise-effect / evidence-bundle 的统一口径
