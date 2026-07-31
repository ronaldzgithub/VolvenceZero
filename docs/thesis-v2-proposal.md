# Volvence Thesis v2 提案：有界产品连续性

> 状态：`preregistered-proposal / not-yet-retained`
>
> 提案日期：2026-07-31
>
> 对应债务：[#93](known-debts.md#93)
>
> 历史终局：[#92 thesis-rejected](thesis%20prove.md)

## 1. 一句话主张

Volvence v2 不再尝试用一个 EXIT 同时证明所有 NL/ETA learned uplift。它只验证：

> 在冻结基底、owner 隔离、可审计跨 session 持久化与可回滚边界下，Gate 8
> wake/sleep 与 Gate 11 per-user state 所产生的连续性差异，能否被非项目人类评审者
> 在 matched 纵向 transcript 中稳定感知。

Gate 2 仅作受限 open-loop `z_t` causal 机制证据，不再尝试进入 v2 纵向主张。

## 2. 继承但不改写的证据

| 来源 | v2 继承的结论 | v2 不允许的改写 |
|---|---|---|
| #92 终局 | mechanism coverage 完整；Gate 2/8/11 有局部支持；full-chain rollback 通过 | 不把 `thesis-rejected` 改成 retained |
| Ecology L1 | station1 结构门成立，alignment 3/4 终局 `BLOCK` | 不冒充 station2/P1/P2 已执行 |
| Gate 2 L3 | 新 Relationship-conditioned carrier 真实存在 | 不忽略 seed 1301 stop-loss，不宣称 longitudinal gain |
| Gate 8 | wake/sleep 在 deterministic owner readout 上 longitudinal-supported | 不冒充人类关系质量真值 |
| Gate 11 | isolated per-user continuity longitudinal-supported | 不冒充“用户感到被记住” |

权威 source hashes 在 machine-readable v2 preregistration 中冻结。
该 preregistration 为
[`artifacts/thesis_v2_product_continuity_prereg_20260731T181500Z.json`](../artifacts/thesis_v2_product_continuity_prereg_20260731T181500Z.json)，
SHA-256 `8bcabb75a6d63068d3dc40e6cbd7e9497560f17cab364017a1cfb76b6fb8f3c2`。

## 3. v2 保留的四类主张

1. **有界 per-user continuity**：Gate 11 的 owner state 可隔离、持久化、跨重启恢复，且
   correct-state 在冻结 owner metrics 上优于 stateless/swapped/shuffled。
2. **wake/sleep consolidation**：Gate 8 的 sleep arm 在冻结 owner metrics 上优于
   no-sleep 与 single-owner controls。
3. **受限 temporal causal control**：Gate 2 v35 支持 open-loop residual action-value selector；仅此而已。
4. **可治理基础设施**：owner、immutable snapshot、PE lineage、replay、persistence 与 rollback
   可审计。这是安全/存在性证据，不是 learned uplift 证据。

## 4. 排除项

下列主张不属于 v2 EXIT，也不能在 v2 中作为任何其他门的 substitute：

- 多频 CMS / ATLAS / Titans 相对单频有稳定、有意义的净增益；
- PE 驱动 runtime behavior 稳定优于 PE-off；
- learned 主动学习稳定节省人类标签；
- SSL→RL takeover 产生 terminal/composition 净增益；
- M3 慢更新优于 plain momentum/SGD/Adam；
- rare-heavy candidate 可自动晋升 production；
- Ecology medium 或一般物理自主性；
- Gate 2 longitudinal takeover 或 live relationship-conditioned selector。

这些条目只能以“新 owner 机制 + 新 schema + fresh source + 独立 prereg”重开各自的研究线。

## 5. 唯一新证据面：Gate 8/11 人类 anchor

L4-A 已冻结 `gate811-human-anchor-prereg.v1`：

- Gate 11：`correct-user-state vs stateless`；
- Gate 8：`sleep-consolidation vs no-sleep`；
- 每份 transcript 三 session / 30 turns，共享 user turns、source lineage、persona、seed、
  model/adapter fingerprint，只改 comparison arm；
- pilot 每 contrast 24 pairs、3 ratings/pair、至少 6 名非项目 rater；
- formal 样本量由 pilot power 分析在 60–300 pairs/contrast 间冻结；
- preference、Likert composite、boundary non-inferiority、ordinal agreement 四门全过才能
  认定单个 gate human-anchored。

L4-B 已落地 typed capture 校验、consent/PII attestations、exact matched pairing、
盲包/内部 key 分离和评分 CSV。工具完成不等于人类证据完成。

## 6. 总 EXIT

v2 只有两个终态，不保留“再跑一点看看”的中间态：

### `product-continuity-retained`

同时满足：

1. 所有 inherited source hashes 与 L4 prereg 未漂移；
2. pilot 一致性门过，power artifact 在 formal outcome 前冻结；
3. Gate 8 formal human anchor 全门过；
4. Gate 11 formal human anchor 全门过；
5. reconciliation 明确 human ratings 只是 evaluation readout，未写入 reward/credit/runtime state；
6. reconciliation 显式 `production_live_promotion_authorized=false`。

### `product-continuity-rejected`

任一 human-anchor gate 失败即进入此终态。失败 gate 的关系质量主张永久限定为
`system_self_eval` / owner metric；另一 gate 如果通过，可保留该 gate 的独立 human-anchored
措辞，但不足以让 v2 整体 retained。

## 7. 不授权的事

v2 即使 retained，也**不自动授权**：

- 翻转任何 `WiringLevel`；
- 安装 Gate 2 L3 selector；
- 打开 CMS Torch、SSL→RL takeover、M3 或 rare-heavy 自动晋升；
- 把 human rating 用作 reward、credit 或训练 label；
- 声称完整 NL+ETA thesis、一般世界模型或物理自主性已被证明。

任何 production 变更仍需独立组件级 promotion gate、rollback plan 与人工 review。

## 8. 当前状态

```text
proposal = thesis-v2-product-continuity
status = preregistered-proposal
inherited_negative_evidence = true
l1_ecology = terminal-block
l3_gate2_longitudinal = single-seed-stoploss
l4a_protocol = complete
l4b_tooling = complete
l4c_pilot_power = pending-external
l4d_formal_human_anchor = pending-external
production_live_promotion_authorized = false
```

当前阻塞不是代码：是 consented fresh transcript 采集与真实非项目评审者。
