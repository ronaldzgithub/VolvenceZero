# Companion Prediction Thesis v3 — 预注册与独立 EXIT

> Status: preregistered; formal run not yet eligible
> Frozen: 2026-08-01
> Claim boundary: 只讨论预测性与成本，不讨论“用户感到被记住”、关系质量或 AGI
> Amendment: 2026-08-01 物证审计后移除 §8 的 pilot 数字；命题、阈值与 EXIT 未改

## 1. 命题

对同一位真人的多 session 对话，若系统形成了有用且有界的关系状态，则它对
该真人下一轮话语表示的预测，应随可用关系历史增加而改善；即使质量不超过
全量长上下文，它也可能以显著更低的 token 与延迟成本达到等质量。

未来本身是唯一 target。禁止使用 evaluation、owner readout、模拟用户分数或
LLM judge 生成 N+1 标签。MSC 的角色扮演与后续 session 换 worker 威胁使本研究
只能声称 predictive continuity，不能声称自然形成的人际关系。

## 2. 冻结数据与身份

- Corpus: official Multi-Session Chat v0.1，非商业研究用途；原文不入库。
- Train: 1001 dyads / 4 sessions，sorted-id SHA-256
  `59f7abe7c2625a6d5629c1bbfa09557ae3559b8de3c1b6eed47bec2092975fca`。
- Validation: 500 dyads / 5 sessions，sorted-id SHA-256
  `8976ab8e8859a2861b1f038469c317780beb4b5af2c7676362639b98dc2443a5`。
- Heldout: 501 dyads / 5 sessions，sorted-id SHA-256
  `58a61e1b08a9d0ae384b413a677e161d2e809cacc1bd81ba79beb557588e5777`。
- 固定把 `speaker_1` 视为被预测的人；每个 sample 截止在其真实话语之前。

## 3. 四臂与共同控制

四臂必须使用同一冻结基底/表示 encoder、同一目标表示、同一 split、同一
sample ids、同一 seeds 与相同 PE head 训练预算；只改变可用状态：

1. `volvence`：完整运行时 owner/snapshot 主链，必须有 full-stack attestation；
2. `stateless`：persona/system 条件与最新 partner message；
3. `long_context`：全部历史直接进入冻结基底，超窗只按 recency 截断；
4. `summary_retrieval`：同一冻结 encoder 支持的摘要/检索状态。

每臂至少 3 个预先声明 seed。长上下文必须报告 raw/accepted/truncated tokens；
各臂报告端到端 latency。离线 bounded-state prototype 可以调试 PE owner 与评估
器，但不得把 `volvence_full_stack` 标为真。

## 4. 容量选择

只在 train 学习、validation 选择 `n_z ∈ {3,16,64,256}`；heldout 禁止参与选择。
容量曲线的最小有意义增益为 mean cosine `0.01`（相对 `n_z=3`）。该 ladder
直接授权的只是 N+1 表示 head 容量；除非另有 temporal-owner 因果实验，禁止
据此修改 temporal controller 默认值或删除 `_step_impl_legacy`。

## 5. 主指标与统计

- 主质量指标：predicted/observed N+1 representation cosine；MSE 为共同 loss 与
  次指标；persistence 使用最新话语表示并在同 target 上结算。
- 横轴：MSC session index（1–5）；同时保留 `history_turns`。
- 置信区间：先在 `(seed, sample)` 配对计算 Volvence−long-context，再按 dyad
  聚合；以固定 seed `20260801 + session_index` 做 2000 次 dyad bootstrap。
- 成本：accepted context tokens、truncated tokens、端到端 latency 的 session 曲线。

## 6. 两种胜利条件

### EXIT A — `QUALITY_ADVANTAGE`

在 session 5 同时满足：

- mean cosine advantage ≥ `0.02`；
- dyad bootstrap 95% CI lower > `0`；
- session 1–5 advantage 的线性 slope > `0`。

### EXIT B — `SCALING_ADVANTAGE`

在 session 5 同时满足：

- cosine advantage ≥ `-0.01`；
- Volvence/long-context accepted-token ratio ≤ `0.10`；
- Volvence/long-context latency ratio ≤ `0.50`。

若 A/B 都不满足，独立总 EXIT 为 `REJECT_AND_SIMPLIFY`：关系状态/owner 架构
未挣得复杂度，停止扩大该 thesis，不改写 #92。

## 7. Evidence eligibility

正式 adjudicator 必须同时看到：官方 heldout hash、501/501 heldout dyads、四臂
完全 sample-matched、≥3 seeds、冻结 encoder fingerprint、完整 Volvence runtime
attestation。任一缺失都强制 `INELIGIBLE_PILOT`，即使点估计超过阈值也不能
选择 thesis。

## 8. 执行状态（不是结果）

**正式四臂实验尚未执行，thesis v3 没有结果。** 预注册正文不得冻结或转述
mechanism pilot 的点估计、置信区间、模型 fingerprint 或容量曲线；正式结果只能
来自满足 §7 全部资格条件的独立 artifact。

2026-08-01 的小样本 CPU run 已从官方 MSC 语料重跑并保存在
[`artifacts/msc_n_plus_one_mechanism_pilot_20260801/`](../artifacts/msc_n_plus_one_mechanism_pilot_20260801/)。
其 `artifact_manifest.json` 对语料 provenance、六个结果文件与关键源码逐文件
SHA-256；同时硬标记 `evidence_level=mechanism-only-pilot`、
`thesis_status=not-evaluated`、`formal_experiment_executed=false`。

它不构成本文结果，原因不是样本量一项，而是实验对象仍不成立：

1. `volvence` 臂是绕过 `propagate` 的 bounded-state prototype，不是完整 runtime；
2. `long_context` 受 MiniLM 的 256-token 上限约束，不是现代长上下文 steelman；
3. target 是外部 MiniLM 句向量，不是冻结 substrate 的 N+1 内部表示；
4. `n_z` ladder 改变的是 forward head 容量，不是 temporal controller 容量。

后续 target-only 收敛包已把 N+1 target 改为 substrate-owned Qwen residual，并
单独保存于
[`artifacts/msc_n_plus_one_substrate_target_smoke_20260801/`](../artifacts/msc_n_plus_one_substrate_target_smoke_20260801/)；
它只关闭上列第 3 项，不改变本预注册的结果状态，也不把 smoke 指标写回本文。

其余接线完成并产出新的正式 bundle 前，本文唯一合法状态是
`formal run not yet eligible`；pilot 不授权 thesis、temporal promotion、legacy
controller 退役或产品关系质量主张。
