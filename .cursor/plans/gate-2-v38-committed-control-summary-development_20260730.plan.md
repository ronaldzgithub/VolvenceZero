# Gate 2 v38：committed-control summary state development

## 1. 冻结前提

- v35 open-loop causal packet 的 `promotion_allowed=true` 保持只读继承。
- v37 recent-k=2 fresh formal 已因 locked confirmation 的
  selector−permutation `−0.011889` 按单 seed stop-loss 判定 NO-GO；
  不补跑 seeds 1/2，不重用 v37 validation/confirmation 做开发或调参。
- `committed_control_window=2`、22 个 candidate、learned control basis、
  ridge ladder、阈值、Qwen2.5-0.5B、CPU、width 896、prefix 8 与 epochs 2
  全部冻结。
- 本包只改变 selector state representation 与为该 state 生成 train-only
  counterfactual rows 的方式；live injection 继续 disabled。

## 2. 唯一变量与 owner

`vz-temporal` 是 selector state representation 的唯一 owner，新增
`residual-state+committed-control-summary.v1`：

- 原 full residual mean/latest/trend 与 12 维 summary 保持逐字节不变；
- 追加 10 个有界坐标：
  - 最近两个 committed controls 的 aggregate sum 经 `tanh`：3 维；
  - latest committed control 经 `tanh`：3 维；
  - latest−previous 经 `tanh`：3 维；
  - `active_count / 2`：1 维；
- 空历史严格发布十维零；维度、window、非有限值或 control shape 漂移必须
  fail loudly。

Gate 2 full-width 896 的输入从 `8076` 维增至 `8086` 维。该 feature 只进入
冻结 selector artifact，不新增 runtime slot，不修改 substrate 或生产 session。

## 3. Train-only 两阶段采集

1. 按 v37 以前的正式路径只用 16 条 train routes 拟合并 round-trip 冻结
   原 selector，作为 bootstrap behavior；其 fingerprint 只作 lineage。
2. bootstrap selector 在同 16 条 train route 上顺序产生 k=2 committed
   histories。每个 prefix 重新对全部 22 个 candidate 执行真实 residual
   forward；zero candidate 表示提交零控制后的 k=2 aggregate，其他 candidate
   使用同一 prior history 加当前 decoded control。
3. EnvironmentOutcome → PredictionError → `pe:action` credit owner 链保持
   不变；summary selector 只消费这些 train-only candidate rows。evaluation、
   heldout、validation、confirmation 不得进入 fit 或 model selection。
4. summary artifact 必须携带 feature contract、bootstrap fingerprint、
   control-basis fingerprint、window 与 train-only provenance。

## 4. Development evidence 与止损

本包仅复用已经观察的 v36 routes，schema
`eta-gate2-control-summary-diagnostic.v1`，seed 0，不能产生 formal promotion
或 SHADOW admission verdict。

发展门要求 train / validation / confirmation / development-heldout 四个 split
的以下三项全部 `>= 1e-6`：

- selector-minus-zero；
- selector-minus-permutation；
- selected step realized delta mean。

任一失败即冻结该 summary contract 为 development NO-GO，不调 scale、不改
summary 坐标、不搜索 window。全部通过才允许另开 v39 fresh formal 包；v39
必须使用全新 validation 与 locked confirmation。

## 5. 文件与验证

- `vz-temporal/internal_rl/counterfactual_selector.py`：feature owner。
- `vz-runtime/eta_proof_benchmark.py`：train-only history/counterfactual
  collection 与 SHADOW consumer。
- `vz-runtime/eta_gate2_residual_evidence.py`：development manifest/export。
- `scripts/run_eta_gate2_residual_evidence.py`：显式 development-only 入口。
- temporal/runtime targeted tests、Ruff、diff check。
- 真 Qwen seed 0 development run；不运行 3 seeds。

## 6. 回滚

feature flag 默认关闭；关闭后 selector fit、artifact、闭环 feature 与 v37
逐字节保持原行为。删除 v38 builder/flag 与 summary feature helper 即完成回滚，
生产 runtime 无迁移。

## 7. 结果（2026-07-30）

真 Qwen2.5-0.5B、CPU、width 896、prefix 8、epochs 2、seed 0 development
run 已完成。summary selector 输入为 8086 维，bootstrap selector fingerprint
`ef360e0e…`，summary selector fingerprint `8546fa15…`，control basis
fingerprint `326aecdd…`。483 条三臂记录全部 k=2、active count≤2 且
side-effect free；只使用 v36 已观察 routes，未使用 v37 formal routes。

| split | selector−zero | selector−permutation | selected step mean |
|---|---:|---:|---:|
| train | +0.163939 | +0.135222 | +0.029145 |
| validation-v36 | +0.041222 | +0.058689 | +0.006870 |
| confirmation-v36 | +0.094909 | +0.069266 | +0.015818 |
| eval（诊断） | +0.036992 | +0.145781 | +0.006726 |
| heldout（止损门） | **−0.022690** | **−0.022389** | **−0.003782** |

summary state 相比 v36 k=2 将 heldout 三项从
`−0.058184 / −0.057883 / −0.009697` 收窄，但没有翻正。预注册四分区门因此
失败，`development_gate_passed=false`。本 feature contract 判定 development
NO-GO：不进入 v39 fresh formal，不改 summary scale/坐标，不重搜 k，live
injection 继续 disabled。v35 open-loop causal packet 保留为只读历史证据，
Gate 2 closed-loop SHADOW admission 仍未闭合。

权威 artifact：
`artifacts/eta_gate2_v38_control_summary_development_fullwidth896_qwen25_05b_cpu_seed0_20260730/`。
