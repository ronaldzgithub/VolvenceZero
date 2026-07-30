# Gate 2 v37：recent-k=2 fresh formal SHADOW admission

## 1. 冻结前提

- v35 causal packet 已锁定为 `promotion_allowed=true`，本包只继承，不重跑。
- v36 full-history `shadow_observation_passed=false` 与 recent-k development
  结果保持不可变。
- development 已唯一选择 `committed_control_window=2`；本包禁止再比较或调整
  k，禁止修改 selector features、basis、候选集、阈值与 prefix 长度。
- `counterfactual_action_selector_live_injection=disabled`，本包仍只运行
  evidence harness，不接真实 session。

## 2. Fresh corpus

保留原 16 条 train 与既有 eval/heldout development routes；用 4 条全新的
`validation-v37-*` 与 4 条全新的 `confirmation-v37-*` 替换 v36 正式分区。
v35/v36 的 validation/confirmation 全部登记为 superseded，不参与 v37 门。
新分区在任何模型 forward 前冻结，并通过词汇不相交契约测试。

## 3. Formal gate

schema：`eta-gate2-residual-causal.v37`。

单 seed stop-loss（seed 0）要求 train / fresh validation / locked confirmation：

- 三臂记录完整且 lineage / fingerprint 有效；
- selector-minus-zero `>= 1e-6`；
- selector-minus-permutation `>= 1e-6`；
- selected step realized delta mean `>= 1e-6`；
- seed 方向为正。

任一失败立即 NO-GO，不运行 3 seeds，不调阈值。

单 seed GO 后运行 seeds `(0, 1, 2)`；`shadow_observation_passed=true` 还要求：

- 三分区上述门全部满足；
- 每分区 3/3 seed 为正；
- selector artifact、basis、runtime lineage 完整；
- 所有记录 side-effect free。

heldout/eval 继续作为已观察 development diagnostics，不进入 formal 门，但负向
结果必须作为 residual risk 报告，禁止隐去。

## 4. Owner 与文件

- `vz-runtime/eta_proof_benchmark.py`：v37 corpus owner 与 case selection。
- `vz-runtime/eta_gate2_residual_evidence.py`：v37 manifest/schema/verdict owner。
- `scripts/run_eta_gate2_residual_evidence.py`：显式 `--recent-k2-formal` 入口。
- runtime Gate 2 tests：freshness、manifest、窗口与单/三 seed gate。
- `docs/specs/evidence_program.md`：预注册与结果回写。

不修改 `vz-temporal` selector，不修改 `vz-substrate`，不修改 session wiring。

## 5. 执行与止损

1. 先写本计划与 evidence spec 预注册段。
2. 实现 corpus / manifest / schema，运行 targeted tests、Ruff、diff check。
3. 真 Qwen CPU seed 0 fresh probe。
4. seed 0 GO 才运行 3-seed formal。
5. formal 失败则 recent-k 方向止损，按 v36 预注册剩余方向转向 committed-control
   summary state features；formal 通过才允许另开 runtime SHADOW wiring 包。

## 6. 回滚

v37 是独立 manifest/corpus；删除 v37 builder 与 routes 即回到 v36 默认
full-history。生产 runtime 从未接入，因此无需线上数据迁移。
