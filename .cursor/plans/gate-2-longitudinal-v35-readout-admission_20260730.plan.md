# Gate 2 longitudinal：v35 open-loop readout admission

## 1. 目标与边界

本包只解决 Gate 2 longitudinal 的输入接缝：把 v35
`ef360e0e72e00d235e7fc0df39b249178e080bf2065c6443dad801dfd77f4293`
selector 冻结为只读 readout，审计既有
`gate11-longitudinal-settled-trace.v2` 是否已经具备运行 readout 和计算
matched `validation_delta` 的充分字段。

禁止事项：

- 不拟合、更新或替换 selector；
- 不向 substrate 注入 control；
- 不修改 shared source、memory、temporal policy 或 session state；
- 不把 selector 未执行时的固定环境结果归因给 selector；
- 不用 action 名、场景文本或关键词重建 22 臂 outcome。

## 2. Owner 与正式交换

- 真 trace、settlement lineage 与 session 边界：既有 runtime source owner；
- 8076 维 residual readout state：必须由 full-width real-substrate capture
  经 temporal 的 `residual_action_state_vector()` 正式发布；
- selector 预测：v35 frozen temporal artifact，只读；
- selector-aligned matched outcome：runtime evidence harness 在隔离的
  counterfactual lane 发布，必须同时含 selected / zero / permutation-null
  的 realized-continuation outcome；
- admission 与 verdict：`vz-runtime` evidence owner。

现有 source 不得原地扩写。后续 capture 包以 companion JSONL 发布
`selector_readout_inputs.jsonl` 与 `selector_matched_outcomes.jsonl`，通过
`transition_id` 一对一 join。

## 3. 预注册 admission gates

### 3.1 Source gates

每个 seed 必须同时满足：

1. settled transition `>=500`；
2. real substrate rate `=1.0`；
3. fallback rate `=0.0`；
4. frozen substrate mutation count `=0`；
5. consumer session count `>=2`；
6. transition id 唯一，既有 source digest / registry prefix 校验通过。

### 3.2 Frozen selector gates

1. selector schema=`residual-action-selector.v1`；
2. model kind=`linear-kernel-ridge-v1`；
3. input dim=`8076`；
4. action count=`22`；
5. fingerprint 精确等于 v35 锁定值；
6. control basis fingerprint 精确等于
   `326aecddc8d0b7e81161568121d457267d8473c22dc74c11b0dc1396b4d9761b`；
7. artifact JSON round-trip fingerprint 校验通过。

### 3.3 Readout readiness gates

每条 source transition 必须有且只有一条 companion input 和 outcome：

- input 携带 8076 个有限值、real capture、fallback=false、mutation=false，
  并绑定 v35 selector / basis fingerprint；
- outcome 携带同一 `transition_id` 的 selected action index、selected
  realized delta、zero delta、permutation-null mean 与完整 lineage；
- selected action index 位于 `[0, 22)`；
- matched outcome 必须声明由 isolated counterfactual lane 产生，不能复用
  source 固定 `task_progress/action_payoff`。

任一缺失时：

- `readout_ready=false`；
- `promotion_allowed=false`；
- longitudinal verdict 保持 `not-supported`；
- v35 原 causal verdict 只作 inherited evidence，不撤销也不升级。

## 4. 后续正式 longitudinal 门（本 admission 包不计算）

companion 输入与 outcome 齐备后，才允许一次性计算：

- primary：跨 session 的 selector selected realized delta 相对
  permutation-null mean；
- Gate 2 EXIT：`validation_delta >= 0.02`；
- matched controls：zero、permutation-null，以及既有 no-replacement /
  no-optimize / ETA-off 对照；
- 规模：每 seed `>=500` real settled traces，关键结论跨 seed；
- 额外不变量：readout 期间 selector / basis / source SHA256 前后一致，
  runtime mutation count=0。

## 5. 止损与回滚

- 若 source gates 失败，先修 source provenance，不启动 capture；
- 若 source gates 通过但 companion 缺失，判 `capture-required`，不得用
  稳定性、action 分布或固定 outcome 代替 `validation_delta`；
- capture 完成后若正式 delta 低于 `0.02`，按 Gate 2 EXIT 收缩到
  “latent controller 可运行”，不调阈值续命；
- 本包只有新 evidence harness、测试、计划和只读 artifact。回滚为删除这些
  新文件；shared source、v35 artifact 与 live wiring 均无需恢复。
