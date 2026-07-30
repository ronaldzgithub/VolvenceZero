# Gate 2 longitudinal：v35 read-only companion capture

## 1. 本包唯一目标

为已经 source-admitted 的
`gate11-longitudinal-settled-trace.v2` 发布独立 companion：

- `selector_readout_inputs.jsonl`：clean full-width residual →
  v35 的 8076 维 state；
- `selector_matched_outcomes.jsonl`：隔离前向中的
  zero / v35-selected / preregistered permutation control outcome。

v35 selector 固定为只读 readout。它不更新、不进入 session、不写 temporal /
memory / credit owner；control 只在 frozen Qwen 的隔离 teacher-forced scoring
forward 内存在，不进入 source 或 live wiring。

## 2. 冻结执行契约

- model：`Qwen/Qwen2.5-0.5B-Instruct`，strict local，CPU；
- hook layers：`20 / 21 / 22`，activation width=`896`；
- selector：v35 kernel ridge，fingerprint
  `ef360e0e72e00d235e7fc0df39b249178e080bf2065c6443dad801dfd77f4293`，
  input/action shape=`8076×22`；
- learned basis：只从 v35 train routes 重新捕获 transition delta 并按既有
  `train-transition-pca-v1` 算法重建；fingerprint 必须精确为
  `326aecddc8d0b7e81161568121d457267d8473c22dc74c11b0dc1396b4d9761b`；
- candidate-index → applied-control：从 v35
  `counterfactual_outcomes.jsonl` 冻结。每个 index 必须只有一个 3 维有限
  control，22 个 index 齐全，index 0 为严格 zero；
- track scale：`(0.7, 0.7, 0.7)`，与 v35 realized-continuation lane 一致；
- outcome：source 的 `prediction_turn` 作 prefix，`settlement_turn` 作
  teacher-forced continuation，raw delta=`zero_nll - arm_nll`。

任一 fingerprint、shape、candidate mapping、source digest 或 prefix
tokenization 不匹配，立即失败，不写伪兼容 fallback。

## 3. Matched permutation null

为避免每条重跑 22 臂，同时保持与 selector 无关的可交换 null，冻结：

`permutation_action_index = (global_index + seed_rank * 7) mod 22`

其中 seed rank 按 `1201 / 1213 / 1223 -> 0 / 1 / 2`。每 seed 510 条使每臂
被分配 23 或 24 次；schedule 只依赖预注册 index/seed，不读取文本、state、
selector value 或 outcome。row-level `permutation_null_mean` 是该 matched
permutation draw 的 realized delta；seed/session aggregate 才是 null
expectation 的估计。另报告 selected 恰等于 permutation 的碰撞率，不能删行。

## 4. 正式指标与止损

每 10 transitions 为一个 consumer session，整条 source 不挑行。

每 seed primary：

- `selector_minus_permutation_mean`；
- `selector_minus_zero_mean`；
- 51 个 session mean 的正向率；
- selector top-1 action coverage / entropy（诊断，不是门）。

正式 readout gate：

1. 每 seed settled/readout/matched count 均 `>=500`；
2. readout input dim/fingerprint/source lineage 100%；
3. fallback=0、mutation=0，source/selector SHA256 前后一致；
4. 每 seed `selector_minus_permutation_mean >= 0.02`；
5. 每 seed `selector_minus_zero_mean >= 0.02`；
6. 3 seed primary mean 的 95% t-CI 下界 `>=0.02`；
7. 每 seed session primary 正向率 `>=0.60`。

先跑 seed 1201。若它在完整 510 条上的第 4、5 或 7 门失败，判 single-seed
stop-loss，不启动 1213/1223；不得调 selector、basis、permutation schedule、
track scale 或阈值重跑同一 source。通过才补另外两 seed。

本包通过只产生 `longitudinal-readout-supported` 子证据；Gate 2 官方
`longitudinal-supported` 还要在后续 reconciliation 中继承 v35 causal
controls 并对齐 Gate 2 EXIT 的 abstraction-quality 条款。本包失败则
Gate 2 longitudinal 保持 `not-supported`，v35 causal verdict 不撤销。

## 5. Artifact、续跑与回滚

- companion root 不写进 immutable Gate 11 source；
- 每条 JSONL 含 canonical `record_sha256`，逐条 append + flush；
- resume 只接受 registry 的严格 prefix；重复/跳行/摘要漂移均 fail loudly；
- formal artifact 记录 source、selector、candidate artifact 的前后 SHA256；
- 回滚为删除 companion artifact。source、selector、模型权重和 runtime
  owner state 不需要恢复。
