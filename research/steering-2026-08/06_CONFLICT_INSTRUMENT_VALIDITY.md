# 06 · 冲突映射仪器有效性（P2c · C1，已执行）

- owner 模块：[`eta_conflict_instrument.py`](../../packages/vz-runtime/src/volvence_zero/agent/eta_conflict_instrument.py)（只读；不训练、不加 bias、不改 production）
- run 脚本：[`scripts/run_eta_conflict_instrument_validity.py`](../../scripts/run_eta_conflict_instrument_validity.py)
- 结构产物：`artifacts/eta_conflict_instrument_20260804/`（`report.json` / `report.md` / `train_headroom.json` / `artifact_manifest.json`）
- 测试：[`test_eta_conflict_instrument.py`](../../packages/vz-runtime/tests/test_eta_conflict_instrument.py)（5 passed）
- corpus：与 S1/S2 同图同参（seed 20260802 / obj 8 / corridor 2 / edge 0.35 / heldout 24 / lengths 3,4）
- 基底：frozen merged S1 模型 `artifacts/eta_stage2_merged_v2_20260803`，layer 20 / 896

## 一句话结论

**仪器有效性 = VALID。** P2a 说 V4 仪器"对条件干预无余量"——那是**仪器**问题，不是概念死路。把观测**目标剥离**（只留当前节点 + 出边，不announce目标）后，同一局部视图在不同 active_subgoal 下要求不同动作，形成**冲突映射**：恒定算子按结构必然错 46%，而 subgoal 完美消歧（残余 0）。基底在目标隐藏时对动作**高度不确定（NLL 2.81）**，一旦告知 subgoal 立刻近确定（0.22）——**2.60 NLL 的可 steer 余量因果归属于 subgoal 这一比特**。这正是条件化学习式 steering 需要且此前 V4 缺失的空间。

## 量化对比

| 指标 | V4 staged-plan（P2a，无余量） | V5 goal-ambiguous junction（C1） |
|---|---|---|
| 恒定算子错误率 | ≈0%（打满） | **0.461** |
| (view, subgoal) → action 残余歧义 | — | **0** |
| oracle 条件算子错误率 | — | **0.000** |
| 冲突行占比 | — | **1.000** |
| 基底 expert NLL（目标隐藏） | 0.168（mean noop） | **2.813**（median 2.787） |
| 基底 expert NLL（目标揭示） | — | **0.218** |
| 可 steer 余量（隐藏 − 揭示） | 静态增益 +0.0002 | **2.595 NLL** |
| 基底不确定占比（NLL>0.1） | — | **0.806** |

## 为什么这是对的诊断（第一性）

1. **恒定算子错误 46% + oracle 条件错误 0%**：这两个数放在一起是关键——它证明 heldout 上"无条件映射"必然错近一半，而"加上 subgoal 条件"降到 0。即：可填的余量 = 46 个百分点，且**唯一缺失的信息就是 active_subgoal**。V4 仪器这两个数都≈0，所以 permuted-z 惩罚为 0、切换冗余（P2a）。
2. **2.60 NLL 的余量因果归属 subgoal**：目标剥离 vs 目标揭示是配对对照（同前缀、同动作、只差一句 `Objective: X`），NLL 从 2.81 掉到 0.22。这不是"任务饱和"（P2a 的 V4 是饱和），而是"信息缺失"——缺的正好是 S1 能读出的那个 subgoal。
3. **decodable→steerable 的桥**：P1 说 S1 轴"可解码不可静态 steer"；但 B screen 证明**学习式**低秩算子有因果作用。C1 现在证明**存在一个足够大的、因果归属明确的目标可供 steer**。三者合起来把 C2 的假设收窄成一个干净问题：*学习式条件算子能否把这 2.60 NLL 的 gap 关上*。

## 守门与边界

- 只读诊断，`trainable_parameter_count=0`、`free_bias_present=false`、不改 production wiring、不回灌 evaluation。
- 不改写任何已封存 verdict（`kill-eta` / S2 / B screen 均不动）。
- C1 只判"仪器值不值得跑 C2"，本身不产出 steering 能力结论。

## 解锁的下一包（C2）

条件化学习式 steering screen（[05 预注册骨架](05_CONDITIONAL_STEERING_PREREG_SKELETON.md)）现在有了合法仪器：
- 传感器：冻结 S1 subgoal readout（condition）。
- 执行器：rank-8 ReFT 乘性写入（B screen 血统，重init，no free bias，zero-code strict no-op）。
- 门控：显式 `{noop, apply}` 由 condition 驱动。
- matched-budget 判定门：`boundary-gated > always-on > random-gate > noop`，主判即"能否把目标剥离下 2.60 NLL 的 gap 关上，且条件性优于无条件"。
