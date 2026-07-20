# validation_delta 0.0139 < 0.02 根因分析（RCA）

调研日期：2026-07-20
输入 artifact：`artifacts/learned_active_evidence_root_510_ecore_force/`
（510-turn real-trace soak，Qwen2.5-1.5B CUDA，E-core lane，2026-07-15 产出）
关联：known-debts #86 / #88 / #89、`docs/specs/prediction-error-loop.md` CP-11 段、
`docs/specs/learned-vs-heuristic-coverage.md` §6。

## 0. 结论先行

**promotion 差距不是"训练不够久"，也不是 capacity ladder 所扫的那些容量旋钮，而是
gate 度量对象（CP-11 predictive heads）的特征贫困 + 度量基准偏保守的叠加**。具体：

1. gate 指标在 ~turn 100 后即平台化——再加 real-trace 长度不会关闭差距（排除"轨迹长度"根因）。
2. 该 510 artifact 产自 **CP-11 特征加宽（7→18 维）与 trailing-window 度量落地之前**的代码；
   两项修复已于 2026-07-15 落库但**从未在真 trace 上测过**。下一次 soak（Lane C）是决定性测量。
3. capacity ladder 的 9 个臂（pe-critic 1/2/4 维 × COCOA 8/32/128 维）validation_delta
   全部恒等 0.0138——因为这些旋钮**根本不进入 gate 度量的计算路径**。当前 ladder 无法诊断
   promotion gate；需要重设计。
4. 一个此前未被强调的信号：kill-criteria 窗口内 `prediction_target_correlation = −0.354`
   ——learned head 赢 baseline 靠的是**偏差收缩（shrinkage）而非追踪目标变化**。
   这独立佐证"瓶颈在特征"，也提示 0.02 绝对 MAE 门槛需要对照 target std 校准。

## 1. gate 指标的确切含义（读代码确认）

`validation_delta = max(window_world_improvement, window_self_improvement)`，来自
PE owner 的 CP-11 world/self predictive heads（线性 head，SHADOW，report-only）相对
手工 `_PredictionErrorHead` baseline 的 MAE 改善（`scripts/run_learned_shadow_soak.py:517-540`）。

关键事实：**这个指标只度量 PE owner 的两个线性预测 head**，不度量 temporal torch 三后端、
CMS torch band 或 Internal RL 的任何学习效果——但它是四个组件 ACTIVE flip 共用的
`validation_delta<0.02` 门槛输入。

## 2. 证据一：改善轨迹在 ~turn 100 平台化

510 artifact 的 `cp11_predictive_heads.readout_series`（累计 MAE 口径，21 个采样点）：

| turn | world_improvement | self_improvement |
|---|---|---|
| 25 | −0.0570 | 0.0091 |
| 75 | −0.0121 | 0.0128 |
| 150 | −0.0007 | 0.0132 |
| 300 | 0.0050 | 0.0135 |
| 510 | 0.0075 | 0.0139 |

self 轴从 turn 75 起就停在 0.013x；world 轴的持续爬升只是冷启动误差（turn 25 时
learned MAE 0.098 vs baseline 0.041）被累计均值慢慢稀释。**用累计口径永远追不回冷启动**。

用差分法估计 turn 400→510 的 trailing 110-turn 边际 MAE：

- world：learned ≈ 0.0296 vs baseline ≈ 0.0407 → trailing improvement ≈ **0.0111**
- self：learned ≈ 0.0247 vs baseline ≈ 0.0394 → trailing improvement ≈ **0.0146**

即使切到 trailing-window 口径（已落库），按此 artifact 的收敛水平仍是 0.015 量级，
差距 ~0.005。加长轨迹无济于事；必须提高 head 的拟合能力或校准门槛。

## 3. 证据二：负的 prediction-target 相关

`final_kill_criteria`（最后 64 样本窗口）：
`prediction_self_autocorrelation = −0.051`，`prediction_target_correlation = −0.354`，
`kill_triggered = False`。

解读：learned head 的逐样本预测与 realized target **负相关**，但 MAE 仍优于 baseline
——说明它赢在把预测收缩到目标均值附近（baseline 手工 head 偏差更大），而不是捕捉
target 的逐 turn 变化。7 维聚合特征（该 run 的实际配置）不携带足以解释 turn-to-turn
变化的信息。这正是 CP-11 特征加宽（增加 substrate spread/peak、delta max-abs、
4 轴滞后 realized outcome、4 轴滞后 signed error 共 18 维）针对的病灶；AR 滞后特征
尤其直接针对"负相关"症状（给 head 提供上一轮误差方向）。

## 4. 证据三：capacity ladder 与 gate 指标完全解耦

9 个 ladder 臂（`nz16-pe{1,2,4}-cocoa{8,32,128}-runtime+ssl+internal-rl+cms-torch-t500`，
synthetic substrate，seed0）的 `validation_delta` **全部恒等 0.0138**，
`cp11_predictive_heads` 读数逐字节相同。原因：

- pe-critic 维度（pe1/2/4）影响 `_PELearnedCritic`（epistemic 分解），不影响 CP-11 heads；
- COCOA 维度（cocoa8/32/128）影响 credit 的 `_RewardingStateHead`，不影响 CP-11 heads；
- `n_z` 固定 16 未扫描；synthetic lane `real_trace_turns=0` 本身就被 gate BLOCK。

**结论：现有 ladder 扫的是与 gate 无关的旋钮。** 要么 ladder 加入 gate 相关维度
（CP-11 特征宽度 / learning rate / window size / target 定义），要么 promotion gate
改为对各组件用各自的学习证据（如 SSL prediction-loss 改善、CMS update-outcome parity、
Internal RL no-optimize 差分——这些在同一 artifact 里其实都是健康的：
`torch_prediction_loss 0.169 vs pure 0.391`、`internal_rl full_return_improvement 1.55 vs
no-optimize 0.0`、CMS `old_knowledge_retention 0.9997`），而不是让四个组件共用一个
只反映 PE heads 的标量。

## 5. 证据四：门槛校准缺口

0.02 是**绝对 MAE 改善**门槛。self 轴 baseline MAE 仅 0.039——learned head 已经拿到了
36% 的相对改善（0.0139/0.039）。若 target 本身方差小（该 artifact 缺 `window_target_stds`
与 `window_persistence_maes`，两者已在加宽版 readout 中补充），绝对门槛可能结构性偏紧。
下一次 soak 必须核对：若 lag-1 persistence baseline 逼近 learned head，说明 target 由慢漂移
主导，0.02 绝对门槛应改为相对 persistence 的边际改善口径（预注册新阈值 = 新观察窗口，
遵守 `credit-and-self-modification.md` 阈值预注册纪律，不得在窗口内调阈值）。

## 6. 行动建议（按证据强度排序）

1. **［决定性测量，Lane C］** 用已落库的 18 维特征 + trailing-window 口径重跑 ≥500 turn
   real-trace soak（Linux/GPU lane）。预期：AR 滞后特征应把 `prediction_target_correlation`
   拉正；若 trailing self ≥ 0.02 则差距直接关闭。
2. **［ladder 重设计，本地可做］** capacity ladder 增加 gate 相关轴：CP-11 head learning
   rate、window size、（若扫 n_z 则同时记录其对 heads 无影响这一事实作为对照）；
   保留 pe/cocoa 维度扫描但明确其归属（#86 capacity→gain 曲线，不是 promotion gate 证据）。
3. **［gate 结构性讨论，spec 级］** 评估把 `validation_delta<0.02` 从"四组件共用 PE-heads
   标量"改为 per-component 学习证据门（SSL loss 改善 / CMS parity+抗遗忘 / RL no-optimize
   差分），与 `learned-vs-heuristic-coverage.md` §8 的升级路线对齐。任何更改先预注册。
4. **［校准检查，随 1 一起］** 下一 soak 读 `window_target_stds` / `window_persistence_maes`，
   决定绝对 vs 相对门槛。

## 7. 诚实边界

- 本 RCA 全部基于既有 artifact 的只读分析与当前代码核对，未跑新实验；
  trailing-window 估计（§2）是累计 MAE 的差分近似，非直接测量。
- 结论 1（特征贫困）有两条独立证据（平台化 + 负相关）但仍待 18 维 real-trace 复测证实；
  在复测前不得宣称"差距已定位并解决"。
- promotion 仍 BLOCKED 的另两个 missing gate（`pe_off_control` / `eta_off_control`）
  属于 #87 同基底消融范畴，不在本 RCA 范围内。
