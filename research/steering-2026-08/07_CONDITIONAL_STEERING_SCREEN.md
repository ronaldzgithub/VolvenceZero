# 07 · 条件化学习式 steering screen（P2c · C2，已执行）= **PASS**

- owner 模块：[`eta_conditional_steering_screen.py`](../../packages/vz-runtime/src/volvence_zero/agent/eta_conditional_steering_screen.py)
- run 脚本：[`scripts/run_eta_conditional_steering_screen.py`](../../scripts/run_eta_conditional_steering_screen.py)
- 产物：`artifacts/eta_conditional_steering_screen_20260804/`（`report.json` / `report.md` / `artifact_manifest.json`）
- 测试：[`test_eta_conditional_steering_screen.py`](../../packages/vz-runtime/tests/test_eta_conditional_steering_screen.py)（5）+ C1 5 = 10 passed
- 基底：frozen merged S1 模型 `artifacts/eta_stage2_merged_v2_20260803`，layer 20 / 896，MPS
- 配置：rank 8，seeds (0,1,2)，updates 80，lr 0.01，batch 32，307 train / 165 heldout junction rows，K=8 subgoal

## 一句话结论

**条件化学习式 steering 成立，且条件性本身有因果价值。** 在 C1 验证过的目标剥离路口仪器上，一个 rank-8、无 free bias、zero-code strict no-op 的**学习式乘性写入**，只要**按 active_subgoal 条件化**，就能把基底 2.81 的 expert NLL 关到 **0.02**（比"直接把目标写进 prompt"的 0.218 天花板还低）；而**等预算的无条件恒定算子只能到 1.36**。这回答了整条主线最初的问题：P1 说"可读不可静态 steer"、B screen 说"学习式有因果作用但 ETA 时间切换冗余"、P2a 说"V4 仪器无条件余量"——**出路正是"读残差(S1) + 有界学习式执行器(ReFT) + 按 subgoal 条件出手(CAST)"，三者合起来能 steer，而且"学何时/往哪出手"确实优于恒定出手。**

## 主判（heldout，3 seed 平均；括号内 min）

| arm（matched budget，同 norm cap 24.06） | expert NLL |
|---|---:|
| noop（目标隐藏基线） | **2.813** |
| subgoal-revealed 文本天花板 | 0.218 |
| **conditional（本方案）** | **0.027** |
| unconditional（等预算恒定算子） | 1.357 |
| random-condition（错条件） | 7.377 |

| 判定门 | 值（min） | 阈值 | 结果 |
|---|---:|---:|:--:|
| gap closed = noop − cond | 2.786（2.776） | ≥0.30 | ✅ |
| conditional advantage = uncond − cond | 1.330（1.329） | ≥0.15 | ✅ |
| condition specificity = rand − cond | 7.350 | ≥0.15 | ✅ |
| gap closed fraction | 1.074 | ≥0.30 | ✅ |
| structural（no free bias / zero-code no-op=0.0 / substrate frozen / params changed） | — | — | ✅ |

**screen admission = PASS（4 门 + 结构，3 seed 全过，余量巨大）。**

## 三个数字的含义（第一性）

1. **conditional 0.027 vs noop 2.813**：学习式条件写入几乎完全恢复了"知道 subgoal"才有的信息——甚至优于文本天花板 0.218，因为残差层的直接写入比一句 `Objective: X` 的自然语言提示更"干净"。这证明 P1 的"decodable≠**static**-steerable"不是"不可 steer"：换成**学习式**执行器就能 steer。
2. **conditional 0.027 vs unconditional 1.357（advantage 1.33）**：同 rank、同预算、同 updates，唯一差别是"是否按 subgoal 条件化"。条件化赢 1.33 NLL ⇒ **"学何时/往哪出手"有独立因果价值**，不是靠更大写入蒙的。这正是 B screen 的 permuted-z=0 / never-switch 想测却因 V4 无余量测不到的东西。
3. **random-condition 7.377（比 noop 还差 4.6）**：喂错 subgoal 会主动把动作推向错误分叉 ⇒ 算子学到的是**方向性、条件特异**的写入，不是"随便写点什么都好"。这排除了 confound，也说明条件读出（S1 sensor）的正确性是关键。

## 与既往结论的关系（不改写任何封存 verdict）

- `kill-eta`（Stage-3）不变：本 screen 不复活 ETA 的 rate-distortion 时间切换 claim。C2 证明的是**条件化学习式 steering**，与 ETA z_t 的"涌现时间抽象"是不同机制——subgoal 是基底**已有**的线性可读结构（见 [02 §](02_VZ_IMPLICATIONS.md) / C1），我们做的是**识别 + 有界干预 + 学策略**，不是压出新抽象。
- S2 static FAIL 不变：C2 恰好解释了 S2 为何 FAIL——静态 probe 轴不可 steer（P1），而学习式条件算子可以（本 screen）。
- B screen FAIL 不变：B screen 的正面资产（学习式低秩写入有因果作用）在 C2 被放大为完整的"条件 > 无条件"证据。

## 边界与下一步

- screen 只判"是否准入独立权威 sweep / 是否解锁 S3 Internal RL"，**不安装控制器、不改 production WiringLevel、不回灌 evaluation**；`substrate_trainable=0`、`free_bias=false`、`zero_code_strict_noop=0.0`。
- 本 screen 的 condition 用的是 **oracle active_subgoal**（此处解耦地只测执行器+条件门）。**已在 [08_READ_STEER_S3_PREREQ.md](08_READ_STEER_S3_PREREQ.md) 完成权威化**：condition 换成**在线非 oracle sensor**（refit 冻结线性 reader，从上下文残差读到 heldout 1.0），`conditional-online` NLL 0.023 = 完全等于 oracle，比 unconditional 优 1.37，route-level bootstrap CI 下界全 >0，5 seed 全过 = **PASS**。
- S3 Internal RL 的意义此时才成立：C2 证明了"扳得动且条件性有价值"，S3 才是"用 PE/结局信用在线学**何时**扳"——方向盘已验证，接下来学开车。
