# 08 · 读→扳闭环（P2c · S3 前置，已执行）= **PASS**

- owner 模块：[`eta_read_steer_prereq.py`](../../packages/vz-runtime/src/volvence_zero/agent/eta_read_steer_prereq.py)
- run 脚本：[`scripts/run_eta_read_steer_prereq.py`](../../scripts/run_eta_read_steer_prereq.py)
- 产物：`artifacts/eta_s3prereq_readloop_20260805/`（`report.json` / `report.md` / `artifact_manifest.json`）
- 测试：[`test_eta_read_steer_prereq.py`](../../packages/vz-runtime/tests/test_eta_read_steer_prereq.py)（9 passed）
- 基底：frozen merged S1 模型 `artifacts/eta_stage2_merged_v2_20260803`，layer 20 / 896，MPS
- 配置：rank 8，seeds (0,1,2,3,4)，updates 80，lr 0.01，batch 32，reader ridge λ=10，bootstrap 5000 / 95%CI，307 train / 165 heldout junction rows，K=8 subgoal

## 一句话结论

**读→扳闭环成立：条件不再靠 oracle。** C2 只证明了"给对 subgoal 就扳得动"，条件是 oracle，不能算可部署闭环。本包用一个**重新拟合的冻结线性 sensor**，从**携带目标的上下文残差**在线读出 subgoal（heldout top-1 = **1.000**），再交给 C2 执行器扳**目标剥离**路口的动作——结果 `conditional-online` 的 expert NLL = **0.023，与 oracle 完全相同**，比等预算无条件算子（1.391）优 **1.37 NLL**，route-level bootstrap 95%CI 下界（最差 seed）仍分别为 **2.40 / 1.26**，全部 > 0。**方向盘（执行器）+ 读盘（sensor）都验证过了，S3 Internal RL 可以准入。**

## 两个 cheap 审计钉死了设计（都是只读）

| 审计 | 数值（chance=0.125） | 含义 |
|---|---:|---|
| 现成 S1 v2 probe 在 C2 面（揭示）top-1 | 0.145 | **S1 不迁移**：probe 在 V4 staged-plan 文本面拟合，不能原样当 C2 面 sensor |
| 现成 S1 v2 probe 在 C2 面（剥离）top-1 | 0.145 | 同上 |
| refit 线性 reader 在**揭示**上下文残差 top-1 | **1.000** | subgoal 在携带目标的残差里**线性完美可读** |
| refit 线性 reader 在**剥离**残差 top-1 | 0.164 | 目标剥离后谁也读不出——条件**必须**来自上下文/记忆（如设计） |

结论：不是"信息不在"，是"要么读错面、要么读被剥离的输入"。把 sensor **在正确的面上重拟合**、从**上下文**读，就闭环。

## 主判（heldout，5 seed 平均；括号内取最差 seed 的 bootstrap CI 下界）

| arm（matched budget） | expert NLL |
|---|---:|
| noop（目标隐藏基线） | 2.813 |
| subgoal-revealed 文本天花板 | 0.218 |
| conditional-oracle（C2 口径） | 0.023 |
| **conditional-online（读→扳，非 oracle）** | **0.023** |
| unconditional（等预算恒定算子） | 1.391 |
| random-condition（错条件） | 7.620 |

| 判定门 | 值 | 阈值 | 结果 |
|---|---:|---:|:--:|
| reader heldout top-1 | 1.000（5 seed 全 1.0） | ≥0.80 | ✅ |
| online gap closed = noop − online | 2.790 | ≥0.30 | ✅ |
| online conditional advantage = uncond − online | 1.368 | ≥0.15 | ✅ |
| online − noop bootstrap CI 下界(min) | 2.398（>0） | >0 | ✅ |
| online − uncond bootstrap CI 下界(min) | 1.259（>0） | >0 | ✅ |
| structural（no free bias / zero-code no-op=0.0 / substrate frozen / params changed） | — | — | ✅ |

**S3 前置 admission = PASS（5 门 + 结构，5 seed 全过）。**

## 三个数字的第一性含义

1. **online 0.023 == oracle 0.023**：因为 reader heldout=1.0，逐行预测的条件与真值**逐行相同**，执行器拿到的条件与 C2 的 oracle 无差别。这把 C2 里"用 oracle 是否作弊"的疑问彻底消掉——**条件是被读出来的，不是被告知的**。
2. **online 0.023 vs unconditional 1.391（advantage 1.37）**：同 rank、同预算、同 updates，唯一差别是"是否按读出的 subgoal 条件化"。条件化赢 1.37 NLL ⇒ **"往哪出手"有独立因果价值**，且此价值在**在线读出**下仍然存在。
3. **route-level bootstrap 下界全 > 0**：不是靠某几条 heldout 路线蒙的；按 case 聚合再重采样，效应对路线分布稳健。

## 这层闭环对应的系统语义（R3/R4/R-PE）

- **识别（慢变/可冻）= sensor**：冻结线性 reader，从上下文残差读"当前 subgoal 在哪个方向"。已证 heldout=1.0。
- **干预接口（有界执行器）= executor**：冻结基底上的 rank-8 乘性写入，no free bias、zero-code strict no-op=0.0，可回滚。已证扳得动且优于恒定出手。
- **策略学习（Internal RL）= S3 本体，尚未做**：π(a_t | readout, PE, …)，动作 {noop, steer(+d,s), steer(−d,s)}，信用来自结局/PE，在线改的是有界控制器（增益/门控/**何时**出手），不是整模。**本包只把"扳得动 + 读得到"钉死，S3 才是"学何时扳"。**

## 边界（严格保持）

- sensor 是 refit 冻结线性读出（无 LM 训练、steer 时冻结）；executor `substrate_trainable=0`、`free_bias=false`、`zero_code_strict_noop=0.0`。
- condition 读自**上下文携带目标**的残差（= agent 从记忆/上下文知道自己的 subgoal），应用于**目标剥离**路口动作前向——这是可部署读→扳环的最小忠实建模，不是把答案塞进 prompt。
- PASS 只表示**可准入 S3 Internal RL 的正式预注册**；`production_promotion_authorized=false`、不安装控制器、不改 WiringLevel、不回灌 evaluation。
- 不改写任何封存 verdict：`kill-eta`（Stage-3）、S2-static FAIL、B screen FAIL 均不变——本包证明的是"识别 + 有界干预 + 待学策略"这条线可闭环，与 ETA 的 z_t 涌现时间抽象是不同机制。

## 下一步（S3 本体，需另起预注册）

用 PE/结局信用在线学"**何时/多大力**扳"，动作空间 {noop, steer(±d,s)}，先在本仪器上验证：(a) 稀疏结局信用能否收敛，(b) 学到的门控是否复现"该出手时才出手"。此前的三层（读得到 / 扳得动 / 条件有价值）已全部就绪。
