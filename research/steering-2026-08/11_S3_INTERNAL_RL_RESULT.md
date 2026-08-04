# 11 · S3 学「何时扳」Internal RL 权威结果 = **实质学习性已证；预注册 worst-seed 稳健门未过**

- owner 模块：[`eta_when_to_steer_rl.py`](../../packages/vz-runtime/src/volvence_zero/agent/eta_when_to_steer_rl.py)（单 owner，自写 minibatch REINFORCE + advantage 归一化 + 熵正则）
- run 脚本：[`scripts/run_eta_when_to_steer_rl.py`](../../scripts/run_eta_when_to_steer_rl.py)；单元测试 [`test_eta_when_to_steer_rl.py`](../../packages/vz-runtime/tests/test_eta_when_to_steer_rl.py)（8 passed，含端到端 REINFORCE 可学性）
- 冻结 prereg：`artifacts/eta_s3_internal_rl_prereg_20260805.json`（SHA `62454418…`）
- 产物：`artifacts/eta_s3_when_to_steer_rl_20260805/`（report.json/md + manifest；git `b77812a5`）
- 基底：frozen merged S1 模型，layer 20/896，MPS；seeds `[0,1,2,3,4]`，1200 episodes，bootstrap 5000/95%

## 一句话结论

**「从稀疏结局信用学会何时扳」在代理上被证明是可学的——但不是在所有 seed 上稳健收敛。** 冻结 sensor（08 reader）+ 冻结 executor（C2 rank-8），唯一在线更新的门控策略只观测 PE 代理（reader margin、belief/fresh 一致性、base 熵）、只拿**每-episode 终局稀疏信用** `R=-mean(route NLL)`、从不给每步对错标签。**5 个 seed 中 4 个稳健学出 selective gate**（pe_gated 0.61–0.92 ≪ always-on 1.79，selectivity 0.35–0.56，全部 route-level bootstrap CI 强正，甚至优于 oracle 1.09）；**1 个 seed（seed 1）落入 always-steer 探索塌缩**（pe_gated 1.79、selectivity 0）。预注册要求 **worst-seed** CI>0，故 literal admission = **FAIL**；但实质学习性证据为强正。

## 聚合（5 seed 平均，heldout expert NLL）

| arm | NLL | 说明 |
|---|---:|---|
| noop | 2.813 | 目标隐藏基线 |
| always-on-belief | 1.791 | 过期条件恒定出手 |
| random-gate | 2.026 | 同频率随机出手 |
| **pe-gated-online（学到的门控）** | **0.951** | 胜所有基线，且 < oracle |
| oracle-gate（上界诊断） | 1.090 | belief 正确才出手 |
| pe-hard-gate（硬规则上界诊断） | 1.090 | belief==fresh 才出手 |
| fresh-ceiling | 0.027 | 每步 fresh 正确条件 |

- 收敛改善（初始→最终，seed 平均）**0.787** ≥ 0.20 ✅
- 门控选择性 steer(非切换)−steer(切换)（seed 平均）**0.382** ≥ 0.30 ✅
- gain CI 下界（**worst-seed**）：vs noop **−0.133**、vs always-on **0.000**、vs random **0.000** ❌（被 seed 1 拖累）

## 逐 seed（暴露 seed 1 的探索塌缩）

| seed | pe_gated | selectivity | steer(post) | steer(non) | CI vs noop | CI vs always-on |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.680 | 0.535 | 0.46 | 1.00 | +1.678 | +0.605 |
| **1** | **1.791** | **0.000** | **1.00** | **1.00** | **−0.133** | **0.000** |
| 2 | 0.754 | 0.563 | 0.39 | 0.96 | +1.621 | +0.512 |
| 3 | 0.922 | 0.354 | 0.38 | 0.73 | +1.512 | +0.281 |
| 4 | 0.607 | 0.458 | 0.52 | 0.98 | +1.739 | +0.665 |

4/5 seed 学出「fresh 行几乎必扳、post-switch 行大幅收手」的 selective 策略，CI 强正；seed 1 早期被 fresh 行奖励锁进 always-steer 吸引盆（steer_post=steer_non=1.0），后续稀疏信用+探索未能逃逸。

## 诚实的机制解读

- **正面**：这是「稀疏、快速、主动学习」claim 的直接支撑——门控策略仅凭**每-episode 终局标量**（无每步标签）就在有界预算内学到逼近/超过 oracle 的择时，且施加集中在「该出手」处（selectivity 正）。pe_gated 0.95 < oracle 1.09 说明学到的门按**真实 NLL 结局**优化，比「belief 正确」粗规则更细。
- **局限**：on-policy REINFORCE + 全局（route）终局奖励存在**探索塌缩**风险——fresh 行的巨大即时收益会快速拉满共享 steer-bias，个别 seed 在 selective 判别学成前锁进 always-steer。加 advantage 归一化 + 熵正则把成功率从 0/5（早期）提到 4/5，但未达 5/5。这是算法级稳健性问题，非仪器/信号问题（同一套特征下 4 seed 成功证明信号充分；S3-A 已证 staleness 完全可检测）。

## 为什么这是诚实的 FAIL（不 p-hack）

预注册 worst-seed CI 门是**严格稳健性**标准。看到 seed 1 塌缩后，我**未**调阈值/聚合口径/动作空间/seeds/预算去翻盘；尝试过的探索增强（熵 0.2、保守 noop 初始化）均在**失败 seed 上诊断性验证**后如实保留结论。literal verdict 按预注册记为 FAIL。

## 建议（程序级决策，交用户裁定；本文件不擅自改写 verdict）

与本项目 Gate-2「literal FAIL + 实质证据正」的处置模式一致，可选：

- **A（推荐）** 采纳实质结论「稀疏信用可学何时扳（4/5 稳健）」进入下一层，并把 seed 1 的探索塌缩记为**已知稳健性风险**；若要满足 worst-seed 门，用**多随机重启取最优**或**熵退火**（标准 REINFORCE 稳健化，不改判据）另立小收敛包重跑一次。
- **B** 严格按 literal FAIL 封存，声明「学习性已展示但预注册稳健门未过」，不进入下一层。

无论 A/B：`substrate_trainable=0`、reader/executor 冻结（未变）、no free bias、zero-code strict no-op、production 未提升、evaluation 未回灌；不改写任何封存 verdict（kill-eta / S2 / B screen / C2 / 08 / S3-A）。

## 三层是否闭环（回答用户先前问题）

| 层 | 状态 |
|---|---|
| 识别 sensor（08 冻结 reader） | ✅ heldout 1.0 |
| 干预 executor（C2 rank-8） | ✅ 关掉 2.79、条件性优 1.37 |
| 策略「何时扳」（S3 本文件） | ⚠️ **可学已证（4/5），worst-seed 稳健门未过（1/5 塌缩）** |

方向盘与读盘钉死；「学开车」在多数 seed 上成立，稳健化是唯一剩余缺口。
