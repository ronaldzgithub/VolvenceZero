# 05 · 条件化学习式 steering screen 预注册骨架（P2b）

状态：**骨架/设计冻结**（尚未固定 source SHA 与 model run；因 P2a 判定当前仪器无余量，本 screen 必须先落"仪器重设计"实现，属独立收敛包）。

本 screen 是继 B screen `kill-eta` FAIL 之后的**新 claim**，不重解释已封存 verdict。它把 `research/steering-2026-08` 文献落到一个可判死的实验：**在有余量的冲突映射仪器上，学习式条件干预能否证明"识别 + 有界执行 + 学何时出手"的因果价值。**

## claim_scope

`conditional-learned-steering-directional-screen`（screen 只决定是否准入另立权威 sweep；不改 production WiringLevel，不回灌 evaluation）。

## source_lineage（绑定既有封存证据）

- Stage-3 `kill-eta`：`artifacts/eta_stage3_rate_distortion_20260803/report.json`
- B screen 早停：`artifacts/eta_faithful_rewrite_screen_20260804/EARLY_STOP_SEAL.md`（ETA FAIL 锁死 + 学习式因果正面资产）
- P1 预检：`research/steering-2026-08/steerability-precheck-result.json`（decodable≠steerable）
- P2a 余量审计：`research/steering-2026-08/04_HEADROOM_AUDIT.md`（当前仪器无条件余量）

## 仪器重设计（前置，必须先实现）

当前 corpus 前缀几乎唯一决定动作 → 恒定算子即最优。重设计目标：让恒定算子**必然**在冲突点出错。

1. **冲突映射（核心）**：构造环境使**同一累积前缀**在不同 active_subgoal 下**最优下一动作不同**（例如"去当前子目标 vs 让路给走廊"在同一路口因子目标不同而分叉）。
2. **基底非天花板主判点**：主判限定在基底 `noop_nll` **未近确定**（如中位数区间、剔除 noop_nll<1e-2 的饱和前缀）的冲突路口。
3. **仪器有效性门（instrument-validity gate，先跑）**：一个恒定最优算子在冲突主判点的错误率必须 > 阈值（证明"无条件"确实不足）；否则 screen 判 `instrument-invalid`，回到重设计，不产出机制结论。

## configuration（拟冻结）

- 传感器（识别，冻结）：S1 v2 readout artifact `086a8f3d…` 的 `axis_for` / class logits 作 **condition**（CAST 式条件向量），只读、不回灌、不随 screen 更新。
- 执行器（有界写入）：B screen 同款 rank-8 乘性写入 `A·diag(tanh(C·z_t))·Bᵀ·e_t`，**重新初始化**、`free_bias=false`、`zero_code_strict_noop=true`；layer 20 / 896。
- 门控（学何时出手）：**显式小动作空间** `{noop, apply(+d), apply(−d), scale-levels}`，由 condition 驱动的门决定，**不经 rate/KL 涌现**（B screen 已三证该路径塌缩为 never-switch）。
- 判定臂（matched budget，同施力预算下比较）：
  - `boundary-gated`（仅在 condition 指示子目标边界/冲突时施加）
  - `always-on`（每步施加）
  - `random-gate`（同频率随机施加）
  - `noop`
- corpus：重设计冲突映射版；train/heldout 路由与 v4 生成期不相交，带 SHA provenance。

## thresholds / decision_rules（拟定）

1. **instrument-validity**（先决）：恒定最优算子在冲突主判点错误率 ≥ `min_constant_operator_error`（如 0.20），否则 `instrument-invalid`。
2. **因果主判**：heldout 上 `boundary-gated` 相对 `noop` 的动作正确率/负 NLL 增益 > 0 且 bootstrap 95% CI **不跨 0**。
3. **优于无条件**：`boundary-gated` > `always-on`（同预算下条件性带来净收益，CI 不跨 0）——这是"学何时出手"有价值的关键门。
4. **优于随机门**：`boundary-gated` > `random-gate`（CI 不跨 0），排除"施加频率"混淆。
5. **结构完整性**：no free bias、zero-code strict no-op、执行器参数确有更新、condition 只读。

全过 → 准入独立权威 sweep（再决定是否解锁 S3）；任一未过 → 封存 FAIL，不改写 `kill-eta`。

## prohibited_after_execution（拟定）

- 读结果后改阈值 / 主判臂 / 动作空间 / seeds / updates；
- 加 additive/常数 bias 或让 zero-code 非 no-op；
- 用 active_subgoal 真值进训练损失（只可作 readout 与 oracle 评价）；
- 训练 substrate 或把 evaluation 准入回灌学习；
- 重贴已封存 `kill-eta`/S2/ B screen verdict；
- 安装 screen 控制器或改 production WiringLevel。

## frozen_source_files

待实现"仪器重设计 + 条件门"后于正式 prereg JSON 固定各 source SHA（含新 corpus 生成器、screen 模块、run 脚本）。

## 退出 / 回滚

全为 evidence lane。screen 不安装、不写回、默认 wiring 不变；回滚即删除 screen artifact 与新代码路径。仪器重设计代码以独立收敛包提交（单 owner：corpus/instrument；单契约：冲突映射语义），与执行器/门控分包，避免一次改动跨越多个 owner。
