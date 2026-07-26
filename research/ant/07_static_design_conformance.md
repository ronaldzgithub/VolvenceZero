# 数字蚂蚁 Ecology 静态设计一致性审计

> **审计日期**：2026-07-26
> **审计对象**：`packages/vz-embodiment-ant`（约 24.8k 行）＋它消费的通用内核契约（`vz-temporal` / `vz-runtime` / `vz-cognition` / `vz-memory` / `vz-contracts`）＋ `scripts/run_ant_*.py`、`scripts/audit_ant_ecology_mechanisms.py`、`scripts/measure_ant_*.py` 驱动层。
> **设计基准（SSOT）**：
> - [`docs/specs/digital-ant-embodiment.md`](../../docs/specs/digital-ant-embodiment.md) —— 冻结契约（边界、两个冻结向量函数、Outcome→PE 接缝、ecology-v2 三物体语义、temporal/action-head profile、冻结 gates、kill conditions）
> - [`05_ecology_p0_p1_p2_plan.md`](05_ecology_p0_p1_p2_plan.md) —— P0→P1→P2 串行执行与验收计划
> - [`04_digital_ant_feasibility.md`](04_digital_ant_feasibility.md) —— VZ 模块 ↔ 蚂蚁神经子系统映射、架构草图、差距清单
> **参照物（非基准）**：[`06_ecology_implementation_status.md`](06_ecology_implementation_status.md) —— 一律视为"待核对的声明"，不作为事实来源；文档与代码不一致处单列一节。
> **方法**：**纯静态**。未运行任何训练、脚本或 pytest；结论全部来自读码 + 已提交 artifact 的内容核对 + 代码常数的算术推演。凡只能由运行时判定的条目一律标 `undecidable-static`，不猜。

---

## 0. 结论摘要

### 0.1 一句话

**架构层面能承载这个设计；但按当前代码，设计要求的那条"可审计因果证据链"产不出来。** 三处一等断点：奖励接缝在 optimizer 侧漏成了密集食物塑形、P0 的 `PASS` 是空的（19 个 snapshot 是同一个冷 checkpoint）、P2 的 23 门晋级判决与 demo loader 完全脱钩；再加一条可行性硬伤：被能力门打分的确定性策略在运动学上达不到 medium/far 所需的累计转向。

### 0.2 三层判断

| 层 | 问题 | 判断 | 依据 |
|---|---|---|---|
| **架构层** | 冻结基底＋有界控制器能不能承载这个设计 | **能** | 两个冻结向量函数是真纯函数、19 维镜像置换逐通道正确、spec 要求的 temporal 契约（rank=16 / effective_dims / contrast_pairs / exclusive steering / 镜像等变 / 16 步 segment / posterior_sample_scale）在四条 lane 都真落地、SSOT 边界在 `src/` 内干净、双槽 journal / resume / 容量淘汰 / P2 两层 bootstrap＋Holm 都是真代码 |
| **证据链层** | 能不能产出设计要求的可审计 PASS/BLOCK | **不能** | P0 的 PASS 由回滚制造且 P0-B 基本未实现；P1 的 60% 门可被 1 蚂蚁 1 布局的 smoke 退化通过并解锁 P2；P2 的晋级门无人消费，loader 用的是另一个可在任意小预算下 PASS 的报告 |
| **可行性层** | 门槛本身达不达得到 | **有 2 个真结构问题** | ① 训练开探索、评估关探索且 exclusive steering 把 base contrast 归零 → 被打分的确定性策略转向权威 ≤0.033 rad/拍，40 拍最多 ~1.3 rad，medium/far 定向往返需 ~3.4–4.0 rad；② R5-R6（连续记忆谱）在本测试床没有任何可测端点 |

### 0.3 审计规模

| 项 | 数量 |
|---|---|
| 设计条目逐条核对 | **146** 条 |
| ├ 已实现（`implemented`） | 81 |
| ├ 部分实现（`partial`） | 43 |
| ├ 缺失（`missing`） | 13 |
| └ 与设计相矛盾（`contradicted`） | 9 |
| 阻塞级判断 | 32 条 → 经独立对抗性复核 **24 CONFIRMED / 8 REFUTED** |
| 复核者补充的漏项 | 35 条 |
| 簇级结论 | `conformant-with-gaps` ×4（A/C/F/G）、`non-conformant` ×4（B/D/E/H） |

### 0.4 一等阻塞速查（按建议修复顺序）

| # | 阻塞 | 类型 | 复核 | 破坏的设计条款 |
|---|---|---|---|---|
| **BLK-01** | `measurement=None` 的拍上，optimizer reward 退化成 PE owner 内部合成的 action 轴（食物传感器的函数）；`dense-local-shaping-off` 消融被反转 | 一致性 | CONFIRMED（+ 我自查） | spec:70、spec:78-84 |
| **BLK-02** | `PE-off` 实际是 `reward-off`：环境 payoff 与 segment bonus 被同时归零 | 完整性 | CONFIRMED（+ 我自查） | spec:377-380 kill condition |
| **BLK-03** | P0 action-chain 的 `PASS` 由逐 episode 回滚制造：19 个 snapshot 是同一个冷 checkpoint | 完整性 | CONFIRMED | plan:119-124 |
| **BLK-04** | P0-C 收 8 个 owner 指纹只 gate 2 个；artifact 自证 24 条 `learning_enabled=False` 下的逐拍变化仍 `passed: true` | 完整性 | CONFIRMED | plan:185、plan:193 |
| **BLK-05** | exclusive steering 后冷启 head 精确为零，P0 仍对 shared-initial 施加 `turn_delta>=1e-4` → 今天跑 P0 必然 BLOCK；`25%` retention 同时变空门 | 一致性 | CONFIRMED | plan:121、spec:326-332 |
| **BLK-06** | P0-B 基本未实现：无 transition protocol、无 negative control 与 switch-rate 上限、无 switch↔状态定位、无 timeout 占比、无 segment-credit on/off parity | 一致性 | CONFIRMED | plan:137-165 |
| **BLK-07** | `layouts_per_tier` 未冻结 → 1 ant × 1 layout 使 60% 门退化为 1/1，可发 `verdict=PASS`；`load_p1_prerequisite` 不看 config，该 PASS 直接解锁 8-ant 正式矩阵 | 一致性 | CONFIRMED ×2 | plan:284-286、plan:429 |
| **BLK-08** | Stage 3 heat-route 从未被评：`heat_route_foraging` 与 `composite` 映射到字节相同的 `(COMPOSITE, FAR)`，`HEAT_ROUTE_AVOIDANCE` 在 P1/P2 层是死枚举 | 一致性 | CONFIRMED | plan:265-268、spec:344-349 |
| **BLK-09** | P2 的 23 门晋级判决无人消费；demo loader 只看 15 门 curriculum 报告，而后者可在 1 ant / 1 round / 1 seed 下 PASS 并被真实加载 | 一致性 | CONFIRMED | plan:419、plan:431 |
| **BLK-10** | ETA-off 是混淆的超集消融（同分支连带关掉 reflection/memory/regime consolidation），同时又不完全（SSL/RL 仍在优化 `_world_policy`） | 一致性 | CONFIRMED（+补充） | spec:377-380、plan:370 |
| **BLK-11** | "任何代码/门槛变化使整批失效"未实现：shard 不记 commit/源码 hash，预注册 digest 不覆盖 `outcome_score` 权重与 gate 逻辑；shard 与 P1 报告都无 manifest | 完整性 | CONFIRMED ×2 | plan:381-386、plan:429、plan:471 |
| **BLK-12** | P1/P2 报告无 provenance、无 manifest、无 device 字段；默认路径固定 → 重跑原地覆盖既有 BLOCK artifact；resume 的 compatibility 丢了 sense schema 与 input dim | 一致性/完整性 | CONFIRMED ×3 | plan:30-31、plan:47-52、spec:65、spec:459 |
| **BLK-13** | `joint_learning_enabled=False` 未关闭 `apply_reflection_prior_update` 的结构写回；当前只因 AntSession 从不调 `begin_new_context` 而侥幸安全 | 一致性 | CONFIRMED | spec:118-124 |
| **FEA-01** | 训练开探索（8 拍常曲率 arc、std floor 0.4、单拍可达 ~0.5 rad）／评估关探索且 base contrast 归零（实测 0.0055–0.0327 rad/拍）→ 策略在一个运动学 regime 被优化、在另一个被打分 | 可行性 | CONFIRMED ×2 | P1 全部 6 条能力门 |
| **FEA-02** | far **训练**局 24 拍 × 实测 ~0.225/拍 = 5.4 单位，对 far 往返 4.2（均值）／5.6（最坏）→ 训练分布里几乎采不到 far delivery（注：plant 上限本身够，见 §2.3） | 可行性 | 部分 REFUTED（强版本被推翻） | spec:320-322 |
| **FEA-03** | near 假阳性比 spec 承认的更糟：body 生成在巢心，37.5% 的 near layout 首拍即 pickup+delivery；`forced_approach` 只修出发段，回巢段仍免费 | 可行性 | 补充发现 | spec:340-343 |
| **FEA-04** | 四路转向驱动共享一根反对称执行器轴，而 `food_steering_alignment` 要求食物驱动在探针状态上**绝对**压过其余全部驱动；无时间复用机制 | 可行性 | 补充发现（"代数互斥"版本被 REFUTED） | spec:354-360 |
| **FEA-05** | R5-R6（连续记忆谱）无任何可测端点：无 memory 臂、无 memory 端点，steering 通路契约上零历史 | 可行性 | 部分 CONFIRMED | spec:9-16 |

### 0.5 判定口径

- **`implemented`**：代码实现了设计条款，且（多数情况下）有测试锚点。
- **`partial`**：主路径对，但存在未覆盖的旁路、未钉住的常数、或语义上少一半。
- **`missing`**：设计明确要求、代码里找不到任何对应物。
- **`contradicted`**：代码存在且与设计条款方向相反，或使该条款在当前配置下不可能成立/变成空门。
- **`undecidable-static`**：只有运行时测量能判定。
- 文档声明"已落地/已验证/PASS"一律不计分；只有代码 + 测试 + 已提交 artifact 三者能对上才算。
- 反向也查：代码是否**超出**冻结 spec（冻结后被放宽的门、被翻转的默认值、能让 BLOCK artifact 通过的 fallback、渗进通用 owner 的 ant 语义）。

---

## 1. 一等阻塞详述

### BLK-01 奖励接缝在 optimizer 侧没守住

**设计要求**：`semantic_*_pull` 只是 substrate 发布的感觉/动机预测通道，**不能冒充环境任务结果**（spec:70）；明确禁止任何"离食物越近奖励越高"的连续势能塑形，因为"如何朝食物走"恰是控制器必须自己学的技能（spec:78-84）。

**代码事实**（本条我亲自逐跳复核过）：

1. `environment_action_payoff = measurement.action_payoff if measurement is not None else None` —— [`final_wiring.py:1523-1526`](../../packages/vz-runtime/src/volvence_zero/integration/final_wiring.py:1523)
2. PE owner 只在该字段非 None 时覆盖自算值：`action_payoff = self._axis_value("action", ...)`，随后 `if action_context.environment_action_payoff is not None: action_payoff = _clamp_signed(...)` —— [`error.py:694-698`](../../packages/vz-cognition/src/volvence_zero/prediction/error.py:694)
3. optimizer 直接消费该值：`realized_action_payoff = _clamp(prediction_error_snapshot.actual_outcome.action_payoff)`，`reward = _clamp(realized_action_payoff + segment_bonus)` —— [`sandbox.py:2204-2212`](../../packages/vz-temporal/src/volvence_zero/internal_rl/sandbox.py:2204)
4. 那条自算的 `"action"` 轴基座是 `_action_signal = 0.34*family + 0.26*directive_pull + 0.18*exploration_pull + 0.22*residual_shift`（[`error.py:826-830`](../../packages/vz-cognition/src/volvence_zero/prediction/error.py:826)），其中 `directive_pull` / `exploration_pull` 读的是 `semantic_directive_pull` / `semantic_exploration_pull`（[`error.py:2253-2267`](../../packages/vz-cognition/src/volvence_zero/prediction/error.py:2253)）
5. 而蚂蚁发布的正是：`commit_pull = |food_left - food_right| * 4`、`explore_pull = 1 - food_center`、`forage_pull = food_center` —— [`sense_encode.py:194-203`](../../packages/vz-embodiment-ant/src/volvence_ant/substrate/sense_encode.py:194) → [`ant_adapter.py:125-129`](../../packages/vz-embodiment-ant/src/volvence_ant/substrate/ant_adapter.py:125)
6. settlement 从不以 measurement 是否存在为条件：只有 `EnvironmentOutcome` 对象为 None 才丢弃（[`sandbox.py:2142-2151`](../../packages/vz-temporal/src/volvence_zero/internal_rl/sandbox.py:2142)），而 AntSession 永远返回 outcome（非里程碑拍 `measurement=None`）

**触发面（精确版，比初判更窄也更锋利）**：ecology 的 measurement 只在 `valenced` 为真时构造，而 local valence 打开时（learned 臂默认）几乎每拍 food/home/cooling 三个 delta 里都有非零项（[`ant_session.py:655-676`](../../packages/vz-embodiment-ant/src/volvence_ant/runtime/ant_session.py:655)）。所以：

- **learned 臂**：fallback 只在"位移被木棍完全阻挡、三个 delta 恰为 0"的拍触发 —— 此时 `new_x == body.x`（[`ant_world.py:444-446`](../../packages/vz-embodiment-ant/src/volvence_ant/env/ant_world.py:444)），中性木棍因此换来一个约 0.38–0.9 的正奖励。这直接抵消 spec:258-259 / spec:268-269 冻结的"木棍接触不产生任何 payoff/valence"。
- **`dense-local-shaping-off` 消融臂**（`ecology_local_valence_enabled=False`）：**每个非里程碑、非热事件拍都走 fallback**。也就是说这条"关掉密集塑形"的消融，实际把密集塑形换成了另一种密集食物塑形。该臂的全部历史结论（06 文档:63 的"关闭 dense local shaping 没有造成 near 崩溃也没有修复远距"）在因果上不可解释。

**可观测性也一并失效**：导出的 `nonzero_ecology_payoffs` 取自 `AntStepRecord.ecology_action_payoff`（measurement 侧，fallback 拍恒为 0，[`ecology_curriculum.py:1032-1035`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_curriculum.py:1032)），`nonzero_reward_steps` 取自 PE signed residual（[`matched_control.py:329-331`](../../packages/vz-embodiment-ant/src/volvence_ant/proofs/matched_control.py:329)）；optimizer 真正消费的 reward 流（`ZTransition.reward_components`，[`sandbox.py:2221-2226`](../../packages/vz-temporal/src/volvence_zero/internal_rl/sandbox.py:2221)）从未离开 `vz-temporal`。整个仓库对 `realized_action_payoff` / `raw_reward` / `reward_components` 在 ant 侧零引用 → 泄漏在现有 evidence 里结构性不可见。

**复核补充（同簇，独立发现）**：`evaluation` 默认仍耦合进 PE 的 actual outcome —— `error.py:2229-2231` 保留 evaluation 派生的 `family_signals`，除非 `VZ_PE_EVALUATION_DECOUPLED` 为真，而全仓库只有 `docs/specs/evaluation.md:126`、`docs/specs/prediction-error-loop.md:70/77/280` 提到该开关，**没有任何 ant 脚本/配置/profile 设置它**；evaluation backbone 确实发布被消费的 family（`backbone.py:244`、`:505`、`:535`）。这直接触碰 spec:89 的"禁止 evaluation 反灌 reward"。

**修法**：settlement 时若 `measurement is None`，reward 必须为 0（或按 spec:95-96 预留的做法新增 typed reward eligibility，并逐条审计 nonzero reward 的 outcome lineage）；同时把 optimizer 实际消费的 reward 及其分量导出到 `AntStepRecord`，让"稀疏性"成为可验证量而不是假设。

---

### BLK-02 `PE-off` 实际是 `reward-off`

**设计要求**：PE / ETA 因果贡献的正式 claim 需要"同感知、同预算的真实 PE-off 与 ETA-off"；kill condition 是"以 random 代理消融，或策略参数未按预期改变"（spec:377-380）。curriculum 自己的 docstring 也声明 PE-off 后"PE 仍是 readout"（[`ecology_curriculum.py:596-604`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_curriculum.py:596)）。

**代码事实**（本条我亲自复核）：

`prediction_error_enabled=False` → `external_prediction_error_drive=False`（[`ecology_curriculum.py:618`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_curriculum.py:618)）→ `runtime_replay_prediction_error_enabled=external_prediction_error_drive`（[`session.py:935-938`](../../packages/vz-runtime/src/volvence_zero/agent/session.py:935)）→ 每次 settlement 都转发给两个 sandbox（`joint_loop/runtime.py:233-235`、`:374-390`）→ sandbox 里：

```python
segment_bonus = 0.0
if prediction_error_reward_enabled and segment_records: ...
realized_action_payoff = _clamp(...) if prediction_error_reward_enabled else 0.0
reward = _clamp(realized_action_payoff + segment_bonus)
```

（[`sandbox.py:2194-2212`](../../packages/vz-temporal/src/volvence_zero/internal_rl/sandbox.py:2194)）

即：**PE-off 臂的每一条 runtime-replay transition 的 reward 恒等于 0**，pickup(+0.5)/delivery(+1.0)/热事件 payoff 一个都到不了 optimizer。复核者另找了唯一可能改写 reward 的钩子 `_apply_delayed_credit_assignments`（`sandbox.py:2629-2655`），确认它只从 proof lane 可达，不会补偿。

**注意一处表面冲突**：F 簇（P2）把 pe_off 判为"真实消融"，理由是 `session.py:1703-1738` 把 PE 从 `set_external_learning_signals` 摘掉并只留 readout key、`joint_loop/runtime.py:258-276` 把 temporal switch pressure 归零。这两件事都成立 —— 但它们是**同一个 flag 的另外两个后果**，并不排除 reward 归零。正确表述是：**PE-off 做的事远多于其名义**（关 switch pressure + 关 external learning signals + 归零 replay reward），因此 `learned vs pe_off` 测的是"有奖励 vs 无奖励"，不是"PE 的因果贡献"。

**修法**：把 replay reward 的启停与 `external_prediction_error_drive` 解耦 —— PE-off 应只切断 PE 对学习信号/switch pressure 的驱动，环境 payoff 与 segment credit 保持匹配；或改用 spec:399-401 给出的构造。任一方案都要新增"消融杠杆确实生效且只生效于名义机制"的 gate（见 BLK-10）。

---

### BLK-03 P0 的 action-chain `PASS` 由逐 episode 回滚制造

**设计要求**：plan:119-124 的冻结门 —— food/heat 左右 `code_l1_delta > 1e-8`、`turn_delta >= 1e-4` 且符号在重复运行中一致、训练后转向敏感性不低于 shared-initial 的 25%、learned 的 action-head 必须有有限非 NaN 更新、no-optimize 指纹不变、每 body 报告且群体 ≥80% 通过。

**代码事实**：

1. guard 失败即回滚：`runner.restore_learning_checkpoints(checkpoints)` 后以 `...:action-chain-rollback` 重新导出，置 `action_chain_rollback_applied = True` —— [`ecology_curriculum.py:994-1016`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_curriculum.py:994)
2. guard 在 P0 是**开着**的：`_curriculum_config` 不设 `action_probe_guard_enabled`，默认 `True`（`ecology_curriculum.py:141`），且 audit 传了 baseline（`ecology_mechanism_audit.py:376-379`）。对比 P1/P2 显式关掉它（`ecology_p1.py:566`、`ecology_p2.py:743`）
3. guard 结果只进 telemetry，四个 gate 一个都不读 `action_chain_guard_passed` / `action_chain_rollback_applied`（[`ecology_mechanism_audit.py:299-305`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_mechanism_audit.py:299) vs [`:632-690`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_mechanism_audit.py:632)）；全仓库这两个字段只有那两处出现
4. **已提交 artifact 自证**：`results/ecology_recovery/p0/ecology_mechanism_audit.v1.json` 中 `"action_chain_rollback_applied": true` 出现 18 次（9 learned + 9 no-optimize，即 18/18），19 个 `action_chain_snapshots` 的 4 个 body `policy_fingerprint` 前缀全是 `d90197ed`（learned final == shared initial），而 `"verdict": "PASS"`
5. 06 文档:14 把同一件事写成"18 个会破坏群体敏感性的候选训练更新被拒绝并回滚"——与"19/19 snapshot 通过"是同一现象的两种叙述

**结论**：`action_chain` 通过的是"同一个冷 checkpoint 被测了 19 次"。plan:121 的 25% retention 实际失败 18/18（记录为 `heat:retention=0.0024/0.0117`，即 20% < 25%），响应是静默回滚而非 BLOCK。plan:122 的"learned 必须有有限非 NaN 更新"在 P0 完全没有对应 gate（curriculum 侧有 `policy_changed`，`ecology_curriculum.py:2023-2028`，P0 没引入）。

**附带（同簇复核补充）**：
- P0-C 是在**回滚后的冷 checkpoint** 上跑的（`_frozen_evaluation_audit(checkpoints=learned)`，`ecology_mechanism_audit.py:592-604`，而 learned 与 initial 字节相同）→ 零参数控制器上的冻结审计检不出参数非平凡后才出现的泄漏。
- `no_optimize` 臂并非行为冻结：`_session_config(optimize=False)` 只清 `joint_apply_policy_optimization`，`joint_learning_enabled=True`、`joint_apply_writeback=True` 仍然为真（`ecology_curriculum.py:608-625`）；artifact 里 9/9 个 no_optimize episode 的 heat `turn_delta` 从基线 `0.0117` 塌到 `0.00124–0.00252`，证明 SSL/writeback 确实改了电机输出，而 gate 只比初末 `policy_fingerprint` 且全部被回滚 → 只能返回 True。
- 任何一层都无法因 P0 BLOCK 而失败：驱动脚本无条件 `return 0`（`scripts/audit_ant_ecology_mechanisms.py:51-53`），唯一回归测试接受任一 verdict（`tests/test_ecology_mechanism_audit.py:120-122`）。
- 提交的证据与代码错版：artifact 声明 `digital-ant-ecology-mechanism-audit.v1`，而模块在同一 commit 已发 `...v2`（`ecology_mechanism_audit.py:33-35`），驱动默认输出路径是 `ecology_mechanism_audit.v2.json`（不存在）。即 06 文档:17 指向的"正式报告"**无法由树内代码复现**。

---

### BLK-04 P0-C 只 gate 2/8 owner，artifact 自证违规仍 PASS

**设计要求**：plan:185 要求显式区分"允许变化的 episode/runtime state"与"禁止变化的 learned state"；plan:193 要求 `learning_enabled=False` 时**所有** learned-owner 指纹**逐拍**不变。

**代码事实**：

- 内核发布 8 个 `learning_owner_fingerprints`：`joint-loop/policy`、`joint-loop/temporal-learning`、`joint-loop/memory`、`prediction`、`credit`、`regime`、`dual-track-gate`、`reflection`（[`session.py:1136-1169`](../../packages/vz-runtime/src/volvence_zero/agent/session.py:1136)）——名字本身就说明它们都是 learning owner。
- audit 逐拍采集全部 8 个的 first-difference（`ecology_mechanism_audit.py:456-478`），但 `passed` 只由 `policy_stable and temporal_stable and replay coverage` 构成，`first_differences` 只报不 gate（[`:510-516`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_mechanism_audit.py:510)）。
- 被 gate 的那 2 个还只比"restore vs final"，不是逐拍（`:488-496`）→ 中途漂移后又回来的情况看不见，正相反于 plan:193 的"逐拍"。
- **artifact 自证**：两个场景各 24 条 `first_differences` 且 `passed: true`；owner/tick 分布为 `joint-loop/memory`(tick 0, body 0-3)、`regime`(tick 0, body 0-3)、`dual-track-gate`(tick 1)、`prediction`(tick 1)、`reflection`(tick 1)、`credit`(tick 2) = 6 owner × 4 body，全部发生在 `learning_enabled=False` 的会话里。
- 全仓库没有任何 allow-list 可以主张这 6 个 owner 合法可变 → 连"它们属于允许变化的 runtime state"这个辩护都无法成立。

**附带**：plan:186-188 指定的两个最小复现是 `butter_only / seed=307` 与 `heat_forced_escape / seed=101`，代码硬编码为 `BUTTER_ONLY, seed=config.seed+101` 与 `HEAT_FORCED_ESCAPE, seed=config.seed+211`（`:595`、`:602`）→ 没有任何 `config.seed` 取值能产生 (307, 101) 这一对；seed 307 是真实 held-out seed（`ecology_curriculum.py:137`）却从未被跑到。

---

### BLK-05 P0 的冷启门未按 exclusive steering 重推导

**设计事实**：spec:326-332 在 exclusive steering 落地后**显式重推导**了冷启探针门 —— 冷启 head 参数精确为零、确定性策略无转向，因此冷启只验 input reachability，转向能力改由训练后的 `paired_action_sensitivity` / `food_steering_alignment` 硬门验收。

**代码事实**：这次重推导只应用到了 P1 curriculum 的开跑前门（`ecology_curriculum.py:2220-2237`），**没有**应用到 P0 mechanism audit：

- 冷启 head 精确为零可从内核独立确认：`_initial_causal_action_head_parameters` 的 `output_factors` 与 `bias` 全零（`interface.py:2344-2353`），residual = `strength * tanh(bias + factors.basis)` = 0（`interface.py:799-810`）；exclusive steering 在 exploration>0 与 ==0 两个分支都把 base contrast 投影掉（`interface.py:3924-3962`），blend gate 也按 pair 取共享均值（`interface.py:4032-4045`）→ 镜像探针对的 contrast 轴两 lane 相同，`turn_delta` 精确为 0。
- 已提交测试把这个后果硬编码了：`assert not any(item.action_sensitive for item in probes)`，docstring 写明"a cold controller emits zero turn by construction"（[`tests/test_ecology_world.py:217-237`](../../packages/vz-embodiment-ant/tests/test_ecology_world.py:217)）。
- 而 P0 audit 仍用同一个 `_evaluate_action_snapshot` 评 shared-initial，并把它计入 `action_chain_ok`（`ecology_mechanism_audit.py:561-570`、`:605-609`），其中含硬门 `turn_delta < turn_delta_threshold`（`:203-217`）。

**双向后果**：(a) shared-initial 必然失败 → 今天跑 P0 只能 BLOCK；(b) `retention_floor = initial_turn_delta × 0.25 = 0` → 25% 门同时变成空门。这两个后果方向相反，说明该门当前既不能通过也不能约束任何东西。

**判定（这是我让审计特别裁定的一条设计完整性问题）**：spec 侧的重推导本身是**合法的**——它伴随一个被声明的设计不变量变更（转向所有权转移到 head），并把转向验收明确移到训练后硬门，而不是为了过门而放宽。真正的违规是**只改了一半**：P0 audit 与其逐 episode guard 未同步，导致 plan §2.1"阈值只能因修实现而收紧"的语义在 P0 侧既未收紧也未重推导，而是留下一个不可能通过的死门 + 一个空门。

---

### BLK-06 P0-B 基本未实现

**设计要求**：plan:137-165 是"temporal abstraction 成立"这一 claim 的全部证据基础：确定性 transition protocol（稳态巡航／food approaching／pickup-carrying／home approaching-delivery／safe→harmful／harmful→cooling）、逐拍两轨记录（beta 连续值/阈值/二值 switch、external switch pressure、`steps_since_switch`、segment 开闭 tick、闭合原因分类）、steady-state 负对照＋预声明 switch-rate 上限、positive control 能把 switch 定位到状态变化附近、"正常 trace 不能全部依赖 timeout"、segment-credit on/off 的 sense / pre-credit action / rollout lineage 对齐。

**代码事实**：

- 整个 temporal gate 是三个聚合存在性判断：`any(switch_count>0)` ∧ `beta-switch closures>0` ∧ `environment-milestone closures>0`（[`ecology_mechanism_audit.py:620-630`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_mechanism_audit.py:620)）。artifact 里的观测串就是 `{'switches': 31, 'close_reasons': (('beta-switch', 25), ('environment-milestone', 30))}` —— 纯聚合、无协议上下文。
- `_segment_telemetry` 只发聚合量（switch_count、closed_segment_count、longest_segment_length、close_reason_counts、switch_gate min/mean/max、两轨 gate 区间），plan:144-150 的逐拍记录一条都没进报告（`:246-306`）。
- 全仓库对 `negative_control|steady_state|switch_rate|transition_protocol|positive_control|timeout_dominance` 的检索，在 ant 的 ecology 模块内**零命中**（唯一命中是 `proofs/matched_control.py:69,233` 的 `switch_rate` 字段，无上限、无稳态协议）。
- `segment_credit_enabled=True` 在 audit 里是硬编码（`:375`）→ P0 不存在 off lane，plan:157/165 的 parity 从未比较。
- 值得记录的是：**substrate 侧的数据是齐的** —— `AntStepRecord` 已携带 `switch_gate` / `is_switching` / `steps_since_switch` / 两轨 gate / 最后闭合原因 / 闭合原因计数 / 两个 switch-pressure delta（`ant_session.py:115-117`、`:163-168`）。缺的是 audit 侧的协议、对照与门，不是感知/记录能力。

**后果**：negative control 与上限缺失 → 高频抖动与"真正的时间抽象"不可区分（gate 是单侧的，switch 越多越好）；无协议 → 任何 closure 都无法归因到具名状态边界；plan:172 明确说"只有 milestone closure 时不得主张 temporal abstraction"，而当前 artifact 的 30 次 milestone closure 与 25 次 beta-switch closure 之间没有任何区分性证据。

---

### BLK-07 P1 的能力门可被 smoke 退化通过，并直接解锁 P2

**设计要求**：plan:284-286 的第一条冻结门槛是"每个 medium/far tier 至少 5 个独立 layout seed"，随后才是"至少 60% layout 成功""成功 layout 中至少 60% body 完成（4 ants 时即 ≥3）"。plan:429 要求 P2 只在 P1 完整 PASS 后开始。

**代码事实**：

- `EcologyP1Config.__post_init__` 冻死了三个**比例**（0.6/0.6/0.05，任何其他值直接 raise），但对预算只校验 `min(training_rounds, evaluation_rounds, layouts_per_tier) >= 1` 与 `n_ants >= 1`（[`ecology_p1.py:88-102`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_p1.py:88)）。`layouts_per_tier: int = 5` 只是**默认值**，不是冻结门槛。
- 门的算术随之退化：`required_layouts = ceil(layouts_per_tier * 0.6)`（`:1271-1273`）在 `layouts_per_tier=1` 时 = 1；`required = max(1, ceil(n_ants * 0.6))`（`:665`）在 `n_ants=1` 时 = 1。门自己的 `threshold` 串会诚实地写成"≥1 layouts; each requires ≥1 bodies"（`:1281-1285`）—— 报告是诚实的，但没有任何东西拒绝它。
- `ECOLOGY_P1_GATE_NAMES`（`:56-72`）共 16 项，没有 `formal_configuration` 之类的预算门；`verdict = "PASS" if not breakpoints else "BLOCK"`（`:1530`）。
- CLI 把旋钮直接交给操作者，无下限（`scripts/run_ant_ecology_p1.py` 的 `--n-ants` / `--layouts-per-tier`）。
- **串行链因此可被绕过**：`load_p1_prerequisite`（[`ecology_p2.py:510-573`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_p2.py:510)）只校验 schema_version、gate 名元组、每个 `gate['passed'] is True`、`verdict == "PASS"`，然后返回它自己算的 sha256；**从不检查 P1 报告里的 `config`**，也从不调用已存在的 `verify_ant_artifact_manifest`。
- 而 P1 报告本身是裸 JSON（`scripts/run_ant_ecology_p1.py:71-73`，无 bundle/manifest/provenance）→ 一个手改的 `passed: true` 全绿 JSON，或一个真实的 1-ant smoke PASS，都能解锁 8-ant 正式矩阵。

**对比**：P2 自己把预算做成了硬门（`ECOLOGY_P2_FORMAL_*`，`ecology_p2.py:270-278`、`:650-690`、`:1854-1872`），所以这不是"团队不知道要冻预算"，而是 P1 这一层漏了，并且恰好是解锁 P2 的那一层。

---

### BLK-08 Stage 3 heat-route avoidance 从未被评

**设计要求**：plan:265-268 把 Stage 3（热区路线回避 + 觅食）作为独立能力：必须完成 pickup→delivery 且 harmful tick rate 受控，"没进热区但也没完成任务"记失败；spec:344-349 要求 held-out 拆成**五类**：butter-only、butter-with-neutral-stick、burning-match route avoidance、burning-match forced escape、composite。

**代码事实**：

```python
("heat_route_foraging", EcologyEvaluationScenario.COMPOSITE, EcologyTrainingTier.FAR)   # :652
("composite",           EcologyEvaluationScenario.COMPOSITE, EcologyTrainingTier.FAR)   # :658
```
（[`ecology_p1.py:645-661`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_p1.py:645)）—— 六个 gate 名，只有**五个不同的 scenario/tier 组合**。

- `EcologyEvaluationScenario.HEAT_ROUTE_AVOIDANCE` 确实存在且可构造（`ecology_curriculum.py:120` → `EcologyStage.BURNING_MATCH`，`:1386-1391`），且 `_scene_objects` 给 BURNING_MATCH 的是"黄油 + 火柴、**无木棍**"，与 COMPOSITE（含木棍）是真正不同的布局类（`:388-423`）。
- 它在 P1/P2 层是死枚举：P2 里只出现在 `_scenario_stage` 的字典字面量中（`ecology_p2.py:851-863`），而 P2 的 scenario 全部来自导入的 `_evaluation_specs`（`ecology_p2.py:65`，用于 `:590/:637/:1658/:1666/:1938-1941/:2470`）→ 那条字典项不可达。

**后果**：`heat_route_foraging` gate 实际在"黄油+木棍+火柴"的 composite 世界上打分，无法区分"火柴单独存在时的路线回避"与 composite 表现；`learned_not_worse_than_no_optimize` 与 `paired_capability_effect_positive` 对 composite 双计权；缺陷经 `_evaluation_specs` 原样传入 P2 的 `heat_route_foraging_and_exposure` primary endpoint。

---

### BLK-09 P2 的晋级判决与 loader 完全脱钩

**设计要求**：plan:419/431 —— 只有全部 P2 条件满足，checkpoint 才能标记 `PASS` 并允许 demo loader 加载；任一项失败即 BLOCK，BLOCK artifact 保留诊断但 loader 必须拒绝。

**代码事实**：

- 全仓库检索 `EcologyP2Report|ecology_p2`：只有 `ecology_p2.py`、`experiments/__init__.py`、`tests/test_ecology_p2.py`、`scripts/run_ant_ecology_p2.py`。`aggregate` 只写一个裸 JSON 并 `exit 0/1`（`scripts/run_ant_ecology_p2.py:142-158`）——**P2 PASS 不产出任何 promotion bundle/archive/manifest**。
- 可加载 artifact 只由 curriculum lane 产出（`scripts/train_ant_ecology.py:47-55` → `write_ecology_checkpoint_bundle`），准入完全由 `_validated_report_verdict` 决定，它只校验 schema_version + 15 个 `ECOLOGY_REQUIRED_GATE_NAMES` + verdict 自洽（[`ecology_checkpoint.py:111-125`](../../packages/vz-embodiment-ant/src/volvence_ant/evidence/ecology_checkpoint.py:111)、`:201-220`）。该文件里检索 `p2|P2`：零命中。
- 被加载的 checkpoint 会真的恢复进 live demo runner（`app/server.py:326-334` → `app/runner.py:161`、`:208-209`）。
- 而 curriculum PASS 的预算无任何下限：`EcologyCurriculumConfig.__post_init__` 接受 `n_ants>=1`、`stage_rounds>=1`、任意非空 `heldout_seeds`（`ecology_curriculum.py:147-175`）→ 1 ant / 1 round / 1 held-out seed 的 PASS 合法可加载，而 ≥8 ant / ≥3 seed 的正式下限只存在于那条无人消费的 P2 通路里。

**公平地记录已经做对的部分**：`load_promoted_ecology_checkpoint` 本身很严 —— 校验 manifest、要求 `digital-ant-ecology-checkpoint.v4`、按冻结 gate 集重算 verdict、拒绝 verdict/gate 矛盾、对 BLOCK 抛 "not promoted"、复核 archive digest / compatibility / aggregate fingerprint（`ecology_checkpoint.py:201-245`）；FixedRule 分支硬编码 verdict=BLOCK 且理由为"fixed-rule is a baseline, not learned evidence"（`app/runner.py:500-506`）。问题不在这些检查的强度，而在于它们检查的是**错误的报告**。

**较弱的旁路（复核补充）**：`AntAppManager.from_evidence_artifact`（`app/runner.py:702-725`）读任意 JSON，不做 manifest 校验。

---

### BLK-10 ETA-off 既混淆又不完全

**设计要求**：spec:377-380 要求"同感知/预算的真实 PE-off 与 ETA-off"，kill condition 含"以 random 代理消融，或策略参数未按预期改变"；plan:370 要求各臂只在具名机制上不同。spec:399-401 还给出了 eta_off 的**具体构造**：冻结 learned-lite 策略 + `ssl_interval=rl_interval=0`（旧 matched-control lane 就是这么做的，`scripts/run_ant_matched_control.py:87-100`）。

**代码事实（混淆）**：`temporal_writeback_enabled=False` → `joint_apply_writeback=False`（`ecology_curriculum.py:619`）→ `ant_session.py:223` → 内核里**一个** `owner_writeback_enabled` 布尔同时门控：

1. 两个 temporal 模块的 `_apply_temporal_reflection_writeback`（`joint_loop/runtime.py:1508-1518`，失败即短路成 BLOCK 记录 `:596-610`）
2. memory store 上的 `ReflectionEngine.apply` ＋ `regime_module.apply_policy_consolidation`（`:1533-1552`）

内核自己的注释就写着这件事：`runtime.py:1050` "This is separate from `apply_writeback`, which gates reflection/memory consolidation."

**代码事实（不完全，复核补充）**：`joint_apply_writeback=False` **并不阻止 temporal metacontroller 继续学习** —— SSL 在 `joint_loop/runtime.py:1040-1041` 优化 `self._world_policy`，internal RL 在 `:1282`、`:1306` 优化同一个策略，两者都不受 `apply_writeback` 门控（no-optimize 控制走的是另一套 checkpoint/restore，`:1047-1056`）。而 `_world_policy` 就是 temporal policy（`:163`、`:169`、`:684-688`）。因此被描述为"涌现时间抽象无法适应"的臂（`ecology_p2.py:158-160`、`ecology_curriculum.py:602-605`）每个周期仍在适应它。

**并且没有 gate 能发现**：`ECOLOGY_P2_GATE_NAMES`（`ecology_p2.py:245-269`）没有 curriculum 侧 `policy_changed` / `no_optimize_policy_stable` 的对应物；每个 shard 算了 `policy_digest`（`:1512-1520`、`:1687`）但只用于 resume 比较（`:1574`）和 `e2e_rl_baseline_present` 的非空检查（`:2180`），**从不跨臂比较**。所以"策略参数未按预期改变"这条 kill condition 在正式通路上无人看守。

**另一处不对称（复核补充）**：`random` 臂在正式矩阵里烧满预算，却不进任何 gate 或配对比较（`"random"` 在该 lane 只出现两次：arm spec `:186-191` 与 baseline 分支 `:939`）。P1 反而有 `forced_escape_above_random_floor`（`ecology_p1.py:63`）→ **正式阶段比解锁它的阶段更宽松**：learned 可以通过 P2 全部能力门而从未被证明高于 random floor。

---

### BLK-11 "任何代码/门槛变化使整批失效"未实现

**设计要求**：plan:381-386（任何代码或门槛变化使整批失效并重新开始）、plan:471（每个 shard 独立 manifest，聚合器只接受 config digest 一致的完整 shard）、plan:429（provenance 完整、artifact hash 可复核）。

**代码事实**：

- `EcologyP2ShardReport`（`ecology_p2.py:614-639`）带 `preregistration_digest` / `schedule_sha256` / `prerequisite` / `device` / `policy_digest` / archive digest —— **没有 commit、没有 tree hash、没有源码 hash、没有 provenance 对象**；写盘是裸 `path.write_text(json.dumps(...))`（`scripts/run_ant_ecology_p2.py:63-71`、`:112-118`）。
- 聚合器只按 `preregistration_digest != digest or schema mismatch or config != resolved` 收 shard（`ecology_p2.py:1833-1852`）→ 只要这三个串相同，**来自不同 commit 的 shard 会静默合并**。
- `preregistration_digest`（`:620-647`）覆盖 config、arm-spec 表、配对比较、endpoint/gate **名**列表、capabilities、held-out layout seeds、每 seed 的 schedule digest、alpha、memory capacity —— **不覆盖**冻结的 `outcome_score` 权重（0.5/1/0.25/0.02，`:693-716`）、任何 gate **逻辑**、世界/环境动力学、evaluation 内部实现。唯一相关的测试只把 P2 的副本 pin 到 curriculum owner 的 `_ecology_outcome_score`（`tests/test_ecology_p2.py:390-434`），不冻任何数值 → 改共享权重后测试仍绿、digest 不变。
- provenance/commit 只在**聚合时**从聚合进程所在工作树采集一次（`scripts/run_ant_ecology_p2.py:136-146`），而 `provenance_clean` 只是调用方传入的布尔（`ecology_p2.py:1813-1817` → gate `:2353-2360`）→ 该门证明的是"聚合时工作树干净"，不是"每个 shard 在干净树上训练与评估"。
- shard、confirmatory、preflight 三种产物一律无 manifest（`_write_json`），`p1_prerequisite_pass` 门只是把 shard 自己写下的字符串再读一遍（`:1875-1895`）。
- 附带：`shard_digests` 算在重解码后的 dataclass 上而非磁盘字节（`:2388-2396`）—— 语义绑定完整，但事后无法对文件复验。

---

### BLK-12 artifact 卫生：无 provenance、无 manifest、可覆盖 BLOCK

**设计要求**：plan:30-31（每次运行必须记录 git SHA、dirty、配置 digest、依赖版本、设备、训练 seed、布局 seed、模型 fingerprint；每个产物使用新文件名，不覆盖已有 BLOCK artifact）、plan:47-52（canonical JSON + manifest + 人类可读 summary + 原始 trace + 测试命令与结果）。

**代码事实**：

- **P1**：`scripts/run_ant_ecology_p1.py:71-75` 裸 `json.dumps` 落盘，无 bundle、无 provenance、无 sidecar；`EcologyP1Report`（`ecology_p1.py:148-162`）与 `EcologyP1Config`（`:77-87`）都没有 provenance/device 字段。**磁盘可证**：`results/ecology_recovery/p1/` 下四个 artifact 的顶层键恰为 `['config','description','diagnostic_breakpoints','diagnostic_results','gates','layout_results','schedule','schema_version','verdict']` —— 无 `provenance`；该目录 `.manifest.json` 数量为 **0**；而 `ecology_p1.preflight.v4.json` 的 `verdict` 是 `BLOCK`，即一个真实 BLOCK artifact 无法回溯到产生它的代码。
- **P2**：三个子命令全部走 `_write_json`，同样无 manifest。
- 唯一的 manifest 生产者 `write_ant_artifact_bundle`（`evidence/provenance.py:159-185`）只被 `write_ecology_checkpoint_bundle` 调用（`ecology_checkpoint.py:161-167`），即只覆盖 train/promotion 通路。
- `AntRunProvenance`（`provenance.py:38-48`）**没有 device 字段**（只有 platform）→ CUDA/MPS 与 CPU 运行在 provenance 上不可区分，尽管 `train_ant_ecology.py` 会按 `--device` 设 `VZ_TENSOR_DEVICE`。training seed 与 layout seed 也塌成一个 `seed_schedule`。
- **覆盖保护为零**：`atomic_write_json` / `_atomic_write_bytes` / `_write_json` 全部 `os.replace` 或 `write_text` 无条件覆盖；`evidence/` 与三个 ecology 驱动里检索 `exists()` 只有两处临时文件清理（`provenance.py:114`、`ecology_checkpoint.py:78`）；检索 `already exists|overwrite|exist_ok=False|FileExistsError|refuse` 无任何 artifact 写入守卫。而所有驱动的默认路径都是常量（`train_ant_ecology.py:20-21`、`run_ant_ecology_p1.py:19-25`、`run_ant_ecology_p2.py:38-41`）→ 用默认参数重跑一次即原地销毁上一份 BLOCK 报告与 archive，且因为 P1/P2 没有 manifest，替换不留痕迹。BLOCK 保护只存在于读侧、且只在 promotion 通路（`ecology_checkpoint.py:219-223`）。
- **resume compatibility 缺两项**：`_progress_compatibility` 只发 `artifact_kind / n_ants / latent_dim / runtime_replay [+ memory_entry_capacity]`（`ecology_p1.py:250-267`、`ecology_p2.py:1052-1064`），**没有 `sense_schema` 与 `input_dim`**，而 promotion bundle 的那条元组两者都有（`ecology_checkpoint.py:48-58`）。spec:65 与 spec:459 明确要求 archive compatibility 同时绑定 sense schema / input dim / latent dim（/ ant count）。缓解因素：ecology lane 里 `AntSenseSchema.ECOLOGY_V2` 是硬编码字面量（`ecology_curriculum.py:616`、`:623`），没有配置或 CLI 能改，所以目前不可被利用 —— 但这正是那条 compatibility 元组要防的**纵深防御被拆掉**。
- 附带（复核补充）：`provenance.atomic_write_json` / `stable_json_digest` / `_stable_json_bytes` / 两个驱动写盘全部用默认 `allow_nan=True`（其中两处还带 `default=str`），而 archive lane 是严格的（`canonical_json.py` 的 `allow_nan=False` + 拒绝重复键）→ 报告侧可能写出非有限常量。
- 附带：P1 驱动对 BLOCK 无条件 `return 0`（`scripts/run_ant_ecology_p1.py:76-82`），而 P2 子命令会返回非零（`run_ant_ecology_p2.py:92`、`:158`）→ 任何按退出码编排的 CI 会把 P1 BLOCK 当成功。串行约束本身仍安全（`load_p1_prerequisite` 会拒绝非 PASS），所以这是自动化面偏差而非门绕过。

---

### BLK-13 `joint_learning_enabled=False` 未关闭 reflection-prior 结构写回

**设计要求**：spec:118-124 —— 该 flag 是 joint-loop 的硬边界，即使恢复出 pending replay/batch，调度器也只能发布 `frozen-evidence-only`，不得执行 SSL、Internal RL、writeback 或 rare-heavy。

**代码事实**：SSL / RL / rare-heavy 确实被真正关掉了（`joint_loop/runtime.py:1706-1732` 的早返回先于 `_pe_rare_heavy_due`，且返回 `rare_heavy_review_recommended=False`）。但 writeback 只在 `run_scheduled_step` 内被关：`MetacontrollerParameterStore.apply_reflection_prior_update`（`temporal/interface.py:1728-1832`）的 `switch` / `persistence` / `learning-rate` / `beta-threshold` / `encoder` / `decoder` / `track-world` / `track-self` / `track-shared` / `action-families` / `action-family-structure` 各分支**全部无 `_learning_writes_enabled` 检查**直接改状态，其中 `:1757-1760`、`:1773-1780`、`:1786-1793` 正是 `n_z=16` 实际服务的 ndim 参数 scaler。

一处修正（复核给出）：`base-weights` 分支**是**被间接门控的，因为它经 `fit_temporal_from_signals`，后者在学习写入关闭时早返回（`interface.py:1712-1726`，守卫在 `:1719`）。其余分支如上所述。

复核还关掉了两条可能的辩护：ant 的具体策略类没有覆写带守卫的版本（`FullLearnedTemporalPolicy.apply_reflection_prior_update` 于 `interface.py:3635-3644`、`LearnedLiteTemporalPolicy` 于 `:2834-2843` 均直接委托给 store）；而 `session.py:628-631`、`:665-668` 确实对两个策略调了 `set_learning_writes_enabled(joint_learning_enabled)` —— 所以这是**接缝 bug 而非布线遗漏**。

**当前为什么没炸**：ant 的 ecology 运行安全，只是因为 `AntSession` 从不调 `begin_new_context`，因此不会入队 session-post 作业 —— 边界靠 embodiment 的回合形状**侥幸**成立，不靠 owner 契约。而这条路径写的每个参数都在 `learning_parameter_fingerprint` 内，一旦触发就会同时污染冻结评估与指纹门。

---

## 2. 可行性分析：门槛本身达不达得到

本节回答的不是"代码是否符合设计"，而是"即使完全符合，设计的冻结门槛能否被达到"。全部结论都从代码常数推演，并标注哪一部分被对抗性复核推翻。

### 2.1 FEA-01 训练与冻结评估不是同一个动力学（决定性）

**训练侧**：exploration proposal 是 `sha256(f"{context_digest}:{segment}:option:{index}")` 映射到 `[-1,1]`，其中 `segment = step // _RUNTIME_EXPLORATION_OPTION_STEPS`，而 `_RUNTIME_EXPLORATION_OPTION_STEPS = 8` → 每一维是**保持 8 拍的常数**，即常曲率圆弧。`strength=1.0`（`ANT_RUNTIME_EXPLORATION_STRENGTH`，[`runtime_profile.py:14`](../../packages/vz-embodiment-ant/src/volvence_ant/evidence/runtime_profile.py:14)）时混合项 `original*(1-strength)` 恰为 0，encoder 自身噪声被完全替换；`posterior_std` 下界 = `0.4*strength = 0.4`。按 06 文档:139 记录的实测 z 值 0.015–0.024，`candidate = clamp(mean + std*noise)` 可达约 0.42 → 单拍 `turn = atan2(4×0.42, 1+4×0.42) = atan2(1.68, 2.68) = 0.560 rad ≈ 32°`，并保持 8 拍。

**评估侧**：held-out builder 传 `sparse_exploration_enabled=False`（[`ecology_curriculum.py:1410-1419`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_curriculum.py:1410)）→ `strength = 0.0` → `if latent_override is None and strength > 0.0` 分支被跳过，`z_candidate = encoded.z_tilde`；随后 exclusive steering 的 `else` 分支令 `deterministic_candidate = z_candidate`，于是 `z_candidate := z_candidate + project_base_code_off_contrast(z_candidate) - z_candidate` = **pair 的共模** —— base 对 contrast 轴的贡献精确为零（`interface.py:3924-3962`）。beta gate 也被 pair-mean 投影（`interface.py:4041-4044`），无法凭空造出 contrast。世界侧不加扰动（ecology 从不填 `motor_distortions`，`ant_world.py:157` 默认 `()`）。

→ **被打分的确定性策略的全部转向 = head residual**，v25 的 16 行 probe 实测 **0.0055–0.0327 rad/拍**，而 plant 上限是 0.675 rad（45°）。

**算术**：40 拍 × 0.033 ≈ **1.3 rad** 累计可用转向；一次定向 medium/far 往返（出发对齐 + 趋近食物 + 掉头回巢）需要约 **3.4–4.0 rad** → **2.5–8× 的运动学缺口**；单拍幅度差 17–100×。

**解释力**：这一条足以解释 06 文档里 v10→v25 的整个模式 —— 训练侧 near/medium pickup 数字健康、冻结 medium/far 恒为 0/5；也解释了为什么唯一通过的能力恰好是 `forced_escape`（只需约 3 拍直行、不需要转向）。它同时意味着"继续加训练量"或"再找一个衰减因子"都不会奏效：**策略在一个运动学 regime 里被优化、在另一个里被打分**，而 `food_steering_alignment` / `paired_action_sensitivity` 探针也是在确定性策略上测的，所以探针可以转正而被打分的行为仍然接近弹道直行。

**复核修正（务必带上）**：P2 的冻结 held-out 是 **120 rounds**（`ECOLOGY_P2_FORMAL_MIN_HELDOUT_ROUNDS = 120`，[`ecology_p2.py:278`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_p2.py:278)，默认见 `:306`），累计可用转向三倍到约 4.0 rad → **纯累计转向在 P2 不再是硬阻塞**；P1 的 40 rounds 才是。任何引用本节数字的写作都不要把 40 拍的结论搬进 P2 endpoint；P2 当前被阻塞的是程序性原因（`p1_prerequisite_pass`，`ecology_p2.py:248`、`:1884`）。

**附加结构约束（复核补充）**：被打分的控制器**结构上无法减速或停下** —— `_clamp` 把 code 界在 `[0,1]`（`interface.py:1907-1908`），故 `speed_unit = sigmoid(4·z2) ∈ [0.500, 0.982]`、`step_command ∈ [0.200, 0.393]`（`motor_decode.py:75-82`、`ant_actuator.py:47-56`）。而证明布局可解的预注册 oracle 恰恰在需要的修正超过 `1.25×max_turn_rate` 时设 `step_command = 0.0` **原地转向**。即"可解性证明"用了一个被打分策略不具备的自由度。

### 2.2 FEA-04 单轴多路复用与"绝对对齐"判据

**先记录被推翻的强版本**：审计初判"food/heat/home/obstacle 四条转向驱动在 exclusive steering + 镜像等变后代数互斥"被 **REFUTED** —— head 是对完整 19 维镜像观测的非线性两层映射，四个探针是四个不同状态，且 v25 已在 30 局同一正式 probe 分布上同时报出 food/heat/home/obstacle 全部 `4/4`（06:145）。

**但真实约束仍然存在**：`contrast_pairs = ((0,1),)`（[`runtime_profile.py:26-28`](../../packages/vz-embodiment-ant/src/volvence_ant/evidence/runtime_profile.py:26)）让 food / heat / home-PI / obstacle 竞争**同一根标量轴**，而 `food_steering_alignment` 要求食物驱动在探针状态上**绝对**压过其余全部转向驱动（`ecology_p2.py:2239-2262`），能力门又同时要求携食时 homing 驱动赢下同一根轴。已测比值：`authority/baseline` near `0.48`、medium `0.19`（06:139）；通道增益审计显示 food 的 `turn_gain` 0.00851 已是四通道最高（heat 0.00429、home 0.00309、obstacle 0.00058），压住绝对判据的是**逐体常数直流** `head_off`（约 `-0.005…+0.002`，按固定比例传导 `head_off×0.45=code_off`、`code_off×(-3.8)=turn_off`，06:141）。

**判断**：不是构造上不可能，但要求"食物增益 > 其余通道之和"在**每个被探测状态上**成立，而系统里没有任何时间复用或状态门控仲裁机制。若要稳定提高通过比例，方向是通道仲裁/时间分离/第二执行器轴，而不是继续加训练量（v23→v24：authority +16%，baseline 同步 +30%）。

### 2.3 FEA-02 far 训练预算（强版本被推翻，弱版本成立）

**被 REFUTED 的强版本**："far 往返连完美控制器都采不到"。按 plant 常数重算：`step_command = sigmoid(z2·code_gain) × step_size`，`code_gain=4.0`、`step_size=0.4`（ecology 从不覆盖 `AntWorldConfig` 默认值），`z2 ∈ [0,1]` → `step ∈ [0.200, 0.393]`/拍。24 拍：满速 **9.43 单位**，零码也有 **4.80 单位**。需求（`2·(d - pickup_radius) - nest_radius`，pickup 见 `ant_world.py:595`、delivery 见 `:579`）：`d=3.0` → 2.8；`d=3.7`（均值）→ 4.20；`d=4.4`（最坏）→ 5.60。满速余量 1.7–3.4×，即使下限速度也过均值需求。预注册 oracle 的最坏 far 往返约 4 拍转向 + 9 拍出发 + 4 拍转向 + 6 拍回程 = 23 拍 < 24。

**成立的弱版本**：以当前策略的**实测**速度（z2≈0.02 → `sigmoid(0.08)≈0.52` → ~0.21–0.225/拍），24 拍只有约 5.4 单位，对 4.20（均值）/5.60（最坏）→ far delivery 在训练分布里几乎不可采样，far 能力只能依赖 near/medium 的零样本迁移。这与 spec:320-322 的"near→medium→far mastery 且每阶段达到预声明 pickup/delivery 样本量"直接冲突。

**我的独立核算**（与上一致，但需统一一个常数）：训练 24 拍 = 9.6 单位（按 `step_size` 满值），held-out 40 拍 = 16.0 单位；far 最短往返按 `food_pickup_radius=1.2`（[`ant_world.py:153`](../../packages/vz-embodiment-ant/src/volvence_ant/env/ant_world.py:153)）算是 6.4 单位，按 `ButterSource.radius=1.1`（`world_objects.py:143`）算是 6.6 单位。**两个拾取半径常数并存**，取哪个会改变约 0.2 单位余量 —— 建议在环境 owner 侧统一，并让 spec 里的 1.1 与代码一致。

### 2.4 FEA-03 near 假阳性的真实规模

- body 生成在**巢心** → 约 **37.5% 的 near layout 首拍即 pickup+delivery**（零位移即同时满足两个里程碑）；最小轨道半径 0.287 使"紧圆巡游"覆盖 near 全程。spec:340-343 只承认"拾取盘与巢重叠"，实际比这更强。
- `forced_approach` 的随机环确实堵住了出发段的退化解：`uniform(1.45, 2.9) × butter.radius` 半径、`uniform(0.4π, 0.8π)` 偏角、按 `(seed+body_id)%2` 左右交替，逐 body 从 layout seed 抽样（[`ecology_curriculum.py:519-540`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_curriculum.py:519)），最近距离 ≥1.518 > 1.1 → 直线必然错过。**但回巢段仍然免费**（pickup 后 body 已在巢附近），所以 forced_approach 只对"出发趋近"施加了压力。
- 相关指标退化：`encountered_food` 判据是 `nearest_food_distance <= 4.4`（[`ecology_curriculum.py:705-706`](../../packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_curriculum.py:705)），即 far tier 的最大食物距离，而非有效感知半径（`antenna_reach=0.9`）→ 几乎每个 body 在每个布局都"encounter"，该 metric 无区分力。
- 反向被低估的难度（复核补充）：`neutral_stick` 与 `composite` 在 **FAR** 上打分，而木棍是**运动阻挡**（`WorldObstacle`，`world_objects.py:404`，由 `_resolve_obstacle_motion` 解析，`ant_world.py:236/266/299`），放在食物方位 `0.55d` 处、半长 `min(1.35, max(0.8, 0.35d))`（`ecology_curriculum.py:393-410`）→ `d=3.7` 时约 2.6 单位宽的墙，门仍要求 3/4 body 绕行并在 40 拍内完成往返。spec 把木棍描述为"中性几何、无 contact mastery"，但它在几何上是硬约束。

### 2.5 FEA-05 证伪端点覆盖

| 证伪目标 | 端点是否存在 | 说明 |
|---|---|---|
| R2 冻结基底 + 有界控制器 | **可证伪** | 冻结函数纯度、probe、`effective_dims` / `contrast_pairs` 门都在 |
| R3-R4 时间抽象 | **端点存在但证据薄弱** | switch / 非 timeout closure 有 gate，但 P0-B 未实装（BLK-06）→ 无法区分抖动与抽象 |
| R5-R6 连续记忆谱 | **无端点** | 无 memory 臂、无 memory endpoint；head 只读 zero-history `causal_action_head_state`（`interface.py:753-810`），steering 通路契约上零历史 |
| R-PE | **端点被污染** | BLK-01（fallback reward）与 BLK-02（PE-off=reward-off）使 PE 因果结论不可解释 |
| SSOT 快照隔离 | **部分** | `src/` 内边界干净且有测试，但测试自身与驱动脚本不受约束（A 簇 A-1） |

**同时记录一条被 REFUTED 的担忧**："四个配对消融都被钉在恒零的能力端点上，所以 P2 无论真相如何都会返回 no effect"——不成立。`_paired_differences` 差的是 `outcome_score = pickups×0.5 + deliveries×1.0 + heat_escapes×0.25 − harmful_heat_ticks×0.02`，是带 0.02 分辨率的**部分信用连续量**；`forced_escape` 又是唯一通过的能力，为每个臂贡献 `heat_escapes×0.25`；各臂在 pickups 上已经分离（06:63、:65、:145）。此外 `food_steering_alignment` 与 `carrying_home_action_alignment` 本身就是"对齐 body 比例"型连续端点，已在冻结 gate 列表内。

### 2.6 FEA-06 P2 算力包线

按 P2 冻结预算（8 ants、55 episodes × 80 stage rounds 训练、6 capabilities × 5 layouts × 120 rounds held-out、10 个 arm spec、≥3 training seed）估算：

- 训练每 `(arm, seed)`：55 × 80 × 8 ≈ **35,200** kernel step ≈ 2.2 h
- held-out 每 `(arm, seed)`：6 × 5 × 120 × 8 ≈ **28,800** step ≈ 1.8 h
- 全矩阵合计约 **1.1M** 次内核 `AntSession.step`，约 **77 CPU·h** 的内核步进

→ 可运行，但属于"多日单次战役"，主成本是内核步进本身而非 artifact I/O（双槽 archive 各约 31 MB/arm，06:147）。参考点：当前 CPU 上 4-ant × 10-layout × 40-round 冻结评估约 6 分钟（06:151）。本机 PyTorch 支持 MPS 但运行环境报 MPS 不可用（06:153）→ 实际按 CPU 规划，且 plan:468 要求的 MPS/CUDA parity smoke 目前无代码实现（G 簇 G-10）。

---

## 3. 逐簇一致性矩阵

每簇给出：结论、逐条目表（状态 / 主要证据 / 测试锚点）、该簇阻塞与对抗性复核结论、复核者补充发现。表中"测试"列为空白或 `—` 表示**该条目没有任何测试锁定**。

### 3.A 冻结基底与 SSOT 边界 — `conformant-with-gaps`

**结论**：这一簇实质到位。两个冻结向量函数是真纯函数（模块级、无闭包状态、无 RNG、常数全为字面量）；`ant-sense.v1` 恰好 14 通道、`ecology-v2` 用 `*SENSE_CHANNELS_V1` splat 在尾部追加 5 个热通道（尾部增长是结构性的而非约定）；`motor_decode` 与 spec 的 opponent residual 完全一致，被删除的 `atan2(z1,z0)` 形式在全仓库不存在；没有任何对象坐标/方位/几何进入 substrate（真值被隔离在 `eval_` 前缀后，`sense_encode` 一个都不读，热值在触角尖坐标采样）。**19 维镜像映射我逐通道核对过，19/19 正确**，且导出对偶一致（`mirrored[2] = mirrored[0] - mirrored[1]`）、swap 对符号相等因此对索引约定鲁棒。所有权干净：`vz-temporal`/`vz-runtime`/`vz-substrate` 中对 `volvence_ant` 与 ant 通道名（`ant_sense`/`heat_left`/`food_diff`）检索全部零命中，内核侧把置换当 schema 无关代数重新校验。

| 条目 | 状态 | 主要证据 | 测试 |
|---|---|---|---|
| A-1 禁止直接 import 内核内部，由 `test_import_boundaries.py` 强制 | **partial** | `tests/test_import_boundaries.py:15,18-35,49-60`；`pyproject.toml:12-24` | `test_import_boundaries.py:49` |
| A-2 只允许 vz-contracts / vz-substrate / vz-runtime facade | implemented | `test_import_boundaries.py:66-73`；`final_wiring.py:315-318` | `test_import_boundaries.py:63` |
| A-3 `ant-sense.v1` 恰 14 通道且不可静默升级 | implemented | `sense_encode.py:36-51,99-114,125-126` | `test_frozen_functions.py:50`；`test_ecology_world.py:161-163` |
| A-4 `ecology-v2` = v1 + 尾部 5 热通道（19） | implemented | `sense_encode.py:54-61,115-124` | `test_ecology_world.py:161`；`:203-210`（闭世界断言） |
| A-5 两个函数纯 numpy、无可学习参数 | implemented | `sense_encode.py:85-128`；`motor_decode.py:46-88` | `test_frozen_functions.py:44`（仅确定性） |
| A-6 无坐标/方位/几何进入 substrate | implemented | `ant_world.py:94-102,337-342`；`ant_adapter.py:70-113` | `test_ecology_world.py:203-210` |
| A-7 `motor_decode` opponent residual 且旧 atan2 形式不残留 | implemented | `motor_decode.py:67-70,71-79,81-82` | `test_frozen_functions.py:62-88` |
| A-8 `n_input` 必须声明完整感觉宽度、不得绑定 `n_z`、不得截断尾部 | **partial** | `ant_session.py:228-231`；`session.py:598-617`；`metacontroller_components.py:1394` | `test_ecology_world.py:168` |
| A-9 archive compatibility 绑定 sense schema + input dim + latent dim | implemented | `ecology_checkpoint.py:48-58,144,236`；`checkpoint_archive.py:279-281` | **—**（元组内容与拒绝路径均未被测试钉住） |
| A-10 `n_z` 与 `n_input` 分离（19 进 encoder、压到独立 latent） | implemented | `test_ecology_world.py:177-181`；`metacontroller_components.py:1478-1480` | `test_ecology_world.py:168` |
| A-11 navigator 从不读世界真值位置；无 live lane 用估计推进真值 | **partial** | `navigator.py:106-135`；`ant_world.py:434-447,484-486` | `test_frozen_functions.py:91-102` |
| A-12 天空罗盘只是带噪航向 + 互补滤波；gain=0 退回纯 dead reckoning；两臂默认一致 | implemented | `navigator.py:124-128`；`ant_session.py:75-76`；`fixed_rule_ant.py:37-40,160-164` | `test_frozen_functions.py:105-145` |
| A-13 镜像变换由冻结 substrate owner 发布，内核只当代数消费 | implemented | `sense_encode.py:139-182`；`causal_action_projection.py:8-84,172-205` | `test_runtime_profile.py:33-45,172` |
| A-14 镜像映射与真实通道顺序一致（19/19） | implemented | `sense_encode.py:150-181`（name-keyed 构造 + 缺通道守卫 `:171-175`） | `test_runtime_profile.py:172-186`（**仅部分**） |
| A-15 镜像 lane 用同一 encoder/head + 真镜像输入 | implemented | `interface.py:3794-3819,1919-1998` | `test_temporal_contracts.py:1148-1149` |

**阻塞与复核**：

- **A-1 CONFIRMED**（→ 见 BLK 附属）：`_SRC_ROOT` 硬绑到 `src/volvence_ant`，两个测试体都只遍历它（`:51`、`:77`）。`volvence_zero.memory` 与 `volvence_zero.joint_loop` 都在禁止前缀里，而**同目录的测试文件**（`test_runtime_profile.py:6`、`test_ecology_mechanism_audit.py:23`、`test_dynamic_colony.py:21`、`test_matched_control.py:91`）与 7/7 个驱动脚本（`run_ant_matched_control.py:61-62`、`run_ant_active_evidence.py:32-33`、`run_ant_demos.py:64-65`、`run_ant_dynamic_colony.py:109`、`run_ant_colony.py:34`、`run_ant_caste.py:107`、`run_ant_theater.py:37`）恰好 import 这些。`pyproject.toml:12-24` 只声明 vz-contracts / vz-substrate / vz-runtime / numpy，无 dev/test extra，包内也没有 conftest.py → **该包的测试套件无法在其声明的依赖闭包上运行**。复核补的范围限定：`src/volvence_ant`（含 app/ 与 viz/）确实干净；`scripts/` 位于仓库根驱动目录、可争议是否属于 owner 边界；**不可争议的是包内 tests/**。
- **A-11 REFUTED**：审计初判"live curriculum 用 `navigator.sync_to` 写真值位姿，违反 spec"被推翻，两条独立反证：(1) spec:333-334 **显式授权** —— "P1 固定 schedule 含两类强制起点 bootstrap，均只初始化身体状态并同步 body-side PI"，"同步 body-side PI"正是 `sync_to`，且代码严格门控在那两个 flag 上（`ecology_curriculum.py:971`）；spec:48-49 禁止的是"用 sync_to 构造的 lane 承载 AntBot/Ardin claim"，是 claim 范围规则而非调用禁令。(2) 前提不成立：`ecology_curriculum.py` 里对 `home_vector|home_error|pi_error|path_integration` 检索为空，AntBot claim lane 是另一条未受影响的通路（`experiments/phase0.py:104-163` 的 `homing_precision_experiment`，对 0.0067 判定，全文无 `sync_to`/`set_body_pose`）。
- **A-8 REFUTED**：两个结构事实为真（声明条件化于 `temporal_policy is None`；vz-runtime 在非 `FullLearnedTemporalPolicy` 时跳过校验），但推论不成立 —— eta_off 注入的是 `LearnedLiteTemporalPolicy`（`run_ant_matched_control.py:97-99`），该类从不触碰 `n_input` 或 ndim encoder：其 `step` 用 `_residual_signature` 造 3 元组 `base_code` 再 `_project_to_ndim(base_code, n_z)`（`interface.py:2874-2904`、`:2465-2474`），tile 的是 3 维向量而非 14 维感觉向量；全仓库没有任何调用方注入 `FullLearnedTemporalPolicy`。**结论**：条目留 `partial`（不设防的旁路真实存在），但当前无可达受害者。
- **A-14 CONFIRMED**：`sense_mirror_transform` 在全仓库只被一个测试触及（`test_runtime_profile.py:29,172-186`），其语义断言只有 2 个 swap（food、heat）与 3 个取反（`food_diff`、`home_ego_sin`、`last_turn_command`）；`:184-186` 的循环只断言 `permutation[source]==index` 与 `signs[index]*signs[source]==1`，**任何"自映射 + 符号 +1"都满足**。冻结契约 spec:306-308 实际更大（还含 obstacle 左右 swap、`heat_diff`、`home_pher_diff`、`trail_pher_diff` 取反）→ 把 `obstacle_left → obstacle_left` 或把 `trail_pher_diff` 符号翻成 +1 都能过全部现有断言，而内核侧校验（`causal_action_projection.py:22-62`）只查长度/双射/符号域/对合，是 schema 无关的。也没有行为级等变测试（镜像 `WorldObservation` 后与变换后的感觉向量比对）。**性质**：测试覆盖缺口，非现行缺陷（当前映射 19/19 正确）。

**复核补充发现**：

1. **（阻塞）天空罗盘与 PI 噪声在各臂间不一致** —— 违反 spec:56-58"罗盘是所有导航共用的 substrate 传感器，同一 frozen substrate 在 matched-control 各臂间一致"。kernel 臂用 `heading_noise=0.01 / step_noise=0.01 / compass_gain=0.85 / compass_noise=0.007`（`ant_session.py:70-76`，接线 `:187-192`），`FixedRuleAnt` 精确匹配并在注释里明写该规则（`fixed_rule_ant.py:37-40`）；但 **E2E-RL 臂**用 `heading_noise=0.0 / step_noise=0.0` 且**不设 compass_gain**（`AntNavigator` 默认 0.0）（`controllers/e2e_rl_ant.py:116-129`）→ 该臂跑在一个无噪声、无罗盘的不同基底上。
2. **（非阻塞，文档缺口）冻结函数里有一个未声明的 plant 常数**：spec:35-39 把 steering 代数冻结为 `forward=1+z0+z1, left=z1-z0`，实现却先把两通道乘以 `code_gain`（默认 4.0）再做该代数（`motor_decode.py:46-70`，`AntSessionConfig.code_gain=4.0` 于 `ant_session.py:77`）。`code_gain` 在 spec 与 plan 中检索均为零命中 → **冻结转向几何上的 4× 增益未被 SSOT 声明**。符号结构与"相等即直行"性质不变，故这改变的是转向幅度而非能力；但它还是一个 per-run 可变字段（无测试把它钉在 4.0）。

### 3.B Outcome→PE 接缝 — `non-conformant`

**结论**：measurement 侧非常干净 —— 世界 owner 完全没有 reward 词汇（`env/ant_world.py` 与 `env/world_objects.py` 对 `reward|payoff` 检索零命中），`EnvironmentMeasurement` 是 frozen value 且注释写明"normalized signed facts, not rewards"，FORAGING 恰好两个里程碑，木棍 contact 只进 facts 不进 payoff、纯 contact 不构造 measurement，携食期 home 项确实用的是方向无关的 PI 进度（不是外部信息素）。**但 optimizer 侧漏了**（BLK-01），且 PE-off 语义与文档不符（BLK-02）。这一簇的评级由此从"接缝干净"降为 `non-conformant`。

| 条目 | 状态 | 主要证据 | 测试 |
|---|---|---|---|
| B-1 typed/immutable measurement，只含可观察事实；世界不给 reward | implemented | `environment.py:112-131`；`ant_world.py:105-132,421-481`；`ant_session.py:322-329` | `test_ecology_world.py:334-344` |
| B-2 运行时不另建 mismatch slot；evaluation 不反灌 | implemented（但见补充 1） | `ant_session.py:330-340`；`joint_loop/runtime.py:1057-1112` | **—** |
| B-3 FORAGING 恰两个里程碑（pickup 0.5 非终局 / delivery 1.0 终局） | implemented | `ant_session.py:580-607`；`ant_world.py:106-121` | `test_ant_session_smoke.py:93-106`（仅 delivery） |
| B-4 任务结果路径无距离/势能塑形 | **partial** | measurement 侧干净；optimizer 侧见 BLK-01 | **—** |
| B-5 ECOLOGY 只增三个稀疏物理事实 | **contradicted**（见复核 B-3） | `ant_session.py:636-649` | **—** |
| B-6 有界 `action_payoff` 组合规则 | implemented | `ant_session.py:650-677` | `test_ecology_world.py:270-302` |
| B-7 携食期 home 项 = PI 进度而非信息素 | implemented | `ant_session.py:317-319,654`；`navigator.py:42-58` | `test_ecology_world.py:347` |
| B-8 木棍 contact 无 payoff、纯 contact 不构造 measurement | **partial**（字面满足，后果被 BLK-01 抵消） | `ant_session.py:640-643,667-676` | `test_ecology_world.py:305-344` |
| B-9 replay reward 优化已发布 `action_payoff`，PE 残差仅诊断 | implemented（机制正确） | `sandbox.py:2122-2132,2204-2226`（lineage 硬校验 `:2157-2184`） | **—** |
| B-10 `ACTIVE` 只选 transition 来源；样本不足报 `waiting-for-runtime-replay`，禁止合成回退 | implemented | `final_wiring.py:271`；`joint_loop/runtime.py:1057-1166` | `test_runtime_profile.py:33-45` |
| B-11 探索契约（默认 0.0、ant 1.0、只改 sample residual、std floor `0.4×strength`、不透明 context、匹配臂一致） | implemented | `final_wiring.py:333`；`interface.py:3021-3024,3873-3906` | `test_runtime_profile.py:48-67,189-215` |
| B-12 `commanded_turn/applied_turn` 只进审计记录 | implemented | `ant_world.py:425-434,466-467`；`ant_session.py:384-396` | **—** |

**阻塞与复核**：B-1（BLK-01）CONFIRMED；B-2（BLK-02）CONFIRMED；B-4（导出计数器与 optimizer 实际消费量不同）CONFIRMED；**B-3 REFUTED** —— "有害热区内每拍 -0.4 是第四个非稀疏 valence 项，违反 spec:85-88 的三稀疏事实"被推翻：审计只读了 §4 的 FORAGING 段，漏了 §5.3 明确冻结的**密集 local valence 通道** —— spec:257-261 冻结"燃烧火柴是 aversive（有害热暴露产生负学习信号，脱离/降温为正）"，spec:265-272 明确 AntSession 把 owner 发布的前后局部信号压成有界 payoff、"升温/有害暴露为负"。而被授权的通道（`0.45·tanh(food)`、`0.45·tanh(home)`、`0.7·tanh(cooling)`，默认开）本身比这条 -0.4 更密更大，故"-0.4 会压过 0.35/0.45 尺度项"的量级论证也不成立。**副产物**：spec §4 的"木棍真实碰撞…产生负 payoff"与 §5.3 的"contact 不产生任何 payoff/valence"互相矛盾 —— 这是 **spec 内部不一致**，代码遵循的是 §5.3（正确的那一条）。

**复核补充发现**：

1. **（阻塞）evaluation 默认耦合进 PE actual outcome** —— 直接触碰 spec:89 的"禁止 evaluation 反灌 reward"。`error.py:2229-2231` 在 `VZ_PE_EVALUATION_DECOUPLED` 为假时保留 evaluation 派生的 `family_signals`，而 `error.py:2196-2208` 自述默认是 SHADOW（legacy = evaluation 耦合）；全仓库该环境变量只出现在 `docs/specs/evaluation.md:126`、`docs/specs/prediction-error-loop.md:70/77/280` 与 `error.py` 自身 —— **没有任何 ant 脚本/配置/profile 设置它**。evaluation backbone 确实发布被消费的 family（`evaluation/backbone.py:244`、`:505`、`:535` 发 `family='abstraction'`），该 family 以 0.34 权重进 `_action_signal`（`error.py:827`）。
2. **（阻塞）"木棍无价态"的冻结在 optimizer 侧被抵消**：`ant_session.py:640-643` 遵守字面（只进 facts、无 payoff），纯 contact 时返回 `measurement=None`（`:667-676`）；但按 BLK-01 的机制，`measurement=None` 恰恰是 fallback 生效的情形。完全被阻挡的移动使位置不变（`ant_world.py:444-446`），三个 delta 全为 0、`valenced=False` → 蚂蚁因撞木棍得到约 0.38–0.9 的正奖励。
3. **（一致性）owner 发布的 home-pheromone 通道是死代码**：spec:265-267 指定 AntSession 只压缩 owner 发布的前后局部 food / home-pheromone / heat 信号，而 `local_home_signal_before/after`（`ant_world.py:422,450,480-481`）在构造之外**从未被读取**；实际用的是 `navigator_before.home_distance - nav_state.home_distance`（`ant_session.py:317-318,654`）。**注**：这与 06:47 记录的修复方向一致（从信息素改为方向无关的 PI 进度），且 PI 是 body 侧自估、不含坐标，所以是"spec 文本未跟上正确修复"，而非泄漏。
4. **（阻塞）整条 Outcome→PE→reward 接缝零测试覆盖**：`observe_runtime_transition` 在全仓库只出现在 `sandbox.py` / `joint_loop/runtime.py` / `agent/session.py`；`prediction_error_reward_enabled` 只出现在前两者 —— **没有任何测试文件触及**。ant 侧看似守这条的测试只断言配置布尔（`test_ecology_p2.py:451-472`、`test_runtime_profile.py:235`）和一个 fixture 字面量（`test_ecology_p2.py:412`）。因此"measurement 缺失即无 optimizer reward""PE-off 保留环境 payoff""里程碑 payoff 幅度冻结"三条都无人看守。
5. **（可行性提示）** reward = `clamp(realized + segment_bonus)`，而 fallback 轴是中性输入下约 0.38 的 `[0,1]` 量 → 无里程碑 episode 的 reward 流是"高均值、低方差的正常数 + 传感器噪声"，advantage 正由该流计算（`sandbox.py:1594` → `_aggregate_batch_targets`）。里程碑分量（0.5 / 1.0 / 热项）在总回报中占少数，需运行时测量真实 reward 分布才能定量 —— 而按 B-4 那正是唯一未导出的量。

### 3.C temporal / causal-action-head 跨库契约 — `conformant-with-gaps`

**结论**：这是本次审计里落地质量最高的一簇。spec §5.3 要求的每一项都真实存在，且多数有测试：head ACTIVE + `rank=n_z=16` + identity input + 零 output/bias 初始化、`causal_action_head_state` 零历史编码并在 live/pure/torch/pending capture/open segment 间共用同一持久化状态、`effective_dims` 在三处严格置零、`contrast_pairs` 正交差分在四条 lane 同构、exclusive steering 只作用确定性均值（保留探索噪声的反对称分量，这是 head 梯度不死锁的关键不变量）、镜像等变 `0.5(f(s)-f(mirror s))` + 速度轴对称半和 + 持久化 v5 schema 严格拒绝、`posterior_sample_scale` 逐 transition 持久化且 pure/torch 共同消费、segment credit 上限 16 / OR 闭合 / 按 transition 计批（ant=4，通用=1）、`n_z>3` 不再执行 legacy alignment、aggregate 发布 owner track weights。通用默认全部保持历史行为（字节等价回滚），且 `vz-temporal` 内对 ant 词汇检索只有一条注释命中。

| 条目 | 状态 | 主要证据 | 测试 |
|---|---|---|---|
| C-1 head ACTIVE / rank=16 / identity input / 零 output+bias；通用默认 DISABLED+低秩；已学 mapping 禁改 rank | implemented | `runtime_profile.py:22-23,90-96`；`final_wiring.py:285-292`；`interface.py:2307-2353` | `test_runtime_profile.py:73-99`；`test_runtime_transition_replay.py:249-300` |
| C-2 `causal_action_head_state` 零历史编码、live 与 replay 共用持久化状态 | implemented | `interface.py:3785-3802,3345-3350`；`sandbox.py:1975-1980,2101-2102` | `test_runtime_transition_replay.py:193-246` |
| C-3 `effective_dims=(0,1,2)` 在 pure/torch 梯度与 live/SHADOW residual 严格置零 | implemented | `interface.py:3994-4000`；`sandbox.py:1479-1484`；`torch_causal_ppo.py:171-176,348-353` | `test_temporal_contracts.py:765-796` |
| C-4 `contrast_pairs=((0,1),)` 四 lane 同一正交投影；`z[2]` 不动 | implemented | `causal_action_projection.py:128-141`；`interface.py:4001-4004`；`sandbox.py:1486-1489`；`torch_causal_ppo.py:388-396` | `test_runtime_transition_replay.py:163-192` |
| C-5 exclusive steering 四 lane 互补投影、只动确定性均值、空 pair 或非 ACTIVE 时 fail loudly | implemented | `interface.py:3923-3962`；`sandbox.py:896-902`；`torch_causal_ppo.py:397-408,429-431,447-450,475-477` | `test_temporal_contracts.py:873-956,996-1030` |
| C-6 exclusive steering 下 beta gate 按 pair 共享（零参数 head 输出精确为 0） | **partial** | live 正确且有测试（`interface.py:4032-4047`）；replay 侧逐维重建 | `test_temporal_contracts.py:960-994`（仅 live） |
| C-7 镜像等变（steering 差半 / 速度和半 / 全链持久化 / 默认 None 回滚） | implemented | `causal_action_projection.py:167-200`；`interface.py:3803-3820,3971-3993`；`sandbox.py:1508-1533` | `test_runtime_transition_replay.py:938-979` |
| C-8 `posterior_sample_scale` 逐 capture 持久化、pure+torch 共同消费、无残留硬编码 0.5 | implemented | `sandbox.py:1793-1799,2104-2105,2286-2287,1147-1149`；`torch_causal_ppo.py:512-520` | `test_runtime_transition_replay.py:1342-1475` |
| C-9 segment credit：上限 16、OR 闭合、按 transition 计批、开闭段进 checkpoint、DISABLED 精确回退 | implemented | `runtime_profile.py:15-21,79-89`；`joint_loop/runtime.py:446-487`；`scheduling.py:192-206` | `test_runtime_profile.py:68-99`；`test_runtime_transition_replay.py:982-1291` |
| C-10 bias 纪律（总幅 0.1 / 单步 0.01 / `0.12×0.05` 尺度 / batch mean 只更新 bias / factor 只吃 centered 协方差） | **partial** | pure owner 路径完全符合（`interface.py:880-931,963-976`）；torch lane 绕过 | `test_temporal_contracts.py:171-215`（仅 pure） |
| C-11 legacy 控制面隔离（`n_z>3` 不写 legacy 字段；aggregate 发 track weights） | implemented | `interface.py:1843-1852,4589-4610`；`joint_loop/runtime.py:2083-2092` | **—** |
| C-12 `vz-temporal` 内无 ant 语义 | implemented | 全仓库检索 `butter` / `pheromone` / `food` / `heat_diff` / `antenna` / `volvence_ant` 于 `vz-temporal/src` 仅 1 条注释命中（`interface.py:4038`） | **—** |

**阻塞与复核**：

- **C-1 CONFIRMED（live/replay clamp 边界不一致）**：live `_clamp = max(0.0, min(1.0, v))`（`interface.py:1907-1908`），在 ndim forward 的每一级都施加（`:1904-1905`、`:3955-3960`、`:4009-4013`、`:4046`）；而 pure `_clamp = max(-1.0, min(1.0, v))`（`sandbox.py:268-269`）用于 `modulated_mean`/`candidate_mean`/`policy_mean`（`:352-378`），torch runtime-replay lane 三处硬编码同样的 `-1.0/1.0`（`torch_causal_ppo.py:429-445`）。复核逐条排除了三种辩护：(a)"latent 是有符号的"——不成立，posterior mean 在 pure encoder（`metacontroller_components.py:223-227`）与加速 ndim backend（`backend_ndim_runtime.py:120-122`）都被 clamp 到 `[0,1]`，`z_tilde` 亦然；spec 里 signed `[-1,1]` 说的是**进入 head 的 state**，不是 latent code。(b)"后续 lane 会修正"——同一个 torch 函数对它的两条合成 lane 用的是 `[0,1]`（`:448-452`、`:462-466`），说明 `[-1,1]` 是 lane 局部不一致而非有意约定。(c) 无测试断言二者相等。**后果**：head 学到非平凡转向后（ant strength=1.0、residual 可达 ±1.0），重建的 mean 可以离开 `[0,1]` 而被捕获的 action 被夹在其内 → `(action - mean)` 在饱和区系统性反号，而 pure head 梯度正是 `(policy_action - policy_mean)/std²`（`sandbox.py:1466-1484`）。
- **C-6 REFUTED**：审计初判"pair-shared gate 只存在于 live，replay 逐维重建会把泄漏放回训练信号"被推翻，且给出了可证明的等价链：(1) capture 除的不是原始 posterior sample，而是 `sampled_candidate = tuple(runtime_state.z_tilde)`（`sandbox.py:1958`），ndim lane 下 owner 发布 `z_tilde = z_candidate`（`interface.py:4074`），即经过 track 调制、exclusive 投影、effective-dims 掩码与 head residual 之后的**最终** candidate。(2) live blend 是 `clamp(gate·cand + (1-gate)·prev)`，而 `cand`、`prev` 都已在 `[0,1]`、`gate ∈ [0,1]` → 凸组合，外层 clamp **永不触界**。(3) 因此 `beta_i = (action_i - prev_i)/(cand_i - prev_i)` 精确等于 `effective_gate[i]`。(4) exclusive steering 下 `effective_gate` 在 blend 前已被 pair-mean 投影 → 重建结果恒有 `beta_left == beta_right`。→ 在 replay 侧再投影是 no-op，不是缺陷。
- **C-3 CONFIRMED（torch lane 绕过全部 bias/factor 纪律）**，且比初判更强：`head_input`/`head_output`/`head_bias` 是普通 `requires_grad` leaf（`torch_causal_ppo.py:309-334`），三者与 track weights、log_std、critic 一起塞进**同一个** `torch.optim.Adam(optimizer_parameters, lr=learning_rate)`（`:554-563`）→ 单一无区分学习率，冻结缩放一个都不存活：无 0.12 bias 因子、无 0.05 state-path 因子、无 ±0.01 单步 bias 上限、无 ±0.1 总幅、无 ±1.5 factor clamp、无均值剔除（batch mean 可自由流入 factor）。写回把 detach 的张量原样交回并 `update_step+1`（`:606-640`），而 `restore_causal_action_head_parameters` 只校验 input rank/宽度、output 宽度/rank、bias 宽度（`interface.py:674-712`）——**没有任何幅度校验**。被绕过的 pure 纪律确实在 `interface.py:880-931`、`:963-976`。触发条件：`internal_rl_backend` 默认 DISABLED，但可由 `VZ_INTERNAL_RL_BACKEND` 打开（`final_wiring.py:266`、`:615`）。
- **C-4 REFUTED**：审计初判"镜像等变是逐调用点 opt-in，调用方忘传 `sense_schema` 就静默关闭"被推翻 —— 决定性事实是 `AntSessionConfig.sense_schema` 本身默认 `AntSenseSchema.V1`（`ant_session.py:96`），所以省略该参数的调用点（`ant_active_evidence.py:86/136/145/153`）构造的是 **V1 会话**，而 spec:311 明确要求非 ecology/V1 profile **不**携带镜像配置 → 这些调用点是合规的，把 ecology 镜像设成工厂默认反而会违反该条。所有真实的 ECOLOGY_V2 内核会话都同时向 rollout config 与 session 传了 schema（`ecology_curriculum.py:616+623`、`ecology_probe.py:185+189`、`scripts/measure_ant_food_steering_gain.py:148+151`、`scripts/measure_ant_carrying_home_conflict.py:204+207`），`app/runner.py:190-201` 是条件传递、正好符合 spec:311；`ecology_p2.py:899` 那处是 E2E-RL 基线控制器自己的 sense schema，不是内核 rollout config。

**复核补充发现**：

1. **M1（一致性）** `temporal-abstraction.md:124` 所称"sandbox 的 `CausalZPolicy` 与 torch PPO 使用同一公式"对**合成 lane** 不成立，且代码注释自认：`torch_causal_ppo.py:454` 写着"Match CausalZPolicy._policy_mean and the live ndim forward"随后 clamp 到 `[0,1]`（`:462-466`），而 `CausalZPolicy._policy_mean` 走的是 `[-1,1]`（`sandbox.py:268-269`，用于 `:869-895`）。更严重的是 torch 的"字节兼容历史回滚 lane"算 `candidate = weights·obs`（`:448-452`），pure lane 算 `hidden·0.40 + surface·0.25 + previous·0.15 + weights·0.20`（`sandbox.py:886-895`）——**不同函数**，不只是不同 clamp。范围限定：ant profile 用 ACTIVE runtime replay，不走合成 lane。
2. **M2（一致性）** pure lane 的 PPO surrogate 对 head 更新是盲的：runtime-replay transition 的 `_surrogate_metrics` 重算 `new_mean/new_std`，却传入**捕获时**的 residual（`action_head_residual=transition.runtime_action_head_residual`，`sandbox.py:1136-1155`）→ 报告的 `approx_kl`、clip fraction、surrogate 目标在 head 参数变动时不会动；head 只由独立的 score-function 路径训练（`:1413-1540`）。torch lane 相反，在图内重算 residual（`torch_causal_ppo.py:359-394`，由 `:435-440` 调用）→ 两个 backend 对同一批报告结构性不同的 KL。
3. **M3（一致性）** 跨 episode 迁移会让 replay 侧的 previous-code 与 owner 侧失步：`joint_loop.restore_learning_checkpoint` 在恢复的 checkpoint 不带 replay report 时对两个 sandbox 调 `reset_runtime_replay_for_episode_transfer()`（`joint_loop/runtime.py:866-868`），后者清零 `_runtime_previous_code`（`sandbox.py:1910-1924`）；而 temporal owner 自己的 previous code 只在 `__init__` 清零（`interface.py:3004`）、在 `:4092` 推进 —— 恢复路径不重置它。`include_runtime_replay=False` 正是 ant 驱动使用的模式（`ecology_curriculum.py:989`、`:1014`；`ecology_p1.py:450`；`ecology_p2.py:1233`）。
4. **M4（可行性）** head 的 score 梯度是 `(action-mean)/max(std²,1e-4)` 并 clamp 到 ±4.0（`sandbox.py:1466-1484`），`policy_std` 下界 0.02（`:379-392`）→ 分母最大 4e-4，任何 `|action-mean| > ~0.04/strength` 都会撞 clamp：**在 contrast 轴上 head 只收到符号信息、没有幅度信息**。步长预算重算：store `learning_rate=0.08`（`interface.py:533`），`factor_learning_rate = lr/len(batch)`（`:900-901`），ant 批为 4 条 settled transition（`runtime_profile.py:21`）并被镜像 lane 追加样本翻倍到 8（`sandbox.py:1507-1533`）→ 0.01；output-factor 单步夹 ±0.05、factor 夹 ±1.5（`:913-931`）→ 需约 20+ 个 optimizer 批（约 80 条 settled transition，约 4 个 episode）才能把 output factor 推到有意义的量级。

### 3.D P0 机制审计 — `non-conformant`

**结论**：25 条里只有 4 条 `implemented`、9 条 `partial`、8 条 `missing`、4 条 `contradicted`。P0-A 的探针基础设施（每 body、每 checkpoint、input reachability、80% 群体门、per-body 失败串）是真的；但三条冻结门在语义上被架空（BLK-03/04/05），P0-B 基本未实装（BLK-06），且 P0 自己的阈值全是可变 dataclass 默认值、无一被测试钉住（对比 P1 把 0.6/0.6/0.05 做成不可表达其他值）。

| 条目 | 状态 | 说明 / 证据 |
|---|---|---|
| D-1 `code_l1_delta > 1e-8` | implemented | `ecology_probe.py:226-235`；`ecology_mechanism_audit.py:47,207-208`；阈值只在可变默认值里，测试只钉更弱的 `>0.0`（`test_ecology_world.py:236`） |
| D-2 no-optimize 指纹不变 | implemented（但当前平凡成立） | `ecology_mechanism_audit.py:610-613`；测试只断言 gate **名**存在（`test_ecology_mechanism_audit.py:123-128`）；因全部 episode 被回滚，该门只能返回 True |
| D-3 first-difference 报告（owner/字段/tick/前后 digest） | implemented（报了但不 gate） | `ecology_mechanism_audit.py:459-486`；artifact `:21621-21630`；字段粒度只到 owner 级（内核每 owner 一个 digest） |
| D-4 冻结评估期 settlement/lineage ≥0.99、drop=0 | implemented | `ecology_mechanism_audit.py:497-516`；测试钉更严的 1.0（`test_ecology_mechanism_audit.py:132-136`）—— **P0-C 唯一有真实测试的子门** |
| D-5 `turn_delta >= 1e-4` | **contradicted** | 双向缺陷，见 BLK-05：P0 对冷启 snapshot 施加该硬门（必 BLOCK）；curriculum 侧的训练后 guard 又不读 `target_aligned` |
| D-6 训练后敏感性 ≥ shared-initial 的 25% | **contradicted** | 算术实现了，但冷启基线为 0 → floor=0 变空门；提交 run 中它触发 18/18 次，响应是静默回滚（BLK-03） |
| D-7 允许变化 vs 禁止变化的显式划分 | **contradicted** | 无 allow-list；6 个 learned owner 逐拍变化仍 PASS（BLK-04） |
| D-8 两个最小复现 `butter_only/307`、`heat_forced_escape/101` | **contradicted** | 硬编码为 `seed+101` 与 `seed+211`（`:595`、`:602`）；无任何 `config.seed` 能产生 (307,101)；seed 307 从未被跑 |
| D-9 有限差分近零空间诊断 | **missing** | ant 包与 scripts 内检索 `finite_difference` / `null_space` / `jacobian` 零命中；最接近的 `measure_ant_*_gain.py` 是另一套通道分解 |
| D-10 重复运行的转向符号一致性 | **missing** | 每个 checkpoint 每种探针只跑一次确定性探测 → 无法计算跨重复符号一致性 |
| D-11 learned head 必须有有限非 NaN 更新 | **missing** | P0 无 isnan/isfinite 检查、无 `policy_changed` 型门（curriculum 侧有 `:2023-2028`） |
| D-12 pure/runtime/torch parity | **missing** | 探针只跑单一 live 路径；P0 无 parity 门、无容差常数、无测试；plan:468 的 MPS/CUDA parity smoke 亦无实现 |
| D-13 确定性 transition protocol（6 个状态段） | **missing** | P0-B telemetry 取自普通训练 episode（BLK-06） |
| D-14 steady-state 负对照 + 预声明 switch-rate 上限 | **missing** | 无负对照运行、无上限常数；gate 单侧（switch 越多越好） |
| D-15 segment-credit on/off parity | **missing** | audit 硬编码 `segment_credit_enabled=True`（`:375`） |
| D-16 P0/P1 阈值必须先写入 schema/test | **missing** | 四个 P0 阈值都是可变 dataclass 默认值，`__post_init__` 接受任意正值；无测试断言 `1e-8 / 1e-4 / 0.25 / 0.8` |
| D-17 探针 checkpoint 时点（初始 / 每 primary / 每 interleaved / 每 stage / 最终） | **partial** | interleaved 从不被探针（`interleaved=False` 硬编码于 `:361-362`）；stage-end 与 final 只与各 stage 最后一局重合 |
| D-18 探针字段清单 | **partial** | 记录了 2 元传感器对、最终 code、turn、head residual、head update_step、checkpoint 指纹；**缺** 原始 19 维 sense、encoder/posterior hidden 摘要、`z_candidate`、track modulation、head 参数 fingerprint/范数/变化范数、motor readout 投影 |
| D-19 每 body 报告 + 群体 ≥80% + 无系统性同向 | **partial** | 前两者实现（`math.ceil(n*0.8)`，floor 1）；"无系统性左右同向"在 P0 完全缺失，尽管数据（`left_turn`/`right_turn`）已在手 |
| D-20 逐拍两轨日志 | **partial** | `AntStepRecord` 侧字段齐全（plan:151 的记录扩展已完成）；但 P0 报告只留聚合量 |
| D-21 positive control 可定位到状态变化附近 | **partial** | 只有 `any(switch_count>0)` + 两类 closure 存在性；无任何东西把 switch 绑到状态变化 tick |
| D-22 正常 trace 不能全部依赖 timeout | **partial** | plan:163 被覆盖；plan:164 无实现（timeout closure 有计数但无上限或占比约束） |
| D-23 owner-scoped 指纹逐拍前后采集 | **partial** | 8 个 owner 的滚动 before/after 已实现，runtime-only 状态被显式排除；但 gate 只用 2 个（BLK-04） |
| D-24 报告 schema `...mechanism-audit.v1` + canonical JSON + manifest/provenance | **partial** | canonical 排序 JSON 有，且 audit 从不导出可晋级 checkpoint（符合 plan:61）；但代码声明 v2、唯一 artifact 是 v1、驱动默认输出路径的 v2 文件不存在 |
| D-25 plan:200-206 建议的三个 per-mechanism 测试文件 | **partial** | 模块与脚本存在；三个测试文件不存在，合并文件只有 3 个测试且唯一涉及 audit 的那个接受任一 verdict |

**四条阻塞全部 CONFIRMED**（详见 §1 BLK-03/04/05/06）。**复核补充发现**（7 条，除 §1 已引用的以外）：

1. P0-C 跑在回滚后的冷 checkpoint 上（`:592-604`，learned 与 initial 字节相同）→ 零参数控制器上的冻结审计检不出参数非平凡后才出现的泄漏。
2. plan:120 的方向符号一半与 plan:124 的"不得有系统性左右同向"在 P0 未实装：`_evaluate_action_snapshot`（`:196-240`）与逐 episode guard（`ecology_curriculum.py:883-907`）只用 `|left_turn - right_turn|` 与 reachability，从不读 `probe.target_aligned` —— 而 P1/P2 lane 是读的（`ecology_p1.py:1418,1453`；`ecology_p2.py:1303`；`ecology_curriculum.py:1573`）。**后果**：一个"朝热源转、背食物转"或"两侧同向但幅度不等"的控制器能通过 P0 action-chain 门。
3. `no_optimize` 臂不是行为冻结（见 §1 BLK-03 附带），且其门双重空转。
4. 树内任何一层都不会因 P0 BLOCK 失败（驱动 `return 0` + 测试接受任一 verdict）。
5. 提交的 P0 证据相对代码已过期，且 06:17 链接的"正式报告"无法由树内代码复现。
6. plan:123 的三后端 parity 在 P0 完全缺席，而 P0 已被宣布完成。
7. P0-A 的探针字段清单缺项（见 D-18），使 plan:130"在首个失败 episode 内做二分 replay 定位导致塌缩的 optimizer update"这一失败分支缺少必要输入。

### 3.E P1 课程与能力门 — `non-conformant`

**结论**：plan §4.2 的数据模型改造基本落地 —— `EcologyBodyEpisodeLineage` 在训练与冻结评估两条路径上都带 `body_id + episode_id + layout_seed`；P1 mastery 是"同 body pickup→delivery 闭环的 per-layout 成功率"而非事件计数；held-out 行按 `(arm, capability, seed)` 只判一次并在 arm checkpoint digest 变化时失效重跑；五个臂从同一 initial checkpoint 分叉并重放同一条 sha256 冻结 schedule 且**无 early stop**；oracle/FixedRule/random 三个诊断基线结构上无法写 learned checkpoint 且复用字符级相同的 held-out seed 表达式；三个 loader 硬拒绝旧 schema。两个 bootstrap 也合规。**非一致性集中在验收定义**，不在管线。

| 条目 | 状态 | 说明 / 证据 |
|---|---|---|
| E-1 每条记录带 body/episode/layout_seed lineage | implemented | `ecology_curriculum.py:275-299,662-676,1537-1549`；无测试断言字段集 |
| E-2 schema 升版且旧 schema 硬拒绝 | implemented | 三处独立硬拒：curriculum artifact（`ecology_curriculum.py:47` + `ecology_checkpoint.py:111-114`）、P2 里的 P1 前置（`ecology_p2.py:535-540`）、progress journal（schema + 完整 config + schedule sha256） |
| E-3 Stage 0 near 不进任何正式能力门 | implemented | butter-NEAR 只出现在 `_fixed_schedule` 训练局；打分能力集只来自 `_evaluation_specs()`（NEAR 仅用于 forced-escape 门） |
| E-4 两个 bootstrap 只初始化身体状态 + 同步 body 侧 PI，不发布坐标/方位/动作标签 | implemented | `ecology_curriculum.py:479-497,555-577,971-972` | 
| E-5 `forced_approach` 随机几何 + 直线必然错过 | implemented | `ecology_curriculum.py:519-540`；测试只经验性复核 `> butter.radius`（弱于冻结的 1.38×）（`test_ecology_curriculum.py:107-121`） |
| E-6 Stage 1 medium/far 分开判定、同 body 闭环、多 layout seed | implemented | `ecology_p1.py:646-648,668-671,1176-1193` |
| E-7 Stage 2 forced escape 只判逃逸率+时延、须过 random floor、平局比 median | implemented | `ecology_p1.py:667-668,1286-1342`；逃逸门结果从不喂给任何 route-avoidance 能力 |
| E-8 Stage 4 neutral stick 只判闭环、contact 不计分 | implemented | `ecology_p1.py:650-654`；`EcologyP1LayoutResult` 根本没有 contact 字段 → 结构上无法进分 |
| E-9 held-out 必须有真实 beta switch 且非全 timeout 闭合 | implemented | `ecology_p1.py:1480-1497`（但见补充 3：被弱化为存在性） |
| E-10 五臂同分叉同 schedule | implemented | `ecology_p1.py:1012-1015,1049-1067,1132-1145` | 
| E-11 诊断基线不写 checkpoint、同 held-out 布局 | implemented | `ecology_p1.py:840-926`（裸控制器直接步进世界，函数内无 AntSession/无 checkpoint 导出） |
| E-12 early stop 的 estimand 冻结 | implemented（以"取消"方式满足） | P1 路径无 early stop，executed == max-budget，schedule sha256 入 journal；唯一停止是 `max_new_work_items` 抛 `EcologyP1ProgressPaused` |
| E-13 held-out 只判一次 | implemented | `ecology_p1.py:1170-1187,1234-1252`；重复行 raise；arm checkpoint digest 变则弃旧行 |
| E-14 schema 版本与 06 文档声明一致 | implemented | 代码 = `development.v25` / `progress.v21` / `curriculum.v7` / bundle `checkpoint.v4`，与 06:143 完全一致 → **过时的是 spec** |
| E-15 mastery 改为成功率 | **partial** | P1 层合规；**curriculum 层仍是 P1 前语义**：`_mastery_reached = pickups>=2 and deliveries>=1` 按 stage 累计，门的描述还写着 "event-sample mastery threshold"，而 schema 已 bump 到 v7 |
| E-16 六类指标分开记录 | **partial** | 六族字段都有，但两项退化：`encountered_food` 判据是 `<=4.4`（far tier 最大食物距离，非 `antenna_reach=0.9`）；路径效率一族未真正计算 |
| E-17 `local-valence-off` → `dense-local-shaping-off` 改名 | **partial** | 只在 P1/P2 的臂名元组里改了；curriculum 报告（v7）仍发 `valence_off_training` / `valence_off_metrics`，臂名字面量仍是 `"valence_off"`，session 杠杆仍叫 `ecology_local_valence_enabled` |
| E-18 报告 median **与 p90** escape latency | **partial** | 只算 median（在 random-floor 门的 observed 串里）；全仓库 ecology 实验内无 p90/percentile/quantile；原始 `escape_latencies` 已持久化 → 可事后算，但不是门的一部分 |
| E-19 冻结 P1 门槛（≥5 layout seed / 60% / 60% / ≤5%） | **partial** | 三个比例不可表达其他值（`__post_init__` raise），但**无测试触发那些 raise**；"≥5 layout seed"完全未冻（BLK-07） |
| E-20 `food_steering_alignment` 硬门 | **partial** | 门算术正确（每 body 一个 FOOD 探针、`target_aligned = left_turn > +θ and right_turn < -θ`、阈值 `ceil(n_ants*0.6)`、只读不回灌）；问题在探针几何（见下 E-20 复核）与探针世界参数（见补充 2） |
| E-21 三命名空间不相交 | **partial** | 分裂偏移保证同一数值 seed 在不同 split 产生不同场景，且 `__post_init__` 拒绝 validation/held-out 重叠；但 **P1 从不跑 validation**（`_curriculum_config` 把 validation 置空） |
| E-22 P1 PASS 定义机器可检 | **partial** | 有：各能力 mastery、`learned_not_worse_than_no_optimize`（覆盖 butter medium/far、heat route、neutral stick、composite）、三个 P0 家族门；缺：checkpoint roundtrip 门、第二次复跑（BLK-08 之外的 E-23） |
| E-23 Stage 3 heat-route 独立评估 | **contradicted** | 见 BLK-08 |
| E-24 Stage 5 composite 仅在 1–4 通过后解锁 | **missing** | 无任何解锁/排序逻辑（`src` 内检索 "unlock" 只命中 P2 前置 docstring）；composite 训练局无条件在冻结 schedule 内，`composite` 门与其他门并列无条件计算 |
| E-25 composite 的 harmful ticks 不高于 matched no-optimize | **missing** | P1 只施加绝对上限 `harmful_rate <= 0.05`（`ecology_p1.py:673-677`）；`harmful` 在 `ecology_p1.py` 内无任何 learned-vs-no_optimize 比较。**P2 恰好实现了这条缺失比较**（`heat_exposure_bounded`），所以是 P1 漏项 |

**阻塞与复核**：E-19/BLK-07 CONFIRMED；E-23/BLK-08 CONFIRMED；E-22（PASS 定义缺 roundtrip 与复跑）CONFIRMED —— `ECOLOGY_P1_GATE_NAMES` 16 项无 roundtrip 成员，最接近的 `replay_lineage` 只查 settlement/lineage/drop（`:1512-1527`）；能力存在于下一层（`_verify_checkpoint_archives`，`ecology_curriculum.py:1591-1620`，驱动 curriculum 的 `checkpoint_archive_roundtrip` 门）但 P1 的 import 列表没拉进来；P1 确实 import 了 `encode_/decode_agent_learning_checkpoint_archive`（`:17-20`），但只在可选 journal 里用（`:367`、`:337`），仅当 `progress_dir` 非 None 且真的 resume 时才走 → **默认单次 P1 运行产生零 roundtrip 证据**，即使带 journal，roundtrip 失败也只是抛异常而非具名门。第二次复跑：`run_ecology_p1` 只返回一个报告，脚本只跑一次，无双报告聚合器。

**E-20 REFUTED**（重要，避免误修）：审计初判"`food_steering_alignment` 探针落在拾取盘内、因此没测它要认证的盘外趋近转向"—— 几何算术是对的（探针黄油在 `(0.6, ±0.35)`，`ButterSource.radius=1.2` 默认，body 在原点，间距 `hypot(0.6,0.35)=0.6946 < 1.2` → `at_food=True`），但**推论不成立**：`at_food` 不是策略的输入。冻结的 19 通道里没有 `at_food` 也没有 `at_nest`（`sense_encode.py:54-60,86-127`），该字段只进诊断用 `AntStepRecord`（`ant_session.py:381`）与下游分析（`proofs/matched_control.py:284-292`）→ 处在盘内不改变策略看到的任何一个数。门读的也不是拾取结果，而是 turn 命令的符号（`ecology_probe.py:236-240`）。

**复核补充发现**：

1. **（阻塞）** plan:293 的 composite 合取项"harmful ticks 不高于 matched no-optimize"在 P1 未实装（= E-25），且 P2 已有该比较 → 解锁阶段比正式阶段少一个条件。
2. **（阻塞）探针世界与训练/评估世界的传感器几何不同**：探针建 `AntWorld(config=AntWorldConfig(seed=seed, step_size=0.4), ...)` 且不覆盖其它字段（`ecology_probe.py:165-171`），因此继承 `antenna_offset_deg=30.0, antenna_reach=0.6`（`ant_world.py:149-150`）；而每个 curriculum 世界都用 `antenna_offset_deg=45.0, antenna_reach=0.9`（`ecology_curriculum.py:453-458`）→ **16 个 P1 门里有 3 个在离分布状态上测 learned 策略**，且同一失配也门控着训练循环里的 guard。
3. **（一致性，非阻塞）** `temporal_non_timeout_closure` 把 plan:296 的"held-out 必须出现真实 beta switch，且不能全部由 timeout 关闭 segment"从**比例**弱化成**存在性**：`sum(switch_count) > 0 and sum(non_timeout_segment_closures) > 0`，在全部 learned held-out 布局上聚合（`ecology_p1.py:1481-1499`）→ 6 能力 × layouts × rounds × ants 里出现**一次**非 timeout 闭合即满足。

### 3.F P2 正式矩阵与晋级门 — `conformant-with-gaps`

**结论**：P2 自己拥有的部分对 plan §5 异常忠实 —— 正式预算（8 ants、latent 16、80/80/120、≥5 held-out layout、≥3 seed）是硬门而非散文；三个 P1 门槛在任何其他取值下都无法表达；按 `(training_seed, arm)` 分片且 journal 钉住 config/digest/schedule/P1 hash；聚合器对缺失、重复、不完整、preflight 单元一律拒绝，配对缺布局时**抛异常而非丢样本**；统计上最要紧的那一点是对的 —— 两层 bootstrap 先重采训练 seed，点估计是"每 seed 均值的均值"，复制单位是 seed 而不是 ant-tick，并对恰好四个预注册比较做 Holm 校正（random 被有意排除在比较之外）。**最大的洞是整个 23 门判决处于空转**（BLK-09）。

| 条目 | 状态 | 说明 / 证据 |
|---|---|---|
| F-1 正式预算硬门 | implemented | `ecology_p2.py:270-278,300-315,650-690,1854-1872`；小于最小值即 BLOCK |
| F-2 正式设备写入 config 且运行前校验 | implemented | `ecology_p2.py:313,1362-1370,637`；无测试覆盖 mismatch 路径 |
| F-3 seed 预注册、排序去重、失败不替换 | implemented | `ecology_p2.py:320-330,577-592,620-648,1337-1341`；held-out seed 是 `capability_index/layout_index` 对基 `5_000_011` 的纯函数 → 无法逐臂手挑 |
| F-4 ≥5 held-out seed、事后不删 seed、缺失不利向插补 | implemented | `ecology_p2.py:1666-1672,1767-1800,1893-1922`；行数不足即 raise |
| F-5 九臂齐备（+`cold` = 十个 arm spec） | implemented | `ecology_p2.py:118-190`；`shard_completeness` 要求每个训练 seed 上十个臂全在 → P2-C 消融不可跳过 |
| F-6 PE-off 是真实消融 | **本报告改判 partial** | F 簇原判 implemented，理由是两处独立消费点（`session.py:1703-1738`、`joint_loop/runtime.py:258-276`）；但同一 flag 还归零了 replay reward（BLK-02）→ 该臂差的不止具名机制 |
| F-7 random 不代理 PE/ETA 消融；存在受训 E2E-RL 臂 | implemented | `ecology_p2.py:197-211,2166-2190,867-915`；四个预注册配对是 learned vs {no_optimize, cold, pe_off, eta_off} |
| F-8 每个可学臂同一 initial checkpoint | implemented | `ecology_p2.py:1237-1262,1404-1409`（bootstrap runner 与 arm spec 无关）；无测试断言各臂初始指纹相同 |
| F-9 训练布局/预算/评估布局全臂匹配 | implemented | schedule 由 P1 builder 产出（测试钉等值）；E2E PPO 基线拿 `episodes=len(schedule)`、同 TRAIN 世界 |
| F-10 分片 + digest 同一性 | implemented | `ecology_p2.py:1044-1103,1836-1852,1893-1922,2541-2580`；journal 漂移即 "progress mismatch" |
| F-11 配对 + 层级统计（seed 为复制单位） | implemented | `ecology_p2.py:1705-1748,1767-1800,747-800` |
| F-12 多重比较校正 | implemented | Holm 逐步下降，显著性要求 `mean>0 and ci_low>0 and holm_p<0.05`；比较表在预注册 digest 内 → 事后不可裁剪 |
| F-13 primary endpoint 按计划顺序 | implemented | `ecology_p2.py:213-241,2360-2378`；八个端点顺序与 plan:393-405 一致（`heat_exposure_bounded` 折进端点 2，无害重排） |
| F-14 loader 拒绝 BLOCK bundle + v3/v2 schema 拒绝 | implemented | `ecology_checkpoint.py:201-245`（校验 manifest、artifact_kind v4、按冻结 gate 集重算 verdict、拒绝矛盾、BLOCK 抛 not promoted） |
| F-15 FixedRule/Canvas 不能冒充通过 | implemented | `app/runner.py:500-506`（硬编码 verdict=BLOCK）、`:210-241`、`:76-91`（默认 `formal_verdict=BLOCK`）；较弱路径见 BLK-09 末段 |
| F-16 preflight 不进最终统计 | implemented | preflight shard 打 `preflight=True` 且被 `shard_completeness` 计为不完整；另跑单布局确定性重放并对漂移失败（`:2468-2500`） |
| F-17 ETA-off 是真实 ETA 消融 | **partial** | 混淆 + 不完全（BLK-10） |
| F-18 每 shard 独立 manifest | **partial** | shard/confirmatory 都是裸 `_write_json`，无 `.manifest.json`、无 per-shard provenance、无 archive hash |
| F-19 代码/门槛变化使整批失效 | **partial** | digest 覆盖 config/arm spec/比较/端点与 gate **名**/capabilities/held-out seed/每 seed schedule digest/alpha/memory capacity；**不覆盖** `outcome_score` 权重（`:692-716`）、gate 逻辑、evaluation 内部 |
| F-20 报告 counts + proportions + effect size + bootstrap CI | **partial** | counts 与 bootstrap CI/p/Holm p 齐全；proportion 只隐式存在；未按名报告标准化 effect size |
| F-21 secondary 不能挽救 primary | **partial** | 构造上成立（verdict 需全门通过，无 secondary 进门）；但多数具名 secondary **不在报告里**（只剩 `escape_latencies`；路径效率、首次事件 tick、per-ant 方差、动作平滑度缺失） |
| F-22 plan 5.7 全部晋级条件编码 | **partial** | 23 门、全或无、门名漂移即 raise（`:2374-2377`）。缺口：① `fixed_rule_safety_floor` 在 `learned曝露 <= max(FixedRule曝露, 0.05)` 时通过 → learned 可严格不如 FixedRule 安全仍过门（`:2133-2165`）；② plan 5.7 的"在预声明学习指标上显示自身优势"无门；③ corruption rollback 无门 |
| F-23 串行 P1 门在代码中强制 | **partial**（机制真实，凭证不可信） | 每个 shard 与 preflight 都拒绝无 P1 PASS 报告即开跑，聚合器亦然（`ecology_p2.py:258-298` 有测试）；但凭证是未经 manifest 校验、未比对 config 的裸 JSON（BLK-07/BLK-11） |
| F-24 CPU parity smoke | **missing** | preflight 记 wall clock/archive size/单布局确定性重放，但无第二设备 parity；`scripts/` 与 ant 包内检索 `parity` 只命中无关的 learned-shadow 项 |
| F-25 loader 准入绑定 P2 判决 | **missing** | BLK-09 |

**四条阻塞全部 CONFIRMED**（BLK-09/10/11 详见 §1）。**复核补充发现**：

1. **（阻塞）无门校验消融杠杆真的生效** → spec:380 的 kill condition"策略参数未按预期改变"在正式通路无人看守（详见 BLK-10）。
2. **（阻塞）eta_off 同时是不完全消融**（详见 BLK-10）。
3. **（阻塞）random 臂烧满预算却不进任何门或比较** → P2 无 random floor 检查，而 P1 有（详见 BLK-10 末）。
4. **（一致性）晋级通路 schema 漂移**：spec:361-362 冻结 loader 面对的 `digital-ant-ecology-checkpoint.v4` / `digital-ant-ecology-curriculum.v3`，而代码的 curriculum 报告 schema —— 也就是 `_validated_report_verdict` 在放行前要求的那个 —— 已是 `digital-ant-ecology-curriculum.v7`（`ecology_curriculum.py:47` → `ecology_checkpoint.py:112-113`）；06:20 只承认 v3→v6。**冻结后被改了四次而 spec 未修订**，且这正是 BLK-09 指出的唯一 demo 加载门。
5. **（完整性）`provenance_clean` 证明的是错误的进程**：`aggregate_ecology_p2_shards(..., worktree_clean: bool)`（`:1813-1817`）直接喂给门（`:2353-2360`），只有 `scripts/run_ant_ecology_p2.py:144` 恰好从真实 provenance 推导；任何其他进程内调用方都可以直接断言干净，而该门无法区分"聚合树干净"与"各 shard 训练树干净"。

### 3.G 工程门：冻结评估、指纹、replay、archive、容量、provenance — `conformant-with-gaps`

**结论**：这一簇承重的机制是真代码而非文档声明。冻结评估确实同时设 `optimize=False` 与 `learning_enabled=False`；内核调度器在任何 cycle/SSL/RL/writeback/rare-heavy 之前硬返回 `frozen-evidence-only`；同一个 flag 在参数 store 层关掉 temporal fast fit、action-family topology/outcome/cache 与 match-head 写入且有真实测试。指纹分类正确：`policy_fingerprint` 只组合 optimizer 拥有的面（update step + critic + causal head），有意不含 track weights 与 persistence，因此 reflection-prior 或 SSL 的 track-weight 写入无法冒充 RL 持久化；PE 驱动的 turn-local mixture 被排除在 temporal-learning 指纹之外。settlement 覆盖率分母严格是 `captured - pending`；settlement/lineage ≥0.99 且 drop==0 在三层报告里都是具名门。archive roundtrip + 篡改回滚、双槽 `.vzac` fsync/rename、SHA256、resume 重校验、已完成不重跑、陈旧评估失效、确定性容量淘汰全部存在且多数有测试。**缺口集中在 artifact 卫生与执行 provenance**（BLK-12/13）。

| 条目 | 状态 | 说明 / 证据 |
|---|---|---|
| G-1 冻结评估同时设两个 flag | implemented | `ecology_curriculum.py:1417-1418,1609-1610,620-621`；`colony_runner.py:57-60` |
| G-2 `frozen-evidence-only` 硬边界（含 rare-heavy 关闭、仍允许恢复 checkpoint 与推理 telemetry） | implemented | `joint_loop/runtime.py:1706-1732`（早返回先于 `_pe_rare_heavy_due`，返回 `rare_heavy_review_recommended=False`）；`joint_learning_enabled=False` 还强制 no-op schedule 并禁止自定义 joint_schedule | 
| G-3 同一 flag 关闭 fast fit / action-family / match-head 写入 | implemented | `session.py:628-631,665-668`；`interface.py:1520-1521,1691-1707`（`observe_family_outcome_feedback` 在触碰 match weights 前返回 False）；测试 `test_temporal_contracts.py:541-579` |
| G-4 三类指纹分离（policy / temporal / memory） | implemented | `session.py:1083-1111,1178-1184`；`checkpoint_archive.py:45-47,86-97`（四个独立 sha256 + 聚合） |
| G-5 `policy_fingerprint` 只覆盖 optimizer 拥有面 | implemented | `session.py:1083-1090`；`sandbox.py:772-806`；`interface.py:615-666`（track_weights 与 persistence 有意缺席） |
| G-6 PE turn-local mixture 不计入 temporal-learning 指纹 | implemented | `interface.py:607-666`（`learning_parameter_fingerprint` 有意省略 `temporal_weights`） |
| G-7 每个 held-out 布局比对前后指纹，任一 body 漂移即阻断 | implemented | `ecology_curriculum.py:1436-1442,1523-1536,1946-1952`；`ecology_p1.py:1500-1514` |
| G-8 settlement 分母 = captured − pending | implemented | `ecology_curriculum.py:1443-1446,1465-1472`；`joint_loop/runtime.py:291-296,310-316`（pending 是逐 sandbox 的 0/1 指示量，与 drop_reasons 结构分离） |
| G-9 lineage/settlement ≥0.99、drop=0 作为门 | implemented | 三层门 + 门名元组冻结（漂移 raise）；`eligible_captures==0` → 覆盖率 0.0（fail closed） |
| G-10 archive roundtrip + 事务回滚且失败后指纹相等 | implemented | `ecology_curriculum.py:1591-1641`（翻转 archive 末字节、要求 `AgentLearningArchiveError`、要求失败后指纹等于失败前）；`colony_runner.py:134-181` |
| G-11 双槽 journal fsync+rename 后才原子推进 | implemented | `ecology_p2.py:1160-1190`；`ecology_p1.py:219-236,373-382`（slot = `completed_episodes % 2`，新局必写非 live 槽） |
| G-12 resume 重校验 config/schedule digest/SHA256/latent+ant count 并拒绝冲突 | implemented | `ecology_p2.py:1067-1126,1414-1428`；`ecology_p1.py:274-311` |
| G-13 已完成不重跑、部分臂只跑缺失后缀 | implemented | `ecology_p2.py:1504-1509,1583-1591,1673-1681` |
| G-14 训练 checkpoint 变化则评估失效 | implemented | `ecology_p2.py:1569-1581,1511-1516`（`policy_digest` 不等即清空 layout_results 全量重评） |
| G-15 确定性 `enforce_artifact_capacity(8192)`，保 CMS learned state | implemented | `memory/store.py:306-350`；`memory/artifacts.py:71-88`；排序键 `(stratum_rank, strength, last_accessed_ms, created_at_ms, entry_id)` 完全确定；`delete_entry` 同步清 pending_promotions/decays 与派生索引 |
| G-16 `joint_learning_enabled=False` 关闭 writeback 路径 | **partial** | `apply_reflection_prior_update` 的结构写回未被门控（BLK-13） |
| G-17 pending 不得掩盖真实 drop reason | **partial** | 发布的 `RuntimeReplayReport` 只带 `drop_reasons[-20:]`，而消费者用 `len(drop_reasons)` 推 drop **计数** → `drop==0` 门不受影响（截断不能把非零变零），但归档计数在每 body 每 episode 20 条处饱和 |
| G-18 archive compatibility 绑定 sense schema + input dim | **partial** | promotion bundle 四项全绑；**P1/P2 resume journal 只绑五项、缺 sense_schema 与 input_dim**（BLK-12） |
| G-19 resume 拒绝冲突而非静默适配 | **partial** | memory-capacity 兼容字段是条件绑定（`include_memory_capacity = ("checkpoint_memory_entry_capacity" in state)`）→ 缺该键的旧 state 会**降级**兼容元组而不是失败 |
| G-20 每次运行记录完整 provenance | **partial** | `AntRunProvenance` 无 device 字段（只有 platform）；training seed 与 layout seed 塌成一个 `seed_schedule`；P1/P2 报告完全无 provenance（BLK-12） |
| G-21 不完整结果不得与完整结果合并 | **partial** | `shard_completeness` 对缺失/不完整/preflight/重复单元失败，digest 门对 config/schema 不匹配失败 → 不完整矩阵到不了 PASS；但行折叠循环（`:1926-1929`）会吞入**所有** shard 的 layout 行 |
| G-22 transient journal 落在被忽略的 `.partials` | **partial** | 只有通用 `SeedPartialStore` 硬编码 `results_root/.partials/<stage>/<fingerprint>`；P1/P2 的 `--progress-dir` 任意（只校验在仓库根内），放 `.partials` 是文档约定而非代码约束 |
| G-23 CPU 确定性路径 + 一次 MPS/CUDA parity smoke | **partial** | 代码里唯一确定性检查是 P2 preflight 的重放，且只重放 `fixed_rule`（非学习臂）单布局 → 不证明 learned CPU 路径确定；设备处理只有 env 读取 + P2 的 config 比对 |
| G-24 append-only trainlog + 崩溃恢复保 lineage | **partial** | stage checkpoint + lineage 是真的（`state.json` 整文件原子覆盖，带 `completed_training_episodes` / `training_complete` / `policy_digest` / `layout_results`，resume 拒绝计数与标志不一致）→ 计数不会静默重启合并；但**不是 append-only 日志** |
| G-25 每个产物新文件名、不覆盖既有 BLOCK artifact | **missing** | BLK-12 |
| G-26 报告 schema 与冻结 spec 一致 | **contradicted** | bundle kind 仍匹配 `checkpoint.v4`，但 curriculum 报告 schema 已是 v7 而 spec 钉 v3、P1 已是 development.v25；loader 只接受 v7（更严，历史 artifact 确实被拒），但**冻结 spec 已过期** |

**四条阻塞全部 CONFIRMED**（BLK-12 三条 + BLK-13）。**复核补充发现**：

1. **（阻塞）** `provenance_clean` 门证明的是聚合进程而非各 shard（与 F 簇补充 5 同源，此处从 shard 字段侧独立确认：`EcologyP2ShardReport` 字段 `:395-417` 与 `shard_report_from_dict` `:2549-2640` 都没有 git SHA/dirty 可读回）。
2. **（一致性，中）** P1 驱动对 BLOCK 退出 0，而 P2 子命令返回非零 → 按退出码编排的 CI 会把 P1 BLOCK 当成功。缓解：`load_p1_prerequisite` 会再读报告并拒绝非 PASS，所以这是自动化面偏差而非门绕过。
3. **（完整性，部分不可静态判定）** 报告/产物写入不是严格 canonical JSON 且可能写出非有限常量：`provenance.atomic_write_json`（`:91-97`）、`stable_json_digest`（`:76-84`）、`_stable_json_bytes`（`ecology_p1.py:203-212`，被 `ecology_p2.py:70` 导入）与两个驱动写盘全部用默认 `allow_nan=True`（其中两处还带 `default=str`）；archive lane 相反是严格的（`canonical_json.py` 的 `allow_nan=False` + 拒绝非有限常量与重复键）。
4. **（记录以免重提，非缺陷）** `shard_digests` 算在重解码后的 dataclass 上（`:2388-2396`）而非磁盘字节；`shard_report_from_dict` 重建了每个声明字段（含 `layout_results` / `probe_summary` / `wall_clock_seconds` / `description`），所以是完整的语义绑定，唯一残余是事后无法对文件复验 —— 已被 F-18 的 manifest 缺失涵盖。

### 3.H 可行性 — `non-conformant`

本簇的 6 个条目对应 §2 的六个问题，此处只给状态与一句话依据，详述见 §2。

| 条目 | 状态 | 一句话依据 |
|---|---|---|
| H-1 行程预算（far 往返在 episode 预算内可行） | **partial** | held-out 40 拍不是问题（两个诊断基线都 4/4 达成 far）；**far 训练局 24 拍在实测速度下才是缺陷**（§2.3） |
| H-2 信用视野（distal 转向的信用能否到 head；长程回巢是否依赖未接线的记忆） | implemented | 本设计**不需要**长程信用：PI 每拍发布 home bearing（`sense_encode.py:103-105`），ecology payoff 逐拍密集（`ant_session.py:656-663`），16 步 GAE segment 比实际所需视野更长。代价是 head 只读零历史 state（`interface.py:753-810`）→ 没有任何依赖历史的搜索行为，也没有任何 memory owner 接到 steering |
| H-3 单轴多路复用与绝对对齐判据 | **contradicted**（强版本被推翻，约束成立） | §2.2 |
| H-4 near 假阳性 | **contradicted**（比 spec 承认的更糟） | §2.4 |
| H-5 P2 算力 | **partial** | 可运行但是多日战役，约 1.1M 内核 step / 77 CPU·h（§2.6） |
| H-6 五个证伪目标的端点覆盖 | **partial** | R5-R6 无端点；R-PE 端点被 BLK-01/02 污染；R3-R4 端点存在但证据薄弱（§2.5） |

**本簇最重要的两条结论（FEA-01 / FEA-02）见 §2.1 / §2.3；两条被推翻的强版本见 §4。**

---

## 4. 已被对抗性复核推翻的疑点（勿按这些去改代码）

| 被推翻的判断 | 为什么不成立 |
|---|---|
| `forced_return` / `forced_approach` 用 `navigator.sync_to` 违反"navigator 从不读真值位置" | spec:333-334 明确授权这两个 bootstrap"同步 body-side PI"；spec:48-49 限制的是**claim 范围**（用 sync_to 构造的 lane 不能承载 AntBot/Ardin claim），而 AntBot claim 走的是 `phase0.py:104-163` 那条不含 sync_to 的通路；且 curriculum 里根本没有 PI 误差/home-vector 指标 |
| eta_off 臂会把 `n_input` 绑成 `n_z` 并截断 ecology 尾部通道 | 注入的是 `LearnedLiteTemporalPolicy`，它从不触碰 `n_input` 或 ndim encoder（`_residual_signature` 返回 3 元组，`_project_to_ndim` tile 的是 3 维向量）；全仓库无人注入 `FullLearnedTemporalPolicy` |
| 有害热区内每拍 `-0.4` 是未授权的第四个 valence 项 | spec §5.3（257-272）显式冻结了密集 local valence 通道并要求"升温/有害暴露为负"；被授权的 `0.45/0.45/0.7·tanh` 三项比它更密更大。副产物：spec §4 与 §5.3 关于木棍 payoff 互相矛盾，代码遵循的是正确的 §5.3 |
| replay 侧逐维 beta 会把零参数 head 的转向泄漏放回训练信号 | 可证明等价：`z_tilde = z_candidate`（最终 candidate）、blend 是 `[0,1]` 内的凸组合故外层 clamp 永不触界 → 重建的 `beta_i` 精确等于 live 的 `effective_gate[i]`，而后者在 exclusive steering 下已 pair-mean 投影 → `beta_left == beta_right` 恒成立 |
| 镜像等变默认关闭、调用方忘传 schema 就静默丢失 | `AntSessionConfig.sense_schema` 默认 V1，而 spec:311 要求非 ecology/V1 profile **不**带镜像配置 → 那些调用点合规；所有真实 ECOLOGY_V2 内核会话都双传了 schema |
| `food_steering_alignment` 探针落在拾取盘内，因此测不到盘外趋近转向 | `at_food` 不在 19 维感知里（只进诊断记录），处在盘内不改变策略看到的任何一个数；门读的是 turn 命令符号，不是拾取结果 |
| far 往返在 24 拍内几何上不可能 | plant 满速 24 拍 = 9.43 单位，最坏需求 5.60 单位；预注册 oracle 约 23 拍完成。成立的是弱版本：**当前策略实测速度**下采不到（§2.3） |
| 四个配对消融都被钉在恒零端点上，P2 必然返回 no effect | `outcome_score` 是带 0.02 分辨率的部分信用连续量，`forced_escape` 为每臂贡献 `heat_escapes×0.25`，各臂在 pickups 上已分离；`food_steering_alignment` / `carrying_home_action_alignment` 本身也是连续比例端点 |

---

## 5. 文档 ↔ 代码不一致清单

| # | 不一致 | 现状 | 建议 |
|---|---|---|---|
| 1 | spec:361-362 钉 `digital-ant-ecology-curriculum.v3`，代码是 **v7**（loader 只收 v7） | 冻结准入契约在冻结后改了四次而 spec 未修订；06:20 只承认 v3→v6 | 修 spec，并在 spec 里写明"该版本号由 `ecology_curriculum.py:47` 单点拥有" |
| 2 | spec §4:85-88（木棍碰撞产生负 payoff）vs §5.3:258-259/268-269（contact 无任何 payoff/valence） | 代码遵循 §5.3（正确） | 删除 §4 里"木棍真实碰撞…产生负 payoff"的措辞 |
| 3 | spec:265-267 指定携食期用 owner 发布的 home-pheromone 信号 | 代码用方向无关的 PI 进度（06:47 记录的正确修复）；`local_home_signal_before/after` 成死字段 | 更新 spec 文本；或保留字段但注明只作诊断 |
| 4 | `code_gain=4.0`（冻结转向几何上的 4× 增益） | 在 spec / plan 中零命中，且是 per-run 可变字段且无测试钉住 | 写入 spec §3 并加测试把它钉在 4.0（matched arms 必须一致） |
| 5 | P0 报告 schema | plan 命名 v1、代码发 v2、唯一 artifact 是 v1、驱动默认路径的 v2 文件不存在 | 重跑 P0 产出 v2 artifact，或把代码退回 v1 并说明 |
| 6 | 06:12 把 plan:193 的"所有 learned-owner"窄化为"policy/temporal-learning owner" | 文档口径与 plan 不一致，恰好掩盖 BLK-04 | 修 06 的措辞，或修 gate（推荐后者） |
| 7 | 06:10 "19/19 通过" 与 06:14 "18 个候选更新被回滚" | 是同一现象的两种叙述（BLK-03） | 06 里合并成一句诚实表述 |
| 8 | 拾取半径两个常数并存：`ant_world.py:153` 的 `food_pickup_radius=1.2` 与 `world_objects.py:143` 的 `ButterSource.radius=1.1`（spec:340 写 1.1） | 影响 far 余量约 0.2 单位与 forced_approach 的 1.38× 判据基准 | 统一到单一 owner 常数 |
| 9 | spec:311 要求非 ecology/V1 profile 不带镜像配置 | `sense_mirror_transform` 接受 V1 并会产出 14 维镜像；只有 `app/runner.py:191-195` 防御性传 None | 让 `sense_mirror_transform` 对 V1 直接拒绝，或在 spec 里放宽措辞 |
| 10 | **对上了，记录以免重复排查**：`development.v25` / `progress.v21` / `curriculum.v7` / `checkpoint.v4` | 代码与 06:143 完全一致 | 无需动作 |

---

## 6. 建议修复顺序与验收方式

顺序按**依赖**而非严重度排：前一步不修，后一步的证据没有意义。

| 步 | 修什么 | 验收（可测） |
|---|---|---|
| **1** | 奖励接缝（BLK-01 + BLK-02）。`measurement is None` 时 reward 置 0（或实装 spec:95-96 的 typed reward eligibility）；把 PE-off 与 replay reward 解耦；把 optimizer 实际消费的 reward 及分量导出到 `AntStepRecord` | 新增测试：① 无 measurement 的拍 → `ZTransition.reward == 0`；② PE-off 臂保留环境 payoff 而只失去 PE 驱动与 switch pressure；③ `nonzero_reward_steps` 改为统计 optimizer 消费的 reward 且与 milestone 计数可对账。此外把 `VZ_PE_EVALUATION_DECOUPLED` 在 ant profile 显式置真并加测试 |
| **2** | P0（BLK-03/04/05/06）。gate 消费 rollback 证据；P0-C 建立 allow-list 并逐拍 gate 全部 8 个 owner；按 exclusive steering 重推导冷启门（冷启只验 reachability，转向移到训练后）；实装 P0-B 的 transition protocol + 负对照 + switch-rate 上限 + timeout 占比 + segment-credit parity；把四个 P0 阈值写进测试 | 从空目录重跑 `audit_ant_ecology_mechanisms.py` 得到**真** PASS 或诚实 BLOCK；驱动对 BLOCK 返回非零；新增三个 per-mechanism 测试文件（plan:200-206） |
| **3** | 决定 train/eval 动力学一致性（FEA-01）。两条路线：(a) 评估保留一个受控、预注册的随机分量并写进 spec；(b) 提高 head 在确定性均值下的转向权威（含通道仲裁或时间分离，见 §2.2） | 冻结评估下 `measure_ant_food_steering_gain` 的 learned `min_authority` 显著 >1e-3 **且** `authority/baseline > 1`；同一 checkpoint 的 medium 冻结重放出现非零 delivery。这一步不通过就不要再投 P1 全量算力 |
| **4** | P1 验收定义（BLK-07/08 + E-24/E-25 + E-18）。冻结 `layouts_per_tier >= 5` 与 `n_ants >= 4`；给 `heat_route_foraging` 接上 `HEAT_ROUTE_AVOIDANCE` 独立场景；加 composite 解锁排序、composite vs no-optimize 曝露比较、checkpoint roundtrip 门、第二次复跑聚合、p90 escape latency；探针世界的 `antenna_offset_deg/reach` 与 curriculum 世界对齐 | `EcologyP1Config.__post_init__` 对小于正式预算的配置 raise 并有测试；`_evaluation_specs()` 返回 6 个**不同**的 scenario/tier；新增双报告聚合器 |
| **5** | P2（BLK-09/10/11 + F-24）。把 loader 准入绑定到 P2 判决（或让 P2 PASS 产出可加载 bundle 并让 loader 只认它）；ETA-off 改用 spec:399-401 的构造（冻结 learned-lite + `ssl_interval=rl_interval=0`）；加"消融杠杆生效"门与 random floor 门；shard 记 commit/tree hash 并走 `write_ant_artifact_bundle`；把 `outcome_score` 权重与 gate 逻辑纳入预注册 digest；补 CPU parity smoke | 测试：① 一个 curriculum PASS 但无 P2 证据的 bundle **被 loader 拒绝**；② 改 `outcome_score` 任一权重使预注册 digest 变化；③ eta_off 与 learned 的差异仅落在 temporal 写回面 |
| **6** | artifact 卫生（BLK-12/13 + G-17/G-19/G-22/G-24）。写入前拒绝覆盖既有 artifact（尤其 BLOCK）；P1/P2 报告走 bundle + manifest + provenance；`AntRunProvenance` 增 device 并拆开 training/layout seed；resume compatibility 补 `sense_schema` + `input_dim`；给 `apply_reflection_prior_update` 的结构写回加 `_learning_writes_enabled` 守卫；报告写入改 `allow_nan=False` | 测试：① 同名再写即抛错；② P1 报告带 provenance 且有 `.manifest.json` 可校验；③ 14 维 v1 archive resume 进 19 维 ecology 运行**被兼容元组拒绝**；④ `learning_enabled=False` 下调用 `apply_reflection_prior_update` 不改任何 `learning_parameter_fingerprint` |

**一条元建议**：本次 146 条中有 **38 条**的"测试锚点"列为空。P0 的四个阈值、archive compatibility 元组、镜像映射的四个通道、Outcome→PE→reward 整条接缝、`policy_fingerprint` 组成、legacy 控制面隔离都属于"实现正确但无人看守"。plan §2.1 的精神是"阈值先入 schema/test，再读结果"——建议把这 38 条按 §6 的步骤逐步补测试锚点，否则每次重构都要重跑一遍本审计。

---

## 7. 附录：审计元数据

- **运行方式**：16 个并行/流水的静态审计 agent（8 个契约簇 × 1 审计 + 1 对抗性复核），复核者独立读码尝试**推翻**审计者的阻塞判断，并补报审计者漏项。
- **规模**：约 2.33M subagent token、1,105 次工具调用、约 40 分钟墙钟。
- **簇级结论**：A `conformant-with-gaps`｜B `non-conformant`｜C `conformant-with-gaps`｜D `non-conformant`｜E `non-conformant`｜F `conformant-with-gaps`｜G `conformant-with-gaps`｜H `non-conformant`。
- **冲突处理规则**：审计者与复核者结论冲突时，本文档以能给出**可复核代码位置**的一方为准，并在正文显式记录两方口径（例如 §3.B 的 B-3、§3.F 的 F-6、§2.1 的 P2 round 数修正）。
- **未运行任何代码**：所有判断来自读码、已提交 artifact 的内容核对、代码常数的算术推演。凡需运行时测量的条目标记为 `undecidable-static`，未计入通过或失败。
- **两条最有后果的判断由主审自查复核**：BLK-01（fallback reward 链，逐跳读了 `final_wiring.py:1523` → `error.py:694-698` → `sandbox.py:2204-2212` → `sense_encode.py:194-203`）与 BLK-02（`ecology_curriculum.py:618` → `session.py:935-938` → `sandbox.py:2194-2212`），并额外读了 `ant_session.py:609-690` 以确定 fallback 的**精确触发面**（learned 臂窄、`dense-local-shaping-off` 臂全覆盖）。
- **本文档的定位**：静态一致性证据，不替代运行时证据。它能证明"设计条款在代码里存在/缺失/被矛盾"，不能证明任何能力结论；P0/P1/P2 的 verdict 仍必须由各自的正式运行产出。


