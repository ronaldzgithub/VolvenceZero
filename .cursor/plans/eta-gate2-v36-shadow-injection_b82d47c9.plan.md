# Gate 2 下一步实现计划：v36 selector SHADOW 注入观测收敛包

> 状态：计划（未开始实现）。前置：v35 已通过（`causal-supported`、
> `promotion_allowed=true`，见
> [`eta-gate2-v35-selector-null_e4a91f27.plan.md`](eta-gate2-v35-selector-null_e4a91f27.plan.md) §8）。
> 本计划对应 v35 计划 §4.2 的第 1、2 步：SHADOW 注入观测 + 多 seed。
>
> 对应债务：`docs/known-debts.md` #92 Gate 2。
> 对应 spec：`docs/specs/evidence_program.md`（新增 v36 小节）、
> `docs/specs/temporal-abstraction.md`（selector artifact 契约）。

---

## 1. v36 要回答的新问题

v35 证明的是**开环、逐前缀**的条件化动作价值：每个前缀独立注入一个候选控制、
独立测量，selector 选出的动作在独立 audit 上超过置换零假设。它没有回答注入
真正上线（哪怕 SHADOW）时的两个新问题：

1. **闭环组合性**：沿一条 route 逐步注入 selector 选择的控制（步 1..i 的注入
   在评分步 i+1 时保持生效），每步的条件化价值是否仍然成立？前缀 i 的注入会
   改变前缀 i+1 的隐态上下文，selector 的 state features 也随之变化——开环
   证据不保证闭环不出现干涉或漂移累积。
2. **分布一致性（无漂移）**：SHADOW 轨迹上逐步 selected credit 的分布是否与
   v35 evidence run 的开环分布一致？若闭环下 selected credit 系统性衰减或
   翻负，说明开环价值不可组合，SHADOW→live 的路径直接关闭。

回答完这两个问题，`selector_injection_allowed` 才能从「SHADOW 级许可」变成
「SHADOW 已观测、无漂移」，为后续 live wiring 决策提供证据。

## 2. 范围与不做的事

- **做**：selector artifact 序列化 + 冻结 provenance；evidence harness 内新增
  闭环 SHADOW 注入 arm；v36 fresh 分区；多 seed（≥3）；预注册轨迹级门。
- **不做**（后续独立包，见 §7）：
  - 线上 session/joint_loop 的 SHADOW wiring（CP-LSS 模式接入真实对话轨迹）
    ——那是 consumer 切换，按收敛包纪律与本包（owner/契约）分离；
  - live injection（`counterfactual_action_selector_live_injection` 保持
    `disabled`，本包不触碰）；
  - Gate 2 longitudinal 层（≥500 real-trace）；
  - 更大基底复测。

## 3. 实现阶段

### 阶段 A：selector artifact 序列化与冻结（owner：`vz-temporal`）

`ResidualActionSelectorArtifact` 已是纯数据 frozen dataclass 且带
`model_fingerprint`，但只存在于单次 run 的进程内。v36 需要「v35 拟合的
selector 原样冻结、跨 run 复用」：

1. 在 `packages/vz-temporal/src/volvence_zero/internal_rl/counterfactual_selector.py`
   增加 `selector_artifact_to_payload(artifact) -> dict` /
   `selector_artifact_from_payload(payload) -> ResidualActionSelectorArtifact`：
   纯 JSON 可序列化、round-trip 后 `model_fingerprint` 逐位一致（fingerprint
   在 from_payload 时重算并与 payload 声明值比对，不一致 fail loudly）。
2. evidence bundle 新增文件 `selector_artifact.json`（进
   `ETA_GATE2_REQUIRED_FILES`），记录 payload + fit split + basis fingerprint
   （selector 依赖 v34 basis，两个指纹必须绑定出现）。
3. 单元测试：round-trip 确定性、维度/指纹失配拒绝、与 v35 run 内拟合结果
   数值一致（用小型合成 examples）。

约束：不改 `fit_residual_action_selector` / `select_counterfactual_actions`
的任何数值行为；本阶段纯增量。

### 阶段 B：闭环 SHADOW 注入 arm（owner：`vz-runtime` evidence harness）

在 `eta_proof_benchmark.py` 的 counterfactual 评测旁新增一个预注册 arm
`shadow-closed-loop`，与现有开环 grid 共用同一 runtime/basis/环境 owner：

1. **闭环 rollout 协议**：对每条 route 按步推进；每个 scoreable 前缀上
   (a) 从当前快照提取 selector state features（复用现有
   `residual_action_state_vector`，不新建第二实现）；
   (b) 用冻结 artifact 预测并选 top-1 候选；
   (c) 将该控制加入「已提交控制集」，此后所有前缀的 forward 都保持步 1..i
   的已提交控制生效（实现上即 applied_control 序列逐步累积）；
   (d) 用环境 owner 的 `measure_realized_continuation_outcome` 测量该步
   realized NLL delta（相对同前缀 zero-control 基线，基线同样在「无任何
   注入」的干净轨迹上测量——两条轨迹分别完整跑完，不混合）。
2. **对照 arm**（同一预注册协议下并跑）：
   - `zero-control` 闭环轨迹（基线）；
   - `permutation-null` 闭环轨迹：每步从同一候选集均匀确定性轮换选择
     （种子固定），回答「闭环增益是否只是任意注入的副产物」。
3. **落盘**：新 jsonl（每步一行：split/route/step、selected index、
   predicted value、realized delta、累积控制数、state features 指纹），
   进 evidence bundle。
4. **零副作用断言**：SHADOW arm 运行前后 runtime descriptor、basis
   fingerprint、selector fingerprint 不变；SHADOW arm 不写回任何
   policy/memory 状态（复用 CP-LSS「SHADOW must be side-effect free」的
   断言风格，见 `learned_shadow_evidence.py`）。

### 阶段 C：v36 fresh 分区与 manifest 预注册（owner：`vz-runtime` evidence）

v35 的 fresh validation/confirmation 已被观察一次并用于通过判定，按污染
纪律不得再作 v36 的一次性检验分区：

1. 在 `eta_proof_benchmark.py` 新增 4 条 `validation-v36-*` fresh routes +
   4 条 `confirmation-v36-*` locked routes（词汇新鲜度契约测试扩展到与
   v35 全部语料不相交）；v35 fresh 分区降级为 development 诊断。
2. `eta_gate2_residual_evidence.py`：schema 升 `eta-gate2-residual-causal.v36`；
   manifest case groups 新增 `shadow_closed_loop_arm`、
   `selector_artifact_fingerprint`、v36 route ids、
   `superseded_validation_route_ids` 追加 v35 分区；
   `counterfactual_action_selector_live_injection` 维持 `disabled`。
3. **seed 升级**：`--seeds 3`（v35 计划 §4.2 第 2 步并入本包；ci-smoke
   tier 不变，先不放宽 `--max-prefix-steps`，轨迹长度扩展留给 runtime
   SHADOW 包，避免一包改两个变量）。

### 阶段 D：预注册门与 verdict（owner：`vz-runtime` evidence）

正式门 `shadow-closed-loop-v1`（全部满足才产 `shadow_observation_passed=true`）：

1. **闭环增益门**：train / fresh v36 validation / locked v36 confirmation
   三分区，闭环轨迹级 realized NLL 总改进（selector arm − zero arm）
   `>= 1e-6`，且逐 seed 方向一致（3/3 seed 为正）。
2. **闭环超零假设门**：三分区 selector arm 轨迹级改进超过 permutation-null
   arm `>= 1e-6`。
3. **无漂移门**：三分区闭环逐步 selected realized delta 的均值符号与 v35
   开环 audit 均值符号一致（均为正）；「前半程 vs 后半程」步位置分解只作
   诊断落盘，不作门槛（避免小样本过度预注册）。

verdict 语义：本门**不改变** v35 的 `promotion_allowed`（causal 层结论
不回溯）；新增独立字段 `shadow_observation_passed`，为下一包（runtime
SHADOW wiring）的准入条件。任一门失败 → `shadow_observation_passed=false`，
live-wiring 路径冻结，按 §6 处理。

## 4. 涉及文件（预计 7 个关键文件，符合 3–8 约束）

| 文件 | 改动 |
|---|---|
| `packages/vz-temporal/src/volvence_zero/internal_rl/counterfactual_selector.py` | 阶段 A：payload 序列化 + 指纹校验 |
| `packages/vz-temporal/src/volvence_zero/internal_rl/__init__.py` | 导出新函数 |
| `packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py` | 阶段 B 闭环 arm + 阶段 C v36 routes |
| `packages/vz-runtime/src/volvence_zero/agent/eta_gate2_residual_evidence.py` | 阶段 C/D：schema v36、manifest、门、verdict、bundle 文件 |
| `packages/vz-temporal/tests/`（新增 selector 序列化测试） | 阶段 A 测试 |
| `packages/vz-runtime/tests/test_eta_residual_causal_controls.py` | v36 契约 + 防污染 + 闭环门测试 |
| `docs/specs/evidence_program.md` | v36 小节（实现前预注册段先行） |

基底层（`vz-substrate`）零改动：闭环注入只是 applied_control 的组合使用，
`install_control_basis` / residual backend 接口不变——若实现中发现 backend
不支持多步累积控制，须停下来单独评估，不得在本包内顺手改基底接口。

## 5. 执行顺序与验证

1. 阶段 A → `pytest packages/vz-temporal/tests/test_counterfactual_selector*.py`
2. 阶段 B/C（先写 v36 预注册段进 `evidence_program.md`，再动代码）→
   `pytest packages/vz-runtime/tests/test_eta_residual_causal_controls.py`
3. `ruff check` 改动路径
4. GO/NO-GO 探针（可选但推荐）：单 seed、仅 train split 跑闭环 arm，确认
   闭环 selected delta 不系统性翻负再触发全量——若探针已翻负，直接记录
   NO-GO，不烧全量 run
5. 全量 run：3 seeds、CPU、v36 语料（预计 36 min × 3 ≈ 2 小时）
6. 结果无论通过与否回写 known-debts / evidence spec / 本计划 §8（预留）

## 6. 失败路径与 kill condition（预注册）

- **闭环增益门失败但 v35 开环结论不受影响**：判定「条件化价值不可组合」。
  允许的后续方向仅两个，各一轮：
  1. 注入衰减策略（只保留最近 k 步已提交控制，k 预注册为 1 与 2 两档）；
  2. selector 状态特征加入「已提交控制摘要」维度（让 selector 看见闭环
     历史，owner 仍是 `vz-temporal`）。
- **kill condition**：两个方向后闭环门仍失败，或出现「train 闭环正、fresh
  分区闭环系统性负」的过拟合翻转 → 主张收缩为「条件化动作价值仅开环成立，
  不可组合注入」，live-wiring 路线长期冻结，Gate 2 推进转向 longitudinal
  层（开环 selector 作为只读 readout 仍可用于 credit 证据）。
- fresh 语料预算沿用 v35 计划 §6 第 3 条的账本：v35 已消耗 8 条，本包再
  消耗 8 条；若本包失败后两个修正方向各需一次性检验，必须再新造，累计
  预算触顶前必须收敛。

## 7. 本包之后的路线（不在本包实现）

> 本包已被编入跨 gate 长周期战役计划
> [`gates-1-6-evidence-campaign_f3a80c15.plan.md`](gates-1-6-evidence-campaign_f3a80c15.plan.md)
> 的 Phase 0；Gate 1/4/5/6 的推进以该战役计划为准。

1. **runtime SHADOW wiring 包**：把冻结 selector 以 `WiringLevel.SHADOW`
   接入真实 session 轨迹（复用 CP-LSS `learned_shadow_evidence` 模式），
   在真实对话快照上记录 would-be 注入与 audit，回答「合成 hierarchical
   语料上的分布是否迁移到真实轨迹」。准入条件：本包
   `shadow_observation_passed=true`。
2. **longitudinal 对表包**：≥500 real-trace、跨 session、`validation_delta
   ≥ 0.02` 的 Gate 2 EXIT 条款，与 #88 合流。
3. **更大基底复测**（触发条件见 v35 计划 §6）。

## 8. 结果（run 完成后回写，当前留空）

待本包执行后填写。
