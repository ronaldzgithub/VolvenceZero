# 认知 Regime Spec

> Status: draft
> Last updated: 2026-07-20
> 对应需求: R14

## 要解决的问题

如何让系统维护持久的交互模式身份，而非将其视为临时 prompt 标签？

## 关键不变量

- Regime 不是 prompt 标签，而是可记忆、可选择、可训练的持久身份
- Regime 在运行时状态中表示（非字符串标签）
- Regime 可从记忆中召回历史效果
- Regime 可被高层控制选择（由抽象控制层选择，而非硬编码规则）
- Regime 可通过延迟结果训练（通过信用分配回路）

## 工程挑战

- 设计 regime 的运行时表示（向量嵌入，不只是字符串标签）
- 实现 regime 的记忆化（可召回历史 regime 及其效果）
- 实现 regime 的高层控制选择（由抽象控制层选择，而非硬编码规则）
- 实现 regime 的延迟结果训练（通过信用分配回路）
- 场景检测必须使用语义级方法，不使用关键词匹配

## 算法候选

Regime 的选择和训练与 ETA 的 metacontroller 紧密相关：
- 控制器代码 `z_t` 的聚类可以对应不同 regime
- 切换单元 `β_t` 的切换时刻可以与 regime 切换对齐
- Internal RL 可以训练 regime 选择策略

### Regime 类型

- casual social contact（日常社交）
- acquaintance building（关系建立）
- emotional support（情感支持）
- guided exploration（引导探索）
- problem solving（问题解决）
- repair and de-escalation（修复与降级）

### Regime 运行时表示

```python
@dataclass(frozen=True)
class RegimeIdentity:
    regime_id: str
    name: str
    embedding: tuple[float, ...]        # 向量表示，非字符串标签
    entry_conditions: str
    exit_conditions: str
    historical_effectiveness: float     # 历史效果评分
```

## 接口契约

**消费的输入**：
- `memory` 快照：regime 历史效果记忆
- `dual_track` 快照：两轨状态（regime 选择需考虑两轨）
- `evaluation` 快照：regime 效果评估分数
- `prediction_error` 快照：上一轮 delayed / per-dimension mismatch，用于更新 historical effectiveness 和当前选择偏置
- `experience_fast_prior` 快照：session-post slow loop 压缩出的 delayed-credit fast bias，用于选择偏置和 sequence payoff 的轻量调节

**产出的输出**：
- `regime` 快照：`RegimeSnapshot`
  - 当前活跃 regime
  - 前一个 regime（如有切换）
  - 切换原因
  - 候选 regime 及评分
  - 当前 regime 持续轮数

**当前实现口径**：

- P04 阶段先保证结构化 identity、稳定 candidate scoring 和可审计切换原因
- 当前选择逻辑基于 `memory`、`dual_track`、`evaluation` 的状态评分基线
- 当前 `RegimeModule` 已新增 owner-side `metacontroller` evidence ingest path：joint loop 可直接用 controller active label / guard rollback evidence 更新 strategy priors，而不改变 `regime` snapshot 契约
- 当前 `RegimeModule` 已新增 owner-side delayed attribution queue：上一轮 regime 选择会在后续 turn 的 evaluation 上结算，并通过 `delayed_outcomes` 发布
- 当前 `RegimeModule` 已直接消费 `prediction_error` slot：`score_regimes()`、historical effectiveness 更新和 turn score 记录都会把 PE 维度偏差纳入 regime 选择与训练
- 当前 `RegimeSnapshot.identity_hints` 作为 typed identity proposal 暴露给 reflection/memory owner；durable identity 写入仍由 reflection/memory owner 决定，Regime owner 不直接越权写 memory
- 当前 `RegimeSnapshot` 还发布 `delayed_attribution_ledger`、`delayed_payoffs`、`sequence_payoffs`、`effectiveness_trend`、`regime_changed` 与 `selection_weights`，供 credit / evaluation / reflection 读取 owner-side delayed outcome 与选择权重证据
- 当前 `RegimeModule.default_wiring_level = SHADOW`；默认类级接线保持 evidence / audit surface，final wiring 可按 rollout 需要显式激活
- 后续可由 temporal / learned selector 替换，但不改变 `regime` snapshot 契约
- 当前 companion evidence 增加 `RGM1 regime_delayed_attribution_visibility` gate：用 dialogue-like repair/support 信号证明 `RegimeModule` 的 delayed attribution / delayed payoff / sequence payoff 能进入 credit 与 evaluation readout，而不是每 turn 静态 prompt 标签重选

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.6 节

## 瞬时 substrate readout vs 持久 regime owner 的分层（2026-07-20）

来源：Anthropic "Emotion Concepts and their Function in an LLM"（`arXiv:2604.07729`），
见 `research/frontier-sweep-2026-07-20.md` §C / §4.3。论文证明情绪概念是残差流中可读出的
内部方向，并对 reward hacking / blackmail / sycophancy 有因果影响——这是 R14"regime 是
内部几何而非 prompt 标签"的强外部证据。但论文同样明确：这些方向编码的是当前 token 处理的
**operative emotion concept**，不是持续存在的主体状态。据此固定三层分工：

| 层 | 载体 | 语义 | 时间尺度 | 约束 |
|---|---|---|---|---|
| substrate readout | `substrate` owner 发布的 persona/emotion geometry 只读信号 | token-local 的"当前有效概念"，可用于风险预警（如 desperation↑ + calm↓ 组合监控） | 瞬时（turn 内） | **只读**；由 producer 发布 typed snapshot，消费者不自行从残差重建；禁止反向成为训练 reward（防 probe→train Goodhart，R12） |
| regime owner（本 spec） | `RegimeSnapshot` 的持久身份 + delayed attribution | 跨 turn / 跨 session 的可训练身份状态 | session ~ longitudinal | regime 状态**不得**由 substrate 瞬时 readout 直接改写；瞬时信号只能作为 evidence 之一进入既有 scoring / delayed attribution 路径 |
| expression 层 | 表达层消费上游 snapshot | 呈现，不推断 | turn | 不自行拼装"情绪/persona"描述（R8） |

把三层合并（例如用 emotion vector 当 regime 真值标签、或用 steering 直接塑造 regime 行为）
会重演 persona-vector 的 Goodhart 风险；该失败模式记为本 spec 的负面锚点。
substrate 侧的 geometry readout 监控面见 Lane D 落地项与 `docs/specs/evaluation.md` 的
`persona_geometry_drift` readout（COG-3）。

## Panorama 参与门（`panorama_level`）

`ParticipationHint.panorama_level` 是回答"这一轮该不该展开全景式决策结构"的**唯一决策面**。任何其他地方出现第二个激活判断都是契约违规——那会让"何时开"变成分散在各处的场景规则。

### 判据：四个与话题无关的结构信号

门的价值等式是 **全景改变最终选择的概率 × 选错的代价**；成本是打断、把情绪对话变成表格、显得冷酷。以下四个信号是这个乘积的可测代理，全部由 typed semantic owner 快照派生（`regime/decision_structure.py`），**不读用户文本、不做话题分类**：

| 特征 | 含义 | 来源 owner |
|---|---|---|
| `option_multiplicity` | 是否存在 ≥2 条互斥且都还活着的路径 | `plan_intent.candidate_plans` / `deferred_intents`，`goal_value.active_tradeoff_count` 佐证 |
| `irreversibility` | 选错是否难以低成本撤回 | `commitment` 的义务暴露 + `boundary_consent` 的外部行动/专业范围压力 + `plan_intent.active_constraints` |
| `ranking_instability` | 用户自己的排序是否在动 | `goal_value.value_conflict` / `goal_shift_pressure` / `decision_readiness` / `conflicted_goal_count`，`plan_intent` 修订churn |
| `unknown_dominance` | 排名是否取决于尚未拿到的信息 | `belief_assumption.verification_needs` / `mean_confidence`，`open_loop` 未决项与确认债 |

四者用**几何平均**合成，不是加权和：四项**同为必要、单项均不充分**。加权和会让三个强轴把归零的第四轴拖过线——那正是系统把别人的痛苦做成表格的机制。几何平均下任一项为 0 即关门。

两个由此**自动得到、无需专门规则**的行为：情绪支持轮次被挡住，是因为没有任何未知在等信息（`unknown_dominance≈0`），不是因为写了"repair 时保持安静"；选餐厅被挡住，是因为没有难以撤回的东西（`irreversibility≈0`）。

两个刻意**不用**的信号：
- `goal_value.reversibility_need`——名字贴切但它由 `value_conflict` / `goal_shift_pressure` / `alignment_score` 算出，与 `ranking_instability` 共线，用它会让"四轴合取"变成一轴取幂。
- "有选项但没选定"——这是最强的不稳定信号，但它是选项数的函数。第一版用了它，导致 `option_multiplicity × ranking_instability` 在审计语料上 r=0.96；移除后降到 0.69。

### 代价不对称

该开没开：用户少一次结构化帮助，下一轮还能开，**可恢复**。不该开却开了：关系性伤害，**用户通常不会说**。因此：

- 入口有负偏置、已开有保持偏置（防抖），且**假阳性率与假阴性率分开报，禁止合成单一 F1**——合成正好掩盖这个不对称；
- 每轮最多升一档（SILENT → BRIEF → STRUCTURED），降档不设限。会话第一轮永远到不了 STRUCTURED：先见到人，再谈结构；
- **"未测量"不等于"测得为零"**。没有任何 semantic owner 提供观测时（冷启动、owner SHADOW/DISABLED 发 `RuntimePlaceholderValue`）门直接判 SILENT；
- 退出条件独立于保持偏置：选项塌缩到一条时强制降档；
- 用户显式覆盖直接胜出（写入通道待 `decision_workspace` 落地）。

### 接线与回滚

`FinalRolloutConfig.panorama_gate_mode`：`"v1"`（生产默认，既有 attention-posture 打分，行为不变）/ `"v2"`（本节的结构门）。两种模式下 `method_level` / `task_level` / `flow_kind` 完全一致，只有 `panorama_level` 变——回滚就是改一个字段。regime owner 新增 7 个 semantic slot 为可选依赖（`.get()` 读取 + isinstance 过滤），未接线的运行时不受影响。防抖状态不入 `RegimeCheckpoint`：重新水合从 None 起步（即从 SILENT 逐档升），把跨会话边界的偏差留在保守方向。

### 证据

`scripts/audit_panorama_gate.py --probe`，语料 `regime/panorama_corpus.py`（负例 5 / 边界 6 / 正例 4），产物在 `research/panorama_gate/`。语料与门由同一人所写，所以单看"v2 全过"没有意义，两个对抗探针才是可证伪的部分：

- **消融**：把某个特征钉成 1.0（门对它失明）后重跑。四个特征各被一条专门的边界例抓住；任何一个消融后语料仍全过，说明该特征只是注释。
- **共线**：四特征两两 Pearson。真实决策情境本就同向变动，所以要求的是"不重复"而非"不相关"，上限 0.85。

当前读数：v1 ceiling=6 / floor=2（15 例），v2 ceiling=0 / floor=0；最大共线 0.687。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时（5.5）| 通过快照发布 regime 状态 |
| 依赖 | Prediction Error 主链 | delayed / per-dimension PE 直接驱动 regime historical effectiveness 与选择偏置 |
| 依赖 | 连续记忆（5.3）| 从记忆中召回 regime 历史效果 |
| 依赖 | 双轨学习（5.4）| regime 选择需考虑两轨状态 |
| 被依赖 | 评估体系（5.7）| F5 中的 regime 对齐度评估 |
| 协作 | 信用分配（5.6）| regime 效果通过信用分配回路训练 |

## 变更日志

- 2026-07-30: regime owner 增加 frozen learning gate 与 owner-authored
  `learning_fingerprint()`。冻结模式保留 active regime 选择、切换及推理快照，关闭
  delayed attribution/effectiveness/selection/score-learner 写入；学习指纹不再混入
  active identity、turn index、持续轮数等合法运行时 telemetry。`RegimeScoreLearner`
  的零权重冷启动预测改为只读，只有真实 settlement 才实例化可持久权重。

- 2026-07-27: Panorama 参与门 v2（见上节）。`regime/decision_structure.py` 新增四个话题无关的决策结构特征；`hint_readout.readout_panorama_level` 用几何平均合成并施加不对称代价偏置、一档升级上限、选项塌缩退出与用户覆盖；`FinalRolloutConfig.panorama_gate_mode` 默认 `"v1"`（行为不变）。审计语料 + 消融/共线探针见 `regime/panorama_corpus.py`、`regime/panorama_audit.py`、`scripts/audit_panorama_gate.py`。过程中修掉一处真实缺陷：`ranking_instability` 原本读选项数，与 `option_multiplicity` 在语料上 r=0.96，使"四轴合取"名不副实；解耦后 0.69。测试 `tests/test_panorama_gate.py`。
- 2026-07-20 (score_regimes learned 化 SHADOW 升级): `RegimeScoreLearner` SHADOW dual-run 候选从"baseline 残差 4 维"扩到消费全部共享 per-turn 特征（4+36 维）：新增 `scoring.compute_regime_feature_values` 作为特征单一计算点（`score_regimes` 固定公式与 learner 共用，行为字节不变），`_ADJUSTMENT_FEATURE_NAMES` 冻结 bootstrap 调整面，learner `_SHARED_FEATURE_ORDER` 冻结 checkpoint 权重对齐（旧 4 维 checkpoint zero-pad 兼容）。live regime 选择不读 learner；ready 门槛 settled≥50 + MAE 领先≥0.02，kill=劣化≥0.10。对应 debt #80/#44；测试 `tests/test_regime_score_learner.py`。
- 2026-07-20: 新增 §"瞬时 substrate readout vs 持久 regime owner 的分层"（Anthropic Emotion Concepts 同步）：token-local operative concept / 持久 regime owner / expression 三层分工，geometry readout 只读、禁止反向训练。来源 `research/frontier-sweep-2026-07-20.md` §6 同步项。
- 2026-05-02: 增加 RGM1 companion evidence gate，冻结 regime delayed attribution visibility 的自动证据口径（RegimeSnapshot delayed attribution → credit → evaluation readout）
- 2026-04-25: 同步当前 `RegimeSnapshot` delayed payoff / sequence payoff / selection weight 字段，并补充 `experience_fast_prior` 输入与默认 `SHADOW` 接线
- 2026-04-20: 接口契约按当前代码收敛为直接消费 `memory + dual_track + evaluation + prediction_error`；当前实现口径明确 regime owner 已直接用 PE 更新 selection bias 与 historical effectiveness
- 2026-04-09: next_gen_emogpt v2: regimes positioned as prediction spaces within the dual-track framework; regime selection weight updates driven by prediction error from delayed outcomes; repo default term: `abstract action` (paper synonym: `subgoal`)
- 2026-04-09: U03 Regime A/B verification: RegimeSelectionWeights confirmed to diverge from uniform (1.0) after delayed outcomes accumulate via process_standalone loop. Learned weights stay within [0.3, 2.0] range. effectiveness_trend published in RegimeSnapshot and verified non-empty after 4 turns.
- 2026-04-06: 补充 owner-side metacontroller evidence ingest 的当前实现口径
- 2026-03-25: 初始版本，从 SYSTEM_DESIGN.md 和 next_gen_emogpt.md 提取
