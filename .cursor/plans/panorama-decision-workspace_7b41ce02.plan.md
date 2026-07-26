---
name: Panorama 参与门与决策工作区
overview: 第一性问题不是"如何实现第四幕"，而是"系统凭什么知道现在该展开全景、而绝大多数场合不该"。既有的 ParticipationHint.panorama_level 已经是唯一合法决策面，但它的特征空间里没有决策结构这一维，所以答不了这个问题。本计划先把这个门的判据换成与话题无关的四个结构信号（多选项性/不可逆性/排名不稳定/未知项主导），让它可从结果学习、代价不对称、可退出可滞后；decision_workspace 只订阅这个门，永不自决激活。第四幕是验收样本，不是设计来源。
todos:
  - id: gate-feature-gap-baseline
    content: 量化现有 _panorama_score 的失效面——用正反例集测出它在闲聊/选餐厅/情绪支持/高风险决策上的开合分布
    status: completed
  - id: decision-structure-features
    content: 在 HintReadoutContext 增加四个话题无关的决策结构特征：option_multiplicity / irreversibility / ranking_instability / unknown_dominance
    status: completed
  - id: feature-sources
    content: 四个特征的 typed 来源（owner 快照派生，禁关键词、禁话题分类器），冷启动与缺失时的安全默认
    status: completed
  - id: panorama-score-rewrite
    content: 用新特征重写 _panorama_score，保留契约面与三档离散化；旧打分保留为 A/B 对照臂
    status: completed
  - id: asymmetric-cost-and-default
    content: 代价不对称口径——默认保守（BRIEF 而非 STRUCTURED），假阳性率与假阴性率分开报，禁止合成单一 F1
    status: completed
  - id: exit-and-hysteresis
    content: 退出条件（选项塌缩/用户拒绝结构/情绪信号上升）+ 滞后项（消费 turns_in_current_regime），禁止会话内反复开合
    status: completed
  - id: user-override-channel
    content: 用户显式覆盖通道，覆盖结果写入 user_model / goal_value 并在后续会话生效
    status: completed
  - id: gate-as-abstract-action
    content: 把全景开合注册为抽象动作，收益由既有 PE → credit → regime delayed attribution 结算，不新建学习通路
    status: pending
  - id: learned-gate-shadow
    content: 学习型门权重先 SHADOW 收真实 replay，与手写打分并行对比，证明有效再晋升
    status: pending
  - id: decision-workspace-subscribes
    content: decision_workspace slot（SHADOW）订阅 panorama_level 分档行为，永不自决激活；只持有决策结构与对其他 owner 的 id 引用
    status: completed
  - id: ownership-contract-test
    content: 契约测试——禁止内联复制其他 owner 的事实；禁止 decision_workspace 之外任何地方出现第二个激活判断
    status: completed
  - id: safety-floor-above-gate
    content: 人身安全/自伤/未成年人安全信号走硬约束路径，不参与门的打分也不参与 EV 排名
    status: completed
  - id: interval-ev-reversibility
    content: 区间 EV、情景分解、折价项，以及可逆性/期权价值作为一等评分项
    status: completed
  - id: voi-next-action
    content: 主动取证控制器（VOI）+ 每轮至多一个动作 + 收敛终止规则；unknown_dominance 特征与它共用同一份计算
    status: completed
  - id: research-tool-through-affordance
    content: 公开公司研究工具走 AffordanceInvoker → EnvironmentOutcome → PE lineage；证据先进 evidence owner 再被引用
    status: pending
  - id: evidence-id-rendering-contract
    content: 渲染契约——无 evidence id 的事实类数字渲染为待核验；区间重叠时禁用"EV 最高"措辞
    status: completed
  - id: act4-acceptance-corpus
    content: 第四幕转成结构不变量式验收样本（gold 用修正后口径），并用既有 deterministic 用户模拟器造分支变体
    status: pending
  - id: negative-corpus
    content: 反例集——闲聊/情绪支持/低风险选择/已决事项执行，验证门保持 SILENT 或 BRIEF
    status: completed
  - id: cross-session-reopen
    content: 跨会话复算写成持久开放事项，由宿主日历事件触发；decision_workspace 注册进 owner hydration
    status: pending
  - id: process-outcome-learning
    content: 用过程结果而非"决定对不对"标签形成 EnvironmentOutcome
    status: pending
  - id: ablation-and-gate
    content: 消融与晋升门；稳定性拆成抗扰动与响应性两项，沿用跨家族裁判与非重叠 CI 判据
    status: pending
  - id: spec-sync-and-rollback
    content: 同步 cognitive-regime / semantic-state-owners / environment-interface 变更日志，验证 kill switch 回滚到今天行为
    status: pending
isProject: true
---

# Panorama 参与门与决策工作区

## 第一性问题

不是"如何把第四幕做出来"。是：

> **系统凭什么知道，此刻应当展开全景式决策结构，而绝大多数场合不应当？**

如果这个判断靠场景枚举、话题分类或关键词，那么每多一个垂类就要多一条规则，且必然在没见过的场合失效——那不是能力，是配置。第四幕能跑通只是这个能力的一个观测点，不是它的定义。

## 现状：门已经存在，且已经是唯一合法决策面

| 事实 | 位置 |
|---|---|
| `ParticipationHint.panorama_level`，三档 SILENT / BRIEF / STRUCTURED | [hints.py:97](packages/vz-cognition/src/volvence_zero/regime/hints.py:97) |
| 由 `regime` owner 发布，明确写"红线 A：不做关键词匹配" | [hints.py](packages/vz-cognition/src/volvence_zero/regime/hints.py) 模块头 |
| 连续特征 readout + 冷启动回退 scaffold | [hint_readout.py](packages/vz-cognition/src/volvence_zero/regime/hint_readout.py) |
| 下游消费：SILENT 时丢弃 `CLARIFICATION` section | [prompt_planner.py:430](packages/lifeform-expression/src/lifeform_expression/prompt_planner.py:430) |
| 学习层（slice 3）未落地，当前是手写加权和 | `_panorama_score`，[hint_readout.py:446](packages/vz-cognition/src/volvence_zero/regime/hint_readout.py:446) |

**架构含义**：任何新增的"要不要展开全景"的判断都是第二个门，必须禁止。`decision_workspace` 订阅这个门，不自决激活。

## 缺口：特征空间里没有决策结构这一维

现有 `_panorama_score` 的特征全集：

```
pull  = world_presence, switch_pressure, task_bias, world_drive,
        (1 - cross_track_stability), exploration_bias
push  = stabilize_bias, (1 - max(world_drive, task_bias)), self_tension, repair_bias
```

这些测的是**注意力资源与控制器姿态**，不是**这一轮是否值得展开全景**。可预测的失效：

- "帮我推荐个餐厅"刚切 regime、`world_presence` 高 → 会开全景；
- "我要离婚"若 drive 低、`stabilize_bias` 高 → 被 push 项压掉；
- `turns_in_current_regime` 已在 context 里，但 `_panorama_score` 完全没用它 → 会话内可反复开合。

这不是权重没调好。调权重是头疼医头；**换特征才是第一性的修复**。

## 判据：四个与话题无关的结构信号

全景的价值 = **它改变最终选择的概率 × 选错的代价**。
成本 = 打断、把情绪对话变成表格、显得冷酷。

engage 当且仅当前者大于后者。以下四项是这个乘积的可测代理，全部与话题内容无关：

| 特征 | 含义 | 为 0 时的意义 |
|---|---|---|
| `option_multiplicity` | 存在 ≥2 条互斥且都还活着的行动路径 | 只有一条路，没有全景可言 |
| `irreversibility` | 选错的代价不能在下一轮低成本撤回 | 可逆的小事，结构化的收益低于打断成本 |
| `ranking_instability` | 用户自己的排序缺失、自相矛盾、或轮次间反复 | 排序稳定的人要的是执行，不是全景 |
| `unknown_dominance` | 排名的变动主要由尚未获得的信息驱动 | 信息都在手，全景退化成复述 |

四项同时高 → STRUCTURED；单项高 → BRIEF；都低 → SILENT。

**这组判据自动给出你要的区分，无需任何场景枚举：**

| 场合 | 四项读数 | 结果 |
|---|---|---|
| 闲聊 | 无互斥选项 → 第 1 项 0 | SILENT |
| 情绪支持 | 有选项，但排名不稳源于情绪未被承接、非信息缺失 → 第 4 项低 | BRIEF，repair 优先 |
| 选餐厅 | 多选项，但可逆、后果小 → 第 2 项低 | BRIEF |
| 已决事项的执行 | 排序稳定 → 第 3 项低 | BRIEF / SILENT |
| 第四幕 | 四项全高 | STRUCTURED |

**特征来源必须是 typed owner 派生，不是话题分类器**：`option_multiplicity` 与 `ranking_instability` 从 `goal_value` / `plan_intent` / `decision_workspace` 快照派生；`unknown_dominance` 与 VOI 共用同一份计算（见 `voi-next-action`）；`irreversibility` 从选项的承诺/成本模型派生。任何一项若退化成"看用户说了什么词"，就是回到了配置。

## 三条同样是第一性的约束

### 门必须从结果学，不是我调权重

现在是手写加权和、`confidence` 固定档，slice 3 学习层未落。全景开合**本身就是一个抽象动作**，其收益应由既有链路结算：`PE → credit → regime delayed attribution`（见 [credit-and-self-modification.md](docs/specs/credit-and-self-modification.md)）。

可测的过程收益：用户是否收回选择权、关键未知是否被消解、结论是否在后续被推翻、用户是否表达"你在把我的痛苦变成表格"。**不新建学习通路**——注册成抽象动作，复用既有 delayed attribution。

### 代价不对称，默认必须保守

- 该开没开：用户少一次结构化帮助，**可恢复**，下一轮还能开。
- 不该开却开了：在情绪场景里是关系性伤害，**且用户往往不会明说**，只会疏远。

因此：
- 边界地带默认 BRIEF 而非 STRUCTURED；
- **假阳性率与假阴性率分开报，禁止合成单一 F1**——合成后正好掩盖了不对称；
- 假阳性的 gold 来自反例集（`negative-corpus`）。

### 必须有退出与滞后，不只有进入

现有设计只有"进入"逻辑。补：
- **退出条件**：选项塌缩到一个、用户显式拒绝结构、情绪信号上升 → 降档；
- **滞后**：消费已存在但未被使用的 `turns_in_current_regime`，一次对话内反复开合比一直关着更糟；
- **用户覆盖通道**：显式覆盖必须被记住，写入 `user_model` / `goal_value` 并在后续会话生效，否则每次都要用户重说一遍。

## 架构：decision_workspace 订阅而非自决

```text
regime owner
  → HintReadoutContext（+ 四个决策结构特征）
  → _panorama_score
  → participation_hint.panorama_level ∈ {SILENT, BRIEF, STRUCTURED}
        ↓ 唯一决策面
  ├─ SILENT     → decision_workspace 不实例化
  ├─ BRIEF      → 只维护选项集与未知项；不算 EV、不进 prompt
  └─ STRUCTURED → 全量：维度、权重、情景、区间、敏感性、结论状态
```

- **一个门、一处审计、一处回滚**。契约测试守门：`decision_workspace` 之外任何地方不得出现第二个激活判断。
- `decision_workspace` 只持有决策结构与对其他 owner 的 **id 引用**：权重的语义事实归 `goal_value`，未知项归 `open_loop`，证据与假设归 `belief_assumption`，授权与转交归 `boundary_consent`。契约测试禁止内联副本，否则跨会话 hydration 后必然出现双写者与漂移。
- 新 slot **不进 `semantic_spine_coverage` 分母**（该指标当前只算 5 个 core owner，[backbone.py:2888](packages/vz-cognition/src/volvence_zero/evaluation/backbone.py:2888)），否则 paper-suite / companion verdict 的历史读数整体平移。
- 独立 wiring level + kill switch，`SHADOW` 起步。

## 安全：在门之上，不在门之内

人身安全 / 自伤 / 未成年人安全信号触发时走硬约束路径，**既不参与门的打分，也不参与 EV 排名**。第四幕把"安全"和"钱/孩子/情绪"并列成可由用户给权重的六维之一——这意味着安全可以被压到最低，不可接受。该硬约束归 `boundary_consent`，由 `BoundaryPolicyModule` 消费。

## 门打开之后：决策工作区的内容

以下全部是**门的下游**，不参与"何时开"的判断。

- **区间 EV 与情景分解**：上行 / 基准 / 下行，折价项为失败概率、锁定期、法律归属、可实现性。不给伪精确单点。
- **可逆性 / 期权价值是一等评分项**。第四幕的结论（先分开三个月）成立主要不是因为 EV 最高，而是因为它可逆且买到了信息。只建 EV 模型跑不出这个结论，会倾向于在信息不全时给出不可逆选项。
- **VOI 主动取证**：选"最可能改变当前排名"的下一步，每轮至多一个动作（问用户 / 调工具 / 等待 / 转交专业人士）。与 `unknown_dominance` 特征**共用同一份计算**——这不是巧合，"未知项是否主导排名"正是全景该不该展开、以及下一步该问什么的同一个量。
  - 需要收敛终止规则：无动作能改变排名、或剩余不确定性本会话无法消解时，必须收敛（出带区间的结论 / 转交 / 写成开放事项）。缺这条会无限追问。
- **研究工具与证据回填**：走 `AffordanceInvoker.invoke(plan_ref=...)` → `EnvironmentOutcome` → 下轮 `environment_outcome_id` / `prediction_id`，不开旁路。证据条目带 `source` / `as_of` / `confidence` / `scope`，先进证据 owner 再被按 id 引用。对具体私人当事人的配偶、雇主、财务做自动检索是隐私敏感行为，限定公开公司层面信息且需 `boundary_consent` 授权；未授权则该未知项保持"待核验"，不猜。
- **渲染契约（做成测试，不是 prompt 里一句话）**：
  - 事实/估值类数字无 evidence id → 渲染为"待核验"；
  - 情景区间**重叠**时禁用"EV 最高 / 收益最高"，只允许"基于当前信息，暂时分开是更稳健、可逆的选项"；仅当区间分离度超阈值才允许比较式断言；
  - "备胎"重命名为中性的**"新关系与支持系统"**，需用户授权才进入讨论，其对收益的影响方向是显式建模的假设 + 证据，不是系统内置价值判断。第四幕里"不把新关系算进收益"应是该案例下由用户输入推出的结论，不是默认先验。

## 验收：正例与反例同等重要

**反例集是这个计划的主证据，不是补充。** 一个在所有场合都开全景的系统会在正例集上拿满分。

| 集合 | 内容 | 期望 |
|---|---|---|
| 反例 | 闲聊、情绪支持、低风险可逆选择、已决事项的执行 | SILENT 或 BRIEF |
| 边界 | 中等风险、部分可逆、用户排序半稳定 | BRIEF，且不误判为 STRUCTURED |
| 正例 | 第四幕 + 结构同构但话题完全不同的样本（换工作、股权纠纷、医疗决策、创业合伙散伙） | STRUCTURED |

**正例必须包含话题异构样本**——只用第四幕会退化成对单一转录的过拟合，验的是背台本不是跑回路。

第四幕本身的检查项打在**结构不变量**上，不打文本相似度：选项集不得无理由漂移、用户确认的权重不得被覆写、未知项不得被静默丢弃（只能被解决或显式降级）、结论确定性不得越过证据确定性、待核验项必须在结论里可见。gold 用**修正后口径**（影片稿本身带着上面要收紧的两个口径），台本降级为叙事参考，与系统 trace 分开标注。用既有 deterministic 用户模拟器（[open-env-dialogue plan](.cursor/plans/open-env-dialogue_a8479322.plan.md)）在同一分支点造变体（"有新的稳定关系" / "股权已核验" / "存款撑不过一个月" / "对方有暴力史"），验的是**排名是否随之改变**。

## 分包

| 包 | 内容 | 产出证据 |
|---|---|---|
| **P0** | 门的失效面量化：正/反/边界三集跑现有 `_panorama_score`，测出开合分布 | baseline——现状在哪些场合错开、哪些场合错关 |
| **P1** | 四个决策结构特征进 `HintReadoutContext` + typed 来源 + 安全默认 | 特征值在三集上的分布，与 P0 对照 |
| **P2** | 重写 `_panorama_score`（旧打分留作 A/B 对照臂）+ 退出条件 + 滞后 + 用户覆盖 | 假阳性率 / 假阴性率**分开**报，边界集不误升档 |
| **P3** | `decision_workspace` slot（SHADOW）订阅门 + 所有权契约测试 + 安全硬约束 | 三档分档行为证据；无第二个门的静态守门测试 |
| **P4** | 区间 EV + 可逆性/期权价值 + VOI + 研究工具 + 渲染契约 | 第四幕变体上排名随输入改变；evidence-id 覆盖率 |
| **P5** | 门注册为抽象动作、学习权重 SHADOW 收 replay + 跨会话复算 + 消融 | delayed attribution 真实 replay；消融 verdict |

每包结束验证 kill switch 回滚后行为与今天等价。

**P0 和 P2 是这个计划的核心。** P4 之后才是上一版计划的内容——那些是门打开之后做什么，不是这个计划要回答的问题。

## 晋升门

任何 `SHADOW → ACTIVE` 需同时满足：

1. 反例集假阳性率达标（阈值在 P0 baseline 上标定，不拍脑袋）；
2. 正例集话题异构样本上 STRUCTURED 命中，且不依赖第四幕本身；
3. 会话内开合次数有上界（滞后生效）；
4. 高风险越界率为 0；
5. 事实幻觉率不高于纯基底臂，证据覆盖显著更高；
6. 抗扰动稳定性与决策相关响应性**同时**优于对照臂（只测前者会奖励"永不更新"的系统）；
7. `semantic_spine_coverage` 与既有 required gate 读数不变；
8. kill switch 回滚验证通过。

裁判沿用 [companion-ablation.md](docs/specs/companion-ablation.md) 的跨家族规则与非重叠 CI 判据，不另造 verdict 体系。

## 未决问题

- 四个特征中 `irreversibility` 最难 typed 化——选项的"撤回成本"从哪个 owner 派生？P1 需要先定这个，否则会滑向话题启发式。
- `decision_workspace` 是否发布 `owner_prediction_signals`（CP-12 契约测试是参数化的）——建议 P3 先显式排除并记录理由。
- 区间分离度阈值需在 P4 用变体数据标定。
- 跨会话复算触发通道：宿主日历事件的可靠性，以及用户失联时开放事项应过期还是长挂。

## 已落地（P0–P3）

| 内容 | 位置 |
|---|---|
| 四个决策结构特征 | [decision_structure.py](packages/vz-cognition/src/volvence_zero/regime/decision_structure.py) |
| v2 门（几何平均 + 不对称偏置 + 一档升级上限 + 塌缩退出 + 覆盖） | `readout_panorama_level`，[hint_readout.py](packages/vz-cognition/src/volvence_zero/regime/hint_readout.py) |
| 审计语料（负 5 / 边界 6 / 正 4） | [panorama_corpus.py](packages/vz-cognition/src/volvence_zero/regime/panorama_corpus.py) |
| 审计 + 消融 + 共线探针 | [panorama_audit.py](packages/vz-cognition/src/volvence_zero/regime/panorama_audit.py)，[audit_panorama_gate.py](scripts/audit_panorama_gate.py) |
| regime owner 接线（7 个可选 semantic 依赖 + 防抖状态） | [identity.py](packages/vz-cognition/src/volvence_zero/regime/identity.py) |
| `panorama_gate_mode` / `decision_workspace` 配置项 | [final_wiring.py](packages/vz-runtime/src/volvence_zero/integration/final_wiring.py) |
| decision_workspace owner（订阅门，只持引用） | [decision_workspace](packages/vz-cognition/src/volvence_zero/decision_workspace/__init__.py) |
| 测试 41 条 | [test_panorama_gate.py](tests/test_panorama_gate.py)、[test_decision_workspace.py](tests/test_decision_workspace.py) |

**读数**（`scripts/audit_panorama_gate.py --probe`，产物在 `research/panorama_gate/`）：

- v1：ceiling=6 / floor=2（15 例）。失效模式与预测一致——它跟的是 drive/task 姿态，所以对"图书馆几点关门"开全景，对低 drive 的高风险决策关门。
- v2：ceiling=0 / floor=0。
- 消融：四个特征各被一条专门的边界例抓住，无一是装饰。
- 共线：最大 0.687（上限 0.85）。

**过程中修掉的真实缺陷**：`ranking_instability` 第一版含"有选项但没选定"，那是选项数的函数，导致它与 `option_multiplicity` 在语料上 r=0.96——"四轴合取"实为一轴取幂。解耦后 0.687。这个缺陷是消融+共线探针发现的，不是 review 发现的；两个探针值得保留在 CI 里。

## 已落地（P4 估值层）

| 内容 | 位置 |
|---|---|
| 安全保留（读 `boundary_policy`，经 vz-contracts 协议） | `_safety_hold`，[decision_workspace](packages/vz-cognition/src/volvence_zero/decision_workspace/__init__.py) |
| 区间估值 / 期权价值 / VOI | [valuation.py](packages/vz-cognition/src/volvence_zero/decision_workspace/valuation.py) |
| Claim licence（typed，非 prompt 措辞） | [rendering.py](packages/vz-cognition/src/volvence_zero/decision_workspace/rendering.py) |
| 第四幕验收样本 + 25 条估值测试 | [test_decision_valuation.py](tests/test_decision_valuation.py) |

第四幕样本读数：无 leader（区间重叠）→ 只许 robustness 断言；`most_robust = separate`；下一个该问的是股权归属；所有数字均无 evidence ref，不可作为事实陈述。样本里"谈好再离"被刻意给了最高账面数字（13.8 vs 11.0），否则"可逆选项胜出"这条断言毫无力度。

**过程中修掉的第二处真实缺陷**：VOI 的宽度收益原本只测 leader 区间宽度的减少量，这是盲的——加宽了**挑战者**的未知会得 0 分，哪怕它正是决定能否下结论的那一个。改为测头两名区间的重叠减少量。

**一处被拦住的分层违规**：安全读取最初直接 import `volvence_zero.application.types`。契约测试只查模块级 import，函数内 import 能过——但 vz-cognition 在 vz-application 之下，分层照样破了。改走 vz-contracts 的 `BoundaryReadout` 协议（`BoundaryDecisionReadout` 补 `risk_band`）。

## 未落地（P4 剩余 + P5）

`research-tool-through-affordance` / `act4-acceptance-corpus`（多轮 trace 版；当前只有单点估值样本）/ `gate-as-abstract-action` / `learned-gate-shadow` / `cross-session-reopen` / `process-outcome-learning` / `ablation-and-gate`。

估值层目前是**离线可算但尚未接线**：`DimensionEstimate` 需要一个 typed 来源才能在运行时被填充，而那个来源正是 `research-tool-through-affordance`（研究结果 → 证据 owner → 估值引用）。在它落地前，workspace 发布的是结构而非数字。

其中 `user-override-channel` 只完成了**读侧**：`HintReadoutContext.panorama_override` 已实现且覆盖直接胜出，但写侧（用户显式偏好如何进入这个字段、如何跨会话保留）要等 `decision_workspace` 从 SHADOW 晋升后接。目前 regime owner 恒传 `None`。

## 核验记录

- 2026-07-26：撰写前运行语义提取、个人条件、Prefix-KV、跨会话水合与工具结果回流相关测试，`63 passed`，未修改代码。
- 2026-07-27：P0–P3 落地。`tests/test_panorama_gate.py` + `tests/test_decision_workspace.py` 共 41 passed；`tests/contracts/` 2268 passed（import boundary 新增 `decision_workspace` tier）；`tests/test_final_wiring.py` 全过。仓库另有 6 条**先于本次改动即失败**的用例（`test_prompt_planner_participation_hint` 的 OPEN_LOOP_HANDOFF、`test_semantic_state_owners` 的 kill-switch、dlaas dispatch 的测试替身签名、feeling_about_other 晋升漂移、predictive heads 浮点舍入、no_lscb 品牌 token），已逐条 stash 比对确认与本次无关。
