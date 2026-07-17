# Semantic Grounding Evidence Spec（语义落地证据）

> Status: implemented（synthetic smoke lane 验证通过；hf substrate 真实 trace 运行 pending）
> Last updated: 2026-07-17
> 对应需求: R3, R4, R8, R11, R12, R15, R-PE

## 要解决的问题

当前系统有两个悬而未决、且现有 gate 体系不直接回答的问题：

1. **潜在抽象是否 grounded**：`z_t / β_t` 发现的 action family 是否对应真实语义动态（关系移动、承诺生命周期、common ground 变化），而不是表面措辞噪声的聚类。现有 promotion gate（`evaluate_learned_active_candidate`）检查 validation delta / latency / rollback / 控制臂方向，但没有一项直接测"latent family ↔ 语义 owner 状态变化"的对齐。
2. **语义追踪对 LLM 的真实依赖度**：9/9 semantic owner 的 typed proposal 当前主要来自 `LLMSemanticProposalRuntime`（hf 部署下开启，synthetic 下 NoOp）。如果关掉这个通道语义状态质量塌掉，说明 grounding 目前靠 LLM 的语义理解在撑，"涌现语义"的对外主张必须更保守；如果不塌，则是 typed 结构 + PE 闭环承担语义追踪的强架构证据。当前没有 matched 对照能回答这个问题（synthetic 与 hf 的轨迹分布不同，构不成对照）。

本 spec 冻结两个实验的设计。两者都是 **evaluation readout / evidence lane**（R12），不是学习源，不新增 runtime slot、owner 或 schema。

## 关键不变量

- **只读既有快照**：两个实验的全部输入来自已发布的 `temporal_abstraction` / `prediction_error` / `credit` / 语义 owner 快照与 snapshot replay export，属于 out-of-turn 分析，与 `emergent-action-abstraction.md` 的 replay 同一口径。分析层不重建任何 owner 内部状态（R8）。
- **readout-only**：grounding report 与消融结果不回灌任何学习路径（R12）；它们是 promotion 证据输入，不是 reward。
- **matched 对照强制**：消融两臂必须同 substrate fingerprint、同 seed schedule、同 scenario 脚本、同 turn budget；grounding 判据必须带 shuffled control，禁止只报单臂绝对值。
- **语义 delta 只来自 typed owner snapshot**：禁止从 transcript 文本用关键词重建"发生了什么语义变化"（no-keyword-matching 规则）。
- **diagnostic-first 升级路径**（R15）：两个 artifact 首版均为 non-gating reference artifact；只有在真实 trace 上口径稳定后，才通过独立收敛包把 grounding gate 加入 temporal backend 的 ACTIVE 晋升要求。

---

## 实验 1：Latent–Semantic Grounding Readout

### 回答的问题

> 一个 latent action family 激活/切换时，语义 owner 的状态变化是否呈现出该 family 特有的、可预测的、可迁移的签名？

### 数据面（全部既有产物）

按 turn 对齐三条序列：

1. **latent 序列**：`temporal_abstraction` 快照的 active family / `action_family_version`、`closed_segments`（`β_t` segment closure），以及 PE action context 中的 `segment_id / abstract_action_id / z_t_digest`；
2. **语义 delta 序列**：各语义 owner 相邻 turn 公共快照的 typed diff——`relationship_state`、`commitment`（lifecycle 迁移）、`open_loop`（新开/关闭）、`common_ground`（atom 增减）、`user_model`、`boundary_consent`。diff 是 owner 已发布字段的结构化比较，每个 owner 的 delta 向量化规则在导出器内一处定义；
3. **结果序列**：`prediction_error` 快照（magnitude、外部 outcome lineage）与 PE 派生的 segment credit records。

### 三个可测判据

**D1 区分性（discrimination）**：按 active family 分组语义 delta 向量，组间差异必须显著大于组内差异。
- 指标：`family_delta_separation` = 组间/组内方差比（或等价 cluster purity）；
- 控制：把 family 标签在 segment 边界内随机重排（shuffled-label control），真实标签的 separation 必须稳定高于 shuffled 分布（多次重排的高分位）。

**D2 领先性（lead）**：`β_t` segment closure 应领先或同步于语义 owner 的状态迁移，而不是滞后跟随（滞后意味着 latent 层只是在事后复述语义层已发生的变化）。
- 指标：switch 事件与语义 delta 事件的互相关峰值滞后 `switch_semantic_lag`（单位 turn，负值 = 领先）；
- 控制：对 switch 时刻做循环平移（shuffled-timing control），真实序列的领先性必须区别于平移分布。

**D3 迁移性（transfer）**：held-out 场景中被复用的 family，其语义 delta 签名必须与训练场景中同一 family 的签名一致。
- 指标：`signature_transfer_similarity` = 同 family 跨场景签名相似度，必须显著高于跨 family 相似度；
- 复用既有 proof harness 的 held-out family reuse 指标作为前置（family 得先被复用，才谈签名一致）。

### Artifact

`semantic_grounding_report.json`（schema_version `semantic-grounding-report.v1`），核心字段：

- `family_delta_separation` + shuffled-control 分布摘要；
- `switch_semantic_lag` 分布 + shuffled-timing 对照；
- `signature_transfer_similarity`（per-family + aggregate）；
- 每个 family 的语义签名摘要（owner 维度的 delta 中心向量，人类可审阅）；
- 覆盖统计：closed segment 数、非空语义 delta turn 占比、参与判定的 family 数；
- manifest / provenance（git sha、substrate fingerprint、seed、来源 artifact sha256），对齐 evidence bundle 口径。

导出器建议落位 `volvence_zero.agent`（与 `dialogue_option_discovery_report` / snapshot replay export 同层），进入 `EvidenceBundle.reference_artifacts`。

### Verdict 规则

- **retain**：D1、D2、D3 全部通过各自 shuffled control，且覆盖统计达门槛（≥50 closed segments、≥3 个被复用 family、非空语义 delta turn 占比 ≥ 0.3）；
- **weak**：D1 通过但 D2 或 D3 证据不足（覆盖不够或 CI 未站稳）；
- **fail / kill**：真实标签与 shuffled control 不可区分（family 与语义无关，抽象是表面聚类），或 D2 系统性滞后（latent 层只是语义层的回声）。fail 是对 "涌现抽象已 grounded" 主张的直接 kill 信号，必须如实进入 claim verdict，不允许换口径重跑到通过。

### 前置条件

- hf substrate 真实 trace（synthetic 下语义 owner 大多为空，报告会因覆盖门槛不足直接标 `insufficient-coverage`，不产生 verdict）；
- 语义 proposal 通道开启（即实验 2 的 on 臂配置）；
- 场景分布需覆盖多种深层结构（破裂-修复、承诺-兑现/违约、延迟结果），否则 D1 的组间差异没有机会出现——这是证据计划中"经历分布声明"的直接消费者。

---

## 实验 2：LLM-Proposal 依赖消融

### 回答的问题

> 同一批轨迹、同一 substrate 下，关掉 `LLMSemanticProposalRuntime`，语义 owner 状态质量和下游认知链路掉多少？

### 臂设计

| 臂 | 配置 | 说明 |
|---|---|---|
| `semantic-proposal-on` | hf substrate + `LLMSemanticProposalRuntime`（vertical 工厂现状） | 基线臂 |
| `semantic-proposal-off` | 同 hf substrate + 显式注入 `NoOpSemanticProposalRuntime` | 消融臂 |

注意耦合事实：session 构造时 LLM-backed semantic runtime 会自动派生 ToM 与 common-ground 的 LLM proposal runtime，因此本消融是**通道级**开关——off 臂同时关闭 semantic / ToM / common-ground 三个 proposal 面。v1 接受这个粒度（它回答的正是"LLM 语义感知通道整体的依赖度"）；细粒度分面消融（只关 semantic 保留 ToM，需显式注入拆开 auto-wire）留作 follow-up 臂。

后续可选第三臂 `semantic-proposal-swap`：换一个不同家族的 provider，测 grounding 对特定 LLM 的敏感度。v1 不要求。

### 控制变量（缺一不可，否则结果不可引用）

- 同 substrate model + weights fingerprint（residual 捕获路径两臂完全一致，只有 proposal 通道不同）；
- 同 seed schedule、同 scenario 脚本、同 turn budget、同 scripted user simulator；
- 两臂 artifact 均带 provenance，进入同一 comparison report。

### 指标（三层）

1. **语义 owner 状态质量**（主指标）：在带 ground-truth 的 scripted probe 上测——场景脚本预先声明"本场景应产生 1 个 commitment CREATE、第 5 turn 应 COMPLETE、应登记 1 条 boundary"，然后读 owner 快照核对。指标：per-slot 的 lifecycle 命中率、非空 record 覆盖率、错误状态率。复用 companion evidence C1–C5 与 `semantic_state.quality` harness 的既有 probe 形状，扩到 9 slot。
2. **下游链路**：PE 分布（语义来源的 prediction/settlement 数量与方向）、followup 触发精度（open_loop / commitment 驱动的 followup 是否还能形成）、`semantic_record_counts` 诊断读数。
3. **grounding 交叉读数**：两臂各跑一次实验 1 的 `semantic_grounding_report`——这是两个实验的核心耦合点，见下节。

### 结果解读（两个方向都可行动，不存在"实验失败"）

| 观察 | 结论 | 行动 |
|---|---|---|
| off 臂主指标大幅下降（lifecycle 命中率掉 > 0.3 级） | 语义追踪当前主要由 LLM proposal 承担 | 对外主张降级为"LLM-assisted typed semantic tracking"；把 proposal 质量纳入正式评估面；投资非 LLM typed source（外部结构化事件、embedding 提案）作为冗余 |
| off 臂主指标基本持平（掉 < 0.1 级） | typed 结构 + PE 闭环承担语义追踪 | 强架构证据，可支撑"语义 owner 不依赖单一 LLM 理解"的主张；LLM proposal 定位为增强而非地基 |
| 中间地带 | 按 slot 分化：报告 per-slot 依赖度排名 | 对高依赖 slot 单独补非 LLM source，低依赖 slot 维持现状 |

### Artifact

`semantic_proposal_ablation_report.json`（schema_version `semantic-proposal-ablation.v1`）：两臂 per-slot 质量指标、pairwise delta + bootstrap CI、下游链路读数、两臂各自的 grounding report 引用（sha256）、manifest / provenance。进入 `EvidenceBundle.reference_artifacts`，non-gating。

### 实施成本

低。off 臂只需 vertical 工厂暴露一个"跳过 `_build_llm_semantic_runtime_from_runtime`"的注入参数（或直接显式传 `NoOpSemanticProposalRuntime`），场景与 runner 全部复用现有 harness；主要成本是 scripted probe 的 ground-truth 声明扩到 9 slot。

---

## 两个实验的耦合关系

最重要的交叉读数是：**grounding 对 LLM 通道的敏感度**。

- 若 on 臂 grounding retain、off 臂 fail：latent 抽象的语义对齐本身依赖 LLM 喂进来的语义 delta——涌现抽象的"语义性"是二手的，主张必须收敛到"latent 层压缩了 LLM 感知的语义"；
- 若两臂 grounding 都 retain（off 臂靠外部 typed 事件与 PE 仍保持对齐）：latent 抽象独立于 LLM 感知捕获了语义动态——这是"涌现语义抽象"最强的一条证据；
- 若两臂都 fail：问题在 latent 层本身（或经历分布覆盖不足），与 LLM 无关，优先修 temporal 证据线而不是 proposal 通道。

## 接口契约

**消费的输入**：`temporal_abstraction` / `prediction_error` / `credit` / 九个语义 owner / ToM / common_ground 的已发布快照序列；snapshot replay export；scripted scenario ground-truth 声明；substrate fingerprint 与 seed manifest。

**产出的输出**：`semantic_grounding_report.json`、`semantic_proposal_ablation_report.json`，均为 read-only reference artifact，经 evidence bundle 分发。

**显式不产出**：不新增 runtime slot、不写任何 owner、不进入 reward、不改变 `evaluate_learned_active_candidate` 的现有 gate 列表（升级为 gate 输入需独立收敛包）。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|---|---|---|
| 依赖 | 时间抽象与内部控制 | 消费 `closed_segments` / family / `z_t_digest`；grounding fail 是 temporal backend 晋升的 kill 信号候选 |
| 依赖 | 语义状态一等 Owner | 语义 delta 全部来自 owner 公共快照；消融臂开关的是 proposal source，owner 单写者语义不变 |
| 依赖 | Prediction Error 主链 | 领先性与结果序列消费 PE 快照与 segment credit |
| 依赖 | 证据计划 | claim 注册、provenance、shuffled/matched control 口径、经历分布覆盖声明 |
| 协作 | Emergent Action Abstraction | 与 snapshot replay export 同一"既有快照 out-of-turn 导出"口径 |
| 协作 | 评估体系 | 属 F5 抽象质量 / F4 学习质量的证据面，readout-only |

## 实现映射

| 组件 | 位置 |
|---|---|
| 实验 1 核心（turn 证据抽取 + D1/D2/D3 + shuffled controls） | `packages/vz-runtime/src/volvence_zero/agent/semantic_grounding.py` |
| 实验 1 CLI（capture / offline 分析 + manifest） | `scripts/build_semantic_grounding_report.py` |
| 实验 2 通道级开关（vertical 工厂 choke point） | `lifeform_service.verticals`：`VZ_SEMANTIC_PROPOSAL_CHANNEL=llm\|noop`（非法值 fail loudly） |
| 实验 2 harness（9-slot scripted probe + 双臂 runner + 报告） | `packages/lifeform-evolution/src/lifeform_evolution/semantic_proposal_ablation.py` |
| 实验 2 CLI（双臂 + 两臂 grounding 交叉读数 + manifest） | `scripts/build_semantic_proposal_ablation_report.py` |
| 流水线编排（unit / smoke / hf 三 lane + summary.json） | `run_semantic_grounding_evidence.sh` → `scripts/run_semantic_grounding_evidence.py`（用法见根 `README.md`） |
| 测试 | `packages/vz-runtime/tests/test_semantic_grounding.py`、`tests/lifeform_e2e/test_semantic_proposal_ablation.py`、`packages/lifeform-service/tests/test_semantic_proposal_channel_switch.py` |

实验 2 的 probe 判据全部是 OBSERVE-immune 的 typed 字段检查（blocked / deferred / completed 状态、revision 计数、confidence-floored commitment slot），并按证据通道分层：`proposal-channel`（只有语义感知 runtime 能从用户话语满足）vs `typed-event`（外部 typed 事件经 adapter runtime 在两臂都生效，作为不变性对照）。synthetic smoke lane 的 on 臂用 scripted probe runtime 精确重放 ground-truth 声明；hf lane 用共享 substrate 包装的 `LLMSemanticProposalRuntime`（`--arm-runtime hf-llm`）。

## 变更日志

- 2026-07-17: 两个实验包实现完成。实验 1：`semantic_grounding` 模块 + CLI；实验 2：通道级 env 开关、9-slot scripted probe（proposal-channel / typed-event 双分层）、双臂 runner、`semantic_proposal_ablation_report.json`（含 case-level bootstrap CI、per-slot 依赖度排名、两臂 grounding 交叉读数）+ CLI。synthetic smoke lane 验证：on 臂 proposal-channel 命中率 1.000 / off 臂 0.000、typed-event 两臂持平（差异 0.000）——差分设计成立。hf substrate 真实 trace 运行 pending。
- 2026-07-17: 初始设计冻结。两个实验（latent–semantic grounding readout、LLM-proposal 依赖消融）的判据、控制、artifact schema 草案、verdict 规则与耦合解读确定；实现与真实 trace 运行 pending。
