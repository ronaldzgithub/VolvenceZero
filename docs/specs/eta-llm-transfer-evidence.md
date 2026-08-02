# ETA 迁移 LLM 四级阶梯证据 Spec

> Status: active（Stage 1 reduced Gate 1 = FAIL；Stage 2–3 未开跑；Stage 4 仅骨架）
> Last updated: 2026-08-02
> 对应需求: R2, R3, R4, R8, R10, R12, R15
> Owner 能力域: `vz-temporal`（元控制器 / rate–distortion）、`vz-substrate`（续训 merge / 分类 probe）、evidence lane（本 spec）
> 执行草稿: [`.cursor/plans/eta_迁移_llm_阶梯_59d33511.plan.md`](../../.cursor/plans/eta_迁移_llm_阶梯_59d33511.plan.md)
> 研究日志: [`research/eta/eta-segment-credit-evidence-plan.zh.md`](../../research/eta/eta-segment-credit-evidence-plan.zh.md)
> 能力边界: [`temporal-abstraction.md`](./temporal-abstraction.md)

## 要解决的问题

2026-08-01 的 rate-distortion 读数给出 mechanism-grade `kill-eta`
（`artifacts/eta_rate_distortion_20260801`），但留下两处未解释异常：rate 轴对
alpha 弱/非单调，以及 3 条 train 路线被记忆化。更关键的是：**ETA 原论文并不在
成熟 LLM 残差流上证明**，而是在作者从零预训练的小领域模型上证明；LLM 迁移是
论文展望，不是已完成实验。因此直接用裸 Qwen 判决 ETA，对主张不构成公平证伪。

本 spec 冻结一条 **claim-to-evidence 四级阶梯**：先校准仪器与数据（Gate 1），
再补领域前提并用线性 probe 前置检验（Gate 2），再在补课冻结基底上重审
rate–distortion（Gate 3），对话迁移仅在前三级全过后预注册（Stage 4）。
死在哪级停在哪级。`kill-eta` 在 Stage 3 通过前保持有效；本程序是
**evaluation / evidence lane**（R12），不改任何 production `WiringLevel`。

## 关键不变量

- **先补前提、再审判**：不得在 Gate 1/2 未过时用 Stage 3 结果撤销或坐实
  `kill-eta`。
- **尺子不行 ≠ 杀主张**：Gate 1 FAIL 的规定动作是修 posterior 方差参数化 /
  仪器数值稳定后重跑，**不是**永久摘除 ETA。
- **Gate 2 杀的是迁移路线**：补课后 heldout probe 仍读不出子目标 →
  “ETA-on-LLM”整体 kill，出报告收官；不等于否定论文在从零小模型上的结果。
- **Gate 3 才是主张终审**：补课冻结基底上出现近垂直 gap（且 joint 臂无 gap、
  gap 内 boundary F1 更高）→ 改判 `retain-eta-on-llm`；否则 kill 升级为
  “跨原生小模型与领域内预训练 LLM 两种机制均不成立”，主张永久摘除。
- **readout-only**：本阶梯产物不回灌学习、不进 reward、不自动 ACTIVE
  晋升 `vz-temporal` torch 路径（R12 / R15）。
- **稀有重路径唯一**：continued pretraining 必须走
  `continued_pretrain_and_merge` / rare-heavy + `ModificationGate` 口径；
  禁止另开基底训练后门（R2 / R10）。
- **预注册冻结**：每级跑前冻结 gate 阈值与 sweep 配置；禁止事后改 alpha
  网格 / 门槛冒充通过。
- **不进 runtime evaluation 包**：本程序不属于
  `vz-cognition.evaluation` 六族 readout；归 evidence_program 与本 spec。

## Claim

| Claim ID | 主张 | 所需门 |
|---|---|---|
| `claim_eta_rate_axis_instrument_valid` | 在 seeded 富语料上，frozen 臂 rate 轴对 alpha 做信息–精度交易 | Gate 1 |
| `claim_llm_residual_carries_subgoal_hierarchy` | 领域续训后的冻结 LLM 残差流可线性解码 active subgoal，且优于裸基座 | Gate 2 |
| `claim_eta_rate_distortion_on_domain_pretrained_llm` | 补课冻结 LLM + 元控制器出现论文式近垂直 gap | Gate 3 |
| `claim_eta_dialogue_transfer` | （contingent）对话域同样可迁移 | Stage 4；仅 1–3 全过 |

## 四级阶梯

### Stage / Gate 1 — 数据机制 / 仪器校准

**问题**：rate 不响应，是题太少背答案，还是尺子坏了？

**做法**：

- 环境 owner：`generate_hierarchical_environment` /
  `generate_hierarchical_routes`（train/heldout 目标顺序组合不相交）
- harness：`RateAxisResponse`（spearman(α, rate) + rate span）
- 跑 frozen 臂即可（joint 臂不是 Gate 1 必要条件；双臂 incomplete-sweep
  不自动等于 Gate 1 FAIL）

**通过门槛**（预注册）：

- `spearman(alpha, rate) ≤ -0.8`
- `rate_span ≥ 0.30`
- rate span 明显高于 7 路线基线（~0.20）

**FAIL 处置**：修 posterior 方差参数化 / 高 α 数值稳定 / 加预算后重跑；
**禁止进入 Stage 2**。

**预注册 / 产物**：

| 角色 | 路径 |
|---|---|
| 全量 Gate 1 预注册 | `artifacts/eta_stage1_rate_axis_prereg_20260802/` |
| 缩减 Gate 1 预注册 | `artifacts/eta_stage1_gate1_reduced_prereg_20260802/` |
| 可行性 pilot（非正式） | `artifacts/eta_stage1_rate_axis_pilot_20260802/` |
| 缩减权威扫（当前） | `artifacts/eta_stage1_gate1_reduced_20260802/` |

**当前状态（2026-08-02）**：缩减权威扫 **Gate 1 = FAIL**。

- spearman = −0.657（门槛 −0.8）
- rate_span = 0.585（过 0.30，约 3× 基线）
- 记忆化已消除（train ≈ heldout distortion）
- 非单调点在 α=1.0 回弹；下一步按预注册修方差参数化后重跑，不开续训

**2026-08-02 smooth/v2 重跑结果**：smooth posterior 与 v2 观测协议修复了 rate
轴（Spearman `-1.000`、rate span `0.691`），但 frozen 臂仍无 boundary F1 / hard
switch，rate–distortion 曲线近水平，Gate 1 仍 **FAIL**。因此不进入 Stage 2。
后续 `switch-gated` rate 仅作为论文 Eq.3 的独立候选修正：它把 KL 乘以实际 code
mix gate，使保持旧 code 的 segment 不付重复 rate；必须另行预注册、先过 surrogate
screen，再用真实 substrate 证伪，不能把 surrogate 通过写成 ETA 证据。

同日 step-0 plan identity probe 仅作 capacity diagnostic：target-2 线性 probe
准确率 `0.170`，shuffle null 为 `0.136`，仍接近 8 类多数基线 `0.193`，不能支撑
“latent 已携带第二子目标”主张。该结果提示 16 维 substrate summary 可能是瓶颈，
但不改变 Gate 1 失败规则，也不授权扩大训练预算。

### Stage / Gate 2 — 领域续训 + 线性分类 probe

**问题**：补课后的 Qwen 残差流是否携带子目标信念？（对齐论文附录 B）

**做法**：

- 语料：`scripts/run_eta_stage2_corpus.py`（仅 train-split ordering，与
  Stage 3 heldout 不相交，带 SHA provenance）
- 续训：`continued_pretrain_and_merge`（LoRA → merge → 冻结落盘 + 权重指纹）
- probe：`fit_linear_classification_probe` + `scripts/run_eta_stage2_probe.py`

**通过门槛**：

- 补课 heldout probe 准确率 ≥ 2× 随机水平
- 准确率随观测前缀增长而上升
- 显著高于未补课原始 Qwen（因果对照）

**FAIL 处置**：LLM 迁移路线整体 kill，出报告收官；不进入 Stage 3。

**预注册**：`artifacts/eta_stage2_gate2_prereg_20260802/`  
**当前状态**：机器与预注册已就绪；**Gate 1 未过，禁止开跑**。

### Stage / Gate 3 — 补课基底上重审 rate–distortion

**问题**：前提补齐后，ETA 在冻结 LLM 上是否成立？

**做法**：`scripts/run_eta_rate_distortion.py` 使用 Stage 2
`--model-source` + Stage 1 规模语料；规则沿用
`eta-rate-distortion-evidence.v1`。

**通过门槛**：

- frozen 臂近垂直 gap（drop share ≥ 0.5 且 rate share ≤ 0.25）
- joint 臂无 gap（有效性对照）
- gap 内 boundary F1 高于 gap 外

**PASS** → 撤销 mechanism `kill-eta`，改判 `retain-eta-on-llm`；复活
`vz-temporal` 须另开收敛包。  
**FAIL** → 永久摘除 ETA 主张；处置包：删主张、保留记忆/连续性、
`vz-temporal` → legacy（独立收敛包，不在本程序内自动执行）。

**预注册**：`artifacts/eta_stage3_prereg_20260802/`  
**当前状态**：未开跑。

### Stage 4 — 对话迁移（contingent，仅设计）

仅当前三级全过后启用。动作 = 对话行为，语料 = MSC 等；对话无子目标真值，
boundary F1 不可作门，退回 gap + heldout 泛化。

**本程序内只产出骨架，不跑实验**：
[`research/eta/eta-stage4-dialogue-transfer-prereg-skeleton.md`](../../research/eta/eta-stage4-dialogue-transfer-prereg-skeleton.md)

## 与其它证据面的边界

| 面 | 关系 |
|---|---|
| `temporal-abstraction.md` | 能力边界与实现 changelog；本 spec 是迁移证据 SSOT |
| `evidence_program.md` | claim registry / bundle 总则；本 claim 在此挂接 |
| `evaluation.md` / `vz-cognition.evaluation` | 六族 runtime readout；**不承载**本阶梯 |
| `companion-prediction-thesis-v3.md` | ETA 永久 kill 后的主科研/产品退路（预测式连续性） |
| Anthropic emotion-concepts 研究包 | 支持“残差可读可控”，**不**替代 Gate 2/3；affect ≠ `z_t` |

## 回滚

- 全部为 evidence lane；production WiringLevel 不变
- 补课基底为独立 artifact，原始 Qwen 路径不受影响
- Gate 1/2 FAIL 不触发主张永久摘除；仅 Gate 3 FAIL 或后续独立处置包才摘除
- `switch-gated` 只是可回滚的 rate 定义候选；默认 `per-step` 保留历史路径，候选
  失败时恢复默认，不得修改已封存的 smooth/v2 负结果

## 变更日志

- 2026-08-02: 本 spec 建立为四级阶梯 SSOT；收录缩减 Gate 1 FAIL
  （spearman −0.657 / span 0.585，α=1.0 回弹）与“先修方差参数化、不开 Stage 2”
  的当前处置。机器、预注册与 Stage 4 骨架此前已在 temporal /
  research evidence plan changelog 中记录。
- 2026-08-02: smooth/v2 复跑使 rate 轴通过但 switching gate 失败；新增
  `switch-gated` 作为 Eq.3 rate economics 候选，明确 surrogate 非权威与不启动
  Stage 2 的退出条件。
