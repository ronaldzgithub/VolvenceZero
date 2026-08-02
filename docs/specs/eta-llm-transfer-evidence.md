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
| 缩减权威扫 | `artifacts/eta_stage1_gate1_reduced_20260802/` |
| smooth/v2 权威扫（rate 轴过、switching FAIL） | `artifacts/eta_stage1_gate1_smooth_v2_20260802/` |
| v3+gated 预注册（冻结，门槛不变） | `artifacts/eta_stage1_gate1_v3_gated_20260802_prereg.json` |
| v3+gated 权威扫（排队/进行中） | `artifacts/eta_stage1_gate1_v3_gated_20260802/` |

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

随后在同一 frozen Gate-1 corpus 上把 route 数提高到 88、capture width 提高到 64，
并使用 Qwen `hf-local` 的 step-0 probe 复核容量：target-2 准确率 `0.148`，shuffle
null `0.102`，仍低于 8 类多数基线 `0.193`；target-1 也未超过多数基线
（`0.784` 对 `0.875`）。这只是更宽激活摘要的非权威诊断，不能升级为 Gate 2/3
证据，也没有改变 `kill-eta` 或生产 wiring。

后续把 capture width 提到完整 896 维，并将 train route 提到 200、heldout 保持 24，
在同一共享 capture 上比较 `n_input={16,64,192}`：target-2 分别为 `0.174`、
`0.179`、`0.174`，对应多数基线 `0.161`；target-1 分别为 `0.848`、`0.732`、
`0.714`，均不高于多数基线 `0.866`。该 bundle 仍是非权威容量诊断，target-2
没有形成可接受的计划身份读出，不能作为 Stage 2/3 的前置通过门。

同一全宽、224-route capture 还对比了当前 token-mean 与被丢弃的 last-token
residual；三个 `n_input` 窗口的 target-1/target-2 结果逐项相同。这说明当前
`residual_sequence` 在该 step-0 协议下没有提供与 token-mean 不同的可用观测，不能
把 pooling 选择当作已定位的根因；该结论同样只是诊断，不改变 Gate 1。

诊断脚本现另提供 plan-at-end-last-token counterfactual：保持 route 内容不变，仅把
计划文本移到 step-0 observation 的末尾，用来分离“信息未进入 capture”与“因果 recency
导致末 token 不携带计划”的解释。该对照尚未产生可权威 artifact；若运行，必须与
真实协议共享同一 corpus、fold 与 null seed，并独立记录为非权威诊断。

随后完成的 corrected pooling bundle
`artifacts/eta_step0_plan_probe_pooling_20260802` 对 `residual_sequence` 做显式
逐步平均，并加入 plan-at-end counterfactual（224 routes、896 维 capture、
`n_input={16,64}`）：last-token 的 target-2 为 `0.174/0.179`，token-mean 为
`0.125/0.147`，plan-at-end-last-token 为 `0.107/0.161`，8 类多数基线均为
`0.161`。这修正了前一份 one-step legacy 对照“逐项相同”的适用范围：它只说明
旧 `residual_sequence[-1]` 取法与旧 summary 相同，不代表完整序列平均也相同。三种
pooling 都没有形成可接受的计划身份读出，仍是非权威诊断。

最后一轮控制还修正了 probe label：`route_signature` 包含 corridor hop，不能直接把
前两项当作 objective targets；脚本现在从 environment 中筛出有序 objective 序列，并
另报 `plan_length_control`。224-route、`n_input=64` 的 control bundle 与 fixed-label
bundle 均未通过计划身份门：fixed-label target-2 在四种 pooling 下为
`0.080/0.125/0.147/0.116`，多数基线约 `0.143`，而 plan-length control 也低于其
多数基线 `0.665`。这只证明标签污染已被隔离，不能把 control/null 差异升级为 latent
计划身份证据。

脚本随后补充了 plan-alone（只保留计划文本）与 last-transition readability controls；
这两组只作为方法学自检，尚未形成新的物证 bundle，也不能替代 fixed-label target
结果。

最终的 224-route、896 维 bundle 已封存六种 pooling/control 组合（`n_input=16`）。
fixed-label target-2 最高为 `0.138`，plan-length control 最高为 `0.647`，均未超过
对应多数基线（约 `0.143` 与 `0.665`）；last-transition control 的真实与 null 均为
`1.000`，因此不提供可辨别性。该 bundle 仍是非权威诊断；后续新增的 raw-PCA 变体
尚未产生独立 artifact。

随后封存的
`artifacts/eta_step0_plan_probe_pca_20260802/step0_plan_probe.json` 在同一
224-route、896 维、protocol v2 corpus 上加入 32 维 raw-PCA 对照。raw-PCA 的
fixed-label target-2 在各 pooling/counterfactual 组合中最高为 `0.120536`，低于
8 类多数基线 `0.142857`；plan-length control 最高为 `0.522321`，低于多数基线
`0.665179`。因此“folding 丢失了计划信号”没有被支持，当前 v2 的 step-0 surface
本身仍不能读出第二子目标；该物证继续保持非权威诊断，不开启 Stage 2/3。

上述诊断还暴露了 v2 的协议缺口：v2 所谓的 `Route plan` 实际填入的是
`case.source_text` 的哈希指纹，并没有把 objective 顺序写给冻结 substrate。运行时
现在注册 `partially-observable-explicit-plan.v3` 作为候选协议，并由
`scripts/probe_eta_step0_plan_identity.py --observation-protocol ...`、预注册脚本和
执行脚本共享同一个 SSOT renderer。v3 只在 step-0 写出可读的 ordered objectives，
后续仍只暴露 current location 与 out-edges，不泄露 completed objectives 或每步
fingerprint；它尚未经过正式 Gate 1，不能把可读计划候选或任何 probe 改善写成 ETA
通过证据。v2 结果、v3 候选和生产 wiring 继续隔离，退出条件仍是先取得独立的
surrogate/真实 substrate 证据，再决定是否预注册新的正式 sweep。

随后产生的 v3 capacity diagnostic
`artifacts/eta_step0_plan_probe_v3_20260802/step0_plan_probe.json` 仅使用
`activation_width=8` 做开发性 sanity check，并非 Gate 1：token-mean 的 fixed-label
target-2 为 `0.727679`，raw-PCA 变体最高为 `0.933036`，均明显高于 8 类多数基线
约 `0.143`；plan-length control 最高为 `1.000`。这说明“把 objective 顺序以可读
计划写入 step-0”确实改变了输入面的可辨识性，但不能证明 latent 已跨边界携带计划、
不能证明 switch-gated rate objective，也不能授权 Stage 2/3。该 artifact 的
`production_promotion_authorized` 等 Gate 字段不存在，故继续按非权威 capacity
diagnostic 管理。

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
- 2026-08-02: step-0 probe 定位 v2 协议缺口（`Route plan` 为哈希指纹，冻结
  substrate 不可读）；注册 v3 协议（step-0 明文 ordered objectives，后续步保持
  v2 locality），v3 capacity diagnostic 显示 target-2 读出恢复
  （`0.554–0.933` 对 null 约 `0.1`）。冻结 v3+smooth+switch-gated 预注册
  （`artifacts/eta_stage1_gate1_v3_gated_20260802_prereg.json`，门槛不变：
  spearman ≤ −0.8、rate_span ≥ 0.30、switching gate 同前），权威扫在 MPS
  设备释放后启动。任何通过判定仍以该权威扫的 `gate1_assessment` 为准。
