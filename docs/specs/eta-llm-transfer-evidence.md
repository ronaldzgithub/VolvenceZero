# ETA 迁移 LLM 四级阶梯证据 Spec

> Status: active（Stage 1 Gate 1 = **PASS**（2026-08-03）。Stage 2 三轮判读全部封存：v1 FAIL 定罪仪器（哈希指纹计划载体，信息天花板 0.18 < 0.25）；v2 FAIL 定罪判据 regime 错配（实质两条件 PASS，0.944 / 因果 +4.3pp）；v3（新 seed 20260804 + retention 判据，prereg `2f3b3bf4…`）字面仍 **FAIL** 但败因**反转**——`2×chance`（0.967 = 7.7×）与 retention（late 0.918 / 衰减 0.077）双 PASS，败在因果对照：**未经任何续训的裸 Qwen 基底自己读出 0.977**，续训无超越余量。实质命题"0.5B 残差流可线性承载 active subgoal"被基底臂直接证实且跨 seed 复现（0.901 / 0.944 / 0.977 / 0.967）；败掉的是"续训必要性"这一对照设计前提。字面 verdict 按预注册封存。**2026-08-03 用户程序级裁定**：Gate-2 看门目的（进入 Stage 3 前确认残差可承载层级）已实质达成，Stage 3 解锁——在续训基底上重审 rate–distortion（预注册 `artifacts/eta_stage3_prereg_v2_20260803/`，权威扫已启动））
> Last updated: 2026-08-03
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
| `claim_eta_rate_axis_instrument_valid` | 在 seeded 富语料上，frozen 臂 rate 轴对 alpha 做信息–精度交易 | Gate 1 → **PASS**（2026-08-03） |
| `claim_llm_residual_carries_subgoal_hierarchy` | 领域续训后的冻结 LLM 残差流可线性解码 active subgoal，且优于裸基座 | Gate 2：v1 FAIL（仪器定罪）；v2 FAIL（判据 regime 错配）；v3 FAIL（因果对照反向失效：裸基底 0.977 已在天花板，无超越余量）。**前半主张（残差可承载）被基底臂跨 seed 证实**；后半（续训必要）被证伪。阶梯处置待决策 |
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
| v3+gated 权威扫（部分完成后中止，见 v4 根因） | `artifacts/eta_stage1_gate1_v3_gated_20260802/` |
| v4 staged-plan probe（step-0 无泄漏 + arrival 可读） | `artifacts/eta_step0_plan_probe_v4_20260802/` |
| v4 缩减筛查（非权威，40 updates） | `artifacts/eta_stage1_gate1_v4_screen_20260802/` |
| v4 预算诊断（非权威，300 updates，暴露连续门走私） | `artifacts/eta_stage1_gate1_v4_budget_20260803/` |
| v4+hard-st 缩减筛查（非权威） | `artifacts/eta_stage1_gate1_v4_hardst_20260803/` |
| v4+hard-st 权威预注册（冻结，门槛不变） | `artifacts/eta_stage1_gate1_v4_hardst_20260803_prereg.json` |
| **v4+hard-st 权威扫（Gate 1 = PASS）** | `artifacts/eta_stage1_gate1_v4_hardst_auth_20260803/` |

**当前状态（2026-08-03）**：v4+smooth+switch-gated+hard-st 权威扫
**Gate 1 = PASS**（`artifacts/eta_stage1_gate1_v4_hardst_auth_20260803/
gate1_assessment.json`）。

- spearman = −1.000（门槛 −0.8）、rate_span = 1.933（门槛 0.30，约 10× 基线）
- never-switch 崩塌解除：hard switch 频率 0.12–0.96，heldout boundary F1
  在每个 alpha 均 > 0（0.240–0.671），为首个 switching 存活的权威扫
- 边界对比在高 rate 压力下涌现（alpha=3.0：边界门概率 0.199 vs 延续 0.050）
- **方向性加分**：frozen 臂检出近垂直 gap（74.4% distortion 改善集中于
  19.6% rate 区间，alpha 1.0→0.3，过预注册 drop/rate/noise 门槛）；但
  gap 区内 boundary F1（0.394）未高于区外（0.537），且缺 joint 臂对照——
  完整 gap 判据属于 Gate 3，不得从本 artifact 主张
- 处置：Stage 2（领域续训 + probe）按阶梯解锁

**历史（2026-08-02）**：缩减权威扫 Gate 1 = FAIL（spearman −0.657、span
0.585、α=1.0 回弹、零切换）；该负结果保留封存，被上述 v4+hard-st 权威扫
按"修尺子后重跑"的规定动作取代。

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

**v3 的结构性激励缺陷（v3+gated 权威扫 7/18 cell 后中止）**：v3 把全部计划信息
一次性写在 step-0，而 step-0 门本就强制为 1、16 维 z 足以容纳约 6 bits 的计划身份，
因此 Eq.3 的最优解是“step-0 编码一次、之后永远 keep”——中途切换传输不了编码器
没有新收到的信息，只会白付 KL。部分权威扫的读数与该论证一致：alpha 0.01→0.1 间
rate 从 `0.463` 单调压到 `0.086`（gated rate 定义工作正常），但 distortion 钉在
`1.43–1.47`（baseline `2.54`）、hard switch 频率与 boundary F1 全为 0。这不是优化
失败，是该协议下 never-switch 即最优；论文的边界涌现前提是**信息在轨迹中途持续
到达**且 decoder 无法自行做条件推理。据此注册
`partially-observable-staged-plan.v4`（分段揭示）：step-0 只写第一个 objective，
到达每个 objective 的观测步写出下一个（"Reached white. Next objective:
purple."），走廊步保持严格局部。never-switch 控制器在第二段物理上无法携带
第二目标（该信息不在 step-0 输入中），在真边界处重新编码可赢得实测约 1.0 nats 的
steering gap，激励恰好落在真边界。v4 probe
（`artifacts/eta_step0_plan_probe_v4_20260802/`，200 train routes、width 8、
n_input 16）验证信息时刻表：step-0 target-1 = `1.000`、target-2 =
`0.089–0.103`（null `0.085–0.121`，无泄漏）；first-arrival 步 target-2 =
`1.000`（null 约 `0.07`，揭示可读）。v4 尚未经过正式 Gate 1；任何缩减筛查
通过都不得写成 ETA 证据，正式判定须另行冻结 v4 预注册后重跑权威扫。

### Stage / Gate 2 — 领域续训 + 线性分类 probe

**问题**：补课后的 Qwen 残差流是否携带子目标信念？（对齐论文附录 B）

**做法**：

- 语料：`scripts/run_eta_stage2_corpus.py`（仅 train-split ordering，与
  Stage 3 heldout 不相交，带 SHA provenance）
- 续训：`continued_pretrain_and_merge`（LoRA → merge → 冻结落盘 + 权重指纹）
- probe：`fit_linear_classification_probe` + `scripts/run_eta_stage2_probe.py`

**仪器版本**（2026-08-03 审计后分化，runner 以 `--document-protocol` /
`--probe-protocol` / `--layer-selection` 显式选择，legacy 保留可回滚）：

- **v1（legacy）**：文档 `render_eta_route_documents`、probe
  `eta_route_probe_rows`（单步观测），计划载体是语料生成器的哈希指纹
  `_context_sentence`（ordering 哈希采样 8 个主题词，构造性不可读——与
  Gate-1 定罪的 v2 协议缺陷同类）；读出层固定最后一层。
- **v2**：文档 `eta_stage2_documents`、probe `eta_stage2_probe_rows`
  （owner：`eta_rate_distortion_evidence.py`，直接复用 rate-distortion v4
  staged-plan 协议渲染作 SSOT 表面）。probe 输入为累积轨迹前缀
  （过去步含专家动作 + 当前步裸观测，对齐论文附录 B 的轨迹处理 regime），
  标签在 v4 下是可见文本的确定函数；读出层由 train rows 的确定性
  case-id fit/val 切分预选（heldout 不参与）；超长文本 fail-loud 不截断。

**通过门槛**：

- 补课 heldout probe 准确率 ≥ 2× 随机水平
- 第二条件按仪器版本分化：v1/v2 为"准确率随观测前缀增长而上升"
  （推断累积 regime）；v3 起为 `retention.v3`（分段揭示 regime：late 桶
  ≥ 2×chance 且 early−late ≤ 0.15）
- 显著高于未补课原始 Qwen（因果对照）

**FAIL 处置**：LLM 迁移路线整体 kill，出报告收官；不进入 Stage 3。

**预注册**：`artifacts/eta_stage2_gate2_prereg_20260803/`（取代 08-02 草稿）
**产物**：
`artifacts/eta_stage2_corpus_20260803/`（语料 provenance，content sha256
`a89b7015…`）、`artifacts/eta_stage2_pretrain_20260803/` 与
`artifacts/eta_stage2_merged_20260803/`（续训 merged 冻结基底，权重指纹
`08472c6d…`）、`artifacts/eta_stage2_probe_20260803/probe_manifest.json`
（双臂 probe + gate2 判定）。

**v1 仪器读数（封存）**：**FAIL（2026-08-03，`gate-2-fail-kill-llm-transfer`）**。
续训把语料 next-token loss 由 2.610 压到 0.119（2000 步 / 120 文档 /
15737 token，near-memorization），但预注册最后一层线性 probe 在 8 类
（chance 0.125 / majority 0.166）heldout 上：续训臂 acc `0.131`、裸 Qwen 臂
`0.166`（后者恰等于 majority，probe 塌到多数类）。三条件里
`2×chance（≥0.25）` **否**、`续训 > 基线` **否**（0.131 < 0.166）、
`随前缀上升` 是——两否即 FAIL。即便用"全 24 层挑最优"读法
（base 0.214 / pretrained 0.202，非合规读出），前两条仍双否。

**仪器审计（2026-08-03，收窄 v1 FAIL 的解释权）**：v1 仪器的 heldout 信息
天花板经离线复算为**构造性低于及格线**——把非指纹可见信息（当前位置 /
可走边 / 已完成目标）在 train 上拟合到贝叶斯极限的查表预测器，heldout 只有
`0.1805`（30.2% 的 heldout 局部特征组合 train 中未出现；heldout 自身 oracle
上界 0.558，多行组 0.367），**低于 2×chance = 0.25 的及格线**。实测两臂
0.131–0.214 恰在该天花板附近：模型已把可读信息基本榨干，**是尺子没有把计划
可读地呈现，不是残差流装不下**。因此 v1 FAIL **定罪仪器而非基底命题**；
"0.5B 残差不载子目标"的早先措辞过强，据此收窄。v1 封存 artifact 不改动。

**v2 仪器重审**：预注册
`artifacts/eta_stage2_gate2_prereg_v2_20260803/`（sha256 `c0a54454…`，
8 个 owner 源文件真实 sha256 锁定，含仪器审计数字与 v2 天花板验证：heldout
文本→标签 499/499 组全确定、745/745 行 subgoal 在前缀显式揭示、ceiling
1.0；probe 文本最长 583 token → `--max-length 640`，文档最长 442 token →
续训 `--max-length 512`）。v2 下三门槛不变；**由于计划已可读（ceiling
1.0），v2 的 FAIL 才真正定罪基底命题**。语料
`artifacts/eta_stage2_corpus_v2_20260803/`（120 文档 / 20110 词 / 重叠 0 /
content sha256 `1caeac3a…`）；续训 merged
`artifacts/eta_stage2_merged_v2_20260803/`（权重指纹 `063077b7…`，
initial_loss 0.967 → final_loss 0.020，2000 步 / 35436 token）。

**v2 仪器读数（2026-08-03，`artifacts/eta_stage2_probe_v2_20260803/`，
封存）**：verdict 按预注册字面 = **FAIL**，但结构与 v1 完全不同——两条
**实质**条件决定性通过，仅第二条件败：

- `2×chance（≥0.25）`：**PASS**——续训臂读出层（train-split 选层 4）heldout
  acc `0.944`（= 7.5× chance；majority 0.188）；裸 Qwen 臂（选层 6）
  `0.901`。全层扫描：base 0.795–0.976、pretrained 0.788–1.000（后段层
  0.99–1.00）。**残差流大幅承载 active subgoal，v1 审计结论被实测证实**。
- `续训 > 基线`：**PASS**——0.944 > 0.901，因果对照成立（+4.3pp，且后段层
  续训臂达 1.00）。
- `随前缀上升`：**FAIL**——early 0.979 / late 0.879（base：0.985/0.747）。
  诊断：该条件为 v1/论文的**推断累积** regime 设计（子目标须从行为轨迹
  逐步推断，前缀越长信念越准）；v2 分段揭示制度下标签是显式给出的，无可
  累积推断，只有跨 token 的**保持衰减**（late 桶 0.879 仍为 chance 的
  7 倍）。属仪器判据与制度的错配，非"残差装不下层级"的证据。
- 附注：train-split 选层规则在 val 全饱和（多层 val=1.0）时平局取最低层，
  选中层 4/6 而非 heldout 更强的后段层（0.976/1.00）；这是预注册规则的
  诚实代价，不回溯更改。

**v2 处置**：v2 verdict 按 `prohibited_after_execution`（执行后禁改条件）封存
为 FAIL；实质读数（0.944 / 0.901 / 因果 +4.3pp）作为"残差可承载"的强方向性
证据登记。经用户决策注册 v3 预注册，把第二条件修为分段揭示制度下的保持类判据。

**v3 重审（2026-08-03，用户授权）**：预注册
`artifacts/eta_stage2_gate2_prereg_v3_20260803/`（sha256 `2f3b3bf4…`，8 个
owner 源文件 SHA 锁定）。仅改动第二条件：`rises-with-prefix.v1` →
`retention.v3`（late 桶 ≥ 2×chance **且** early−late ≤ 0.15；由
`assess_gate2 --gate-conditions` 承载，三分支单测覆盖）。**forking-paths
防护**：因判据修订动机产生于观察 v2 读数之后，v3 终审使用**全新 corpus
seed 20260804**（新环境图 + 新路线切分，判据在任何新 seed 模型读数存在前
冻结）；v1/v2 封存 artifact 不做再判读。新 seed 天花板复验：heldout
文本→标签 439/439 组全确定、664/664 行显式揭示、ceiling 1.0；probe 文本
最长 507 token（< 640）、文档最长 444（< 512）。语料
`artifacts/eta_stage2_corpus_v3_20260803/`（120 文档 / 重叠 0 / content
sha256 `d78281b5…`）；续训 merged
`artifacts/eta_stage2_merged_v3_20260803/`（权重指纹 `0e387aba…`，
initial_loss 1.034 → final_loss 0.023，2000 步）。

**v3 仪器读数（2026-08-03，`artifacts/eta_stage2_probe_v3_20260803/`，
封存）**：verdict 按预注册字面 = **FAIL**，但败因与 v1/v2 完全不同且
**方向反转**：

- `2×chance（≥0.25）`：**PASS**——续训臂（train-split 选层 12）heldout acc
  `0.967`（= 7.7× chance）。
- `retention.v3`：**PASS**——early 0.995 / late 0.918，衰减 0.077 ≤ 0.15，
  late 桶为 chance 的 7.3 倍。修正后的第二条件在新 seed 上一次通过。
- `续训 > 基线`：**FAIL**——**未经任何领域续训的裸 Qwen 基底（选层 21）
  自己读出 `0.977`**，高于续训臂 0.967。因果对照失效的方向不是"续训没
  学到"，而是**基底已在天花板、无超越余量**（v2 seed 上 base 0.901 尚有
  余量故该条通过；新 seed 图结构更利于基底）。
- 诊断：`beats_base` 的设计前提是"裸基底弱、续训补齐领域前提"。v4 可读
  协议下计划就在文本里，任何有能力的 LM 都会把它线性编码进残差——基底
  臂 0.977 恰恰是**实质命题"残差可承载子目标层级"的最强证据**（无需任何
  领域适配）。败掉的是"续训必要性"这一对照假设，不是基底命题。

**Gate-2 三轮总结**：实质命题"0.5B 残差流可线性承载 active subgoal"在
两个独立 seed、四个臂上复现（0.901 / 0.944 / 0.977 / 0.967，chance
0.125）；三次字面 FAIL 分别定罪仪器（v1，计划不可读）、判据（v2，regime
错配）、对照设计（v3，基底天花板）。v3 verdict 按
`prohibited_after_execution` 封存，不注册 v4 重判。**阶梯层面处置待用户
决策**：预注册决策规则的字面含义是 kill LLM 迁移路线；但该规则的论证
前提（"可读仪器 + 匹配判据下的 FAIL 定罪基底"）被 v3 的失败方向证伪——
基底不是装不下，是已经装着。可选处置：(a) 按字面收官出报告；(b) 在
程序层面裁定 Gate-2 的看门目的（"进入 Stage 3 前确认残差可承载层级"）
已实质达成，Stage 3 直接在裸 Qwen（或续训基底）上重审 rate–distortion
——那才是 ETA 机制本身的终审。

**2026-08-03 用户程序级裁定：取 (b)**。裁定依据：Gate-2 的看门功能是防止
"在不满足论文前提（残差携带子目标信念，论文附录 B）的基底上跑 Stage 3
然后把 FAIL 错误归因于 ETA 机制"；该前提已被两 seed 四臂 0.90+ 的读数
确立，且裸基底与续训基底都满足。三个字面 FAIL verdict 全部原样封存、
不改判；解锁的是阶梯推进权而非任何 claim 的 PASS 状态。Stage 3 基底取
**Stage-2 v2 merged**（seed 20260802 语料续训，与 Stage-3 评估语料同
seed，保持"补课基底"的原阶梯设计；裸基底对照留给 Stage-3 FAIL 时的
敏感性分析）。

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

**预注册**：`artifacts/eta_stage3_prereg_v2_20260803/`（取代 08-02 草稿：
草稿早于 v4 协议 / switch-gated / hard-st / 300 updates 的 Gate-1 修尺，
参数已过时）。v2 预注册镜像 Gate-1 权威扫全部参数（v4 + smooth +
switch-gated + hard-st，6 alpha × 3 seed，corpus seed 20260802 / 64+24
routes），仅两处不同：`--model-source` 指向 Stage-2 v2 merged 基底
（权重指纹 `063077b7…`），`--arms frozen joint`（joint 为 gap 判据的
强制有效性对照，36 cells）。

**当前状态**：权威扫已启动（2026-08-03），产物
`artifacts/eta_stage3_rate_distortion_20260803/`。

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
- 2026-08-03: v3+gated 权威扫 7/18 cell 后中止：rate 轴单调但 distortion 平坦、
  零切换。定位下一层根因为 v3 的信息到达时刻表（全部计划 step-0 一次到达 →
  never-switch 即 Eq.3 最优）；注册 v4 staged-plan 协议（step-0 只揭示第一
  objective，各 arrival 揭示下一个），v4 probe 证实 step-0 无 target-2 泄漏且
  arrival 步 target-2 完全可读。v4 缩减筛查与后续权威扫须另行冻结预注册。
- 2026-08-03: v4 预算诊断（300 updates，`eta_stage1_gate1_v4_budget_20260803`）
  证明 40 updates 是 distortion 瓶颈（alpha=0.01 distortion 1.248→0.919，
  distortion 首次响应 rate：0.92↔1.08），但同时暴露**连续门走私漏洞**：
  alpha=0.01 时门塌缩至 0.03 且边界/延续门概率零对比（新增
  `boundary_switch_probability` / `continuation_switch_probability` point 级
  遥测），控制器以微小门幅度逐步混入新后验信息、按门值分数支付 switch-gated
  KL，离散切换永不值得。修复：`StoreSSLTrainingSession` 新增
  `steered_gate_mode`（`continuous` legacy / `hard-st` 论文式离散开关 +
  straight-through 梯度），runner/CLI/预注册新增 `gate_mode` 字段并纳入预注册
  校验。单测覆盖：hard-st 下门不 fire 时 KL 塌缩至 step-0、ST 梯度可达 switch
  FFN、未知模式拒绝。v4+hard-st 缩减筛查为非权威诊断；正式判定须冻结含
  `gate_mode` 的新预注册后重跑权威扫。
- 2026-08-03: v4+hard-st 缩减筛查（`eta_stage1_gate1_v4_hardst_20260803`，
  300 updates，非权威）首次打破 never-switch：hard switch 真实发生
  （alpha=0.1/1.0 频率 0.172/0.388），heldout boundary F1 首次 > 0
  （0.352–0.678），门出现边界对比（alpha=1.0：0.454 vs 0.394），且 alpha=1.0
  以最低 rate 0.131 取得最低 distortion 0.947（选择性切换的 rate-distortion
  效率形态）。rate 轴 Spearman −1.0、span 2.377。据此冻结 v4+smooth+
  switch-gated+hard-st 权威预注册
  （`artifacts/eta_stage1_gate1_v4_hardst_20260803_prereg.json`，
  sha256 `b0d18f60…`，18 cells，updates=300，门槛不变：spearman ≤ −0.8、
  rate_span ≥ 0.30、switching gate 同前），权威扫
  `artifacts/eta_stage1_gate1_v4_hardst_auth_20260803/` 已启动。任何通过
  判定以该权威扫的 `gate1_assessment` 为准；筛查数字不得引用为证据。
- 2026-08-03: 权威扫完成，**Gate 1 = PASS**：spearman −1.000、span 1.933、
  heldout boundary F1 全 alpha > 0（0.240–0.671）、hard switch 0.12–0.96。
  frozen 臂另检出近垂直 gap（drop share 0.744 / rate share 0.196，
  alpha 1.0→0.3），但 gap 区内 F1 未高于区外且无 joint 臂，只作方向性
  记录，不构成 Gate 3 证据。判定文件
  `gate1_assessment.{json,md}` 已随 artifact 封存；Stage 2 解锁。运行途中
  首实例因机器休眠中断，经 `--resume` 从 12/18 checkpoint 续跑完成，
  configuration fingerprint 校验一致。
- 2026-08-03: 执行 Stage 2 全链（语料 → 续训+merge → probe），**Gate 2 = FAIL**
  （`gate-2-fail-kill-llm-transfer`）。参数与定稿预注册
  `artifacts/eta_stage2_gate2_prereg_20260803/`（sha256 `a2561f3b…`）逐字节一致：
  corpus_seed 20260802 / objective_count 8 / train_routes 120 /
  heldout_routes 60 / train_lengths [2,3] / heldout_lengths [3,4] /
  ridge_alpha 1.0 / 最后一层读出。语料 content sha256 `a89b7015…`（120 文档、
  9539 词、train/heldout 重叠 0）；续训 merged 权重指纹 `08472c6d…`
  （initial_loss 2.610 → final_loss 0.119，2000 步 / 15737 token）。probe：
  8 类（chance 0.125 / majority 0.166），续训臂最后一层 heldout acc `0.131`、
  裸 Qwen 臂 `0.166`（= majority，probe 塌到多数类）；三条件
  `2×chance≥0.25` 否 / `续训>基线` 否（0.131<0.166）/ `随前缀上升` 是。稳健性：
  全 24 层最优（base 0.214 / pretrained 0.202，非合规读出）下 `2×chance` 与
  `续训>基线` 仍双否。判读：0.5B 残差流领域续训后仍无线性可解码 subgoal 层级，
  next-token 近记忆化甚至略微恶化最后一层读出。据预注册 `decision_rules`：
  claim   `claim_llm_residual_carries_subgoal_hierarchy` 在 0.5B 被驳，整条
  LLM 迁移路线 kill、Stage 3 不跑；ETA 主张未永久摘除（保留 Gate 3 / 独立处置
  包）。规模敏感性为独立开放问题，须另立新预注册，不撤销本封存结果。
  `probe_manifest.json` 已随 `artifacts/eta_stage2_probe_20260803/` 封存。
- 2026-08-03: **Stage-2 仪器审计**，收窄同日 v1 FAIL 的解释权。代码审计发现
  v1 语料/probe 的计划载体是 `_context_sentence` 哈希指纹（ordering 哈希采样
  8 主题词，构造性不可读——与 Gate-1 定罪的观测协议 v2 缺陷同类，且该管线
  早于 v3/v4 修复、被 v1 预注册连同缺陷一起 SHA 冻结）。离线复算 heldout
  信息天花板：非指纹可见信息的 train-拟合贝叶斯查表在 heldout 仅 `0.1805`
  （未见局部键 30.2%；oracle 上界 0.558 / 多行组 0.367），低于 2×chance =
  0.25 及格线——**v1 的 2×chance 条件构造性不可过**，实测两臂 0.131–0.214
  恰在天花板附近（可读信息已被榨干）。判读修正：v1 FAIL 定罪仪器而非基底
  命题，"0.5B 残差不载子目标"措辞收窄；v1 封存 artifact 与其 verdict 字符串
  不改动。修复（仪器 v2，owner `eta_rate_distortion_evidence.py`）：
  `eta_stage2_documents` / `eta_stage2_probe_rows` 复用 rate-distortion v4
  staged-plan 渲染作 SSOT 表面（`_rate_distortion_observation_texts` 扩展
  返回 per-step active subgoal，v4 文本逐字节不变，全 42 项 runner 单测通过
  + 新增 3 项 v2 不变量单测：文本→标签确定、无未来目标泄漏、文档-probe 表面
  一致）；probe 输入改累积轨迹前缀（对齐论文附录 B）；读出层改 train-split
  确定性预选（v1 固定末层输出头对齐，base 臂实测中层 0.214 > 末层 0.166）；
  超长 fail-loud。runner 新增 `--document-protocol` / `--probe-protocol` /
  `--layer-selection` / `--max-length`，legacy 默认保留可回滚。v2 预注册
  `artifacts/eta_stage2_gate2_prereg_v2_20260803/`（sha256 `c0a54454…`）
  冻结仪器审计数字与 v2 天花板验证（ceiling 1.0），明确 v2 FAIL 才定罪基底
  命题；v2 全链已启动。
- 2026-08-03: **v2 仪器重审完成**（语料 `1caeac3a…` → 续训 merged
  `063077b7…`，initial 0.967 → final 0.020 → probe
  `artifacts/eta_stage2_probe_v2_20260803/`，参数与 v2 预注册逐字节一致）。
  实测：裸 Qwen heldout acc `0.901`（全层 0.795–0.976）、续训臂 `0.944`
  （全层 0.788–1.000，后段层 0.99–1.00）——**残差流大幅承载 active
  subgoal，仪器审计的"v1 定罪仪器"结论被实测证实**（v1 同一命题读数仅
  0.131/0.166）。三条件：`2×chance` PASS（7.5×）、`续训>基线` PASS
  （+4.3pp）、`随前缀上升` FAIL（early 0.979 / late 0.879，保持衰减而非
  推断累积——该条件为 v1 推断 regime 设计，在 v2 显式揭示制度下 regime
  错配）。按预注册字面 v2 verdict = **FAIL** 封存（执行后禁改条件）；实质
  读数登记为"残差可承载"的强方向性证据。待决策：是否注册修正第二条件为
  保持类判据的 v3 预注册；Stage 3 在正式 PASS 前保持锁定。另：train-split
  选层在 val 饱和时平局取低层（4/6），为预注册规则的诚实代价。
- 2026-08-03: **v3 重审完成（用户授权）**。判据修订：`assess_gate2` 新增
  `retention.v3` 第二条件（late ≥ 2×chance 且 early−late ≤ 0.15，CLI
  `--gate-conditions`，三分支单测）。forking-paths 防护：v3 预注册
  （`artifacts/eta_stage2_gate2_prereg_v3_20260803/`，sha256 `2f3b3bf4…`）
  改用全新 corpus seed 20260804 并在任何新 seed 读数存在前冻结，透明登记
  修订动机与时序；v1/v2 封存件不再判读。新 seed 天花板 1.0（439/439 组
  确定、664/664 行揭示）。全链：语料 `d78281b5…` → 续训 merged
  `0e387aba…`（1.034 → 0.023）→ probe
  `artifacts/eta_stage2_probe_v3_20260803/`。读数：`2×chance` PASS
  （0.967 = 7.7×，选层 12）、`retention.v3` PASS（0.995/0.918，衰减
  0.077）、`续训>基线` **FAIL——裸 Qwen 基底（选层 21）0.977 反超续训臂
  0.967**。verdict 按字面 = FAIL 封存，但败因方向反转：基底不是装不下
  子目标层级，而是**无需任何领域续训就已在天花板携带**（v4 可读协议下
  计划在文本中，能干的 LM 自然线性编码之）。实质命题跨两 seed 四臂复现
  （0.901/0.944/0.977/0.967）；三轮字面 FAIL 分别定罪仪器、判据、对照
  设计。不注册 v4 重判；阶梯处置（按字面 kill 迁移路线，或裁定 Gate-2
  看门目的已实质达成、Stage 3 以裸/续训基底解锁）升级为程序级用户决策。
- 2026-08-03: **用户程序级裁定 Stage 3 解锁**（取处置 (b)）：Gate-2 看门
  前提（残差携带子目标信念）已被两 seed 四臂 0.90+ 确立；三个字面 FAIL
  verdict 原样封存不改判，解锁的是阶梯推进权。冻结 Stage-3 v2 预注册
  `artifacts/eta_stage3_prereg_v2_20260803/`（取代 08-02 过时草稿）：
  Gate-1 权威扫同款参数（v4 + smooth + switch-gated + hard-st，300
  updates，corpus seed 20260802），基底换 Stage-2 v2 merged
  （`063077b7…`），双臂 frozen + joint 共 36 cells。权威扫
  `artifacts/eta_stage3_rate_distortion_20260803/` 已启动；判读以其
  verdict 为准（retain-eta-on-llm / kill 升级永久摘除）。
