# 人类世界模型 Thesis 第一阶段 —— Claim Registry（冻结版）

> Status: frozen claim registry（文档级；真跑 gate on GPU/keys）
> Last updated: 2026-07-14
> 对应需求: R2（稳定基底 + 自适应控制器）, R3/R4（时间抽象 / 潜在控制器）, R-PE, R7（双轨/关系）, R12（评估只读）, R15（可解释 + 可回滚证据）
> 关联 debt: **#87**（本 spec 的宿主债）· #29 / #37（EQ-Bench P10）· #48 / #71（judge bias）· #68（drive ablation）· #72（synthetic 不可外引）· #82 / #84（reference SUT / baseline）· #86（learned coverage / capacity→gain）
> 实现载体: [`companion-ablation.md`](./companion-ablation.md)（同基底 9-track 工具链）· [`evidence_program.md`](./evidence_program.md)（claim verdict / evidence bundle 宿主）

## 本 spec 的定位

`docs/specs/companion-ablation.md` 描述**怎么跑**（同基底 9-track 工程），`docs/specs/evidence_program.md` 是 claim verdict / evidence bundle 的**机器口径宿主**。本 spec 只做一件事：**把"人类世界模型 thesis 第一阶段"这句对外承诺冻结成一组可证伪、有 kill 条件、有结果分级的 retain claim**，杜绝把"流程跑通 / 单 benchmark 分数 / 局部 demo"误写成 "thesis proven"（debt #87 的根本诉求）。

> **红线口径**：真跑（P1/P2）产出 `first-stage-retained` verdict 之前，任何 pitch / deck / memo / press / leaderboard **不得**声明"人类世界模型已证明 / 认知回路已验证 / 低数据持续个体化已成立"。当前命中态见文末"当前状态"。

## 1. 冻结的 retain claim（5 条）

每条 claim 必须由 matched-control 的 bootstrap CI 保守下界判决：`ci_low = system.ci95_lo − control.ci95_hi`。`retain` = `delta > 0 且 ci_low > 0`；`weak` = `delta > 0 但 CI 触零`；`fail` = `delta ≤ 0`；`insufficient_data` = arc/seed 不足。

| Claim ID | 命题 | 对照臂 | retain 条件 |
|---|---|---|---|
| `claim_pipeline_gt_raw` | 完整 pipeline 优于裸 substrate（任何层是否有用的下界） | `raw` | delta>0 且 ci_low>0 |
| `claim_gt_standard_layers` | 优于标准 memory wrapper **且** 优于标准 agent 框架（回应"套个 wrapper 你们还赢吗"） | `memory-only` / `RAG` / `agent-framework` | 三条 pairwise 均 retain |
| `claim_component_causal_contribution` | PE / ETA / 主动学习各自对增益有**正向因果贡献**（不是只靠 raw substrate 或 prompt wrapper） | `PE-off` / `ETA-off` / `active-learning-off (random-sampling)` / `LoRA-adapter` | 每条组件消融 pairwise 均 retain（关掉该组件显著变差） |
| `claim_training_adds_value` | 训练 bootstrap 有增量 | `volvence-cold` | delta>0 且 ci_low>0 |
| `claim_heldout_cohort_stable` | 优势在 held-out 场景跨 seed 稳定 | held-out + 多 seed | 相对 CI 半宽足够紧 + arc/seed 足量 |

> `claim_component_causal_contribution` 是 debt #87 相对既有 companion-ablation 四 claim **新增的一条**——它把 thesis 的三条护城河（#88 时间抽象 / #89 记忆 / #90 主动学习）逐个做因果切分，防止"完整 pipeline 赢了但赢在别处"。

## 2. 统一 matched-control matrix（8 臂）

所有臂**同一份冻结 substrate、同 prompt budget、同 context budget、同 judge / human protocol**。分差只能归因于 substrate 之上的层。

| 臂 | substrate 之上的层 | 隔离的变量 | 当前实现 |
|---|---|---|---|
| `raw` | 无（裸 substrate） | 任何层 vs 无层的下界 | ✅ [`companion-ablation.md`](./companion-ablation.md) `raw` track |
| `memory-only` | 标准 memory wrapper（summary+user_model+episodic） | "标准记忆封装"基线 | ✅ `ref-harness`（H-A/H-C 子集） |
| `RAG` | embed 检索增强 | "标准检索"基线 | ✅ `ref-harness` H-B（embed retrieval） |
| `agent-framework` | 标准开源 agent 框架 | "标准 agent 框架"基线 | ✅ `camel` track |
| `LoRA-adapter` | 冻结 substrate + persona LoRA，无控制器层 | "只微调不控制"基线 | ✅ serving arm `companion-lora-adapter`（真 LoRA artifact 仍 gate on #41 GPU bake） |
| `PE-off` | 完整 pipeline 关 PE 主链 | PE 的因果贡献 | ✅ serving arm `companion-pe-drive-off` |
| `ETA-off` | 完整 pipeline 关时间抽象 | ETA 的因果贡献 | ✅ serving arm `companion-eta-off`（关联 #88） |
| `active-learning-off` | 完整 pipeline 用随机采样替代主动请求反馈 | 主动学习的因果贡献 | ✅ serving arm `companion-active-learning-off`（random-sampling baseline） |
| `volvence-cold` | 完整 pipeline，无训练 bootstrap | 控制器结构本身 | ✅ `volvence-cold` track |
| `volvence` | 完整 pipeline + 训练 bootstrap | **被验证系统** | ✅ `volvence` track |

**实现说明**：同基底 5-track 工具链（`raw / ref-harness / camel / volvence-cold / volvence`）已 land 并通过 P0 wiring smoke（`claim_pipeline_gt_raw` / `claim_gt_standard_layers` / `claim_training_adds_value` / `claim_heldout_cohort_stable` 就位）。**2026-07-12 起，`claim_component_causal_contribution` 的四臂已完成 P0 wiring**：`compare_companion_ablation.py` 支持 `pe-off` / `eta-off` / `active-learning-off` / `lora-adapter` 四个 component track 并输出第 5 条 claim verdict（任一臂 fail → kill；缺臂 → 最多 weak，且 `first-stage-retained` 必须四臂全 retain）。**2026-07-14 起，四臂真实 serving profile 已迁到同基底服务矩阵**：`scripts/companion_bench/reference_systems.same_substrate_ablation.yaml` 声明 9 track；`serve_same_substrate_ablation.{sh,ps1}` 以单个 `lifeform-serve --ablation-bundle` 进程承载 `volvence / cold / component` 六个 vertical，并通过 `?vertical=` 路由共享同一冻结 HF runtime；ref-harness/camel 仍独立进程 upstream 到 `:8000/v1?mode=raw`。P1/P2 runner 的 health / topology / fingerprint / summary gate 也扩到 9 track。**仍缺**：真 LoRA artifact（#41 GPU bake）与完整 P1/P2 真跑 verdict，故 claim 3 的因果 verdict 依旧 pending。

## 3. 证据门槛（每条 claim verdict 必备）

任一缺失 → verdict 不可外引：

- **provenance**：manifest + git sha + 依赖版本 + seed schedule + artifact sha256/size（对齐 `evidence_program.md` provenance 口径）。
- **matched control**：同 substrate（`assert_same_substrate.py` fingerprint fail-loud 校验）、同 prompt/context budget、同 judge/user-sim protocol。
- **统计**：pairwise effect + uncertainty interval（bootstrap CI）+ 保守下界 `ci_low`。
- **leakage attestation**：held-out 文本无泄漏；hidden family label 不出现在 transcript / user input / response。
- **judge-bias check**：裁判 / 用户模拟器非同家族 substrate（#71/#72）；judge robustness/calibration 证据在档（#48）。
- **human anchor**：至少一组 blinded human review 或跨家族 judge sanity check（对齐 `claim_external_human_legibility`）。

## 4. 结果口径分级（四态，只有第三态可宣称 thesis）

| 态 | 含义 | 可对外说的话 |
|---|---|---|
| `wiring-ready` | 流程/schema/回归跑通，未跑真 substrate | "验证框架与 ablation 设计已就位" |
| `weak-positive` | claim 1/2/3 mean delta 全正但未全 retain（或 stability 未站稳） | "有方向性正信号，未达 retain" |
| `first-stage-retained` | claim 1–5 全 retain + 证据门槛全绿 | "人类世界模型 thesis **第一阶段** retained" |
| `world-model-extension-ready` | **本链永不自动给出** | 需物理/具身侧独立 benchmark；人类侧成功不能外推 |

## 5. Kill 条件（显式化）

命中任一 → 按 #87 把 thesis **收缩为 product-memory / companion company 口径**，并同步降级 [`human-world-model-thesis-2026-06.md`](../../research/strategy/human-world-model-thesis-2026-06.md)：

1. `claim_pipeline_gt_raw` fail —— 层相对裸 substrate 无净增益。
2. `claim_gt_standard_layers` fail —— 打不过标准 memory/RAG/agent-framework（"套 wrapper 就够了"）。
3. `claim_component_causal_contribution` fail —— 主动学习不优于随机采样，或控制器不优于 memory/RAG，或 PE 信号无法稳定定义，或增益主要来自 raw substrate / prompt wrapper。
4. `claim_training_adds_value` fail —— 训练 bootstrap 无增量。

> Kill 条件不是失败叙事，而是**诚实边界**：它保证 verdict 一旦为负，对外口径立即收缩而不是被埋进总通过率。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|---|---|---|
| 实现依赖 | [`companion-ablation.md`](./companion-ablation.md) | 同基底 9-track 工具链承载 core controls（claim 1/2/4/5）与 component arms（claim 3） |
| verdict 宿主 | [`evidence_program.md`](./evidence_program.md) | claim verdict / evidence bundle / provenance 机器口径 |
| 组件因果来源 | #88（ETA torch）· #89（CMS）· #90（主动学习） | `claim_component_causal_contribution` 的三条组件消融对应这三条护城河债 |
| 只读锚 | `claim_external_human_legibility`（#51 external anchor） | human review 只作对照真值，不回灌学习链路（R12） |
| 明确排除 | 物理 / 具身世界模型 | `world-model-extension-ready` 需独立 benchmark，不在本链 |

## 当前状态（2026-07-14）

- **命中态**：`wiring-ready`。claim 1–5 工具链全部 land + 9-track P0 wiring smoke 通过（含 component 四臂 + 第 5 条 claim verdict）。
- **未跑**：全部真跑 verdict（P1/P2）gate on GPU + 跨家族裁判 keys（预算批准，运维步骤）；component 四臂 serving 已落地并收敛到单 substrate owner，但真 LoRA artifact（#41）与真 trace verdict 仍 pending。
- **可外引结论**：**无**。在 `first-stage-retained` 之前只能说"设计与 ablation 框架已就位"。

## 变更日志

- 2026-07-14: 9-track serving 拓扑收敛为单 `lifeform-serve --ablation-bundle`
  substrate owner + `?vertical=` route selection，替代 8002–8005 component
  进程。`serve_topology.json` / manifest topology / vertical probes 成为 P1
  readiness 的一部分，确保同基底证据不再退化为"同权重、多 GPU 副本"。
- 2026-07-12: component 四臂 P0 wiring 落地。`compare_companion_ablation.py` 新增 `pe-off` / `eta-off` / `active-learning-off` / `lora-adapter` track 与 `claim_component_causal_contribution` verdict（fail → kill；缺臂 cap 至 weak；`first-stage-retained` 现要求 5 claim 全 retain——之前 4-claim 即可 first-stage 的口径是对冻结 registry 的实现落差，已修正）；`run_same_substrate_ablation.py` p0-smoke 扩到 9 track，P1/P2 依 roster submission id 自动纳入 component 臂。测试：`tests/companion_ablation/`（component retain / kill / 缺臂 cap / 9-track smoke）。
- 2026-07-05: 初始冻结版。把 debt #87 推荐的 5 条 retain claim（新增 `claim_component_causal_contribution`）、8 臂 matched-control matrix、6 项证据门槛、4 态结果分级、4 条 kill 条件一次性冻结为 thesis 第一阶段 claim registry SSOT；登记进 [`00_INDEX.md`](./00_INDEX.md)。文档级交付（零 GPU），真跑 verdict 仍 gate on GPU/keys。
