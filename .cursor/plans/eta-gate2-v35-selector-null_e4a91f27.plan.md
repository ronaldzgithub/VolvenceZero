# Gate 2 完整分析计划：v35 selector-vs-permutation-null 收敛包及后续路径

> 状态：**v35 已完成并通过**（2026-07-29，run 36 min）。§1–§6 在读取结果之前写定
> （预注册计划，判定标准未回溯修改）；结果见 §8。
> verdict `causal-supported`、`promotion_allowed=true`——按 §3.3 第一行判定，
> Gate 2 causal 层在本 packet 范围内闭合，下一步进入 §4.2 的 SHADOW 注入观测阶段。
>
> 对应债务：`docs/known-debts.md` #92 Gate 2（ETA 的 `z_t / beta_t` 涌现与因果残差控制）。
> 对应 spec：`docs/specs/evidence_program.md` §eta-gate2-residual-causal.v34 / v35。

---

## 1. 问题定义

Gate 2 的命题是：`z_t` 携带**可复用**的抽象动作，decoder 输出 `U_t` 对冻结基底
residual stream 产生**可测因果作用**。晋升要求两件事同时成立：

1. **行为级因果功率**：残差注入能以 ≥ `0.02`（NLL 单位）的效应全距改变环境结局。
2. **可迁移动作价值**：在 train 上学到的动作知识，在从未观察过的冻结分区上仍然有效
   （否则只是逐前缀噪声拟合，等价 max-of-noise）。

第 1 件事已在 v33 证明（134/134 前缀效应全距 ≥ `0.02`）。第 2 件事是 v30 以来
唯一未闭合的失败面，也是本计划的全部内容。

## 2. 证伪链摘要（为什么走到 v35）

| 版本 | 假设 | 反证结果 | 存活的结论 |
|---|---|---|---|
| v30 | oracle 逐前缀改善 = 可达解 | fresh validation audit 为负，oracle 增益是 max-of-noise | self-NLL 通道整体关闭 |
| v31 | 环境结局（真实 downstream outcome）作 target | 可复现但效应 ~7e-4，远低于 0.02 | PE→credit 链路正确，读出通道饱和 |
| v32 | 预注册 oracle-vs-permutation-null 门 | （门本身，非假设） | 防 max-of-noise 的统计纪律固化 |
| v33 | 天花板在读出链路（对执行器近盲） | 修复后效应全距 ×100（134/134 ≥ 0.02），但 oracle 门 validation 失败 | 因果功率证明；效应真实但**上下文条件化**，动作边际≈0 |
| v34 | 迁移失败根因是执行器坐标系任意（固定正弦基与行为子空间无关） | 换 train-transition PCA basis 后 v32 严格 selector 门首次四分区全绿（audit `+0.006~+0.017`），但 oracle 门 validation 仍负 | 可迁移动作价值存在，**载体是条件化 selector，不是动作边际** |

v34 的核心发现：oracle 门检验的是「每前缀 argmax 动作的边际迁移」，在效应真实但
上下文条件化的系统里，argmax 恰好是对 max-of-noise 最脆弱的读出；而同一 artifact 内
train-fit selector 的独立 audit 均值在全部四个冻结分区超过同一 permutation null
（validation `+0.0173` vs null `+0.0058`）。即：**门检验的对象（动作边际）与系统
实际的迁移机制（条件化动作价值）不匹配**。

纪律约束：v32 oracle 门是预注册的，validation 失败即 promotion 拒绝，不回溯改门。
所以 v34 的 selector 证据只能记为诊断；要把它变成正式证据，必须**预注册一个新门 +
一批从未观察过的分区**，这就是 v35。

## 3. v35 收敛包设计（已实现，实现细节冻结）

### 3.1 预注册新门：`selector-vs-permutation-null-v1`

- **检验对象**：条件化动作价值的迁移——train-only 拟合的 selector 在冻结分区上，
  其被选动作的**独立 audit credit**（同一控制在下一 prefix 对 subsequent realized
  segment 的测量，与选择时使用的信号相互独立）是否超过 permutation-null 基线。
- **null 构造**：对每条 selector 选择，null = 同一 `(split, route, prefix)` 反事实
  格点上**全部候选**的 audit credit 均值（可交换零假设下，任何无信号 selector 的
  期望恰为该值）。实现见 `_selector_permutation_null_by_split`
  （`packages/vz-runtime/src/volvence_zero/agent/eta_gate2_residual_evidence.py`）。
- **通过条件**（全部满足才有 `reachable_solution_evidence=true`）：
  - train / validation / confirmation 三个分区 selector 选择格点均存在；
  - 三个分区的 `selected_excess_over_null_mean ≥ 1e-6`
    （`ETA_GATE2_MIN_MEASURABLE_EFFECT`）。
- **不变项**：v34 learned control basis（train-transition PCA，fingerprint 进
  provenance）、v33 realized-continuation NLL 结局、PE→credit 链路、v32 双门
  （mechanism/causal + 严格 selector 门）全部原样保留；
  `counterfactual_action_selector_live_injection=disabled` 不变，线上 wiring 不开。

### 3.2 新鲜分区（防证据污染）

v33/v34 的 validation routes 已被观察两次，不得再作一次性检验。v35 新造：

- **fresh validation**（4 条，从未观察）：`validation-v35-librarian-catalog-shift` /
  `validation-v35-clinic-handoff-review` / `validation-v35-museum-lighting-plan` /
  `validation-v35-harbor-manifest-update`；
- **locked confirmation**（4 条，manifest 声明 `confirmation_split_locked=true`）：
  `confirmation-v35-orchestra-rehearsal` / `confirmation-v35-greenhouse-irrigation` /
  `confirmation-v35-courier-route-repair` / `confirmation-v35-kiln-temperature-log`；
- 旧 validation 降级为 `superseded_validation_route_ids`，只作诊断；
- 词汇新鲜度契约测试保证 fresh 分区源文本与全部既有 case 的内容词不相交
  （`test_eta_gate2_v35_fresh_splits_are_lexically_fresh`）。

### 3.3 判定矩阵（预注册，不回溯）

| 结果 | 判定 | 动作 |
|---|---|---|
| 三分区 excess 全部 ≥ 1e-6 且 mechanism/causal/selector 门全绿 | `causal-supported`，`promotion_allowed=true` | Gate 2 causal 层首次闭合；进入 §5 的 shadow-injection 阶段 |
| train 通过、fresh validation 或 confirmation 失败 | `mechanism-supported`，promotion 拒绝 | 判定「条件化动作价值不跨新鲜语料迁移」，v34 selector 证据收缩为 development 诊断；触发 §6 kill condition 评估 |
| train 也失败 | 同上 | selector 拟合本身不稳，回查 v34 basis 在新语料上的适用性 |

## 4. 修改之后的目标（post-modification goals）

### 4.1 v35 直接目标

1. **主张升级目标**：把 v34 的「可迁移动作价值以 train-fit selector 为载体获得正向
   证据（诊断级）」升级为「条件化动作价值在从未观察的 fresh validation + locked
   confirmation 上通过预注册置换零假设检验（正式证据级）」。
2. **数值目标**（预注册最小值，非期望值）：三分区
   `selected_excess_over_null_mean ≥ 1e-6`；参考量级：v34 在旧 validation 上
   selector audit `+0.0173` vs null `+0.0058`（excess ≈ `+0.0115`）。
3. **不变量目标**：效应全距 134+ 前缀 ≥ `0.02` 不缩水；fallback=0、hook
   coverage=1.0、prefix protocol=1.0；basis fingerprint 与 v34 逐位一致。

### 4.2 v35 通过后的 Gate 2 路径（顺序不可调换）

1. **SHADOW 注入观测**（下一收敛包，实现计划见
   [`eta-gate2-v36-shadow-injection_b82d47c9.plan.md`](eta-gate2-v36-shadow-injection_b82d47c9.plan.md)）：
   `selector_injection_allowed=true` 仅为 SHADOW 级许可；在 SHADOW wiring
   下并跑 selector-on/off，比对快照差异，不影响线上行为。退出条件：SHADOW
   轨迹上 selected audit credit 分布与 evidence run 一致（无分布漂移）。
2. **多 seed + 更长轨迹**：单 seed CPU ci-smoke 升级为 ≥3 seed、
   `--max-prefix-steps` 放宽，确认 excess 不是单 seed 波动。
3. **Gate 2 EXIT 对表**：对照 known-debts Gate 2 EXIT（≥500 real-trace、
   `validation_delta ≥ 0.02`、抽象质量优于 matched controls、不靠语义 action
   label scaffold）逐项补齐；当前 v35 即使通过也只闭合 causal 层，
   longitudinal 层（真 trace、跨 session）仍是独立前置。
4. **更大基底复测**（可选，触发条件见 §6）：0.5B 上 excess 若显著但绝对量小，
   在更大 Qwen 基底上复测条件化迁移是否放大。

### 4.3 v35 失败后的目标（同样预注册）

- 不加语义 action label、不降阈值、不复用已观察分区重跑——这三条是 v30 以来的
  固定纪律。
- 允许的后续方向（按优先级）：
  1. selector 状态特征从 12 维稳定统计升级为 basis-aligned 投影特征
     （执行器坐标系与读出坐标系对齐，v34 只改了执行器侧）；
  2. `control_scale` 钳制上限从 `0.12` 提到 `0.30` 复测效应-迁移权衡；
  3. 若两者均不改善 fresh 分区 excess，Gate 2 主张长期收缩为
     「行为级因果功率已证明；可复用动作价值在 0.5B 无证据」，
     并把复测挂到更大基底的 rare-heavy 计划。

## 5. 涉及 owner 与契约（已落地部分）

| 文件 | 角色 |
|---|---|
| `packages/vz-substrate/src/volvence_zero/substrate/control_basis.py` | basis owner：`fit_transition_control_basis` + fingerprint（v34） |
| `packages/vz-substrate/src/volvence_zero/substrate/residual_backend.py` | `install_control_basis` + provenance（v34） |
| `packages/vz-runtime/src/volvence_zero/agent/eta_proof_benchmark.py` | fresh routes / corpus owner（v35） |
| `packages/vz-runtime/src/volvence_zero/agent/eta_gate2_residual_evidence.py` | schema v35、selector-null 门、manifest、verdict |
| `packages/vz-runtime/tests/test_eta_residual_causal_controls.py` | v35 契约测试 + fresh split 防污染测试 |
| `docs/specs/evidence_program.md` | v34/v35 evidence 契约 |
| `docs/known-debts.md` #92 Gate 2 | 证伪链记录 + 本计划引用 |

回滚方式：v35 仅改 evidence 契约与判定逻辑，线上 wiring 零改动
（live injection 始终 disabled）；回滚 = 恢复 v34 schema 常量与 manifest
case groups，fresh routes 为纯新增可直接移除。

## 6. Kill condition（何时停止在此方向投入）

满足任一条则停止 selector 方向迭代，主张收缩并转向更大基底/rare-heavy 计划：

1. v35 失败后，§4.3 的两个后续方向（读出对齐、control_scale 放宽）各一轮
   仍无法使 fresh validation excess 为正；
2. 任何一轮出现「train excess 显著为正、fresh 分区系统性为负」的翻转
   （过拟合签名而非量级不足）；
3. 三轮内消耗掉的 fresh 分区总数达到 12 条 route 而无一次通过
   （fresh 语料供给纪律：每轮必须新造，不可回收）。

## 7. 验证清单

- [x] `pytest packages/vz-runtime/tests/test_eta_residual_causal_controls.py`（v35 契约 + 防污染）
- [x] `pytest packages/vz-substrate/tests/test_control_basis.py`（v34 basis）
- [x] `ruff check` 改动路径（无新增违例）
- [x] v35 CPU evidence run 完成 → 读取 `promotion_verdict.json` /
      `ablation_results.json` 的 `selector_permutation_null_by_split`
- [x] 结果回写 known-debts Gate 2 bullet 与 `evidence_program.md` v35 小节

## 8. 结果（2026-07-29 run 完成后回写）

artifact：`artifacts/eta_gate2_residual_causal_v35_selector_null_fresh_fullwidth896_qwen25_05b_cpu_1seed_20260729`
（Qwen2.5-0.5B、full-width 896、CPU、单 seed、`--max-prefix-steps 8`，36 min）。

### 8.1 预注册门判定：通过（§3.3 第一行）

| 分区 | selection_count | selected audit mean | null mean | **excess over null** | positive rate |
|---|---|---|---|---|---|
| train | 90 | `+0.0062` | `+0.0041` | **`+0.0020`** | 0.61 |
| fresh validation | 23 | `+0.0151` | `+0.0005` | **`+0.0146`** | 0.61 |
| locked confirmation | 24 | `+0.0038` | `-0.0023` | **`+0.0061`** | 0.58 |
| eval（诊断） | 11 | `+0.0135` | `-0.0105` | `+0.0240` | 0.73 |
| heldout（诊断） | 12 | `+0.0072` | `+0.0030` | `+0.0042` | 0.50 |

三个正式分区 excess 全部 ≥ `1e-6`，五分区全正；mechanism / causal / selector
门全绿 → `reachable_solution_evidence=true`、verdict **`causal-supported`**、
**`promotion_allowed=true`**。

### 8.2 不变量核对

- 效应全距：160/160 前缀 ≥ `0.02`（median train/eval/heldout/validation/
  confirmation = `0.074/0.098/0.070/0.076/0.081`）——因果功率不缩水。
- basis fingerprint 与 v34 逐位一致：`train-transition-pca-v1:326aecdd…`。
- fallback=0、hook coverage=1.0、prefix protocol=1.0。
- oracle 边际诊断（非门槛）依旧混合：validation `+0.0107` 正、eval `-0.0081` /
  confirmation `-0.0055` 负——继续印证「迁移载体是条件化 selector，不是动作边际」。

### 8.3 已达成目标与边界

- §4.1 目标 1（主张升级为正式证据级）与目标 3（不变量）达成；目标 2 的数值
  远超预注册最小值（fresh validation excess `+0.0146` vs 最小 `1e-6`）。
- 边界不扩大：单 seed、CPU、ci-smoke 短前缀、合成 hierarchical 语料；
  `selector_injection_allowed=true` 仅 SHADOW 级，live injection 保持 disabled；
  longitudinal 层（≥500 real-trace、跨 session）与 Gate 2 EXIT 其余条款未闭合。
- 下一收敛包：§4.2 第 1 步 SHADOW 注入观测。

### 8.4 实现收口审计（结果读取后，不改变预注册阈值）

交付复核补上 selector lineage 的 fail-closed 完整性门：train / fresh
validation / locked confirmation 的每条 selection 都必须能定位同一
`(split, route, prefix)` 候选格点，`selected_action_index` 必须存在，且
`audit_selected_raw_delta` 必须与该候选的 `audit_action_credit` 一致。缺格点、
缺候选、冲突重复值或数值不一致均禁止晋升；这不改变 §3.1 的统计对象和
`1e-6` 阈值，只防止部分配对或 lineage 漂移误绿。

v35 原始 JSONL 重放结果：train / validation / confirmation 输入 selection
分别为 `90 / 23 / 24`，有效配对同为 `90 / 23 / 24`；三分区
`missing_counterfactual_grid_count / missing_selected_candidate_count /
selected_audit_lineage_mismatch_count` 全部为 `0`。因此新增的
`*_selector_lineage_complete` gates 全绿，§8.1 verdict 不变。
