# Gate 1–6 证据战役：长周期完整执行计划

> 状态：执行中（Phase 0–4 已完成；Gate 1 / Gate 4 / Gate 5 causal 按
> 预注册 NO-GO 收缩；Phase 5 Gate 6 数值门已冻结，进入实现）。
> 目标：把 `docs/known-debts.md` #92 的系统级证据门推进到
> **Gate 1、2、4、5、6 全部拿到 mechanism + causal verdict**（Gate 2 另加
> SHADOW 观测闭环）。注：known-debts 当前没有 `### Gate 3` 小节（编号从 2 跳到
> 4），所以「跑完 Gate 6」= Gate 1 / 2 / 4 / 5 / 6 五个门。
>
> 前置计划：
> - [`eta-gate2-v35-selector-null_e4a91f27.plan.md`](eta-gate2-v35-selector-null_e4a91f27.plan.md)（已完成，v35 通过）
> - [`eta-gate2-v36-shadow-injection_b82d47c9.plan.md`](eta-gate2-v36-shadow-injection_b82d47c9.plan.md)（Phase 0 的第一包）
>
> 本计划是多收敛包的编排层：每个 Phase 内部仍遵守收敛包纪律（一包一个 owner、
> 3–8 个关键文件、可回滚），任何一包失败不连坐其他包。

---

## 0. 现状盘点（2026-07-30 摸底结论）

| Gate | 命题 | 现有证据 | 离 verdict 最近的缺口 |
|---|---|---|---|
| 1 PE/LSS | PE 是一级原始信号，与真梯度 LSS 对齐 | owner 代码 + 单元测试 + `pe-drive-off` serving 臂 | 无 gold case suite、无 LSS bridge 验证、无 `full-no-pe-drive` matched run |
| 2 ETA 因果 | `z_t/beta_t` 涌现 + 残差因果控制 | **v35 已过 causal**（`promotion_allowed=true`） | SHADOW 闭环观测（v36 已计划）、longitudinal |
| 4 主动学习 | 建立在 ETA 段上的 segment-aware active | apprenticeship Stage 0 probe；注意 `learned_active_*` 是 #88 后端晋升不是本门 | 四臂同预算 matched run + 打乱边界负对照 |
| 5 CMS 抗遗忘 | 多频记忆吸收-保持 Pareto 优于单频 | M2 代码闭合 + 单元级 A/B + shadow smoke | ≥500 settled 真 trace 上的多臂 Pareto run |
| 6 Meta-init | slow→fast 是可迁移初始化先验不是复制 | 只有 owner 内 telemetry（`slow_to_fast_init_benefit`） | 四臂 episode harness + swapped-user 污染负对照，全部从零建 |

**共同缺口**：Gate 4/5/6 的 causal 层都需要「同基底、同经历」的 ≥500 settled
transitions 真 trace。这是本战役的关键基建（Phase 2），建一次、三个门复用，
符合统一实验纪律第 2 条（同经历）。

## 1. 总路线图

```text
Phase 0  Gate 2 SHADOW 闭环          [v36 包已计划，先执行]
Phase 1  Gate 1 mechanism + causal   [两个包，与 Phase 0 的长 run 可并行推进实现]
Phase 2  共享真 trace 工厂            [基建包，Gate 4/5/6 的前置；最长的单次 run]
Phase 3  Gate 5 CMS Pareto           [依赖 Phase 2]
Phase 4  Gate 4 主动学习四臂          [依赖 Phase 2；复用 Gate 2 的 segment 机器]
Phase 5  Gate 6 meta-init 四臂        [依赖 Phase 2；依赖 Gate 5 的 CMS 慢层结论]
Phase 6  总对账 checkpoint            [聚合 verdict、更新 known-debts、决定是否推 Gate 7/8]
```

依赖关系：Phase 1 与 Phase 0 无依赖可交错；Phase 3/4/5 都依赖 Phase 2 的
trace，但三者互相独立、可按机器空闲穿插。每个 Phase 结束是一个可停顿的
checkpoint——中途停下不会留半成品状态。

预算量级（全部 CPU、本机，参考 v35 单 run 36 min）：

| Phase | 实现工作量 | 长 run 时长（估） |
|---|---|---|
| 0 | v36 计划已写，中等 | GO 探针 ~40 min + 全量 3 seed ~2 h |
| 1 | 中等（两包） | mechanism 套件分钟级；causal 对照 run ~1–3 h |
| 2 | 中等偏大 | **trace 生成 ~4–10 h**（500+ transitions 逐 turn 真 substrate） |
| 3 | 中等 | 5 臂 × 3 seed 重放 ~3–6 h |
| 4 | 中等偏大 | 4 臂 + 负对照 × 3 seed ~3–6 h |
| 5 | 大（从零建 harness） | 4 臂 × 多 episode × 3 seed ~4–8 h |
| 6 | 小 | 无 |

## 2. Phase 0 — Gate 2 SHADOW 闭环（收尾包）

按 [`eta-gate2-v36-shadow-injection_b82d47c9.plan.md`](eta-gate2-v36-shadow-injection_b82d47c9.plan.md)
执行，不在此重复。本战役只记两条对账线：

- **产出**：`shadow_observation_passed` 字段 + v36 bundle。
- **对本战役的影响**：v36 失败不阻塞 Phase 1–5（那些门不依赖闭环注入）；只
  影响 Gate 2 自身的 live-wiring 路线（按 v36 计划 §6 处理）。

## 3. Phase 1 — Gate 1：PE 原始数值与真 LSS grounding

拆两个包。owner 均为 `vz-cognition` 的 `prediction` 子包，evidence harness
在 `vz-runtime`。

### 包 1a：mechanism 套件（gold case + LSS bridge + lineage）

1. **Gold case suite**：数值 / 概率 / 枚举 / 向量 / 分布五类 outcome 各建
   deterministic gold case（输入、prediction、actual、期望 `pe_raw`、归一化
   值、component decomposition 全部手工冻结），同输入重跑逐位一致。落
   `packages/vz-cognition/tests/test_pe_gold_cases.py`。
2. **LSS bridge 验证**：MSE case 断言 `runtime_signed_pe == -dL/doutput`
   （容差预注册 `1e-9`）；对既有 `bridge_runtime_pe_to_lss`
   （`prediction/torch_lss.py`）用 torch autograd 算真梯度对拍。非 MSE 类型
   逐个登记 link function，量纲不同的误差禁止直接相加（契约测试断言）。
3. **lineage coverage 审计器**：新增 evidence 脚本
   `scripts/run_gate1_pe_mechanism_evidence.py`，对一次真 session run 的
   `predictions/outcomes/prediction_errors` jsonl 做一一 join，输出
   `lineage_coverage`（要求 =100%）、错配接受数（要求 =0）、重复结算数
   （要求 =0），产 12 文件 bundle（复用 #92 强制格式）。
4. **evaluation-decoupled 对照**：同一 trace 下改变 evaluation 配置
   （`VZ_PE_EVALUATION_DECOUPLED` 两态），断言 actual outcome / PE /
   learning credit 逐字节不变。

预注册门（mechanism verdict）：gold 全绿 + LSS 容差内 + coverage=100% +
decoupled 逐字节不变。全绿 → Gate 1 `mechanism` verdict 落账。

### 包 1b：causal 对照（`full-no-pe-drive` vs `full`）

1. 载体用现有 dialogue suite 的 `pe-drive-off` profile
   （`packages/vz-runtime/.../dialogue/_legacy.py` 已有该臂），不新建平行
   实现；差距是没有预注册门与 bundle 化。
2. 预注册 primary metric：同基底、同 seed schedule、同 scenario split 下，
   `full` 相对 `full-no-pe-drive` 在 held-out 场景的预注册学习增益指标
   （沿用 dialogue suite 现有 held-out 指标，实现前在 evidence spec 冻结
   具体指标名与最小效应 + 3 seed 方向一致性）。
3. 失败语义（预注册）：增益不出现 → 按 Gate 1 EXIT 收缩「PE 是有效学习源」
   主张为「PE 是可审计信号」，Gate 1 只保留 mechanism verdict；**不调参重跑**。

### Phase 1 文件面（两包合计 ≤8 关键文件）

`vz-cognition` prediction 子包测试 ×2、`scripts/run_gate1_pe_mechanism_evidence.py`、
dialogue suite 门注册处、`tests/contracts` 契约测试、`docs/specs/evidence_program.md`
Gate 1 小节、known-debts Gate 1 bullet。

## 4. Phase 2 — 共享真 trace 工厂（Gate 4/5/6 基建包）

**这是本战役最重要的一包**：产出一份不可变、可指纹化、≥500 settled
transitions 的真 substrate trace 语料，此后 Gate 4/5/6（以及未来 Gate 2/7/8
longitudinal）全部离线 replay 它，保证「同经历」。

1. **owner 与位置**：新模块 `packages/vz-runtime/src/volvence_zero/agent/`
   下的 trace factory（复用现有 session/joint_loop 正式路径逐 turn 真跑，
   禁止自建简化环境——遵守预训练附加约束「复用线上正式模块」）。
2. **语料设计**（实现前冻结进 evidence spec）：
   - 覆盖交错的旧知识/新知识 episode（Gate 5 吸收-保持需要）、多用户/多
     情境 context 块（Gate 6 meta-init 需要 train-context vs held-out-context
     切分）、含可结算 outcome 的对话轮（Gate 1/4 需要 settled PE）；
   - 规模：≥500 settled transitions，按 3 个 seed 各生成一份（多 seed 是
     Gate 6 EXIT 硬要求）；
   - 切分在生成时就冻结：`trace-train / trace-heldout-context /
     trace-locked-confirmation`，写进 manifest，locked 部分任何设计决策
     不得触碰。
3. **不可变契约**：落盘为 jsonl + `substrate_fingerprint` + trace 指纹；
   消费方只能整体加载，禁止逐条挑选（防 cherry-pick）。
4. **验收**（本包自身的门）：settled transition 计数 ≥500/seed、lineage
   join 100%（直接复用包 1a 的审计器——这是 Phase 1 先行的原因之一）、
   turn latency 与 slow-job latency 分开统计正常。
5. **run 预算**：逐 turn 真 Qwen CPU 前向，约 4–10 h/全部 seed；建议夜间
   跑，脚本支持断点续写（逐 transition append + 已完成计数，中断重启不
   重算）。

## 5. Phase 3 — Gate 5：CMS 多时间尺度吸收-保持 Pareto

1. **臂矩阵**（预注册）：`nested-CMS(full)` / `single-timescale`（同参数
   预算折叠为单频层）/ `no-ATLAS-replay` / `no-PE-write-gate` /
   `memory-only` 五臂，全部离线 replay Phase 2 同一 trace。
2. **双指标门**：`new_knowledge_absorption` 与 `old_knowledge_retention`
   同报（复用 `test_cms_anti_forgetting_evidence.py` 已有 proxy 定义，升级
   为 trace 级）；附 memory churn、错误晋升率、检索命中。禁止只报 loss。
3. **预注册通过条件**：full 在吸收-保持二维上 Pareto 不劣于任何对照臂，
   且至少一维显著优于 `single-timescale`（3 seed 方向一致）；cadence 断言
   （快层每轮、中/慢层只在声明边界、background 不阻塞 turn）作为
   mechanism 门同 bundle 产出。
4. **失败语义**：Pareto 不成立 → 按 Gate 5 EXIT 主张收缩为「多频 CMS 可
   运行且可回滚」，进 Phase 6 对账，不调参重跑同一 locked 分区。
5. **owner**：`vz-memory`（臂内学习）+ `vz-runtime`（replay harness 与
   verdict）。新脚本 `scripts/run_gate5_cms_pareto_evidence.py`。

## 6. Phase 4 — Gate 4：ETA 段上的主动学习四臂

1. **臂矩阵**（同标签预算，预注册）：`segment-aware-active`（以 ETA
   `closed_segments / z-family / beta_t` 为经验单位选样）/
   `turn-level-active` / `random-feedback`（复现 seed）/ `no-feedback`。
   载体复用 `vz-cognition` apprenticeship owner（`core.py` 已有
   `_random_feedback_sample`），segment 单位从 Gate 2 的 ETA segment 机器
   读快照，不重建。
2. **指标**（预注册）：达到同等 held-out alignment 所需标签数（primary）、
   累计 regret、无效请求率、漏问高风险事件率。
3. **关键负对照**：`shuffled-segment-boundary` 臂——打乱 segment 边界但保留
   PE 数值；若 segment-aware 优势不消失，判「主动学习并未建立在涌现抽象上」，
   Gate 4 主张降为「PE 驱动的反馈请求」（这是预注册的 kill 判据，不是可
   商量项）。
4. **边界纪律**：反馈请求只产 typed open loop / proposal，不越过
   consent/boundary owner——harness 断言之。
5. **owner**：`vz-cognition` apprenticeship + `vz-runtime` harness。新脚本
   `scripts/run_gate4_active_learning_evidence.py`。注意与 #88 的
   `learned_active_*` 命名区分（脚本名/artifact 名带 `gate4`，避免混淆）。

## 7. Phase 5 — Gate 6：nested meta-init 四臂 episode harness

从零建，是纯新增（现状只有 telemetry）：

1. **episode 协议**：`train-context 适应 → 快层 reset → held-out-context
   重新适应`，重复多 episode；held-out 用户/情境（来自 Phase 2 的
   `trace-heldout-context` 切分）不得出现在慢层训练标签中。
2. **臂矩阵**（同参数预算、同可见历史）：`meta-init`（慢层学到的初始化）/
   `copy-init`（直接复制快层终态）/ `random-init` / `no-init`。
3. **指标**（预注册）：达到目标误差所需步数（primary）、前 K 步 adaptation
   AUC、最终质量、负迁移率。
4. **污染负对照**：`swapped-user-slow-state`——换成另一用户的慢层状态，
   具体事实泄漏计数必须 =0；若 swapped 臂表现与正确臂无差，说明慢层没学到
   用户相关先验（诊断，不是门）。
5. **通过条件**：meta-init 在 3 seed held-out context 上减少适应步数或提高
   前 K 步 AUC，且不增加最终误差/泄漏/负迁移；「reset 后状态不同」不算通过。
6. **owner**：`vz-memory`（meta-init 本体在 CMS 慢层）+ `vz-runtime`
   harness。新脚本 `scripts/run_gate6_meta_init_evidence.py`。

## 8. Phase 6 — 总对账 checkpoint

1. 汇总 Gate 1/2/4/5/6 的 verdict 矩阵（mechanism / causal 两层），逐门
   回写 known-debts 与 `evidence_program.md`；主张按各门 EXIT 措辞收缩或
   确立，禁止越过 verdict 措辞。
2. 检查 #92 总 EXIT 第 1 条的进度（Gate 1-10 mechanism、Gate 1-8+10
   causal）——本战役完成后剩 Gate 7/8/9/10/11。
3. 产出下一战役建议：Gate 7（SSL→RL 已有半套 harness，缺 reverse-order /
   joint-unfrozen 两臂）和 Gate 8（wake/sleep，缺 next-session matched
   harness）可复用 Phase 2 trace，是自然的下一步。

## 9. 全局纪律（每个 Phase 都适用）

1. **预注册先行**：每包动代码前，先把门/指标/最小效应/kill condition 写进
   `docs/specs/evidence_program.md` 对应小节；未预注册的数字只能是诊断。
2. **fresh/locked 分区账本**：locked 分区只消费一次；被用于任何设计决策的
   分区立即降级为 development。Gate 2 的教训（v33/v34 validation 污染）
   直接适用。
3. **bundle 格式**：全部 run 产 #92 强制 12 文件 bundle；`report.md` 只引用
   bundle 内可复算结果。
4. **失败即收缩，不调参续命**：任何门失败按该门 EXIT 措辞收缩主张并进入
   Phase 6 对账；每个门最多允许一轮预注册的修正方向（学 Gate 2 v35 计划
   §4.3 的模式），修正仍失败即记 kill。
5. **长 run 工程习惯**：全部长 run 走托管后台 + 断点续写 + 逐行 jsonl
   落盘；GO/NO-GO 探针先行（单 seed 小规模确认方向再烧全量）；机器负载高
   时 Phase 3/4/5 的 run 串行排队，不并跑两个真 substrate 前向任务。
6. **验证范围**：每包只跑直接相关测试 + `ruff check` 改动路径；改共享
   契约/schema 时加 `pytest tests/contracts`；不因包大而全仓回归。

## 10. 建议执行顺序（可直接照此推进）

1. v36 包（Phase 0）：实现 → 探针 → 3-seed 全量 → 回写。
2. 包 1a（Gate 1 mechanism）：纯测试 + 审计器，快，且 Phase 2 依赖它。
3. 包 1b（Gate 1 causal）：预注册 + 对照 run。
4. Phase 2 trace 工厂：实现 + 夜间长 run（战役中最长等待）。
5. Phase 3（Gate 5）→ Phase 4（Gate 4）→ Phase 5（Gate 6）：按此序，因为
   Gate 5 的 CMS 结论是 Gate 6 meta-init 的语义前提，Gate 4 独立可穿插。
6. Phase 6 对账。

每步完成即回写本计划的 §11 进度表；中断后从进度表恢复。

## 11. 进度表（执行中维护）

| 包 | 状态 | artifact / 结论 |
|---|---|---|
| Phase 0 v36 SHADOW 闭环 | 已完成（NO-GO） | 单 seed 真实 Qwen 探针：train/confirmation 超过 zero 与 permutation-null；fresh validation `selector-permutation=-0.040979`，`shadow_observation_passed=false`，按预注册不跑 3 seed；v35 causal promotion 保留 |
| 包 1a Gate 1 mechanism | 已完成（PASS） | `gate1-pe-mechanism.v1`：五类 gold、真 LSS 最大误差 `1.39e-17`、lineage `1.0`、mismatch/duplicate `0/0`、ACTIVE evaluation-decoupled 字节一致；artifact `artifacts/gate1_pe_mechanism_20260730` |
| 包 1b Gate 1 causal | 已完成（NO-GO） | `gate1-pe-causal.v1` seed 101：四个 held-out case 的纯行为成功率 `pe-eta=1.0 / pe-drive-off=1.0`，gain `0.0 < 0.25`；按预注册停止 211/307，不调参；Gate 1 收缩为 mechanism-only。artifact `artifacts/gate1_pe_causal_20260730` |
| Phase 2 trace 工厂 | 已完成（PASS） | `gate456-shared-settled-trace.v1`：Qwen2.5-0.5B strict-local 三 seed 各 510、合计 1530 settled transitions；分区 900/450/180，lineage `1.0`、mismatch/duplicate `0/0`，fallback/empty residual/mutation 均 0，跨 seed fingerprint 一致；consumer admission allowed。artifact `artifacts/gate456_shared_settled_trace_20260730` |
| Phase 3 Gate 5 Pareto | 已完成（NO-GO） | `gate5-cms-pareto.v1`：五臂 × 三 seed 共 7650 arm-transitions；mechanism/cadence/rollback 全绿，full 对所有 control Pareto 不劣，但相对 single-timescale 双指标增益 `+2.508e-7 / +1.173e-6 < 0.02`，causal claim 收缩；artifact `artifacts/gate5_cms_pareto_20260730` |
| Phase 4 Gate 4 四臂 | 已完成（NO-GO） | `gate4-active-learning.v1`：修正版 adjunct trace 三 seed × 510 closure coverage 全绿；五臂 typed request/OpenLoop/no-write 机制门通过，但所有 heldout/locked balanced accuracy=`0.5`、labels-needed=`61`，segment 对 turn/random/shuffled gain 均 0，PE-driven diagnostic 也失败；主张收缩为 typed request 机制可运行。artifacts `artifacts/gate4_segment_settled_trace_20260730`、`artifacts/gate4_active_learning_20260730` |
| Phase 5 Gate 6 四臂 | 预注册已冻结，实施中 | `gate6-meta-init.v1`：四个 primary 初始化臂 + swapped-user diagnostic；locked 尚未读取 |
| Phase 6 总对账 | 未开始 | — |
