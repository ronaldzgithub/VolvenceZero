# Coding-Lab 证据 lane Spec（编程域持续学习）

> 状态：Packet 0 / Packet 1 判词已产出；Packet 2 进行中（注入臂 + 双门）；
> Packet 3 前置 a（基底预检）已 PASS。
> 计划源头：`.cursor/plans/编程域持续学习证据_bd518941.plan.md`；战略依据见
> `docs/moving forward/主线提升方案_2026-08.md`（A1 判词收窄后本 lane 升格为主证据 lane）。

## 1. 职责边界

| 部件 | 位置 | 唯一职责 |
|---|---|---|
| 环境生成器 | `packages/lifeform-domain-coding/src/lifeform_domain_coding/lab/generation.py` | 确定性合成 Python 包 + pytest 套件；隐藏不变量注册表 |
| 任务链 | `.../lab/tasks.py` | 种子化 change-request 序列；参考解 / 两类 sabotage / bug 注入 |
| Oracle | `.../lab/oracle.py` | 在干净评测目录结算 episode（原装测试重生成 + 隐藏验收测试注入） |
| 链工作区 | `.../lab/workspace.py` | 内嵌 git 仓库 + 每 episode worktree；pass 合并包树、fail 丢弃 |
| 轨迹存储 | `.../lab/trajectory.py` | 内容寻址 JSONL；重放基底（API 手不可确定，重放靠日志不靠重跑） |
| 手 | `.../lab/hands.py` | `ScriptedHand`（机器标定）/ `OpenAICompatHand`（真实 coder API） |
| 标定 runner | `.../lab/calibration.py` + `scripts/run_coding_lab_calibration.py` | Packet 0 判词工件 |
| Held-out 封存 | `.../lab/heldout.py` | 结构变体哈希封存不开封（迁移包用） |

`lab` 不被 coding vertical 的产品路径 import；Packet 0 无脑核依赖。SHADOW 观察者
（Packet 1）经 `Brain` facade 正式通路进入脑核，`vz-*` 不反向依赖 lab。

## 2. 环境不变量（防作弊纪律）

- **真代码真测试**：生成物是可运行包 + 真 pytest；oracle 由环境判卷，无 judge。
- **隐藏不变量**：5 类 latent 约束（config 大小写 / index 幂等 / half-up 舍入 /
  报表插入序 / 隐藏 consumer）不写进任务描述，仅由 `tests/full` 机械判决。
- **oracle 防篡改**：评测只取 agent 工作区的**包树**；测试树按 spec 重新生成，
  验收测试评测时注入——workspace 里改测试不影响判决（`tests_tampered` 单独记账）。
- **确定性拆分**：环境侧（生成 / 任务链 / oracle）比特级确定（同 spec 同 tree hash）；
  手侧不可确定，轨迹全量落盘 + sha256 内容寻址。
- **磁盘纪律**：runner preflight 检查剩余磁盘；逐 episode bytes 记账；worktree 用完即删。

### 2.1 House 约定（记忆可兑现难度旋钮，2026-08-13）

隐藏不变量由**仓库内可见**的回归测试强制——强手提交前跑全套测试即可全部
自查（2026-08-12 qwen3-coder-next 标定 0.9375 饱和的根因）。House 约定
关闭该回路：owner 偏好级契约，**只**由 oracle 注入的隐藏验收测试
`test_house_<id>` 强制，仓库零信号、集内 `run_test` 不可发现、跨集稳定
复现——跨集记忆是停止违反的唯一合法通道。

- `EnvSpec.convention_ids`（默认空 = 旧行为逐比特不变）；穿线到
  `CalibrationConfig` / `ArmChainConfig` / 标定 CLI `--conventions` /
  Packet 2 prereg `convention_ids`（必须与过带标定一致）。
- 首个约定 `convention_export_all`：新公共符号必须登记进模块 `__all__`
  （覆盖 add_helper / extend_report / config_feature）。参考解携带合规
  编辑（可解性保持）；invariant sabotage 同样合规（其"过验收、破回归"
  定义性质不变）。
- 违反经 oracle 归因进 `invariant_violations` 同一通道（observer/记忆
  流水线零 schema 变更）。

## 3. Packet 0 判词（已产出）

工件：`artifacts/coding_lab/coding_lab_calibration_scripted_20260812/report.json`。

- `environment_deterministic = True`（同 spec 同树同链；异 seed 异树）
- `oracle_band = True`（脚本手 4 链 × 8 episode，pass rate 0.656 ∈ [0.2, 0.8]，
  跨链方差存在；**scope = machinery-only**，冻结 API 手必须在 Packet 2 prereg 前重跑本标定）
- `heldout_sealed = True`（2 个结构变体哈希封存，树已销毁）

API 手标定（qwen3-coder-next，2026-08-13）：

- 无约定：pass rate 0.9375 超带上限——任务对强手饱和，`oracle_band = False`
  （`coding_lab_calibration_api_qwen3codernext_20260812`）。
- 加 `convention_export_all` 后：pass rate 0.4375 ∈ [0.2, 0.8]，三判词全过；
  16 次违反全部归因 `convention_export_all`，覆盖类别（add_helper /
  extend_report / config_feature）无记忆手 100% 失败——难度旋钮按设计咬合
  （`coding_lab_calibration_api_qwen3codernext_hard_20260813`）。

## 4. Episode 终局进入内核的契约（Packet 1）

**复用判决**：外部结局走既有唯一合法通道 `dialogue_external_outcome`
（owner `DialogueExternalOutcomeModule`，vz-runtime），**不新建 slot、不建第二 owner**。
`DialogueExternalOutcomeKind` 按其文档化扩展协议（W3-A 先例）追加 task-execution 组：

- `TASK_VERIFIED` / `TASK_REGRESSED`，只能由确定性 oracle 经 `ENVIRONMENT` typed
  source 注入；禁止 chat text 推断、禁止 LLM 提案。
- 关系轴零污染是新增值的存在理由（`MISSED` 关系 delta −0.60 且 task delta 为 0）。
- 五张下游表的显式语义（PE bias / regime score / 结构投影 / rupture opt-out /
  repair opt-out）与 DLaaS 镜像暂不扩展的决定，见
  `docs/DATA_CONTRACT.md` §"DialogueExternalOutcomeKind vocabulary"。

采集器提交面：`BrainSession.submit_dialogue_outcome(kind=..., source=ENVIRONMENT,
confidence=1.0, evidence_ref=<trajectory sha256>, description=<结构化结局摘要>,
action_turn_index=...)`。PE 在自身 `process` 内经 `_apply_external_outcome_bias`
把它融入 `ActualOutcome`（provenance 走 `external_outcome_refs`）。

## 5. SHADOW 观察者（Packet 1 结构）

- Orchestrator 在 `lifeform-evolution`（对齐 `seven_day_companion.py` 先例）；
  只经 `Brain.create_session` / `BrainSession.run_turn_async` / typed submit API 进入脑核。
- 经历写入只走 `MemoryStore` 正式 API 与 `submit_semantic_events` / `submit_tool_result`。
- **行动前下注**：episode 内 turn 流驱动 PE owner 发布 `next_prediction`
  （`predicted_task_progress` 等），时间戳先于 oracle 结算落盘；oracle 结局经
  §4 通道进入下一 turn 的 `ActualOutcome`，由 `_settle_owner_predictions` 结算。
- 全程 SHADOW：观察者脑核不影响手的行为；零 ACTIVE 变更。

判词（Packet 1，已产出）：工件
`artifacts/coding_lab/coding_lab_observer_scripted_20260812/`。

- `pe_discrimination = True`（signed_reward 分离 p≈1e-4；task_progress 分离 p≈1e-4）
- `cross_process_recovery = True`（4 链全部：scoped store 重载后记忆非空）
- `external_outcome_channel = True`（每 episode 的终局证据均带 provenance 进入 ActualOutcome）
- `forecast_skill = False`（**如实封存**，scope = synthetic substrate × scripted
  trajectories）。根因探针（`forecast_skill_diagnostic.md`）：PE owner 的
  `next_prediction` 前向面在合成基底上不消费 execution_result 证据——三连
  失败工具流的预报不低于三连成功流。修复路线：(1) publisher 侧丰富
  （vz-cognition PE 预测头纳入执行证据，独立收敛包）；(2) 真实基底 + 冻结
  API 手重跑。**不降判据、不在本 lane 内 hack。**

## 6. 注入包与三臂对照（Packet 2 结构）

模块：`lifeform-evolution/coding_lab_arms.py` + `scripts/run_coding_lab_packet2.py`
（`freeze-prereg` / `smoke` / `formal` 三个子命令；formal 只认冻结 prereg 的 sha256）。

- **三臂同手同链**：`brain`（记忆摘要注入）/ `steelman`（全历史转录注入，
  上下文随 episode 线性增长）/ `stateless`（空上下文资格臂）。手在三臂间冻结，
  只有 `HandContext.context_preamble` 不同。
- **recall 契约**：经验召回走 memory owner 的官方 `MemoryStore.retrieve`
  （`RetrievalQuery(strata=(EPISODIC, DURABLE), facets=("coding-lab",
  "category:<cat>"))`），**不走对话回合**——对话回合会先把召回提示写进
  TRANSIENT，随后提示以满相似度检回自身，把全部经验条目挤出 top-k
  （2026-08-12 smoke 实测 5/5 命中均为提示自拷贝）。提示与 facets 只含
  harness 已知元数据（category + 目标文件），任务描述不入提示（防自指泄漏）。
- **注入包 SSOT**：每行引用 owner 生成面（召回条目 `content`、快照
  `description`、记忆层摘要）；harness 只加标签、排序与预算截断。召回经验
  置于包首，预算截断先丢通用摘要。
- **写入契约**：每 episode 终局后观察者经 `MemoryStore.write` 写一条
  EPISODIC / `Track.WORLD` 经验（task id + category + 结局 + 不变量违规 +
  任务描述），失败经历 strength 更高（0.85 vs 0.60）。
- **smoke 已知效应**：`MemoryAwareScriptedHand` 在针类别上确定性行为
  （前导含针 → 正确；不含 → acceptance sabotage），非针类别保留随机抽签
  提供基线方差。smoke 判词 = 已知效应方向（brain>stateless）+ 预期空
  （brain≈steelman）+ 标度结构（brain 有界 / steelman 线性增长 / 比值递减）。
- **formal 三门**（2026-08-13 prereg 冻结，任何 formal 运行之前）：
  `memory_gate` = brain vs stateless 斜率差 bootstrap 5% 下界 > 0
  （记忆有效的正主张）；`quality_gate` = brain vs steelman 斜率差下界
  > −0.05（非劣界，结构化记忆不劣于堆上下文）；`scaling_gate` =
  token/latency 比值门。首个冻结 prereg：
  `artifacts/coding_lab/prereg/coding_lab_packet2_prereg_20260813.json`
  （8 链 × 10 集，conventions=convention_export_all，与过带标定一致）。
- **formal 断点续跑**：按 (chain, arm) 格写 `rows.json` 检查点；
  `--resume` 已完成格直接加载，中断格整格重跑。长跑配 `caffeinate`
  防休眠（2026-08-12 夜跑被机器休眠杀掉的教训）。

## 6.5 断点续跑与瞬时故障纪律（API 手长跑）

- **链级检查点**：标定 runner 每条链完成时写 `chains/chain-XX/rows.json`；
  `--resume` 重启时已完成链直接加载，中断链整链清除重跑（保持链仓库
  合并历史确定性，不重建 git 中间态）。
- **API 手有界重试**：`OpenAICompatHand._post` 对瞬时传输错误
  （SSL/超时/连接断）与 429/5xx 指数退避重试 ≤4 次；其余 4xx 立即
  fail loudly；重试耗尽带上下文抛错。不吞契约违反。

## 7. Packet 3 前置 a 判词（已产出）

工件：`artifacts/coding_lab/coding_lab_packet3_substrate_check_20260812/`；
脚本 `scripts/run_coding_lab_packet3_substrate_check.py`（只读诊断）。

- `Qwen/Qwen2.5-Coder-1.5B-Instruct` fp32 全门 PASS：块解析 28 层、
  hidden 1536、middle 捕获层 {13,14,15}、scorer 注入层 13、
  `baseline_action_nll` 有限、峰值 RSS 6.0 GiB（远低于 24 GiB 界）。
- 判词：Packet 3 采用 Coder-1.5B fp32 几何；0.5B 回退路线保留但不启用。

## 7.5 Junction 语料与前置 b（Packet 3 结构）

模块：`lab/junctions.py`；审计脚本 `scripts/run_coding_lab_packet3_margin.py`。

- **junction 定义**：episode 日志中每个手决策点。动作词表是五个沙盒
  工具上的协议枚举（`investigate`=read/list/grep、`edit`=write_file、
  `test`=run_test、`submit`；协议外工具请求记为 `invalid`——真实 API
  手会请求不存在的工具，它进入历史状态但永不成为标签）。
- **状态键**：`(category, reads_bucket≤3, has_edited, test_state)` 协议级
  状态，不做文本语义匹配。
- **对照式标注（R-CL4 防线）**：仅当同一状态键同时存在通过分支与失败
  分支、且二者动作不同才产出标签；expert = 通过分支众数动作（并列取
  最短分支）。脚本手轨迹按构造无对照信号（过/挂动作序列相同），语料
  必须来自 API 手轨迹——manifest 如实记账。
- **split**：按状态键 sha256 确定性切 train/eval（默认 30% eval）。
- **前置 b 三门**：`corpus_sufficient`（≥24 对照 junction，不足不启动
  模型）→ `expert_margin`（NLL 间隙中位数 > 0 且 bootstrap CI 下界 > 0）
  → `steer_headroom`（norm-capped 随机方向干预平均 |ΔNLL| ≥ 0.01 nats）。
  任一不过 → 不开 RL，封存"该决策面无择时余量"。

## 7.6 Packet 2.5：层 A 黑盒择时 gate（结构）

模块：`lifeform_evolution/coding_lab_blackbox_gate.py`；runner
`scripts/run_coding_lab_packet25_blackbox_gate.py`。目的：直接检验
**Learnable 轴不依赖白盒残差**。

- **离线 bandit 表**：从轨迹日志冻结 (协议状态键, 动作类) → episode
  终局通过率；训练期间表不再更新。
- **特征全黑盒**：junction owner 发布的结构化协议状态（category 哈希
  桶 one-hot + reads_bucket + has_edited + test_state one-hot，维度 13），
  不碰残差。
- **信用只来自 episode 终局 oracle**（环境结算，非 evaluation/judge，
  R12 无泄漏）；动作采样掩码到有观测支撑的动作（无反事实证据不外推）。
- **策略**：softmax gate（S3-E `_GatePolicy` 骨架），无 free bias，
  minibatch REINFORCE + 熵正则 + multi-restart 取 train 最优。
- **判词**：held-out 状态键上 gate 期望通过率 vs uniform-over-supported
  的 uplift bootstrap CI 下界 > 0；同时报告 vs 全局众数动作基线。
  状态键 < 8 train / 4 eval 时 fail loudly 不出读数。

## 7.7 Packet 3：S3-E 编程域复刻（结构）

模块：`lifeform_evolution/coding_lab_s3e.py`；runner
`scripts/run_coding_lab_packet3_s3e.py`（`freeze-prereg` / `run`）。

- **结构映射**：route = 通过的 episode；junction = 决策点；潜在条件
  （subgoal 等价物）= 该点 expert 下一步动作类（4 类）；观测
  goal-stripped（无 category/任务描述），revealed 文本携带任务；相位
  切换（investigate→edit→test→submit）提供真实 post_switch。
- **算法链整体复用** vz-runtime `eta_when_to_steer_rl`（SSOT，不复制
  平行实现）：`_capture_examples` → ridge reader fit → 低秩条件
  executor（无 free bias）→ `_precompute_records` 反事实表 →
  `_run_seed`（REINFORCE × multi-restart=4）→ `_aggregate`
  （worst-seed CI）→ `assess_when_to_steer` 六门。复用合法性：这些
  函数对 row 只做鸭子字段访问，`CodingJunctionRow` 提供同名字段。
- **几何重绑定**（前置 a）：Qwen2.5-Coder-1.5B fp32、注入层 13、宽
  1536、rank 8；artifact 不可从 0.5B 几何迁移，必须重 fit。
- **prereg**：`artifacts/coding_lab/coding_lab_packet3_prereg.json`
  （schema `coding-lab-when-to-steer-prereg.v1`）；六门阈值沿用 S3-E
  先例数值并由 runner 与模块常量互检；语料门 train ≥ 60 行 /
  heldout ≥ 20 行；split 按 case（episode sha）不跨界。
- **run 前置**：前置 b margin 审计 overall_pass 必须为真
  （`--skip-margin-check` 仅与 `--smoke` 联用，冒烟不产生判词）。
- 产物：report.json/md + artifact_manifest.json（prereg sha、六源文件
  sha、语料指纹、MPS attestation、margin attestation）。

## 7.8 Packet 4：自改提案经 ModificationGate.OFFLINE（结构）

模块：`lifeform_evolution/coding_lab_packet4.py`；runner
`scripts/run_coding_lab_packet4_gate.py`。照
`steering_promotion_gate` 先例。

- **候选** = admitted 的 Packet 3 运行（smoke 运行显式拒收）；
  `validation_delta` = 判词聚合的 `gain_vs_noop_ci_lower_min`
  （worst-seed CI 下界，环境结算出身）。
- **门** = cognition 拥有的 `evaluate_gate_reasons`（OFFLINE 档
  validation ≥ 0.05、rollback 证据必须在场，fail-closed）；三条安全
  读数（contract_integrity / rollback_resilience / fallback_reliance）
  是只读结构证据，不进学习环路（R12）。
- **部署** = 单文件注册表指针
  （`artifacts/coding_lab/active_artifact.json`）写入；**回滚** =
  candidate-bound checkpoint 恢复（genesis = 删除指针）；runner 在
  ALLOW 后强制演示 apply → rollback → verify → re-apply 全程。

## 7.9 全阶段流水线（机制冒烟）

`scripts/run_coding_lab_pipeline.py`：一条命令按序驱动 P0 → P1 → P2
→ P2.5 → P3a(margin) → P3b(S3-E smoke) → P4(gate probe)，每阶段调用
正式 runner（不复制逻辑），机制级小参数（约 30 分钟）。

- **判定口径**：机制完整性 = 每阶段跑完并产出 artifact；判词数值如实
  记录，玩具规模下的 FAIL 不作为管线失败、不改阈值。
- **P4 探针纪律**：`--mechanism-probe` 把 apply/rollback 演示打到
  probe 注册表（`probe_active_artifact.json`），永不触碰正式指针。
- 2026-08-13 首跑：7/7 站产出；P0/P2 判词全过；P1 的 forecast/PE 门、
  P2.5、P3a expert 间隙、P3b 三门在玩具规模如实 FAIL；P4 以正确理由
  BLOCK（validation 余量 + contract integrity）。

## 8. 变更纪律

**2026-08-12 注 bug 锚定修复（非模板变更，版本不递增）**：fix_bug 的
prestate 注入从精确文本锚（`FileEdit`）改为 AST 函数名锚
（`tasks.FunctionReplace` + `workspace._apply_function_replace`）。动机：
API 手在演进式链仓库中会用自己的风格改写函数并在 PASS 后合并，精确文本
锚随之漂移（首次 API 标定在第二个 fix_bug episode 的 `begin_episode`
崩溃）。等价性证据：pristine 树上新旧注入逐字节相同（探针验证），故
既有脚本手判词（Packet 0/1/2 smoke）全部保持有效；`GENERATOR_VERSION`
未动，held-out 封存不作废。hand 侧编辑仍保持文本锚（它们作用于注入的
已知输出）。

- 环境模板 / 不变量 / 任务模板变更 ⇒ `GENERATOR_VERSION` 递增 + 重新标定 +
  重新封存 held-out（旧封存作废，记录于本文件）。
- 换手（模型 / provider）= 新 lineage ⇒ 重跑 Packet 0 标定。
- 判据先冻结再跑；任何包触发退出条件即如实封存负结果。
