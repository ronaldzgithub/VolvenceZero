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
  防休眠（2026-08-12 夜跑被机器休眠杀掉的教训）。单写者进程锁
  `.formal.lock`（pid 活性检测）——两个 `--resume` 实例赛跑会互删
  对方在跑的格目录（2026-08-13 v2 首启实测）。
- **失败证据可行动化**（v2 仪器修正，对三臂对称）：v1 formal 实测
  裸违反 id 不可行动——steelman 全文历史在手违反率 0.53→0.50，
  brain 仅 0.45。oracle 现捕获 junit 断言消息为 `failure_details`
  （隐藏测试自带明确断言文本），流进 steelman 转录 `[oracle-failure]`
  行与 brain 记忆条目 `ci evidence` 段。

### 6.6 Packet 2 正式判词（已产出，2026-08-13）

工件：`coding_lab_packet2_formal_v2_qwen3codernext_20260813/report.json`；
prereg v2 sha256 `f72dc17e…`（8 链 × 10 集 × 3 臂，qwen3-coder-next，
conventions=convention_export_all，digest 预算 3500 字符）。

- `memory_gate = True`：brain vs stateless 斜率差 bootstrap 5% 下界
  +0.0061 > 0（均值 +0.0235，v1 的 2 倍）。
- `quality_gate = True`：brain vs steelman 下界 −0.0045 > −0.05；
  均值 +0.003 为正——结构化记忆在同等信息下不劣、甚至略优。
- `scaling_gate = True`：上下文 token 比 0.0977 ≤ 0.10
  （brain 882 vs steelman 9027 token）。
- 机制曲线（约定违反率 前半→后半）：brain 0.50→**0.17**、
  steelman 0.53→0.50、stateless 0.55→0.50。关键失败证据在 9000+
  token 原始转录中段被淹没，而 brain 的结构化召回把相关经验置于
  包首——**结构化记忆不只便宜 9 倍，同等信息下更有效**。
- v1 运行（`coding_lab_packet2_formal_qwen3codernext_20260813`）保留
  为仪器迭代证据：2/3 门过，scaling 0.1088 惜败，暴露裸 id 不可
  行动问题。

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
- **信用式标注（R-CL4 防线，2026-08-13 修正）**：owner 先发布
  `(状态键, 动作) → episode 终局通过率` 的信用表
  （`build_action_outcome_table`）；expert = 该状态下条件通过率最高且
  支撑 ≥ `min_action_support`(5) 的动作，non-expert = 同样有支撑但通过
  率低出 ≥ `min_pass_rate_margin`(0.10) 的动作。两侧统计随
  `ContrastiveJunction` 一起发布，消费者不重算。
  `credit_expert_actions` 是 expert 目标的**唯一定义**，margin 审计与
  S3-E 复刻共用。
  - **为什么改**：原规则是"expert = 某条通过分支在此处做的动作"。基线
    通过率约 0.5 时这等于幸存者偏差——它把"没跑测试就提交"标成 expert，
    使 margin 审计正间隙比例只有 0.51（抛硬币）。信用式标注后专家均值
    通过率 0.594 vs 非专家 0.270。
  - **诚实边界**：动作未随机化，条件通过率是**观测量**而非干预量，会
    与局面难度/手的信心混淆（如"未测直接提交"在容易局面上占优）。这是
    离线语料的固有局限，如实登记，不当作因果最优策略。
- **状态键记账**：2026-08-13 语料 77 个协议状态中 22 个有信用标签、
  22 个被判**无择时余量**（同状态两动作通过率差 < 0.10，如
  `fix_bug|tests=passed` 下 submit 1.0 vs test 0.95）、33 个单元支撑不足。
  "无余量"是结论而非数据缺口，两者在 `state_key_accounting` 中分开记。
- **动作表面文本 SSOT**：`ACTION_SURFACES`（裸标识符）与
  `NEUTRAL_STATE_TEXT` 由 `lab/junctions.py` 拥有，审计与 RL 复刻共用
  ——否则审计测的不是它要审的仪器。
- **split**：按状态键 sha256 确定性切 train/eval（默认 30% eval）。
- **前置 b 三门**：`corpus_sufficient`（≥20 对照 junction，不足不启动
  模型）→ `expert_resolution` → `steer_headroom`（norm-capped 随机方向
  干预平均 |ΔNLL| ≥ 0.01 nats）。任一不过 → 不开 RL，封存"该决策面无
  择时余量"。
  - `expert_resolution`：动作间 |PMI 间隙| 的 bootstrap CI 下界
    ≥ 0.3 nats（锚定 prereg `gain_vs_noop` 的改进量下限——仪器分辨尺度
    不得小于所声称的改进），且同输入重复打分逐位一致。读数取
    domain-conditional PMI `nll(a|state) − nll(a|neutral)`，抵掉各选项
    自身的表面似然。
  - **基底对齐度只报不设门**（同日修正）：原 `expert_margin` 门要求带
    符号间隙中位数 > 0，即要求冻结基底**已经**偏好信用专家动作。这是
    规格错误——steering 实验的前提正是基底存在偏差，基底若已对便无可
    扳之处。对齐度作为决策面性质记入 `base_alignment`（PMI 与原始读数
    正比例均为 0.36，即基底多数时候扳错方向 → 这是 steering 余量）。
  - 前置 b 的阈值是**预检 CLI 默认值**，不属 prereg 冻结判则；Packet 3
    的可证伪主张全部落在六条冻结判则上，本次修正未放宽任何一条。

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
  （subgoal 等价物）= 该状态键的**信用式 expert 动作**（4 类，取自
  `credit_expert_actions`，不是"该 episode 恰好做了什么"）；无信用标签
  的状态键不产出 row。观测 goal-stripped（无 category/任务描述），
  revealed 文本携带任务——被剥掉的正是 category，因此"读潜在条件"就是
  从残差里读隐藏的任务类别；相位切换提供真实 post_switch。
- **算法链整体复用** vz-runtime `eta_when_to_steer_rl`（SSOT，不复制
  平行实现）：`_capture_examples` → ridge reader fit → 低秩条件
  executor（无 free bias）→ `_precompute_records` 反事实表 →
  `_run_seed`（REINFORCE × multi-restart=4）→ `_aggregate`
  （worst-seed CI）→ `assess_when_to_steer` 六门。复用合法性：这些
  函数对 row 只做鸭子字段访问，`CodingJunctionRow` 提供同名字段。
- **几何重绑定**（前置 a）：Qwen2.5-Coder-1.5B fp32、注入层 13、宽
  1536、rank 8；artifact 不可从 0.5B 几何迁移，必须重 fit。
- **prereg**：v1 `coding_lab_packet3_prereg.json`（原样留档，仅用于
  机制冒烟）；formal 用 v2
  `coding_lab_packet3_prereg_v2.json`（sha256 `6c9819fe…`），在任何
  formal 运行之前冻结，`amendments` 字段记录本次修正（`expert_source`
  由 `passing-episodes-only` 改为 `conditional-pass-rate-credit`、前置 b
  门由对齐改为分辨力）及其理由；六门阈值与 credit source 未动。prereg
  文件一次性写入（重复冻结 fail loudly）。六门阈值沿用 S3-E 先例数值并
  由 runner 与模块常量互检；语料门 train ≥ 60 行 / heldout ≥ 20 行；
  split 按 case（episode sha）不跨界。
- **run 前置**：前置 b margin 审计 overall_pass 必须为真
  （`--skip-margin-check` 仅与 `--smoke` 联用，冒烟不产生判词）。
- 产物：report.json/md + artifact_manifest.json（prereg sha、六源文件
  sha、语料指纹、MPS attestation、margin attestation）。
- **首次 formal 判词（2026-08-13，`coding_lab_packet3_s3e_formal_20260813`）**：
  prereg v2、margin attestation = `coding_lab_packet3_margin_resolution_20260813`、
  语料 885 train / 362 heldout 行（288 / 125 case，来自 838 条轨迹）、
  5 seed、**admitted=true，六门全过**：

  | 读数 | 值 | 门 |
  |---|---|---|
  | noop NLL | 2.4831 | — |
  | gate 在线 NLL | 1.1760 | — |
  | always-on NLL | 5.3183 | — |
  | random-gate NLL | 4.4834 | — |
  | oracle 上限 NLL | 1.0123 | — |
  | convergence 改进 | 2.3461 | ≥ 0.2 |
  | gain vs noop（worst-seed CI 下界） | 0.9239 | ≥ 0.3 |
  | gain vs always-on（同上） | 4.6786 | ≥ 0.2 |
  | gain vs random-gate（同上） | 3.6472 | ≥ 0.2 |
  | gate selectivity | 0.9052 | ≥ 0.3 |

  结构完整性：无 free bias、zero-code 严格 noop、基底可训参数 0、
  RL 期间 reader / executor 参数未变。
  **读法**：always-on（5.32）比 noop（2.48）**更差**，随机择时（4.48）
  同样更差，而学到的 gate 做到 1.18、逼近 oracle 上限 1.01——赢的是
  **何时扳**这件事本身，不是"扳就有好处"。此前 margin 审计记录的基底
  反向对齐（正比例 0.36）在这里被兑现为可用余量。
- **诚实边界**：本判词是 junction 决策面上的 **NLL 读数**，不是端到端
  episode 通过率提升；语料标签是观测量（动作未随机化）；SHADOW 接线，
  未进 production ACTIVE。

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
- **首次 formal 判词（2026-08-13，`coding_lab_packet4_formal_20260813`）**：
  候选 `coding_lab_packet3_s3e_formal_20260813`，decision=**allow**，
  blocking_reasons 空；`validation_delta` 0.9239（= 该运行 worst-seed
  `gain_vs_noop_ci_lower_min`）；contract_integrity 1.0 /
  rollback_resilience 1.0 / fallback_reliance 0.0；
  `rollback_verified=true`，incumbent 为 genesis（首次上位）。

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

## 7.10 Packet 3.5：junction 介入式 RCT 标定（结构就绪，smoke 已过）

背景：§7.5 的 `credit_expert_actions` 是观察性条件通过率，已登记
survivorship/难度混杂。本包把 (状态键, 动作) 单元改为**随机化**测量，
是 Packet 3.6（episode 通过率判据）的 expert 目标来源。

模块：`lab/hands.py` 的 `ForcedActionAssignment` / `ForcedActionHand` /
`ConstraintAwareScriptedHand`（闭环 smoke 手）；`lab/junctions.py` 的
`transcript_protocol_state` / `state_key_for`（与 `extract_junctions`
同一状态机 SSOT）、`extract_forced_assignment` / ITT 表 /
`interventional_expert_actions` / `wilson_interval`；runner
`lifeform_evolution/coding_lab_interventional.py` + CLI
`scripts/run_coding_lab_packet35_interventional.py`
（smoke / freeze-prereg / formal，formal 强制 prereg SHA-256）。

- **随机化单位**：每 episode 由 seeded draw 指定 control 或一个协议动作；
  在首个命中目标状态键的决策点**一次性**实现：submit/investigate 由包装器
  直接实现（零真值泄漏），edit/test 经 context 指令约束内层手（有界重试，
  intention-to-treat：不服从保留自然动作与记录）。assignment 以 metadata
  记入轨迹 `hand_decision`，轨迹是唯一分析 SSOT，重复标记 fail loudly。
- **诚实边界**：ITT 通过率对"给定已到达状态的指派动作"是因果的；哪些状态
  被到达仍是观察性的。介入表与观察表分歧是**发现**（混杂实证），不是失败。
  本包不授权任何能力主张。
- **smoke 判词（2026-08-27，`packet35_smoke_20260827a`，development）**：
  4 链 × 8 集 scripted，32/32 触发（首决策状态键），intervention 28 /
  control 4，四类动作 compliance 全 1.0；expert map 空（单元支持度不足，
  预期行为——formal 规模才有资格产出 map）。修复两个机制 bug 并回归：
  相对 output root 会让 `git worktree add` 把 worktree 落进链 repo（现强制
  绝对化）；开环 ScriptedHand 被注入步打乱（smoke 手改为闭环 transcript 推进）。
- **formal 前置**：prereg 冻结目标状态键（取自既有 838 轨迹语料的高支持键）、
  cell 最小支持、margin、seed 与 API 手；预算随攒批确认。

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
