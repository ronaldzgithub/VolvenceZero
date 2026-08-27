# 四能力因果主线执行计划（Four-Able Mainline Execution Plan）

> 状态：**计划冻结稿（2026-08-26）**。本文件是"在本机持续实施、修正并以独立长陪伴、多 session、长 context
> 的因果实验，诚实证明 Volvence 的 Appendable / Readable / Learnable / Steerable 四项能力都实际起作用"
> 这一目标的唯一主线路线图与执行纪律 SSOT。
>
> 本文件**不授权**任何 formal run、CUDA 执行、公开锚发布、WiringLevel 变更或 production 晋升；
> 一切执行仍由各阶段独立冻结的 prereg / protocol / admission 授权。
> 本文件不改写任何既有冻结判词（attempt03、v5 失败封存、P1j/P1k/P1m、A1、S2/S3-D、#92 等原样保留）。
>
> 与 [`主线提升方案_2026-08.md`](../moving%20forward/主线提升方案_2026-08.md) 的关系：该方案定义
> A1/A2/B/C 工作流与全局晋升路线，本文件不复述、不修改；本文件收敛的是 **Relationship Lab
> 四能力因果实验主线**的阶段顺序与执行纪律，并把 2026-08-25/26 的执行复盘教训固化为硬约束。

---

## 0. 目标原文（不可改写）

> 在本机持续实施、修正并以独立长陪伴、多 session、长 context 的因果实验，诚实证明 Volvence 的
> Appendable、Readable、Learnable、Steerable 四项能力都实际起作用；完成前持续推进，
> 不以小样本、prompt 对比或代理指标冒充结论。

目标禁止的是**冒充结论**（小样本当充分样本、prompt 对比当因果证据、代理指标当产品效果）；
它不要求每次迭代都执行最重的锚定仪式。证据强度分档见 §2。

## 1. 现状快照（2026-08-26，证据基线 commit `7381743b`）

### 1.1 已有判词（全部冻结，不回改）

| 事实 | 数据 | 含义 |
|---|---|---|
| attempt03 终局 | `typed_control_product_horizon_executed_effect_not_observed` | full 对两条 strong baseline +22.92pp、对 credit-withheld / strict-noop +36.46pp 过门；对 frozen -9.38pp、对 unnamed reader -4.17pp、安全门、durability 门全败 |
| 消融退化 | credit_withheld 臂 `gate_update=0 / credit_applied=0`，对 full 的 action divergence 与 strict_noop 完全相同（163/192） | 不应用 credit ⇒ gate 从不更新 ⇒ 行为等价 noop；**Learnable 的独立贡献至今没有被任何实验测量过**，+36.46pp 不得归因于学习 |
| 失败面极小 | full vs frozen / vs unnamed 各 36 个分歧决策（union 50）；主窗口净差 -9/96 与 -4/96；`return_after_gap` -18.75% 来自 16 个决策中的净 3 个 | 失败可逐条审计；同时说明当前功效在掷硬币边缘 |
| reader 塌缩 | full 臂 named reader 192/192 全部输出 `agency_displacement`；reader-error 子集净差 -15/48、correct 子集 +6/48（correctness 与 condition 完全混杂，不作因果归因） | 命名读出内容失效是最可疑根因，但需 qualification + 写回审计分别闭合 |
| v5 资格执行失败 | launcher `SystemDrive/SystemRoot` 与 CPython `SYSTEMDRIVE/SYSTEMROOT` 键大小写在 parent 环境门 fail closed；root/nonce 已消耗，判 `invalid/incomplete` | 一个冻结协议 + 公开 Gist 锚被一个"彩排一次即可发现"的工程 bug 烧掉；v6 修复已提交（`7381743b`），尚无新 protocol/Gist/root/CUDA run |
| P4.6 双层状态 | 历史 development artifact `6b6f5e4f…34a99`：136 receipts、544 arm evaluations、13/13 checks PASS；当前 BIOS `A.B0`、microcode `0x120 < 0x12F`，且无真实 host-qualification root | **开发 CUDA 可继续；正式宿主资格仍未通过。**该历史 artifact 只证明 development-only synthetic proxy actuation，不授权正式 host-dependent Steerable 槽或四能力主张 |
| 四轴 ledger | 12 槽（每轴 mechanism → SHADOW → unseen single-axis）当前 **0/12** 正式 PASS | 任何 integrated run 无准入 |

### 1.2 执行复盘结论（2026-08-26 审计，本计划的直接动机）

约 40+ 小时 agent 工作时间中，产生新因果信息的实验仅 attempt03 一次（约 30 分钟）。
主要损耗：验证机器复杂度增长快于科学信息增长、公开锚确认逐个串行排队、
收敛包纪律失守（单 commit 584 文件 +44K 行）、多任务并发互相阻塞、
v5 因缺彩排烧掉公开锚。§3 的纪律条款逐条针对这些损耗。

## 2. 证据分档（开发档 / 正式档）

诚实的最低要求是：判据先于结果冻结、结果无论好坏原样封存、不选择性报告。
满足这三条即可产生合法证据；锚定仪式的重量按证据用途分档：

| | **开发档（development tier）** | **正式档（formal tier）** |
|---|---|---|
| 用途 | 消融修正、功效估计、写回审计、彩排、成本探针、诊断 | 每轴 unseen 单轴门、integrated Product Horizon、对外主张 |
| 预注册 | prereg JSON + git commit hash + seeds，冻结后不改判据 | 全套：独立 protocol identity、公开 Gist 首版锚、sealed evaluator 分离 |
| 运行时闭包 | 记录 model/weights/协议 hash 即可 | 完整 source-tree / runtime / process 闭包（现行 relationship-lab 口径） |
| 复验 | 可重跑脚本 + 固定 seed | 独立 validate-existing / fresh-clone replay / 第三进程比对 |
| 结论上限 | `development`，不得进入四轴 ledger 正式槽、不得对外引用 | 按各协议冻结判词 |
| 人工确认 | 本地收敛包 commit 已获持续授权，完成即提交 | 仅外部副作用（push、Gist、formal execution、硬件操作）按 §3.4 攒批确认 |

**硬规则**：开发档结果**永远不能**升级为正式档证据——需要正式结论时用正式档重跑；
反之，正式档协议冻结前的一切探索必须在开发档完成，禁止拿正式锚做调试。

## 3. 执行纪律（八条硬约束）

1. **正式档默认彩排，显式豁免时承担工程失败风险**：未来"冻结协议 + 公开锚 + 一次性执行"链路
   默认先用开发档一次性协议跑通相同代码路径；但本轮 Product Horizon gate/pulse/theta/admission 路线与
   reader v6 已按用户裁决关闭 rehearsal，除非用户重新开启，不生成 2–3 item rehearsal artifact。无彩排
   只接受更高的执行失败/烧锚风险，不放宽预注册门、失败封存、分歧门、lineage 或禁止事后调参的纪律；
   public/outcome-free treatment-reachability admission 是永久机制门，不是 rehearsal，仍必须执行。
2. **收敛包尺寸**：恢复 AGENTS.md §8 口径——一包一 owner、3–8 个关键文件、完成即提交。
   禁止再出现数百文件级 dirty tree；实验 artifact 与代码分包提交。
3. **单写入者**：同一时间只有一个任务/agent 拥有本工作区写权限；其他任务只读。
   跨任务协调走 goal / 阶段汇报，不靠互发停止指令。
4. **确认攒批**：待人工授权的外部动作（push、Gist 发布、formal execution、硬件操作）维护成
   单张待办清单，攒批一次确认；本地 3–8 文件收敛包完成即提交，不再逐次询问。agent 在等待期间
   不做重复性只读盘点，转而推进不依赖该授权的并行工作或结束 turn。
5. **提交节奏**：任何超过一个收敛包的未提交改动都是裸露风险；
   每包完成即提交（按 AGENTS.md §13 中文规范）。
6. **硬件前置不悬置**：任何硬件/宿主前置条件只允许两种状态——已完成，或已正式
   记录"推迟 + 对应证据线暂停"。禁止维持"永远差一步"的默认绕行。
7. **周校准问题**："本周产生了多少新的因果信息？"每周自查；若连续一周答案为零而
   脚手架代码仍在增长，暂停脚手架、优先执行最近的开发档实验。
8. **三本账分离**：A1（七日 fixed-script 传导线，R-A1b 5.93e-05 只属于此账）、
   A2（MSC 预算线，501 dyad ≈880h 为 Mac/MPS 历史计价，50–100 dyad 中等矩阵属于此账）、
   Product Horizon（24/112/160/537 roots 与本计划各阶段属于此账）。跨账只共享方法论与
   诚实边界，禁止共享 estimand、N、protocol identity 或 evidence claim。

## 4. 阶段路线图

每阶段列出：入口条件 → 交付物 → 出口判据。阶段内允许并行；跨阶段依赖不得跳过。
除 Phase 0 外，每阶段的执行档位已标注。

### Phase 0 · 工作区收口（已完成：`15887ed1`，无实验）

入口：无。
交付：

1. dirty tree 已按 owner 分包提交：reader qualification v4–v6 谱系/preflight `96feb2ae`、
   attempt03 事后诊断 `c3850986`、power planning scaffold `61c5575f`、计划 SSOT
   `08684268`/`9a0b7d43`/`5406c035`。
2. 单写入者已声明：当前主任务是唯一写入者，其余 agent/任务只读。

出口已于 `15887ed1` 登记。判定口径是“本包目标路径清洁 + 既有 dirty-path allowlist 明确”，
不是为了追求全仓零噪声而回退用户 submodule、删除历史 artifact 或反复清理后续阶段的未完成根。

### Phase 1 · attempt03 诊断闭合 + reader qualification 执行（开发档 + 正式档各一件）

入口：Phase 0 完成。
两条线并行，互不阻塞：

**1a（开发档，已完成）attempt03 写回审计闭合**：`7c2f24ba` 已对 full/frozen 全部
384 个 transition 做机械重放，
逐行闭合 settlement→PE→credit→gate→owner→下一拍；中间 gate checkpoint 标记
`rederived_endpoint_bound`。同时闭合 reader 塌缩根因（为什么 192/192 全输出
`agency_displacement`：`b7d1796e` 已封存常数偏移下的几何诊断）。判词上限仍为
`post_hoc_mechanical_writeback_replay_only`，不作因果归因，不重复运行。

**1b（正式档，当前推迟）reader qualification v6 执行**：rehearsal 已按用户裁决关闭；恢复时先跑
model-free protocol/runtime preflight，随后直接冻结新 execution protocol、发新 Gist（不复用 v5 锚）并
一次性执行 CUDA qualification，同时接受执行路径 bug 可能再次消耗该锚的风险。判词上限
`exact_source_reader_development_admitted`（224/224 row + 28/28 group + margin ≥ 0.01，
预注册门不改）。2026-08-26 的首个 rehearsal 在完成 formal-path 多轮完整性读取期间由用户停止；
未完成 root 原样保留为 non-evidence，不消耗 protocol/Gist/formal execution。该线不再阻塞 Phase 2/2.5，
但在 Readable unseen formal gate 前仍须回到此处完成资格链。

出口：1a 审计 artifact 已封存，因此 Phase 2/2.5 可继续；1b 是独立 Readable formal 前置，
恢复时必须得到 PASS 或 FAIL 终局（FAIL 原样封存，回到 reader 设计修正，不换门重试）。

### Phase 2 · 协议修正设计（开发档为主）

入口：Phase 1a 完成（1b 可仍在途）。
交付四个互相独立的收敛包：

1. **Learnable/Steerable 新消融拓扑（prereg-ready 机制已完成）**：`741e7b1b` 冻结
   nonzero theta0、forced exposure、exact PE-credit batch 与 atomic apply/withhold；`7d476101` /
   `f6ae1f6a` 冻结 preference owner settlement replay 与 canonical persistence identity；
   `bc9f1b0a` 冻结共同 candidate 后 executor-only apply/strict-noop、actual delivered action 与完整
   owner replay receipt。未来 protocol 中 `frozen_theta0` 臂必须与 full 共享同一
   theta0、同 forced-action、同 reaction/outcome/PE/credit 流，唯一差异是参数
   atomic apply bit。receipt 必须证明 `campaign_online_update_count=0`、参数 delta
   精确为零、评估期至少一次非 noop steer；strict-noop 臂与 frozen 臂共享 gate decision
   与 intervention candidate，只切 executor apply bit。任一证明缺失判
   `arm_degeneracy_invalid_contrast_no_claim`。对比定义冻结为：
   Learnable = full − frozen_theta0；Steerable = frozen_theta0 − strict_noop。
   当前只完成 owner/consumer 机制；campaign-level `campaign_online_update_count=0`、参数零 delta、
   非 noop opportunity 与 arm divergence 仍须由新 development protocol 证明，不能登记能力效果。
2. **source-v3 campaign admission（已完成）**：development protocol
   `98d51d84…6338` 只复用现有 reactive environment，不重建 engine；独立收敛包物化 32 onboarding、
   192 decisions、576 条 sealed action-counterfactual commitments，双 model-free worker replay + 第三进程
   byte-exact 比对。artifact `982d6f8e…0f59` 精确绑定实现提交 `d8119f25…` 并通过独立
   `validate-existing`；上限 `campaign_input_admitted=true`，不授权 campaign execution。环境 draw hash
   包含 selected action，明确是 action-specific potential-outcome randomness，不冒充 common-random-number。
   注意
   source-v3 exact texts 已被 v5/v6 资格链见过，只能用于 admission 与 SHADOW；
   各轴 unseen 单轴门必须另立 disjoint source revision。
   **112-root source-v4 owner 与 input admission（已完成）**：独立 owner 不修改上述 8/24 source-v3
   envelope 或 admission closure；冻结 112 synthetic roots ×（4 onboarding + 8 collection + 40 evaluation），
   onboarding 为旧 base policy，collection 与 evaluation 共享反转后的 complementary policy，避免在冻结
   evaluation gate 时把过时学习预先设计成负效应；五个 evaluation segment 各 8 条。public trajectory 与
   去身份 causal tape 均须 112/112 唯一，5,376 个 environment seed 全异；现 owner 可确定性 replay
   16,128 条三动作 branch、不拥有 collection forced action。development admission protocol
   `b3988b21…2102` 以 implementation commit `b55ce6ad…` 生成单份六文件 compact commitment artifact
   `4cc1ec45…dd54`，pre-manifest 第二次全重建/磁盘 byte compare 与独立外部双 ID
   `validate-existing` 均通过，因此只提升 `campaign_input_admitted=true`。source protocol/public/sealed identity
   为 `dbf05262…1d91 / f46336a9…86d6 / 51900ec7…8936`；reader table、theta0、forced-action schedule、
   campaign、模型/CUDA 与效果 authority 仍全部为 false/zero。
3. **功效修正**：下一版 Horizon 每个 durability segment ≥ 6–8 个决策，或增加
   matched worlds；每轴的 N 由该轴独立 power 选择（既有 screen 已显示某 5pp 场景
   需 N=538，禁止用 537 冒充充分样本）。分段样本量必须随报告发布。
4. **A2 预算门（独立账）**：先做 no-model / cost-only CUDA 探针重新计价（880h 是
   MPS 历史价），再决定是否冻结 50–100 dyad 中等矩阵 prereg；中等矩阵只产出
   方向/方差/ICC/可行性，不作正式证据，也不与 Horizon roots 混账。

出口：四包各自提交，协议设计冻结为 prereg-ready 草案（尚不执行）。

### Phase 2.5 · 修正后的 Product Horizon 开发矩阵（开发档，下一因果目标）

入口：Phase 2 包 1 的 gate owner 与 executor consumer 分包完成，且所选 Product Horizon source 已
通过对应 development admission；不等待 reader v6 formal 链。现有 27 项 owner/executor/gate 定向测试已覆盖
机制与 receipt，因此按用户裁决不另启 reader rehearsal 或独立 2–3 item 彩排 artifact；新 worker 先通过
model-free `validate-existing`。natural-APPLY dynamic collection-prefix、forced common-batch protocol、首个唯一
materialization、external-ID `validate-existing` 与 campaign-input public seam 均已闭合。三臂 campaign protocol
`bc4c0882…587c`（raw `9374a71b…19fa`）由实现提交 `73e9ed7a…64f91` 完成首个唯一中等
root-cluster matrix；artifact `05e97726…6300` 与纯读取 exact replay 已由证据提交 `41ebea0f…64edf` 封存。
该运行没有通过 Learnable manipulation gate，因此当前阶段从“跑矩阵”转为“修复唯一 gate owner 后建立新的
treatment-reachability prerequisite”；禁止重跑旧矩阵、降低分歧门，或把 unit test/参数变化称为效果证据：

当前 forced common-batch 收敛包已选择 112 个 **root-local** schedule/batch，而不是一个 896-credit global
batch。protocol `dd0d28a7…aff93` 预注册 public-position schedule 父索引 `13c3f8e2…4e9f`、每 root 只执行一次
八拍 collection、共同 terminal owner persistence，以及 full/frozen/strict 的 APPLY/WITHHOLD/WITHHOLD transition。
它只读取 dynamic PASS manifest，dynamic natural trace/outcome/credit 读取数为零。implementation
`5d028d9c…2798` 的首次且唯一 112×8 materialization 已发布 artifact `92880fb7…40d7` 并 PASS；显式双 ID
`validate-existing` 重放通过且 6 文件 bytes/size/mtime 不变。112/112 roots 的 APPLY 参数均非零且无 cap-hit，
但该 PASS 仍只授权 campaign protocol freeze。按用户裁决未生成 2–3 item experimental rehearsal artifact；普通
unit tests、schedule identity 校验和 model-free `validate-existing` 保留。

该 campaign 的 112 roots × 3 arms × 40 evaluation 已完整执行；collection 0–7 仍排除于 effect estimand，
evaluation credit 仍全部不 apply。full 与 frozen_theta0 的 4,480 个 actual action 全同，尽管 learned/cold
checkpoint 在 112/112 root 不同，Learnable 仍必须判
`arm_degeneracy_invalid_contrast_no_claim`；参数移动不是 treatment。frozen_theta0 与 strict_noop 有
4,325/4,480 actual divergence，Steerable 仅得到 development GO candidate，不升级为能力效果。

theta0 v3 的机制路线冻结为三个小包加一个 cold evaluation 接缝；gate、pulse、bootstrap materialization/validation
与 condensed-theta0 authorization 均已闭合。protocol.v1 attempt01 因 typed assignment
ID 误用在首条 persisted preaction 前失败；attempt02 虽完成 112×8 materialization，却因 temporal publisher
墙钟进入 forced receipt 而未通过 byte-exact `validate-existing`。两根均禁止续跑/覆盖且无下游 authority。
live protocol.v2 另立 identity，冻结 owner-side logical clock 与完整 publisher/pulse lineage；theta0 v3 bootstrap
attempt03 已在新 create-only 根完成正式 materialization，并在 detached exact implementation commit 上通过 external
双 ID 完整重放；全程没有恢复 2–3 item 彩排：

1. **已闭合** versioned gate operator v2 与 add-only federated parent：无 free bias、移除恒定 support、四个 centred typed feature；只消费完整
   `RelationshipActionCommonBaselineCredit`，以完整 fixed-balanced schedule membership 计算事前 half-centred
   development feature-moment score，推荐动作为 noop 时 zero-information；精确 balance 不冒充随机化/ATE/CATE，
   每个实例只允许一次 cold APPLY/WITHHOLD；learned theta0 的构造/恢复必须由 parent+batch+APPLY receipt
   完整重放且拒绝净零参数，advisory 必须由 frozen policy+forecast 重放 frozen decision。flat 896-entry batch 因
   O(N²) serialization/content-ID 成本已在 theta output 前否决；新 parent 以连续 offset 绑定 112 个完整 8-credit
   provenance child、全局校验唯一性/时间顺序，并只产生一个 unique plan identity、一次 parent APPLY/WITHHOLD 与
   一个 terminal checkpoint。APPLY 固定 `atomic=1 / child_transition=0`；federated transition 必须先凝结为 cold theta0
   才能 evaluation。同序 flat reference 的 candidate checkpoint byte-exact；旧 flat v2 golden ID 与 v1 全部原样保留；
2. **已闭合** v2 pulse/forced consumer 与 add-only federation join：以独立 `.v2` authorization/command/receipt/snapshot mechanically 绑定
   完整 assignment receipt、policy+forecast replay 后的 v2 frozen decision、actual temporal delivery、
   owner→Social PE→parent/common-baseline credit；pulse-owned continuous collected batch 保留完整 settlement、
   owner forecast-publication replay 与全程连续 post→input handoff；add-only segmented batch 绑定唯一 segment scope、
   显式 immutable owner start 与段内连续 handoff，并在 flatten 后完整顺序耗尽同一 schedule。generic pulse 只证明
   caller 提供的 segment start，不把它命名为 root reset；theta0 v3 owner 必须另证每 root 从 `None` 精确重放四条
   onboarding。raw gate-only batch 不可提交，continuous/segmented 各用互不兼容的专用 transition/commit；
   APPLY/WITHHOLD 必须由同一 collection 成对派生并 exact-match plan/pre/candidate checkpoint。strict-noop 只切
   executor disposition，外部 base authorization 不能直接打开 ACTIVE；v1 与既有 continuous v2 receipt/payload/ID
   均不复用、不改名。两类 collection 还在构造时 seal owner provenance，并在 ID/export/transition 边界重放校验，
   防止 shallow-frozen `OwnerPersistenceSnapshot.payload` 的 nested mutation；有效 continuous 的公开 constructor
   signature、canonical payload 与 ID 不变。federated collection 必须显式取得完整 parent schedule 与按 parent order
   排列的完整 segmented child components；全局拒绝重复 collection/segment/scope 与非递增 credit time，loader 必须
   同时取得完整 typed parent/children，不能靠 compact ID 重建。matched wrapper 只调用 gate-owned parent factory，
   只产生一个 APPLY/WITHHOLD pair，固定 `child_transition_count=0`，不循环 child transition、不生成旧
   `FrozenPolicy`。本机制只能证明 membership/order/timestamp ordering；parent 在首个 forecast/outcome 前 create-only
   持久化必须由第三包的唯一 materialization receipt 另证；该 receipt 只能从 actual APPLY/WITHHOLD 建立
   accepted persisted lineage 内 `child_transition_count=0`，不证明 external/OS 路径不存在；
3. **已闭合 protocol.v2 materialization + validation** theta0 v3 owner：live protocol
   `f5c33f5c…1d26`（raw `83060179…e38f`）事前冻结 bootstrap `1/512`、root-online `1/4`、
   cap `4.0`、一个含 112 个 provenance child / 896 credit 的全局 federated development parent，以及
   cap-hit、零 informative update、非有限/全零参数与不完整计数的 FAIL 门。materialize 与 validate 都必须证明
   supplied commit 存在且等于 HEAD，runtime/temporal/pulse/theta/protocol/CLI 六个 git blob 属于该 commit，且 `packages` 与本 CLI
   无 tracked/untracked 漂移；输出根还必须与 source/reader 冻结根及仓库 `packages/scripts` 代码域隔离。parent schedule create-only 写入、
   flush/fsync、同句柄回读、close/reopen byte-exact 后，durable owner-path receipt 才能成为 ledger row 0；
   每 root 从字面 `None` 重放四条 onboarding，448 次 onboarding write、784 次段内 handoff 和 896 次
   outcome writeback 均为终局门。每拍 temporal delivery 时间固定 `root*20+4+2*decision`，credit 固定晚 1；
   896 个 producer timestamp 全局递增，禁止 wall clock、consumer replace 与 validator normalization。收齐后只调用一次 parent transition factory，最终 child count 必须来自
   APPLY/WITHHOLD actual receipt，不得硬编码为零；closed trace 还必须在 transition/manifest 前 reopen byte-exact。
   v1 attempt01 三文件 partial root 与报告 `421ad709…e733e` 原样保留。attempt02 六文件约 41MB 根也只在本机
   原样保留：materialization manifest `0c596cd5…076c` 的正字段被随后 validation FAIL 否决；首差在 ledger row 6
   `temporal_projection.timestamp_ms` 及派生 forced receipt ID，compact 报告 `a795fdcf…5cf7`。theta0 v3 bootstrap
   attempt03 在 implementation `af6cc60b…ad74` 上完成 112 roots / 896 credits，发布 artifact
   `dde0fc78…9777` 与 learned theta0 `dded60c6…2f7`；APPLY 为 896 update / 839 informative / 0 cap-hit，
   WITHHOLD 为 0 update，accepted-lineage child transition 为 0。自动 post-materialization validator 因并发
   docs/business commit 将 main HEAD 推进到 `e7817b93`，在 `_run_bootstrap` 前 exact-HEAD guard 正确拒绝，未观察
   artifact content drift；随后 detached `af6cc60b` worktree 的 external 双 ID `validate-existing` 全量 replay PASS，
   六文件 fingerprints 不变。compact acceptance receipt `4c3c414b…aab5`（raw `cae85ed0…af6c`）只接受 development mechanism terminal，
   `effect_scientific_terminal=false`。
4. **已闭合 condensed learned-theta0 cold evaluation authorization seam**：新增 add-only
   `RelationshipProductV2CondensedTheta0FrozenPulseAuthorization`，构造与 payload 恢复都必须取得完整 pulse
   federated matched wrapper，并在 load 边界重放 wrapper integrity、parent APPLY/WITHHOLD，再由 gate canonical
   factory 重导完整 learned artifact 并 exact-equal（含 learning rate/cap）；随后只保存深不可变 lineage
   projection、learned artifact 与 cold policy，不把含
   shallow-frozen owner payload 的 producer graph 带入 consumer snapshot。evaluation policy 必须为
   `update=0 / informative=0 / processed_credit_ids=()` 且无 transition batch/receipt；source APPLY 只记录在新
   authorization payload，executor receipt 的 evaluation `gate_transition_disposition=null`，不得伪装成 APPLY/WITHHOLD
   在线更新。candidate 与 strict-noop 共享 exact forecast/policy/decision/advisory，唯一 command treatment 仍是
   executor disposition。legacy continuous v2 authorization/payload/ID 分支保持原样。model-free fixture 为构造
   authorization 重放了 synthetic forced settlement→PE→credit 与一次 parent APPLY/WITHHOLD；condensation 后的
   candidate/strict evaluation preaction 没有新增 outcome、PE、credit 或 gate update。全程未运行模型、CUDA、campaign
   或彩排，也尚未把 attempt03 artifact 用作新 source 的 treatment-reachability 证据。

三个机制包与 cold evaluation authorization seam 现已闭合。下一包另立 permanent
treatment-reachability admission；它只读新 source 的 public/outcome-free surface，不生成
outcome/PE/credit/update，不是 rehearsal。当前 `treatment_reachability_admitted=false`。

permanent admission 未闭合前不冻结或启动下一次 integrated/development campaign。新效果矩阵必须使用新的、
预先 sealed reactive source；当前 source-v4 只可作为已花费的 adaptive development training lineage。

1. 冻结选择 N=112 的 development operational marker；每 root 的 8 条 matched collection 不进入效果
   estimand，随后五段各 8 条、共 40 条 evaluation。112 是 synthetic roots，不是真人或 formal N；
2. `full − frozen_theta0` 估计 Learnable 的方向、方差与 root-level ICC；
3. `frozen_theta0 − strict_noop` 估计 Steerable 的方向、方差与 root-level ICC；
4. 同时报告非 noop 机会数、arm divergence、每 segment 决策数、wall/GPU 成本与 full-N feasibility；
5. 方向为负或接近零则停止对应 formal 大运行；方向为正但低于 practical floor 仍如实记为
   `directionally_positive_below_practical_floor`，不得改门；
6. 本矩阵使用 Product Horizon roots，**不是** A2 的 50–100 dyad 中等矩阵，二者不共享 N、estimand、
   protocol identity 或 evidence claim。

出口：development 终局和 stop/go 建议封存；无论结果如何都不进入 12 槽 formal ledger。

### Phase 3 · 四轴单轴门（正式档）

入口：Phase 2 包 1、2 完成；Phase 2.5 对应 contrast 已封存且没有触发 stop；对应轴的 SHADOW receipt 就绪。
内容：按 12 槽 prerequisite ledger 逐轴推进 mechanism → SHADOW → unseen single-axis：

| 轴 | 当前机制层 | 缺口 |
|---|---|---|
| Appendable | hydration/owner-history 前置证据（`675815b9…2052` 等） | SHADOW 槽 + unseen 单轴门；须直面 attempt03 的 frozen-better 负信号 |
| Readable | Phase 1b 的 reader admission | 合格 SHADOW + unseen 系统门（disjoint source） |
| Learnable | PE→credit→gate 可重放 | `frozen_theta0` 新臂的 unseen 单轴证据 |
| Steerable | fit + P4.6 development-only physical artifact `6b6f5e4f…34a99` | 产品 SHADOW + 非 noop receipt + unseen 单轴效果；依赖真实 residual/host 的 formal 槽在 host qualification 前暂停 |

每个 unseen 单轴门使用与既有资格链 disjoint 的 source revision，正式档全套锚定。
ledger 授权字段由 validator 派生，当前合法终局为
`valid_incomplete_prerequisite_ledger_integrated_run_not_authorized`。

出口：12/12 槽 PASS ⇒ 仅授权 integrated protocol freeze（不直接授权执行）。

### Phase 4 · integrated Product Horizon（正式档，唯一四轴合取实验)

入口：Phase 3 出口 + source admission + power 冻结 + host admission + execution admission 各自独立过门。
内容：全新 protocol identity（不复用 attempt03/v2 任何 root），包含 Phase 2 的全部修正
（frozen_theta0 臂、加大功效、分段样本量披露），经 model-free 真实执行路径校验后冻结、公开锚定、一次性执行。
彩排默认关闭；只有未来独立事前裁决证明不可替代时，才另立不进入效果证据的彩排协议。

诚实终局集合必须包含：
- 预注册门 PASS；
- FAIL（含 `directionally_positive_below_practical_floor`：方向为正但低于预注册最小效应，
  一律判 FAIL，同时单独报告方向、CI 与门槛缺口）；
- `arm_degeneracy_invalid_contrast_no_claim`（机制 receipt 不闭合时）。

若重复出现同向小正效应：唯一合法升级路径是"更大冻结基底 + 全新 prereg + 全新
held-out source + 全新 evidence chain"；禁止事后降门槛、重解释旧 run 或把换基底
混入同一 estimand。

出口：终局判词封存。PASS 仍限定为 synthetic typed environment 的系统级证据。

### Phase 5 · 真人验证（本机范围之外，仅登记）

integrated PASS 之后的产品级验证需要独立真人受试者、知情同意、去标识化、安全审查与
用户可见生成，合成环境结果不能替代。本计划只登记该边界，不设计其协议。

## 5. 决策门总表

| 门 | 时点 | 选项 | 默认 |
|---|---|---|---|
| 宿主裁决（已完成） | Phase 0 | 推迟正式 host qualification；development CUDA 继续；host-dependent formal Steerable/Integrated 线暂停 | 不以 BIOS settings 变更替代 microcode/qualification |
| reader FAIL 分支 | Phase 1b 后 | 修 reader 设计（开发档迭代）再走新资格链；不换门重试 | — |
| Horizon development stop/go | Phase 2.5 | Learnable/Steerable 方向、方差/ICC、成本与 practical-floor 缺口 | 负或近零停止 formal 大运行 |
| A2 预算 | Phase 2.4 | cost probe 后：中等矩阵 / 换硬件计价 / 压采集成本 | 先 probe，禁止直开 501 |
| frozen-better 复现分支 | Phase 3 Appendable | 若修正后 Appendable 单轴门仍为负：如实封存"持续写回在该环境无净增益"，四轴主张按缺口收窄，不得绕过该轴 | — |
| integrated 授权 | Phase 4 入口 | 仅 ledger 12/12 + 各 admission 齐全 | 缺任一即 `not_authorized` |

## 6. 禁止清单（继承并固化）

- 用 attempt03 的 +36.46pp 声称 Learnable 独立贡献（消融已证退化）。
- 用 A1 的 5.93e-05 诊断 Product Horizon source（跨账）。
- 用开发档结果、彩排产物、pilot/中等矩阵结果冒充正式证据。
- 在四轴 ledger 未齐前启动 integrated Horizon CUDA run。
- 复用已消耗的 root/nonce/公开锚；编辑或删除失败锚。
- 事后修改任何已冻结判据、降低门槛、选择性报告分段结果。
- 把 evaluation/judge 分数回灌学习环路（R12，全程有效）。

## 7. 变更纪律

本文件更新时机：任一阶段出口达成、决策门裁决、或某阶段判词触发路线变更时，
追加带日期的变更记录，不回改历史段落。各阶段执行细节与判词以对应 prereg JSON
与 artifact 为准，本文件只做路线图、纪律与对账。

- 2026-08-26 · 初版冻结：固化 attempt03 复盘（36/50 分歧、消融退化、reader 塌缩、
  功效量化）、v5 失败教训（彩排强制）、证据分档、八条执行纪律与 Phase 0–5 路线。
  本记录不授权任何 run。
- 2026-08-26 · 按用户确认宿主/BIOS 问题已解决，从计划中整体移除相关裁决项与约束
  （§1.1 宿主行、Phase 0 BIOS 裁决、§5 BIOS 门、Steerable 物理线约束）；§3.6 保留为
  通用"硬件前置不悬置"纪律。既有 host-block receipt 等历史 artifact 不受本次删除影响，
  原样保留。
- 2026-08-26 · Phase 0 出口达成：dirty tree 已按 owner 分包提交（attempt03 事后诊断
  `c3850986`、功效规划脚手架 `61c5575f`、本计划 spec `08684268`/`9a0b7d43`/`5406c035`），
  `git status` 除 `external/vz-bundle` 既有改动外清洁。用户已裁决单写入者：本主线由当前
  Cursor 主任务独占写入，其余任务（含既有 Codex 任务）转只读；用户同时裁决 32K 长上下文
  操作化不纳入本计划 Phase 2，留给 P4.7 线。本记录不授权任何 run。
- 2026-08-26 · 事实修正：`9a0b7d43` 把“用户已修改 BIOS settings、CUDA 可重试”误记为
  “正式宿主前提已解决”。后续实测仍为 BIOS `A.B0` / microcode `0x120 < 0x12F`，且不存在
  真实 host-qualification root；同时历史 P4.6 artifact `6b6f5e4f…34a99` 已以 13/13 checks
  封存 development-only PASS。现明确裁决为“development CUDA 继续；formal host qualification
  推迟；host-dependent formal 线暂停”，不重写上述错误提交或任何历史 artifact。
- 2026-08-26 · Phase 0/1 状态校正：Phase 0 不重复；Phase 1a 已由 `7c2f24ba`/`b7d1796e`
  闭合。首个 reader v6 rehearsal 由用户停止，未完成根保留为 non-evidence，formal reader 链推迟且
  不阻塞新消融。新增 Phase 2.5 Product Horizon development root-cluster 矩阵作为下一因果目标；
  本地小包 commit 已获持续授权，外部动作才攒批确认。本记录不授权 formal run。
- 2026-08-26 · Phase 2 机制推进：`741e7b1b`、`7d476101`、`f6ae1f6a`、`bc9f1b0a`
  已把 frozen-theta0 的 gate owner、preference settlement/persistence identity 与 executor-only strict-noop
  consumer 闭合到 prereg-ready 边界；尚无 campaign-level non-noop/update/divergence/effect 证据。
  source-v3 campaign-input admission protocol/owner 冻结为 `98d51d84…6338`；精确绑定 `d8119f25…` 的
  artifact `982d6f8e…0f59` 已由 model-free cooperative 双 worker 物化并通过第三进程与独立
  `validate-existing` 的 byte-exact 复验。PID 只作 self-reported receipt，
  `fresh_process_independence_proven=false`；唯一新增正判词为 `campaign_input_admitted=true`。
  用户关闭的 reader rehearsal 继续不启动。本记录不授权 campaign、CUDA、formal 或 integrated run。
- 2026-08-26 · source-v4 medium input 完成：`36466b59` 冻结 112 roots × 48 decisions 的独立 owner；
  `b55ce6ad` 冻结轻量 development admission；artifact `4cc1ec45…dd54` 由 protocol
  `b3988b21…2102` 生成并通过独立外部双 ID `validate-existing`。正式树只保留一份 public/evaluator 与
  16,128 个 branch digest，未复制 replay A/B/PID comparator；结论上限仅 `campaign_input_admitted=true`，
  不授权 reader/theta0/forced action/campaign/model/CUDA 或任何能力效果。
- 2026-08-26 · Phase 2.5 当前入口裁决为 NO-GO：在冻结 campaign protocol 前进行的未预注册、post-hoc
  public-only scan 中，最终 `f44eb267…3398` cold theta0 对 source-v4 的 5,376 个 reset-state probe 全部
  gate `noop` 且实际交付 `neutral_noop`；只有 112 个是可达首 preaction，其余 5,264 个是 counterfactual
  stress probe。对其自身 source-v3 校准 forecast 的终态回放也为 192/192 noop，说明在线 trace 的
  164/192 steer 不能代表最终冻结 policy；terminal-order/重复截距仅作诊断，不冒充已证明的唯一根因。
  当前 112-root 三臂 campaign 保持 `campaign_blocked_effect_not_tested`，须在全新 development theta0 lineage
  中事前冻结 bootstrap 与一次性 opportunity 门后才能恢复；reader rehearsal 继续关闭。本记录不授权
  campaign、qualification、formal、integrated 或任何四轴效果主张。
- 2026-08-26 · theta0 v2 forced-batch bootstrap 设计冻结：新 adaptive development lineage 显式声明
  source-v3/f44e 双重使用；192-entry public-position schedule 以块状公式固定为 symbolic role 96/96，实际
  action 经 executor/environment/owner PE-credit 链结算但不在线 apply，唯一 batch 对同一 plan 同时发布
  APPLY/WITHHOLD receipt。完整 batch terminal、非零 delta、未触 cap、cold 0/0/0 与 training-support physical
  nonnoop 均过门才发布候选 v2；FAIL 禁止换顺序/seed/threshold/bias/早期 checkpoint 重试。该包尚未执行，
  成功也只进入另行冻结的 source-v4 transductive public opportunity scanner，不授权 campaign；reader rehearsal
  继续关闭。冻结 protocol `dfefb9fa…841e`（raw `fd3fa87b…2867e`）；forced actual 与 frozen training-support
  nonnoop 口径分离，全零 terminal 也封存为 no-consumable FAIL，不留下 incomplete root。
- 2026-08-26 · theta0 v2 首个终局已封存并 exact replay：implementation `66c2d83a…8dd3`、artifact
  `9012b52f…d00e`、candidate theta0 `3acd6e4f…9344`；forced actual nonnoop 95/192，唯一 batch 的
  APPLY/WITHHOLD 分别为 `192 updates + 1 commit` / `0 updates + 0 commit`，candidate 在双重使用的
  training-support 上为 190/192 nonnoop。判词严格为
  `development_theta0_v2_materialized_training_support_opportunity_only_effect_not_tested`：它只解除 theta0
  本地退化阻断，不是独立能力效果。source-v4 transductive public opportunity、collection-prefix 动态门与
  campaign 仍未过门，Phase 2.5 继续 NO-GO；reader rehearsal 继续关闭。
- 2026-08-26 · source-v4 transductive public opportunity scanner 已冻结待一次性执行：protocol
  `4471c9ab…4d83f`（raw `9ce246b2…35419`）只读 source-v4 public plan、development reader table/artifact 与
  theta0 v2 cold policy。112 个 index-0 首拍与 5,264 个 reset-state stress probe 分账；5,376 个
  APPLY_CANDIDATE 全部走真实 typed placeholder executor，PASS 只按 temporal delivered nonnoop，并要求 reachable/
  evaluation 各一个相同 prestate 下的 strict-noop actual-action divergence witness。PASS 只提升 protocol-freeze
  authority，collection-prefix execution 仍为 false；未来另行冻结的 8-decision dynamic protocol 中，112 条首
  preaction 的 owner/forecast/decision/action/executor/advisory/
  temporal-controller/cold-policy v1 projection 必须 exact match；不授权 evaluation/
  campaign/effect/formal/四轴。reader rehearsal 按用户裁决继续关闭，本包不另建 2–3 item 彩排 artifact。
- 2026-08-26 · transductive scanner 首个且唯一终局已封存并 exact replay：implementation
  `0ffda0a1…1373`、artifact `2dec2e3f…774e4c`。5,376 个 APPLY probe 的 temporal delivered nonnoop 为
  reachable `107/112`、collection reset-state stress `740/784`、evaluation reset-state stress `4,279/4,480`；后两类仍是
  post-onboarding reset-state counterfactual probes。两个 canonical APPLY-vs-strict witness 均闭合，112 个
  evaluation roots 均至少一个 delivered nonnoop，cold checkpoint 全程 `0/0/0`；复验不改变 artifact bytes/mtime。
  因此只提升 `source_v4_opportunity_established=true` 与
  `collection_prefix_protocol_freeze_authorized=true`。`collection_prefix_execution_authorized=false` 保持不变，
  Phase 2.5 的下一合法动作是另立 8-decision collection-only dynamic protocol 并 exact-match 23 字段首拍 seam；
  5 个首拍 noop root 不得筛除/替换，两个 witness 不作 aggregate Steerable。新协议必须在 natural APPLY sequential
  gate 与 forced common-collection batch materializer 之间明确选择；若后者不匹配 seam 就停止重设计，不能削弱门。
  evaluation/campaign/effect/formal/四轴继续 NO-GO，reader rehearsal 继续关闭。
- 2026-08-26 · scanner 后续 lane 已明确选择 natural `APPLY_CANDIDATE` sequential dynamic gate，未把 forced
  common-collection 混入首拍 frozen seam。新 protocol `47cea5fa…0dbb2`（raw `3f3f7ad3…0d97f3`）保留全部
  112 roots，每 root onboarding 一次、只跑 index 0–7；112 个首拍逐行 exact-match 23 字段 projection，后
  784 拍必须消费 exact prior settlement owner state。每拍 preaction append+fsync 后才按 temporal delivered
  action 打开 precommitted source-v4 branch并生成 owner writeback→social PE→credit，credit 不在线 apply，cold
  checkpoint 始终 `0/0/0`。PASS 也只授权冻结 forced common-batch protocol，不授权其执行、evaluation、campaign
  或任何能力效果。实现与定向测试已就绪，待冻结提交后首个唯一 materialization；reader rehearsal 继续关闭。
- 2026-08-27 · natural-APPLY dynamic collection-prefix 首个且唯一终局已封存并 exact replay：implementation
  `c275bd90…ade4`、artifact `f1a5b2f6…aaf4`、1,906-row trace `e9cf896b…f295`。112 roots / 448 onboarding /
  896 pre+postaction / 112 first-seam / 784 owner handoff / 896 writeback、branch、PE、credit 与四类唯一 ID 全部
  闭合；temporal action 为 stay 389 / space 353 / noop 154，742 nonnoop 覆盖 112/112 roots，cold checkpoint
  896/896 不变且 failure reasons 为空。外传双 ID 的 `validate-existing` 通过，产物 bytes/size/mtime 无写入。
  该 PASS 只提升 `forced_common_batch_protocol_freeze_authorized=true`：下一合法动作是冻结共同 forced-collection
  schedule/batch protocol，**不是执行** forced batch、40-decision evaluation 或 Phase 2.5 effect matrix；reader
  rehearsal 继续关闭，formal/unseen/integrated/四轴 effect 继续 NO-GO。
- 2026-08-27 · forced common-batch 设计已冻结为 112 个 root-local 8-credit batch：protocol
  `dd0d28a7…aff93`（raw `0a1cec80…51fe`）固定 112 份 local sequence `0..7` schedule 及父索引
  `13c3f8e2…4e9f`，禁止 global 896-credit batch、跨 root learned policy 或三臂重跑 collection。每 root 的
  full/frozen/strict 必须共享 exact batch/plan 与同一 terminal owner bytes，transition 为 APPLY/WITHHOLD/
  同一 WITHHOLD receipt；full 只允许 theta0+batch+APPLY owner replay。dynamic artifact 只读 PASS manifest，
  natural trace/outcome/credit 读取为零。本包仍未执行 materialization/evaluation，不授权 campaign/effect；按用户
  裁决不生成实验彩排 artifact，提交冻结后才进行首个唯一 112×8 development run。
- 2026-08-27 · 首个唯一 forced common-batch materialization 已由 implementation `5d028d9c…2798` 发布为
  artifact `92880fb7…40d7`，证据提交 `327ee3f4…`。112 roots、112 个 unique 8-credit batch、112 APPLY 与
  112 canonical WITHHOLD 全部 exact join；每个 APPLY 为 8 updates + 1 atomic commit，112/112 参数变化非零，
  最大绝对终态参数 `3.028261 < cap 4.0`，零 cap-hit，full 全部可由 theta0+batch+receipt 重放。symbolic role
  为 448 recommendation/448 noop；actual action 为 stay 218 / space 209 / noop 469，即 427 physical nonnoop，
  覆盖 112/112 roots。双 ID 只读复验及三路审计 GO。终局仅提升 `campaign_protocol_freeze_authorized=true`；
  campaign execution/effect/四轴/reader qualification/formal/unseen/真人/production 继续 false。
- 2026-08-27 · Phase 2.5 三臂 development campaign 已冻结待提交、未执行：protocol
  `bc4c0882…587c`（raw `9374a71b…19fa`）完整绑定 forced artifact `92880fb7…40d7` 与 38 项 direct
  lineage；dynamic owner 新增只暴露 typed action/outcome 的 public selected-branch facade。矩阵固定
  112 roots × full/frozen_theta0/strict_noop × decision 8–47，Learnable=`full−frozen_theta0`、
  Steerable=`frozen_theta0−strict_noop`，三臂 evaluation credit 均不 apply。每拍三臂 preaction/postaction
  分别以 durable group receipt 隔离 source open 与 owner advance；报告使用 20,000 次 whole-root paired
  bootstrap、0.05 practical floor、五段 point-estimate durability 与 0.02 safety upper 门。用户裁决的
  `rehearsal=false` 已写入 protocol。离线 validator 还会从上游 frozen initialization 重放完整 executor
  receipt，并由 typed external evidence 重算 settlement→matching social PE→credit；同 action 跨臂必须复用
  exact source branch。本包只执行 unit/contract 与纯内存统计校验，不运行完整 campaign、模型、CUDA 或
  Gist。当前不存在 outcome/effect/stop-go；提交本冻结包后才允许首个唯一 materialization。
- 2026-08-27 · Phase 2.5 首个唯一 development campaign 已终局并封存：implementation
  `73e9ed7a…64f91`、artifact `05e97726…6300`、证据提交 `41ebea0f…64edf`；336 terminal state、
  36,066 trace row 经外传双 ID 纯读取 exact replay。full/frozen 的 4,480 个 actual action 全同，故
  Learnable 为 `arm_degeneracy_invalid_contrast_no_claim`，参数/checkpoint 变化不得冒充 treatment；
  frozen/strict 有 4,325 个 actual divergence，Steerable 只保留 development GO candidate。campaign 总状态
  `development_campaign_completed_contrast_invalid_no_claim`，不授权 power prereg、formal 或任一 able 效果。
- 2026-08-27 · Learnable 修复裁决：拒绝事后挑早期 checkpoint、移动 threshold/bias 或仅靠小 LR 制造 crossing。
  新 versioned gate operator v2 保留旧 v1 全谱系，删除 free bias/恒定 support，使用四个 centred typed
  forecast feature；forced assignment 只消费 `e2e25718` 冻结的 common-noop-baseline typed credit，并按完整
  fixed-balanced schedule 的 half-centred development feature moment 对称更新，owner recommendation 为 noop
  时发布 zero-information receipt。learned theta0 load 必须携完整 parent/batch/APPLY receipt 重放且净更新非零；
  SHADOW advisory 必须携 policy+forecast 重放 decision。该 score 不是随机化/因果效应/ATE/CATE/Learnable 证明。bootstrap/online
  rate 分别预定 `1/512` 与 `1/4`，但须等 v2
  pulse seam 与 theta0 v3 protocol 各自冻结后才能物化。执行前只读 public/outcome-free treatment-reachability
  admission 是永久 manipulation gate，不是实验 rehearsal；reader/model/CUDA 继续不启动。
- 2026-08-27 · v2 pulse/forced consumer seam 已闭合到 model-free owner-contract 边界：forced authorization
  保存完整 fixed-balanced assignment artifact/entry 与 cold policy，调用方不能提交 concrete action；executor 在
  temporal delivery 前重放 policy+forecast，settlement 只按 actual delivered action 经既有 owner persistence、
  Social PE、parent action-credit 派生 exact common-baseline credit。pulse-owned collected batch 保留完整且有序的
  同 schedule settlement，并由 owner helper 重放 forecast publication、验证相邻 post→input handoff；raw gate-only
  batch 不能进入 transition。APPLY/WITHHOLD 从同一 collection 成对派生并锁定同一 plan/pre/candidate checkpoint，
  strict-noop 只改变 executor bit；外部 base authorization 不能绕过 v2 outer authorization 打开 ACTIVE。
  新类型均为 `.v2`，legacy v1 ID/payload 不变。本包不运行 reader rehearsal、模型、CUDA 或 campaign，也不证明
  treatment reachability/Learnable/Steerable；下一包仍是 theta0 v3，随后才执行永久 public/outcome-free admission。
- 2026-08-27 · 为 theta0 v3 的 112 个独立 root 增加 add-only segmented collection contract：每段显式绑定唯一
  `segment_scope_id`、immutable owner start、段内 forecast scope 与 post→input handoff；所有段 flatten 后仍须从
  sequence 0 开始完整耗尽同一 fixed-balanced schedule。segmented collection 使用独立 transition/matched-pair ID 与
  commit 入口，旧 continuous v2 签名、payload 与 identity 保持不变。generic pulse 不声称 segment start 是空 root；
  theta0 v3 owner 仍须逐 root 证明从 `None` 重放四条 onboarding。当前只闭合 model-free 机制，不运行彩排、模型、
  CUDA 或 campaign。红队发现 owner snapshot payload 可被构造后原地修改；continuous/segmented 现均以 construction-time
  provenance seal + 边界重放 fail closed，有效旧 constructor signature/canonical payload/ID 不变。segmented compact
  payload 只持 `gate_batch_id`，但 896-entry gate batch 本身仍有 O(N²) serialization/content-ID 成本；v3 protocol
  必须事前披露，若不可接受只能另立 gate schema，禁止看见 theta 输出后改。
- 2026-08-27 · 算力/serialization 裁决在任何 theta0 v3 输出前完成：拒绝单一 flat 896-entry v2 schedule/batch，
  选择 add-only gate federation。parent schedule 以连续 global offset 绑定 112 个完整 root-local 8-entry child；
  post-collection parent batch 保留全部 typed child components，并在全局重放 decision/forecast/assignment/exposure/
  credit 唯一性、timestamp、同 cold artifact/checkpoint/policy 与 448/448 balance。更新数学、学习率和逐项 cap 顺序未改，
  小型同序 flat reference 的 candidate checkpoint byte-exact。matched APPLY/WITHHOLD 共享一个 unique parent plan identity、
  pre/candidate checkpoint；APPLY 为一次 atomic parent commit，receipt 报 `child_transition_count=0`，federated terminal
  必须先凝结为 cold learned theta0 才能 evaluation。该字段只证明本 parent 内无 child transition；future owner 仍须证明
  pre-outcome timing，并从 actual APPLY/WITHHOLD receipts 建立 accepted persisted lineage 内
  `child_transition_count=0`，不扩展为 external/OS 路径不存在。当前仅 gate owner + model-free unit contract 闭合；pulse federation
  consumer、theta0 v3、permanent admission、模型/CUDA/campaign/effect 全部待完成，reader/campaign 彩排继续关闭。
- 2026-08-27 · pulse federation consumer 已以 `1e267a61` 提交闭合；theta0 v3 development owner、
  protocol `9c48a8e3…c12b`（raw `c7e2d75f…883f`）、只含 `materialize / validate-existing` 的 CLI 与 model-free
  定向测试随本提交冻结。协议将 source-v4 和未资格化 development reader 明示标为已花费 adaptive
  input，不读旧 theta0-v2 / forced-common outcome-credit，不运行模型或 CUDA。当前仅冻结实现：未产生
  theta0 v3 artifact，未建立 durable/root-reset/no-child-transition execution claim，也未计算任何 effect。
  提交前红队发现任意 40-hex SHA 可冒充 implementation commit，且 child-transition 失败 artifact 会将实测计数
  硬写为 0；本包已将两者 fail closed，并增加 input/output 根隔离、closed-trace 回读、
  448/784/896 owner-state 变化/交接门。这些仍是 implementation safeguards，不是实验结果。
- 2026-08-27 · theta0 v3 commit `9a6dfc04` 的首个 create-only materialization 在首条 persisted preaction 前
  implementation-failed：typed assignment receipt 的正式 ID 是 `assignment_id`（并由 forced exposure 发布为
  `assignment_receipt_id`），workflow 却访问了不存在的 `receipt_id`。partial root 只有 protocol、parent schedule 与
  6-row trace（parent durable、root begin、四次 onboarding），无 persisted preaction/manifest/transition/theta；一个
  public preaction/forecast/forced exposure 已在内存构造，但 environment/sealed outcome/PE/credit 均为零。报告
  `421ad709…e733e` 明确 `scientific_terminal=false / effect_claim_authorized=false`。该根不续跑、不覆盖；本修正只使用
  owner 发布的 typed ID，新 attempt 必须保持同一 protocol/scientific pins 并使用新 create-only 输出根。彩排仍关闭。
  下一合法动作是在修正 implementation commit 上做第二次 attempt；它若完成，才是首个完成态 create-only、model-free
  112×8 materialization，随即用外部
  protocol/artifact ID 纯读取复验；之后才冻结 permanent public/outcome-free treatment-reachability admission。
  彩排继续关闭，admission 仍不得被命名为 rehearsal。
- 2026-08-27 · 上述第二次 attempt 已完成 112×8 materialization，但自动 `validate-existing` 在 ledger row 6
  因 forced receipt 内 `temporal_projection.timestamp_ms` 使用墙钟而 byte drift；该 timestamp 继续传染 child/federated
  collection、transition、theta 与 manifest identity，故拒绝 validator normalization。attempt02 manifest
  `0c596cd5…076c` 不被接受，六文件根本机原样保留，compact failure report `a795fdcf…5cf7`，全部下游 claim=false。
  add-only runtime `publish_at`、TrackTemporal standalone publisher 与 v2 forced pulse logical-time seam 已分别闭合；live
  theta protocol.v2 `f5c33f5c…1d26`（raw `83060179…e38f`）冻结 `temporal=root*20+4+2*decision`、
  `credit=temporal+1`、runtime/temporal/pulse/theta/protocol/CLI 六 blob lineage，以及 materialize/validate 共用 exact
  implementation HEAD + clean scope。下一动作是新 create-only 根的正式
  materialization + external 双 ID validation；不插入彩排，v1 两次失败均不重写。
- 2026-08-27 · theta0 v3 bootstrap attempt03 已在 protocol.v2 与 implementation `af6cc60b…ad74` 上完成
  112 roots / 896 credits 的 create-only materialization，artifact `dde0fc78…9777` 发布 cold learned theta0
  `dded60c6…2f7`。自动 post-materialization validation 因两个 docs/business-only commit 使 main HEAD 前移，
  在 artifact replay 前的 exact-HEAD guard 拒绝；主工作区未回退。随后 detached exact-commit worktree 以外传
  protocol/artifact 双 ID 完整 replay PASS，六文件 size/mtime/SHA fingerprints 不变；compact acceptance receipt
  `4c3c414b…aab5`（raw `cae85ed0…af6c`）。本终局只接受 development mechanism，reader qualification、treatment reachability、
  campaign/effect/formal/unseen/integrated/四轴/真人/production 均未建立，model/CUDA/rehearsal count 仍为 0。
  当时下一步为先闭合 condensed learned-theta0 cold evaluation authorization seam，再冻结独立 permanent
  public/outcome-free treatment-reachability admission。
- 2026-08-27 · condensed learned-theta0 cold evaluation authorization seam 已以 add-only pulse contract
  闭合：full federated wrapper 只在构造/恢复边界用于完整 replay，冻结授权仅保留不可变 lineage projection 与
  `0/0/()` cold policy；source APPLY 与 evaluation transition 分账，后者固定为 `null`。candidate/strict 只差
  executor disposition，旧 continuous v2 payload/identity 保持原样。15 项 pulse v2 model-free 测试全过，未运行
  model/CUDA/rehearsal/campaign；下一动作只剩独立 permanent public/outcome-free treatment-reachability admission。
