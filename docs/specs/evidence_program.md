# 证据计划 Spec

> Status: draft
> Last updated: 2026-04-25
> 对应需求: R12, R15

## 要解决的问题

如何把内部 benchmark / proof harness 升级为可复现、可审阅、可回放的证据生产线，使系统的对外主张能被明确映射到 gate、artifact、盲评和统计结论，而不是靠单次 run 或主观叙述。

## 关键不变量

- 对外主张必须先冻结成可证伪 claim，再绑定 required gates、artifact 和 verdict 规则
- claim verdict 必须回溯到 manifest、seed、git sha、依赖版本和原始 artifact
- dialogue / ETA paper-suite 共享统一的 evidence bundle 口径，不各自发明一套 summary schema
- 盲评外发包不得泄漏 profile label 或内部 case 标识；profile 映射只存在 internal key
- 人评不是替代自动评估，而是额外证据面；自动指标、人评与 claim verdict 必须能并列审阅
- open-environment widening evidence 必须区分 `open_core`、`open_families`、`open_heldout`，不能把单一固定场景误写成开放泛化

## 工程挑战

- 设计 claim registry，把抽象宣传口径压成具体 gate
- 统一 dialogue / ETA aggregate 报告的 pairwise effect、claim verdict 和 evidence bundle 导出
- 让 blind review packet 真正可外发，同时保留内部 unblinding key
- 为人评建立最小协议和可机读 aggregate，而不是只导出一组 transcript
- 让 repeated-run summary 不只给 interval，还能给 matched-control effect 与 retained / weak / fail verdict

## 算法候选

证据计划属于评估与 rollout 审计层，受 R12 / R15 约束：

- evaluation 仍是 PE-first 主链的 readout / gate / widening evidence，不替代 learning primitive
- claim verdict 基于 matched-control comparisons、longitudinal evidence、blind review 与 provenance
- open-environment 作为 widening surface，只能在 held-out 覆盖与统计口径满足时支撑更强 claim

## 接口契约

**消费的输入**：
- dialogue comprehensive / paper-suite aggregate
- ETA proof paper-suite aggregate
- NL essence / ETA acceptance gates
- blind review packet、human rating entries、human rating aggregate
- manifest / provenance / repeated-run summaries / pairwise metric effects

**产出的输出**：
- claim registry / claim verdicts
- external-safe blind review packet
- internal unblinding key
- human rating template / aggregate
- unified evidence bundle
- `snapshot_replay_bundle.json`（Phase 1 runtime export shape started）：导出 existing snapshots 的 action replay section（`PredictionActionContext` / `temporal_abstraction.closed_segments` / `prediction_error` / `credit` summaries）用于再现与证据审阅，不引入 trace runtime schema；完整 paper-suite bundle 挂载仍可后续扩展

当前实现口径：

- `volvence_zero.agent.paper_suite` 提供共享 `ClaimVerdict` 与 `EvidenceBundle`
- `volvence_zero.agent.baseline_manifest` 发布
  `DefaultBehaviorBaselineManifest`，分别冻结 package-facing `BrainConfig`
  默认面与 dialogue `pe-eta` paper baseline。两者不得被描述成同一运行时：
  当前产品默认是 synthetic substrate、无 persistence root、禁止 live substrate
  mutation；paper baseline 使用独立 benchmark schedule、builtin-only runtime，
  且允许实验性 live substrate mutation。manifest 同时冻结
  `FinalRolloutConfig` 的所有 `WiringLevel`、capability wiring digest 与
  `learned-vs-heuristic-coverage.md` 版本，行为面变化会改变 manifest digest。
- retain-level 外发必须先调用
  `validate_evidence_bundle_for_external_use(...)`。该 gate 只阻断至少包含一个
  `status="retain"` claim 的 bundle；`weak` / `fail` / SHADOW artifact 仍可导出供审阅。
  默认校验 clean working tree、有效 git identity、非空依赖清单与 digest、
  manifest hash 一致和 seed schedule；同基底或正式外发路径必须额外要求
  verified substrate fingerprint 与每个原始 artifact 的 sha256/size。
  缺失时抛 `RetainProvenanceError`，禁止静默降级为可外引 retain。
- dialogue / ETA paper-suite aggregate 会额外发布 pairwise effects 与 claim verdicts
- dialogue paper-suite export 会同时导出 blinded packet、internal key、rating template、rating aggregate 与 unified evidence bundle
- dialogue emergence dashboard / paper-suite metric values 发布 `canonical_mean_semantic_spine_coverage`、`canonical_mean_cognitive_loop_readiness` 以及 open-environment 对应读数；这些是 semantic owner 快照的证据读数，不作为学习源头
- dialogue NL essence assessment 发布 `semantic-spine-ready` gate，用于审计核心 semantic owner spine 是否具备完整 coverage 与非零 readiness；该 gate 目前不进入默认 required gate 列表
- dialogue paper-suite manifest 将 `canonical_mean_semantic_spine_coverage` 与 `canonical_mean_cognitive_loop_readiness` 列为 secondary metrics；companion stateful relationship verdict 优先消费 repeated-run summary，reference dashboard 只作为 fallback
- dialogue paper-suite export 可额外导出 `semantic_proposal_quality_shadow.json`，并在 `EvidenceBundle.reference_artifacts` 中登记同一 payload；该 payload 是 non-gating shadow diagnostic，不参与 retain/fail verdict
- ETA paper-suite export 会导出统一 evidence bundle，复用相同的 claim verdict / pairwise effect 口径
- Gate 1 PE mechanism packet 使用 schema `gate1-pe-mechanism.v1`，在实现与
  evidence run 前冻结以下门：
  - gold surface 固定为 `numeric / probability / enum / vector /
    distribution` 五类；numeric/vector 使用
    `0.5 * ||prediction-actual||²`，probability 使用 Bernoulli
    cross-entropy 的 logit link，enum 使用 one-hot categorical
    cross-entropy 的 logit link，distribution 使用 soft-target categorical
    cross-entropy 的 logit link。每类都发布 component decomposition，
    runtime signed PE 固定为 `actual-prediction`，并必须在 `1e-9` 内等于
    对应真 autograd LSS 的负值。不同 link 的梯度量纲禁止聚合成单一训练量。
  - 同一 gold case 重跑必须逐字段相同；非法概率、非归一化分布、enum
    目标越界与 prediction/actual 维度不一致必须 fail loudly。
  - lineage auditor 以 `prediction_id` 为主键，一一连接 prediction、
    outcome 与 prediction-error，并同时核对 `environment_event_id /
    environment_outcome_id / observed_at`。`lineage_coverage == 1.0`、
    accepted mismatch `== 0`、duplicate settlement `== 0` 才通过。
  - `VZ_PE_EVALUATION_DECOUPLED=ACTIVE` 下，仅改变 evaluation 内容时，
    actual outcome、PE 与 PE-derived learning credit 的 canonical JSON
    必须逐字节不变；SHADOW 仅作旧行为回滚对照，不进入通过判据。
  - packet 复用 Gate 2 的 12 文件名：
    `manifest.yaml / predictions.jsonl / outcomes.jsonl /
    prediction_errors.jsonl / segments.jsonl / credit.jsonl /
    state_diff.jsonl / action_selection.jsonl / ablation_results.json /
    promotion_verdict.json / rollback_evidence.json / report.md`。
    本包最多产 `mechanism-supported`，不得产 causal 或 thesis verdict。
  - 2026-07-30 实跑：五类 gold 与四类 fail-loud probe 全通过，真 LSS 最大
    bridge error=`1.3877787807814457e-17`；lineage coverage=`1.0`、
    mismatch/duplicate=`0/0`；evaluation-decoupled 两臂 canonical payload
    SHA256 均为
    `1151657afa05b76671d9196d00e1b50c2d3e64c1bf4297610233865cfd7a9537`。
    verdict=`mechanism-supported`，artifact：
    `artifacts/gate1_pe_mechanism_20260730`。causal 层仍由包 1b 单独判定。
- Gate 1 PE causal packet 使用 schema `gate1-pe-causal.v1`，在实现与
  evidence run 前冻结以下门：
  - matched arms 固定为 dialogue owner 的 `pe-eta`（full）与
    `pe-drive-off`（full-no-pe-drive）；两臂共享同一个 frozen builtin
    substrate runtime、同一 seed、同一 scenario 和相同 turn budget，唯一
    行为差异是 PE 是否驱动 schedule、credit 与 optimizer。
  - primary split 只含现有 dialogue suite 中
    `split == "open_heldout"` 的四个冻结场景：
    `open_repair_heldout / open_clarification_heldout /
    open_failure_loop_heldout / open_goal_shift_heldout`。不得混入
    `open_core / open_families / demo`，也不得按实跑结果删选场景。
  - primary metric 沿用 open dialogue owner 已有的行为 acceptance check
    `late-episode-stabilization-or-improvement`，定义
    `heldout_learning_success_rate` 为四个 held-out case 中该检查的通过率，
    `heldout_learning_gain = rate(pe-eta) - rate(pe-drive-off)`。通用
    `open_pass_rate` 含 `pe-schedule-observed` 机制检查，会与 PE-off 干预
    形成循环，因此只作诊断、禁止进入本 causal gate。seed schedule 冻结为
    `101 / 211 / 307`；最小效应冻结为一个 held-out case，即每个 seed 的 gain
    `>= 0.25`，且三 seed 必须全部同向达到门槛；跨 seed mean gain
    `>= 0.25`。均值、逐 case acceptance checks 与 metric means 同时发布，
    但不替代 primary gate。
  - 单 seed `101` 是预注册 GO/NO-GO 探针，也同时是三 seed 矩阵的第一个
    正式 seed：若 gain `< 0.25`，立即停止，不运行其余 seed，不调参重跑，
    causal verdict=`not-supported`；Gate 1 主张收缩为“PE 是可审计原始
    信号”，保留 `gate1-pe-mechanism.v1` 的 mechanism verdict。
  - 若探针通过，才运行 `211 / 307`；任一 seed 不达门或场景/profile/
    runtime fingerprint 不匹配均判 `not-supported`。三 seed 全通过时最多
    产 `causal-supported`，不得自动升级为纵向或 thesis verdict。
  - packet 复用同一 12 文件名；逐 seed/profile/scenario 的 pass、reasons、
    acceptance checks 与 owner telemetry 落入 jsonl，bundle 必须能从这些
    原始记录重算 verdict。rollback 是保持 `pe-drive-off` 臂可用并将 causal
    claim 收缩，不修改 runtime owner。
  - 2026-07-30 实跑：正式 GO/NO-GO seed `101` 在四个
    `open_heldout` case 上，`pe-eta` 与 `pe-drive-off` 的
    `heldout_learning_success_rate` 均为 `1.0`，primary gain=`0.0`，
    未达到预注册最小效应 `0.25`，因此立即停止且未运行 `211 / 307`。
    作为诊断，通用 `open_pass_rate` 为 `1.0 vs 0.0`，但 PE-off 的失败
    reasons 仅来自 `pe-schedule-observed` 与
    `runtime-backbone-evidence-observed`，不能作为行为学习增益。
    verdict=`not-supported`；Gate 1 保留 mechanism verdict，causal 主张
    收缩为“PE 是可审计原始信号”。artifact：
    `artifacts/gate1_pe_causal_20260730`。
- Gate 4/5/6 共享真 trace 工厂使用 schema
  `gate456-shared-settled-trace.v1`，在实现与生成前冻结以下契约：
  - 本工厂是 `vz-runtime` 对既有 session/public snapshot 的 out-of-turn
    export，不新增 runtime slot 或第二 learning owner。每个正式样本是独立
    两 turn 微会话：第一 turn 走正式 `AgentSessionRunner.run_turn()` 发布
    owner prediction；环境 adapter 随后通过
    `submit_environment_outcome()` 提交携带该 `prediction_id` 的 typed
    outcome；第二 turn 再走正式主链结算 actual、PE、credit 与 temporal
    snapshot。bootstrap turn 不计 transition。
  - substrate 固定为本机缓存的
    `Qwen/Qwen2.5-0.5B-Instruct`、CPU、`strict-local`、frozen、
    fallback denied；任何 seed 出现非 `hf-local` origin、fallback、
    live substrate mutation、空 residual sequence 或 substrate
    fingerprint 漂移均 fail closed。
  - seed schedule 固定为 `401 / 409 / 419`，每 seed 精确生成 `510`
    settled transitions：`trace-train=300`、`trace-heldout-context=150`、
    `trace-locked-confirmation=60`。三分区按 registry 顺序整体生成，
    consumer 只能按 manifest 的完整 partition 加载，禁止逐条挑选。
  - corpus registry 固定为 18 个 user/context block：train 10 个、
    heldout 5 个、locked 3 个；每个 block 内交错
    `old-recall / new-introduce / new-revision / old-retention` 四类 episode，
    并同时发布 knowledge key、context/user id 与可观察 environment
    measurement。locked 分区只允许生成、指纹和契约验收，本包不得根据其
    指标修改 corpus、阈值或实现。
  - append/resume 单位是一个已结算 transition：只有两 turn、typed
    outcome、session-post slow loop 全部完成后，才向
    `transitions.jsonl` 追加一条 canonical record 并刷新 progress；重启
    必须验证已有 prefix 的 transition id、seed、partition 与 digest，
    然后从下一条继续，不重算已完成样本。
  - 每条 transition 必须携带 `session_id / prediction_id /
    prediction_ref /
    environment_event_id / environment_outcome_id / observed_at`，以及
    prediction、actual、PE、credit、temporal segment/action、memory 与
    substrate public readout；lineage join coverage 必须 `1.0`，accepted
    mismatch 与 duplicate settlement 必须 `0`。每 seed settled count
    必须 `510`，三 seed substrate fingerprint 必须相同。
    `PredictionErrorModule` 的 owner-issued `prediction_id` 作用域是单
    session，因此跨微会话 join 的主键固定为
    `prediction_ref = session_id + "::" + prediction_id`；原始
    `prediction_id` 仍逐字保留，禁止改写 owner id 冒充全局 id。
  - turn latency 与 session-post slow-job latency 分开统计；它们只作运行
    诊断，不进入 learning verdict。通过本包只产
    `trace-contract-supported`，不得产 Gate 4/5/6 causal verdict。
  - `transitions.jsonl` 是不可变 source corpus；常规 12 文件 evidence
    packet 由它确定性派生。回滚为停止生成并保留已完成 prefix；任何未达门
    corpus 标记 invalid，Gate 4/5/6 禁止消费。
  - 2026-07-30 实跑：seed `401 / 409 / 419` 各生成 `510` 条、合计
    `1530` 条 settled Qwen trace；aggregate 分区为
    `trace-train=900 / trace-heldout-context=450 /
    trace-locked-confirmation=180`。三 seed lineage coverage 均为
    `1.0`、mismatch/duplicate 均为 `0/0`，runtime origin 均为
    `hf-local`，fallback/empty-residual/substrate-mutation 均为 `0`，
    runtime fingerprint 统一为
    `runtime-descriptor-sha256:6c26ccf9224adf3400b01a4c626d4f53b54de4f2b588ecfc1fad48b07e2c1966`。
    aggregate verdict=`trace-contract-supported`、consumer
    admission=`allowed`；本包没有产 Gate 4/5/6 causal verdict。artifact：
    `artifacts/gate456_shared_settled_trace_20260730`。
- Gate 5 CMS Pareto 使用 schema `gate5-cms-pareto.v1`，在实现与首次读取
  `trace-locked-confirmation` 指标前冻结以下协议：
  - consumer 必须按 seed `401 / 409 / 419` 分别顺序加载完整
    `trace-train=300`、`trace-heldout-context=150`、
    `trace-locked-confirmation=60`，不得重排、抽样或按结果删行。每条 replay
    observation 的学习向量只能取该 transition 的 public
    `memory_snapshot.attribute_summary` 中最新
    `substrate_feature_digest`；PE 只能从同条 public
    `prediction_error` 重建 typed snapshot。两者都必须保留
    `transition_id / prediction_ref / record_sha256` lineage。该 digest 是
    memory owner 已发布的 substrate-derived readout；harness 不遍历原始
    runtime 私有状态，也不从自然语言重造 substrate。
  - 五臂固定为：`nested-CMS(full)`（nested、cadence `1/2/4`、ATLAS
    `8/4/2`、Titans PE features on）；`single-timescale`（同三 band MLP
    参数预算，independent、cadence `1/1/1`，其余 uplift 保持）；
    `no-ATLAS-replay`（仅将 replay 关闭）；`no-PE-write-gate`（仅将 CMS
    Titans PE features 关闭；MemoryStore 的正式 artifact PE admission
    gate 不变）；`memory-only`（同一 MemoryStore artifact/retrieval owner，
    learned CMS core 关闭）。所有臂消费同一顺序、同一 typed PE 与同一
    knowledge write/retrieval protocol。
  - primary 只在 locked 60 条上计算，前 450 条只用于顺序适应。每条
    `new-introduce / new-revision` 读取 owner 发布的
    `cms_new_knowledge_absorption`，每条 `old-retention` 读取
    `cms_old_knowledge_retention`；arm/seed 指标是对应样本算术均值。
    `memory-only` 没有 learned core，按公共 snapshot 默认语义记为
    absorption `0.0`、retention `1.0`。这两个指标沿用 #89 Stage 0
    per-update band-drift proxy，只能支持 CMS 更新/抗漂移 claim，不能冒充
    deployment-time recall。
  - full 对每个 control 的 Pareto 门为：三 seed aggregate 的 absorption
    与 retention 都不低于 control 超过 `0.01`（即
    `full - control >= -0.01`），并且每个 seed 都满足相同容差。相对
    `single-timescale`，至少一个维度的三 seed aggregate 增益必须
    `>= 0.02`，且该维逐 seed 增益全部 `> 0`。`0.01 / 0.02` 在 locked
    首次读取前冻结，不因结果调整。
  - 同 bundle 报告 diagnostic（不替代 primary）：owner-published
    `memory_updater_touched_param_ratio` 的均值作为 memory churn、
    negative `new-revision` 后进入 durable stratum 的 trace knowledge
    entry 比率作为错误晋升率、按正式 `RetrievalQuery` 对
    `knowledge_key` 的 top-k 命中率、命中时的跨 session
    `action_payoff`、以及 trace 中
    `session_post_slow_job_ms` 与 turn latency 的分位数。cadence mechanism
    必须断言 full 的 public intervals 为 `1/2/4`、single 为 `1/1/1`，
    每臂 frozen substrate mutation count=`0`，background work 不被合并
    进 turn latency。
  - 单 seed `401` 的 train+heldout replay 可作代码/方向探针，但不得读取
    locked；探针失败只能修复契约、lineage 或实现错误，不能改五臂、阈值或
    metric。正式 run 首次且仅一次消费三 seed locked，产强制 12 文件
    bundle。任一 lineage/partition/cadence/frozen-substrate 门失败则
    verdict=`invalid`；机制门全绿但 Pareto 或最小效应失败则
    verdict=`not-supported`，主张收缩为“多频 CMS 可运行、可审计且可
    回滚”，禁止调参重跑同一 locked 分区。全部通过最多得到
    `causal-supported`；回滚为恢复 `memory-only` 或显式关闭
    `cms_pe_features_enabled / cms_replay_window_size`，不修改 frozen
    substrate。
  - 2026-07-30 正式 run 已完成，locked 分区按协议仅消费一次。五臂 × 三
    seed 共 replay `7650` arm-transitions；全部 lineage、`1/2/4` 与
    `1/1/1` cadence、matched CMS 参数预算、frozen substrate mutation=0、
    latency 分离与 checkpoint rollback 门通过。full 对全部四个 control
    均满足 `0.01` Pareto 不劣；但相对 `single-timescale` 的 absorption /
    retention 增益仅为 `+0.000000251 / +0.000001173`，虽三 seed 同向，
    仍远低于冻结最小效应 `0.02`。因此
    verdict=`not-supported`，不调参、不重跑 locked；Gate 5 causal
    主张收缩为“多频 CMS 可运行、可审计且可回滚”。diagnostic 中五臂
    retrieval hit 均为 `1.0`、错误晋升率均为 `0.0`，不能补救 primary
    effect 缺失。artifact：`artifacts/gate5_cms_pareto_20260730`。
- ETA proof suite 当前还区分 `eta-internal-rl-proof` 与 `eta-open-weight-residual-proof` 两类 manifest；真实 residual-control claim 必须绑定 `transformers-open-weight` capture / actual hook fire rate / fallback rate / prefix-aligned intervention 证据，不能由 trace 或 synthetic backend 单独支撑。当前 claim gate 要求 fallback rate 为 `0.0`、actual hook fire rate 至少 `0.75`、residual sequence 非空、intervention protocol valid；显式 fallback smoke run 必须保持 fail/quarantine 语义。`planned_layer_fraction` 只说明选了多少层，不作为 hook 健康硬门槛
- ETA Gate 2 residual causal packet 使用
  `volvence_zero.agent.eta_gate2_residual_evidence` 冻结
  `identity / zero / shuffled / reversed` 四臂。四臂必须共享 policy、
  optimizer 初态、场景、seed、prefix 与 frozen substrate；identity 只训练
  一次，三个 control 从同一训练后 checkpoint 恢复且不得继续更新。
  transition 同时记录
  `control_before_ablation` 和实际 `applied_control`。bundle 必须产出
	  `manifest.yaml / predictions.jsonl / outcomes.jsonl /
	  prediction_errors.jsonl / segments.jsonl / credit.jsonl /
	  state_diff.jsonl / action_selection.jsonl / ablation_results.json /
	  promotion_verdict.json / rollback_evidence.json / report.md`，缺任一文件
	  或 episode 缺逐步
	  intervention record 时 fail loudly。trace smoke 最多只能得到
	  `wiring-ready`；真 open-weight 且 hook/fallback/prefix/变换门全通过时最多
	  得到 `mechanism-supported`。v29 按 `train / eval /
  development-heldout / confirmation` 分区发布 continuation NLL 与有效
  score count；eval 必须方向为正，且从未参与设计或停止决策的 locked
  confirmation split 相对最佳 matched control 达到预注册 `0.02`，才允许
  `causal-supported`。manifest 必须显式携带
  `confirmation_split_locked=true`，只有 split 名称而没有锁定声明仍应
  fail。反复观察过的 heldout 不得继续充当确认集。
- 2026-07-28 Qwen2.5-0.5B CPU 单 seed 实测：
  control-summary 与 baseline capture 统一取末 token 后，zero-control
  downstream effect 严格为 0，机制门通过；proof completion 因 observation
  cosine 主导已从 causal outcome 移除。只读 observed continuation NLL
  驱动的 counterfactual PE 训练保持 SSL 结构冻结、仅更新 causal policy。
  v19 的 90 个 train prefix 上，22-action oracle 平均可达改善
  `0.032274`、正改善率 `97.78%`，最佳固定候选也为 `+0.005697`。
  counterfactual optimizer 改为逐状态 direct best-action 后，v20 的 8-update
  候选在 train 成为最佳臂，development-heldout 为 `+0.005979`，但 eval
  仍为 `-0.016155`。v22 的只读分区扫描显示 eval oracle 为 `+0.021150`、
  正改善率 `83.33%`，而最佳固定候选为 zero；因此候选空间可达，但需要
  context-conditioned selection。48 维 signed state（v21）和 24 维 hybrid
  state（v23）都使未见分区退化，默认回滚到 12 维 v22 候选。v25 保持
  12→3、direct best-action、8 updates 和原 eval/development-heldout 不变，
  仅把无标签训练分布从 8 条 route / 45 个 prefix 扩到 16 条 route / 96 个
  prefix；新增 8 条 route 的内容词与非训练分区内容词不相交，confirmation
  仍关闭。真实运行产生 `768` 条 optimizer transition、`16896` 次候选评分和
  `2640` 个去重 counterfactual score。扩大训练分布后 eval 从
  `-0.016155` 退化到 `-0.017924`，development-heldout 从 `+0.005979`
  退化到 `-0.001057`，train identity 也不再胜过最佳 control；与此同时
  eval oracle 升至 `+0.022156`，development-heldout oracle 升至
	  `+0.035008`。v26 用 train-only PCA16 + ridge 建立 prefix residual 到 22
	  个动作的 selector；v27 只用 route-grouped train CV 在
	  `PCA={8,16,32,64} × ridge={0.1,1,10}` 中选型。两者在 eval 的 selected
	  delta 都为 `-0.013828`，没有达到 shadow injection 门。v28 去掉 hash/PCA，
	  直接消费 substrate 发布的全部坐标，却发现输入仅 `84` 维：Qwen 每层
	  `896` 维 hidden 在 publisher 内被默认压到 `8` 维。v29 因此把 evidence
	  runtime 的显式 `activation_width` 设为 `896`；生产默认仍为 `8`，真实导出
	  会核对 manifest 与 provenance，宽度不一致 fail loudly。三层
	  mean/latest/trend 最终给 selector `8076` 维输入，仍只得到 train route-CV
	  `-0.002488`、eval `-0.019454`、development-heldout `-0.001422`；
	  top-3 命中率分别为 `0.135417 / 0.166667 / 0.142857`，接近 22 动作下
	  的随机基线 `0.136364`。这排除了“只因手工 state、PCA/hash 或 8 维捕获
	  丢信息”作为充分根因。

	  v30 把这一时序假设做成可证伪实验：每个 prefix 从冻结 Qwen 的零控制
	  分布按 SHA256(case/prefix/cohort/index) 固定种子采样多条 continuation，
	  target cohort 只负责拟合动作价值，完全独立的 audit cohort 只负责
	  train route-CV 模型选择与只读验证；fresh validation 在运行前冻结。
	  transformers substrate 对同一 prefix/control 的 cohort 做一次 batch
	  forward，并由等价测试保证与逐条 NLL 一致。2026-07-28 的 full-width
	  `896`、CPU、单 seed、2 target + 2 audit 校准生成 `576` 条 continuation，
	  selector 在 train/fresh-validation/eval target 上的 selected delta 为
	  `-0.001990 / -0.003317 / -0.005052`，对应独立 audit 为
	  `+0.001393 / -0.001125 / -0.000569`；development-heldout audit 只有
	  `+0.000077`，近于零。fresh validation audit 为负，top-3
	  `0.136364` 等于 22 动作随机基线；尽管 validation target/audit oracle
	  分别为 `+0.011997 / +0.014453`，selector 仍未学到可泛化映射。
	  因此 verdict 保持 `mechanism-supported`，
	  `selector_ready_for_shadow_injection=false`。2+2 只属于校准/反证，
	  不替代 manifest 默认 4+4、跨 seed 或 locked confirmation；当前不再投入
	  4+4 来调同一个 self-distribution NLL 目标，下一包转向真实 downstream
	  outcome / environment PE。
	  当前最新 falsification artifact 为
	  `artifacts/eta_gate2_residual_causal_v30_prefix_expected_2x2_calibration_fullwidth896_qwen25_05b_cpu_1seed_20260728`；
	  v24 仍是当前最佳 12 维候选基线：
  `artifacts/eta_gate2_residual_causal_v24_canonical_state12_direct_8updates_train8_qwen25_05b_cpu_1seed_20260728`。

	  v31 不再把 self-distribution NLL 当作 selector target。对同一冻结 prefix
	  的 22 个候选逐一执行真实 residual forward；environment owner 用干预后
	  residual snapshot 对预注册 proof subgoal target 计算
	  `EnvironmentMeasurement(task_progress, action_payoff)`，且 payoff 明确排除
	  requested control。zero-control 的实际 measurement 作为冻结事前预测，
	  候选 measurement 依次经过 Prediction Error owner 和 credit owner，最终
	  `pe:action` signed credit 才作为 direct action-value target。真实 prefix
	  calibration 在该模式下禁用；下一 prefix 的未干预 residual trajectory
	  另作 audit，只用于 train-route CV 选型与冻结验证。builtin 数值探针在
	  1 train route 上得到 88 条候选记录、primary/audit 各 85 个不同 credit；
	  primary `[-0.049736, +0.025664]`、audit
	  `[-0.045159, +0.033482]`，primary/audit target 重合率为 `0.0`。这只证明
	  目标链可区分、未塌缩，不等于真实 Qwen selector 已泛化。

	  v31 bundle schema 为 `eta-gate2-residual-causal.v31`，新增
	  `counterfactual_outcomes.jsonl`。每条记录必须带
	  observation/segment/candidate lineage、实际 residual forward、primary 与
	  audit outcome、PE magnitude、action credit；缺任一链路时
	  `environment_outcome_reaches_pe_credit` 必须失败。v30
	  `sampled-prefix-expected-value` 仍可显式回滚复现，但默认 manifest 使用
	  `environment-outcome-pe-credit`，并要求
	  `self_nll_excluded_from_selector_target=true`。

	  v32（schema `eta-gate2-residual-causal.v32`）在 bundle 契约中新增
	  signal gates（oracle-vs-permutation-null 门）。动机：2026-07-29 对
	  v30/v31 落盘数据的复检证明，v19-v30 的 continuation-NLL oracle 正值
	  （`+0.012~+0.020`）全部低于 22 候选纯噪声取 max 的期望上限
	  （单点测量噪声 `σ≈0.011` → `E[max22]≈+0.021~+0.024`），且同一动作在
	  target 与独立 audit 续写上的效果相关性仅 ~0.1——此前「oracle 为正即
	  可达解存在」的解读是 max-of-noise 选择偏差。v32 的机器化防线：对每个
	  prefix 取 target cohort 的 argmax 候选，在独立 audit cohort 上复测，
	  其 audit action credit 必须超过 audit 候选均值（可交换零假设下的
	  期望）至少 measurement floor，train 与 validation 两个 split 都要
	  通过；`ablation_results.json` / `promotion_verdict.json` 新增
	  `signal_gates` / `reachable_solution_evidence` /
	  `oracle_permutation_null_by_split`。`reachable_solution_evidence=false`
	  时 verdict 记「无可达解证据」，即使 mechanism/causal gates 全绿也
	  拒绝 causal 晋升，失败的 signal gate 进 kill_conditions。契约测试：
	  `test_eta_gate2_signal_gates_reject_max_of_noise_oracle` 与
	  `test_eta_gate2_reachable_solution_evidence_gates_causal_promotion`
	  （`packages/vz-runtime/tests/test_eta_residual_causal_controls.py`）。

	  2026-07-29 的 Qwen2.5-0.5B、full-width 896、CPU、单 seed v31 完整
	  outcome run 产出 `3168` 条候选记录；MPS 在当前 Torch 2.10 runtime
	  `is_built=true` 但 `is_available=false`，因此 provenance 明确记录 CPU，
	  不把 CPU 结果冒充 MPS。v32 对该批不可变原始记录做只读 gate replay，
	  benchmark 与 `counterfactual_outcomes.jsonl` 的 SHA256 均与 v31 源包
	  一致，未重跑 forward、未修改 measurement / PE / credit。oracle 在独立
	  audit 上相对 permutation null 的 excess 为 train `+0.000249169`、
	  eval `+0.000488147`、heldout `+0.000277891`、validation
	  `+0.000064799`，train/validation signal gates 均通过，故
	  `reachable_solution_evidence=true`：动作空间中的可迁移信号不是单纯
	  max-of-noise。

	  当前 selector 仍未完成：train / eval / heldout / validation 的 audit
	  selected credit 分别为 `+0.000044220 / -0.000131504 /
	  +0.000138137 / +0.000015486`。旧门只检查 train+validation，会错误给出
	  injection allowed；v32 严格门要求四个冻结分区 audit 全部为正，因此
	  `selector_ready_for_shadow_injection=false`、
	  `selector_injection_allowed=false`。confirmation 仍未锁定，最终 verdict
	  为 `mechanism-supported`，不是 Gate 2 完成。权威重判 artifact：
	  `artifacts/eta_gate2_residual_causal_v32_environment_outcome_strict_replay_fullwidth896_qwen25_05b_cpu_1seed_20260729`。

	  v33（schema `eta-gate2-residual-causal.v33`）把环境结局从
	  residual-signature 对齐迁到 realized-continuation NLL。动机（2026-07-29
	  去天花板诊断）：v31 的 `measure_residual_outcome` 读出链路有三重衰减
	  ——(a) `summarize_residual_activations` 对 896 维激活取逐层均值，而
	  control basis 是近零均值的单位范数正弦行，均值读出与执行器可写
	  子空间近正交；(b) 摘要三维中 max/spread 两维被 `clamp_unit` 饱和在
	  1.0，完全失灵；(c) `(cos+1)/2` 映射再压掉一半动态范围（实测
	  cos=0.905→progress=0.9525，与 v31 落盘值逐字节一致）。三者叠加使
	  最强候选 `(1,1,1)×0.7` 也只能移动摘要签名 ~1.3e-4，oracle 物理全距
	  被封在 3e-4 以下——不是「环境太容易」，是「读出对干预近盲」。任何
	  selector 都不可能通过这条通道展示 `0.02` 量级的因果效应。
	  真 Qwen CPU 探针证明行为级读出没有此问题：对 22 个候选控制，
	  realized next segment 的 teacher-forced 确定性 NLL 效应全距为
	  `0.048~0.137`（4 前缀 × 2 route），测量零采样噪声（同输入同输出），
	  max-of-noise 机制在测量层不成立；跨前缀候选排序 Spearman 中位数仅
	  ~0.14，说明「哪个动作好」依赖前缀，正是 selector 要学的条件结构。
	  v33 契约：primary outcome = 候选控制下 realized next segment（route
	  真实到达的下一段文本）的逐 token teacher-forced NLL 相对冻结
	  zero-control forward 的带符号改进，由环境 owner
	  `measure_realized_continuation_outcome` 折算成 [-1,1] 的
	  `EnvironmentMeasurement`（NLL 单位、斜率 1，`0.02` 阈值可直接比较）；
	  audit outcome = 同一控制在下一 prefix 上对 subsequent realized
	  segment 的同款测量（跨段迁移审计，非重复测量）。由于 audit 需要
	  `i+2` 前缀，grid 只发射 primary/audit 都已实现的行，run 入口新增
	  `--max-prefix-steps`（v33 运行用 8）补回行数。manifest 预注册项
	  `counterfactual_outcome_chain / counterfactual_primary_target /
	  counterfactual_audit_surface` 同步改写；PE→credit 链路、v32 signal
	  gates（oracle-vs-permutation-null）与四分区严格注入门全部保留且
	  作用于新单位。

	  2026-07-29 v33 完整 run（Qwen2.5-0.5B、full-width 896、CPU、单
	  seed、77 min）：134 前缀 × 22 候选（2948 行），效应全距 median
	  train/eval/heldout/validation = `0.071/0.079/0.074/0.084`，134/134
	  前缀 ≥ 预注册 `0.02`——较 v31 `~7e-4` 增长约 100 倍，去天花板
	  完成，预注册「效应停留 1e-3 则收缩」的 kill condition 不触发。
	  oracle target 均值 `+0.023~+0.046` 全为正，但 v32 门在 validation
	  失败：train transfer excess `+0.0025` 通过，validation `-0.0024`
	  （eval `-0.0025`、heldout `-0.0065` 诊断同为负），
	  `reachable_solution_evidence=false`，verdict `mechanism-supported`，
	  `validation_oracle_transfer_exceeds_permutation_null` 进
	  kill_conditions。因测量确定性，该失败不是 v30 式测量噪声而是
	  上下文特异性：动作效应前缀内真实且大，但同一动作收益不跨
	  realized segment 迁移。结论：0.5B 上 3 维 `z_t` 残差注入的
	  行为级因果功率已证明（每前缀全距 ≥ `0.02`），可复用（跨段迁移）
	  动作价值未证明；selector 注入维持禁止。权威 artifact：
	  `artifacts/eta_gate2_residual_causal_v33_realized_continuation_fullwidth896_qwen25_05b_cpu_1seed_20260729`。

	  v34（schema `eta-gate2-residual-causal.v34`）把控制 basis 从固定
	  正弦换成从冻结基底自身学到的 train-transition PCA basis。动机
	  （2026-07-29 v33 方差分解诊断）：v33 的迁移失败不是测量噪声，
	  而是执行器方向任意——`_build_control_basis` 的三行单位范数正弦
	  与行为意义子空间无关，等价于「随机转向」，局部效应真实但跨
	  上下文不组成一致的动作价值（global action-main train→validation
	  R² `-0.0103`、within-route leave-one-prefix-out R² 中位 `-0.3063`、
	  相邻段 Spearman `-0.039`）。修根因 = 换执行器坐标系，不是调
	  selector。新 basis 由 substrate owner
	  `volvence_zero.substrate.control_basis.fit_transition_control_basis`
	  离线拟合：状态坐标取 hook 层（20/21/22）最后 token 隐态的逐层
	  均值（896 维，capture full-width 精确值），样本为 train-split
	  路线相邻前缀的转移增量 `h_{i+1}-h_i`；row0 = 归一化均值转移方向
	  （「route 前进」的平均方向），row1-2 = 居中增量的前两主成分并对
	  row0 正交化。实现纯 Python + 固定种子幂迭代（Gram trick），逐位
	  确定，`control_basis_fingerprint` 进 provenance。运行时通过
	  `TransformersOpenWeightResidualRuntime.install_control_basis`
	  安装（仅旋转有界 `applied_control` 的可写方向，不动模型权重、
	  control-scale clamp 或 capture 语义；行重归一化保证与正弦默认
	  可比）。冻结分区纪律：basis 只从 train split 拟合，由
	  `_install_learned_control_basis` 强制；manifest 预注册
	  `control_basis_mode=train-transition-pca-v1 /
	  control_basis_fit_split=train / control_basis_rank=3 /
	  control_basis_state_coordinate=hook-layer-mean-last-token-hidden`，
	  运行时把 fingerprint、转移样本数与 fit 路线写入
	  runtime_descriptor。v33 的 realized-continuation 测量、PE→credit
	  链路、v32 signal gates 与四分区严格注入门全部原样保留。
	  GO/NO-GO 探针（scripts/probe_learned_control_basis.py，真 Qwen
	  CPU，47 上下文 × 22 候选，成对比较，validation 未触碰）：learned
	  basis 下 train→dev 候选平均信用 Spearman 从 `0.257` 升到
	  `0.597`，dev 跨前缀两两 Spearman 从 `-0.011` 升到 `+0.098`，
	  同路线相邻段 Spearman 从 `-0.020` 升到 `+0.100`，每前缀效应
	  全距中位 `0.075→0.091`（因果功率未缩水）——首次出现跨上下文
	  可复用结构，触发全量 run。

	  2026-07-29 v34 完整 run（Qwen2.5-0.5B、full-width 896、CPU、单
	  seed、`--max-prefix-steps 8`，14 min）产出同一 learned basis
	  fingerprint `326aecddc8d0b7e8...`，fit 样本为 train split 的
	  `106` 个 transition delta。134/134 前缀效应全距仍 ≥ `0.02`
	  （median train/eval/heldout/validation = `0.074/0.098/0.070/0.095`），
	  因果功率没有被 learned basis 压缩。严格 selector 门首次四分区全绿：
	  train-only 拟合 selector 的独立 audit selected credit 为 train
	  `+0.0062`、eval `+0.0135`、heldout `+0.0072`、validation
	  `+0.0173`，`selector_ready_for_shadow_injection=true` 且
	  `selector_injection_allowed=true`。这只是 SHADOW 级许可；manifest
	  仍预注册 `counterfactual_action_selector_live_injection=disabled`，
	  线上 substrate injection 未开启。

	  同一 run 的 oracle-vs-permutation-null 门仍拒绝 promotion：
	  train excess `+0.0056` 与 heldout `+0.0022` 为正，但 eval
	  `-0.0081`、validation `-0.0049` 为负，故
	  `validation_oracle_transfer_exceeds_permutation_null=false`、
	  `reachable_solution_evidence=false`、verdict 仍为
	  `mechanism-supported`、`promotion_allowed=false`。解释上，v34 证明
	  可迁移动作价值开始出现在条件化 selector 载体上，而不是动作边际
	  oracle 上；但预注册 v32 oracle 门未通过，不能回溯改门或宣称 Gate 2
	  完成。v35 若要把 selector-vs-permutation-null 作为正式门，必须
	  在运行前预注册并使用 fresh validation/locked confirmation routes；
	  v33/v34 已观察过的 validation routes 不得作为下一次一次性检验。

	  v35（schema `eta-gate2-residual-causal.v35`）把 reachable-solution
	  证据从「边际 action oracle 是否跨 prefix 迁移」改成预注册的
	  `selector-vs-permutation-null-v1`：train-only 拟合 selector 在每个
	  frozen split 的 prefix 上选出的 candidate，其 independent audit
	  credit 必须超过同一 prefix 全候选 audit credit 的可交换零假设均值。
	  旧 oracle-vs-null 保留为诊断，但不再作为正式 `signal_gates`，因为
	  v34 证明可复用价值的载体是 `Q(context, action)` selector 而非边际
	  action。为避免 v33/v34 validation 污染，v35 manifest 切到
	  `case_corpus=eta-gate2-selector-fresh-v4`：保留 16 条 train 与原
	  eval/heldout development routes，替换 4 条 fresh validation routes，
	  并新增 4 条 locked confirmation routes；manifest 同时记录
	  `superseded_validation_route_ids`，且声明
	  `confirmation_split_locked=true`、`validation_frozen_before_run=true`。
	  v34 learned control basis、realized-continuation NLL outcome、PE→credit
	  链路与 live injection disabled 均保持不变。promotion 条件变为：
	  mechanism gates 全绿、fresh confirmation causal gates 全绿、selector
	  gates 全绿，且 train/validation/confirmation 的
	  `selected_excess_over_null_mean >= 1e-6`。三分区还必须逐条验证
	  selector lineage：每条 selection 都能定位同一
	  `(split, route, prefix)` 的候选格点，`selected_action_index` 存在于
	  该格点，且 `audit_selected_raw_delta` 与该候选发布的
	  `audit_action_credit` 数值一致；缺格点、缺候选或数值不一致均使
	  对应 `*_selector_lineage_complete` signal gate 失败，禁止用部分可配对
	  selection 的正 excess 晋升。完整预注册计划（含通过/失败后的目标与
	  kill conditions）见
	  `.cursor/plans/eta-gate2-v35-selector-null_e4a91f27.plan.md`。

	  2026-07-29 v35 完整 run（同 v33/v34 配置，36 min）：预注册
	  selector-vs-permutation-null 门三个正式分区全部通过——selected
	  excess over null 为 train `+0.0020`、fresh validation `+0.0146`、
	  locked confirmation `+0.0061`（诊断分区 eval `+0.0240`、heldout
	  `+0.0042` 同为正）。因果功率不变量保持：160/160 前缀效应全距 ≥
	  `0.02`（median train/eval/heldout/validation/confirmation =
	  `0.074/0.098/0.070/0.076/0.081`），basis fingerprint 与 v34 逐位
	  一致（`train-transition-pca-v1:326aecdd…`）。oracle 边际诊断依旧
	  混合（validation `+0.0107` 正、eval `-0.0081` / confirmation
	  `-0.0055` 负），继续支持「迁移载体是条件化 selector」的解读。
	  machine verdict 首次为 `causal-supported`、
	  `promotion_allowed=true`、`reachable_solution_evidence=true`。
	  边界：单 seed、CPU、ci-smoke 短前缀、合成 hierarchical 语料；
	  `selector_injection_allowed=true` 仅为 SHADOW 级许可，
	  `counterfactual_action_selector_live_injection=disabled` 不变；
	  longitudinal 层与 Gate 2 EXIT 其余条款仍未闭合。artifact：
	  `artifacts/eta_gate2_residual_causal_v35_selector_null_fresh_fullwidth896_qwen25_05b_cpu_1seed_20260729`。

	  v36（schema `eta-gate2-residual-causal.v36`）预注册
	  `shadow-closed-loop-v1`，只回答 v35 条件化动作价值在 evidence harness
	  内是否可逐步组合，不接入真实 session，也不改变 v35 causal verdict。
	  每个 seed 的 train-only kernel selector 在拟合后通过
	  `residual-action-selector.v1` JSON round-trip 冻结；artifact 与
	  `train-transition-pca-v1` basis fingerprint、run id、seed 和 fit split
	  一并写入 `selector_artifact.json`。闭环 arm 在每个 scoreable prefix
	  复用 `residual_action_state_vector` 选 top-1，将 decoded control 加入
	  committed set，并以 committed controls 的逐维和作为后续 prefix 的
	  aggregate applied control；zero arm 始终为零，permutation-null arm 由
	  `(run_seed, route, step)` 的 SHA256 固定轮换候选。三臂均只做
	  teacher-forced realized-continuation NLL 读出，不写 policy/memory，
	  运行前后 runtime descriptor、basis 与 selector fingerprint 必须不变。
	  每步证据写 `shadow_closed_loop.jsonl`，包含 run/seed/split/route/step、
	  arm、selected index/predicted value、step/aggregate control、realized
	  delta、committed count、state features fingerprint 与三项 provenance。

	  v36 使用 4 条新 fresh validation 与 4 条 locked confirmation routes；
	  v35 validation/confirmation 均降级为 superseded development diagnostics。
	  正式 run 为 3 seeds，ci-smoke prefix 长度保持不变。独立字段
	  `shadow_observation_passed` 仅在下列门全部满足时为 true：
	  train/validation/confirmation 的 selector-minus-zero 轨迹总改进均
	  `>=1e-6` 且每分区 3/3 seed 为正；selector-minus-permutation-null
	  同样均 `>=1e-6`；三分区逐步 selected realized delta 均值与 v35 open-loop
	  audit 同为正；provenance 不变且三臂记录完整。失败只冻结后续 runtime
	  SHADOW/live wiring，不回溯 v35 `promotion_allowed=true`。完整预注册与
	  kill conditions 见
	  `.cursor/plans/eta-gate2-v36-shadow-injection_b82d47c9.plan.md`。

	  2026-07-29 v36 单 seed、全正式分区 GO/NO-GO 探针已完成（Qwen2.5-0.5B、
	  CPU、activation width 896、max prefix 8）。三臂共 `483` 条记录，所有
	  route/step 均完整且 `side_effect_free=true`；selector fingerprint
	  `ef360e0e…` 与 learned basis fingerprint `326aecdd…` 在全包内唯一。
	  train 的 selector-minus-zero / selector-minus-permutation 为
	  `+0.126917 / +0.058562`，locked confirmation 为
	  `+0.060983 / +0.054931`；fresh validation 虽相对 zero 为
	  `+0.104359`，但相对 permutation-null 为 `-0.040979`，因此预注册
	  `validation_selector_beats_permutation_null` 失败，
	  `shadow_observation_passed=false`。按算力止损协议不启动三 seed 全量，
	  不把单 seed provenance 缺口混写成算法失败，也不调低阈值。v35
	  `promotion_allowed=true` 继续继承；runtime SHADOW/live wiring 仍冻结。
	  artifact：
	  `artifacts/eta_gate2_residual_causal_v36_shadow_fullwidth896_qwen25_05b_cpu_1seed_probe_20260729`。
- NL slow-loop 支持 ETA fast path 的 claim 需要读取 memory / credit / family payoff / long-horizon coverage 等 runtime evidence，不能只用“有 slow loop job 完成”作为结论
- Phase 2/3 SHADOW candidate smoke 现在有独立 artifact schema：`phase2_shadow_evidence_smoke.json`，`schema_version="phase2-shadow-evidence-smoke.v1"`。该 artifact 由 `scripts/run_phase2_shadow_evidence_smoke.py` 生成，覆盖 SYS-1 / COG-1 / COG-2 / COG-3 单项 profile 与可选 Phase 3 组合 profile；它是 SHADOW review artifact，不是 retain/fail claim verdict 的替代。
- Phase 2/3 multi-seed evidence 现在有独立 artifact schema：`phase2_shadow_evidence_multiseed.json`，`schema_version="phase2-shadow-evidence-multiseed.v1"`；阶段 D decision report schema 为 `phase2_shadow_decision_report.json`，`schema_version="phase2-shadow-decision-report.v1"`。二者仍是 SHADOW/decision-support artifact，不直接替代完整 paper-suite claim verdict。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 评估体系 | claim verdict 消费 evaluation / benchmark evidence |
| 依赖 | 契约式运行时 | provenance 与 artifact 必须回溯到真实 runtime 产物 |
| 依赖 | 多时间尺度学习 / 时间抽象 | claim registry 需要把这些设计命题绑定到可观测 gate |
| 协作 | 调试体系 | blind review / paper-suite 工件是 widening 与审计面 |
| 协作 | Emergent Action Abstraction | `snapshot_replay_bundle.json` 是该能力域 Phase 1 落地的 artifact，进入 unified evidence bundle |
| 被依赖 | rollout / 外部汇报 | evidence bundle 是对外结论、候选比较与审稿材料的统一入口 |

## 初始 Claim Registry

- `claim_pe_multi_timescale_default`
  - 命题：`PE-first + multi-timescale` 是默认路径上的机制事实
  - 需要：`pe-first`、`multi-timescale-default`、`judge-gated-evolution`、`cross-session-growth`
- `claim_temporal_advantage_over_controls`
  - 命题：时间抽象与在线适应在 matched controls 前有稳定优势
  - 需要：dialogue / ETA pairwise effects 为正，且 gap 不只是单次最好结果
- `claim_beyond_scripted_canonical`
  - 命题：优势不只存在于 canonical scripted cases
  - 需要：perturbation / systematic replay / open-environment / held-out families 共同给证据
- `claim_external_human_legibility`
  - 命题：优势能被外部人类评审感知
  - 需要：blinded packet、多评审员评分、inter-rater agreement 与自动指标相关性
- `claim_eta_real_open_weight_residual_control`
  - 命题：ETA residual-control evidence 来自真实 open-weight runtime，而非 synthetic proof harness
  - 需要：`transformers-open-weight` primary backend、低 fallback rate、actual hook fire rate、prefix-aligned before/after control、matched trace fallback control、open-weight paper-suite manifest；`ci-smoke` 只能证明连线与可诊断失败，`retain` 需要 repeated full-suite aggregate
- `claim_nl_slow_loop_improves_eta_fast_path`
  - 命题：NL slow loop 对 ETA fast path 的初始化、family reuse 或 held-out payoff 有可测增益
  - 需要：slow-loop writeback evidence、credit-to-family write count、long-horizon payoff coverage、matched no-fast-prior / non-nested control
- `claim_companion_stateful_relationship`
  - 命题：companion 不是静态 support prompt，而是能感知当前对象状态、在 session 内调整、并在显式用户范围 memory 下跨 session 保留偏好
  - 需要：C1 state sensitivity、C2 within-session adaptation、C3 explicit cross-session retention、C4 default memory isolation、C5 default social scope isolation
  - 当前轻量 verdict 先绑定 `semantic-spine-ready`、`canonical_mean_semantic_spine_coverage`、`canonical_mean_cognitive_loop_readiness` 与 `cross-session-growth`，作为完整 companion 证据前的地基门
- repeated-run verdict 优先使用 paper-suite secondary metric summary 的 sample count / mean，避免只看单次 reference dashboard

### Semantic grounding claims（设计冻结；详见 [`semantic-grounding-evidence.md`](./semantic-grounding-evidence.md)）

- `claim_latent_abstraction_semantically_grounded`
  - 命题：latent action family 对应真实语义动态（区分性 / 领先性 / 迁移性），不是表面措辞聚类
  - 需要：`semantic_grounding_report.json` 的 D1/D2/D3 全部通过 shuffled control 且覆盖统计达门槛；fail 是该主张的 kill 信号，不允许换口径重跑
- `claim_semantic_tracking_not_llm_dependent`
  - 命题：语义 owner 状态追踪由 typed 结构 + PE 闭环承担，不依赖单一 LLM proposal 通道
  - 需要：`semantic_proposal_ablation_report.json` 中 off 臂 per-slot lifecycle 命中率相对 on 臂无大幅退化，两臂同 substrate fingerprint / seed / scenario；off 臂大幅退化时该 claim 必须降级为 "LLM-assisted typed semantic tracking"，不得静默保留原口径

### Same-substrate Companion Bench ablation claims（debt #87；详见 [`companion-ablation.md`](./companion-ablation.md)）

> **冻结的 thesis 第一阶段 claim registry SSOT 见 [`human-world-model-ablation.md`](./human-world-model-ablation.md)**（5 条 retain claim + 8 臂 matched-control matrix + 证据门槛 + 4 态分级 + kill 条件）。下面是同基底工具链已实现的 4 条；registry 相对本节新增 `claim_component_causal_contribution`（PE/ETA/主动学习逐个因果切分，四臂待迁同基底）。

这组 claim 把"人类世界模型 thesis 第一阶段"压成可证伪、同基底的 matched-control。所有 track 跑同一份冻结 Qwen，由 `compare_companion_ablation.py` 产出 verdict（retain 需 delta>0 且 bootstrap CI 非重叠下界 ci_low>0）。

- `claim_pipeline_gt_raw`
  - 命题：volvence 完整 pipeline 在 Companion Bench 上优于裸 Qwen
  - 需要：`raw` 与 `volvence` track 同基底；final_mean delta>0 且 ci_low>0
- `claim_gt_standard_layers`
  - 命题：volvence 优于标准 memory wrapper（ref-harness）**且**优于标准开源 agent 框架（camel）——回应"给 GPT/Claude 套个 wrapper 你们还赢吗"
  - 需要：`ref-harness` 与 `camel` track 同基底；两条 pairwise effect 都 retain
- `claim_training_adds_value`
  - 命题：训练 bootstrap 有增量（volvence > volvence-cold）
  - 需要：`volvence-cold` 与 `volvence` 同基底；delta>0 且 ci_low>0
- `claim_heldout_cohort_stable`
  - 命题：优势在 held-out 场景跨 seed 稳定
  - 需要：held-out + 多 seed 跑（arc_count 足够）+ 相对 CI 半宽足够紧
- 红线（缺一不可，否则 verdict 不可外引）：same-substrate guard VERIFIED；裁判/用户模拟器非 Qwen（#71/#72）；4 条 CompanionBench attestation 全 True；judge robustness/calibration 证据在档；held-out 文本无泄漏
- 四态结论：`kill-criteria-triggered` / `wiring-ready` / `weak-positive` / `first-stage-retained`；`world-model-extension-ready` 需物理侧独立 benchmark，本链不产出

## Dialogue Paper-Suite Evidence Map

本节冻结 dialogue paper-suite 当前 claim 的可操作证据口径。这里的 claim verdict 是 rollout / 外部汇报层的证据读数，不改变 runtime slot，也不成为学习源头。

### `claim_temporal_advantage_over_controls`

**命题**：`pe-eta` 的 canonical scripted 表现优于 matched controls，且优势不是只有结果分数，没有机制证据。

**retain 条件**：
- `canonical_pass_rate` 对 `pe-drive-off` 的 pairwise effect `ci_low > 0`
- `canonical_pass_rate` 对 `eta-off` 的 pairwise effect `ci_low > 0`
- `canonical_runtime_backbone_evidence_rate > 0`
- `canonical_mean_runtime_backbone_signal_quality > 0`

**weak 条件**：
- 对 `pe-drive-off` 与 `eta-off` 的 mean delta 为正，但 runtime backbone consistency 不完整，或 CI 尚未站稳

**主要 artifact**：
- `paper_suite_aggregate.json`
- `evidence_bundle.json`

**轻量测试节点**：
- `tests/test_dialogue_benchmark.py::test_dialogue_temporal_advantage_claim_requires_runtime_backbone_consistency`

### `claim_beyond_scripted_canonical`

**命题**：优势不只存在于 canonical scripted cases，还能扩展到 perturbation / held-out open environment，并经过一个不读 runtime telemetry 的 user policy 检查。

**retain 条件**：
- `open_pass_rate` 对 `pe-drive-off` 的 pairwise effect `ci_low > 0`
- open scenarios 包含 `open_heldout`
- 至少一个 open case 的 `OpenDialogueEpisodeState.user_policy_kind == "transcript-only"`
- hidden perturbation family label 不得出现在 runtime transcript / user input / assistant response 中（只作为 evidence-layer 分组）
- 至少一个 expected repair open case 的 `repair_observable` 为 true
- 至少一个 open case 的 `runtime_adaptation_evidence_observed` 为 true
- `perturbation_pass_rate_pe_eta > 0`

**weak 条件**：
- open 或 perturbation mean delta 为正，但 held-out 或 transcript-only policy evidence 不完整

**主要 artifact**：
- `paper_suite_aggregate.json`
- `reference_emergence_dashboard.json`
- `evidence_bundle.json`

**轻量测试节点**：
- `tests/test_dialogue_benchmark.py::test_transcript_only_user_simulator_ignores_runtime_telemetry`
- `tests/test_dialogue_benchmark.py::test_build_open_dialogue_case_report_uses_open_acceptance_surface`
- `tests/test_dialogue_benchmark.py::test_claim_beyond_scripted_requires_open_repair_and_no_hidden_label_leak`

### `claim_companion_stateful_relationship`

**命题**：companion 不是静态 support prompt，而是至少具备可审计的 semantic owner spine，可在后续 C1-C5 完整证据前证明“状态感知地基”存在。

**retain 条件（当前轻量口径）**：
- `semantic-spine-ready` gate 通过
- `canonical_mean_semantic_spine_coverage >= 1.0`
- `canonical_mean_cognitive_loop_readiness > 0`
- `cross-session-growth` gate 通过

**weak 条件**：
- `semantic-spine-ready` gate 通过
- `canonical_mean_semantic_spine_coverage >= 1.0`
- `canonical_mean_cognitive_loop_readiness > 0`

**主要 artifact**：
- `paper_suite_aggregate.json`
- `reference_emergence_dashboard.json`
- `evidence_bundle.json`

**轻量测试节点**：
- `tests/test_dialogue_benchmark.py::test_run_dialogue_paper_suite_repeated_benchmark_emits_interval_summaries`

### `claim_external_human_legibility`

**命题**：优势能被外部评审者在 blinded transcripts 上感知，而不只存在于内部 telemetry。

**retain 条件**：
- `human_ratings_aggregate.rater_count >= 3`
- `inter_rater_agreement >= 0.6`
- 满足以下任一项：
  - 任一维度 `correlation_with_automatic > 0.1`
  - 所有 pairwise preferences 都有 `pair_count > 0`、`win_rate >= 0.5`、`mean_score_delta > 0`

**weak 条件**：
- `rater_count >= 2`
- `inter_rater_agreement >= 0.4`

**主要 artifact**：
- `expert_review_packet_blinded.json`
- `expert_review_key_internal.json`（内部保留，不外发）
- `human_rating_template.csv`
- `human_ratings_aggregate.json`
- `evidence_bundle.json`

**轻量测试节点**：
- `tests/test_dialogue_benchmark.py::test_dialogue_human_rating_csv_aggregate_exports_external_claim`

### `claim_rare_heavy_net_benefit`

**命题**：rare-heavy / slow artifact 路径带来可测净收益，而不是把收益藏在总通过率中。

**retain 条件**：
- 存在 `pe-eta` vs `pe-eta-no-rare-heavy` matched control
- `canonical_pass_rate` 对 `pe-eta-no-rare-heavy` 的 pairwise effect `sample_count > 0`
- 该 effect 的 `ci_low > 0`

**weak 条件**：
- 存在 matched control 且 `mean_delta > 0`，但 CI 尚未站稳

**fail 条件**：
- 缺少 `pe-eta-no-rare-heavy` control，或 no-rare-heavy 对照没有正向 gap

**主要 artifact**：
- `paper_suite_aggregate.json`
- `evidence_bundle.json`

**轻量测试节点**：
- `tests/test_dialogue_benchmark.py::test_dialogue_rare_heavy_claim_requires_no_rare_heavy_control`

### Artifact 使用边界

- `expert_review_packet_blinded.json` 可以外发；`expert_review_key_internal.json` 只用于内部 unblinding 与 aggregate，不得进入 blind-review 包。
- `paper_suite_aggregate.json` 是 claim verdict 的主入口；它必须由 manifest / provenance / pairwise effects / optional human ratings 重新计算，而不是手工改写。
- `evidence_bundle.json` 是跨系统消费入口，包含 manifest、provenance、run summaries、aggregate metrics、pairwise effects、blind review packet 与 claim verdicts。
- 轻量测试只验证 claim 规则和 artifact shape；完整 empirical 结论仍必须来自 `paper-suite-small` / `paper-suite-full` repeated-run aggregate。

### `phase2_shadow_evidence_smoke.json`

**用途**：Phase 2/3 SHADOW profile 的最小 review artifact。用于检查 profile wiring、candidate readout、metric extraction、deterministic head-to-head 与 cross-generation gate evidence shape 是否完整。

**生成命令**：

```bash
python scripts/run_phase2_shadow_evidence_smoke.py --synthetic-runner --case-limit 1
```

真实 runner evidence 去掉 `--synthetic-runner`；组合 profile 加 `--include-phase3-combos`。

**稳定字段**：

- `schema_version == "phase2-shadow-evidence-smoke.v1"`
- `artifact_kind == "phase2_shadow_evidence_smoke"`
- `runner_kind`: `synthetic` / `default`
- `include_phase3_combos: bool`
- `provenance`
  - `git_sha`
  - `git_branch`
  - `working_tree_dirty`
  - `python_version`
  - `platform`
- `profile_labels`
- `focus_metric_means`
- `focus_metric_deltas_from_baseline`
- `head_to_head_results`
- `cross_generation_gate_evidence`

**Sidecar manifest**：

- 文件名：`phase2_shadow_evidence_manifest.json`
- `schema_version == "phase2-shadow-evidence-manifest.v1"`
- `artifact_kind == "phase2_shadow_evidence_manifest"`
- `source_schema_version == "phase2-shadow-evidence-smoke.v1"`
- `artifacts[]`
  - `path`
  - `sha256`
  - `size_bytes`
- `provenance`（与主 JSON payload 一致）

**Manifest 校验命令**：

```bash
python scripts/verify_phase2_shadow_evidence_manifest.py artifacts/phase2_shadow_evidence_smoke/phase2_shadow_evidence_manifest.json
```

该命令 fail-loudly 校验所有登记 artifact 的 `sha256` / `size_bytes`。reviewer 在引用 JSON / Markdown evidence 前应先跑此命令。

**边界**：

- Synthetic artifact 只验证 schema / wiring / metric surface；不能作为 retain 证据。
- Default runner artifact 可以作为 SHADOW review evidence，但升 ACTIVE 仍需 multi-seed paper-suite-small 或等价 evidence。
- Markdown sibling `phase2_shadow_evidence_smoke.md` 是人类 review 面；JSON 是机器可读 SSOT。
- Manifest 是 artifact 完整性 sidecar；reviewer 应优先用 manifest 校验 JSON / Markdown 是否被改写。

### `phase2_shadow_evidence_multiseed.json` 与 decision report

**用途**：聚合 Phase 2/3 SHADOW profiles 的多次 run，给阶段 D 决策提供稳定输入。

**生成命令**：

```bash
python scripts/run_phase2_shadow_evidence_multiseed.py --case-limit 4 --seeds 0 1 2 3 4 --output-dir artifacts/phase2-shadow-real-multiseed
python scripts/build_phase2_shadow_decision_report.py artifacts/phase2-shadow-real-multiseed/phase2_shadow_evidence_multiseed.json --output-dir artifacts/phase2-shadow-real-multiseed
```

**边界**：

- `phase2_shadow_evidence_multiseed.json` 汇总 mean / std / stderr 与 deterministic head-to-head；它不做 ACTIVE 决策。
- `phase2_shadow_decision_report.json` 给出 `ACTIVE_CANDIDATE` / `REMAIN_SHADOW` / `DISABLED` 建议；真实合并仍需人工 review + rollback plan。
- Synthetic runner 的 decision 永远只能是 `REMAIN_SHADOW`。

### Blind Review External Dispatch（recruitment-agnostic）

本节描述 `claim_external_human_legibility` 的实际外发流程。**该流程刻意不绑定具体招募/分发平台**（Google Form / Prolific / 内部团队都可以承载），只规定文件清单、rater 任务说明与回收 schema，让运营层灵活替换。

**步骤**

1. 生成 packet：跑 `bash scripts/run_dialogue_paper_suite.sh artifacts/dialogue_paper_suite paper-suite-small`，产物里包含
   - `expert_review_packet_blinded.json` —— 外发安全
   - `expert_review_key_internal.json` —— 仅内部
   - `human_rating_template.csv` —— 1 个空模板，含 header `rater_id,item_id,sample_id,blinded_label,dimension_id,score`
2. 招募 rater（≥3 人，独立填写）：把 `expert_review_packet_blinded.json` 与 rater 操作说明分发给 rater；操作说明里至少包括 packet 里的 `review_dimensions[*].prompt`、量表 1–5、不允许查看 packet 之外的内部 telemetry。**任何分发渠道都可以**（云盘 / 邮件 / 表单 / 平台），只要 rater 最终交回**一个 CSV per rater**（保留 header）。
3. 回收 + 合并：把所有 rater 的 CSV 放到一个目录，例如 `artifacts/dialogue_paper_suite/ratings/`。每个 rater 一个 CSV，rater_id 不冲突。然后用 `volvence_zero.agent.load_dialogue_human_rating_entries_csv_dir(csv_dir)` 一次性加载（rater_id 冲突会抛错防止重复计票，紧急情况可用 `forbid_rater_id_collision=False`）。
4. Aggregate + 重导：把合并后的 entries 与 `expert_review_key_internal.json` + 原始 `reference_run_report` 喂给 `aggregate_dialogue_human_ratings(packet, entries, internal_key, reference_report)`，得到 `human_ratings_aggregate`；再调用 `export_dialogue_paper_suite_artifact_bundle(report, output_dir, human_ratings_aggregate=aggregate)` 重导 bundle，让 `claim_external_human_legibility` verdict 反映真实评分。
5. 验收：`paper_suite_aggregate.json` 中
   - `claim_external_human_legibility.status == "retain"` 需要 `rater_count >= 3` AND `inter_rater_agreement >= 0.6` AND（任一维度 `correlation_with_automatic > 0.1` 或所有 pairwise preferences `win_rate >= 0.5` 且 `mean_score_delta > 0`）
   - `weak` 仅需 `rater_count >= 2` 且 `inter_rater_agreement >= 0.4`

**安全护栏**

- `tests/test_dialogue_benchmark.py::test_blind_packet_transcripts_have_no_profile_label_leak` —— 验证 packet 里的 transcript 不含 `pe-eta` / `pe-drive-off` / `eta-off` 这类 profile-label 字符串（case_id 是英文常用词如 `repair` / `goal_drift` 时不能误报，因为它们必然出现在话题中）
- `tests/test_dialogue_benchmark.py::test_dialogue_paper_suite_artifact_bundle_exports_expert_review_packet` —— 验证 `source_profile_label` 不出现在 packet JSON 全文中
- `human_ratings_aggregate.json` 里包含真实 profile_label，**只能内部消费**，不要随 packet 一起外发

**当前 inter-rater agreement 算法**

实现见 `_dialogue_inter_rater_agreement`：对每个 `(item_id, sample_id, dimension_id)` 单元，计算所有 rater 两两绝对差的均值，映射到 [0, 1]（除以 4，假定 1–5 量表），最后对所有单元再取均值。这与 Krippendorff's alpha / Cohen's kappa 不直接可比。Krippendorff alpha v2 是 backlog 项；外发数据对外引用时请明确指出口径。

## Companion Evidence Map

本节冻结 companionship claim 的最小自动证据口径。它回答“陪伴能力是否只是固定 prompt”的问题，但不把固定 scripted benchmark pass 误写成人类级关系成熟。

### `claim_companion_stateful_relationship`

**命题**：companion 能根据当前对话对象状态调整表达，在同一 session 中形成连续状态，并在显式同一用户 / 同一生命体范围的共享 memory 下跨 session 保留偏好；默认 session 仍保持隔离，避免多租户串记忆。

**retain 条件**：
- C1 `state_sensitivity`: 同一 companion 面对 task vs emotional context 时，`interlocutor_state` 至少在 task focus / directness / rapport warmth 上分化，且 readout confidence 达标
- C2 `within_session_adaptation`: `low-mood-disclosure` 多轮内出现至少两个 expression intents，PE 有变化，且最终 interlocutor readout confidence 达标
- C3 `explicit_cross_session_retention`: 注入共享 `MemoryStore` 后，session B 能检索到 session A 写入的偏好
- C4 `default_memory_isolation`: 未显式注入共享 store 时，两个 session 的 `MemoryStore` 不共享
- C5 `default_social_scope_isolation`: 默认 companion turn 的 R16 scope 固定为 `primary/self`，`multi_party_identity` / `social_prediction` / `social_prediction_error` 以 SHADOW readout 发布且 social PE 默认为空
- AAC1 `alignment_pe_repair_visibility`: commitment alignment 从 AGREE→REJECT 的转变能进入 relationship PE，并产生 `DEFER_ONLY` repair follow-up policy
- RGM1 `regime_delayed_attribution_visibility`: dialogue-like repair/support regime evidence 能产生 delayed attribution、delayed credit records 与 evaluation readout metrics，且不要求硬编码 regime 切换
- RFL1 `reflection_writeback_stability`: reflection 能消费 dialogue slow-loop evidence，并通过 checkpoint / rollback 的 bounded apply path 写入 memory / regime evidence，不直接绕过 owner
- AAC1 / RGM1 / RFL1 在 dialogue paper-suite 层是**外部注入**：上游 `lifeform_evolution.companion_evidence.run_companion_evidence()` 生产 `CompanionEvidenceGate` 列表后，由编排层（如 lifeform-bench）通过 `export_dialogue_paper_suite_artifact_bundle(..., companion_structural_gates=((gate_id, passed), ...))` 注入。dialogue 层不导入 `lifeform_evolution`，只读不可变 `(gate_id, passed)` 元组（R8 / SSOT 合规）。当未注入时，verdict 的 retain 检查回退到 4 项地基（`semantic-spine-ready` + 两个 spine 指标 + `cross-session-growth`），与历史行为一致；注入后 retain 还要求所有提供的 structural gate 通过
- v2 `composite_score` 记录 C1-C5 gate score + widening transcript diversity；v2 transcript diversity 是 widening diagnostic，不单独作为 retain 硬门槛

**weak 条件**：
- C1/C2 通过，但 C3 只能通过人工脚本或临时状态证明，尚无显式 shared-memory gate

**fail 条件**：
- C1 不分化，或 C2 没有 session 内状态变化，或 C3 需要默认跨 session 串记忆才能通过，或 C5 显示默认 social scope 不是 `primary/self`

**主要 artifact**：
- `companion_evidence_report.json`
- `lifeform-bench --companion-evidence-report` stdout
- `companion_evidence_report.json.transcripts[]`：paraphrase / tone shift / delayed return / preference conflict 微场景 transcript，用于后续 blind review / human rating
- `dialogue_option_discovery_report.json`（Phase A trajectory diagnostic, non-gating）：从 `DialogueBenchmarkTurn` 与 optional snapshot replay `action_replay` 读取 `switch_gate`、`active_abstract_action`、`prediction_error`、`closed_segments` / `z_t_digest`，报告 `termination_event_count`、`option_duration_mean`、`abstract_action_diversity`、`pe_spike_near_termination_rate`、`option_reuse_across_cases`。该 artifact 验证 ETA 时间抽象是否在 dialogue trajectory 上有可审计形状，但当前不改变 retain 条件。
- `pe_counterfactual_closure_report.json`（Phase A trajectory diagnostic, non-gating）：复用现有 `pe-eta` / `pe-drive-off` / `pe-eta-pe-readout-only` / `eta-off` profile comparison，报告 `pe_to_credit_drop`、`pe_to_behavior_drop`、`readout_only_gap`、`eta_dependency_gap`、`closure_strength`。该 artifact 验证 PE drive 与 readout-only / ETA-off 对照之间的因果闭环迹象，但当前不替代后续 delayed / shuffled / wrong-sign PE harness。
- `longitudinal_dialogue_report.json`（PhaseB longitudinal trajectory diagnostic, non-gating）：按 persona/session 聚合多 session 证据，报告 `retention_rate`、`isolation_pass_rate`、`adaptation_trend`、`drift_risk_score`、`trajectory_strength`、`cross_session_verdict`。首版支持从 `DialogueLongitudinalBenchmarkReport` 派生，也支持显式 `LongitudinalDialogueSessionEvidence`（shared-memory / isolation evidence 由上游 owner/report 产生后注入）。该 artifact 验证同一 virtual user 多 session 轨迹的连续性地基，但当前不改变 retain 条件。
- `nl_ablation_matrix_report.json`（PhaseC NL diagnostic, non-gating）：聚合 `full-nl` / `no-ssl` / `no-rl` / `no-reflection` / `no-rare-heavy` / `no-fast-prior` / `timescale-off` 的 structured metrics，报告 `cross_session_growth_score`、`heldout_payoff_score`、`memory_churn_risk`、`behavior_drift_risk`、`slow_to_fast_transfer_gain`、`full_nl_advantage`。首版支持从 dialogue comparison report 或 explicit variant metrics 生成；没有真实 profile 的 variant 必须显式输入，不能伪造。
- `memory_stratum_flow_report.json`（PhaseC memory diagnostic, non-gating）：读取 `MemorySnapshot` 或 normalized dict evidence，报告 stratum progression、promotion/decay pressure、derived index activity、lifecycle signal strength 与 `memory_flow_strength`。它只读 memory owner 发布的 snapshot/readout，不推断 raw text。
- `regime_lockin_report.json`（PhaseC regime diagnostic, non-gating）：读取 `RegimeSnapshot` 或 normalized dict evidence，报告 `lockin_strength`、`switch_rate`、`hysteresis_proxy`、`delayed_attribution_strength`、`sequence_payoff_strength`、`regime_identity_stability`。当前没有 runtime-level hysteresis owner，`hysteresis_proxy` 只由候选 regime 波动但 active regime 保持的结构化证据派生。

**Phase A trajectory evidence 边界**：

- 两个 Phase A artifact 均通过 `export_dialogue_paper_suite_artifact_bundle(..., include_phase_a_trajectory_reports=True)` 或显式传入 report 后进入 `EvidenceBundle.reference_artifacts`。
- 它们是 readout / evidence artifact，不写 owner、不成为 learning source、不新增 runtime slot。
- `dialogue_option_discovery_report` 首版允许 `evidence_quality="turn-telemetry-only"`，因为完整 per-turn replay accumulation 尚未进入 runtime；若提供 `snapshot_replay_artifact`，则升级为 `snapshot-replay+turn-telemetry`。
- `pe_counterfactual_closure_report` 首版只覆盖已有 profile counterfactual；delayed / shuffled / wrong-sign PE 仍是后续 harness，不在本阶段伪造。
- `claim_companion_stateful_relationship` 的 retain 条件暂不消费这两个 artifact；它们为后续从机制 gate 升级到 trajectory gate 提供证据输入。

**PhaseB longitudinal trajectory evidence 边界**：

- `longitudinal_dialogue_report` 通过 `export_dialogue_paper_suite_artifact_bundle(..., include_phase_b_longitudinal_report=True)` 或显式传入 report 后进入 `EvidenceBundle.reference_artifacts`。
- 它是 readout / evidence artifact，不写 owner、不新增 runtime slot、不成为 learning source。
- 首版 v1 personas 是 `direct-but-overwhelmed`、`slow-trust-repair`、`boundary-sensitive`、`preference-conflict`、`delayed-return`，每个 persona 先 3 sessions，作为 synthetic trajectory surface；不能把 v1 结果夸大成人类级 companion retain。
- Shared-memory retention 与 default isolation 是双轨指标：`explicit_retention_observed` / `retrieved_preference_count` 证明显式共享路径存在，`default_isolation_preserved` 证明默认隔离仍保持。
- `claim_companion_stateful_relationship` 的 retain 条件暂不消费该 artifact；未来升级 retain 需要至少同时满足 multi-session trend、memory retention、default isolation、preference-conflict repair 与 human review anchor。

**PhaseC NL / memory / regime longitudinal evidence 边界**：

- `nl_ablation_matrix_report`、`memory_stratum_flow_report`、`regime_lockin_report` 可显式传入 `export_dialogue_paper_suite_artifact_bundle(...)`，或在后续完整 paper-suite runner 中按需生成，最终进入 `EvidenceBundle.reference_artifacts`。
- 三个 artifact 都是 non-gating diagnostics，不写 owner、不新增 runtime slot、不成为 learning source。
- `nl_ablation_matrix_report` 的首版 explicit variant metrics 是 proof harness 入口；若缺少 no-SSL / no-RL / no-reflection 等真实 profile，不得把 full-NL claim 写成 retain，只能作为待补对照。
- `memory_stratum_flow_report` 的 stratum flow 只基于 `MemorySnapshot.total_entries_by_stratum`、`pending_promotions`、`pending_decays`、`cms_band_vectors`、`lifecycle_metrics` 等 owner-owned readouts。
- `regime_lockin_report` 的 lock-in / hysteresis 是从 `RegimeSnapshot.turns_in_current_regime`、`regime_changed`、`candidate_regimes`、`delayed_attributions`、`sequence_payoffs` 派生的 readout，不改变 regime selection policy。
- 后续把 `claim_companion_stateful_relationship` 升级到 trajectory retain 时，至少需要同时具备：PhaseA ETA/PE trajectory evidence、PhaseB longitudinal dialogue evidence、PhaseC NL positive ablation gap、memory stratum flow、regime lock-in，以及 external human review anchor。

**轻量测试节点**：
- `tests/lifeform_e2e/test_companion_learning_evidence.py`
- `tests/lifeform_e2e/test_companion_evidence_report.py`
- `tests/test_dialogue_benchmark.py::test_build_dialogue_option_discovery_report_from_turns_and_replay`
- `tests/test_dialogue_benchmark.py::test_build_pe_counterfactual_closure_report_from_existing_profiles`
- `tests/test_dialogue_benchmark.py::test_dialogue_paper_suite_export_writes_phase_a_trajectory_reports`
- `tests/test_dialogue_benchmark.py::test_build_longitudinal_dialogue_report_from_session_evidence`
- `tests/test_dialogue_benchmark.py::test_dialogue_paper_suite_export_writes_phase_b_longitudinal_report`
- `tests/test_dialogue_benchmark.py::test_build_nl_ablation_matrix_report_from_explicit_variant_metrics`
- `tests/test_dialogue_benchmark.py::test_build_memory_stratum_flow_report_from_dict_snapshots`
- `tests/test_dialogue_benchmark.py::test_build_regime_lockin_report_from_dict_snapshots`
- `tests/test_dialogue_benchmark.py::test_dialogue_paper_suite_export_writes_phase_c_reports`
- `tests/test_eta_nl_clean_action_abstraction.py::test_dialogue_option_discovery_accepts_snapshot_replay_context`

**运行入口**：
- `lifeform-bench --companion-evidence-report`
- 可选 JSON：`lifeform-bench --companion-evidence-report --companion-evidence-json companion_evidence_report.json`

## Social Cognition Evidence Map

本节冻结 R16-R20 的最小自动证据口径。早期 slice 的目标不是证明人类级社会理解，而是证明社会认知状态没有退化为 renderer 文案分支或单一 `user_model` bucket。

### `claim_tom_owner_separation`

**命题**：Theory of Mind 状态至少在契约与显式 proposal path 上区分 belief / intent / feeling / preference；belief conflict 不会写入 preference owner，preference conflict 也不会伪装成 belief。

**retain 条件**：
- R16A `active_identity_memory_scope`: explicit `EnvironmentEvent.frame` publishes ACTIVE `multi_party_identity`, and memory writes inherit the same subject / audience scope without renderer inference
- R16B `active_social_pe_memory_visibility`: ACTIVE `social_prediction` / `social_prediction_error` publish MEMORY_VISIBILITY prediction/error and negative credit when active scope suppresses cross-subject memory
- T1 `tom_owner_contract`: `OtherMindRecordKind` 有限枚举覆盖 belief / intent / feeling / preference，四类 snapshot 会拒绝 wrong-kind record
- T2 `explicit_tom_proposal_path`: 显式 proposal 可以填充目标 ToM owner，final wiring 默认不把通用 semantic runtime 当 ToM classifier
- T3 `false_belief_preference_separation_probe`: 人工 false-belief + preference-conflict probe 中，belief 与 preference 分别落入各自 owner，record kind 不混写
- T4 `structured_tom_runtime_path`: `LLMToMProposalRuntime` 的结构化 JSON 输出能填充目标 ToM owner， malformed / low-confidence 输出不会伪造 durable record
- T5 `affect_preference_separation_probe`: 同一输入中的 transient feeling 与 durable preference 分别进入 `feeling_about_other` / `preference_about_other`，不混写
- R1 `wrong_addressee_role_pe_credit`: 人工 wrong-addressee role PE 可以转成 shared credit，证明 role mistake 能进入 PE/credit 链路而不是 renderer 规则
- R2 `role_prediction_diagnostic_visibility`: EnvironmentEvent role frame 产生 `ROLE_ASSIGNMENT` prediction，且 `response_assembly.semantic_record_counts` 可诊断性显示 role prediction count
- R18A `active_role_frame_diagnostics`: default final wiring 中 ACTIVE `conversational_role` 消费 EnvironmentEvent role frame，并发布 diagnostics，不泄漏到 renderer 文案
- G1 `common_ground_diagnostic_visibility`: 显式 dyad/group common-ground atoms 可以进入 `common_ground` owner，并在 `response_assembly.semantic_record_counts` 诊断性显示 atom count
- G2 `structured_common_ground_runtime_path`: 结构化 common-ground runtime 可以把 dyad/group JSON proposal 写入 `common_ground` owner，并显示为 diagnostic count
- G3 `reference_repair_common_ground_probe`: repair / clarification evidence 可以进入 dyad common-ground atom，而不是由 renderer 文案推断
- GROUP1 `group_diagnostic_visibility`: 显式 group identity / joint commitment 可以进入 `groups` owner，并在 `response_assembly.semantic_record_counts` 诊断性显示 group 与 joint commitment count

**fail 条件**：
- ToM proposal 进入 `user_model.stable_preferences` 才能通过；或四类 ToM 状态共享同一 untyped owner；或 renderer / raw text 分支直接决定 belief/preference 行为。

**轻量测试节点**：
- `tests/contracts/test_social_cognition_contracts.py`
- `tests/test_social_tom.py`
- `tests/test_social_cognition_evidence.py`

**主要 artifact**：
- `social_cognition_evidence_report`（Python API: `lifeform_evolution.run_social_cognition_evidence()`）
- `social_cognition_evidence_report_to_dict(report)`：T1-T3 gate payload，可被后续 CLI / bundle 引用
- CLI stdout: `lifeform-bench --social-cognition-evidence-report`
- JSON artifact: `lifeform-bench --social-cognition-evidence-report --social-cognition-evidence-json social_cognition_evidence_report.json`

## EQ Evidence-Chain Closure Bundle (Wave E1-E5)

Wave E1-E5 把 `default NoOp` 路径下不出 evidence 的链路打通；所有面都通过单一 manifest 暴露，下游 cite 按 `evidence_bundle.json` gate 读，不再各家拼装。

**单命令入口**：`scripts/run_eq_evidence_bundle.sh [ROUNDS] [BUNDLE_DIR]`

**Bundle 装配器**：`python -m lifeform_evolution.evidence_bundle assemble --bundle-dir <dir> --output <path>`

### Gates

| Gate id | 关闭条件 | 关联 debt |
|---|---|---|
| `debt_10b_item3` | 至少一个 long-form scenario 的 `tom_records_total_last > 0` 且 `common_ground_dyad_atoms_total_last > 0`（必须用真 LLM proposal runtime） | #10B item 3 |
| `debt_10c_il_rapport_snr` | 跨 scenario 的 `cross_scenario_summary.il_rapport_trend_snr_mean ≥ 1.5` | #10C |
| `debt_11_long_form_coverage` | 至少一次 bundle 报告 `pe_window_filled_scenario_ratio ≥ 0.5` | #11 follow-up |
| `wave_e4_multi_party_keying` | 3-party scenario artifact 存在；F3 facets 由 `tests/contracts/test_multi_party_shadow_evidence.py` 静态守门 | — |
| `debt_6_rewarding_state_head_promotion` | rollback drill test 存在且通过；promotion 仍需真 trace evidence | #6 |
| `debt_7_pe_critic_head_promotion` | rollback drill test 存在且通过；promotion 仍需真 trace evidence | #7 |

### Required artifacts

bundle dir 中必须有：

- `<scenario_id>_longitudinal.json` × 4（4 个 long-form scenario）
- `long-form-three-party-arc_longitudinal.json`（multi-party SHADOW probe）
- 装配器自动产 `evidence_bundle.json`

每个 artifact 的 SHA-256 + 大小记录在 `evidence_bundle.json.artifact_provenance` 内，外部审阅可通过 sha256 验真。

### 不变量

- 所有 evidence gate 必须 typed JSON readout，不允许从文本输出推断（`no-keyword-matching-hacks` rule）
- bundle 装配器是纯只读脚本，不 mutate 任何 owner / runtime；它只读 artifact 并写 manifest
- 每个 gate 的 `passed` 字段必须能从 typed metric 读出，不依赖人工判读

## State KV P6 冻结尽调包

单命令 `python scripts/run_state_kv_due_diligence.py --mode all` 先执行或复用
新增 evidence lane，再生成 `state-kv-freeze-manifest.v2` 与
`state-kv-due-diligence.v1`。manifest
冻结 Qwen 权重 SHA-256、Prefix-KV artifact id、profile 清单、三 generation
seed、场景、指标、judge panel，以及每份输入 evidence 的路径/schema/state/
SHA-256；结构由
`packages/vz-runtime/src/volvence_zero/schemas/state_kv_freeze_manifest.schema.json`
校验。report 生成前重新计算所有 evidence 指纹，任一文件变化即 fail loudly，
禁止用后改 artifact 复写已冻结结论。

v2 同时冻结 prefix manifest 自身 SHA-256 与 canonical resolved experiment
config（profiles、完整 persona/probe material、generation 参数、阈值、judge
配置），并校验 freeze id。总编排实际调用 quality、bank-gain、control-dim
和 credit-longitudinal runner；旧的昂贵 lane 必须逐份存在、可解析并进入哈希
清单，不能静默跳过。汇总器只映射结构化门，不把机制分叉、局部通过或缺失臂
包装成完整结论。

## 变更日志

- 2026-07-28: 归档 ETA Gate 2 v30 prefix expected-value 2+2 校准。新增固定
  seed target/audit cohort、fresh validation、action selection audit readout
  与 batch continuation scoring 证据；Qwen2.5-0.5B full-width 896 CPU 单 seed
  validation audit 为负，维持 `mechanism-supported` 和 injection disabled，
  并把下一包冻结为真实 downstream outcome target。
- 2026-07-17: semantic grounding claims 的两个实验包实现完成：`volvence_zero.agent.semantic_grounding`（D1/D2/D3 + shuffled controls）+ `scripts/build_semantic_grounding_report.py`；`lifeform_evolution.semantic_proposal_ablation`（9-slot scripted probe、on/off matched 双臂、case-level bootstrap CI、两臂 grounding 交叉读数）+ `scripts/build_semantic_proposal_ablation_report.py` + vertical 工厂 `VZ_SEMANTIC_PROPOSAL_CHANNEL` 通道级开关。synthetic smoke lane 差分设计验证通过；hf substrate 真实 trace 运行 pending。
- 2026-07-17: 新增 semantic grounding claims（`claim_latent_abstraction_semantically_grounded` / `claim_semantic_tracking_not_llm_dependent`），绑定 `docs/specs/semantic-grounding-evidence.md` 的两个实验设计与 non-gating artifact；实现与真实 trace 运行 pending。
- 2026-07-14: evidence bundle v2 统一入口（CP-00 / GAP-10）。新增
  `volvence_zero.agent.evidence_bundle_v2.assemble_evidence_bundle_v2` +
  单命令 `python scripts/assemble_evidence_bundle_v2.py`：把
  dialogue/ETA paper-suite、EQ longitudinal、learned-shadow、CompanionBench
  P1 manifest 五个 lane 聚合为一份 `evidence_bundle_v2.json`（共享
  git/runtime provenance + 每 lane sha256 fingerprint）；被请求的 lane
  缺文件 / 非 JSON / 缺该 lane 必需 provenance key 时
  `EvidenceBundleV2Error` fail loudly，不产出半成品。聚合器不重算任何
  lane verdict（gate 语义留在各 lane owner）。同轮：EQ
  `evidence_bundle assemble` 补 `provenance` block（git sha / dirty tree /
  seed_schedule，对齐 paper-suite 契约）；`baseline_manifest` 新增
  `build_runtime_behavior_baseline`（CP-01 运行时指标冻结：turn latency、
  PE 四轴、evaluation family means、credit closure、regime、switch 读数，
  digest 排除延迟字段且必须可重放）；`learned_coverage_version` 同步为
  `v0.1@2026-07-14`。测试：`packages/vz-runtime/tests/test_evidence_bundle_v2.py`、
  `test_agi_uplift_baselines.py::test_runtime_behavior_baseline_replays_identical_digest`。
- 2026-07-12: CP-LSS-01（learned-shadow closure packet）落地统一 learned-shadow 证据面：
  冻结 operator profile `build_learned_shadow_rollout_config()`（n_z=16 + 四个 torch
  autograd backend 全 SHADOW，生产默认仍全 DISABLED）；单命令
  `python scripts/run_learned_shadow_evidence_smoke.py` 产出
  `learned-shadow-evidence-smoke.v1` artifact + sha256 manifest。artifact 由
  `volvence_zero.agent.learned_shadow_evidence.collect_learned_shadow_evidence()`
  只读装配四个 owner 的本地证据面（temporal runtime `latest_runtime_shadow_report`
  / SSL `latest_report` torch\_\* / internal RL `latest_optimization_report` torch\_\*
  / CMS `latest_cms_backend_evidence`），任一 owner 缺证据或在 SHADOW 下写回即
  `LearnedShadowEvidenceError` fail loudly。**定级为 P0 wiring / synthetic-CPU
  tier**：只证明接线与 SHADOW 无副作用语义，不构成 ACTIVE 晋升证据（后者仍
  gate on Linux CUDA 真 trace lane，见 #88/#89）。验收测试
  `packages/vz-runtime/tests/test_learned_shadow_evidence.py`（含 CP-02 n_z=16
  全 DISABLED 纯路径稳定性门）。
- 2026-05-09: Wave E5 (Evidence-Chain Closure milestone) 落地 EQ Evidence-Chain Closure Bundle 段：单命令入口 `scripts/run_eq_evidence_bundle.sh`、装配器 `python -m lifeform_evolution.evidence_bundle assemble`、6 条 typed gate verdict、artifact provenance（sha256 + size）。E1-E4 落地的所有 owner / family / longitudinal 字段已通过 bundle 暴露，外部 cite 可通过单一 `evidence_bundle.json` 路径读取所有 verdict。
- 2026-05-02: Social Cognition evidence report 增加 R16A active identity memory scope gate，覆盖 EnvironmentEvent frame → ACTIVE multi_party_identity → memory subject/audience scope 链路
- 2026-05-02: Social Cognition evidence report 增加 R16B active social PE memory visibility gate，覆盖 ACTIVE social_prediction/social_prediction_error → memory visibility PE → negative credit 链路
- 2026-05-02: 强化 `claim_beyond_scripted_canonical`，新增 hidden perturbation label non-leak、repair observable 与 runtime adaptation evidence 条件，仍复用现有 dialogue benchmark / paper-suite / evidence bundle
- 2026-05-01: 新增 Dialogue Paper-Suite Evidence Map，冻结 temporal advantage、beyond scripted、external human legibility、rare-heavy net benefit 四类 dialogue claim 的 retain / weak / fail 条件、artifact 边界与轻量测试入口
- 2026-05-02: 新增 Companion Evidence Map，冻结 C1-C4 companion stateful-relationship claim、运行入口与轻量测试节点；v2 增加 widening transcript artifact 与 composite score（diagnostic，不替代 C1-C4 retain gate）
- 2026-05-02: 增加 C5 default social scope isolation gate，覆盖 R16 `primary/self` 默认 scope 与空 social PE 链路，避免 companion v1 在多人化迁移中隐式串人
- 2026-05-02: 增加 AAC1 alignment PE repair visibility gate，覆盖 AAC commitment alignment reject → relationship PE → defer-only repair follow-up 链路
- 2026-05-02: 增加 RGM1 regime delayed attribution visibility gate，覆盖 dialogue repair/support signal → RegimeSnapshot delayed attribution → credit/evaluation readout 链路
- 2026-05-02: 增加 RFL1 reflection writeback stability gate，覆盖 dialogue slow-loop evidence → bounded reflection apply → checkpoint/rollback 链路
- 2026-05-02: 新增 Social Cognition Evidence Map，冻结 R17 ToM owner separation 的 T1-T3 轻量证据门槛
- 2026-05-02: 增加 Social Cognition evidence report artifact，T1-T3 由 `lifeform_evolution.run_social_cognition_evidence()` 汇总输出；CLI 支持 `--social-cognition-evidence-report` 与 `--social-cognition-evidence-json`
- 2026-05-02: Social Cognition evidence report 增加 R1 wrong-addressee role PE credit gate，覆盖 R18 role mistake → credit 链路
- 2026-05-02: Social Cognition evidence report 增加 T4/T5 structured ToM gates，覆盖 LLMToMProposalRuntime 结构化输出和 affect/preference 分离
- 2026-05-02: Social Cognition evidence report 增加 R2 role prediction diagnostic visibility gate，覆盖 R18 role prediction → response_assembly diagnostic count 链路
- 2026-05-02: Social Cognition evidence report 增加 R18A active role frame diagnostics gate，覆盖 ACTIVE conversational_role → diagnostics 且不进入 renderer 文案
- 2026-05-02: Social Cognition evidence report 增加 G1 common-ground diagnostic visibility gate，覆盖 R19 explicit atoms → response_assembly diagnostic count 链路
- 2026-05-02: Social Cognition evidence report 增加 G2/G3 structured common-ground gates，覆盖 R19 structured runtime → owner atom 与 reference repair → dyad atom 链路
- 2026-05-02: Social Cognition evidence report 增加 GROUP1 group diagnostic visibility gate，覆盖 R20 explicit group state → response_assembly diagnostic count 链路
- 2026-05-02: 重写 `docs/specs/emergent-action-abstraction.md` 的 replay artifact 为 `snapshot_replay_bundle.json`，由 existing snapshots 导出，不依赖 trace owner
- 2026-04-25: 补充 ETA open-weight residual-control 与 NL slow-loop-support claim 的 evidence 边界，明确 synthetic / trace backend 不能单独支撑真实 residual-control claim
- 2026-04-26: 细化 real open-weight gate：把 planned layer fraction 与 actual hook fire rate 分离，新增 prefix-aligned intervention 与 smoke/full evidence tier 边界
- 2026-04-25: 初始版本，建立 claim-to-evidence / blind-review / pairwise-effect / evidence-bundle 的统一口径

## 变更日志补充

- 2026-07-13: CP-24 capacity→gain ladder manifest. 新增
  `volvence_zero.agent.capacity_ladder`：`CapacityLadderArm` /
  `CapacityLadderManifest` / `build_capacity_ladder_manifest(...)`，冻结计划轴
  `n_z ∈ {3,16,64,256}`、PE critic capacity `{1,2,4}`、COCOA hidden
  `{8,32,128}`、backend combo、500/1000 turns、0.5B/1.5B/7B substrate 与
  3 seed，默认生成 864 arms。该 manifest 只定义 evidence lane 任务，不产生
  retain claim；实际跑分仍 gate on Linux CUDA / judge budget / artifact provenance。
  测试：`packages/vz-runtime/tests/test_capacity_ladder_manifest.py`。

## 变更日志补充（production verdict）

- 2026-07-13: production verdict typed evaluator. 新增
  `volvence_zero.agent.production_verdict`：`ProductionEvidenceSummary` /
  `ComponentGateEvidence` / `evaluate_production_verdict(...)`，将计划的四个
  最终口径（`first-stage-retained` / `product-companion-retained` /
  `architecture-platform-only` / `inconclusive`）和逐组件 ACTIVE / SHADOW /
  DISABLED 决策固化为 read-only typed 判定。缺 human anchor / longitudinal / P2
  claim 时返回 `inconclusive`；安全或 rollback gate 失败的组件为 DISABLED；
  证据不足但安全可回滚的组件为 SHADOW。测试：
  `packages/vz-runtime/tests/test_production_verdict.py`。

## 变更日志补充（longitudinal + human anchor）

- 2026-07-13: longitudinal + human-anchor study manifest. 新增
  `volvence_zero.agent.longitudinal_human_anchor`：`LongitudinalPersonaPlan` /
  `HumanAnchorProtocol` / `build_longitudinal_human_anchor_manifest()`，冻结计划
  §14 的 5 persona × 20 sessions、每 session 8–15 turns、shared-memory hydration
  vs default isolation comparison、核心 continuity/repair/contamination/followup
  metrics，以及 3 blinded raters + inter-rater agreement >= 0.6 的 human anchor
  protocol。该 manifest 只定义研究形状，不产生 claim；真实 transcript、rating
  CSV 与 aggregate artifacts 仍需外部 run。测试：
  `packages/vz-runtime/tests/test_longitudinal_human_anchor_manifest.py`。
