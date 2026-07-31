# 连续记忆系统 Spec

> Status: draft
> Last updated: 2026-07-20
> 对应需求: R5, R6

## 要解决的问题

如何实现跨时间尺度的连续记忆谱，并通过反思将经验沉淀为持久结构？

## 关键不变量

- **NL 范式层**：记忆是任何由输入引起的 neural update，分布在所有参数中（NL §A.8）
- **运行时工程层**：系统使用显式 owner 模块（MemoryStore, CMS, snapshots）使范式层的分布式记忆可检查、可回滚
- 记忆是连续谱，不是二元短期/长期分割
- 记忆写入通过正式 owner 和 API，不可绕过
- 慢反思产出两类产物：记忆沉淀 + 策略沉淀
- 慢反思异步运行，不阻塞实时交互
- `MemoryModule` 必须继续是唯一 memory owner；更强的 NL-style memory 只能作为 owner 内部 tower 演进，不能通过新增第二 owner 获得
- query / learned core / artifact retrieval 必须走同一条 owner-side memory signal contract，而不是多套语义空间松散拼接
- public `cms_state` 可以继续保留三层摘要，但 owner 内部允许更深的 nested tower，并通过 machine-readable tower profile 对外发布
- public `cms_state` 除兼容三带摘要外，还必须发布 machine-readable `continuum_profile`，至少包含 `bands`、`reconstruction_edges` 与 `readout_band_id`

### Continuum frequency contract

当前 runtime contract 明确区分两层：

1. **兼容摘要层**
   - `online_fast`
   - `session_medium`
   - `background_slow`

2. **连续谱层**
   - `continuum_profile.profile_id`
   - `continuum_profile.bands[*]`
   - `continuum_profile.reconstruction_edges[*]`
   - `continuum_profile.readout_band_id`

其中：

- `bands` 负责表达当前 owner 发布的频率带、更新频率、持久性偏置、检索权重与 pending signal
- `reconstruction_edges` 负责表达 `slow->fast reset`、`meta-init`、`associative readout` 等跨层恢复/迁移路径
- consumer 若需要理解“连续谱位置”，应优先读取 `continuum_profile`，而不是自行从三带字段反推

## 工程挑战

- 设计多层记忆（瞬态 → 情景 → 持久 → 派生索引），每层有不同的更新频率
- 实现提升/衰减/部分重建机制
- 实现异步慢反思路径，产出记忆沉淀和策略沉淀两类产物
- 确保记忆写入通过正式 owner 和 API，不可绕过

## 算法候选

来自 `docs/next_gen_emogpt.md`：

### CMS 多频率层（NL 附录 A.5）

```
y_t = MLP^(ν_K)(MLP^(ν_{K-1})(... MLP^(ν_1)(x_t)))
```

第 `i` 层 MLP 的参数每 `c^(i)` 步更新一次。

### 记忆层级映射

| 层 | 内容 | 更新频率 | 算法对应 |
|----|------|----------|----------|
| 瞬态工作状态 | 当前 wave、场景、活跃帧状态 | 每 turn | CMS 最高频层 |
| 会话情景状态 | 近期交互轨迹、未解决张力 | 每场景/会话 | CMS 中频层 |
| 持久语义记忆 | 稳定的用户/自我/关系/世界知识 | background-slow | CMS 低频层 |
| 派生索引 | 可重建的检索辅助、摘要、任务投影 | 按需重建 | — |

### 记忆操作

- **写入**：仅通过正式 owner 和 API
- **提升**：低频层从高频层提取持久知识
- **衰减**：有限容量迫使遗忘非关键信息
- **部分重建**：遗忘后可从低频层回流知识到高频层
- **抗遗忘**：CMS 确保低频层保留被高频层遗忘的知识

### 慢反思路径（R6 + NL/ETA）

异步运行于交互窗口之后：
- NL 的 CMS 低频层天然对应慢反思——参数每 `c^(K)` 步才更新一次
- ETA 的 SSL-RL 交替循环提供反思-强化交替模式
- M3 优化器的慢动量 `m^(2)` 每 `ν` 步聚合梯度，是优化器层面的"反思"

**两类反思产物**：

| 产物类型 | 内容 | 写入目标 |
|----------|------|----------|
| 记忆沉淀 | 持久卡片、信念、开放循环、偏好轨迹 | 持久语义记忆 |
| 策略沉淀 | 抽象控制器更新、路径先验、策略偏好 | 控制器参数 |

当前实现口径：

- P07 先输出 consolidation proposal 和审计信息
- 第二阶段已补充 bounded apply path：memory owner 支持 checkpoint / rollback
- 当前真实写回首先落在低风险目标：durable promotion、belief writeback、decay、promotion threshold 调整
- CMS 当前已升级为 machine-readable 多频带 owner state：`online-fast / session-medium / background-slow` 都带有 `cadence_interval / observations_since_update / pending_signal`
- `session_medium / background_slow` 不再每 turn 同频更新，而是通过显式 cadence gating 分层推进
- reflection apply 已从散落阈值收敛成统一 bounded consolidation score 路径，memory/regime writeback 幅度由同一条 score 决定
- 当前 memory owner 已内部重分层成 `learned core + artifact store + derived index`：CMS learned core 是主记忆基底；显式 entries 退到 durable artifact / audit 层；semantic index 明确是可重建 derived layer
- retrieval 已升级为 owner-side learned-core-guided recall：query 先与 CMS state 融合形成 recall signal，再用 artifact entries 作为可解释锚点；semantic index 继续纳入 checkpoint / restore，但不再被视为主记忆本体
- runtime query facets 当前可组合 `user_text + substrate + 上一轮 temporal_abstraction + 上一轮 dual_track` 的已发布快照上下文，不引入同轮循环依赖
- 当前 reflection 已可消费 regime owner 发布的 `delayed_outcomes + identity_hints`，并在 credit gate 允许时把 typed identity proposal 沉淀为 durable identity entries；memory owner 仍是唯一持久写入 owner
- 当前 `MemoryModule` 已直接消费 `prediction_error` slot：owner 会把 PE 写成 `prediction_error:*` 记忆事件、调节 `promotion_threshold`，并把主导误差维度纳入 retrieval facets
- 当前 turn-time memory ingest 不再只依赖 substrate 序列重建；orchestrator 会把当前 `user_text` 作为 owner-side turn input 直接交给 `MemoryModule`，使用户原话本身进入 transient write / retrieval query
- 当前默认 memory owner 已携带 nested MLP CMS profile；context boundary 与
  rare-heavy import 后会通过 owner-side `reset_nested_context()` 触发
  slow->fast 初始化，并把 `nested_context_reset_count` /
  `last_nested_reset_applied` / `slow_to_fast_init_benefit` 发布到 lifecycle
  telemetry。`slow_to_fast_init_benefit` 的正式语义是 reset 后最多 5 次
  replay 中 `copy-init shadow loss - active-init loss` 的均值；没有 matched
  shadow 时为 `0`，禁止再解释为 reset 前后向量位移。
- 当前 CMS 的 fast / session / slow / nested 更新律已开始通过 owner-side `learned_update_rule` 统一发布：`CMSState` / checkpoint 现可携带 updater state、effective learning rate、update gate、slow-mix 与 update summary，使 memory 不再只是“learned core + hand-tuned write rule”的混合体
- 当前默认慢反思已改为 session-post queued orchestration：turn-time wiring 只生成 deferred consolidation request，真正的 durable promotion / decay / belief writeback / temporal-prior apply 在 context boundary 后由 session-post slow loop 执行；memory owner 仍是唯一 durable 写入 owner
- 当前 session-post slow loop 还会通过独立 `session_post_slow_loop` slot 发布 queue state 与 recent completion summaries，使 memory/cadence 证明面不再依赖 runner 私有 telemetry；queue owner 对完全一致的 session-post job 实施 session-lifetime 幂等并公开 duplicate count，避免重试导致 memory / temporal 双重写回，同 ID 不同 payload 则显式拒绝
- 当前 temporal->memory 的 CMS feedback 已收敛到正式 owner 路径：final wiring 不再直接 side-effect `memory_store.observe_encoder_feedback(...)`，而是由 temporal owner 通过已发布快照携带 feedback signal，memory owner 在自己的 processing path 中消费
- 当前 memory owner 内部 retrieval 已切到统一 owner-side signal contract：query、entry、tower readout 都通过同一条 vector projection / fusion law 进入 recall scoring，artifact index 不再维持一套独立主语义
- 当前 `CMSState` 已开始发布 machine-readable tower profile：除了兼容的 `online_fast / session_medium / background_slow` 摘要外，还会公开 nested meta-init levels 与 tower readout
- 当前 reflection durable apply 已升级为 tower-native consolidation：slow loop 对 memory 的写回不再只做 `reflect_lessons()`，而会把 promoted/durable/belief lessons 显式压进 learned tower 的 online/session/background readout 路径
- 当前 companion evidence 增加 `RFL1 reflection_writeback_stability` gate：证明 dialogue slow-loop evidence 可通过 bounded reflection apply 写入 memory / regime，并保留 checkpoint / rollback，而不是绕过 owner 直接突变
- 当前 tiny Hope owner-side proof 已收敛到 `MemoryStore` / `CMSMemoryCore` 内部：`LearnedUpdateRule` 生成有界 write / step / decay / reset 系数，`CMSState.hope_self_modification_state` 发布机器可读 self-mod evidence，checkpoint / restore 会同时回滚 band state 与 Hope meta-state；这不是独立 `HopeModule`，也不新增第二 memory owner。
- Gate 6 evidence-only 初始化控制继续位于同一 owner：
  `MemoryStore.initialize_nested_context_for_evidence()` 只允许
  `meta-init / copy-init / random-init / no-init / external-meta-init` 五种显式
  模式，发布不可变 `CMSContextInitializationEvidence`，并保证初始化不改
  background-slow state、MLP 权重、nested target 或 updater 的 learned
  parameter state（reset decision telemetry 仍按 owner 规则更新）。生产
  `reset_nested_context()` 当前委托 `copy-init` 作为生产回滚路径；其余模式
  不能成为 live policy 或第二 memory owner。conditioned meta-init 只在
  显式 evidence 构造中启用。
- 2026-07-30 `gate6-meta-init.v1` 正式结果为 mechanism PASS / causal
  NO-GO：meta-init 胜过 random/no-init，但与 copy-init 的 locked
  steps-to-target 均为 `0`，早期 AUC 还低 `0.000171`；paired/swapped
  diagnostic 无可区分优势。当前只能证明 owner-side nested initializer
  可运行、可审计、零事实泄漏且可 checkpoint rollback，不能声称 slow
  meta-prior 相比直接复制更快适应，也不能声称 user-related prior。
- 2026-07-31 `gate6-conditioned-meta-init-v3-retest.v1` 在同一 owner 内
  增加 context prototype（最近原型 EMA、reset 时相似度混合），并提供
  `context_conditioned/prototype_count/context_match_score` 不可变证据；
  `k=1` 且关闭 conditioning 时回到旧 global EMA。机制门全绿，但相对
  copy-init effect=`-0.0490765`、negative-transfer rate=`1.0`，故触发
  kill condition，生产继续使用 copy-init。

## 当前 proof surface

当前 repo 在连续记忆系统上优先证明 4 条工程命题：

1. `PE-memory closure`
   - memory owner 不只是被动存储，而是直接消费 `prediction_error`，把 PE 写入 memory event、调节 `promotion_threshold`，并影响 retrieval facets
2. `slow-shapes-fast`
   - nested CMS 的 slow band 会在 context boundary 或 rare-heavy import 后，通过 owner-side `reset_nested_context()` 对快层初始化产生可观察影响
3. `learned-core-guided recall`
   - retrieval 不再主要证明“artifact store 能被搜到”，而是证明 recall 先由 learned core 驱动，再落到 durable artifacts 作为解释性 readout；owner lifecycle telemetry 会发布 recall confidence / core-guided recall evidence
4. `tower-native consolidation`
   - background-slow writeback 不再只把 lesson count 打到 learned core，而是把 promoted entries、durable entries、belief updates 与 lesson-derived pressure 显式融合成 tower consolidation update，并留下 checkpoint / rollback 可验证证据
5. `tiny-Hope bounded self-modification`
   - memory owner 会发布 `hope_self_modification_state` 与 lifecycle metrics（如 generated learning / decay / reset rate），证明自修改发生在 owner 内部、可检查、可 checkpoint / rollback，而不是通过新 owner 或 substrate live mutation 完成
6. `runtime evidence escapes the owner`
   - `MemoryStore` 现在会把 `SubstrateSnapshot` 统一压缩成 runtime-grounded lifecycle telemetry：例如 `last_runtime_backbone_signal_quality`、`last_runtime_backbone_signal_strength`、`last_runtime_backbone_hook_coverage` 与 `last_fast_memory_runtime_alignment`；`evaluation` 与 `dialogue_benchmark` 默认优先读取这组读数，而不是把 tower telemetry 当作唯一 headline
7. `review-only fast memory still leaves evidence`
   - frozen / review-only doctrine 不再等同于“没有 online-fast 证据”；只要 runtime 产出了合法 `fast_memory_signal`，session owner 也会正式写入 `MemoryStore`，并把它与最近一次 runtime backbone signal 对齐
8. `tower evidence reaches rollout artifacts`
   - tower telemetry 仍会进入 emergence dashboard、paper-suite summary 与 NL-essence gate，但它现在被降级为从属证据面，需要和 runtime-grounded evidence 一起成立，而不是单独代表默认主证据面

这里的 proof surface 依赖 owner 发布的 lifecycle telemetry 与 machine-readable tower profile，而不是仅凭文本输出推断。当前 benchmark 能证明“慢层影响快层的证据面已存在，而且 recall / consolidation 已经开始围绕统一 tower 组织”；同时，默认主证据已开始转向 runtime/backbone readout 与 fast-memory alignment，tower depth/alignment/consolidation 则退到辅助约束位。它仍不能被表述成论文级 distributed memory / self-modifying learner 的完整复现。

### R5/R6 Behavioural Proof Surface (CMA-2 / Phase 2 W2.6)

来源：Logan J. *Continuum Memory Architectures for Long-Horizon LLM Agents*. arXiv:2601.09913, 2026.

上面的 proof surface 都是 **owner-internal lifecycle telemetry**——可以证明记忆 owner 在执行预定行为，但不能证明这些行为对外部观察者来说**真的有用**。CMA-2 加一层**行为级证据面**：4 个 deterministic probe，跨 N 个 simulated session 共享同一个 `MemoryStore`，断言 retrieval 端的「长时间窗连续性」。Probe 失败是 R5/R6 在长 horizon 上的真实违约信号，而不是 readout 缺失。

| Probe | 文件 | 紧扣的不变量 | 当前状态 |
|---|---|---|---|
| Update | `tests/longitudinal/test_vz_memprobe_update.py` | 后续 belief override 在 retrieval rank 中胜过早期同主题 belief（不被 stale fact 主导） | PASS |
| Temporal | `tests/longitudinal/test_vz_memprobe_temporal.py` | 给定 anchor event，其时间邻域（前后 turn）能被召回；topically-unrelated 邻居不会污染 anchor 查询 | PASS |
| Assoc | `tests/longitudinal/test_vz_memprobe_assoc.py` | 跨 owner 的 3 跳 chain（user_model → relationship_state → boundary_consent）查询能完整覆盖；distal end (boundary_consent.granted) 不丢失 | PASS |
| Context | `tests/longitudinal/test_vz_memprobe_context.py` | 同关键词在不同 regime 下，`RetrievalQuery.facets` 能 disambiguate top-1 | PASS（debt #10D 关闭后转 PASS：`_score_entry` 加 `+5` per matched facet 的显式 boost） |

#### 与现有 owner-internal proof 的关系

- 不替代上述 8 条 owner-internal proof，而是补它们看不到的「行为面」。owner 再合规，retrieval 出口端能不能撑住长 horizon 是另一回事。
- 不调任何 LLM runtime（默认 `NoOpSemanticProposalRuntime` 路径）。这意味着 VZ-MemProbe 在 debt #10B 关闭之前就有意义，而不是和 ToM owner records 一起卡住。
- ToM owner 的 records 本身需要 LLM runtime 才能写满；当 #10B 关闭后，VZ-MemProbe 的 setup 可以从「deterministic write 模拟 consolidation」切换到「真实 turn 流 + drain slow loop」，assertion 表面不变。

#### Read-only invariant

VZ-MemProbe 失败**只**作为长 horizon 连续性 evidence；R12 的「评估只读」在长 horizon 上仍然成立：

- 失败**不**驱动 ModificationGate 决策。
- 失败**不**反向更新 retrieval scoring 算法（owner 自己内部演化）。
- 失败**先**写 known-debts，再考虑改 owner——R8 owner 边界优先于 evidence 修补。

### Substrate-feature retrieval & attribute readout（Phase 1.C, NL-aligned）

NL-对齐裁判线（与 A-Mem 的对照）：

1. CMS / `MemoryModule` 仍是唯一 memory owner，没有 LLM-curated 第二 owner；
2. 记忆写入与组织规则只来自 owner 自身（PE owner readout + substrate feature_surface），不外包给外置 LLM；
3. retrieval 的 dense embedding 来自 substrate `feature_surface` 的 owner-side 下游使用，不引入与 substrate 表征并行的独立嵌入空间。

落点：

- `vz-memory/memory/retrieval.py` 新增 `_substrate_embedding(feature_surface, dim)`：按稳定字典序拼接 feature_surface → L2 归一化 → 截 / 补到 `dim`。`feature_surface` 为空时返回零向量，调用方退回到 hash 风格的 `_semantic_embedding`，保证 bootstrap / 无 substrate 测试桩仍可工作。
- `vz-memory/memory/store.py::MemoryStore._owner_signal(...)` 现在优先把 substrate-derived signal 作为 dense 维度的主成分（substrate 缺席时权重退化为 0），semantic + metadata 仍保留稳定贡献，消费侧 cosine 行为不变。
- `MemoryStore.observe_substrate(...)` 在 turn 起始处缓存 `feature_surface`，使写入与查询 hit 同一份；`MemoryStore.apply_prediction_error_signal(...)` 缓存 PE intensity / primary axis / regime_id / `pe_decomposition.epistemic_magnitude` / `pe_decomposition.aleatoric_magnitude`。
- `MemoryStore.write(...)` 在每条新 entry 上 pin 一个 owner-internal `MemoryAttributeReadout`：PE 字段来自缓存，`substrate_feature_digest` 来自 `_substrate_embedding(..., dim=min(learned_signal_dim, 8))`。
- `MemorySnapshot.attribute_summary` 是 capped tuple（默认 16，按 `timestamp_ms` 倒序），让下游有结构化窗口可读。**`MemoryEntry` schema 不动**；`MemoryStoreCheckpoint.entry_attributes` 以默认空 tuple 的 additive 字段保存 owner-coupled attribute 投影，使删除/改写回滚保持 artifact、semantic index 与 attribute index 原子一致。

下游兼容性：

- 现有 `MemoryEntry.tags`-driven retrieval / write paths 行为保持不变（tags 仍是规则字符串）。
- attribute index 写入 checkpoint / persistence；旧 checkpoint 缺字段时按空 tuple 恢复。它仍是 Memory owner 内部投影，不成为新的事实 owner 或学习信号源。
- 不参与 acceptance gate；未来 Phase 2.A / 2.B 可让 credit / temporal 在自家 owner 内消费 attribute_summary 以做更精细的 contribution 估计。

**NL 路线明确不做**：

- 不引入 A-Mem 的 LLM 卡片演化路径（违反裁判线 2）。
- 不把 HippoRAG 2 当作独立 memory 架构（违反裁判线 1、3）；只保留它的图召回 / 多跳算法作为 `DerivedRetrievalIndex` 内部未来候选。
- 不替换 substrate 基底；不改 CMS update rule 的算法主干（Titans / ATLAS 的 update-rule 子集留作 `docs/specs/cms-atlas-titans-uplift.md` 的中期专项）。

### NL 关键洞察

NL 重新定义"记忆 = 任何由输入引起的神经更新"（附录 A.8），记忆分布在所有参数中而非独立模块。灾难性遗忘是压缩的自然后果——有限容量迫使模型遗忘以容纳新信息。

## 接口契约

**消费的输入**：
- `substrate` 快照：当前可实现的 substrate surface（用于记忆检索的上下文）
- `temporal_abstraction` 快照：当前控制器状态（用于记忆检索的意图上下文）
- `dual_track` 快照：轨道标记（用于按轨道写入记忆）
- `prediction_error` 快照：上一轮 outcome mismatch，用于 owner-side memory write、promotion threshold 调节和 retrieval facets

**产出的输出**：
- `memory` 快照：`MemorySnapshot`
  - 各层级摘要（模块自身生成）
  - 本轮检索到的相关记忆
  - 统计信息
  - pending promotion / decay 状态
  - machine-readable `cms_state`
  - nested lifecycle telemetry（如 reset 次数与 slow->fast init benefit）
  - tower lifecycle telemetry（如 tower depth、tower alignment、tower-native consolidation count）
  - `attribute_summary`（optional, Phase 1.C）：最近 N 条 owner-internal `MemoryAttributeReadout`，每条携带 `pe_intensity / pe_primary_axis / regime_id / substrate_feature_digest / epistemic_magnitude / aleatoric_magnitude / timestamp_ms`

**owner API**：

- 所有写入通过 `MemoryWriteRequest` 进入 Memory owner
- longitudinal evidence replay 只能通过
  `MemoryStore.observe_replay_signal(...)` 把 owner 已发布的不可变 signal
  送回 CMS；该 evidence-only ingress 与 live `observe_substrate` 共用同一
  cadence / replay / PE-gate / anti-forgetting update kernel，空或非有限
  signal 必须 fail loudly，不能伪造 `SubstrateSnapshot`
- retrieval contract 区分“本轮检索结果”与“持久状态摘要”
- retrieval ranking 由 Memory owner 内部统一负责，可组合 `user_text + substrate facets + owner query facets` 做 tower-guided retrieval；artifact / lexical 只是同一 owner-side fusion law 的不同证据源。`RetrievalQuery.facets` 与 entry tag 的命中走显式 `+5` per match 的 tie-breaker boost（debt #10D 关闭后落地，`_score_entry` 内），不只走 embedding 通道，确保 regime-context disambiguation 在 lexical/semantic 几乎并列时仍可决定 top-1
- promotion / decay / reconstruction 先建模为显式状态，不作为隐式副作用暴露给消费者
- 第二阶段补充 `checkpoint / restore` 与 bounded `apply_reflection_consolidation`
- `promotion_threshold` 属于 Memory owner 的可回滚低风险自适应参数
- 显式 `MemoryEntry` 属于 artifact / durable explanation layer，不等同于主记忆真相
- semantic index 属于 Memory owner 内部 derived index，不向外暴露独立 owner

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.3 节

当前 `DATA_CONTRACT.md` 3.3 已同步 `CMSState.tower_profile`、`tower_depth`、`continuum_profile`、`update_rule_state` 与 `hope_self_modification_state`；本 spec 的连续谱 contract 以这些公共字段为读数入口，不能要求 consumer 读取 `CMSMemoryCore` 私有结构。

## 时间定位、证据区间与记忆 claim 的行为证据阶梯（2026-07-20）

来源：Meta S-EMBER（`arXiv:2607.02689`）与 CMU/MBZUAI Beyond Perplexity（`arXiv:2607.00368`），
见 `research/frontier-sweep-2026-07-20.md` §B1 / §H4 / §4.5。本节是 spec motivation +
评估口径要求，**不改 schema**；任何新增字段仍须先进 `docs/DATA_CONTRACT.md`。

### 时间定位与证据区间（S-EMBER）

S-EMBER 的关键结果：模型规模 / context 长度提升能改善"发生了什么"，却不能改善"**何时**发生"
（grounding precision 长期 < 3%，recall 随证据间隔十分钟内从 ~50% 衰减到 ~22%）。对本 owner 的含义：

- 情景记忆（episodic stratum）的评估必须把"**回答正确**"和"**证据定位正确**"拆成两个只读指标：
  后者要求检索结果能给出时间定位（哪个 session / 哪个 turn 区间）与证据区间（支持该记忆的
  原始 entry lineage）。
- R5/R6 不能退化为"大 context + 向量检索"：`MemoryEntry` 的 lineage / provenance 字段
  是时间定位能力的载体，retrieval readout 评估应覆盖 temporal localization 命中率
  （VZ-MemProbe 的 `mp.temporal.*` 是现有起点，缺"证据区间正确性"维度）。

### 记忆 claim 的行为证据阶梯（Beyond Perplexity）

任何"系统学会记住 / 个性化 / 持续学习"的对外 claim，必须区分三层且不得互相冒充：

1. stream/domain adaptation（loss/perplexity 下降）——**不构成记忆证据**；
2. bridge internalization（内部表征吸收）——中间证据；
3. deployment-time behavioral learning（行为级）——唯一可对外引用层，至少要求：
   **later recall**（延迟自由回忆）、**paraphrase robustness**（改写鲁棒）、**retention**（保持）、
   **locality**（不污染无关知识）、**conflict handling**（冲突更新），且配 matched
   explicit-memory baseline 对照。

该阶梯是 R12/R15 的证据门：防止把 TTT / CMS band 更新的 loss 改善升格为
"CMS / relationship learning 成立"的 claim。落地位置为评估侧（readout-only），
见 `docs/specs/evaluation.md` 与 Lane C 记忆行为证据阶梯工件。

**首包已落地（2026-07-20）**：`tests/longitudinal/test_memory_behavior_ladder.py`
以 VZ-MemProbe 同款 deterministic 协议实现五级 `mb.*` 探针（later_recall /
paraphrase / retention / locality / conflict）。当前结果：retrieval 机制侧
6/7 PASS；**`mb.paraphrase.top1_semantic`（零词面重叠改写）strict xfail**——
默认 hash-stub 语义嵌入下改写查询输给共享表面 token 的干扰项，需真实
substrate-backed `SemanticEmbeddingBackend`（#91 真 substrate 增益证据）才能翻绿。
诚实边界：本套探针只覆盖"模拟 consolidation + 检索机制"层；对外 deployment-memory
claim 还需真实 turn-flow 写入路径 + matched explicit-memory baseline 对照臂。

### retrieval policy 版本化（spec motivation）

PAHF / SkillOS / EvolveMem 共同说明：记忆 owner 不只拥有内容，还拥有 retrieval / retention
policy 本身，且 policy 也会老化。未来 memory snapshot 至少应能发布：retrieval policy 的
版本与 wiring level、最近一次 gated change 与回退点。**本节不扩 schema**；若落地，
新字段先进 `DATA_CONTRACT.md` 3.3。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时（5.5）| 通过快照发布记忆状态 |
| 依赖 | Prediction Error 主链 | 用于 owner-side PE 写入、threshold 调节和 query facets |
| 依赖 | 多时间尺度学习（5.1）| 各层按不同频率更新 |
| 被依赖 | 双轨学习（5.4）| 提供按轨道隔离的记忆存储 |
| 被依赖 | 认知 Regime（5.8）| 提供 regime 历史效果的记忆 |
| 协作 | 时间抽象（5.2）| 为 metacontroller 提供记忆上下文 |
| 协作 | 信用分配（5.6）| 反思输入包含信用分配记录 |
| 协作 | 评估体系（5.7）| F4 学习质量中的记忆沉淀质量评估 |

## 变更日志

- 2026-07-30: frozen evaluation 的 `learning_enabled=False` 进入 `MemoryModule`
  owner。该模式跳过 substrate/temporal/PE 观察、artifact 写入与访问触碰，只执行
  `record_access=False` 的只读检索并发布快照；因此 CMS、artifact、索引、PE write gate
  与 recall 学习统计均保持 checkpoint 指纹不变。

- 2026-07-30: 新增 evidence-only `observe_replay_signal` owner ingress，
  使 Gate 5 可从共享真 trace 的 public substrate-derived digest 重放完整
  CMS update law，而不伪造 substrate snapshot 或建立第二 memory owner。
- 2026-07-20: 新增 §"时间定位、证据区间与记忆 claim 的行为证据阶梯"：S-EMBER 时间定位/证据区间双指标要求、Beyond Perplexity 三层行为证据阶梯（loss↓ 不构成记忆 claim）、retrieval policy 版本化 motivation。不改 schema；来源 `research/frontier-sweep-2026-07-20.md` §6 同步项。
- 2026-07-17: G4 reflection consolidation learned SHADOW 候选（反思沉淀节）。新增 `vz-cognition.reflection.consolidation_learner.ConsolidationScoreLearner`：有界线性残差 head，输入为既有 consolidation 特征（memory_pressure / cross_tension / alert_pressure / positive・negative credit / pe_penalty），固定公式 = 初始化 + 回滚点；session-held（`AgentSessionRunner` 持有，经 final wiring 注入 per-turn `ReflectionEngine`），`ConsolidationScore.learned_promotion_score`（默认 None）report-only 发布；settlement 用下一 turn realized PE magnitude（一 turn 窗口，gate-learner 语义）。promote / decay / writeback gate 全部不读 learner。测试：`tests/test_reflection_consolidation_learner.py`。
- 2026-05-06: Phase 1.C 上线 substrate-feature retrieval embedding 与 owner-internal `MemoryAttributeReadout`；新 `MemorySnapshot.attribute_summary` 字段，`MemoryEntry` schema 不动；明确 A-Mem / HippoRAG 2 不进路线图，Titans / ATLAS update-rule 子集走 `cms-atlas-titans-uplift.md` 中期专项。
- 2026-08-01: P2 relationship-memory console 增加 scope-guarded 单条删除/改写；`MemoryStore.delete_artifact_entry` 原子清理 artifact、semantic index 与 attribute index，checkpoint 新增向后兼容的 `entry_attributes`，保证持久化失败可完整回滚。
- 2026-04-25: 与 `DATA_CONTRACT.md` 3.3 对齐，明确 continuum / tower / learned-update 字段已经是 public `cms_state` 读数，不是 consumer 侧推断
- 2026-04-20: 接口契约改为直接消费 `prediction_error`；当前实现口径明确 memory owner 已将 PE 用于记忆事件写入、promotion threshold 调节和 retrieval facets
- 2026-04-22: 默认慢反思执行路径切到 session-post slow loop；memory durable apply 不再与 turn 主链同频，而由 context boundary 后的 queued consolidation 驱动
- 2026-04-09: next_gen_emogpt v2: explicit distinction between NL paradigm-level memory (any neural update) and runtime-level memory (explicit owner modules + snapshots); both layers are valid and complementary
- 2026-04-09: U04 Persistence load path fix: `MemoryStore.load_from_backend()` now calls `_reconstruct_checkpoint()` to rebuild `MemoryStoreCheckpoint` from deserialized JSON dict and `restore_checkpoint()` to restore full store state including entries, strata, semantic index, and CMS. Verified: save→restart→load roundtrip preserves entries (including track/stratum), CMS MLP parameters, and handles version incompatibility via safe degradation. CMS nested meta-targets also survive persistence roundtrip.
- 2026-04-09: U02 CMS MLP Upgrade + Nested variant: CMS bands upgraded from fixed-dim vector (`dim=3`) to optional MLP mode (`mode="mlp"`) with configurable `d_in`/`d_hidden`. Each band is a `CMSBandMLP` (2-layer residual MLP: `y = x + W1 @ tanh(W2 @ x)`, W1 zero-init for identity at start). `CMSVariant.NESTED` added: background band meta-learns session init target, session meta-learns online init target (`_update_nested_meta_targets`); `reset_context()` re-initializes from these meta-learned targets (not simple copy). Verified: init error decreases across repeated context resets. `CMSCheckpointState` extended with `nested_session_init_target`/`nested_online_init_target`. Parameter budget: ~1K params/band at d_in=16, d_hidden=32.
- 2026-04-06: P19 N-dim CMS + Gradient-Style Updates: upgraded from EMA blending to gradient-style per-band updates with configurable learning rates (online_lr, session_lr, background_lr), momentum (momentum_beta), and anti-forgetting backflow from slow→fast bands. CMSBandState now includes learning_rate, momentum, and anti_forgetting_strength. encoder_feedback now handles dim mismatch via projection.
- 2026-04-06: P10 CMS-enhanced encoder: CMS bands feed into SequenceEncoder prior; encoder output feeds back to CMS via observe_encoder_feedback; cadence gating preserved
- 2026-04-06: 补充 machine-readable CMS bands + cadence gating、bounded consolidation score、hybrid semantic retrieval 的当前实现口径
- 2026-04-06: 补充 checkpoint/rollback、promotion_threshold 和最小 CMS 核的当前实现口径
- 2026-03-25: 初始版本，从 SYSTEM_DESIGN.md 和 next_gen_emogpt.md 提取
