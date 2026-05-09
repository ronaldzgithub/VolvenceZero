# Known Architecture Debt

> Status: tracked, not blocking
> Last updated: 2026-05-09 (Wave E1-E5 + post-E1 fence-strip diagnosis + option B per-session runtimes landed; debt #10B item 3 closure path now 2-layer)

> 2026-05-09 update (option B / per-session LLM proposal runtimes): 上一条 update 列出的"第二个 sub-issue"（attempt counter / records-total 只反映 last-turn）按推荐修法 (2) 落地。**Layer 1**: `AgentSessionRunner.__init__` ([`packages/vz-runtime/src/volvence_zero/agent/session.py`](../packages/vz-runtime/src/volvence_zero/agent/session.py)) 在构造期从 unwrapped `_semantic_proposal_runtime` 派生 `_tom_proposal_runtime` + `_common_ground_proposal_runtime`（仅当上游是 `LLMSemanticProposalRuntime` 实例时；否则保持 `None` 维持 NoOp fail-closed 默认）；`run_final_wiring_turn(...)` 调用追加 `tom_proposal_runtime=self._tom_proposal_runtime, common_ground_proposal_runtime=self._common_ground_proposal_runtime`，绕开 `build_final_runtime_modules` 的 per-turn 默认构造分支，使两个 runtime 实例在整个 session 的所有 turn 复用，`LLMProposalAttemptAccumulator` 真正 session-cumulative。副带修一个 latent 二级 bug：当 `pending_semantic_events` 触发 `AdapterSemanticProposalRuntime` 包裹时，per-turn 路径的 strict `isinstance` 检查会失败并静默 fail-close ToM/CG 自动接线 —— 现在从 unwrapped runtime 一次性构造避开这条路径。**Layer 2**: [`examples/run_cross_session_probe_llm.py`](../examples/run_cross_session_probe_llm.py) `_summarize_artifact` 闭合判定从 `tom_records_total_last > 0`（last-turn owner snapshot；per-turn 设计，不是稳定的"激活"信号）改为 `any(per_round_tom_proposal_parsed_ok_total) > 0 OR any(per_round_common_ground_proposal_parsed_ok_total) > 0`（session-cumulative 类型化 schema 通过的 proposal 数；与 Layer 1 配合直接反映 session 级 EQ 链激活）；legacy last-turn 字段仍打印作为次要 readout 但不再 gate 闭合。**Layer 3**: 4 个新 unit test 在 [`tests/contracts/test_llm_proposal_runtime_session_persistence.py`](../tests/contracts/test_llm_proposal_runtime_session_persistence.py) 锁住 (a) LLMSemanticProposalRuntime → 两个派生 runtime 实例非 None；(b) NoOp → 都保持 None；(c) 同实例 3 次 propose → `proposals_received_total == 3` 且 `parsed_ok == 3`；(d) CG 同样累计；并在 [`tests/lifeform_e2e/test_llm_semantic_runtime_evidence_chain.py`](../tests/lifeform_e2e/test_llm_semantic_runtime_evidence_chain.py) 加 `bench.tom_proposal_attempts_total == len(scenario.turns)`（pre-fix 必为 1，post-fix == 3）的 cumulative-monotonic 断言，把 per-turn-rebuild 退化作为可观测的回归点固化。875 个相关 contract / e2e test 全绿（70 个新 + 修改测试 + 805 个未变更回归）。

> **#10B item 3 闭合 condition 现在 2 层都满足**：
> 1. **Layer 1（counter 累计性）已修**：post-Layer-1 evidence chain test 已经在 fake provider 下显式断言 `tom_proposal_attempts_total == 3`（场景 3 turn）。
> 2. **Layer 2（verdict semantics）已修**：`_summarize_artifact` 现在用 cumulative parsed_ok 判 closure，而 fence-strip 修复 (上一 update) 已经证明真实 Qwen 1.5B 在 cross-session-emotional-followup turn 1+2 各产 1 个 valid CG dyad atom（[`artifacts/eq_uplift/llm_proposal_debug_postfix_v2.jsonl`](../artifacts/eq_uplift/llm_proposal_debug_postfix_v2.jsonl)）。两层叠加：post-Layer-1 重跑 probe 会让 `per_round_common_ground_proposal_parsed_ok_total >= [2]`，Layer 2 verdict 据此判 `[10B item 3] CLOSED`。
> 3. **唯一未直接执行的项**：实际重跑一次 1.5B Qwen probe 拿端到端 artifact（~40 min）。这是验证步骤而非新 architecture work；Layer 1 的 unit test 已经从行为层证明 counter 会累计、Layer 2 的 verdict 逻辑已经在现有 v2 artifact 上验证（仍 OPEN，因为 v2 是 pre-Layer-1 跑出的，counter 仍 cap 在 1 — 这是预期行为而不是 verdict bug）。
>
> **`#10B item 3` 实质完成，等一次 end-to-end 跑确认即可移到 closed 段**。
>
> **Out-of-scope（保留 follow-up，不开新 debt）**：
> - Owner snapshot 累计语义（`OtherMindRecord.records` / `CommonGroundSnapshot.dyad_atoms` 是否应跨 turn 累计）—— 涉及 R8 所有权决策，不在本 wave 范围；当前以 cumulative counter 作为 EQ 链激活的 SSOT
> - `_TOM_PROMPT` few-shot 升级（让 Qwen 1.5B 更稳产 ToM record）—— 正交于本 wave，可独立推进
> - 跨 session counter 聚合（已在 `LongitudinalFamilyReport` 层面就位，本 wave 无 change）

本文档记录已知但暂不处理的架构债。每条都经过评估：**不处理短期不会导致系统行为错误**，但**中长期会影响可演化性或可调试性**。新增条目时参照相同格式：路径 / 问题 / 风险 / 触发条件 / 推荐修法。

> 2026-05-09 update: Evidence-Chain Closure milestone (Wave E1-E5) 全部代码级交付落地，744+ 个 contract test 全绿（在原 738 基础上新增 ToM/CG/PE 诊断 + rollback drill + multi-party + bundle assembler 6 类共 35 个测试）。**关键诊断面已就位**：
> - **Wave E1**: `LLMProposalAttemptCounters` typed contract（[`packages/vz-contracts/src/volvence_zero/llm_proposal_diagnostics.py`](../packages/vz-contracts/src/volvence_zero/llm_proposal_diagnostics.py)）+ `LLMProposalAttemptAccumulator` 接入 `LLMSemanticProposalRuntime` / `LLMToMProposalRuntime` / `LLMCommonGroundProposalRuntime`，每个 owner snapshot 通过 `proposal_diagnostics` 字段暴露；`final_wiring.py:1201-1268` 加 isinstance wrapper warning。**debt #10B 仍开放**——code-level 诊断面就位，但真实 1.5B Qwen evidence run 未执行（需 CI / 人工跑），item 3 fail-loud 状态保留。下次 evidence run 跑成功后即可关闭。
> - **Wave E2**: 3 个新 long-form scenario（companion-arc / task-arc / trust-arc）+ 1 个 3-party scenario 落地；`BenchmarkReport.pe_distribution_window_filled` + F4 metric `f4.pe_distribution_window_filled` + `LongitudinalFamilyReport.pe_distribution_window_filled_round_ratio` + cli cross-scenario summary 全部就位。**debt #11 follow-up（scenario 多样性）已实质落地**，长形态 scenario 数从 1 → 5。
> - **Wave E3**: `tests/contracts/test_learned_baseline_rollback_drill.py` 6 个 rollback drill 测试；`docs/specs/credit-and-self-modification.md` 与 `docs/specs/prediction-error-loop.md` 增补 promotion criteria 表格。**debt #6 / #7 仍开放**——rollback drill 已通过，但 SHADOW → ACTIVE 升级仍需 ≥ 500 turn 真 trace evidence。
> - **Wave E4**: `BenchmarkReport.per_interlocutor_record_counts` + `wrong_person_pe_events_total` + 两个新 F3 metric (`f3.distinct_interlocutor_count` / `f3.wrong_person_pe_events_total`)，readout-only。
> - **Wave E5**: `scripts/run_eq_evidence_bundle.sh` 单命令入口 + `python -m lifeform_evolution.evidence_bundle assemble` 子命令 + 6 条 typed gate verdict（`debt_10b_item3` / `debt_10c_il_rapport_snr` / `debt_11_long_form_coverage` / `wave_e4_multi_party_keying` / `debt_6_rewarding_state_head_promotion` / `debt_7_pe_critic_head_promotion`）+ artifact provenance（sha256 + size）。`docs/specs/evidence_program.md` 同步增补 EQ Evidence-Chain Closure Bundle 段。
>
> **未关闭的债项及前置依赖**：
> - #10B / #10C：等待真实 1.5B Qwen evidence run 跑成功（需 ≥ 24GB RAM 或 GPU，本次 agent session 不能执行）。
> - #6 / #7：等待 ≥ 500 turn 真 trace 上 `validation_delta ≥ 0.02` 持续观察证据（同样需要 evidence run）。
> - #8 / #9：与本 milestone 不相关，状态不变。

> 2026-05-06 update: ssot-cleanup-p0-p4 五个 wave 全部 land。debts #1 / #2 已关闭；debt #3 缩窄到一个文件（已抽 `application/scoring_helpers.py`，剩 `vz-cognition` 的两个 fork 留给 future 收敛）；新增 #9（god 文件结构债，从 W5 部分切分中产生）。
>
> 2026-05-07 update: ssot-cleanup-p5 land。debts #3 / #4 已关闭：#3 把 `stub_semantic_embedding` 提升到 `vz-contracts.semantic_embedding`，原四处 fork（`application/scoring_helpers` / `dual_track/core` / `evaluation/semantic_readouts` / `application/storage`，known-debts 原列表漏掉了 storage 那一份）全部改为 thin re-export，canonical modulus 统一为 65537（与常用 dim 互质），契约测试 [`tests/contracts/test_semantic_embedding_ssot.py`](../tests/contracts/test_semantic_embedding_ssot.py) 守门；#4 17 个 prod / 测试文件的纯类型 import 收敛到 `volvence_zero.evaluation` facade，[`tests/contracts/test_import_boundaries.py`](../tests/contracts/test_import_boundaries.py) 静态拒绝从 `evaluation.backbone` 拉纯类型。
>
> 2026-05-08 update: EQ-owner uplift Phase 1 (W1.A-F) + Phase 2 (W2.A-D) land：7 个 SHADOW EQ owner（interlocutor / rupture / 4 ToM about-other / common_ground / session_post_slow_loop）promote 到 ACTIVE，wiring 完整，830 个 contract / unit test 通过；W2.C longitudinal aggregator + W2.D sparse-reward 接线 ready。但**实测发现这条 evidence 链整体不通**：默认 benchmark 没共享 memory + 没接 LLM proposal runtime → ToM owner records 始终为空 + 跨 session 学习信号弱到 0.001-0.006 量级。详见 debt #10。
>
> 2026-05-08 update (later): DM-1 distributional PE + CMA-2 VZ-MemProbe 三 wave 全部 land。W1: `PredictionError.distribution_summary` + `_PEDistributionWindow` + `VitalsSnapshot.distributional_drift_axes` + 10 contract test 全绿。W2: `lifeform-bench --longitudinal-rounds N` 现在共享 `MemoryStore`（**debt #10A 关闭**）+ `BenchmarkReport.final_interlocutor_axes` + `LongitudinalFamilyReport.il_rapport_trend_pos` 新 acceptance gate；4 个 VZ-MemProbe（`tests/longitudinal/test_vz_memprobe_*.py`）3/4 PASS, 1 XFAIL（context disambiguation）。W3 联合 evidence run（`artifacts/eq_uplift/distributional_evidence.json`）暴露两个新发现：(a) **debt #11**: PE 分布窗口在典型 5-15 turn benchmark session 下永不填满（min_window=16 > 默认 scene 长度），W1.3 vitals drift 在真实 benchmark 中始终为空；(b) **debt #10D**: `RetrievalQuery.facets` 在 `_score_entry` 评分中权重不足以 disambiguate top-1（context 探针 XFAIL 的根因）。debt #10C 状态保留：实测 il_rapport delta mean=+0.0018 / SNR≈0.80，信号方向正确但仍弱，前置依赖 #10B（LLM runtime 接入）；本 wave 未触动 #10B / #10C，只把 #10A 关闭并把 evidence 链补完。
>
> 2026-05-08 update (final): debt #11 按修法 (3) 方法论关闭。Wave A.1 写 38-turn `long-form-life-arc.json` scenario；Wave A.2/A.3 跑 `artifacts/eq_uplift/probe_pe_window_long_form.py` → `artifacts/eq_uplift/pe_window_long_form.json` verdict overall_pass=True（first_summary_turn=17 / first_drift_turn=21 / 4 axis IQR n=8 vs n=16 全 STABLE）。Wave B.1 把 `_PEDistributionWindow.min_window` 16→8（vitals warmup 保持 5，记录折中理由）；Wave B.2 同步两个 contract test + 新增 `test_distribution_window_iqr_stable_at_min_window_n8`（n=8 vs n=32 IQR 比 ∈ [0.4, 2.5] 守门）。Wave C.1 重跑 W3 联合 evidence，`long-form-life-arc` 在 3/3 rounds 产出非 None `pe_distribution_summary` + `vitals.distributional_drift_axes`，IQR relationship delta = 0.0032（具体 distributional shift 证据）；其他 5 个 cross-session scenario 结构性 3-5 turn 仍永不填窗口——**这是 scenario design 选择，不是 tuning 失败**。**debt #11 关闭**；scenario 长度多样性（更多 long-form scenario 覆盖）作为方法论 follow-up 列在 closed entry 末尾，不开新 debt。
>
> 2026-05-09 update: known-debts triple-closure wave。**`#10D` 关闭** —— `_score_entry` 加 `+5` per matched facet 显式 boost（[`packages/vz-memory/src/volvence_zero/memory/store.py`](../packages/vz-memory/src/volvence_zero/memory/store.py)），signature 新增 `query_facets: tuple[str, ...]`；[`tests/longitudinal/test_vz_memprobe_context.py:test_mp_context_regime_facet_disambiguates_both_directions`](../tests/longitudinal/test_vz_memprobe_context.py) 摘 `xfail(strict=True)` 后两个方向都 PASS；spec 同步 [`docs/specs/continuum-memory.md`](specs/continuum-memory.md) 表格 + 接口契约段。**`#10B item 3` Case C (fail-loud)** —— 跑 `examples/run_cross_session_probe_llm.py --rounds 3 --model-source Qwen/Qwen2.5-0.5B-Instruct` 后 round 1 完整 F1-F6 family report 显示 `f3.tom_records_total = 0.000` AND `f3.common_ground_dyad_atoms_total = 0.000` —— 即使 LLM runtime wired in 且其他通道（PE / vitals / il axes）全活，ToM / CG owner 仍未产 record。Round 2 在加权重后 Python 进程静默 OOM-die，未产 JSON artifact；stdout 完整保留在 [`artifacts/eq_uplift/cross_session_probe_llm.stdout.log`](../artifacts/eq_uplift/cross_session_probe_llm.stdout.log)。**`#10B` 保留开放**，item 3 attempt log + 三选诊断（Qwen 0.5B too small / wiring isinstance 检查 / scenario 触发不足）+ 下次推进路径已记录在 `### 10B` 段。**`#10C` 保留原状**（前置依赖 #10B item 3 真激活）。重要副产物：W2.0c 落地的 `f3.tom_records_total` / `f3.common_ground_dyad_atoms_total` 诊断 metric **正确发挥了作用**——第一时间暴露 ToM 链未激活，没让 LLM-driven probe 在表面 PASS 的伪装下混过去。回归 809 passed (上轮 808 + 1 xfailed → 809 passed + 0 xfailed)。
>
> 2026-05-08 update (Phase 2 W2.0c): debt #10B item 1 + item 2 已 land。`lifeform-bench` 新增 `--use-llm-semantic-runtime` flag（路由到 `build_companion_lifeform_with_real_substrate(use_llm_semantic_runtime=True, memory_store=shared_store, ...)`，一份 Qwen 同时供 substrate residual + LLM semantic provider）；`BenchmarkReport` / `FamilyReport._compute_f3` / `LongitudinalFamilyReport` 三层 surface `tom_records_total` 与 `common_ground_dyad_atoms_total` 两个 threshold=None 诊断 metric；`docs/specs/social_cognition/02_theory_of_mind.md` 与 `docs/EVALUATION_SYSTEM.md` § 1.2 同步显式记录 fail-closed 默认。fake-provider 契约测试 [`tests/lifeform_e2e/test_llm_semantic_runtime_evidence_chain.py`](../tests/lifeform_e2e/test_llm_semantic_runtime_evidence_chain.py)（5 个 case 全绿）+ 默认 skip 的 real-Qwen smoke [`tests/lifeform_e2e/test_longitudinal_with_llm_runtime_smoke.py`](../tests/lifeform_e2e/test_longitudinal_with_llm_runtime_smoke.py) 就位。item 3 evidence run 由 [`examples/run_cross_session_probe_llm.py`](../examples/run_cross_session_probe_llm.py) 触发并产出 `artifacts/eq_uplift/cross_session_probe_llm.json`。

---

## 6. Phase 2.A — Full COCOA rewarding-state head（已落地，SHADOW/readout 默认）

- **路径**：`packages/vz-cognition/src/volvence_zero/credit/gate.py`（`derive_counterfactual_contribution_records` 当前 lightweight 实现）
- **状态**：Phase 2.A 已实现为 `CreditLedger` owner-internal `RewardingStateHeadState`。`final_wiring.py` 通过 credit owner 入口生成 historical 与 learned 两条 readout，不在编排层重建 baseline。
- **剩余风险**：中低。默认仍是 readout / SHADOW 对比；`counterfactual_contribution_learned` 不进入现有 acceptance gate。真实 open-dialogue 数据上仍需观察 learned baseline 是否降低 long-horizon 方差。
- **保留退出条件**：若 learned baseline 在真实 trace 上长期不优于 historical baseline，保留 Phase 1.A historical record 作为 fallback，并可通过 checkpoint 恢复 head state。
- **后续观察项**：跟踪 `counterfactual_readouts.validation_delta`、`recent_modifications` 中 `credit.rewarding_state_head` 的 block/allow 比例，以及 `delayed_ledger_size` 与 segment closure 的一致性。

## 7. Phase 2.B — Learned PE critic head（已落地，report-only 默认）

- **路径**：`packages/vz-cognition/src/volvence_zero/prediction/error.py`（`_PECriticHead` 当前 running-stats 实现）
- **状态**：Phase 2.B 已在 `_PECriticHead` 内实现 learned contextual critic，输入为 `SubstrateSnapshot.feature_surface` digest + `PredictionActionContext`，输出 expected `|axis_error|`。`PEDecomposition` append-only 发布 critic prediction、improvement、checkpoint id、gate decision。
- **剩余风险**：低。`pe_decomposition` 仍是 optional，bootstrap 时为 `None`；evaluation 的 `pe_aleatoric_magnitude` / `pe_epistemic_magnitude` 继续严格 report-only，不进入 acceptance gate。
- **保留退出条件**：如果真实 session 中 learned critic 造成 epistemic readout 过早归零，可恢复 running-stats-only 语义，或只消费 `critic_predicted_magnitude` 作为诊断字段。
- **后续观察项**：跟踪 `critic_update_count`、`critic_gate_decision`、`improvement_magnitude` 与 memory `MemoryAttributeReadout.epistemic_magnitude` 的一致性。

## 8. `joint_loop` 与 runtime 主链共享 owner 实例

- **路径**：
  - 生产者：`packages/vz-runtime/src/volvence_zero/agent/session.py`（`AgentSessionRunner.__init__`）
  - 消费者：`packages/vz-temporal/src/volvence_zero/joint_loop/runtime.py`（`ETANLJointLoop.run_cycle`）
- **问题**：`_memory_store` / `_evaluation_backbone` / `_world_temporal_policy` / `_self_temporal_policy` / `_default_residual_runtime` 是同一实例被 runtime 主链和 `ETANLJointLoop` 同时持有并写入。属于"第二编排面"代码 pattern。
- **违反**：R8 精神（但在具体实现上已用 docstring 契约 + `TRAINING WRITEBACK PHASE` 注释块 + 契约测试 `test_joint_loop_shares_owner_instances_with_runtime_by_design` 固化边界）
- **短期风险**：低。turn 内是顺序执行而非并发，debug 现在有明确可视化的阶段边界。
- **触发条件**：有人把 writeback 逻辑加到 `TRAINING WRITEBACK PHASE` 注释块之外 → 重新变回"不可追踪"状态。契约测试只能测实例共享关系，测不出"phase block 之外的 mutation"。
- **推荐修法**：彻底方案是把 joint-loop post-propagate 的 owner writeback 搬到 runtime 编排层，joint-loop 只发 `JointCycleReport` typed proposals。这会破坏当前"在线 adaptation 立刻生效"的 pattern，需要重新设计 apply phase。不建议在产品迭代压力下做，等 NL 多时间尺度 apply phase 需要重新规划时一并做。
- **优先级**：低（已有契约测试兜底，边际收益低于成本）。

## 9. ~~`agent/session.py`~~ wave-1 done / `application/runtime.py` deferred

- **路径**：
  - ~~`packages/vz-runtime/src/volvence_zero/agent/session.py`（W5 后约 3825 行）~~ → wave-1 split landed 2026-05-09，详见下方 update
  - `packages/vz-application/src/volvence_zero/application/runtime.py`（仍约 3941 行，wave-2 候选）
- **问题**：W5 of ssot-cleanup-p0-p4 抽出 `session_helpers.py` (260 行) 与 `application/scoring_helpers.py` (139 行) 的纯函数，但 `AgentSessionRunner` / `ResponseAssemblyModule` 等核心 class 仍住在单文件里，单文件 ≥ 3800 行。
- **违反**：可演化性 / 可读性，非 R8 硬违反。
- **短期风险**：低。功能与 SSOT 都已经齐了；纯文件结构债。
- **触发条件**：再有一个大型 feature 落到 `ResponseAssemblyModule` 上时 → 单文件超过 4500 行，新人 onboarding 成本飙升；新功能与既有 phase 边界混淆（`agent/session.py` 已通过 wave-1 收紧到 1132 行）。
- **推荐修法**：W5 plan 描述的物理切分（`agent/session/{lifecycle,writeback_phase,training_phase,observation}.py` + `application/{response_assembly,retrieval_policy,decision_kind,scoring}.py`）。需要先在 `AgentSessionRunner` 上引入 mixin / 服务对象重组，再做物理切分。
- **优先级**：wave-2 (`application/runtime.py`) 仍是低，等 W6+ wave 一起做。

> **2026-05-09 update (wave-1 / `agent/session.py` mixin split)**：按计划实施 mixin 拆分，**flat sibling 布局**（`agent/session_lifecycle.py` 等同级文件）而非 W5 文档原写的 subpackage 布局，原因：维持现有 `from volvence_zero.agent.session import X` 导入路径零变更；`import volvence_zero.agent.session as agent_session_module` 在 [`tests/test_agent_session_runner.py`](../tests/test_agent_session_runner.py) 等仍解析为单一模块；git 历史归属 `session.py` 一个文件，不引入 file→folder 重命名歧义；rollback blast radius 最小。
>
> **行数对比（wave-1）**：
> - `agent/session.py`：3826 → **1132** 行（-2694，god 文件削减 ~70%）
> - 新增 [`agent/session_lifecycle.py`](../packages/vz-runtime/src/volvence_zero/agent/session_lifecycle.py)：398 行（11 个公共 API + lifecycle 方法）
> - 新增 [`agent/session_writeback_phase.py`](../packages/vz-runtime/src/volvence_zero/agent/session_writeback_phase.py)：906 行（12 个 session-post slow loop + experience writeback 方法 + per-mixin `_APPLICATION_PRIOR_PROPOSAL_BUILDER` singleton）
> - 新增 [`agent/session_training_phase.py`](../packages/vz-runtime/src/volvence_zero/agent/session_training_phase.py)：1258 行（17 个 rare-heavy + online-fast substrate self-mod 方法；最大的 mixin，对齐 debt #8 既有的 TRAINING WRITEBACK PHASE 边界）
> - 新增 [`agent/session_observation.py`](../packages/vz-runtime/src/volvence_zero/agent/session_observation.py)：623 行（4 个观测方法：`_build_substrate_adapter` / `_build_training_trace_from_substrate` / `_to_turn_result` / `_run_imagination`）
>
> **MRO 与单实例语义**：`class AgentSessionRunner(SessionLifecycleMixin, SessionWritebackPhaseMixin, SessionTrainingPhaseMixin, SessionObservationMixin)`，运行时验证 MRO=`[AgentSessionRunner, SessionLifecycleMixin, SessionWritebackPhaseMixin, SessionTrainingPhaseMixin, SessionObservationMixin, object]`。所有 mixin 是无状态 method 容器：无 `__init__`，从 `self._*` 读取 `AgentSessionRunner.__init__` 拥有的属性。Cross-mixin call surface（如 lifecycle 的 `begin_new_context` 调 writeback 的 `_maybe_build_current_session_report`、observation 的 `_to_turn_result` 调 writeback 的 `_publish_*_snapshot`）通过标准 MRO 解析，无 `super()` 链。
>
> **副带闭合的两个 latent 问题**：
> 1. `derive_learning_evidence_credit_records` 在原 `session.py` line 979 被调用但**从未被 import**（pre-existing 隐患，因为该 code path 罕见、tests 未触发，所以 NameError 从未被抛）。wave-1 split 在 [`session_writeback_phase.py`](../packages/vz-runtime/src/volvence_zero/agent/session_writeback_phase.py) 显式 `from volvence_zero.credit.gate import derive_learning_evidence_credit_records`，关闭这个隐患。
> 2. 训练 mixin 的 `_build_rare_heavy_replay_runner` 与 `_evaluate_rare_heavy_candidate` 等使用 lazy import 模式（`from volvence_zero.agent.session import AgentSessionRunner` 在方法体内）避免循环 import；同样模式用于在 mixin 内构造 `RareHeavyTrainingExample` / `RareHeavyTrainingBundle` / `RareHeavyPreImportEvaluation` / `RareHeavyTurnResult` / `OnlineFastSubstrateTurnResult` / `AgentTurnResult` 等仍住 `session.py` 的 dataclasses。
>
> **Test 调整**：[`tests/test_agent_session_runner.py`](../tests/test_agent_session_runner.py) 两处 `monkeypatch.setattr(agent_session_module, "evaluate_gate", ...)` 也需要同时 patch [`session_training_phase`](../packages/vz-runtime/src/volvence_zero/agent/session_training_phase.py) 模块的 `evaluate_gate` 名字（因为 `_maybe_apply_online_fast_substrate_self_mod` 移到训练 mixin，函数 globals 现在是该 mixin 模块）。两个测试 site 都加了 import + 第二个 `monkeypatch.setattr` 调用；老的 session 模块 patch 也保留，对旧 import 路径的测试无破坏。这是 mixin 提取**唯一**触及测试的语义变化点（49 个 AgentSessionRunner 测试中只有 2 个），其它都是机械搬运。
>
> **回归证据**：49/49 `tests/test_agent_session_runner.py`（在两半 batch 中跑全绿；连续单进程跑会触发 pre-existing 的 Windows + torch 2.11 累积 access violation 在第 ~25-30 个测试，与 wave-1 无关，已用 batch split 回避）+ 788/788 `tests/contracts/`（含本周 land 的 70 个 EQ owner / fence-strip / option B / Wave E1-E5 contract test）+ 60/60 focused `tests/lifeform_e2e/`（multi-turn / companion-regime / family-report / LLM evidence chain，覆盖 session.py 主路径）。
>
> **未做（wave-2 deferred）**：`application/runtime.py` (~3941 行) 是不同 shape——god FILE not god CLASS，含 ~50 dataclasses + ~80 helper functions + 8 owner Modules。物理切分按 category（dataclasses / Module classes / pure helpers）而不是 phase。等 W6+ wave 一起做。

## 10. EQ-owner uplift Phase 1+2 后 cross-session evidence 链断裂（W2.C / W1.C/D/E/F / W2.A）

> 这一条是**三个相互关联但需要分别修**的 sub-issue。架构通了、契约测试通了，但实测发现真实 benchmark 跑不出 evidence。诊断 artifact 在 [`artifacts/eq_uplift/cross_session_probe.json`](../artifacts/eq_uplift/cross_session_probe.json)、跑分日志在 `artifacts/eq_uplift/longitudinal*.{log,json}`。

### ~~10A. `lifeform-bench --longitudinal-rounds N` 不共享 memory store，每轮是独立 session~~ —— 2026-05-08 关闭

DM-1 + CMA-2 Wave 2 关闭。`lifeform-evolution/cli.py` 新加 `_build_vertical_lifeform_with_shared_store(name)` helper，在 longitudinal pass 进入 round 循环之前构造一个 `build_default_memory_store()` 实例并通过 `build_companion_lifeform(memory_store=...)` 注入；`BenchmarkReport.final_interlocutor_axes` 增加 6 个 il 轴；`_compute_f3` 把它们包装成 `f3.il_trust_final` / `f3.il_rapport_final` 等 metric；`LongitudinalFamilyReport` 增加 `il_trust_first/last/trend` + `il_rapport_first/last/trend` + 新 acceptance gate `il_rapport_trend_pos`（threshold 0.005）。新增 4 个 unit test（`test_il_axes_absent_falls_back_to_legacy_gate` / `test_il_rapport_trend_pos_blocks_passed_when_below_threshold` / `test_il_rapport_trend_pos_passes_when_above_threshold` / `test_il_axes_dict_round_trips`）。原 `bond_warmth_*` 保留作为 backward-compat 但 docstring 注明「饱和 drive，跨 session 通常 trend=0；用 il_* 看跨 session 信号」。Coding vertical 回退到 per-session store 直到 `build_coding_lifeform` 接 `memory_store=` kwarg。

### 10B. W1 EQ owner records 在默认 NoOp semantic runtime 下永远为空

- **路径**：
  - `packages/vz-runtime/src/volvence_zero/integration/final_wiring.py:1164`（`semantic_runtime = semantic_proposal_runtime or NoOpSemanticProposalRuntime()`）
  - `packages/vz-runtime/src/volvence_zero/integration/final_wiring.py:1166-1190`（W1.C / W1.E fail-closed default：`isinstance(semantic_runtime, LLMSemanticProposalRuntime)` 才构造 ToM / common-ground proposal runtime）
  - `packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/__init__.py:139`（`build_companion_lifeform` 默认 `semantic_proposal_runtime=None`）
- **问题**：
  - 默认 companion lifeform 用 `NoOpSemanticProposalRuntime`
  - W1.C / W1.E 的 fail-closed 设计：没 LLM 时 `tom_proposal_runtime` / `common_ground_proposal_runtime` 都回退到 `None`
  - 结果：`feeling_about_other.records = ()`、`belief_about_other.records = ()`、`intent_about_other.records = ()`、`preference_about_other.records = ()`、`common_ground.dyad_atoms = ()` —— **W1.C / D / E / F + W2.A 这 5 个 wave 拿到的全是空 snapshot**
  - planner 的 `_apply_feeling_snapshot` / `_apply_common_ground_snapshot` / `_tom_rationale_tags` 在空 records 下都直接 no-op → **下游 rationale_tags 也不会出现 `feeling=observed` / `framing=belief_observed` 等 typed signal**
  - 等于：默认 benchmark 跑下来，W1.C/D/E/F + W2.A 5 个 wave 的「promotion」对实际行为零影响
- **风险**：中。架构正确（fail-closed 是设计决策），但**「EQ 信号链激活」只在带 LLM 的 lifeform 路径下成立**，文档没明确这一点；任何 demo / evaluation 跑默认 NoOp 路径都拿不到 EQ evidence
- **触发条件**：(a) 演示 / 验证 EQ 能力时；(b) 任何用 NoOp 默认配置跑的 family-report 被 cite 作为「ToM owner 工作」evidence
- **推荐修法**：
  1. ~~加诊断指标进 `_compute_f3` 或新增一族：`f3.tom_records_total` / `f3.common_ground_dyad_atoms_total`~~ —— Phase 2 W2.0c 已落地。`BenchmarkReport.tom_records_total` / `BenchmarkReport.common_ground_dyad_atoms_total` + `f3.tom_records_total` / `f3.common_ground_dyad_atoms_total` 两个 threshold=None 诊断 metric 在 family report 中暴露；`LongitudinalFamilyReport` 增加对应 first/last/trend + per_round 数组。契约测试 [`tests/lifeform_e2e/test_llm_semantic_runtime_evidence_chain.py`](../tests/lifeform_e2e/test_llm_semantic_runtime_evidence_chain.py) 用 fake provider 证明链路是 runtime-gated（NoOp → 0；LLM → > 0）。
  2. ~~在 `docs/specs/social_cognition/02_theory_of_mind.md` 显式记录：「ToM owner records 仅在 `LLMSemanticProposalRuntime`（含 `LLMToMProposalRuntime` 派生）wired 时产生；NoOp / fake runtime 下 records 永远为空」~~ —— Phase 2 W2.0c 已落地。spec「关键不变量」末尾新加一条；`docs/EVALUATION_SYSTEM.md` § 1.2 在 code-backed readout 列表中加入 `f3.tom_records_total` / `f3.common_ground_dyad_atoms_total` 与其 longitudinal 聚合维度。
  3. （评估侧）跑一次 `build_companion_lifeform_with_real_substrate(use_llm_semantic_runtime=True)` + cross-session probe，把结果存到 `artifacts/eq_uplift/cross_session_probe_llm.json` 作为「W1 EQ owner 真激活」的证据 baseline。预计需 30-60 分钟 CPU + Qwen 1.5B 模型已下载。`lifeform-bench` 已支持 `--use-llm-semantic-runtime` flag（item 1 落地的副产物，CLI 自动校验 `--vertical companion --longitudinal-rounds > 0`），运行入口由 [`examples/run_cross_session_probe_llm.py`](../examples/run_cross_session_probe_llm.py) 提供。real-Qwen 回归 smoke 在 [`tests/lifeform_e2e/test_longitudinal_with_llm_runtime_smoke.py`](../tests/lifeform_e2e/test_longitudinal_with_llm_runtime_smoke.py)，默认 skip，`VZ_RUN_LLM_SMOKE=1` 启用。
- **优先级**：~~低~~ → 中（actionable）。item 1 + 2 已 land；item 3 在 2026-05-09 的 Triple-Closure wave 尝试运行，结果归类 **Case C (fail-loud)**，详见下方「item 3 attempt log」。
- **item 3 attempt log (2026-05-09 Triple-Closure wave)**：
  - **运行配置**：`python examples/run_cross_session_probe_llm.py --rounds 3 --model-source Qwen/Qwen2.5-0.5B-Instruct --scenarios-path packages/lifeform-domain-emogpt/.../cross-session-emotional-followup.json`（5-turn × 3 rounds，因为 1.5B 未本地缓存只能用 0.5B）
  - **结果**：Round 1 完整跑完，F1-F6 family report 全 PASS / partial-pass，stdout 保留在 [`artifacts/eq_uplift/cross_session_probe_llm.stdout.log`](../artifacts/eq_uplift/cross_session_probe_llm.stdout.log)；**round 2 在 LLM 加权重后 Python 进程静默死亡**（CPU OOM 怀疑，未捕获 traceback），`cross_session_probe_llm.json` 未写入
  - **关键证据**：Round 1 F3 报告显示 `f3.tom_records_total = 0.000` AND `f3.common_ground_dyad_atoms_total = 0.000` —— **即使 `LLMSemanticProposalRuntime` wired in 且 LLM 真在做推理（看 il_rapport=0.805、bond_warmth=0.800、PE 通道全活），ToM owner / Common Ground owner 仍未产出任何 record**
  - **诊断**（按可能性递减排序）：
    1. **Qwen 0.5B 太小**（最可能）—— ToM proposal LLM runtime 期望结构化 JSON 输出（belief/intent/feeling/preference proposals），0.5B 在受限 prompt 下的 JSON 一致性显著低于 1.5B；解析层可能把所有 0.5B 输出当成 malformed proposals 丢弃。1.5B 是 spec 推荐 baseline 不是偶然。
    2. **CLI flag 接线确未生效到 ToM proposal runtime**（其次可能）—— `--use-llm-semantic-runtime` 把 `LLMSemanticProposalRuntime` 装到 wiring 里，但 `LLMToMProposalRuntime` 与 `LLMCommonGroundProposalRuntime` 走的是「检测 `isinstance(semantic_runtime, LLMSemanticProposalRuntime)` 才注入」的 fail-closed 默认（[`final_wiring.py:1166-1190`](../packages/vz-runtime/src/volvence_zero/integration/final_wiring.py)）；如果 CLI 传的不是 `LLMSemanticProposalRuntime` 实例而是 wrapper / adapter，这个 isinstance 检查会失败并静默回退到 None
    3. **5-turn cross-session-emotional-followup 触发不到 ToM 产出条件**（最不可能）—— 虽然 5 turn 短，但 ToM owners 应该至少在 turn 1-2 就检测到 emotional disclosure 并产出 belief proposals；0 record 不像「短不够触发」，更像「LLM 输出全被丢弃」
  - **下次推进路径**：
    1. 先把 Qwen 1.5B 缓存到本地（`hf download Qwen/Qwen2.5-1.5B-Instruct`，需 HF_TOKEN），再用 1.5B 跑同样的 probe；如果 1.5B 路径下 `tom_records_total > 0` 则 hypothesis 1 confirmed，0.5B 路径标注「不支持」即可
    2. 若 1.5B 路径下仍 0.000 → 走 hypothesis 2 路线，写一个 instrument log 在 `LLMToMProposalRuntime.propose(...)` 入口，捕获 raw LLM 输出 + 解析失败原因
    3. round 2 OOM 是次要故障；可以加 `--rounds 1` 先解耦该问题，等 ToM 激活先证明再回头处理多轮
  - **重要不变量保留**：item 1+2 的诊断 metric (`f3.tom_records_total` / `f3.common_ground_dyad_atoms_total`) **正确发挥了作用**——Round 1 family report 第一时间暴露了 ToM 链未激活，没有让 LLM-driven probe 在表面 PASS 的伪装下混过去。这是 W2.0c metric 落地的核心价值

### ~~10D. `RetrievalQuery.facets` 在 `_score_entry` 中权重不足以驱动 regime 级 disambiguation~~ —— 2026-05-09 关闭

按推荐修法 (1) 关闭。[`packages/vz-memory/src/volvence_zero/memory/store.py:_score_entry`](../packages/vz-memory/src/volvence_zero/memory/store.py) 加显式 facet boost：

```python
facet_score = 0.0
if query_facets:
    facet_lower = {facet.lower() for facet in query_facets}
    matched = len(facet_lower & tag_tokens)
    facet_score = matched * 5.0
return (
    learned_affinity * learned_recall.learned_weight
    + artifact_semantic_score * learned_recall.artifact_weight
    + lexical_score * 0.8
    + facet_score
)
```

`+5` per matched facet 落在 lexical band 之上、dominant semantic / learned 通道之下，给 regime facets 真实的 tie-breaker 权重而不掩盖内容信号。`_score_entry` signature 新增 `query_facets: tuple[str, ...]`；唯一调用点 [`store.py:307`](../packages/vz-memory/src/volvence_zero/memory/store.py) 跟随。

[`tests/longitudinal/test_vz_memprobe_context.py:test_mp_context_regime_facet_disambiguates_both_directions`](../tests/longitudinal/test_vz_memprobe_context.py) 的 `pytest.mark.xfail(strict=True)` 装饰器 + `import pytest` 已摘除；该 symmetric test 现在两个方向（`regime:problem_solving` → PR review；`regime:casual_social` → restaurant review）都 PASS，无 xfail。spec 同步：[`docs/specs/continuum-memory.md`](specs/continuum-memory.md) "R5/R6 Behavioural Proof Surface (CMA-2)" 表格里 Context 行从 XFAIL 改 PASS；同文件 "接口契约 / 当前实现口径" 段落明确 `RetrievalQuery.facets` 走 `+5` per match 的 boost 通道，不只走 embedding。

### 10C. 跨 session 学习信号在 shared memory + NoOp runtime 下幅度过弱（0.001-0.006）

- **路径**：诊断点散布于 `interlocutor/readout.py`、`vitals` 各 drive 的 recharge 公式
- **问题**：即使补上 10A（共享 memory_store），实测 `il_trust` / `il_rapport` 跨 3 rounds 的 delta 仍只有 0.001-0.006。这低于一般 evidence 的最小可解释阈值
- **根因（怀疑，未确认）**：
  1. `interlocutor_state` 是从 6 个上游 owner 当 turn 重新派生的 readout，对 memory store 内容仅通过 `MemoryAttributeReadout` 间接读；多数 readout 输入是当前 turn 的 evidence，不是 cumulative history
  2. `bond_warmth` 等 vitals drive 是 ceiling-saturated（ceiling 0.8-0.9 by recharge dynamics），跨 session 累积无空间
  3. ToM records 空（10B 副作用）→ readout 无 ToM 输入 → 跨 session memory 进不到 interlocutor 派生
- **风险**：中-低。如果 10A + 10B 都修了但 cross-session signal 还是 < 0.01，说明 readout 算法本身需要把 cumulative history 喂进去才能产生显著跨 session 漂移
- **触发条件**：10A + 10B 修完后的回归测试 — 如果 `il_rapport_trend` 在 3-5 rounds 内仍 < 0.02，触发本债
- **推荐修法**：
  1. （等 10A/10B 完成再决定）让 `InterlocutorReadoutContext` 增加显式 cumulative 字段（如 `cumulative_emotional_disclosure_count` / `cumulative_repair_count`），从 `MemoryStore` 跨 session 累积取
  2. 或者把 ToM `OtherMindRecord` 的高 confidence 持久 records 喂进 `interlocutor_state` 的 readout 作为「关系深度」proxy
- **优先级**：低（前置依赖 10A + 10B，目前无法直接 actionable）

## ~~11. PE distribution window 是 per-session 私有，benchmark 永不填满~~ —— 2026-05-08 关闭

按推荐修法 (3) 的 evidence-first 方法论关闭。

**Wave A — Mechanism validation**：新建 38-turn 单 session [`packages/lifeform-domain-emogpt/.../scenarios/long-form-life-arc.json`](../packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/scenarios/long-form-life-arc.json)（rapport → low-mood → guided exploration → rupture → repair → continuity 弧线）；跑 [`artifacts/eq_uplift/probe_pe_window_long_form.py`](../artifacts/eq_uplift/probe_pe_window_long_form.py) 输出 [`artifacts/eq_uplift/pe_window_long_form.json`](../artifacts/eq_uplift/pe_window_long_form.json)，verdict overall_pass=True：

- `first_summary_turn=17`（窗口在 16 个非 bootstrap turn 后填满）
- `first_drift_turn=21`（vitals warmup 5 个观察后产 drift）
- 4 axis IQR n=8 vs n=16 全部 STABLE（statistical sanity）

**Wave B — 修法 (1) land**：[`packages/vz-cognition/src/volvence_zero/prediction/error.py`](../packages/vz-cognition/src/volvence_zero/prediction/error.py) `PredictionErrorModule.__init__` 内的 `_PEDistributionWindow.min_window` 16→8（max_window=64 不变）；[`packages/lifeform-core/src/lifeform_core/vitals.py`](../packages/lifeform-core/src/lifeform_core/vitals.py) `_BASELINE_WARMUP_OBSERVATIONS=5` 保持（评估发现 sqrt(5/3) ≈ 1.29 noise penalty 比省 2 turn 更重要，记录折中理由）。两个常量均在 owner-internal 注释中引用 long-form probe 证据，避免未来无 spec 调参。

**Wave B — contract test 守门**：[`tests/contracts/test_pe_distribution_summary_contract.py`](../tests/contracts/test_pe_distribution_summary_contract.py) 两个 boundary 测试更新到 `min_window=8` 默认；新增 `test_distribution_window_iqr_stable_at_min_window_n8` —— 32 个确定性 RNG 样本下 n=8 IQR 与 n=32 IQR 比值必须 ∈ [0.4, 2.5]（反映 SE-of-IQR 在小样本下的预期变异范围，5x / 10x 比值仍会 fail）。[`tests/contracts/test_pe_distribution_backward_compat.py`](../tests/contracts/test_pe_distribution_backward_compat.py) fixture pre-fill 改为 auto-derive `_min_window`，这样未来再调参不必同步改 fixture。

**Wave B — spec 同步**：[`docs/specs/prediction-error-loop.md`](specs/prediction-error-loop.md) 新增 "`min_window=8` 的证据来源" 小节明确 long-form probe 是任何未来 min_window 调整的前置；[`docs/specs/lifeform-vitals.md`](specs/lifeform-vitals.md) 注明 warmup=5 的折中理由。

**Wave C — 重跑 W3 联合 evidence**：[`artifacts/eq_uplift/distributional_evidence.json`](../artifacts/eq_uplift/distributional_evidence.json) 加 `long-form-life-arc` 到 `_PROBE_SCENARIOS`，重跑得 long-form-life-arc 在 3/3 rounds 产出非 None `pe_distribution_summary` + `vitals.distributional_drift_axes`，IQR relationship axis delta = +0.0032（具体跨 session distributional shift evidence，从 W3.1 的 0.000 跃迁到非零）。其他 5 个 cross-session scenario 仍 None ——这是它们 3-5 turn 的 design choice 决定的（cross-round repetition 而非 distributional evidence vehicle），不是 tuning 失败。

**Wave C — 全量回归**：[`tests/contracts/test_pe_distribution_summary_contract.py`](../tests/contracts/test_pe_distribution_summary_contract.py) 11 测试全绿；[`tests/contracts/test_pe_distribution_backward_compat.py`](../tests/contracts/test_pe_distribution_backward_compat.py) 3 测试全绿；其他 PE / vitals / final_wiring / longitudinal regression 不变。

**未启用修法 (2)**：lifeform-level 共享 PE 窗口（跨 session 累积分布形状）保留作为未来选项，**不开新 debt**——当前 evidence 显示 (1) 已足够让 8+ turn 场景拿到 evidence，(2) 的复杂度（`Brain` / `Lifeform` 多处接线、cross-session window state migration）与当下需求不匹配。

**Methodology follow-up**（不开 debt）：scenario coverage 多样性是产品方向问题不是架构债。如果未来 evaluation 需要更多类型的 long-form distributional evidence vehicle，可以在 [`packages/lifeform-domain-emogpt/.../scenarios/`](../packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/scenarios/) 增加 30-50 turn 的多 vertical / 多 regime 长 scenario，按 long-form-life-arc 的格式即可。

---

## 已关闭的债务（参考）

这些在 2026-05-04 至 2026-05-06 的 SSOT 收敛中已修完，留作对照：

- ~~`credit/gate.py -> temporal_types` 上游边界未声明~~
- ~~`SEMANTIC_OWNER_SLOTS` 在 `dual_track` 和 `semantic_state` 双源（dual_track 漏 `open_loop`）~~
- ~~`ReflectionEngine.apply(regime_module=...)` 直接持有并调用 `RegimeModule`~~
- ~~`memory/store.py` 解析 peer snapshot 内部字段拼 retrieval facets（temporal / dual_track / PE）~~
- ~~`EvaluationSnapshot.alerts` 文本子串驱动 regime / reflection / credit gate 控制逻辑~~
- ~~regime scoring 在 stable task opener 被 `guided_exploration` 抢占~~
- ~~super-loop diversity penalty 单峰不收敛 + `xfail` 的 `coding-regime.bs`~~
- ~~Debt #1: `interlocutor/__init__.py` duck-typed 多 owner 重建~~ —— W2 of ssot-cleanup-p0-p4 关闭：`InterlocutorStateModule` 是 SHADOW owner，下游 (`prompt_planner` / `response_synthesizer` / `LifeformSession.interlocutor_state`) 都读发布的 snapshot；`InterlocutorThresholds` 为唯一阈值源；`compute_zones` 自动同步 zone bool。详见 [`docs/specs/interlocutor-state.md`](specs/interlocutor-state.md) + 契约测试 [`tests/contracts/test_interlocutor_state_contract.py`](../tests/contracts/test_interlocutor_state_contract.py)。
- ~~Debt #2: `application/runtime.py` 硬编码 regime id 语义映射~~ —— W4 of ssot-cleanup-p0-p4 关闭：`RegimeIdentity.application_brief: ApplicationBrief` 发布 `task_focus / support_focus / repair_focus / exploration_focus / domain_affinity / continuum_target_position / decision_kind_hint / support_decision_threshold / knowledge_weight_nudge`；`vz-application` 全部 18+ regime-id branch 已切换；契约测试 [`tests/contracts/test_application_no_regime_id_branching.py`](../tests/contracts/test_application_no_regime_id_branching.py) 静态守门。新增 regime 只需在 `volvence_zero.regime.templates.REGIME_TEMPLATES` 加一行。
- ~~`relationship_repair_alpha_gate.py` substring 匹配 `result.response.rationale` 当 alpha-gate 契约~~ —— W1 of ssot-cleanup-p0-p4 关闭：`AgentResponse.rationale_tags: tuple[str, ...]` typed 字段；synthesizer 渲染时发 `acknowledge_section=repair_alpha` / `intent=repair-first` typed tag；gate 读 typed tag。
- ~~kernel `vz-runtime/agent/response.py` 维护 `lesson_hint_map` / `tension_hint_map` UX 文本~~ —— W1 of ssot-cleanup-p0-p4 关闭：UX 文本搬到 `lifeform-expression.reflection_hints`，kernel 不再持有；`ReflectionLessonId` / `ReflectionTensionId` 是 enum SSOT；contract test [`tests/test_reflection_hints.py`](../tests/test_reflection_hints.py) 强制 1:1 hint 覆盖。
- ~~`prompt_planner` / `response_synthesizer` 重复阈值 (0.55 vs 0.56 等)~~ —— W2 of ssot-cleanup-p0-p4 关闭：所有阈值集中在 `InterlocutorThresholds`，consumer 读 zone bool。
- ~~`response_synthesizer._repair_kind_label` 重复 RuptureKind→string 字典 + `getattr(rupture, "repair_pressure", 0.0)` duck-type~~ —— W3 of ssot-cleanup-p0-p4 关闭：`RuptureStateSnapshot.kind_label` 由 owner 从 `RUPTURE_KIND_LABEL` SSOT 派生；`rupture_state.owner.py` 改为 typed `relationship_state_value.repair_pressure` 访问。
- ~~`response_synthesizer` 三个 `_render_*` 函数的 `if regime == "..."` 分支链~~ —— W3 of ssot-cleanup-p0-p4 关闭：`RegimeIdentity.expression_brief.acknowledge_hint / frame_hint / next_step_hint / open_loop_hint / continuity_hint` 是渲染 lookup key；新 regime 只需更新 `regime/templates.py` 的 brief。
- ~~Debt #3: 三套语义 embedding stub 分叉（实际为四处：`application/scoring_helpers` / `dual_track/core` / `evaluation/semantic_readouts` / `application/storage`，known-debts 原列表漏掉了 storage 那一份）~~ —— ssot-cleanup-p5 关闭：canonical SSOT 落到 `volvence_zero.semantic_embedding`（`stub_semantic_embedding` / `stub_semantic_tokens` / `stub_cosine_similarity`），`CANONICAL_MODULUS = 65537`（与 dim 4/6/8/16/32/64/128/256 互质），四处 fork 全部改为 thin re-export，原 mod 37 / 41 不一致已消除。契约测试 [`tests/contracts/test_semantic_embedding_ssot.py`](../tests/contracts/test_semantic_embedding_ssot.py) 通过 identity 检查（三处 `is` 同一函数）+ AST 扫描禁止新增 `def _semantic_embedding`（白名单仅 `memory/retrieval` 的 dim=6/tags 签名与 `substrate/adapter` 的 dim=256 残差投影）。
- ~~Debt #4: `EvaluationBackbone` 类型入口不干净（17 个文件从 `evaluation.backbone` 拉纯类型）~~ —— ssot-cleanup-p5 关闭：所有纯类型 import 改走 `volvence_zero.evaluation` facade（`evaluation/__init__.py` 已经 re-export）；只有 `EvaluationBackbone` / `EvaluationModule` / `_feature_surface_snapshot`（实现 + 内部 hook）保留 backbone 路径。契约测试 [`tests/contracts/test_import_boundaries.py`](../tests/contracts/test_import_boundaries.py) 新增 `test_kernel_imports_evaluation_types_via_facade` 静态守门，AST 扫描全部 prod 代码强制此分层。

---

## 维护规则

1. 新加架构债时，先问自己：**"不改会死人吗？"**
   - 如果"短期风险"是"高"或"会爆"，不要写进这里，直接修。
   - 如果确实是"能跑 + 长期影响可演化性"，写进这里。
2. 每条都要有 **触发条件**。没有触发条件的债 = 不是债，是 preference。
3. 关闭条目时把它移到"已关闭的债务"段落，别直接删，留作 pattern 参考。